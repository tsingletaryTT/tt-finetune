"""Canonical TTNN O-projection path nested in one artifact-v4 owner."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_program_composition import compiled_ttl_program


ttl = None
tt_lang_operation = None


def _compute_config(*, math_approx_mode: bool) -> dict:
    return {
        "math_fidelity": "HiFi4",
        "math_approx_mode": bool(math_approx_mode),
        "fp32_dest_acc_en": True,
        "packer_l1_acc": False,
        "dst_full_sync_en": False,
    }


@lru_cache(maxsize=16)
def build_tt_lang_native_lora_o_path(
    *,
    tokens: int,
    hidden_features: int,
    logical_rank: int,
    scale: float,
    debug_stage: str = "full",
    memory_mode: str = "l1",
    frozen_weight_transposed: bool = False,
    rank_k_tiles_per_phase: int = 2,
) -> Callable:
    """Build exact ``base + scale * ((head @ A.T) @ B.T)``."""

    if tokens <= 0 or tokens % 32:
        raise ValueError("tokens must be positive and tile aligned")
    if hidden_features <= 0 or hidden_features % 32:
        raise ValueError("hidden_features must be positive and tile aligned")
    if not 1 <= logical_rank <= 128:
        raise ValueError("canonical O path supports logical rank up to 128")
    physical_rank = ((logical_rank + 31) // 32) * 32
    rank_tiles = physical_rank // 32
    if (
        rank_k_tiles_per_phase <= 0
        or (hidden_features // 32) % rank_k_tiles_per_phase
    ):
        raise ValueError("rank K phase must divide the hidden tile count")
    if debug_stage not in {"base", "delta", "scaled", "full"}:
        raise ValueError("debug_stage must be base, delta, scaled or full")
    if memory_mode not in {
        "l1", "all_dram", "base_dram", "delta_dram", "scaled_dram"
    }:
        raise ValueError(
            "memory_mode must be l1, all_dram, base_dram, delta_dram or "
            "scaled_dram"
        )

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError("native LoRA O path requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid="full")
    def _probe(head, frozen_weight, lora_a, lora_b, output, rank_hidden):
        source = ttl.make_dataflow_buffer_like(
            head, shape=(1, 1), block_count=2
        )
        wide = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=1
        )
        narrow = ttl.make_dataflow_buffer_like(
            rank_hidden, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                transaction = ttl.copy(head[0:1, 0:1], tile)
                transaction.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with wide.reserve() as result:
                    result.store(tile)
                with narrow.reserve() as result:
                    result.store(tile)

        @ttl.datamovement()
        def write():
            with wide.wait() as tile:
                transaction = ttl.copy(tile, output[0:1, 0:1])
                transaction.wait()
            with narrow.wait() as tile:
                transaction = ttl.copy(tile, rank_hidden[0:1, 0:1])
                transaction.wait()

    variant = (
        f"t{tokens}.h{hidden_features}.r{logical_rank}."
        f"s{float(scale):.9g}.{debug_stage}.{memory_mode}."
        f"wt{int(frozen_weight_transposed)}."
        f"rk{rank_k_tiles_per_phase}.rt{rank_tiles}.v7"
    )

    # Public table: head, W, A, B, output, H.  Private L1 table:
    # base, delta, scalar, scaled-delta.  H is deliberately a public result in
    # this qualification owner; the attention owner will map it into its
    # private checkpoint arena.
    rank_program_config = {
        "grid": [11, 10],
        "in0_block_w": rank_k_tiles_per_phase,
        "out_subblock_h": 1,
        "out_subblock_w": 1,
        "out_block_h": 1,
        "out_block_w": 1,
        "per_core_m": 1,
        "per_core_n": 1,
        "fuse_batch": False,
    }
    if logical_rank >= 32:
        # The rank-32 canonical graph selects the explicit 2-D family.  The
        # legacy rank-8/16 fingerprints were qualified through TTNN's older
        # mcast/gather spelling and must retain that exact schedule.
        rank_program_config.update({
            "family": "2d",
            "transpose_mcast": False,
        })
    else:
        rank_program_config.update({
            "mcast_in0": False,
            "gather_in0": False,
            "untilize_out": False,
        })
    rank_program = {
        "operation": "ttnn.matmul",
        "operand_indices": [0, 2, 5],
        "transpose_a": False,
        "transpose_b": True,
        "compute_config": _compute_config(math_approx_mode=True),
    }
    if physical_rank == 32:
        # Preserve the frozen resident runtime contract.  Wider ranks deliberately use
        # TTNN's shape-aware selector: forcing one N tile per core reproduces
        # rank-32 arithmetic independently for each tile, while the reference
        # R64+ graph selects a different (and observable) reduction schedule.
        rank_program["program_config"] = rank_program_config
    # Match the all-DRAM TTNN selector for both accepted production widths.
    # Blackhole has eleven worker columns, so the canonical 2-D schedule
    # widens each worker block just enough to cover the output N tiles:
    # SmolLM2 18 tiles -> 2, Qwen 28 tiles -> 3.
    base_n_block = math.ceil((hidden_features // 32) / 11)
    base_program = {
        "operation": "ttnn.matmul",
        "operand_indices": [0, 1, 4 if debug_stage == "base" else 6],
        "transpose_a": False,
        "transpose_b": not frozen_weight_transposed,
        # Match TTNN's all-DRAM canonical schedule even though the full owner
        # keeps this result in L1.  Auto-selection sees the L1 destination and
        # otherwise switches to a 1-D reduction with different BF16 bits.
        "program_config": {
            "family": "2d",
            "grid": [11, 10],
            "in0_block_w": 1,
            "out_subblock_h": 1,
            "out_subblock_w": base_n_block,
            "out_block_h": 1,
            "out_block_w": base_n_block,
            "per_core_m": 1,
            "per_core_n": base_n_block,
            "fuse_batch": False,
            "transpose_mcast": False,
        },
        "compute_config": _compute_config(math_approx_mode=False),
    }
    delta_program = {
        "operation": "ttnn.matmul",
        "operand_indices": [
            5, 3, 4 if debug_stage == "delta" else 7
        ],
        "transpose_a": False,
        "transpose_b": True,
        "compute_config": _compute_config(math_approx_mode=False),
    }
    scale_program = {
        "operation": "ttnn.full",
        "operand_indices": [8],
        "fill_value": float(scale),
    }
    multiply_program = {
        "operation": "ttnn.multiply",
        "operand_indices": [7, 8, 4 if debug_stage == "scaled" else 9],
    }
    add_program = {
        "operation": "ttnn.add",
        "operand_indices": [6, 9, 4],
    }
    programs = [rank_program]
    if debug_stage == "base":
        programs.append(base_program)
    elif debug_stage == "delta":
        programs.append(delta_program)
    elif debug_stage == "scaled":
        programs.extend([delta_program, scale_program, multiply_program])
    else:
        programs.extend([
            base_program,
            delta_program,
            scale_program,
            multiply_program,
            add_program,
        ])

    @tt_lang_operation(
        operation_id=f"tt_finetune.native_lora_o_path.{variant}",
        arg_roles=("in",) * 4 + ("out", "out"),
        version_tag=f"native-lora-o-path-{variant}",
    )
    def operation(head, frozen_weight, lora_a, lora_b, output, rank_hidden):
        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(
            head, frozen_weight, lora_a, lora_b, output, rank_hidden
        )
        compiled = compiled_ttl_program(_probe)
        compiled.program_sequence = programs
        base_memory = (
            "clone" if memory_mode in {"all_dram", "base_dram"} else "l1"
        )
        delta_memory = (
            "clone" if memory_mode in {"all_dram", "delta_dram"} else "l1"
        )
        scaled_memory = (
            "clone" if memory_mode in {"all_dram", "scaled_dram"} else "l1"
        )
        compiled.internal_tensors = [
            {"source_operand": 4, "memory": base_memory},
            {"source_operand": 4, "memory": delta_memory},
            {
                "source_operand": 4,
                "memory": "l1",
                "shape": [1, 1],
            },
            {"source_operand": 4, "memory": scaled_memory},
        ]
        return result

    operation.artifact_format = 4
    operation.physical_programs = len(programs)
    operation.public_custom_calls = 1
    operation.compiler_owned_l1_tensors = 4
    operation.safe_default = False
    operation.experimental_backend = "native_lora_o_path_v1"
    operation.debug_stage = debug_stage
    operation.memory_mode = memory_mode
    operation.frozen_weight_transposed = bool(frozen_weight_transposed)
    operation.rank_k_tiles_per_phase = int(rank_k_tiles_per_phase)
    operation.program_sequence = programs
    operation._ttl_program = _probe
    return operation


__all__ = ["build_tt_lang_native_lora_o_path"]
