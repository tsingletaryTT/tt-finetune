"""Canonical TTNN SwiGLU/down sequence behind one TT-Lang owner call.

The sequence deliberately delegates every numerically observable arithmetic
operation to the same TTNN operation family as the safe graph.  Intermediate
activations are compiler-owned interleaved L1 tensors, so they never become
PJRT values or make an intermediate GDDR6 round trip.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import torch

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_program_composition import compiled_ttl_program


ttl = None
tt_lang_operation = None


def _compute_config() -> dict[str, object]:
    return {"math_fidelity": "HiFi4", "fp32_dest_acc_en": True}


def _mcast_1d_rank_program_config() -> dict[str, object]:
    return {
        "family": "1d",
        # The canonical all-DRAM 128xI @ rankxI.T selector assigns one
        # token tile to each of four workers.  Pin that physical reduction
        # schedule even though the artifact-v4 destination is private L1.
        "grid": [4, 1],
        "in0_block_w": 2,
        "out_subblock_h": 1,
        "out_subblock_w": 1,
        "out_block_h": 1,
        "out_block_w": 1,
        "per_core_m": 1,
        "per_core_n": 1,
        "mcast_in0": False,
        "gather_in0": False,
        "untilize_out": False,
        "fuse_batch": False,
    }


def _mcast_2d_down_program_config(hidden_features: int) -> dict[str, object]:
    per_core_n = {576: 2, 896: 3}.get(hidden_features)
    if per_core_n is None:
        raise ValueError(
            f"no canonical down matmul schedule for hidden={hidden_features}"
        )
    return {
        "family": "2d",
        # Four token tiles by ceil(N/per_core_n) column groups is the exact
        # TTNN auto-selected DRAM program geometry.
        "grid": [
            (hidden_features // 32 + per_core_n - 1) // per_core_n,
            4,
        ],
        "in0_block_w": 1,
        "out_subblock_h": 1,
        "out_subblock_w": per_core_n,
        "out_block_h": 1,
        "out_block_w": per_core_n,
        "per_core_m": 1,
        "per_core_n": per_core_n,
        "transpose_mcast": False,
        "fuse_batch": False,
    }


@lru_cache(maxsize=16)
def build_tt_lang_native_swiglu_down(
    *,
    scale: float,
    tokens: int,
    hidden_features: int,
    intermediate_features: int,
    physical_rank: int,
    logical_rank: int | None = None,
) -> Callable:
    """Build an exact one-call TTNN-native SwiGLU/down projection."""

    dimensions = (tokens, hidden_features, intermediate_features, physical_rank)
    if any(value <= 0 or value % 32 for value in dimensions):
        raise ValueError("native SwiGLU/down dimensions must be positive /32")
    if physical_rank > 128:
        raise ValueError("native SwiGLU/down currently supports rank through 128")
    logical_rank_features = (
        physical_rank if logical_rank is None else int(logical_rank)
    )
    if not 1 <= logical_rank_features <= physical_rank:
        raise ValueError("native SwiGLU/down logical rank must fit physical rank")
    # See the paired gate/up owner: compiler-owned tiled L1 values must retain
    # their full physical rank tile until the allocator learns padded sizes.
    rank_features = physical_rank

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("native SwiGLU/down requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid="full")
    def _probe(gate, up, frozen_weight_t, lora_a, lora_b, output):
        source = ttl.make_dataflow_buffer_like(gate, shape=(1, 1), block_count=1)
        target = ttl.make_dataflow_buffer_like(output, shape=(1, 1), block_count=1)

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                tx = ttl.copy(gate[0:1, 0:1], tile)
                tx.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with target.reserve() as result:
                    result.store(tile)

        @ttl.datamovement()
        def write():
            with target.wait() as tile:
                tx = ttl.copy(tile, output[0:1, 0:1])
                tx.wait()

    variant = (
        f"m{tokens}.h{hidden_features}.i{intermediate_features}."
        f"r{physical_rank}.logical{logical_rank_features}."
        f"s{float(scale):.9g}.v12-rank-scaled-auto-delta"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.native_swiglu_down.{variant}",
        arg_roles=("in", "in", "in", "in", "in", "out"),
        version_tag=f"native-swiglu-down-{variant}",
    )
    def operation(gate, up, frozen_weight_t, lora_a, lora_b, output):
        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(gate, up, frozen_weight_t, lora_a, lora_b, output)
        compiled = compiled_ttl_program(_probe)
        compute = _compute_config()
        # External IDs 0..5; private IDs 6..12.
        rank_program = {
            "operation": "ttnn.matmul",
            "operand_indices": [7, 3, 8],
            "transpose_a": False,
            "transpose_b": True,
            "compute_config": compute,
        }
        if physical_rank == 32:
            rank_program["program_config"] = _mcast_1d_rank_program_config()
        delta_program = {
            "operation": "ttnn.matmul",
            "operand_indices": [8, 4, 10],
            "transpose_a": False,
            "transpose_b": True,
            "compute_config": compute,
        }
        if physical_rank == 32:
            delta_program["program_config"] = (
                _mcast_2d_down_program_config(hidden_features)
            )
        compiled.program_sequence = [
            {"operation": "ttnn.silu", "operand_indices": [0, 6]},
            {"operation": "ttnn.multiply", "operand_indices": [6, 1, 7]},
            rank_program,
            delta_program,
            {
                "operation": "ttnn.full",
                "operand_indices": [11],
                "fill_value": float(
                    torch.tensor(scale, dtype=torch.bfloat16).float().item()
                ),
            },
            {"operation": "ttnn.multiply", "operand_indices": [10, 11, 12]},
            {
                "operation": "ttnn.matmul",
                "operand_indices": [7, 2, 9],
                "transpose_a": False,
                "transpose_b": False,
                "compute_config": compute,
                "program_config": _mcast_2d_down_program_config(
                    hidden_features
                ),
            },
            {"operation": "ttnn.add", "operand_indices": [9, 12, 5]},
        ]
        specs = (
            ([tokens, intermediate_features], "BFloat16"),  # silu
            ([tokens, intermediate_features], "BFloat16"),  # product
            ([tokens, rank_features], "BFloat16"),          # H
            ([tokens, hidden_features], "BFloat16"),        # base
            ([tokens, hidden_features], "BFloat16"),        # delta
            ([1, 1], "BFloat16"),                           # scale
            ([tokens, hidden_features], "BFloat16"),        # scaled
        )
        compiled.internal_tensors = [
            {
                "source_operand": 0,
                "memory": "l1",
                "shape": shape,
                "data_format": data_format,
                "layout": "tile",
            }
            for shape, data_format in specs
        ]
        return result

    operation.artifact_format = 4
    operation.physical_programs = 8
    operation.public_custom_calls = 1
    operation.compiler_owned_l1_tensors = 7
    operation.internal_tensor_specs = [
        {
            "source_operand": 0,
            "memory": "l1",
            "shape": shape,
            "data_format": data_format,
            "layout": "tile",
        }
        for shape, data_format in (
            ([tokens, intermediate_features], "BFloat16"),
            ([tokens, intermediate_features], "BFloat16"),
            ([tokens, rank_features], "BFloat16"),
            ([tokens, hidden_features], "BFloat16"),
            ([tokens, hidden_features], "BFloat16"),
            ([1, 1], "BFloat16"),
            ([tokens, hidden_features], "BFloat16"),
        )
    ]
    operation.intermediate_dram_bytes = 0
    operation.safe_default = False
    operation.experimental_backend = "native_swiglu_down_ttnn_v1"
    operation._ttl_program = _probe
    return operation


__all__ = ["build_tt_lang_native_swiglu_down"]
