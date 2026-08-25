"""Canonical TTNN gate/up LoRA projections behind one public owner call.

The two projection results are independent compiler-owned L1 tensors when
this sequence is embedded in the resident layer.  TTNN owns the observable
matmul/add schedules, while the enclosing artifact owns their lifetime.
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


def _rank_program_config() -> dict[str, object]:
    # Canonical TTNN auto-selection for [128,K] @ [8/16,K]^T uses one
    # token-row tile per each of four cores and streams K two tiles at a time.
    return {
        "family": "reuse",
        "grid": [4, 1],
        "in0_block_w": 2,
        "out_subblock_h": 1,
        "out_subblock_w": 1,
        "out_block_h": 1,
        "out_block_w": 1,
        "per_core_m": 1,
        "per_core_n": 1,
        "fuse_batch": False,
    }


def _wide_program_config(*, in0_block_w: int) -> dict[str, object]:
    # Canonical wide-N projection: one output-column tile per core, all four
    # token tiles resident on that core, with input multicast across N.
    return {
        "family": "1d",
        "grid": [11, 10],
        "in0_block_w": in0_block_w,
        "out_subblock_h": 4,
        "out_subblock_w": 1,
        "out_block_h": 4,
        "out_block_w": 1,
        "per_core_m": 4,
        "per_core_n": 1,
        "mcast_in0": True,
        "gather_in0": False,
        "untilize_out": False,
        "fuse_batch": False,
    }


@lru_cache(maxsize=16)
def build_tt_lang_native_paired_gate_up(
    *,
    scale: float,
    tokens: int,
    hidden_features: int,
    intermediate_features: int,
    physical_rank: int,
    logical_rank: int | None = None,
) -> Callable:
    dimensions = (tokens, hidden_features, intermediate_features, physical_rank)
    if any(value <= 0 or value % 32 for value in dimensions):
        raise ValueError("native gate/up dimensions must be positive /32")
    if physical_rank > 128:
        raise ValueError("native gate/up currently supports rank through 128")
    logical_rank_features = (
        physical_rank if logical_rank is None else int(logical_rank)
    )
    if not 1 <= logical_rank_features <= physical_rank:
        raise ValueError("native gate/up logical rank must fit physical rank")
    # TTNN tensors carry a logical shape, but artifact-v4 compiler-owned L1
    # allocations are currently sized from that logical element count.  A
    # [M,8] tiled tensor therefore reserves only one quarter of its required
    # physical tile storage and corrupts the following private tensors.  Keep
    # the resident ABI physically padded to one full rank tile.  Zero-padded
    # K lanes are numerically exact to TTNN's logical-rank matmul schedule.
    rank_features = physical_rank

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("native gate/up requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid="full")
    def _probe(
        inputs,
        gate_weight_t, gate_a, gate_b,
        up_weight_t, up_a, up_b,
        gate_output, up_output,
    ):
        source = ttl.make_dataflow_buffer_like(inputs, shape=(1, 1), block_count=1)
        target = ttl.make_dataflow_buffer_like(
            gate_output, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                tx = ttl.copy(inputs[0:1, 0:1], tile)
                tx.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with target.reserve() as result:
                    result.store(tile)

        @ttl.datamovement()
        def write():
            with target.wait() as tile:
                tx = ttl.copy(tile, gate_output[0:1, 0:1])
                tx.wait()

    variant = (
        f"m{tokens}.h{hidden_features}.i{intermediate_features}."
        f"r{physical_rank}.logical{logical_rank_features}."
        f"s{float(scale):.9g}.v5-rank-scaled"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.native_paired_gate_up.{variant}",
        arg_roles=("in",) * 7 + ("out", "out"),
        version_tag=f"native-paired-gate-up-{variant}",
    )
    def operation(
        inputs,
        gate_weight_t, gate_a, gate_b,
        up_weight_t, up_a, up_b,
        gate_output, up_output,
    ):
        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(
            inputs,
            gate_weight_t, gate_a, gate_b,
            up_weight_t, up_a, up_b,
            gate_output, up_output,
        )
        compiled = compiled_ttl_program(_probe)
        compute = _compute_config()
        scale_value = float(
            torch.tensor(scale, dtype=torch.bfloat16).float().item()
        )
        # Public IDs 0..8; private IDs 9..17.
        compiled.program_sequence = [
            {
                "operation": "ttnn.matmul",
                "operand_indices": [0, 2, 9],
                "transpose_a": False,
                "transpose_b": True,
                "compute_config": compute,
            },
            {
                "operation": "ttnn.matmul",
                "operand_indices": [9, 3, 11],
                "transpose_a": False,
                "transpose_b": True,
                "compute_config": compute,
            },
            {
                "operation": "ttnn.matmul",
                "operand_indices": [0, 1, 10],
                "transpose_a": False,
                "transpose_b": False,
                "compute_config": compute,
            },
            {"operation": "ttnn.full", "operand_indices": [13], "fill_value": scale_value},
            {"operation": "ttnn.multiply", "operand_indices": [11, 13, 14]},
            {"operation": "ttnn.add", "operand_indices": [10, 14, 7]},
            {
                "operation": "ttnn.matmul",
                "operand_indices": [0, 5, 15],
                "transpose_a": False,
                "transpose_b": True,
                "compute_config": compute,
            },
            {
                "operation": "ttnn.matmul",
                "operand_indices": [15, 6, 17],
                "transpose_a": False,
                "transpose_b": True,
                "compute_config": compute,
            },
            {
                "operation": "ttnn.matmul",
                "operand_indices": [0, 4, 16],
                "transpose_a": False,
                "transpose_b": False,
                "compute_config": compute,
            },
            {"operation": "ttnn.multiply", "operand_indices": [17, 13, 12]},
            {"operation": "ttnn.add", "operand_indices": [16, 12, 8]},
        ]
        specs = (
            [tokens, rank_features],
            [tokens, intermediate_features],
            [tokens, intermediate_features],
            [tokens, intermediate_features],
            [1, 1],
            [tokens, intermediate_features],
            [tokens, rank_features],
            [tokens, intermediate_features],
            [tokens, intermediate_features],
        )
        compiled.internal_tensors = [
            {
                "source_operand": 0,
                "memory": "l1",
                "shape": shape,
                "data_format": "BFloat16",
                "layout": "tile",
            }
            for shape in specs
        ]
        return result

    operation.artifact_format = 4
    operation.physical_programs = 11
    operation.public_custom_calls = 1
    operation.compiler_owned_l1_tensors = 9
    operation.internal_tensor_specs = [
        {
            "source_operand": 0,
            "memory": "l1",
            "shape": shape,
            "data_format": "BFloat16",
            "layout": "tile",
        }
        for shape in (
            [tokens, rank_features],
            [tokens, intermediate_features],
            [tokens, intermediate_features],
            [tokens, intermediate_features],
            [1, 1],
            [tokens, intermediate_features],
            [tokens, rank_features],
            [tokens, intermediate_features],
            [tokens, intermediate_features],
        )
    ]
    operation.intermediate_dram_bytes = 0
    operation.safe_default = False
    operation.experimental_backend = "native_paired_gate_up_ttnn_v1"
    operation._ttl_program = _probe
    return operation


__all__ = ["build_tt_lang_native_paired_gate_up"]
