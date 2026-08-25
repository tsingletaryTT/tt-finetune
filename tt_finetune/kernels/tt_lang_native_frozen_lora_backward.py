"""Shape-parametric TTNN LoRA projection backward behind one public call.

This is the scalable native geometry counterpart to the hand-written rank-32 exact
projection kernels.  It deliberately expresses the same frozen Linear + LoRA
backward graph as native TTNN children while the enclosing artifact owns all
intermediate tensors.  The implementation is rank-tile and token-count
parametric; qualification points are selected by callers, not by this file.

For ``Y = linear(X, W) + scale * linear(linear(X, A), B)`` it returns only
``dX``, ``dA`` and ``dB``.  ``H``, scaled ``dY``, ``dH`` and both dX branches
remain compiler-private L1 tensors.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import torch

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_program_composition import compiled_ttl_program


TILE_SIZE = 32
ttl = None
tt_lang_operation = None


def _compute_config() -> dict[str, object]:
    return {
        "math_fidelity": "HiFi4",
        "fp32_dest_acc_en": True,
    }


def _mcast_2d_config(
    *,
    out_subblock_h: int,
    out_block_h: int,
    per_core_m: int,
    out_subblock_w: int = 1,
    out_block_w: int = 1,
    per_core_n: int = 1,
) -> dict[str, object]:
    """Canonical all-DRAM TTNN schedule, retained for an L1 destination."""

    return {
        "family": "2d",
        "grid": [11, 10],
        "in0_block_w": 1,
        "out_subblock_h": out_subblock_h,
        "out_subblock_w": out_subblock_w,
        "out_block_h": out_block_h,
        "out_block_w": out_block_w,
        "per_core_m": per_core_m,
        "per_core_n": per_core_n,
        "transpose_mcast": False,
        "fuse_batch": False,
    }


def _db_mcast_1d_config(*, rank_tiles: int) -> dict[str, object]:
    """Canonical dB raw-gradient schedule for R rank tiles."""

    return {
        "family": "1d",
        "grid": [11, 10],
        "in0_block_w": 2,
        "out_subblock_h": rank_tiles,
        "out_subblock_w": 1,
        "out_block_h": rank_tiles,
        "out_block_w": 1,
        "per_core_m": rank_tiles,
        "per_core_n": 1,
        "mcast_in0": True,
        "gather_in0": False,
        "untilize_out": False,
        "fuse_batch": False,
    }


def _da_program_config(*, rank_tiles: int) -> dict[str, object]:
    """Canonical dA schedule selected from the physical rank tile count."""

    if rank_tiles <= 2:
        return {
            "family": "1d",
            "grid": [11, 10],
            "in0_block_w": 2,
            "out_subblock_h": 1,
            "out_subblock_w": rank_tiles,
            "out_block_h": 1,
            "out_block_w": rank_tiles,
            "per_core_m": 1,
            "per_core_n": rank_tiles,
            "mcast_in0": False,
            "gather_in0": False,
            "untilize_out": False,
            "fuse_batch": False,
        }
    return _mcast_2d_config(
        out_subblock_h=2, out_block_h=2, per_core_m=2
    )


def _rank_projection_config(*, rank_tiles: int) -> dict[str, object]:
    if rank_tiles == 1:
        return {
            "family": "1d",
            "grid": [11, 10],
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
    return _mcast_2d_config(
        out_subblock_h=1, out_block_h=1, per_core_m=1
    )


@lru_cache(maxsize=64)
def build_tt_lang_native_frozen_lora_backward(
    *,
    scale: float,
    tokens: int,
    input_features: int,
    output_features: int,
    physical_rank: int,
    logical_rank: int | None = None,
    private_memory: str = "l1",
    include_base_gradient: bool = True,
) -> Callable:
    """Build one three-output native frozen-Linear + LoRA backward owner."""

    dimensions = (tokens, input_features, output_features, physical_rank)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError(
            "native frozen-LoRA backward dimensions must be positive /32"
        )
    logical = physical_rank if logical_rank is None else int(logical_rank)
    if not 1 <= logical <= physical_rank:
        raise ValueError("logical rank must fit the physical rank")
    if private_memory not in {"l1", "clone"}:
        raise ValueError("private memory must be 'l1' or 'clone'")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError(
                "native frozen-LoRA backward requires TT-XLA"
            ) from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid="full")
    def _probe(
        inputs,
        output_gradient,
        weight,
        lora_a,
        lora_b,
        input_gradient_output,
        lora_a_gradient_output,
        lora_b_gradient_output,
    ):
        source = ttl.make_dataflow_buffer_like(
            inputs, shape=(1, 1), block_count=1
        )
        target = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=1
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
                tx = ttl.copy(tile, input_gradient_output[0:1, 0:1])
                tx.wait()

    scale_value = float(
        torch.tensor(scale, dtype=torch.bfloat16).float().item()
    )
    variant = (
        f"m{tokens}.k{input_features}.n{output_features}."
        f"r{physical_rank}.logical{logical}.s{scale_value:.9g}."
        f"mem{private_memory}.base{int(include_base_gradient)}.v9"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.native_frozen_lora_backward.{variant}",
        arg_roles=("in",) * 5 + ("out",) * 3,
        version_tag=f"native-frozen-lora-backward-{variant}",
    )
    def operation(
        inputs,
        output_gradient,
        weight,
        lora_a,
        lora_b,
        input_gradient_output,
        lora_a_gradient_output,
        lora_b_gradient_output,
    ):
        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(
            inputs,
            output_gradient,
            weight,
            lora_a,
            lora_b,
            input_gradient_output,
            lora_a_gradient_output,
            lora_b_gradient_output,
        )
        compiled = compiled_ttl_program(_probe)
        compute = _compute_config()
        def exact_matmul(
            operands: list[int], *, transpose_a: bool = False,
            transpose_b: bool = False,
        ) -> dict[str, object]:
            # Internal results stay in L1, but the runtime derives the
            # numerical program from the canonical all-DRAM contract.  This
            # keeps reduction order parametric in M/K/N/R rather than
            # hard-coding qualification shapes in this planner.
            return {
                "operation": "ttnn.matmul",
                "operand_indices": operands,
                "transpose_a": transpose_a,
                "transpose_b": transpose_b,
                "compute_config": compute,
                "reference_dram_schedule": True,
            }
        # Public tensor IDs 0..7.  The full projection owner uses private IDs
        # 8..15; the LoRA-branch owner uses 8..13.
        #
        # Keep the reference operation order and output orientation.  Scaling
        # dY is an observable BF16 boundary; dA/dB deliberately compute the
        # canonical raw matrices and then use TTNN's physical permute.
        if include_base_gradient:
            compiled.program_sequence = [
                exact_matmul([0, 3, 8], transpose_b=True),
                {
                    "operation": "ttnn.full",
                    "operand_indices": [9],
                    "fill_value": scale_value,
                },
                {
                    "operation": "ttnn.multiply",
                    "operand_indices": [1, 9, 10],
                },
                exact_matmul([8, 10, 14], transpose_a=True),
                {
                    "operation": "ttnn.permute",
                    "operand_indices": [14, 7],
                    "permutation": [1, 0],
                },
                exact_matmul([10, 4, 11]),
                exact_matmul([0, 11, 15], transpose_a=True),
                exact_matmul([1, 2, 12]),
                exact_matmul([11, 3, 13]),
                {"operation": "ttnn.add", "operand_indices": [12, 13, 5]},
                {
                    "operation": "ttnn.permute",
                    "operand_indices": [15, 6],
                    "permutation": [1, 0],
                },
            ]
        else:
            # Keep the base Linear outside the opaque autograd boundary.  Its
            # dX and the LoRA dX then meet at the same outer AddBackward node
            # as ordinary PEFT, preserving cross-projection accumulation
            # order while this one call owns every LoRA-branch intermediate.
            # Operand 2 is the forward-saved rank-space activation in this
            # variant (the full owner uses it as the frozen base weight).
            compiled.program_sequence = [
                {
                    "operation": "ttnn.full",
                    "operand_indices": [8],
                    "fill_value": scale_value,
                },
                {
                    "operation": "ttnn.multiply",
                    "operand_indices": [1, 8, 9],
                },
                exact_matmul([2, 9, 11], transpose_a=True),
                {
                    "operation": "ttnn.permute",
                    "operand_indices": [11, 7],
                    "permutation": [1, 0],
                },
                exact_matmul([9, 4, 10]),
                exact_matmul([0, 10, 12], transpose_a=True),
                {
                    "operation": "ttnn.permute",
                    "operand_indices": [12, 6],
                    "permutation": [1, 0],
                },
                exact_matmul([10, 3, 5]),
            ]
        # Preserve the logical rank in every observable reduction.  TTNN's
        # tiled layout still rounds the backing storage to `physical_rank`,
        # but exposing that padding as a logical K dimension changes the dX
        # reduction tree for tail ranks (for example R73 -> R96) and is not
        # bit-exact with ordinary PEFT autograd.
        specs = (
            (
                [tokens, logical],
                [1, 1],
                [tokens, output_features],
                [tokens, logical],
                [tokens, input_features],
                [tokens, input_features],
                [logical, output_features],
                [input_features, logical],
            )
            if include_base_gradient
            else (
                [1, 1],
                [tokens, output_features],
                [tokens, logical],
                [logical, output_features],
                [input_features, logical],
            )
        )
        compiled.internal_tensors = [
            {
                "source_operand": 0,
                "memory": private_memory,
                "shape": shape,
                "data_format": "BFloat16",
                "layout": "tile",
            }
            for shape in specs
        ]
        return result

    operation.artifact_format = 4
    operation.physical_programs = 11 if include_base_gradient else 8
    operation.public_custom_calls = 1
    operation.compiler_owned_tensors = 8 if include_base_gradient else 5
    operation.intermediate_dram_bytes = 0 if private_memory == "l1" else None
    operation.safe_default = False
    operation.experimental_backend = "native_frozen_lora_backward_ttnn_v7"
    operation._ttl_program = _probe
    return operation


__all__ = ["build_tt_lang_native_frozen_lora_backward"]
