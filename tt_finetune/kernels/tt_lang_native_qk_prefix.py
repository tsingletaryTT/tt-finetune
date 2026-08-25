"""Canonical native TTNN QK prefix for a resident attention owner."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import torch

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_owner_aligned_copy import build_tt_lang_owner_aligned_copy
from .tt_lang_program_composition import (
    compile_serialized_child,
    compiled_ttl_program,
    make_serialized_gqa_head_expand_copy_bitwise,
    remap_serialized_child,
    serialized_child_tensor_ids,
)


ttl = None
tt_lang_operation = None


def _placeholder_like(tensor, *, shape: tuple[int, ...], dtype):
    try:
        return type(tensor)(
            shape=shape,
            dtype=dtype,
            layout=tensor.layout,
            memory_config=tensor.memory_config(),
            device=tensor.device(),
        )
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            "native QK prefix requires shape-only compile operands"
        ) from exc


def _remap_all(program: dict, mapping: dict[int, int]) -> dict:
    used = serialized_child_tensor_ids(program)
    return remap_serialized_child(
        program, {tensor: mapping[tensor] for tensor in used}
    )


def _compute_config() -> dict[str, object]:
    return {
        "math_fidelity": "HiFi4",
        "math_approx_mode": True,
        "fp32_dest_acc_en": True,
        "packer_l1_acc": False,
        "dst_full_sync_en": False,
    }


@lru_cache(maxsize=8)
def build_tt_lang_native_qk_prefix(
    *,
    sequence_length: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> Callable:
    """Typecast Q/K, expand GQA K in L1, then run native FP32 QK.

    Public qualification ABI:
    ``BF16 Q, compact BF16 K -> FP32 raw score``.
    The two casts and expanded key are compiler-owned L1 tensors.  The final
    matmul pins the canonical all-DRAM QK schedule.  TTNN otherwise selects
    the single-core ``MatmulMultiCoreProgramConfig`` for L1 operands, whereas
    the untouched PJRT graph uses a 2-D multicast schedule.  The two programs
    reduce FP32 tiles in a different order and are therefore not bit-exact.
    """

    if sequence_length <= 0 or sequence_length % 32:
        raise ValueError("sequence length must be positive and tile aligned")
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by positive KV heads")
    if head_dim <= 0 or head_dim % 32:
        raise ValueError("head dimension must be positive and tile aligned")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError("native QK prefix requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    page_copy = build_tt_lang_owner_aligned_copy(
        rows=sequence_length,
        columns=query_heads * head_dim,
    )

    @ttl.operation(grid="full")
    def _probe(query, compact_key, raw_score):
        source = ttl.make_dataflow_buffer_like(
            query, shape=(1, 1), block_count=1
        )
        target = ttl.make_dataflow_buffer_like(
            raw_score, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                transaction = ttl.copy(query[0, 0, 0:1, 0:1], tile)
                transaction.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with target.reserve() as result:
                    result.store(ttl.math.typecast(tile, torch.float32))

        @ttl.datamovement()
        def write():
            with target.wait() as tile:
                transaction = ttl.copy(
                    tile, raw_score[0, 0, 0:1, 0:1]
                )
                transaction.wait()

    variant = (
        f"s{sequence_length}.q{query_heads}.kv{kv_heads}.d{head_dim}.v2"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.native_qk_prefix.{variant}",
        arg_roles=("in", "in", "out"),
        version_tag=f"native-qk-prefix-{variant}",
    )
    def operation(query, compact_key, raw_score):
        query_shape = (1, query_heads, sequence_length, head_dim)
        compact_key_shape = (1, kv_heads, sequence_length, head_dim)
        expanded_key_shape = query_shape
        query_fp32 = _placeholder_like(
            raw_score, shape=query_shape, dtype=torch.float32
        )
        compact_key_fp32 = _placeholder_like(
            raw_score, shape=compact_key_shape, dtype=torch.float32
        )
        expanded_key_source = _placeholder_like(
            raw_score,
            shape=(sequence_length, query_heads * head_dim),
            dtype=torch.float32,
        )
        expanded_key_destination = _placeholder_like(
            raw_score,
            shape=(sequence_length, query_heads * head_dim),
            dtype=torch.float32,
        )

        expand_child = compile_serialized_child(
            page_copy,
            (expanded_key_source, expanded_key_destination),
        )
        expand_child = make_serialized_gqa_head_expand_copy_bitwise(
            expand_child,
            sequence_tiles=sequence_length // 32,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dimension_tiles=head_dim // 32,
            tile_bytes=4096,
        )
        expand_child = _remap_all(expand_child, {0: 4, 1: 5})

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(query, compact_key, raw_score)
        compiled = compiled_ttl_program(_probe)
        compiled.program_sequence = [
            {"operation": "ttnn.typecast", "operand_indices": [0, 3]},
            {"operation": "ttnn.typecast", "operand_indices": [1, 4]},
            expand_child,
            {
                "operation": "ttnn.matmul",
                "operand_indices": [3, 5, 2],
                "transpose_a": False,
                "transpose_b": True,
                "program_config": {
                    "family": "2d",
                    "grid": [11, 10],
                    "in0_block_w": 1,
                    "out_subblock_h": 1,
                    "out_subblock_w": 1,
                    "out_block_h": 1,
                    "out_block_w": 1,
                    "per_core_m": 1,
                    "per_core_n": 1,
                    "transpose_mcast": False,
                    "fuse_batch": False,
                },
                "compute_config": _compute_config(),
            },
        ]
        compiled.internal_tensors = [
            {
                "source_operand": 2,
                "memory": "l1",
                "shape": list(query_shape),
                "data_format": "Float32",
                "layout": "tile",
            },
            {
                "source_operand": 2,
                "memory": "l1",
                "shape": list(compact_key_shape),
                "data_format": "Float32",
                "layout": "tile",
            },
            {
                "source_operand": 2,
                "memory": "l1",
                "shape": list(expanded_key_shape),
                "data_format": "Float32",
                "layout": "tile",
            },
        ]
        return result

    operation.artifact_format = 4
    operation.physical_programs = 4
    operation.public_custom_calls = 1
    operation.compiler_owned_l1_tensors = 3
    operation.intermediate_dram_bytes = 0
    operation.output_shape = (
        1, query_heads, sequence_length, sequence_length
    )
    operation.safe_default = False
    operation.experimental_backend = "native_qk_prefix_v2"
    operation._ttl_program = _probe
    return operation


__all__ = ["build_tt_lang_native_qk_prefix"]
