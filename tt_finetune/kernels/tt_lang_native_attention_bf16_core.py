"""Canonical resident BF16 causal-GQA attention for the safe graph.

Every arithmetic stage is emitted as the same TTNN operation used by the
untouched Torch/XLA graph. Q/K/V expansion is NoC-only. Score, FP32 softmax
workspace and BF16 probability remain private within one artifact-v4 call.
"""

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


def _placeholder_like(tensor, *, shape: tuple[int, ...], dtype=torch.bfloat16):
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
            "native BF16 attention core requires shape-only operands"
        ) from exc


def _remap_all(program: dict, mapping: dict[int, int]) -> dict:
    used = serialized_child_tensor_ids(program)
    return remap_serialized_child(
        program, {tensor: mapping[tensor] for tensor in used}
    )


def _compute_config() -> dict[str, object]:
    # Match the safe graph exactly: optional tri-state knobs remain absent.
    return {"math_fidelity": "HiFi4", "fp32_dest_acc_en": True}


def _softmax_compute_config() -> dict[str, object]:
    # Artifact-v4's native softmax descriptor requires concrete values.  These
    # are TTNN's resolution of the safe graph's otherwise absent tri-states.
    return {
        "math_fidelity": "HiFi4",
        "math_approx_mode": True,
        "fp32_dest_acc_en": True,
        "packer_l1_acc": False,
        "dst_full_sync_en": False,
    }


def _canonical_pv_program_config() -> dict[str, object]:
    # L1 changes auto-selection. Pin the all-DRAM safe-graph reduction tree,
    # captured from TTNN's Blackhole program factory for [1,9,128,128] @
    # [1,9,128,64].
    return {
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
    }


def _canonical_qk_program_config() -> dict[str, object]:
    return {
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
    }


@lru_cache(maxsize=16)
def build_tt_lang_native_attention_bf16_core_compact_private(
    *,
    sequence_length: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float,
    private_memory: str = "l1",
    pin_qk_schedule: bool = True,
    pin_pv_schedule: bool = True,
    l1_tensor_indices: tuple[int, ...] = (),
) -> Callable:
    """Build one BF16 attention call accepting compact GQA K/V."""

    if sequence_length <= 0 or sequence_length % 32:
        raise ValueError("sequence length must be positive and tile aligned")
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by positive KV heads")
    if head_dim <= 0 or head_dim % 32:
        raise ValueError("head dimension must be positive and tile aligned")
    if private_memory not in {"l1", "clone", "hybrid"}:
        raise ValueError("private_memory must be 'l1', 'clone' or 'hybrid'")
    if any(index < 0 or index >= 9 for index in l1_tensor_indices):
        raise ValueError("l1_tensor_indices must contain values in [0, 8]")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("native BF16 attention core requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    expanded_copy = build_tt_lang_owner_aligned_copy(
        rows=sequence_length, columns=query_heads * head_dim
    )

    @ttl.operation(grid="full")
    def _probe(query, compact_key, mask, compact_value, head):
        source = ttl.make_dataflow_buffer_like(
            compact_value, shape=(1, 1), block_count=1
        )
        output = ttl.make_dataflow_buffer_like(head, shape=(1, 1), block_count=1)

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                tx = ttl.copy(compact_value[0, 0, 0:1, 0:1], tile)
                tx.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with output.reserve() as target:
                    target.store(tile)

        @ttl.datamovement()
        def write():
            with output.wait() as tile:
                tx = ttl.copy(tile, head[0, 0, 0:1, 0:1])
                tx.wait()

    variant = (
        f"s{sequence_length}.q{query_heads}.kv{kv_heads}.d{head_dim}."
        f"scale{float(scale):.9g}.m{private_memory}."
        f"l1{'-'.join(map(str, l1_tensor_indices))}."
        f"qkpin{int(pin_qk_schedule)}.pvpin{int(pin_pv_schedule)}.v12"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.native_attention_bf16_core.{variant}",
        arg_roles=("in", "in", "in", "in", "out"),
        version_tag=f"native-attention-bf16-core-{variant}",
    )
    def operation(query, compact_key, mask, compact_value, head):
        expanded_shape = (1, query_heads, sequence_length, head_dim)
        score_shape = (1, query_heads, sequence_length, sequence_length)
        source_placeholder = _placeholder_like(
            query, shape=(sequence_length, query_heads * head_dim)
        )
        destination_placeholder = _placeholder_like(
            query, shape=(sequence_length, query_heads * head_dim)
        )
        key_expand = compile_serialized_child(
            expanded_copy, (source_placeholder, destination_placeholder)
        )
        key_expand = make_serialized_gqa_head_expand_copy_bitwise(
            key_expand,
            sequence_tiles=sequence_length // 32,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dimension_tiles=head_dim // 32,
        )
        key_expand = _remap_all(key_expand, {0: 1, 1: 5})
        value_expand = compile_serialized_child(
            expanded_copy, (source_placeholder, destination_placeholder)
        )
        value_expand = make_serialized_gqa_head_expand_copy_bitwise(
            value_expand,
            sequence_tiles=sequence_length // 32,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dimension_tiles=head_dim // 32,
        )
        value_expand = _remap_all(value_expand, {0: 3, 1: 6})

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(query, compact_key, mask, compact_value, head)
        compiled = compiled_ttl_program(_probe)
        config = _compute_config()
        pv_descriptor = {
            "operation": "ttnn.matmul",
            "operand_indices": [13, 6, 4],
            "transpose_a": False,
            "transpose_b": False,
            "compute_config": config,
        }
        if pin_pv_schedule:
            pv_descriptor["program_config"] = _canonical_pv_program_config()
        qk_descriptor = {
            "operation": "ttnn.matmul",
            "operand_indices": [0, 5, 7],
            "transpose_a": False,
            "transpose_b": True,
            "compute_config": config,
        }
        if pin_qk_schedule:
            qk_descriptor["program_config"] = _canonical_qk_program_config()
        compiled.program_sequence = [
            key_expand,
            value_expand,
            qk_descriptor,
            {"operation": "ttnn.full", "operand_indices": [8],
             "fill_value": float(scale)},
            {"operation": "ttnn.multiply", "operand_indices": [7, 8, 9]},
            {"operation": "ttnn.add", "operand_indices": [9, 2, 10]},
            {"operation": "ttnn.typecast", "operand_indices": [10, 11]},
            {
                "operation": "ttnn.softmax",
                "operand_indices": [11, 12],
                "dimension": -1,
                "numeric_stable": True,
                "compute_config": _softmax_compute_config(),
            },
            {"operation": "ttnn.typecast", "operand_indices": [12, 13]},
            pv_descriptor,
        ]
        formats = (
            "BFloat16", "BFloat16", "BFloat16", "BFloat16", "BFloat16",
            "BFloat16", "Float32", "Float32", "BFloat16",
        )
        shapes = (
            expanded_shape, expanded_shape, score_shape, (1, 1, 1, 1),
            score_shape, score_shape, score_shape, score_shape, score_shape,
        )
        def memory_for(index: int) -> str:
            if private_memory == "hybrid":
                return "l1" if index in l1_tensor_indices else "clone"
            return private_memory

        compiled.internal_tensors = [
            {
                "source_operand": 0,
                "memory": memory_for(index),
                "shape": list(shape),
                "data_format": data_format,
                "layout": "tile",
            }
            for index, (shape, data_format) in enumerate(zip(shapes, formats))
        ]
        return result

    operation.artifact_format = 4
    operation.physical_programs = 10
    operation.public_custom_calls = 1
    operation.compiler_owned_l1_tensors = (
        9 if private_memory == "l1" else len(l1_tensor_indices)
        if private_memory == "hybrid" else 0
    )
    operation.intermediate_dram_bytes = 0 if private_memory == "l1" else -1
    operation.published_probability = False
    operation.safe_default = False
    operation.experimental_backend = "native_attention_bf16_ttnn_v2"
    operation.private_memory = private_memory
    operation.pin_qk_schedule = bool(pin_qk_schedule)
    operation.pin_pv_schedule = bool(pin_pv_schedule)
    operation.l1_tensor_indices = tuple(l1_tensor_indices)
    operation._ttl_program = _probe
    return operation


__all__ = ["build_tt_lang_native_attention_bf16_core_compact_private"]
