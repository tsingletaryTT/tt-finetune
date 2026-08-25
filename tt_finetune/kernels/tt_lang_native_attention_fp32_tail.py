"""Canonical FP32 attention tail nested under one artifact-v4 owner."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_owner_aligned_copy import build_tt_lang_owner_aligned_copy
from .tt_lang_program_composition import (
    compile_serialized_child,
    compile_serialized_sequence_children,
    compiled_ttl_program,
    make_serialized_gqa_head_expand_copy_bitwise,
    remap_serialized_child,
    serialized_child_tensor_ids,
)


ttl = None
tt_lang_operation = None


def _compile_placeholder_like(tensor, *, shape: tuple[int, ...]):
    try:
        return type(tensor)(
            shape=shape,
            dtype=tensor.dtype,
            layout=tensor.layout,
            memory_config=tensor.memory_config(),
            device=tensor.device(),
        )
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            "native attention composition requires shape-only compile operands"
        ) from exc


def _remap_all(program: dict, mapping: dict[int, int]) -> dict:
    used = serialized_child_tensor_ids(program)
    return remap_serialized_child(
        program, {tensor: mapping[tensor] for tensor in used}
    )


def _compute_config() -> dict[str, object]:
    return {
        "math_fidelity": "HiFi4",
        # The canonical lowered graph leaves this tri-state knob unset.  TTNN's
        # Blackhole softmax factory resolves that to the approximate SFPU
        # schedule; forcing precise mode changes the reduction/exp contract.
        "math_approx_mode": True,
        "fp32_dest_acc_en": True,
        "packer_l1_acc": False,
        "dst_full_sync_en": False,
    }


@lru_cache(maxsize=8)
def build_tt_lang_native_attention_fp32_tail(
    *,
    sequence_length: int,
    query_heads: int,
    head_dim: int,
    scale: float,
    value_memory: str = "l1",
    head_memory: str = "l1",
) -> Callable:
    """Build ``softmax(raw_qk * scale + mask) @ V`` in canonical FP32.

    The public qualification ABI accepts an already exact raw QK tensor and
    an already GQA-expanded value tensor.  It deliberately publishes the
    probability as a second output so exactness can be checked without a
    compiler dump.  Use :func:`build_tt_lang_native_attention_fp32_tail_private`
    for the actual resident contract, where probability never leaves L1.
    """

    if sequence_length <= 0 or sequence_length % 32:
        raise ValueError("sequence length must be positive and tile aligned")
    if query_heads <= 0:
        raise ValueError("query heads must be positive")
    if head_dim <= 0 or head_dim % 32:
        raise ValueError("head dimension must be positive and tile aligned")
    if value_memory not in ("clone", "l1"):
        raise ValueError("value_memory must be 'clone' or 'l1'")
    if head_memory not in ("clone", "l1"):
        raise ValueError("head_memory must be 'clone' or 'l1'")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError("native attention FP32 tail requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid="full")
    def _probe(raw_score, mask, expanded_value, probability, head):
        source = ttl.make_dataflow_buffer_like(
            raw_score, shape=(1, 1), block_count=1
        )
        output = ttl.make_dataflow_buffer_like(
            probability, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                transaction = ttl.copy(raw_score[0, 0, 0:1, 0:1], tile)
                transaction.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with output.reserve() as target:
                    target.store(tile)

        @ttl.datamovement()
        def write():
            with output.wait() as tile:
                transaction = ttl.copy(tile, probability[0, 0, 0:1, 0:1])
                transaction.wait()

    variant = (
        f"s{sequence_length}.q{query_heads}.d{head_dim}."
        f"scale{float(scale):.9g}.vm{value_memory}.hm{head_memory}.v2"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.native_attention_fp32_tail.{variant}",
        arg_roles=("in", "in", "in", "out", "out"),
        version_tag=f"native-attention-fp32-tail-{variant}",
    )
    def operation(raw_score, mask, expanded_value, probability, head):
        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(raw_score, mask, expanded_value, probability, head)
        compiled = compiled_ttl_program(_probe)
        config = _compute_config()
        compiled.program_sequence = [
            {
                "operation": "ttnn.full",
                "operand_indices": [5],
                "fill_value": float(scale),
            },
            {
                "operation": "ttnn.multiply",
                "operand_indices": [0, 5, 6],
            },
            {
                "operation": "ttnn.typecast",
                "operand_indices": [1, 7],
            },
            {
                "operation": "ttnn.add",
                "operand_indices": [6, 7, 8],
            },
            {
                "operation": "ttnn.softmax",
                "operand_indices": [8, 3],
                "dimension": -1,
                "numeric_stable": True,
                "compute_config": config,
            },
            {
                "operation": "ttnn.typecast",
                "operand_indices": [2, 9],
            },
            {
                "operation": "ttnn.matmul",
                "operand_indices": [3, 9, 10],
                "transpose_a": False,
                "transpose_b": False,
                # Pin the canonical batched-reuse reduction.  Letting TTNN
                # auto-select after either operand/output moves to L1 changes
                # 8,637 final BF16 values even though the input tensors are
                # exact.
                "program_config": {
                    "family": "reuse",
                    "grid": [11, 10],
                    "in0_block_w": 1,
                    "out_subblock_h": 2,
                    "out_subblock_w": 2,
                    "out_block_h": 4,
                    "out_block_w": 2,
                    "per_core_m": 4,
                    "per_core_n": 2,
                    "fuse_batch": False,
                },
                "compute_config": config,
            },
            {
                "operation": "ttnn.typecast",
                "operand_indices": [10, 4],
            },
        ]
        compiled.internal_tensors = [
            {
                "source_operand": 0,
                "memory": "l1",
                "shape": [1, 1, 1, 1],
            },
            {"source_operand": 0, "memory": "l1"},
            {
                "source_operand": 1,
                "memory": "l1",
                "data_format": "Float32",
            },
            {"source_operand": 0, "memory": "l1"},
            {
                "source_operand": 2,
                "memory": value_memory,
                "data_format": "Float32",
            },
            {
                "source_operand": 2,
                "memory": head_memory,
                "data_format": "Float32",
            },
        ]
        return result

    operation.artifact_format = 4
    operation.physical_programs = 8
    operation.public_custom_calls = 1
    operation.compiler_owned_l1_tensors = 6
    # Qualification publishes probability as a real DPS output.  Count that
    # traffic honestly; the private owner below is the zero-intermediate-DRAM
    # production candidate.
    operation.intermediate_dram_bytes = (
        query_heads * sequence_length * sequence_length * 4
    )
    operation.published_probability = True
    operation.safe_default = False
    operation.experimental_backend = "native_attention_fp32_tail_v1"
    operation.value_memory = value_memory
    operation.head_memory = head_memory
    operation._ttl_program = _probe
    return operation


@lru_cache(maxsize=8)
def build_tt_lang_native_attention_fp32_tail_private(
    *,
    sequence_length: int,
    query_heads: int,
    head_dim: int,
    scale: float,
) -> Callable:
    """Build the exact attention tail with only the BF16 head made public.

    The FP32 probability and every other arithmetic intermediate are
    compiler-owned L1 tensors.  This is the contract needed by the full
    resident attention-forward owner: no probability DPS result, no hidden
    PJRT tensor, and no intermediate DRAM round trip.
    """

    if sequence_length <= 0 or sequence_length % 32:
        raise ValueError("sequence length must be positive and tile aligned")
    if query_heads <= 0:
        raise ValueError("query heads must be positive")
    if head_dim <= 0 or head_dim % 32:
        raise ValueError("head dimension must be positive and tile aligned")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError("native attention FP32 tail requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid="full")
    def _probe(raw_score, mask, expanded_value, head):
        source = ttl.make_dataflow_buffer_like(
            expanded_value, shape=(1, 1), block_count=1
        )
        output = ttl.make_dataflow_buffer_like(
            head, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                transaction = ttl.copy(
                    expanded_value[0, 0, 0:1, 0:1], tile
                )
                transaction.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with output.reserve() as target:
                    target.store(tile)

        @ttl.datamovement()
        def write():
            with output.wait() as tile:
                transaction = ttl.copy(tile, head[0, 0, 0:1, 0:1])
                transaction.wait()

    variant = (
        f"s{sequence_length}.q{query_heads}.d{head_dim}."
        f"scale{float(scale):.9g}.private-l1.v1"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.native_attention_fp32_tail.{variant}",
        arg_roles=("in", "in", "in", "out"),
        version_tag=f"native-attention-fp32-tail-{variant}",
    )
    def operation(raw_score, mask, expanded_value, head):
        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(raw_score, mask, expanded_value, head)
        compiled = compiled_ttl_program(_probe)
        config = _compute_config()
        compiled.program_sequence = [
            {
                "operation": "ttnn.full",
                "operand_indices": [4],
                "fill_value": float(scale),
            },
            {
                "operation": "ttnn.multiply",
                "operand_indices": [0, 4, 5],
            },
            {
                "operation": "ttnn.typecast",
                "operand_indices": [1, 6],
            },
            {
                "operation": "ttnn.add",
                "operand_indices": [5, 6, 7],
            },
            {
                "operation": "ttnn.softmax",
                "operand_indices": [7, 8],
                "dimension": -1,
                "numeric_stable": True,
                "compute_config": config,
            },
            {
                "operation": "ttnn.typecast",
                "operand_indices": [2, 9],
            },
            {
                "operation": "ttnn.matmul",
                "operand_indices": [8, 9, 10],
                "transpose_a": False,
                "transpose_b": False,
                "program_config": {
                    "family": "reuse",
                    "grid": [11, 10],
                    "in0_block_w": 1,
                    "out_subblock_h": 2,
                    "out_subblock_w": 2,
                    "out_block_h": 4,
                    "out_block_w": 2,
                    "per_core_m": 4,
                    "per_core_n": 2,
                    "fuse_batch": False,
                },
                "compute_config": config,
            },
            {
                "operation": "ttnn.typecast",
                "operand_indices": [10, 3],
            },
        ]
        compiled.internal_tensors = [
            {
                "source_operand": 0,
                "memory": "l1",
                "shape": [1, 1, 1, 1],
            },
            {"source_operand": 0, "memory": "l1"},
            {
                "source_operand": 1,
                "memory": "l1",
                "data_format": "Float32",
            },
            {"source_operand": 0, "memory": "l1"},
            {"source_operand": 0, "memory": "l1"},
            {
                "source_operand": 2,
                "memory": "l1",
                "data_format": "Float32",
            },
            {
                "source_operand": 2,
                "memory": "l1",
                "data_format": "Float32",
            },
        ]
        return result

    operation.artifact_format = 4
    operation.physical_programs = 8
    operation.public_custom_calls = 1
    operation.compiler_owned_l1_tensors = 7
    operation.intermediate_dram_bytes = 0
    operation.published_probability = False
    operation.safe_default = False
    operation.experimental_backend = "native_attention_fp32_tail_private_v1"
    operation._ttl_program = _probe
    return operation


@lru_cache(maxsize=8)
def build_tt_lang_native_attention_fp32_tail_compact_value_private(
    *,
    sequence_length: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float,
) -> Callable:
    """Private exact tail accepting compact GQA value heads.

    A NoC-only child fans each compact KV head out to its query-head group in
    compiler-owned L1.  The exact private FP32 tail then consumes that tensor
    directly.  The only public result remains the final BF16 attention head.
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
            raise RuntimeError("native attention FP32 tail requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    private_tail = build_tt_lang_native_attention_fp32_tail_private(
        sequence_length=sequence_length,
        query_heads=query_heads,
        head_dim=head_dim,
        scale=scale,
    )
    page_copy = build_tt_lang_owner_aligned_copy(
        rows=sequence_length,
        columns=query_heads * head_dim,
    )

    @ttl.operation(grid="full")
    def _probe(raw_score, mask, compact_value, head):
        source = ttl.make_dataflow_buffer_like(
            compact_value, shape=(1, 1), block_count=1
        )
        output = ttl.make_dataflow_buffer_like(
            head, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                transaction = ttl.copy(
                    compact_value[0, 0, 0:1, 0:1], tile
                )
                transaction.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with output.reserve() as target:
                    target.store(tile)

        @ttl.datamovement()
        def write():
            with output.wait() as tile:
                transaction = ttl.copy(tile, head[0, 0, 0:1, 0:1])
                transaction.wait()

    variant = (
        f"s{sequence_length}.q{query_heads}.kv{kv_heads}.d{head_dim}."
        f"scale{float(scale):.9g}.compact-private-l1.v1"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.native_attention_fp32_tail.{variant}",
        arg_roles=("in", "in", "in", "out"),
        version_tag=f"native-attention-fp32-tail-{variant}",
    )
    def operation(raw_score, mask, compact_value, head):
        expanded_shape = (1, query_heads, sequence_length, head_dim)
        expanded_value = _compile_placeholder_like(
            compact_value, shape=expanded_shape
        )
        # Keep source and destination as distinct compile-time tensor objects.
        # Passing one placeholder twice makes TT-Lang intentionally alias the
        # two operation operands into a single tensor ID.  After remapping,
        # the NoC fan-out then writes back into the compact public input while
        # the private expanded tensor remains uninitialized.
        flat_expanded_source = _compile_placeholder_like(
            compact_value,
            shape=(sequence_length, query_heads * head_dim),
        )
        flat_expanded_destination = _compile_placeholder_like(
            compact_value,
            shape=(sequence_length, query_heads * head_dim),
        )
        copy_child = compile_serialized_child(
            page_copy, (flat_expanded_source, flat_expanded_destination)
        )
        copy_child = make_serialized_gqa_head_expand_copy_bitwise(
            copy_child,
            sequence_tiles=sequence_length // 32,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dimension_tiles=head_dim // 32,
        )
        copy_child = _remap_all(copy_child, {0: 2, 1: 4})

        tail_children = compile_serialized_sequence_children(
            private_tail, (raw_score, mask, expanded_value, head)
        )
        if len(tail_children) != private_tail.physical_programs:
            raise RuntimeError("private attention-tail child count changed")
        tail_map = {0: 0, 1: 1, 2: 4, 3: 3}
        tail_map.update({index: index + 1 for index in range(4, 11)})
        children = [
            copy_child,
            *(_remap_all(child, tail_map) for child in tail_children),
        ]

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(raw_score, mask, compact_value, head)
        compiled = compiled_ttl_program(_probe)
        compiled.program_sequence = children
        compiled.internal_tensors = [
            {
                # Derive the physical TILE encoding from raw_score.  The
                # compact GQA input may arrive row-major; cloning that layout
                # makes the downstream native matmul invalid even though the
                # NoC page payload itself is already tiled.
                "source_operand": 0,
                "memory": "l1",
                "shape": list(expanded_shape),
                "data_format": "BFloat16",
                "layout": "tile",
            },
            {
                "source_operand": 0,
                "memory": "l1",
                "shape": [1, 1, 1, 1],
                "layout": "tile",
            },
            {"source_operand": 0, "memory": "l1", "layout": "tile"},
            {
                "source_operand": 1,
                "memory": "l1",
                "data_format": "Float32",
                "layout": "tile",
            },
            {"source_operand": 0, "memory": "l1", "layout": "tile"},
            {"source_operand": 0, "memory": "l1", "layout": "tile"},
            {
                "source_operand": 0,
                "memory": "l1",
                "shape": list(expanded_shape),
                "data_format": "Float32",
                "layout": "tile",
            },
            {
                "source_operand": 0,
                "memory": "l1",
                "shape": list(expanded_shape),
                "data_format": "Float32",
                "layout": "tile",
            },
        ]
        return result

    operation.artifact_format = 4
    operation.physical_programs = 1 + private_tail.physical_programs
    operation.public_custom_calls = 1
    operation.compiler_owned_l1_tensors = 8
    operation.intermediate_dram_bytes = 0
    operation.gqa_expand_noc_bytes = (
        query_heads * sequence_length * head_dim * 2 * 2
    )
    operation.published_probability = False
    operation.safe_default = False
    operation.experimental_backend = (
        "native_attention_fp32_tail_compact_value_private_v1"
    )
    operation._ttl_program = _probe
    return operation


@lru_cache(maxsize=8)
def build_tt_lang_native_attention_fp32_tail_diagnostic(
    *,
    sequence_length: int,
    query_heads: int,
    head_dim: int,
    scale: float,
) -> Callable:
    """Expose every native tail boundary to locate the first divergence.

    This is intentionally a qualification-only path: the corresponding
    production builder above keeps the same tensors compiler-private in L1.
    """

    if sequence_length <= 0 or sequence_length % 32:
        raise ValueError("sequence length must be positive and tile aligned")
    if query_heads <= 0:
        raise ValueError("query heads must be positive")
    if head_dim <= 0 or head_dim % 32:
        raise ValueError("head dimension must be positive and tile aligned")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError("native attention FP32 tail requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid="full")
    def _probe(
        raw_score,
        mask,
        expanded_value,
        scaled_score,
        fp32_mask,
        masked_score,
        probability,
        fp32_value,
        fp32_head,
        head,
    ):
        source = ttl.make_dataflow_buffer_like(
            raw_score, shape=(1, 1), block_count=1
        )
        output = ttl.make_dataflow_buffer_like(
            scaled_score, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                transaction = ttl.copy(raw_score[0, 0, 0:1, 0:1], tile)
                transaction.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with output.reserve() as target:
                    target.store(tile)

        @ttl.datamovement()
        def write():
            with output.wait() as tile:
                transaction = ttl.copy(tile, scaled_score[0, 0, 0:1, 0:1])
                transaction.wait()

    variant = (
        f"s{sequence_length}.q{query_heads}.d{head_dim}."
        f"scale{float(scale):.9g}.diagnostic.v1"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.native_attention_fp32_tail.{variant}",
        arg_roles=("in", "in", "in", *("out",) * 7),
        version_tag=f"native-attention-fp32-tail-{variant}",
    )
    def operation(
        raw_score,
        mask,
        expanded_value,
        scaled_score,
        fp32_mask,
        masked_score,
        probability,
        fp32_value,
        fp32_head,
        head,
    ):
        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(
            raw_score,
            mask,
            expanded_value,
            scaled_score,
            fp32_mask,
            masked_score,
            probability,
            fp32_value,
            fp32_head,
            head,
        )
        compiled = compiled_ttl_program(_probe)
        config = _compute_config()
        compiled.program_sequence = [
            {
                "operation": "ttnn.full",
                "operand_indices": [10],
                "fill_value": float(scale),
            },
            {"operation": "ttnn.multiply", "operand_indices": [0, 10, 3]},
            {"operation": "ttnn.typecast", "operand_indices": [1, 4]},
            {"operation": "ttnn.add", "operand_indices": [3, 4, 5]},
            {
                "operation": "ttnn.softmax",
                "operand_indices": [5, 6],
                "dimension": -1,
                "numeric_stable": True,
                "compute_config": config,
            },
            {"operation": "ttnn.typecast", "operand_indices": [2, 7]},
            {
                "operation": "ttnn.matmul",
                "operand_indices": [6, 7, 8],
                "transpose_a": False,
                "transpose_b": False,
                "compute_config": config,
            },
            {"operation": "ttnn.typecast", "operand_indices": [8, 9]},
        ]
        compiled.internal_tensors = [
            {
                "source_operand": 0,
                "memory": "l1",
                "shape": [1, 1, 1, 1],
            }
        ]
        return result

    operation.artifact_format = 4
    operation.physical_programs = 8
    operation.public_custom_calls = 1
    operation.safe_default = False
    operation.experimental_backend = "native_attention_fp32_tail_diagnostic_v1"
    operation._ttl_program = _probe
    return operation


__all__ = [
    "build_tt_lang_native_attention_fp32_tail",
    "build_tt_lang_native_attention_fp32_tail_compact_value_private",
    "build_tt_lang_native_attention_fp32_tail_private",
    "build_tt_lang_native_attention_fp32_tail_diagnostic",
]
