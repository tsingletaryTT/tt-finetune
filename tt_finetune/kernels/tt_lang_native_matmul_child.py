"""Artifact-v4 owner whose physical child is a real TTNN matmul.

This is intentionally a small qualification primitive.  A normal TT-Lang
probe establishes the public PJRT/custom-call ABI, then artifact-v4 replaces
its physical program with a ``ttnn.matmul`` descriptor.  The compiler lowers
that descriptor to the ordinary TTNN op, allowing TTNN's program factory to
own the complete reduction and packing context while the enclosing resident
owner remains one public call.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_program_composition import compiled_ttl_program


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=16)
def build_tt_lang_native_matmul_child(
    *,
    tokens: int,
    input_features: int,
    output_features: int,
    grid_x: int = 11,
    grid_y: int = 10,
    in0_block_w: int = 2,
    out_subblock_h: int = 1,
    out_subblock_w: int = 1,
    out_block_h: int = 1,
    out_block_w: int = 1,
    per_core_m: int = 1,
    per_core_n: int = 1,
) -> Callable:
    """Build ``X @ weight.T`` through a native child of one v4 owner."""

    if tokens <= 0 or input_features <= 0 or output_features <= 0:
        raise ValueError("matmul dimensions must be positive")
    if tokens % 32 or input_features % 32:
        raise ValueError("tokens and input_features must be tile aligned")
    if not 1 <= output_features <= 32:
        raise ValueError("the qualification owner supports one output tile")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError("native matmul child requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid="full")
    def _probe(inputs, weight, output):
        source = ttl.make_dataflow_buffer_like(
            inputs, shape=(1, 1), block_count=1
        )
        target = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                transaction = ttl.copy(inputs[0:1, 0:1], tile)
                transaction.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with target.reserve() as result:
                    result.store(tile)

        @ttl.datamovement()
        def write():
            with target.wait() as tile:
                transaction = ttl.copy(tile, output[0:1, 0:1])
                transaction.wait()

    variant = (
        f"t{tokens}.k{input_features}.n{output_features}."
        f"g{grid_x}x{grid_y}.bw{in0_block_w}."
        f"sb{out_subblock_h}x{out_subblock_w}."
        f"ob{out_block_h}x{out_block_w}."
        f"pc{per_core_m}x{per_core_n}.v1"
    )

    descriptor = {
        "operation": "ttnn.matmul",
        "operand_indices": [0, 1, 2],
        "transpose_a": False,
        "transpose_b": True,
        "program_config": {
            "grid": [grid_x, grid_y],
            "in0_block_w": in0_block_w,
            "out_subblock_h": out_subblock_h,
            "out_subblock_w": out_subblock_w,
            "out_block_h": out_block_h,
            "out_block_w": out_block_w,
            "per_core_m": per_core_m,
            "per_core_n": per_core_n,
            "fuse_batch": False,
            "mcast_in0": False,
            "gather_in0": False,
            "untilize_out": False,
        },
        "compute_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": True,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
    }

    @tt_lang_operation(
        operation_id=f"tt_finetune.native_matmul_child.{variant}",
        arg_roles=("in", "in", "out"),
        version_tag=f"native-matmul-child-{variant}",
    )
    def operation(inputs, weight, output):
        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(inputs, weight, output)
        compiled = compiled_ttl_program(_probe)
        compiled.program_sequence = [descriptor]
        compiled.internal_tensors = []
        return result

    operation.artifact_format = 4
    operation.physical_programs = 1
    operation.public_custom_calls = 1
    operation.output_shape = (tokens, output_features)
    operation.native_descriptor = descriptor
    operation.safe_default = False
    operation.experimental_backend = "native_ttnn_matmul_child_v1"
    operation._ttl_program = _probe
    return operation


__all__ = ["build_tt_lang_native_matmul_child"]


@lru_cache(maxsize=16)
def build_tt_lang_native_batched_qk_matmul(
    *,
    sequence_length: int,
    query_heads: int,
    head_dim: int,
) -> Callable:
    """Build canonical FP32 ``Q @ K.T`` as one native TTNN child.

    Q and K use ``[1, heads, sequence, head_dim]`` TILE tensors.  Keeping the
    transpose as a matmul attribute lets TTNN own the reader, reduction and
    pack context exactly as it does for the untouched PyTorch/XLA graph.
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
            raise RuntimeError("native batched QK child requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid="full")
    def _probe(query, key, output):
        source = ttl.make_dataflow_buffer_like(
            query, shape=(1, 1), block_count=1
        )
        target = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=1
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
                    result.store(tile)

        @ttl.datamovement()
        def write():
            with target.wait() as tile:
                transaction = ttl.copy(tile, output[0, 0, 0:1, 0:1])
                transaction.wait()

    variant = f"s{sequence_length}.q{query_heads}.d{head_dim}.fp32.v1"
    descriptor = {
        "operation": "ttnn.matmul",
        "operand_indices": [0, 1, 2],
        "transpose_a": False,
        "transpose_b": True,
        "compute_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": True,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
    }

    @tt_lang_operation(
        operation_id=f"tt_finetune.native_batched_qk_matmul.{variant}",
        arg_roles=("in", "in", "out"),
        version_tag=f"native-batched-qk-matmul-{variant}",
    )
    def operation(query, key, output):
        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(query, key, output)
        compiled = compiled_ttl_program(_probe)
        compiled.program_sequence = [descriptor]
        compiled.internal_tensors = []
        return result

    operation.artifact_format = 4
    operation.physical_programs = 1
    operation.public_custom_calls = 1
    operation.output_shape = (
        1, query_heads, sequence_length, sequence_length
    )
    operation.native_descriptor = descriptor
    operation.safe_default = False
    operation.experimental_backend = "native_batched_qk_matmul_v1"
    operation._ttl_program = _probe
    return operation


__all__.append("build_tt_lang_native_batched_qk_matmul")


@lru_cache(maxsize=16)
def build_tt_lang_native_batched_pv_matmul(
    *, sequence_length: int, query_heads: int, head_dim: int,
) -> Callable:
    """Canonical BF16 ``probability @ value`` qualification child."""
    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid="full")
    def _probe(probability, value, output):
        source = ttl.make_dataflow_buffer_like(probability, shape=(1, 1), block_count=1)
        target = ttl.make_dataflow_buffer_like(output, shape=(1, 1), block_count=1)
        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                tx = ttl.copy(probability[0, 0, 0:1, 0:1], tile); tx.wait()
        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with target.reserve() as result: result.store(tile)
        @ttl.datamovement()
        def write():
            with target.wait() as tile:
                tx = ttl.copy(tile, output[0, 0, 0:1, 0:1]); tx.wait()

    variant = f"s{sequence_length}.q{query_heads}.d{head_dim}.bf16.auto.v2"
    descriptor = {
        "operation": "ttnn.matmul", "operand_indices": [0, 1, 2],
        "transpose_a": False, "transpose_b": False,
        "compute_config": {
            "math_fidelity": "HiFi4", "fp32_dest_acc_en": True,
        },
    }
    @tt_lang_operation(
        operation_id=f"tt_finetune.native_batched_pv_matmul.{variant}",
        arg_roles=("in", "in", "out"),
        version_tag=f"native-batched-pv-matmul-{variant}",
    )
    def operation(probability, value, output):
        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(probability, value, output)
        compiled = compiled_ttl_program(_probe)
        compiled.program_sequence = [descriptor]
        compiled.internal_tensors = []
        return result
    operation.artifact_format = 4
    operation.physical_programs = 1
    operation.public_custom_calls = 1
    operation.native_descriptor = descriptor
    operation._ttl_program = _probe
    return operation


__all__.append("build_tt_lang_native_batched_pv_matmul")
