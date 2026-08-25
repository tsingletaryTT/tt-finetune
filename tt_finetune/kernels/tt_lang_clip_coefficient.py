"""Exact device clip coefficient from a packed BF16 total-norm tile."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_adamw_exact import _install_native_compute


ttl = None
tt_lang_operation = None


def _load_tt_lang() -> None:
    global ttl, tt_lang_operation
    if ttl is not None:
        return
    try:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("clip coefficient requires TT-XLA") from exc
    ttl, tt_lang_operation = ttl_module, decorator


@lru_cache(maxsize=8)
def build_tt_lang_clip_coefficient(*, norm_page_offset: int = 0) -> Callable:
    if norm_page_offset < 0:
        raise ValueError("norm_page_offset must be non-negative")
    _load_tt_lang()

    @ttl.operation(grid=(1, 1), fp32_dest_acc_en=False)
    def coefficient_program(norm_in, constants_in, coefficient_out):
        norm = ttl.make_dataflow_buffer_like(
            norm_in, shape=(1, 1), block_count=1
        )
        constants = ttl.make_dataflow_buffer_like(
            constants_in, shape=(3, 1), block_count=1
        )
        denominator = ttl.make_dataflow_buffer_like(
            coefficient_out, shape=(1, 1), block_count=1
        )
        ratio = ttl.make_dataflow_buffer_like(
            coefficient_out, shape=(1, 1), block_count=1
        )
        scalar = ttl.make_dataflow_buffer_like(
            coefficient_out, shape=(1, 1), block_count=1
        )
        output = ttl.make_dataflow_buffer_like(
            coefficient_out, shape=(1, 1), block_count=1
        )

        @ttl.compute()
        def compute():
            with norm.wait() as value:
                with denominator.reserve() as temp:
                    temp.store(value)
            with denominator.wait() as value:
                with ratio.reserve() as temp:
                    temp.store(value)
            constants.wait().pop()
            with ratio.wait() as value:
                with scalar.reserve() as result:
                    result.store(value)
            with scalar.wait() as value:
                with output.reserve() as result:
                    result.store(value)

        @ttl.datamovement()
        def read():
            with norm.reserve() as value:
                tx = ttl.copy(
                    norm_in[norm_page_offset : norm_page_offset + 1, 0:1],
                    value,
                )
                tx.wait()
            with constants.reserve() as value:
                tx = ttl.copy(constants_in[0:3, 0:1], value)
                tx.wait()

        @ttl.datamovement()
        def write():
            with output.wait() as value:
                tx = ttl.copy(value, coefficient_out[0:1, 0:1])
                tx.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.clip_coefficient.page{norm_page_offset}.v1",
        arg_roles=("in", "in", "out"),
        version_tag=f"clip-coefficient-page{norm_page_offset}-v1",
    )
    def operation(norm_in, constants_in, coefficient_out):
        _clear_ttl_compile_wrapper_cache(coefficient_program)
        result = coefficient_program(norm_in, constants_in, coefficient_out)
        _install_native_compute(
            coefficient_program,
            source_name="clip_coefficient_compute.cpp",
            expected_cbs=6,
            replacements={},
        )
        return result

    operation.norm_page_offset = norm_page_offset
    operation.safe_default = False
    operation.experimental_backend = "packed_exact_clip_coefficient_v1"
    operation._ttl_program = coefficient_program
    return operation
