"""Exact scalar power stage over a packed BF16 norm arena."""

from __future__ import annotations

import hashlib
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
        raise RuntimeError("gradient scalar power requires TT-XLA") from exc
    ttl, tt_lang_operation = ttl_module, decorator


@lru_cache(maxsize=128)
def build_tt_lang_gradient_scalar_power(
    *, input_page_offset: int, output_page_offset: int,
    arena_pages: int,
) -> Callable:
    if min(input_page_offset, output_page_offset) < 0:
        raise ValueError("page offsets must be non-negative")
    if arena_pages <= max(input_page_offset, output_page_offset):
        raise ValueError("arena_pages must contain both scalar pages")
    _load_tt_lang()

    @ttl.operation(grid=(1, 1), fp32_dest_acc_en=False)
    def power_program(sum_in, exponent_in, arena_in, arena_out):
        source = ttl.make_dataflow_buffer_like(
            sum_in, shape=(1, 1), block_count=1
        )
        exponent = ttl.make_dataflow_buffer_like(
            exponent_in, shape=(1, 1), block_count=1
        )
        output = ttl.make_dataflow_buffer_like(
            arena_out, shape=(1, 1), block_count=1
        )
        carry = ttl.make_dataflow_buffer_like(
            arena_in, shape=(1, 1), block_count=2
        )

        @ttl.compute()
        def compute():
            with source.wait() as value:
                exponent.wait().pop()
                with output.reserve() as result:
                    # Compile-safe shell; native source uses binary power.
                    result.store(ttl.sqrt(value))

        @ttl.datamovement()
        def read():
            with source.reserve() as value:
                tx = ttl.copy(
                    sum_in[input_page_offset : input_page_offset + 1, 0:1],
                    value,
                )
                tx.wait()
            with exponent.reserve() as value:
                tx = ttl.copy(exponent_in[0:1, 0:1], value)
                tx.wait()
            for page in range(arena_pages):
                if page != output_page_offset:
                    with carry.reserve() as value:
                        tx = ttl.copy(arena_in[page : page + 1, 0:1], value)
                        tx.wait()

        @ttl.datamovement()
        def write():
            for page in range(arena_pages):
                if page != output_page_offset:
                    with carry.wait() as value:
                        tx = ttl.copy(value, arena_out[page : page + 1, 0:1])
                        tx.wait()
            with output.wait() as value:
                tx = ttl.copy(
                    value,
                    arena_out[output_page_offset : output_page_offset + 1, 0:1],
                )
                tx.wait()

    digest = hashlib.sha256(
        f"{input_page_offset}:{output_page_offset}:{arena_pages}:v1".encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.gradient_scalar_power.{digest}",
        arg_roles=("in", "in", "in", "out"),
        version_tag=f"gradient-scalar-power-{digest}",
    )
    def operation(sum_in, exponent_in, arena_in, arena_out):
        _clear_ttl_compile_wrapper_cache(power_program)
        result = power_program(sum_in, exponent_in, arena_in, arena_out)
        _install_native_compute(
            power_program,
            source_name="gradient_scalar_power_compute.cpp",
            expected_cbs=4,
            replacements={},
        )
        return result

    operation.input_page_offset = input_page_offset
    operation.output_page_offset = output_page_offset
    operation.arena_pages = arena_pages
    operation.safe_default = False
    operation.experimental_backend = "packed_exact_scalar_power_v1"
    operation._ttl_program = power_program
    return operation
