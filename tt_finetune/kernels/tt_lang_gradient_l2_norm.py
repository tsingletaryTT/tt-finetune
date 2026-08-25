"""Exact per-logical-tensor L2 norm over packed gradient pages."""

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
    if ttl is not None and tt_lang_operation is not None:
        return
    try:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("gradient L2 norm requires TT-XLA") from exc
    ttl, tt_lang_operation = ttl_module, decorator


@lru_cache(maxsize=64)
def build_tt_lang_gradient_l2_norm(
    *, page_offset: int, height_tiles: int, width_tiles: int,
    apply_sqrt: bool = True,
    output_page_offset: int = 0,
    norm_arena_pages: int = 2,
) -> Callable:
    if page_offset < 0:
        raise ValueError("page_offset must be non-negative")
    if height_tiles <= 0 or width_tiles <= 0:
        raise ValueError("tile dimensions must be positive")
    if output_page_offset < 0:
        raise ValueError("output_page_offset must be non-negative")
    if norm_arena_pages <= output_page_offset:
        raise ValueError("norm_arena_pages must contain the output page")
    _load_tt_lang()
    pages = height_tiles * width_tiles

    # Match the TTNN generic HW-reduce contract.  The public reduction keeps
    # its BF16 input/output boundary, but the GMPOOL destination accumulator is
    # FP32.  Leaving this at TT-Lang's default changes the reduction rounding
    # even though the tile traversal is otherwise identical.
    @ttl.operation(grid=(1, 1), fp32_dest_acc_en=True)
    def norm_program(gradient_in, scaler_in, norm_arena_in, norm_out):
        gradient = ttl.make_dataflow_buffer_like(
            gradient_in, shape=(1, 1), block_count=2
        )
        scaler = ttl.make_dataflow_buffer_like(
            scaler_in, shape=(1, 1), block_count=1
        )
        square = ttl.make_dataflow_buffer_like(
            # TT-Lang's source-level validator caps block_count at 32.  The
            # native exact reduction intentionally materializes the complete
            # BF16 square tensor in L1 (matching the public multiply -> reduce
            # boundary), so the descriptor is expanded to ``pages`` below.
            gradient_in, shape=(1, 1), block_count=min(pages, 32)
        )
        reduced = ttl.make_dataflow_buffer_like(
            norm_out, shape=(1, 1), block_count=1
        )
        norm = ttl.make_dataflow_buffer_like(
            norm_out, shape=(1, 1), block_count=1
        )
        carry = ttl.make_dataflow_buffer_like(
            norm_arena_in, shape=(1, 1), block_count=2
        )
        exponent = ttl.make_dataflow_buffer_like(
            scaler_in, shape=(1, 1), block_count=1
        )

        @ttl.compute()
        def compute():
            for _ in range(pages):
                with gradient.wait() as value:
                    with square.reserve() as output:
                        output.store(value * value)
            with reduced.reserve() as output:
                with scaler.wait() as value:
                    output.store(value)
            with reduced.wait() as value:
                exponent.wait().pop()
                with norm.reserve() as output:
                    output.store(ttl.sqrt(value))

        @ttl.datamovement()
        def read():
            for page in range(norm_arena_pages):
                if page != output_page_offset:
                    with carry.reserve() as value:
                        tx = ttl.copy(
                            norm_arena_in[page : page + 1, 0:1], value
                        )
                        tx.wait()
            with scaler.reserve() as value:
                tx = ttl.copy(scaler_in[0:1, 0:1], value)
                tx.wait()
            with exponent.reserve() as value:
                tx = ttl.copy(scaler_in[1:2, 0:1], value)
                tx.wait()
            for page in range(pages):
                with gradient.reserve() as value:
                    tx = ttl.copy(
                        gradient_in[
                            page_offset + page : page_offset + page + 1,
                            0:1,
                        ],
                        value,
                    )
                    tx.wait()

        @ttl.datamovement()
        def write():
            # Preserve every previously written norm page in the new output
            # allocation, then replace exactly this operation's scalar page.
            for page in range(norm_arena_pages):
                if page != output_page_offset:
                    with carry.wait() as value:
                        tx = ttl.copy(
                            value, norm_out[page : page + 1, 0:1]
                        )
                        tx.wait()
            with norm.wait() as value:
                tx = ttl.copy(
                    value,
                    norm_out[
                        output_page_offset : output_page_offset + 1,
                        0:1,
                    ],
                )
                tx.wait()

    digest = hashlib.sha256(
        f"{page_offset}:{height_tiles}:{width_tiles}:{int(apply_sqrt)}:"
        f"out{output_page_offset}:arena{norm_arena_pages}:v4".encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.gradient_l2_norm.{digest}",
        # The previous norm arena is an explicit input and the same tensor may
        # be supplied as the output template.  This is TT-Lang's two-operand
        # spelling of an in-place state edge (the decorator has no ``inout``).
        arg_roles=("in", "in", "in", "out"),
        version_tag=f"gradient-l2-norm-{digest}",
    )
    def operation(gradient_in, scaler_in, norm_arena_in, norm_out):
        _clear_ttl_compile_wrapper_cache(norm_program)
        result = norm_program(
            gradient_in, scaler_in, norm_arena_in, norm_out
        )
        _install_native_compute(
            norm_program,
            source_name="gradient_l2_norm_compute.cpp",
            expected_cbs=7,
            replacements={
                "@HEIGHT_TILES@": str(height_tiles),
                "@WIDTH_TILES@": str(width_tiles),
                "@APPLY_SQRT@": str(int(apply_sqrt)),
            },
        )
        closure = norm_program.__closure__ or ()
        for name, cell in zip(norm_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 7:
                    raise RuntimeError(
                        "gradient L2 norm CB descriptor contract changed"
                    )
                compiled.cb_configs[2].block_count = pages
            break
        return result

    operation.page_offset = page_offset
    operation.height_tiles = height_tiles
    operation.width_tiles = width_tiles
    operation.pages = pages
    operation.apply_sqrt = apply_sqrt
    operation.output_page_offset = output_page_offset
    operation.norm_arena_pages = norm_arena_pages
    operation.safe_default = False
    operation.experimental_backend = "packed_exact_gradient_l2_norm_v1"
    operation._ttl_program = norm_program
    return operation
