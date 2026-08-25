"""Exact BF16 scalar scaling over a packed gradient-arena page range."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_adamw_exact import _install_native_compute


TILE_SIZE = 32
ttl = None
tt_lang_operation = None


def _load_tt_lang() -> None:
    global ttl, tt_lang_operation
    if ttl is not None and tt_lang_operation is not None:
        return
    try:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
    except ImportError as exc:  # pragma: no cover - runtime-only path
        raise RuntimeError(
            "packed gradient scaling requires the TT-XLA environment"
        ) from exc
    ttl, tt_lang_operation = ttl_module, decorator


@lru_cache(maxsize=16)
def build_tt_lang_gradient_scale(
    *, gradient_page_offset: int, gradient_pages: int, granularity: int = 1
) -> Callable:
    """Return a graph-stable exact BF16 scale operation.

    Only ``gradient_pages`` beginning at ``gradient_page_offset`` are read and
    written.  The coefficient is a pre-materialized BF16 tile.  Keeping this
    operation separate while the exact norm reduction is qualified lets us
    prove the scale half of clipping without changing its numerical boundary.
    """

    if gradient_page_offset < 0:
        raise ValueError("gradient_page_offset must be non-negative")
    if gradient_pages <= 0:
        raise ValueError("gradient_pages must be positive")
    if granularity <= 0 or gradient_pages % granularity:
        raise ValueError("granularity must divide gradient_pages")
    _load_tt_lang()

    @ttl.operation(grid=(8, 8))
    def scale_program(gradient_in, coefficient_in, gradient_out):
        rows_per_block = granularity
        total_blocks = gradient_pages // rows_per_block
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        blocks_per_node = -(-total_blocks // total_nodes)
        block_shape = (rows_per_block, 1)
        gradient_dfb = ttl.make_dataflow_buffer_like(
            gradient_in, shape=block_shape, block_count=2
        )
        coefficient_dfb = ttl.make_dataflow_buffer_like(
            coefficient_in, shape=(1, 1), block_count=1
        )
        output_dfb = ttl.make_dataflow_buffer_like(
            gradient_out, shape=block_shape, block_count=2
        )

        @ttl.compute()
        def compute():
            coefficient = coefficient_dfb.wait()
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    with (
                        gradient_dfb.wait() as gradient,
                        output_dfb.reserve() as output,
                    ):
                        output.store(gradient * coefficient)
            coefficient.pop()

        @ttl.datamovement()
        def read():
            coefficient = coefficient_dfb.reserve()
            coefficient_tx = ttl.copy(
                coefficient_in[0:1, 0:1], coefficient
            )
            coefficient_tx.wait()
            coefficient.push()

            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    start = gradient_page_offset + block * rows_per_block
                    stop = start + rows_per_block
                    with gradient_dfb.reserve() as gradient:
                        tx = ttl.copy(
                            gradient_in[start:stop, 0:1], gradient
                        )
                        tx.wait()

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    start = gradient_page_offset + block * rows_per_block
                    stop = start + rows_per_block
                    with output_dfb.wait() as output:
                        tx = ttl.copy(output, gradient_out[start:stop, 0:1])
                        tx.wait()

    digest = hashlib.sha256(
        f"{gradient_page_offset}:{gradient_pages}:{granularity}:v1".encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.gradient_scale.{digest}",
        arg_roles=("in", "in", "out"),
        version_tag=f"gradient-scale-{digest}",
    )
    def operation(gradient_in, coefficient_in, gradient_out):
        _clear_ttl_compile_wrapper_cache(scale_program)
        result = scale_program(gradient_in, coefficient_in, gradient_out)
        total_blocks = gradient_pages // granularity
        _install_native_compute(
            scale_program,
            source_name="gradient_scale_compute.cpp",
            expected_cbs=3,
            replacements={
                "@TOTAL_BLOCKS@": str(total_blocks),
                "@BLOCKS_PER_NODE@": str(-(-total_blocks // 64)),
                "@ROWS_PER_BLOCK@": str(granularity),
            },
        )
        return result

    operation.gradient_page_offset = gradient_page_offset
    operation.gradient_pages = gradient_pages
    operation.granularity = granularity
    operation.safe_default = False
    operation.experimental_backend = "packed_exact_gradient_scale_v1"
    operation._ttl_program = scale_program
    return operation
