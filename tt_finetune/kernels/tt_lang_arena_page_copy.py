"""Copy one tile-contiguous matrix slice out of a larger BF16 arena."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=16)
def build_tt_lang_arena_page_copy(
    *, rows: int, columns: int, source_page_offset: int
) -> Callable:
    """Copy ``rows*columns`` beginning at an arbitrary arena tile page."""
    if rows <= 0 or rows % 32 or columns <= 0 or columns % 32:
        raise ValueError("arena page copy requires positive tile-aligned dimensions")
    if source_page_offset < 0:
        raise ValueError("arena source page offset must be non-negative")
    row_tiles, column_tiles = rows // 32, columns // 32
    total_tiles = row_tiles * column_tiles
    worker_count = 110
    jobs_per_worker = (total_tiles + worker_count - 1) // worker_count

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid=BLACKHOLE_P150A_GRID)
    def _program(source_tensor, output_tensor):
        source = ttl.make_dataflow_buffer_like(
            source_tensor, shape=(1, 1), block_count=2
        )
        output = ttl.make_dataflow_buffer_like(
            output_tensor, shape=(1, 1), block_count=2
        )

        @ttl.datamovement()
        def read():
            x, y = ttl.node(dims=2)
            node = y * 11 + x
            for job in range(jobs_per_worker):
                tile = job * worker_count + node
                if tile < total_tiles:
                    page = source_page_offset + tile
                    source_row = page // column_tiles
                    source_column = page % column_tiles
                    with source.reserve() as target:
                        tx = ttl.copy(
                            source_tensor[
                                source_row:source_row + 1,
                                source_column:source_column + 1,
                            ],
                            target,
                        )
                        tx.wait()

        @ttl.compute()
        def compute():
            x, y = ttl.node(dims=2)
            node = y * 11 + x
            for job in range(jobs_per_worker):
                tile_index = job * worker_count + node
                if tile_index < total_tiles:
                    with source.wait() as tile:
                        with output.reserve() as target:
                            target.store(tile)

        @ttl.datamovement()
        def write():
            x, y = ttl.node(dims=2)
            node = y * 11 + x
            for job in range(jobs_per_worker):
                tile_index = job * worker_count + node
                if tile_index < total_tiles:
                    output_row = tile_index // column_tiles
                    output_column = tile_index % column_tiles
                    with output.wait() as tile:
                        tx = ttl.copy(
                            tile,
                            output_tensor[
                                output_row:output_row + 1,
                                output_column:output_column + 1,
                            ],
                        )
                        tx.wait()

    @tt_lang_operation(
        operation_id=(
            f"tt_finetune.arena_page_copy.{rows}x{columns}."
            f"page{source_page_offset}.v2"
        ),
        arg_roles=("in", "out"),
        version_tag=(
            f"arena-page-copy-{rows}x{columns}-page{source_page_offset}-v2"
        ),
    )
    def operation(source_tensor, output_tensor):
        _clear_ttl_compile_wrapper_cache(_program)
        return _program(source_tensor, output_tensor)

    operation._ttl_program = _program
    operation.source_page_offset = source_page_offset
    operation.safe_default = False
    return operation


@lru_cache(maxsize=32)
def build_tt_lang_linear_packed_page_copy(
    *, rows: int, columns: int, source_page_offset: int
) -> Callable:
    """Copy a matrix from a canonical one-tile-wide packed page tensor."""
    if rows <= 0 or rows % 32 or columns <= 0 or columns % 32:
        raise ValueError("packed page copy requires positive tile-aligned dimensions")
    if source_page_offset < 0:
        raise ValueError("packed source page offset must be non-negative")
    row_tiles, column_tiles = rows // 32, columns // 32
    total_tiles = row_tiles * column_tiles

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid=BLACKHOLE_P150A_GRID)
    def _program(source_tensor, output_tensor):
        source = ttl.make_dataflow_buffer_like(
            source_tensor, shape=(1, 1), block_count=2
        )
        output = ttl.make_dataflow_buffer_like(
            output_tensor, shape=(1, 1), block_count=2
        )

        @ttl.datamovement()
        def read():
            x, y = ttl.node(dims=2)
            node = y * 11 + x
            for column in range(column_tiles):
                tile = node * column_tiles + column
                if tile < total_tiles:
                    page = source_page_offset + tile
                    with source.reserve() as target:
                        tx = ttl.copy(
                            source_tensor[page:page + 1, 0:1], target
                        )
                        tx.wait()

        @ttl.compute()
        def compute():
            x, y = ttl.node(dims=2)
            node = y * 11 + x
            for column in range(column_tiles):
                if node * column_tiles + column < total_tiles:
                    with source.wait() as tile:
                        with output.reserve() as target:
                            target.store(tile)

        @ttl.datamovement()
        def write():
            x, y = ttl.node(dims=2)
            node = y * 11 + x
            for column in range(column_tiles):
                if node * column_tiles + column < total_tiles:
                    with output.wait() as tile:
                        tx = ttl.copy(
                            tile,
                            output_tensor[node:node + 1, column:column + 1],
                        )
                        tx.wait()

    variant = f"{rows}x{columns}.page{source_page_offset}.v1"

    @tt_lang_operation(
        operation_id=f"tt_finetune.linear_packed_page_copy.{variant}",
        arg_roles=("in", "out"),
        version_tag=f"linear-packed-page-copy-{variant}",
    )
    def operation(source_tensor, output_tensor):
        _clear_ttl_compile_wrapper_cache(_program)
        return _program(source_tensor, output_tensor)

    operation._ttl_program = _program
    operation.source_page_offset = source_page_offset
    operation.safe_default = False
    return operation
