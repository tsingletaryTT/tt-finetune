"""Row-owner-aligned BF16 copy used to qualify private program handoffs."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID

ttl = None
tt_lang_operation = None


@lru_cache(maxsize=16)
def build_tt_lang_owner_aligned_copy(
    *, rows: int, columns: int, source_row_offset: int = 0,
    source_column_offset: int = 0,
) -> Callable:
    if rows % 32 or columns % 32:
        raise ValueError("owner-aligned copy requires 32-aligned dimensions")
    if source_row_offset < 0 or source_row_offset % 32:
        raise ValueError("owner-aligned source row offset must be non-negative /32")
    if source_column_offset < 0 or source_column_offset % 32:
        raise ValueError(
            "owner-aligned source column offset must be non-negative /32"
        )
    row_tiles, column_tiles = rows // 32, columns // 32
    source_row_tile_offset = source_row_offset // 32
    source_column_tile_offset = source_column_offset // 32
    total_tiles = row_tiles * column_tiles
    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid=BLACKHOLE_P150A_GRID)
    def _program(source_tensor, output_tensor):
        source = ttl.make_dataflow_buffer_like(source_tensor, shape=(1, 1), block_count=2)
        output = ttl.make_dataflow_buffer_like(output_tensor, shape=(1, 1), block_count=2)

        @ttl.datamovement()
        def read():
            x, y = ttl.node(dims=2)
            node = y * 11 + x
            for row in range(node, row_tiles, 110):
                for column in range(column_tiles):
                    with source.reserve() as target:
                        tx = ttl.copy(
                            source_tensor[
                                row + source_row_tile_offset:
                                row + source_row_tile_offset + 1,
                                source_column_tile_offset + column:
                                source_column_tile_offset + column + 1,
                            ],
                            target,
                        )
                        tx.wait()

        @ttl.compute()
        def compute():
            x, y = ttl.node(dims=2)
            node = y * 11 + x
            for row in range(node, row_tiles, 110):
                for column in range(column_tiles):
                    with source.wait() as tile:
                        with output.reserve() as target:
                            target.store(tile)

        @ttl.datamovement()
        def write():
            x, y = ttl.node(dims=2)
            node = y * 11 + x
            for row in range(node, row_tiles, 110):
                for column in range(column_tiles):
                    with output.wait() as tile:
                        tx = ttl.copy(
                            tile,
                            output_tensor[row:row + 1, column:column + 1],
                        )
                        tx.wait()

    @tt_lang_operation(
        operation_id=(
            f"tt_finetune.owner_aligned_copy.{rows}x{columns}."
            f"row{source_row_offset}.column{source_column_offset}.v3"
        ),
        arg_roles=("in", "out"),
        version_tag=(
            f"owner-aligned-copy-{rows}x{columns}-"
            f"row{source_row_offset}-column{source_column_offset}-v3"
        ),
    )
    def operation(source_tensor, output_tensor):
        _clear_ttl_compile_wrapper_cache(_program)
        return _program(source_tensor, output_tensor)

    operation._ttl_program = _program
    operation.safe_default = False
    operation.source_row_offset = source_row_offset
    operation.source_column_offset = source_column_offset
    return operation
