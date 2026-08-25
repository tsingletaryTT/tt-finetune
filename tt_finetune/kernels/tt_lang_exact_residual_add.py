"""Exact tile-distributed BF16 residual addition for resident owners.

The primitive is intentionally tiny, but unlike an optimization leaf it is a
composition building block: its output may be remapped to compiler-owned L1
and consumed by the next compact TT-Metal program without entering PJRT.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=16)
def build_tt_lang_exact_residual_add(
    *, rows: int, columns: int, owner_aligned: bool = False
) -> Callable:
    if rows <= 0 or columns <= 0 or rows % 32 or columns % 32:
        raise ValueError("residual add dimensions must be positive and 32-aligned")
    row_tiles = rows // 32
    column_tiles = columns // 32
    total_tiles = row_tiles * column_tiles

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError("exact residual add requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = decorator

    @ttl.operation(grid=BLACKHOLE_P150A_GRID if owner_aligned else "full")
    def _program(lhs, rhs, output):
        grid_columns, grid_rows = ttl.grid_size(dims=2)
        workers = grid_columns * grid_rows
        local_tiles = column_tiles if owner_aligned else -(-total_tiles // workers)
        depth = 1 if owner_aligned else 2
        lhs_cb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=depth)
        rhs_cb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=depth)
        output_cb = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=depth
        )

        @ttl.datamovement()
        def read():
            worker_column, worker_row = ttl.node(dims=2)
            worker = worker_row * grid_columns + worker_column
            for local_tile in range(local_tiles):
                tile_id = worker * local_tiles + local_tile
                if tile_id < total_tiles:
                    tile_row = tile_id // column_tiles
                    tile_column = tile_id - tile_row * column_tiles
                    with lhs_cb.reserve() as tile:
                        tx = ttl.copy(
                            lhs[
                                tile_row : tile_row + 1,
                                tile_column : tile_column + 1,
                            ],
                            tile,
                        )
                        tx.wait()
                    with rhs_cb.reserve() as tile:
                        tx = ttl.copy(
                            rhs[
                                tile_row : tile_row + 1,
                                tile_column : tile_column + 1,
                            ],
                            tile,
                        )
                        tx.wait()

        @ttl.compute()
        def compute():
            worker_column, worker_row = ttl.node(dims=2)
            worker = worker_row * grid_columns + worker_column
            for local_tile in range(local_tiles):
                tile_id = worker * local_tiles + local_tile
                if tile_id < total_tiles:
                    with lhs_cb.wait() as lhs_tile:
                        with rhs_cb.wait() as rhs_tile:
                            with output_cb.reserve() as target:
                                target.store(lhs_tile + rhs_tile)

        @ttl.datamovement()
        def write():
            worker_column, worker_row = ttl.node(dims=2)
            worker = worker_row * grid_columns + worker_column
            for local_tile in range(local_tiles):
                tile_id = worker * local_tiles + local_tile
                if tile_id < total_tiles:
                    tile_row = tile_id // column_tiles
                    tile_column = tile_id - tile_row * column_tiles
                    with output_cb.wait() as tile:
                        tx = ttl.copy(
                            tile,
                            output[
                                tile_row : tile_row + 1,
                                tile_column : tile_column + 1,
                            ],
                        )
                        tx.wait()

    @tt_lang_operation(
        operation_id=(
            f"tt_finetune.exact_residual_add.{rows}x{columns}."
            f"{'owner' if owner_aligned else 'distributed'}.v2"
        ),
        arg_roles=("in", "in", "out"),
        version_tag=f"exact-residual-add-{rows}x{columns}-v1",
    )
    def operation(lhs, rhs, output):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(lhs, rhs, output)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                for path, thread in compiled.kernel_paths:
                    if thread != "compute":
                        continue
                    source_path = Path(path)
                    source = source_path.read_text()
                    marker = "void kernel_main() {\n"
                    if source.count(marker) != 1:
                        raise RuntimeError("residual-add compute entry changed")
                    source = source.replace(
                        marker,
                        marker
                        + "  compute_kernel_hw_startup("
                        + "get_compile_time_arg_val(0), "
                        + "get_compile_time_arg_val(1), "
                        + "get_compile_time_arg_val(2));\n",
                        1,
                    )
                    source_path.write_text(source)
            break
        return result

    operation.rows = rows
    operation.columns = columns
    operation.bit_exact_contract = True
    operation.owner_aligned = bool(owner_aligned)
    operation._ttl_program = _program
    operation.safe_default = False
    return operation
