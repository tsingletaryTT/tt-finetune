"""Qualification operation for compiler-owned artifact-v4 handoff tensors.

The public custom-call ABI is ``input, output``.  TT-MLIR allocates a private
third tensor in L1.  Two compact TT-Metal programs execute in order:

``external input -> private L1 scratch -> external output``.

This is deliberately small: it proves ownership, TensorAccessor remapping and
cross-program visibility before the same mechanism is used for full layer
activations.  It is experimental and never selected by the trainer backend.
"""

from __future__ import annotations

import copy
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache

ttl = None
tt_lang_operation = None


def build_tt_lang_internal_sequence_smoke(
    *, rows: int = 32, columns: int = 32
) -> Callable:
    if rows <= 0 or columns <= 0 or rows % 32 or columns % 32:
        raise ValueError("internal sequence smoke dimensions must be positive /32")
    row_tiles = rows // 32
    column_tiles = columns // 32
    total_tiles = row_tiles * column_tiles

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError("internal sequence smoke requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = decorator

    @ttl.operation(grid="full")
    def _copy_program(input_tensor, output_tensor):
        grid_columns, grid_rows = ttl.grid_size(dims=2)
        total_workers = grid_columns * grid_rows
        local_tiles = -(-total_tiles // total_workers)
        source = ttl.make_dataflow_buffer_like(
            input_tensor, shape=(1, 1), block_count=2
        )
        output = ttl.make_dataflow_buffer_like(
            output_tensor, shape=(1, 1), block_count=2
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
                    with source.reserve() as tile:
                        transaction = ttl.copy(
                            input_tensor[
                                tile_row : tile_row + 1,
                                tile_column : tile_column + 1,
                            ],
                            tile,
                        )
                        transaction.wait()

        @ttl.compute()
        def compute():
            worker_column, worker_row = ttl.node(dims=2)
            worker = worker_row * grid_columns + worker_column
            for local_tile in range(local_tiles):
                tile_id = worker * local_tiles + local_tile
                if tile_id < total_tiles:
                    with source.wait() as tile:
                        with output.reserve() as target:
                            target.store(tile)

        @ttl.datamovement()
        def write():
            worker_column, worker_row = ttl.node(dims=2)
            worker = worker_row * grid_columns + worker_column
            for local_tile in range(local_tiles):
                tile_id = worker * local_tiles + local_tile
                if tile_id < total_tiles:
                    tile_row = tile_id // column_tiles
                    tile_column = tile_id - tile_row * column_tiles
                    with output.wait() as tile:
                        transaction = ttl.copy(
                            tile,
                            output_tensor[
                                tile_row : tile_row + 1,
                                tile_column : tile_column + 1,
                            ],
                        )
                        transaction.wait()

    @tt_lang_operation(
        operation_id=(
            f"tt_finetune.internal_sequence_smoke.{rows}x{columns}.v2"
        ),
        arg_roles=("in", "out"),
        version_tag=f"internal-sequence-smoke-{rows}x{columns}-v2",
    )
    def operation(input_tensor, output_tensor):
        _clear_ttl_compile_wrapper_cache(_copy_program)
        result = _copy_program(input_tensor, output_tensor)

        closure = _copy_program.__closure__ or ()
        for name, cell in zip(_copy_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                producer = copy.copy(compiled)
                consumer = copy.copy(compiled)

                producer.kernel_tensor_indices = [
                    [2 if tensor_index == 1 else tensor_index for tensor_index in indices]
                    for indices in compiled.kernel_tensor_indices
                ]
                consumer.kernel_tensor_indices = [
                    [2 if tensor_index == 0 else tensor_index for tensor_index in indices]
                    for indices in compiled.kernel_tensor_indices
                ]

                compiled.program_sequence = [producer, consumer]
                compiled.internal_tensors = [
                    {"source_operand": 0, "memory": "l1"}
                ]
            break
        return result

    operation.artifact_format = 4
    operation.compiler_owned_internal_tensors = 1
    operation.experimental_backend = "internal_sequence_smoke_v2"
    operation.shape = (rows, columns)
    operation.safe_default = False
    return operation
