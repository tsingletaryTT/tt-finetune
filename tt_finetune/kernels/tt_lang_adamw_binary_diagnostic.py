"""Standalone native FP32 binary programs for numerical qualification."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_adamw_exact import (
    _install_native_compute,
    _set_native_unpack_fp32,
)


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=8)
def build_tt_lang_adamw_binary_diagnostic(
    *, rows: int, operation: str
) -> Callable:
    if rows <= 0 or rows % 32:
        raise ValueError("rows must be positive and divisible by 32")
    if operation not in {"add", "divide"}:
        raise ValueError("operation must be add or divide")
    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
        ttl, tt_lang_operation = ttl_module, decorator

    total_tiles = rows // 32

    @ttl.operation(
        grid=(8, 8), fp32_dest_acc_en=True,
        options="--no-ttl-maximize-dst",
    )
    def _program(lhs_in, rhs_in, output):
        grid_columns, grid_rows = ttl.grid_size(dims=2)
        tiles_per_node = -(-total_tiles // (grid_columns * grid_rows))
        lhs = ttl.make_dataflow_buffer_like(
            lhs_in, shape=(1, 1), block_count=2
        )
        rhs = ttl.make_dataflow_buffer_like(
            rhs_in, shape=(1, 1), block_count=2
        )
        result = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=2
        )

        @ttl.compute()
        def compute():
            column, row = ttl.node(dims=2)
            node = row * grid_columns + column
            for local in range(tiles_per_node):
                tile = node * tiles_per_node + local
                if tile < total_tiles:
                    with lhs.wait() as a, rhs.wait() as b, result.reserve() as c:
                        c.store(a + b)

        @ttl.datamovement()
        def read():
            column, row = ttl.node(dims=2)
            node = row * grid_columns + column
            for local in range(tiles_per_node):
                tile = node * tiles_per_node + local
                if tile < total_tiles:
                    with lhs.reserve() as a, rhs.reserve() as b:
                        txa = ttl.copy(lhs_in[tile:tile + 1, 0:1], a)
                        txb = ttl.copy(rhs_in[tile:tile + 1, 0:1], b)
                        txa.wait()
                        txb.wait()

        @ttl.datamovement()
        def write():
            column, row = ttl.node(dims=2)
            node = row * grid_columns + column
            for local in range(tiles_per_node):
                tile = node * tiles_per_node + local
                if tile < total_tiles:
                    with result.wait() as value:
                        tx = ttl.copy(value, output[tile:tile + 1, 0:1])
                        tx.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.adamw_binary_diag.{operation}.r{rows}.v1",
        arg_roles=("in", "in", "out"),
        version_tag=f"adamw-binary-diag-{operation}-r{rows}-v1",
    )
    def op(lhs, rhs, output):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(lhs, rhs, output)
        _install_native_compute(
            _program,
            source_name="adamw_fp32_binary_diagnostic_compute.cpp",
            expected_cbs=3,
            replacements={
                "@TOTAL_TILES@": str(total_tiles),
                "@TILES_PER_NODE@": str(-(-total_tiles // 64)),
                "@OP_INIT@": (
                    "add_binary_tile_init();"
                    if operation == "add" else "div_binary_tile_init();"
                ),
                "@OP_CALL@": (
                    "add_binary_tile(0, 1, 0);"
                    if operation == "add" else "div_binary_tile(0, 1, 0);"
                ),
            },
        )
        _set_native_unpack_fp32(_program, {0, 1, 2})
        return result

    op._ttl_program = _program
    op.safe_default = False
    return op
