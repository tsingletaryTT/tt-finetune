"""Native logical-to-physical packing for a two-member LoRA group.

The execution arena is persistent and zero-initialized once:

    rows [0:64]                  = grouped A, width K
    rows [64:64 + 2N], cols 0:64 = block-diagonal grouped B

Only nonzero tiles are copied each step. Off-diagonal B tiles remain resident
zeros, avoiding graph-level padding, concatenation, and zero materialization.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID
from .tt_lang_lora import TILE_SIZE


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=32)
def build_tt_lang_grouped2_lora_pack(
    *,
    input_features: int,
    output_features: int,
) -> Callable:
    if (
        input_features <= 0
        or output_features <= 0
        or input_features % TILE_SIZE
        or output_features % TILE_SIZE
    ):
        raise ValueError("grouped pack dimensions must be positive /32")
    k_tiles = input_features // TILE_SIZE
    n_tiles = output_features // TILE_SIZE

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import (
                tt_lang_operation as tt_lang_operation_decorator,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("native grouped pack requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    variant = hashlib.sha256(
        (
            f"{input_features}:{output_features}:"
            "grouped2-logical-pack-v2-three-stage"
        ).encode("ascii")
    ).hexdigest()[:16]

    @ttl.operation(grid=BLACKHOLE_P150A_GRID)
    def _pack_program(a0, b0, a1, b1, arena):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        total_jobs = 2 * k_tiles + 2 * n_tiles
        jobs_per_node = -(-total_jobs // total_nodes)
        tile_dfb = ttl.make_dataflow_buffer_like(
            arena,
            shape=(1, 1),
            block_count=max(2, jobs_per_node),
        )
        staged_dfb = ttl.make_dataflow_buffer_like(
            arena,
            shape=(1, 1),
            block_count=max(2, jobs_per_node),
        )

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(jobs_per_node):
                job = node_index * jobs_per_node + local_job
                if job < total_jobs:
                    with tile_dfb.reserve() as tile:
                        if job < k_tiles:
                            tx = ttl.copy(
                                a0[0:1, job : job + 1], tile
                            )
                            tx.wait()
                        if job >= k_tiles and job < 2 * k_tiles:
                            k_tile = job - k_tiles
                            tx = ttl.copy(
                                a1[0:1, k_tile : k_tile + 1], tile
                            )
                            tx.wait()
                        if (
                            job >= 2 * k_tiles
                            and job < 2 * k_tiles + n_tiles
                        ):
                            n_tile = job - 2 * k_tiles
                            tx = ttl.copy(
                                b0[n_tile : n_tile + 1, 0:1], tile
                            )
                            tx.wait()
                        if job >= 2 * k_tiles + n_tiles:
                            n_tile = job - 2 * k_tiles - n_tiles
                            tx = ttl.copy(
                                b1[n_tile : n_tile + 1, 0:1], tile
                            )
                            tx.wait()

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(jobs_per_node):
                job = node_index * jobs_per_node + local_job
                if job < total_jobs:
                    with tile_dfb.wait() as tile:
                        with staged_dfb.reserve() as staged:
                            staged.store(tile)

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(jobs_per_node):
                job = node_index * jobs_per_node + local_job
                if job < total_jobs:
                    with staged_dfb.wait() as tile:
                        if job < k_tiles:
                            tx = ttl.copy(
                                tile, arena[0:1, job : job + 1]
                            )
                            tx.wait()
                        if job >= k_tiles and job < 2 * k_tiles:
                            k_tile = job - k_tiles
                            tx = ttl.copy(
                                tile, arena[1:2, k_tile : k_tile + 1]
                            )
                            tx.wait()
                        if (
                            job >= 2 * k_tiles
                            and job < 2 * k_tiles + n_tiles
                        ):
                            n_tile = job - 2 * k_tiles
                            tx = ttl.copy(
                                tile,
                                arena[
                                    2 + n_tile : 3 + n_tile,
                                    0:1,
                                ],
                            )
                            tx.wait()
                        if job >= 2 * k_tiles + n_tiles:
                            n_tile = job - 2 * k_tiles - n_tiles
                            tx = ttl.copy(
                                tile,
                                arena[
                                    2 + n_tiles + n_tile :
                                    3 + n_tiles + n_tile,
                                    1:2,
                                ],
                            )
                            tx.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.grouped2_lora_pack.{variant}",
        arg_roles=("in", "in", "in", "in", "out"),
        version_tag=f"grouped2-lora-pack-{variant}",
    )
    def grouped_pack(a0, b0, a1, b1, arena):
        _clear_ttl_compile_wrapper_cache(_pack_program)
        return _pack_program(a0, b0, a1, b1, arena)

    return grouped_pack
