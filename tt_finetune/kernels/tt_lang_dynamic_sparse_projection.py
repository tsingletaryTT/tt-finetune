"""Parallel selected-block expert projection for Blackhole P150a.

The upstream TTNN sparse matmul assigns output columns to cores and then
serializes every active token/expert block behind those cores.  This program
uses the opposite ownership: every (token block, expert, output tile) is an
independent job.  The reader checks the device routing bitmap, skips inactive
jobs without doing a matmul, and spreads the active jobs over the complete
worker grid.  Expert weights remain in their canonical expert-major storage;
no per-assignment or per-block weight tensor is materialized.

The first implementation intentionally covers a single routing tile (up to 32
token blocks and 32 experts).  Unsupported shapes fall back before lowering.
"""

from __future__ import annotations

import hashlib
import inspect
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import (
    BLACKHOLE_P150A_CORE_COUNT,
    BLACKHOLE_P150A_GRID,
)
from .tt_lang_lora import TILE_SIZE


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=24)
def build_tt_lang_dynamic_sparse_projection(
    *,
    token_blocks: int,
    expert_count: int,
    input_features: int,
    output_features: int,
) -> Callable:
    if min(token_blocks, expert_count, input_features, output_features) <= 0:
        raise ValueError("dynamic sparse projection dimensions must be positive")
    if token_blocks > TILE_SIZE or expert_count > TILE_SIZE:
        raise ValueError("parallel sparse projection currently supports one routing tile")
    if input_features % TILE_SIZE or output_features % TILE_SIZE:
        raise ValueError("dynamic sparse projection features must be tile aligned")

    k_tiles = input_features // TILE_SIZE
    n_tiles = output_features // TILE_SIZE
    candidate_blocks = token_blocks * expert_count
    total_jobs = candidate_blocks * n_tiles
    jobs_per_node = -(-total_jobs // BLACKHOLE_P150A_CORE_COUNT)
    if jobs_per_node > 64:
        raise ValueError("parallel sparse projection exceeds the bounded job owner")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - silicon only
            raise RuntimeError("parallel sparse projection requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    variant = hashlib.sha256(
        (
            f"{token_blocks}:{expert_count}:{input_features}:"
            f"{output_features}:parallel-dynamic-sparse-v1"
        ).encode("ascii")
    ).hexdigest()[:16]
    operation_options = {
        "grid": BLACKHOLE_P150A_GRID,
        "fp32_dest_acc_en": True,
        "options": "--ttl-matmul-full-fp32",
    }
    if "math_approx_mode" in inspect.signature(ttl.operation).parameters:
        operation_options["math_approx_mode"] = False

    @ttl.operation(**operation_options)
    def _program(inputs, expert_weight_t, routing, output):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        local_jobs = -(-total_jobs // total_nodes)
        x_dfb = ttl.make_dataflow_buffer_like(
            inputs, shape=(1, k_tiles), block_count=2
        )
        weight_dfb = ttl.make_dataflow_buffer_like(
            expert_weight_t, shape=(k_tiles, 1), block_count=2
        )
        route_dfb = ttl.make_dataflow_buffer_like(
            routing, shape=(1, 1), block_count=1
        )
        result_dfb = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=2
        )

        @ttl.datamovement()
        def read():
            # This body establishes tensor accessors/CBs.  It is replaced by
            # the routing-aware TT-Metal reader below before compilation.
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            with route_dfb.reserve() as route:
                tx_route = ttl.copy(
                    routing[0, 0, 0:token_blocks, 0:expert_count],
                    route,
                )
                tx_route.wait()
            for local_job in range(local_jobs):
                job = node_index * local_jobs + local_job
                if job < total_jobs:
                    block = job // n_tiles
                    n_tile = job - block * n_tiles
                    expert = block - (block // expert_count) * expert_count
                    with (
                        x_dfb.reserve() as x_block,
                        weight_dfb.reserve() as weight,
                    ):
                        tx_x = ttl.copy(
                            inputs[block : block + 1, 0:k_tiles], x_block
                        )
                        weight_row = expert * k_tiles
                        tx_w = ttl.copy(
                            expert_weight_t[
                                weight_row : weight_row + k_tiles,
                                n_tile : n_tile + 1,
                            ],
                            weight,
                        )
                        tx_x.wait()
                        tx_w.wait()

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(local_jobs):
                job = node_index * local_jobs + local_job
                if job < total_jobs:
                    with x_dfb.wait() as x_block, weight_dfb.wait() as weight:
                        with result_dfb.reserve() as result:
                            result.store(ttl.matmul(x_block, weight))

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(local_jobs):
                job = node_index * local_jobs + local_job
                if job < total_jobs:
                    block = job // n_tiles
                    n_tile = job - block * n_tiles
                    with result_dfb.wait() as result:
                        tx = ttl.copy(
                            result,
                            output[block : block + 1, n_tile : n_tile + 1],
                        )
                        tx.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.dynamic_sparse_projection.{variant}",
        arg_roles=("in", "in", "in", "out"),
        version_tag=f"dynamic-sparse-projection-{variant}",
    )
    def operation(inputs, expert_weight_t, routing, output):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(inputs, expert_weight_t, routing, output)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 4:
                    raise RuntimeError(
                        "parallel sparse projection expected four circular buffers"
                    )
                for (_, thread), config in zip(
                    compiled.kernel_paths, compiled.kernel_configs
                ):
                    if thread != "compute":
                        continue
                    if hasattr(config, "math_fidelity"):
                        enum_type = type(config.math_fidelity)
                        config.math_fidelity = getattr(enum_type, "HiFi4")
                    if hasattr(config, "math_approx_mode"):
                        config.math_approx_mode = False
                if os.environ.get(
                    "TT_FINETUNE_DYNAMIC_MOE_PARALLEL_SPARSE_SCAFFOLD"
                ) == "1":
                    continue
                grid = compiled.core_ranges.bounding_box().grid_size()
                replacements = {
                    "@JOBS_PER_NODE@": str(jobs_per_node),
                    "@TOTAL_JOBS@": str(total_jobs),
                    "@TOKEN_BLOCKS@": str(token_blocks),
                    "@EXPERTS@": str(expert_count),
                    "@K_TILES@": str(k_tiles),
                    "@N_TILES@": str(n_tiles),
                    "@GRID_COLS@": str(int(grid.x)),
                }
                names = {
                    "noc": (
                        "dynamic_sparse_projection_reader.cpp",
                        "dynamic_sparse_projection_writer.cpp",
                    ),
                    "compute": ("dynamic_sparse_projection_compute.cpp",),
                }
                noc_index = 0
                for (path, thread) in compiled.kernel_paths:
                    source_name = (
                        names["compute"][0]
                        if thread == "compute"
                        else names["noc"][noc_index]
                    )
                    if thread != "compute":
                        noc_index += 1
                    source = (
                        Path(__file__).with_name("cpp") / source_name
                    ).read_text(encoding="utf-8")
                    for marker, value in replacements.items():
                        source = source.replace(marker, value)
                    if any(marker in source for marker in replacements):
                        raise RuntimeError(
                            f"unexpanded placeholder in {source_name}"
                        )
                    Path(path).write_text(source, encoding="utf-8")
            break
        return result

    operation.candidate_blocks = candidate_blocks
    operation.total_jobs = total_jobs
    operation.jobs_per_node = jobs_per_node
    operation.safe_default = False
    operation.experimental_backend = "parallel_dynamic_sparse_v1"
    operation._ttl_program = _program
    return operation


__all__ = ["build_tt_lang_dynamic_sparse_projection"]
