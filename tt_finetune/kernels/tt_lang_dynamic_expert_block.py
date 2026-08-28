"""Exact block-parallel frozen expert projection for Blackhole P150a.

Each logical tile row is one homogeneous selected-expert assignment block.
Unlike TTNN's current sparse 1-D program, blocks are independent jobs spread
over the full worker grid instead of a batch loop serialized behind one input
multicast.  The weight tensor is already gathered once per block by the
device graph; the TT-Metal program keeps the established two-tile FP32
reduction/one-final-BF16-pack schedule.
"""

from __future__ import annotations

import hashlib
import inspect
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import (
    BLACKHOLE_P150A_CORE_COUNT,
    BLACKHOLE_P150A_GRID,
)
from .tt_lang_lora import TILE_SIZE, _FP32_DTYPE_ONLY_TENSOR


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=24)
def build_tt_lang_dynamic_expert_block_forward(
    *,
    block_count: int,
    expert_count: int,
    input_features: int,
    output_features: int,
    reduction_tiles_per_phase: int = 2,
    accumulation_schedule: str = "explicit_partial",
    math_approx_mode: bool = False,
) -> Callable:
    dimensions = (block_count, expert_count, input_features, output_features)
    if min(dimensions) <= 0:
        raise ValueError("expert block dimensions must be positive")
    if input_features % TILE_SIZE or output_features % TILE_SIZE:
        raise ValueError("expert block features must be tile aligned")
    k_tiles = input_features // TILE_SIZE
    n_tiles = output_features // TILE_SIZE
    if reduction_tiles_per_phase <= 0 or k_tiles % reduction_tiles_per_phase:
        raise ValueError("expert reduction phase must divide K tiles")
    reduction_blocks = k_tiles // reduction_tiles_per_phase
    if accumulation_schedule not in {
        "explicit_partial",
        "retain_destination",
        "packer_l1",
    }:
        raise ValueError("unsupported expert block accumulation schedule")
    total_jobs = block_count * n_tiles
    jobs_per_node = -(-total_jobs // BLACKHOLE_P150A_CORE_COUNT)
    if jobs_per_node > 32:
        raise ValueError("expert block wave count exceeds the bounded CB owner")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - silicon only
            raise RuntimeError("native expert blocks require TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    variant = hashlib.sha256(
        (
            f"{block_count}:{expert_count}:{input_features}:{output_features}:"
            f"{reduction_tiles_per_phase}:{accumulation_schedule}:"
            f"approx{int(math_approx_mode)}:dynamic-expert-block-v3"
        ).encode("ascii")
    ).hexdigest()[:16]

    operation_options = {
        "grid": BLACKHOLE_P150A_GRID,
        "fp32_dest_acc_en": True,
        "options": "--ttl-matmul-full-fp32",
    }
    if "math_approx_mode" in inspect.signature(ttl.operation).parameters:
        operation_options["math_approx_mode"] = math_approx_mode

    @ttl.operation(**operation_options)
    def _program(inputs, expert_weight_t, block_experts, output):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        local_jobs = -(-total_jobs // total_nodes)
        x_dfb = ttl.make_dataflow_buffer_like(
            inputs, shape=(1, reduction_tiles_per_phase), block_count=2
        )
        weight_dfb = ttl.make_dataflow_buffer_like(
            expert_weight_t,
            shape=(reduction_tiles_per_phase, 1),
            block_count=2,
        )
        partial_dfb = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=2
        )
        result_dfb = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=2
        )
        expert_dfb = ttl.make_dataflow_buffer_like(
            block_experts, shape=(1, 2), block_count=1
        )

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            with expert_dfb.reserve() as expert_ids:
                tx_ids = ttl.copy(
                    block_experts[0:1, 0:block_count], expert_ids
                )
                tx_ids.wait()
            for local_job in range(local_jobs):
                job = node_index * local_jobs + local_job
                if job < total_jobs:
                    block = job // n_tiles
                    n_tile = job - block * n_tiles
                    for phase in range(reduction_blocks):
                        k_start = phase * reduction_tiles_per_phase
                        # TT-Lang scaffold only.  The routing-aware reader
                        # replaces this static owner before kernel build.
                        weight_k_start = (
                            (block - (block // expert_count) * expert_count)
                            * k_tiles
                            + k_start
                        )
                        with (
                            x_dfb.reserve() as x_block,
                            weight_dfb.reserve() as weight,
                        ):
                            tx_x = ttl.copy(
                                inputs[
                                    block : block + 1,
                                    k_start : k_start + reduction_tiles_per_phase,
                                ],
                                x_block,
                            )
                            tx_w = ttl.copy(
                                expert_weight_t[
                                    weight_k_start : weight_k_start
                                    + reduction_tiles_per_phase,
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
                    with partial_dfb.reserve() as partial:
                        partial.store(ttl.block.fill(0, shape=partial.shape))
                    for _ in range(reduction_blocks):
                        with (
                            x_dfb.wait() as x_block,
                            weight_dfb.wait() as weight,
                            partial_dfb.wait() as previous,
                        ):
                            with partial_dfb.reserve() as next_value:
                                next_value.store(
                                    previous + ttl.matmul(x_block, weight)
                                )
                    with partial_dfb.wait() as partial:
                        with result_dfb.reserve() as result:
                            result.store(partial)

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
        operation_id=f"tt_finetune.dynamic_expert_block.{variant}",
        arg_roles=("in", "in", "in", "out"),
        version_tag=f"dynamic-expert-block-{variant}",
    )
    def operation(inputs, expert_weight_t, block_experts, output):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(inputs, expert_weight_t, block_experts, output)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 5:
                    raise RuntimeError(
                        "exact expert block expected five circular buffers"
                    )
                compiled.cb_configs[2].tensor = _FP32_DTYPE_ONLY_TENSOR
                for (_, thread), config in zip(
                    compiled.kernel_paths, compiled.kernel_configs
                ):
                    if thread != "compute":
                        continue
                    if hasattr(config, "math_fidelity"):
                        enum_type = type(config.math_fidelity)
                        config.math_fidelity = getattr(enum_type, "HiFi4")
                    if hasattr(config, "math_approx_mode"):
                        config.math_approx_mode = bool(math_approx_mode)
                grid = compiled.core_ranges.bounding_box().grid_size()
                replacements = {
                    "@JOBS_PER_NODE@": str(jobs_per_node),
                    "@TOTAL_JOBS@": str(total_jobs),
                    "@BLOCK_COUNT@": str(block_count),
                    "@EXPERT_COUNT@": str(expert_count),
                    "@K_TILES@": str(n_tiles),
                    "@N_TILES@": str(k_tiles),
                    "@REDUCTION_TILES_PER_PHASE@": str(
                        reduction_tiles_per_phase
                    ),
                    "@GRID_COLS@": str(int(grid.x)),
                }
                reader_paths = [
                    Path(path)
                    for path, thread in compiled.kernel_paths
                    if thread == "noc"
                ]
                if len(reader_paths) != 2:
                    raise RuntimeError("expert block expected reader and writer kernels")
                source = (
                    Path(__file__).with_name("cpp")
                    / "dynamic_expert_block_reader.cpp"
                ).read_text(encoding="utf-8")
                for marker, value in replacements.items():
                    source = source.replace(marker, value)
                if any(marker in source for marker in replacements):
                    raise RuntimeError("unexpanded expert block placeholder")
                reader_paths[0].write_text(source, encoding="utf-8")
                compute_paths = [
                    Path(path)
                    for path, thread in compiled.kernel_paths
                    if thread == "compute"
                ]
                if len(compute_paths) != 1:
                    raise RuntimeError("expert block expected one compute kernel")
                compute_source = (
                    Path(__file__).with_name("cpp")
                    / "frozen_linear_backward_dx_exact_compute.cpp"
                ).read_text(encoding="utf-8")
                compute_replacements = {
                    "@JOBS_PER_NODE@": str(jobs_per_node),
                    "@TOTAL_JOBS@": str(total_jobs),
                    "@K_TILES@": str(n_tiles),
                    "@N_TILES@": str(k_tiles),
                    "@REDUCTION_TILES_PER_PHASE@": str(
                        reduction_tiles_per_phase
                    ),
                    "@RETAIN_DESTINATION_ACROSS_REDUCTION@": str(
                        accumulation_schedule == "retain_destination"
                    ).lower(),
                    "@PACKER_L1_ACCUMULATE_ACROSS_REDUCTION@": str(
                        accumulation_schedule == "packer_l1"
                    ).lower(),
                    "@GRID_COLS@": str(int(grid.x)),
                }
                for marker, value in compute_replacements.items():
                    compute_source = compute_source.replace(marker, value)
                if any(
                    marker in compute_source
                    for marker in compute_replacements
                ):
                    raise RuntimeError("unexpanded expert compute placeholder")
                compute_paths[0].write_text(compute_source, encoding="utf-8")
            break
        return result

    operation.block_count = block_count
    operation.total_jobs = total_jobs
    operation.jobs_per_node = jobs_per_node
    operation.reduction_tiles_per_phase = reduction_tiles_per_phase
    operation.accumulation_schedule = accumulation_schedule
    operation.math_approx_mode = math_approx_mode
    operation.safe_default = False
    operation.experimental_backend = "dynamic_expert_block_exact_v1"
    operation._ttl_program = _program
    return operation


__all__ = ["build_tt_lang_dynamic_expert_block_forward"]
