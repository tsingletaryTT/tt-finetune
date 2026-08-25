"""Planner for a fused frozen-base-linear + exact LoRA TT-Metal program.

This is deliberately separate from the accepted LoRA-only planner. A frozen
linear projection has enough output-feature work to distribute jobs over both
M and N. Each ``(M tile, N block)`` job recomputes the small rank projection
H, trading a little redundant rank work for much better Blackhole core
occupancy and no cross-core H multicast in the first implementation.
"""

from __future__ import annotations

import math
import hashlib
import inspect
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_lora import (
    BF16_TILE_BYTES,
    TILE_SIZE,
    _FP32_DTYPE_ONLY_TENSOR,
)


ttl = None
tt_lang_operation = None

BLACKHOLE_P150A_GRID = (11, 10)
BLACKHOLE_P150A_CORE_COUNT = (
    BLACKHOLE_P150A_GRID[0] * BLACKHOLE_P150A_GRID[1]
)


def _exact_biasless_add_source() -> str:
    return r"""
    add_tiles_init(cb_base, cb_scaled_delta);
    disable_fp32_dest_acc();
    for (uint32_t local_job = 0; local_job < jobs_per_node; ++local_job) {
        const uint32_t job = node_index * jobs_per_node + local_job;
        if (job >= total_jobs) {
            continue;
        }
        base.wait_front(n_tiles_per_block);
        scaled_delta.wait_front(n_tiles_per_block);
        output.reserve_back(n_tiles_per_block);
        for (uint32_t tile = 0; tile < n_tiles_per_block; ++tile) {
            tile_regs_acquire();
            add_tiles_bf16(
                cb_base, cb_scaled_delta, tile, tile, 0);
            tile_regs_commit_bf16();
            tile_regs_wait();
            pack_reconfig_data_format_bf16(cb_output);
            pack_tile_bf16(0, cb_output, tile);
            tile_regs_release_bf16();
        }
        output.push_back(n_tiles_per_block);
        scaled_delta.pop_front(n_tiles_per_block);
        base.pop_front(n_tiles_per_block);
    }
    enable_fp32_dest_acc();
"""


def _exact_bf16_bias_then_lora_add_source() -> str:
    """Match ``BF16(base) + bias`` before the final LoRA add.

    Qwen's full StableHLO graph materializes the frozen matmul as BF16, adds
    the BF16 bias, and only then adds the BF16-scaled LoRA delta.  Keeping the
    base accumulator in FP32 through the bias add is mathematically close but
    does not preserve that graph's two explicit rounding boundaries.
    """

    return r"""
    add_tiles_init(cb_base, cb_bias);
    disable_fp32_dest_acc();
    for (uint32_t local_job = 0; local_job < jobs_per_node; ++local_job) {
        const uint32_t job = node_index * jobs_per_node + local_job;
        if (job >= total_jobs) {
            continue;
        }
        base.wait_front(n_tiles_per_block);
        bias.wait_front(n_tiles_per_block);
        biased_base.reserve_back(n_tiles_per_block);
        for (uint32_t tile = 0; tile < n_tiles_per_block; ++tile) {
            tile_regs_acquire();
            add_tiles_bf16(cb_base, cb_bias, tile, tile, 0);
            tile_regs_commit_bf16();
            tile_regs_wait();
            pack_reconfig_data_format_bf16(cb_biased_base);
            pack_tile_bf16(0, cb_biased_base, tile);
            tile_regs_release_bf16();
        }
        biased_base.push_back(n_tiles_per_block);
        bias.pop_front(n_tiles_per_block);
        base.pop_front(n_tiles_per_block);
    }

    add_tiles_init(cb_biased_base, cb_scaled_delta);
    for (uint32_t local_job = 0; local_job < jobs_per_node; ++local_job) {
        const uint32_t job = node_index * jobs_per_node + local_job;
        if (job >= total_jobs) {
            continue;
        }
        biased_base.wait_front(n_tiles_per_block);
        scaled_delta.wait_front(n_tiles_per_block);
        output.reserve_back(n_tiles_per_block);
        for (uint32_t tile = 0; tile < n_tiles_per_block; ++tile) {
            tile_regs_acquire();
            add_tiles_bf16(
                cb_biased_base, cb_scaled_delta, tile, tile, 0);
            tile_regs_commit_bf16();
            tile_regs_wait();
            pack_reconfig_data_format_bf16(cb_output);
            pack_tile_bf16(0, cb_output, tile);
            tile_regs_release_bf16();
        }
        output.push_back(n_tiles_per_block);
        scaled_delta.pop_front(n_tiles_per_block);
        biased_base.pop_front(n_tiles_per_block);
    }
    enable_fp32_dest_acc();
"""


@dataclass(frozen=True)
class FrozenLinearLoRAPlan:
    n_tiles_per_block: int
    total_jobs: int
    scheduling_waves: int
    estimated_relative_work_per_wave: int
    estimated_l1_bytes: int
    l1_budget_bytes: int


def estimate_frozen_linear_lora_l1_bytes(
    *,
    n_tiles_per_block: int,
    rank_tiles: int = 1,
    k_tiles_per_phase: int = 2,
    base_k_tiles_per_phase: int | None = None,
    base_partial_fp32: bool = True,
    staged_jobs_per_node: int = 2,
    has_bias: bool = False,
) -> int:
    """Conservative CB estimate in BF16-equivalent tiles.

    FP32 partial buffers count as two BF16 tiles. The estimate includes
    separate long-K partial storage for H and the frozen base projection plus
    complete N-block staging for the exact BF16 scale/add boundary.
    """

    base_k_tiles = (
        k_tiles_per_phase
        if base_k_tiles_per_phase is None
        else base_k_tiles_per_phase
    )
    values = (
        n_tiles_per_block,
        rank_tiles,
        k_tiles_per_phase,
        base_k_tiles,
        staged_jobs_per_node,
    )
    if min(values) <= 0:
        raise ValueError("all tile dimensions must be positive")
    n_tiles = n_tiles_per_block
    tiles = (
        2 * k_tiles_per_phase  # H X, depth two
        + 2 * rank_tiles * k_tiles_per_phase  # A, depth two
        + 4 * rank_tiles  # FP32 H partial, depth two
        + 2 * rank_tiles  # BF16 H
        + 2 * base_k_tiles  # base X, depth two
        + 2 * n_tiles * base_k_tiles  # frozen W, depth two
        + (4 if base_partial_fp32 else 2) * n_tiles
        # frozen-linear partial, FP32 or BF16, depth two
        + staged_jobs_per_node * n_tiles  # staged BF16 base
        + 2 * n_tiles * rank_tiles  # B, depth two
        + 2 * n_tiles  # LoRA delta
        + staged_jobs_per_node * n_tiles  # staged scaled delta
        + 2 * n_tiles  # final output
    )
    # Keep these two CBs in both generated variants. TT-Lang resolves both
    # sides of captured compile-time branches, so defining them unconditionally
    # preserves the accepted five-argument biasless ABI while the exact C++
    # replacement simply leaves the buffers unused there.
    tiles += 2 * max(2, staged_jobs_per_node) * n_tiles
    return tiles * BF16_TILE_BYTES


def plan_frozen_linear_lora(
    *,
    tokens: int,
    input_features: int,
    output_features: int,
    physical_rank: int,
    core_count: int = BLACKHOLE_P150A_CORE_COUNT,
    worker_l1_bytes: int = int(1.46 * 1024 * 1024),
    l1_fraction: float = 0.75,
    maximum_n_tiles_per_block: int = 24,
) -> FrozenLinearLoRAPlan:
    """Choose the first 2-D job schedule for silicon autotuning.

    The cost model minimizes ``waves * (N block + rank tiles)``. Base-linear
    work grows with the N block; recomputed H work grows with rank. This keeps
    enough independent jobs to occupy Blackhole without blindly choosing the
    smallest N block and repeating the long-K rank reduction excessively.
    """

    dimensions = (tokens, input_features, output_features, physical_rank)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError(
            "all fused frozen-linear dimensions must be positive /32"
        )
    if input_features // TILE_SIZE % 2:
        raise ValueError("exact long-K reduction requires an even K tile count")
    if physical_rank > 3 * TILE_SIZE:
        raise ValueError(
            "grouped fused frozen-linear planner supports at most 3 rank tiles"
        )
    if core_count <= 0 or worker_l1_bytes <= 0:
        raise ValueError("core count and worker L1 must be positive")
    if not 0.0 < l1_fraction <= 1.0:
        raise ValueError("l1_fraction must be in (0, 1]")

    m_tiles = tokens // TILE_SIZE
    n_tiles = output_features // TILE_SIZE
    rank_tiles = physical_rank // TILE_SIZE
    budget = int(worker_l1_bytes * l1_fraction)
    candidates: list[tuple[int, int, int, int, int]] = []
    for n_block in range(
        1, min(n_tiles, maximum_n_tiles_per_block) + 1
    ):
        if n_tiles % n_block:
            continue
        # Four-wide destination subblocks avoid a tail variant in the first
        # exact base-projection compute kernel.
        if n_block > 4 and n_block % 4:
            continue
        jobs = m_tiles * (n_tiles // n_block)
        waves = math.ceil(jobs / core_count)
        required = estimate_frozen_linear_lora_l1_bytes(
            n_tiles_per_block=n_block,
            rank_tiles=rank_tiles,
            staged_jobs_per_node=waves,
        )
        if required > budget:
            continue
        relative = waves * (n_block + rank_tiles)
        candidates.append(
            (relative, -n_block, jobs, waves, required)
        )
    if not candidates:
        raise ValueError("no fused frozen-linear schedule fits worker L1")
    relative, negative_n_block, jobs, waves, required = min(candidates)
    return FrozenLinearLoRAPlan(
        n_tiles_per_block=-negative_n_block,
        total_jobs=jobs,
        scheduling_waves=waves,
        estimated_relative_work_per_wave=relative,
        estimated_l1_bytes=required,
        l1_budget_bytes=budget,
    )


@lru_cache(maxsize=32)
def build_tt_lang_frozen_linear_lora_forward_prototype(
    *,
    scale: float,
    tokens: int,
    input_features: int,
    output_features: int,
    physical_rank: int,
    n_tiles_per_block: int | None = None,
    lora_k_tiles_per_phase: int = 2,
    base_k_tiles_per_phase: int = 2,
    base_partial_fp32: bool = True,
    math_approx_mode: bool = False,
    exact_reduction: bool = False,
    lora_init_schedule: int = 0,
    has_bias: bool = False,
    bias_after_bf16_base: bool = False,
    worker_grid: tuple[int, int] = BLACKHOLE_P150A_GRID,
) -> Callable:
    """Build the first compiler-generated 2-D fused-forward prototype.

    This operation is intentionally not the exact production kernel yet. It
    validates the M×N job mapping and measures whether replacing the frozen
    base projection crosses dispatch break-even before hand-writing its exact
    two-K-tile compute schedule.
    """

    plan = plan_frozen_linear_lora(
        tokens=tokens,
        input_features=input_features,
        output_features=output_features,
        physical_rank=physical_rank,
        core_count=int(worker_grid[0]) * int(worker_grid[1]),
    )
    worker_count = int(worker_grid[0]) * int(worker_grid[1])
    m_tiles = tokens // TILE_SIZE
    n_tiles = output_features // TILE_SIZE
    k_tiles = input_features // TILE_SIZE
    rank_tiles = physical_rank // TILE_SIZE
    n_block = (
        plan.n_tiles_per_block
        if n_tiles_per_block is None
        else n_tiles_per_block
    )
    if n_block <= 0 or n_tiles % n_block:
        raise ValueError("invalid explicit fused-forward N block")
    staged_jobs_per_node = -(
        -(m_tiles * (n_tiles // n_block)) // worker_count
    )
    if (
        estimate_frozen_linear_lora_l1_bytes(
            n_tiles_per_block=n_block,
            rank_tiles=rank_tiles,
            base_k_tiles_per_phase=base_k_tiles_per_phase,
            base_partial_fp32=base_partial_fp32,
            staged_jobs_per_node=staged_jobs_per_node,
            has_bias=has_bias,
        )
        > plan.l1_budget_bytes
    ):
        raise ValueError("invalid explicit fused-forward N block")
    if exact_reduction and n_block != 1:
        raise ValueError(
            "accepted exact fused frozen-linear kernel requires N block 1"
        )
    n_blocks = n_tiles // n_block
    if lora_k_tiles_per_phase <= 0 or k_tiles % lora_k_tiles_per_phase:
        raise ValueError("LoRA K phase must be a positive K-tile divisor")
    k_block = lora_k_tiles_per_phase
    k_blocks = k_tiles // k_block
    if (
        base_k_tiles_per_phase <= 0
        or k_tiles % base_k_tiles_per_phase
    ):
        raise ValueError("base K phase must be a positive K-tile divisor")
    base_k_block = base_k_tiles_per_phase
    base_k_blocks = k_tiles // base_k_block
    # P150a exposes an 11x10 logical worker grid. Base and scaled-delta tiles
    # remain queued
    # across all local jobs so the exact kernel switches destination mode
    # only once per core, avoiding SyncHalf corruption between jobs.
    staged_jobs_per_node = -(
        -(m_tiles * n_blocks) // worker_count
    )
    if exact_reduction and staged_jobs_per_node > 32:
        raise ValueError(
            "exact fused frozen-linear staging exceeds TT-Lang's 32-block "
            "CB limit; split/overflow staging is required"
        )
    if lora_init_schedule not in (0, 1, 2, 4):
        raise ValueError(
            "LoRA init schedule must be 0, 1, 2 or 4; schedule 3 is "
            "rejected because it deadlocks the resident CB protocol"
        )

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import (
                tt_lang_operation as tt_lang_operation_decorator,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The fused frozen-linear prototype requires TT-XLA"
            ) from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    scale_bf16 = float(
        torch.tensor(scale, dtype=torch.bfloat16).float().item()
    )
    variant = hashlib.sha256(
        (
            f"{scale_bf16:.17g}:{tokens}:{input_features}:"
            f"{output_features}:{physical_rank}:{n_block}:"
            f"{k_block}:"
            f"{base_k_block}:{int(base_partial_fp32)}:"
            f"{int(math_approx_mode)}:"
            f"{int(exact_reduction)}:"
            f"{lora_init_schedule}:"
            f"{int(has_bias)}:"
            f"{int(bias_after_bf16_base)}:"
            f"{worker_grid[0]}x{worker_grid[1]}:"
            "frozen-linear-prototype-v8-explicit-bias-boundaries"
        ).encode("ascii")
    ).hexdigest()[:16]
    scale_bits = struct.unpack(
        "<I", struct.pack("<f", scale_bf16)
    )[0]

    operation_options = {
        "grid": worker_grid,
        "fp32_dest_acc_en": True,
        "options": "--ttl-matmul-full-fp32",
    }
    if "math_approx_mode" in inspect.signature(
        ttl.operation
    ).parameters:
        operation_options["math_approx_mode"] = math_approx_mode
    elif math_approx_mode:
        raise RuntimeError(
            "This TT-Lang build cannot request math_approx_mode=True"
        )

    # TT-Lang's AST frontend accepts captured integer constants in control
    # flow, but does not classify a captured Python bool as an integer value.
    bias_enabled = int(has_bias)

    @ttl.operation(**operation_options)
    def _fused_program(
        inputs,
        frozen_weight,
        bias,
        lora_a,
        lora_b,
        arena,
    ):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        total_jobs = m_tiles * n_blocks
        jobs_per_node = -(-total_jobs // total_nodes)

        h_x_dfb = ttl.make_dataflow_buffer_like(
            inputs, shape=(1, k_block), block_count=2
        )
        a_dfb = ttl.make_dataflow_buffer_like(
            lora_a, shape=(rank_tiles, k_block), block_count=2
        )
        h_acc_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=2
        )
        h_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=2
        )
        saved_h_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=1
        )
        base_x_dfb = ttl.make_dataflow_buffer_like(
            inputs, shape=(1, base_k_block), block_count=2
        )
        w_dfb = ttl.make_dataflow_buffer_like(
            frozen_weight,
            shape=(base_k_block, n_block),
            block_count=2,
        )
        base_acc_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, n_block), block_count=2
        )
        base_dfb = ttl.make_dataflow_buffer_like(
            arena,
            shape=(1, n_block),
            block_count=max(2, staged_jobs_per_node),
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            lora_b, shape=(n_block, rank_tiles), block_count=2
        )
        delta_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, n_block), block_count=2
        )
        scaled_delta_dfb = ttl.make_dataflow_buffer_like(
            arena,
            shape=(1, n_block),
            block_count=max(2, staged_jobs_per_node),
        )
        output_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, n_block), block_count=2
        )
        bias_dfb = ttl.make_dataflow_buffer_like(
            bias,
            shape=(1, n_block),
            block_count=max(2, staged_jobs_per_node),
        )
        biased_base_dfb = ttl.make_dataflow_buffer_like(
            arena,
            shape=(1, n_block),
            block_count=max(2, staged_jobs_per_node),
        )

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(jobs_per_node):
                job = node_index * jobs_per_node + local_job
                if job < total_jobs:
                    m_tile = job // n_blocks
                    n_index = job - m_tile * n_blocks
                    n_start = n_index * n_block
                    for k_index in range(k_blocks):
                        k_start = k_index * k_block
                        with (
                            h_x_dfb.reserve() as x_block,
                            a_dfb.reserve() as a_block,
                        ):
                            tx_x = ttl.copy(
                                inputs[
                                    m_tile : m_tile + 1,
                                    k_start : k_start + k_block,
                                ],
                                x_block,
                            )
                            tx_a = ttl.copy(
                                lora_a[
                                    0:rank_tiles,
                                    k_start : k_start + k_block,
                                ],
                                a_block,
                            )
                            tx_x.wait()
                            tx_a.wait()
                    for k_index in range(base_k_blocks):
                        k_start = k_index * base_k_block
                        with (
                            base_x_dfb.reserve() as x_block,
                            w_dfb.reserve() as w_block,
                        ):
                            tx_x = ttl.copy(
                                inputs[
                                    m_tile : m_tile + 1,
                                    k_start : k_start + base_k_block,
                                ],
                                x_block,
                            )
                            tx_w = ttl.copy(
                                frozen_weight[
                                    k_start : k_start + base_k_block,
                                    n_start : n_start + n_block,
                                ],
                                w_block,
                            )
                            tx_w.wait()
                            tx_x.wait()
                    with b_dfb.reserve() as b_block:
                        tx_b = ttl.copy(
                            lora_b[
                                n_start : n_start + n_block,
                                0:rank_tiles,
                            ],
                            b_block,
                        )
                        tx_b.wait()
                    if bias_enabled == 1:
                        with bias_dfb.reserve() as bias_block:
                            tx_bias = ttl.copy(
                                bias[0:1, n_start : n_start + n_block],
                                bias_block,
                            )
                            tx_bias.wait()

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(jobs_per_node):
                job = node_index * jobs_per_node + local_job
                if job < total_jobs:
                    m_tile = job // n_blocks
                    n_index = job - m_tile * n_blocks
                    with h_acc_dfb.reserve() as partial:
                        partial.store(
                            ttl.block.fill(0, shape=partial.shape)
                        )
                    for _ in range(k_blocks):
                        with (
                            h_x_dfb.wait() as x_block,
                            a_dfb.wait() as a_block,
                            h_acc_dfb.wait() as previous,
                        ):
                            with h_acc_dfb.reserve() as next_value:
                                next_value.store(
                                    previous
                                    + ttl.matmul(
                                        x_block,
                                        a_block,
                                        transpose_rhs=True,
                                    )
                                )
                    with h_acc_dfb.wait() as partial:
                        with h_dfb.reserve() as hidden:
                            hidden.store(partial)
                        if n_index == 0:
                            with saved_h_dfb.reserve() as saved:
                                saved.store(partial)

                    with base_acc_dfb.reserve() as partial:
                        partial.store(
                            ttl.block.fill(0, shape=partial.shape)
                        )
                    for _ in range(base_k_blocks):
                        with (
                            base_x_dfb.wait() as x_block,
                            w_dfb.wait() as w_block,
                            base_acc_dfb.wait() as previous,
                        ):
                            with base_acc_dfb.reserve() as next_value:
                                next_value.store(
                                    previous
                                    + ttl.matmul(x_block, w_block)
                                )
                    with base_acc_dfb.wait() as partial:
                        with base_dfb.reserve() as base:
                            base.store(partial)

                    with (
                        h_dfb.wait() as hidden,
                        b_dfb.wait() as b_block,
                    ):
                        with delta_dfb.reserve() as delta:
                            delta.store(
                                ttl.matmul(
                                    hidden,
                                    b_block,
                                    transpose_rhs=True,
                                )
                            )
                    with delta_dfb.wait() as delta:
                        with scaled_delta_dfb.reserve() as scaled:
                            scaled.store(delta * scale_bf16)
                    with (
                        base_dfb.wait() as base,
                        scaled_delta_dfb.wait() as scaled,
                    ):
                        if bias_enabled == 1:
                            with bias_dfb.wait() as bias_block:
                                with biased_base_dfb.reserve() as biased:
                                    biased.store(base + bias_block)
                            with biased_base_dfb.wait() as biased:
                                with output_dfb.reserve() as output:
                                    output.store(biased + scaled)
                        else:
                            with output_dfb.reserve() as output:
                                output.store(base + scaled)

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(jobs_per_node):
                job = node_index * jobs_per_node + local_job
                if job < total_jobs:
                    m_tile = job // n_blocks
                    n_index = job - m_tile * n_blocks
                    n_start = n_index * n_block
                    with output_dfb.wait() as output:
                        tx = ttl.copy(
                            output,
                            arena[
                                m_tile : m_tile + 1,
                                n_start : n_start + n_block,
                            ],
                        )
                        tx.wait()
                    if n_index == 0:
                        with saved_h_dfb.wait() as saved:
                            tx_h = ttl.copy(
                                saved,
                                arena[
                                    m_tile : m_tile + 1,
                                    n_tiles : n_tiles + rank_tiles,
                                ],
                            )
                            tx_h.wait()

    def invoke(inputs, frozen_weight, bias, lora_a, lora_b, arena):
        _clear_ttl_compile_wrapper_cache(_fused_program)
        result = _fused_program(
            inputs, frozen_weight, bias, lora_a, lora_b, arena
        )
        if exact_reduction:
            closure = _fused_program.__closure__ or ()
            for name, cell in zip(
                _fused_program.__code__.co_freevars,
                closure,
            ):
                if name != "cache" or not isinstance(
                    cell.cell_contents, dict
                ):
                    continue
                for compiled in cell.cell_contents.values():
                    expected_cb_count = 15
                    if len(compiled.cb_configs) != expected_cb_count:
                        raise RuntimeError(
                            "exact frozen-linear LoRA expected "
                            f"{expected_cb_count} CBs, got "
                            f"{len(compiled.cb_configs)}"
                        )
                    compiled.cb_configs[2].tensor = (
                        _FP32_DTYPE_ONLY_TENSOR
                    )
                    if base_partial_fp32:
                        compiled.cb_configs[7].tensor = (
                            _FP32_DTYPE_ONLY_TENSOR
                        )
                    compute_paths = [
                        Path(path)
                        for path, thread_type in compiled.kernel_paths
                        if thread_type == "compute"
                    ]
                    if len(compute_paths) != 1:
                        raise RuntimeError(
                            "exact frozen-linear LoRA expected one compute "
                            "kernel"
                        )
                    grid = compiled.core_ranges.bounding_box().grid_size()
                    total_jobs = m_tiles * n_blocks
                    total_nodes = int(grid.x) * int(grid.y)
                    rounded_bias = has_bias and bias_after_bf16_base
                    replacements = {
                        "@NUM_K_BLOCKS@": str(k_blocks),
                        "@H_K_BLOCK_TILES@": str(k_block),
                        "@RANK_TILES@": str(rank_tiles),
                        "@DELTA_RANK_TILES@": str(rank_tiles),
                        "@DELTA_H_TILE_OFFSET@": "0",
                        "@BASE_K_BLOCK_TILES@": str(base_k_block),
                        "@NUM_BASE_K_BLOCKS@": str(base_k_blocks),
                        "@BASE_TRANSPOSE_RHS@": "false",
                        "@NUM_N_BLOCKS@": str(n_blocks),
                        "@N_TILES_PER_BLOCK@": str(n_block),
                        "@JOBS_PER_NODE@": str(
                            -(-total_jobs // total_nodes)
                        ),
                        "@TOTAL_JOBS@": str(total_jobs),
                        "@GRID_COLS@": str(int(grid.x)),
                        "@SCALE_BITS@": str(scale_bits),
                        "@LORA_INIT_SCHEDULE@": str(lora_init_schedule),
                        "@BASE_BIAS_WAIT@": "",
                        "@BASE_BIAS_ACCUMULATE@": "",
                        "@BASE_BIAS_POP@": "",
                        "@BASE_FINAL_TARGET@": (
                            "cb_base_partial"
                            if has_bias and not rounded_bias
                            else "(last ? cb_base : cb_base_partial)"
                        ),
                        "@BASE_RESERVE@": (
                            "base_partial.reserve_back("
                            "n_tiles_per_block);"
                            if has_bias and not rounded_bias
                            else (
                                "if (last) { base.reserve_back("
                                "n_tiles_per_block); } else { "
                                "base_partial.reserve_back("
                                "n_tiles_per_block); }"
                            )
                        ),
                        "@BASE_PUSH@": (
                            "base_partial.push_back(n_tiles_per_block);"
                            if has_bias and not rounded_bias
                            else (
                                "if (last) { base.push_back("
                                "n_tiles_per_block); } else { "
                                "base_partial.push_back("
                                "n_tiles_per_block); }"
                            )
                        ),
                        "@BASE_BIAS_FINISH@": (
                            "if (last) {\n"
                            "                base_partial.wait_front("
                            "n_tiles_per_block);\n"
                            "                bias.wait_front("
                            "n_tiles_per_block);\n"
                            "                base.reserve_back("
                            "n_tiles_per_block);\n"
                            "                reconfig_data_format(\n"
                            "                    cb_w, cb_base_partial,\n"
                            "                    cb_base_x, cb_bias);\n"
                            "                add_tiles_init("
                            "cb_base_partial, cb_bias);\n"
                            "                tile_regs_acquire();\n"
                            "                for (uint32_t tile = 0; tile < "
                            "n_tiles_per_block; ++tile) {\n"
                            "                    add_tiles(\n"
                            "                        cb_base_partial, cb_bias,\n"
                            "                        tile, tile, tile);\n"
                            "                }\n"
                            "                tile_regs_commit();\n"
                            "                tile_regs_wait();\n"
                            "                PACK((pack_reconfig_data_format("
                            "cb_base)));\n"
                            "                pack_tile_block(\n"
                            "                    0, cb_base, "
                            "n_tiles_per_block);\n"
                            "                tile_regs_release();\n"
                            "                base.push_back("
                            "n_tiles_per_block);\n"
                            "                bias.pop_front("
                            "n_tiles_per_block);\n"
                            "                base_partial.pop_front("
                            "n_tiles_per_block);\n"
                            "            }"
                            if has_bias and not rounded_bias
                            else ""
                        ),
                        "@BIAS_DECLARATIONS@": (
                            "constexpr uint32_t cb_bias = "
                            "get_compile_time_arg_val(13);\n"
                            "    constexpr uint32_t cb_biased_base = "
                            "get_compile_time_arg_val(14);"
                            if has_bias
                            else ""
                        ),
                        "@BIAS_BUFFER_OBJECTS@": (
                            "CircularBuffer bias(cb_bias);\n"
                            "    CircularBuffer biased_base(cb_biased_base);"
                            if has_bias
                            else ""
                        ),
                        "@FINAL_ADD_BODY@": (
                            _exact_bf16_bias_then_lora_add_source()
                            if rounded_bias
                            else _exact_biasless_add_source()
                        ),
                    }
                    source = (
                        Path(__file__).with_name("cpp")
                        / "frozen_linear_lora_forward_exact_compute.cpp"
                    ).read_text(encoding="utf-8")
                    for placeholder, value in replacements.items():
                        source = source.replace(placeholder, value)
                    if any(key in source for key in replacements):
                        raise RuntimeError(
                            "unexpanded exact frozen-linear placeholder"
                        )
                    compute_paths[0].write_text(
                        source, encoding="utf-8"
                    )
                break
        return result

    operation_metadata = {
        "operation_id": f"tt_finetune.frozen_linear_lora.{variant}",
        "version_tag": f"frozen-linear-lora-{variant}",
    }
    if has_bias:
        @tt_lang_operation(
            **operation_metadata,
            arg_roles=("in", "in", "in", "in", "in", "out"),
        )
        def fused_forward(
            inputs, frozen_weight, bias, lora_a, lora_b, arena
        ):
            return invoke(
                inputs, frozen_weight, bias, lora_a, lora_b, arena
            )
    else:
        @tt_lang_operation(
            **operation_metadata,
            arg_roles=("in", "in", "in", "in", "in", "out"),
        )
        def fused_forward_with_dead_bias(
            inputs, frozen_weight, dead_bias, lora_a, lora_b, arena
        ):
            return invoke(
                inputs, frozen_weight, dead_bias, lora_a, lora_b, arena
            )

        def fused_forward(inputs, frozen_weight, lora_a, lora_b, arena):
            # The inner TT-Lang program has one six-tensor signature. Its bias
            # operand is dead in this variant; duplicating the read-only input
            # gives PJRT the required operand without aliasing the output and
            # preserves the established public five-argument operation ABI.
            return fused_forward_with_dead_bias(
                inputs, frozen_weight, inputs, lora_a, lora_b, arena
            )

    return fused_forward
