"""Fused exact frozen-base dX plus LoRA dH/dB for Blackhole P150a.

One custom call returns an arena with this logical layout:

    rows [0:M],   cols [0:K]   = frozen-base dX
    rows [0:M],   cols [K:K+R] = LoRA dH
    rows [M:M+N], cols [0:R]   = LoRA dB

The unused bottom-right arena region is never read or written.  dW is absent
by construction because the base Linear is frozen.
"""

from __future__ import annotations

import hashlib
import re
import struct
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_backward import (
    estimate_frozen_linear_backward_l1_bytes,
)
from .tt_lang_frozen_linear_lora import (
    BLACKHOLE_P150A_CORE_COUNT,
    BLACKHOLE_P150A_GRID,
)
from .tt_lang_lora import TILE_SIZE, _FP32_DTYPE_ONLY_TENSOR
from .tt_lang_lora_backward_arena import (
    estimate_lora_backward_arena_l1_bytes,
)


ttl = None
tt_lang_operation = None


def estimate_fused_frozen_lora_backward_l1_bytes(
    *,
    token_tiles: int,
    input_tiles: int,
    output_tiles: int,
    rank_tiles: int = 1,
    base_reduction_tiles_per_phase: int = 1,
) -> int:
    if min(token_tiles, input_tiles, output_tiles, rank_tiles) <= 0:
        raise ValueError("all tile dimensions must be positive")
    base_jobs = token_tiles * input_tiles
    staged = -(-base_jobs // BLACKHOLE_P150A_CORE_COUNT)
    return estimate_frozen_linear_backward_l1_bytes(
        staged_jobs_per_node=staged,
        reduction_tiles_per_phase=base_reduction_tiles_per_phase,
    ) + estimate_lora_backward_arena_l1_bytes(
        token_tiles=token_tiles,
        output_tiles=output_tiles,
        rank_tiles=rank_tiles,
    )


def _shift_compile_time_cb_indices(source: str, offset: int) -> str:
    """Shift numeric CB compile-time arguments in a generated phase source."""

    pattern = re.compile(r"get_compile_time_arg_val\((\d+)\)")
    return pattern.sub(
        lambda match: (
            "get_compile_time_arg_val("
            f"{int(match.group(1)) + offset})"
        ),
        source,
    )


@lru_cache(maxsize=32)
def build_tt_lang_frozen_linear_lora_backward(
    *,
    scale: float,
    tokens: int,
    input_features: int,
    output_features: int,
    physical_rank: int,
    base_reduction_tiles_per_phase: int = 1,
) -> Callable:
    dimensions = (tokens, input_features, output_features, physical_rank)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError("all fused backward dimensions must be positive /32")
    m_tiles = tokens // TILE_SIZE
    k_tiles = input_features // TILE_SIZE
    n_tiles = output_features // TILE_SIZE
    if (
        base_reduction_tiles_per_phase <= 0
        or n_tiles % base_reduction_tiles_per_phase
    ):
        raise ValueError(
            "base dX reduction phase must divide output-feature tiles"
        )
    base_reduction_blocks = (
        n_tiles // base_reduction_tiles_per_phase
    )
    rank_tiles = physical_rank // TILE_SIZE
    if rank_tiles != 1 or m_tiles % 2 or n_tiles % 2:
        raise ValueError(
            "exact fused backward requires rank 32 and even M/N tiles"
        )
    base_jobs = m_tiles * k_tiles
    base_jobs_per_node = -(
        -base_jobs // BLACKHOLE_P150A_CORE_COUNT
    )
    l1_bytes = estimate_fused_frozen_lora_backward_l1_bytes(
        token_tiles=m_tiles,
        input_tiles=k_tiles,
        output_tiles=n_tiles,
        rank_tiles=rank_tiles,
        base_reduction_tiles_per_phase=(
            base_reduction_tiles_per_phase
        ),
    )
    l1_budget = int(1.46 * 1024 * 1024 * 0.75)
    if l1_bytes > l1_budget:
        raise ValueError(
            "fused frozen LoRA backward exceeds 75% worker L1: "
            f"{l1_bytes} > {l1_budget}"
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
                "The fused frozen LoRA backward requires TT-XLA"
            ) from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    scale_bf16 = float(
        torch.tensor(scale, dtype=torch.bfloat16).float().item()
    )
    scale_bits = struct.unpack(
        "<I", struct.pack("<f", scale_bf16)
    )[0]
    variant = hashlib.sha256(
        (
            f"{scale_bf16:.17g}:{tokens}:{input_features}:"
            f"{output_features}:{physical_rank}:"
            f"{base_reduction_tiles_per_phase}:"
            "frozen-lora-backward-v2-phased-base-dx-dh-db"
        ).encode("ascii")
    ).hexdigest()[:16]

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _backward_program(
        output_gradient,
        frozen_weight,
        lora_b,
        hidden,
        arena,
    ):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        local_base_jobs = -(-base_jobs // total_nodes)
        m_tiles_per_node = -(-m_tiles // total_nodes)
        n_tiles_per_node = -(-n_tiles // total_nodes)

        base_dy_dfb = ttl.make_dataflow_buffer_like(
            output_gradient,
            shape=(1, base_reduction_tiles_per_phase),
            block_count=2,
        )
        base_weight_dfb = ttl.make_dataflow_buffer_like(
            frozen_weight,
            shape=(base_reduction_tiles_per_phase, 1),
            block_count=2,
        )
        base_partial_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=2
        )
        base_result_dfb = ttl.make_dataflow_buffer_like(
            arena,
            shape=(1, 1),
            block_count=max(2, base_jobs_per_node),
        )

        dh_dy_dfb = ttl.make_dataflow_buffer_like(
            output_gradient, shape=(1, n_tiles), block_count=1
        )
        dh_b_dfb = ttl.make_dataflow_buffer_like(
            lora_b, shape=(n_tiles, rank_tiles), block_count=1
        )
        dh_scaled_dy_dfb = ttl.make_dataflow_buffer_like(
            output_gradient, shape=(1, n_tiles), block_count=1
        )
        dh_partial_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=1
        )
        dh_output_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=1
        )

        db_dy_dfb = ttl.make_dataflow_buffer_like(
            output_gradient, shape=(m_tiles, 1), block_count=1
        )
        db_hidden_dfb = ttl.make_dataflow_buffer_like(
            hidden, shape=(m_tiles, rank_tiles), block_count=1
        )
        db_scaled_dy_dfb = ttl.make_dataflow_buffer_like(
            output_gradient, shape=(m_tiles, 1), block_count=1
        )
        db_transposed_hidden_dfb = ttl.make_dataflow_buffer_like(
            hidden, shape=(rank_tiles, m_tiles), block_count=1
        )
        db_partial_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(rank_tiles, 1), block_count=1
        )
        db_pretranspose_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(rank_tiles, 1), block_count=1
        )
        db_output_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=1
        )

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(local_base_jobs):
                job = node_index * local_base_jobs + local_job
                if job < base_jobs:
                    m_tile = job // k_tiles
                    k_tile = job - m_tile * k_tiles
                    for block in range(base_reduction_blocks):
                        n_start = (
                            block * base_reduction_tiles_per_phase
                        )
                        with (
                            base_dy_dfb.reserve() as dy,
                            base_weight_dfb.reserve() as weight,
                        ):
                            tx_dy = ttl.copy(
                                output_gradient[
                                    m_tile : m_tile + 1,
                                    n_start : n_start
                                    + base_reduction_tiles_per_phase,
                                ],
                                dy,
                            )
                            tx_weight = ttl.copy(
                                frozen_weight[
                                    n_start : n_start
                                    + base_reduction_tiles_per_phase,
                                    k_tile : k_tile + 1,
                                ],
                                weight,
                            )
                            tx_dy.wait()
                            tx_weight.wait()

            for local_m in range(m_tiles_per_node):
                m_tile = node_index * m_tiles_per_node + local_m
                if m_tile < m_tiles:
                    with (
                        dh_dy_dfb.reserve() as dy,
                        dh_b_dfb.reserve() as b,
                    ):
                        tx_dy = ttl.copy(
                            output_gradient[
                                m_tile : m_tile + 1, 0:n_tiles
                            ],
                            dy,
                        )
                        tx_b = ttl.copy(
                            lora_b[0:n_tiles, 0:rank_tiles], b
                        )
                        tx_dy.wait()
                        tx_b.wait()

            for local_n in range(n_tiles_per_node):
                n_tile = node_index * n_tiles_per_node + local_n
                if n_tile < n_tiles:
                    with (
                        db_dy_dfb.reserve() as dy,
                        db_hidden_dfb.reserve() as hidden_block,
                    ):
                        tx_dy = ttl.copy(
                            output_gradient[
                                0:m_tiles, n_tile : n_tile + 1
                            ],
                            dy,
                        )
                        tx_hidden = ttl.copy(
                            hidden[0:m_tiles, 0:rank_tiles],
                            hidden_block,
                        )
                        tx_dy.wait()
                        tx_hidden.wait()

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(local_base_jobs):
                job = node_index * local_base_jobs + local_job
                if job < base_jobs:
                    with base_partial_dfb.reserve() as partial:
                        partial.store(
                            ttl.block.fill(0, shape=partial.shape)
                        )
                    for _ in range(base_reduction_blocks):
                        with (
                            base_dy_dfb.wait() as dy,
                            base_weight_dfb.wait() as weight,
                            base_partial_dfb.wait() as previous,
                        ):
                            with base_partial_dfb.reserve() as next_value:
                                next_value.store(
                                    previous + ttl.matmul(dy, weight)
                                )
                    with base_partial_dfb.wait() as partial:
                        with base_result_dfb.reserve() as result:
                            result.store(partial)

            for local_m in range(m_tiles_per_node):
                m_tile = node_index * m_tiles_per_node + local_m
                if m_tile < m_tiles:
                    with dh_dy_dfb.wait() as dy:
                        with dh_scaled_dy_dfb.reserve() as scaled:
                            scaled.store(dy * scale_bf16)
                    with (
                        dh_scaled_dy_dfb.wait() as scaled,
                        dh_b_dfb.wait() as b,
                    ):
                        with dh_partial_dfb.reserve() as partial:
                            partial.store(ttl.matmul(scaled, b))
                    with dh_partial_dfb.wait() as partial:
                        with dh_output_dfb.reserve() as dh:
                            dh.store(partial)

            for local_n in range(n_tiles_per_node):
                n_tile = node_index * n_tiles_per_node + local_n
                if n_tile < n_tiles:
                    with db_dy_dfb.wait() as dy:
                        with db_scaled_dy_dfb.reserve() as scaled:
                            scaled.store(dy * scale_bf16)
                    with db_hidden_dfb.wait() as hidden_block:
                        with db_transposed_hidden_dfb.reserve() as transposed:
                            transposed.store(ttl.transpose(hidden_block))
                    with (
                        db_transposed_hidden_dfb.wait() as transposed,
                        db_scaled_dy_dfb.wait() as scaled,
                    ):
                        with db_partial_dfb.reserve() as partial:
                            partial.store(ttl.matmul(transposed, scaled))
                    with db_partial_dfb.wait() as partial:
                        with db_pretranspose_dfb.reserve() as before:
                            before.store(partial)
                    with db_pretranspose_dfb.wait() as before:
                        with db_output_dfb.reserve() as db:
                            db.store(before)

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(local_base_jobs):
                job = node_index * local_base_jobs + local_job
                if job < base_jobs:
                    m_tile = job // k_tiles
                    k_tile = job - m_tile * k_tiles
                    with base_result_dfb.wait() as result:
                        tx = ttl.copy(
                            result,
                            arena[
                                m_tile : m_tile + 1,
                                k_tile : k_tile + 1,
                            ],
                        )
                        tx.wait()
            for local_m in range(m_tiles_per_node):
                m_tile = node_index * m_tiles_per_node + local_m
                if m_tile < m_tiles:
                    with dh_output_dfb.wait() as dh:
                        tx = ttl.copy(
                            dh,
                            arena[
                                m_tile : m_tile + 1,
                                k_tiles : k_tiles + rank_tiles,
                            ],
                        )
                        tx.wait()
            for local_n in range(n_tiles_per_node):
                n_tile = node_index * n_tiles_per_node + local_n
                if n_tile < n_tiles:
                    with db_output_dfb.wait() as db:
                        tx = ttl.copy(
                            db,
                            arena[
                                m_tiles + n_tile : m_tiles + n_tile + 1,
                                0:rank_tiles,
                            ],
                        )
                        tx.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.frozen_lora_backward.{variant}",
        arg_roles=("in", "in", "in", "in", "out"),
        version_tag=f"frozen-lora-backward-{variant}",
    )
    def fused_backward(
        output_gradient,
        frozen_weight,
        lora_b,
        hidden,
        arena,
    ):
        _clear_ttl_compile_wrapper_cache(_backward_program)
        result = _backward_program(
            output_gradient, frozen_weight, lora_b, hidden, arena
        )
        closure = _backward_program.__closure__ or ()
        for name, cell in zip(
            _backward_program.__code__.co_freevars, closure
        ):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 16:
                    raise RuntimeError(
                        "fused frozen backward expected 16 CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                for index in (2, 7, 13):
                    compiled.cb_configs[index].tensor = (
                        _FP32_DTYPE_ONLY_TENSOR
                    )
                compute_paths = [
                    Path(path)
                    for path, thread_type in compiled.kernel_paths
                    if thread_type == "compute"
                ]
                if len(compute_paths) != 1:
                    raise RuntimeError(
                        "fused backward expected one compute kernel"
                    )
                grid = compiled.core_ranges.bounding_box().grid_size()
                total_nodes = int(grid.x) * int(grid.y)
                replacements = {
                    "@JOBS_PER_NODE@": str(
                        -(-base_jobs // total_nodes)
                    ),
                    "@TOTAL_JOBS@": str(base_jobs),
                    "@K_TILES@": str(k_tiles),
                    "@N_TILES@": str(n_tiles),
                    "@REDUCTION_TILES_PER_PHASE@": str(
                        base_reduction_tiles_per_phase
                    ),
                    "@GRID_COLS@": str(int(grid.x)),
                    "@NUM_N_BLOCKS@": str(n_tiles // 2),
                    "@M_TILES_PER_NODE@": str(
                        -(-m_tiles // total_nodes)
                    ),
                    "@TOTAL_M_TILES@": str(m_tiles),
                    "@N_TILES_PER_NODE@": str(
                        -(-n_tiles // total_nodes)
                    ),
                    "@TOTAL_N_TILES@": str(n_tiles),
                    "@REDUCTION_TILES@": str(m_tiles),
                    "@SCALE_BITS@": str(scale_bits),
                }
                cpp_dir = Path(__file__).with_name("cpp")
                lora_source = (
                    cpp_dir / "lora_backward_arena_exact_compute.cpp"
                ).read_text(encoding="utf-8")
                lora_source = _shift_compile_time_cb_indices(
                    lora_source, 4
                ).replace(
                    "void kernel_main()",
                    "FORCE_INLINE void lora_backward_phase()",
                    1,
                )
                base_source = (
                    cpp_dir / "frozen_linear_backward_base_phase.cpp"
                ).read_text(encoding="utf-8")
                source = lora_source + "\n" + base_source
                for placeholder, value in replacements.items():
                    source = source.replace(placeholder, value)
                if any(key in source for key in replacements):
                    raise RuntimeError(
                        "unexpanded fused backward placeholder"
                    )
                compute_paths[0].write_text(source, encoding="utf-8")
            break
        return result

    return fused_backward
