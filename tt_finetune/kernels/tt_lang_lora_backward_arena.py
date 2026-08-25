"""Exact single-output ``[dH; dB]`` LoRA backward arena for Blackhole.

The installed Shardy version cannot propagate tuple-valued TT-Lang custom
calls.  Both gradients have physical rank 32, so they can instead share one
tiled output tensor:

    rows [0:M]     = dH
    rows [M:M+N]   = dB

The hand-written compute kernel preserves the already-validated, bit-exact
dH and dB reduction schedules while removing one custom call and one Python
execution boundary from the experimental autograd bridge.
"""

from __future__ import annotations

import hashlib
import struct
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
from .tt_lang_lora_weight_backward import plan_lora_db


ttl = None
tt_lang_operation = None


def estimate_lora_backward_arena_l1_bytes(
    *,
    token_tiles: int,
    output_tiles: int,
    rank_tiles: int = 1,
) -> int:
    """Estimate all dH+dB CB storage in BF16-equivalent tiles."""

    if min(token_tiles, output_tiles, rank_tiles) <= 0:
        raise ValueError("all tile dimensions must be positive")
    dh_tiles = (
        output_tiles  # dY row
        + output_tiles * rank_tiles  # B
        + output_tiles  # scaled dY
        + 2 * rank_tiles  # FP32 partial
        + rank_tiles  # dH output
    )
    db_tiles = (
        token_tiles  # dY column
        + token_tiles * rank_tiles  # H
        + token_tiles  # scaled dY
        + token_tiles * rank_tiles  # transposed H
        + 2 * rank_tiles  # FP32 partial
        + rank_tiles  # pretranspose
        + rank_tiles  # dB output
    )
    return (dh_tiles + db_tiles) * BF16_TILE_BYTES


@lru_cache(maxsize=32)
def build_tt_lang_lora_backward_arena(
    *,
    scale: float,
    tokens: int,
    output_features: int,
    physical_rank: int,
) -> Callable:
    """Build one exact custom operation returning ``[dH; dB]``."""

    plan_lora_db(
        tokens=tokens,
        output_features=output_features,
        physical_rank=physical_rank,
    )
    dimensions = (tokens, output_features, physical_rank)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError(
            "tokens, output features, and rank must be positive /32"
        )
    m_tiles = tokens // TILE_SIZE
    n_tiles = output_features // TILE_SIZE
    rank_tiles = physical_rank // TILE_SIZE
    if rank_tiles != 1 or m_tiles % 2 or n_tiles % 2:
        raise ValueError(
            "exact backward arena requires rank 32 and even M/N tile counts"
        )
    l1_budget = int(1.46 * 1024 * 1024 * 0.75)
    l1_bytes = estimate_lora_backward_arena_l1_bytes(
        token_tiles=m_tiles,
        output_tiles=n_tiles,
        rank_tiles=rank_tiles,
    )
    if l1_bytes > l1_budget:
        raise ValueError(
            "exact backward arena exceeds the 75% worker L1 budget: "
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
                "The native LoRA backward arena requires the TT-XLA worker"
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
            f"{scale_bf16:.17g}:{tokens}:{output_features}:"
            f"{physical_rank}:backward-arena-v1"
        ).encode("ascii")
    ).hexdigest()[:16]

    @ttl.operation(
        grid="full",
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _backward_arena_program(
        output_gradient,
        lora_b,
        hidden,
        arena,
    ):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        m_tiles_per_node = -(-m_tiles // total_nodes)
        n_tiles_per_node = -(-n_tiles // total_nodes)

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
                            lora_b[0:n_tiles, 0:rank_tiles],
                            b,
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
                        with (
                            db_transposed_hidden_dfb.reserve()
                            as transposed_hidden
                        ):
                            transposed_hidden.store(
                                ttl.transpose(hidden_block)
                            )
                    with (
                        db_transposed_hidden_dfb.wait()
                        as transposed_hidden,
                        db_scaled_dy_dfb.wait() as scaled,
                    ):
                        with db_partial_dfb.reserve() as partial:
                            partial.store(
                                ttl.matmul(transposed_hidden, scaled)
                            )
                    with db_partial_dfb.wait() as partial:
                        with db_pretranspose_dfb.reserve() as pretranspose:
                            pretranspose.store(partial)
                    with db_pretranspose_dfb.wait() as pretranspose:
                        with db_output_dfb.reserve() as db:
                            db.store(pretranspose)

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_m in range(m_tiles_per_node):
                m_tile = node_index * m_tiles_per_node + local_m
                if m_tile < m_tiles:
                    with dh_output_dfb.wait() as dh:
                        tx = ttl.copy(
                            dh,
                            arena[m_tile : m_tile + 1, 0:rank_tiles],
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
        operation_id=f"tt_finetune.lora_backward_arena.{variant}",
        arg_roles=("in", "in", "in", "out"),
        version_tag=f"lora-backward-arena-{variant}",
    )
    def backward_arena(output_gradient, lora_b, hidden, arena):
        _clear_ttl_compile_wrapper_cache(_backward_arena_program)
        result = _backward_arena_program(
            output_gradient, lora_b, hidden, arena
        )
        closure = _backward_arena_program.__closure__ or ()
        for name, cell in zip(
            _backward_arena_program.__code__.co_freevars,
            closure,
        ):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 12:
                    raise RuntimeError(
                        "exact backward arena expected 12 CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                compiled.cb_configs[3].tensor = _FP32_DTYPE_ONLY_TENSOR
                compiled.cb_configs[9].tensor = _FP32_DTYPE_ONLY_TENSOR
                compute_paths = [
                    Path(path)
                    for path, thread_type in compiled.kernel_paths
                    if thread_type == "compute"
                ]
                if len(compute_paths) != 1:
                    raise RuntimeError(
                        "exact backward arena expected one compute kernel"
                    )
                grid = compiled.core_ranges.bounding_box().grid_size()
                total_nodes = int(grid.x) * int(grid.y)
                replacements = {
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
                    "@GRID_COLS@": str(int(grid.x)),
                    "@SCALE_BITS@": str(scale_bits),
                }
                source = (
                    Path(__file__).with_name("cpp")
                    / "lora_backward_arena_exact_compute.cpp"
                ).read_text(encoding="utf-8")
                for placeholder, value in replacements.items():
                    source = source.replace(placeholder, value)
                if any(key in source for key in replacements):
                    raise RuntimeError(
                        "unexpanded placeholder in backward arena source"
                    )
                compute_paths[0].write_text(source, encoding="utf-8")
            break
        return result

    return backward_arena
