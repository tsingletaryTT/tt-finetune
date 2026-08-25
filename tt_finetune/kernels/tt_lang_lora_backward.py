"""Experimental native LoRA backward primitives for Blackhole.

The first accepted unit is the activation-gradient path:

    scaled_dy = dy * scale
    dH = scaled_dy @ B
    dX = dH @ A

It is deliberately one TT-Metal reader/compute/writer program.  ``scaled_dy``
and ``dH`` are explicit BF16 L1 circular-buffer boundaries so the operation
matches the generic PyTorch/XLA numerical contract without writing either
intermediate to DRAM.

The transpose-reduction weight gradients (dA and dB) are kept separate until
their reduction schedules are characterized and validated.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache


TILE_SIZE = 32
BF16_TILE_BYTES = TILE_SIZE * TILE_SIZE * 2
ttl = None
tt_lang_operation = None


@dataclass(frozen=True)
class _DTypeOnlyTensor:
    dtype: torch.dtype


_FP32_DTYPE_ONLY_TENSOR = _DTypeOnlyTensor(torch.float32)


@dataclass(frozen=True)
class LoraBackwardDxPlan:
    m_tiles_per_block: int
    n_tiles_per_block: int
    k_tiles_per_block: int
    rank_tiles: int
    estimated_l1_bytes: int
    l1_budget_bytes: int


def estimate_lora_backward_dx_l1_bytes(
    *,
    m_tiles_per_block: int,
    n_tiles_per_block: int,
    k_tiles_per_block: int,
    rank_tiles: int,
    block_count: int = 1,
) -> int:
    """Estimate explicit BF16 CB storage for the native dX program."""

    values = (
        m_tiles_per_block,
        n_tiles_per_block,
        k_tiles_per_block,
        rank_tiles,
        block_count,
    )
    if any(value <= 0 for value in values):
        raise ValueError("all tile dimensions must be positive")
    tiles = (
        m_tiles_per_block * n_tiles_per_block  # dY
        + n_tiles_per_block * rank_tiles  # B
        + m_tiles_per_block * n_tiles_per_block  # scaled dY
        + m_tiles_per_block * rank_tiles  # dH
        + rank_tiles * k_tiles_per_block  # A
        + m_tiles_per_block * k_tiles_per_block  # dX
    )
    return block_count * tiles * BF16_TILE_BYTES


def plan_lora_backward_dx(
    *,
    tokens: int,
    input_features: int,
    output_features: int,
    physical_rank: int,
    worker_l1_bytes: int = int(1.46 * 1024 * 1024),
    l1_fraction: float = 0.75,
) -> LoraBackwardDxPlan:
    """Choose the largest full-feature dX block that fits the L1 budget.

    The initial native primitive intentionally keeps the complete output and
    input feature reductions in one block.  This avoids introducing arbitrary
    BF16 partial-reduction boundaries before exactness is measured.  M is
    distributed across cores one tile at a time.
    """

    dimensions = (tokens, input_features, output_features, physical_rank)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError(
            "tokens, feature dimensions, and physical rank must be positive "
            "multiples of 32"
        )
    if not 0.0 < l1_fraction <= 1.0:
        raise ValueError("l1_fraction must be in (0, 1]")
    if worker_l1_bytes <= 0:
        raise ValueError("worker_l1_bytes must be positive")

    plan = LoraBackwardDxPlan(
        m_tiles_per_block=1,
        n_tiles_per_block=output_features // TILE_SIZE,
        k_tiles_per_block=input_features // TILE_SIZE,
        rank_tiles=physical_rank // TILE_SIZE,
        estimated_l1_bytes=estimate_lora_backward_dx_l1_bytes(
            m_tiles_per_block=1,
            n_tiles_per_block=output_features // TILE_SIZE,
            k_tiles_per_block=input_features // TILE_SIZE,
            rank_tiles=physical_rank // TILE_SIZE,
        ),
        l1_budget_bytes=int(worker_l1_bytes * l1_fraction),
    )
    if plan.estimated_l1_bytes > plan.l1_budget_bytes:
        raise ValueError(
            "full-feature native dX schedule exceeds the configured L1 "
            f"budget: {plan.estimated_l1_bytes} > {plan.l1_budget_bytes}"
        )
    return plan


def _variant_id(
    scale: float,
    n_tiles: int,
    k_tiles: int,
    rank_tiles: int,
    exact_reduction: bool,
) -> str:
    payload = (
        f"{scale:.17g}:{n_tiles}:{k_tiles}:{rank_tiles}:"
        f"{int(exact_reduction)}:native-dx-v2"
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


@lru_cache(maxsize=32)
def build_tt_lang_lora_backward_dx(
    *,
    scale: float,
    input_features: int,
    output_features: int,
    physical_rank: int,
    exact_reduction: bool = False,
) -> Callable:
    """Build a one-output native LoRA activation-backward operation."""

    dimensions = (input_features, output_features, physical_rank)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError(
            "feature dimensions and physical rank must be positive "
            "multiples of 32"
        )

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import (
                tt_lang_operation as tt_lang_operation_decorator,
            )
        except ImportError as exc:  # pragma: no cover - host-only environment
            raise RuntimeError(
                "The native LoRA backward kernel requires the TT-XLA worker "
                "environment"
            ) from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    scale_bf16 = float(
        torch.tensor(scale, dtype=torch.bfloat16).float().item()
    )
    n_tiles = output_features // TILE_SIZE
    k_tiles = input_features // TILE_SIZE
    rank_tiles = physical_rank // TILE_SIZE
    if exact_reduction and (rank_tiles != 1 or n_tiles % 2):
        raise ValueError(
            "exact dX currently requires physical rank 32 and an even "
            "output-feature tile count"
        )
    variant = _variant_id(
        scale_bf16,
        n_tiles,
        k_tiles,
        rank_tiles,
        exact_reduction,
    )
    scale_bits = struct.unpack(
        "<I",
        struct.pack("<f", scale_bf16),
    )[0]

    @ttl.operation(
        grid="full",
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _backward_dx_program(
        output_gradient,
        lora_b,
        lora_a,
        input_gradient,
    ):
        m_tiles = output_gradient.shape[0] // TILE_SIZE
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        m_tiles_per_node = -(-m_tiles // total_nodes)

        dy_dfb = ttl.make_dataflow_buffer_like(
            output_gradient,
            shape=(1, n_tiles),
            block_count=1,
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            lora_b,
            shape=(n_tiles, rank_tiles),
            block_count=1,
        )
        scaled_dy_dfb = ttl.make_dataflow_buffer_like(
            output_gradient,
            shape=(1, n_tiles),
            block_count=1,
        )
        dh_dfb = ttl.make_dataflow_buffer_like(
            input_gradient,
            shape=(1, rank_tiles),
            block_count=1,
        )
        a_dfb = ttl.make_dataflow_buffer_like(
            lora_a,
            shape=(rank_tiles, k_tiles),
            block_count=1,
        )
        dx_dfb = ttl.make_dataflow_buffer_like(
            input_gradient,
            shape=(1, k_tiles),
            block_count=1,
        )

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_m in range(m_tiles_per_node):
                m_tile = node_index * m_tiles_per_node + local_m
                if m_tile < m_tiles:
                    with (
                        dy_dfb.reserve() as dy,
                        b_dfb.reserve() as b,
                        a_dfb.reserve() as a,
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
                        tx_a = ttl.copy(
                            lora_a[0:rank_tiles, 0:k_tiles],
                            a,
                        )
                        tx_dy.wait()
                        tx_b.wait()
                        tx_a.wait()

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_m in range(m_tiles_per_node):
                m_tile = node_index * m_tiles_per_node + local_m
                if m_tile < m_tiles:
                    with dy_dfb.wait() as dy:
                        with scaled_dy_dfb.reserve() as scaled_dy:
                            scaled_dy.store(dy * scale_bf16)
                    with (
                        scaled_dy_dfb.wait() as scaled_dy,
                        b_dfb.wait() as b,
                    ):
                        with dh_dfb.reserve() as dh:
                            dh.store(ttl.matmul(scaled_dy, b))
                    with (
                        dh_dfb.wait() as dh,
                        a_dfb.wait() as a,
                    ):
                        with dx_dfb.reserve() as dx:
                            dx.store(ttl.matmul(dh, a))

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_m in range(m_tiles_per_node):
                m_tile = node_index * m_tiles_per_node + local_m
                if m_tile < m_tiles:
                    with dx_dfb.wait() as dx:
                        tx = ttl.copy(
                            dx,
                            input_gradient[
                                m_tile : m_tile + 1, 0:k_tiles
                            ],
                        )
                        tx.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.lora_backward_dx.{variant}",
        arg_roles=("in", "in", "in", "out"),
        version_tag=f"lora-backward-dx-{variant}",
    )
    def backward_dx(output_gradient, lora_b, lora_a, input_gradient):
        _clear_ttl_compile_wrapper_cache(_backward_dx_program)
        result = _backward_dx_program(
            output_gradient,
            lora_b,
            lora_a,
            input_gradient,
        )
        if exact_reduction:
            closure = _backward_dx_program.__closure__ or ()
            for name, cell in zip(
                _backward_dx_program.__code__.co_freevars,
                closure,
            ):
                if name != "cache" or not isinstance(
                    cell.cell_contents, dict
                ):
                    continue
                for compiled in cell.cell_contents.values():
                    if len(compiled.cb_configs) != 6:
                        raise RuntimeError(
                            "exact native dX expected 6 circular buffers, "
                            f"got {len(compiled.cb_configs)}"
                        )
                    compiled.cb_configs[3].tensor = (
                        _FP32_DTYPE_ONLY_TENSOR
                    )
                    compute_paths = [
                        Path(path)
                        for path, thread_type in compiled.kernel_paths
                        if thread_type == "compute"
                    ]
                    if len(compute_paths) != 1:
                        raise RuntimeError(
                            "exact native dX expected one compute kernel, "
                            f"got {len(compute_paths)}"
                        )
                    grid = compiled.core_ranges.bounding_box().grid_size()
                    total_m_tiles = output_gradient.shape[0] // TILE_SIZE
                    total_nodes = int(grid.x) * int(grid.y)
                    m_tiles_per_node = -(
                        -total_m_tiles // total_nodes
                    )
                    second_subblock_w = min(4, k_tiles)
                    while k_tiles % second_subblock_w:
                        second_subblock_w -= 1
                    source_path = (
                        Path(__file__).with_name("cpp")
                        / "lora_backward_dx_exact_compute.cpp"
                    )
                    source = source_path.read_text(encoding="utf-8")
                    replacements = {
                        "@NUM_N_BLOCKS@": str(n_tiles // 2),
                        "@K_TILES@": str(k_tiles),
                        "@SECOND_SUBBLOCK_W@": str(second_subblock_w),
                        "@M_TILES_PER_NODE@": str(m_tiles_per_node),
                        "@TOTAL_M_TILES@": str(total_m_tiles),
                        "@GRID_COLS@": str(int(grid.x)),
                        "@SCALE_BITS@": str(scale_bits),
                    }
                    for placeholder, value in replacements.items():
                        source = source.replace(placeholder, value)
                    if any(
                        placeholder in source
                        for placeholder in replacements
                    ):
                        raise RuntimeError(
                            "unexpanded placeholder in exact native dX source"
                        )
                    compute_paths[0].write_text(source, encoding="utf-8")
                break
        return result

    return backward_dx
