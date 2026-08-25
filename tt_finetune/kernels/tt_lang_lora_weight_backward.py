"""Experimental native LoRA weight-gradient primitives for Blackhole.

The first primitive computes the B gradient without materializing ``dY.T``:

    scaled_dY = BF16(dY * scale)
    dB = scaled_dY.T @ H

where ``H`` is the BF16 rank-space activation saved by the LoRA forward
operation.  Work is distributed by output-feature tile, so each core owns
complete token reductions and produces disjoint rows of packed dB.
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
class LoraWeightGradientPlan:
    reduction_tiles: int
    output_tiles_per_node: int
    rank_tiles: int
    estimated_l1_bytes: int
    l1_budget_bytes: int


def estimate_lora_db_l1_bytes(
    *,
    reduction_tiles: int,
    rank_tiles: int,
    output_tiles_per_node: int = 1,
) -> int:
    """Estimate BF16 CB storage for the transpose-aware dB operation."""

    values = (reduction_tiles, rank_tiles, output_tiles_per_node)
    if any(value <= 0 for value in values):
        raise ValueError("all tile dimensions must be positive")
    tiles = output_tiles_per_node * (
        reduction_tiles  # dY column
        + reduction_tiles * rank_tiles  # H
        + reduction_tiles  # transposed dY
        + rank_tiles  # dB
    )
    return tiles * BF16_TILE_BYTES


def plan_lora_db(
    *,
    tokens: int,
    output_features: int,
    physical_rank: int,
    worker_l1_bytes: int = int(1.46 * 1024 * 1024),
    l1_fraction: float = 0.75,
) -> LoraWeightGradientPlan:
    """Plan one complete token reduction per output-feature tile."""

    dimensions = (tokens, output_features, physical_rank)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError(
            "tokens, output features, and physical rank must be positive "
            "multiples of 32"
        )
    if not 0.0 < l1_fraction <= 1.0:
        raise ValueError("l1_fraction must be in (0, 1]")
    reduction_tiles = tokens // TILE_SIZE
    rank_tiles = physical_rank // TILE_SIZE
    budget = int(worker_l1_bytes * l1_fraction)
    required = estimate_lora_db_l1_bytes(
        reduction_tiles=reduction_tiles,
        rank_tiles=rank_tiles,
    )
    if required > budget:
        raise ValueError(
            "complete native dB reduction exceeds the configured L1 budget: "
            f"{required} > {budget}"
        )
    return LoraWeightGradientPlan(
        reduction_tiles=reduction_tiles,
        output_tiles_per_node=1,
        rank_tiles=rank_tiles,
        estimated_l1_bytes=required,
        l1_budget_bytes=budget,
    )


def estimate_lora_da_l1_bytes(
    *,
    reduction_tiles: int,
    rank_tiles: int,
    input_tiles_per_node: int = 1,
) -> int:
    """Estimate BF16 CB storage for the transpose-aware dA operation."""

    values = (reduction_tiles, rank_tiles, input_tiles_per_node)
    if any(value <= 0 for value in values):
        raise ValueError("all tile dimensions must be positive")
    tiles = input_tiles_per_node * (
        reduction_tiles * rank_tiles  # dH
        + reduction_tiles  # X column
        + reduction_tiles * rank_tiles  # transposed dH
        + rank_tiles  # dA
    )
    return tiles * BF16_TILE_BYTES


def plan_lora_da(
    *,
    tokens: int,
    input_features: int,
    physical_rank: int,
    worker_l1_bytes: int = int(1.46 * 1024 * 1024),
    l1_fraction: float = 0.75,
) -> LoraWeightGradientPlan:
    """Plan one complete token reduction per input-feature tile."""

    dimensions = (tokens, input_features, physical_rank)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError(
            "tokens, input features, and physical rank must be positive "
            "multiples of 32"
        )
    if not 0.0 < l1_fraction <= 1.0:
        raise ValueError("l1_fraction must be in (0, 1]")
    reduction_tiles = tokens // TILE_SIZE
    rank_tiles = physical_rank // TILE_SIZE
    budget = int(worker_l1_bytes * l1_fraction)
    required = estimate_lora_da_l1_bytes(
        reduction_tiles=reduction_tiles,
        rank_tiles=rank_tiles,
    )
    if required > budget:
        raise ValueError(
            "complete native dA reduction exceeds the configured L1 budget: "
            f"{required} > {budget}"
        )
    return LoraWeightGradientPlan(
        reduction_tiles=reduction_tiles,
        output_tiles_per_node=1,
        rank_tiles=rank_tiles,
        estimated_l1_bytes=required,
        l1_budget_bytes=budget,
    )


def _variant_id(
    scale: float,
    reduction_tiles: int,
    output_tiles: int,
    rank_tiles: int,
) -> str:
    payload = (
        f"{scale:.17g}:{reduction_tiles}:{output_tiles}:{rank_tiles}:"
        "native-db-v1"
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


@lru_cache(maxsize=32)
def build_tt_lang_lora_backward_db(
    *,
    scale: float,
    tokens: int,
    output_features: int,
    physical_rank: int,
    exact_reduction: bool = False,
) -> Callable:
    """Build a one-output transpose-aware native dB custom operation."""

    plan = plan_lora_db(
        tokens=tokens,
        output_features=output_features,
        physical_rank=physical_rank,
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
                "The native LoRA dB kernel requires the TT-XLA worker "
                "environment"
            ) from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    scale_bf16 = float(
        torch.tensor(scale, dtype=torch.bfloat16).float().item()
    )
    m_tiles = plan.reduction_tiles
    n_tiles = output_features // TILE_SIZE
    rank_tiles = plan.rank_tiles
    if exact_reduction and (m_tiles % 2 or rank_tiles != 1):
        raise ValueError(
            "exact dB currently requires an even token tile count and one "
            "physical rank tile"
        )
    variant = _variant_id(
        scale_bf16,
        m_tiles,
        n_tiles,
        rank_tiles,
    )
    if exact_reduction:
        variant = hashlib.sha256(
            f"{variant}:exact-v1".encode("ascii")
        ).hexdigest()[:16]
    scale_bits = struct.unpack(
        "<I",
        struct.pack("<f", scale_bf16),
    )[0]

    @ttl.operation(
        grid="full",
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _backward_db_program(output_gradient, hidden, db_output):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        n_tiles_per_node = -(-n_tiles // total_nodes)

        dy_dfb = ttl.make_dataflow_buffer_like(
            output_gradient,
            shape=(m_tiles, 1),
            block_count=1,
        )
        hidden_dfb = ttl.make_dataflow_buffer_like(
            hidden,
            shape=(m_tiles, rank_tiles),
            block_count=1,
        )
        scaled_dy_dfb = ttl.make_dataflow_buffer_like(
            output_gradient,
            shape=(m_tiles, 1),
            block_count=1,
        )
        transposed_dy_dfb = ttl.make_dataflow_buffer_like(
            output_gradient,
            shape=(1, m_tiles),
            block_count=1,
        )
        partial_dfb = ttl.make_dataflow_buffer_like(
            db_output,
            shape=(1, rank_tiles),
            block_count=1,
        )
        pretranspose_db_dfb = ttl.make_dataflow_buffer_like(
            db_output,
            shape=(rank_tiles, 1),
            block_count=1,
        )
        db_dfb = ttl.make_dataflow_buffer_like(
            db_output,
            shape=(1, rank_tiles),
            block_count=1,
        )

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_n in range(n_tiles_per_node):
                n_tile = node_index * n_tiles_per_node + local_n
                if n_tile < n_tiles:
                    with (
                        dy_dfb.reserve() as dy,
                        hidden_dfb.reserve() as hidden_block,
                    ):
                        tx_dy = ttl.copy(
                            output_gradient[0:m_tiles, n_tile : n_tile + 1],
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
            for local_n in range(n_tiles_per_node):
                n_tile = node_index * n_tiles_per_node + local_n
                if n_tile < n_tiles:
                    with dy_dfb.wait() as dy:
                        with scaled_dy_dfb.reserve() as scaled_dy:
                            scaled_dy.store(dy * scale_bf16)
                    with scaled_dy_dfb.wait() as scaled_dy:
                        with transposed_dy_dfb.reserve() as transposed_dy:
                            transposed_dy.store(ttl.transpose(scaled_dy))
                    with (
                        transposed_dy_dfb.wait() as transposed_dy,
                        hidden_dfb.wait() as hidden_block,
                    ):
                        with partial_dfb.reserve() as partial:
                            partial.store(
                                ttl.matmul(transposed_dy, hidden_block)
                            )
                    with partial_dfb.wait() as partial:
                        with pretranspose_db_dfb.reserve() as pretranspose:
                            pretranspose.store(partial)
                    with pretranspose_db_dfb.wait() as pretranspose:
                        with db_dfb.reserve() as db:
                            db.store(pretranspose)

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_n in range(n_tiles_per_node):
                n_tile = node_index * n_tiles_per_node + local_n
                if n_tile < n_tiles:
                    with db_dfb.wait() as db:
                        tx = ttl.copy(
                            db,
                            db_output[
                                n_tile : n_tile + 1, 0:rank_tiles
                            ],
                        )
                        tx.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.lora_backward_db.{variant}",
        arg_roles=("in", "in", "out"),
        version_tag=f"lora-backward-db-{variant}",
    )
    def backward_db(output_gradient, hidden, db_output):
        _clear_ttl_compile_wrapper_cache(_backward_db_program)
        result = _backward_db_program(
            output_gradient,
            hidden,
            db_output,
        )
        if exact_reduction:
            closure = _backward_db_program.__closure__ or ()
            for name, cell in zip(
                _backward_db_program.__code__.co_freevars,
                closure,
            ):
                if name != "cache" or not isinstance(
                    cell.cell_contents, dict
                ):
                    continue
                for compiled in cell.cell_contents.values():
                    if len(compiled.cb_configs) != 7:
                        raise RuntimeError(
                            "exact native dB expected 7 circular buffers, "
                            f"got {len(compiled.cb_configs)}"
                        )
                    compiled.cb_configs[4].tensor = (
                        _FP32_DTYPE_ONLY_TENSOR
                    )
                    compute_paths = [
                        Path(path)
                        for path, thread_type in compiled.kernel_paths
                        if thread_type == "compute"
                    ]
                    if len(compute_paths) != 1:
                        raise RuntimeError(
                            "exact native dB expected one compute kernel, "
                            f"got {len(compute_paths)}"
                        )
                    grid = compiled.core_ranges.bounding_box().grid_size()
                    total_nodes = int(grid.x) * int(grid.y)
                    n_tiles_per_node = -(-n_tiles // total_nodes)
                    source_path = (
                        Path(__file__).with_name("cpp")
                        / "lora_backward_db_exact_compute.cpp"
                    )
                    source = source_path.read_text(encoding="utf-8")
                    replacements = {
                        "@REDUCTION_TILES@": str(m_tiles),
                        "@RANK_TILES@": str(rank_tiles),
                        "@N_TILES_PER_NODE@": str(n_tiles_per_node),
                        "@TOTAL_N_TILES@": str(n_tiles),
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
                            "unexpanded placeholder in exact native dB source"
                        )
                    compute_paths[0].write_text(source, encoding="utf-8")
                break
        return result

    return backward_db


@lru_cache(maxsize=32)
def build_tt_lang_lora_backward_da(
    *,
    tokens: int,
    input_features: int,
    physical_rank: int,
    exact_reduction: bool = False,
) -> Callable:
    """Build a one-output transpose-aware native dA custom operation."""

    plan = plan_lora_da(
        tokens=tokens,
        input_features=input_features,
        physical_rank=physical_rank,
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
                "The native LoRA dA kernel requires the TT-XLA worker "
                "environment"
            ) from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    m_tiles = plan.reduction_tiles
    k_tiles = input_features // TILE_SIZE
    rank_tiles = plan.rank_tiles
    if exact_reduction and (m_tiles % 2 or rank_tiles != 1):
        raise ValueError(
            "exact dA currently requires an even token tile count and one "
            "physical rank tile"
        )
    variant_payload = (
        f"{m_tiles}:{k_tiles}:{rank_tiles}:"
        f"{int(exact_reduction)}:native-da-v1"
    )
    variant = hashlib.sha256(
        variant_payload.encode("ascii")
    ).hexdigest()[:16]

    @ttl.operation(
        grid="full",
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _backward_da_program(hidden_gradient, inputs, da_output):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        k_tiles_per_node = -(-k_tiles // total_nodes)

        dh_dfb = ttl.make_dataflow_buffer_like(
            hidden_gradient,
            shape=(m_tiles, rank_tiles),
            block_count=1,
        )
        x_dfb = ttl.make_dataflow_buffer_like(
            inputs,
            shape=(m_tiles, 1),
            block_count=1,
        )
        transposed_dh_dfb = ttl.make_dataflow_buffer_like(
            hidden_gradient,
            shape=(rank_tiles, m_tiles),
            block_count=1,
        )
        partial_dfb = ttl.make_dataflow_buffer_like(
            da_output,
            shape=(rank_tiles, 1),
            block_count=1,
        )
        pretranspose_da_dfb = ttl.make_dataflow_buffer_like(
            da_output,
            shape=(1, rank_tiles),
            block_count=1,
        )
        da_dfb = ttl.make_dataflow_buffer_like(
            da_output,
            shape=(rank_tiles, 1),
            block_count=1,
        )

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_k in range(k_tiles_per_node):
                k_tile = node_index * k_tiles_per_node + local_k
                if k_tile < k_tiles:
                    with (
                        dh_dfb.reserve() as dh,
                        x_dfb.reserve() as x,
                    ):
                        tx_dh = ttl.copy(
                            hidden_gradient[
                                0:m_tiles, 0:rank_tiles
                            ],
                            dh,
                        )
                        tx_x = ttl.copy(
                            inputs[0:m_tiles, k_tile : k_tile + 1],
                            x,
                        )
                        tx_dh.wait()
                        tx_x.wait()

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_k in range(k_tiles_per_node):
                k_tile = node_index * k_tiles_per_node + local_k
                if k_tile < k_tiles:
                    with dh_dfb.wait() as dh:
                        with transposed_dh_dfb.reserve() as transposed_dh:
                            transposed_dh.store(ttl.transpose(dh))
                    with (
                        transposed_dh_dfb.wait() as transposed_dh,
                        x_dfb.wait() as x,
                    ):
                        with partial_dfb.reserve() as partial:
                            partial.store(ttl.matmul(transposed_dh, x))
                    with partial_dfb.wait() as partial:
                        with pretranspose_da_dfb.reserve() as pretranspose:
                            pretranspose.store(partial)
                    with pretranspose_da_dfb.wait() as pretranspose:
                        with da_dfb.reserve() as da:
                            da.store(pretranspose)

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_k in range(k_tiles_per_node):
                k_tile = node_index * k_tiles_per_node + local_k
                if k_tile < k_tiles:
                    with da_dfb.wait() as da:
                        tx = ttl.copy(
                            da,
                            da_output[
                                0:rank_tiles, k_tile : k_tile + 1
                            ],
                        )
                        tx.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.lora_backward_da.{variant}",
        arg_roles=("in", "in", "out"),
        version_tag=f"lora-backward-da-{variant}",
    )
    def backward_da(hidden_gradient, inputs, da_output):
        _clear_ttl_compile_wrapper_cache(_backward_da_program)
        result = _backward_da_program(
            hidden_gradient,
            inputs,
            da_output,
        )
        if exact_reduction:
            closure = _backward_da_program.__closure__ or ()
            for name, cell in zip(
                _backward_da_program.__code__.co_freevars,
                closure,
            ):
                if name != "cache" or not isinstance(
                    cell.cell_contents, dict
                ):
                    continue
                for compiled in cell.cell_contents.values():
                    if len(compiled.cb_configs) != 6:
                        raise RuntimeError(
                            "exact native dA expected 6 circular buffers, "
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
                            "exact native dA expected one compute kernel, "
                            f"got {len(compute_paths)}"
                        )
                    grid = compiled.core_ranges.bounding_box().grid_size()
                    total_nodes = int(grid.x) * int(grid.y)
                    k_tiles_per_node = -(-k_tiles // total_nodes)
                    source_path = (
                        Path(__file__).with_name("cpp")
                        / "lora_backward_da_exact_compute.cpp"
                    )
                    source = source_path.read_text(encoding="utf-8")
                    replacements = {
                        "@REDUCTION_TILES@": str(m_tiles),
                        "@RANK_TILES@": str(rank_tiles),
                        "@K_TILES_PER_NODE@": str(k_tiles_per_node),
                        "@TOTAL_K_TILES@": str(k_tiles),
                        "@GRID_COLS@": str(int(grid.x)),
                    }
                    for placeholder, value in replacements.items():
                        source = source.replace(placeholder, value)
                    if any(
                        placeholder in source
                        for placeholder in replacements
                    ):
                        raise RuntimeError(
                            "unexpanded placeholder in exact native dA source"
                        )
                    compute_paths[0].write_text(source, encoding="utf-8")
                break
        return result

    return backward_da
