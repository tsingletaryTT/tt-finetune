"""Experimental native LoRA forward kernel for TT-XLA through TT-Lang.

This module is intentionally not wired into the public trainer yet.  It is the
first real matmul-bearing custom operation on the native-kernel path:

    output = base + scale * ((inputs @ lora_a.T) @ lora_b.T)

The operation is compiled by TT-Lang, carried through the TT-Torch/PJRT custom
call bridge, and executed as one TT-Metal reader/compute/writer program.

TT hardware operates on 32x32 tiles.  The initial prototype therefore requires
all matrix dimensions, including the LoRA rank, to be padded to multiples of
32.  Padding rank 8 or 16 with zeros preserves the mathematical result.  The
public integration will hide this representation detail once correctness and
performance are accepted.
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
    """Minimal tensor likeness used only while serializing an internal CB."""

    dtype: torch.dtype


_FP32_DTYPE_ONLY_TENSOR = _DTypeOnlyTensor(torch.float32)


@dataclass(frozen=True)
class LoraForwardBlockPlan:
    m_tiles_per_block: int
    n_tiles_per_block: int
    k_tiles_per_block: int
    estimated_l1_bytes: int
    l1_budget_bytes: int


def estimate_lora_forward_l1_bytes(
    *,
    m_tiles_per_block: int,
    n_tiles_per_block: int,
    k_tiles_per_block: int,
    rank_tiles: int = 1,
) -> int:
    """Estimate the seven depth-two BF16 circular buffers in the program."""

    values = (
        m_tiles_per_block,
        n_tiles_per_block,
        k_tiles_per_block,
        rank_tiles,
    )
    if any(value <= 0 for value in values):
        raise ValueError("all tile dimensions must be positive")
    tiles_per_buffer_set = (
        m_tiles_per_block * k_tiles_per_block  # inputs
        + rank_tiles * k_tiles_per_block  # A
        + m_tiles_per_block * rank_tiles  # hidden accumulator
        + n_tiles_per_block * rank_tiles  # B
        + 3 * m_tiles_per_block * n_tiles_per_block  # base/delta/output
    )
    return 2 * tiles_per_buffer_set * BF16_TILE_BYTES


def estimate_exact_lora_forward_l1_bytes(
    *,
    n_tiles_per_block: int,
    rank_tiles: int = 1,
) -> int:
    """Estimate explicit CB storage for the exact M=1/K=2 program.

    The FP32 partial CB counts as two BF16 tiles per physical tile. The input
    CB is enlarged to retain a complete scaled-delta N block during the
    one-time BF16 destination-mode add phase.
    """

    if n_tiles_per_block <= 0 or rank_tiles <= 0:
        raise ValueError("tile dimensions must be positive")
    input_tiles = max(4, n_tiles_per_block)
    bf16_equivalent_tiles = (
        input_tiles
        + 4 * rank_tiles  # double-buffered A, K=2
        + 4 * rank_tiles  # depth-two FP32 partial
        + 2 * rank_tiles  # final BF16 hidden
        + 8 * n_tiles_per_block  # B, base, delta, output
    )
    return bf16_equivalent_tiles * BF16_TILE_BYTES


def plan_exact_lora_forward_blocks(
    *,
    tokens: int,
    input_features: int,
    output_features: int,
    physical_rank: int,
    worker_l1_bytes: int = int(1.46 * 1024 * 1024),
    l1_fraction: float = 0.75,
    maximum_n_tiles_per_block: int = 32,
) -> LoraForwardBlockPlan:
    """Choose a safe exact schedule without requiring manual block flags."""

    dimensions = (tokens, input_features, output_features, physical_rank)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError(
            "tokens, feature dimensions, and physical rank must be positive "
            "multiples of 32"
        )
    if input_features // TILE_SIZE % 2:
        raise ValueError("exact forward requires an even K tile count")
    if physical_rank != TILE_SIZE:
        raise ValueError(
            "exact forward currently supports one physical rank tile"
        )
    if not 0.0 < l1_fraction <= 1.0:
        raise ValueError("l1_fraction must be in (0, 1]")
    if worker_l1_bytes <= 0 or maximum_n_tiles_per_block <= 0:
        raise ValueError("L1 and N-block limits must be positive")

    total_n_tiles = output_features // TILE_SIZE
    budget = int(worker_l1_bytes * l1_fraction)
    candidates = [
        n_tiles
        for n_tiles in range(
            1,
            min(total_n_tiles, maximum_n_tiles_per_block) + 1,
        )
        if total_n_tiles % n_tiles == 0
        and estimate_exact_lora_forward_l1_bytes(
            n_tiles_per_block=n_tiles
        )
        <= budget
    ]
    if not candidates:
        raise ValueError("no exact native forward N block fits L1")
    n_tiles = max(candidates)
    return LoraForwardBlockPlan(
        m_tiles_per_block=1,
        n_tiles_per_block=n_tiles,
        k_tiles_per_block=2,
        estimated_l1_bytes=estimate_exact_lora_forward_l1_bytes(
            n_tiles_per_block=n_tiles
        ),
        l1_budget_bytes=budget,
    )


def plan_lora_forward_blocks(
    *,
    tokens: int,
    input_features: int,
    output_features: int,
    physical_rank: int,
    worker_l1_bytes: int = int(1.46 * 1024 * 1024),
    l1_fraction: float = 0.75,
) -> LoraForwardBlockPlan:
    """Choose a shape-generic, L1-budgeted starting schedule.

    This is a planner, not a substitute for silicon autotuning.  It encodes
    the two measured constraints that generalized across Pythia and Qwen:
    use M=2 tiles when divisible, and do not make the K block wider than
    16 tiles.  Among valid N/K divisors it maximizes work per synchronization
    while keeping explicit CBs within 75% of the reported worker L1.
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

    m_tiles = tokens // TILE_SIZE
    n_tiles = output_features // TILE_SIZE
    k_tiles = input_features // TILE_SIZE
    rank_tiles = physical_rank // TILE_SIZE
    m_block = 2 if m_tiles % 2 == 0 else 1
    budget = int(worker_l1_bytes * l1_fraction)

    candidates: list[tuple[int, int, int]] = []
    for n_block in range(1, n_tiles + 1):
        if n_tiles % n_block:
            continue
        for k_block in range(1, min(k_tiles, 16) + 1):
            if k_tiles % k_block:
                continue
            required = estimate_lora_forward_l1_bytes(
                m_tiles_per_block=m_block,
                n_tiles_per_block=n_block,
                k_tiles_per_block=k_block,
                rank_tiles=rank_tiles,
            )
            if required <= budget:
                # Prefer a wider K block when synchronization-work scores tie;
                # the measured 24x12 Pythia schedule leaves more L1 margin
                # than 36x8 and performs equivalently.
                candidates.append((n_block * k_block, k_block, n_block))
    if not candidates:
        raise ValueError(
            "no native LoRA block schedule fits the configured L1 budget"
        )
    _, k_block, n_block = max(candidates)
    required = estimate_lora_forward_l1_bytes(
        m_tiles_per_block=m_block,
        n_tiles_per_block=n_block,
        k_tiles_per_block=k_block,
        rank_tiles=rank_tiles,
    )
    return LoraForwardBlockPlan(
        m_tiles_per_block=m_block,
        n_tiles_per_block=n_block,
        k_tiles_per_block=k_block,
        estimated_l1_bytes=required,
        l1_budget_bytes=budget,
    )


def _variant_id(
    scale: float,
    m_tiles_per_block: int,
    n_tiles_per_block: int,
    k_tiles_per_block: int,
    matmul_full_fp32: bool,
    math_fidelity: str,
    fp32_dest_acc_en: bool,
    exact_reduction: bool,
) -> str:
    payload = (
        f"{scale:.17g}:{m_tiles_per_block}:{n_tiles_per_block}:"
        f"{k_tiles_per_block}:{int(matmul_full_fp32)}:"
        f"{math_fidelity}:{int(fp32_dest_acc_en)}:"
        f"{int(exact_reduction)}:v21"
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


def padded_rank(rank: int) -> int:
    """Return the physical tile-aligned rank used by the native prototype."""

    if rank <= 0:
        raise ValueError("rank must be positive")
    return ((rank + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


def pad_lora_weights(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad PEFT-shaped A=[R,K], B=[N,R] weights to a tile-aligned rank."""

    if lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError("LoRA A and B must both be rank-2 tensors")
    rank, input_features = lora_a.shape
    output_features, b_rank = lora_b.shape
    if rank != b_rank:
        raise ValueError(
            f"LoRA rank mismatch: A has {rank}, B has {b_rank}"
        )
    physical_rank = padded_rank(rank)
    if physical_rank == rank:
        return lora_a, lora_b
    # Keep padding functional.  The old ``new_zeros(); slice.copy_()`` form
    # works eagerly on CPU but the in-place slice update is not represented as
    # a data dependency by the current lazy TT-XLA backend.  Inside a custom
    # autograd forward it therefore reached the device as an all-zero padded
    # operand.  Concatenation preserves the same logical/physical layout and
    # makes the dependency explicit in StableHLO on every backend.
    rank_padding = physical_rank - rank
    padded_a = torch.cat(
        (lora_a, lora_a.new_zeros((rank_padding, input_features))), dim=0
    )
    padded_b = torch.cat(
        (lora_b, lora_b.new_zeros((output_features, rank_padding))), dim=1
    )
    return padded_a.contiguous(), padded_b.contiguous()


@lru_cache(maxsize=32)
def build_tt_lang_lora_forward(
    scale: float,
    m_tiles_per_block: int = 2,
    n_tiles_per_block: int = 4,
    k_tiles_per_block: int = 4,
    matmul_full_fp32: bool = True,
    math_fidelity: str = "HiFi4",
    fp32_dest_acc_en: bool = True,
    exact_reduction: bool = False,
) -> Callable:
    """Build a fused native LoRA forward custom operation.

    Inputs use PEFT orientation:

    * ``inputs``: ``[M, K]``
    * ``lora_a``: ``[R_padded, K]``
    * ``lora_b``: ``[N, R_padded]``
    * ``base`` and ``output``: ``[M, N]``

    ``R_padded`` is normally 32 for LoRA rank 8 or 16.  The low-rank
    intermediate is kept in L1 and is never materialized in DRAM.
    """

    for name, value in (
        ("m_tiles_per_block", m_tiles_per_block),
        ("n_tiles_per_block", n_tiles_per_block),
        ("k_tiles_per_block", k_tiles_per_block),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if math_fidelity not in {"LoFi", "HiFi2", "HiFi3", "HiFi4"}:
        raise ValueError(
            "math_fidelity must be LoFi, HiFi2, HiFi3, or HiFi4"
        )
    if exact_reduction:
        if m_tiles_per_block != 1 or k_tiles_per_block != 2:
            raise ValueError(
                "exact_reduction currently requires M block 1 and K block 2 "
                "to match TT-NN's measured Blackhole reduction schedule"
            )
        if math_fidelity != "HiFi4" or not fp32_dest_acc_en:
            raise ValueError(
                "exact_reduction requires HiFi4 and FP32 destination "
                "accumulation"
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
                "The native LoRA kernel requires the TT-XLA worker environment"
            ) from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    scale_bf16 = float(
        torch.tensor(scale, dtype=torch.bfloat16).float().item()
    )
    variant = _variant_id(
        scale_bf16,
        m_tiles_per_block,
        n_tiles_per_block,
        k_tiles_per_block,
        matmul_full_fp32,
        math_fidelity,
        fp32_dest_acc_en,
        exact_reduction,
    )
    scale_bits = struct.unpack(
        "<I",
        struct.pack("<f", scale_bf16),
    )[0]

    @ttl.operation(
        grid="full",
        fp32_dest_acc_en=fp32_dest_acc_en,
        options=(
            "--ttl-matmul-full-fp32"
            if matmul_full_fp32
            else "--no-ttl-matmul-full-fp32"
        ),
    )
    def _lora_forward_program(inputs, lora_a, lora_b, base, output):
        m_blocks = (
            inputs.shape[0] // TILE_SIZE // m_tiles_per_block
        )
        n_blocks = (
            lora_b.shape[0] // TILE_SIZE // n_tiles_per_block
        )
        k_blocks = (
            inputs.shape[1] // TILE_SIZE // k_tiles_per_block
        )
        rank_tiles = lora_a.shape[0] // TILE_SIZE

        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        m_blocks_per_node = -(-m_blocks // total_nodes)

        inputs_dfb = ttl.make_dataflow_buffer_like(
            inputs,
            shape=(m_tiles_per_block, k_tiles_per_block),
            # Exact forward reuses this CB to stage a complete N block of
            # scaled BF16 deltas before switching destination mode once for
            # the final add.
            block_count=(
                max(
                    2,
                    -(
                        -(m_tiles_per_block * n_tiles_per_block)
                        // (
                            m_tiles_per_block
                            * k_tiles_per_block
                        )
                    ),
                )
                if exact_reduction
                else 2
            ),
        )
        a_dfb = ttl.make_dataflow_buffer_like(
            lora_a,
            shape=(rank_tiles, k_tiles_per_block),
            block_count=2,
        )
        hidden_acc_dfb = ttl.make_dataflow_buffer_like(
            output,
            shape=(m_tiles_per_block, rank_tiles),
            block_count=2,
        )
        hidden_final_dfb = ttl.make_dataflow_buffer_like(
            output,
            shape=(m_tiles_per_block, rank_tiles),
            block_count=2,
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            lora_b,
            shape=(n_tiles_per_block, rank_tiles),
            block_count=2,
        )
        base_dfb = ttl.make_dataflow_buffer_like(
            base,
            shape=(m_tiles_per_block, n_tiles_per_block),
            block_count=2,
        )
        # The current TT-XLA TT-Lang bridge cannot serialize compiler-created
        # anonymous CBs (their buffer_index remains -1).  Tie the second
        # matmul's L1-only result to the output tensor's dtype/layout
        # explicitly.  This is still an L1 intermediate, not a DRAM write.
        delta_dfb = ttl.make_dataflow_buffer_like(
            output,
            shape=(m_tiles_per_block, n_tiles_per_block),
            block_count=2,
        )
        output_dfb = ttl.make_dataflow_buffer_like(
            output,
            shape=(m_tiles_per_block, n_tiles_per_block),
            block_count=2,
        )

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_m_block in range(m_blocks_per_node):
                m_block = node_index * m_blocks_per_node + local_m_block
                if m_block < m_blocks:
                    m_start = m_block * m_tiles_per_block
                    m_stop = m_start + m_tiles_per_block
                    for k_block in range(k_blocks):
                        k_start = k_block * k_tiles_per_block
                        k_stop = k_start + k_tiles_per_block
                        with (
                            inputs_dfb.reserve() as inputs_block,
                            a_dfb.reserve() as a_block,
                        ):
                            tx_inputs = ttl.copy(
                                inputs[m_start:m_stop, k_start:k_stop],
                                inputs_block,
                            )
                            tx_a = ttl.copy(
                                lora_a[0:rank_tiles, k_start:k_stop],
                                a_block,
                            )
                            tx_inputs.wait()
                            tx_a.wait()

                    for n_block in range(n_blocks):
                        n_start = n_block * n_tiles_per_block
                        n_stop = n_start + n_tiles_per_block
                        with (
                            b_dfb.reserve() as b_block,
                            base_dfb.reserve() as base_block,
                        ):
                            tx_b = ttl.copy(
                                lora_b[n_start:n_stop, 0:rank_tiles],
                                b_block,
                            )
                            tx_base = ttl.copy(
                                base[m_start:m_stop, n_start:n_stop],
                                base_block,
                            )
                            tx_b.wait()
                            tx_base.wait()

        @ttl.compute()
        def compute():
            for node_m_block in range(m_blocks_per_node):
                node_col, node_row = ttl.node(dims=2)
                node_index = node_row * grid_cols + node_col
                m_block = node_index * m_blocks_per_node + node_m_block
                if m_block < m_blocks:
                    with hidden_acc_dfb.reserve() as hidden_acc:
                        hidden_acc.store(
                            ttl.block.fill(0, shape=hidden_acc.shape)
                        )

                    for _ in range(k_blocks):
                        with (
                            inputs_dfb.wait() as inputs_block,
                            a_dfb.wait() as a_block,
                            hidden_acc_dfb.wait() as previous_hidden,
                        ):
                            with hidden_acc_dfb.reserve() as next_hidden:
                                next_hidden.store(
                                    previous_hidden
                                    + ttl.matmul(
                                        inputs_block,
                                        a_block,
                                        transpose_rhs=True,
                                    )
                                )

                    # The exact TT-Metal source replaces this BF16 copy with
                    # the final pack from its FP32 partial-result CB. Keeping
                    # the DFB explicit here gives the bridge a stable BF16 CB
                    # descriptor without materializing H in DRAM.
                    with hidden_acc_dfb.wait() as hidden_acc:
                        with hidden_final_dfb.reserve() as hidden_final:
                            hidden_final.store(hidden_acc)

                    # Keep X@A.T live in L1 while all output-feature blocks
                    # consume it.  No rank-space tensor is written to DRAM.
                    with hidden_final_dfb.wait() as hidden:
                        for _ in range(n_blocks):
                            with (
                                b_dfb.wait() as b_block,
                                base_dfb.wait() as base_block,
                            ):
                                with delta_dfb.reserve() as delta_block:
                                    delta_block.store(
                                        ttl.matmul(
                                            hidden,
                                            b_block,
                                            transpose_rhs=True,
                                        )
                                    )
                                with (
                                    delta_dfb.wait() as delta,
                                    output_dfb.reserve() as output_block,
                                ):
                                    output_block.store(
                                        base_block + delta * scale_bf16
                                    )

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_m_block in range(m_blocks_per_node):
                m_block = node_index * m_blocks_per_node + local_m_block
                if m_block < m_blocks:
                    m_start = m_block * m_tiles_per_block
                    m_stop = m_start + m_tiles_per_block
                    for n_block in range(n_blocks):
                        n_start = n_block * n_tiles_per_block
                        n_stop = n_start + n_tiles_per_block
                        with output_dfb.wait() as output_block:
                            tx = ttl.copy(
                                output_block,
                                output[m_start:m_stop, n_start:n_stop],
                            )
                            tx.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.lora_forward.{variant}",
        arg_roles=("in", "in", "in", "in", "out"),
        version_tag=f"lora-forward-{variant}",
    )
    def lora_forward(inputs, lora_a, lora_b, base, output):
        _clear_ttl_compile_wrapper_cache(_lora_forward_program)
        result = _lora_forward_program(
            inputs,
            lora_a,
            lora_b,
            base,
            output,
        )
        # TT-Lang currently defaults every generic compute descriptor to
        # HiFi4 and does not expose fidelity on @ttl.operation. During PJRT's
        # compile-only callback the compiled object is available in this
        # wrapper's private cache; adjust only its compute descriptor before
        # tt_torch serializes it when a benchmark requests another variant.
        closure = _lora_forward_program.__closure__ or ()
        for name, cell in zip(
            _lora_forward_program.__code__.co_freevars,
            closure,
        ):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if exact_reduction:
                    if len(compiled.cb_configs) != 8:
                        raise RuntimeError(
                            "exact native LoRA expected 8 circular buffers, "
                            f"got {len(compiled.cb_configs)}"
                        )
                    partial_config = compiled.cb_configs[2]
                    expected_shape = (1, 1)
                    if tuple(partial_config.shape) != expected_shape:
                        raise RuntimeError(
                            "exact native LoRA partial CB moved: "
                            f"expected {expected_shape}, got "
                            f"{tuple(partial_config.shape)}"
                        )
                    partial_config.tensor = _FP32_DTYPE_ONLY_TENSOR

                    compute_paths = [
                        Path(path)
                        for path, thread_type in compiled.kernel_paths
                        if thread_type == "compute"
                    ]
                    if len(compute_paths) != 1:
                        raise RuntimeError(
                            "exact native LoRA expected one compute kernel, "
                            f"got {len(compute_paths)}"
                        )
                    grid = compiled.core_ranges.bounding_box().grid_size()
                    total_m_blocks = inputs.shape[0] // TILE_SIZE
                    total_nodes = int(grid.x) * int(grid.y)
                    m_blocks_per_node = -(
                        -total_m_blocks // total_nodes
                    )
                    template_path = (
                        Path(__file__).with_name("cpp")
                        / "lora_forward_exact_compute.cpp"
                    )
                    source = template_path.read_text(encoding="utf-8")
                    second_subblock_w = min(4, n_tiles_per_block)
                    while n_tiles_per_block % second_subblock_w:
                        second_subblock_w -= 1
                    replacements = {
                        "@NUM_K_BLOCKS@": str(
                            inputs.shape[1] // TILE_SIZE // 2
                        ),
                        "@NUM_N_BLOCKS@": str(
                            lora_b.shape[0]
                            // TILE_SIZE
                            // n_tiles_per_block
                        ),
                        "@N_TILES_PER_BLOCK@": str(
                            n_tiles_per_block
                        ),
                        "@SECOND_SUBBLOCK_W@": str(
                            second_subblock_w
                        ),
                        "@M_BLOCKS_PER_NODE@": str(
                            m_blocks_per_node
                        ),
                        "@TOTAL_M_BLOCKS@": str(total_m_blocks),
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
                            "unexpanded placeholder in exact native LoRA "
                            "TT-Metal source"
                        )
                    compute_paths[0].write_text(
                        source,
                        encoding="utf-8",
                    )
                for config in compiled.kernel_configs:
                    if not hasattr(config, "math_fidelity"):
                        continue
                    enum_type = type(config.math_fidelity)
                    config.math_fidelity = getattr(enum_type, math_fidelity)
            break
        return result

    return lora_forward
