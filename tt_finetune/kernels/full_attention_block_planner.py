"""Hardware plan for a resident Llama-family attention autograd owner.

The existing attention-core owner deliberately stops at dQ/dK/dV.  This
planner describes the larger boundary that starts at the gradient after the
output projection and ends at the decoder-block input plus the eight LoRA
parameter gradients.  It is intentionally independent from Transformers so
the TT-Metal implementation can be selected by tensor geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

from .attention_block_planner import (
    BLACKHOLE_L1_BYTES,
    TILE_SIZE,
    plan_attention_block,
)
from .full_attention_projection_handoff import (
    plan_attention_projection_l1_handoff,
    plan_qkv_resident_projection,
)


BF16_TILE_BYTES = TILE_SIZE * TILE_SIZE * 2


@dataclass(frozen=True)
class FullAttentionBlockPlan:
    batch_size: int
    sequence_length: int
    hidden_features: int
    query_heads: int
    kv_heads: int
    head_dim: int
    logical_rank: int
    tokens: int
    query_features: int
    kv_features: int
    token_tiles: int
    hidden_tiles: int
    query_tiles: int
    kv_tiles: int
    rank_tiles: int
    attention_workers: int
    projection_workers: int
    phase_peak_l1_bytes: int
    l1_budget_bytes: int
    resident_handoff_bytes: int
    qkv_projection_noc_bytes: int
    qkv_projection_phases: tuple[str, ...]
    old_intermediate_dram_bytes: int
    final_output_bytes: int
    final_tensor_count: int

    @property
    def removes_intermediate_dram_bytes(self) -> int:
        return self.old_intermediate_dram_bytes


def _tiles(rows: int, columns: int) -> int:
    return (rows // TILE_SIZE) * (columns // TILE_SIZE)


def plan_full_attention_block(
    *,
    batch_size: int,
    sequence_length: int,
    hidden_features: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    logical_rank: int,
    l1_fraction: float = 0.75,
) -> FullAttentionBlockPlan:
    """Return a conservative phase-reused L1 and traffic plan.

    The full owner is phased.  It never attempts to place all model weights or
    all activations in one core's L1.  ``phase_peak_l1_bytes`` is therefore the
    maximum of the output-projection, attention-core and Q/K/V-projection
    phases plus the local handoff window, rather than their (incorrect) sum.
    """

    if hidden_features <= 0 or hidden_features % TILE_SIZE:
        raise ValueError("hidden_features must be positive and tile aligned")
    if logical_rank <= 0 or logical_rank > TILE_SIZE:
        raise ValueError("logical_rank must be in [1, 32]")
    if not 0.0 < l1_fraction <= 0.80:
        raise ValueError("l1_fraction must be in (0, 0.80]")

    core = plan_attention_block(
        batch_size=batch_size,
        sequence_length=sequence_length,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        l1_fraction=min(l1_fraction, 0.80),
    )
    projection_handoff = plan_attention_projection_l1_handoff(
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    qkv_projection = plan_qkv_resident_projection(
        tokens=batch_size * sequence_length,
        hidden_features=hidden_features,
        kv_features=kv_heads * head_dim,
    )
    if core.kv_group_size % 2 != 1:
        raise ValueError(
            "exact resident GQA reduction currently requires an odd group"
        )
    tokens = batch_size * sequence_length
    if tokens % TILE_SIZE:
        raise ValueError("flattened token count must be tile aligned")
    query_features = query_heads * head_dim
    kv_features = kv_heads * head_dim
    if query_features != hidden_features:
        raise ValueError(
            "first full owner requires query width == hidden width"
        )
    if kv_features % TILE_SIZE:
        raise ValueError("KV width must be tile aligned")

    token_tiles = tokens // TILE_SIZE
    hidden_tiles = hidden_features // TILE_SIZE
    query_tiles = query_features // TILE_SIZE
    kv_tiles = kv_features // TILE_SIZE
    rank_tiles = 1  # logical ranks are padded to one physical Tensix tile.

    # Projection phases stream long-K weights.  Per worker we retain a
    # double-buffered operand pair, FP32 partials, rank-space dH, one dX tile,
    # transpose scratch, and output staging.  The explicit 20-tile reserve
    # covers CB bookkeeping and exact BF16 pack boundaries.
    largest_projection_width_tiles = max(
        hidden_tiles, query_tiles, kv_tiles
    )
    projection_tiles_per_worker = (
        2 * 2                         # double-buffered A/B operands
        + 2                           # FP32 partial spill/reload
        + token_tiles * rank_tiles    # resident LoRA dH window
        + 2                           # base and LoRA dX tiles
        + largest_projection_width_tiles
        + 20
    )
    projection_peak = projection_tiles_per_worker * BF16_TILE_BYTES

    # The core planner accounts FP32 buffers at their true 4-byte width.
    # Add two BF16 handoff windows: post-o dX and post-RoPE dQ/dK/dV.  They are
    # sharded, so a worker owns at most one tile from each window at a time.
    handoff_peak = 2 * BF16_TILE_BYTES
    # The attention owner retains a dQ row on each active query worker and
    # dQ+dK+dV on each member-zero GQA owner until the projection island has
    # consumed the sequential Q/V/K phases.  This is local per-core storage,
    # not aggregate L1.
    retained_projection_input_peak = (
        projection_handoff.retained_l1_bytes_per_kv_owner
    )
    qkv_sum_handoff_peak = max(
        qkv_projection.peak_source_residency_bytes,
        qkv_projection.peak_target_residency_bytes,
    )
    phase_peak = max(
        core.estimated_peak_l1_bytes + handoff_peak
        + retained_projection_input_peak,
        projection_peak + handoff_peak + qkv_sum_handoff_peak,
    )
    budget = int(BLACKHOLE_L1_BYTES * l1_fraction)
    if phase_peak > budget:
        raise ValueError(
            "full attention phase exceeds worker L1 budget: "
            f"{phase_peak} > {budget} bytes"
        )

    q_bytes = tokens * query_features * 2
    kv_bytes = tokens * kv_features * 2
    # Boundaries removed relative to the current accepted path:
    #   o-projection dX write/read, dQ/dK/dV write/read, and inverse-RoPE
    #   dQ/dK write/read before Q/K/V projection backward.
    old_intermediate = 2 * (
        q_bytes                  # post-o gradient
        + q_bytes + 2 * kv_bytes # attention-core dQ/dK/dV
        + q_bytes + kv_bytes     # inverse-RoPE dQ/dK
    )
    resident_handoff = (
        q_bytes
        + q_bytes + 2 * kv_bytes
        + q_bytes + kv_bytes
    )

    # dHidden plus A/B gradients for Q, K, V and O.  Frozen weights and biases
    # deliberately have no gradient outputs.
    adapter_elements = logical_rank * (
        # A and B for each projection.
        (hidden_features + query_features)
        + (hidden_features + kv_features)
        + (hidden_features + kv_features)
        + (query_features + hidden_features)
    )
    final_bytes = (tokens * hidden_features + adapter_elements) * 2

    return FullAttentionBlockPlan(
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        logical_rank=logical_rank,
        tokens=tokens,
        query_features=query_features,
        kv_features=kv_features,
        token_tiles=token_tiles,
        hidden_tiles=hidden_tiles,
        query_tiles=query_tiles,
        kv_tiles=kv_tiles,
        rank_tiles=rank_tiles,
        attention_workers=core.active_workers,
        projection_workers=projection_handoff.projection_workers,
        phase_peak_l1_bytes=phase_peak,
        l1_budget_bytes=budget,
        resident_handoff_bytes=resident_handoff,
        qkv_projection_noc_bytes=qkv_projection.noc_bytes,
        qkv_projection_phases=qkv_projection.phases,
        old_intermediate_dram_bytes=old_intermediate,
        final_output_bytes=final_bytes,
        final_tensor_count=9,
    )


def full_attention_block_support_error(**kwargs: object) -> str | None:
    try:
        plan_full_attention_block(**kwargs)  # type: ignore[arg-type]
    except ValueError as error:
        return str(error)
    return None
