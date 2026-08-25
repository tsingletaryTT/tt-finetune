"""Hardware plan for the next resident attention-forward ownership boundary."""

from __future__ import annotations

from dataclasses import dataclass

from .full_attention_forward_handoff import (
    FullAttentionForwardHandoffPlan,
    plan_full_attention_forward_handoff,
)
from .tt_lang_lora import TILE_SIZE


BF16_TILE_BYTES = TILE_SIZE * TILE_SIZE * 2
FP32_TILE_BYTES = TILE_SIZE * TILE_SIZE * 4
P150A_WORKERS = 110
PROJECTION_WORKERS = 50
ATTENTION_WORKERS = 60


@dataclass(frozen=True)
class AttentionForwardArenaSlice:
    """One logical tensor stored in the owner's BF16 physical page arena.

    ``pages_per_tile`` is two for the raw FP32 probability checkpoint.  Those
    pages are opaque to PJRT: the TT-Metal writer splits a 4096-byte FP32 tile
    over two adjacent 2048-byte pages and the backward reader joins them.
    """

    name: str
    page_offset: int
    page_count: int
    logical_tiles: int
    pages_per_tile: int
    dtype: str


@dataclass(frozen=True)
class AttentionForwardArenaLayout:
    physical_shape: tuple[int, int]
    hidden_tiles: int
    used_pages: int
    allocated_pages: int
    output_pages: int
    checkpoint_pages: int
    checkpoint_dram_bytes: int
    padding_bytes: int
    slices: tuple[AttentionForwardArenaSlice, ...]

    def slice(self, name: str) -> AttentionForwardArenaSlice:
        for region in self.slices:
            if region.name == name:
                return region
        raise KeyError(name)


@dataclass(frozen=True)
class FullAttentionForwardPlan:
    sequence_tiles: int
    hidden_tiles: int
    kv_tiles: int
    query_heads: int
    kv_heads: int
    group_size: int
    physical_rank: int
    projection_workers: int
    attention_workers: int
    phases: tuple[str, ...]
    checkpoint_rank_tiles: int
    checkpoint_rank_bytes: int
    score_tiles_per_query_owner: int
    attention_l1_bytes_per_query_owner: int
    intermediate_dram_bytes: int
    final_output_bytes: int
    qkv_bias: bool
    handoff: FullAttentionForwardHandoffPlan
    arena: AttentionForwardArenaLayout


def _plan_forward_arena(
    *,
    tokens: int,
    hidden_features: int,
    kv_features: int,
    query_heads: int,
    sequence_tiles: int,
    hidden_tiles: int,
    kv_tiles: int,
    rank_tiles: int,
    include_attention_block: bool = False,
) -> AttentionForwardArenaLayout:
    """Pack all outputs/checkpoints into one rectangular BF16 tensor.

    Only the first ``output`` region is interpreted by the framework.  The
    remaining pages are autograd-owned checkpoint storage and are consumed
    directly by the resident backward program.
    """

    regions: list[AttentionForwardArenaSlice] = []
    cursor = 0

    def allocate(
        name: str, logical_tiles: int, *, dtype: str = "bfloat16"
    ) -> None:
        nonlocal cursor
        pages_per_tile = 2 if dtype == "float32_raw" else 1
        page_count = logical_tiles * pages_per_tile
        regions.append(
            AttentionForwardArenaSlice(
                name=name,
                page_offset=cursor,
                page_count=page_count,
                logical_tiles=logical_tiles,
                pages_per_tile=pages_per_tile,
                dtype=dtype,
            )
        )
        cursor += page_count

    output_tiles = sequence_tiles * hidden_tiles
    allocate("output", output_tiles)
    allocate("query", output_tiles)
    allocate("key", sequence_tiles * kv_tiles)
    allocate("value", sequence_tiles * kv_tiles)
    allocate("head_output", output_tiles)
    for projection in ("q", "k", "v", "o"):
        allocate(f"{projection}_rank_hidden", sequence_tiles * rank_tiles)
    # [query_heads, sequence_tiles, sequence_tiles] FP32 tile matrix.
    allocate(
        "probability_fp32",
        query_heads * sequence_tiles * sequence_tiles,
        dtype="float32_raw",
    )
    if include_attention_block:
        allocate("post_attention_residual", output_tiles)
        allocate("post_attention_normalized", output_tiles)
        allocate(
            "post_attention_rstd_fp32", sequence_tiles,
            dtype="float32_raw",
        )

    used_pages = cursor
    allocated_rows = (
        (used_pages + hidden_tiles - 1) // hidden_tiles
    ) * TILE_SIZE
    allocated_pages = (allocated_rows // TILE_SIZE) * hidden_tiles
    output_pages = output_tiles
    checkpoint_pages = used_pages - output_pages
    return AttentionForwardArenaLayout(
        physical_shape=(allocated_rows, hidden_features),
        hidden_tiles=hidden_tiles,
        used_pages=used_pages,
        allocated_pages=allocated_pages,
        output_pages=output_pages,
        checkpoint_pages=checkpoint_pages,
        checkpoint_dram_bytes=checkpoint_pages * BF16_TILE_BYTES,
        padding_bytes=(allocated_pages - used_pages) * BF16_TILE_BYTES,
        slices=tuple(regions),
    )


def plan_full_attention_forward(
    *,
    tokens: int,
    hidden_features: int,
    kv_features: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    physical_rank: int = 32,
    qkv_bias: bool = False,
    include_attention_block: bool = False,
) -> FullAttentionForwardPlan:
    """Plan one Q/K/V -> RoPE -> causal attention -> O TT-Metal owner.

    The plan deliberately accounts for the forward checkpoints consumed by
    the accepted v90 backward. Rank activations are final owner outputs for
    autograd purposes; Q/K/V, scores, probabilities and head output remain
    internal L1/NoC values.
    """

    dimensions = (tokens, hidden_features, kv_features, head_dim, physical_rank)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError("forward dimensions must be positive and tile aligned")
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if query_heads * head_dim != hidden_features:
        raise ValueError("query heads do not cover hidden width")
    if kv_heads * head_dim != kv_features:
        raise ValueError("KV heads do not cover KV width")
    group = query_heads // kv_heads
    if group % 2 != 1:
        raise ValueError("qualified causal-GQA schedule requires an odd group")
    sequence_tiles = tokens // TILE_SIZE
    if query_heads * sequence_tiles > ATTENTION_WORKERS:
        raise ValueError("attention row owners exceed the 60-core island")

    hidden_tiles = hidden_features // TILE_SIZE
    kv_tiles = kv_features // TILE_SIZE
    rank_tiles = physical_rank // TILE_SIZE
    handoff = plan_full_attention_forward_handoff(
        sequence_length=tokens,
        hidden_features=hidden_features,
        kv_features=kv_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    # Q, K, V and O each save one [tokens, physical_rank] checkpoint.
    checkpoint_tiles = 4 * sequence_tiles * rank_tiles
    arena = _plan_forward_arena(
        tokens=tokens,
        hidden_features=hidden_features,
        kv_features=kv_features,
        query_heads=query_heads,
        sequence_tiles=sequence_tiles,
        hidden_tiles=hidden_tiles,
        kv_tiles=kv_tiles,
        rank_tiles=rank_tiles,
        include_attention_block=include_attention_block,
    )
    # One row owner retains a causal score/probability row, Q/K/V head tiles,
    # and two FP32 online-softmax accumulator tiles. This is intentionally a
    # conservative peak; the implementation must remain below 75% of L1.
    score_tiles = sequence_tiles
    attention_l1 = (
        (2 * score_tiles + 6) * FP32_TILE_BYTES
        + (4 * score_tiles + 8) * BF16_TILE_BYTES
    )
    return FullAttentionForwardPlan(
        sequence_tiles=sequence_tiles,
        hidden_tiles=hidden_tiles,
        kv_tiles=kv_tiles,
        query_heads=query_heads,
        kv_heads=kv_heads,
        group_size=group,
        physical_rank=physical_rank,
        projection_workers=PROJECTION_WORKERS,
        attention_workers=ATTENTION_WORKERS,
        phases=(
            "qkv_frozen_lora_projection",
            "rope",
            "causal_qk",
            "online_softmax",
            "probability_v",
            "output_frozen_lora_projection",
        ),
        checkpoint_rank_tiles=checkpoint_tiles,
        checkpoint_rank_bytes=checkpoint_tiles * BF16_TILE_BYTES,
        score_tiles_per_query_owner=score_tiles,
        attention_l1_bytes_per_query_owner=attention_l1,
        intermediate_dram_bytes=0,
        final_output_bytes=tokens * hidden_features * 2,
        qkv_bias=bool(qkv_bias),
        handoff=handoff,
        arena=arena,
    )
