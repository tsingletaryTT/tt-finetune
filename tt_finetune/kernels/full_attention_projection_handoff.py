"""L1/NoC ownership plan for attention-core -> Q/K/V backward.

The accepted output-attention owner leaves one remaining device boundary:
the resident attention core writes dQ/dK/dV before the projection backward
program reads them again.  This module describes the replacement boundary in
physical tiles.  It deliberately plans per-core L1 ownership and NoC readers;
it never treats aggregate chip L1 as a shared cache.

The first complete schedule retains the final attention gradients on their
60-core source island. During each sequential V, K, then Q projection phase,
the 50-core projection island pulls every required gradient tile directly
from the owning source core.  A tile is transferred at most once to a given
projection core in a phase and is then reused by base-dX, LoRA-dH and dB work
on that core.  No intermediate tensor is assigned a DRAM address.
"""

from __future__ import annotations

from dataclasses import dataclass

from .full_attention_handoff import (
    ATTENTION_WORKERS,
    PROJECTION_WORKERS,
)
from .tt_lang_lora import TILE_SIZE


BF16_TILE_BYTES = TILE_SIZE * TILE_SIZE * 2


@dataclass(frozen=True)
class ProjectionGradientTile:
    projection: str
    token_tile: int
    output_tile: int
    source_attention_node: int
    source_tile: int
    projection_consumers: tuple[int, ...]


@dataclass(frozen=True)
class AttentionProjectionHandoffPlan:
    sequence_tiles: int
    hidden_tiles: int
    query_heads: int
    kv_heads: int
    dimension_tiles: int
    group_size: int
    projection_workers: int
    attention_workers: int
    q_tiles: int
    k_tiles: int
    v_tiles: int
    noc_tile_reads: int
    noc_bytes: int
    intermediate_dram_bytes: int
    retained_tiles_per_query_worker: int
    retained_tiles_per_kv_owner: int
    retained_l1_bytes_per_kv_owner: int
    projection_order: tuple[str, ...]
    exact_dx_edges: tuple[tuple[str, str, str], ...]
    mappings: tuple[ProjectionGradientTile, ...]


@dataclass(frozen=True)
class QKVInputGradientHandoffTile:
    """One exact K/V-sum tile transferred into the Q projection island."""

    job: int
    token_tile: int
    hidden_tile: int
    kv_source_node: int
    kv_source_slot: int
    q_target_node: int
    q_target_slot: int


@dataclass(frozen=True)
class QKVResidentProjectionPlan:
    """Two-phase, one-program replacement for serial V -> K -> Q reuse.

    K and V execute in the silicon-qualified 55+55-core resident program.
    Its exact three-edge result is retained on the K/member-zero island and
    sent once to the 50-core Q island.  Q then performs the final two BF16
    edges.  This avoids re-entering one large projection compute body three
    times, which is not a safe Blackhole schedule.
    """

    token_tiles: int
    hidden_tiles: int
    kv_tiles: int
    kv_member_workers: int
    q_workers: int
    kv_jobs_per_node: int
    q_jobs_per_node: int
    phases: tuple[str, ...]
    custom_calls: int
    intermediate_dram_bytes: int
    local_direct_tiles: int
    local_copy_tiles: int
    noc_tiles: int
    noc_bytes: int
    peak_source_residency_bytes: int
    peak_target_residency_bytes: int
    exact_dx_edges: tuple[tuple[str, str, str], ...]
    handoff: tuple[QKVInputGradientHandoffTile, ...]


def plan_qkv_resident_projection(
    *, tokens: int, hidden_features: int, kv_features: int,
) -> QKVResidentProjectionPlan:
    """Plan the accepted K/V-parallel -> Q resident projection schedule."""

    if any(
        value <= 0 or value % TILE_SIZE
        for value in (tokens, hidden_features, kv_features)
    ):
        raise ValueError("QKV projection dimensions must be positive and /32")
    token_tiles = tokens // TILE_SIZE
    hidden_tiles = hidden_features // TILE_SIZE
    kv_tiles = kv_features // TILE_SIZE
    kv_member_workers = 55
    q_workers = PROJECTION_WORKERS
    jobs = token_tiles * hidden_tiles
    kv_jobs_per_node = -(-jobs // kv_member_workers)
    q_jobs_per_node = -(-jobs // q_workers)
    handoff = []
    for job in range(jobs):
        token_tile, hidden_tile = divmod(job, hidden_tiles)
        kv_source_node, kv_source_slot = divmod(job, kv_jobs_per_node)
        q_target_node, q_target_slot = divmod(job, q_jobs_per_node)
        handoff.append(
            QKVInputGradientHandoffTile(
                job=job,
                token_tile=token_tile,
                hidden_tile=hidden_tile,
                kv_source_node=kv_source_node,
                kv_source_slot=kv_source_slot,
                q_target_node=q_target_node,
                q_target_slot=q_target_slot,
            )
        )
    exact_dx_edges = (
        ("v_base", "v_lora", "v_sum"),
        ("v_sum", "k_lora", "vk_lora_sum"),
        ("vk_lora_sum", "k_base", "kv_sum"),
        ("kv_sum", "q_lora", "kvq_lora_sum"),
        ("kvq_lora_sum", "q_base", "dhidden"),
    )
    local_direct_tiles = sum(
        tile.kv_source_node == tile.q_target_node
        and tile.kv_source_slot == tile.q_target_slot
        for tile in handoff
    )
    local_copy_tiles = sum(
        tile.kv_source_node == tile.q_target_node
        and tile.kv_source_slot != tile.q_target_slot
        for tile in handoff
    )
    noc_tiles = jobs - local_direct_tiles - local_copy_tiles
    return QKVResidentProjectionPlan(
        token_tiles=token_tiles,
        hidden_tiles=hidden_tiles,
        kv_tiles=kv_tiles,
        kv_member_workers=kv_member_workers,
        q_workers=q_workers,
        kv_jobs_per_node=kv_jobs_per_node,
        q_jobs_per_node=q_jobs_per_node,
        phases=("parallel_kv", "q_tail"),
        custom_calls=1,
        intermediate_dram_bytes=0,
        local_direct_tiles=local_direct_tiles,
        local_copy_tiles=local_copy_tiles,
        noc_tiles=noc_tiles,
        noc_bytes=noc_tiles * BF16_TILE_BYTES,
        peak_source_residency_bytes=kv_jobs_per_node * BF16_TILE_BYTES,
        peak_target_residency_bytes=q_jobs_per_node * BF16_TILE_BYTES,
        exact_dx_edges=exact_dx_edges,
        handoff=tuple(handoff),
    )


def _projection_consumers(
    *, token_tile: int, output_tile: int, sequence_tiles: int,
    hidden_tiles: int, output_tiles: int,
) -> tuple[int, ...]:
    """Return the 50-core consumers for one dY tile.

    The accepted exact projection schedule has three uses of dY:

    * every base-dX job ``(token_tile, hidden_tile)`` reduces over output;
    * one row owner computes the rank-space dH;
    * one output-column owner computes dB.

    The same core may own more than one role.  Deduplicating here is what
    turns the remote tile into a resident local input instead of repeating a
    NoC read for each mathematical consumer.
    """

    jobs = sequence_tiles * hidden_tiles
    jobs_per_node = -(-jobs // PROJECTION_WORKERS)
    consumers = {
        (token_tile * hidden_tiles + hidden_tile) // jobs_per_node
        for hidden_tile in range(hidden_tiles)
    }
    # dH uses the first M workers.  dB distributes output columns in the same
    # contiguous ceil-div schedule as resident_down_backward.
    consumers.add(token_tile)
    output_tiles_per_node = -(-output_tiles // PROJECTION_WORKERS)
    consumers.add(output_tile // output_tiles_per_node)
    if min(consumers) < 0 or max(consumers) >= PROJECTION_WORKERS:
        raise ValueError("projection consumer escaped the 50-core island")
    return tuple(sorted(consumers))


def plan_attention_projection_l1_handoff(
    *, sequence_length: int, hidden_features: int,
    query_heads: int, kv_heads: int, head_dim: int,
) -> AttentionProjectionHandoffPlan:
    """Plan exact dQ/dK/dV residency through Q/K/V projection backward."""

    dimensions = (sequence_length, hidden_features, head_dim)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError("handoff dimensions must be positive and tile aligned")
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by positive KV heads")
    if query_heads * head_dim != hidden_features:
        raise ValueError("query-head geometry must exactly cover hidden width")

    sequence_tiles = sequence_length // TILE_SIZE
    hidden_tiles = hidden_features // TILE_SIZE
    dimension_tiles = head_dim // TILE_SIZE
    kv_output_tiles = kv_heads * dimension_tiles
    group = query_heads // kv_heads
    active_attention_workers = query_heads * sequence_tiles
    if active_attention_workers > ATTENTION_WORKERS:
        raise ValueError("attention geometry exceeds the 60-core island")

    mappings: list[ProjectionGradientTile] = []
    # V -> K -> Q is not a performance preference.  It is the framework's
    # visible BF16 input-gradient tree and therefore part of the exactness
    # contract.  Each edge below packs before the next edge consumes it.
    projection_order = ("v", "k", "q")
    exact_dx_edges = (
        ("v_base", "v_lora", "v_sum"),
        ("v_sum", "k_lora", "vk_lora_sum"),
        ("vk_lora_sum", "k_base", "kv_sum"),
        ("kv_sum", "q_lora", "kvq_lora_sum"),
        ("kvq_lora_sum", "q_base", "dhidden"),
    )
    for projection, heads, output_tiles in (
        ("v", kv_heads, kv_output_tiles),
        ("k", kv_heads, kv_output_tiles),
        ("q", query_heads, hidden_tiles),
    ):
        for token_tile in range(sequence_tiles):
            for output_tile in range(output_tiles):
                head = output_tile // dimension_tiles
                dim = output_tile - head * dimension_tiles
                if projection == "q":
                    source = head * sequence_tiles + token_tile
                else:
                    # Exact GQA dK/dV reduction is retained by member-zero's
                    # query-row worker for this KV head.
                    source = head * group * sequence_tiles + token_tile
                mappings.append(
                    ProjectionGradientTile(
                        projection=projection,
                        token_tile=token_tile,
                        output_tile=output_tile,
                        source_attention_node=source,
                        source_tile=dim,
                        projection_consumers=_projection_consumers(
                            token_tile=token_tile,
                            output_tile=output_tile,
                            sequence_tiles=sequence_tiles,
                            hidden_tiles=hidden_tiles,
                            output_tiles=output_tiles,
                        ),
                    )
                )

    q_tiles = sequence_tiles * hidden_tiles
    kv_tiles = sequence_tiles * kv_output_tiles
    noc_tile_reads = sum(len(item.projection_consumers) for item in mappings)
    retained_query = dimension_tiles
    # A member-zero source owns its ordinary dQ row and the reduced dK/dV row.
    retained_owner = 3 * dimension_tiles
    return AttentionProjectionHandoffPlan(
        sequence_tiles=sequence_tiles,
        hidden_tiles=hidden_tiles,
        query_heads=query_heads,
        kv_heads=kv_heads,
        dimension_tiles=dimension_tiles,
        group_size=group,
        projection_workers=PROJECTION_WORKERS,
        attention_workers=ATTENTION_WORKERS,
        q_tiles=q_tiles,
        k_tiles=kv_tiles,
        v_tiles=kv_tiles,
        noc_tile_reads=noc_tile_reads,
        noc_bytes=noc_tile_reads * BF16_TILE_BYTES,
        intermediate_dram_bytes=0,
        retained_tiles_per_query_worker=retained_query,
        retained_tiles_per_kv_owner=retained_owner,
        retained_l1_bytes_per_kv_owner=retained_owner * BF16_TILE_BYTES,
        projection_order=projection_order,
        exact_dx_edges=exact_dx_edges,
        mappings=tuple(mappings),
    )
