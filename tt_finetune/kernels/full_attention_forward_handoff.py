"""Physical Q/K/V projection -> attention-island L1/NoC schedule.

The full resident forward uses the same 50+60 worker split as the accepted
attention backward.  Projection results are token-major, while the attention
workers own one ``(query_head, query_row_tile)`` each.  This module makes the
otherwise easy-to-hide redistribution explicit:

* Q is sent exactly once to its query-row owner.
* K and V are sent once to a canonical row owner for each KV-head/token row.
* that owner fans the tile out inside the attention island to all query rows
  sharing the KV head.

No route has a DRAM address.  The route and credit counts are consumed by the
TT-Metal source generator and are also an executable guard against accidentally
reintroducing PJRT-visible Q/K/V tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

from .full_attention_handoff import ATTENTION_WORKERS, PROJECTION_WORKERS
from .tt_lang_lora import TILE_SIZE


BF16_TILE_BYTES = TILE_SIZE * TILE_SIZE * 2


@dataclass(frozen=True)
class ForwardProjectionTileRoute:
    projection: str
    token_tile: int
    output_tile: int
    head: int
    dimension_tile: int
    source_projection_node: int
    source_slot: int
    staging_attention_node: int
    final_attention_nodes: tuple[int, ...]


@dataclass(frozen=True)
class FullAttentionForwardHandoffPlan:
    sequence_tiles: int
    hidden_tiles: int
    kv_tiles: int
    dimension_tiles: int
    query_heads: int
    kv_heads: int
    group_size: int
    projection_workers: int
    attention_workers: int
    projection_to_attention_tiles: int
    attention_fanout_tiles: int
    noc_tile_writes: int
    noc_bytes: int
    intermediate_dram_bytes: int
    q_tiles_per_query_owner: int
    k_tiles_per_query_owner: int
    v_tiles_per_query_owner: int
    retained_tiles_per_query_owner: int
    retained_bytes_per_query_owner: int
    routes: tuple[ForwardProjectionTileRoute, ...]

    def routes_for(self, projection: str) -> tuple[ForwardProjectionTileRoute, ...]:
        return tuple(route for route in self.routes if route.projection == projection)


def _source_for_job(job: int, total_jobs: int) -> tuple[int, int]:
    jobs_per_node = -(-total_jobs // PROJECTION_WORKERS)
    return divmod(job, jobs_per_node)


def plan_full_attention_forward_handoff(
    *, sequence_length: int, hidden_features: int, kv_features: int,
    query_heads: int, kv_heads: int, head_dim: int,
) -> FullAttentionForwardHandoffPlan:
    """Plan Q/K/V residency from the projection to attention island."""

    dimensions = (sequence_length, hidden_features, kv_features, head_dim)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError("forward handoff dimensions must be positive and tile aligned")
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by positive KV heads")
    if query_heads * head_dim != hidden_features:
        raise ValueError("query heads do not cover hidden width")
    if kv_heads * head_dim != kv_features:
        raise ValueError("KV heads do not cover KV width")

    sequence_tiles = sequence_length // TILE_SIZE
    hidden_tiles = hidden_features // TILE_SIZE
    kv_tiles = kv_features // TILE_SIZE
    dimension_tiles = head_dim // TILE_SIZE
    group_size = query_heads // kv_heads
    active_attention_workers = query_heads * sequence_tiles
    if active_attention_workers > ATTENTION_WORKERS:
        raise ValueError("attention geometry exceeds the 60-core island")

    routes: list[ForwardProjectionTileRoute] = []
    for projection, output_tiles, heads in (
        ("q", hidden_tiles, query_heads),
        ("k", kv_tiles, kv_heads),
        ("v", kv_tiles, kv_heads),
    ):
        total_jobs = sequence_tiles * output_tiles
        for token_tile in range(sequence_tiles):
            for output_tile in range(output_tiles):
                job = token_tile * output_tiles + output_tile
                source_node, source_slot = _source_for_job(job, total_jobs)
                head, dimension_tile = divmod(output_tile, dimension_tiles)
                if head >= heads:
                    raise AssertionError("projection tile escaped its head geometry")
                if projection == "q":
                    staging = head * sequence_tiles + token_tile
                    destinations = (staging,)
                else:
                    first_query_head = head * group_size
                    # The canonical node owns this K/V token row.  After K is
                    # RoPE-transformed (V needs no transform), it multicasts
                    # the tile to every query-row owner in its GQA group.
                    staging = first_query_head * sequence_tiles + token_tile
                    destinations = tuple(
                        query_head * sequence_tiles + query_row
                        for query_head in range(
                            first_query_head, first_query_head + group_size
                        )
                        for query_row in range(sequence_tiles)
                    )
                routes.append(
                    ForwardProjectionTileRoute(
                        projection=projection,
                        token_tile=token_tile,
                        output_tile=output_tile,
                        head=head,
                        dimension_tile=dimension_tile,
                        source_projection_node=source_node,
                        source_slot=source_slot,
                        staging_attention_node=staging,
                        final_attention_nodes=destinations,
                    )
                )

    projection_to_attention = len(routes)
    # A staged K/V tile is already resident at its canonical destination, so
    # only the remaining destinations consume NoC fanout writes.
    attention_fanout = sum(
        len(route.final_attention_nodes) - 1
        for route in routes
        if route.projection in ("k", "v")
    )
    noc_writes = projection_to_attention + attention_fanout
    q_per_owner = dimension_tiles
    k_per_owner = sequence_tiles * dimension_tiles
    v_per_owner = sequence_tiles * dimension_tiles
    retained = q_per_owner + k_per_owner + v_per_owner
    return FullAttentionForwardHandoffPlan(
        sequence_tiles=sequence_tiles,
        hidden_tiles=hidden_tiles,
        kv_tiles=kv_tiles,
        dimension_tiles=dimension_tiles,
        query_heads=query_heads,
        kv_heads=kv_heads,
        group_size=group_size,
        projection_workers=PROJECTION_WORKERS,
        attention_workers=ATTENTION_WORKERS,
        projection_to_attention_tiles=projection_to_attention,
        attention_fanout_tiles=attention_fanout,
        noc_tile_writes=noc_writes,
        noc_bytes=noc_writes * BF16_TILE_BYTES,
        intermediate_dram_bytes=0,
        q_tiles_per_query_owner=q_per_owner,
        k_tiles_per_query_owner=k_per_owner,
        v_tiles_per_query_owner=v_per_owner,
        retained_tiles_per_query_owner=retained,
        retained_bytes_per_query_owner=retained * BF16_TILE_BYTES,
        routes=tuple(routes),
    )
