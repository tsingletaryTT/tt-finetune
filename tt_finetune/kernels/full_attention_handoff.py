"""Tile ownership map for output-projection -> attention-core L1 handoff.

The fused program partitions Blackhole's 110 workers into a 50-core
projection island and a 60-core attention island.  Projection dX is produced
as token-major ``[M, H]`` tiles, while attention consumes head-major
``[QH, S, D]`` tiles.  This module makes that permutation and its semaphore
credit contract explicit before it is encoded in TT-Metal C++.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tt_lang_lora import TILE_SIZE


PROJECTION_GRID_COLUMNS = 5
PROJECTION_WORKERS = 50
ATTENTION_GRID_COLUMNS = 6
ATTENTION_LOGICAL_X_OFFSET = 5
ATTENTION_WORKERS = 60


@dataclass(frozen=True)
class AttentionHandoffTile:
    source_job: int
    token_tile: int
    hidden_tile: int
    query_head: int
    head_dim_tile: int
    full_destination_nodes: tuple[int, ...]
    row_destination_node: int


@dataclass(frozen=True)
class AttentionHandoffPlan:
    sequence_tiles: int
    dimension_tiles: int
    query_heads: int
    hidden_tiles: int
    source_tiles: int
    noc_tile_writes: int
    credits_per_attention_worker: int
    tiles_per_projection_worker: int
    mappings: tuple[AttentionHandoffTile, ...]


def attention_worker_logical_coord(node: int) -> tuple[int, int]:
    if not 0 <= node < ATTENTION_WORKERS:
        raise ValueError("attention worker node is outside the 60-core island")
    return (
        ATTENTION_LOGICAL_X_OFFSET + node % ATTENTION_GRID_COLUMNS,
        node // ATTENTION_GRID_COLUMNS,
    )


def plan_output_attention_l1_handoff(
    *, sequence_length: int, hidden_features: int,
    query_heads: int, head_dim: int,
) -> AttentionHandoffPlan:
    dimensions = (sequence_length, hidden_features, head_dim)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError("handoff dimensions must be positive and tile aligned")
    if query_heads * head_dim != hidden_features:
        raise ValueError("query-head geometry must exactly cover hidden width")
    sequence_tiles = sequence_length // TILE_SIZE
    dimension_tiles = head_dim // TILE_SIZE
    hidden_tiles = hidden_features // TILE_SIZE
    active_attention_workers = query_heads * sequence_tiles
    if active_attention_workers > ATTENTION_WORKERS:
        raise ValueError("attention geometry exceeds the 60-core island")

    mappings: list[AttentionHandoffTile] = []
    for token_tile in range(sequence_tiles):
        for hidden_tile in range(hidden_tiles):
            head = hidden_tile // dimension_tiles
            dim = hidden_tile - head * dimension_tiles
            destinations = tuple(
                head * sequence_tiles + row
                for row in range(sequence_tiles)
            )
            mappings.append(
                AttentionHandoffTile(
                    source_job=token_tile * hidden_tiles + hidden_tile,
                    token_tile=token_tile,
                    hidden_tile=hidden_tile,
                    query_head=head,
                    head_dim_tile=dim,
                    full_destination_nodes=destinations,
                    row_destination_node=head * sequence_tiles + token_tile,
                )
            )

    source_tiles = sequence_tiles * hidden_tiles
    # Each source tile is placed into every query-row owner's full-head CB and
    # once more into the matching row owner's narrow dO-row CB.
    noc_writes = source_tiles * (sequence_tiles + 1)
    return AttentionHandoffPlan(
        sequence_tiles=sequence_tiles,
        dimension_tiles=dimension_tiles,
        query_heads=query_heads,
        hidden_tiles=hidden_tiles,
        source_tiles=source_tiles,
        noc_tile_writes=noc_writes,
        credits_per_attention_worker=sequence_tiles * dimension_tiles,
        tiles_per_projection_worker=(
            source_tiles + PROJECTION_WORKERS - 1
        ) // PROJECTION_WORKERS,
        mappings=tuple(mappings),
    )
