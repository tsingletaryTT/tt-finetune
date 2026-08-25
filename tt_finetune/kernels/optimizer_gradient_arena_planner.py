"""Tile-page plan for backward-to-optimizer resident gradient arenas.

The accepted attention owner needs a token-by-hidden scratch prefix while its
input RMSNorm backward is still pending.  Everything after that prefix is a
dense stream of final LoRA-gradient tiles.  Keeping the stream dense is what
lets clipping and AdamW consume it without reconstructing 14 Python tensors
per Transformer layer.

The plan records physical tile pages, not logical matrix rows.  Every page is
one 32x32 BF16 tile and the public arena is represented as ``[pages * 32, 32]``.
"""

from __future__ import annotations

from dataclasses import dataclass


TILE_SIZE = 32


@dataclass(frozen=True)
class GradientArenaSlice:
    name: str
    page_offset: int
    page_count: int
    logical_shape: tuple[int, int]

    @property
    def page_stop(self) -> int:
        return self.page_offset + self.page_count


@dataclass(frozen=True)
class LayerGradientArenaPlan:
    tokens: int
    hidden_features: int
    intermediate_features: int
    kv_features: int
    scratch_pages: int
    slices: tuple[GradientArenaSlice, ...]
    total_pages: int

    @property
    def gradient_page_offset(self) -> int:
        return self.scratch_pages

    @property
    def gradient_pages(self) -> int:
        return self.total_pages - self.scratch_pages

    @property
    def physical_shape(self) -> tuple[int, int]:
        return self.total_pages * TILE_SIZE, TILE_SIZE

    def slice(self, name: str) -> GradientArenaSlice:
        matches = [item for item in self.slices if item.name == name]
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]


@dataclass(frozen=True)
class MultiLayerGradientArenaPlan:
    layer_count: int
    scratch_pages: int
    layer_gradient_pages: int
    slices: tuple[GradientArenaSlice, ...]
    total_pages: int

    @property
    def physical_shape(self) -> tuple[int, int]:
        return self.total_pages * TILE_SIZE, TILE_SIZE

    def slice(self, layer: int, name: str) -> GradientArenaSlice:
        key = f"layer{layer}.{name}"
        matches = [item for item in self.slices if item.name == key]
        if len(matches) != 1:
            raise KeyError(key)
        return matches[0]


def _require_tiled(name: str, value: int) -> None:
    if value <= 0 or value % TILE_SIZE:
        raise ValueError(f"{name} must be positive and divisible by 32")


def plan_layer_optimizer_gradient_arena(
    *,
    tokens: int,
    hidden_features: int,
    intermediate_features: int,
    kv_features: int,
    physical_rank: int = TILE_SIZE,
) -> LayerGradientArenaPlan:
    """Plan one exact layer's scratch prefix and 14 dense gradients.

    Gradient ordering follows the already-qualified physical writers.  No
    transpose or numerical operation changes are needed to target this plan.
    """

    for name, value in (
        ("tokens", tokens),
        ("hidden_features", hidden_features),
        ("intermediate_features", intermediate_features),
        ("kv_features", kv_features),
        ("physical_rank", physical_rank),
    ):
        _require_tiled(name, value)

    token_tiles = tokens // TILE_SIZE
    hidden_tiles = hidden_features // TILE_SIZE
    intermediate_tiles = intermediate_features // TILE_SIZE
    kv_tiles = kv_features // TILE_SIZE
    rank_tiles = physical_rank // TILE_SIZE
    scratch_pages = token_tiles * hidden_tiles
    cursor = scratch_pages
    slices: list[GradientArenaSlice] = []

    def append(name: str, pages: int, shape: tuple[int, int]) -> None:
        nonlocal cursor
        slices.append(GradientArenaSlice(name, cursor, pages, shape))
        cursor += pages

    # Attention projection writer order.
    append("o_a", rank_tiles * hidden_tiles, (physical_rank, hidden_features))
    append("o_b", hidden_tiles * rank_tiles, (hidden_features, physical_rank))
    append("q_a", rank_tiles * hidden_tiles, (physical_rank, hidden_features))
    append("k_a", rank_tiles * hidden_tiles, (physical_rank, hidden_features))
    append("v_a", rank_tiles * hidden_tiles, (physical_rank, hidden_features))
    append("q_b", hidden_tiles * rank_tiles, (hidden_features, physical_rank))
    append("k_b", kv_tiles * rank_tiles, (kv_features, physical_rank))
    append("v_b", kv_tiles * rank_tiles, (kv_features, physical_rank))

    # Full-MLP writer order.
    append("gate_a", rank_tiles * hidden_tiles, (physical_rank, hidden_features))
    append("gate_b", intermediate_tiles * rank_tiles, (intermediate_features, physical_rank))
    append("up_a", rank_tiles * hidden_tiles, (physical_rank, hidden_features))
    append("up_b", intermediate_tiles * rank_tiles, (intermediate_features, physical_rank))
    append("down_a", rank_tiles * intermediate_tiles, (physical_rank, intermediate_features))
    append("down_b", hidden_tiles * rank_tiles, (hidden_features, physical_rank))

    return LayerGradientArenaPlan(
        tokens=tokens,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        kv_features=kv_features,
        scratch_pages=scratch_pages,
        slices=tuple(slices),
        total_pages=cursor,
    )


def plan_multi_layer_optimizer_gradient_arena(
    *,
    layer_count: int,
    tokens: int,
    hidden_features: int,
    intermediate_features: int,
    kv_features: int,
    physical_rank: int = TILE_SIZE,
) -> MultiLayerGradientArenaPlan:
    """Share one reusable attention scratch prefix across reverse layers."""

    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    layer = plan_layer_optimizer_gradient_arena(
        tokens=tokens,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        kv_features=kv_features,
        physical_rank=physical_rank,
    )
    cursor = layer.scratch_pages
    slices: list[GradientArenaSlice] = []
    for index in range(layer_count):
        for item in layer.slices:
            relative = item.page_offset - layer.scratch_pages
            slices.append(
                GradientArenaSlice(
                    f"layer{index}.{item.name}",
                    cursor + relative,
                    item.page_count,
                    item.logical_shape,
                )
            )
        cursor += layer.gradient_pages
    return MultiLayerGradientArenaPlan(
        layer_count=layer_count,
        scratch_pages=layer.scratch_pages,
        layer_gradient_pages=layer.gradient_pages,
        slices=tuple(slices),
        total_pages=cursor,
    )
