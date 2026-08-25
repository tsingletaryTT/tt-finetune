"""Packing utilities for shared-input transformer projection groups.

Llama attention Q/K/V and MLP gate/up projections consume the same activation.
They can be represented as one larger frozen projection plus one block-diagonal
LoRA projection.  This lets a single TT-Metal program load/schedule the shared
input once and amortize a custom-call boundary across the group.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .tt_lang_lora import TILE_SIZE


@dataclass(frozen=True)
class GroupedProjectionLayout:
    output_features: tuple[int, ...]
    physical_rank_per_projection: int

    @property
    def group_size(self) -> int:
        return len(self.output_features)

    @property
    def total_output_features(self) -> int:
        return sum(self.output_features)

    @property
    def total_physical_rank(self) -> int:
        return self.group_size * self.physical_rank_per_projection

    def split_outputs(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(value.split(self.output_features, dim=-1))


def pack_shared_input_projection_group(
    frozen_weights: tuple[torch.Tensor, ...],
    lora_as: tuple[torch.Tensor, ...],
    lora_bs: tuple[torch.Tensor, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    GroupedProjectionLayout,
]:
    """Pack 2-3 independent LoRA projections into one block-diagonal form."""

    group_size = len(frozen_weights)
    if group_size not in (2, 3):
        raise ValueError("native projection groups require 2 or 3 members")
    if len(lora_as) != group_size or len(lora_bs) != group_size:
        raise ValueError("grouped weights must have equal tuple lengths")
    if any(
        tensor.ndim != 2
        for tensor in (*frozen_weights, *lora_as, *lora_bs)
    ):
        raise ValueError("grouped projection weights must be rank-2")

    input_features = frozen_weights[0].shape[1]
    physical_rank = lora_as[0].shape[0]
    output_features = tuple(weight.shape[0] for weight in frozen_weights)
    if physical_rank != TILE_SIZE:
        raise ValueError(
            "each grouped LoRA member must use physical rank 32"
        )
    for weight, lora_a, lora_b, outputs in zip(
        frozen_weights, lora_as, lora_bs, output_features
    ):
        if (
            weight.shape != (outputs, input_features)
            or lora_a.shape != (physical_rank, input_features)
            or lora_b.shape != (outputs, physical_rank)
        ):
            raise ValueError("incompatible grouped projection shapes")
        if outputs % TILE_SIZE:
            raise ValueError("grouped output features must be /32")

    layout = GroupedProjectionLayout(
        output_features=output_features,
        physical_rank_per_projection=physical_rank,
    )
    grouped_weight = torch.cat(frozen_weights, dim=0)
    grouped_a = torch.cat(lora_as, dim=0)
    grouped_b = lora_bs[0].new_zeros(
        layout.total_output_features,
        layout.total_physical_rank,
    )
    output_offset = 0
    rank_offset = 0
    for lora_b, outputs in zip(lora_bs, output_features):
        grouped_b[
            output_offset : output_offset + outputs,
            rank_offset : rank_offset + physical_rank,
        ] = lora_b
        output_offset += outputs
        rank_offset += physical_rank
    return grouped_weight, grouped_a, grouped_b, layout
