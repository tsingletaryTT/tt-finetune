"""Autograd-compatible native execution packing for logical PEFT adapters."""

from __future__ import annotations

import torch

from .tt_lang_grouped_lora_pack import build_tt_lang_grouped2_lora_pack


class NativeGrouped2LoRAPackFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a0, b0, a1, b1, arena):
        rank = int(a0.shape[0])
        input_features = int(a0.shape[1])
        output_features = int(b0.shape[0])
        if (
            a1.shape != a0.shape
            or b1.shape != b0.shape
            or b0.shape[1] != rank
            or rank <= 0
            or rank > 32
        ):
            raise ValueError("invalid logical grouped adapter shapes")
        operation = build_tt_lang_grouped2_lora_pack(
            input_features=input_features,
            output_features=output_features,
        )
        packed = operation(a0, b0, a1, b1, arena)
        ctx.rank = rank
        ctx.input_features = input_features
        ctx.output_features = output_features
        grouped_a = packed[:64, :input_features]
        grouped_b = packed[
            64 : 64 + 2 * output_features, :64
        ]
        return grouped_a, grouped_b

    @staticmethod
    def backward(ctx, grouped_a_gradient, grouped_b_gradient):
        rank = ctx.rank
        outputs = ctx.output_features
        return (
            grouped_a_gradient[:rank, : ctx.input_features],
            grouped_b_gradient[:outputs, :rank],
            grouped_a_gradient[32 : 32 + rank, : ctx.input_features],
            grouped_b_gradient[
                outputs : 2 * outputs, 32 : 32 + rank
            ],
            None,
        )


def native_pack_grouped2_logical_adapters(
    a0: torch.Tensor,
    b0: torch.Tensor,
    a1: torch.Tensor,
    b1: torch.Tensor,
    arena: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return NativeGrouped2LoRAPackFunction.apply(a0, b0, a1, b1, arena)

