"""Generic explicit-autograd LoRA projection prototypes."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def reference_lora_delta(
    inputs: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """The standard PEFT LoRA projection order."""

    rank_hidden = functional.linear(inputs, lora_a)
    return functional.linear(rank_hidden, lora_b) * scale


class LayoutAwareLoRAFunction(torch.autograd.Function):
    """Express LoRA backward directly, without generic Linear autograd.

    This is a compiler-level prototype, not yet a native Metal kernel. It
    retains the rank-space forward activation and makes all four required
    backward matmuls explicit so StableHLO structure and exactness can be
    measured before committing to a lower-level implementation.
    """

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        rank_hidden = functional.linear(inputs, lora_a)
        output = functional.linear(rank_hidden, lora_b) * scale
        ctx.save_for_backward(inputs, rank_hidden, lora_a, lora_b)
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        inputs, rank_hidden, lora_a, lora_b = ctx.saved_tensors
        scaled_gradient = output_gradient * ctx.scale
        flat_gradient = scaled_gradient.reshape(
            -1, scaled_gradient.shape[-1]
        )
        flat_hidden = rank_hidden.reshape(-1, rank_hidden.shape[-1])
        lora_b_gradient = flat_gradient.transpose(0, 1) @ flat_hidden
        hidden_gradient = scaled_gradient @ lora_b
        flat_hidden_gradient = hidden_gradient.reshape(
            -1, hidden_gradient.shape[-1]
        )
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        lora_a_gradient = (
            flat_hidden_gradient.transpose(0, 1) @ flat_inputs
        )
        input_gradient = hidden_gradient @ lora_a
        return (
            input_gradient,
            lora_a_gradient,
            lora_b_gradient,
            None,
        )


def layout_aware_lora_delta(
    inputs: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return LayoutAwareLoRAFunction.apply(inputs, lora_a, lora_b, scale)
