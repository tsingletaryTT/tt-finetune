"""Single autograd owner for exact native SwiGLU LoRA MLP training."""

from __future__ import annotations

import torch

from .tt_lang_full_mlp_backward import build_tt_lang_full_mlp_backward
from .tt_lang_grouped2_logical_forward import (
    build_tt_lang_grouped2_logical_forward,
)
from .tt_lang_resident_swiglu_down import (
    build_tt_lang_resident_swiglu_down_forward,
)


class NativeResidentFullMLPFunction(torch.autograd.Function):
    """Own gate/up forward and the complete seven-output MLP backward."""

    @staticmethod
    def forward(
        ctx,
        inputs,
        grouped_weight_t,
        gate_weight,
        gate_a,
        gate_b,
        up_weight,
        up_a,
        up_b,
        gate_up_scale,
        gate_up_base_phase,
        down_weight,
        down_weight_t,
        down_a,
        down_b,
        down_scale,
        grouped_forward_arena,
        down_forward_arena,
        input_gradient_output,
        gate_a_gradient_output,
        gate_b_gradient_output,
        up_a_gradient_output,
        up_b_gradient_output,
        down_a_gradient_output,
        down_b_gradient_output,
    ):
        tokens, hidden = inputs.shape
        intermediate = gate_weight.shape[0]
        grouped_forward = build_tt_lang_grouped2_logical_forward(
            scale=float(gate_up_scale),
            tokens=tokens,
            input_features=hidden,
            member_output_features=intermediate,
            base_k_tiles_per_phase=int(gate_up_base_phase),
            save_input_checkpoint=True,
        )
        grouped_arena = grouped_forward(
            inputs,
            grouped_weight_t,
            gate_a,
            gate_b,
            up_a,
            up_b,
            grouped_forward_arena,
        )
        gate = grouped_arena[:, :intermediate]
        up = grouped_arena[:, intermediate : 2 * intermediate]
        gate_hidden = grouped_arena[
            :, 2 * intermediate : 2 * intermediate + 32
        ]
        up_hidden = grouped_arena[
            :, 2 * intermediate + 32 : 2 * intermediate + 64
        ]
        down_forward = build_tt_lang_resident_swiglu_down_forward(
            scale=float(down_scale),
            tokens=tokens,
            hidden_features=hidden,
            intermediate_features=intermediate,
            physical_rank=down_a.shape[0],
            # Match the accepted resident MLP numerical contract: sigmoid,
            # gate multiplication, and up multiplication cross the same BF16
            # pack boundaries as the reference path.  The fused activation
            # schedule is mathematically equivalent but not bit-identical.
            fp32_activation=False,
            decomposed_activation=True,
            # Preserve the accepted forward program completion/layout
            # contract while full backward recomputes H independently.
            save_rank_hidden=True,
            # Gate/up are views into a checkpoint arena whose per-token row
            # also contains two rank tiles and the saved input.  Consume the
            # arena directly with explicit native page IDs; treating either
            # view as a tightly packed PJRT tensor crosses row boundaries.
            packed_gate_up_arena=True,
            packed_gate_up_row_tiles=(
                2 * (intermediate // 32) + 2 + (hidden // 32)
            ),
        )
        down_arena = down_forward(
            grouped_arena,
            grouped_arena,
            down_weight_t,
            down_a,
            down_b,
            down_forward_arena,
        )
        # Preserve the accepted sliced-output numerical contract while the H
        # columns remain unwritten padding.  Backward recomputes H internally.
        output = down_arena[:, :hidden]
        ctx.save_for_backward(
            grouped_arena,
            gate_weight,
            gate_a,
            gate_b,
            up_weight,
            up_a,
            up_b,
            down_weight,
            down_a,
            down_b,
            down_arena,
            input_gradient_output,
            gate_a_gradient_output,
            gate_b_gradient_output,
            up_a_gradient_output,
            up_b_gradient_output,
            down_a_gradient_output,
            down_b_gradient_output,
        )
        ctx.gate_up_scale = float(gate_up_scale)
        ctx.down_scale = float(down_scale)
        return output

    @staticmethod
    def backward(ctx, output_gradient):
        (
            grouped_checkpoint_arena,
            gate_weight,
            gate_a,
            gate_b,
            up_weight,
            up_a,
            up_b,
            down_weight,
            down_a,
            down_b,
            down_arena,
            input_gradient_output,
            gate_a_gradient_output,
            gate_b_gradient_output,
            up_a_gradient_output,
            up_b_gradient_output,
            down_a_gradient_output,
            down_b_gradient_output,
        ) = ctx.saved_tensors
        tokens = grouped_checkpoint_arena.shape[0]
        hidden = gate_weight.shape[1]
        intermediate = gate_weight.shape[0]
        operation = build_tt_lang_full_mlp_backward(
            down_scale=ctx.down_scale,
            gate_up_scale=ctx.gate_up_scale,
            tokens=tokens,
            hidden_features=hidden,
            intermediate_features=intermediate,
            physical_rank=down_a.shape[0],
        )
        (
            input_gradient,
            gate_a_gradient,
            gate_b_gradient,
            up_a_gradient,
            up_b_gradient,
            down_a_gradient,
            down_b_gradient,
        ) = operation(
            output_gradient,
            down_weight,
            down_b,
            down_arena,  # Lifetime anchor; H is recomputed internally.
            down_a,
            # The complete grouped checkpoint replaces the old external
            # product tensor. The TT-Metal program reads its packed gate/up
            # regions and recreates product tiles directly in local L1.
            grouped_checkpoint_arena,
            grouped_checkpoint_arena,
            grouped_checkpoint_arena,
            gate_weight,
            gate_b,
            grouped_checkpoint_arena,
            gate_a,
            up_weight,
            up_b,
            grouped_checkpoint_arena,
            up_a,
            grouped_checkpoint_arena,
            input_gradient_output,
            gate_a_gradient_output,
            gate_b_gradient_output,
            up_a_gradient_output,
            up_b_gradient_output,
            down_a_gradient_output,
            down_b_gradient_output,
        )
        return (
            input_gradient,       # inputs
            None,                 # grouped_weight_t
            None,                 # gate_weight
            gate_a_gradient,      # gate_a
            gate_b_gradient,      # gate_b
            None,                 # up_weight
            up_a_gradient,        # up_a
            up_b_gradient,        # up_b
            None,                 # gate_up_scale
            None,                 # gate_up_base_phase
            None,                 # down_weight
            None,                 # down_weight_t
            down_a_gradient,      # down_a
            down_b_gradient,      # down_b
            None,                 # down_scale
            None,                 # grouped_forward_arena
            None,                 # down_forward_arena
            None,                 # input_gradient_output
            None,                 # gate_a_gradient_output
            None,                 # gate_b_gradient_output
            None,                 # up_a_gradient_output
            None,                 # up_b_gradient_output
            None,                 # down_a_gradient_output
            None,                 # down_b_gradient_output
        )


def native_resident_full_mlp(
    inputs: torch.Tensor,
    grouped_weight_t: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    up_weight: torch.Tensor,
    up_a: torch.Tensor,
    up_b: torch.Tensor,
    gate_up_scale: float,
    gate_up_base_phase: int,
    down_weight: torch.Tensor,
    down_weight_t: torch.Tensor,
    down_a: torch.Tensor,
    down_b: torch.Tensor,
    down_scale: float,
    grouped_forward_arena: torch.Tensor,
    down_forward_arena: torch.Tensor,
    input_gradient_output: torch.Tensor,
    gate_a_gradient_output: torch.Tensor,
    gate_b_gradient_output: torch.Tensor,
    up_a_gradient_output: torch.Tensor,
    up_b_gradient_output: torch.Tensor,
    down_a_gradient_output: torch.Tensor,
    down_b_gradient_output: torch.Tensor,
) -> torch.Tensor:
    return NativeResidentFullMLPFunction.apply(
        inputs,
        grouped_weight_t,
        gate_weight,
        gate_a,
        gate_b,
        up_weight,
        up_a,
        up_b,
        float(gate_up_scale),
        int(gate_up_base_phase),
        down_weight,
        down_weight_t,
        down_a,
        down_b,
        float(down_scale),
        grouped_forward_arena,
        down_forward_arena,
        input_gradient_output,
        gate_a_gradient_output,
        gate_b_gradient_output,
        up_a_gradient_output,
        up_b_gradient_output,
        down_a_gradient_output,
        down_b_gradient_output,
    )
