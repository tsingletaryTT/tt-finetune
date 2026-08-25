"""Experimental autograd owner for resident causal-GQA attention core."""

from __future__ import annotations

import torch

from .attention_block_planner import plan_attention_block
from .native_attention_core import (
    make_attention_gradient_arena,
    native_attention_core_backward,
)


def _validate_attention_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    if query.dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError("resident attention requires float32 or bfloat16")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("resident attention Q/K/V must share one dtype")
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("Q/K/V must be [B,H,S,D], with equal K/V shapes")
    batch, query_heads, sequence, head_dim = query.shape
    if key.shape[0] != batch or key.shape[2:] != (sequence, head_dim):
        raise ValueError("key geometry must match query batch/sequence/dim")
    plan_attention_block(
        batch_size=batch,
        sequence_length=sequence,
        query_heads=query_heads,
        kv_heads=key.shape[1],
        head_dim=head_dim,
    )
    if batch != 1 or sequence != 128:
        raise ValueError(
            "first resident attention autograd schedule requires B=1, S=128"
        )


class NativeResidentAttentionCoreFunction(torch.autograd.Function):
    """Attach one-call core backward to an already exact forward result.

    This boundary is intentionally opt-in. The safe generic graph remains the
    default even though the qualified SmolLM2/Qwen schedules are bit-exact.
    """

    @staticmethod
    def forward(
        ctx,
        exact_output,
        probability,
        query,
        key,
        value,
        scale,
        gradient_arena,
        resident_backward,
    ):
        # The Metal reader owns a fixed tiled page contract.  Save explicit
        # contiguous checkpoints so XLA cannot reuse a transposed/view layout
        # chosen by another forward consumer for the custom-call inputs.  The
        # generic A/B control uses this identical checkpoint boundary and lets
        # the original exact-output edge handle backward.
        ctx.query_shape = tuple(query.shape)
        ctx.key_shape = tuple(key.shape)
        ctx.save_for_backward(
            probability.detach().reshape(-1, probability.shape[-1]).clone(),
            query.detach().reshape(-1, query.shape[-1]).clone(),
            key.detach().reshape(-1, key.shape[-1]).clone(),
            value.detach().reshape(-1, value.shape[-1]).clone(),
            gradient_arena,
        )
        ctx.scale = float(scale)
        ctx.resident_backward = bool(resident_backward)
        return exact_output

    @staticmethod
    def backward(ctx, output_gradient):
        if not ctx.resident_backward:
            # Safe control: preserve the ordinary Transformers autograd graph.
            # The remaining inputs are checkpoint-only and receive no direct
            # gradients from this identity boundary.
            return output_gradient, None, None, None, None, None, None, None
        (
            probability,
            query,
            key,
            value,
            gradient_arena,
        ) = ctx.saved_tensors
        query = query.reshape(ctx.query_shape)
        key = key.reshape(ctx.key_shape)
        value = value.reshape(ctx.key_shape)
        probability = probability.reshape(
            ctx.query_shape[0], ctx.query_shape[1],
            ctx.query_shape[2], ctx.query_shape[2],
        )
        output_gradient = output_gradient.reshape(
            -1, output_gradient.shape[-1]
        ).clone().reshape(ctx.query_shape)
        dq, dk, dv = native_attention_core_backward(
            probability,
            output_gradient,
            query,
            key,
            value,
            scale=ctx.scale,
            arena=gradient_arena,
            resident_owner=ctx.resident_backward,
        )
        # Do not continue through exact_output/probability's generic autograd
        # graph. Q/K/V receive the replacement native gradients directly.
        return None, None, dq, dk, dv, None, None, None


def attach_resident_attention_backward(
    exact_output: torch.Tensor,
    probability: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    gradient_arena: torch.Tensor,
    resident_backward: bool = True,
) -> torch.Tensor:
    """Keep an exact framework forward and replace only its Q/K/V backward."""

    _validate_attention_inputs(query, key, value)
    if exact_output.shape != query.shape:
        raise ValueError("exact attention output must match query shape")
    if probability.shape != (*query.shape[:-1], query.shape[-2]):
        raise ValueError("probability must have [B,QH,S,S] shape")
    # The native owner replaces the complete Q/K/V backward.  Keeping the
    # differentiable ``exact_output`` edge alive makes AOTAutograd retain the
    # original attention backward as zero-valued work and add it to the native
    # gradients.  Detach only for the opt-in owner; the safe control still
    # returns through the untouched framework graph.
    owner_output = exact_output.detach() if resident_backward else exact_output
    return NativeResidentAttentionCoreFunction.apply(
        owner_output,
        probability.detach(),
        query,
        key,
        value,
        float(scale),
        gradient_arena,
        bool(resident_backward),
    )


def native_resident_attention_core(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
    gradient_arena: torch.Tensor | None = None,
    resident_backward: bool = False,
) -> torch.Tensor:
    """Run the opt-in resident causal-GQA attention-core autograd boundary."""

    _validate_attention_inputs(query, key, value)
    if scale is None:
        scale = query.shape[-1] ** -0.5
    if gradient_arena is None:
        gradient_arena = make_attention_gradient_arena(query, key)
    group = query.shape[1] // key.shape[1]
    expanded_key = key.repeat_interleave(group, dim=1)
    expanded_value = value.repeat_interleave(group, dim=1)
    scores = (query @ expanded_key.transpose(-2, -1)) * float(scale)
    causal_mask = torch.triu(
        torch.full_like(scores, float("-inf")), diagonal=1
    )
    probability_fp32 = torch.softmax(
        scores + causal_mask, dim=-1, dtype=torch.float32
    )
    probability = probability_fp32.to(query.dtype)
    exact_output = probability @ expanded_value
    owner_output = exact_output.detach() if resident_backward else exact_output
    return NativeResidentAttentionCoreFunction.apply(
        # These values are forward checkpoints, not differentiable edges.
        # Q/K/V below are the only gradient destinations of the owner.
        owner_output,
        probability_fp32.detach(),
        query,
        key,
        value,
        float(scale),
        gradient_arena,
        bool(resident_backward),
    )


native_resident_attention_core.experimental_backend = (
    "resident_attention_core_v2_exact_opt_in"
)
