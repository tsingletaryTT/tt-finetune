"""Reference numerical schedule for the full resident attention owner.

These helpers are deliberately small and explicit.  They are the executable
contract used while replacing each framework boundary with TT-Metal phases;
changing an association or a BF16 boundary here is a numerical-contract
change, not a harmless refactor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    if half * 2 != value.shape[-1]:
        raise ValueError("RoPE head dimension must be even")
    first = value[..., :half]
    second = value[..., half:]
    return torch.cat((-second, first), dim=-1)


def inverse_rope_gradient_exact(
    output_gradient: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Match autograd for ``x*cos + rotate_half(x)*sin`` exactly.

    In particular, both multiplies cross their ordinary dtype boundary before
    the final add.  A native implementation must not accumulate both terms in
    FP32 and pack once, because that changes BF16 results.
    """

    sine_term = output_gradient * sine
    cosine_term = output_gradient * cosine
    return cosine_term - rotate_half(sine_term)


@dataclass(frozen=True)
class FrozenLoRAForward:
    output: torch.Tensor
    rank_hidden: torch.Tensor


def frozen_lora_forward_exact(
    inputs: torch.Tensor,
    frozen_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
    bias: torch.Tensor | None = None,
) -> FrozenLoRAForward:
    """Reference forward boundary for one frozen PEFT Linear.

    ``F.linear`` intentionally owns the optional base bias.  The native
    projection reproduces TTNN's FP32 final-partial reload, BF16 bias add and
    BF16 pack before adding the separately packed LoRA branch.
    """

    base = F.linear(inputs, frozen_weight, bias)
    rank_hidden = inputs @ lora_a.transpose(0, 1)
    delta = (rank_hidden @ lora_b.transpose(0, 1)) * float(scale)
    return FrozenLoRAForward(output=base + delta, rank_hidden=rank_hidden)


@dataclass(frozen=True)
class FullAttentionForward:
    output: torch.Tensor
    probability_fp32: torch.Tensor
    probability_bf16: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    head_output: torch.Tensor
    q_rank_hidden: torch.Tensor
    k_rank_hidden: torch.Tensor
    v_rank_hidden: torch.Tensor
    o_rank_hidden: torch.Tensor


def full_attention_forward_exact(
    inputs: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float,
    q_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor | None],
    k_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor | None],
    v_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor | None],
    o_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor | None],
) -> FullAttentionForward:
    """Executable exact schedule for the single-call attention owner.

    The returned values are precisely the final tensor plus checkpoints needed
    by the already-qualified resident backward.  Softmax remains FP32 for the
    checkpoint and crosses a distinct BF16 boundary before ``P @ V``.
    """

    if inputs.ndim != 2:
        raise ValueError("full attention reference expects [tokens, hidden]")
    tokens, hidden = inputs.shape
    if query_heads * head_dim != hidden:
        raise ValueError("query heads do not cover hidden width")
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")

    q_linear = frozen_lora_forward_exact(inputs, *q_parts)
    k_linear = frozen_lora_forward_exact(inputs, *k_parts)
    v_linear = frozen_lora_forward_exact(inputs, *v_parts)
    query = q_linear.output.reshape(tokens, query_heads, head_dim).transpose(0, 1)
    key = k_linear.output.reshape(tokens, kv_heads, head_dim).transpose(0, 1)
    value = v_linear.output.reshape(tokens, kv_heads, head_dim).transpose(0, 1)
    query = query * cosine + rotate_half(query) * sine
    key = key * cosine + rotate_half(key) * sine

    group = query_heads // kv_heads
    expanded_key = key.repeat_interleave(group, dim=0)
    expanded_value = value.repeat_interleave(group, dim=0)
    score = (query @ expanded_key.transpose(-2, -1)) * float(scale)
    if attention_mask is not None:
        score = score + attention_mask[..., :tokens]
    probability_fp32 = F.softmax(score, dim=-1, dtype=torch.float32)
    probability_bf16 = probability_fp32.to(query.dtype)
    head_output_heads = probability_bf16 @ expanded_value
    head_output = head_output_heads.transpose(0, 1).reshape(tokens, hidden)
    o_linear = frozen_lora_forward_exact(head_output, *o_parts)
    return FullAttentionForward(
        output=o_linear.output,
        probability_fp32=probability_fp32,
        probability_bf16=probability_bf16,
        query=query,
        key=key,
        value=value,
        head_output=head_output,
        q_rank_hidden=q_linear.rank_hidden,
        k_rank_hidden=k_linear.rank_hidden,
        v_rank_hidden=v_linear.rank_hidden,
        o_rank_hidden=o_linear.rank_hidden,
    )


@dataclass(frozen=True)
class FrozenLoRABackward:
    input_gradient: torch.Tensor
    base_input_gradient: torch.Tensor
    lora_input_gradient: torch.Tensor
    a_gradient: torch.Tensor
    b_gradient: torch.Tensor


def frozen_lora_backward_exact(
    inputs: torch.Tensor,
    output_gradient: torch.Tensor,
    frozen_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
) -> FrozenLoRABackward:
    """Reference PEFT Linear backward with a frozen base weight.

    The schedule mirrors the observable PyTorch/PEFT graph: scale dY before
    both LoRA branches, materialize the rank-space gradient, then add LoRA dX
    to frozen-base dX.  dA/dB retain their final transpose boundaries.
    """

    scaled_output_gradient = output_gradient * float(scale)
    rank_hidden = inputs @ lora_a.transpose(0, 1)
    rank_gradient = scaled_output_gradient @ lora_b
    base_input_gradient = output_gradient @ frozen_weight
    lora_input_gradient = rank_gradient @ lora_a
    input_gradient = lora_input_gradient + base_input_gradient
    a_gradient = (inputs.transpose(0, 1) @ rank_gradient).transpose(0, 1)
    # PEFT's Linear(B) weight gradient is emitted as H.T @ scaled_dY and only
    # then transposed back to B's [out, rank] parameter layout.  Computing the
    # algebraically equivalent scaled_dY.T @ H changes TTNN's reduction tree.
    b_gradient = (
        rank_hidden.transpose(0, 1) @ scaled_output_gradient
    ).transpose(0, 1)
    return FrozenLoRABackward(
        input_gradient=input_gradient,
        base_input_gradient=base_input_gradient,
        lora_input_gradient=lora_input_gradient,
        a_gradient=a_gradient,
        b_gradient=b_gradient,
    )


def accumulate_qkv_input_gradients_exact(
    query: FrozenLoRABackward,
    key: FrozenLoRABackward,
    value: FrozenLoRABackward,
) -> torch.Tensor:
    """Preserve the five-edge Llama/PEFT order observed in StableHLO.

    AOTAutograd does not first finish Q, K and V and then add three values. It
    emits V's completed gradient, followed by K-LoRA, K-base, Q-LoRA and
    Q-base. This distinction is visible in BF16 to every preceding layer.
    """

    result = value.input_gradient
    result = result + key.lora_input_gradient
    result = result + key.base_input_gradient
    result = result + query.lora_input_gradient
    result = result + query.base_input_gradient
    return result
