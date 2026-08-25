"""Deterministic long-sequence causal attention contract for streamed-attention.

Forward uses TTNN's native FlashAttention-2 prefill operation on XLA. Backward
recomputes one query tile at a time and never saves or creates a full SxS
probability tensor. The blockwise backward is the safe reference contract for
the later resident TT-Metal implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


TILE_SIZE = 32
DEFAULT_QUERY_CHUNK_TOKENS = 128
SAFE_EXACT_BACKWARD = "safe_exact"
NATIVE_EXPERIMENTAL_BACKWARD = "native_experimental"
_BACKWARD_BACKENDS = frozenset(
    (SAFE_EXACT_BACKWARD, NATIVE_EXPERIMENTAL_BACKWARD)
)
_TT_SDPA_AUTOGRAD_REGISTERED = False
_ACTIVE_QUERY_CHUNK_TOKENS = DEFAULT_QUERY_CHUNK_TOKENS


def configure_streamed_query_chunk_tokens(value: int) -> None:
    """Select the compile-time safe-backward query geometry.

    TT-XLA traces the registered autograd formula in a serialized worker. The
    value is therefore a real graph/runtime choice, not a host-only reference
    hint. Already compiled executables retain the value captured in their graph.
    """

    if value <= 0 or value % TILE_SIZE:
        raise ValueError("query chunk must be a positive tile multiple")
    global _ACTIVE_QUERY_CHUNK_TOKENS
    _ACTIVE_QUERY_CHUNK_TOKENS = int(value)


def _ensure_tt_sdpa_safe_autograd() -> None:
    """Teach compiled graphs the same safe backward as the Python owner.

    Eager autograd preserves ``StreamedCausalAttentionFunction`` as the owner.
    TT's ``torch.compile`` path traces through that Python function and sees the
    inner ``tt::scaled_dot_product_attention`` custom op directly. The frontend
    op intentionally ships without an autograd formula, so register our exact
    blockwise contract before the compiled graph is captured.
    """

    global _TT_SDPA_AUTOGRAD_REGISTERED
    if _TT_SDPA_AUTOGRAD_REGISTERED:
        return
    import tt_torch.custom_ops  # noqa: F401

    def setup_context(ctx, inputs, output):
        del output
        query, key, value, attention_mask, is_causal, scale, _window = inputs
        if attention_mask is not None or not is_causal:
            raise ValueError(
                "native geometry safe SDPA autograd requires internal causal masking"
            )
        ctx.save_for_backward(query, key, value)
        ctx.scale = (
            1.0 / math.sqrt(query.shape[-1])
            if scale is None else float(scale)
        )

    def backward(ctx, output_gradient):
        query, key, value = ctx.saved_tensors
        dq, dk, dv = streamed_causal_attention_backward_reference(
            query,
            key,
            value,
            output_gradient,
            scale=ctx.scale,
            query_chunk_tokens=_ACTIVE_QUERY_CHUNK_TOKENS,
        )
        return dq, dk, dv, None, None, None, None

    torch.library.register_autograd(
        "tt::scaled_dot_product_attention",
        backward,
        setup_context=setup_context,
    )
    _TT_SDPA_AUTOGRAD_REGISTERED = True


def _validate(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("streamed attention requires [B,H,S,D] tensors")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("query/key batch, sequence and head dimension must match")
    if query.shape[1] % key.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    if query.shape[2] % TILE_SIZE or query.shape[3] % TILE_SIZE:
        raise ValueError("sequence and head dimension must be tile aligned")
    if query.dtype != torch.bfloat16 or key.dtype != query.dtype:
        raise ValueError("exact streamed contract requires BF16 Q/K/V")


def _expand_gqa(tensor: torch.Tensor, query_heads: int) -> torch.Tensor:
    groups = query_heads // tensor.shape[1]
    return tensor if groups == 1 else tensor.repeat_interleave(groups, dim=1)


def streamed_causal_attention_backward_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_gradient: torch.Tensor,
    *,
    scale: float,
    query_chunk_tokens: int = DEFAULT_QUERY_CHUNK_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute exact blockwise dQ/dK/dV without a full SxS tensor.

    The loop order, FP32 conversion points, GQA expansion/reduction order and
    final BF16 pack are part of the canonical streamed-attention numerical contract.
    """

    _validate(query, key, value)
    if output_gradient.shape != query.shape:
        raise ValueError("output gradient must have query shape")
    if query_chunk_tokens <= 0 or query_chunk_tokens % TILE_SIZE:
        raise ValueError("query chunk must be a positive tile multiple")
    batch, query_heads, sequence_length, head_dim = query.shape
    kv_heads = key.shape[1]
    group = query_heads // kv_heads

    query_fp32 = query.float()
    key_fp32 = _expand_gqa(key, query_heads).float()
    value_fp32 = _expand_gqa(value, query_heads).float()
    gradient_fp32 = output_gradient.float()
    key_transposed = key_fp32.transpose(-2, -1)
    value_transposed = value_fp32.transpose(-2, -1)

    dq_parts: list[torch.Tensor] = []
    dk_expanded = torch.zeros_like(key_fp32)
    dv_expanded = torch.zeros_like(value_fp32)
    key_positions = torch.arange(sequence_length, device=query.device).view(
        1, 1, 1, sequence_length
    )
    for start in range(0, sequence_length, query_chunk_tokens):
        end = min(sequence_length, start + query_chunk_tokens)
        q_block = query_fp32[:, :, start:end, :]
        do_block = gradient_fp32[:, :, start:end, :]
        scores = torch.matmul(q_block, key_transposed) * float(scale)
        query_positions = torch.arange(
            start, end, device=query.device
        ).view(1, 1, end - start, 1)
        scores = scores.masked_fill(key_positions > query_positions, -torch.inf)
        probability = torch.softmax(scores, dim=-1)

        dp = torch.matmul(do_block, value_transposed)
        row_dot = (dp * probability).sum(dim=-1, keepdim=True)
        ds = probability * (dp - row_dot)
        dq_parts.append(torch.matmul(ds, key_fp32) * float(scale))
        dk_expanded = dk_expanded + (
            torch.matmul(ds.transpose(-2, -1), q_block) * float(scale)
        )
        dv_expanded = dv_expanded + torch.matmul(
            probability.transpose(-2, -1), do_block
        )

    dq = torch.cat(dq_parts, dim=2).to(query.dtype)
    if group == 1:
        dk, dv = dk_expanded, dv_expanded
    else:
        dk = dk_expanded.reshape(
            batch, kv_heads, group, sequence_length, head_dim
        ).sum(dim=2)
        dv = dv_expanded.reshape(
            batch, kv_heads, group, sequence_length, head_dim
        ).sum(dim=2)
    return dq, dk.to(key.dtype), dv.to(value.dtype)


@torch.library.custom_op(
    "tt_finetune::scaled_dot_product_attention_backward_native",
    mutates_args=[],
    device_types=["cpu", "xla"],
)
def native_causal_attention_backward(
    output_gradient: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Emit the experimental native TT-Train Flash-attention backward op."""

    _validate(query, key, value)
    if output_gradient.shape != query.shape:
        raise ValueError("output gradient must have query shape")
    if query.device.type == "xla":
        from torch_xla.experimental import stablehlo_custom_call

        outputs = stablehlo_custom_call.stablehlo_custom_call(
            [output_gradient, query, key, value],
            "tt.scaled_dot_product_attention_backward",
            [list(query.shape), list(key.shape), list(value.shape)],
            [query.dtype, key.dtype, value.dtype],
            frontend_attributes={"is_causal": "True"},
        )
        return tuple(outputs)

    return streamed_causal_attention_backward_reference(
        query,
        key,
        value,
        output_gradient,
        scale=1.0 / math.sqrt(query.shape[-1]),
    )


@native_causal_attention_backward.register_fake
def _native_causal_attention_backward_fake(
    output_gradient: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)


class StreamedCausalAttentionFunction(torch.autograd.Function):
    """Native streamed forward with recompute-only blockwise backward."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        query_chunk_tokens: int,
        backward_backend: str,
    ) -> torch.Tensor:
        _validate(query, key, value)
        ctx.save_for_backward(query, key, value)
        ctx.scale = float(scale)
        ctx.query_chunk_tokens = int(query_chunk_tokens)
        if backward_backend not in _BACKWARD_BACKENDS:
            raise ValueError(f"unsupported streamed backward backend: {backward_backend}")
        standard_scale = 1.0 / math.sqrt(query.shape[-1])
        if (
            backward_backend == NATIVE_EXPERIMENTAL_BACKWARD
            and float(scale) != standard_scale
        ):
            raise ValueError(
                "native attention backward currently requires 1/sqrt(head_dim) scale"
            )
        ctx.backward_backend = backward_backend
        if query.device.type == "xla":
            # Importing registers torch.ops.tt without forcing callers to know
            # about the frontend package.
            _ensure_tt_sdpa_safe_autograd()

            return torch.ops.tt.scaled_dot_product_attention(
                query, key, value, None, True, float(scale)
            )
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=float(scale),
            enable_gqa=query.shape[1] != key.shape[1],
        )

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        query, key, value = ctx.saved_tensors
        if ctx.backward_backend == NATIVE_EXPERIMENTAL_BACKWARD:
            if query.device.type == "xla":
                dq, dk, dv = native_causal_attention_backward(
                    output_gradient, query, key, value
                )
            else:
                # Preserve the selected blockwise numerical contract on the
                # safe CPU fallback.  The XLA custom call has a fixed native
                # schedule and intentionally does not expose this host-only
                # reference tuning parameter in its ABI.
                dq, dk, dv = streamed_causal_attention_backward_reference(
                    query,
                    key,
                    value,
                    output_gradient,
                    scale=ctx.scale,
                    query_chunk_tokens=ctx.query_chunk_tokens,
                )
        else:
            dq, dk, dv = streamed_causal_attention_backward_reference(
                query,
                key,
                value,
                output_gradient,
                scale=ctx.scale,
                query_chunk_tokens=ctx.query_chunk_tokens,
            )
        return dq, dk, dv, None, None, None


def streamed_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
    query_chunk_tokens: int = DEFAULT_QUERY_CHUNK_TOKENS,
    backward_backend: str = SAFE_EXACT_BACKWARD,
) -> torch.Tensor:
    _validate(query, key, value)
    resolved_scale = (
        1.0 / math.sqrt(query.shape[-1]) if scale is None else float(scale)
    )
    return StreamedCausalAttentionFunction.apply(
        query, key, value, resolved_scale, query_chunk_tokens, backward_backend
    )


__all__ = [
    "DEFAULT_QUERY_CHUNK_TOKENS",
    "NATIVE_EXPERIMENTAL_BACKWARD",
    "SAFE_EXACT_BACKWARD",
    "StreamedCausalAttentionFunction",
    "configure_streamed_query_chunk_tokens",
    "native_causal_attention_backward",
    "streamed_causal_attention",
    "streamed_causal_attention_backward_reference",
]
