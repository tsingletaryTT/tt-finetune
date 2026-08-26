"""Transformers attention-interface bridge for nativeB streamed training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tt_finetune.architecture import map_model
from tt_finetune.runtime.geometry_planner import GeometryPlan

from tt_finetune.kernels.streamed_causal_attention import (
    DEFAULT_QUERY_CHUNK_TOKENS,
    NATIVE_EXPERIMENTAL_BACKWARD,
    SAFE_EXACT_BACKWARD,
    configure_streamed_query_chunk_tokens,
    streamed_causal_attention,
)


ATTENTION_BACKEND = "tt_streamed"
NATIVE_EXPERIMENTAL_ATTENTION_BACKEND = (
    "tt_streamed_native_backward_experimental"
)


def streamed_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    return _streamed_attention_forward(
        module, query, key, value, attention_mask, dropout, scaling,
        is_causal, backward_backend=SAFE_EXACT_BACKWARD, **kwargs,
    )


def streamed_native_backward_experimental_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    return _streamed_attention_forward(
        module, query, key, value, attention_mask, dropout, scaling,
        is_causal, backward_backend=NATIVE_EXPERIMENTAL_BACKWARD, **kwargs,
    )


def _streamed_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float | None,
    is_causal: bool | None,
    *,
    backward_backend: str,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if float(dropout) != 0.0:
        raise ValueError("nativeB streamed attention requires zero dropout")
    if attention_mask is not None:
        raise ValueError(
            "nativeB causal masking is internal; dense masks are forbidden"
        )
    if is_causal is False or not getattr(module, "is_causal", True):
        raise ValueError("nativeB streamed attention requires causal attention")
    if kwargs.get("output_attentions", False):
        raise ValueError("streamed attention does not publish attention weights")
    output = streamed_causal_attention(
        query,
        key,
        value,
        scale=scaling,
        query_chunk_tokens=int(getattr(
            module, "_tt_query_chunk_tokens", DEFAULT_QUERY_CHUNK_TOKENS
        )),
        backward_backend=backward_backend,
    )
    return output.transpose(1, 2).contiguous(), None


@dataclass(frozen=True)
class StreamedAttentionInstallReport:
    backend: str
    layer_count: int
    full_sxs_mask_allocated: bool
    safe_default: bool
    query_chunk_tokens: int
    backward_backend: str


def install_streamed_attention(
    model: nn.Module,
    *,
    native_backward_experimental: bool = False,
    query_chunk_tokens: int = DEFAULT_QUERY_CHUNK_TOKENS,
) -> StreamedAttentionInstallReport:
    """Register and select the streamed interface on one decoder model."""

    if query_chunk_tokens <= 0 or query_chunk_tokens % 32:
        raise ValueError("streamed attention query chunk must be tile aligned")
    configure_streamed_query_chunk_tokens(query_chunk_tokens)
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError as exc:  # pragma: no cover - integration environment
        raise RuntimeError("Transformers attention registry is unavailable") from exc
    selected_backend = (
        NATIVE_EXPERIMENTAL_ATTENTION_BACKEND
        if native_backward_experimental else ATTENTION_BACKEND
    )
    selected_function = (
        streamed_native_backward_experimental_forward
        if native_backward_experimental else streamed_attention_forward
    )
    ALL_ATTENTION_FUNCTIONS.register(selected_backend, selected_function)
    canonical = map_model(model)
    layers = tuple(model.get_submodule(layer.path) for layer in canonical.layers)
    if not layers:
        raise ValueError("model has no decoder layers")
    for layer in layers:
        attention = layer.self_attn
        if int(attention.head_dim) % 32:
            raise ValueError("streamed attention requires tile-aligned head dim")
        query_heads = int(attention.q_proj.out_features) // int(attention.head_dim)
        kv_heads = int(attention.k_proj.out_features) // int(attention.head_dim)
        if query_heads % kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
    model.config._attn_implementation = selected_backend
    decoder_path = canonical.decoder_path.rsplit(".layers", 1)[0]
    decoder = model.get_submodule(decoder_path)
    if hasattr(decoder, "config"):
        decoder.config._attn_implementation = selected_backend
    for layer in layers:
        layer.self_attn.config._attn_implementation = selected_backend
        layer.self_attn._tt_query_chunk_tokens = int(query_chunk_tokens)
    return StreamedAttentionInstallReport(
        backend=selected_backend,
        layer_count=len(layers),
        full_sxs_mask_allocated=False,
        safe_default=False,
        query_chunk_tokens=int(query_chunk_tokens),
        backward_backend=(
            NATIVE_EXPERIMENTAL_BACKWARD
            if native_backward_experimental else SAFE_EXACT_BACKWARD
        ),
    )


def install_attention_for_plan(
    model: nn.Module,
    plan: GeometryPlan,
    *,
    qualification_override: str | None = None,
) -> StreamedAttentionInstallReport | None:
    """Install the attention schedule selected by the native planner.

    Normal callers provide no override.  Qualification tools may explicitly
    request the safe or native numerical contract so both sides can be
    captured independently without turning those choices into user-facing
    trainer knobs.
    """

    if plan.attention_strategy == "dense_causal":
        if qualification_override is not None:
            raise ValueError("cannot override a dense attention plan")
        return None
    selected = qualification_override or plan.attention_backward_backend
    if selected not in ("safe_streamed_exact", "native_streamed_exact"):
        raise ValueError(f"unsupported native attention backend: {selected}")
    if (
        selected == "native_streamed_exact"
        and plan.attention_backward_backend != "native_streamed_exact"
    ):
        raise ValueError(
            "native attention override is outside the planner-admitted geometry"
        )
    return install_streamed_attention(
        model,
        native_backward_experimental=(selected == "native_streamed_exact"),
    )


__all__ = [
    "ATTENTION_BACKEND",
    "NATIVE_EXPERIMENTAL_ATTENTION_BACKEND",
    "StreamedAttentionInstallReport",
    "install_attention_for_plan",
    "install_streamed_attention",
    "streamed_attention_forward",
    "streamed_native_backward_experimental_forward",
]
