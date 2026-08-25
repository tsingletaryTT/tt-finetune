"""Capability validation for canonical architecture decoder primitives."""

from __future__ import annotations

from dataclasses import dataclass

from .descriptor import CanonicalArchitectureDescriptor


@dataclass(frozen=True)
class UnsupportedCapability:
    primitive: str
    semantic: str
    reason: str


@dataclass(frozen=True)
class ArchitectureCapabilityReport:
    supported: bool
    required_primitives: tuple[str, ...]
    unsupported: tuple[UnsupportedCapability, ...]
    safe_fallback_primitives: tuple[str, ...]

    def require_supported(self) -> None:
        if self.supported:
            return
        details = "; ".join(
            f"{item.primitive}={item.semantic}: {item.reason}"
            for item in self.unsupported
        )
        raise ValueError(f"unsupported architecture architecture capabilities: {details}")

    def to_dict(self) -> dict:
        return {
            "supported": self.supported,
            "required_primitives": list(self.required_primitives),
            "safe_fallback_primitives": list(self.safe_fallback_primitives),
            "unsupported": [vars(item) for item in self.unsupported],
        }


SUPPORTED_ACTIVATIONS = frozenset(
    {"silu", "gelu", "gelu_new", "relu", "relu2", "tanh"}
)


def architecture_capabilities(
    descriptor: CanonicalArchitectureDescriptor,
    *,
    sequence_length: int,
) -> ArchitectureCapabilityReport:
    """Validate semantic support independently of optimized schedules."""

    unsupported: list[UnsupportedCapability] = []
    required = [
        (
            "embedding"
            if descriptor.embedding_scale == 1.0
            else f"embedding:scale={descriptor.embedding_scale:g}"
        ),
        descriptor.norm_kind,
        f"position:{descriptor.rope.kind}",
        f"qkv:{descriptor.qkv_projection_layout}",
        f"attention:{descriptor.attention_kind}",
        "attention_output_projection",
        "residual",
        f"mlp:{descriptor.mlp_kind}:{descriptor.gate_up_layout}",
        "final_norm" if descriptor.final_norm else "final_norm:identity",
        f"lm_head:{descriptor.lm_head_layout}",
        "causal_loss",
        "lora_projection",
    ]
    if descriptor.attention_qk_norm is not None:
        required.append(f"attention_qk_norm:{descriptor.attention_qk_norm}")
    if descriptor.local_attention_layers:
        required.append(
            f"local_attention:window={descriptor.attention_window}:"
            f"layers={','.join(str(index) for index in descriptor.local_attention_layers)}"
        )
    if descriptor.moe is not None:
        moe = descriptor.moe
        required.extend((
            f"moe_router:{moe.router_type}:{moe.router_logits_dtype}",
            f"moe_topk:{moe.top_k}",
            f"moe_expert:{moe.expert_mlp_kind}",
            "moe_device_dispatch",
            "moe_weighted_combine",
            "expert_lora_projection",
        ))
        supported_router_types = {"linear_topk"}
        if moe.router_type not in supported_router_types:
            unsupported.append(UnsupportedCapability(
                "moe_router", moe.router_type,
                "selected-only device execution requires a linear top-k router",
            ))
        if moe.router_logits_dtype not in {"float32", "fp32"}:
            unsupported.append(UnsupportedCapability(
                "moe_router_dtype", moe.router_logits_dtype,
                "deterministic routing contract requires FP32 logits",
            ))
        supported_normalizations = {
            "softmax_selected",
            "softmax_all_selected",
            "softmax_all_renormalized_selected",
            "sigmoid_selected",
            "sigmoid_selected_unnormalized",
            "unnormalized_selected",
        }
        if moe.router_normalization not in supported_normalizations:
            unsupported.append(UnsupportedCapability(
                "moe_router_normalization", moe.router_normalization,
                "router normalization has no selected-only device contract",
            ))
        if moe.expert_mlp_kind not in {"swiglu", "ordinary"}:
            unsupported.append(UnsupportedCapability(
                "moe_expert_mlp", moe.expert_mlp_kind,
                "selected-only owner supports SwiGLU and ordinary MLP experts",
            ))
        if (
            moe.shared_expert_count
            and moe.shared_expert_mlp_kind not in {"swiglu", "ordinary"}
        ):
            unsupported.append(UnsupportedCapability(
                "moe_shared_experts", str(moe.shared_expert_mlp_kind),
                "shared expert kind has no canonical selected-only execution",
            ))
        if moe.expert_group_policy not in {None, "group_limited_topk"}:
            unsupported.append(UnsupportedCapability(
                "moe_expert_groups", str(moe.expert_group_policy),
                "expert grouping policy is not a max-score group-limited top-k",
            ))
        if moe.auxiliary_loss_kind not in {"none", "load_balancing"}:
            unsupported.append(UnsupportedCapability(
                "moe_auxiliary_loss", moe.auxiliary_loss_kind,
                "router auxiliary loss has no device training-loss contract",
            ))
        if moe.expert_bias:
            unsupported.append(UnsupportedCapability(
                "moe_expert_bias", "enabled",
                "canonical selected expert projection bias is not implemented",
            ))
        if moe.overflow_policy == "overflow_expert":
            unsupported.append(UnsupportedCapability(
                "moe_capacity_overflow", moe.overflow_policy,
                "a dedicated overflow expert is not mapped into the routed set",
            ))
        if moe.router_jitter_noise != 0.0:
            unsupported.append(UnsupportedCapability(
                "moe_router_jitter", str(moe.router_jitter_noise),
                "device RNG replay for router jitter is not yet qualified",
            ))
    if not descriptor.decoder_only or not descriptor.causal:
        unsupported.append(UnsupportedCapability(
            "model", "non_causal_decoder", "only decoder-only causal LMs are in scope"
        ))
    if descriptor.rope.kind not in {
        "standard", "scaled", "learned_absolute", "sinusoidal_absolute",
        "alibi", "none",
    }:
        unsupported.append(UnsupportedCapability(
            "position", descriptor.rope.kind,
            "position scheme has no architecture-safe execution contract",
        ))
    if descriptor.mlp_activation not in SUPPORTED_ACTIVATIONS:
        unsupported.append(UnsupportedCapability(
            "mlp_activation", descriptor.mlp_activation,
            "activation is not in the qualified safe primitive set",
        ))
    if descriptor.lm_head_layout not in {"linear", "conv1d"}:
        unsupported.append(UnsupportedCapability(
            "lm_head", descriptor.lm_head_layout,
            "only dense linear vocabulary heads are supported",
        ))
    safe_items: list[str] = []
    if descriptor.rope.kind == "alibi":
        safe_items.append("alibi_attention_bias")
    if (
        descriptor.attention_window is not None
        and sequence_length > descriptor.attention_window
    ):
        # The semantic is representable but the current long-sequence native
        # owner implements global causal attention only. Keep the architecture
        # admitted through the safe HF attention contract.
        safe_items.append("sliding_local_attention")
    return ArchitectureCapabilityReport(
        supported=not unsupported,
        required_primitives=tuple(required),
        unsupported=tuple(unsupported),
        safe_fallback_primitives=tuple(safe_items),
    )


__all__ = [
    "ArchitectureCapabilityReport",
    "UnsupportedCapability",
    "architecture_capabilities",
]
