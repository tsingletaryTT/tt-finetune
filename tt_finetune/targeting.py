"""Canonical, model-name-independent adapter target selection.

The public API accepts semantic roles (``q``, ``v``, ``mlp``...) while the
architecture mapper owns the translation to concrete Hugging Face module
paths.  No model or repository name participates in this decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


ATTENTION_ROLES = frozenset({"q", "k", "v", "kv", "qk", "qkv", "o"})
MLP_ROLES = frozenset(
    {"gate", "up", "gate_up", "down", "mlp_up", "mlp_down"}
)
CANONICAL_ROLES = ATTENTION_ROLES | MLP_ROLES

_GROUPS = {
    "attention": ATTENTION_ROLES,
    "attention_only": ATTENTION_ROLES,
    "mlp": MLP_ROLES,
    "mlp_only": MLP_ROLES,
    "qv": frozenset({"q", "v"}),
    "q_v": frozenset({"q", "v"}),
    # Published IA3 applies output scaling to K/V and input scaling to the
    # feed-forward down projection.  Both gated and ordinary MLP dialects are
    # represented; a concrete architecture exposes only the applicable one.
    "ia3": frozenset({"k", "v", "down", "mlp_down"}),
    "ia3_feedforward": frozenset({"down", "mlp_down"}),
}


@dataclass(frozen=True)
class CanonicalTargetSelection:
    paths: tuple[str, ...]
    roles: tuple[str, ...]
    policy: str


def _normalize(requested: str | Sequence[str]) -> tuple[str, ...]:
    values = (requested,) if isinstance(requested, str) else tuple(requested)
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("target_modules must contain one or more non-empty strings")
    return tuple(value.lower().replace("-", "_") for value in values)


def resolve_canonical_role_targets(
    requested: str | Sequence[str],
    *,
    weight_map: Any,
    eligible_modules: Mapping[str, Any],
) -> CanonicalTargetSelection | None:
    """Resolve semantic target policies to exact qualified module paths.

    ``None`` means the request is an explicit module-name request and should be
    handled by the caller's compatibility path.  Semantic requests never fall
    back to leaf-name guessing.
    """

    values = _normalize(requested)
    if len(values) == 1 and values[0] in {
        "auto", "auto_qv", "auto_attention", "auto_all_linear", "all_linear"
    }:
        # Legacy automatic policies retain their existing resolver because it
        # deliberately supports fused-QKV fallback and PEFT all-linear syntax.
        return None

    expanded: set[str] = set()
    semantic = True
    for value in values:
        if value in _GROUPS:
            expanded.update(_GROUPS[value])
        elif value in CANONICAL_ROLES:
            expanded.add(value)
        else:
            semantic = False
            break
    if not semantic:
        return None

    targets = tuple(
        item
        for item in weight_map.lora_targets
        if item.path in eligible_modules and item.role in expanded
    )
    available_roles = {
        item.role
        for item in weight_map.lora_targets
        if item.path in eligible_modules
    }

    # A combined QKV projection cannot honestly implement Q/V-only LoRA:
    # adapting the module also changes K.  The legacy ``auto`` policy may use
    # fused QKV for broad compatibility; the explicit semantic contract may not.
    strict_component_request = not any(
        value in {"attention", "attention_only"} for value in values
    )
    requested_qkv_components = (
        expanded & {"q", "k", "v"} if strict_component_request else set()
    )
    missing = requested_qkv_components - available_roles
    fused_roles = sorted(available_roles & {"qkv", "kv", "qk"})
    if missing and fused_roles:
        fused_description = (
            "fused into role 'qkv'"
            if fused_roles == ["qkv"]
            else f"overlap fused roles {fused_roles}"
        )
        raise ValueError(
            "The requested separate canonical roles "
            f"{sorted(requested_qkv_components)} {fused_description} in this "
            "architecture. Select the fused canonical "
            "role explicitly; a component-only adapter would otherwise alter "
            "another attention component."
        )

    matched_roles = {item.role for item in targets}
    # Group policies intentionally tolerate architecture-equivalent variants
    # (gate/up vs gate_up, GELU mlp_up/down). Explicit roles do not.
    explicit_roles = {value for value in values if value in CANONICAL_ROLES}
    missing_explicit = explicit_roles - matched_roles
    if missing_explicit:
        raise ValueError(
            f"Requested canonical adapter roles do not exist: {sorted(missing_explicit)}. "
            f"Available roles: {sorted(available_roles)}"
        )
    if not targets:
        raise ValueError(
            f"Canonical adapter policy {values!r} matched no projections. "
            f"Available roles: {sorted(available_roles)}"
        )

    policy = values[0] if len(values) == 1 else "canonical_roles"
    return CanonicalTargetSelection(
        paths=tuple(sorted(item.path for item in targets)),
        roles=tuple(sorted(matched_roles)),
        policy=policy,
    )
