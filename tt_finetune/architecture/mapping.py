"""Map heterogeneous Hugging Face module trees into canonical decoder roles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable

import torch
from torch import nn

from .descriptor import (
    CanonicalArchitectureDescriptor,
    MoEDescriptor,
    ProjectionBiases,
    RopeDescriptor,
    describe_config,
    with_observed_topology,
)


ATTENTION_NAMES = ("self_attn", "attention", "attn", "self_attention")
MLP_NAMES = ("mlp", "feed_forward", "ffn", "feedforward")
MOE_NAMES = (
    "block_sparse_moe", "moe", "sparse_moe", "mixture_of_experts",
    "feed_forward_moe",
)
ROLE_ALIASES = {
    "q": ("q_proj", "q_attn", "query", "q"),
    "k": ("k_proj", "key", "k"),
    "v": ("v_proj", "value", "v"),
    "qkv": ("query_key_value", "qkv_proj", "wqkv", "Wqkv", "c_attn", "qkv"),
    # GPTBigCode exposes a separate ``q_attn`` plus a Conv1D ``c_attn``
    # containing only K/V.  ``c_attn`` remains a valid fully fused QKV alias
    # for GPT-2 when no separate query projection is present; topology, not a
    # checkpoint name, disambiguates the two cases in ``_map_layer``.
    "kv": ("kv_proj", "key_value", "key_value_proj", "wkv", "c_attn"),
    "qk": ("qk_proj", "query_key", "query_key_proj", "wqk"),
    "o": ("o_proj", "out_proj", "dense", "c_proj"),
    "gate": ("gate_proj", "w1", "gate"),
    "up": ("up_proj", "w3"),
    "gate_up": ("gate_up_proj", "w1w3", "gate_up"),
    "down": ("down_proj", "w2", "out_proj"),
    "mlp_up": (
        "dense_h_to_4h", "c_fc", "fc1", "fc_in", "up_proj",
    ),
    "mlp_down": (
        "dense_4h_to_h", "c_proj", "fc2", "fc_out", "down_proj",
    ),
}


@dataclass(frozen=True)
class CanonicalLinearRole:
    role: str
    path: str
    input_features: int
    output_features: int
    bias: bool
    storage_layout: str
    module_type: str
    layer_index: int
    parameter_alias_group: str | None = None


@dataclass(frozen=True)
class CanonicalExpertLinearRole:
    role: str
    path: str
    expert_count: int
    input_features: int
    output_features: int
    bias: bool
    storage_layout: str
    module_type: str
    layer_index: int
    # Fused tensors expose one path; separate ModuleList experts expose one
    # path per expert while retaining the same logical role.
    paths: tuple[str, ...] = ()
    parameter_alias_groups: tuple[str | None, ...] = ()


@dataclass(frozen=True)
class CanonicalMoELayerMap:
    path: str
    router_path: str
    router_input_features: int
    expert_count: int
    top_k: int
    expert_mlp_kind: str
    projections: tuple[CanonicalExpertLinearRole, ...]
    expert_storage_layout: str = "fused"
    shared_expert_path: str | None = None
    shared_expert_gate_path: str | None = None
    shared_expert_mlp_kind: str | None = None
    shared_expert_projections: tuple[CanonicalExpertLinearRole, ...] = ()
    expert_group_count: int | None = None
    expert_group_top_k: int | None = None

    def projection(self, role: str) -> CanonicalExpertLinearRole:
        matches = [item for item in self.projections if item.role == role]
        if len(matches) != 1:
            raise KeyError(f"MoE block {self.path} has no unique {role} projection")
        return matches[0]


@dataclass(frozen=True)
class CanonicalLayerMap:
    index: int
    path: str
    attention_path: str
    mlp_path: str | None
    input_norm_path: str
    post_attention_norm_path: str | None
    query_norm_path: str | None
    key_norm_path: str | None
    projections: tuple[CanonicalLinearRole, ...]
    moe: CanonicalMoELayerMap | None = None

    def projection(self, role: str) -> CanonicalLinearRole:
        matches = [item for item in self.projections if item.role == role]
        if len(matches) != 1:
            raise KeyError(f"layer {self.index} has no unique {role} projection")
        return matches[0]


@dataclass(frozen=True)
class CanonicalModelMap:
    decoder_path: str
    embedding_path: str
    final_norm_path: str | None
    lm_head_path: str
    layers: tuple[CanonicalLayerMap, ...]
    global_projections: tuple[CanonicalLinearRole, ...]
    tied_embeddings: bool
    shared_parameter_views: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def lora_targets(self) -> tuple[CanonicalLinearRole, ...]:
        return (
            tuple(item for layer in self.layers for item in layer.projections)
            + self.global_projections
        )

    @property
    def expert_lora_targets(self) -> tuple[CanonicalExpertLinearRole, ...]:
        return tuple(
            projection
            for layer in self.layers
            if layer.moe is not None
            for projection in layer.moe.projections
        )

    @property
    def shared_expert_lora_targets(
        self,
    ) -> tuple[CanonicalExpertLinearRole, ...]:
        return tuple(
            projection
            for layer in self.layers
            if layer.moe is not None
            for projection in layer.moe.shared_expert_projections
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoder_path": self.decoder_path,
            "embedding_path": self.embedding_path,
            "final_norm_path": self.final_norm_path,
            "lm_head_path": self.lm_head_path,
            "layer_count": len(self.layers),
            "tied_embeddings": self.tied_embeddings,
            "shared_parameter_views": {
                group: list(paths)
                for group, paths in self.shared_parameter_views
            },
            "global_projections": [
                vars(item) for item in self.global_projections
            ],
            "layers": [
                {
                    "index": layer.index,
                    "path": layer.path,
                    "attention_path": layer.attention_path,
                    "mlp_path": layer.mlp_path,
                    "input_norm_path": layer.input_norm_path,
                    "post_attention_norm_path": layer.post_attention_norm_path,
                    "query_norm_path": layer.query_norm_path,
                    "key_norm_path": layer.key_norm_path,
                    "projections": [vars(item) for item in layer.projections],
                    "moe": (
                        {
                            "path": layer.moe.path,
                            "router_path": layer.moe.router_path,
                            "router_input_features": layer.moe.router_input_features,
                            "expert_count": layer.moe.expert_count,
                            "top_k": layer.moe.top_k,
                            "expert_mlp_kind": layer.moe.expert_mlp_kind,
                            "expert_storage_layout": (
                                layer.moe.expert_storage_layout
                            ),
                            "shared_expert_path": layer.moe.shared_expert_path,
                            "shared_expert_gate_path": (
                                layer.moe.shared_expert_gate_path
                            ),
                            "shared_expert_mlp_kind": (
                                layer.moe.shared_expert_mlp_kind
                            ),
                            "expert_group_count": layer.moe.expert_group_count,
                            "expert_group_top_k": layer.moe.expert_group_top_k,
                            "projections": [
                                vars(item) for item in layer.moe.projections
                            ],
                            "shared_expert_projections": [
                                vars(item)
                                for item in layer.moe.shared_expert_projections
                            ],
                        }
                        if layer.moe is not None else None
                    ),
                }
                for layer in self.layers
            ],
        }


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _named_identity(model: nn.Module, wanted: nn.Module | None) -> str | None:
    if wanted is None:
        return None
    for name, module in model.named_modules():
        if module is wanted:
            return name
    return None


def _is_linear_like(module: nn.Module) -> bool:
    base = getattr(module, "base_layer", module)
    weight = getattr(base, "weight", None)
    return isinstance(weight, torch.Tensor) and weight.ndim == 2


def _linear_geometry(module: nn.Module) -> tuple[int, int, str, bool, str]:
    base = getattr(module, "base_layer", module)
    weight = getattr(base, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError(f"{type(module).__name__} is not a linear projection")
    module_type = type(base).__name__
    if isinstance(base, nn.Linear) or (
        hasattr(base, "in_features") and hasattr(base, "out_features")
    ):
        input_features = int(base.in_features)
        output_features = int(base.out_features)
        layout = "out_in"
    elif module_type == "Conv1D" or (
        hasattr(base, "nf") and not hasattr(base, "out_features")
    ):
        input_features = int(weight.shape[0])
        output_features = int(weight.shape[1])
        layout = "in_out"
    else:
        raise TypeError(
            f"cannot infer linear storage convention for {module_type}"
        )
    return (
        input_features,
        output_features,
        layout,
        getattr(base, "bias", None) is not None,
        module_type,
    )


def _child_by_alias(module: nn.Module, aliases: Iterable[str]) -> tuple[str, nn.Module] | None:
    children = dict(module.named_children())
    for alias in aliases:
        child = children.get(alias)
        if child is not None:
            return alias, child
    lowered = {name.lower(): (name, child) for name, child in children.items()}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def _descendant_by_alias(
    module: nn.Module, aliases: Iterable[str], *, max_depth: int = 2,
) -> tuple[str, nn.Module] | None:
    alias_set = {item.lower() for item in aliases}
    for name, child in module.named_modules():
        if not name or name.count(".") >= max_depth:
            continue
        if name.rsplit(".", 1)[-1].lower() in alias_set:
            return name, child
    return None


def _decoder_layers(model: nn.Module, expected_layers: int) -> tuple[str, nn.ModuleList]:
    candidates: list[tuple[int, str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or not module:
            continue
        first = module[0]
        attention = _child_by_alias(first, ATTENTION_NAMES)
        mlp = _child_by_alias(first, MLP_NAMES)
        moe = _child_by_alias(first, MOE_NAMES)
        score = 0
        if len(module) == expected_layers:
            score += 100
        if attention is not None:
            score += 20
        if mlp is not None:
            score += 20
        if moe is not None:
            score += 20
        if name.rsplit(".", 1)[-1] in {"layers", "h", "blocks", "block"}:
            score += 10
        if sum(1 for child in first.modules() if _is_linear_like(child)) >= 4:
            score += 5
        candidates.append((score, name, module))
    if not candidates:
        raise ValueError("could not locate a decoder-layer ModuleList")
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score = candidates[0][0]
    best = [item for item in candidates if item[0] == best_score]
    if len(best) != 1:
        names = [name for _score, name, _module in best]
        raise ValueError(f"ambiguous decoder-layer ModuleLists: {names}")
    _score, path, layers = best[0]
    if len(layers) != expected_layers:
        raise ValueError(
            f"config declares {expected_layers} layers but module tree has {len(layers)}"
        )
    return path, layers


def _norm_children(layer: nn.Module) -> list[tuple[str, nn.Module]]:
    return [
        (name, child)
        for name, child in layer.named_children()
        if "norm" in type(child).__name__.lower() or "norm" in name.lower()
    ]


def _projection(
    parent: nn.Module,
    parent_path: str,
    role: str,
    layer_index: int,
) -> CanonicalLinearRole | None:
    found = _child_by_alias(parent, ROLE_ALIASES[role])
    if found is None:
        # Some HF dialects wrap the actual attention projection container one
        # level below the block's public ``attn`` module (GPT-Neo is the
        # motivating case).  Alias discovery remains semantic-name based and
        # confined to this attention/MLP subtree; it never chooses a backend.
        found = _descendant_by_alias(
            parent, ROLE_ALIASES[role], max_depth=3
        )
    if found is None:
        return None
    name, module = found
    if not _is_linear_like(module):
        return None
    input_features, output_features, layout, bias, module_type = _linear_geometry(module)
    return CanonicalLinearRole(
        role=role,
        path=_join(parent_path, name),
        input_features=input_features,
        output_features=output_features,
        bias=bias,
        storage_layout=layout,
        module_type=module_type,
        layer_index=layer_index,
    )


def _three_dimensional_weight_modules(
    module: nn.Module,
) -> list[tuple[str, nn.Module, torch.Tensor]]:
    output = []
    for name, child in module.named_modules():
        if not name:
            continue
        weight = getattr(child, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 3:
            output.append((name, child, weight))
    return output


_EXPERT_CONTAINER_NAMES = (
    "experts", "local_experts", "routed_experts", "expert_list",
)
_SHARED_EXPERT_NAMES = (
    "shared_expert", "shared_experts", "shared_mlp",
)
_SHARED_EXPERT_GATE_NAMES = (
    "shared_expert_gate", "shared_gate", "shared_expert_router",
)


def _expert_container(
    module: nn.Module,
) -> tuple[str, nn.ModuleList] | None:
    found = _descendant_by_alias(
        module, _EXPERT_CONTAINER_NAMES, max_depth=3
    )
    if found is None or not isinstance(found[1], nn.ModuleList):
        return None
    if not found[1]:
        raise ValueError("MoE expert container is empty")
    return found[0], found[1]


def _looks_like_moe(module: nn.Module) -> bool:
    return (
        len(_three_dimensional_weight_modules(module)) >= 2
        or _expert_container(module) is not None
    )


def _find_moe(layer: nn.Module) -> tuple[str, nn.Module] | None:
    by_alias = _child_by_alias(layer, MOE_NAMES)
    if by_alias is not None:
        return by_alias
    candidates = []
    for name, child in layer.named_children():
        if _looks_like_moe(child):
            candidates.append((name, child))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _find_router(
    moe: nn.Module,
    *,
    hidden_size: int,
    expert_count: int,
) -> tuple[str, nn.Module]:
    aliases = ("router", "gate", "gating", "router_layer")
    candidate = _descendant_by_alias(moe, aliases, max_depth=3)
    linear_candidates = []
    for name, child in moe.named_modules():
        if not name or not _is_linear_like(child):
            continue
        try:
            input_features, output_features, _layout, _bias, _kind = (
                _linear_geometry(child)
            )
        except TypeError:
            continue
        if input_features == hidden_size and output_features == expert_count:
            linear_candidates.append((name, child))
    if candidate is not None:
        candidate_name, candidate_module = candidate
        if _is_linear_like(candidate_module):
            return candidate_name, candidate_module
        nested = _descendant_by_alias(
            candidate_module, ("layer", "linear", "proj"), max_depth=2
        )
        if nested is not None and _is_linear_like(nested[1]):
            return _join(candidate_name, nested[0]), nested[1]
    if len(linear_candidates) != 1:
        raise ValueError("could not locate one canonical MoE router projection")
    return linear_candidates[0]


def _first_int_attribute(module: nn.Module, *names: str) -> int | None:
    for owner in (module, getattr(module, "router", None)):
        if owner is None:
            continue
        for name in names:
            value = getattr(owner, name, None)
            if value is not None:
                return int(value)
    return None


def _fused_expert_geometry(
    weight: torch.Tensor,
    *,
    hidden_size: int,
    is_output: bool,
) -> tuple[int, int, str]:
    """Return logical input/output and physical expert storage order."""

    second, third = map(int, weight.shape[1:])
    if is_output:
        if second == hidden_size:
            return third, second, "experts_out_in"
        if third == hidden_size:
            return second, third, "experts_in_out"
    else:
        if third == hidden_size:
            return third, second, "experts_out_in"
        if second == hidden_size:
            return second, third, "experts_in_out"
    raise ValueError(
        "expert tensor geometry does not contain the residual hidden width "
        f"{hidden_size}: {tuple(weight.shape)}"
    )


def _map_fused_experts(
    weighted: list[tuple[str, nn.Module, torch.Tensor]],
    *,
    path: str,
    layer_index: int,
    hidden_size: int,
) -> tuple[tuple[CanonicalExpertLinearRole, ...], int, str]:
    expert_counts = {int(weight.shape[0]) for _name, _child, weight in weighted}
    if len(expert_counts) != 1:
        raise ValueError(f"MoE block {path} has inconsistent expert counts")
    expert_count = next(iter(expert_counts))
    down_aliases = {"output_linear", "down_proj", "w2", "down", "fc2"}
    gate_up_aliases = {
        "input_linear", "gate_up_proj", "w1w3", "gate_up",
    }
    gate_aliases = {"gate_proj", "w1", "gate"}
    up_aliases = {"up_proj", "w3", "up", "fc1"}
    named_down = [
        item for item in weighted
        if item[0].rsplit(".", 1)[-1].lower() in down_aliases
    ]
    if len(named_down) == 1:
        down = named_down[0]
    else:
        candidates = []
        for item in weighted:
            try:
                _input, output, _layout = _fused_expert_geometry(
                    item[2], hidden_size=hidden_size, is_output=True
                )
            except ValueError:
                continue
            if output == hidden_size:
                candidates.append(item)
        if len(candidates) != 1:
            raise ValueError(
                f"MoE block {path} has no unique fused expert down projection"
            )
        down = candidates[0]
    down_name, down_module, down_weight = down
    down_input, down_output, down_layout = _fused_expert_geometry(
        down_weight, hidden_size=hidden_size, is_output=True
    )
    if down_output != hidden_size:
        raise ValueError("expert down projection does not return hidden width")

    inputs = [item for item in weighted if item is not down]
    projections: list[CanonicalExpertLinearRole] = []

    def append_input(item, role: str) -> int:
        name, module, weight = item
        input_features, output_features, layout = _fused_expert_geometry(
            weight, hidden_size=hidden_size, is_output=False
        )
        projections.append(CanonicalExpertLinearRole(
            role=role,
            path=_join(path, name),
            expert_count=expert_count,
            input_features=input_features,
            output_features=output_features,
            bias=getattr(module, "bias", None) is not None,
            storage_layout=layout,
            module_type=type(module).__name__,
            layer_index=layer_index,
        ))
        return output_features

    named_gate_up = [
        item for item in inputs
        if item[0].rsplit(".", 1)[-1].lower() in gate_up_aliases
    ]
    named_gate = [
        item for item in inputs
        if item[0].rsplit(".", 1)[-1].lower() in gate_aliases
    ]
    named_up = [
        item for item in inputs
        if item[0].rsplit(".", 1)[-1].lower() in up_aliases
    ]
    if len(inputs) == 1:
        width = append_input(inputs[0], "expert_gate_up")
        if width == 2 * down_input:
            expert_kind = "swiglu"
        elif width == down_input:
            # An unfused ordinary expert input is still represented by one
            # tensor; use the semantically accurate ordinary role.
            projections[-1] = CanonicalExpertLinearRole(
                **{
                    **vars(projections[-1]),
                    "role": "expert_up",
                }
            )
            expert_kind = "ordinary"
        else:
            raise ValueError(
                f"MoE block {path} has unsupported fused expert input width"
            )
    elif (
        len(inputs) == 2
        and len(named_gate) == 1
        and len(named_up) == 1
    ):
        gate_width = append_input(named_gate[0], "expert_gate")
        up_width = append_input(named_up[0], "expert_up")
        if gate_width != down_input or up_width != down_input:
            raise ValueError("separate fused expert gate/up widths disagree")
        expert_kind = "swiglu"
    elif len(named_gate_up) == 1 and len(inputs) == 1:
        # Covered above; retained as an explicit diagnostic guard.
        raise AssertionError("unreachable fused gate/up mapping")
    else:
        raise ValueError(
            f"MoE block {path} has ambiguous fused expert input projections"
        )
    projections.append(CanonicalExpertLinearRole(
        role="expert_down",
        path=_join(path, down_name),
        expert_count=expert_count,
        input_features=down_input,
        output_features=down_output,
        bias=getattr(down_module, "bias", None) is not None,
        storage_layout=down_layout,
        module_type=type(down_module).__name__,
        layer_index=layer_index,
    ))
    return tuple(projections), expert_count, expert_kind


def _one_expert_roles(
    expert: nn.Module,
    *,
    path: str,
    layer_index: int,
    hidden_size: int,
) -> tuple[str, tuple[CanonicalLinearRole, ...]]:
    gate = _projection(expert, path, "gate", layer_index)
    up = _projection(expert, path, "up", layer_index)
    gate_up = _projection(expert, path, "gate_up", layer_index)
    down = _projection(expert, path, "down", layer_index)
    if down is None:
        down = _projection(expert, path, "mlp_down", layer_index)
    if gate is not None and up is not None and down is not None:
        if (
            gate.input_features != hidden_size
            or up.input_features != hidden_size
            or gate.output_features != down.input_features
            or up.output_features != down.input_features
            or down.output_features != hidden_size
        ):
            raise ValueError("separate expert gate/up/down geometry disagrees")
        return "swiglu", (gate, up, down)
    if gate_up is not None and down is not None:
        if (
            gate_up.input_features != hidden_size
            or gate_up.output_features != 2 * down.input_features
            or down.output_features != hidden_size
        ):
            raise ValueError("separate expert fused gate/up geometry disagrees")
        return "swiglu", (gate_up, down)
    ordinary_up = _projection(expert, path, "mlp_up", layer_index)
    if ordinary_up is None:
        ordinary_up = up
    if ordinary_up is not None and down is not None:
        if (
            ordinary_up.input_features != hidden_size
            or ordinary_up.output_features != down.input_features
            or down.output_features != hidden_size
        ):
            raise ValueError("separate ordinary expert geometry disagrees")
        return "ordinary", (ordinary_up, down)
    raise ValueError(f"expert {path} has no canonical MLP projections")


def _map_separate_experts(
    experts: nn.ModuleList,
    *,
    path: str,
    layer_index: int,
    hidden_size: int,
) -> tuple[tuple[CanonicalExpertLinearRole, ...], str]:
    mapped = [
        _one_expert_roles(
            expert,
            path=_join(path, str(index)),
            layer_index=layer_index,
            hidden_size=hidden_size,
        )
        for index, expert in enumerate(experts)
    ]
    kinds = {kind for kind, _roles in mapped}
    role_sets = [tuple(role.role for role in roles) for _kind, roles in mapped]
    if len(kinds) != 1 or len(set(role_sets)) != 1:
        raise ValueError("routed experts have heterogeneous MLP semantics")
    expert_count = len(mapped)
    output = []
    for role_index, base_role in enumerate(mapped[0][1]):
        instances = [roles[role_index] for _kind, roles in mapped]
        geometry = {
            (
                item.input_features,
                item.output_features,
                item.storage_layout,
                item.bias,
            )
            for item in instances
        }
        if len(geometry) != 1:
            raise ValueError("routed expert projection geometry is heterogeneous")
        canonical_role = {
            "gate": "expert_gate",
            "up": "expert_up",
            "gate_up": "expert_gate_up",
            "down": "expert_down",
            "mlp_up": "expert_up",
            "mlp_down": "expert_down",
        }[base_role.role]
        output.append(CanonicalExpertLinearRole(
            role=canonical_role,
            path=path,
            expert_count=expert_count,
            input_features=base_role.input_features,
            output_features=base_role.output_features,
            bias=base_role.bias,
            storage_layout="separate_" + base_role.storage_layout,
            module_type=base_role.module_type,
            layer_index=layer_index,
            paths=tuple(item.path for item in instances),
        ))
    return tuple(output), next(iter(kinds))


def _map_shared_experts(
    moe: nn.Module,
    *,
    path: str,
    layer_index: int,
    hidden_size: int,
) -> tuple[
    str | None,
    str | None,
    str | None,
    tuple[CanonicalExpertLinearRole, ...],
]:
    found = _descendant_by_alias(moe, _SHARED_EXPERT_NAMES, max_depth=3)
    if found is None:
        return None, None, None, ()
    relative, module = found
    shared_path = _join(path, relative)
    if isinstance(module, nn.ModuleList):
        experts = module
        projections, kind = _map_separate_experts(
            experts,
            path=shared_path,
            layer_index=layer_index,
            hidden_size=hidden_size,
        )
    else:
        kind, roles = _one_expert_roles(
            module,
            path=shared_path,
            layer_index=layer_index,
            hidden_size=hidden_size,
        )
        projections = tuple(CanonicalExpertLinearRole(
            role={
                "gate": "shared_expert_gate",
                "up": "shared_expert_up",
                "gate_up": "shared_expert_gate_up",
                "down": "shared_expert_down",
                "mlp_up": "shared_expert_up",
                "mlp_down": "shared_expert_down",
            }[role.role],
            path=role.path,
            expert_count=1,
            input_features=role.input_features,
            output_features=role.output_features,
            bias=role.bias,
            storage_layout="separate_" + role.storage_layout,
            module_type=role.module_type,
            layer_index=layer_index,
            paths=(role.path,),
        ) for role in roles)
    # Rewrite routed-style names produced by the ModuleList helper.
    if isinstance(module, nn.ModuleList):
        projections = tuple(CanonicalExpertLinearRole(
            **{
                **vars(item),
                "role": item.role.replace("expert_", "shared_expert_", 1),
            }
        ) for item in projections)
    gate = _descendant_by_alias(
        moe, _SHARED_EXPERT_GATE_NAMES, max_depth=3
    )
    gate_path = _join(path, gate[0]) if gate is not None else None
    return shared_path, gate_path, kind, projections


def _map_moe(
    moe: nn.Module,
    path: str,
    layer_index: int,
    *,
    hidden_size: int,
) -> CanonicalMoELayerMap:
    weighted = _three_dimensional_weight_modules(moe)
    container = _expert_container(moe)
    if weighted and container is not None:
        # A shared expert may contain ordinary 2-D tensors, but routed experts
        # must have one authoritative storage representation.
        raise ValueError(
            f"MoE block {path} ambiguously exposes fused and separate routed experts"
        )

    if weighted:
        projections, expert_count, expert_kind = _map_fused_experts(
            weighted,
            path=path,
            layer_index=layer_index,
            hidden_size=hidden_size,
        )
        storage_layout = "fused"
    elif container is not None:
        container_name, experts = container
        projections, expert_kind = _map_separate_experts(
            experts,
            path=_join(path, container_name),
            layer_index=layer_index,
            hidden_size=hidden_size,
        )
        expert_count = len(experts)
        storage_layout = "separate"
    else:
        raise ValueError(f"MoE block {path} lacks routed expert projections")

    router_name, router_module = _find_router(
        moe, hidden_size=hidden_size, expert_count=expert_count
    )
    router_input, router_output, _layout, _bias, _kind = _linear_geometry(
        router_module
    )
    if router_output != expert_count:
        raise ValueError("MoE router output does not equal expert count")
    router_object = getattr(moe, "router", None)
    top_k = int(
        next(
            (
                value
                for value in (
                    getattr(router_object, "top_k", None),
                    getattr(moe, "top_k", None),
                    getattr(moe, "num_experts_per_tok", None),
                    getattr(moe, "num_experts_per_token", None),
                )
                if value is not None
            ),
            1,
        )
    )
    shared = _map_shared_experts(
        moe,
        path=path,
        layer_index=layer_index,
        hidden_size=hidden_size,
    )
    shared_path, shared_gate_path, shared_kind, shared_projections = shared
    group_count = _first_int_attribute(
        moe, "n_group", "num_expert_groups", "expert_group_count"
    )
    group_top_k = _first_int_attribute(
        moe, "topk_group", "num_expert_groups_per_token", "expert_group_top_k"
    )
    return CanonicalMoELayerMap(
        path=path,
        router_path=_join(path, router_name),
        router_input_features=router_input,
        expert_count=expert_count,
        top_k=top_k,
        expert_mlp_kind=expert_kind,
        projections=projections,
        expert_storage_layout=storage_layout,
        shared_expert_path=shared_path,
        shared_expert_gate_path=shared_gate_path,
        shared_expert_mlp_kind=shared_kind,
        shared_expert_projections=shared_projections,
        expert_group_count=group_count,
        expert_group_top_k=group_top_k,
    )


def _map_layer(layer: nn.Module, path: str, index: int) -> CanonicalLayerMap:
    attention_found = _child_by_alias(layer, ATTENTION_NAMES)
    mlp_found = _child_by_alias(layer, MLP_NAMES)
    moe_found = _find_moe(layer)
    if mlp_found is not None and _looks_like_moe(mlp_found[1]):
        if moe_found is None:
            moe_found = mlp_found
        if moe_found[1] is mlp_found[1]:
            mlp_found = None
    flat_mlp_up = (
        _projection(layer, path, "mlp_up", index)
        if mlp_found is None and moe_found is None else None
    )
    flat_mlp_down = (
        _projection(layer, path, "mlp_down", index)
        if mlp_found is None and moe_found is None else None
    )
    flat_mlp = flat_mlp_up is not None and flat_mlp_down is not None
    if (
        attention_found is None
        or (mlp_found is None and moe_found is None and not flat_mlp)
    ):
        raise ValueError(f"layer {path} lacks recognizable attention or MLP/MoE")
    attention_name, attention = attention_found
    attention_path = _join(path, attention_name)
    mlp_path = (
        _join(path, mlp_found[0])
        if mlp_found is not None else path if flat_mlp else None
    )

    projections: list[CanonicalLinearRole] = []
    separate = [
        _projection(attention, attention_path, role, index)
        for role in ("q", "k", "v")
    ]
    if all(item is not None for item in separate):
        projections.extend(item for item in separate if item is not None)
    else:
        q = _projection(attention, attention_path, "q", index)
        k = _projection(attention, attention_path, "k", index)
        v = _projection(attention, attention_path, "v", index)
        kv = _projection(attention, attention_path, "kv", index)
        qk = _projection(attention, attention_path, "qk", index)
        if q is not None and kv is not None and k is None and v is None:
            projections.extend((q, kv))
        elif qk is not None and v is not None and q is None and k is None:
            projections.extend((qk, v))
        else:
            fused = _projection(attention, attention_path, "qkv", index)
            if fused is None:
                raise ValueError(
                    f"layer {path} has neither separate nor fused QKV and "
                    "does not expose a valid partial fusion"
                )
            projections.append(fused)
    output = _projection(attention, attention_path, "o", index)
    if output is None:
        raise ValueError(f"layer {path} lacks an attention output projection")
    projections.append(output)

    mapped_moe = None
    if mlp_found is not None or flat_mlp:
        if mlp_found is not None:
            mlp_name, mlp = mlp_found
        else:
            mlp_name, mlp = "", layer
        assert mlp_path is not None
        gate = _projection(mlp, mlp_path, "gate", index)
        up = _projection(mlp, mlp_path, "up", index)
        down = _projection(mlp, mlp_path, "down", index)
        gate_up = _projection(mlp, mlp_path, "gate_up", index)
        if gate is not None and up is not None and down is not None:
            projections.extend((gate, up, down))
        elif gate_up is not None and down is not None:
            projections.extend((gate_up, down))
        elif gate is not None and down is not None and (
            gate.input_features == down.output_features
            and gate.output_features == 2 * down.input_features
        ):
            # A fused gate/up tensor is sometimes published under the ordinary
            # ``gate_proj`` name.  Geometry proves the fused semantic without
            # relying on a repository/family name.
            projections.extend((
                CanonicalLinearRole(
                    role="gate_up",
                    path=gate.path,
                    input_features=gate.input_features,
                    output_features=gate.output_features,
                    bias=gate.bias,
                    storage_layout=gate.storage_layout,
                    module_type=gate.module_type,
                    layer_index=gate.layer_index,
                ),
                down,
            ))
        else:
            ordinary_up = (
                flat_mlp_up
                if flat_mlp and mlp is layer
                else _projection(mlp, mlp_path, "mlp_up", index)
            )
            ordinary_down = (
                flat_mlp_down
                if flat_mlp and mlp is layer
                else _projection(mlp, mlp_path, "mlp_down", index)
            )
            if ordinary_up is None or ordinary_down is None:
                raise ValueError(f"layer {path} has an unsupported MLP projection topology")
            projections.extend((ordinary_up, ordinary_down))
    if moe_found is not None:
        moe_name, moe_module = moe_found
        q_projection = next(item for item in projections if item.role in {"q", "qkv"})
        mapped_moe = _map_moe(
            moe_module,
            _join(path, moe_name),
            index,
            hidden_size=q_projection.input_features,
        )

    norms = _norm_children(layer)
    if not norms:
        raise ValueError(f"layer {path} has no normalization module")
    input_aliases = (
        "input_layernorm", "ln_1", "self_attn_layer_norm", "attention_norm",
    )
    post_aliases = (
        "post_attention_layernorm", "ln_2", "final_layer_norm", "ffn_norm",
    )
    input_norm = _child_by_alias(layer, input_aliases)
    post_norm = _child_by_alias(layer, post_aliases)
    if input_norm is None:
        input_norm = norms[0]
    if post_norm is None and len(norms) > 1:
        post_norm = norms[1]
    query_norm = _descendant_by_alias(
        attention, ("q_norm", "query_norm"), max_depth=3
    )
    key_norm = _descendant_by_alias(
        attention, ("k_norm", "key_norm"), max_depth=3
    )
    return CanonicalLayerMap(
        index=index,
        path=path,
        attention_path=attention_path,
        mlp_path=mlp_path,
        input_norm_path=_join(path, input_norm[0]),
        post_attention_norm_path=(
            _join(path, post_norm[0]) if post_norm is not None else None
        ),
        query_norm_path=(
            _join(attention_path, query_norm[0])
            if query_norm is not None else None
        ),
        key_norm_path=(
            _join(attention_path, key_norm[0])
            if key_norm is not None else None
        ),
        projections=tuple(projections),
        moe=mapped_moe,
    )


def map_model(model: nn.Module) -> CanonicalModelMap:
    """Discover decoder layers and weights from structure, not model name."""

    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("model has no Hugging Face config")
    descriptor = describe_config(config)
    decoder_path, layers = _decoder_layers(model, descriptor.layer_count)
    mapped_layers = tuple(
        _map_layer(layer, _join(decoder_path, str(index)), index)
        for index, layer in enumerate(layers)
    )

    input_embedding = (
        model.get_input_embeddings()
        if hasattr(model, "get_input_embeddings") else None
    )
    output_embedding = (
        model.get_output_embeddings()
        if hasattr(model, "get_output_embeddings") else None
    )
    embedding_path = _named_identity(model, input_embedding)
    lm_head_path = _named_identity(model, output_embedding)
    if embedding_path is None or lm_head_path is None:
        raise ValueError("could not locate input embedding or LM head")

    layer_prefixes = tuple(f"{decoder_path}.{index}." for index in range(len(layers)))
    outside_norms = [
        name
        for name, module in model.named_modules()
        if name
        and not name.startswith(layer_prefixes)
        and ("norm" in name.lower() or "norm" in type(module).__name__.lower())
    ]
    preferred_final = [
        name for name in outside_norms
        if name.rsplit(".", 1)[-1] in {"norm", "final_layer_norm", "ln_f"}
    ]
    final_norm_path = (preferred_final or outside_norms or [None])[-1]
    if final_norm_path is None and descriptor.block_norm_placement != "post_norm":
        raise ValueError("could not locate final normalization")

    input_weight = getattr(input_embedding, "weight", None)
    output_weight = getattr(output_embedding, "weight", None)
    tied = input_weight is not None and input_weight is output_weight
    global_projections: list[CanonicalLinearRole] = []
    global_aliases = {
        "embedding_in": ("project_in", "embedding_projection", "embed_proj"),
        "embedding_out": (
            "project_out", "output_projection", "unembed_proj",
        ),
    }
    for role, aliases in global_aliases.items():
        candidates = [
            (name, module)
            for name, module in model.named_modules()
            if name
            and name.rsplit(".", 1)[-1] in aliases
            and _is_linear_like(module)
            and not name.startswith(layer_prefixes)
        ]
        if len(candidates) > 1:
            raise ValueError(
                f"multiple global {role} projection candidates: "
                f"{[name for name, _module in candidates]}"
            )
        if candidates:
            name, module = candidates[0]
            input_features, output_features, layout, bias, module_type = (
                _linear_geometry(module)
            )
            global_projections.append(CanonicalLinearRole(
                role=role,
                path=name,
                input_features=input_features,
                output_features=output_features,
                bias=bias,
                storage_layout=layout,
                module_type=module_type,
                layer_index=-1,
            ))
    parameter_paths: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        parameter_paths.setdefault(id(parameter), []).append(name)
    shared_parameter_views = []
    for paths in parameter_paths.values():
        unique = tuple(sorted(set(paths)))
        if len(unique) < 2:
            continue
        identity = hashlib.sha256(
            "\0".join(unique).encode("utf-8")
        ).hexdigest()[:16]
        shared_parameter_views.append((identity, unique))
    return CanonicalModelMap(
        decoder_path=decoder_path,
        embedding_path=embedding_path,
        final_norm_path=final_norm_path,
        lm_head_path=lm_head_path,
        layers=mapped_layers,
        global_projections=tuple(global_projections),
        tied_embeddings=tied,
        shared_parameter_views=tuple(sorted(shared_parameter_views)),
    )


def describe_model(
    model: nn.Module,
    *,
    dtype: str | None = None,
) -> tuple[CanonicalArchitectureDescriptor, CanonicalModelMap]:
    """Combine config geometry with authoritative module-tree topology."""

    base = describe_config(model.config, dtype=dtype)
    mapping = map_model(model)
    first = mapping.layers[0]
    roles = tuple(item.role for item in first.projections)
    if {"q", "k", "v"} <= set(roles):
        qkv_layout = "separate"
    elif {"q", "kv"} <= set(roles):
        qkv_layout = "q_plus_fused_kv"
    elif {"qk", "v"} <= set(roles):
        qkv_layout = "fused_qk_plus_v"
    else:
        qkv = first.projection("qkv")
        qkv_layout = "fused_conv1d" if qkv.storage_layout == "in_out" else "fused_linear"
    observed_moe_layers = tuple(
        layer for layer in mapping.layers if layer.moe is not None
    )
    if observed_moe_layers:
        first_moe = observed_moe_layers[0].moe
        assert first_moe is not None
        mlp_kind = first_moe.expert_mlp_kind
        gate_up_layout = "fused" if mlp_kind != "ordinary" else "none"
    elif {"gate", "up", "down"} <= set(roles):
        mlp_kind = "swiglu" if base.mlp_activation == "silu" else "gated"
        gate_up_layout = "separate"
    elif {"gate_up", "down"} <= set(roles):
        mlp_kind = "swiglu" if base.mlp_activation == "silu" else "gated"
        gate_up_layout = "fused"
    else:
        mlp_kind = "ordinary"
        gate_up_layout = "none"

    by_role = {item.role: item for item in first.projections}
    q_width = base.attention_heads * base.head_dimension
    kv_width = base.kv_heads * base.head_dimension
    if qkv_layout == "separate":
        expected = {
            "q": (base.hidden_size, q_width),
            "k": (base.hidden_size, kv_width),
            "v": (base.hidden_size, kv_width),
            "o": (q_width, base.hidden_size),
        }
    elif qkv_layout == "q_plus_fused_kv":
        expected = {
            "q": (base.hidden_size, q_width),
            "kv": (base.hidden_size, 2 * kv_width),
            "o": (q_width, base.hidden_size),
        }
    elif qkv_layout == "fused_qk_plus_v":
        expected = {
            "qk": (base.hidden_size, q_width + kv_width),
            "v": (base.hidden_size, kv_width),
            "o": (q_width, base.hidden_size),
        }
    else:
        expected = {
            "qkv": (base.hidden_size, q_width + 2 * kv_width),
            "o": (q_width, base.hidden_size),
        }
    for role, (wanted_input, wanted_output) in expected.items():
        projection = by_role[role]
        if (
            projection.input_features != wanted_input
            or projection.output_features != wanted_output
        ):
            raise ValueError(
                f"layer {first.path} {role} projection geometry "
                f"{projection.input_features}x{projection.output_features} "
                f"does not match canonical {wanted_input}x{wanted_output}"
            )

    if observed_moe_layers and "down" not in by_role:
        # A fully sparse layer has no dense MLP projection to observe.  Its
        # expert width is represented separately by MoEDescriptor, while this
        # dense-width field retains the canonical config value for any dense
        # layers in a mixed stack.
        observed_intermediate = base.intermediate_size
    elif mlp_kind == "ordinary":
        observed_intermediate = by_role["mlp_up"].output_features
        if by_role["mlp_down"].input_features != observed_intermediate:
            raise ValueError("ordinary MLP input/output projections disagree")
    elif gate_up_layout == "fused":
        observed_intermediate = by_role["down"].input_features
        if by_role["gate_up"].output_features != 2 * observed_intermediate:
            raise ValueError("fused gate/up width is not twice MLP intermediate")
    else:
        observed_intermediate = by_role["gate"].output_features
        if (
            by_role["up"].output_features != observed_intermediate
            or by_role["down"].input_features != observed_intermediate
        ):
            raise ValueError("separate gated MLP projection widths disagree")

    first_norm = model.get_submodule(first.input_norm_path)
    norm_kind = (
        "rms_norm"
        if "rms" in type(first_norm).__name__.lower()
        else "layer_norm"
    )
    qkv_bias = by_role.get("qkv")
    query_bias = by_role.get("q") or by_role.get("qk") or qkv_bias
    key_bias = by_role.get("k") or by_role.get("kv") or by_role.get("qk") or qkv_bias
    value_bias = by_role.get("v") or by_role.get("kv") or qkv_bias
    mlp_input = (
        by_role.get("gate")
        or by_role.get("gate_up")
        or by_role.get("mlp_up")
    )
    mlp_output = by_role.get("down") or by_role.get("mlp_down")
    expert_targets = mapping.expert_lora_targets
    shared_expert_targets = mapping.shared_expert_lora_targets
    expert_bias = any(
        item.bias for item in (*expert_targets, *shared_expert_targets)
    )
    biases = ProjectionBiases(
        query=bool(query_bias and query_bias.bias),
        key=bool(key_bias and key_bias.bias),
        value=bool(value_bias and value_bias.bias),
        output=by_role["o"].bias,
        mlp_input=bool(mlp_input and mlp_input.bias) or expert_bias,
        mlp_output=bool(mlp_output and mlp_output.bias) or expert_bias,
    )
    lm_head = model.get_submodule(mapping.lm_head_path)
    _in, _out, lm_layout, _bias, _type = _linear_geometry(lm_head)
    observed_moe = base.moe
    if observed_moe_layers:
        first_moe = observed_moe_layers[0].moe
        assert first_moe is not None
        if any(
            layer.moe is not None
            and (
                layer.moe.expert_count != first_moe.expert_count
                or layer.moe.top_k != first_moe.top_k
                or layer.moe.expert_mlp_kind != first_moe.expert_mlp_kind
                or layer.moe.expert_storage_layout
                    != first_moe.expert_storage_layout
                or bool(layer.moe.shared_expert_projections)
                    != bool(first_moe.shared_expert_projections)
                or layer.moe.expert_group_count
                    != first_moe.expert_group_count
                or layer.moe.expert_group_top_k
                    != first_moe.expert_group_top_k
            )
            for layer in observed_moe_layers
        ):
            raise ValueError("heterogeneous per-layer MoE geometry is not yet canonical")
        moe_indices = tuple(layer.index for layer in observed_moe_layers)
        dense_indices = tuple(
            layer.index for layer in mapping.layers if layer.moe is None
        )
        expert_down = first_moe.projection("expert_down")
        shared_down = next(
            (
                item
                for item in first_moe.shared_expert_projections
                if item.role == "shared_expert_down"
            ),
            None,
        )
        shared_count = (
            shared_down.expert_count
            if shared_down is not None
            else (base.moe.shared_expert_count if base.moe else 0)
        )
        original_moe_module = model.get_submodule(first_moe.path)
        observed_router_normalization = (
            base.moe.router_normalization
            if base.moe else "softmax_selected"
        )
        # A separate-expert block with no explicit normalization flag follows
        # the common full-softmax -> top-k -> selected renormalization
        # contract. Fused routers retain their explicit/configured semantics.
        # This is a structural semantic canonicalization, never an execution
        # decision based on a checkpoint or repository name.
        if (
            first_moe.expert_storage_layout == "separate"
            and not hasattr(original_moe_module, "norm_topk_prob")
            and observed_router_normalization == "softmax_selected"
        ):
            observed_router_normalization = (
                "softmax_all_renormalized_selected"
            )
        observed_moe = MoEDescriptor(
            expert_count=first_moe.expert_count,
            top_k=first_moe.top_k,
            router_type=(base.moe.router_type if base.moe else "linear_topk"),
            router_logits_dtype=(
                base.moe.router_logits_dtype if base.moe else "float32"
            ),
            router_normalization=observed_router_normalization,
            expert_mlp_kind=first_moe.expert_mlp_kind,
            expert_intermediate_size=expert_down.input_features,
            shared_expert_count=shared_count,
            expert_group_count=(
                first_moe.expert_group_count
                if first_moe.expert_group_count is not None
                else (base.moe.expert_group_count if base.moe else None)
            ),
            expert_bias=expert_bias,
            auxiliary_loss_kind=(
                base.moe.auxiliary_loss_kind if base.moe else "none"
            ),
            auxiliary_loss_coefficient=(
                base.moe.auxiliary_loss_coefficient if base.moe else 0.0
            ),
            expert_lora_roles=tuple(
                dict.fromkeys(item.role for item in expert_targets)
            ),
            moe_layer_indices=moe_indices,
            dense_layer_indices=dense_indices,
            expert_storage_layout=first_moe.expert_storage_layout,
            shared_expert_storage_layout=(
                "separate" if shared_down is not None else None
            ),
            shared_expert_mlp_kind=first_moe.shared_expert_mlp_kind,
            shared_expert_intermediate_size=(
                shared_down.input_features
                if shared_down is not None else None
            ),
            shared_expert_gate=first_moe.shared_expert_gate_path is not None,
            expert_group_top_k=(
                first_moe.expert_group_top_k
                if first_moe.expert_group_top_k is not None
                else (base.moe.expert_group_top_k if base.moe else None)
            ),
            expert_group_policy=(
                base.moe.expert_group_policy if base.moe else None
            ),
            router_z_loss_coefficient=(
                base.moe.router_z_loss_coefficient if base.moe else 0.0
            ),
            expert_capacity_factor=(
                base.moe.expert_capacity_factor if base.moe else None
            ),
            expert_min_capacity=(
                base.moe.expert_min_capacity if base.moe else None
            ),
            overflow_policy=(
                base.moe.overflow_policy if base.moe else "none"
            ),
            router_jitter_noise=(
                base.moe.router_jitter_noise if base.moe else 0.0
            ),
        )
    lora_roles = tuple(dict.fromkeys(
        roles
        + tuple(item.role for item in mapping.global_projections)
        + tuple(item.role for item in expert_targets)
        + tuple(item.role for item in shared_expert_targets)
    ))

    norm_count = sum(
        path is not None
        for path in (first.input_norm_path, first.post_attention_norm_path)
    )
    observed_placement = (
        "parallel_pre_norm" if norm_count == 1
        else base.block_norm_placement
    )

    query_norm_module = (
        model.get_submodule(first.query_norm_path)
        if first.query_norm_path is not None else None
    )
    key_norm_module = (
        model.get_submodule(first.key_norm_path)
        if first.key_norm_path is not None else None
    )
    if (query_norm_module is None) != (key_norm_module is None):
        raise ValueError("attention must expose both query and key norms")
    if query_norm_module is None:
        qk_norm_kind = None
        qk_norm_epsilon = None
    else:
        query_kind = (
            "rms_norm"
            if "rms" in type(query_norm_module).__name__.lower()
            else "layer_norm"
        )
        key_kind = (
            "rms_norm"
            if "rms" in type(key_norm_module).__name__.lower()
            else "layer_norm"
        )
        if query_kind != key_kind:
            raise ValueError("query and key norms use different semantics")
        qk_norm_kind = query_kind
        qk_norm_epsilon = float(
            getattr(
                query_norm_module,
                "variance_epsilon",
                getattr(query_norm_module, "eps", base.norm_epsilon),
            )
        )

    observed_rope = base.rope
    position_modules = tuple(model.named_modules())
    sinusoidal = any(
        "sinusoidal" in type(module).__name__.lower()
        and "position" in type(module).__name__.lower()
        for _name, module in position_modules
    )
    learned_position = any(
        (
            name.rsplit(".", 1)[-1].lower() in {
                "wpe", "embed_positions", "position_embeddings",
            }
            or "learnedpositional" in type(module).__name__.lower()
        )
        and isinstance(getattr(module, "weight", None), torch.Tensor)
        for name, module in position_modules
    )
    if sinusoidal:
        observed_rope = RopeDescriptor(
            kind="sinusoidal_absolute",
            theta=None,
            rotary_dimension=0,
        )
    elif learned_position:
        observed_rope = RopeDescriptor(
            kind="learned_absolute",
            theta=None,
            rotary_dimension=0,
        )

    embedding_module = model.get_submodule(mapping.embedding_path)
    embedding_weight = getattr(embedding_module, "weight", None)
    observed_embedding_dimension = (
        int(embedding_weight.shape[1])
        if isinstance(embedding_weight, torch.Tensor)
        and embedding_weight.ndim == 2
        else base.embedding_dimension
    )
    descriptor = with_observed_topology(
        base,
        qkv_projection_layout=qkv_layout,
        norm_kind=norm_kind,
        mlp_kind=mlp_kind,
        gate_up_layout=gate_up_layout,
        biases=biases,
        lora_roles=lora_roles,
        tied_embeddings=mapping.tied_embeddings,
        lm_head_layout="conv1d" if lm_layout == "in_out" else "linear",
        final_norm=mapping.final_norm_path is not None,
        intermediate_size=observed_intermediate,
        block_norm_placement=observed_placement,
        rope=observed_rope,
        attention_qk_norm=qk_norm_kind,
        attention_qk_norm_epsilon=qk_norm_epsilon,
        embedding_dimension=observed_embedding_dimension,
        moe=observed_moe,
    )
    return descriptor, mapping


def canonical_lora_target_paths(model: nn.Module) -> tuple[str, ...]:
    """Return all canonical decoder projection paths, excluding the LM head."""

    return tuple(item.path for item in map_model(model).lora_targets)


def grouped_lora_target_geometry(
    mapping: CanonicalModelMap,
    *,
    selected_paths: Iterable[str] | None = None,
) -> tuple[tuple[str, int, int, int], ...]:
    """Aggregate canonical target instances by semantic role and shape."""

    selected = set(selected_paths) if selected_paths is not None else None
    counts: dict[tuple[str, int, int], int] = {}
    for item in mapping.lora_targets:
        if selected is not None and item.path not in selected:
            continue
        key = (item.role, item.input_features, item.output_features)
        counts[key] = counts.get(key, 0) + 1
    return tuple(
        (role, input_features, output_features, count)
        for (role, input_features, output_features), count in sorted(counts.items())
    )


def grouped_expert_lora_target_geometry(
    mapping: CanonicalModelMap,
) -> tuple[tuple[str, int, int, int], ...]:
    """Aggregate 3-D expert projections by role and logical expert instance."""

    counts: dict[tuple[str, int, int], int] = {}
    for item in mapping.expert_lora_targets:
        key = (item.role, item.input_features, item.output_features)
        counts[key] = counts.get(key, 0) + item.expert_count
    return tuple(
        (role, input_features, output_features, count)
        for (role, input_features, output_features), count in sorted(counts.items())
    )


__all__ = [
    "CanonicalLayerMap",
    "CanonicalExpertLinearRole",
    "CanonicalMoELayerMap",
    "CanonicalLinearRole",
    "CanonicalModelMap",
    "canonical_lora_target_paths",
    "describe_model",
    "grouped_lora_target_geometry",
    "grouped_expert_lora_target_geometry",
    "map_model",
]
