"""Canonical, model-name-independent causal-decoder architecture discovery.

Hugging Face configuration classes use different field names and occasionally
omit topology facts that are obvious from the instantiated module tree.  This
module converts those dialects into one immutable semantic descriptor.  The
``model_type`` and ``architectures`` strings are retained for diagnostics only;
planner decisions consume the canonical fields and semantic fingerprint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from typing import Any, Mapping


_MISSING = object()


@dataclass(frozen=True)
class RopeDescriptor:
    kind: str
    theta: float | None
    rotary_dimension: int
    scaling: tuple[tuple[str, str], ...] = ()

    @property
    def uses_rope(self) -> bool:
        return self.kind in {"standard", "scaled"}


@dataclass(frozen=True)
class ProjectionBiases:
    query: bool
    key: bool
    value: bool
    output: bool
    mlp_input: bool
    mlp_output: bool


@dataclass(frozen=True)
class MoEDescriptor:
    """Canonical sparse-expert semantics independent of checkpoint naming."""

    expert_count: int
    top_k: int
    router_type: str
    router_logits_dtype: str
    router_normalization: str
    expert_mlp_kind: str
    expert_intermediate_size: int
    shared_expert_count: int
    expert_group_count: int | None
    expert_bias: bool
    auxiliary_loss_kind: str
    auxiliary_loss_coefficient: float
    expert_lora_roles: tuple[str, ...]
    moe_layer_indices: tuple[int, ...]
    dense_layer_indices: tuple[int, ...]
    # autotuner extensions.  Defaults intentionally describe the already
    # accepted Granite-style contract and are omitted from semantic_dict so
    # frozen architecture/6.5 fingerprints remain stable.
    expert_storage_layout: str = "fused"
    shared_expert_storage_layout: str | None = None
    shared_expert_mlp_kind: str | None = None
    shared_expert_intermediate_size: int | None = None
    shared_expert_gate: bool = False
    expert_group_top_k: int | None = None
    expert_group_policy: str | None = None
    router_z_loss_coefficient: float = 0.0
    expert_capacity_factor: float | None = None
    expert_min_capacity: int | None = None
    overflow_policy: str = "none"
    router_jitter_noise: float = 0.0

    @property
    def selected_expert_compute_fraction(self) -> float:
        """Ideal routed-expert work fraction before load imbalance."""

        return self.top_k / self.expert_count


@dataclass(frozen=True)
class CanonicalArchitectureDescriptor:
    """Semantic and geometric contract for one decoder-only causal LM."""

    hidden_size: int
    intermediate_size: int
    layer_count: int
    vocabulary_size: int
    attention_heads: int
    kv_heads: int
    head_dimension: int
    attention_kind: str
    qkv_projection_layout: str
    norm_kind: str
    norm_epsilon: float
    block_norm_placement: str
    mlp_kind: str
    mlp_activation: str
    gate_up_layout: str
    rope: RopeDescriptor
    tied_embeddings: bool
    lm_head_layout: str
    final_norm: bool
    attention_window: int | None
    local_attention_layers: tuple[int, ...]
    attention_qk_norm: str | None
    attention_qk_norm_epsilon: float | None
    embedding_dimension: int
    embedding_scale: float
    dtype: str
    biases: ProjectionBiases
    lora_roles: tuple[str, ...]
    moe: MoEDescriptor | None = None
    decoder_only: bool = True
    causal: bool = True
    diagnostics: tuple[tuple[str, str], ...] = field(
        default=(), compare=False, hash=False,
    )

    def validate(self) -> None:
        dimensions = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "layer_count": self.layer_count,
            "vocabulary_size": self.vocabulary_size,
            "attention_heads": self.attention_heads,
            "kv_heads": self.kv_heads,
            "head_dimension": self.head_dimension,
        }
        invalid = {name: value for name, value in dimensions.items() if value <= 0}
        if invalid:
            raise ValueError(f"architecture dimensions must be positive: {invalid}")
        if self.attention_heads % self.kv_heads:
            raise ValueError("attention heads must be divisible by KV heads")
        # The query projection width is heads * head_dimension, but it need not
        # equal the residual-stream width.  Qwen3 is a normal dense decoder
        # whose learned query space is wider than ``hidden_size``.  Projection
        # geometry is checked against the instantiated module tree by the
        # structural mapper rather than imposing the older equality here.
        if self.attention_kind not in {"mha", "gqa", "mqa"}:
            raise ValueError(f"unsupported attention kind: {self.attention_kind}")
        if self.qkv_projection_layout not in {
            "separate", "fused_linear", "fused_conv1d",
            "q_plus_fused_kv", "fused_qk_plus_v",
        }:
            raise ValueError(
                f"unsupported QKV layout: {self.qkv_projection_layout}"
            )
        if self.norm_kind not in {"rms_norm", "layer_norm"}:
            raise ValueError(f"unsupported norm kind: {self.norm_kind}")
        if self.norm_epsilon <= 0:
            raise ValueError("normalization epsilon must be positive")
        if self.block_norm_placement not in {
            "sequential_pre_norm", "parallel_pre_norm", "post_norm",
        }:
            raise ValueError(
                f"unsupported norm placement: {self.block_norm_placement}"
            )
        if self.mlp_kind not in {"swiglu", "gated", "ordinary"}:
            raise ValueError(f"unsupported MLP kind: {self.mlp_kind}")
        if self.gate_up_layout not in {"separate", "fused", "none"}:
            raise ValueError(f"unsupported gate/up layout: {self.gate_up_layout}")
        if self.mlp_kind == "ordinary" and self.gate_up_layout != "none":
            raise ValueError("ordinary MLP must not expose a gate/up layout")
        if self.mlp_kind != "ordinary" and self.gate_up_layout == "none":
            raise ValueError("gated MLP requires separate or fused gate/up")
        if self.rope.rotary_dimension < 0:
            raise ValueError("rotary dimension cannot be negative")
        if self.rope.rotary_dimension > self.head_dimension:
            raise ValueError("rotary dimension cannot exceed head dimension")
        if self.attention_window is not None and self.attention_window <= 0:
            raise ValueError("attention window must be positive")
        if any(
            index < 0 or index >= self.layer_count
            for index in self.local_attention_layers
        ):
            raise ValueError("local attention layer index is outside the decoder")
        if len(set(self.local_attention_layers)) != len(
            self.local_attention_layers
        ):
            raise ValueError("local attention layer indices must be unique")
        if self.local_attention_layers and self.attention_window is None:
            raise ValueError("local attention layers require an attention window")
        if self.attention_qk_norm not in {None, "rms_norm", "layer_norm"}:
            raise ValueError(
                f"unsupported attention Q/K norm: {self.attention_qk_norm}"
            )
        if (
            self.attention_qk_norm is None
            and self.attention_qk_norm_epsilon is not None
        ):
            raise ValueError("Q/K norm epsilon requires a Q/K norm")
        if (
            self.attention_qk_norm is not None
            and (
                self.attention_qk_norm_epsilon is None
                or self.attention_qk_norm_epsilon <= 0
            )
        ):
            raise ValueError("Q/K norm requires a positive epsilon")
        if self.embedding_dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        if not math.isfinite(self.embedding_scale) or self.embedding_scale <= 0:
            raise ValueError("embedding scale must be finite and positive")
        if not self.decoder_only or not self.causal:
            raise ValueError("architecture requires a decoder-only causal LM")
        if self.moe is not None:
            if self.moe.expert_count <= 0:
                raise ValueError("MoE expert count must be positive")
            if not 0 < self.moe.top_k <= self.moe.expert_count:
                raise ValueError("MoE top-k must be in [1, expert_count]")
            if self.moe.expert_intermediate_size <= 0:
                raise ValueError("MoE expert intermediate size must be positive")
            if self.moe.shared_expert_count < 0:
                raise ValueError("MoE shared expert count cannot be negative")
            if self.moe.expert_storage_layout not in {
                "fused", "separate",
            }:
                raise ValueError(
                    "MoE expert storage must be fused or separate"
                )
            if self.moe.shared_expert_storage_layout not in {
                None, "fused", "separate",
            }:
                raise ValueError(
                    "shared expert storage must be fused, separate, or absent"
                )
            if (
                self.moe.shared_expert_count
                and self.moe.shared_expert_mlp_kind not in {
                    "swiglu", "gated", "ordinary",
                }
            ):
                raise ValueError(
                    "shared experts require a canonical MLP kind"
                )
            if (
                self.moe.shared_expert_count
                and (
                    self.moe.shared_expert_intermediate_size is None
                    or self.moe.shared_expert_intermediate_size <= 0
                )
            ):
                raise ValueError(
                    "shared experts require a positive intermediate size"
                )
            if (
                self.moe.expert_group_count is not None
                and self.moe.expert_group_count <= 0
            ):
                raise ValueError("MoE expert group count must be positive")
            if self.moe.expert_group_top_k is not None:
                if self.moe.expert_group_count is None:
                    raise ValueError(
                        "group top-k requires an expert group count"
                    )
                if not 0 < self.moe.expert_group_top_k <= self.moe.expert_group_count:
                    raise ValueError(
                        "expert group top-k must be in [1, group count]"
                    )
            if self.moe.router_z_loss_coefficient < 0:
                raise ValueError("router z-loss coefficient cannot be negative")
            if self.moe.expert_capacity_factor is not None and (
                self.moe.expert_capacity_factor <= 0
            ):
                raise ValueError("expert capacity factor must be positive")
            if self.moe.expert_min_capacity is not None and (
                self.moe.expert_min_capacity < 0
            ):
                raise ValueError("expert minimum capacity cannot be negative")
            if self.moe.overflow_policy not in {
                "none", "drop", "residual", "overflow_expert",
            }:
                raise ValueError(
                    f"unsupported MoE overflow policy: {self.moe.overflow_policy}"
                )
            if self.moe.router_jitter_noise < 0:
                raise ValueError("router jitter noise cannot be negative")
            layer_indices = self.moe.moe_layer_indices + self.moe.dense_layer_indices
            if len(set(layer_indices)) != len(layer_indices):
                raise ValueError("MoE and dense layer patterns overlap")
            if any(index < 0 or index >= self.layer_count for index in layer_indices):
                raise ValueError("MoE layer pattern contains an invalid layer index")
            if set(layer_indices) != set(range(self.layer_count)):
                raise ValueError("MoE layer pattern must classify every decoder layer")

    def semantic_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("diagnostics", None)
        # Preserve the frozen architecture dense fingerprints exactly.  Sparse
        # semantics participate in the fingerprint whenever they are present.
        if payload.get("moe") is None:
            payload.pop("moe", None)
        else:
            moe = payload["moe"]
            # Do not perturb historical sparse fingerprints for the original
            # implicit defaults.  Non-default autotuner MoE semantics remain
            # part of the stable semantic identity.
            defaults = {
                "expert_storage_layout": "fused",
                "shared_expert_storage_layout": None,
                "shared_expert_mlp_kind": None,
                "shared_expert_intermediate_size": None,
                "shared_expert_gate": False,
                "expert_group_top_k": None,
                "expert_group_policy": None,
                "router_z_loss_coefficient": 0.0,
                "expert_capacity_factor": None,
                "expert_min_capacity": None,
                "overflow_policy": "none",
                "router_jitter_noise": 0.0,
            }
            for name, default in defaults.items():
                if moe.get(name) == default:
                    moe.pop(name, None)
        # architecture.7 adds semantics that are absent from the already-qualified
        # architecture models.  Omit their identity/default forms so those frozen
        # semantic fingerprints remain byte-for-byte stable.
        if not payload.get("local_attention_layers"):
            payload.pop("local_attention_layers", None)
        if payload.get("attention_qk_norm") is None:
            payload.pop("attention_qk_norm", None)
            payload.pop("attention_qk_norm_epsilon", None)
        if payload.get("embedding_dimension") == payload.get("hidden_size"):
            payload.pop("embedding_dimension", None)
        if payload.get("embedding_scale") == 1.0:
            payload.pop("embedding_scale", None)
        if payload.get("final_norm") is True:
            payload.pop("final_norm", None)
        return payload

    @property
    def semantic_fingerprint(self) -> str:
        encoded = json.dumps(
            self.semantic_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]

    def to_dict(self) -> dict[str, Any]:
        payload = self.semantic_dict()
        payload["diagnostics"] = dict(self.diagnostics)
        payload["semantic_fingerprint"] = self.semantic_fingerprint
        return payload

    @property
    def query_width(self) -> int:
        return self.attention_heads * self.head_dimension

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dimension


@dataclass(frozen=True)
class _TopologyHint:
    qkv_layout: str
    norm_kind: str
    mlp_kind: str
    gate_up_layout: str
    position_kind: str = "rope"
    lm_head_layout: str = "linear"


# These are thin Hugging Face config-dialect adapters. They supply facts absent
# from some serialized configs; no execution backend is selected from this map.
_HF_TOPOLOGY_HINTS: dict[str, _TopologyHint] = {
    "llama": _TopologyHint("separate", "rms_norm", "swiglu", "separate"),
    "mistral": _TopologyHint("separate", "rms_norm", "swiglu", "separate"),
    "qwen2": _TopologyHint("separate", "rms_norm", "swiglu", "separate"),
    "qwen3": _TopologyHint("separate", "rms_norm", "swiglu", "separate"),
    "gemma": _TopologyHint("separate", "rms_norm", "gated", "separate"),
    "gemma2": _TopologyHint("separate", "rms_norm", "gated", "separate"),
    "gemma3_text": _TopologyHint(
        "separate", "rms_norm", "gated", "separate"
    ),
    "smollm": _TopologyHint("separate", "rms_norm", "swiglu", "separate"),
    "gpt_neox": _TopologyHint(
        "fused_linear", "layer_norm", "ordinary", "none"
    ),
    "gpt2": _TopologyHint(
        "fused_conv1d", "layer_norm", "ordinary", "none",
        position_kind="learned_absolute", lm_head_layout="linear",
    ),
    "gpt_neo": _TopologyHint(
        "separate", "layer_norm", "ordinary", "none",
        position_kind="learned_absolute", lm_head_layout="linear",
    ),
    "opt": _TopologyHint(
        "separate", "layer_norm", "ordinary", "none",
        position_kind="learned_absolute", lm_head_layout="linear",
    ),
    "xglm": _TopologyHint(
        "separate", "layer_norm", "ordinary", "none",
        position_kind="sinusoidal_absolute", lm_head_layout="linear",
    ),
    "codegen": _TopologyHint(
        "fused_linear", "layer_norm", "ordinary", "none"
    ),
    "stablelm_alpha": _TopologyHint(
        "fused_linear", "layer_norm", "swiglu", "fused"
    ),
    "phi": _TopologyHint("separate", "layer_norm", "ordinary", "none"),
    "phi3": _TopologyHint("fused_linear", "rms_norm", "swiglu", "fused"),
    "falcon": _TopologyHint(
        "fused_linear", "layer_norm", "ordinary", "none"
    ),
    "granitemoe": _TopologyHint(
        "separate", "rms_norm", "swiglu", "fused"
    ),
    "olmoe": _TopologyHint(
        "separate", "rms_norm", "swiglu", "separate"
    ),
}


def _mapping(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return vars(config)


def _read(values: Mapping[str, Any], *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        value = values.get(name, _MISSING)
        if value is not _MISSING and value is not None:
            return value
    if default is _MISSING:
        raise ValueError(f"configuration is missing all of: {', '.join(names)}")
    return default


def _canonical_activation(value: Any) -> str:
    normalized = str(value).lower().replace("-", "_")
    aliases = {
        "swish": "silu",
        "gelu_fast": "gelu_new",
        "gelu_pytorch_tanh": "gelu_new",
        "gelu_approximate": "gelu_new",
    }
    return aliases.get(normalized, normalized)


def _canonical_scaling(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        return (("value", repr(value)),)
    return tuple(
        sorted((str(key), json.dumps(item, sort_keys=True)) for key, item in value.items())
    )


def _diagnostics(values: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    architectures = values.get("architectures") or ()
    return tuple(
        sorted(
            {
                "model_type": str(values.get("model_type", "unknown")),
                "architectures": ",".join(str(item) for item in architectures),
                "name_or_path": str(values.get("_name_or_path", "")),
            }.items()
        )
    )


def describe_config(
    config: Any,
    *,
    dtype: str | None = None,
) -> CanonicalArchitectureDescriptor:
    """Derive a canonical descriptor from a HF config or plain mapping."""

    values = _mapping(config)
    model_type = str(values.get("model_type", "")).lower()
    hint = _HF_TOPOLOGY_HINTS.get(model_type)

    if bool(values.get("is_encoder_decoder", False)):
        raise ValueError("encoder-decoder architectures are outside architecture")
    architectures = tuple(str(item) for item in values.get("architectures") or ())
    if any("ForConditionalGeneration" in name for name in architectures):
        raise ValueError("conditional-generation architectures are outside architecture")
    if model_type in {"bert", "roberta", "deberta", "vit", "t5", "bart"}:
        raise ValueError(f"{model_type} is not a dense decoder-only causal LM")

    hidden = int(_read(values, "hidden_size", "n_embd", "d_model"))
    layers = int(
        _read(values, "num_hidden_layers", "n_layer", "num_layers", "n_layers")
    )
    heads = int(
        _read(
            values,
            "num_attention_heads",
            "n_head",
            "num_heads",
            "attention_heads",
        )
    )
    kv_heads = int(
        _read(
            values,
            "num_key_value_heads", "num_kv_heads", "n_head_kv",
            default=(1 if values.get("multi_query") else heads),
        )
    )
    head_dim = int(_read(values, "head_dim", default=hidden // heads))
    if hidden % heads and "head_dim" not in values:
        raise ValueError("hidden size is not divisible by attention heads")
    intermediate = _read(
        values, "intermediate_size", "ffn_dim", "n_inner", default=None
    )
    if intermediate is None:
        expansion = float(values.get("ffn_ratio", 4.0))
        intermediate = int(hidden * expansion)
    intermediate = int(intermediate)
    vocabulary = int(_read(values, "vocab_size", "padded_vocab_size"))

    qkv_layout = str(
        _read(
            values, "qkv_projection_layout", "qkv_layout",
            default=(hint.qkv_layout if hint else "separate"),
        )
    )
    norm_kind = str(
        _read(
            values, "norm_kind", default=(
                "rms_norm" if values.get("rms_norm_eps") is not None
                else hint.norm_kind if hint else "layer_norm"
            )
        )
    )
    norm_epsilon = float(
        _read(
            values,
            "rms_norm_eps",
            "layer_norm_eps",
            "layer_norm_epsilon",
            "norm_eps",
            default=1e-5,
        )
    )
    explicit_pre_norm = values.get("do_layer_norm_before", _MISSING)
    placement = str(
        _read(
            values, "block_norm_placement", default=(
                "post_norm"
                if explicit_pre_norm is not _MISSING
                and explicit_pre_norm is not None
                and not bool(explicit_pre_norm)
                else "parallel_pre_norm"
                if bool(values.get("use_parallel_residual", False))
                else "sequential_pre_norm"
            )
        )
    )
    activation = _canonical_activation(
        _read(values, "hidden_act", "activation_function", "activation", default="gelu")
    )
    mlp_kind = str(
        _read(values, "mlp_kind", "mlp_type", default=(
            hint.mlp_kind if hint else (
                "swiglu" if activation == "silu" else "ordinary"
            )
        ))
    )
    gate_up_layout = str(
        _read(values, "gate_up_layout", default=(
            hint.gate_up_layout if hint else (
                "none" if mlp_kind == "ordinary" else "separate"
            )
        ))
    )

    partial = _read(
        values, "partial_rotary_factor", "rotary_pct", default=None
    )
    rotary_dimension = _read(values, "rotary_dim", default=None)
    if rotary_dimension is None:
        rotary_dimension = (
            int(round(head_dim * float(partial)))
            if partial is not None else head_dim
        )
    position_kind = str(
        _read(
            values, "position_embedding_kind", default=(
                hint.position_kind if hint else "rope"
            )
        )
    )
    rope_scaling = values.get("rope_scaling")
    if position_kind in {
        "learned_absolute", "sinusoidal_absolute", "alibi", "none",
    }:
        rope_kind = position_kind
        rotary_dimension = 0
        theta = None
    else:
        rope_kind = "scaled" if rope_scaling else "standard"
        theta = float(_read(values, "rope_theta", "rotary_emb_base", default=10000.0))

    attention_kind = (
        "mha" if kv_heads == heads else "mqa" if kv_heads == 1 else "gqa"
    )
    attention_bias = bool(
        _read(values, "attention_bias", "qkv_bias", "use_bias", default=False)
    )
    output_bias = bool(
        _read(values, "attention_output_bias", "o_proj_bias", default=attention_bias)
    )
    mlp_bias = bool(_read(values, "mlp_bias", "use_mlp_bias", default=False))
    window_enabled = values.get("use_sliding_window")
    window = _read(values, "sliding_window", "window_size", default=None)
    attention_window = (
        int(window)
        if window is not None and window_enabled is not False
        else None
    )
    attention_layers = values.get("attention_layers")
    if attention_layers is None:
        local_attention_layers = (
            tuple(range(layers)) if attention_window is not None else ()
        )
    else:
        flattened_attention_layers: list[str] = []
        for item in attention_layers:
            if isinstance(item, (list, tuple)):
                flattened_attention_layers.extend(str(value) for value in item)
            else:
                flattened_attention_layers.append(str(item))
        if len(flattened_attention_layers) != layers:
            raise ValueError(
                "attention layer pattern length does not match decoder layers"
            )
        local_attention_layers = tuple(
            index
            for index, item in enumerate(flattened_attention_layers)
            if item.lower() in {"local", "sliding", "sliding_window"}
        )
    embedding_dimension = int(values.get("word_embed_proj_dim") or hidden)
    embedding_scale = (
        math.sqrt(hidden) if bool(values.get("scale_embedding", False)) else 1.0
    )
    resolved_dtype = str(
        dtype
        or values.get("torch_dtype")
        or values.get("dtype")
        or "bfloat16"
    ).replace("torch.", "")

    if qkv_layout == "separate":
        attention_roles = ("q", "k", "v", "o")
    elif qkv_layout == "q_plus_fused_kv":
        attention_roles = ("q", "kv", "o")
    elif qkv_layout == "fused_qk_plus_v":
        attention_roles = ("qk", "v", "o")
    else:
        attention_roles = ("qkv", "o")
    if mlp_kind == "ordinary":
        mlp_roles = ("mlp_up", "mlp_down")
    elif gate_up_layout == "fused":
        mlp_roles = ("gate_up", "down")
    else:
        mlp_roles = ("gate", "up", "down")

    expert_count = int(
        _read(
            values,
            "num_local_experts",
            "num_experts",
            "n_routed_experts",
            default=0,
        )
    )
    moe = None
    if expert_count > 0:
        top_k = int(
            _read(
                values,
                "num_experts_per_tok",
                "num_experts_per_token",
                "top_k",
                "num_selected_experts",
                default=1,
            )
        )
        shared_experts = int(
            _read(
                values,
                "num_shared_experts",
                "shared_expert_count",
                default=0,
            )
        )
        group_count_value = _read(
            values, "n_group", "num_expert_groups", default=None
        )
        group_top_k_value = _read(
            values,
            "topk_group",
            "num_expert_groups_per_token",
            "expert_group_top_k",
            default=None,
        )
        explicit_pattern = values.get("moe_layer_indices")
        if explicit_pattern is not None:
            moe_indices = tuple(int(index) for index in explicit_pattern)
        else:
            frequency = int(
                _read(
                    values,
                    "moe_layer_frequency",
                    "decoder_sparse_step",
                    default=1,
                )
            )
            offset = int(values.get("moe_layer_offset", 0))
            moe_indices = tuple(
                index for index in range(layers)
                if index >= offset and (index - offset) % frequency == 0
            )
        dense_indices = tuple(
            index for index in range(layers) if index not in set(moe_indices)
        )
        auxiliary_coefficient = float(
            _read(
                values,
                "router_aux_loss_coef",
                "router_aux_loss_coefficient",
                default=0.0,
            )
        )
        expert_kind = "swiglu" if activation == "silu" else (
            "ordinary" if mlp_kind == "ordinary" else mlp_kind
        )
        expert_roles = (
            ("expert_up", "expert_down")
            if expert_kind == "ordinary"
            else ("expert_gate_up", "expert_down")
        )
        normalization = values.get("router_normalization")
        if normalization is None:
            scoring = str(values.get("scoring_func", "softmax")).lower()
            normalized_topk = _read(
                values,
                "norm_topk_prob",
                "normalize_router_probabilities",
                default=None,
            )
            if scoring == "sigmoid":
                normalization = (
                    "sigmoid_selected"
                    if normalized_topk is not False
                    else "sigmoid_selected_unnormalized"
                )
            elif normalized_topk is True:
                # Qwen-style routers softmax over all experts first, select,
                # and then renormalize the selected probabilities.
                normalization = "softmax_all_renormalized_selected"
            elif normalized_topk is False:
                normalization = "softmax_all_selected"
            else:
                # Historical fused Granite-style router contract.
                normalization = "softmax_selected"
        shared_intermediate = _read(
            values,
            "shared_expert_intermediate_size",
            "shared_moe_intermediate_size",
            default=(intermediate if shared_experts else None),
        )
        capacity_factor = _read(
            values,
            "expert_capacity_factor",
            "capacity_factor",
            default=None,
        )
        minimum_capacity = _read(
            values,
            "expert_min_capacity",
            "min_capacity",
            default=None,
        )
        overflow_policy = str(
            _read(values, "expert_overflow_policy", default=(
                "drop" if bool(values.get("drop_tokens", False)) else "none"
            ))
        )
        moe = MoEDescriptor(
            expert_count=expert_count,
            top_k=top_k,
            router_type=str(values.get("router_type", "linear_topk")),
            router_logits_dtype=str(
                values.get("router_logits_dtype", "float32")
            ).replace("torch.", ""),
            router_normalization=str(normalization),
            expert_mlp_kind=expert_kind,
            expert_intermediate_size=int(
                _read(
                    values,
                    "moe_intermediate_size",
                    "expert_intermediate_size",
                    default=intermediate,
                )
            ),
            shared_expert_count=shared_experts,
            expert_group_count=(
                int(group_count_value) if group_count_value is not None else None
            ),
            expert_bias=bool(values.get("expert_bias", False)),
            auxiliary_loss_kind=(
                "load_balancing" if auxiliary_coefficient != 0.0 else "none"
            ),
            auxiliary_loss_coefficient=auxiliary_coefficient,
            expert_lora_roles=expert_roles,
            moe_layer_indices=moe_indices,
            dense_layer_indices=dense_indices,
            expert_storage_layout=str(
                values.get("expert_storage_layout", "fused")
            ),
            shared_expert_storage_layout=(
                str(values.get("shared_expert_storage_layout", "separate"))
                if shared_experts else None
            ),
            shared_expert_mlp_kind=(expert_kind if shared_experts else None),
            shared_expert_intermediate_size=(
                int(shared_intermediate)
                if shared_intermediate is not None else None
            ),
            shared_expert_gate=bool(
                values.get("shared_expert_gate", False)
                or values.get("shared_expert_gate_enabled", False)
            ),
            expert_group_top_k=(
                int(group_top_k_value)
                if group_top_k_value is not None else None
            ),
            expert_group_policy=(
                str(values.get("expert_group_policy", "group_limited_topk"))
                if group_count_value is not None else None
            ),
            router_z_loss_coefficient=float(
                _read(
                    values,
                    "router_z_loss_coef",
                    "router_z_loss_coefficient",
                    default=0.0,
                )
            ),
            expert_capacity_factor=(
                float(capacity_factor) if capacity_factor is not None else None
            ),
            expert_min_capacity=(
                int(minimum_capacity)
                if minimum_capacity is not None else None
            ),
            overflow_policy=overflow_policy,
            router_jitter_noise=float(
                _read(
                    values,
                    "router_jitter_noise",
                    "router_jitter_noise_coef",
                    default=0.0,
                )
            ),
        )

    descriptor = CanonicalArchitectureDescriptor(
        hidden_size=hidden,
        intermediate_size=intermediate,
        layer_count=layers,
        vocabulary_size=vocabulary,
        attention_heads=heads,
        kv_heads=kv_heads,
        head_dimension=head_dim,
        attention_kind=attention_kind,
        qkv_projection_layout=qkv_layout,
        norm_kind=norm_kind,
        norm_epsilon=norm_epsilon,
        block_norm_placement=placement,
        mlp_kind=mlp_kind,
        mlp_activation=activation,
        gate_up_layout=gate_up_layout,
        rope=RopeDescriptor(
            kind=rope_kind,
            theta=theta,
            rotary_dimension=int(rotary_dimension),
            scaling=_canonical_scaling(rope_scaling),
        ),
        tied_embeddings=bool(values.get("tie_word_embeddings", False)),
        lm_head_layout=str(
            _read(
                values, "lm_head_layout",
                default=(hint.lm_head_layout if hint else "linear"),
            )
        ),
        final_norm=True,
        attention_window=attention_window,
        local_attention_layers=local_attention_layers,
        attention_qk_norm=None,
        attention_qk_norm_epsilon=None,
        embedding_dimension=embedding_dimension,
        embedding_scale=embedding_scale,
        dtype=resolved_dtype,
        biases=ProjectionBiases(
            query=attention_bias,
            key=attention_bias,
            value=attention_bias,
            output=output_bias,
            mlp_input=mlp_bias,
            mlp_output=mlp_bias,
        ),
        lora_roles=attention_roles + mlp_roles,
        moe=moe,
        diagnostics=_diagnostics(values),
    )
    descriptor.validate()
    return descriptor


def with_observed_topology(
    descriptor: CanonicalArchitectureDescriptor,
    *,
    qkv_projection_layout: str,
    norm_kind: str,
    mlp_kind: str,
    gate_up_layout: str,
    biases: ProjectionBiases,
    lora_roles: tuple[str, ...],
    tied_embeddings: bool,
    lm_head_layout: str,
    final_norm: bool = True,
    intermediate_size: int | None = None,
    block_norm_placement: str | None = None,
    rope: RopeDescriptor | None = None,
    attention_qk_norm: str | None = None,
    attention_qk_norm_epsilon: float | None = None,
    embedding_dimension: int | None = None,
    moe: MoEDescriptor | None = None,
) -> CanonicalArchitectureDescriptor:
    """Replace config hints with facts observed from an instantiated model."""

    updated = replace(
        descriptor,
        qkv_projection_layout=qkv_projection_layout,
        norm_kind=norm_kind,
        mlp_kind=mlp_kind,
        gate_up_layout=gate_up_layout,
        biases=biases,
        lora_roles=lora_roles,
        tied_embeddings=tied_embeddings,
        lm_head_layout=lm_head_layout,
        final_norm=final_norm,
        intermediate_size=(
            descriptor.intermediate_size
            if intermediate_size is None else intermediate_size
        ),
        block_norm_placement=(
            descriptor.block_norm_placement
            if block_norm_placement is None else block_norm_placement
        ),
        rope=descriptor.rope if rope is None else rope,
        attention_qk_norm=attention_qk_norm,
        attention_qk_norm_epsilon=attention_qk_norm_epsilon,
        embedding_dimension=(
            descriptor.embedding_dimension
            if embedding_dimension is None else embedding_dimension
        ),
        moe=moe if moe is not None else descriptor.moe,
    )
    updated.validate()
    return updated


__all__ = [
    "CanonicalArchitectureDescriptor",
    "ProjectionBiases",
    "MoEDescriptor",
    "RopeDescriptor",
    "describe_config",
    "with_observed_topology",
]
