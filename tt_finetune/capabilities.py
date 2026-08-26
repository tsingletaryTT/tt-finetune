"""Public, compilation-free Blackhole capability and admission query."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class SupportResult:
    supported: bool
    model: str
    architecture_recognized: bool
    canonical_semantics: dict[str, Any] | None
    lora_targets: tuple[str, ...]
    rank: int
    batch_size: int
    sequence_length: int
    precision: str
    adapter_type: str
    attention_strategy: str | None
    execution_path: str | None
    memory_admitted: bool
    estimated_device_bytes: int | None
    estimated_device_gib: float | None
    reason: str
    suggested_geometry: dict[str, int] | None
    evidence: str = "statically_predicted"
    memory_mode_requested: str = "bf16"
    memory_mode_selected: str = "bf16"
    memory_breakdown: dict[str, Any] | None = None
    use_rslora: bool = False
    use_dora: bool = False
    lora_plus_lr_ratio: float | None = None
    layers_to_transform: tuple[int, ...] | None = None
    ia3_feedforward_modules: tuple[str, ...] | None = None
    automatic_plan: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **vars(self),
            "lora_targets": list(self.lora_targets),
            "silicon_qualified": self.evidence == "silicon_qualified",
        }

    @property
    def silicon_qualified(self) -> bool:
        return self.evidence == "silicon_qualified"

    def __bool__(self) -> bool:
        return self.supported


def _model_name(model: Any) -> str:
    if isinstance(model, (str, Path)):
        return str(model)
    value = getattr(model, "name_or_path", None)
    if value:
        return str(value)
    raise TypeError("model must be a Hugging Face ID/path or BlackholeModel")


def _resolve_targets(model, requested: str | Sequence[str], weight_map=None):
    import torch.nn as nn
    from transformers.pytorch_utils import Conv1D

    linear = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, Conv1D))
    }
    output = model.get_output_embeddings()
    eligible = {
        name: module
        for name, module in linear.items()
        if module is not output and name.rsplit(".", 1)[-1] != "lm_head"
    }
    if weight_map is not None:
        from tt_finetune.targeting import resolve_canonical_role_targets

        canonical = resolve_canonical_role_targets(
            requested,
            weight_map=weight_map,
            eligible_modules=eligible,
        )
        if canonical is not None:
            return {path: eligible[path] for path in canonical.paths}
    if isinstance(requested, str):
        requested = [requested]
    requested = list(requested)
    policy = requested[0] if len(requested) == 1 else None
    if policy in {"all-linear", "auto_all_linear", "auto-all-linear"}:
        return eligible
    attention = {
        name: module
        for name, module in eligible.items()
        if any(
            marker in part.lower()
            for part in name.split(".")
            for marker in ("attn", "attention")
        )
    }
    if policy in {
        "auto", "auto_qv", "auto-qv", "auto_attention", "auto-attention"
    }:
        leaves = {name.rsplit(".", 1)[-1] for name in attention}
        if policy in {"auto_attention", "auto-attention"}:
            selected_leaves = leaves
        else:
            selected_leaves = next(
                (
                    set(pair)
                    for pair in (("q_proj", "v_proj"), ("query", "value"), ("q", "v"))
                    if set(pair) <= leaves
                ),
                None,
            )
            if selected_leaves is None:
                fused = leaves & {
                    "query_key_value", "qkv", "qkv_proj", "wqkv", "c_attn"
                }
                selected_leaves = fused if len(fused) == 1 else set()
        selected = {
            name: module
            for name, module in attention.items()
            if name.rsplit(".", 1)[-1] in selected_leaves
        }
    else:
        selected = {
            name: module
            for name, module in eligible.items()
            if name in requested or name.rsplit(".", 1)[-1] in requested
        }
    if not selected:
        available = sorted({name.rsplit(".", 1)[-1] for name in eligible})
        raise ValueError(
            f"LoRA targets {requested!r} matched no supported projection; "
            f"available targets include {available}"
        )
    return selected


def query_support(
    *,
    model: Any,
    rank: int = 8,
    batch_size: int = 1,
    sequence_length: int = 128,
    target_modules: str | Sequence[str] = "auto",
    adapter_type: str = "lora",
    feedforward_modules: str | Sequence[str] = "ia3_feedforward",
    use_rslora: bool = False,
    use_dora: bool = False,
    lora_dropout: float = 0.0,
    lora_bias: str = "none",
    lora_plus_lr_ratio: float | None = None,
    layers_to_transform: Sequence[int] | None = None,
    precision: str = "bfloat16",
    revision: str | None = None,
    trust_remote_code: bool = False,
    memory_mode: str = "auto",
) -> SupportResult:
    """Predict support and memory admission without touching TT silicon."""

    name = _model_name(model)
    normalized_precision = precision.lower().replace("torch.", "")
    method = adapter_type.lower().replace("-", "_")
    if method == "rslora":
        use_rslora = True
    elif method == "dora":
        use_dora = True
    elif method == "lora_plus" and lora_plus_lr_ratio is None:
        return SupportResult(
            False,
            model=name,
            rank=rank,
            batch_size=batch_size,
            sequence_length=sequence_length,
            precision=normalized_precision,
            adapter_type=method,
            architecture_recognized=False,
            canonical_semantics=None,
            lora_targets=(),
            attention_strategy=None,
            execution_path=None,
            memory_admitted=False,
            estimated_device_bytes=None,
            estimated_device_gib=None,
            reason="adapter_type='lora_plus' requires lora_plus_lr_ratio",
            suggested_geometry=None,
            memory_mode_requested=memory_mode,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_plus_lr_ratio=lora_plus_lr_ratio,
            layers_to_transform=(
                tuple(layers_to_transform)
                if layers_to_transform is not None else None
            ),
        )
    base = dict(
        model=name,
        rank=rank,
        batch_size=batch_size,
        sequence_length=sequence_length,
        precision=normalized_precision,
        adapter_type=method,
        memory_mode_requested=memory_mode,
        use_rslora=use_rslora,
        use_dora=use_dora,
        lora_plus_lr_ratio=lora_plus_lr_ratio,
        layers_to_transform=(
            tuple(layers_to_transform)
            if layers_to_transform is not None else None
        ),
        ia3_feedforward_modules=None,
    )
    if rank <= 0 or batch_size <= 0 or sequence_length <= 0:
        return SupportResult(
            False, architecture_recognized=False, canonical_semantics=None,
            lora_targets=(), attention_strategy=None, execution_path=None,
            memory_admitted=False, estimated_device_bytes=None,
            estimated_device_gib=None,
            reason=(
                "rank, batch_size and sequence_length must all be positive; "
                f"got R{rank}/B{batch_size}/S{sequence_length}"
            ),
            suggested_geometry=None, **base,
        )
    if method in {
        "prefix", "prefix_tuning", "prompt", "prompt_tuning", "p_tuning"
    }:
        return SupportResult(
            False, architecture_recognized=False, canonical_semantics=None,
            lora_targets=(), attention_strategy=None, execution_path=None,
            memory_admitted=False, estimated_device_bytes=None,
            estimated_device_gib=None,
            reason=(
                f"{adapter_type} is not supported by the current execution "
                "ABI because it changes the sequence/attention-state ABI "
                "(virtual tokens or prefix key/value state). The current "
                "single-P150a engine owns a fixed real-token causal-attention "
                "contract, so this requires a new architecture execution "
                "semantic rather than a projection adapter wrapper."
            ),
            suggested_geometry=None, **base,
        )
    if method not in {"lora", "rslora", "dora", "lora_plus", "ia3"}:
        return SupportResult(
            False, architecture_recognized=False, canonical_semantics=None,
            lora_targets=(), attention_strategy=None, execution_path=None,
            memory_admitted=False, estimated_device_bytes=None,
            estimated_device_gib=None,
            reason=f"adapter_type={adapter_type!r} is not supported",
            suggested_geometry=None, **base,
        )
    if lora_plus_lr_ratio is not None and (
        lora_plus_lr_ratio <= 0 or lora_plus_lr_ratio == 1
    ):
        return SupportResult(
            False, architecture_recognized=False, canonical_semantics=None,
            lora_targets=(), attention_strategy=None, execution_path=None,
            memory_admitted=False, estimated_device_bytes=None,
            estimated_device_gib=None,
            reason=(
                "lora_plus_lr_ratio must be positive and different from 1"
            ),
            suggested_geometry=None, **base,
        )
    if normalized_precision not in {"bfloat16", "bf16", "float32", "fp32"}:
        return SupportResult(
            False, architecture_recognized=False, canonical_semantics=None,
            lora_targets=(), attention_strategy=None, execution_path=None,
            memory_admitted=False, estimated_device_bytes=None,
            estimated_device_gib=None,
            reason=f"precision={precision!r} is not supported by the BF16 trainer",
            suggested_geometry=None, **base,
        )
    architecture_recognized = False
    canonical_semantics = None
    resolved_targets: tuple[str, ...] = ()
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from tt_finetune.architecture import (
            describe_model,
            grouped_lora_target_geometry,
        )
        from tt_finetune.runtime import (
            DataShapeProfile,
            PlannerRequest,
            HardwareCapabilities,
            LoRATargetGeometry,
            estimate_memory,
            plan_autotuned_execution,
            plan_execution,
        )

        config = AutoConfig.from_pretrained(
            name,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        with torch.device("meta"):
            meta_model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=trust_remote_code,
            )
        descriptor, weight_map = describe_model(meta_model)
        descriptor = replace(
            descriptor,
            dtype=(
                "bfloat16"
                if normalized_precision in {"bfloat16", "bf16"}
                else "float32"
            ),
        )
        architecture_recognized = True
        canonical_semantics = descriptor.to_dict()
        selected = _resolve_targets(meta_model, target_modules, weight_map)
        selected_feedforward: dict[str, Any] = {}
        if method == "ia3":
            selected_feedforward = _resolve_targets(
                meta_model, feedforward_modules, weight_map
            )
        if layers_to_transform is not None:
            requested_layers = tuple(layers_to_transform)
            invalid_layers = sorted(
                index
                for index in requested_layers
                if not isinstance(index, int)
                or index < 0
                or index >= descriptor.layer_count
            )
            if invalid_layers:
                raise ValueError(
                    "requested adapter layers are outside the canonical "
                    f"range 0..{descriptor.layer_count - 1}: {invalid_layers}"
                )
            layer_by_path = {
                item.path: item.layer_index for item in weight_map.lora_targets
            }
            selected = {
                path: module
                for path, module in selected.items()
                if layer_by_path.get(path) in set(requested_layers)
            }
            if not selected:
                raise ValueError(
                    "the requested target roles do not exist in the selected "
                    f"layers {sorted(requested_layers)}"
                )
            if selected_feedforward:
                selected_feedforward = {
                    path: module
                    for path, module in selected_feedforward.items()
                    if layer_by_path.get(path) in set(requested_layers)
                }
        if method == "ia3":
            missing_feedforward = sorted(
                set(selected_feedforward) - set(selected)
            )
            if missing_feedforward:
                raise ValueError(
                    "IA3 feedforward modules must be a subset of targets: "
                    f"{missing_feedforward}"
                )
            if not selected_feedforward:
                raise ValueError(
                    "IA3 requires a canonical feedforward target"
                )
        resolved_targets = tuple(sorted(selected))
        canonical = {item.path for item in weight_map.lora_targets}
        noncanonical = sorted(set(selected) - canonical)
        if noncanonical:
            raise ValueError(
                "selected projections lack canonical training semantics: "
                f"{noncanonical}"
            )
        geometry = grouped_lora_target_geometry(
            weight_map,
            selected_paths=set(selected),
        )
        targets = tuple(
            LoRATargetGeometry(role, input_size, output_size, instances)
            for role, input_size, output_size, instances in geometry
        )
        plan = plan_execution(
            descriptor,
            rank=(rank if method != "ia3" else 1),
            batch_size=batch_size,
            sequence_length=sequence_length,
            lora_targets=targets,
        )
        adapter_parameter_count = plan.lora_parameter_count
        if method == "ia3":
            from peft import IA3Config as PeftIA3Config, get_peft_model

            topology = get_peft_model(
                meta_model,
                PeftIA3Config(
                    target_modules=list(selected),
                    feedforward_modules=list(selected_feedforward),
                    task_type="CAUSAL_LM",
                    init_ia3_weights=False,
                ),
            )
            adapter_parameter_count = sum(
                parameter.numel()
                for parameter in topology.parameters()
                if parameter.requires_grad
            )
        memory = estimate_memory(
            base_parameter_count=plan.estimated_base_parameter_count,
            lora_parameter_count=adapter_parameter_count,
            logical_tokens=plan.logical_tokens,
            hidden_size=descriptor.hidden_size,
            layer_count=descriptor.layer_count,
            requested_mode=memory_mode,
        )
        try:
            from .runtime.memory_planner import (
                P150A_ALLOCATABLE_DRAM_BYTES,
            )

            execution_profile = plan_autotuned_execution(PlannerRequest(
                architecture=descriptor,
                safe_plan=plan,
                # architecture intentionally retained its older conservative
                # 24-GiB planning default. low-memory measured the actual P150a
                # allocatable capacity and admits guarded 7B reduced-weight
                # plans against it; autotuner must consume that same physical
                # capacity or it silently drops every valid 7B candidate.
                hardware=HardwareCapabilities(
                    dram_bytes=P150A_ALLOCATABLE_DRAM_BYTES,
                ),
                data=DataShapeProfile.fixed(sequence_length),
                memory_mode=memory.selected_mode,
                adapter_contract=tuple(sorted({
                    "method": method,
                    "use_rslora": bool(use_rslora),
                    "use_dora": bool(use_dora),
                    "dropout": float(lora_dropout),
                    "bias": str(lora_bias),
                }.items())),
                estimated_device_bytes=memory.estimated_peak_bytes,
            ))
        except ValueError:
            # The accepted safe planner remains authoritative if no optimized
            # autotuner candidate survives analytical pruning.
            execution_profile = None
        attention = plan.selection("attention").implementation
        suggestion = None
        if plan.capabilities.supported and not memory.admitted:
            candidate = sequence_length
            while candidate > 32:
                candidate = max(32, (candidate // 2 // 32) * 32)
                smaller = plan_execution(
                    descriptor,
                    rank=(rank if method != "ia3" else 1),
                    batch_size=batch_size,
                    sequence_length=candidate,
                    lora_targets=targets,
                )
                smaller_memory = estimate_memory(
                    base_parameter_count=smaller.estimated_base_parameter_count,
                    lora_parameter_count=adapter_parameter_count,
                    logical_tokens=smaller.logical_tokens,
                    hidden_size=descriptor.hidden_size,
                    layer_count=descriptor.layer_count,
                    requested_mode=memory_mode,
                )
                if smaller_memory.admitted:
                    suggestion = {
                        "rank": rank,
                        "batch_size": batch_size,
                        "sequence_length": candidate,
                    }
                    break
        return SupportResult(
            supported=bool(plan.capabilities.supported and memory.admitted),
            architecture_recognized=True,
            canonical_semantics=canonical_semantics,
            lora_targets=resolved_targets,
            attention_strategy=attention,
            execution_path=(
                "generic_peft_ia3"
                if method == "ia3"
                else plan.selected_stack_backend
            ),
            memory_admitted=memory.admitted,
            estimated_device_bytes=memory.estimated_peak_bytes,
            estimated_device_gib=memory.estimated_peak_bytes / 1024**3,
            reason=(
                memory.selection_reason
                if plan.capabilities.supported
                else plan.admission_reason
            ),
            suggested_geometry=suggestion,
            memory_mode_selected=memory.selected_mode,
            memory_breakdown=memory.to_dict(),
            automatic_plan=(
                execution_profile.to_dict()
                if execution_profile is not None else None
            ),
            **{
                **base,
                "ia3_feedforward_modules": (
                    tuple(sorted(selected_feedforward))
                    if method == "ia3" else None
                ),
            },
        )
    except Exception as error:
        return SupportResult(
            False, architecture_recognized=architecture_recognized,
            canonical_semantics=canonical_semantics,
            lora_targets=resolved_targets,
            attention_strategy=None, execution_path=None,
            memory_admitted=False, estimated_device_bytes=None,
            estimated_device_gib=None,
            reason=f"{type(error).__name__}: {error}",
            suggested_geometry=None, **base,
        )
