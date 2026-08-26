"""Apply an architecture plan to an instantiated PEFT model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from tt_finetune.architecture import (
    CanonicalArchitectureDescriptor,
    CanonicalModelMap,
    describe_model,
    grouped_expert_lora_target_geometry,
    grouped_lora_target_geometry,
)
from tt_finetune.runtime import (
    ExecutionPlan,
    LoRATargetGeometry,
    plan_execution,
)

from .streamed_attention import (
    StreamedAttentionInstallReport,
    install_streamed_attention,
)
from .stable_layer_norm import install_stable_layer_norm
from .sparse_moe import SparseMoEInstallReport, install_sparse_moe


@dataclass(frozen=True)
class ArchitectureInstallReport:
    architecture: CanonicalArchitectureDescriptor
    weight_map: CanonicalModelMap
    plan: ExecutionPlan
    selected_lora_paths: tuple[str, ...]
    selected_expert_lora_paths: tuple[str, ...]
    attention_install: StreamedAttentionInstallReport | None
    moe_install: SparseMoEInstallReport | None
    executor_backend: str
    executor_fallback_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture.to_dict(),
            "weight_map": self.weight_map.to_dict(),
            "plan": self.plan.to_dict(),
            "selected_lora_paths": list(self.selected_lora_paths),
            "selected_expert_lora_paths": list(self.selected_expert_lora_paths),
            "attention_install": (
                vars(self.attention_install)
                if self.attention_install is not None else None
            ),
            "moe_install": (
                vars(self.moe_install) if self.moe_install is not None else None
            ),
            "executor_backend": self.executor_backend,
            "executor_fallback_reason": self.executor_fallback_reason,
        }


def _has_lora_adapter(module: nn.Module) -> bool:
    return bool(
        hasattr(module, "lora_A")
        or hasattr(module, "lora_embedding_A")
        or type(module).__name__.lower().startswith("lora")
    )


def plan_and_install_architecture(
    model: nn.Module,
    *,
    rank: int,
    alpha: int | None = None,
    batch_size: int,
    sequence_length: int,
    requested_attention_backend: str = "auto",
    requested_moe_backend: str = "auto",
    attention_query_chunk_tokens: int = 128,
) -> ArchitectureInstallReport:
    """Discover, plan and install only architecture-qualified integrations."""

    if requested_attention_backend not in {"auto", "safe", "native"}:
        raise ValueError("attention backend must be auto, safe or native")
    if requested_moe_backend not in {"auto", "safe", "sparse", "dynamic"}:
        raise ValueError("MoE backend must be auto, safe, sparse or dynamic")
    descriptor, weight_map = describe_model(model)
    first_parameter = next(model.parameters(), None)
    observed_dtype = (
        str(first_parameter.dtype).replace("torch.", "")
        if first_parameter is not None else descriptor.dtype
    )
    # Preserve every previously accepted semantic fingerprint.  The one new
    # dialect that advertises FP32 checkpoint weights but is explicitly loaded
    # as BF16 must nevertheless be planned for its real execution footprint.
    if descriptor.dtype == "float32" and observed_dtype != "float32":
        descriptor, weight_map = describe_model(model, dtype=observed_dtype)
    adapted = tuple(
        item.path
        for item in weight_map.lora_targets
        if _has_lora_adapter(model.get_submodule(item.path))
    )
    selected_paths = adapted or tuple(item.path for item in weight_map.lora_targets)
    grouped = grouped_lora_target_geometry(
        weight_map, selected_paths=selected_paths
    )
    expert_grouped = grouped_expert_lora_target_geometry(weight_map)
    plan = plan_execution(
        descriptor,
        rank=rank,
        batch_size=batch_size,
        sequence_length=sequence_length,
        lora_targets=tuple(
            LoRATargetGeometry(role, input_features, output_features, instances)
            for role, input_features, output_features, instances
            in grouped + expert_grouped
        ),
    )
    plan.require_admitted()

    attention = plan.selection("attention")
    normalization = plan.selection("normalization")
    attention_install = None
    moe_install = None
    if normalization.implementation == "safe_explicit_layer_norm":
        install_stable_layer_norm(model)
    if requested_attention_backend == "native" and attention.implementation != "native_streamed_causal":
        raise ValueError(
            "native streamed attention was requested outside its qualified "
            f"capability: {attention.reason}"
        )
    if sequence_length > 128 and descriptor.qkv_projection_layout == "separate":
        use_native = (
            attention.implementation == "native_streamed_causal"
            and requested_attention_backend != "safe"
        )
        attention_install = install_streamed_attention(
            model,
            native_backward_experimental=use_native,
            query_chunk_tokens=attention_query_chunk_tokens,
        )
    if descriptor.moe is not None:
        moe_selection = plan.selection("moe")
        selected_moe_backend = (
            "dynamic"
            if requested_moe_backend == "auto"
            and plan.moe_execution_schedule
            == "dynamic_native_expert_bucketed"
            else
            "sparse"
            if requested_moe_backend == "auto"
            and moe_selection.implementation == "device_sparse_selected_experts"
            else "safe"
            if requested_moe_backend == "auto"
            else requested_moe_backend
        )
        if selected_moe_backend == "dynamic" and (
            descriptor.moe.expert_capacity_factor is not None
        ):
            raise ValueError(
                "dynamic no-capacity expert scheduling cannot replace an "
                "explicit capacity/drop routing contract"
            )
        if (
            selected_moe_backend == "sparse"
            and moe_selection.implementation != "device_sparse_selected_experts"
        ):
            raise ValueError(
                "sparse MoE was requested outside its qualified capability: "
                f"{moe_selection.reason}"
            )
        moe_install = install_sparse_moe(
            model,
            weight_map,
            architecture=descriptor,
            rank=rank,
            alpha=alpha if alpha is not None else 2 * rank,
            backend=selected_moe_backend,
        )

    # This is the emitted executor plan, not a retrospective guess about what
    # the worker ran.  The public worker must either construct this resident
    # ABI or fail the job; product-semantic exclusions are applied explicitly
    # before model transfer and replace this field with the safe backend plus
    # a concrete reason.  Silent post-plan fallback is forbidden.
    if plan.selected_stack_backend == "resident_exact":
        executor_backend = "resident_exact"
        executor_fallback_reason = None
    else:
        executor_backend = plan.selected_stack_backend
        executor_fallback_reason = None
    report = ArchitectureInstallReport(
        architecture=descriptor,
        weight_map=weight_map,
        plan=plan,
        selected_lora_paths=selected_paths,
        selected_expert_lora_paths=tuple(
            item.path for item in weight_map.expert_lora_targets
        ),
        attention_install=attention_install,
        moe_install=moe_install,
        executor_backend=executor_backend,
        executor_fallback_reason=executor_fallback_reason,
    )
    model._tt_architecture_report = report
    return report


__all__ = ["ArchitectureInstallReport", "plan_and_install_architecture"]
