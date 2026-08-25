"""Capacity-driven single-P150a memory planning for public admission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


P150A_ALLOCATABLE_DRAM_BYTES = 34_217_131_008
P150A_GUARDED_DRAM_BYTES = int(P150A_ALLOCATABLE_DRAM_BYTES * 0.88)
FROZEN_WEIGHT_EXECUTABLE_COPY_FACTOR = 2.0


@dataclass(frozen=True)
class MemoryEstimate:
    requested_mode: str
    selected_mode: str
    frozen_weight_dtype: str
    frozen_weights_bytes_per_copy: int
    frozen_weights_cached_bytes: int
    trainable_parameter_bytes: int
    gradient_bytes: int
    optimizer_moment_bytes: int
    activation_workspace_bytes: int
    runtime_reserve_bytes: int
    estimated_peak_bytes: int
    allocatable_dram_bytes: int
    guarded_dram_bytes: int
    admitted: bool
    selection_reason: str
    retained_bf16_source_bytes: int = 0
    bf4_fraction_of_reduced_weights: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


def _weight_ratio(dtype: str) -> float:
    # One 32x32 tile: BF16=2048 B, BF8=1088 B, BF4=576 B.
    return {"bf16": 1.0, "bfp_bf8": 1088 / 2048, "bfp_bf4": 576 / 2048}[dtype]


def estimate_memory(
    *,
    base_parameter_count: int,
    lora_parameter_count: int,
    logical_tokens: int,
    hidden_size: int,
    layer_count: int,
    requested_mode: str = "auto",
) -> MemoryEstimate:
    """Estimate peak by semantic geometry, never checkpoint identity."""

    if requested_mode not in {"auto", "bf16", "bf8_frozen", "bf4_frozen"}:
        raise ValueError(f"unsupported memory_mode: {requested_mode}")
    trainable = lora_parameter_count * 2
    gradients = lora_parameter_count * 2
    moments = lora_parameter_count * 4
    # Six checkpoint-equivalent hidden streams per layer is conservative for
    # the generic path; resident/streamed schedules reuse a subset.
    activations = logical_tokens * hidden_size * 2 * layer_count * 6
    runtime_reserve = 768 * 1024**2

    def values(dtype: str) -> tuple[int, int]:
        per_copy = int(base_parameter_count * 2 * _weight_ratio(dtype))
        cached = int(per_copy * FROZEN_WEIGHT_EXECUTABLE_COPY_FACTOR)
        retained_source = (
            base_parameter_count * 2 if dtype != "bf16" else 0
        )
        total = (
            cached
            + retained_source
            + trainable
            + gradients
            + moments
            + activations
            + runtime_reserve
        )
        return per_copy, total

    candidates = {
        dtype: values(dtype) for dtype in ("bf16", "bfp_bf8", "bfp_bf4")
    }
    if requested_mode == "bf16":
        selected = "bf16"
        reason = "explicit BF16 memory contract"
    elif requested_mode == "bf8_frozen":
        selected = "bfp_bf8"
        reason = "explicit native BF8 frozen-weight contract"
    elif requested_mode == "bf4_frozen":
        selected = "bfp_bf4"
        reason = "explicit native BF4 frozen-weight contract"
    elif candidates["bf16"][1] <= P150A_GUARDED_DRAM_BYTES:
        selected = "bf16"
        reason = "BF16 estimate fits the guarded single-P150a budget"
    elif candidates["bfp_bf8"][1] <= P150A_GUARDED_DRAM_BYTES:
        selected = "bfp_bf8"
        reason = "BF16 exceeds the guarded budget and native BF8 fits"
    elif candidates["bfp_bf4"][1] <= P150A_GUARDED_DRAM_BYTES:
        selected = "mixed_bfp_bf8_bf4"
        reason = (
            "BF16 and full BF8 exceed the guarded budget; the planner will "
            "lower the largest frozen matmul tensors from BF8 to BF4 only "
            "until the guarded single-P150a budget fits"
        )
    else:
        selected = "bfp_bf4"
        reason = "even native BF4 exceeds the guarded single-P150a budget"
    if selected == "mixed_bfp_bf8_bf4":
        bf8_per_copy, bf8_total = candidates["bfp_bf8"]
        bf4_per_copy, bf4_total = candidates["bfp_bf4"]
        required = bf8_total - P150A_GUARDED_DRAM_BYTES
        available = bf8_total - bf4_total
        bf4_fraction = min(1.0, max(0.0, required / available))
        per_copy = int(
            bf8_per_copy
            - (bf8_per_copy - bf4_per_copy) * bf4_fraction
        )
        total = P150A_GUARDED_DRAM_BYTES
    else:
        per_copy, total = candidates[selected]
        bf4_fraction = 1.0 if selected == "bfp_bf4" else 0.0
    selected_mode = {
        "bf16": "bf16",
        "bfp_bf8": "bf8_frozen",
        "bfp_bf4": "bf4_frozen",
        "mixed_bfp_bf8_bf4": "hybrid_frozen",
    }[selected]
    return MemoryEstimate(
        requested_mode=requested_mode,
        selected_mode=selected_mode,
        frozen_weight_dtype=selected,
        frozen_weights_bytes_per_copy=per_copy,
        frozen_weights_cached_bytes=int(
            per_copy * FROZEN_WEIGHT_EXECUTABLE_COPY_FACTOR
        ),
        trainable_parameter_bytes=trainable,
        gradient_bytes=gradients,
        optimizer_moment_bytes=moments,
        activation_workspace_bytes=activations,
        runtime_reserve_bytes=runtime_reserve,
        estimated_peak_bytes=total,
        allocatable_dram_bytes=P150A_ALLOCATABLE_DRAM_BYTES,
        guarded_dram_bytes=P150A_GUARDED_DRAM_BYTES,
        admitted=total <= P150A_GUARDED_DRAM_BYTES,
        selection_reason=reason,
        retained_bf16_source_bytes=(
            base_parameter_count * 2 if selected != "bf16" else 0
        ),
        bf4_fraction_of_reduced_weights=bf4_fraction,
    )


__all__ = [
    "FROZEN_WEIGHT_EXECUTABLE_COPY_FACTOR",
    "MemoryEstimate",
    "P150A_ALLOCATABLE_DRAM_BYTES",
    "P150A_GUARDED_DRAM_BYTES",
    "estimate_memory",
]
