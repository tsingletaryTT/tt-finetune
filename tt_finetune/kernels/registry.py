"""Tracks which operations use compiler-generated or custom native kernels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelCapability:
    operation: str
    implementation: str
    status: str
    reason: str


KERNELS = (
    KernelCapability(
        operation="lora_linear",
        implementation="tt_xla_generated",
        status="hardware_passed",
        reason="Linear q/v LoRA forward and backward passed on P150a.",
    ),
    KernelCapability(
        operation="lora_conv1d",
        implementation="tt_xla_generated",
        status="hardware_passed",
        reason="GPT-2 fused c_attn LoRA passed on P150a.",
    ),
    KernelCapability(
        operation="fused_lora_linear",
        implementation="tt_metal_custom",
        status="experimental_hardware_passed",
        reason=(
            "Exact native forward passed Pythia and Qwen on P150a with "
            "3.19x-6.61x isolated speedup."
        ),
    ),
    KernelCapability(
        operation="fused_lora_backward_dx",
        implementation="tt_metal_custom",
        status="experimental_hardware_passed",
        reason=(
            "Exact native dY@B->dH@A passed Pythia and Qwen on P150a with "
            "2.05x-4.38x isolated speedup."
        ),
    ),
    KernelCapability(
        operation="lora_weight_gradients",
        implementation="tt_metal_custom",
        status="experimental_hardware_passed",
        reason=(
            "Production Linear-autograd operand order is bit-exact on "
            "Pythia and Qwen. Native dA/dB are faster on measured Pythia; "
            "Qwen retains its generic fallback."
        ),
    ),
    KernelCapability(
        operation="native_lora_projection_pipeline",
        implementation="tt_metal_custom_persistent_arenas",
        status="experimental_hardware_passed",
        reason=(
            "One-call [output|H] and [dH;dB] arenas are bit-exact. The "
            "backward arena is 2.63x-3.92x faster in isolation and dependent "
            "custom calls now run safely without forced synchronization. "
            "Complete autograd remains 0.86x Pythia / 0.68x Qwen, so the "
            "generic production fallback remains active."
        ),
    ),
    KernelCapability(
        operation="fused_frozen_linear_lora",
        implementation="tt_metal_custom",
        status="hardware_passed_experimental_integration",
        reason=(
            "Exact Qwen and Pythia forward passed with zero output/H "
            "mismatches at 4.06x and 2.65x. The Pythia K1 reduction "
            "contract and full 11x10 P150a grid are validated."
        ),
    ),
    KernelCapability(
        operation="fused_frozen_linear_lora_backward",
        implementation="tt_metal_custom_persistent_arena",
        status="hardware_passed_experimental_integration",
        reason=(
            "One call computes frozen-base dX plus LoRA dH/dB without dW. "
            "Pythia is 5.09x and Qwen 2.49x faster in isolation; every "
            "returned segment is bit-identical."
        ),
    ),
    KernelCapability(
        operation="native_frozen_lora_autograd",
        implementation="tt_metal_custom_plus_exact_generic_tail",
        status="hardware_passed_auto_eligible",
        reason=(
            "Full projection output/dX/dA/dB are bit-identical. The current "
            "three-family regression measures 1.22x Pythia and 1.49x Qwen; "
            "auto may select validated XLA shapes while CPU/unknown shapes "
            "remain generic. Trainer module replacement is still pending."
        ),
    ),
    KernelCapability(
        operation="grouped_shared_input_lora_forward",
        implementation="tt_metal_block_diagonal_rank_arena",
        status="hardware_passed_experimental_integration",
        reason=(
            "Two-member SmolLM2 Q+K and gate+up groups are bit-exact. "
            "Grouped forward is 4.09x and 3.48x faster than two separate "
            "native calls. Three-member QKV is numerically rejected."
        ),
    ),
    KernelCapability(
        operation="grouped_shared_input_lora_autograd",
        implementation="tt_metal_grouped_forward_member_backward",
        status="hardware_passed_experimental_integration",
        reason=(
            "SmolLM2 gate+up output/dX/dA/dB are bit-identical. Complete "
            "autograd improves 3.264 ms to 2.021 ms (1.62x) without dW."
        ),
    ),
    KernelCapability(
        operation="layout_aware_lora_autograd",
        implementation="torch_autograd_prototype",
        status="rejected",
        reason=(
            "Removed two transpose nodes but was slower and not bit-exact "
            "for LoRA weight gradients."
        ),
    ),
)
