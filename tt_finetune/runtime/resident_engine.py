"""Execution bridge for the experimental resident runtime resident stack.

This owner deliberately separates *model edges* from the resident Transformer
core.  Embedding, final normalization, LM head and causal loss may initially
remain ordinary device operations.  A callback converts the final resident
hidden state into ``(loss, dHidden)``; the latter feeds the packed resident
backward/clip/AdamW pipeline without a host transfer or per-layer PJRT tensor.

The safe trainer remains the default.  This class is an explicit experimental
backend and fails construction for any shape that is not represented by a
qualified source/optimizer descriptor bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from tt_finetune.kernels.tt_lang_adamw_descriptor_owner import (
    build_tt_lang_adamw_descriptor_owner,
)
from tt_finetune.kernels.tt_lang_arena_page_copy import (
    build_tt_lang_arena_page_copy,
)
from tt_finetune.kernels.tt_lang_multi_layer_gradient_arena_schedule import (
    build_tt_lang_multi_layer_gradient_arena_schedule,
)
from tt_finetune.kernels.tt_lang_output_attention_resident_backward import (
    full_attention_backward_arena_shape,
)
from tt_finetune.kernels.tt_lang_packed_gradient_clip_owner import (
    build_tt_lang_packed_gradient_clip_owner,
)
from tt_finetune.kernels.tt_lang_resident_layer_forward_trainable_owner import (
    build_tt_lang_resident_layer_forward_trainable_owner,
)
from tt_finetune.kernels.tt_lang_resident_layer_forward_compact_owner import (
    build_tt_lang_resident_layer_forward_compact_owner,
)

from .packed_state import PackedTransformerStack
from .causal_lm import (
    CausalLMDeviceEdges,
    CausalLMNativeWorkspace,
)


@dataclass(frozen=True)
class AdamWScalars:
    decay_factor: torch.Tensor
    denominator_scale: torch.Tensor
    scaled_epsilon: torch.Tensor


@dataclass(frozen=True)
class ResidentStepResult:
    loss: torch.Tensor
    final_hidden: torch.Tensor
    final_input_gradient: torch.Tensor
    optimizer_state: torch.Tensor


class PackedUpdateEngine:
    """Rank-scalable exact clipping/AdamW fallback for generic autograd.

    This keeps the canonical packed state and device-side update path when a
    Transformer shape has not yet been admitted by the resident forward and
    backward owners.  It deliberately owns no model math and is therefore a
    safe fallback rather than a second numerical implementation.
    """

    def __init__(
        self,
        packed: PackedTransformerStack,
        *,
        descriptor_bundle: str | Path,
        device: torch.device,
        grouped_norms: bool = True,
    ) -> None:
        self.packed = packed
        shape = packed.shape
        self.clip_owner = build_tt_lang_packed_gradient_clip_owner(
            layer_count=shape.layers,
            sequence_length=shape.batch_size * shape.sequence_length,
            hidden_features=shape.hidden_features,
            intermediate_features=shape.intermediate_features,
            kv_features=shape.kv_features,
            physical_rank=shape.physical_rank,
            grouped_norms=grouped_norms,
        )
        self.adamw_owner = build_tt_lang_adamw_descriptor_owner(
            descriptor_bundle=str(descriptor_bundle)
        )
        rows = packed.trainable_pages.shape[0]
        if (
            self.adamw_owner.rows != rows
            or self.adamw_owner.gradient_page_offset
            != packed.gradient_plan.scratch_pages
            or self.adamw_owner.gradient_pages
            != shape.layers * packed.gradient_plan.layer_gradient_pages
        ):
            raise ValueError("AdamW descriptor does not match packed rank")
        bf16 = torch.bfloat16
        self.optimizer_state = torch.cat((
            packed.trainable_pages.to(device),
            torch.zeros_like(packed.trainable_pages).to(device),
            torch.zeros_like(packed.trainable_pages).to(device),
        ))
        self.clip_scaler = torch.cat((
            torch.ones((32, 32), dtype=bf16),
            torch.full((32, 32), 0.5, dtype=bf16),
        )).to(device)
        norm_rows = len(packed.gradient_plan.slices) * 32
        self.clip_norm_template = _device_seed(
            (norm_rows, 32), dtype=bf16, device=device, nonce=41
        )
        self.clip_norm_output_template = _device_seed(
            (norm_rows, 32), dtype=bf16, device=device, nonce=42
        )
        self.clip_constants = torch.cat((
            torch.full((32, 32), 1.0e-6, dtype=bf16),
            torch.ones((32, 32), dtype=bf16),
            torch.ones((32, 32), dtype=bf16),
        )).to(device)
        self.clip_coefficient_template = _device_seed(
            (32, 32), dtype=bf16, device=device, nonce=43
        )
        self.clipped_gradient_output = _device_seed(
            packed.gradient_plan.physical_shape,
            dtype=bf16, device=device, nonce=44,
        )
        self.moments_template = _device_seed(
            (4 * rows, 32), dtype=bf16, device=device, nonce=45
        )
        self.denominator_template = _device_seed(
            (rows, 32), dtype=torch.float32, device=device, nonce=46
        )
        self.optimizer_output = _device_seed(
            (3 * rows, 32), dtype=bf16, device=device, nonce=47
        )

    def clip_adamw(
        self, gradients: torch.Tensor, scalars: AdamWScalars,
    ) -> torch.Tensor:
        if tuple(gradients.shape) != self.packed.gradient_plan.physical_shape:
            raise ValueError("raw gradient arena has the wrong physical shape")
        clipped = self.clip_owner(
            gradients, self.clip_scaler, self.clip_norm_template,
            self.clip_norm_output_template, self.clip_constants,
            self.clip_coefficient_template, self.clipped_gradient_output,
        )
        updated = self.adamw_owner(
            self.optimizer_state, clipped, scalars.decay_factor,
            scalars.denominator_scale, scalars.scaled_epsilon,
            self.moments_template, self.denominator_template,
            self.optimizer_output,
        )
        self.clipped_gradient_output = clipped
        self.optimizer_state = updated
        return updated

    def update_logical_gradients(
        self,
        gradients: dict[str, torch.Tensor],
        scalars: AdamWScalars,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Device-pack logical gradients and apply the exact packed update."""

        arena = self.packed.pack_logical_gradients(gradients)
        return arena, self.clip_adamw(arena, scalars)


def _device_seed(
    shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device,
    nonce: int = 0,
) -> torch.Tensor:
    value = torch.zeros(shape, dtype=dtype)
    if nonce:
        value.reshape(-1)[0] = float(nonce) / 128.0
    return value.to(device)


class ResidentStackEngine:
    """Stateful forward→loss-seed→backward→clip→AdamW device bridge."""

    def __init__(
        self,
        packed: PackedTransformerStack,
        *,
        source_bundle: str | Path,
        descriptor_bundle: str | Path,
        device: torch.device,
        grouped_norms: bool = True,
        compact_forward: bool = False,
        canonical_exact_attention: bool = False,
        o_lora_k_tiles_per_phase: int | None = None,
        native_bounded_causal_lm_edge: bool = False,
        diagnostic_top_full_backward: bool = False,
        diagnostic_top_attention_parameter_source: str = "packed",
        diagnostic_top_attention_stage: str | None = None,
        backward_direct_public_checkpoints: bool | None = None,
        diagnostic_forward_separate_parameters: bool = False,
    ) -> None:
        self.packed = packed
        self.device = device
        source_bundle = Path(source_bundle)
        descriptor_bundle = Path(descriptor_bundle)
        if not (source_bundle / "manifest.json").is_file():
            raise ValueError(
                f"missing accepted resident runtime source manifest: {source_bundle}"
            )
        if not (descriptor_bundle / "manifest.json").is_file():
            raise ValueError(
                f"missing accepted AdamW descriptor manifest: {descriptor_bundle}"
            )

        shape = packed.shape
        if shape.batch_size != 1:
            raise ValueError(
                "the frozen resident runtime resident specialization requires batch 1"
            )
        if backward_direct_public_checkpoints is None:
            # Both qualified checkpoint layouts carry raw FP32 halves inside
            # a BF16 arena.  A semantic private import may preserve ordinary
            # BF16 pages while canonicalizing those raw halves, which first
            # appears as a small Q/K/V backward divergence.  Consume the
            # device-produced parity arenas directly; the backward builder
            # still reserves the legacy private tensor IDs so the qualified
            # reverse L1 address plan is unchanged.
            backward_direct_public_checkpoints = True
        common = dict(
            layer_count=shape.layers,
            block_page_offsets=packed.block_page_offsets,
            source_bundle=str(source_bundle),
            sequence_length=shape.sequence_length,
            hidden_features=shape.hidden_features,
            intermediate_features=shape.intermediate_features,
            kv_features=shape.kv_features,
            query_heads=shape.query_heads,
            kv_heads=shape.kv_heads,
            head_dim=shape.head_dim,
            physical_rank=shape.physical_rank,
            lora_scale=shape.lora_scale,
            attention_scale=shape.attention_scale,
            rmsnorm_epsilon=shape.rmsnorm_epsilon,
            has_bias=shape.has_bias,
            o_lora_k_tiles_per_phase=o_lora_k_tiles_per_phase,
        )
        if canonical_exact_attention and not compact_forward:
            raise ValueError(
                "canonical exact attention requires compact_forward=True"
            )
        exact_schedule = dict(
            canonical_exact_attention=canonical_exact_attention,
            logical_rank=shape.logical_rank,
        )
        self.forward_owner = build_tt_lang_multi_layer_gradient_arena_schedule(
            **common,
            **exact_schedule,
            compact_forward=compact_forward,
            reuse_backward_scratch=True,
            execution_phase="forward",
            debug_forward_checkpoint_separate_parameters=(
                diagnostic_forward_separate_parameters
            ),
        )
        self.backward_owner = build_tt_lang_multi_layer_gradient_arena_schedule(
            **common,
            logical_rank=shape.logical_rank,
            execution_phase="backward",
            debug_top_full_backward=diagnostic_top_full_backward,
            debug_top_attention_stage=diagnostic_top_attention_stage,
            backward_direct_public_checkpoints=(
                backward_direct_public_checkpoints
            ),
            debug_top_attention_parameter_source=(
                diagnostic_top_attention_parameter_source
            ),
            # Qwen's biased/rank-16 backward has a one-layer delayed TTNN
            # consumer on the private attention scratch.  Consecutively
            # reusing one DPS/L1 group lets the next layer overwrite that
            # storage before the prior consumer has retired.  Ping-pong the
            # two groups; bias-free Smol keeps the smaller one-group plan.
            reuse_backward_scratch=(4 if shape.has_bias else True),
        )
        self.clip_owner = build_tt_lang_packed_gradient_clip_owner(
            layer_count=shape.layers,
            sequence_length=shape.sequence_length,
            hidden_features=shape.hidden_features,
            intermediate_features=shape.intermediate_features,
            kv_features=shape.kv_features,
            physical_rank=shape.physical_rank,
            grouped_norms=grouped_norms,
        )
        self.adamw_owner = build_tt_lang_adamw_descriptor_owner(
            descriptor_bundle=str(descriptor_bundle)
        )
        optimizer_rows = packed.trainable_pages.shape[0]
        if (
            self.adamw_owner.rows != optimizer_rows
            or self.adamw_owner.gradient_page_offset
            != packed.gradient_plan.scratch_pages
            or self.adamw_owner.gradient_pages
            != shape.layers * packed.gradient_plan.layer_gradient_pages
        ):
            raise ValueError(
                "AdamW descriptor does not match the real model's packed state"
            )

        per_layer_forward_builder = (
            build_tt_lang_resident_layer_forward_compact_owner
            if compact_forward
            else build_tt_lang_resident_layer_forward_trainable_owner
        )
        exact_forward_kwargs = (
            {
                "canonical_native_o_path": True,
                "exact_attention_tail": True,
                # Match the accepted per-layer autograd oracle.  Its Smol
                # rank-32+ path keeps the original FP32 attention tail and
                # ordinary per-projection LoRA schedule; only rank 8 uses the
                # packed Q/V/K specialization.
                "canonical_bf16_attention": False,
                "canonical_grouped_qkv_rank": bool(
                    not shape.has_bias and shape.logical_rank == 8
                ),
                "canonical_separate_qkv_rank": False,
                "logical_rank": shape.logical_rank,
            }
            if canonical_exact_attention else {}
        )
        per_layer_forward = per_layer_forward_builder(
            sequence_length=shape.sequence_length,
            hidden_features=shape.hidden_features,
            intermediate_features=shape.intermediate_features,
            kv_features=shape.kv_features,
            query_heads=shape.query_heads,
            kv_heads=shape.kv_heads,
            head_dim=shape.head_dim,
            physical_rank=shape.physical_rank,
            lora_scale=shape.lora_scale,
            attention_scale=shape.attention_scale,
            has_bias=shape.has_bias,
            input_rmsnorm_epsilon=shape.rmsnorm_epsilon,
            post_attention_rmsnorm_epsilon=shape.rmsnorm_epsilon,
            hidden_page_base=0,
            publish_layer_output_at_page_zero=False,
            o_lora_k_tiles_per_phase=o_lora_k_tiles_per_phase,
            **exact_forward_kwargs,
        )
        final_layer = shape.layers - 1
        final_page_offset = (
            (final_layer // 2) * per_layer_forward.layer_allocated_pages
            + per_layer_forward.layer_output_page_offset
        )
        self.final_checkpoint_parity = final_layer % 2
        self.layer_checkpoint_rows = per_layer_forward.output_shape[0]
        self.layer_checkpoint_columns = per_layer_forward.output_shape[1]
        self.layer_allocated_pages = per_layer_forward.layer_allocated_pages
        self.layer_output_page_offset = (
            per_layer_forward.layer_output_page_offset
        )
        self.final_hidden_copy = build_tt_lang_arena_page_copy(
            rows=shape.sequence_length,
            columns=shape.hidden_features,
            source_page_offset=final_page_offset,
        )

        bf16 = torch.bfloat16
        hidden_shape = (shape.sequence_length, shape.hidden_features)
        attention_shape = full_attention_backward_arena_shape(
            tokens=shape.sequence_length,
            hidden_features=shape.hidden_features,
            kv_features=shape.kv_features,
        )
        mlp_shapes = (
            (32, shape.hidden_features),
            (shape.intermediate_features, 32),
            (32, shape.hidden_features),
            (shape.intermediate_features, 32),
            (32, shape.intermediate_features),
            (shape.hidden_features, 32),
        )
        self.fp32_seed_a = _device_seed(
            hidden_shape, dtype=torch.float32, device=device
        )
        self.fp32_seed_b = _device_seed(
            hidden_shape, dtype=torch.float32, device=device, nonce=1
        )
        self.bf16_seed = _device_seed(hidden_shape, dtype=bf16, device=device)
        self.checkpoint_template = _device_seed(
            per_layer_forward.output_shape, dtype=bf16, device=device
        )
        self.hidden_template = _device_seed(
            hidden_shape, dtype=bf16, device=device, nonce=2
        )
        self.attention_template = _device_seed(
            attention_shape, dtype=bf16, device=device, nonce=3
        )
        self.mlp_templates = tuple(
            _device_seed(shape_, dtype=bf16, device=device, nonce=4 + index)
            for index, shape_ in enumerate(mlp_shapes)
        )
        self.gradient_template = _device_seed(
            packed.gradient_plan.physical_shape,
            dtype=bf16,
            device=device,
            nonce=10,
        )
        self.frozen = packed.frozen_pages.to(device)
        parameters = packed.trainable_pages.to(device)
        self.optimizer_state = torch.cat((
            parameters,
            torch.zeros_like(parameters),
            torch.zeros_like(parameters),
        ))
        self.block_templates = tuple(value.to(device) for value in packed.block_template)

        self.checkpoints = [
            _device_seed(
                arena_shape,
                dtype=bf16,
                device=device,
                nonce=20 + index,
            )
            for index, arena_shape in enumerate(
                self.forward_owner.checkpoint_arena_shapes
            )
        ]
        self.final_hidden_output = _device_seed(
            hidden_shape, dtype=bf16, device=device, nonce=22
        )
        self.gradient_output = _device_seed(
            packed.gradient_plan.physical_shape,
            dtype=bf16,
            device=device,
            nonce=23,
        )
        self.final_input_gradient_output = _device_seed(
            hidden_shape, dtype=bf16, device=device, nonce=24
        )
        self.clip_scaler = torch.cat((
            torch.ones((32, 32), dtype=bf16),
            torch.full((32, 32), 0.5, dtype=bf16),
        )).to(device)
        norm_rows = len(packed.gradient_plan.slices) * 32
        self.clip_norm_template = _device_seed(
            (norm_rows, 32), dtype=bf16, device=device, nonce=25
        )
        self.clip_norm_output_template = _device_seed(
            (norm_rows, 32), dtype=bf16, device=device, nonce=26
        )
        self.clip_constants = torch.cat((
            torch.full((32, 32), 1.0e-6, dtype=bf16),
            torch.ones((32, 32), dtype=bf16),
            torch.ones((32, 32), dtype=bf16),
        )).to(device)
        self.clip_coefficient_template = _device_seed(
            (32, 32), dtype=bf16, device=device, nonce=27
        )
        self.clipped_gradient_output = _device_seed(
            packed.gradient_plan.physical_shape,
            dtype=bf16,
            device=device,
            nonce=28,
        )
        self.moments_template = _device_seed(
            (4 * optimizer_rows, 32), dtype=bf16, device=device, nonce=29
        )
        self.denominator_template = _device_seed(
            (optimizer_rows, 32), dtype=torch.float32, device=device, nonce=30
        )
        self.optimizer_output = _device_seed(
            (3 * optimizer_rows, 32), dtype=bf16, device=device, nonce=31
        )
        self._pending_checkpoints: tuple[torch.Tensor, torch.Tensor] | None = None
        self.compact_forward = bool(compact_forward)
        self.canonical_exact_attention = bool(canonical_exact_attention)
        self.native_bounded_causal_lm_edge = bool(
            native_bounded_causal_lm_edge
        )
        self._native_causal_lm_workspace: (
            CausalLMNativeWorkspace | None
        ) = None

    @property
    def public_custom_calls_per_step(self) -> int:
        # Forward, final-hidden extraction, backward, clipping, AdamW.  Generic
        # final norm/head/loss operations are reported separately by StableHLO.
        return 7 if self.native_bounded_causal_lm_edge else 5

    @property
    def physical_programs_per_step(self) -> int:
        return (
            self.forward_owner.physical_programs
            + 1
            + self.backward_owner.physical_programs
            + self.clip_owner.physical_programs
            + self.adamw_owner.physical_programs
        )

    def forward_hidden(self, initial_hidden: torch.Tensor) -> torch.Tensor:
        shape = self.packed.shape
        flat = initial_hidden.reshape(shape.sequence_length, shape.hidden_features)
        checkpoints = self.forward_owner(
            flat,
            self.bf16_seed,
            self.fp32_seed_a,
            self.fp32_seed_b,
            self.bf16_seed,
            self.checkpoint_template,
            self.hidden_template,
            self.attention_template,
            *self.mlp_templates,
            self.gradient_template,
            self.frozen,
            self.optimizer_state,
            *self.block_templates,
            self.checkpoints[0],
            self.checkpoints[1],
        )
        self.checkpoints[:] = checkpoints
        self._pending_checkpoints = checkpoints
        final_hidden = self.final_hidden_copy(
            checkpoints[self.final_checkpoint_parity],
            self.final_hidden_output,
        )
        self.final_hidden_output = final_hidden
        return final_hidden.reshape_as(initial_hidden)

    def backward_clip_adamw(
        self,
        initial_hidden: torch.Tensor,
        top_gradient: torch.Tensor,
        scalars: AdamWScalars,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gradients, final_input_gradient = self.backward_raw(
            initial_hidden, top_gradient
        )
        updated = self.clip_adamw(gradients, scalars)
        return updated, final_input_gradient

    def backward_raw(
        self,
        initial_hidden: torch.Tensor,
        top_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the resident reverse stack and expose only its packed ABI."""

        if self._pending_checkpoints is None:
            raise RuntimeError("resident runtime backward requires a preceding forward")
        shape = self.packed.shape
        gradients, final_input_gradient = self.backward_owner(
            initial_hidden.reshape(shape.sequence_length, shape.hidden_features),
            top_gradient.reshape(shape.sequence_length, shape.hidden_features),
            self.fp32_seed_a,
            self.fp32_seed_b,
            self.bf16_seed,
            self._pending_checkpoints[0],
            self._pending_checkpoints[1],
            self.attention_template,
            *self.mlp_templates,
            self.gradient_template,
            self.frozen,
            self.optimizer_state,
            *self.block_templates,
            self.gradient_output,
            self.final_input_gradient_output,
        )
        self.gradient_output = gradients
        self.final_input_gradient_output = final_input_gradient
        self._pending_checkpoints = None
        return gradients, final_input_gradient

    def clip_adamw(
        self,
        gradients: torch.Tensor,
        scalars: AdamWScalars,
    ) -> torch.Tensor:
        """Consume one raw packed arena and evolve the persistent state."""

        if tuple(gradients.shape) != self.packed.gradient_plan.physical_shape:
            raise ValueError("raw gradient arena has the wrong physical shape")
        clipped = self.clip_owner(
            gradients,
            self.clip_scaler,
            self.clip_norm_template,
            self.clip_norm_output_template,
            self.clip_constants,
            self.clip_coefficient_template,
            self.clipped_gradient_output,
        )
        updated = self.adamw_owner(
            self.optimizer_state,
            clipped,
            scalars.decay_factor,
            scalars.denominator_scale,
            scalars.scaled_epsilon,
            self.moments_template,
            self.denominator_template,
            self.optimizer_output,
        )
        self.clipped_gradient_output = clipped
        self.optimizer_state = updated
        return updated

    def step(
        self,
        initial_hidden: torch.Tensor,
        loss_and_top_gradient: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ],
        scalars: AdamWScalars,
    ) -> ResidentStepResult:
        final_hidden = self.forward_hidden(initial_hidden)
        loss, top_gradient = loss_and_top_gradient(final_hidden)
        state, final_input_gradient = self.backward_clip_adamw(
            initial_hidden, top_gradient, scalars
        )
        return ResidentStepResult(
            loss=loss,
            final_hidden=final_hidden,
            final_input_gradient=final_input_gradient,
            optimizer_state=state,
        )

    def step_tokens(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        edges: CausalLMDeviceEdges,
        scalars: AdamWScalars,
    ) -> ResidentStepResult:
        """Run a real causal-LM step without exposing a synthetic dHidden.

        The public method is intentionally narrow while resident runtime remains
        experimental: batch one, the qualified fixed sequence length, and a
        frozen causal-LM edge contract.  Unsupported shapes fail before any
        device program is submitted and callers retain their safe trainer.
        """

        shape = self.packed.shape
        expected_tokens = (1, shape.sequence_length)
        if tuple(input_ids.shape) != expected_tokens:
            raise ValueError(
                f"resident runtime input_ids must have shape {expected_tokens}"
            )
        if tuple(labels.shape) != expected_tokens:
            raise ValueError(
                f"resident runtime labels must have shape {expected_tokens}"
            )
        if edges.hidden_features != shape.hidden_features:
            raise ValueError("causal-LM edges do not match the resident stack")
        initial_hidden = edges.embed(input_ids)
        if self.native_bounded_causal_lm_edge:
            if self._native_causal_lm_workspace is None:
                self._native_causal_lm_workspace = (
                    edges.allocate_native_workspace(shape.sequence_length)
                )
            loss_edge = lambda final_hidden: (
                edges.native_bounded_loss_and_top_gradient(
                    final_hidden,
                    labels,
                    self._native_causal_lm_workspace,
                )
            )
        else:
            loss_edge = lambda final_hidden: edges.loss_and_top_gradient(
                final_hidden, labels
            )
        return self.step(
            initial_hidden,
            loss_edge,
            scalars,
        )


__all__ = [
    "AdamWScalars",
    "PackedUpdateEngine",
    "ResidentStackEngine",
    "ResidentStepResult",
]
