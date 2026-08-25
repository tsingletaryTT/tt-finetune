"""Experimental rank/batch-parametric PEFT projection backward.

The ordinary PEFT module remains the source of truth for parameters and
checkpoints.  This integration only replaces the autograd boundary of an
eligible frozen Linear + LoRA projection.  Forward keeps PEFT's canonical
BF16 operation order; backward is owned by one TT-Lang custom call whose
intermediates are compiler-private L1 tensors.

Unsupported calls execute the untouched PEFT forward.  Nothing in this file
is selected by the safe trainer unless the experimental installer is called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MethodType
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from tt_finetune.kernels.tt_lang_native_frozen_lora_backward import (
    build_tt_lang_native_frozen_lora_backward,
)


TILE_SIZE = 32


def _active_adapter(module: nn.Module) -> str | None:
    active = getattr(module, "active_adapter", "default")
    if isinstance(active, str):
        return active
    if isinstance(active, Iterable):
        values = tuple(active)
        return values[0] if len(values) == 1 else None
    return None


def _dropout_probability(module: nn.Module, adapter: str) -> float:
    dropouts = getattr(module, "lora_dropout", {})
    dropout = (
        dropouts.get(adapter)
        if hasattr(dropouts, "get")
        else dropouts[adapter]
        if adapter in dropouts
        else None
    )
    return float(getattr(dropout, "p", 0.0)) if dropout is not None else 0.0


def _dora_enabled(module: nn.Module, adapter: str) -> bool:
    values = getattr(module, "use_dora", {})
    if hasattr(values, "get"):
        return bool(values.get(adapter, False))
    return bool(values[adapter]) if adapter in values else False


def _support_error(module: nn.Module, inputs: torch.Tensor) -> str | None:
    adapter = _active_adapter(module)
    if adapter is None:
        return "requires exactly one active adapter"
    if bool(getattr(module, "disable_adapters", False)):
        return "adapters are disabled"
    if bool(getattr(module, "merged", False)):
        return "merged adapters are unsupported"
    if _dora_enabled(module, adapter):
        return "DoRA is unsupported"
    if _dropout_probability(module, adapter) != 0.0:
        return "requires zero LoRA dropout"
    try:
        base = module.base_layer
        weight = base.weight
        lora_a = module.lora_A[adapter].weight
        lora_b = module.lora_B[adapter].weight
    except (AttributeError, KeyError, TypeError):
        return "not a PEFT LoRA Linear"
    if inputs.ndim < 2:
        return "input rank must be at least two"
    tokens = math.prod(inputs.shape[:-1])
    input_features = inputs.shape[-1]
    output_features, weight_features = weight.shape
    logical_rank = lora_a.shape[0]
    if (
        weight.ndim != 2
        or weight_features != input_features
        or tuple(lora_a.shape) != (logical_rank, input_features)
        or tuple(lora_b.shape) != (output_features, logical_rank)
    ):
        return "incompatible Linear/LoRA shapes"
    if logical_rank <= 0:
        return "rank must be positive"
    if any(
        value <= 0 or value % TILE_SIZE
        for value in (tokens, input_features, output_features)
    ):
        return "flattened tokens and feature dimensions must be tile aligned"
    if inputs.device.type != "xla":
        return "native projection requires XLA tensors"
    tensors = (inputs, weight, lora_a, lora_b)
    if any(value.dtype != torch.bfloat16 for value in tensors):
        return "exact native projection requires BF16 tensors"
    if any(value.device != inputs.device for value in tensors[1:]):
        return "projection tensors must share one device"
    if bool(weight.requires_grad):
        return "base weight must be frozen"
    return None


def _pad_adapter(
    lora_a: torch.Tensor, lora_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    logical_rank = lora_a.shape[0]
    physical_rank = math.ceil(logical_rank / TILE_SIZE) * TILE_SIZE
    padding = physical_rank - logical_rank
    if padding == 0:
        return lora_a, lora_b, physical_rank
    return (
        torch.cat(
            (lora_a, lora_a.new_zeros((padding, lora_a.shape[1]))),
            dim=0,
        ),
        torch.cat(
            (lora_b, lora_b.new_zeros((lora_b.shape[0], padding))),
            dim=1,
        ),
        physical_rank,
    )


class _ScalableFrozenLoRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        frozen_weight: torch.Tensor,
        bias: torch.Tensor | None,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        original_shape = tuple(inputs.shape)
        flat_inputs = inputs.reshape(-1, original_shape[-1])
        physical_rank = math.ceil(lora_a.shape[0] / TILE_SIZE) * TILE_SIZE

        # Match PEFT's observable order: base Linear, A Linear, B Linear,
        # scale, then add.  Keep the logical rank visible: padding rank tails
        # before the second matmul changes the reduction schedule for dX even
        # when every padded value is zero.
        base = F.linear(flat_inputs, frozen_weight, bias)
        hidden = F.linear(flat_inputs, lora_a)
        output = base + F.linear(hidden, lora_b) * float(scale)

        ctx.save_for_backward(
            flat_inputs, frozen_weight, lora_a, lora_b,
        )
        ctx.scale = float(scale)
        ctx.logical_rank = int(lora_a.shape[0])
        ctx.physical_rank = physical_rank
        ctx.original_shape = original_shape
        return output.reshape(*original_shape[:-1], frozen_weight.shape[0])

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        flat_inputs, frozen_weight, lora_a, lora_b = ctx.saved_tensors
        flat_gradient = output_gradient.reshape(-1, output_gradient.shape[-1])
        operation = build_tt_lang_native_frozen_lora_backward(
            scale=ctx.scale,
            tokens=flat_inputs.shape[0],
            input_features=flat_inputs.shape[1],
            output_features=flat_gradient.shape[1],
            physical_rank=ctx.physical_rank,
            logical_rank=ctx.logical_rank,
            private_memory="l1",
        )
        input_gradient, a_gradient, b_gradient = operation(
            flat_inputs,
            flat_gradient,
            frozen_weight,
            lora_a,
            lora_b,
            torch.zeros_like(flat_inputs),
            torch.zeros_like(lora_a),
            torch.zeros_like(lora_b),
        )
        return (
            input_gradient.reshape(ctx.original_shape),
            None,
            None,
            a_gradient,
            b_gradient,
            None,
        )


class _ScalableLoRABranchFunction(torch.autograd.Function):
    """Own only the LoRA branch while PEFT's frozen base branch stays visible."""

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        frozen_weight: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        original_shape = tuple(inputs.shape)
        flat_inputs = inputs.reshape(-1, original_shape[-1])
        hidden = F.linear(flat_inputs, lora_a)
        output = F.linear(hidden, lora_b) * float(scale)
        ctx.save_for_backward(flat_inputs, hidden, lora_a, lora_b)
        ctx.scale = float(scale)
        ctx.logical_rank = int(lora_a.shape[0])
        ctx.physical_rank = math.ceil(lora_a.shape[0] / TILE_SIZE) * TILE_SIZE
        ctx.original_shape = original_shape
        return output.reshape(*original_shape[:-1], lora_b.shape[0])

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        flat_inputs, saved_hidden, lora_a, lora_b = ctx.saved_tensors
        flat_gradient = output_gradient.reshape(-1, output_gradient.shape[-1])
        operation = build_tt_lang_native_frozen_lora_backward(
            scale=ctx.scale,
            tokens=flat_inputs.shape[0],
            input_features=flat_inputs.shape[1],
            output_features=flat_gradient.shape[1],
            physical_rank=ctx.physical_rank,
            logical_rank=ctx.logical_rank,
            private_memory="l1",
            include_base_gradient=False,
        )
        input_gradient, a_gradient, b_gradient = operation(
            flat_inputs,
            flat_gradient,
            saved_hidden,
            lora_a,
            lora_b,
            torch.zeros_like(flat_inputs),
            torch.zeros_like(lora_a),
            torch.zeros_like(lora_b),
        )
        return (
            input_gradient.reshape(ctx.original_shape),
            None,
            a_gradient,
            b_gradient,
            None,
        )


def _native_projection_forward(
    module: nn.Module, inputs: torch.Tensor, *args, **kwargs,
) -> torch.Tensor:
    original = module._tt_scalable_projection_original_forward
    if args or kwargs:
        return original(inputs, *args, **kwargs)
    if _support_error(module, inputs) is not None:
        return original(inputs)
    adapter = _active_adapter(module)
    assert adapter is not None
    base = module.base_layer
    base_output = base(inputs)
    delta = _ScalableLoRABranchFunction.apply(
        inputs,
        base.weight,
        module.lora_A[adapter].weight,
        module.lora_B[adapter].weight,
        float(module.scaling[adapter]),
    )
    return base_output + delta


@dataclass(frozen=True)
class ScalableProjectionInstallReport:
    installed_paths: tuple[str, ...]
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def installed_count(self) -> int:
        return len(self.installed_paths)


def install_scalable_native_projection_backward(
    model: nn.Module,
) -> ScalableProjectionInstallReport:
    """Patch eligible PEFT Linear modules with the experimental boundary.

    Admission is checked again for every invocation because the flattened
    token count is runtime-shape dependent.  Unsupported shapes use the saved
    PEFT method, providing an explicit safe fallback.
    """

    installed: list[str] = []
    skipped: dict[str, str] = {}
    for path, module in tuple(model.named_modules()):
        if not path or not hasattr(module, "lora_A"):
            continue
        if hasattr(module, "_tt_scalable_projection_original_forward"):
            installed.append(path)
            continue
        adapter = _active_adapter(module)
        if adapter is None:
            skipped[path] = "requires exactly one active adapter"
            continue
        try:
            base = module.base_layer
            lora_a = module.lora_A[adapter].weight
            lora_b = module.lora_B[adapter].weight
        except (AttributeError, KeyError, TypeError):
            skipped[path] = "not a PEFT LoRA Linear"
            continue
        # gate/up are parallel consumers whose B-gradient reductions are
        # scheduled together by the ordinary graph.  Owning either projection
        # independently changes that contextual matmul schedule (most visibly
        # at R128/B2), even though each isolated projection is exact.  Keep the
        # device-side PEFT path until the scalable grouped-MLP owner can admit
        # both siblings under one numerical schedule.
        if path.rsplit(".", 1)[-1] in {"gate_proj", "up_proj"}:
            skipped[path] = "requires grouped gate/up exact schedule"
            continue
        if (
            base.weight.ndim != 2
            or lora_a.ndim != 2
            or lora_b.ndim != 2
            or base.weight.shape[1] % TILE_SIZE
            or base.weight.shape[0] % TILE_SIZE
        ):
            skipped[path] = "feature dimensions are not tile aligned"
            continue
        module._tt_scalable_projection_original_forward = module.forward
        module.forward = MethodType(_native_projection_forward, module)
        installed.append(path)
    return ScalableProjectionInstallReport(
        installed_paths=tuple(installed), skipped=skipped,
    )


__all__ = [
    "ScalableProjectionInstallReport",
    "install_scalable_native_projection_backward",
]
