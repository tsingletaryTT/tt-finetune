"""Numerically stable, architecture-general LayerNorm for Blackhole training."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class _StableLayerNormFunction(torch.autograd.Function):
    """Explicit FP32-moment LayerNorm with a deterministic analytic backward."""

    @staticmethod
    def forward(
        ctx,
        value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        source_dtype = value.dtype
        value_fp32 = value.float()
        mean = value_fp32.mean(dim=-1, keepdim=True)
        centered = value_fp32 - mean
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        reciprocal_std = torch.rsqrt(variance + float(epsilon))
        normalized = centered * reciprocal_std
        ctx.save_for_backward(normalized, reciprocal_std, weight)
        output = normalized * weight.float() + bias.float()
        return output.to(source_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        normalized, reciprocal_std, weight = ctx.saved_tensors
        grad_fp32 = grad_output.float()
        weighted = grad_fp32 * weight.float()
        width = normalized.shape[-1]
        mean_weighted = weighted.sum(dim=-1, keepdim=True) / float(width)
        mean_projected = (
            (weighted * normalized).sum(dim=-1, keepdim=True) / float(width)
        )
        grad_input = (
            weighted - mean_weighted - normalized * mean_projected
        ) * reciprocal_std
        reduction_dims = tuple(range(grad_fp32.ndim - 1))
        grad_weight = (grad_fp32 * normalized).sum(dim=reduction_dims)
        grad_bias = grad_fp32.sum(dim=reduction_dims)
        return (
            grad_input.to(grad_output.dtype),
            grad_weight.to(weight.dtype),
            grad_bias.to(weight.dtype),
            None,
        )


class StableLayerNorm(nn.Module):
    """Drop-in 1-D normalized-shape LayerNorm using the stable contract."""

    def __init__(
        self,
        normalized_shape: int | Iterable[int],
        *,
        eps: float,
        weight: nn.Parameter,
        bias: nn.Parameter,
    ) -> None:
        super().__init__()
        shape = (
            (int(normalized_shape),)
            if isinstance(normalized_shape, int)
            else tuple(int(item) for item in normalized_shape)
        )
        if len(shape) != 1:
            raise ValueError("stable LayerNorm currently requires one normalized axis")
        self.normalized_shape = shape
        self.eps = float(eps)
        self.elementwise_affine = True
        self.weight = weight
        self.bias = bias

    @classmethod
    def from_module(cls, module: nn.LayerNorm) -> "StableLayerNorm":
        if module.weight is None or module.bias is None:
            raise ValueError("stable LayerNorm requires affine weight and bias")
        return cls(
            module.normalized_shape,
            eps=module.eps,
            weight=module.weight,
            bias=module.bias,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _StableLayerNormFunction.apply(
            value, self.weight, self.bias, self.eps
        )


def install_stable_layer_norm(model: nn.Module) -> tuple[str, ...]:
    """Replace every standard LayerNorm by the canonical stable primitive."""

    replacements: list[tuple[str, StableLayerNorm]] = []
    for path, module in model.named_modules():
        if path and isinstance(module, nn.LayerNorm):
            replacements.append((path, StableLayerNorm.from_module(module)))
    for path, replacement in replacements:
        parent_path, _, child_name = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, child_name, replacement)
    return tuple(path for path, _replacement in replacements)


__all__ = ["StableLayerNorm", "install_stable_layer_norm"]
