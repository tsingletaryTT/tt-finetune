"""Autograd integration for the exact experimental Blackhole LoRA backend.

This module composes the accepted exact TT-Metal primitives into one PyTorch
autograd contract. It intentionally keeps the physical rank tile (normally
32) visible: padding lanes are zero, receive zero gradients, and are sliced
only when exporting a PEFT checkpoint.

The forward integration returns ``[output | H]`` through one single-output
arena, and backward returns ``[dH; dB]`` through another. This is the
tuple-free compatibility design for the installed Shardy bridge.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .tt_lang_lora import (
    TILE_SIZE,
)
from .tt_lang_lora_arena import build_tt_lang_lora_forward_arena
from .tt_lang_lora_backward_arena import (
    build_tt_lang_lora_backward_arena,
)
from .tt_lang_lora_weight_backward import (
    build_tt_lang_lora_backward_da,
)


@dataclass
class NativeLoRADeviceArena:
    base_init: torch.Tensor
    forward_arena_init: torch.Tensor
    backward_arena_init: torch.Tensor
    lora_a_gradient_init: torch.Tensor


def make_native_lora_device_arena(
    inputs: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> NativeLoRADeviceArena:
    """Materialize persistent TT-Lang DPS destinations as device data."""

    tokens = inputs.shape[0]
    physical_rank = lora_a.shape[0]
    output_features = lora_b.shape[0]

    def device_zeros(shape: tuple[int, int]) -> torch.Tensor:
        return torch.zeros(shape, dtype=inputs.dtype).to(inputs.device)

    return NativeLoRADeviceArena(
        base_init=device_zeros((tokens, output_features)),
        forward_arena_init=device_zeros(
            (tokens, output_features + physical_rank)
        ),
        backward_arena_init=device_zeros(
            (tokens + output_features, physical_rank)
        ),
        lora_a_gradient_init=device_zeros(tuple(lora_a.shape)),
    )


def _sync_custom_boundary(tensor: torch.Tensor) -> None:
    """Diagnostic compatibility boundary for older TT-MLIR behavior.

    Persistent single-output arenas now pass without this boundary on the
    installed stack. Keeping the switch makes the former tensor-pool failure
    reproducible and provides a conservative fallback for another runtime.
    """

    import torch_xla.core.xla_model as xm

    # A targeted materialization keeps the custom-call result as an explicit
    # graph root. A global step marker from inside autograd forward can run
    # before the local result has become a root and reproduces the same
    # tensor-pool lifetime failure.
    xm.unlazy([tensor])


def _validate_native_shapes(
    inputs: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> tuple[int, int, int, int]:
    if inputs.ndim != 2 or lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError(
            "native exact LoRA currently requires flattened rank-2 tensors"
        )
    tokens, input_features = inputs.shape
    physical_rank, a_features = lora_a.shape
    output_features, b_rank = lora_b.shape
    if input_features != a_features or physical_rank != b_rank:
        raise ValueError("incompatible native LoRA tensor shapes")
    if physical_rank != TILE_SIZE:
        raise ValueError(
            "native exact LoRA currently requires physical rank 32"
        )
    dimensions = (tokens, input_features, output_features, physical_rank)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError(
            "native exact LoRA dimensions must be positive multiples of 32"
        )
    if input_features // TILE_SIZE % 2:
        raise ValueError(
            "native exact LoRA requires an even input-feature tile count"
        )
    return tokens, input_features, output_features, physical_rank


class NativeExactLoRAFunction(torch.autograd.Function):
    """Bit-exact experimental native LoRA delta with explicit backward."""

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scale: float,
        native_da: bool,
        sync_boundaries: bool,
        base_init: torch.Tensor,
        forward_arena_init: torch.Tensor,
        backward_arena_init: torch.Tensor,
        lora_a_gradient_init: torch.Tensor,
    ) -> torch.Tensor:
        (
            tokens,
            input_features,
            output_features,
            physical_rank,
        ) = _validate_native_shapes(inputs, lora_a, lora_b)

        forward_op = build_tt_lang_lora_forward_arena(
            tokens=tokens,
            input_features=input_features,
            output_features=output_features,
            physical_rank=physical_rank,
            scale=scale,
        )
        forward_arena = forward_op(
            inputs,
            lora_a,
            lora_b,
            base_init,
            forward_arena_init,
        )
        if sync_boundaries:
            _sync_custom_boundary(forward_arena)
        output = forward_arena[:, :output_features]
        hidden = forward_arena[:, output_features:]

        ctx.save_for_backward(
            inputs,
            hidden,
            lora_a,
            lora_b,
            backward_arena_init,
            lora_a_gradient_init,
        )
        ctx.scale = float(scale)
        ctx.native_da = bool(native_da)
        ctx.sync_boundaries = bool(sync_boundaries)
        return output

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        (
            inputs,
            hidden,
            lora_a,
            lora_b,
            backward_arena_init,
            lora_a_gradient_init,
        ) = ctx.saved_tensors
        tokens, input_features = inputs.shape
        output_features, physical_rank = lora_b.shape

        backward_op = build_tt_lang_lora_backward_arena(
            scale=ctx.scale,
            tokens=tokens,
            output_features=output_features,
            physical_rank=physical_rank,
        )
        backward_arena = backward_op(
            output_gradient,
            lora_b,
            hidden,
            backward_arena_init,
        )
        if ctx.sync_boundaries:
            _sync_custom_boundary(backward_arena)
        hidden_gradient = backward_arena[:tokens]
        lora_b_gradient = backward_arena[
            tokens : tokens + output_features
        ]

        # This rank-one-tile matmul is already exact and small. The combined
        # dX+dH native kernel remains available for the future multi-output
        # path; recomputing the expensive long-N reduction twice would be
        # slower than this short generic matmul.
        input_gradient = hidden_gradient @ lora_a

        if ctx.native_da:
            da_op = build_tt_lang_lora_backward_da(
                tokens=tokens,
                input_features=input_features,
                physical_rank=physical_rank,
                exact_reduction=True,
            )
            lora_a_gradient = da_op(
                hidden_gradient,
                inputs,
                lora_a_gradient_init,
            )
            if ctx.sync_boundaries:
                _sync_custom_boundary(lora_a_gradient)
        else:
            lora_a_gradient = (
                inputs.transpose(0, 1) @ hidden_gradient
            ).transpose(0, 1)

        return (
            input_gradient,
            lora_a_gradient,
            lora_b_gradient,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def native_exact_lora_delta(
    inputs: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
    *,
    native_da: bool | None = None,
    sync_boundaries: bool = False,
    allow_unsafe_bridge: bool = False,
    arena: NativeLoRADeviceArena | None = None,
) -> torch.Tensor:
    """Apply the experimental exact native LoRA autograd backend.

    ``native_da=None`` keeps dA in the generic graph. The native dA kernel is
    exact and faster in isolation for long-token shapes, but its additional
    custom-call dispatch is slower in the complete autograd projection.
    """

    if not allow_unsafe_bridge:
        raise RuntimeError(
            "Native LoRA Python autograd integration is experimental and "
            "disabled by default. It is bit-exact with persistent device-data "
            "arenas, but the complete projection is not yet faster than the "
            "generic graph. Use the manual phase benchmark or generic trainer "
            "until the fused base-linear path is accepted."
        )
    if native_da is None:
        native_da = False
    if arena is None:
        arena = make_native_lora_device_arena(inputs, lora_a, lora_b)
    return NativeExactLoRAFunction.apply(
        inputs,
        lora_a,
        lora_b,
        float(scale),
        bool(native_da),
        bool(sync_boundaries),
        arena.base_init,
        arena.forward_arena_init,
        arena.backward_arena_init,
        arena.lora_a_gradient_init,
    )


class NativeExactLoRAExecutor:
    """Persistent-arena callable for experimental trainer/module wiring."""

    def __init__(
        self,
        *,
        native_da: bool | None = None,
        sync_boundaries: bool = False,
    ):
        self.native_da = native_da
        self.sync_boundaries = sync_boundaries
        self.arena: NativeLoRADeviceArena | None = None
        self.signature: tuple[object, ...] | None = None

    def __call__(
        self,
        inputs: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        signature = (
            tuple(inputs.shape),
            tuple(lora_a.shape),
            tuple(lora_b.shape),
            inputs.dtype,
            inputs.device,
        )
        if self.signature != signature:
            self.arena = make_native_lora_device_arena(
                inputs, lora_a, lora_b
            )
            self.signature = signature
        assert self.arena is not None
        return native_exact_lora_delta(
            inputs,
            lora_a,
            lora_b,
            scale,
            native_da=self.native_da,
            sync_boundaries=self.sync_boundaries,
            allow_unsafe_bridge=True,
            arena=self.arena,
        )
