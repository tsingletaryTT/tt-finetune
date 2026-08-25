"""Exact fused frozen-Linear + LoRA autograd integration for P150a.

The native boundary replaces both the base Linear and LoRA projection in
forward, then replaces frozen-base dX plus LoRA dH/dB in backward.  Frozen dW
is never requested, allocated, or computed.  Small dX-LoRA, dA, and the final
dX add remain generic exact operations until fusing them proves faster.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model_workloads import (
    EXACT_BASE_DX_K_PHASES,
    EXACT_BASE_K_PHASES,
)
from .native_lora_autograd import _sync_custom_boundary
from .tt_lang_frozen_linear_lora import (
    build_tt_lang_frozen_linear_lora_forward_prototype,
)
from .tt_lang_frozen_linear_lora_backward import (
    build_tt_lang_frozen_linear_lora_backward,
)
from .tt_lang_frozen_lora_backward_tail import (
    build_tt_lang_frozen_lora_backward_tail,
)
from .tt_lang_lora import TILE_SIZE


@dataclass
class NativeFrozenLoRADeviceArena:
    frozen_weight_transposed: torch.Tensor
    forward_arena_init: torch.Tensor
    backward_arena_init: torch.Tensor
    backward_tail_arena_init: torch.Tensor


def exact_base_k_phase(
    input_features: int, output_features: int
) -> int | None:
    """Return a silicon-validated exact schedule, or None for fallback."""

    return EXACT_BASE_K_PHASES.get(
        (input_features, output_features)
    )


def exact_base_dx_k_phase(
    input_features: int, output_features: int
) -> int | None:
    """Return the exact dX phase for an original forward projection."""

    return EXACT_BASE_DX_K_PHASES.get(
        (input_features, output_features)
    )


def _validate_shapes(
    inputs: torch.Tensor,
    frozen_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> tuple[int, int, int, int]:
    if any(
        tensor.ndim != 2
        for tensor in (inputs, frozen_weight, lora_a, lora_b)
    ):
        raise ValueError("native frozen LoRA requires rank-2 tensors")
    tokens, input_features = inputs.shape
    output_features, weight_features = frozen_weight.shape
    physical_rank, a_features = lora_a.shape
    b_outputs, b_rank = lora_b.shape
    if (
        weight_features != input_features
        or a_features != input_features
        or b_outputs != output_features
        or b_rank != physical_rank
    ):
        raise ValueError("incompatible frozen Linear + LoRA shapes")
    dimensions = (tokens, input_features, output_features, physical_rank)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError("all native fused dimensions must be positive /32")
    if physical_rank != TILE_SIZE:
        raise ValueError("native fused LoRA requires physical rank 32")
    if exact_base_k_phase(input_features, output_features) is None:
        raise ValueError(
            "no bit-exact native frozen-Linear schedule is registered for "
            f"{input_features}->{output_features}"
        )
    if exact_base_dx_k_phase(input_features, output_features) is None:
        raise ValueError(
            "no bit-exact native frozen-Linear dX schedule is registered "
            f"for {input_features}->{output_features}"
        )
    return tokens, input_features, output_features, physical_rank


def make_native_frozen_lora_device_arena(
    inputs: torch.Tensor,
    frozen_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> NativeFrozenLoRADeviceArena:
    tokens, input_features, output_features, physical_rank = (
        _validate_shapes(inputs, frozen_weight, lora_a, lora_b)
    )

    def device_zeros(shape: tuple[int, int]) -> torch.Tensor:
        return torch.zeros(shape, dtype=inputs.dtype).to(inputs.device)

    return NativeFrozenLoRADeviceArena(
        frozen_weight_transposed=(
            frozen_weight.transpose(0, 1).contiguous()
        ),
        forward_arena_init=device_zeros(
            (tokens, output_features + physical_rank)
        ),
        backward_arena_init=device_zeros(
            (
                tokens + output_features,
                input_features + physical_rank,
            )
        ),
        backward_tail_arena_init=device_zeros(
            (tokens + physical_rank, input_features)
        ),
    )


class NativeExactFrozenLoRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        frozen_weight: torch.Tensor,
        frozen_weight_transposed: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scale: float,
        sync_boundaries: bool,
        forward_arena_init: torch.Tensor,
        backward_arena_init: torch.Tensor,
        backward_tail_arena_init: torch.Tensor,
        full_native_tail: bool,
    ) -> torch.Tensor:
        tokens, input_features, output_features, physical_rank = (
            _validate_shapes(inputs, frozen_weight, lora_a, lora_b)
        )
        base_k_phase = exact_base_k_phase(
            input_features, output_features
        )
        assert base_k_phase is not None
        operation = build_tt_lang_frozen_linear_lora_forward_prototype(
            scale=float(scale),
            tokens=tokens,
            input_features=input_features,
            output_features=output_features,
            physical_rank=physical_rank,
            n_tiles_per_block=1,
            base_k_tiles_per_phase=base_k_phase,
            base_partial_fp32=True,
            math_approx_mode=False,
            exact_reduction=True,
        )
        arena = operation(
            inputs,
            frozen_weight_transposed,
            lora_a,
            lora_b,
            forward_arena_init,
        )
        if sync_boundaries:
            _sync_custom_boundary(arena)
        output = arena[:, :output_features]
        hidden = arena[:, output_features:]
        ctx.save_for_backward(
            inputs,
            frozen_weight,
            lora_a,
            lora_b,
            hidden,
            backward_arena_init,
            backward_tail_arena_init,
        )
        ctx.scale = float(scale)
        ctx.sync_boundaries = bool(sync_boundaries)
        ctx.full_native_tail = bool(full_native_tail)
        return output

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        (
            inputs,
            frozen_weight,
            lora_a,
            lora_b,
            hidden,
            backward_arena_init,
            backward_tail_arena_init,
        ) = ctx.saved_tensors
        tokens, input_features = inputs.shape
        output_features, physical_rank = lora_b.shape
        operation = build_tt_lang_frozen_linear_lora_backward(
            scale=ctx.scale,
            tokens=tokens,
            input_features=input_features,
            output_features=output_features,
            physical_rank=physical_rank,
            base_reduction_tiles_per_phase=exact_base_dx_k_phase(
                input_features, output_features
            ),
        )
        arena = operation(
            output_gradient,
            frozen_weight,
            lora_b,
            hidden,
            backward_arena_init,
        )
        if ctx.sync_boundaries:
            _sync_custom_boundary(arena)

        base_input_gradient = arena[:tokens, :input_features]
        hidden_gradient = arena[
            :tokens, input_features : input_features + physical_rank
        ]
        lora_b_gradient = arena[
            tokens : tokens + output_features, :physical_rank
        ]

        if ctx.full_native_tail:
            tail_operation = build_tt_lang_frozen_lora_backward_tail(
                tokens=tokens,
                input_features=input_features,
                physical_rank=physical_rank,
            )
            tail_arena = tail_operation(
                base_input_gradient,
                hidden_gradient,
                lora_a,
                inputs,
                backward_tail_arena_init,
            )
            if ctx.sync_boundaries:
                _sync_custom_boundary(tail_arena)
            input_gradient = tail_arena[:tokens, :input_features]
            lora_a_gradient = tail_arena[
                tokens : tokens + physical_rank, :input_features
            ]
        else:
            lora_input_gradient = hidden_gradient @ lora_a
            input_gradient = base_input_gradient + lora_input_gradient
            lora_a_gradient = (
                inputs.transpose(0, 1) @ hidden_gradient
            ).transpose(0, 1)

        return (
            input_gradient,
            None,  # frozen_weight: no dW
            None,  # cached transposed frozen weight
            lora_a_gradient,
            lora_b_gradient,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def native_exact_frozen_lora_linear(
    inputs: torch.Tensor,
    frozen_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
    *,
    arena: NativeFrozenLoRADeviceArena | None = None,
    sync_boundaries: bool = False,
    full_native_tail: bool = False,
) -> torch.Tensor:
    """Run the accepted exact native fused projection.

    Unsupported shapes raise here so a module-level policy can select the
    generic fallback before entering autograd.
    """

    if arena is None:
        arena = make_native_frozen_lora_device_arena(
            inputs, frozen_weight, lora_a, lora_b
        )
    return NativeExactFrozenLoRAFunction.apply(
        inputs,
        frozen_weight,
        arena.frozen_weight_transposed,
        lora_a,
        lora_b,
        float(scale),
        bool(sync_boundaries),
        arena.forward_arena_init,
        arena.backward_arena_init,
        arena.backward_tail_arena_init,
        bool(full_native_tail),
    )


class NativeExactFrozenLoRAExecutor:
    """Cache persistent device arenas per projection signature."""

    def __init__(
        self,
        *,
        sync_boundaries: bool = False,
        full_native_tail: bool = False,
    ):
        self.sync_boundaries = bool(sync_boundaries)
        self.full_native_tail = bool(full_native_tail)
        self.arena: NativeFrozenLoRADeviceArena | None = None
        self.signature: tuple[object, ...] | None = None

    def supports(
        self,
        inputs: torch.Tensor,
        frozen_weight: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
    ) -> bool:
        try:
            _validate_shapes(inputs, frozen_weight, lora_a, lora_b)
        except ValueError:
            return False
        return True

    def __call__(
        self,
        inputs: torch.Tensor,
        frozen_weight: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        _validate_shapes(inputs, frozen_weight, lora_a, lora_b)
        signature = (
            tuple(inputs.shape),
            tuple(frozen_weight.shape),
            tuple(lora_a.shape),
            tuple(lora_b.shape),
            inputs.dtype,
            inputs.device,
            id(frozen_weight),
        )
        if self.arena is None or self.signature != signature:
            self.arena = make_native_frozen_lora_device_arena(
                inputs, frozen_weight, lora_a, lora_b
            )
            self.signature = signature
        return native_exact_frozen_lora_linear(
            inputs,
            frozen_weight,
            lora_a,
            lora_b,
            scale,
            arena=self.arena,
            sync_boundaries=self.sync_boundaries,
            full_native_tail=self.full_native_tail,
        )


class FrozenLoRALinearDispatcher:
    """Policy boundary used by future module replacement/integration.

    ``auto`` selects native only for a silicon-validated shape already on an
    XLA device. CPU tensors and unknown shapes stay generic.
    ``native_experimental`` remains an explicit strict request.
    """

    _BACKENDS = {"auto", "generic", "native_experimental"}

    def __init__(
        self,
        backend: str = "auto",
        *,
        sync_boundaries: bool = False,
    ):
        if backend not in self._BACKENDS:
            raise ValueError(
                "backend must be auto, generic, or native_experimental"
            )
        self.backend = backend
        self.native = NativeExactFrozenLoRAExecutor(
            sync_boundaries=sync_boundaries
        )

    @staticmethod
    def generic(
        inputs: torch.Tensor,
        frozen_weight: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        hidden = inputs @ lora_a.transpose(0, 1)
        return (
            inputs @ frozen_weight.transpose(0, 1)
            + (hidden @ lora_b.transpose(0, 1)) * scale
        )

    def __call__(
        self,
        inputs: torch.Tensor,
        frozen_weight: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        if self.backend == "generic":
            return self.generic(
                inputs, frozen_weight, lora_a, lora_b, scale
            )
        if self.backend == "auto":
            if (
                inputs.device.type == "xla"
                and self.native.supports(
                    inputs, frozen_weight, lora_a, lora_b
                )
            ):
                return self.native(
                    inputs, frozen_weight, lora_a, lora_b, scale
                )
            return self.generic(
                inputs, frozen_weight, lora_a, lora_b, scale
            )
        if not self.native.supports(
            inputs, frozen_weight, lora_a, lora_b
        ):
            raise ValueError(
                "native_experimental was requested for an unvalidated "
                "shape; select generic/auto instead"
            )
        return self.native(
            inputs, frozen_weight, lora_a, lora_b, scale
        )
