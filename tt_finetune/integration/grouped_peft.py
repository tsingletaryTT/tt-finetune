"""PEFT-compatible installation of exact grouped LoRA transformer blocks.

The native representation is deliberately an execution view.  The ordinary
PEFT ``lora_A`` and ``lora_B`` parameters remain registered under their
original names, so optimizers, trainable-only checkpoints, resume, and
``save_pretrained`` continue to use the standard adapter format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
from types import MethodType
from typing import Iterable

import torch
from torch import nn

from tt_finetune.kernels.grouped_lora import GroupedProjectionLayout
from tt_finetune.kernels.model_workloads import (
    EXACT_BASE_DX_K_PHASES,
    EXACT_BASE_K_PHASES,
)
from tt_finetune.kernels.native_grouped_lora_autograd import (
    NativeGrouped2Arena,
    make_native_grouped2_arena,
    native_exact_grouped2_lora,
)
from tt_finetune.kernels.native_grouped_lora_pack import (
    native_pack_grouped2_logical_adapters,
)
from tt_finetune.kernels.tt_lang_lora import TILE_SIZE
from tt_finetune.kernels.tt_lang_swiglu import native_exact_swiglu
from tt_finetune.kernels.native_resident_mlp_autograd import (
    native_resident_swiglu_down,
)
from tt_finetune.kernels.native_full_mlp_autograd import (
    native_resident_full_mlp,
)
from tt_finetune.kernels.full_mlp_backward_planner import (
    full_mlp_backward_support_error,
    plan_full_mlp_backward,
)
from tt_finetune.kernels.tt_lang_resident_swiglu_down import (
    resident_swiglu_down_support_error,
)
from tt_finetune.kernels.tt_lang_resident_down_backward import (
    resident_down_backward_support_error,
)
from tt_finetune.kernels.tt_lang_resident_down_activation_backward import (
    resident_down_activation_backward_support_error,
)


SUPPORTED_BACKENDS = {"auto", "generic", "native_experimental"}
# Exactness is a property of the complete reduction tile grid, not only K/N
# feature widths.  Four token tiles (128 flattened tokens) are currently the
# only grouped backward schedule accepted by full-model silicon parity.
EXACT_GROUPED_BACKWARD_TOKEN_TILES = frozenset({4})
# These scratch values are fully overwritten, but each same-shaped DPS output
# must begin as a distinct graph value so lazy XLA cannot CSE storage across
# layers. Reserve sixteen BF16 bit-pattern slots per MLP (one base plus seven
# final outputs); arithmetic offsets on consecutive layer integers collide.
_RESIDENT_MLP_ARENA_NONCES = itertools.count(256, 16)


def _resident_mlp_nonce(code: int) -> float:
    """Map an integer code to a unique, exactly representable BF16 value."""

    exponent, mantissa = divmod(code, 128)
    return math.ldexp(1.0 + mantissa / 128.0, exponent)


@dataclass(frozen=True)
class GroupedLoRAInstallReport:
    backend: str
    installed_paths: tuple[str, ...]
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def installed_count(self) -> int:
        return len(self.installed_paths)


def _active_adapter(module: nn.Module) -> str | None:
    active = getattr(module, "active_adapter", "default")
    if isinstance(active, str):
        return active
    if isinstance(active, Iterable):
        values = tuple(active)
        return values[0] if len(values) == 1 else None
    return None


def _adapter_linear_parts(
    module: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
    """Return frozen W, logical A/B, scale, and rank for one PEFT linear."""

    adapter = _active_adapter(module)
    if adapter is None:
        raise ValueError("exact grouped execution requires one active adapter")
    try:
        base_layer = module.base_layer
        lora_a = module.lora_A[adapter].weight
        lora_b = module.lora_B[adapter].weight
        scale = float(module.scaling[adapter])
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("module is not a supported PEFT LoRA linear") from error
    weight = base_layer.weight
    if getattr(base_layer, "bias", None) is not None:
        raise ValueError("native grouped projection does not yet support bias")
    if bool(getattr(module, "disable_adapters", False)):
        raise ValueError("PEFT adapters are disabled")
    if bool(getattr(module, "merged", False)):
        raise ValueError("merged PEFT adapters are not supported")
    use_dora = getattr(module, "use_dora", {})
    dora_enabled = (
        use_dora.get(adapter, False)
        if hasattr(use_dora, "get")
        else use_dora[adapter]
        if adapter in use_dora
        else False
    )
    if bool(dora_enabled):
        raise ValueError("DoRA is not supported by the exact grouped kernel")
    dropouts = getattr(module, "lora_dropout", {})
    dropout = (
        dropouts.get(adapter)
        if hasattr(dropouts, "get")
        else dropouts[adapter]
        if adapter in dropouts
        else None
    )
    if dropout is not None and float(getattr(dropout, "p", 0.0)) != 0.0:
        raise ValueError("native grouped execution requires zero LoRA dropout")
    rank = int(lora_a.shape[0])
    if (
        weight.ndim != 2
        or lora_a.shape != (rank, weight.shape[1])
        or lora_b.shape != (weight.shape[0], rank)
    ):
        raise ValueError("unsupported PEFT LoRA tensor shapes")
    if rank <= 0 or rank > TILE_SIZE:
        raise ValueError("logical LoRA rank must be between 1 and 32")
    return weight, lora_a, lora_b, scale, rank


def _pad_lora_a(value: torch.Tensor) -> torch.Tensor:
    padding = TILE_SIZE - value.shape[0]
    if padding == 0:
        return value
    return torch.cat(
        (value, value.new_zeros((padding, value.shape[1]))),
        dim=0,
    )


def _pad_lora_b(value: torch.Tensor) -> torch.Tensor:
    padding = TILE_SIZE - value.shape[1]
    if padding == 0:
        return value
    return torch.cat(
        (value, value.new_zeros((value.shape[0], padding))),
        dim=1,
    )


def _pack_two_logical_adapters(
    first_a: torch.Tensor,
    first_b: torch.Tensor,
    second_a: torch.Tensor,
    second_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_a = _pad_lora_a(first_a)
    second_a = _pad_lora_a(second_a)
    first_b = _pad_lora_b(first_b)
    second_b = _pad_lora_b(second_b)
    grouped_a = torch.cat((first_a, second_a), dim=0)
    grouped_b = torch.cat(
        (
            torch.cat(
                (
                    first_b,
                    first_b.new_zeros((first_b.shape[0], TILE_SIZE)),
                ),
                dim=1,
            ),
            torch.cat(
                (
                    second_b.new_zeros((second_b.shape[0], TILE_SIZE)),
                    second_b,
                ),
                dim=1,
            ),
        ),
        dim=0,
    )
    return grouped_a, grouped_b


class NativeGroupedLlamaMLP(nn.Module):
    """Llama MLP whose gate/up PEFT projections share one native TT program."""

    def __init__(
        self,
        original: nn.Module,
        *,
        backend: str = "auto",
        native_backward: bool = True,
        grouped_native_phase_one: bool = False,
        full_native_tail: bool = False,
        grouped_native_tail: bool = False,
        resident_grouped_backward: bool = False,
        native_forward: bool = True,
        native_swiglu: bool = False,
        native_swiglu_decomposed: bool = False,
        resident_mlp_forward: bool = False,
        resident_mlp_fp32_activation: bool = False,
        resident_down_backward: bool = False,
        resident_activation_backward: bool = False,
        resident_down_activation_backward: bool = False,
        resident_full_mlp_backward: bool = False,
        native_packing: bool = False,
        direct_logical: bool = False,
        input_gradient_edge_order: tuple[int, int, int, int] = (
            2,
            3,
            1,
            0,
        ),
        separate_input_gradients: bool = True,
        grouped_base_input_gradient: bool = False,
    ):
        super().__init__()
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"backend must be one of {sorted(SUPPORTED_BACKENDS)}"
            )
        if direct_logical and not native_forward:
            raise ValueError(
                "direct logical adapter reads require the native forward"
            )
        if native_swiglu and resident_mlp_forward:
            raise ValueError(
                "native_swiglu and resident_mlp_forward are mutually exclusive"
            )
        if resident_activation_backward and not resident_mlp_forward:
            raise ValueError(
                "resident activation backward requires resident MLP forward"
            )
        if resident_down_activation_backward and not resident_mlp_forward:
            raise ValueError(
                "resident down/activation backward requires resident MLP "
                "forward"
            )
        if resident_full_mlp_backward and not resident_mlp_forward:
            raise ValueError(
                "resident full MLP backward requires resident MLP forward"
            )
        if native_swiglu or resident_mlp_forward:
            activation_name = type(original.act_fn).__name__.lower()
            function_name = getattr(original.act_fn, "__name__", "").lower()
            if "silu" not in activation_name and function_name != "silu":
                raise ValueError("native SwiGLU requires a SiLU activation")
        gate = _adapter_linear_parts(original.gate_proj)
        up = _adapter_linear_parts(original.up_proj)
        if gate[0].shape[1] != up[0].shape[1]:
            raise ValueError("gate/up projections must share input width")
        if gate[3] != up[3]:
            raise ValueError("gate/up LoRA scales must match")
        if gate[4] != up[4]:
            raise ValueError("gate/up logical ranks must match")

        input_features = gate[0].shape[1]
        output_features = (gate[0].shape[0], up[0].shape[0])
        if (
            input_features % TILE_SIZE
            or any(value % TILE_SIZE for value in output_features)
        ):
            raise ValueError("native grouped dimensions must be tile aligned")
        phase = EXACT_BASE_K_PHASES.get(
            (input_features, output_features[0])
        )
        if phase is None or EXACT_BASE_K_PHASES.get(
            (input_features, output_features[1])
        ) != phase:
            raise ValueError("gate/up shape has no exact silicon schedule")

        # Keep these children directly under the MLP so every PEFT state key
        # remains byte-for-byte identical to the unwrapped model.
        self.gate_proj = original.gate_proj
        self.up_proj = original.up_proj
        self.down_proj = original.down_proj
        self.act_fn = original.act_fn
        if hasattr(original, "config"):
            self.config = original.config

        self.backend = backend
        self.native_backward = bool(native_backward)
        self.grouped_native_phase_one = bool(
            grouped_native_phase_one
        )
        self.full_native_tail = bool(full_native_tail)
        self.grouped_native_tail = bool(grouped_native_tail)
        self.resident_grouped_backward = bool(
            resident_grouped_backward
        )
        self.native_forward = bool(native_forward)
        self.native_swiglu = bool(native_swiglu)
        self.native_swiglu_decomposed = bool(
            native_swiglu_decomposed
        )
        self.resident_mlp_forward = bool(resident_mlp_forward)
        self.resident_mlp_fp32_activation = bool(
            resident_mlp_fp32_activation
        )
        self.resident_down_backward = bool(resident_down_backward)
        self.resident_activation_backward = bool(
            resident_activation_backward
        )
        self.resident_down_activation_backward = bool(
            resident_down_activation_backward
        )
        self.resident_full_mlp_backward = bool(
            resident_full_mlp_backward
        )
        self._resident_mlp_arena_nonce_code = (
            next(_RESIDENT_MLP_ARENA_NONCES)
            if self.resident_mlp_forward
            else 0
        )
        self._resident_mlp_arena_nonce = (
            _resident_mlp_nonce(self._resident_mlp_arena_nonce_code)
            if self.resident_mlp_forward
            else 0.0
        )
        self.native_packing = bool(native_packing)
        self.direct_logical = bool(direct_logical)
        self.input_gradient_edge_order = tuple(
            input_gradient_edge_order
        )
        self.separate_input_gradients = bool(separate_input_gradients)
        self.grouped_base_input_gradient = bool(
            grouped_base_input_gradient
        )
        self.input_features = input_features
        self.output_features = output_features
        self.logical_rank = gate[4]
        self.scale = gate[3]
        self.base_k_tiles_per_phase = phase
        self.layout = GroupedProjectionLayout(
            output_features=output_features,
            physical_rank_per_projection=TILE_SIZE,
        )
        self.register_buffer(
            "_native_grouped_weight",
            torch.cat((gate[0].detach(), up[0].detach()), dim=0),
            persistent=False,
        )
        self._native_arena: NativeGrouped2Arena | None = None
        self._native_arena_signature: tuple[object, ...] | None = None
        self._native_pack_arena: torch.Tensor | None = None
        self._native_pack_signature: tuple[object, ...] | None = None
        self._native_swiglu_arena: torch.Tensor | None = None
        self._native_swiglu_signature: tuple[object, ...] | None = None
        self._resident_mlp_arena: torch.Tensor | None = None
        self._resident_mlp_backward_arena: torch.Tensor | None = None
        self._resident_mlp_da_arena: torch.Tensor | None = None
        self._resident_mlp_activation_backward_arena: (
            torch.Tensor | None
        ) = None
        self._resident_mlp_down_activation_backward_arena: (
            torch.Tensor | None
        ) = None
        self._resident_full_mlp_backward_outputs: (
            tuple[torch.Tensor, ...] | None
        ) = None
        self._resident_mlp_product_recompute_arena: (
            torch.Tensor | None
        ) = None
        self._resident_mlp_signature: tuple[object, ...] | None = None
        if self.resident_mlp_forward:
            down = _adapter_linear_parts(self.down_proj)
            native_down_weight = (
                down[0].detach().transpose(0, 1).contiguous()
            )
        else:
            native_down_weight = None
        self.register_buffer(
            "_native_down_weight_transposed",
            native_down_weight,
            persistent=False,
        )
        self.last_execution_path = "not_run"

    def _apply(self, fn):
        self._native_arena = None
        self._native_arena_signature = None
        self._native_pack_arena = None
        self._native_pack_signature = None
        self._native_swiglu_arena = None
        self._native_swiglu_signature = None
        self._resident_mlp_arena = None
        self._resident_mlp_backward_arena = None
        self._resident_mlp_da_arena = None
        self._resident_mlp_activation_backward_arena = None
        self._resident_mlp_down_activation_backward_arena = None
        self._resident_full_mlp_backward_outputs = None
        self._resident_mlp_product_recompute_arena = None
        self._resident_mlp_signature = None
        return super()._apply(fn)

    def _pack_adapters(
        self,
        gate_a: torch.Tensor,
        gate_b: torch.Tensor,
        up_a: torch.Tensor,
        up_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.native_packing:
            return _pack_two_logical_adapters(
                gate_a, gate_b, up_a, up_b
            )
        if self.output_features[0] != self.output_features[1]:
            raise RuntimeError(
                "native logical packing currently requires equal outputs"
            )
        signature = (
            gate_a.dtype,
            gate_a.device,
            self.input_features,
            self.output_features,
        )
        if (
            self._native_pack_arena is None
            or signature != self._native_pack_signature
        ):
            rows = 64 + sum(self.output_features)
            columns = max(self.input_features, 64)
            self._native_pack_arena = torch.zeros(
                (rows, columns),
                dtype=gate_a.dtype,
                device=gate_a.device,
            )
            self._native_pack_signature = signature
        return native_pack_grouped2_logical_adapters(
            gate_a,
            gate_b,
            up_a,
            up_b,
            self._native_pack_arena,
        )

    def _generic_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.last_execution_path = "generic"
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )

    def _native_rejection(self, hidden_states: torch.Tensor) -> str | None:
        if hidden_states.device.type != "xla":
            return "native grouped execution requires an XLA tensor"
        if hidden_states.dtype != torch.bfloat16:
            return "native grouped execution currently requires bfloat16"
        if hidden_states.shape[-1] != self.input_features:
            return "hidden width changed after native block installation"
        tokens = hidden_states.numel() // self.input_features
        if tokens % TILE_SIZE:
            return "flattened token count must be divisible by 32"
        if (
            self.native_backward
            and tokens // TILE_SIZE
            not in EXACT_GROUPED_BACKWARD_TOKEN_TILES
        ):
            return (
                "no exact grouped backward schedule for "
                f"{tokens // TILE_SIZE} token tiles; validated: "
                f"{sorted(EXACT_GROUPED_BACKWARD_TOKEN_TILES)}"
            )
        try:
            gate = _adapter_linear_parts(self.gate_proj)
            up = _adapter_linear_parts(self.up_proj)
        except ValueError as error:
            return str(error)
        if gate[3] != self.scale or up[3] != self.scale:
            return "LoRA scale changed after native block installation"
        if gate[4] != self.logical_rank or up[4] != self.logical_rank:
            return "LoRA rank changed after native block installation"
        return None

    def _arena(
        self,
        flat_inputs: torch.Tensor,
        grouped_a: torch.Tensor,
        grouped_b: torch.Tensor,
        *,
        validate_packed: bool = True,
    ) -> NativeGrouped2Arena:
        signature = (
            flat_inputs.shape[0],
            flat_inputs.dtype,
            flat_inputs.device,
            validate_packed,
            self.resident_full_mlp_backward,
        )
        if (
            self._native_arena is None
            or signature != self._native_arena_signature
        ):
            self._native_arena = make_native_grouped2_arena(
                flat_inputs,
                self._native_grouped_weight,
                grouped_a,
                grouped_b,
                self.layout,
                validate_packed=validate_packed,
                forward_arena_fill_value=(
                    self._resident_mlp_arena_nonce
                    if self.resident_full_mlp_backward
                    else 0.0
                ),
                input_checkpoint_columns=(
                    self.input_features
                    if self.resident_full_mlp_backward
                    else 0
                ),
            )
            self._native_arena_signature = signature
        return self._native_arena

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.backend == "generic":
            return self._generic_forward(hidden_states)
        rejection = self._native_rejection(hidden_states)
        if rejection is not None:
            if self.backend == "native_experimental":
                raise RuntimeError(rejection)
            return self._generic_forward(hidden_states)

        gate = _adapter_linear_parts(self.gate_proj)
        up = _adapter_linear_parts(self.up_proj)
        if self.direct_logical:
            grouped_a, grouped_b = gate[1], gate[2]
        else:
            grouped_a, grouped_b = self._pack_adapters(
                gate[1], gate[2], up[1], up[2]
            )
        original_shape = hidden_states.shape
        flat_inputs = hidden_states.reshape(-1, self.input_features)
        if self.resident_full_mlp_backward:
            down = _adapter_linear_parts(self.down_proj)
            gate_a = _pad_lora_a(gate[1])
            gate_b = _pad_lora_b(gate[2])
            up_a = _pad_lora_a(up[1])
            up_b = _pad_lora_b(up[2])
            down_a = _pad_lora_a(down[1])
            down_b = _pad_lora_b(down[2])
            support_error = full_mlp_backward_support_error(
                tokens=flat_inputs.shape[0],
                hidden_features=self.input_features,
                intermediate_features=self.output_features[0],
                physical_rank=down_a.shape[0],
            )
            if support_error is not None:
                if self.backend == "native_experimental":
                    raise RuntimeError(support_error)
                return self._generic_forward(hidden_states)
            signature = (
                flat_inputs.shape[0],
                flat_inputs.dtype,
                flat_inputs.device,
                self.input_features,
                self.output_features[0],
                "full_mlp_owner_v1",
            )
            if (
                self._resident_mlp_signature != signature
                or self._resident_mlp_arena is None
                or self._resident_full_mlp_backward_outputs is None
            ):
                self._resident_mlp_arena = torch.full(
                    (
                        flat_inputs.shape[0],
                        self.input_features + down_a.shape[0],
                    ),
                    self._resident_mlp_arena_nonce,
                    dtype=flat_inputs.dtype,
                    device=flat_inputs.device,
                )
                output_shapes = (
                    (flat_inputs.shape[0], self.input_features),
                    (TILE_SIZE, self.input_features),
                    (self.output_features[0], TILE_SIZE),
                    (TILE_SIZE, self.input_features),
                    (self.output_features[0], TILE_SIZE),
                    (TILE_SIZE, self.output_features[0]),
                    (self.input_features, TILE_SIZE),
                )
                self._resident_full_mlp_backward_outputs = tuple(
                    torch.full(
                        shape,
                        _resident_mlp_nonce(
                            self._resident_mlp_arena_nonce_code + index + 1
                        ),
                        dtype=flat_inputs.dtype,
                        device=flat_inputs.device,
                    )
                    for index, shape in enumerate(output_shapes)
                )
                self._resident_mlp_signature = signature
            native_arena = self._arena(
                flat_inputs,
                grouped_a,
                grouped_b,
                validate_packed=not self.direct_logical,
            )
            assert self._native_down_weight_transposed is not None
            assert self._resident_mlp_arena is not None
            assert self._resident_full_mlp_backward_outputs is not None
            result = native_resident_full_mlp(
                flat_inputs,
                native_arena.frozen_weight_transposed,
                gate[0],
                gate_a,
                gate_b,
                up[0],
                up_a,
                up_b,
                gate[3],
                self.base_k_tiles_per_phase,
                down[0],
                self._native_down_weight_transposed,
                down_a,
                down_b,
                down[3],
                native_arena.forward_arena_init,
                self._resident_mlp_arena,
                *self._resident_full_mlp_backward_outputs,
            )
            self.last_execution_path = "native_resident_full_mlp_owner"
            return result.reshape(*original_shape[:-1], result.shape[-1])
        grouped_output = native_exact_grouped2_lora(
            flat_inputs,
            self._native_grouped_weight,
            grouped_a,
            grouped_b,
            self.scale,
            self.layout,
            base_k_tiles_per_phase=self.base_k_tiles_per_phase,
            arena=self._arena(
                flat_inputs,
                grouped_a,
                grouped_b,
                validate_packed=not self.direct_logical,
            ),
            peft_input_gradient_order=True,
            native_backward=self.native_backward,
            grouped_native_phase_one=self.grouped_native_phase_one,
            full_native_tail=self.full_native_tail,
            grouped_native_tail=self.grouped_native_tail,
            resident_grouped_backward=(
                self.resident_grouped_backward
            ),
            native_forward=self.native_forward,
            logical_rank_per_member=self.logical_rank,
            logical_adapters=(gate[1], gate[2], up[1], up[2]),
            direct_logical=self.direct_logical,
            separate_input_gradients=self.separate_input_gradients,
            input_gradient_edge_order=self.input_gradient_edge_order,
            grouped_base_input_gradient=self.grouped_base_input_gradient,
        )
        gate_output, up_output = self.layout.split_outputs(grouped_output)
        if self.resident_mlp_forward:
            down = _adapter_linear_parts(self.down_proj)
            down_a = _pad_lora_a(down[1])
            down_b = _pad_lora_b(down[2])
            support_error = resident_swiglu_down_support_error(
                tokens=gate_output.shape[0],
                hidden_features=down[0].shape[0],
                intermediate_features=gate_output.shape[1],
                physical_rank=down_a.shape[0],
            )
            if support_error is not None:
                if self.backend == "native_experimental":
                    raise RuntimeError(support_error)
                activated = self.act_fn(gate_output) * up_output
                result = self.down_proj(activated)
                self.last_execution_path = "native_grouped2_resident_mlp_fallback"
                return result.reshape(*original_shape[:-1], result.shape[-1])
            if (
                self.resident_down_backward
                or self.resident_down_activation_backward
            ):
                backward_phase = EXACT_BASE_DX_K_PHASES.get(
                    (gate_output.shape[1], down[0].shape[0])
                )
                if backward_phase is None:
                    support_error = (
                        "no exact resident down backward reduction schedule"
                    )
                else:
                    support = (
                        resident_down_activation_backward_support_error
                        if self.resident_down_activation_backward
                        else resident_down_backward_support_error
                    )
                    support_error = support(
                        tokens=gate_output.shape[0],
                        input_features=gate_output.shape[1],
                        output_features=down[0].shape[0],
                        physical_rank=down_a.shape[0],
                        base_reduction_tiles_per_phase=backward_phase,
                    )
                if support_error is not None:
                    if self.backend == "native_experimental":
                        raise RuntimeError(support_error)
                    activated = self.act_fn(gate_output) * up_output
                    result = self.down_proj(activated)
                    self.last_execution_path = (
                        "native_grouped2_resident_backward_fallback"
                    )
                    return result.reshape(
                        *original_shape[:-1], result.shape[-1]
                    )
            signature = (
                gate_output.shape[0], gate_output.dtype, gate_output.device,
                down[0].shape[0], down_a.shape[0],
            )
            if (
                self._resident_mlp_arena is None
                or self._resident_mlp_signature != signature
            ):
                self._resident_mlp_arena = torch.full(
                    (gate_output.shape[0], down[0].shape[0] + down_a.shape[0]),
                    self._resident_mlp_arena_nonce,
                    dtype=gate_output.dtype,
                    device=gate_output.device,
                )
                backward_rows = (
                    gate_output.shape[0]
                    + down[0].shape[0]
                    + (
                        down_a.shape[0]
                        if (
                            self.resident_down_backward
                            or self.resident_down_activation_backward
                        )
                        else 0
                    )
                )
                backward_columns = (
                    gate_output.shape[1]
                    if (
                        self.resident_down_backward
                        or self.resident_down_activation_backward
                    )
                    else gate_output.shape[1] + down_a.shape[0]
                )
                self._resident_mlp_backward_arena = torch.full(
                    (backward_rows, backward_columns),
                    self._resident_mlp_arena_nonce,
                    dtype=gate_output.dtype,
                    device=gate_output.device,
                )
                self._resident_mlp_da_arena = torch.full(
                    (down_a.shape[0], gate_output.shape[1]),
                    self._resident_mlp_arena_nonce,
                    dtype=gate_output.dtype,
                    device=gate_output.device,
                )
                self._resident_mlp_activation_backward_arena = torch.full(
                    (2 * gate_output.shape[0], gate_output.shape[1]),
                    self._resident_mlp_arena_nonce,
                    dtype=gate_output.dtype,
                    device=gate_output.device,
                )
                if self.resident_down_activation_backward:
                    self._resident_mlp_down_activation_backward_arena = (
                        torch.full(
                            (
                                2 * gate_output.shape[0]
                                + down_a.shape[0]
                                + down[0].shape[0],
                                gate_output.shape[1],
                            ),
                            self._resident_mlp_arena_nonce,
                            dtype=gate_output.dtype,
                            device=gate_output.device,
                        )
                    )
                self._resident_mlp_product_recompute_arena = torch.full(
                    gate_output.shape,
                    self._resident_mlp_arena_nonce,
                    dtype=gate_output.dtype,
                    device=gate_output.device,
                )
                self._resident_mlp_signature = signature
            assert self._resident_mlp_backward_arena is not None
            assert self._resident_mlp_da_arena is not None
            assert (
                self._resident_mlp_activation_backward_arena is not None
            )
            if self.resident_down_activation_backward:
                assert (
                    self._resident_mlp_down_activation_backward_arena
                    is not None
                )
            assert self._resident_mlp_product_recompute_arena is not None
            result = native_resident_swiglu_down(
                gate_output,
                up_output,
                down[0],
                self._native_down_weight_transposed,
                down_a,
                down_b,
                down[3],
                self._resident_mlp_arena,
                self._resident_mlp_backward_arena,
                self._resident_mlp_da_arena,
                self._resident_mlp_activation_backward_arena,
                (
                    self._resident_mlp_down_activation_backward_arena
                    if self._resident_mlp_down_activation_backward_arena
                    is not None
                    else self._resident_mlp_activation_backward_arena
                ),
                self._resident_mlp_product_recompute_arena,
                self.resident_mlp_fp32_activation,
                not self.resident_mlp_fp32_activation,
                self.resident_down_backward,
                self.resident_activation_backward,
                self.resident_down_activation_backward,
            )
            self.last_execution_path = (
                "native_grouped2_resident_mlp_mixed_backward"
                if self.resident_down_activation_backward
                else "native_grouped2_resident_mlp_forward"
            )
            return result.reshape(*original_shape[:-1], result.shape[-1])
        if self.native_swiglu:
            swiglu_signature = (
                gate_output.shape,
                gate_output.dtype,
                gate_output.device,
            )
            if (
                self._native_swiglu_arena is None
                or self._native_swiglu_signature != swiglu_signature
            ):
                self._native_swiglu_arena = torch.zeros_like(gate_output)
                self._native_swiglu_signature = swiglu_signature
            activated = native_exact_swiglu(
                gate_output,
                up_output,
                self._native_swiglu_arena,
                diagnostic_stage=(
                    "full"
                    if self.native_swiglu_decomposed
                    else "full_builtin"
                ),
            )
            self.last_execution_path = "native_grouped2_swiglu"
        else:
            activated = self.act_fn(gate_output) * up_output
            self.last_execution_path = "native_grouped2"
        result = self.down_proj(activated)
        return result.reshape(*original_shape[:-1], result.shape[-1])


class NativeGroupedKVCoordinator:
    """Coordinate consecutive PEFT K/V calls without changing state keys.

    Transformers' Llama attention calls ``k_proj(hidden_states)`` immediately
    followed by ``v_proj(hidden_states)``.  The first call launches one
    grouped native program and the second returns the saved sibling output.
    The projection modules themselves remain registered in their original
    locations, so PEFT names and checkpoints are unchanged.
    """

    def __init__(
        self,
        attention: nn.Module,
        *,
        backend: str,
        native_backward: bool,
        grouped_native_phase_one: bool,
        full_native_tail: bool,
        grouped_native_tail: bool,
        resident_grouped_backward: bool,
        input_gradient_edge_order: tuple[int, int, int, int],
        separate_input_gradients: bool,
        grouped_base_input_gradient: bool,
    ):
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"backend must be one of {sorted(SUPPORTED_BACKENDS)}"
            )
        key = _adapter_linear_parts(attention.k_proj)
        value = _adapter_linear_parts(attention.v_proj)
        if key[0].shape != value[0].shape:
            raise ValueError(
                "native K/V grouping requires equal projection shapes"
            )
        if key[3] != value[3]:
            raise ValueError("K/V LoRA scales must match")
        if key[4] != value[4]:
            raise ValueError("K/V LoRA ranks must match")
        input_features = key[0].shape[1]
        output_features = key[0].shape[0]
        if input_features % TILE_SIZE or output_features % TILE_SIZE:
            raise ValueError("native K/V dimensions must be tile aligned")
        phase = EXACT_BASE_K_PHASES.get(
            (input_features, output_features)
        )
        if phase is None:
            raise ValueError("K/V shape has no exact silicon schedule")

        self.attention = attention
        self.backend = backend
        self.native_backward = bool(native_backward)
        self.grouped_native_phase_one = bool(
            grouped_native_phase_one
        )
        self.full_native_tail = bool(full_native_tail)
        self.grouped_native_tail = bool(grouped_native_tail)
        self.resident_grouped_backward = bool(
            resident_grouped_backward
        )
        self.input_gradient_edge_order = tuple(
            input_gradient_edge_order
        )
        self.separate_input_gradients = bool(separate_input_gradients)
        self.grouped_base_input_gradient = bool(
            grouped_base_input_gradient
        )
        self.input_features = input_features
        self.output_features = output_features
        self.logical_rank = key[4]
        self.scale = key[3]
        self.base_k_tiles_per_phase = phase
        # Every mutation-style custom-call output must be a distinct graph
        # value. Otherwise lazy XLA may CSE equal zero arenas across decoder
        # layers; a later K/V backward then overwrites an earlier layer's dB.
        # Reserve the same 16-value BF16 identity block used by resident MLPs.
        self._arena_nonce_code = next(_RESIDENT_MLP_ARENA_NONCES)
        self._arena_nonce = _resident_mlp_nonce(self._arena_nonce_code)
        self.layout = GroupedProjectionLayout(
            output_features=(output_features, output_features),
            physical_rank_per_projection=TILE_SIZE,
        )
        self.generic_forwards = (
            attention.k_proj.forward,
            attention.v_proj.forward,
        )
        self._arena: NativeGrouped2Arena | None = None
        self._arena_signature: tuple[object, ...] | None = None
        self._pending_input: torch.Tensor | None = None
        self._pending_value: torch.Tensor | None = None
        self.last_execution_path = "not_run"

        attention.register_buffer(
            "_native_grouped_kv_weight",
            torch.cat((key[0].detach(), value[0].detach()), dim=0),
            persistent=False,
        )

    def _clear_pending(self) -> None:
        self._pending_input = None
        self._pending_value = None

    def _rejection(
        self,
        hidden_states: torch.Tensor,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> str | None:
        if args or kwargs:
            return "native K/V grouping does not support projection kwargs"
        if hidden_states.device.type != "xla":
            return "native K/V grouping requires an XLA tensor"
        if hidden_states.dtype != torch.bfloat16:
            return "native K/V grouping currently requires bfloat16"
        if hidden_states.shape[-1] != self.input_features:
            return "K/V hidden width changed after native installation"
        tokens = hidden_states.numel() // self.input_features
        if tokens % TILE_SIZE:
            return "flattened K/V token count must be divisible by 32"
        if (
            self.native_backward
            and tokens // TILE_SIZE
            not in EXACT_GROUPED_BACKWARD_TOKEN_TILES
        ):
            return (
                "no exact grouped K/V backward schedule for "
                f"{tokens // TILE_SIZE} token tiles; validated: "
                f"{sorted(EXACT_GROUPED_BACKWARD_TOKEN_TILES)}"
            )
        try:
            key = _adapter_linear_parts(self.attention.k_proj)
            value = _adapter_linear_parts(self.attention.v_proj)
        except ValueError as error:
            return str(error)
        if key[3] != self.scale or value[3] != self.scale:
            return "K/V LoRA scale changed after native installation"
        if key[4] != self.logical_rank or value[4] != self.logical_rank:
            return "K/V LoRA rank changed after native installation"
        return None

    def _native_arena(
        self,
        flat_inputs: torch.Tensor,
        key_a: torch.Tensor,
        key_b: torch.Tensor,
    ) -> NativeGrouped2Arena:
        signature = (
            flat_inputs.shape[0],
            flat_inputs.dtype,
            flat_inputs.device,
        )
        if self._arena is None or signature != self._arena_signature:
            self._arena = make_native_grouped2_arena(
                flat_inputs,
                self.attention._native_grouped_kv_weight,
                key_a,
                key_b,
                self.layout,
                validate_packed=False,
                forward_arena_fill_value=self._arena_nonce,
                scratch_arena_fill_values=tuple(
                    _resident_mlp_nonce(self._arena_nonce_code + slot)
                    for slot in range(1, 8)
                ),
            )
            self._arena_signature = signature
        return self._arena

    def project(
        self,
        member: int,
        hidden_states: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        if member == 1:
            if (
                self._pending_value is not None
                and self._pending_input is hidden_states
            ):
                result = self._pending_value
                self._clear_pending()
                return result
            self._clear_pending()
            return self.generic_forwards[1](
                hidden_states, *args, **kwargs
            )

        # A stale value means a former attention call did not reach v_proj.
        # Never let it leak into another forward.
        self._clear_pending()
        rejection = self._rejection(hidden_states, args, kwargs)
        if rejection is not None:
            if self.backend == "native_experimental":
                raise RuntimeError(rejection)
            self.last_execution_path = "generic"
            return self.generic_forwards[0](
                hidden_states, *args, **kwargs
            )

        key = _adapter_linear_parts(self.attention.k_proj)
        value = _adapter_linear_parts(self.attention.v_proj)
        original_shape = hidden_states.shape
        flat_inputs = hidden_states.reshape(-1, self.input_features)
        grouped_output = native_exact_grouped2_lora(
            flat_inputs,
            self.attention._native_grouped_kv_weight,
            key[1],
            key[2],
            self.scale,
            self.layout,
            base_k_tiles_per_phase=self.base_k_tiles_per_phase,
            arena=self._native_arena(flat_inputs, key[1], key[2]),
            # Separate PEFT branches accumulate V first, followed by the K
            # LoRA path and finally the K frozen-base path. Match that
            # observable BF16 association exactly.
            peft_input_gradient_order=True,
            native_backward=self.native_backward,
            grouped_native_phase_one=self.grouped_native_phase_one,
            full_native_tail=self.full_native_tail,
            grouped_native_tail=self.grouped_native_tail,
            resident_grouped_backward=(
                self.resident_grouped_backward
            ),
            native_forward=True,
            logical_rank_per_member=self.logical_rank,
            logical_adapters=(
                key[1],
                key[2],
                value[1],
                value[2],
            ),
            direct_logical=True,
            separate_input_gradients=self.separate_input_gradients,
            input_gradient_edge_order=self.input_gradient_edge_order,
            grouped_base_input_gradient=self.grouped_base_input_gradient,
        )
        key_output, value_output = self.layout.split_outputs(
            grouped_output
        )
        key_output = key_output.reshape(
            *original_shape[:-1], self.output_features
        )
        value_output = value_output.reshape(
            *original_shape[:-1], self.output_features
        )
        self._pending_input = hidden_states
        self._pending_value = value_output
        self.last_execution_path = "native_grouped_kv"
        return key_output


def _native_grouped_key_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    *args: object,
    **kwargs: object,
) -> torch.Tensor:
    return module._tt_native_kv_coordinator.project(
        0, hidden_states, *args, **kwargs
    )


def _native_grouped_value_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    *args: object,
    **kwargs: object,
) -> torch.Tensor:
    return module._tt_native_kv_coordinator.project(
        1, hidden_states, *args, **kwargs
    )


def _install_native_grouped_kv(
    attention: nn.Module,
    *,
    backend: str,
    native_backward: bool,
    grouped_native_phase_one: bool,
    full_native_tail: bool,
    grouped_native_tail: bool,
    resident_grouped_backward: bool,
    input_gradient_edge_order: tuple[int, int, int, int],
    separate_input_gradients: bool,
    grouped_base_input_gradient: bool,
) -> NativeGroupedKVCoordinator:
    if hasattr(attention.k_proj, "_tt_native_kv_coordinator"):
        raise ValueError("native K/V grouping is already installed")
    coordinator = NativeGroupedKVCoordinator(
        attention,
        backend=backend,
        native_backward=native_backward,
        grouped_native_phase_one=grouped_native_phase_one,
        full_native_tail=full_native_tail,
        grouped_native_tail=grouped_native_tail,
        resident_grouped_backward=resident_grouped_backward,
        input_gradient_edge_order=input_gradient_edge_order,
        separate_input_gradients=separate_input_gradients,
        grouped_base_input_gradient=grouped_base_input_gradient,
    )
    # A plain coordinator attribute and instance-bound methods do not register
    # new PyTorch modules or parameters, preserving every PEFT state key.
    attention.k_proj._tt_native_kv_coordinator = coordinator
    attention.v_proj._tt_native_kv_coordinator = coordinator
    attention.k_proj.forward = MethodType(
        _native_grouped_key_forward, attention.k_proj
    )
    attention.v_proj.forward = MethodType(
        _native_grouped_value_forward, attention.v_proj
    )
    attention._tt_native_kv_coordinator = coordinator
    return coordinator


def install_native_grouped_lora_blocks(
    model: nn.Module,
    *,
    backend: str = "auto",
    native_backward: bool = True,
    grouped_native_phase_one: bool = False,
    full_native_tail: bool = False,
    grouped_native_tail: bool = False,
    resident_grouped_backward: bool = False,
    native_forward: bool = True,
    native_swiglu: bool = False,
    native_swiglu_decomposed: bool = False,
    resident_mlp_forward: bool = False,
    resident_mlp_fp32_activation: bool = False,
    resident_down_backward: bool = False,
    resident_activation_backward: bool = False,
    resident_down_activation_backward: bool = False,
    resident_full_mlp_backward: bool = False,
    native_packing: bool = False,
    direct_logical: bool = False,
    install_mlp: bool = True,
    install_attention: bool = True,
    input_gradient_edge_order: tuple[int, int, int, int] = (
        2,
        3,
        1,
        0,
    ),
    separate_input_gradients: bool = True,
    grouped_base_input_gradient: bool = False,
) -> GroupedLoRAInstallReport:
    """Install exact gate/up execution views without changing PEFT state keys."""

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"backend must be one of {sorted(SUPPORTED_BACKENDS)}"
        )
    installed: list[str] = []
    skipped: dict[str, str] = {}
    candidates = tuple(model.named_modules())
    if install_mlp:
        for path, module in candidates:
            if not path or isinstance(module, NativeGroupedLlamaMLP):
                continue
            if not all(
                hasattr(module, name)
                for name in (
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "act_fn",
                )
            ):
                continue
            if not hasattr(module.gate_proj, "lora_A"):
                continue
            try:
                replacement = NativeGroupedLlamaMLP(
                    module,
                    backend=backend,
                    native_backward=native_backward,
                    grouped_native_phase_one=grouped_native_phase_one,
                    full_native_tail=full_native_tail,
                    grouped_native_tail=grouped_native_tail,
                    resident_grouped_backward=(
                        resident_grouped_backward
                    ),
                    native_forward=native_forward,
                    native_swiglu=native_swiglu,
                    native_swiglu_decomposed=(
                        native_swiglu_decomposed
                    ),
                    resident_mlp_forward=resident_mlp_forward,
                    resident_mlp_fp32_activation=(
                        resident_mlp_fp32_activation
                    ),
                    resident_down_backward=resident_down_backward,
                    resident_activation_backward=(
                        resident_activation_backward
                    ),
                    resident_down_activation_backward=(
                        resident_down_activation_backward
                    ),
                    resident_full_mlp_backward=(
                        resident_full_mlp_backward
                    ),
                    native_packing=native_packing,
                    direct_logical=direct_logical,
                    input_gradient_edge_order=input_gradient_edge_order,
                    separate_input_gradients=separate_input_gradients,
                    grouped_base_input_gradient=(
                        grouped_base_input_gradient
                    ),
                )
            except ValueError as error:
                skipped[path] = str(error)
                continue
            parent_path, _, child_name = path.rpartition(".")
            parent = (
                model.get_submodule(parent_path) if parent_path else model
            )
            setattr(parent, child_name, replacement)
            installed.append(path)

    # Llama-family attention invokes K then V with the same activation. Keep
    # their original PEFT modules in place and coordinate the pair at the call
    # boundary. The generic backend intentionally avoids monkeypatching.
    if install_attention and backend != "generic":
        for path, module in candidates:
            if not path or isinstance(module, NativeGroupedLlamaMLP):
                continue
            if not all(
                hasattr(module, name)
                for name in ("q_proj", "k_proj", "v_proj", "o_proj")
            ):
                continue
            if not hasattr(module.k_proj, "lora_A"):
                continue
            report_path = f"{path}.[k_proj,v_proj]"
            try:
                _install_native_grouped_kv(
                    module,
                    backend=backend,
                    native_backward=native_backward,
                    grouped_native_phase_one=grouped_native_phase_one,
                    full_native_tail=full_native_tail,
                    grouped_native_tail=grouped_native_tail,
                    resident_grouped_backward=(
                        resident_grouped_backward
                    ),
                    input_gradient_edge_order=input_gradient_edge_order,
                    separate_input_gradients=separate_input_gradients,
                    grouped_base_input_gradient=(
                        grouped_base_input_gradient
                    ),
                )
            except ValueError as error:
                skipped[report_path] = str(error)
                continue
            installed.append(report_path)
    return GroupedLoRAInstallReport(
        backend=backend,
        installed_paths=tuple(installed),
        skipped=skipped,
    )
