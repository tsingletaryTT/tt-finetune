"""Autograd boundary for the exact resident SwiGLU + down forward program."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model_workloads import EXACT_BASE_DX_K_PHASES
from .tt_lang_frozen_linear_lora_backward import (
    build_tt_lang_frozen_linear_lora_backward,
)
from .tt_lang_lora_weight_backward import (
    build_tt_lang_lora_backward_da,
)
from .tt_lang_resident_down_backward import (
    build_tt_lang_resident_down_backward,
)
from .tt_lang_resident_down_activation_backward import (
    build_tt_lang_resident_down_activation_backward,
)
from .tt_lang_resident_swiglu_down import (
    build_tt_lang_resident_swiglu_down_forward,
    resident_swiglu_down_support_error,
)
from .tt_lang_swiglu import (
    build_tt_lang_swiglu_backward,
    build_tt_lang_swiglu_forward,
)


class NativeResidentSwiGLUDownFunction(torch.autograd.Function):
    """Resident exact forward with validated native down backward phases.

    The large activation product is recomputed because saving it to DRAM was
    measured slower. Rank-space H remains in the forward arena. The optional
    resident down-backward program computes dProduct/dH/dA/dB in one call and
    consumes dH through L1/NoC rather than materializing it. Activation
    backward remains in the framework graph until its exact tile schedule can
    feed the resident grouped gate/up program directly.
    """

    @staticmethod
    def forward(
        ctx,
        gate: torch.Tensor,
        up: torch.Tensor,
        frozen_weight: torch.Tensor,
        frozen_weight_t: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scale: float,
        arena: torch.Tensor,
        backward_arena: torch.Tensor,
        lora_a_gradient_arena: torch.Tensor,
        activation_backward_arena: torch.Tensor,
        down_activation_backward_arena: torch.Tensor,
        product_recompute_arena: torch.Tensor,
        fp32_activation: bool,
        decomposed_activation: bool,
        resident_down_backward: bool,
        resident_activation_backward: bool,
        resident_down_activation_backward: bool,
    ) -> torch.Tensor:
        tokens, intermediate = gate.shape
        hidden = frozen_weight.shape[0]
        error = resident_swiglu_down_support_error(
            tokens=tokens,
            hidden_features=hidden,
            intermediate_features=intermediate,
            physical_rank=lora_a.shape[0],
        )
        if error is not None:
            raise ValueError(error)
        operation = build_tt_lang_resident_swiglu_down_forward(
            scale=float(scale),
            tokens=tokens,
            hidden_features=hidden,
            intermediate_features=intermediate,
            physical_rank=lora_a.shape[0],
            fp32_activation=bool(fp32_activation),
            decomposed_activation=bool(decomposed_activation),
        )
        result_arena = operation(
            gate, up, frozen_weight_t, lora_a, lora_b, arena
        )
        output = result_arena[:, :hidden]
        rank_hidden = result_arena[:, hidden:]
        ctx.save_for_backward(
            gate,
            up,
            frozen_weight,
            lora_a,
            lora_b,
            rank_hidden,
            backward_arena,
            lora_a_gradient_arena,
            activation_backward_arena,
            down_activation_backward_arena,
            product_recompute_arena,
        )
        ctx.scale = float(scale)
        ctx.resident_down_backward = bool(resident_down_backward)
        ctx.resident_activation_backward = bool(
            resident_activation_backward
        )
        ctx.resident_down_activation_backward = bool(
            resident_down_activation_backward
        )
        return output

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        (
            gate,
            up,
            frozen_weight,
            lora_a,
            lora_b,
            rank_hidden,
            backward_arena,
            lora_a_gradient_arena,
            activation_backward_arena,
            down_activation_backward_arena,
            product_recompute_arena,
        ) = ctx.saved_tensors
        tokens, intermediate = gate.shape
        hidden, physical_rank = lora_b.shape
        base_dx_phase = EXACT_BASE_DX_K_PHASES.get(
            (intermediate, hidden)
        )
        if base_dx_phase is None:
            raise RuntimeError(
                "resident MLP backward has no exact down dX schedule"
            )

        if (
            ctx.resident_activation_backward
            or ctx.resident_down_activation_backward
        ):
            product_operation = build_tt_lang_swiglu_forward(
                tokens=tokens,
                intermediate_features=intermediate,
                diagnostic_stage="full",
            )
            product = product_operation(
                gate, up, product_recompute_arena
            )
        else:
            product = F.silu(gate) * up
        if ctx.resident_down_activation_backward:
            # The 100-core matmul island sends dProduct directly to the
            # 10-core exact-BF16 activation island. The only visible tensors
            # are dGate, dUp, dA and dB; there is no PJRT/DRAM edge between
            # the two phases.
            resident_operation = (
                build_tt_lang_resident_down_activation_backward(
                    scale=ctx.scale,
                    tokens=tokens,
                    input_features=intermediate,
                    output_features=hidden,
                    physical_rank=physical_rank,
                    base_reduction_tiles_per_phase=base_dx_phase,
                )
            )
            backward_result = resident_operation(
                output_gradient,
                frozen_weight,
                lora_b,
                rank_hidden,
                lora_a,
                product,
                gate,
                up,
                down_activation_backward_arena,
            )
            gate_gradient = backward_result[:tokens, :intermediate]
            up_gradient = backward_result[
                tokens : 2 * tokens, :intermediate
            ]
            gradient_rows = 2 * tokens
            lora_a_gradient = backward_result[
                gradient_rows : gradient_rows + physical_rank,
                :intermediate,
            ]
            lora_b_gradient = backward_result[
                gradient_rows
                + physical_rank : gradient_rows
                + physical_rank
                + hidden,
                :physical_rank,
            ]
        elif ctx.resident_down_backward:
            # One call owns dH and consumes it in L1 for both dA and the LoRA
            # dProduct path. Only final dProduct/dA/dB leave the program.
            resident_operation = build_tt_lang_resident_down_backward(
                scale=ctx.scale,
                tokens=tokens,
                input_features=intermediate,
                output_features=hidden,
                physical_rank=physical_rank,
                base_reduction_tiles_per_phase=base_dx_phase,
                # The one-compute-kernel fused prototype is intentionally not
                # selected here. Exact activation math needs a compute kernel
                # compiled with FP32 destination accumulation disabled.
                activation_backward=False,
            )
            backward_result = resident_operation(
                output_gradient,
                frozen_weight,
                lora_b,
                rank_hidden,
                lora_a,
                product,
                backward_arena,
            )
            gradient_rows = tokens
            product_gradient = backward_result[:tokens, :intermediate]
            lora_a_gradient = backward_result[
                gradient_rows : gradient_rows + physical_rank, :intermediate
            ]
            lora_b_gradient = backward_result[
                gradient_rows
                + physical_rank : gradient_rows
                + physical_rank
                + hidden,
                :physical_rank,
            ]
        else:
            # Accepted compatibility path: phase one and dA are separate
            # calls with dH materialized through the arena.
            backward_operation = build_tt_lang_frozen_linear_lora_backward(
                scale=ctx.scale,
                tokens=tokens,
                input_features=intermediate,
                output_features=hidden,
                physical_rank=physical_rank,
                base_reduction_tiles_per_phase=base_dx_phase,
            )
            backward_result = backward_operation(
                output_gradient,
                frozen_weight,
                lora_b,
                rank_hidden,
                backward_arena,
            )
            base_product_gradient = backward_result[:tokens, :intermediate]
            rank_gradient = backward_result[
                :tokens, intermediate : intermediate + physical_rank
            ]
            lora_b_gradient = backward_result[
                tokens : tokens + hidden, :physical_rank
            ]

            lora_product_gradient = rank_gradient @ lora_a
            product_gradient = base_product_gradient + lora_product_gradient
            da_operation = build_tt_lang_lora_backward_da(
                tokens=tokens,
                input_features=intermediate,
                physical_rank=physical_rank,
                exact_reduction=True,
            )
            lora_a_gradient = da_operation(
                rank_gradient,
                product,
                lora_a_gradient_arena,
            )
        if ctx.resident_down_activation_backward:
            pass
        elif ctx.resident_activation_backward:
            activation_operation = build_tt_lang_swiglu_backward(
                tokens=tokens,
                intermediate_features=intermediate,
            )
            activation_result = activation_operation(
                product_gradient,
                gate,
                up,
                activation_backward_arena,
            )
            gate_gradient = activation_result[:tokens]
            up_gradient = activation_result[tokens : 2 * tokens]
        else:
            gate_gradient = torch.ops.aten.silu_backward(
                product_gradient * up, gate
            )
            up_gradient = product_gradient * F.silu(gate)
        return (
            gate_gradient,
            up_gradient,
            None,
            None,
            lora_a_gradient,
            lora_b_gradient,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def native_resident_swiglu_down(
    gate: torch.Tensor,
    up: torch.Tensor,
    frozen_weight: torch.Tensor,
    frozen_weight_t: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
    arena: torch.Tensor,
    backward_arena: torch.Tensor,
    lora_a_gradient_arena: torch.Tensor,
    activation_backward_arena: torch.Tensor,
    down_activation_backward_arena: torch.Tensor,
    product_recompute_arena: torch.Tensor,
    fp32_activation: bool = False,
    decomposed_activation: bool = True,
    resident_down_backward: bool = False,
    resident_activation_backward: bool = False,
    resident_down_activation_backward: bool = False,
) -> torch.Tensor:
    return NativeResidentSwiGLUDownFunction.apply(
        gate,
        up,
        frozen_weight,
        frozen_weight_t,
        lora_a,
        lora_b,
        scale,
        arena,
        backward_arena,
        lora_a_gradient_arena,
        activation_backward_arena,
        down_activation_backward_arena,
        product_recompute_arena,
        bool(fp32_activation),
        bool(decomposed_activation),
        bool(resident_down_backward),
        bool(resident_activation_backward),
        bool(resident_down_activation_backward),
    )
