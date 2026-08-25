"""Exact two-member shared-input LoRA block autograd for Blackhole."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .grouped_lora import GroupedProjectionLayout
from .native_frozen_lora_autograd import exact_base_dx_k_phase
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
from .tt_lang_grouped2_logical_forward import (
    build_tt_lang_grouped2_logical_forward,
)
from .tt_lang_grouped2_frozen_lora_backward_tail import (
    build_tt_lang_grouped2_frozen_lora_backward_tail,
)
from .tt_lang_grouped2_frozen_linear_lora_backward import (
    build_tt_lang_grouped2_frozen_linear_lora_backward,
)
from .tt_lang_grouped2_resident_backward import (
    build_tt_lang_grouped2_resident_backward,
    resident_backward_support_error,
)


@dataclass
class NativeGrouped2Arena:
    frozen_weight_transposed: torch.Tensor
    forward_arena_init: torch.Tensor
    backward_arena_inits: tuple[torch.Tensor, torch.Tensor]
    grouped_backward_arena_init: torch.Tensor
    backward_tail_arena_inits: tuple[torch.Tensor, torch.Tensor]
    backward_tail_zero_inits: tuple[torch.Tensor, torch.Tensor]
    grouped_backward_tail_arena_init: torch.Tensor
    resident_backward_arena_init: torch.Tensor


def make_native_grouped2_arena(
    inputs: torch.Tensor,
    grouped_frozen_weight: torch.Tensor,
    grouped_a: torch.Tensor,
    grouped_b: torch.Tensor,
    layout: GroupedProjectionLayout,
    *,
    validate_packed: bool = True,
    forward_arena_fill_value: float = 0.0,
    scratch_arena_fill_values: tuple[float, ...] | None = None,
    input_checkpoint_columns: int = 0,
) -> NativeGrouped2Arena:
    if layout.group_size != 2 or layout.total_physical_rank != 64:
        raise ValueError("exact grouped autograd requires two rank-32 members")
    tokens, input_features = inputs.shape
    if grouped_frozen_weight.shape != (
        layout.total_output_features,
        input_features,
    ):
        raise ValueError("invalid grouped frozen weight")
    if validate_packed:
        if grouped_a.shape != (64, input_features):
            raise ValueError("invalid grouped LoRA A")
        if grouped_b.shape != (layout.total_output_features, 64):
            raise ValueError("invalid grouped LoRA B")
    if input_checkpoint_columns not in {0, input_features}:
        raise ValueError(
            "grouped input checkpoint must be absent or the full input width"
        )

    if scratch_arena_fill_values is None:
        scratch_arena_fill_values = (0.0,) * 7
    if len(scratch_arena_fill_values) != 7:
        raise ValueError("grouped scratch identity requires seven values")
    scratch = iter(scratch_arena_fill_values)

    def device_output(shape: tuple[int, int]) -> torch.Tensor:
        return torch.full(
            shape, next(scratch), dtype=inputs.dtype, device=inputs.device
        )

    def device_zeros(shape: tuple[int, int]) -> torch.Tensor:
        return torch.zeros(shape, dtype=inputs.dtype, device=inputs.device)

    return NativeGrouped2Arena(
        frozen_weight_transposed=(
            grouped_frozen_weight.transpose(0, 1).contiguous()
        ),
        forward_arena_init=torch.full(
            (
                tokens,
                layout.total_output_features
                + 64
                + input_checkpoint_columns,
            ),
            float(forward_arena_fill_value),
            dtype=inputs.dtype,
            device=inputs.device,
        ),
        backward_arena_inits=tuple(
            device_output(
                (tokens + outputs, input_features + 32)
            )
            for outputs in layout.output_features
        ),
        grouped_backward_arena_init=device_output(
            (
                2 * tokens + sum(layout.output_features),
                input_features + 32,
            )
        ),
        backward_tail_arena_inits=tuple(
            device_output((tokens + 32, input_features))
            for _ in layout.output_features
        ),
        backward_tail_zero_inits=tuple(
            device_zeros((tokens, input_features))
            for _ in layout.output_features
        ),
        grouped_backward_tail_arena_init=device_output(
            (tokens + 64, input_features)
        ),
        resident_backward_arena_init=device_output(
            (
                tokens + 64 + sum(layout.output_features),
                input_features,
            )
        ),
    )


class NativeExactGrouped2LoRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        second_inputs,
        third_inputs,
        fourth_inputs,
        grouped_frozen_weight,
        grouped_frozen_weight_transposed,
        grouped_a,
        grouped_b,
        scale,
        first_output_features,
        base_k_tiles_per_phase,
        sync_boundaries,
        forward_arena_init,
        backward_arena_0,
        backward_arena_1,
        grouped_backward_arena,
        backward_tail_arena_0,
        backward_tail_arena_1,
        backward_tail_zero_0,
        backward_tail_zero_1,
        grouped_backward_tail_arena,
        resident_backward_arena,
        peft_input_gradient_order,
        native_backward,
        grouped_native_phase_one,
        full_native_tail,
        grouped_native_tail,
        resident_grouped_backward,
        native_forward,
        logical_rank_per_member,
        logical_a0,
        logical_b0,
        logical_a1,
        logical_b1,
        direct_logical,
        separate_input_gradients,
        input_gradient_edge_order,
        grouped_base_input_gradient,
    ):
        if any(
            value.shape != inputs.shape
            for value in (second_inputs, third_inputs, fourth_inputs)
        ):
            raise ValueError("grouped member inputs must have equal shapes")
        tokens, input_features = inputs.shape
        total_output_features = grouped_frozen_weight.shape[0]
        second_output_features = (
            total_output_features - first_output_features
        )
        if native_forward:
            if direct_logical:
                if first_output_features != second_output_features:
                    raise ValueError(
                        "direct logical forward requires equal outputs"
                    )
                operation = build_tt_lang_grouped2_logical_forward(
                    scale=float(scale),
                    tokens=tokens,
                    input_features=input_features,
                    member_output_features=int(first_output_features),
                    base_k_tiles_per_phase=int(
                        base_k_tiles_per_phase
                    ),
                )
                arena = operation(
                    inputs,
                    grouped_frozen_weight_transposed,
                    logical_a0,
                    logical_b0,
                    logical_a1,
                    logical_b1,
                    forward_arena_init,
                )
            else:
                operation = (
                    build_tt_lang_frozen_linear_lora_forward_prototype(
                        scale=float(scale),
                        tokens=tokens,
                        input_features=input_features,
                        output_features=total_output_features,
                        physical_rank=64,
                        n_tiles_per_block=1,
                        base_k_tiles_per_phase=int(
                            base_k_tiles_per_phase
                        ),
                        exact_reduction=True,
                    )
                )
                arena = operation(
                    inputs,
                    grouped_frozen_weight_transposed,
                    grouped_a,
                    grouped_b,
                    forward_arena_init,
                )
            if sync_boundaries:
                _sync_custom_boundary(arena)
            output = arena[:, :total_output_features]
            hidden = arena[:, total_output_features:]
        else:
            rank = int(logical_rank_per_member)
            if rank <= 0 or rank > 32:
                raise ValueError("logical member rank must be in [1, 32]")
            member_outputs = (
                int(first_output_features),
                int(second_output_features),
            )
            output_parts = []
            hidden_parts = []
            output_offset = 0
            for member, outputs in enumerate(member_outputs):
                rank_offset = member * 32
                logical_a = grouped_a[
                    rank_offset : rank_offset + rank
                ]
                logical_b = grouped_b[
                    output_offset : output_offset + outputs,
                    rank_offset : rank_offset + rank,
                ]
                logical_hidden = inputs @ logical_a.transpose(0, 1)
                output_parts.append(
                    inputs
                    @ grouped_frozen_weight[
                        output_offset : output_offset + outputs
                    ].transpose(0, 1)
                    + (
                        logical_hidden @ logical_b.transpose(0, 1)
                    )
                    * float(scale)
                )
                hidden_parts.append(
                    torch.cat(
                        (
                            logical_hidden,
                            logical_hidden.new_zeros(
                                (tokens, 32 - rank)
                            ),
                        ),
                        dim=1,
                    )
                )
                output_offset += outputs
            output = torch.cat(output_parts, dim=1)
            hidden = torch.cat(hidden_parts, dim=1)
        ctx.save_for_backward(
            inputs,
            grouped_frozen_weight,
            grouped_a,
            grouped_b,
            hidden,
            backward_arena_0,
            backward_arena_1,
            grouped_backward_arena,
            backward_tail_arena_0,
            backward_tail_arena_1,
            backward_tail_zero_0,
            backward_tail_zero_1,
            grouped_backward_tail_arena,
            resident_backward_arena,
            logical_a0,
            logical_b0,
            logical_a1,
            logical_b1,
        )
        ctx.output_features = (
            int(first_output_features),
            int(second_output_features),
        )
        ctx.scale = float(scale)
        ctx.sync_boundaries = bool(sync_boundaries)
        ctx.peft_input_gradient_order = bool(peft_input_gradient_order)
        ctx.native_backward = bool(native_backward)
        ctx.grouped_native_phase_one = bool(grouped_native_phase_one)
        if ctx.grouped_native_phase_one and not ctx.native_backward:
            raise ValueError(
                "grouped native phase one requires native backward"
            )
        if (
            ctx.grouped_native_phase_one
            and ctx.output_features[0] != ctx.output_features[1]
        ):
            raise ValueError(
                "grouped native phase one requires equal member outputs"
            )
        ctx.full_native_tail = bool(full_native_tail)
        ctx.grouped_native_tail = bool(grouped_native_tail)
        if ctx.grouped_native_tail and not (
            ctx.native_backward
            and ctx.full_native_tail
            and ctx.peft_input_gradient_order
        ):
            raise ValueError(
                "grouped native tail requires native phase one, the full "
                "native tail, and PEFT input-gradient ordering"
            )
        resident_requested = bool(resident_grouped_backward)
        if resident_requested and not (
            ctx.native_backward
            and ctx.full_native_tail
            and ctx.peft_input_gradient_order
        ):
            raise ValueError(
                "resident grouped backward requires native backward, "
                "the full native tail, and PEFT input-gradient ordering"
            )
        if (
            resident_requested
            and ctx.output_features[0] != ctx.output_features[1]
        ):
            raise ValueError(
                "resident grouped backward requires equal member outputs"
            )
        resident_support_error = resident_backward_support_error(
            tokens=tokens,
            input_features=input_features,
            member_output_features=ctx.output_features[0],
            physical_rank=32,
            base_reduction_tiles_per_phase=int(
                base_k_tiles_per_phase
            ),
        )
        ctx.resident_grouped_backward = (
            resident_requested and resident_support_error is None
        )
        ctx.resident_grouped_backward_fallback_reason = (
            resident_support_error if resident_requested else None
        )
        # The resident program already owns both phase-one reductions.  Keeping
        # the legacy phase-one branch live in the autograd graph is redundant
        # and measurably worsens full-step scheduling, even though backward
        # returns through the resident branch before executing it.  Preserve
        # the requested option for unsupported-shape fallback, but remove the
        # dead branch from resident graphs.
        if ctx.resident_grouped_backward:
            ctx.grouped_native_phase_one = False
        ctx.direct_logical = bool(direct_logical)
        ctx.separate_input_gradients = bool(
            separate_input_gradients
        )
        edge_order = tuple(int(value) for value in input_gradient_edge_order)
        if sorted(edge_order) != [0, 1, 2, 3]:
            raise ValueError(
                "input gradient edge order must be a permutation of 0..3"
            )
        ctx.input_gradient_edge_order = edge_order
        ctx.grouped_base_input_gradient = bool(
            grouped_base_input_gradient
        )
        return output

    @staticmethod
    def backward(ctx, output_gradient):
        (
            inputs,
            grouped_frozen_weight,
            grouped_a,
            grouped_b,
            hidden,
            backward_arena_0,
            backward_arena_1,
            grouped_backward_arena,
            backward_tail_arena_0,
            backward_tail_arena_1,
            backward_tail_zero_0,
            backward_tail_zero_1,
            grouped_backward_tail_arena,
            resident_backward_arena,
            logical_a0,
            logical_b0,
            logical_a1,
            logical_b1,
        ) = ctx.saved_tensors
        tokens, input_features = inputs.shape
        output_offsets = (0, ctx.output_features[0])
        rank_offsets = (0, 32)
        member_base_input_gradients = []
        member_lora_input_gradients = []
        member_a_gradients = []
        member_b_gradients = []
        member_hidden_gradients = []
        member_lora_as = []
        grouped_phase_one_arena = None
        if ctx.resident_grouped_backward:
            outputs = ctx.output_features[0]
            dx_phase = exact_base_dx_k_phase(
                input_features, outputs
            )
            if dx_phase is None:
                raise RuntimeError(
                    "resident backward encountered an unvalidated member"
                )
            if ctx.direct_logical:
                first_a, second_a = logical_a0, logical_a1
                first_b, second_b = logical_b0, logical_b1
            else:
                first_a = grouped_a[:32]
                second_a = grouped_a[32:64]
                first_b = grouped_b[:outputs, :32]
                second_b = grouped_b[outputs:, 32:64]
            operation = build_tt_lang_grouped2_resident_backward(
                scale=ctx.scale,
                tokens=tokens,
                input_features=input_features,
                member_output_features=outputs,
                physical_rank=32,
                base_reduction_tiles_per_phase=dx_phase,
            )
            resident_arena = operation(
                output_gradient[:, :outputs],
                grouped_frozen_weight[:outputs],
                first_b,
                hidden[:, :32],
                first_a,
                output_gradient[:, outputs : 2 * outputs],
                grouped_frozen_weight[outputs : 2 * outputs],
                second_b,
                hidden[:, 32:64],
                second_a,
                inputs,
                resident_backward_arena,
            )
            if ctx.sync_boundaries:
                _sync_custom_boundary(resident_arena)
            input_gradient = resident_arena[:tokens, :input_features]
            member_a_gradients = [
                resident_arena[
                    tokens : tokens + 32, :input_features
                ],
                resident_arena[
                    tokens + 32 : tokens + 64, :input_features
                ],
            ]
            member_b_gradients = [
                resident_arena[
                    tokens + 64 : tokens + 64 + outputs, :32
                ],
                resident_arena[
                    tokens
                    + 64
                    + outputs : tokens
                    + 64
                    + 2 * outputs,
                    :32,
                ],
            ]
            first_outputs, second_outputs = ctx.output_features
            if ctx.direct_logical:
                grouped_a_gradient = None
                grouped_b_gradient = None
                rank = int(logical_a0.shape[0])
                logical_gradients = (
                    member_a_gradients[0][:rank],
                    member_b_gradients[0][:, :rank],
                    member_a_gradients[1][:rank],
                    member_b_gradients[1][:, :rank],
                )
            else:
                grouped_a_gradient = torch.cat(
                    member_a_gradients, dim=0
                )
                zero_top_right = torch.zeros(
                    first_outputs,
                    32,
                    dtype=grouped_b.dtype,
                    device=grouped_b.device,
                )
                zero_bottom_left = torch.zeros(
                    second_outputs,
                    32,
                    dtype=grouped_b.dtype,
                    device=grouped_b.device,
                )
                grouped_b_gradient = torch.cat(
                    (
                        torch.cat(
                            (
                                member_b_gradients[0],
                                zero_top_right,
                            ),
                            dim=1,
                        ),
                        torch.cat(
                            (
                                zero_bottom_left,
                                member_b_gradients[1],
                            ),
                            dim=1,
                        ),
                    ),
                    dim=0,
                )
                logical_gradients = (None, None, None, None)
            return (
                input_gradient,              # inputs
                None,                        # second_inputs
                None,                        # third_inputs
                None,                        # fourth_inputs
                None,                        # grouped_frozen_weight
                None,                        # frozen_weight_transposed
                grouped_a_gradient,          # grouped_a
                grouped_b_gradient,          # grouped_b
                None,                        # scale
                None,                        # first_output_features
                None,                        # base_k_tiles_per_phase
                None,                        # sync_boundaries
                None,                        # forward_arena_init
                None,                        # backward_arena_0
                None,                        # backward_arena_1
                None,                        # grouped_backward_arena
                None,                        # backward_tail_arena_0
                None,                        # backward_tail_arena_1
                None,                        # backward_tail_zero_0
                None,                        # backward_tail_zero_1
                None,                        # grouped_backward_tail_arena
                None,                        # resident_backward_arena
                None,                        # peft_input_gradient_order
                None,                        # native_backward
                None,                        # grouped_native_phase_one
                None,                        # full_native_tail
                None,                        # grouped_native_tail
                None,                        # resident_grouped_backward
                None,                        # native_forward
                None,                        # logical_rank_per_member
                logical_gradients[0],        # logical_a0
                logical_gradients[1],        # logical_b0
                logical_gradients[2],        # logical_a1
                logical_gradients[3],        # logical_b1
                None,                        # direct_logical
                None,                        # separate_input_gradients
                None,                        # input_gradient_edge_order
                None,                        # grouped_base_input_gradient
            )
        if ctx.grouped_native_phase_one:
            outputs = ctx.output_features[0]
            dx_phase = exact_base_dx_k_phase(
                input_features, outputs
            )
            if dx_phase is None:
                raise RuntimeError(
                    "grouped phase one encountered an unvalidated member"
                )
            if ctx.direct_logical:
                first_b, second_b = logical_b0, logical_b1
            else:
                first_b = grouped_b[:outputs, :32]
                second_b = grouped_b[outputs:, 32:64]
            grouped_operation = (
                build_tt_lang_grouped2_frozen_linear_lora_backward(
                    scale=ctx.scale,
                    tokens=tokens,
                    input_features=input_features,
                    member_output_features=outputs,
                    physical_rank=32,
                    base_reduction_tiles_per_phase=dx_phase,
                    concurrent_members=True,
                )
            )
            grouped_phase_one_arena = grouped_operation(
                output_gradient[:, :outputs],
                grouped_frozen_weight[:outputs],
                first_b,
                hidden[:, :32],
                output_gradient[:, outputs : 2 * outputs],
                grouped_frozen_weight[outputs : 2 * outputs],
                second_b,
                hidden[:, 32:64],
                grouped_backward_arena,
            )
            if ctx.sync_boundaries:
                _sync_custom_boundary(grouped_phase_one_arena)
        for member, (
            outputs,
            output_offset,
            rank_offset,
            init,
            tail_init,
            tail_zero,
        ) in enumerate(
            zip(
                ctx.output_features,
                output_offsets,
                rank_offsets,
                (backward_arena_0, backward_arena_1),
                (
                    backward_tail_arena_0,
                    backward_tail_arena_1,
                ),
                (
                    backward_tail_zero_0,
                    backward_tail_zero_1,
                ),
            )
        ):
            dy = output_gradient[
                :, output_offset : output_offset + outputs
            ]
            weight = grouped_frozen_weight[
                output_offset : output_offset + outputs
            ]
            if ctx.direct_logical:
                lora_a = (logical_a0, logical_a1)[member]
                lora_b = (logical_b0, logical_b1)[member]
            else:
                lora_a = grouped_a[rank_offset : rank_offset + 32]
                lora_b = grouped_b[
                    output_offset : output_offset + outputs,
                    rank_offset : rank_offset + 32,
                ]
            saved_hidden = hidden[:, rank_offset : rank_offset + 32]
            if ctx.native_backward:
                if ctx.grouped_native_phase_one:
                    assert grouped_phase_one_arena is not None
                    member_rows = tokens + outputs
                    member_offset = member * member_rows
                    arena = grouped_phase_one_arena[
                        member_offset :
                        member_offset + member_rows
                    ]
                else:
                    dx_phase = exact_base_dx_k_phase(
                        input_features, outputs
                    )
                    if dx_phase is None:
                        raise RuntimeError(
                            "grouped backward encountered an "
                            "unvalidated member"
                        )
                    operation = build_tt_lang_frozen_linear_lora_backward(
                        scale=ctx.scale,
                        tokens=tokens,
                        input_features=input_features,
                        output_features=outputs,
                        physical_rank=32,
                        base_reduction_tiles_per_phase=dx_phase,
                    )
                    arena = operation(
                        dy, weight, lora_b, saved_hidden, init
                    )
                    if ctx.sync_boundaries:
                        _sync_custom_boundary(arena)
                base_dx = arena[:tokens, :input_features]
                hidden_gradient = arena[
                    :tokens, input_features : input_features + 32
                ]
                b_gradient = arena[
                    tokens : tokens + outputs, :32
                ]
            else:
                scaled_gradient = dy * ctx.scale
                base_dx = dy @ weight
                hidden_gradient = scaled_gradient @ lora_b
                b_gradient = (
                    scaled_gradient.transpose(0, 1) @ saved_hidden
                )
            if ctx.full_native_tail:
                if not ctx.native_backward:
                    raise RuntimeError(
                        "the full native tail requires native phase one"
                    )
                if ctx.grouped_native_tail:
                    lora_dx = None
                    a_gradient = None
                else:
                    tail_operation = (
                        build_tt_lang_frozen_lora_backward_tail(
                            tokens=tokens,
                            input_features=input_features,
                            physical_rank=32,
                        )
                    )
                    # The compact member-tail program computes lora dX and dA
                    # in one dispatch. A zero base preserves the member lora
                    # dX as an independently ordered BF16 PEFT contribution.
                    tail_arena = tail_operation(
                        tail_zero,
                        hidden_gradient,
                        lora_a,
                        inputs,
                        tail_init,
                    )
                    if ctx.sync_boundaries:
                        _sync_custom_boundary(tail_arena)
                    lora_dx = tail_arena[:tokens, :input_features]
                    a_gradient = tail_arena[
                        tokens : tokens + 32, :input_features
                    ]
            else:
                lora_dx = hidden_gradient @ lora_a
                a_gradient = (
                    inputs.transpose(0, 1) @ hidden_gradient
                ).transpose(0, 1)
            member_base_input_gradients.append(base_dx)
            member_b_gradients.append(b_gradient)
            member_hidden_gradients.append(hidden_gradient)
            member_lora_as.append(lora_a)
            if lora_dx is not None and a_gradient is not None:
                member_lora_input_gradients.append(lora_dx)
                member_a_gradients.append(a_gradient)

        grouped_tail_input_gradient = None
        if ctx.grouped_native_tail:
            tail_operation = (
                build_tt_lang_grouped2_frozen_lora_backward_tail(
                    tokens=tokens,
                    input_features=input_features,
                )
            )
            tail_arena = tail_operation(
                member_base_input_gradients[0],
                member_hidden_gradients[0],
                member_lora_as[0],
                member_base_input_gradients[1],
                member_hidden_gradients[1],
                member_lora_as[1],
                inputs,
                grouped_backward_tail_arena,
            )
            if ctx.sync_boundaries:
                _sync_custom_boundary(tail_arena)
            grouped_tail_input_gradient = tail_arena[
                :tokens, :input_features
            ]
            member_a_gradients = [
                tail_arena[
                    tokens : tokens + 32, :input_features
                ],
                tail_arena[
                    tokens + 32 : tokens + 64, :input_features
                ],
            ]
        elif ctx.peft_input_gradient_order:
            # A real PEFT Llama MLP accumulates the up branch first, then the
            # gate LoRA and finally the gate base contribution.
            input_gradient = (
                (
                    member_base_input_gradients[1]
                    + member_lora_input_gradients[1]
                )
                + member_lora_input_gradients[0]
            ) + member_base_input_gradients[0]
        else:
            # Match PyTorch/XLA's reverse traversal of a direct concatenate.
            input_gradient = (
                (
                    member_base_input_gradients[1]
                    + member_lora_input_gradients[1]
                )
                + member_base_input_gradients[0]
            ) + member_lora_input_gradients[0]
        if ctx.grouped_native_tail:
            first_input_gradient = grouped_tail_input_gradient
            second_input_gradient = None
            third_input_gradient = None
            fourth_input_gradient = None
        elif ctx.separate_input_gradients:
            # Recreate the four original PEFT activation edges. Their
            # accumulation order is observable in BF16 and can depend on the
            # compiled token-tile grid, so the schedule is explicit instead
            # of being hidden in this custom autograd function.
            if ctx.grouped_base_input_gradient:
                # Compiler-level prototype for the next native kernel: reduce
                # both frozen branches through one shared K output and keep
                # the two exact low-rank contributions on separate edges.
                first_input_gradient = (
                    output_gradient @ grouped_frozen_weight
                )
                second_input_gradient = (
                    member_lora_input_gradients[1]
                )
                third_input_gradient = (
                    member_lora_input_gradients[0]
                )
                fourth_input_gradient = None
            else:
                contributions = (
                    member_base_input_gradients[0],
                    member_lora_input_gradients[0],
                    member_base_input_gradients[1],
                    member_lora_input_gradients[1],
                )
                (
                    first_input_gradient,
                    second_input_gradient,
                    third_input_gradient,
                    fourth_input_gradient,
                ) = tuple(
                    contributions[index]
                    for index in ctx.input_gradient_edge_order
                )
        else:
            if ctx.grouped_base_input_gradient:
                grouped_base = output_gradient @ grouped_frozen_weight
                first_input_gradient = (
                    grouped_base + member_lora_input_gradients[1]
                ) + member_lora_input_gradients[0]
            else:
                first_input_gradient = input_gradient
            second_input_gradient = None
            third_input_gradient = None
            fourth_input_gradient = None
        first_outputs, second_outputs = ctx.output_features
        if ctx.direct_logical:
            grouped_a_gradient = None
            grouped_b_gradient = None
            rank = int(logical_a0.shape[0])
            logical_gradients = (
                member_a_gradients[0][:rank],
                member_b_gradients[0][:, :rank],
                member_a_gradients[1][:rank],
                member_b_gradients[1][:, :rank],
            )
        else:
            grouped_a_gradient = torch.cat(member_a_gradients, dim=0)
            zero_top_right = torch.zeros(
                first_outputs,
                32,
                dtype=grouped_b.dtype,
                device=grouped_b.device,
            )
            zero_bottom_left = torch.zeros(
                second_outputs,
                32,
                dtype=grouped_b.dtype,
                device=grouped_b.device,
            )
            grouped_b_gradient = torch.cat(
                (
                    torch.cat(
                        (member_b_gradients[0], zero_top_right), dim=1
                    ),
                    torch.cat(
                        (zero_bottom_left, member_b_gradients[1]), dim=1
                    ),
                ),
                dim=0,
            )
            logical_gradients = (None, None, None, None)
        return (
            first_input_gradient,       # inputs
            second_input_gradient,      # second_inputs
            third_input_gradient,       # third_inputs
            fourth_input_gradient,      # fourth_inputs
            None,                       # grouped_frozen_weight
            None,                       # frozen_weight_transposed
            grouped_a_gradient,         # grouped_a
            grouped_b_gradient,         # grouped_b
            None,                       # scale
            None,                       # first_output_features
            None,                       # base_k_tiles_per_phase
            None,                       # sync_boundaries
            None,                       # forward_arena_init
            None,                       # backward_arena_0
            None,                       # backward_arena_1
            None,                       # grouped_backward_arena
            None,                       # backward_tail_arena_0
            None,                       # backward_tail_arena_1
            None,                       # backward_tail_zero_0
            None,                       # backward_tail_zero_1
            None,                       # grouped_backward_tail_arena
            None,                       # resident_backward_arena
            None,                       # peft_input_gradient_order
            None,                       # native_backward
            None,                       # grouped_native_phase_one
            None,                       # full_native_tail
            None,                       # grouped_native_tail
            None,                       # resident_grouped_backward
            None,                       # native_forward
            None,                       # logical_rank_per_member
            logical_gradients[0],       # logical_a0
            logical_gradients[1],       # logical_b0
            logical_gradients[2],       # logical_a1
            logical_gradients[3],       # logical_b1
            None,                       # direct_logical
            None,                       # separate_input_gradients
            None,                       # input_gradient_edge_order
            None,                       # grouped_base_input_gradient
        )


def native_exact_grouped2_lora(
    inputs: torch.Tensor,
    grouped_frozen_weight: torch.Tensor,
    grouped_a: torch.Tensor,
    grouped_b: torch.Tensor,
    scale: float,
    layout: GroupedProjectionLayout,
    *,
    base_k_tiles_per_phase: int,
    arena: NativeGrouped2Arena | None = None,
    sync_boundaries: bool = False,
    peft_input_gradient_order: bool = False,
    native_backward: bool = True,
    grouped_native_phase_one: bool = False,
    full_native_tail: bool = False,
    grouped_native_tail: bool = False,
    resident_grouped_backward: bool = False,
    native_forward: bool = True,
    logical_rank_per_member: int = 32,
    logical_adapters: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ] | None = None,
    direct_logical: bool = False,
    separate_input_gradients: bool = False,
    input_gradient_edge_order: tuple[int, int, int, int] = (2, 3, 1, 0),
    grouped_base_input_gradient: bool = False,
) -> torch.Tensor:
    if arena is None:
        arena = make_native_grouped2_arena(
            inputs,
            grouped_frozen_weight,
            grouped_a,
            grouped_b,
            layout,
            validate_packed=not direct_logical,
        )
    if logical_adapters is None:
        logical_adapters = (
            grouped_a[:logical_rank_per_member],
            grouped_b[
                : layout.output_features[0],
                :logical_rank_per_member,
            ],
            grouped_a[32 : 32 + logical_rank_per_member],
            grouped_b[
                layout.output_features[0] :,
                32 : 32 + logical_rank_per_member,
            ],
        )
    gradient_inputs = (inputs, inputs, inputs, inputs)
    return NativeExactGrouped2LoRAFunction.apply(
        gradient_inputs[0],
        gradient_inputs[1],
        gradient_inputs[2],
        gradient_inputs[3],
        grouped_frozen_weight,
        arena.frozen_weight_transposed,
        grouped_a,
        grouped_b,
        float(scale),
        layout.output_features[0],
        int(base_k_tiles_per_phase),
        bool(sync_boundaries),
        arena.forward_arena_init,
        arena.backward_arena_inits[0],
        arena.backward_arena_inits[1],
        arena.grouped_backward_arena_init,
        arena.backward_tail_arena_inits[0],
        arena.backward_tail_arena_inits[1],
        arena.backward_tail_zero_inits[0],
        arena.backward_tail_zero_inits[1],
        arena.grouped_backward_tail_arena_init,
        arena.resident_backward_arena_init,
        bool(peft_input_gradient_order),
        bool(native_backward),
        bool(grouped_native_phase_one),
        bool(full_native_tail),
        bool(grouped_native_tail),
        bool(resident_grouped_backward),
        bool(native_forward),
        int(logical_rank_per_member),
        logical_adapters[0],
        logical_adapters[1],
        logical_adapters[2],
        logical_adapters[3],
        bool(direct_logical),
        bool(separate_input_gradients),
        tuple(input_gradient_edge_order),
        bool(grouped_base_input_gradient),
    )
