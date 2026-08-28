"""Block-sparse no-capacity expert projections for the TT-XLA MoE owner.

The TT custom ``sparse_matmul`` operation consumes a device routing bitmap and
therefore schedules only expert/tile pairs which contain selected tokens.  It
does not gather or replicate expert matrices per assignment and it does not
impose a capacity/drop semantic.  The small Python autograd boundaries below
provide the frozen-input and trainable-LoRA gradients which the upstream
inference-oriented custom operation does not currently register itself.

Only complete 32-token tiles enter this path.  Other geometries stay on the
accepted assignment-gather fallback; padding a real training batch is a data
planner decision and is never silently introduced here.
"""

from __future__ import annotations

import os

import torch


TILE = 32
# The current Blackhole sparse-matmul 1D multicast implementation does not
# make forward progress for a single output tile (N=32).  Both sparse-A and
# sparse-B paths are silicon-qualified once the physical output spans sixteen
# tiles.  Keep the model-visible width unchanged and pad only the private
# frozen projection presented to the sparse kernel.  Real expert projections
# at N>=512 pay no padding cost.
MIN_SPARSE_OUTPUT_FEATURES = 16 * TILE

# Keep the previously qualified dense-dispatch backward as the safe default
# until the gather-pack path has completed silicon qualification.  The latter
# reuses the already-qualified forward pack owner: every (expert, position)
# destination is unique, so this is a gather into packed slots, not a reduction
# and does not require a one-hot dispatch GEMM.
USE_GATHER_PACK_BACKWARD = (
    os.environ.get("TT_FINETUNE_DYNAMIC_MOE_GATHER_PACK_BACKWARD") == "1"
)
# Diagnostic only: distinguish arithmetic cost from the serial sparse
# scheduler. This physically evaluates zero-padded slots and is never eligible
# for public/planner selection under the selected-only execution contract.
USE_DENSE_PACKED_FORWARD_DIAGNOSTIC = (
    os.environ.get("TT_FINETUNE_DYNAMIC_MOE_DENSE_PACKED_DIAGNOSTIC") == "1"
)
# The selected-block BMM schedule is the qualified default for fitting
# no-capacity geometries.  Setting the variable to ``0`` retains the former
# sparse-matmul schedule as an operational fallback and comparison authority.
USE_BLOCK_BATCHED_FORWARD = (
    os.environ.get("TT_FINETUNE_DYNAMIC_MOE_BLOCK_BATCHED", "1") != "0"
)
MAX_BLOCK_BATCHED_WEIGHT_BYTES = int(
    os.environ.get("TT_FINETUNE_DYNAMIC_MOE_BLOCK_BATCHED_MAX_BYTES", 256 << 20)
)
USE_NATIVE_BLOCK_FORWARD_EXPERIMENT = (
    os.environ.get("TT_FINETUNE_DYNAMIC_MOE_NATIVE_BLOCKS") == "1"
)
USE_PARALLEL_SPARSE_FORWARD_EXPERIMENT = (
    os.environ.get("TT_FINETUNE_DYNAMIC_MOE_PARALLEL_SPARSE") == "1"
)
NATIVE_BLOCK_REDUCTION_TILES = int(
    os.environ.get("TT_FINETUNE_DYNAMIC_MOE_NATIVE_BLOCK_KT", "2")
)
NATIVE_BLOCK_ACCUMULATION_SCHEDULE = os.environ.get(
    "TT_FINETUNE_DYNAMIC_MOE_NATIVE_BLOCK_ACCUMULATION",
    "explicit_partial",
)
NATIVE_BLOCK_MATH_APPROX = (
    os.environ.get("TT_FINETUNE_DYNAMIC_MOE_NATIVE_BLOCK_APPROX") == "1"
)


def dynamic_sparse_physical_output_features(output_features: int) -> int:
    """Return the private sparse-kernel width for a logical projection."""

    if output_features <= 0:
        raise ValueError("dynamic sparse expert output width must be positive")
    tiled = ((output_features + TILE - 1) // TILE) * TILE
    return max(MIN_SPARSE_OUTPUT_FEATURES, tiled)


def _pad_sparse_output_weight(weight: torch.Tensor) -> torch.Tensor:
    logical = int(weight.shape[1])
    physical = dynamic_sparse_physical_output_features(logical)
    if physical == logical:
        return weight
    zeros = weight.new_zeros(
        (int(weight.shape[0]), physical - logical, int(weight.shape[2]))
    )
    return torch.cat((weight, zeros), dim=1)


def dynamic_sparse_support_error(
    *, tokens: int, experts: int, input_features: int, output_features: int
) -> str | None:
    if tokens <= 0 or tokens % TILE:
        return "dynamic sparse experts require a positive /32 token count"
    if experts <= 0:
        return "dynamic sparse experts require at least one expert"
    if input_features <= 0 or output_features <= 0:
        return "dynamic sparse expert dimensions must be positive"
    if input_features % TILE or output_features % TILE:
        return "dynamic sparse expert dimensions must be tile aligned"
    return None


def expert_block_sparsity(
    selected_indices: torch.Tensor,
    *,
    expert_count: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact [T,E] routing and TT sparse [1,1,T/32,E] blocks."""

    if selected_indices.ndim != 2:
        raise ValueError("selected expert indices must have shape [tokens, top_k]")
    tokens = int(selected_indices.shape[0])
    error = dynamic_sparse_support_error(
        tokens=tokens,
        experts=expert_count,
        input_features=TILE,
        output_features=TILE,
    )
    if error is not None:
        raise ValueError(error)
    expert_ids = torch.arange(
        expert_count,
        dtype=selected_indices.dtype,
        device=selected_indices.device,
    )
    exact = (
        selected_indices.unsqueeze(-1)
        == expert_ids.reshape(1, 1, expert_count)
    ).any(dim=1)
    blocks = (
        exact.reshape(tokens // TILE, TILE, expert_count)
        .to(torch.int32)
        .sum(dim=1)
        .gt(0)
        .to(dtype)
        .reshape(1, 1, tokens // TILE, expert_count)
    )
    return exact, blocks


def _validate_shared_inputs(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    block_sparsity: torch.Tensor,
    selected_indices: torch.Tensor,
) -> tuple[int, int, int, int]:
    if inputs.ndim != 3 or weight.ndim != 3:
        raise ValueError("dynamic shared expert linear requires rank-3 tensors")
    tokens, top_k, input_features = inputs.shape
    experts, output_features, weight_input = weight.shape
    if input_features != weight_input:
        raise ValueError("dynamic expert linear input/weight mismatch")
    if selected_indices.shape != (tokens, top_k):
        raise ValueError("dynamic expert selection/input shape mismatch")
    if block_sparsity.shape != (1, 1, tokens // TILE, experts):
        raise ValueError("dynamic expert block-sparsity shape mismatch")
    error = dynamic_sparse_support_error(
        tokens=tokens,
        experts=experts,
        input_features=input_features,
        output_features=output_features,
    )
    if error is not None:
        raise ValueError(error)
    return tokens, experts, input_features, output_features


def _validate_expert_inputs(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    block_sparsity: torch.Tensor,
    selected_indices: torch.Tensor,
) -> tuple[int, int, int, int]:
    if inputs.ndim != 3 or weight.ndim != 3:
        raise ValueError("dynamic expert-specific linear requires rank-3 tensors")
    tokens, experts, input_features = inputs.shape
    weight_experts, output_features, weight_input = weight.shape
    if experts != weight_experts or input_features != weight_input:
        raise ValueError("dynamic expert-specific input/weight mismatch")
    if selected_indices.ndim != 2 or selected_indices.shape[0] != tokens:
        raise ValueError("dynamic expert selection/input shape mismatch")
    if block_sparsity.shape != (1, 1, tokens // TILE, experts):
        raise ValueError("dynamic expert block-sparsity shape mismatch")
    error = dynamic_sparse_support_error(
        tokens=tokens,
        experts=experts,
        input_features=input_features,
        output_features=output_features,
    )
    if error is not None:
        raise ValueError(error)
    return tokens, experts, input_features, output_features


def _cpu_active_token_mask(
    block_sparsity: torch.Tensor, *, tokens: int, experts: int
) -> torch.Tensor:
    return (
        block_sparsity.reshape(tokens // TILE, experts)
        .unsqueeze(1)
        .expand(-1, TILE, -1)
        .reshape(tokens, experts)
    )


def packed_expert_block_sparsity(
    selected_indices: torch.Tensor,
    positions: torch.Tensor,
    *,
    expert_count: int,
    slot_count: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return active blocks for exact per-expert no-drop assignment slots."""

    if selected_indices.shape != positions.shape or selected_indices.ndim != 2:
        raise ValueError("selected expert indices/positions must share [T,K]")
    if slot_count <= 0 or slot_count % TILE:
        raise ValueError("dynamic expert slot count must be positive and /32")
    assignments = int(selected_indices.numel())
    flat_indices = selected_indices.reshape(assignments)
    flat_positions = positions.reshape(assignments)
    expert_ids = torch.arange(
        expert_count,
        dtype=flat_indices.dtype,
        device=flat_indices.device,
    )
    block_ids = torch.arange(
        slot_count // TILE,
        dtype=flat_positions.dtype,
        device=flat_positions.device,
    )
    return (
        (
            flat_indices.reshape(assignments, 1, 1)
            == expert_ids.reshape(1, 1, expert_count)
        )
        & (
            (flat_positions // TILE).reshape(assignments, 1, 1)
            == block_ids.reshape(1, slot_count // TILE, 1)
        )
    ).any(dim=0).to(dtype).reshape(
        1, 1, slot_count // TILE, expert_count
    )


def maximum_no_drop_assignment_blocks(
    *, tokens: int, assignments: int, expert_count: int
) -> int:
    """Tight shape-only bound for homogeneous 32-row expert blocks."""

    if min(tokens, assignments, expert_count) <= 0:
        raise ValueError("dynamic block geometry must be positive")
    maximum_per_expert = (tokens + TILE - 1) // TILE
    first_blocks = min(assignments, expert_count)
    remaining = assignments - first_blocks
    return min(
        expert_count * maximum_per_expert,
        first_blocks + remaining // TILE,
    )


def dynamic_block_batched_support_error(
    *,
    tokens: int,
    top_k: int,
    expert_count: int,
    input_features: int,
    output_features: int,
    element_size: int,
) -> str | None:
    """Admit only block schedules with a bounded gathered-weight view.

    The execution evaluates homogeneous selected-expert blocks, but the
    current TT-XLA lowering materializes one frozen-weight view per bounded
    block.  Keep that private representation bounded until a native dynamic
    weight reader is both faster and fully qualified.
    """

    error = dynamic_sparse_support_error(
        tokens=tokens,
        experts=expert_count,
        input_features=input_features,
        output_features=output_features,
    )
    if error is not None:
        return error
    if not 0 < top_k <= expert_count:
        return "dynamic block batching requires top_k in [1, expert_count]"
    if element_size <= 0:
        return "dynamic block batching requires a positive element size"
    block_count = maximum_no_drop_assignment_blocks(
        tokens=tokens,
        assignments=tokens * top_k,
        expert_count=expert_count,
    )
    gathered_weight_bytes = (
        block_count * input_features * output_features * element_size
    )
    if gathered_weight_bytes > MAX_BLOCK_BATCHED_WEIGHT_BYTES:
        return (
            "dynamic selected-block weight view exceeds the qualified "
            f"{MAX_BLOCK_BATCHED_WEIGHT_BYTES}-byte bound "
            f"({gathered_weight_bytes} bytes requested)"
        )
    return None


def _block_assignment_plan(
    selected_indices: torch.Tensor,
    positions: torch.Tensor,
    *,
    expert_count: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Map selected assignments to a bounded list of homogeneous blocks."""

    if selected_indices.shape != positions.shape or selected_indices.ndim != 2:
        raise ValueError("block assignment plan requires matching [T,K] maps")
    tokens = int(selected_indices.shape[0])
    assignments = int(selected_indices.numel())
    maximum_blocks = maximum_no_drop_assignment_blocks(
        tokens=tokens,
        assignments=assignments,
        expert_count=expert_count,
    )
    flat_experts = selected_indices.reshape(assignments)
    flat_positions = positions.reshape(assignments)
    expert_ids = torch.arange(
        expert_count,
        dtype=flat_experts.dtype,
        device=flat_experts.device,
    )
    counts = (
        flat_experts.reshape(assignments, 1)
        == expert_ids.reshape(1, expert_count)
    ).to(torch.int32).sum(dim=0)
    blocks_per_expert = (counts + TILE - 1) // TILE
    ends = torch.cumsum(blocks_per_expert, dim=0, dtype=torch.int32)
    offsets = ends - blocks_per_expert
    assignment_offsets = torch.where(
        flat_experts.reshape(assignments, 1)
        == expert_ids.reshape(1, expert_count),
        offsets.reshape(1, expert_count),
        torch.zeros((), dtype=offsets.dtype, device=offsets.device),
    ).sum(dim=1)
    assignment_blocks = assignment_offsets + flat_positions // TILE
    flat_slots = assignment_blocks * TILE + flat_positions % TILE

    block_ids = torch.arange(
        maximum_blocks,
        dtype=offsets.dtype,
        device=offsets.device,
    )
    owners = (
        block_ids.reshape(maximum_blocks, 1)
        >= offsets.reshape(1, expert_count)
    ) & (
        block_ids.reshape(maximum_blocks, 1)
        < ends.reshape(1, expert_count)
    )
    block_experts = (
        owners.to(expert_ids.dtype)
        * expert_ids.reshape(1, expert_count)
    ).sum(dim=1)
    return flat_slots, block_experts, maximum_blocks


@torch.library.custom_op(
    "tt_finetune::dynamic_pack_assignment_blocks",
    mutates_args=[],
    device_types=["cpu", "xla"],
)
def _dynamic_pack_assignment_blocks_op(
    inputs: torch.Tensor,
    flat_slots: torch.Tensor,
    block_count: int,
) -> torch.Tensor:
    """Pack assignments into fixed homogeneous expert blocks."""

    assignments = int(flat_slots.numel())
    if inputs.ndim != 3 or int(inputs.shape[0] * inputs.shape[1]) != assignments:
        raise ValueError("assignment block pack input/map mismatch")
    physical_slots = int(block_count) * TILE
    slot_ids = torch.arange(
        physical_slots,
        dtype=flat_slots.dtype,
        device=flat_slots.device,
    )
    matches = flat_slots.reshape(assignments, 1) == slot_ids.reshape(
        1, physical_slots
    )
    valid = matches.any(dim=0)
    assignment_ids = torch.arange(
        assignments, dtype=torch.float32, device=inputs.device
    )
    inverse = (
        matches.to(torch.float32)
        * assignment_ids.reshape(assignments, 1)
    ).sum(dim=0).to(torch.long)
    packed = inputs.reshape(assignments, inputs.shape[-1])[inverse]
    return (
        packed * valid.to(inputs.dtype).unsqueeze(-1)
    ).reshape(block_count, TILE, inputs.shape[-1])


@_dynamic_pack_assignment_blocks_op.register_fake
def _dynamic_pack_assignment_blocks_fake(
    inputs: torch.Tensor,
    flat_slots: torch.Tensor,
    block_count: int,
) -> torch.Tensor:
    del flat_slots
    return inputs.new_empty((block_count, TILE, inputs.shape[-1]))


def _block_pack_setup_context(ctx, inputs, output) -> None:
    del output
    values, flat_slots, _block_count = inputs
    ctx.save_for_backward(flat_slots)
    ctx.input_shape = tuple(values.shape)


def _block_pack_backward(ctx, grad_packed):
    (flat_slots,) = ctx.saved_tensors
    restored = grad_packed.reshape(-1, grad_packed.shape[-1])[flat_slots]
    return restored.reshape(ctx.input_shape), None, None


_dynamic_pack_assignment_blocks_op.register_autograd(
    _block_pack_backward, setup_context=_block_pack_setup_context
)


@torch.library.custom_op(
    "tt_finetune::dynamic_restore_assignment_blocks",
    mutates_args=[],
    device_types=["cpu", "xla"],
)
def _dynamic_restore_assignment_blocks_op(
    packed: torch.Tensor,
    flat_slots: torch.Tensor,
    tokens: int,
    top_k: int,
) -> torch.Tensor:
    restored = packed.reshape(-1, packed.shape[-1])[flat_slots]
    return restored.reshape(tokens, top_k, packed.shape[-1])


@_dynamic_restore_assignment_blocks_op.register_fake
def _dynamic_restore_assignment_blocks_fake(
    packed: torch.Tensor,
    flat_slots: torch.Tensor,
    tokens: int,
    top_k: int,
) -> torch.Tensor:
    del flat_slots
    return packed.new_empty((tokens, top_k, packed.shape[-1]))


def _block_restore_setup_context(ctx, inputs, output) -> None:
    del output
    packed, flat_slots, _tokens, _top_k = inputs
    ctx.save_for_backward(flat_slots)
    ctx.packed_shape = tuple(packed.shape)


def _block_restore_backward(ctx, grad_selected):
    (flat_slots,) = ctx.saved_tensors
    assignments = int(flat_slots.numel())
    physical_slots = int(ctx.packed_shape[0] * ctx.packed_shape[1])
    slot_ids = torch.arange(
        physical_slots,
        dtype=flat_slots.dtype,
        device=flat_slots.device,
    )
    dispatch = (
        flat_slots.reshape(assignments, 1)
        == slot_ids.reshape(1, physical_slots)
    ).to(grad_selected.dtype)
    packed = dispatch.transpose(0, 1) @ grad_selected.reshape(
        assignments, grad_selected.shape[-1]
    )
    return packed.reshape(ctx.packed_shape), None, None, None


_dynamic_restore_assignment_blocks_op.register_autograd(
    _block_restore_backward, setup_context=_block_restore_setup_context
)


@torch.library.custom_op(
    "tt_finetune::dynamic_native_expert_blocks",
    mutates_args=[],
    device_types=["cpu", "xla"],
)
def _dynamic_native_expert_blocks_op(
    packed: torch.Tensor,
    expert_weight: torch.Tensor,
    block_experts: torch.Tensor,
    reduction_tiles_per_phase: int,
    accumulation_schedule_id: int,
    math_approx_mode: bool,
) -> torch.Tensor:
    if packed.ndim != 3 or expert_weight.ndim != 3:
        raise ValueError("native expert blocks require rank-three tensors")
    if block_experts.shape != (packed.shape[0],):
        raise ValueError("native expert block owner map mismatch")
    if packed.shape[1] != TILE or packed.shape[2] != expert_weight.shape[2]:
        raise ValueError("native expert block projection mismatch")
    if packed.device.type == "xla":
        from .tt_lang_dynamic_expert_block import (
            build_tt_lang_dynamic_expert_block_forward,
        )

        blocks = int(packed.shape[0])
        input_features = int(packed.shape[2])
        output_features = int(expert_weight.shape[1])
        operation = build_tt_lang_dynamic_expert_block_forward(
            block_count=blocks,
            expert_count=int(expert_weight.shape[0]),
            input_features=input_features,
            output_features=output_features,
            reduction_tiles_per_phase=int(reduction_tiles_per_phase),
            accumulation_schedule=(
                "explicit_partial"
                if accumulation_schedule_id == 0
                else "retain_destination"
                if accumulation_schedule_id == 1
                else "packer_l1"
            ),
            math_approx_mode=bool(math_approx_mode),
        )
        flat_input = packed.reshape(blocks * TILE, input_features)
        weight_t = expert_weight.transpose(1, 2).contiguous().reshape(
            int(expert_weight.shape[0]) * input_features,
            output_features,
        )
        owner_map = block_experts.to(torch.int32).reshape(1, blocks)
        output = torch.empty(
            (blocks * TILE, output_features),
            dtype=packed.dtype,
            device=packed.device,
        )
        return operation(flat_input, weight_t, owner_map, output).reshape(
            blocks, TILE, output_features
        )
    selected_weight = expert_weight[block_experts]
    return torch.bmm(packed, selected_weight.transpose(1, 2))


@_dynamic_native_expert_blocks_op.register_fake
def _dynamic_native_expert_blocks_fake(
    packed: torch.Tensor,
    expert_weight: torch.Tensor,
    block_experts: torch.Tensor,
    reduction_tiles_per_phase: int,
    accumulation_schedule_id: int,
    math_approx_mode: bool,
) -> torch.Tensor:
    del block_experts, reduction_tiles_per_phase, accumulation_schedule_id
    del math_approx_mode
    return packed.new_empty(
        (packed.shape[0], packed.shape[1], expert_weight.shape[1])
    )


def _native_blocks_setup_context(ctx, inputs, output) -> None:
    del output
    _packed, expert_weight, block_experts, phase, schedule, approx = inputs
    ctx.save_for_backward(expert_weight, block_experts)
    ctx.phase = int(phase)
    ctx.schedule = int(schedule)
    ctx.approx = bool(approx)


def _native_blocks_backward(ctx, grad_output):
    expert_weight, block_experts = ctx.saved_tensors
    grad_input = _dynamic_native_expert_blocks_op(
        grad_output,
        expert_weight.transpose(1, 2).contiguous(),
        block_experts,
        ctx.phase,
        ctx.schedule,
        ctx.approx,
    )
    return grad_input, None, None, None, None, None


_dynamic_native_expert_blocks_op.register_autograd(
    _native_blocks_backward, setup_context=_native_blocks_setup_context
)


def dynamic_block_batched_expert_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    selected_indices: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Capacity-free block-batched experiment without per-token all-expert work."""

    flat_slots, block_experts, block_count = _block_assignment_plan(
        selected_indices,
        positions,
        expert_count=int(weight.shape[0]),
    )
    packed = _dynamic_pack_assignment_blocks_op(
        inputs, flat_slots, block_count
    )
    projected = (
        _dynamic_native_expert_blocks_op(
            packed,
            weight,
            block_experts,
            NATIVE_BLOCK_REDUCTION_TILES,
            {
                "explicit_partial": 0,
                "retain_destination": 1,
                "packer_l1": 2,
            }[NATIVE_BLOCK_ACCUMULATION_SCHEDULE],
            NATIVE_BLOCK_MATH_APPROX,
        )
        if USE_NATIVE_BLOCK_FORWARD_EXPERIMENT
        else torch.bmm(packed, weight[block_experts].transpose(1, 2))
    )
    return _dynamic_restore_assignment_blocks_op(
        projected,
        flat_slots,
        int(selected_indices.shape[0]),
        int(selected_indices.shape[1]),
    )


@torch.library.custom_op(
    "tt_finetune::dynamic_pack_expert_assignments",
    mutates_args=[],
    device_types=["cpu", "xla"],
)
def _dynamic_pack_expert_assignments_op(
    inputs: torch.Tensor,
    selected_indices: torch.Tensor,
    positions: torch.Tensor,
    expert_count: int,
) -> torch.Tensor:
    """Pack [T,K,H] assignments into [T,E,H] without dense dispatch GEMM."""

    if inputs.ndim != 3 or selected_indices.shape != inputs.shape[:2]:
        raise ValueError("dynamic assignment pack requires [T,K,H] inputs")
    if positions.shape != selected_indices.shape:
        raise ValueError("dynamic assignment positions must match selection")
    tokens, top_k, width = inputs.shape
    assignments = tokens * top_k
    flat_inputs = inputs.reshape(assignments, width)
    flat_indices = selected_indices.reshape(assignments)
    flat_positions = positions.reshape(assignments)
    expert_ids = torch.arange(
        expert_count,
        dtype=flat_indices.dtype,
        device=flat_indices.device,
    )
    slot_ids = torch.arange(
        tokens,
        dtype=flat_positions.dtype,
        device=flat_positions.device,
    )
    matches = (
        flat_indices.reshape(1, 1, assignments)
        == expert_ids.reshape(expert_count, 1, 1)
    ) & (
        flat_positions.reshape(1, 1, assignments)
        == slot_ids.reshape(1, tokens, 1)
    )
    valid = matches.any(dim=-1)
    assignment_ids = torch.arange(
        assignments, dtype=torch.float32, device=inputs.device
    )
    inverse = (
        matches.to(torch.float32)
        * assignment_ids.reshape(1, 1, assignments)
    ).sum(dim=-1).to(torch.long)
    expert_major = flat_inputs[inverse]
    expert_major = expert_major * valid.to(inputs.dtype).unsqueeze(-1)
    return expert_major.transpose(0, 1)


@_dynamic_pack_expert_assignments_op.register_fake
def _dynamic_pack_expert_assignments_fake(
    inputs: torch.Tensor,
    selected_indices: torch.Tensor,
    positions: torch.Tensor,
    expert_count: int,
) -> torch.Tensor:
    del selected_indices, positions
    return inputs.new_empty((inputs.shape[0], expert_count, inputs.shape[2]))


def _assignment_dispatch_matrix(
    selected_indices: torch.Tensor,
    positions: torch.Tensor,
    *,
    expert_count: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the exact assignment-to-packed-slot map used in gradients.

    Advanced multi-axis indexing lowers correctly in forward on the current TT
    backend but produces invalid values when used inside these autograd
    formulas.  Keep the efficient gather in the user-visible forward and use a
    plain matmul dispatch map at the two backward boundaries.  The map has one
    nonzero per assignment and therefore introduces neither capacity loss nor
    expert duplication.
    """

    tokens = int(selected_indices.shape[0])
    flat_slot = (
        positions.reshape(-1) * int(expert_count)
        + selected_indices.reshape(-1)
    )
    packed_slot = torch.arange(
        tokens * int(expert_count),
        dtype=flat_slot.dtype,
        device=flat_slot.device,
    )
    return (
        flat_slot.reshape(-1, 1) == packed_slot.reshape(1, -1)
    ).to(dtype)


def _backward_restore_assignments(
    packed: torch.Tensor,
    selected_indices: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Reliably restore packed gradients to token-major assignments on TT."""

    # Use a single-axis gather over the physical packed-slot dimension.  The
    # TT lowering corruption is specific to the multi-axis advanced indexing
    # form; this flattened form shares the already-qualified forward gather
    # lowering and performs no reduction or rounding.
    flat_slot = (
        positions.reshape(-1) * int(packed.shape[1])
        + selected_indices.reshape(-1)
    )
    return packed.reshape(-1, packed.shape[-1])[flat_slot].reshape(
        *selected_indices.shape, packed.shape[-1]
    )


def _backward_pack_assignments(
    values: torch.Tensor,
    selected_indices: torch.Tensor,
    positions: torch.Tensor,
    *,
    expert_count: int,
) -> torch.Tensor:
    """Reliably pack token-major gradients into no-drop expert slots on TT."""

    if USE_GATHER_PACK_BACKWARD:
        return _dynamic_pack_expert_assignments_op(
            values, selected_indices, positions, expert_count
        )

    dispatch = _assignment_dispatch_matrix(
        selected_indices,
        positions,
        expert_count=expert_count,
        dtype=values.dtype,
    )
    packed = dispatch.transpose(0, 1) @ values.reshape(-1, values.shape[-1])
    return packed.reshape(
        selected_indices.shape[0], expert_count, values.shape[-1]
    )


def _pack_setup_context(ctx, inputs, output) -> None:
    del output
    values, selected_indices, positions, expert_count = inputs
    ctx.save_for_backward(selected_indices, positions)
    ctx.input_shape = tuple(values.shape)
    ctx.expert_count = int(expert_count)


def _pack_backward(ctx, grad_packed):
    selected_indices, positions = ctx.saved_tensors
    grad_inputs = _backward_restore_assignments(
        grad_packed, selected_indices, positions
    )
    return grad_inputs.reshape(ctx.input_shape), None, None, None


_dynamic_pack_expert_assignments_op.register_autograd(
    _pack_backward, setup_context=_pack_setup_context
)


@torch.library.custom_op(
    "tt_finetune::dynamic_restore_expert_assignments",
    mutates_args=[],
    device_types=["cpu", "xla"],
)
def _dynamic_restore_expert_assignments_op(
    packed: torch.Tensor,
    selected_indices: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Restore [T,E,H] expert slots into token-major [T,K,H]."""

    if packed.ndim != 3 or selected_indices.shape != positions.shape:
        raise ValueError("dynamic assignment restore geometry mismatch")
    if packed.shape[0] != selected_indices.shape[0]:
        raise ValueError("dynamic assignment restore token mismatch")
    return packed[positions, selected_indices]


@_dynamic_restore_expert_assignments_op.register_fake
def _dynamic_restore_expert_assignments_fake(
    packed: torch.Tensor,
    selected_indices: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    del positions
    return packed.new_empty(
        (packed.shape[0], selected_indices.shape[1], packed.shape[2])
    )


def _restore_setup_context(ctx, inputs, output) -> None:
    del output
    packed, selected_indices, positions = inputs
    ctx.save_for_backward(selected_indices, positions)
    ctx.expert_count = int(packed.shape[1])


def _restore_backward(ctx, grad_selected):
    selected_indices, positions = ctx.saved_tensors
    grad_packed = _backward_pack_assignments(
        grad_selected,
        selected_indices,
        positions,
        expert_count=ctx.expert_count,
    )
    return grad_packed, None, None


_dynamic_restore_expert_assignments_op.register_autograd(
    _restore_backward, setup_context=_restore_setup_context
)


@torch.library.custom_op(
    "tt_finetune::dynamic_sparse_expert_input_gradient",
    mutates_args=[],
    device_types=["cpu", "xla"],
)
def _dynamic_sparse_expert_input_gradient_op(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    block_sparsity: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token/per-expert dInput without gathering expert weights.

    ``weight[selected_indices]`` lowers to an embedding whose logical output is
    T*K*O*I.  For real MoE weights that temporary is gigabytes and its TTNN
    circular buffer exceeds per-core L1.  The transpose-aware sparse matmul
    below consumes the unchanged expert-major weight once and schedules only
    active token/expert blocks.  Exact selected lanes are gathered only after
    this much smaller T*E*I result exists.
    """

    if grad_output.ndim != 3 or weight.ndim != 3:
        raise ValueError("dynamic sparse input gradient requires rank-3 tensors")
    tokens, experts, output_features = grad_output.shape
    weight_experts, weight_output, input_features = weight.shape
    if experts != weight_experts or output_features != weight_output:
        raise ValueError("dynamic sparse input-gradient geometry mismatch")
    if block_sparsity.shape != (1, 1, tokens // TILE, experts):
        raise ValueError("dynamic sparse input-gradient routing shape mismatch")
    error = dynamic_sparse_support_error(
        tokens=tokens,
        experts=experts,
        input_features=output_features,
        output_features=input_features,
    )
    if error is not None:
        raise ValueError(error)

    if grad_output.device.type == "xla":
        from torch_xla.experimental import stablehlo_custom_call

        tiled_gradient = grad_output.reshape(
            tokens // TILE, TILE, experts, output_features
        ).permute(0, 2, 1, 3).contiguous()
        transposed_weight = weight.transpose(1, 2).contiguous()
        physical_weight = _pad_sparse_output_weight(transposed_weight)
        physical_input_features = int(physical_weight.shape[1])
        right = physical_weight.transpose(1, 2).unsqueeze(0).contiguous()
        tiled = stablehlo_custom_call.stablehlo_custom_call(
            [tiled_gradient, right, block_sparsity],
            "tt.sparse_matmul",
            [[tokens // TILE, experts, TILE, physical_input_features]],
            [grad_output.dtype],
            frontend_attributes={
                "is_input_a_sparse": "True",
                "is_input_b_sparse": "False",
                "nnz": "0",
            },
        )
        return tiled.permute(0, 2, 1, 3).reshape(
            tokens, experts, physical_input_features
        )[:, :, :input_features]

    result = torch.einsum(
        "teo,eoi->tei", grad_output.float(), weight.float()
    ).to(grad_output.dtype)
    active = _cpu_active_token_mask(
        block_sparsity, tokens=tokens, experts=experts
    )
    return result * active.to(result.dtype).unsqueeze(-1)


@_dynamic_sparse_expert_input_gradient_op.register_fake
def _dynamic_sparse_expert_input_gradient_fake(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    block_sparsity: torch.Tensor,
) -> torch.Tensor:
    del block_sparsity
    return grad_output.new_empty(
        (grad_output.shape[0], grad_output.shape[1], weight.shape[2])
    )


@torch.library.custom_op(
    "tt_finetune::dynamic_shared_input_expert_linear",
    mutates_args=[],
    device_types=["cpu", "xla"],
)
def _dynamic_shared_input_expert_linear_op(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    block_sparsity: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    """Selected-expert projection with an autograd-visible routing owner.

    Keeping ``selected_indices`` in this public operator schema is essential.
    Dynamo may trace through a Python ``autograd.Function`` and expose the raw
    inference-only ``tt.sparse_matmul`` to autograd.  This registered operator
    remains opaque, lowers to the same TT sparse custom call, and gives the
    backward formula the exact token/expert assignments instead of only the
    coarser 32-token block bitmap.
    """

    tokens, experts, input_features, output_features = _validate_shared_inputs(
        inputs, weight, block_sparsity, selected_indices
    )
    physical_weight = _pad_sparse_output_weight(weight)
    physical_output_features = int(physical_weight.shape[1])
    if inputs.device.type == "xla":
        from torch_xla.experimental import stablehlo_custom_call

        left = inputs[:, 0, :].reshape(
            1, tokens // TILE, TILE, input_features
        )
        right = physical_weight.transpose(1, 2).unsqueeze(0).contiguous()
        sparsity = block_sparsity.reshape(
            1, tokens // TILE, 1, experts
        )
        tiled = stablehlo_custom_call.stablehlo_custom_call(
            [left, right, sparsity],
            "tt.sparse_matmul",
            [[1, tokens // TILE, 1, experts, TILE, physical_output_features]],
            [inputs.dtype],
            frontend_attributes={
                "is_input_a_sparse": "False",
                "is_input_b_sparse": "True",
                "nnz": "0",
            },
        )
        return tiled.squeeze(0).squeeze(1).permute(0, 2, 1, 3).reshape(
            tokens, experts, physical_output_features
        )[:, :, :output_features]

    base = torch.einsum(
        "ti,eoi->teo", inputs[:, 0, :].float(), weight.float()
    ).to(inputs.dtype)
    active = _cpu_active_token_mask(
        block_sparsity, tokens=tokens, experts=experts
    )
    return base * active.to(base.dtype).unsqueeze(-1)


@_dynamic_shared_input_expert_linear_op.register_fake
def _dynamic_shared_input_expert_linear_fake(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    block_sparsity: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    del block_sparsity, selected_indices
    return inputs.new_empty((inputs.shape[0], weight.shape[0], weight.shape[1]))


def _shared_setup_context(ctx, inputs, output) -> None:
    del output
    values, weight, block_sparsity, selected_indices = inputs
    ctx.save_for_backward(values, weight, block_sparsity, selected_indices)


def _shared_backward(ctx, grad_output):
    inputs, weight, block_sparsity, selected_indices = ctx.saved_tensors
    tokens, top_k = selected_indices.shape
    token_ids = torch.arange(
        tokens, dtype=selected_indices.dtype, device=selected_indices.device
    ).unsqueeze(1)
    if ctx.needs_input_grad[1]:
        raise RuntimeError(
            "dynamic sparse expert base weights must remain frozen"
        )
    per_expert = _dynamic_sparse_expert_input_gradient_op(
        grad_output, weight, block_sparsity
    )
    grad_inputs = per_expert[token_ids, selected_indices]
    return grad_inputs, None, None, None


_dynamic_shared_input_expert_linear_op.register_autograd(
    _shared_backward, setup_context=_shared_setup_context
)


@torch.library.custom_op(
    "tt_finetune::dynamic_parallel_sparse_expert_linear",
    mutates_args=[],
    device_types=["cpu", "xla"],
)
def _dynamic_parallel_sparse_expert_linear_op(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    block_sparsity: torch.Tensor,
) -> torch.Tensor:
    if inputs.ndim != 3 or weight.ndim != 3:
        raise ValueError("parallel sparse expert linear requires rank-3 tensors")
    tokens, experts, input_features = inputs.shape
    weight_experts, output_features, weight_input = weight.shape
    if experts != weight_experts or input_features != weight_input:
        raise ValueError("parallel sparse expert geometry mismatch")
    token_blocks = tokens // TILE
    if block_sparsity.shape != (1, 1, token_blocks, experts):
        raise ValueError("parallel sparse expert routing shape mismatch")
    if token_blocks > TILE or experts > TILE:
        raise ValueError("parallel sparse expert currently requires one routing tile")
    physical_weight = _pad_sparse_output_weight(weight)
    physical_output_features = int(physical_weight.shape[1])
    if inputs.device.type == "xla":
        from .tt_lang_dynamic_sparse_projection import (
            build_tt_lang_dynamic_sparse_projection,
        )

        operation = build_tt_lang_dynamic_sparse_projection(
            token_blocks=token_blocks,
            expert_count=experts,
            input_features=input_features,
            output_features=physical_output_features,
        )
        tiled_inputs = inputs.reshape(
            token_blocks, TILE, experts, input_features
        ).permute(0, 2, 1, 3).contiguous()
        flat_inputs = tiled_inputs.reshape(
            token_blocks * experts * TILE, input_features
        )
        weight_t = physical_weight.transpose(1, 2).contiguous().reshape(
            experts * input_features, physical_output_features
        )
        output = torch.empty(
            (token_blocks * experts * TILE, physical_output_features),
            dtype=inputs.dtype,
            device=inputs.device,
        )
        projected = operation(flat_inputs, weight_t, block_sparsity, output)
        return projected.reshape(
            token_blocks, experts, TILE, physical_output_features
        ).permute(0, 2, 1, 3).reshape(
            tokens, experts, physical_output_features
        )[:, :, :output_features]
    projected = torch.einsum(
        "tei,eoi->teo", inputs.float(), weight.float()
    ).to(inputs.dtype)
    active = _cpu_active_token_mask(
        block_sparsity, tokens=tokens, experts=experts
    )
    return projected * active.to(projected.dtype).unsqueeze(-1)


@_dynamic_parallel_sparse_expert_linear_op.register_fake
def _dynamic_parallel_sparse_expert_linear_fake(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    block_sparsity: torch.Tensor,
) -> torch.Tensor:
    del block_sparsity
    return inputs.new_empty((inputs.shape[0], weight.shape[0], weight.shape[1]))


def _parallel_sparse_setup_context(ctx, inputs, output) -> None:
    del output
    _values, weight, block_sparsity = inputs
    ctx.save_for_backward(weight, block_sparsity)


def _parallel_sparse_backward(ctx, grad_output):
    weight, block_sparsity = ctx.saved_tensors
    grad_inputs = _dynamic_parallel_sparse_expert_linear_op(
        grad_output, weight.transpose(1, 2), block_sparsity
    )
    return grad_inputs, None, None


_dynamic_parallel_sparse_expert_linear_op.register_autograd(
    _parallel_sparse_backward, setup_context=_parallel_sparse_setup_context
)


@torch.library.custom_op(
    "tt_finetune::dynamic_expert_input_linear",
    mutates_args=[],
    device_types=["cpu", "xla"],
)
def _dynamic_expert_input_linear_op(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    block_sparsity: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    tokens, experts, input_features, output_features = _validate_expert_inputs(
        inputs, weight, block_sparsity, selected_indices
    )
    tiled_inputs = inputs.reshape(
        tokens // TILE, TILE, experts, input_features
    ).permute(0, 2, 1, 3).contiguous()
    physical_weight = _pad_sparse_output_weight(weight)
    physical_output_features = int(physical_weight.shape[1])
    if inputs.device.type == "xla":
        from torch_xla.experimental import stablehlo_custom_call

        right = physical_weight.transpose(1, 2).unsqueeze(0).contiguous()
        tiled = stablehlo_custom_call.stablehlo_custom_call(
            [tiled_inputs, right, block_sparsity],
            "tt.sparse_matmul",
            [[tokens // TILE, experts, TILE, physical_output_features]],
            [inputs.dtype],
            frontend_attributes={
                "is_input_a_sparse": "True",
                "is_input_b_sparse": "False",
                "nnz": "0",
            },
        )
        return tiled.permute(0, 2, 1, 3).reshape(
            tokens, experts, physical_output_features
        )[:, :, :output_features]

    base = torch.einsum("tei,eoi->teo", inputs.float(), weight.float()).to(
        inputs.dtype
    )
    active = _cpu_active_token_mask(
        block_sparsity, tokens=tokens, experts=experts
    )
    return base * active.to(base.dtype).unsqueeze(-1)


@_dynamic_expert_input_linear_op.register_fake
def _dynamic_expert_input_linear_fake(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    block_sparsity: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    del block_sparsity, selected_indices
    return inputs.new_empty((inputs.shape[0], weight.shape[0], weight.shape[1]))


def _expert_setup_context(ctx, inputs, output) -> None:
    del output
    values, weight, block_sparsity, selected_indices = inputs
    ctx.save_for_backward(values, weight, block_sparsity, selected_indices)


def _expert_backward(ctx, grad_output):
    _inputs, weight, block_sparsity, _selected_indices = ctx.saved_tensors
    if ctx.needs_input_grad[1]:
        raise RuntimeError(
            "dynamic sparse expert base weights must remain frozen"
        )
    grad_inputs = _dynamic_sparse_expert_input_gradient_op(
        grad_output, weight, block_sparsity
    )
    return grad_inputs, None, None, None


_dynamic_expert_input_linear_op.register_autograd(
    _expert_backward, setup_context=_expert_setup_context
)


def dynamic_shared_input_expert_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    block_sparsity: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    return _dynamic_shared_input_expert_linear_op(
        inputs, weight, block_sparsity, selected_indices
    )


def dynamic_pack_expert_assignments(
    inputs: torch.Tensor,
    selected_indices: torch.Tensor,
    positions: torch.Tensor,
    expert_count: int,
) -> torch.Tensor:
    return _dynamic_pack_expert_assignments_op(
        inputs, selected_indices, positions, expert_count
    )


def dynamic_restore_expert_assignments(
    packed: torch.Tensor,
    selected_indices: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    return _dynamic_restore_expert_assignments_op(
        packed, selected_indices, positions
    )


def dynamic_expert_input_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    block_sparsity: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    if USE_DENSE_PACKED_FORWARD_DIAGNOSTIC:
        return torch.bmm(
            inputs.transpose(0, 1), weight.transpose(1, 2)
        ).transpose(0, 1)
    if USE_PARALLEL_SPARSE_FORWARD_EXPERIMENT:
        return _dynamic_parallel_sparse_expert_linear_op(
            inputs, weight, block_sparsity
        )
    return _dynamic_expert_input_linear_op(
        inputs, weight, block_sparsity, selected_indices
    )


__all__ = [
    "dynamic_block_batched_support_error",
    "dynamic_block_batched_expert_linear",
    "dynamic_expert_input_linear",
    "dynamic_pack_expert_assignments",
    "dynamic_restore_expert_assignments",
    "dynamic_sparse_physical_output_features",
    "dynamic_shared_input_expert_linear",
    "dynamic_sparse_support_error",
    "expert_block_sparsity",
    "maximum_no_drop_assignment_blocks",
    "packed_expert_block_sparsity",
    "USE_BLOCK_BATCHED_FORWARD",
    "USE_PARALLEL_SPARSE_FORWARD_EXPERIMENT",
]
