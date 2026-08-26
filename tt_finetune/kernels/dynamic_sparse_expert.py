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

import torch


TILE = 32
# The current Blackhole sparse-matmul 1D multicast implementation does not
# make forward progress for a single output tile (N=32).  Both sparse-A and
# sparse-B paths are silicon-qualified once the physical output spans sixteen
# tiles.  Keep the model-visible width unchanged and pad only the private
# frozen projection presented to the sparse kernel.  Real expert projections
# at N>=512 pay no padding cost.
MIN_SPARSE_OUTPUT_FEATURES = 16 * TILE


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


def _sparse_matmul(
    left: torch.Tensor,
    right: torch.Tensor,
    sparsity: torch.Tensor,
    *,
    input_a_sparse: bool,
) -> torch.Tensor:
    # Importing tt_torch registers torch.ops.tt.* without making TT-XLA a
    # package import requirement for users who only inspect the planner.
    import tt_torch  # noqa: F401

    return torch.ops.tt.sparse_matmul(
        left,
        right,
        sparsity,
        nnz=0,
        is_input_a_sparse=input_a_sparse,
        is_input_b_sparse=not input_a_sparse,
    )


class _SharedInputExpertLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, block_sparsity, selected_indices):
        # inputs [T,K,I] is the same expanded view used by the safe selected
        # path.  Forward needs one physical token row; backward deliberately
        # returns one gradient per top-k lane so PyTorch performs the final K
        # reduction at the same graph boundary as the reference.
        tokens, top_k, input_features = inputs.shape
        experts, output_features, weight_input = weight.shape
        if input_features != weight_input:
            raise ValueError("dynamic expert linear input/weight mismatch")
        error = dynamic_sparse_support_error(
            tokens=tokens,
            experts=experts,
            input_features=input_features,
            output_features=output_features,
        )
        if error is not None:
            raise ValueError(error)
        physical_weight = _pad_sparse_output_weight(weight)
        physical_output_features = int(physical_weight.shape[1])
        tiled = _sparse_matmul(
            inputs[:, 0, :].reshape(1, 1, tokens, input_features),
            physical_weight.transpose(1, 2).unsqueeze(0).contiguous(),
            block_sparsity,
            input_a_sparse=False,
        )
        # [1,Tb,E,32,O] -> [T,E,O]
        output = tiled.squeeze(0).permute(0, 2, 1, 3).reshape(
            tokens, experts, physical_output_features
        )[:, :, :output_features]
        ctx.save_for_backward(inputs, weight, selected_indices)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, selected_indices = ctx.saved_tensors
        tokens, top_k = selected_indices.shape
        token_ids = torch.arange(
            tokens, dtype=selected_indices.dtype, device=selected_indices.device
        ).unsqueeze(1)
        selected_grad = grad_output[token_ids, selected_indices]
        selected_weight = weight[selected_indices]
        grad_inputs = torch.bmm(
            selected_grad.reshape(-1, 1, selected_grad.shape[-1]),
            selected_weight.reshape(
                -1, selected_weight.shape[-2], selected_weight.shape[-1]
            ),
        ).reshape(tokens, top_k, inputs.shape[-1])
        grad_weight = None
        if ctx.needs_input_grad[1]:
            per_assignment = torch.bmm(
                selected_grad.reshape(-1, selected_grad.shape[-1], 1),
                inputs.reshape(-1, 1, inputs.shape[-1]),
            )
            expert_ids = torch.arange(
                weight.shape[0],
                dtype=selected_indices.dtype,
                device=selected_indices.device,
            )
            selector = (
                selected_indices.reshape(-1, 1)
                == expert_ids.reshape(1, -1)
            ).to(per_assignment.dtype)
            grad_weight = torch.matmul(
                selector.transpose(0, 1),
                per_assignment.reshape(per_assignment.shape[0], -1),
            ).reshape_as(weight)
        return grad_inputs, grad_weight, None, None


class _ExpertInputLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, block_sparsity, selected_indices):
        # inputs [T,E,I], weight [E,O,I]
        tokens, experts, input_features = inputs.shape
        weight_experts, output_features, weight_input = weight.shape
        if experts != weight_experts or input_features != weight_input:
            raise ValueError("dynamic expert-specific input/weight mismatch")
        error = dynamic_sparse_support_error(
            tokens=tokens,
            experts=experts,
            input_features=input_features,
            output_features=output_features,
        )
        if error is not None:
            raise ValueError(error)
        tiled_inputs = inputs.reshape(
            tokens // TILE, TILE, experts, input_features
        ).permute(0, 2, 1, 3).contiguous()
        physical_weight = _pad_sparse_output_weight(weight)
        physical_output_features = int(physical_weight.shape[1])
        tiled = _sparse_matmul(
            tiled_inputs,
            physical_weight.transpose(1, 2).unsqueeze(0).contiguous(),
            block_sparsity,
            input_a_sparse=True,
        )
        output = tiled.permute(0, 2, 1, 3).reshape(
            tokens, experts, physical_output_features
        )[:, :, :output_features]
        ctx.save_for_backward(inputs, weight, selected_indices)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, selected_indices = ctx.saved_tensors
        tokens, top_k = selected_indices.shape
        token_ids = torch.arange(
            tokens, dtype=selected_indices.dtype, device=selected_indices.device
        ).unsqueeze(1)
        selected_grad = grad_output[token_ids, selected_indices]
        selected_weight = weight[selected_indices]
        selected_inputs = inputs[token_ids, selected_indices]
        selected_grad_inputs = torch.bmm(
            selected_grad.reshape(-1, 1, selected_grad.shape[-1]),
            selected_weight.reshape(
                -1, selected_weight.shape[-2], selected_weight.shape[-1]
            ),
        ).reshape(tokens, top_k, inputs.shape[-1])
        expert_ids = torch.arange(
            weight.shape[0],
            dtype=selected_indices.dtype,
            device=selected_indices.device,
        )
        selector_3d = (
            selected_indices.unsqueeze(-1)
            == expert_ids.reshape(1, 1, -1)
        ).to(selected_grad_inputs.dtype)
        grad_inputs = (
            selected_grad_inputs.unsqueeze(2)
            * selector_3d.unsqueeze(-1)
        ).sum(dim=1)
        grad_weight = None
        if ctx.needs_input_grad[1]:
            per_assignment = torch.bmm(
                selected_grad.reshape(-1, selected_grad.shape[-1], 1),
                selected_inputs.reshape(-1, 1, inputs.shape[-1]),
            )
            selector = selector_3d.reshape(-1, weight.shape[0])
            grad_weight = torch.matmul(
                selector.transpose(0, 1),
                per_assignment.reshape(per_assignment.shape[0], -1),
            ).reshape_as(weight)
        return grad_inputs, grad_weight, None, None


def dynamic_shared_input_expert_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    block_sparsity: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    return _SharedInputExpertLinear.apply(
        inputs, weight, block_sparsity, selected_indices
    )


def dynamic_expert_input_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    block_sparsity: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    return _ExpertInputLinear.apply(
        inputs, weight, block_sparsity, selected_indices
    )


__all__ = [
    "dynamic_expert_input_linear",
    "dynamic_sparse_physical_output_features",
    "dynamic_shared_input_expert_linear",
    "dynamic_sparse_support_error",
    "expert_block_sparsity",
]
