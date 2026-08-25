"""Architecture-general deterministic sparse-MoE execution for Blackhole.

The optimized path routes tensors entirely on device and evaluates only the
selected token/expert assignments.  A deliberately dense safe path shares the
same routing and expert arithmetic contract and exists as an exact oracle and
fallback for unqualified schedules.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from tt_finetune.architecture import (
    CanonicalArchitectureDescriptor,
    CanonicalExpertLinearRole,
    CanonicalModelMap,
)


@dataclass(frozen=True)
class SparseMoEInstallReport:
    backend: str
    layers: int
    expert_count: int
    top_k: int
    rank: int
    host_token_routing_loop: bool
    evaluates_all_experts: bool
    deterministic_contract: str


def deterministic_topk(
    logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select experts in descending-logit order, lower expert id on ties.

    A pairwise rank computes the lexicographic order directly.  This avoids
    both stable sort (not implemented by current TTNN) and scatter (not legal
    in the current TT lowering), while keeping routing entirely on device.
    """

    if logits.ndim != 2:
        raise ValueError("router logits must have shape [tokens, experts]")
    if not 0 < top_k <= logits.shape[-1]:
        raise ValueError("top_k must be in [1, expert_count]")
    fp32_logits = logits.float()
    experts = logits.shape[-1]
    expert_ids = torch.arange(experts, dtype=torch.int64, device=logits.device)
    candidate = fp32_logits.unsqueeze(-1)
    other = fp32_logits.unsqueeze(-2)
    earlier_id = expert_ids.reshape(1, 1, experts) < expert_ids.reshape(
        1, experts, 1
    )
    outranks = (other > candidate) | ((other == candidate) & earlier_id)
    rank = outranks.to(torch.int64).sum(dim=-1)
    selected_ranks = torch.arange(
        top_k, dtype=rank.dtype, device=rank.device
    ).reshape(1, top_k, 1)
    selector_bool = rank.unsqueeze(1) == selected_ranks
    indices = (
        selector_bool.to(torch.int64) * expert_ids.reshape(1, 1, experts)
    ).sum(dim=-1).detach()
    # Re-form selected logits through a dense differentiable contraction.
    # Top-k indices are discrete; this preserves the required gradient through
    # selected router logits without emitting unsupported scatter-backward.
    # ``where`` is intentional: group-limited routing masks excluded experts
    # with -inf, and multiplication would turn 0 * -inf into NaN.
    values = torch.where(
        selector_bool,
        fp32_logits.unsqueeze(1),
        torch.zeros((), dtype=fp32_logits.dtype, device=fp32_logits.device),
    ).sum(dim=-1)
    return values, indices


def _expert_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Batched F.linear for [..., input] and [..., output, input]."""

    leading = inputs.shape[:-1]
    input_features = inputs.shape[-1]
    output_features = weight.shape[-2]
    flat_inputs = inputs.reshape(-1, input_features, 1)
    flat_weight = weight.reshape(-1, output_features, input_features)
    return torch.bmm(flat_weight, flat_inputs).squeeze(-1).reshape(
        *leading, output_features
    )


class DeviceSparseMoELoRA(nn.Module):
    """One canonical sparse expert block with expert LoRA adapters."""

    def __init__(
        self,
        original: nn.Module,
        *,
        router_layer: nn.Module,
        input_weight: torch.Tensor,
        output_weight: torch.Tensor,
        rank: int,
        alpha: int,
        top_k: int,
        backend: str,
    ) -> None:
        super().__init__()
        if backend not in {"safe", "sparse"}:
            raise ValueError("MoE backend must be safe or sparse")
        if input_weight.ndim != 3 or output_weight.ndim != 3:
            raise ValueError("expert weights must have [experts, out, in] layout")
        expert_count, fused_width, hidden_size = input_weight.shape
        output_experts, output_hidden, intermediate = output_weight.shape
        if output_experts != expert_count or output_hidden != hidden_size:
            raise ValueError("inconsistent expert input/output geometry")
        if fused_width != 2 * intermediate:
            raise ValueError("initial sparse owner requires fused SwiGLU experts")
        self.router_layer = router_layer
        self.input_weight = nn.Parameter(input_weight.detach(), requires_grad=False)
        self.output_weight = nn.Parameter(output_weight.detach(), requires_grad=False)
        self.expert_count = int(expert_count)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate)
        self.top_k = int(top_k)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.backend = backend
        self._original_type = type(original).__name__
        self.last_route_indices: torch.Tensor | None = None

        self.lora_A_gate_up = nn.Parameter(
            torch.empty(expert_count, rank, hidden_size, dtype=input_weight.dtype)
        )
        self.lora_B_gate_up = nn.Parameter(
            torch.zeros(expert_count, fused_width, rank, dtype=input_weight.dtype)
        )
        self.lora_A_down = nn.Parameter(
            torch.empty(expert_count, rank, intermediate, dtype=output_weight.dtype)
        )
        self.lora_B_down = nn.Parameter(
            torch.zeros(expert_count, hidden_size, rank, dtype=output_weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A_gate_up, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_down, a=math.sqrt(5))

    def _selected_weights(
        self,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        expert_ids = torch.arange(
            self.expert_count, dtype=indices.dtype, device=indices.device
        )
        selector = (indices.unsqueeze(-1) == expert_ids).to(
            self.lora_A_gate_up.dtype
        )

        def differentiable_select(value: torch.Tensor) -> torch.Tensor:
            flat = torch.matmul(
                selector.reshape(-1, self.expert_count),
                value.reshape(self.expert_count, -1),
            )
            return flat.reshape(*indices.shape, *value.shape[1:])

        return (
            self.input_weight[indices],
            self.output_weight[indices],
            differentiable_select(self.lora_A_gate_up),
            differentiable_select(self.lora_B_gate_up),
            differentiable_select(self.lora_A_down),
            differentiable_select(self.lora_B_down),
        )

    def _experts(
        self,
        inputs: torch.Tensor,
        weights: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        base_in, base_out, a_in, b_in, a_out, b_out = weights
        gate_up = _expert_linear(inputs, base_in)
        low_in = _expert_linear(inputs, a_in)
        gate_up = gate_up + _expert_linear(low_in, b_in) * self.scaling
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        output = _expert_linear(hidden, base_out)
        low_out = _expert_linear(hidden, a_out)
        return output + _expert_linear(low_out, b_out) * self.scaling

    def forward(self, layer_input: torch.Tensor):
        original_shape = layer_input.shape
        flat = layer_input.reshape(-1, original_shape[-1])
        router_logits = self.router_layer(flat).float()
        selected_logits, selected_indices = deterministic_topk(
            router_logits, self.top_k
        )
        gates = torch.softmax(selected_logits, dim=-1).to(flat.dtype)
        # Retain device routing metadata for optional, out-of-band statistics.
        self.last_route_indices = selected_indices.detach()

        # Both contracts use the canonical selected-assignment arithmetic so
        # their backward reduction boundaries are bit-identical.  ``safe`` is
        # the unfused Torch/XLA graph; ``sparse`` is the planner's optimized
        # device schedule.  Neither fallback silently turns sparse MoE into
        # dense all-expert compute.
        selected_inputs = flat.unsqueeze(1).expand(-1, self.top_k, -1)
        expert_outputs = self._experts(
            selected_inputs, self._selected_weights(selected_indices)
        )

        combined = (expert_outputs * gates.unsqueeze(-1)).sum(dim=1)
        return combined.reshape(*original_shape), router_logits


class GenericDeviceSparseMoELoRA(nn.Module):
    """Selected-only routed experts for fused or separate storage semantics.

    Separate expert modules are stacked once during installation.  Runtime
    routing, group filtering, capacity ranking, expert selection, routed
    LoRA, shared experts and router losses stay as device tensor operations;
    there is no Python token/expert loop in ``forward``.
    """

    def __init__(
        self,
        original: nn.Module,
        *,
        router_layer: nn.Module,
        routed_weights: Mapping[str, torch.Tensor],
        shared_weights: Mapping[str, torch.Tensor] | None,
        shared_gate: nn.Module | None,
        rank: int,
        alpha: int,
        top_k: int,
        backend: str,
        router_normalization: str,
        expert_group_count: int | None = None,
        expert_group_top_k: int | None = None,
        auxiliary_loss_coefficient: float = 0.0,
        router_z_loss_coefficient: float = 0.0,
        capacity_factor: float | None = None,
        minimum_capacity: int | None = None,
        overflow_policy: str = "none",
        router_jitter_noise: float = 0.0,
        output_contract: str = "hidden_router_tuple",
    ) -> None:
        super().__init__()
        if backend not in {"safe", "sparse"}:
            raise ValueError("MoE backend must be safe or sparse")
        if output_contract not in {"hidden", "hidden_router_tuple"}:
            raise ValueError("unsupported MoE module output contract")
        allowed_normalization = {
            "softmax_selected",
            "softmax_all_selected",
            "softmax_all_renormalized_selected",
            "sigmoid_selected",
            "sigmoid_selected_unnormalized",
            "unnormalized_selected",
        }
        if router_normalization not in allowed_normalization:
            raise ValueError(
                f"unsupported router normalization: {router_normalization}"
            )
        if overflow_policy not in {"none", "drop", "residual"}:
            raise ValueError(
                f"unsupported executable overflow policy: {overflow_policy}"
            )
        self.router_layer = router_layer
        self.routed_weights = nn.ParameterDict({
            role: nn.Parameter(value.detach(), requires_grad=False)
            for role, value in routed_weights.items()
        })
        self.shared_weights = nn.ParameterDict({
            role: nn.Parameter(value.detach(), requires_grad=False)
            for role, value in (shared_weights or {}).items()
        })
        first = next(iter(self.routed_weights.values()), None)
        if first is None or first.ndim != 3:
            raise ValueError("routed expert weights must be [experts, out, in]")
        self.expert_count = int(first.shape[0])
        self.hidden_size = int(first.shape[-1])
        if "down" not in self.routed_weights:
            raise ValueError("routed expert weights require down projection")
        self.intermediate_size = int(self.routed_weights["down"].shape[-1])
        expected_roles = (
            {"gate_up", "down"}
            if "gate_up" in self.routed_weights
            else {"gate", "up", "down"}
            if "gate" in self.routed_weights
            else {"up", "down"}
        )
        if set(self.routed_weights) != expected_roles:
            raise ValueError(
                "routed expert weights do not form a canonical MLP: "
                f"{sorted(self.routed_weights)}"
            )
        if any(
            value.ndim != 3 or int(value.shape[0]) != self.expert_count
            for value in self.routed_weights.values()
        ):
            raise ValueError("routed expert tensors disagree on expert axis")
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.top_k = int(top_k)
        self.backend = backend
        self.router_normalization = router_normalization
        self.expert_group_count = expert_group_count
        self.expert_group_top_k = expert_group_top_k
        self.auxiliary_loss_coefficient = float(auxiliary_loss_coefficient)
        self.router_z_loss_coefficient = float(router_z_loss_coefficient)
        self.capacity_factor = capacity_factor
        self.minimum_capacity = minimum_capacity
        self.overflow_policy = overflow_policy
        self.router_jitter_noise = float(router_jitter_noise)
        self.output_contract = output_contract
        self._original_type = type(original).__name__
        self.last_route_indices: torch.Tensor | None = None
        self.last_router_logits: torch.Tensor | None = None
        self.last_auxiliary_loss: torch.Tensor | None = None
        self.last_router_z_loss: torch.Tensor | None = None
        self.last_execution_schedule: str | None = None
        self.last_packed_slot_count: int | None = None

        self.routed_lora_A = nn.ParameterDict()
        self.routed_lora_B = nn.ParameterDict()
        for role, weight in self.routed_weights.items():
            self.routed_lora_A[role] = nn.Parameter(torch.empty(
                self.expert_count,
                rank,
                int(weight.shape[-1]),
                dtype=weight.dtype,
            ))
            self.routed_lora_B[role] = nn.Parameter(torch.zeros(
                self.expert_count,
                int(weight.shape[-2]),
                rank,
                dtype=weight.dtype,
            ))
            nn.init.kaiming_uniform_(
                self.routed_lora_A[role], a=math.sqrt(5)
            )

        self.shared_lora_A = nn.ParameterDict()
        self.shared_lora_B = nn.ParameterDict()
        for role, weight in self.shared_weights.items():
            if weight.ndim != 3:
                raise ValueError(
                    "shared expert weights must be [experts, out, in]"
                )
            count = int(weight.shape[0])
            self.shared_lora_A[role] = nn.Parameter(torch.empty(
                count, rank, int(weight.shape[-1]), dtype=weight.dtype
            ))
            self.shared_lora_B[role] = nn.Parameter(torch.zeros(
                count, int(weight.shape[-2]), rank, dtype=weight.dtype
            ))
            nn.init.kaiming_uniform_(
                self.shared_lora_A[role], a=math.sqrt(5)
            )
        self.shared_gate = shared_gate

    @staticmethod
    def _select(
        value: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        expert_count = int(value.shape[0])
        expert_ids = torch.arange(
            expert_count, dtype=indices.dtype, device=indices.device
        )
        selector = (indices.unsqueeze(-1) == expert_ids).to(value.dtype)
        flat = torch.matmul(
            selector.reshape(-1, expert_count),
            value.reshape(expert_count, -1),
        )
        return flat.reshape(*indices.shape, *value.shape[1:])

    def _project_selected(
        self,
        role: str,
        inputs: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.routed_weights[role][indices]
        output = _expert_linear(inputs, weight)
        low = _expert_linear(
            inputs, self._select(self.routed_lora_A[role], indices)
        )
        return output + _expert_linear(
            low, self._select(self.routed_lora_B[role], indices)
        ) * self.scaling

    def _project_bucketed(
        self,
        role: str,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Project fixed-capacity expert buckets without copying weights.

        ``inputs`` is [experts, capacity, in].  Each expert matrix therefore
        participates once in a strided batched matmul instead of being gathered
        into a [token, top-k, out, in] tensor for every routed assignment.
        Empty capacity slots are ordinary tile padding, not dense all-expert
        token work.
        """

        weight = self.routed_weights[role]
        output = torch.bmm(inputs, weight.transpose(1, 2))
        low = torch.bmm(
            inputs, self.routed_lora_A[role].transpose(1, 2)
        )
        return output + torch.bmm(
            low, self.routed_lora_B[role].transpose(1, 2)
        ) * self.scaling

    def _routed_experts(
        self,
        inputs: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        if "gate_up" in self.routed_weights:
            gate, up = self._project_selected(
                "gate_up", inputs, indices
            ).chunk(2, dim=-1)
            hidden = F.silu(gate) * up
        elif "gate" in self.routed_weights:
            gate = self._project_selected("gate", inputs, indices)
            up = self._project_selected("up", inputs, indices)
            hidden = F.silu(gate) * up
        else:
            hidden = F.gelu(
                self._project_selected("up", inputs, indices)
            )
        return self._project_selected("down", hidden, indices)

    def _routed_experts_bucketed(
        self,
        inputs: torch.Tensor,
        indices: torch.Tensor,
        positions: torch.Tensor,
        capacity: int,
    ) -> torch.Tensor:
        """Pack selected assignments by expert and restore token-major order.

        Dispatch and restore stay as device tensor operations.  The only
        gathered object on the return path is the small expert activation;
        frozen and trainable expert matrices are never replicated per token.
        The schedule is valid only for an explicit drop/residual capacity
        contract, which gives the static per-expert bucket bound required by
        StableHLO.
        """

        assignment_inputs = inputs.reshape(-1, inputs.shape[-1])
        assignment_indices = indices.reshape(-1)
        assignments = int(assignment_indices.shape[0])
        positions = positions.reshape(-1)

        expert_ids = torch.arange(
            self.expert_count,
            dtype=assignment_indices.dtype,
            device=assignment_indices.device,
        )
        slot_ids = torch.arange(
            capacity, dtype=positions.dtype, device=positions.device
        )
        dispatch = (
            assignment_indices.reshape(1, 1, assignments)
            == expert_ids.reshape(self.expert_count, 1, 1)
        ) & (
            positions.reshape(1, 1, assignments)
            == slot_ids.reshape(1, capacity, 1)
        )
        expert_inputs = torch.matmul(
            dispatch.to(assignment_inputs.dtype), assignment_inputs
        )

        if "gate_up" in self.routed_weights:
            gate, up = self._project_bucketed(
                "gate_up", expert_inputs
            ).chunk(2, dim=-1)
            hidden = F.silu(gate) * up
        elif "gate" in self.routed_weights:
            gate = self._project_bucketed("gate", expert_inputs)
            up = self._project_bucketed("up", expert_inputs)
            hidden = F.silu(gate) * up
        else:
            hidden = F.gelu(self._project_bucketed("up", expert_inputs))
        expert_outputs = self._project_bucketed("down", hidden)

        # Advanced indexing here gathers one activation row per assignment;
        # unlike the old path it never gathers an expert weight matrix.
        valid = positions < capacity
        restored = expert_outputs[
            assignment_indices,
            positions.clamp(max=capacity - 1),
        ]
        restored = restored * valid.to(restored.dtype).unsqueeze(-1)
        return restored.reshape(*indices.shape, expert_outputs.shape[-1])

    def _group_limited_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.expert_group_count is None:
            return logits
        groups = self.expert_group_count
        if self.expert_count % groups:
            raise ValueError("expert count must be divisible by group count")
        group_top_k = self.expert_group_top_k or groups
        per_group = self.expert_count // groups
        group_scores = logits.reshape(
            logits.shape[0], groups, per_group
        ).amax(dim=-1)
        _values, selected_groups = deterministic_topk(
            group_scores, group_top_k
        )
        group_ids = torch.arange(
            groups,
            dtype=selected_groups.dtype,
            device=selected_groups.device,
        )
        group_mask = (
            selected_groups.unsqueeze(-1) == group_ids.reshape(1, 1, groups)
        ).any(dim=1)
        expert_mask = group_mask.unsqueeze(-1).expand(
            -1, -1, per_group
        ).reshape(logits.shape)
        return torch.where(
            expert_mask,
            logits,
            torch.full_like(logits, float("-inf")),
        )

    def _routing_weights(
        self,
        logits: torch.Tensor,
        selected_logits: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.router_normalization == "softmax_selected":
            return torch.softmax(selected_logits, dim=-1)
        if self.router_normalization == "sigmoid_selected":
            scores = torch.sigmoid(selected_logits)
            return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        if self.router_normalization == "sigmoid_selected_unnormalized":
            return torch.sigmoid(selected_logits)
        if self.router_normalization in {
            "softmax_all_selected",
            "softmax_all_renormalized_selected",
        }:
            all_weights = torch.softmax(logits, dim=-1)
            expert_ids = torch.arange(
                self.expert_count,
                dtype=selected_indices.dtype,
                device=selected_indices.device,
            )
            selector = (
                selected_indices.unsqueeze(-1)
                == expert_ids.reshape(1, 1, self.expert_count)
            )
            selected = torch.where(
                selector,
                all_weights.unsqueeze(1),
                torch.zeros(
                    (), dtype=all_weights.dtype, device=all_weights.device
                ),
            ).sum(dim=-1)
            if self.router_normalization == "softmax_all_renormalized_selected":
                selected = selected / selected.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-20)
            return selected
        return selected_logits

    def _expert_capacity(self, assignment_count: int) -> int:
        if self.capacity_factor is None:
            return assignment_count
        capacity = int(math.ceil(
            self.capacity_factor * assignment_count / self.expert_count
        ))
        if self.minimum_capacity is not None:
            capacity = max(capacity, self.minimum_capacity)
        return max(1, capacity)

    def _routing_positions(self, indices: torch.Tensor) -> torch.Tensor:
        """Return the stable zero-based slot for each expert assignment.

        A token-major one-hot prefix scan is O(assignments * experts), unlike
        the original pairwise O(assignments**2) equality matrix.  Lower expert
        IDs and token/rank traversal remain unchanged, so capacity decisions
        and BF16 reduction order are preserved.
        """

        flat = indices.reshape(-1)
        expert_ids = torch.arange(
            self.expert_count, dtype=flat.dtype, device=flat.device
        )
        one_hot = flat.unsqueeze(-1) == expert_ids.reshape(1, -1)
        prefixes = torch.cumsum(
            one_hot.to(torch.int32), dim=0, dtype=torch.int32
        ) - 1
        return torch.where(
            one_hot,
            prefixes,
            torch.zeros((), dtype=prefixes.dtype, device=prefixes.device),
        ).sum(dim=-1).reshape(indices.shape)

    def _capacity_mask(
        self,
        indices: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.capacity_factor is None:
            return torch.ones_like(indices, dtype=torch.bool)
        count = int(indices.numel())
        capacity = self._expert_capacity(count)
        positions = (
            self._routing_positions(indices)
            if positions is None else positions
        )
        return positions < capacity

    def _shared_experts(self, flat: torch.Tensor) -> torch.Tensor:
        if not self.shared_weights:
            return torch.zeros_like(flat)
        count = int(next(iter(self.shared_weights.values())).shape[0])
        inputs = flat.unsqueeze(1).expand(-1, count, -1)

        def project(role: str, value: torch.Tensor) -> torch.Tensor:
            expert_major = value.transpose(0, 1)
            output = torch.bmm(
                expert_major, self.shared_weights[role].transpose(1, 2)
            )
            low = torch.bmm(
                expert_major, self.shared_lora_A[role].transpose(1, 2)
            )
            result = output + torch.bmm(
                low, self.shared_lora_B[role].transpose(1, 2)
            ) * self.scaling
            return result.transpose(0, 1)

        if "gate_up" in self.shared_weights:
            gate, up = project("gate_up", inputs).chunk(2, dim=-1)
            hidden = F.silu(gate) * up
        elif "gate" in self.shared_weights:
            hidden = F.silu(project("gate", inputs)) * project("up", inputs)
        else:
            hidden = F.gelu(project("up", inputs))
        outputs = project("down", hidden)
        if self.shared_gate is None:
            weights = torch.ones(
                flat.shape[0], count, dtype=flat.dtype, device=flat.device
            )
        else:
            weights = torch.sigmoid(self.shared_gate(flat)).to(flat.dtype)
            if weights.shape[-1] == 1 and count > 1:
                weights = weights.expand(-1, count)
        return (outputs * weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, layer_input: torch.Tensor):
        if self.training and self.router_jitter_noise > 0.0:
            layer_input = layer_input * torch.empty_like(layer_input).uniform_(
                1.0 - self.router_jitter_noise,
                1.0 + self.router_jitter_noise,
            )
        original_shape = layer_input.shape
        flat = layer_input.reshape(-1, original_shape[-1])
        router_logits = self.router_layer(flat).float()
        selection_logits = self._group_limited_logits(router_logits)
        selected_logits, selected_indices = deterministic_topk(
            selection_logits, self.top_k
        )
        gates = self._routing_weights(
            router_logits, selected_logits, selected_indices
        ).to(flat.dtype)
        positions = (
            self._routing_positions(selected_indices)
            if self.capacity_factor is not None else None
        )
        capacity = self._capacity_mask(selected_indices, positions)
        if self.overflow_policy != "none":
            gates = gates * capacity.to(gates.dtype)
        self.last_route_indices = selected_indices.detach()
        self.last_router_logits = router_logits
        selected_inputs = flat.unsqueeze(1).expand(-1, self.top_k, -1)
        use_bucketed = (
            self.backend == "sparse"
            and self.capacity_factor is not None
            and self.overflow_policy in {"drop", "residual"}
        )
        if use_bucketed:
            expert_capacity = self._expert_capacity(
                int(selected_indices.numel())
            )
            routed = self._routed_experts_bucketed(
                selected_inputs,
                selected_indices,
                positions,
                expert_capacity,
            )
            self.last_execution_schedule = "expert_capacity_bucketed"
            self.last_packed_slot_count = (
                self.expert_count * expert_capacity
            )
        else:
            routed = self._routed_experts(
                selected_inputs, selected_indices
            )
            self.last_execution_schedule = "assignment_weight_gather"
            self.last_packed_slot_count = None
        combined = (routed * gates.unsqueeze(-1)).sum(dim=1)
        if self.overflow_policy == "residual":
            dropped = (~capacity).all(dim=1).to(flat.dtype).unsqueeze(-1)
            combined = combined + flat * dropped
        combined = combined + self._shared_experts(flat)

        probabilities = torch.softmax(router_logits, dim=-1)
        expert_ids = torch.arange(
            self.expert_count,
            dtype=selected_indices.dtype,
            device=selected_indices.device,
        )
        assignments = (
            selected_indices.unsqueeze(-1)
            == expert_ids.reshape(1, 1, self.expert_count)
        ).float().mean(dim=0)
        self.last_auxiliary_loss = (
            self.expert_count
            * (
                probabilities.mean(dim=0).unsqueeze(0)
                * assignments
            ).sum()
            * self.auxiliary_loss_coefficient
        )
        self.last_router_z_loss = (
            torch.logsumexp(router_logits, dim=-1).square().mean()
            * self.router_z_loss_coefficient
        )
        output = combined.reshape(*original_shape)
        if self.output_contract == "hidden":
            return output
        return output, router_logits


def sparse_moe_regularization_loss(
    model: nn.Module,
    *,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Return device-side auxiliary and z-loss for installed generic owners.

    Owners with the same routing contract are aggregated across layers, which
    matches the common HF load-balancing definition. The returned tensor keeps
    gradients through router probabilities; discrete assignments remain
    detached. Padding is excluded using the original training attention mask.
    """

    owners = [
        module for module in model.modules()
        if isinstance(module, GenericDeviceSparseMoELoRA)
        and module.last_router_logits is not None
        and module.last_route_indices is not None
    ]
    if not owners:
        return None
    total: torch.Tensor | None = None
    grouped: dict[tuple[int, int, float], list[GenericDeviceSparseMoELoRA]] = {}
    for owner in owners:
        key = (
            owner.expert_count,
            owner.top_k,
            owner.auxiliary_loss_coefficient,
        )
        grouped.setdefault(key, []).append(owner)
    for (expert_count, top_k, coefficient), members in grouped.items():
        first_logits = members[0].last_router_logits
        assert first_logits is not None
        if coefficient != 0.0:
            logits = torch.cat([
                member.last_router_logits
                for member in members
                if member.last_router_logits is not None
            ], dim=0)
            indices = torch.cat([
                member.last_route_indices
                for member in members
                if member.last_route_indices is not None
            ], dim=0)
            expert_ids = torch.arange(
                expert_count,
                dtype=indices.dtype,
                device=indices.device,
            )
            expert_mask = (
                indices.unsqueeze(-1)
                == expert_ids.reshape(1, 1, expert_count)
            ).float()
            probabilities = torch.softmax(logits, dim=-1)
            if attention_mask is None:
                tokens_per_expert = expert_mask.mean(dim=0)
                router_probability = probabilities.mean(dim=0)
            else:
                flat_mask = attention_mask.reshape(-1).to(
                    device=logits.device,
                    dtype=probabilities.dtype,
                )
                repeated = flat_mask.repeat(len(members))
                denominator = repeated.sum().clamp_min(1.0)
                tokens_per_expert = (
                    expert_mask * repeated.reshape(-1, 1, 1)
                ).sum(dim=0) / denominator
                router_probability = (
                    probabilities * repeated.unsqueeze(-1)
                ).sum(dim=0) / denominator
            value = (
                expert_count
                * (
                    tokens_per_expert
                    * router_probability.unsqueeze(0)
                ).sum()
                * coefficient
            )
            total = value if total is None else total + value
        z_members = [
            member for member in members
            if member.router_z_loss_coefficient != 0.0
        ]
        for member in z_members:
            assert member.last_router_logits is not None
            value = (
                torch.logsumexp(
                    member.last_router_logits, dim=-1
                ).square().mean()
                * member.router_z_loss_coefficient
            )
            total = value if total is None else total + value
    return total


def _module_weight(module: nn.Module) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"{type(module).__name__} has no expert weight tensor")
    return weight


def _canonical_expert_weight(
    model: nn.Module,
    projection: CanonicalExpertLinearRole,
) -> torch.Tensor:
    paths = projection.paths or (projection.path,)
    values = [_module_weight(model.get_submodule(path)) for path in paths]
    if len(values) == 1 and values[0].ndim == 3:
        result = values[0]
    elif all(value.ndim == 2 for value in values):
        result = torch.stack(values)
    else:
        raise ValueError(
            f"expert role {projection.role} has inconsistent physical tensors"
        )
    if projection.storage_layout.endswith("in_out"):
        result = result.transpose(-1, -2)
    expected = (
        projection.expert_count,
        projection.output_features,
        projection.input_features,
    )
    if tuple(result.shape) != expected:
        raise ValueError(
            f"expert role {projection.role} normalized to {tuple(result.shape)}, "
            f"expected {expected}"
        )
    return result


def _role_key(role: str) -> str:
    for prefix in ("shared_expert_", "expert_"):
        if role.startswith(prefix):
            return role[len(prefix):]
    raise ValueError(f"not an expert projection role: {role}")


def _probe_output_contract(
    original: nn.Module,
    *,
    hidden_size: int,
    router_layer: nn.Module,
) -> str:
    explicit = getattr(original, "_tt_output_contract", None)
    if explicit in {"hidden", "hidden_router_tuple"}:
        return explicit
    parameter = next(router_layer.parameters(), None)
    dtype = parameter.dtype if parameter is not None else torch.bfloat16
    device = parameter.device if parameter is not None else torch.device("cpu")
    was_training = original.training
    try:
        original.eval()
        with torch.no_grad():
            sample = torch.zeros(1, 1, hidden_size, dtype=dtype, device=device)
            result = original(sample)
    except Exception as error:
        raise ValueError(
            "could not infer the MoE block output contract structurally; set "
            "module._tt_output_contract to 'hidden' or 'hidden_router_tuple'"
        ) from error
    finally:
        original.train(was_training)
    if isinstance(result, torch.Tensor):
        return "hidden"
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], torch.Tensor)
        and isinstance(result[1], torch.Tensor)
    ):
        return "hidden_router_tuple"
    raise ValueError(
        "unsupported MoE block output structure; expected Tensor or "
        "(hidden, router_logits)"
    )


def _register_router_output_capture(model: nn.Module) -> None:
    """Teach Transformers' structural recorder about the generic owner.

    Recent Transformers releases collect router logits through an
    ``OutputRecorder`` keyed by the original expert-block class. Replacing the
    block without extending that semantic recorder silently drops auxiliary
    router loss. Mutating the existing recorder list preserves the upstream
    contract without checking a model family or repository name.
    """

    try:
        from transformers.utils.generic import OutputRecorder
    except ImportError:
        return
    for module in model.modules():
        flags = getattr(module, "_can_record_outputs", None)
        if not isinstance(flags, dict) or "router_logits" not in flags:
            continue
        current = flags["router_logits"]
        entries = list(current) if isinstance(current, list) else [current]
        if any(
            isinstance(item, OutputRecorder)
            and item.target_class is GenericDeviceSparseMoELoRA
            and item.index == 1
            for item in entries
        ):
            continue
        entries.append(OutputRecorder(GenericDeviceSparseMoELoRA, index=1))
        # The Transformers registry retains this class-level mapping by
        # reference, so an in-place update also updates already-created model
        # instances without reaching into a private global registry.
        flags["router_logits"] = entries


def install_sparse_moe(
    model: nn.Module,
    mapping: CanonicalModelMap,
    *,
    architecture: CanonicalArchitectureDescriptor | None = None,
    rank: int,
    alpha: int,
    backend: str,
) -> SparseMoEInstallReport | None:
    """Replace mapped MoE blocks without using checkpoint-family branches."""

    moe_layers = [layer for layer in mapping.layers if layer.moe is not None]
    if not moe_layers:
        return None
    for layer in moe_layers:
        assert layer.moe is not None
        moe_map = layer.moe
        original = model.get_submodule(moe_map.path)
        router = model.get_submodule(moe_map.router_path)
        moe_semantics = architecture.moe if architecture is not None else None
        legacy = bool(
            moe_map.expert_storage_layout == "fused"
            and tuple(item.role for item in moe_map.projections)
                == ("expert_gate_up", "expert_down")
            and not moe_map.shared_expert_projections
            and moe_map.expert_group_count is None
            and (
                moe_semantics is None
                or (
                    moe_semantics.router_normalization == "softmax_selected"
                    and moe_semantics.auxiliary_loss_coefficient == 0.0
                    and moe_semantics.router_z_loss_coefficient == 0.0
                    and moe_semantics.expert_capacity_factor is None
                )
            )
        )
        if legacy:
            replacement = DeviceSparseMoELoRA(
                original,
                router_layer=router,
                input_weight=_canonical_expert_weight(
                    model, moe_map.projection("expert_gate_up")
                ),
                output_weight=_canonical_expert_weight(
                    model, moe_map.projection("expert_down")
                ),
                rank=rank,
                alpha=alpha,
                top_k=moe_map.top_k,
                backend=backend,
            )
        else:
            routed_weights = {
                _role_key(item.role): _canonical_expert_weight(model, item)
                for item in moe_map.projections
            }
            shared_weights = {
                _role_key(item.role): _canonical_expert_weight(model, item)
                for item in moe_map.shared_expert_projections
            }
            shared_gate = (
                model.get_submodule(moe_map.shared_expert_gate_path)
                if moe_map.shared_expert_gate_path is not None else None
            )
            replacement = GenericDeviceSparseMoELoRA(
                original,
                router_layer=router,
                routed_weights=routed_weights,
                shared_weights=shared_weights,
                shared_gate=shared_gate,
                rank=rank,
                alpha=alpha,
                top_k=moe_map.top_k,
                backend=backend,
                router_normalization=(
                    moe_semantics.router_normalization
                    if moe_semantics is not None else "softmax_selected"
                ),
                expert_group_count=moe_map.expert_group_count,
                expert_group_top_k=moe_map.expert_group_top_k,
                auxiliary_loss_coefficient=(
                    moe_semantics.auxiliary_loss_coefficient
                    if moe_semantics is not None else 0.0
                ),
                router_z_loss_coefficient=(
                    moe_semantics.router_z_loss_coefficient
                    if moe_semantics is not None else 0.0
                ),
                capacity_factor=(
                    moe_semantics.expert_capacity_factor
                    if moe_semantics is not None else None
                ),
                minimum_capacity=(
                    moe_semantics.expert_min_capacity
                    if moe_semantics is not None else None
                ),
                overflow_policy=(
                    moe_semantics.overflow_policy
                    if moe_semantics is not None else "none"
                ),
                router_jitter_noise=(
                    moe_semantics.router_jitter_noise
                    if moe_semantics is not None else 0.0
                ),
                output_contract=_probe_output_contract(
                    original,
                    hidden_size=moe_map.router_input_features,
                    router_layer=router,
                ),
            )
        parent_path, child_name = moe_map.path.rsplit(".", 1)
        setattr(model.get_submodule(parent_path), child_name, replacement)
    if any(
        isinstance(module, GenericDeviceSparseMoELoRA)
        for module in model.modules()
    ):
        _register_router_output_capture(model)
    first = moe_layers[0].moe
    assert first is not None
    return SparseMoEInstallReport(
        backend=backend,
        layers=len(moe_layers),
        expert_count=first.expert_count,
        top_k=first.top_k,
        rank=rank,
        host_token_routing_loop=False,
        evaluates_all_experts=False,
        deterministic_contract=(
            "fp32_router; deterministic selected-only device routing; lower "
            "expert id wins ties; token-major selected-rank dispatch; "
            "canonical routed/shared expert math; no host token loop"
        ),
    )


__all__ = [
    "DeviceSparseMoELoRA",
    "GenericDeviceSparseMoELoRA",
    "SparseMoEInstallReport",
    "deterministic_topk",
    "install_sparse_moe",
    "sparse_moe_regularization_loss",
]
