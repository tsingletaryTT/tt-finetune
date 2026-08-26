"""Architecture-driven Blackhole planning for dense decoder-only causal LMs.

architecture deliberately leaves the frozen native planner unchanged.  This
planner consumes a canonical semantic descriptor, derives all size-dependent
geometry, and selects only implementations whose structural capability and
qualification constraints match.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from tt_finetune.architecture import (
    ArchitectureCapabilityReport,
    CanonicalArchitectureDescriptor,
    architecture_capabilities,
)


TILE_SIZE = 32
SEQUENCE_BLOCK = 128


@dataclass(frozen=True)
class HardwareCapabilities:
    architecture: str = "blackhole"
    worker_cores: int = 110
    l1_bytes_per_core: int = 1_500_000
    l1_usable_fraction: float = 0.75
    dram_bytes: int = 24 * 1024**3
    native_sdpa_backward: bool = True

    @property
    def usable_l1_bytes_per_core(self) -> int:
        return int(self.l1_bytes_per_core * self.l1_usable_fraction)

    def validate(self) -> None:
        if self.architecture != "blackhole":
            raise ValueError("architecture currently targets Blackhole")
        if self.worker_cores <= 0 or self.l1_bytes_per_core <= 0:
            raise ValueError("worker-core and L1 capacities must be positive")
        if not 0.0 < self.l1_usable_fraction < 1.0:
            raise ValueError("L1 usable fraction must be in (0, 1)")
        if self.dram_bytes <= 0:
            raise ValueError("DRAM capacity must be positive")


@dataclass(frozen=True)
class PrimitiveSelection:
    primitive: str
    implementation: str
    qualification: str
    fallback: str | None
    reason: str


@dataclass(frozen=True)
class LoRATargetGeometry:
    role: str
    input_features: int
    output_features: int
    instances: int

    def parameter_count(self, rank: int) -> int:
        return self.instances * rank * (self.input_features + self.output_features)


@dataclass(frozen=True)
class ExecutionMemoryPlan:
    """Planner-visible residency and arena policy for one execution plan."""

    tile_alignment: int
    l1_bytes_per_core: int
    usable_l1_bytes_per_core: int
    aggregate_usable_l1_bytes: int
    frozen_parameter_placement: str
    trainable_state_placement: str
    activation_placement: str
    attention_scratch_placement: str
    optimizer_state_placement: str
    arena_policy: str
    double_buffered: bool

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


@dataclass(frozen=True)
class ExecutionPlan:
    architecture: CanonicalArchitectureDescriptor
    capabilities: ArchitectureCapabilityReport
    rank: int
    rank_tiles: int
    physical_rank: int
    rank_tail: int
    batch_size: int
    sequence_length: int
    sequence_blocks: int
    sequence_tail: int
    logical_tokens: int
    physical_tokens: int
    lora_targets: tuple[LoRATargetGeometry, ...]
    lora_parameter_count: int
    estimated_base_parameter_count: int
    estimated_device_bytes: int
    moe_expert_weight_bytes: int
    moe_routed_scratch_bytes: int
    moe_router_weight_bytes: int
    moe_shared_expert_weight_bytes: int
    moe_selected_activation_bytes: int
    moe_routing_workspace_bytes: int
    moe_expert_lora_state_bytes: int
    moe_execution_schedule: str | None
    moe_capacity_per_expert: int | None
    moe_packed_assignment_slots: int | None
    moe_weight_replication_avoided_bytes: int
    moe_sparse_compute_ratio: float | None
    memory_admitted: bool
    memory_plan: ExecutionMemoryPlan
    attention_materializes_full_sxs_dram: bool
    checkpoint_strategy: str
    loss_strategy: str
    selections: tuple[PrimitiveSelection, ...]
    selected_stack_backend: str
    safe_fallback: bool
    admission_reason: str

    def selection(self, primitive: str) -> PrimitiveSelection:
        matches = [item for item in self.selections if item.primitive == primitive]
        if len(matches) != 1:
            raise KeyError(f"plan has no unique selection for {primitive}")
        return matches[0]

    def require_admitted(self) -> None:
        self.capabilities.require_supported()
        if not self.memory_admitted:
            raise ValueError(self.admission_reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "rank": self.rank,
            "rank_tiles": self.rank_tiles,
            "physical_rank": self.physical_rank,
            "rank_tail": self.rank_tail,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "sequence_blocks": self.sequence_blocks,
            "sequence_tail": self.sequence_tail,
            "logical_tokens": self.logical_tokens,
            "physical_tokens": self.physical_tokens,
            "lora_targets": [vars(item) for item in self.lora_targets],
            "lora_parameter_count": self.lora_parameter_count,
            "estimated_base_parameter_count": self.estimated_base_parameter_count,
            "estimated_device_bytes": self.estimated_device_bytes,
            "moe_expert_weight_bytes": self.moe_expert_weight_bytes,
            "moe_routed_scratch_bytes": self.moe_routed_scratch_bytes,
            "moe_router_weight_bytes": self.moe_router_weight_bytes,
            "moe_shared_expert_weight_bytes": (
                self.moe_shared_expert_weight_bytes
            ),
            "moe_selected_activation_bytes": (
                self.moe_selected_activation_bytes
            ),
            "moe_routing_workspace_bytes": self.moe_routing_workspace_bytes,
            "moe_expert_lora_state_bytes": self.moe_expert_lora_state_bytes,
            "moe_execution_schedule": self.moe_execution_schedule,
            "moe_capacity_per_expert": self.moe_capacity_per_expert,
            "moe_packed_assignment_slots": self.moe_packed_assignment_slots,
            "moe_weight_replication_avoided_bytes": (
                self.moe_weight_replication_avoided_bytes
            ),
            "moe_sparse_compute_ratio": self.moe_sparse_compute_ratio,
            "memory_admitted": self.memory_admitted,
            "memory_plan": self.memory_plan.to_dict(),
            "attention_materializes_full_sxs_dram": (
                self.attention_materializes_full_sxs_dram
            ),
            "checkpoint_strategy": self.checkpoint_strategy,
            "loss_strategy": self.loss_strategy,
            "selections": [vars(item) for item in self.selections],
            "selected_stack_backend": self.selected_stack_backend,
            "safe_fallback": self.safe_fallback,
            "admission_reason": self.admission_reason,
        }


@dataclass(frozen=True)
class ResidentQualification:
    """One frozen full-stack contract, keyed only by semantics and geometry.

    Bundle paths are execution artifacts, not model-family routing.  A renamed
    or independently published checkpoint with the same canonical fingerprint
    and R/B/S geometry resolves to this same qualification.
    """

    semantic_fingerprint: str
    rank: int
    batch_size: int
    sequence_length: int
    contract: str
    source_bundle: str
    descriptor_bundle: str


# These are structural fingerprints of the frozen resident runtime contracts. Model
# names are intentionally absent. A renamed checkpoint with identical semantic
# architecture receives the same decision.
_RESIDENT_QUALIFICATIONS = (
    ResidentQualification(
        "738434b016f9144dce8cc44c", 8, 1, 128,
        "smol_geometry_20_step_exact",
        "generated/smollm2_resident_layer_backward_sequence_v8",
        "generated/adamw_descriptors_smollm2_30layer_exact_v1",
    ),
    ResidentQualification(
        "491a95b4f0dc7ec66c77a07a", 16, 1, 128,
        "qwen_geometry_20_step_exact",
        "generated/qwen_resident_layer_backward_sequence_v12",
        "generated/adamw_descriptors_qwen_24layer_exact_v1",
    ),
)


# Native long-attention is admitted by tensor geometry. These two head layouts
# are the native silicon-qualified contracts; new layouts remain safe until a
# matching evolving qualification is added.
_NATIVE_ATTENTION_GEOMETRIES = frozenset({
    (64, 9, 3),
    (64, 14, 2),
})


def _lora_geometries(
    descriptor: CanonicalArchitectureDescriptor,
) -> tuple[LoRATargetGeometry, ...]:
    h = descriptor.hidden_size
    q = descriptor.attention_heads * descriptor.head_dimension
    kv = descriptor.kv_heads * descriptor.head_dimension
    i = descriptor.intermediate_size
    layers = descriptor.layer_count
    if descriptor.qkv_projection_layout == "separate":
        attention = (
            LoRATargetGeometry("q", h, q, layers),
            LoRATargetGeometry("k", h, kv, layers),
            LoRATargetGeometry("v", h, kv, layers),
            LoRATargetGeometry("o", q, h, layers),
        )
    elif descriptor.qkv_projection_layout == "q_plus_fused_kv":
        attention = (
            LoRATargetGeometry("q", h, q, layers),
            LoRATargetGeometry("kv", h, 2 * kv, layers),
            LoRATargetGeometry("o", q, h, layers),
        )
    elif descriptor.qkv_projection_layout == "fused_qk_plus_v":
        attention = (
            LoRATargetGeometry("qk", h, q + kv, layers),
            LoRATargetGeometry("v", h, kv, layers),
            LoRATargetGeometry("o", q, h, layers),
        )
    else:
        attention = (
            LoRATargetGeometry("qkv", h, q + 2 * kv, layers),
            LoRATargetGeometry("o", q, h, layers),
        )
    dense_layers = (
        len(descriptor.moe.dense_layer_indices)
        if descriptor.moe is not None else layers
    )
    if descriptor.mlp_kind == "ordinary":
        mlp = (
            LoRATargetGeometry("mlp_up", h, i, dense_layers),
            LoRATargetGeometry("mlp_down", i, h, dense_layers),
        )
    elif descriptor.gate_up_layout == "fused":
        mlp = (
            LoRATargetGeometry("gate_up", h, 2 * i, dense_layers),
            LoRATargetGeometry("down", i, h, dense_layers),
        )
    else:
        mlp = (
            LoRATargetGeometry("gate", h, i, dense_layers),
            LoRATargetGeometry("up", h, i, dense_layers),
            LoRATargetGeometry("down", i, h, dense_layers),
        )
    mlp = tuple(item for item in mlp if item.instances > 0)
    if descriptor.moe is None:
        return attention + mlp
    moe = descriptor.moe
    expert_instances = len(moe.moe_layer_indices) * moe.expert_count
    expert_roles = set(moe.expert_lora_roles)
    if "expert_gate_up" in expert_roles:
        expert = (
            LoRATargetGeometry(
                "expert_gate_up", h, 2 * moe.expert_intermediate_size,
                expert_instances,
            ),
            LoRATargetGeometry(
                "expert_down", moe.expert_intermediate_size, h,
                expert_instances,
            ),
        )
    elif "expert_gate" in expert_roles:
        expert = (
            LoRATargetGeometry(
                "expert_gate", h, moe.expert_intermediate_size,
                expert_instances,
            ),
            LoRATargetGeometry(
                "expert_up", h, moe.expert_intermediate_size,
                expert_instances,
            ),
            LoRATargetGeometry(
                "expert_down", moe.expert_intermediate_size, h,
                expert_instances,
            ),
        )
    else:
        expert = (
            LoRATargetGeometry(
                "expert_up", h, moe.expert_intermediate_size, expert_instances
            ),
            LoRATargetGeometry(
                "expert_down", moe.expert_intermediate_size, h, expert_instances
            ),
        )
    shared_instances = (
        len(moe.moe_layer_indices) * moe.shared_expert_count
    )
    if not shared_instances:
        shared: tuple[LoRATargetGeometry, ...] = ()
    elif moe.shared_expert_mlp_kind == "ordinary":
        shared_i = moe.shared_expert_intermediate_size
        assert shared_i is not None
        shared = (
            LoRATargetGeometry(
                "shared_expert_up", h, shared_i, shared_instances
            ),
            LoRATargetGeometry(
                "shared_expert_down", shared_i, h, shared_instances
            ),
        )
    elif moe.shared_expert_storage_layout == "fused":
        shared_i = moe.shared_expert_intermediate_size
        assert shared_i is not None
        shared = (
            LoRATargetGeometry(
                "shared_expert_gate_up", h, 2 * shared_i,
                shared_instances,
            ),
            LoRATargetGeometry(
                "shared_expert_down", shared_i, h, shared_instances
            ),
        )
    else:
        shared_i = moe.shared_expert_intermediate_size
        assert shared_i is not None
        shared = (
            LoRATargetGeometry(
                "shared_expert_gate", h, shared_i, shared_instances
            ),
            LoRATargetGeometry(
                "shared_expert_up", h, shared_i, shared_instances
            ),
            LoRATargetGeometry(
                "shared_expert_down", shared_i, h, shared_instances
            ),
        )
    return attention + mlp + expert + shared


def _base_parameter_estimate(descriptor: CanonicalArchitectureDescriptor) -> int:
    h = descriptor.hidden_size
    q = descriptor.attention_heads * descriptor.head_dimension
    kv = descriptor.kv_heads * descriptor.head_dimension
    i = descriptor.intermediate_size
    if descriptor.qkv_projection_layout == "separate":
        attention_weights = h * (q + 2 * kv) + q * h
    else:
        attention_weights = h * (q + 2 * kv) + q * h
    dense_mlp_weights = (
        2 * h * i
        if descriptor.mlp_kind == "ordinary"
        else 3 * h * i
    )
    if descriptor.moe is None:
        dense_layers = descriptor.layer_count
        moe_layers = 0
        expert_weights = 0
        shared_weights = 0
        router_weights = 0
        shared_gate_weights = 0
    else:
        dense_layers = len(descriptor.moe.dense_layer_indices)
        moe_layers = len(descriptor.moe.moe_layer_indices)
        expert_i = descriptor.moe.expert_intermediate_size
        per_expert = (
            2 * h * expert_i
            if descriptor.moe.expert_mlp_kind == "ordinary"
            else 3 * h * expert_i
        )
        expert_weights = descriptor.moe.expert_count * per_expert
        shared_i = descriptor.moe.shared_expert_intermediate_size or 0
        shared_per_expert = (
            2 * h * shared_i
            if descriptor.moe.shared_expert_mlp_kind == "ordinary"
            else 3 * h * shared_i
        )
        shared_weights = (
            descriptor.moe.shared_expert_count * shared_per_expert
        )
        router_weights = h * descriptor.moe.expert_count
        shared_gate_weights = (
            h if descriptor.moe.shared_expert_gate else 0
        )
    norms = 2 * h
    if descriptor.attention_qk_norm is not None:
        norms += 2 * descriptor.head_dimension
    biases = (
        (q if descriptor.biases.query else 0)
        + (kv if descriptor.biases.key else 0)
        + (kv if descriptor.biases.value else 0)
        + (h if descriptor.biases.output else 0)
        + (i if descriptor.biases.mlp_input else 0)
        + (h if descriptor.biases.mlp_output else 0)
    )
    embedding_dimension = descriptor.embedding_dimension
    embedding = descriptor.vocabulary_size * embedding_dimension
    head = 0 if descriptor.tied_embeddings else embedding
    embedding_projections = (
        2 * h * embedding_dimension
        if embedding_dimension != h else 0
    )
    return (
        descriptor.layer_count * (attention_weights + norms + biases)
        + dense_layers * dense_mlp_weights
        + moe_layers * (
            expert_weights + shared_weights + router_weights
            + shared_gate_weights
        )
        + embedding + head + embedding_projections
        + (h if descriptor.final_norm else 0)
    )


def _resident_qualification(
    descriptor: CanonicalArchitectureDescriptor,
    rank: int,
    batch_size: int,
    sequence_length: int,
) -> ResidentQualification | None:
    return next(
        (
            item for item in _RESIDENT_QUALIFICATIONS
            if item.semantic_fingerprint == descriptor.semantic_fingerprint
            and item.rank == rank
            and item.batch_size == batch_size
            and item.sequence_length == sequence_length
        ),
        None,
    )


def resident_qualification_for_plan(
    plan: ExecutionPlan,
) -> ResidentQualification | None:
    """Resolve the frozen resident bundle for an already-derived plan."""

    return _resident_qualification(
        plan.architecture,
        plan.rank,
        plan.batch_size,
        plan.sequence_length,
    )


def plan_execution(
    descriptor: CanonicalArchitectureDescriptor,
    *,
    rank: int,
    batch_size: int,
    sequence_length: int,
    lora_targets: tuple[LoRATargetGeometry, ...] | None = None,
    hardware: HardwareCapabilities | None = None,
) -> ExecutionPlan:
    """Plan a model from semantic capabilities and geometry only."""

    descriptor.validate()
    hardware = hardware or HardwareCapabilities()
    hardware.validate()
    if rank <= 0 or batch_size <= 0 or sequence_length <= 0:
        raise ValueError("rank, batch size and sequence length must be positive")
    capabilities = architecture_capabilities(
        descriptor, sequence_length=sequence_length
    )

    rank_tiles = math.ceil(rank / TILE_SIZE)
    physical_rank = rank_tiles * TILE_SIZE
    rank_tail = rank - (rank_tiles - 1) * TILE_SIZE
    sequence_blocks = math.ceil(sequence_length / SEQUENCE_BLOCK)
    sequence_tail = sequence_length - (sequence_blocks - 1) * SEQUENCE_BLOCK
    logical_tokens = batch_size * sequence_length
    physical_tokens = math.ceil(logical_tokens / TILE_SIZE) * TILE_SIZE
    targets = lora_targets or _lora_geometries(descriptor)
    if not targets:
        raise ValueError("at least one canonical LoRA target is required")
    if any(
        item.input_features <= 0
        or item.output_features <= 0
        or item.instances <= 0
        for item in targets
    ):
        raise ValueError("LoRA target geometry must be positive")
    lora_parameters = sum(item.parameter_count(rank) for item in targets)
    base_parameters = _base_parameter_estimate(descriptor)
    dtype_bytes = 2 if descriptor.dtype in {"bfloat16", "float16"} else 4
    # Frozen base weights + LoRA parameter/gradient + FP32 Adam moments + a
    # conservative six hidden-state checkpoint epochs.
    estimated_device_bytes = (
        base_parameters * dtype_bytes
        + lora_parameters * (2 + 2 + 4 + 4)
        + logical_tokens * descriptor.hidden_size * dtype_bytes * 6
    )
    if descriptor.moe is None:
        moe_expert_weight_bytes = 0
        moe_routed_scratch_bytes = 0
        moe_router_weight_bytes = 0
        moe_shared_expert_weight_bytes = 0
        moe_selected_activation_bytes = 0
        moe_routing_workspace_bytes = 0
        moe_expert_lora_state_bytes = 0
        moe_execution_schedule = None
        moe_capacity_per_expert = None
        moe_packed_assignment_slots = None
        moe_weight_replication_avoided_bytes = 0
        moe_sparse_compute_ratio = None
    else:
        moe = descriptor.moe
        moe_capacity_per_expert = None
        moe_packed_assignment_slots = None
        moe_weight_replication_avoided_bytes = 0
        expert_i = moe.expert_intermediate_size
        per_expert_factor = 2 if moe.expert_mlp_kind == "ordinary" else 3
        expert_parameters = (
            moe.expert_count * per_expert_factor
            * descriptor.hidden_size * expert_i
        )
        moe_layers = len(moe.moe_layer_indices)
        moe_expert_weight_bytes = (
            moe_layers * expert_parameters * dtype_bytes
        )
        shared_i = moe.shared_expert_intermediate_size or 0
        shared_factor = (
            2 if moe.shared_expert_mlp_kind == "ordinary" else 3
        )
        moe_shared_expert_weight_bytes = (
            moe_layers * moe.shared_expert_count * shared_factor
            * descriptor.hidden_size * shared_i * dtype_bytes
        )
        moe_router_weight_bytes = (
            moe_layers * descriptor.hidden_size
            * (moe.expert_count + (1 if moe.shared_expert_gate else 0))
            * dtype_bytes
        )
        assignments = logical_tokens * moe.top_k
        per_assignment_weights = (
            per_expert_factor * descriptor.hidden_size * expert_i
        )
        selected_intermediate_width = (
            expert_i if moe.expert_mlp_kind == "ordinary" else 2 * expert_i
        )
        capacity_bucketed = (
            moe.expert_capacity_factor is not None
            and moe.overflow_policy in {"drop", "residual"}
        )
        dynamic_no_capacity = (
            not capacity_bucketed
            and moe.expert_capacity_factor is None
            and descriptor.dtype == "bfloat16"
            and logical_tokens % TILE_SIZE == 0
            and descriptor.hidden_size % TILE_SIZE == 0
            and expert_i % TILE_SIZE == 0
            and moe.expert_count <= 32
            and moe.top_k <= 8
        )
        if capacity_bucketed:
            capacity = int(math.ceil(
                moe.expert_capacity_factor
                * assignments / moe.expert_count
            ))
            if moe.expert_min_capacity is not None:
                capacity = max(capacity, moe.expert_min_capacity)
            capacity = max(1, capacity)
            packed_slots = moe.expert_count * capacity
            # The current graph candidate represents dispatch as a compact
            # boolean [expert, slot, assignment] tensor. Expert matrices stay
            # [expert, out, in] and are consumed once by strided batched GEMM.
            moe_routed_scratch_bytes = (
                packed_slots * assignments
                + assignments * descriptor.hidden_size * dtype_bytes
            )
            moe_selected_activation_bytes = (
                packed_slots
                * (2 * descriptor.hidden_size + selected_intermediate_width)
                * dtype_bytes
            )
            # One-hot prefix positions use bool + int32 [assignment, expert].
            moe_routing_workspace_bytes = assignments * moe.expert_count * 5
            moe_execution_schedule = "expert_capacity_bucketed"
            moe_capacity_per_expert = capacity
            moe_packed_assignment_slots = packed_slots
            moe_weight_replication_avoided_bytes = (
                assignments * per_assignment_weights * dtype_bytes
            )
        elif dynamic_no_capacity:
            # Each expert owns one deterministic slot per token. This is an
            # exact no-drop bound because top-k cannot select the same expert
            # twice for a token. Only occupied 32-row blocks enter sparse
            # matmul; the physical slot tensor is bounded and no expert weight
            # is gathered per routed assignment.
            packed_slots = logical_tokens * moe.expert_count
            dispatch_elements = assignments * packed_slots
            moe_routed_scratch_bytes = (
                dispatch_elements * dtype_bytes
                + assignments * descriptor.hidden_size * dtype_bytes
            )
            moe_selected_activation_bytes = (
                packed_slots
                * (2 * descriptor.hidden_size + selected_intermediate_width)
                * dtype_bytes
            )
            # Boolean assignment map plus compact inverse/position metadata.
            moe_routing_workspace_bytes = (
                dispatch_elements
                + packed_slots * 4
                + assignments * (8 + 8)
            )
            moe_execution_schedule = "dynamic_native_expert_bucketed"
            moe_packed_assignment_slots = packed_slots
            moe_weight_replication_avoided_bytes = (
                assignments * per_assignment_weights * dtype_bytes
            )
        else:
            # Unsupported dynamic geometries retain the exact selected-
            # assignment fallback. Never truncate them to make a plan fit.
            moe_routed_scratch_bytes = (
                assignments * per_assignment_weights * dtype_bytes
            )
            moe_selected_activation_bytes = (
                assignments
                * (2 * descriptor.hidden_size + selected_intermediate_width)
                * dtype_bytes
            )
            moe_routing_workspace_bytes = 0
            moe_execution_schedule = "assignment_weight_gather"
        moe_routing_workspace_bytes += logical_tokens * (
            moe.expert_count * 4
            + moe.top_k * (4 + 8 + 1)
            + (moe.expert_group_count or 0) * 4
        )
        moe_expert_lora_parameters = sum(
            item.parameter_count(rank)
            for item in targets
            if item.role.startswith(("expert_", "shared_expert_"))
        )
        moe_expert_lora_state_bytes = moe_expert_lora_parameters * 12
        moe_sparse_compute_ratio = moe.top_k / moe.expert_count
        estimated_device_bytes += (
            moe_routed_scratch_bytes
            + moe_selected_activation_bytes
            + moe_routing_workspace_bytes
        )
    memory_admitted = estimated_device_bytes <= hardware.dram_bytes

    resident = _resident_qualification(
        descriptor, rank, batch_size, sequence_length
    )
    if resident is not None:
        memory_plan = ExecutionMemoryPlan(
            tile_alignment=TILE_SIZE,
            l1_bytes_per_core=hardware.l1_bytes_per_core,
            usable_l1_bytes_per_core=hardware.usable_l1_bytes_per_core,
            aggregate_usable_l1_bytes=(
                hardware.worker_cores * hardware.usable_l1_bytes_per_core
            ),
            frozen_parameter_placement="packed_dram_streamed_to_owned_cores",
            trainable_state_placement="packed_dram_with_resident_l1_working_sets",
            activation_placement="two_parity_checkpoint_arenas_with_l1_handoffs",
            attention_scratch_placement="per_group_l1_noc_resident",
            optimizer_state_placement="packed_dram_streamed_by_descriptor",
            arena_policy="frozen_resident_stack_abi",
            double_buffered=True,
        )
    elif sequence_length > SEQUENCE_BLOCK:
        memory_plan = ExecutionMemoryPlan(
            tile_alignment=TILE_SIZE,
            l1_bytes_per_core=hardware.l1_bytes_per_core,
            usable_l1_bytes_per_core=hardware.usable_l1_bytes_per_core,
            aggregate_usable_l1_bytes=(
                hardware.worker_cores * hardware.usable_l1_bytes_per_core
            ),
            frozen_parameter_placement="dram_layer_streamed",
            trainable_state_placement="packed_dram_with_l1_rank_blocks",
            activation_placement="layerwise_recompute_and_bounded_checkpoints",
            attention_scratch_placement="query_block_l1_no_full_sxs_dram",
            optimizer_state_placement="packed_dram_prefetched_per_pack",
            arena_policy="sequence_blocked_streamed",
            double_buffered=True,
        )
    else:
        memory_plan = ExecutionMemoryPlan(
            tile_alignment=TILE_SIZE,
            l1_bytes_per_core=hardware.l1_bytes_per_core,
            usable_l1_bytes_per_core=hardware.usable_l1_bytes_per_core,
            aggregate_usable_l1_bytes=(
                hardware.worker_cores * hardware.usable_l1_bytes_per_core
            ),
            frozen_parameter_placement="framework_managed_dram",
            trainable_state_placement="tile_packed_dram_bounded_packs",
            activation_placement="framework_saved_activations",
            attention_scratch_placement=(
                "bounded_dense_tiled_scratch_and_routed_expert_dram"
                if descriptor.moe is not None else "bounded_dense_tiled_scratch"
            ),
            optimizer_state_placement="tile_packed_dram_bounded_packs",
            arena_policy="safe_framework_tiled",
            double_buffered=False,
        )
    selections: list[PrimitiveSelection] = []
    safe_fallback = True
    if resident is not None:
        stack_backend = "resident_exact"
        selections.append(PrimitiveSelection(
            "transformer_stack", stack_backend, resident.contract,
            "safe_torch_xla_autograd",
            "semantic fingerprint and R/B/S exactly match a frozen resident contract",
        ))
    else:
        stack_backend = "safe_torch_xla_autograd"
        selections.append(PrimitiveSelection(
            "transformer_stack", stack_backend, "architecture_safe_contract",
            None,
            "no faster full-stack implementation is qualified for this geometry",
        ))

    complete_blocks = sequence_tail == SEQUENCE_BLOCK
    native_head_geometry = (
        descriptor.head_dimension,
        descriptor.attention_heads,
        descriptor.kv_heads,
    ) in _NATIVE_ATTENTION_GEOMETRIES
    local_attention_active = (
        descriptor.attention_window is not None
        and sequence_length > descriptor.attention_window
    )
    if sequence_length <= SEQUENCE_BLOCK:
        attention_impl = (
            "resident_dense_causal" if resident is not None else "safe_dense_causal"
        )
        full_sxs = False
        attention_qualification = (
            resident.contract if resident is not None else "dense_safe_reference"
        )
        attention_reason = "sequence fits one dense qualification block"
    elif (
        hardware.native_sdpa_backward
        and descriptor.dtype == "bfloat16"
        and complete_blocks
        and native_head_geometry
        and not local_attention_active
    ):
        attention_impl = "native_streamed_causal"
        full_sxs = False
        attention_qualification = "native_streamed_head_geometry"
        attention_reason = "head geometry and block schedule are silicon-qualified"
    elif descriptor.qkv_projection_layout == "separate" and not local_attention_active:
        attention_impl = "safe_streamed_causal"
        full_sxs = False
        attention_qualification = "canonical_safe_streamed_reference"
        attention_reason = (
            "streamed semantic is supported but native head geometry or tail is unqualified"
        )
    else:
        attention_impl = "safe_framework_attention"
        full_sxs = sequence_length > SEQUENCE_BLOCK
        attention_qualification = "huggingface_safe_reference"
        attention_reason = (
            "fused/local attention uses its architecture-defined safe implementation"
        )
    selections.append(PrimitiveSelection(
        "attention", attention_impl, attention_qualification,
        "safe_streamed_causal" if attention_impl == "native_streamed_causal" else None,
        attention_reason,
    ))
    selections.extend((
        PrimitiveSelection(
            "normalization",
            "resident_exact" if resident is not None else (
                "safe_explicit_layer_norm"
                if descriptor.norm_kind == "layer_norm"
                else "safe_torch_xla"
            ),
            resident.contract if resident is not None else (
                "explicit_fp32_moments_analytic_backward"
                if descriptor.norm_kind == "layer_norm"
                else "architecture_safe_contract"
            ),
            "safe_explicit_layer_norm" if resident is not None else None,
            (
                "generic Blackhole LayerNorm dX is numerically unstable; "
                "use explicit FP32 moments and analytic backward"
                if descriptor.norm_kind == "layer_norm" and resident is None
                else "selected by canonical norm semantic and qualified geometry"
            ),
        ),
        PrimitiveSelection(
            "mlp",
            "resident_exact" if resident is not None else "safe_torch_xla",
            resident.contract if resident is not None else "architecture_safe_contract",
            "safe_torch_xla" if resident is not None else None,
            "selected by canonical MLP semantic and qualified geometry",
        ),
        PrimitiveSelection(
            "lora",
            "device_packed_resident" if resident is not None else "peft_device_exact",
            resident.contract if resident is not None else "canonical_role_map",
            "peft_device_exact" if resident is not None else None,
            "canonical target roles and parametric rank determine adapter geometry",
        ),
        PrimitiveSelection(
            "optimizer", "tile_packed_adamw", "parametric_exact",
            "torch_xla_adamw",
            "optimizer packing is independent of model-family naming",
        ),
    ))
    if descriptor.moe is not None:
        moe = descriptor.moe
        capacity_bucketed = (
            moe_execution_schedule == "expert_capacity_bucketed"
        )
        dynamic_no_capacity = (
            moe_execution_schedule == "dynamic_native_expert_bucketed"
        )
        sparse_geometry_qualified = (
            descriptor.dtype == "bfloat16"
            and descriptor.hidden_size % TILE_SIZE == 0
            and moe.expert_intermediate_size % TILE_SIZE == 0
            and moe.expert_count <= 32
            and moe.top_k <= 8
        )
        selections.append(PrimitiveSelection(
            "moe",
            (
                "device_sparse_selected_experts"
                if sparse_geometry_qualified else "safe_device_selected_experts"
            ),
            (
                "canonical_sparse_moe_dynamic_no_drop_v1"
                if sparse_geometry_qualified and dynamic_no_capacity else
                "canonical_sparse_moe_capacity_bucketed_v1"
                if sparse_geometry_qualified and capacity_bucketed else
                "canonical_sparse_moe_v1"
                if sparse_geometry_qualified else "architecture_safe_contract"
            ),
            (
                "safe_device_selected_experts"
                if sparse_geometry_qualified else None
            ),
            (
                "router and exact per-expert positions stay on device; one "
                "slot per token bounds every expert without capacity loss; "
                "only occupied 32-row expert blocks execute"
                if sparse_geometry_qualified and dynamic_no_capacity else
                "router, deterministic top-k, capacity packing, expert-batched "
                "matmuls and weighted combine stay on device; expert matrices "
                "are not replicated per assignment"
                if sparse_geometry_qualified and capacity_bucketed else
                "router, deterministic top-k, dispatch and weighted combine stay "
                "on device; only top-k expert assignments are evaluated"
                if sparse_geometry_qualified else
                "sparse semantics are supported but dtype/tile/expert geometry is "
                "outside the silicon-qualified generic schedule"
            ),
        ))
    dense_logits_bytes = logical_tokens * descriptor.vocabulary_size * 4
    loss_strategy = (
        "streamed_vocab_fp32_causal_loss"
        if dense_logits_bytes > 1024**3 else "dense_fp32_causal_loss"
    )
    checkpoint_strategy = (
        "resident_layer_checkpoints"
        if resident is not None
        else "layerwise_recompute_streamed"
        if sequence_length > SEQUENCE_BLOCK
        else "framework_saved_activations"
    )
    admitted = capabilities.supported and memory_admitted
    if not capabilities.supported:
        reason = "architecture contains unsupported semantic primitives"
    elif not memory_admitted:
        reason = (
            f"estimated device footprint {estimated_device_bytes} exceeds "
            f"Blackhole DRAM budget {hardware.dram_bytes}"
        )
    else:
        reason = "architecture capabilities and estimated memory are admitted"
    return ExecutionPlan(
        architecture=descriptor,
        capabilities=capabilities,
        rank=rank,
        rank_tiles=rank_tiles,
        physical_rank=physical_rank,
        rank_tail=rank_tail,
        batch_size=batch_size,
        sequence_length=sequence_length,
        sequence_blocks=sequence_blocks,
        sequence_tail=sequence_tail,
        logical_tokens=logical_tokens,
        physical_tokens=physical_tokens,
        lora_targets=targets,
        lora_parameter_count=lora_parameters,
        estimated_base_parameter_count=base_parameters,
        estimated_device_bytes=estimated_device_bytes,
        moe_expert_weight_bytes=moe_expert_weight_bytes,
        moe_routed_scratch_bytes=moe_routed_scratch_bytes,
        moe_router_weight_bytes=moe_router_weight_bytes,
        moe_shared_expert_weight_bytes=moe_shared_expert_weight_bytes,
        moe_selected_activation_bytes=moe_selected_activation_bytes,
        moe_routing_workspace_bytes=moe_routing_workspace_bytes,
        moe_expert_lora_state_bytes=moe_expert_lora_state_bytes,
        moe_execution_schedule=moe_execution_schedule,
        moe_capacity_per_expert=moe_capacity_per_expert,
        moe_packed_assignment_slots=moe_packed_assignment_slots,
        moe_weight_replication_avoided_bytes=(
            moe_weight_replication_avoided_bytes
        ),
        moe_sparse_compute_ratio=moe_sparse_compute_ratio,
        memory_admitted=memory_admitted,
        memory_plan=memory_plan,
        attention_materializes_full_sxs_dram=full_sxs,
        checkpoint_strategy=checkpoint_strategy,
        loss_strategy=loss_strategy,
        selections=tuple(selections),
        selected_stack_backend=stack_backend,
        safe_fallback=safe_fallback,
        admission_reason=reason,
    )


__all__ = [
    "ExecutionPlan",
    "HardwareCapabilities",
    "ExecutionMemoryPlan",
    "ResidentQualification",
    "LoRATargetGeometry",
    "PrimitiveSelection",
    "plan_execution",
    "resident_qualification_for_plan",
]
