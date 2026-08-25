"""Per-core Blackhole plan for a resident gated-MLP training boundary.

The planner describes ownership and lifetime; it does not pretend that the
120 worker L1s are one shared cache.  It is deliberately independent of XLA
so unsupported shapes can fall back before a custom-call graph is built.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_CORE_COUNT
from .tt_lang_grouped2_resident_backward import (
    estimate_resident_grouped_backward_l1_bytes,
    resident_backward_support_error,
)
from .tt_lang_lora import BF16_TILE_BYTES, TILE_SIZE
from .tt_lang_resident_down_backward import (
    estimate_resident_down_backward_l1_bytes,
    resident_down_backward_support_error,
)


@dataclass(frozen=True)
class ResidentMLPShape:
    tokens: int
    hidden_features: int
    intermediate_features: int
    physical_rank: int = TILE_SIZE


@dataclass(frozen=True)
class ResidentMLPTraffic:
    """Logical intermediate traffic, excluding weights and final outputs."""

    generic_forward_dram_bytes: int
    resident_forward_dram_bytes: int
    generic_backward_dram_bytes: int
    resident_backward_dram_bytes: int
    eliminated_dram_bytes: int
    forward_peer_noc_bytes: int
    down_input_noc_bytes_upper_bound: int


@dataclass(frozen=True)
class ResidentMLPPlan:
    shape: ResidentMLPShape
    token_tiles: int
    hidden_tiles: int
    intermediate_tiles: int
    member_cores: int
    gate_up_jobs_per_member_core: int
    down_jobs_per_core: int
    fused_matmul_cores: int
    fused_activation_cores: int
    fused_gate_up_member_cores: int
    fused_gate_up_jobs_per_member_core: int
    fused_activation_jobs_per_core: int
    product_tiles_per_owner_core: int
    resident_down_backward_l1_bytes: int
    resident_grouped_backward_l1_bytes: int
    activation_streaming_l1_bytes: int
    naive_unaliased_backward_l1_bytes: int
    estimated_backward_peak_l1_bytes: int
    l1_budget_bytes: int
    activation_checkpoint: str
    product_owner: str
    exact_rounding_contract: tuple[str, ...]
    backward_phase_order: tuple[str, ...]
    final_backward_outputs: tuple[str, ...]
    traffic: ResidentMLPTraffic

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def plan_resident_gated_mlp(
    *,
    tokens: int,
    hidden_features: int,
    intermediate_features: int,
    physical_rank: int = TILE_SIZE,
    gate_up_base_dx_tiles_per_phase: int = 1,
    activation_checkpoint: str = "save_gate_up",
    worker_l1_bytes: int = int(1.46 * 1024 * 1024),
    l1_fraction: float = 0.75,
    fused_activation_cores: int = 10,
) -> ResidentMLPPlan:
    """Plan the first resident gate/up -> SwiGLU -> down boundary.

    Gate and up use two 55-core halves.  Corresponding output tiles meet on
    the gate half, where the SwiGLU product remains distributed until the
    down projection consumes it.  During backward, activation derivatives
    are produced tile-by-tile directly into the grouped gate/up backward dY
    CB.  The three small depth-two activation input CBs are therefore the only
    incremental backward allocation; a second full intermediate row is not.
    """

    shape = ResidentMLPShape(
        tokens=tokens,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        physical_rank=physical_rank,
    )
    dimensions = (
        tokens,
        hidden_features,
        intermediate_features,
        physical_rank,
    )
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError("resident MLP dimensions must be positive /32")
    if physical_rank != TILE_SIZE:
        raise ValueError("resident MLP currently requires physical rank 32")
    if activation_checkpoint not in {"save_gate_up", "recompute_gate_up"}:
        raise ValueError(
            "activation_checkpoint must be save_gate_up or recompute_gate_up"
        )
    if worker_l1_bytes <= 0 or not 0.0 < l1_fraction <= 1.0:
        raise ValueError("invalid worker L1 budget")
    fused_matmul_cores = (
        BLACKHOLE_P150A_CORE_COUNT - fused_activation_cores
    )
    if (
        fused_activation_cores <= 0
        or fused_matmul_cores <= 0
        or fused_matmul_cores % 2
    ):
        raise ValueError(
            "fused activation island must leave a positive even matmul core "
            "count for two equal gate/up groups"
        )

    support_error = resident_backward_support_error(
        tokens=tokens,
        input_features=hidden_features,
        member_output_features=intermediate_features,
        physical_rank=physical_rank,
        base_reduction_tiles_per_phase=(
            gate_up_base_dx_tiles_per_phase
        ),
    )
    if support_error is not None:
        raise ValueError(f"gate/up resident backward unsupported: {support_error}")
    down_support_error = resident_down_backward_support_error(
        tokens=tokens,
        input_features=intermediate_features,
        output_features=hidden_features,
        physical_rank=physical_rank,
        base_reduction_tiles_per_phase=1,
    )
    if down_support_error is not None:
        raise ValueError(
            f"down resident backward unsupported: {down_support_error}"
        )

    m_tiles = tokens // TILE_SIZE
    h_tiles = hidden_features // TILE_SIZE
    i_tiles = intermediate_features // TILE_SIZE
    member_cores = BLACKHOLE_P150A_CORE_COUNT // 2
    gate_up_jobs = m_tiles * i_tiles
    gate_up_jobs_per_core = _ceil_div(gate_up_jobs, member_cores)
    down_jobs_per_core = _ceil_div(
        m_tiles * h_tiles, BLACKHOLE_P150A_CORE_COUNT
    )
    fused_member_cores = fused_matmul_cores // 2
    fused_gate_up_jobs_per_core = _ceil_div(
        gate_up_jobs, fused_member_cores
    )
    fused_activation_jobs_per_core = _ceil_div(
        gate_up_jobs, fused_activation_cores
    )

    grouped_backward_l1 = estimate_resident_grouped_backward_l1_bytes(
        tokens=tokens,
        input_features=hidden_features,
        member_output_features=intermediate_features,
        physical_rank=physical_rank,
        base_reduction_tiles_per_phase=(
            gate_up_base_dx_tiles_per_phase
        ),
    )
    down_backward_l1 = estimate_resident_down_backward_l1_bytes(
        tokens=tokens,
        input_features=intermediate_features,
        output_features=hidden_features,
        physical_rank=physical_rank,
        base_reduction_tiles_per_phase=1,
    )
    # gate, up and dZ are streamed through depth-two one-tile CBs. dgate/dup
    # replace the grouped-backward dY CB contents and are not double-counted.
    activation_streaming_l1 = 3 * 2 * BF16_TILE_BYTES
    # The down and gate/up programs are sequential phases of one custom call.
    # Their CB address ranges must therefore alias by lifetime rather than be
    # allocated side-by-side. Activation's three depth-two streams overlap the
    # producer/consumer handoff and are the only incremental live allocation.
    # This distinction is essential for Qwen: the naive sum exceeds the safe
    # per-worker budget even though the lifetime-aliased schedule fits.
    naive_unaliased_peak = (
        down_backward_l1
        + grouped_backward_l1
        + activation_streaming_l1
    )
    estimated_peak = max(
        down_backward_l1 + activation_streaming_l1,
        grouped_backward_l1 + activation_streaming_l1,
    )
    budget = int(worker_l1_bytes * l1_fraction)
    if estimated_peak > budget:
        raise ValueError(
            "resident MLP backward exceeds configured per-worker L1: "
            f"{estimated_peak} > {budget}"
        )

    mi_tiles = m_tiles * i_tiles
    # Logical boundary transfers. One transfer is one BF16 tile read or write.
    # Generic forward writes/reads gate, up and product (6*MI). Resident
    # forward writes only gate/up for backward checkpointing (2*MI), or none
    # in recompute mode. Generic backward reads product, writes/reads dZ,
    # reads gate/up, and writes/reads dgate/dup (9*MI). Resident backward reads
    # only the two checkpoint tensors; all derivative edges remain local.
    generic_forward_tiles = 6 * mi_tiles
    resident_forward_tiles = (
        2 * mi_tiles if activation_checkpoint == "save_gate_up" else 0
    )
    generic_backward_tiles = 9 * mi_tiles
    resident_backward_tiles = (
        2 * mi_tiles if activation_checkpoint == "save_gate_up" else 0
    )
    traffic = ResidentMLPTraffic(
        generic_forward_dram_bytes=generic_forward_tiles * BF16_TILE_BYTES,
        resident_forward_dram_bytes=(
            resident_forward_tiles * BF16_TILE_BYTES
        ),
        generic_backward_dram_bytes=generic_backward_tiles * BF16_TILE_BYTES,
        resident_backward_dram_bytes=(
            resident_backward_tiles * BF16_TILE_BYTES
        ),
        eliminated_dram_bytes=(
            generic_forward_tiles
            + generic_backward_tiles
            - resident_forward_tiles
            - resident_backward_tiles
        )
        * BF16_TILE_BYTES,
        # One member's tile crosses to its paired owner for SwiGLU.
        forward_peer_noc_bytes=mi_tiles * BF16_TILE_BYTES,
        # Conservative unicast-equivalent accounting. The implementation can
        # turn this into row multicast; it is not an L1-capacity estimate.
        down_input_noc_bytes_upper_bound=(
            mi_tiles * h_tiles * BF16_TILE_BYTES
        ),
    )
    return ResidentMLPPlan(
        shape=shape,
        token_tiles=m_tiles,
        hidden_tiles=h_tiles,
        intermediate_tiles=i_tiles,
        member_cores=member_cores,
        gate_up_jobs_per_member_core=gate_up_jobs_per_core,
        down_jobs_per_core=down_jobs_per_core,
        fused_matmul_cores=fused_matmul_cores,
        fused_activation_cores=fused_activation_cores,
        fused_gate_up_member_cores=fused_member_cores,
        fused_gate_up_jobs_per_member_core=(
            fused_gate_up_jobs_per_core
        ),
        fused_activation_jobs_per_core=fused_activation_jobs_per_core,
        product_tiles_per_owner_core=gate_up_jobs_per_core,
        resident_down_backward_l1_bytes=down_backward_l1,
        resident_grouped_backward_l1_bytes=grouped_backward_l1,
        activation_streaming_l1_bytes=activation_streaming_l1,
        naive_unaliased_backward_l1_bytes=naive_unaliased_peak,
        estimated_backward_peak_l1_bytes=estimated_peak,
        l1_budget_bytes=budget,
        activation_checkpoint=activation_checkpoint,
        product_owner="gate-half core; matching up tile arrives by NoC",
        exact_rounding_contract=(
            "gate/up projection outputs pack to BF16 at reference boundary",
            "SiLU output packs to BF16 before multiply",
            "SwiGLU product packs to BF16 before down projection",
            "dZ packs to BF16 before activation backward",
            "dgate/dup pack at the generic autograd edge",
            "grouped gate/up dX retains the accepted staged BF16 add tree",
        ),
        backward_phase_order=(
            "FP32 matmul island: resident down backward",
            "NoC dProduct stream to BF16 activation island",
            "BF16 activation island: exact SwiGLU derivative",
            "NoC dGate/dUp stream to FP32 matmul island",
            "FP32 matmul island: resident grouped gate/up backward",
            "final_input_gradient_accumulation",
        ),
        final_backward_outputs=(
            "input_gradient",
            "gate_lora_a_gradient",
            "gate_lora_b_gradient",
            "up_lora_a_gradient",
            "up_lora_b_gradient",
            "down_lora_a_gradient",
            "down_lora_b_gradient",
        ),
        traffic=traffic,
    )
