"""Hardware plan for a complete resident SwiGLU MLP backward program.

The accepted mixed-island edge uses 100 FP32-destination matmul workers and
10 exact-BF16 activation workers.  The next program keeps that split, then
reuses the 100 matmul workers as two 50-core gate/up groups.  dGate and dUp
remain banked in the activation island's L1 and are read over NoC; they are
never materialized as PJRT outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .model_workloads import EXACT_BASE_DX_K_PHASES, EXACT_BASE_K_PHASES
from .tt_lang_frozen_linear_lora_backward import (
    estimate_fused_frozen_lora_backward_l1_bytes,
)
from .tt_lang_grouped2_resident_backward import (
    estimate_resident_grouped_backward_l1_bytes,
)
from .tt_lang_lora import TILE_SIZE
from .tt_lang_resident_down_activation_backward import (
    resident_down_activation_backward_support_error,
)


MATMUL_CORES = 100
ACTIVATION_CORES = 10
MEMBER_CORES = MATMUL_CORES // 2
MAX_CB_BLOCKS = 32
BF16_TILE_BYTES = TILE_SIZE * TILE_SIZE * 2
L1_BUDGET_BYTES = int(1.46 * 1024 * 1024 * 0.75)


@dataclass(frozen=True)
class FullMLPBackwardPlan:
    tokens: int
    hidden_features: int
    intermediate_features: int
    physical_rank: int
    down_base_dx_phase: int
    gate_up_base_dx_phase: int
    down_jobs_per_core: int
    activation_jobs_per_core: int
    activation_banks_per_output: int
    activation_resident_l1_bytes: int
    gate_up_jobs_per_member_core: int
    matmul_peak_l1_bytes: int
    activation_peak_l1_bytes: int
    output_tile_shape: tuple[int, int]
    mlp_dx_row: int
    gate_da_row: int
    up_da_row: int
    down_da_row: int
    gate_db_row: int
    up_db_row: int
    down_db_row: int

    @property
    def custom_calls(self) -> int:
        return 1

    @property
    def intermediate_dram_bytes(self) -> int:
        return 0


def full_mlp_backward_support_error(
    *,
    tokens: int,
    hidden_features: int,
    intermediate_features: int,
    physical_rank: int,
) -> str | None:
    dimensions = (
        tokens,
        hidden_features,
        intermediate_features,
        physical_rank,
    )
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        return "resident full MLP dimensions must be positive and /32"
    if physical_rank != TILE_SIZE:
        return "resident full MLP currently requires physical rank 32"

    down_phase = EXACT_BASE_DX_K_PHASES.get(
        (intermediate_features, hidden_features)
    )
    gate_up_phase = EXACT_BASE_DX_K_PHASES.get(
        (hidden_features, intermediate_features)
    )
    if down_phase is None:
        return "no exact down-projection dX schedule"
    if gate_up_phase is None:
        return "no exact gate/up dX schedule"
    if EXACT_BASE_K_PHASES.get(
        (hidden_features, intermediate_features)
    ) is None:
        return "no exact gate/up forward schedule"

    error = resident_down_activation_backward_support_error(
        tokens=tokens,
        input_features=intermediate_features,
        output_features=hidden_features,
        physical_rank=physical_rank,
        base_reduction_tiles_per_phase=down_phase,
    )
    if error is not None:
        return error

    m_tiles = tokens // TILE_SIZE
    hidden_tiles = hidden_features // TILE_SIZE
    intermediate_tiles = intermediate_features // TILE_SIZE
    down_jobs = math.ceil(
        m_tiles * intermediate_tiles / MATMUL_CORES
    )
    activation_jobs = max(
        sum(
            min(
                down_jobs,
                m_tiles * intermediate_tiles - producer * down_jobs,
            )
            for producer in range(
                activation_node, MATMUL_CORES, ACTIVATION_CORES
            )
            if producer * down_jobs < m_tiles * intermediate_tiles
        )
        for activation_node in range(ACTIVATION_CORES)
    )
    activation_banks = math.ceil(activation_jobs / MAX_CB_BLOCKS)
    if activation_banks > 8:
        return "activation L1 handoff needs more than eight CB banks/output"
    gate_up_jobs = math.ceil(
        m_tiles * hidden_tiles / MEMBER_CORES
    )
    if gate_up_jobs > MAX_CB_BLOCKS:
        return "gate/up schedule exceeds the 32-block CB limit"

    try:
        plan = plan_full_mlp_backward(
            tokens=tokens,
            hidden_features=hidden_features,
            intermediate_features=intermediate_features,
            physical_rank=physical_rank,
            _skip_support_check=True,
        )
    except ValueError as error:
        return str(error)
    if plan.matmul_peak_l1_bytes > L1_BUDGET_BYTES:
        return (
            "full MLP matmul island exceeds 75% L1: "
            f"{plan.matmul_peak_l1_bytes} > {L1_BUDGET_BYTES}"
        )
    if plan.activation_peak_l1_bytes > L1_BUDGET_BYTES:
        return (
            "full MLP activation island exceeds 75% L1: "
            f"{plan.activation_peak_l1_bytes} > {L1_BUDGET_BYTES}"
        )
    return None


def plan_full_mlp_backward(
    *,
    tokens: int,
    hidden_features: int,
    intermediate_features: int,
    physical_rank: int,
    _skip_support_check: bool = False,
) -> FullMLPBackwardPlan:
    if not _skip_support_check:
        error = full_mlp_backward_support_error(
            tokens=tokens,
            hidden_features=hidden_features,
            intermediate_features=intermediate_features,
            physical_rank=physical_rank,
        )
        if error is not None:
            raise ValueError(error)

    m_tiles = tokens // TILE_SIZE
    hidden_tiles = hidden_features // TILE_SIZE
    intermediate_tiles = intermediate_features // TILE_SIZE
    down_phase = EXACT_BASE_DX_K_PHASES[
        (intermediate_features, hidden_features)
    ]
    gate_up_phase = EXACT_BASE_DX_K_PHASES[
        (hidden_features, intermediate_features)
    ]
    down_jobs = math.ceil(
        m_tiles * intermediate_tiles / MATMUL_CORES
    )
    activation_jobs = max(
        sum(
            min(
                down_jobs,
                m_tiles * intermediate_tiles - producer * down_jobs,
            )
            for producer in range(
                activation_node, MATMUL_CORES, ACTIVATION_CORES
            )
            if producer * down_jobs < m_tiles * intermediate_tiles
        )
        for activation_node in range(ACTIVATION_CORES)
    )
    activation_banks = math.ceil(activation_jobs / MAX_CB_BLOCKS)
    gate_up_jobs = math.ceil(
        m_tiles * hidden_tiles / MEMBER_CORES
    )

    down_l1 = estimate_fused_frozen_lora_backward_l1_bytes(
        token_tiles=m_tiles,
        input_tiles=intermediate_tiles,
        output_tiles=hidden_tiles,
        rank_tiles=1,
        base_reduction_tiles_per_phase=down_phase,
    )
    grouped_l1_55 = estimate_resident_grouped_backward_l1_bytes(
        tokens=tokens,
        input_features=hidden_features,
        member_output_features=intermediate_features,
        physical_rank=physical_rank,
        base_reduction_tiles_per_phase=gate_up_phase,
        member_core_count=MEMBER_CORES,
    )
    matmul_peak = max(down_l1, grouped_l1_55)

    activation_resident = (
        2 * activation_jobs * BF16_TILE_BYTES
    )
    # Thirteen two-slot BF16 CBs are used by the accepted exact activation
    # derivative pipeline. The true full-owner product producer adds six
    # depth-two streaming CBs (gate, saved gate, up, sigmoid, SiLU, output).
    # Crucially, none scales with the 61/63 local Qwen product jobs.
    activation_scratch = (13 + 6) * 2 * BF16_TILE_BYTES
    activation_peak = activation_resident + activation_scratch

    output_rows = (
        m_tiles
        + 3
        + 2 * intermediate_tiles
        + hidden_tiles
    )
    mlp_dx_row = 0
    gate_da_row = m_tiles
    up_da_row = gate_da_row + 1
    down_da_row = up_da_row + 1
    gate_db_row = down_da_row + 1
    up_db_row = gate_db_row + intermediate_tiles
    down_db_row = up_db_row + intermediate_tiles
    if down_db_row + hidden_tiles != output_rows:
        raise RuntimeError("full MLP final arena layout is inconsistent")
    return FullMLPBackwardPlan(
        tokens=tokens,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        physical_rank=physical_rank,
        down_base_dx_phase=down_phase,
        gate_up_base_dx_phase=gate_up_phase,
        down_jobs_per_core=down_jobs,
        activation_jobs_per_core=activation_jobs,
        activation_banks_per_output=activation_banks,
        activation_resident_l1_bytes=activation_resident,
        gate_up_jobs_per_member_core=gate_up_jobs,
        matmul_peak_l1_bytes=matmul_peak,
        activation_peak_l1_bytes=activation_peak,
        output_tile_shape=(output_rows, intermediate_tiles),
        mlp_dx_row=mlp_dx_row,
        gate_da_row=gate_da_row,
        up_da_row=up_da_row,
        down_da_row=down_da_row,
        gate_db_row=gate_db_row,
        up_db_row=up_db_row,
        down_db_row=down_db_row,
    )
