"""Silicon-validated model-family workloads for native LoRA kernels.

The native backend is accepted per projection shape, not merely per model
name.  A family entry documents why each shape exists in a real transformer
and the exact long-K reduction phase measured against TT-NN on Blackhole.
Unknown shapes deliberately have no native contract and must fall back.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectionWorkload:
    name: str
    input_features: int
    output_features: int
    logical_rank: int
    exact_base_k_tiles_per_phase: int
    exact_base_dx_k_tiles_per_phase: int
    block_group: str


@dataclass(frozen=True)
class ModelFamilyWorkload:
    family: str
    model: str
    architecture: str
    tokens: int
    projections: tuple[ProjectionWorkload, ...]


MODEL_FAMILY_WORKLOADS = (
    ModelFamilyWorkload(
        family="pythia",
        model="EleutherAI/pythia-160m",
        architecture="GPTNeoXForCausalLM",
        tokens=960,
        projections=(
            ProjectionWorkload(
                name="combined_qkv",
                input_features=768,
                output_features=2304,
                logical_rank=8,
                exact_base_k_tiles_per_phase=1,
                exact_base_dx_k_tiles_per_phase=1,
                block_group="attention_combined",
            ),
        ),
    ),
    ModelFamilyWorkload(
        family="qwen",
        model="Qwen-family notebook and Qwen2.5-0.5B workloads",
        architecture="QwenForCausalLM",
        tokens=128,
        projections=(
            ProjectionWorkload(
                name="notebook_projection",
                input_features=1024,
                output_features=2048,
                logical_rank=16,
                exact_base_k_tiles_per_phase=2,
                exact_base_dx_k_tiles_per_phase=1,
                block_group="notebook_target",
            ),
            ProjectionWorkload(
                name="qwen2_5_q_or_o_proj",
                input_features=896,
                output_features=896,
                logical_rank=16,
                exact_base_k_tiles_per_phase=1,
                exact_base_dx_k_tiles_per_phase=1,
                block_group="attention_qkv",
            ),
            ProjectionWorkload(
                name="qwen2_5_k_or_v_proj",
                input_features=896,
                output_features=128,
                logical_rank=16,
                exact_base_k_tiles_per_phase=1,
                exact_base_dx_k_tiles_per_phase=1,
                block_group="attention_qkv",
            ),
            ProjectionWorkload(
                name="qwen2_5_gate_or_up_proj",
                input_features=896,
                output_features=4864,
                logical_rank=8,
                exact_base_k_tiles_per_phase=2,
                exact_base_dx_k_tiles_per_phase=1,
                block_group="mlp_gate_up",
            ),
            ProjectionWorkload(
                name="qwen2_5_down_proj",
                input_features=4864,
                output_features=896,
                logical_rank=8,
                exact_base_k_tiles_per_phase=1,
                exact_base_dx_k_tiles_per_phase=2,
                block_group="mlp_down",
            ),
        ),
    ),
    ModelFamilyWorkload(
        family="smollm2",
        model="HuggingFaceTB/SmolLM2-135M",
        architecture="LlamaForCausalLM",
        tokens=128,
        projections=(
            ProjectionWorkload(
                name="q_proj",
                input_features=576,
                output_features=576,
                logical_rank=8,
                exact_base_k_tiles_per_phase=1,
                exact_base_dx_k_tiles_per_phase=1,
                block_group="attention_qkv",
            ),
            ProjectionWorkload(
                name="k_or_v_proj",
                input_features=576,
                output_features=192,
                logical_rank=8,
                exact_base_k_tiles_per_phase=1,
                exact_base_dx_k_tiles_per_phase=1,
                block_group="attention_qkv",
            ),
            ProjectionWorkload(
                name="gate_or_up_proj",
                input_features=576,
                output_features=1536,
                logical_rank=8,
                exact_base_k_tiles_per_phase=2,
                exact_base_dx_k_tiles_per_phase=1,
                block_group="mlp_gate_up",
            ),
            ProjectionWorkload(
                name="down_proj",
                input_features=1536,
                output_features=576,
                logical_rank=8,
                exact_base_k_tiles_per_phase=1,
                exact_base_dx_k_tiles_per_phase=2,
                block_group="mlp_down",
            ),
        ),
    ),
)


def exact_base_k_phase_registry() -> dict[tuple[int, int], int]:
    """Build a shape registry and reject contradictory family contracts."""

    result: dict[tuple[int, int], int] = {}
    for family in MODEL_FAMILY_WORKLOADS:
        for projection in family.projections:
            key = (
                projection.input_features,
                projection.output_features,
            )
            previous = result.setdefault(
                key, projection.exact_base_k_tiles_per_phase
            )
            if previous != projection.exact_base_k_tiles_per_phase:
                raise RuntimeError(
                    f"conflicting exact K phases for projection {key}"
                )
    return result


EXACT_BASE_K_PHASES = exact_base_k_phase_registry()


def exact_base_dx_k_phase_registry() -> dict[tuple[int, int], int]:
    """Build exact dX schedules keyed by the original forward shape."""

    result: dict[tuple[int, int], int] = {}
    for family in MODEL_FAMILY_WORKLOADS:
        for projection in family.projections:
            key = (
                projection.input_features,
                projection.output_features,
            )
            previous = result.setdefault(
                key, projection.exact_base_dx_k_tiles_per_phase
            )
            if previous != projection.exact_base_dx_k_tiles_per_phase:
                raise RuntimeError(
                    f"conflicting exact dX K phases for projection {key}"
                )
    return result


EXACT_BASE_DX_K_PHASES = exact_base_dx_k_phase_registry()


def model_family_workload(family: str) -> ModelFamilyWorkload:
    for workload in MODEL_FAMILY_WORKLOADS:
        if workload.family == family:
            return workload
    raise KeyError(f"unknown model family: {family}")
