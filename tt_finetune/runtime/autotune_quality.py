"""autotuner adaptive search and geometry-quality acceptance contracts.

This module deliberately separates three concepts that were previously easy
to conflate:

* candidate generation says which schedules are executable;
* a normal autotune portfolio says what a user is allowed to measure;
* a broader qualification portfolio estimates the best known valid plan.

All final comparisons consume synchronized complete-step measurements.  Region
or micro-kernel timings may order candidates, but are not accepted as winners.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import statistics
from typing import Any, Mapping, Sequence

from .autotuner import ExecutionGeometryCandidate


def candidate_runtime_signature(
    candidate: ExecutionGeometryCandidate,
) -> tuple[Any, ...]:
    """Return the knobs which are genuinely executable for this candidate.

    Rejected candidates intentionally retain their speculative fields for the
    audit, but only admitted candidates may enter a measured portfolio.
    """

    if not candidate.admitted:
        raise ValueError("rejected candidates have no executable signature")
    return (
        candidate.graph_partition,
        candidate.attention_strategy,
        candidate.peft_strategy,
        candidate.attention_query_block,
        candidate.attention_kv_block,
        candidate.attention_cores,
        candidate.mlp_cores,
        candidate.lora_cores,
        candidate.moe_cores,
        candidate.layer_group_size,
        candidate.rank_tile_width,
        candidate.activation_policy,
        candidate.frozen_weight_policy,
        candidate.moe_schedule,
        candidate.prefetch_depth,
        candidate.double_buffered,
        candidate.sequence_bucket,
        candidate.optimizer_packing_strategy,
        candidate.optimizer_pack_bytes,
    )


def _macro_signature(candidate: ExecutionGeometryCandidate) -> tuple[Any, ...]:
    return (
        candidate.graph_partition,
        candidate.attention_strategy,
        candidate.peft_strategy,
        candidate.activation_policy,
        candidate.frozen_weight_policy,
        candidate.moe_schedule,
        candidate.attention_query_block,
        candidate.attention_kv_block,
        candidate.attention_cores,
        candidate.mlp_cores,
        candidate.layer_group_size,
        candidate.prefetch_depth,
        candidate.double_buffered,
        candidate.sequence_bucket,
        candidate.rank_tile_width,
    )


def adaptive_candidate_portfolio(
    candidates: Sequence[ExecutionGeometryCandidate],
    *,
    static_candidate_id: str,
    budget: int,
    qualification_oracle: bool = False,
) -> tuple[ExecutionGeometryCandidate, ...]:
    """Choose a bounded macro-first measurement portfolio.

    The static plan is always first and every runtime-equivalent candidate is
    deduplicated.  Normal search maximizes macro diversity before considering
    nearby numerical pack sizes.  Qualification-oracle search uses the same
    ordering but exhausts the supplied broader budget.
    """

    if budget <= 0:
        raise ValueError("candidate budget must be positive")
    executable = [item for item in candidates if item.admitted]
    if not executable:
        raise ValueError("the portfolio needs at least one admitted candidate")
    by_runtime: dict[tuple[Any, ...], ExecutionGeometryCandidate] = {}
    for item in sorted(
        executable, key=lambda value: (value.analytical_score, value.candidate_id)
    ):
        by_runtime.setdefault(candidate_runtime_signature(item), item)
    unique = list(by_runtime.values())
    static = next(
        (item for item in unique if item.candidate_id == static_candidate_id),
        None,
    )
    if static is None:
        raise ValueError("static candidate is absent from admitted candidates")

    selected = [static]
    remaining = [item for item in unique if item.candidate_id != static.candidate_id]
    while remaining and len(selected) < min(budget, len(unique)):
        macros = {_macro_signature(item) for item in selected}
        partitions = {item.graph_partition for item in selected}
        memory = {
            (item.activation_policy, item.frozen_weight_policy)
            for item in selected
        }
        attention = {
            (
                item.attention_query_block,
                item.attention_kv_block,
                item.attention_cores,
            )
            for item in selected
        }
        mlp = {(item.mlp_cores, item.layer_group_size) for item in selected}
        moe = {item.moe_schedule for item in selected}
        peft = {item.peft_strategy for item in selected}
        optimizer = {
            (item.optimizer_packing_strategy, item.optimizer_pack_bytes)
            for item in selected
        }
        data = {item.sequence_bucket for item in selected}

        def novelty(item: ExecutionGeometryCandidate) -> tuple[float, ...]:
            # Whole schedule categories dominate. Pack-size proximity is last.
            macro_new = _macro_signature(item) not in macros
            score = (
                1_000_000.0 * macro_new
                + 100_000.0 * (item.graph_partition not in partitions)
                + 50_000.0 * (
                    (item.activation_policy, item.frozen_weight_policy)
                    not in memory
                )
                + 40_000.0 * (
                    (
                        item.attention_query_block,
                        item.attention_kv_block,
                        item.attention_cores,
                    ) not in attention
                )
                + 30_000.0 * (
                    (item.mlp_cores, item.layer_group_size) not in mlp
                )
                # Expert scheduling changes the physical work performed for a
                # routed layer.  On an MoE request it must enter the normal
                # bounded portfolio before smaller graph/packing variants;
                # otherwise a budget of four can entirely miss the native
                # selected-expert schedule while spending all four slots on
                # dense-boundary variations.
                + 200_000.0 * (item.moe_schedule not in moe)
                # A native PEFT owner changes the dominant forward/backward
                # dataflow.  It must be sampled before a second nearby graph
                # boundary or optimizer packing layout; otherwise small
                # budgets can completely miss the only candidate that owns
                # new device work.
                + 150_000.0 * (item.peft_strategy not in peft)
                + 10_000.0 * (item.sequence_bucket not in data)
                + 1_000.0 * (
                    (item.optimizer_packing_strategy, item.optimizer_pack_bytes)
                    not in optimizer
                )
            )
            # The broader oracle prefers exhaustive diversity; the normal path
            # uses analytical rank only to break equal-coverage ties.
            analytical = -float(item.analytical_score)
            return (score, analytical, -float(len(item.candidate_id)))

        chosen = max(remaining, key=novelty)
        selected.append(chosen)
        remaining.remove(chosen)

    # The flag is semantic/documentary today: a qualification runner supplies
    # a larger budget. It exists so call sites cannot accidentally label a
    # normal bounded search as the oracle authority.
    del qualification_oracle
    return tuple(selected)


def measurement_objective(measurement: Mapping[str, Any]) -> float:
    """Lower-is-better complete-step cost per useful token."""

    median_ms = float(measurement["median_ms"])
    useful_tps = measurement.get("median_useful_tokens_per_second")
    if useful_tps is not None:
        useful_tps = float(useful_tps)
        if not math.isfinite(useful_tps) or useful_tps <= 0:
            raise ValueError("useful-token throughput must be finite and positive")
        return 1000.0 / useful_tps
    if not math.isfinite(median_ms) or median_ms <= 0:
        raise ValueError("median step latency must be finite and positive")
    return median_ms


def select_non_regressing_winner(
    measurements: Sequence[Mapping[str, Any]],
    *,
    static_candidate_id: str,
    maximum_static_regression: float = 0.03,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Select the best measured plan with an explicit static safety rail."""

    if maximum_static_regression < 0:
        raise ValueError("maximum regression must be non-negative")
    by_id = {str(item["candidate_id"]): item for item in measurements}
    if static_candidate_id not in by_id:
        raise ValueError("static measurement is required for safe selection")
    static = by_id[static_candidate_id]
    static_cost = measurement_objective(static)
    eligible = [
        item for item in measurements
        if measurement_objective(item)
        <= static_cost * (1.0 + maximum_static_regression)
    ]
    winner = min(eligible, key=measurement_objective)
    winner_cost = measurement_objective(winner)
    return winner, {
        "static_candidate_id": static_candidate_id,
        "static_objective": static_cost,
        "selected_objective": winner_cost,
        "maximum_static_regression": maximum_static_regression,
        "static_fallback_selected": (
            str(winner["candidate_id"]) == static_candidate_id
        ),
        "selected_speedup": static_cost / winner_cost,
    }


@dataclass(frozen=True)
class GeometryStressCase:
    case_id: str
    semantics: Mapping[str, Any]
    rank: int
    batch: int
    sequence: int
    gradient_accumulation: int
    target_modules: tuple[str, ...]
    memory_mode: str
    data_shape: str
    difficulty: str
    performance_cliff_case: bool = False

    @property
    def semantic_fingerprint(self) -> str:
        payload = json.dumps(
            dict(self.semantics), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "semantic_fingerprint": self.semantic_fingerprint}


def default_geometry_stress_matrix() -> tuple[GeometryStressCase, ...]:
    """Return the frozen bounded, non-Cartesian autotuner matrix specification."""

    dense = {
        "kind": "dense", "norm": "rmsnorm", "attention": "gqa",
        "qkv": "separate", "mlp": "swiglu", "hidden": 128,
        "intermediate": 384, "heads": 4, "kv_heads": 2, "head_dim": 32,
        "layers": 2,
    }
    cases = [
        ("gqa-r4-b1-s128", dense, 4, 1, 128, 1, "friendly"),
        ("gqa-r8-b2-s256", dense, 8, 2, 256, 1, "friendly"),
        ("mqa-r16-b3-s128", {**dense, "attention": "mqa", "kv_heads": 1}, 16, 3, 128, 2, "awkward"),
        ("layernorm-fused-r32-b4-s256", {**dense, "norm": "layernorm", "attention": "mha", "qkv": "fused_conv1d", "mlp": "ordinary", "intermediate": 512, "kv_heads": 4}, 32, 4, 256, 1, "friendly"),
        ("asymmetric-fused-mqa-r73-b1-s128", {**dense, "norm": "layernorm", "attention": "mqa", "qkv": "fused_asymmetric", "mlp": "ordinary", "intermediate": 512, "kv_heads": 1}, 73, 1, 128, 1, "awkward"),
        ("tail-hidden-r128-b2-s256", {**dense, "hidden": 192, "intermediate": 352, "heads": 6, "kv_heads": 2}, 128, 2, 256, 1, "awkward"),
        ("pythia-mha-r8-b1-s1024", {**dense, "norm": "layernorm", "attention": "mha", "qkv": "fused_linear", "mlp": "ordinary", "intermediate": 512, "kv_heads": 4}, 8, 1, 1024, 1, "awkward"),
        ("gqa-r64-b2-s512", dense, 64, 2, 512, 1, "awkward"),
        ("gqa-r32-b3-s1024", dense, 32, 3, 1024, 2, "awkward"),
        ("layernorm-fused-r16-b1-s2048", {**dense, "norm": "layernorm", "attention": "mha", "qkv": "fused_conv1d", "mlp": "ordinary", "intermediate": 512, "kv_heads": 4}, 16, 1, 2048, 2, "awkward"),
        ("mqa-r4-b1-s4096", {**dense, "attention": "mqa", "kv_heads": 1}, 4, 1, 4096, 4, "awkward"),
        ("layernorm-mha-r4-b2-s1024", {**dense, "norm": "layernorm", "attention": "mha", "qkv": "fused_linear", "mlp": "ordinary", "intermediate": 512, "kv_heads": 4}, 4, 2, 1024, 1, "friendly"),
        ("tail-r128-b4-s128", {**dense, "hidden": 192, "intermediate": 352, "heads": 6, "kv_heads": 2}, 128, 4, 128, 1, "awkward"),
        ("packed-data-r8-b3-s256", dense, 8, 3, 256, 2, "awkward"),
        ("reduced-boundary-r8-b1-s32", {**dense, "hidden": 3584, "intermediate": 18944, "heads": 28, "kv_heads": 4, "head_dim": 128, "layers": 28}, 8, 1, 32, 1, "performance_cliff"),
    ]
    result: list[GeometryStressCase] = []
    for case_id, semantics, rank, batch, sequence, accumulation, difficulty in cases:
        result.append(GeometryStressCase(
            case_id=case_id,
            semantics=semantics,
            rank=rank,
            batch=batch,
            sequence=sequence,
            gradient_accumulation=accumulation,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
            memory_mode="hybrid_frozen" if case_id.startswith("reduced-") else "auto",
            data_shape="variable_packed" if case_id.startswith("packed-") else "fixed",
            difficulty=difficulty,
            performance_cliff_case=difficulty == "performance_cliff",
        ))
    moe_base = {
        "kind": "moe", "norm": "rmsnorm", "attention": "gqa",
        "qkv": "separate", "mlp": "swiglu", "hidden": 128,
        "intermediate": 128, "heads": 4, "kv_heads": 2, "head_dim": 32,
        "layers": 1,
    }
    moe_cases = (
        ("moe-capacity-r8-b1-s128", {**moe_base, "experts": 4, "top_k": 2, "capacity": 1.25, "overflow": "drop", "expert_storage": "fused"}, 8, 1, 128),
        ("moe-dynamic-r16-b2-s256", {**moe_base, "hidden": 512, "intermediate": 512, "heads": 8, "kv_heads": 2, "head_dim": 64, "experts": 4, "top_k": 2, "capacity": None, "expert_storage": "fused"}, 16, 2, 256),
        ("moe-shared-r32-b3-s128", {**moe_base, "experts": 4, "top_k": 2, "shared_experts": 1, "capacity": None, "expert_storage": "separate"}, 32, 3, 128),
        ("moe-fused-r73-b1-s1024", {**moe_base, "experts": 4, "top_k": 2, "expert_storage": "fused", "capacity": None}, 73, 1, 1024),
        ("moe-tail-r128-b4-s256", {**moe_base, "experts": 4, "top_k": 2, "expert_storage": "fused", "capacity": None}, 128, 4, 256),
    )
    for case_id, semantics, rank, batch, sequence in moe_cases:
        result.append(GeometryStressCase(
            case_id=case_id, semantics=semantics, rank=rank, batch=batch,
            sequence=sequence, gradient_accumulation=1,
            target_modules=("q_proj", "v_proj", "experts.gate_proj", "experts.up_proj", "experts.down_proj"),
            memory_mode="auto", data_shape="capacity_skewed" if semantics.get("capacity") else "expert_skewed",
            difficulty="awkward",
        ))
    return tuple(result)


def qualify_autotuner_case(
    *,
    case: GeometryStressCase,
    static: Mapping[str, Any],
    automatic: Mapping[str, Any],
    oracle: Mapping[str, Any],
    selected_candidate: Mapping[str, Any],
    candidate_count: int,
    rejected_candidates: Sequence[Mapping[str, Any]],
    tuning_wall_seconds: float,
    cache_status: str,
    correctness_contract: str,
) -> dict[str, Any]:
    static_ms = float(static["median_ms"])
    auto_ms = float(automatic["median_ms"])
    oracle_ms = float(oracle["median_ms"])
    if min(static_ms, auto_ms, oracle_ms) <= 0:
        raise ValueError("authority medians must be positive")
    potential = static_ms / oracle_ms
    static_gap = static_ms - oracle_ms
    recovered = (
        (static_ms - auto_ms) / static_gap if static_gap > 0 else 1.0
    )
    return {
        **case.to_dict(),
        "static_baseline_median_ms": static_ms,
        "automatic_plan_median_ms": auto_ms,
        "qualification_oracle_median_ms": oracle_ms,
        "automatic_speedup": static_ms / auto_ms,
        "oracle_potential_speedup": potential,
        "auto_vs_oracle_ratio": auto_ms / oracle_ms,
        "recovered_gap": recovered,
        "selected_candidate": dict(selected_candidate),
        "candidate_count": int(candidate_count),
        "rejected_candidates": [dict(item) for item in rejected_candidates],
        "tuning_wall_seconds": float(tuning_wall_seconds),
        "cache_status": cache_status,
        "correctness_contract": correctness_contract,
    }


def aggregate_autotuner_quality(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute the hard autotuner gates without performance-cliff inflation."""

    fitting = [item for item in cases if not item.get("capacity_failure")]
    general = [
        item for item in fitting if not item.get("performance_cliff_case", False)
    ]
    if not general:
        raise ValueError("at least one non-cliff fitting case is required")
    awkward = [
        item for item in general
        if float(item["static_baseline_median_ms"])
        > 1.10 * float(item["qualification_oracle_median_ms"])
    ]
    within_5 = statistics.fmean(
        float(item["auto_vs_oracle_ratio"]) <= 1.05 for item in general
    )
    within_10 = statistics.fmean(
        float(item["auto_vs_oracle_ratio"]) <= 1.10 for item in general
    )
    worst_regression = max(
        float(item["automatic_plan_median_ms"])
        / float(item["static_baseline_median_ms"]) - 1.0
        for item in general
    )
    recovered = [float(item["recovered_gap"]) for item in awkward]
    result = {
        "case_count": len(cases),
        "fitting_general_case_count": len(general),
        "performance_cliff_case_count": len(fitting) - len(general),
        "awkward_available_gap_case_count": len(awkward),
        "median_normal_autotuner_speedup": statistics.median(
            float(item["automatic_speedup"]) for item in general
        ),
        "awkward_subset_median_speedup": (
            statistics.median(float(item["automatic_speedup"]) for item in awkward)
            if awkward else None
        ),
        "median_recovered_gap": statistics.median(recovered) if recovered else None,
        "percentage_within_5_percent_of_oracle": 100.0 * within_5,
        "percentage_within_10_percent_of_oracle": 100.0 * within_10,
        "worst_auto_vs_static_regression": worst_regression,
        "median_tuning_wall_seconds": statistics.median(
            float(item["tuning_wall_seconds"]) for item in general
        ),
        "cache_reuse_rate": statistics.fmean(
            str(item.get("cache_status", "")).startswith("hit")
            for item in general
        ),
    }
    recovered_gate = (
        statistics.median(recovered) >= 0.70 if recovered else True
    )
    result["gates"] = {
        "at_least_90_percent_within_10_percent_oracle": within_10 >= 0.90,
        "no_more_than_3_percent_static_regression": worst_regression <= 0.03,
        "awkward_median_recovered_gap_at_least_70_percent": recovered_gate,
    }
    result["awkward_gap_gate_status"] = (
        "measured"
        if recovered
        else "not_applicable_no_general_case_has_more_than_10_percent_oracle_gap"
    )
    result["accepted"] = all(result["gates"].values())
    return result


__all__ = [
    "GeometryStressCase",
    "adaptive_candidate_portfolio",
    "aggregate_autotuner_quality",
    "candidate_runtime_signature",
    "default_geometry_stress_matrix",
    "measurement_objective",
    "qualify_autotuner_case",
    "select_non_regressing_winner",
]
