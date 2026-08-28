"""Deterministic semantic-to-physical compilation for Blackhole training.

Normal training must not benchmark a portfolio of plans or consume a stale
measured winner.  This module lowers one admitted semantic training request to
one executable worker/resident ABI using only architecture, geometry, memory,
adapter, data-shape and P150a resource contracts.  Measured autotuning remains
available separately as an offline calibration and qualification tool.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

from .autotuner import (
    ExecutionGeometryCandidate,
    PlannerRequest,
    candidate_worker_arguments,
    generate_candidates,
)


_SCHEMA = "tt-finetune-deterministic-execution-plan-v1"


def _digest(value: Mapping[str, Any], *, length: int = 24) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


@dataclass(frozen=True)
class CompiledPhysicalPlan:
    """One executable physical contract emitted without online measurement."""

    plan_id: str
    semantic_fingerprint: str
    execution_backend: str
    selection_authority: str
    geometry: dict[str, Any]
    graph_partition: str
    memory_policy: dict[str, Any]
    component_policies: dict[str, Any]
    numerical_contract: dict[str, str]
    execution_candidate: dict[str, Any] | None
    worker_arguments: tuple[str, ...]
    request_contract: dict[str, Any]
    online_candidate_search: bool = False
    measured_profile_cache_consulted: bool = False
    schema: str = _SCHEMA

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["worker_arguments"] = list(self.worker_arguments)
        return value


def _first_executable_static_candidate(
    request: PlannerRequest,
) -> ExecutionGeometryCandidate:
    """Return the safe deterministic runtime baseline for this request.

    ``generate_candidates`` preserves the worker's accepted/default ABI as its
    first candidate and appends experimental macro alternatives afterwards.
    Normal compilation consumes only that baseline.  Offline qualification may
    measure the alternatives and use the resulting calibration to improve this
    deterministic rule in a later source revision.
    """

    candidates = generate_candidates(request)
    for candidate in candidates:
        if candidate.admitted:
            return candidate
    # ``generate_candidates`` already rejects this condition; retain a local
    # invariant so a future implementation cannot emit an empty physical plan.
    raise ValueError("deterministic compiler found no executable static plan")


def compile_physical_training_plan(
    request: PlannerRequest,
) -> CompiledPhysicalPlan:
    """Compile exactly one executable Blackhole training plan.

    The result is independent of checkpoint/repository names.  It is not a
    performance claim: it is the deterministic physical ABI chosen by the
    current calibrated compiler source.  The direct backend later binds it to
    the concrete compiler/runtime source hashes before execution.
    """

    plan = request.safe_plan
    plan.require_admitted()
    semantic_contract = request.semantic_key_document()
    selections = {
        item.primitive: item.implementation for item in plan.selections
    }
    numerical = {
        item.primitive: item.qualification for item in plan.selections
    }
    geometry = {
        "rank": plan.rank,
        "rank_tiles": plan.rank_tiles,
        "physical_rank": plan.physical_rank,
        "rank_tail": plan.rank_tail,
        "batch_size": plan.batch_size,
        "sequence_length": plan.sequence_length,
        "sequence_blocks": plan.sequence_blocks,
        "sequence_tail": plan.sequence_tail,
        "logical_tokens": plan.logical_tokens,
        "physical_tokens": plan.physical_tokens,
    }

    candidate: ExecutionGeometryCandidate | None
    if plan.selected_stack_backend == "resident_exact":
        candidate = None
        graph_partition = "resident_stack_five_call"
        authority = "qualified semantic resident contract"
        worker_arguments: tuple[str, ...] = ()
    else:
        candidate = _first_executable_static_candidate(request)
        graph_partition = candidate.graph_partition
        authority = "deterministic calibrated static cost model"
        worker_arguments = candidate_worker_arguments(candidate)

    identity = {
        "schema": _SCHEMA,
        "semantic_fingerprint": plan.architecture.semantic_fingerprint,
        "execution_backend": plan.selected_stack_backend,
        "geometry": geometry,
        "graph_partition": graph_partition,
        "memory_policy": plan.memory_plan.to_dict(),
        "component_policies": selections,
        "numerical_contract": numerical,
        "execution_candidate": (
            candidate.to_dict() if candidate is not None else None
        ),
        "worker_arguments": list(worker_arguments),
        "request_contract": semantic_contract,
    }
    return CompiledPhysicalPlan(
        plan_id="cp-" + _digest(identity),
        semantic_fingerprint=plan.architecture.semantic_fingerprint,
        execution_backend=plan.selected_stack_backend,
        selection_authority=authority,
        geometry=geometry,
        graph_partition=graph_partition,
        memory_policy=plan.memory_plan.to_dict(),
        component_policies=selections,
        numerical_contract=numerical,
        execution_candidate=(
            candidate.to_dict() if candidate is not None else None
        ),
        worker_arguments=worker_arguments,
        request_contract=semantic_contract,
    )


def bind_compiled_plan(
    plan: Mapping[str, Any],
    *,
    compiler_hash: str,
    runtime_source_hash: str,
) -> dict[str, Any]:
    """Bind a semantic physical plan to the executable implementation source."""

    bound = json.loads(json.dumps(plan, sort_keys=True, default=str))
    bound["compiler_hash"] = str(compiler_hash)
    bound["runtime_source_hash"] = str(runtime_source_hash)
    identity = dict(bound)
    identity.pop("physical_plan_id", None)
    bound["physical_plan_id"] = "p-" + _digest(identity)
    return bound


__all__ = [
    "CompiledPhysicalPlan",
    "bind_compiled_plan",
    "compile_physical_training_plan",
]
