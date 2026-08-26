"""Geometry-aware autotuner candidate planning and measured profile selection.

The accepted architecture planner remains the semantic and memory-safety authority.
This module builds a bounded set of whole-step execution candidates on top of
that safe plan, rejects impossible schedules analytically, and only calls a
candidate *autotuned* after synchronized silicon measurements select it.

Checkpoint and repository names are deliberately absent from every key and
decision.  Equivalent architectures with the same tensor/training geometry
share the same candidate set and measured profile.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from tt_finetune.architecture import CanonicalArchitectureDescriptor

from .execution_planner import (
    ExecutionPlan,
    HardwareCapabilities,
    TILE_SIZE,
)


_CACHE_SCHEMA = 1


def _ceil_to(value: int, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment


def _stable_digest(value: Any, *, length: int = 24) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def semantic_profile_key(contract: Mapping[str, Any]) -> str:
    """Return the stable cache key for a JSON-compatible semantic contract."""

    normalized = json.loads(json.dumps(
        contract, sort_keys=True, default=str
    ))
    return _stable_digest(normalized, length=40)


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("a quantile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    index = fraction * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class DataShapeProfile:
    """Planner-visible sequence distribution, independent of dataset identity."""

    example_count: int
    minimum_tokens: int
    median_tokens: float
    p90_tokens: float
    maximum_tokens: int
    mean_tokens: float
    padding_fraction: float
    packing_opportunity: float

    @classmethod
    def fixed(cls, sequence_length: int) -> "DataShapeProfile":
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        return cls(
            example_count=0,
            minimum_tokens=sequence_length,
            median_tokens=float(sequence_length),
            p90_tokens=float(sequence_length),
            maximum_tokens=sequence_length,
            mean_tokens=float(sequence_length),
            padding_fraction=0.0,
            packing_opportunity=0.0,
        )

    @classmethod
    def from_lengths(
        cls,
        lengths: Iterable[int],
        *,
        sequence_length: int,
    ) -> "DataShapeProfile":
        values = [min(int(value), sequence_length) for value in lengths]
        if not values or any(value <= 0 for value in values):
            raise ValueError("token lengths must contain positive values")
        mean = statistics.fmean(values)
        padding = max(0.0, 1.0 - mean / sequence_length)
        # The upper bound assumes deterministic first-fit packing and is used
        # only to prioritize a measured data candidate, never as a speed claim.
        packing = max(0.0, min(1.0, padding - (1.0 / sequence_length)))
        return cls(
            example_count=len(values),
            minimum_tokens=min(values),
            median_tokens=_quantile(values, 0.5),
            p90_tokens=_quantile(values, 0.9),
            maximum_tokens=max(values),
            mean_tokens=mean,
            padding_fraction=padding,
            packing_opportunity=packing,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlannerContext:
    """State that invalidates measured profiles when execution changes."""

    compiler_hash: str
    runtime_hash: str
    source_hash: str
    hardware_identity: str = "blackhole-p150a-single"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class PlannerRequest:
    architecture: CanonicalArchitectureDescriptor
    safe_plan: ExecutionPlan
    hardware: HardwareCapabilities
    data: DataShapeProfile
    gradient_accumulation_steps: int = 1
    memory_mode: str = "auto"
    optimizer: str = "adamw"
    scheduler: str = "constant"
    # Canonical PEFT semantics which can change whether a native schedule is
    # executable.  Keep this name-free and JSON-shaped so it participates in
    # the measured-profile identity without coupling the planner to PEFT
    # classes.
    adapter_contract: tuple[tuple[str, Any], ...] = ()
    estimated_device_bytes: int | None = None
    context: PlannerContext = PlannerContext(
        compiler_hash="unknown",
        runtime_hash="unknown",
        source_hash="unknown",
    )

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.estimated_device_bytes is not None and self.estimated_device_bytes <= 0:
            raise ValueError("estimated_device_bytes must be positive")
        if self.safe_plan.architecture.semantic_fingerprint != (
            self.architecture.semantic_fingerprint
        ):
            raise ValueError("safe plan and architecture descriptor disagree")

    def semantic_key_document(self) -> dict[str, Any]:
        document = {
            "schema": _CACHE_SCHEMA,
            "hardware": {
                **asdict(self.hardware),
                "identity": self.context.hardware_identity,
            },
            "compiler_hash": self.context.compiler_hash,
            "runtime_hash": self.context.runtime_hash,
            "source_hash": self.context.source_hash,
            "architecture": self.architecture.semantic_dict(),
            "geometry": {
                "rank": self.safe_plan.rank,
                "batch": self.safe_plan.batch_size,
                "sequence": self.safe_plan.sequence_length,
                "gradient_accumulation": self.gradient_accumulation_steps,
                "dtype": self.architecture.dtype,
                "memory_mode": self.memory_mode,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "estimated_device_bytes": self.estimated_device_bytes,
            },
            "adapter_topology": [
                asdict(item) for item in self.safe_plan.lora_targets
            ],
            "adapter_contract": dict(self.adapter_contract),
            "data_shape": self.data.to_dict(),
        }
        # Normalize tuples and any other JSON-compatible containers so a
        # freshly computed request compares exactly with its persisted form.
        return json.loads(json.dumps(document, sort_keys=True, default=str))

    @property
    def profile_key(self) -> str:
        return semantic_profile_key(self.semantic_key_document())


@dataclass(frozen=True)
class ExecutionGeometryCandidate:
    candidate_id: str
    attention_query_block: int
    attention_kv_block: int
    attention_cores: int
    mlp_cores: int
    lora_cores: int
    moe_cores: int
    layer_group_size: int
    rank_tile_width: int
    optimizer_pack_bytes: int
    optimizer_packing_strategy: str
    graph_partition: str
    activation_policy: str
    frozen_weight_policy: str
    moe_schedule: str | None
    prefetch_depth: int
    double_buffered: bool
    sequence_bucket: int
    estimated_l1_peak_per_core: int
    estimated_dram_peak_bytes: int
    estimated_noc_bytes_per_step: int
    estimated_sync_points: int
    tile_utilization: float
    useful_token_fraction: float
    analytical_score: float
    admitted: bool
    rejection_reasons: tuple[str, ...]
    attention_strategy: str = "auto"
    peft_strategy: str = "framework_safe"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateMeasurement:
    candidate_id: str
    samples_ms: tuple[float, ...]
    region_ms: tuple[tuple[str, float], ...] = ()
    useful_tokens_per_step: float | None = None
    padded_tokens_per_step: float | None = None
    data_wait_ms: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.samples_ms or any(
            not math.isfinite(value) or value <= 0
            for value in self.samples_ms
        ):
            raise ValueError("measurements must be finite positive milliseconds")
        for name, value in (
            ("useful_tokens_per_step", self.useful_tokens_per_step),
            ("padded_tokens_per_step", self.padded_tokens_per_step),
        ):
            if value is not None and (
                not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.data_wait_ms and (
            len(self.data_wait_ms) != len(self.samples_ms)
            or any(value < 0 or not math.isfinite(value) for value in self.data_wait_ms)
        ):
            raise ValueError("data wait samples must align with step samples")

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def p10_ms(self) -> float:
        return _quantile(self.samples_ms, 0.1)

    @property
    def p90_ms(self) -> float:
        return _quantile(self.samples_ms, 0.9)

    @property
    def median_useful_tokens_per_second(self) -> float | None:
        if self.useful_tokens_per_step is None:
            return None
        return self.useful_tokens_per_step / (self.median_ms / 1000.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "sample_count": len(self.samples_ms),
            "median_ms": self.median_ms,
            "p10_ms": self.p10_ms,
            "p90_ms": self.p90_ms,
            "samples_ms": list(self.samples_ms),
            "region_ms": dict(self.region_ms),
            "useful_tokens_per_step": self.useful_tokens_per_step,
            "padded_tokens_per_step": self.padded_tokens_per_step,
            "median_useful_tokens_per_second": (
                self.median_useful_tokens_per_second
            ),
            "data_wait_ms": list(self.data_wait_ms),
        }


@dataclass(frozen=True)
class AutotuneProfile:
    profile_key: str
    selection_status: str
    cache_status: str
    cache_reason: str
    selected: ExecutionGeometryCandidate
    candidates_generated: int
    candidates_admitted: int
    candidates_profiled: int
    candidate_audit: tuple[ExecutionGeometryCandidate, ...]
    measurements: tuple[CandidateMeasurement, ...]
    safe_fallback: dict[str, Any]
    request_contract: dict[str, Any]

    @property
    def measured(self) -> bool:
        return self.selection_status in {"measured", "cache_hit_measured"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_key": self.profile_key,
            "selection_status": self.selection_status,
            "cache_status": self.cache_status,
            "cache_reason": self.cache_reason,
            "selected": self.selected.to_dict(),
            "candidates_generated": self.candidates_generated,
            "candidates_admitted": self.candidates_admitted,
            "candidates_profiled": self.candidates_profiled,
            "candidate_audit": [item.to_dict() for item in self.candidate_audit],
            "measurements": [item.to_dict() for item in self.measurements],
            "safe_fallback": self.safe_fallback,
            "request_contract": self.request_contract,
        }


def _candidate_signature(values: Mapping[str, Any]) -> str:
    return "g-" + _stable_digest(dict(values), length=16)


def _peft_topology_rejections(
    strategy: str,
    target_roles: set[str],
) -> tuple[str, ...]:
    """Reject native owners that cannot bind the requested adapter graph."""

    reasons: list[str] = []
    has_mlp_group = {"gate", "up", "down"}.issubset(target_roles)
    has_kv_group = {"k", "v"}.issubset(target_roles)
    has_attention_group = {"q", "k", "v", "o"}.issubset(target_roles)
    if strategy == "native_grouped" and not (
        has_mlp_group or has_kv_group
    ):
        reasons.append(
            "native grouped PEFT requires gate/up/down or K/V targets"
        )
    if strategy in {
        "native_resident_mlp",
        "native_resident_mlp_attention",
    } and not has_mlp_group:
        reasons.append("resident MLP PEFT requires gate/up/down targets")
    if strategy == "native_resident_mlp_attention" and (
        not has_attention_group
    ):
        reasons.append("resident attention PEFT requires Q/K/V/O targets")
    return tuple(reasons)


def _largest_core_multiple(limit: int, parallelism: int) -> int:
    parallelism = max(1, min(parallelism, limit))
    value = limit - (limit % parallelism)
    return max(parallelism, value)


def _candidate_knobs(request: PlannerRequest) -> list[dict[str, Any]]:
    """Return a bounded, non-Cartesian set of meaningful schedule variants."""

    plan = request.safe_plan
    descriptor = request.architecture
    sequence = plan.sequence_length
    valid_blocks = tuple(
        value for value in (32, 64, 128, 256, 512)
        if value <= max(32, _ceil_to(sequence, 32))
    ) or (32,)
    default_block = min(
        128,
        max(value for value in valid_blocks if value <= max(32, sequence)),
    )
    attention_cores = _largest_core_multiple(
        request.hardware.worker_cores, descriptor.attention_heads
    )
    kv_parallelism = max(1, descriptor.kv_heads)
    moe_cores = (
        _largest_core_multiple(request.hardware.worker_cores, kv_parallelism)
        if descriptor.moe is not None else 0
    )
    # Start from the execution contract that the current worker can actually
    # express.  Optimizer packing and graph partitioning are runtime ABI
    # controls today; resident activations, weight prefetch and layer grouping
    # are future candidates and must not leak into otherwise executable
    # variants.  Keeping those two classes separate is important: an
    # analytically attractive but unwired residency knob must never cause the
    # real graph/optimizer alternatives to be pruned with it.
    runtime_base = {
        "attention_strategy": "auto",
        "peft_strategy": "framework_safe",
        "attention_query_block": default_block,
        "attention_kv_block": default_block,
        "attention_cores": attention_cores,
        "mlp_cores": request.hardware.worker_cores,
        "lora_cores": request.hardware.worker_cores,
        "moe_cores": moe_cores,
        "layer_group_size": 1,
        "rank_tile_width": _ceil_to(plan.rank, TILE_SIZE),
        "optimizer_pack_bytes": 1024 * 1024,
        "optimizer_packing_strategy": "contiguous",
        "graph_partition": _default_graph_partition(request),
        "activation_policy": "framework_safe",
        "frozen_weight_policy": "framework_managed",
        "moe_schedule": plan.moe_execution_schedule,
        "prefetch_depth": 0,
        "double_buffered": False,
        "sequence_bucket": _ceil_to(sequence, 32),
    }
    variants = [runtime_base]

    def vary(**updates: Any) -> None:
        variants.append({**runtime_base, **updates})

    def explore(source: Mapping[str, Any], **updates: Any) -> None:
        variants.append({**source, **updates})

    # These variations lower to distinct, already implemented worker ABIs.
    # They therefore remain independently measurable even while the more
    # ambitious residency candidates below are still rejected.
    for pack in (256 * 1024, 4 * 1024 * 1024):
        vary(optimizer_pack_bytes=pack)
    vary(optimizer_packing_strategy="shape_grouped")
    partitions = ["monolithic", "phase_split_auto", "phase_split_forward"]
    if request.memory_mode in {
        "bf8_frozen",
        "bf4_frozen",
        "hybrid_frozen",
    }:
        # Reduced frozen weights have an explicit deterministic (not
        # BF16-bit-exact) numerical contract.  Keeping forward and backward
        # in one lazy graph prevents a phase marker from exporting every
        # autograd-saved reduced-weight view.  Backward, clipping and AdamW
        # retain their established device barriers.
        partitions.append("forward_backward_fused")
    for partition in partitions:
        if partition != runtime_base["graph_partition"]:
            vary(graph_partition=partition)

    optimized = {
        **runtime_base,
        "activation_policy": (
            "recompute" if sequence > 128 else "save_bounded"
        ),
        "frozen_weight_policy": (
            "persistent_reduced"
            if request.memory_mode in {
                "bf8_frozen", "bf4_frozen", "hybrid_frozen"
            }
            else "stream_bf16"
        ),
        "prefetch_depth": 1,
        "double_buffered": True,
    }
    variants.append(optimized)

    if sequence > 128 and descriptor.qkv_projection_layout == "separate":
        vary(attention_strategy="safe_streamed")
        for block in valid_blocks:
            if block != default_block:
                vary(
                    attention_strategy="safe_streamed",
                    attention_query_block=block,
                )
        if plan.selection("attention").implementation == "native_streamed_causal":
            vary(attention_strategy="native_streamed")
    # These are real public-worker execution paths.  They are deliberately
    # macro candidates: the tuner first asks whether grouped native LoRA or a
    # larger resident MLP/attention ownership boundary is beneficial, then
    # tunes smaller packing choices only inside a promising region.
    vary(peft_strategy="native_grouped")
    vary(peft_strategy="native_resident_mlp")
    vary(peft_strategy="native_resident_mlp_attention")
    for group in (2, 4, 8):
        if group <= descriptor.layer_count:
            explore(optimized, layer_group_size=group)
    for prefetch in (0, 2):
        explore(
            optimized,
            prefetch_depth=prefetch,
            double_buffered=prefetch > 0,
        )
    for policy in ("save_bounded", "recompute"):
        if policy != optimized["activation_policy"]:
            explore(optimized, activation_policy=policy)
    if plan.rank > TILE_SIZE:
        for width in (TILE_SIZE, 2 * TILE_SIZE, 4 * TILE_SIZE):
            if width != optimized["rank_tile_width"]:
                explore(optimized, rank_tile_width=width)
    if descriptor.moe is not None:
        # These are distinct execution candidates, not model-family routes.
        # Unsupported/dynamic schedules are retained in the audit and pruned
        # with an explicit reason until a native implementation is qualified.
        for schedule in (
            "assignment_weight_gather",
            "expert_capacity_bucketed",
            "dynamic_native_expert_bucketed",
        ):
            if schedule != runtime_base["moe_schedule"]:
                vary(moe_schedule=schedule)
    if request.data.packing_opportunity >= 0.1:
        bucket = max(
            32,
            min(sequence, _ceil_to(int(request.data.p90_tokens), 32)),
        )
        explore(optimized, sequence_bucket=bucket)
    # Two curated macro candidates explore the interactions that matter most
    # without forming the Cartesian product of every knob.
    explore(
        optimized,
        layer_group_size=min(4, descriptor.layer_count),
        prefetch_depth=2,
        optimizer_pack_bytes=4 * 1024 * 1024,
        activation_policy="recompute",
    )
    explore(
        optimized,
        attention_query_block=min(valid_blocks),
        attention_kv_block=default_block,
        layer_group_size=min(2, descriptor.layer_count),
        prefetch_depth=1,
    )

    unique: dict[str, dict[str, Any]] = {}
    for item in variants:
        unique[_candidate_signature(item)] = item
    return list(unique.values())


def _large_reduced_boundary_requires_fusion(
    request: PlannerRequest,
) -> bool:
    """Return whether exporting the forward boundary is not capacity-safe.

    Reduced frozen-weight execution retains a BF16 source representation and
    compiled block-float representations. A forward phase boundary also
    exports autograd-saved reduced-weight views. On a large model this both
    recreates the old pathological execution boundary and leaves too little
    contiguous DRAM for later lazy-graph allocations, even when a coarse
    aggregate byte estimate is below physical capacity.

    The threshold is the same geometry limit used by the admitted monolithic
    graph. The decision intentionally depends only on precision, semantic
    graph shape and estimated memory -- never on checkpoint identity.
    """

    plan = request.safe_plan
    return bool(
        request.memory_mode in {
            "bf8_frozen", "bf4_frozen", "hybrid_frozen",
        }
        and plan.sequence_length <= 128
        and request.architecture.moe is None
        and max(
            int(plan.estimated_device_bytes),
            int(request.estimated_device_bytes or 0),
        ) > 8 * 1024**3
    )


def _default_graph_partition(request: PlannerRequest) -> str:
    """Choose the safe static graph boundary from executable geometry."""

    plan = request.safe_plan
    if (
        plan.estimated_device_bytes <= 8 * 1024**3
        and plan.sequence_length <= 128
        and request.architecture.moe is None
    ):
        return "monolithic"
    if _large_reduced_boundary_requires_fusion(request):
        return "forward_backward_fused"
    return "phase_split_auto"


def _evaluate_candidate(
    request: PlannerRequest,
    knobs: Mapping[str, Any],
) -> ExecutionGeometryCandidate:
    descriptor = request.architecture
    plan = request.safe_plan
    hardware = request.hardware
    dtype_bytes = 2 if descriptor.dtype in {"bfloat16", "float16"} else 4
    q_block = int(knobs["attention_query_block"])
    kv_block = int(knobs["attention_kv_block"])
    rank_width = int(knobs["rank_tile_width"])
    layer_group = int(knobs["layer_group_size"])
    hidden_physical = _ceil_to(descriptor.hidden_size, TILE_SIZE)
    intermediate_physical = _ceil_to(descriptor.intermediate_size, TILE_SIZE)
    token_physical = _ceil_to(
        plan.batch_size * int(knobs["sequence_bucket"]), TILE_SIZE
    )
    rank_physical = _ceil_to(plan.rank, TILE_SIZE)

    token_util = plan.logical_tokens / max(1, token_physical)
    hidden_util = descriptor.hidden_size / hidden_physical
    intermediate_util = descriptor.intermediate_size / intermediate_physical
    rank_util = plan.rank / rank_physical
    tile_utilization = max(
        1e-6,
        statistics.fmean((token_util, hidden_util, intermediate_util, rank_util)),
    )
    useful_token_fraction = max(
        1e-6,
        min(1.0, 1.0 - request.data.padding_fraction),
    )

    attention_l1 = (
        (q_block + 2 * kv_block)
        * _ceil_to(descriptor.head_dimension, TILE_SIZE)
        * dtype_bytes
        * 2
        + q_block * kv_block * 4
    )
    mlp_l1 = (
        _ceil_to(max(32, token_physical // max(1, int(knobs["mlp_cores"]))), 32)
        * (hidden_physical + 2 * intermediate_physical)
        * dtype_bytes
        * (2 if bool(knobs["double_buffered"]) else 1)
    )
    lora_l1 = (
        (hidden_physical + intermediate_physical)
        * max(TILE_SIZE, rank_width)
        * dtype_bytes
        // max(1, int(knobs["lora_cores"]))
        * 4
    )
    optimizer_l1 = (
        int(knobs["optimizer_pack_bytes"])
        * 4  # parameter, gradient, first and second moment working streams
        // max(1, hardware.worker_cores)
    )
    moe_l1 = 0
    if descriptor.moe is not None:
        expert_i = _ceil_to(
            descriptor.moe.expert_intermediate_size, TILE_SIZE
        )
        moe_l1 = (
            (hidden_physical + 2 * expert_i)
            * TILE_SIZE * dtype_bytes
            * (2 if bool(knobs["double_buffered"]) else 1)
        )
    if knobs["activation_policy"] == "framework_safe":
        # TTNN owns bounded operator-local scratch for the accepted fallback;
        # it does not reserve a resident MLP/attention working set.
        attention_l1 = min(attention_l1, 256 * 1024)
        mlp_l1 = min(mlp_l1, 256 * 1024)
        lora_l1 = min(lora_l1, 192 * 1024)
        moe_l1 = min(moe_l1, 256 * 1024)
    runtime_margin = 128 * 1024
    l1_peak = max(attention_l1, mlp_l1, lora_l1, optimizer_l1, moe_l1)
    l1_peak += runtime_margin

    checkpoint_bytes = (
        plan.logical_tokens * descriptor.hidden_size * dtype_bytes
        * (layer_group if knobs["activation_policy"] == "save_bounded" else 1)
    )
    prefetch_bytes = int(knobs["prefetch_depth"]) * (
        descriptor.hidden_size * descriptor.intermediate_size * dtype_bytes
    )
    dram_peak = (
        request.estimated_device_bytes
        if request.estimated_device_bytes is not None
        else plan.estimated_device_bytes
    ) + checkpoint_bytes + prefetch_bytes
    if (
        descriptor.moe is not None
        and knobs["moe_schedule"] == "dynamic_native_expert_bucketed"
    ):
        # The current single-card sparse_matmul ABI materializes only the
        # active layer's block-sparse expert-major activations.  Account for
        # that real transient instead of pretending the TTML variable-row ABI
        # is already available through PJRT.
        dram_peak += (
            plan.logical_tokens
            * descriptor.moe.expert_count
            * (
                3 * descriptor.moe.expert_intermediate_size
                + descriptor.hidden_size
                + 2 * rank_physical
            )
            * dtype_bytes
        )
    noc_bytes = descriptor.layer_count * plan.logical_tokens * dtype_bytes * (
        descriptor.hidden_size * 4 + descriptor.intermediate_size * 2
    )
    noc_bytes //= max(1, layer_group)
    if descriptor.moe is not None:
        noc_bytes += plan.moe_selected_activation_bytes * descriptor.layer_count

    attention_blocks = math.ceil(plan.sequence_length / q_block)
    sync_points = math.ceil(descriptor.layer_count / layer_group) * (
        3 + 2 * attention_blocks
    )
    if int(knobs["prefetch_depth"]) == 0:
        sync_points += descriptor.layer_count

    reasons: list[str] = []
    if l1_peak > hardware.usable_l1_bytes_per_core:
        reasons.append(
            f"estimated per-core L1 {l1_peak} exceeds guarded budget "
            f"{hardware.usable_l1_bytes_per_core}"
        )
    if dram_peak > hardware.dram_bytes:
        reasons.append(
            f"estimated DRAM peak {dram_peak} exceeds {hardware.dram_bytes}"
        )
    if q_block <= 0 or kv_block <= 0 or q_block % TILE_SIZE or kv_block % TILE_SIZE:
        reasons.append("attention blocks must be positive 32-token multiples")
    if rank_width < TILE_SIZE or rank_width % TILE_SIZE:
        reasons.append("rank tile width must be a positive tile multiple")
    default_attention_block = min(
        128, _ceil_to(plan.sequence_length, TILE_SIZE)
    )
    streamed_query_override = bool(
        plan.sequence_length > 128
        and descriptor.qkv_projection_layout == "separate"
        and knobs["attention_strategy"] == "safe_streamed"
        and q_block <= plan.sequence_length
    )
    if q_block != default_attention_block and not streamed_query_override:
        reasons.append(
            "attention-block override is not wired to the current runtime"
        )
    if kv_block != default_attention_block:
        reasons.append(
            "KV-block override is audit-only in the current streamed runtime"
        )
    if knobs["attention_strategy"] not in {
        "auto", "safe_streamed", "native_streamed"
    }:
        reasons.append("unsupported attention execution strategy")
    if knobs["attention_strategy"] in {"safe_streamed", "native_streamed"} and not (
        plan.sequence_length > 128
        and descriptor.qkv_projection_layout == "separate"
    ):
        reasons.append(
            "streamed-attention override requires long separate-QKV semantics"
        )
    if knobs["attention_strategy"] == "native_streamed" and (
        plan.selection("attention").implementation != "native_streamed_causal"
    ):
        reasons.append("native streamed attention is outside the safe plan")
    if knobs["attention_strategy"] == "native_streamed" and (
        q_block != default_attention_block
    ):
        reasons.append("native streamed attention has a fixed query schedule")
    if knobs["peft_strategy"] not in {
        "framework_safe",
        "native_grouped",
        "native_resident_mlp",
        "native_resident_mlp_attention",
    }:
        reasons.append("unsupported PEFT execution strategy")
    if knobs["peft_strategy"] != "framework_safe":
        adapter = dict(request.adapter_contract)
        target_roles = {item.role for item in plan.lora_targets}
        if not (
            adapter.get("method", "lora") == "lora"
            and not adapter.get("use_rslora", False)
            and not adapter.get("use_dora", False)
            and float(adapter.get("dropout", 0.0)) == 0.0
            and adapter.get("bias", "none") == "none"
        ):
            reasons.append(
                "native PEFT requires ordinary zero-dropout, bias-free LoRA"
            )
        if descriptor.moe is not None:
            reasons.append("native grouped PEFT is not wired for routed experts")
        if plan.rank > 128:
            reasons.append("native grouped PEFT supports logical rank <=128")
        reasons.extend(_peft_topology_rejections(
            knobs["peft_strategy"], target_roles
        ))
        if knobs["peft_strategy"] == "native_resident_mlp_attention" and (
            descriptor.qkv_projection_layout != "separate"
        ):
            reasons.append(
                "resident attention PEFT requires separate Q/K/V projections"
            )
    if layer_group != 1:
        reasons.append("layer-group override is not wired to the current runtime")
    if knobs["activation_policy"] != "framework_safe":
        reasons.append("activation-policy override is not wired to the runtime")
    if knobs["frozen_weight_policy"] != "framework_managed":
        reasons.append("frozen-weight residency override is not wired to the runtime")
    if int(knobs["prefetch_depth"]) != 0:
        reasons.append("prefetch-depth override is not wired to the runtime")
    if bool(knobs["double_buffered"]):
        reasons.append("double-buffering override is not wired to the runtime")
    if int(knobs["sequence_bucket"]) != _ceil_to(
        plan.sequence_length, TILE_SIZE
    ):
        reasons.append("sequence-bucket override is not wired to the runtime")
    if rank_width != _ceil_to(plan.rank, TILE_SIZE):
        reasons.append("rank-tile override is not wired to the current runtime")
    if knobs["optimizer_packing_strategy"] not in {"contiguous", "shape_grouped"}:
        reasons.append("unsupported optimizer packing strategy")
    if knobs["graph_partition"] not in {
        "monolithic",
        "forward_backward_fused",
        "phase_split_auto",
        "phase_split_forward",
    }:
        reasons.append("unsupported graph partition")
    if knobs["graph_partition"] == "monolithic" and not (
        plan.estimated_device_bytes <= 8 * 1024**3
        and plan.sequence_length <= 128
        and descriptor.moe is None
    ):
        reasons.append("monolithic graph exceeds its admitted geometry contract")
    if knobs["graph_partition"] == "forward_backward_fused" and not (
        request.memory_mode in {
            "bf8_frozen",
            "bf4_frozen",
            "hybrid_frozen",
        }
        and plan.sequence_length <= 128
        and descriptor.moe is None
    ):
        reasons.append(
            "forward/backward fusion requires an admitted short-sequence "
            "dense reduced-weight contract"
        )
    if (
        knobs["graph_partition"] in {
            "phase_split_auto", "phase_split_forward",
        }
        and _large_reduced_boundary_requires_fusion(request)
    ):
        reasons.append(
            "a large reduced-weight graph must keep forward and backward "
            "together; exporting the autograd-saved reduced-weight boundary "
            "is not fragmentation-safe on one P150a"
        )
    if knobs["moe_schedule"] == "expert_capacity_bucketed":
        if descriptor.moe is None:
            reasons.append("expert-bucketed schedule requires MoE semantics")
        elif not (
            descriptor.moe.expert_capacity_factor is not None
            and descriptor.moe.overflow_policy in {"drop", "residual"}
        ):
            reasons.append(
                "static expert buckets require an explicit capacity/drop contract"
            )
    if knobs["moe_schedule"] == "dynamic_native_expert_bucketed":
        if descriptor.moe is None:
            reasons.append("dynamic expert scheduling requires MoE semantics")
        else:
            if descriptor.moe.expert_capacity_factor is not None:
                reasons.append(
                    "dynamic no-capacity scheduling cannot replace an explicit "
                    "capacity/drop contract"
                )
            if plan.logical_tokens % TILE_SIZE:
                reasons.append("dynamic experts require a /32 token geometry")
            if any(
                value % TILE_SIZE
                for value in (
                    descriptor.hidden_size,
                    descriptor.moe.expert_intermediate_size,
                )
            ):
                reasons.append(
                    "dynamic expert sparse matmuls require tile-aligned "
                    "hidden/intermediate dimensions"
                )
    elif knobs["moe_schedule"] not in {
        None,
        "assignment_weight_gather",
        "expert_capacity_bucketed",
    }:
        reasons.append("unsupported MoE execution schedule")
    if knobs["moe_schedule"] == "assignment_weight_gather":
        if plan.moe_routed_scratch_bytes > hardware.dram_bytes // 4:
            reasons.append(
                "per-assignment expert weight replication exceeds guarded scratch"
            )

    # Unitless analytical ranking: bytes and barriers are normalized so no
    # unmeasured candidate is presented as a millisecond prediction.
    compute_term = (
        descriptor.layer_count
        * plan.logical_tokens
        * (descriptor.hidden_size + descriptor.intermediate_size)
        / tile_utilization
    )
    movement_term = (dram_peak + noc_bytes) / (1024 * 1024)
    sync_term = sync_points * 100_000
    padding_term = compute_term * (1.0 / useful_token_fraction - 1.0)
    score = compute_term + movement_term * 10_000 + sync_term + padding_term
    if knobs["activation_policy"] == "framework_safe":
        score *= 1.5
    if reasons:
        score = float("inf")
    candidate_id = _candidate_signature(knobs)
    return ExecutionGeometryCandidate(
        candidate_id=candidate_id,
        **dict(knobs),
        estimated_l1_peak_per_core=l1_peak,
        estimated_dram_peak_bytes=dram_peak,
        estimated_noc_bytes_per_step=noc_bytes,
        estimated_sync_points=sync_points,
        tile_utilization=tile_utilization,
        useful_token_fraction=useful_token_fraction,
        analytical_score=score,
        admitted=not reasons,
        rejection_reasons=tuple(reasons),
    )


def generate_candidates(
    request: PlannerRequest,
) -> tuple[ExecutionGeometryCandidate, ...]:
    candidates = tuple(
        _evaluate_candidate(request, knobs)
        for knobs in _candidate_knobs(request)
    )
    if not any(item.admitted for item in candidates):
        # The architecture safe plan is still authoritative, but autotuner must not
        # invent an optimized plan when every generated geometry was pruned.
        details = "; ".join(
            f"{item.candidate_id}: {', '.join(item.rejection_reasons)}"
            for item in candidates
        )
        raise ValueError(f"no execution candidate admitted: {details}")
    return candidates


def candidate_worker_arguments(
    candidate: ExecutionGeometryCandidate,
) -> tuple[str, ...]:
    """Translate an admitted candidate into the existing worker ABI."""

    if not candidate.admitted:
        raise ValueError(
            f"candidate {candidate.candidate_id} is not admitted: "
            + "; ".join(candidate.rejection_reasons)
        )
    arguments = [
        "--optimizer-tile-packed",
        "--optimizer-pack-bytes", str(candidate.optimizer_pack_bytes),
        "--optimizer-packing-strategy", candidate.optimizer_packing_strategy,
        "--attention-query-block", str(candidate.attention_query_block),
        "--attention-backend", (
            "safe" if candidate.attention_strategy == "safe_streamed" else
            "native" if candidate.attention_strategy == "native_streamed" else
            "auto"
        ),
        "--peft-execution-strategy", candidate.peft_strategy,
        "--moe-execution-strategy", (
            "dynamic"
            if candidate.moe_schedule == "dynamic_native_expert_bucketed"
            else "sparse"
            if candidate.moe_schedule == "expert_capacity_bucketed"
            else "safe"
            if candidate.moe_schedule == "assignment_weight_gather"
            else "auto"
        ),
    ]
    if candidate.graph_partition == "monolithic":
        arguments.append("--fuse-step")
    elif candidate.graph_partition == "forward_backward_fused":
        arguments.extend([
            "--split-clip-optimizer",
            "--split-optimizer-phases",
        ])
    else:
        arguments.extend([
            "--split-clip-optimizer",
            "--split-after-loss",
            "--split-optimizer-phases",
        ])
        arguments.append(
            "--auto-split-after-forward"
            if candidate.graph_partition == "phase_split_auto"
            else "--split-after-forward"
        )
    return tuple(arguments)


class ExecutionProfileCache:
    """Atomic inventory of measured winners keyed by semantic compatibility."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _read(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"schema_version": _CACHE_SCHEMA, "entries": {}}
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid execution profile cache: {self.path}") from error
        if value.get("schema_version") != _CACHE_SCHEMA:
            raise ValueError(
                f"unsupported execution profile cache schema in {self.path}"
            )
        if not isinstance(value.get("entries"), dict):
            raise ValueError("profile cache entries must be an object")
        return value

    def lookup(
        self, request: PlannerRequest
    ) -> tuple[dict[str, Any] | None, str]:
        document = self._read()
        entry = document["entries"].get(request.profile_key)
        if entry is not None:
            return entry, "exact semantic/runtime key matched"
        request_doc = request.semantic_key_document()
        architecture = request_doc["architecture"]
        geometry = request_doc["geometry"]
        for candidate in document["entries"].values():
            contract = candidate.get("request_contract", {})
            if contract.get("architecture") != architecture:
                continue
            if contract.get("geometry") != geometry:
                return None, "architecture matched but training geometry changed"
            for field in ("compiler_hash", "runtime_hash", "source_hash"):
                if contract.get(field) != request_doc.get(field):
                    return None, f"{field} changed"
            return None, "data shape or hardware contract changed"
        return None, "no compatible measured semantic profile"

    def lookup_contract(
        self, contract: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any] | None]:
        key = semantic_profile_key(contract)
        return key, self._read()["entries"].get(key)

    def lookup_contract_explain(
        self, contract: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any] | None, str]:
        """Lookup an already serialized contract with an exact miss reason."""

        request = json.loads(json.dumps(
            contract, sort_keys=True, default=str
        ))
        key = semantic_profile_key(request)
        document = self._read()
        entry = document["entries"].get(key)
        if entry is not None:
            return key, entry, "exact semantic/runtime key matched"
        architecture = request.get("architecture")
        geometry = request.get("geometry")
        architecture_matches = [
            candidate.get("request_contract") or {}
            for candidate in document["entries"].values()
            if (candidate.get("request_contract") or {}).get("architecture")
            == architecture
        ]
        if not architecture_matches:
            return key, None, "no compatible measured semantic profile"
        geometry_matches = [
            cached for cached in architecture_matches
            if cached.get("geometry") == geometry
        ]
        if not geometry_matches:
            return key, None, (
                "architecture matched but training geometry changed"
            )
        for field in ("compiler_hash", "runtime_hash", "source_hash"):
            if not any(
                cached.get(field) == request.get(field)
                for cached in geometry_matches
            ):
                return key, None, f"{field} changed"
            geometry_matches = [
                cached for cached in geometry_matches
                if cached.get(field) == request.get(field)
            ]
        if not any(
            cached.get("hardware") == request.get("hardware")
            for cached in geometry_matches
        ):
            return key, None, "hardware contract changed"
        hardware_matches = [
            cached for cached in geometry_matches
            if cached.get("hardware") == request.get("hardware")
        ]
        if not any(
            cached.get("data_shape") == request.get("data_shape")
            for cached in hardware_matches
        ):
            return key, None, "data shape changed"
        return key, None, "adapter or execution semantics changed"

    def store(self, profile: AutotuneProfile) -> None:
        if not profile.measured:
            raise ValueError("only measured execution winners may enter the cache")
        document = self._read()
        document["entries"][profile.profile_key] = {
            **profile.to_dict(),
            "stored_unix_seconds": time.time(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{time.time_ns()}.tmp"
        )
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def store_document(
        self,
        *,
        profile_key: str,
        profile: Mapping[str, Any],
    ) -> None:
        if profile.get("selection_status") != "measured":
            raise ValueError("only measured execution winners may enter the cache")
        if semantic_profile_key(profile.get("request_contract", {})) != profile_key:
            raise ValueError("profile key does not match its request contract")
        document = self._read()
        document["entries"][profile_key] = {
            **dict(profile),
            "stored_unix_seconds": time.time(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{time.time_ns()}.tmp"
        )
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def inventory(self) -> dict[str, Any]:
        return self._read()

    def describe(self) -> dict[str, Any]:
        """Return a stable user-facing inventory without internal payloads."""

        document = self._read()
        entries = []
        for key, value in sorted(document.get("entries", {}).items()):
            contract = value.get("request_contract") or {}
            geometry = contract.get("geometry") or {}
            architecture = contract.get("architecture") or {}
            selected = value.get("selected") or {}
            measurements = value.get("measurements") or []
            winner = next((
                item for item in measurements
                if item.get("candidate_id") == selected.get("candidate_id")
            ), None)
            entries.append({
                "profile_key": key,
                "stored_unix_seconds": value.get("stored_unix_seconds"),
                "architecture_fingerprint": architecture.get(
                    "semantic_fingerprint"
                ),
                "geometry": geometry,
                "selected_candidate_id": selected.get("candidate_id"),
                "graph_partition": selected.get("graph_partition"),
                "median_ms": (winner or {}).get("median_ms"),
                "estimated_compile_seconds": (winner or {}).get(
                    "estimated_compile_seconds"
                ),
                "compiler_hash": contract.get("compiler_hash"),
                "runtime_hash": contract.get("runtime_hash"),
                "source_hash": contract.get("source_hash"),
            })
        return {
            "schema_version": document.get("schema_version"),
            "cache_path": str(self.path),
            "profile_count": len(entries),
            "entries": entries,
        }


RegionProfiler = Callable[
    [ExecutionGeometryCandidate], Mapping[str, float]
]
FullStepProfiler = Callable[
    [ExecutionGeometryCandidate], CandidateMeasurement
]


def _candidate_from_dict(value: Mapping[str, Any]) -> ExecutionGeometryCandidate:
    fields = ExecutionGeometryCandidate.__dataclass_fields__
    return ExecutionGeometryCandidate(**{
        name: (
            tuple(value.get(name, ())) if name == "rejection_reasons" else
            value.get(name, field.default)
        )
        for name, field in fields.items()
    })


def execution_candidate_from_dict(
    value: Mapping[str, Any],
) -> ExecutionGeometryCandidate:
    """Rehydrate a public plan candidate for runtime/profile execution."""

    return _candidate_from_dict(value)


def plan_autotuned_execution(
    request: PlannerRequest,
    *,
    cache: ExecutionProfileCache | None = None,
    region_profiler: RegionProfiler | None = None,
    full_step_profiler: FullStepProfiler | None = None,
    maximum_full_step_candidates: int = 4,
) -> AutotuneProfile:
    """Generate, prune and optionally measure a bounded candidate set.

    With no profiler callbacks this is a transparent analytical plan, not an
    autotuning claim.  A measured cache entry or full-step profiler is required
    for ``selection_status`` to say measured.
    """

    candidates = generate_candidates(request)
    generated_admitted = tuple(item for item in candidates if item.admitted)
    admitted = sorted(
        (item for item in candidates if item.admitted),
        key=lambda item: item.analytical_score,
    )
    static = generated_admitted[0]
    cache_reason = "cache disabled"
    if cache is not None:
        cached, cache_reason = cache.lookup(request)
        if cached is not None:
            selected = _candidate_from_dict(cached["selected"])
            current = {item.candidate_id: item for item in admitted}
            if selected.candidate_id in current:
                measurements = tuple(
                    CandidateMeasurement(
                        candidate_id=item["candidate_id"],
                        samples_ms=tuple(item["samples_ms"]),
                        region_ms=tuple(sorted(
                            (item.get("region_ms") or {}).items()
                        )),
                        useful_tokens_per_step=item.get(
                            "useful_tokens_per_step"
                        ),
                        padded_tokens_per_step=item.get(
                            "padded_tokens_per_step"
                        ),
                        data_wait_ms=tuple(item.get("data_wait_ms") or ()),
                    )
                    for item in cached.get("measurements", [])
                )
                return AutotuneProfile(
                    profile_key=request.profile_key,
                    selection_status="cache_hit_measured",
                    cache_status="hit",
                    cache_reason=cache_reason,
                    selected=current[selected.candidate_id],
                    candidates_generated=len(candidates),
                    candidates_admitted=len(admitted),
                    candidates_profiled=len(measurements),
                    candidate_audit=candidates,
                    measurements=measurements,
                    safe_fallback=request.safe_plan.to_dict(),
                    request_contract=request.semantic_key_document(),
                )
            cache_reason = "cached candidate is absent from the current candidate set"

    from .autotune_quality import adaptive_candidate_portfolio

    finalists = list(adaptive_candidate_portfolio(
        admitted,
        static_candidate_id=static.candidate_id,
        budget=maximum_full_step_candidates,
    ))
    region_values: dict[str, tuple[tuple[str, float], ...]] = {}
    if region_profiler is not None:
        ranked = []
        for candidate in admitted:
            values = dict(region_profiler(candidate))
            if not values or any(
                not math.isfinite(float(value)) or float(value) < 0
                for value in values.values()
            ):
                raise ValueError("region profiler returned invalid timing data")
            region_values[candidate.candidate_id] = tuple(sorted(
                (name, float(value)) for name, value in values.items()
            ))
            ranked.append((
                sum(values.values()),
                candidate.analytical_score,
                candidate.candidate_id,
                candidate,
            ))
        region_ranked = [item[3] for item in sorted(ranked)]
        # Region timings prune ties but cannot displace the static authority or
        # collapse the final budget into one nearby macro schedule.
        finalists = list(adaptive_candidate_portfolio(
            region_ranked,
            static_candidate_id=static.candidate_id,
            budget=maximum_full_step_candidates,
        ))

    measurements: list[CandidateMeasurement] = []
    if full_step_profiler is not None:
        for candidate in finalists:
            measured = full_step_profiler(candidate)
            if measured.candidate_id != candidate.candidate_id:
                raise ValueError("full-step profiler returned the wrong candidate id")
            if not measured.region_ms and candidate.candidate_id in region_values:
                measured = replace(
                    measured, region_ms=region_values[candidate.candidate_id]
                )
            measurements.append(measured)
        by_id = {item.candidate_id: item for item in admitted}
        from .autotune_quality import select_non_regressing_winner

        winner_document, _selection_safety = select_non_regressing_winner(
            [item.to_dict() for item in measurements],
            static_candidate_id=static.candidate_id,
        )
        winner = next(
            item for item in measurements
            if item.candidate_id == winner_document["candidate_id"]
        )
        selected = by_id[winner.candidate_id]
        status = "measured"
        cache_status = "miss_measured" if cache is not None else "disabled"
    else:
        selected = admitted[0]
        status = "analytical_unmeasured"
        cache_status = "miss" if cache is not None else "disabled"

    profile = AutotuneProfile(
        profile_key=request.profile_key,
        selection_status=status,
        cache_status=cache_status,
        cache_reason=cache_reason,
        selected=selected,
        candidates_generated=len(candidates),
        candidates_admitted=len(admitted),
        candidates_profiled=len(measurements),
        candidate_audit=candidates,
        measurements=tuple(measurements),
        safe_fallback=request.safe_plan.to_dict(),
        request_contract=request.semantic_key_document(),
    )
    if cache is not None and profile.measured:
        cache.store(profile)
    return profile


__all__ = [
    "CandidateMeasurement",
    "DataShapeProfile",
    "ExecutionGeometryCandidate",
    "AutotuneProfile",
    "PlannerContext",
    "PlannerRequest",
    "ExecutionProfileCache",
    "generate_candidates",
    "execution_candidate_from_dict",
    "candidate_worker_arguments",
    "plan_autotuned_execution",
    "semantic_profile_key",
]
