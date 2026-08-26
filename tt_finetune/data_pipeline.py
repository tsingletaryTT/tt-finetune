"""Deterministic host-side data planning for Blackhole training.

The module deliberately has no Torch or tokenizer dependency.  It owns the
parts of the input pipeline that must be reproducible across workers and
resume: epoch shuffling, length bucketing, batch order, and data-contract
identity.  The JSONL dataset adapter consumes these plans when it constructs a
Torch DataLoader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Any, Iterable, Mapping, Sequence


DATA_PLAN_VERSION = "tt-finetune-data-plan-v1"


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_sequence_buckets(
    max_length: int,
    configured: Sequence[int] | None = None,
    *,
    tile_size: int = 32,
) -> tuple[int, ...]:
    """Return validated tile-aligned bucket ceilings ending at max_length."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    physical_max = math.ceil(max_length / tile_size) * tile_size
    if configured is None:
        values = []
        boundary = tile_size
        while boundary < physical_max:
            values.append(boundary)
            boundary *= 2
        values.append(physical_max)
    else:
        values = [int(value) for value in configured]
        if not values:
            raise ValueError("sequence_buckets cannot be empty")
        if any(value <= 0 or value % tile_size for value in values):
            raise ValueError(
                f"sequence_buckets must be positive {tile_size}-token multiples"
            )
        if any(value > physical_max for value in values):
            raise ValueError("sequence_buckets cannot exceed max_length rounded to a tile")
        values.append(physical_max)
    return tuple(sorted(set(values)))


def bucket_for_length(length: int, boundaries: Sequence[int]) -> int:
    if length <= 0:
        raise ValueError("example lengths must be positive")
    for boundary in boundaries:
        if length <= boundary:
            return int(boundary)
    raise ValueError(
        f"example length {length} exceeds largest bucket {boundaries[-1]}"
    )


@dataclass(frozen=True)
class PlannedBatch:
    indices: tuple[int, ...]
    sequence_length: int
    useful_tokens: int
    padded_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpochBatchPlan:
    epoch: int
    seed: int
    batches: tuple[PlannedBatch, ...]
    dropped_indices: tuple[int, ...]
    algorithm: str = DATA_PLAN_VERSION

    @property
    def useful_tokens(self) -> int:
        return sum(item.useful_tokens for item in self.batches)

    @property
    def padded_tokens(self) -> int:
        return sum(item.padded_tokens for item in self.batches)

    @property
    def padding_fraction(self) -> float:
        return 1.0 - self.useful_tokens / max(1, self.padded_tokens)

    @property
    def order_hash(self) -> str:
        return _stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "epoch": self.epoch,
            "seed": self.seed,
            "batches": [item.to_dict() for item in self.batches],
            "dropped_indices": list(self.dropped_indices),
            "useful_tokens": self.useful_tokens,
            "padded_tokens": self.padded_tokens,
            "padding_fraction": self.padding_fraction,
        }


@dataclass(frozen=True)
class DataGeometryCandidate:
    candidate_id: str
    batch_size: int
    sequence_length: int
    gradient_accumulation_steps: int
    sequence_bucketing: bool
    packing: bool
    prefetch_batches: int
    expected_batches: int
    expected_optimizer_steps: int
    expected_useful_tokens: int
    expected_padded_tokens: int
    expected_padding_fraction: float
    projected_device_bytes: int | None
    projected_step_seconds: float | None
    projected_useful_tokens_per_second: float | None
    projected_epoch_seconds: float | None
    projection_contract: str
    admitted: bool
    rejection_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def best_fit_pack_lengths(
    lengths: Sequence[int], max_length: int, *, order: Sequence[int] | None = None
) -> tuple[int, ...]:
    """Return deterministic no-split best-fit bin lengths."""

    source_order = list(range(len(lengths))) if order is None else list(order)
    bins: list[int] = []
    for index in source_order:
        length = int(lengths[index])
        if length <= 0 or length > max_length:
            raise ValueError("pack lengths must be in [1, max_length]")
        candidates = [
            (max_length - used - length, bin_index)
            for bin_index, used in enumerate(bins)
            if used + length <= max_length
        ]
        if candidates:
            _remaining, selected = min(candidates)
            bins[selected] += length
        else:
            bins.append(length)
    return tuple(bins)


def plan_data_geometries(
    lengths: Sequence[int],
    *,
    batch_size: int,
    sequence_length: int,
    gradient_accumulation_steps: int,
    seed: int,
    packing_enabled: bool,
    packing_semantically_safe: bool,
    estimated_device_bytes: int | None = None,
    device_capacity_bytes: int | None = None,
    measured_step_seconds: float | None = None,
    preparation_seconds_per_batch: float = 0.0,
) -> dict[str, Any]:
    """Build a bounded non-Cartesian data/hardware candidate portfolio.

    Projections are intentionally labelled analytical.  A selected production
    winner still requires synchronized whole-step measurement by the common
    autotuner autotuner.
    """

    if not lengths:
        raise ValueError("data planning requires at least one token length")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    raw_lengths = tuple(min(int(value), sequence_length) for value in lengths)
    effective_batch = batch_size * gradient_accumulation_steps
    policies: list[tuple[int, int, bool, bool, int]] = [
        (batch_size, gradient_accumulation_steps, False, False, 0),
        (batch_size, gradient_accumulation_steps, True, packing_enabled, 0),
        (batch_size, gradient_accumulation_steps, True, packing_enabled, 2),
    ]
    for candidate_batch in (1, 2, 3, 4):
        if candidate_batch == batch_size:
            continue
        candidate_accumulation = max(
            1, math.ceil(effective_batch / candidate_batch)
        )
        policies.append((
            candidate_batch,
            candidate_accumulation,
            True,
            packing_enabled,
            2,
        ))
    if packing_semantically_safe and not packing_enabled:
        policies.append((
            batch_size,
            gradient_accumulation_steps,
            True,
            True,
            2,
        ))

    unique = []
    seen = set()
    for policy in policies:
        if policy not in seen:
            unique.append(policy)
            seen.add(policy)

    candidates = []
    baseline_physical = None
    for index, (
        candidate_batch,
        candidate_accumulation,
        bucketed,
        packed,
        prefetch,
    ) in enumerate(unique):
        candidate_lengths = (
            best_fit_pack_lengths(raw_lengths, sequence_length)
            if packed
            else raw_lengths
        )
        epoch_plan = plan_epoch_batches(
            candidate_lengths,
            batch_size=candidate_batch,
            max_length=sequence_length,
            seed=seed,
            epoch=0,
            shuffle=True,
            bucketed=bucketed,
            drop_last=True,
        )
        physical_per_step = (
            epoch_plan.padded_tokens / max(1, len(epoch_plan.batches))
            * candidate_accumulation
        )
        if baseline_physical is None:
            baseline_physical = physical_per_step
        token_scale = physical_per_step / max(1.0, baseline_physical)
        # Persistent model/optimizer state dominates most LoRA jobs.  Model
        # 90% as fixed and 10% as token-geometry-dependent; this is a guarded
        # admission projection, never a measured peak claim.
        projected_memory = (
            int(estimated_device_bytes * (0.90 + 0.10 * token_scale))
            if estimated_device_bytes is not None
            else None
        )
        reasons = []
        if packed and not packing_semantically_safe:
            reasons.append("packing would change independent-example attention semantics")
        if not epoch_plan.batches:
            reasons.append("candidate produces zero complete batches")
        if (
            projected_memory is not None
            and device_capacity_bytes is not None
            and projected_memory > device_capacity_bytes
        ):
            reasons.append("projected single-card memory exceeds guarded capacity")
        projected_step = (
            measured_step_seconds * token_scale
            if measured_step_seconds is not None
            else None
        )
        optimizer_steps = math.ceil(
            len(epoch_plan.batches) / candidate_accumulation
        )
        exposed_prepare = (
            preparation_seconds_per_batch
            * len(epoch_plan.batches)
            * (0.25 if prefetch else 1.0)
        )
        projected_epoch = (
            optimizer_steps * projected_step + exposed_prepare
            if projected_step is not None
            else None
        )
        useful_tps = (
            epoch_plan.useful_tokens / projected_epoch
            if projected_epoch is not None and projected_epoch > 0
            else None
        )
        document = {
            "batch_size": candidate_batch,
            "gradient_accumulation_steps": candidate_accumulation,
            "sequence_bucketing": bucketed,
            "packing": packed,
            "prefetch_batches": prefetch,
        }
        candidates.append(DataGeometryCandidate(
            candidate_id="d-" + _stable_hash(document)[:16],
            batch_size=candidate_batch,
            sequence_length=sequence_length,
            gradient_accumulation_steps=candidate_accumulation,
            sequence_bucketing=bucketed,
            packing=packed,
            prefetch_batches=prefetch,
            expected_batches=len(epoch_plan.batches),
            expected_optimizer_steps=optimizer_steps,
            expected_useful_tokens=epoch_plan.useful_tokens,
            expected_padded_tokens=epoch_plan.padded_tokens,
            expected_padding_fraction=epoch_plan.padding_fraction,
            projected_device_bytes=projected_memory,
            projected_step_seconds=projected_step,
            projected_useful_tokens_per_second=useful_tps,
            projected_epoch_seconds=projected_epoch,
            projection_contract=(
                "analytical_fixed90_activation10_token_scale; whole-step "
                "measurement required before selection"
            ),
            admitted=not reasons,
            rejection_reasons=tuple(reasons),
        ))
    admitted = [item for item in candidates if item.admitted]
    projected = [
        item for item in admitted
        if item.projected_useful_tokens_per_second is not None
    ]
    selected = max(
        projected,
        key=lambda item: item.projected_useful_tokens_per_second,
        default=(admitted[0] if admitted else None),
    )
    return {
        "selection_status": "analytical_unmeasured",
        "selection_rule": (
            "maximum projected useful tokens/sec; production acceptance "
            "requires synchronized whole-step measurement"
        ),
        "selected": selected.to_dict() if selected is not None else None,
        "candidates": [item.to_dict() for item in candidates],
        "candidate_count": len(candidates),
        "admitted_count": len(admitted),
        "effective_batch_target": effective_batch,
    }


def plan_epoch_batches(
    lengths: Sequence[int],
    *,
    batch_size: int,
    max_length: int,
    seed: int,
    epoch: int,
    shuffle: bool,
    bucketed: bool,
    boundaries: Sequence[int] | None = None,
    drop_last: bool = True,
) -> EpochBatchPlan:
    """Build a fully deterministic epoch order with minimal bucket padding."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if any(int(length) <= 0 or int(length) > max_length for length in lengths):
        raise ValueError("lengths must be in the inclusive range [1, max_length]")
    normalized = (
        normalize_sequence_buckets(max_length, boundaries)
        if bucketed
        else (math.ceil(max_length / 32) * 32,)
    )
    rng = random.Random(int(seed) + int(epoch))
    groups: dict[int, list[int]] = {item: [] for item in normalized}
    indices = list(range(len(lengths)))
    if shuffle:
        rng.shuffle(indices)
    for index in indices:
        groups[bucket_for_length(int(lengths[index]), normalized)].append(index)

    batches: list[PlannedBatch] = []
    dropped: list[int] = []
    for boundary in normalized:
        group = groups[boundary]
        # ``indices`` was globally shuffled above.  A second independent
        # shuffle avoids source-length correlations within each bucket while
        # remaining exactly reconstructible from seed+epoch.
        if shuffle:
            rng.shuffle(group)
        complete = len(group) if not drop_last else len(group) - len(group) % batch_size
        for start in range(0, complete, batch_size):
            batch = tuple(group[start : start + batch_size])
            if len(batch) < batch_size and drop_last:
                dropped.extend(batch)
                continue
            useful = sum(int(lengths[index]) for index in batch)
            batches.append(PlannedBatch(
                indices=batch,
                sequence_length=boundary,
                useful_tokens=useful,
                padded_tokens=boundary * len(batch),
            ))
        dropped.extend(group[complete:])
    if shuffle:
        rng.shuffle(batches)
    return EpochBatchPlan(
        epoch=int(epoch),
        seed=int(seed),
        batches=tuple(batches),
        dropped_indices=tuple(dropped),
    )


def data_contract_hash(
    *,
    tokenized_rows: Iterable[Mapping[str, Sequence[int]]],
    configuration: Mapping[str, Any],
) -> str:
    """Hash the semantic data state consumed by an exact-resume run."""

    digest = hashlib.sha256()
    digest.update(DATA_PLAN_VERSION.encode("utf-8"))
    digest.update(json.dumps(
        dict(configuration),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8"))
    for row in tokenized_rows:
        for key in ("input_ids", "attention_mask", "labels"):
            digest.update(key.encode("ascii"))
            digest.update(json.dumps(
                [int(value) for value in row[key]],
                separators=(",", ":"),
            ).encode("ascii"))
    return digest.hexdigest()


def project_data_throughput(
    *,
    plan: EpochBatchPlan,
    accelerator_step_seconds: float,
    gradient_accumulation_steps: int = 1,
    preparation_seconds_per_batch: float = 0.0,
    overlap_fraction: float = 1.0,
) -> dict[str, float]:
    """Project useful throughput without pretending overlap is free."""

    if accelerator_step_seconds <= 0:
        raise ValueError("accelerator_step_seconds must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if preparation_seconds_per_batch < 0:
        raise ValueError("preparation_seconds_per_batch cannot be negative")
    if not 0 <= overlap_fraction <= 1:
        raise ValueError("overlap_fraction must be in [0, 1]")
    batch_count = len(plan.batches)
    optimizer_steps = math.ceil(batch_count / gradient_accumulation_steps)
    exposed_prepare = (
        batch_count
        * preparation_seconds_per_batch
        * (1.0 - overlap_fraction)
    )
    wall = optimizer_steps * accelerator_step_seconds + exposed_prepare
    return {
        "optimizer_steps": float(optimizer_steps),
        "useful_tokens": float(plan.useful_tokens),
        "padded_tokens": float(plan.padded_tokens),
        "projected_wall_seconds": wall,
        "projected_useful_tokens_per_second": plan.useful_tokens / max(wall, 1e-12),
        "projected_physical_tokens_per_second": plan.padded_tokens / max(wall, 1e-12),
    }
