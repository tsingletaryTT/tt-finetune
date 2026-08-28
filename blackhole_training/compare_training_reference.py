#!/usr/bin/env python3
"""Compare an evolving TT loss trajectory with a matched PyTorch authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from blackhole_training.compare_training_resume import _checkpoint_state


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"training metrics must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(metrics: dict[str, Any]) -> dict[int, dict[str, Any]]:
    losses = metrics.get("step_losses") or []
    rates = metrics.get("step_learning_rates") or []
    batches = metrics.get("batch_identity_history") or []
    if not isinstance(losses, list) or not isinstance(batches, list):
        raise TypeError("step losses and batch history must be lists")
    if len(losses) != len(batches):
        raise ValueError("step losses and batch history have different lengths")
    if rates and len(rates) != len(losses):
        raise ValueError("learning-rate history has a different length")
    result: dict[int, dict[str, Any]] = {}
    for index, (loss, batch) in enumerate(zip(losses, batches)):
        step = int(batch.get("step"))
        result[step] = {
            "loss": loss,
            "learning_rate": rates[index] if rates else None,
            "batch": batch.get("host"),
        }
    return result


def compare(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    max_step: int | None = None,
    minimum_correlation: float = 0.90,
    maximum_mean_absolute_error: float = 0.50,
    initial_checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_records = _records(candidate)
    reference_records = _records(reference)
    common = sorted(set(candidate_records) & set(reference_records))
    if max_step is not None:
        common = [step for step in common if step <= max_step]
    observed = [
        step
        for step in common
        if candidate_records[step]["loss"] is not None
        and reference_records[step]["loss"] is not None
    ]
    candidate_losses = [
        float(candidate_records[step]["loss"]) for step in observed
    ]
    reference_losses = [
        float(reference_records[step]["loss"]) for step in observed
    ]
    errors = [
        left - right
        for left, right in zip(candidate_losses, reference_losses)
    ]
    correlation = (
        statistics.correlation(candidate_losses, reference_losses)
        if len(candidate_losses) >= 2
        and statistics.pstdev(candidate_losses) > 0
        and statistics.pstdev(reference_losses) > 0
        else None
    )
    mae = (
        statistics.fmean(abs(value) for value in errors)
        if errors else None
    )
    rmse = (
        math.sqrt(statistics.fmean(value * value for value in errors))
        if errors else None
    )
    batch_match = all(
        candidate_records[step]["batch"] == reference_records[step]["batch"]
        for step in common
    )
    rate_match = all(
        candidate_records[step]["learning_rate"]
        == reference_records[step]["learning_rate"]
        for step in common
    )
    finite = bool(errors) and all(
        math.isfinite(value)
        for value in (*candidate_losses, *reference_losses)
    )
    checks = {
        "overlapping_steps_present": bool(common),
        "observed_loss_pairs_present": bool(observed),
        "same_batch_order": batch_match,
        "same_learning_rate_schedule": rate_match,
        "finite_loss_pairs": finite,
        "trajectory_correlation_within_contract": (
            correlation is not None
            and correlation >= minimum_correlation
        ),
        "mean_absolute_error_within_contract": (
            mae is not None and mae <= maximum_mean_absolute_error
        ),
    }
    if initial_checkpoint is not None:
        expected_hashes = {
            initial_checkpoint["trainable"]["sha256"],
            initial_checkpoint["trainable_runtime_wrapper"]["sha256"],
        }
        candidate_initial = (
            (candidate.get("initial_training_state") or {})
            .get("trainable", {})
            .get("sha256")
        )
        reference_initial = (
            (reference.get("initial_training_state") or {})
            .get("trainable", {})
            .get("sha256")
        )
        checks.update({
            "candidate_initial_state_matches_serialized_authority": (
                candidate_initial in expected_hashes
            ),
            "reference_initial_state_matches_serialized_authority": (
                reference_initial in expected_hashes
            ),
        })
    result = {
        "format": "tt-finetune-training-reference-comparison-v1",
        "status": "accepted" if all(checks.values()) else "failed",
        "contract": {
            "equality": "numerical trajectory, not bit identity",
            "minimum_pearson_correlation": minimum_correlation,
            "maximum_mean_absolute_error": maximum_mean_absolute_error,
            "maximum_step": max_step,
        },
        "checks": checks,
        "common_step_count": len(common),
        "observed_loss_pair_count": len(observed),
        "first_observed_step": observed[0] if observed else None,
        "last_observed_step": observed[-1] if observed else None,
        "pearson_correlation": correlation,
        "mean_absolute_error": mae,
        "root_mean_square_error": rmse,
        "maximum_absolute_error": (
            max((abs(value) for value in errors), default=None)
        ),
        "candidate_mean_loss": (
            statistics.fmean(candidate_losses)
            if candidate_losses else None
        ),
        "reference_mean_loss": (
            statistics.fmean(reference_losses)
            if reference_losses else None
        ),
    }
    if initial_checkpoint is not None:
        result["serialized_initial_state_authority"] = initial_checkpoint
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-step", type=int)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--minimum-correlation", type=float, default=0.90)
    parser.add_argument(
        "--maximum-mean-absolute-error", type=float, default=0.50
    )
    arguments = parser.parse_args()
    candidate_path = arguments.candidate.expanduser().resolve()
    reference_path = arguments.reference.expanduser().resolve()
    result = compare(
        _load(candidate_path),
        _load(reference_path),
        max_step=arguments.max_step,
        minimum_correlation=arguments.minimum_correlation,
        maximum_mean_absolute_error=arguments.maximum_mean_absolute_error,
        initial_checkpoint=(
            _checkpoint_state(
                arguments.initial_checkpoint.expanduser().resolve()
            )
            if arguments.initial_checkpoint is not None else None
        ),
    )
    result["candidate"] = {
        "path": str(candidate_path),
        "sha256": _sha256(candidate_path),
    }
    result["reference"] = {
        "path": str(reference_path),
        "sha256": _sha256(reference_path),
    }
    destination = arguments.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
