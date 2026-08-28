#!/usr/bin/env python3
"""Compare trained-adapter behavior with a matched PyTorch/PEFT authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"evaluation must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_identity(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value)
    for part in Path(raw).parts:
        if part.startswith("models--"):
            pieces = part.removeprefix("models--").split("--")
            if len(pieces) >= 2:
                return "/".join(pieces)
    return raw if "/" not in raw else Path(raw).name


def _rows(payload: dict[str, Any]) -> dict[int, float]:
    result: dict[int, float] = {}
    for row in payload.get("per_example") or []:
        index = int(row["index"])
        value = row.get("mean_nll")
        if value is not None:
            result[index] = float(value)
    return result


def _common_prefix_fraction(left: list[int], right: list[int]) -> float:
    denominator = max(len(left), len(right), 1)
    common = 0
    for first, second in zip(left, right):
        if first != second:
            break
        common += 1
    return common / denominator


def compare(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    minimum_correlation: float = 0.90,
    maximum_mean_absolute_error: float = 0.50,
    maximum_normalized_mean_absolute_error: float = 0.40,
) -> dict[str, Any]:
    candidate_rows = _rows(candidate)
    reference_rows = _rows(reference)
    common = sorted(set(candidate_rows) & set(reference_rows))
    candidate_values = [candidate_rows[index] for index in common]
    reference_values = [reference_rows[index] for index in common]
    errors = [
        left - right
        for left, right in zip(candidate_values, reference_values)
    ]
    finite = bool(errors) and all(
        math.isfinite(value)
        for value in (*candidate_values, *reference_values)
    )
    correlation = (
        statistics.correlation(candidate_values, reference_values)
        if len(common) >= 2
        and statistics.pstdev(candidate_values) > 0
        and statistics.pstdev(reference_values) > 0
        else None
    )
    mae = (
        statistics.fmean(abs(value) for value in errors)
        if errors else None
    )
    reference_scale = (
        statistics.fmean(abs(value) for value in reference_values)
        if reference_values else None
    )
    normalized_mae = (
        mae / reference_scale
        if mae is not None and reference_scale not in {None, 0.0}
        else None
    )
    effective_mae_limit = (
        max(
            maximum_mean_absolute_error,
            maximum_normalized_mean_absolute_error * reference_scale,
        )
        if reference_scale is not None else maximum_mean_absolute_error
    )
    signed_bias = statistics.fmean(errors) if errors else None
    centered_mae = (
        statistics.fmean(abs(value - signed_bias) for value in errors)
        if errors and signed_bias is not None else None
    )
    candidate_generations = {
        int(row["index"]): list(row.get("generated_token_ids") or [])
        for row in candidate.get("generations") or []
    }
    reference_generations = {
        int(row["index"]): list(row.get("generated_token_ids") or [])
        for row in reference.get("generations") or []
    }
    generation_indices = sorted(
        set(candidate_generations) & set(reference_generations)
    )
    exact_generations = sum(
        candidate_generations[index] == reference_generations[index]
        for index in generation_indices
    )
    prefix_fractions = [
        _common_prefix_fraction(
            candidate_generations[index], reference_generations[index]
        )
        for index in generation_indices
    ]
    same_contract = all((
        _model_identity(candidate.get("model"))
        == _model_identity(reference.get("model")),
        candidate.get("dataset", {}).get("sha256")
        == reference.get("dataset", {}).get("sha256"),
        candidate.get("configuration") == reference.get("configuration"),
        candidate.get("supervised_tokens")
        == reference.get("supervised_tokens"),
    ))
    checks = {
        "same_evaluation_contract": same_contract,
        "all_examples_compared": (
            bool(common)
            and len(common) == len(candidate_rows) == len(reference_rows)
        ),
        "finite_per_example_nll": finite,
        "per_example_nll_correlation_within_contract": (
            correlation is not None
            and correlation >= minimum_correlation
        ),
        "per_example_nll_mae_within_contract": (
            mae is not None and mae <= effective_mae_limit
        ),
    }
    return {
        "format": "tt-finetune-adapter-reference-comparison-v1",
        "status": "accepted" if all(checks.values()) else "failed",
        "contract": {
            "authority": "matched PyTorch/Transformers/PEFT CPU evaluation",
            "equality": "numerical trained behavior, not bit identity",
            "minimum_per_example_nll_correlation": minimum_correlation,
            "maximum_per_example_nll_mean_absolute_error": (
                maximum_mean_absolute_error
            ),
            "maximum_normalized_per_example_nll_mean_absolute_error": (
                maximum_normalized_mean_absolute_error
            ),
            "effective_per_example_nll_mean_absolute_error_limit": (
                effective_mae_limit
            ),
            "generation_agreement": "reported, not an acceptance gate",
        },
        "checks": checks,
        "model": _model_identity(candidate.get("model")),
        "example_count": len(common),
        "per_example_nll_correlation": correlation,
        "per_example_nll_mean_absolute_error": mae,
        "per_example_nll_normalized_mean_absolute_error": normalized_mae,
        "per_example_nll_signed_bias": signed_bias,
        "per_example_nll_bias_centered_mean_absolute_error": centered_mae,
        "per_example_nll_root_mean_square_error": (
            math.sqrt(statistics.fmean(value * value for value in errors))
            if errors else None
        ),
        "candidate_mean_nll": candidate.get("mean_negative_log_likelihood"),
        "reference_mean_nll": reference.get("mean_negative_log_likelihood"),
        "generation_count": len(generation_indices),
        "exact_generation_count": exact_generations,
        "mean_generation_common_prefix_fraction": (
            statistics.fmean(prefix_fractions)
            if prefix_fractions else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-correlation", type=float, default=0.90)
    parser.add_argument(
        "--maximum-mean-absolute-error", type=float, default=0.50
    )
    parser.add_argument(
        "--maximum-normalized-mean-absolute-error",
        type=float,
        default=0.40,
    )
    arguments = parser.parse_args()
    candidate_path = arguments.candidate.expanduser().resolve()
    reference_path = arguments.reference.expanduser().resolve()
    result = compare(
        _load(candidate_path),
        _load(reference_path),
        minimum_correlation=arguments.minimum_correlation,
        maximum_mean_absolute_error=arguments.maximum_mean_absolute_error,
        maximum_normalized_mean_absolute_error=(
            arguments.maximum_normalized_mean_absolute_error
        ),
    )
    result["candidate"] = {
        "artifact": candidate_path.name,
        "sha256": _sha256(candidate_path),
    }
    result["reference"] = {
        "artifact": reference_path.name,
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
