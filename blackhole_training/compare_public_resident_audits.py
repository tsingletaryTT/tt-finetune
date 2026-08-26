#!/usr/bin/env python3
"""Compare two isolated public-resident evolving-state qualifications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


_CATEGORIES = (
    "loss",
    "raw_gradients",
    "clipped_gradients",
    "parameters",
    "first_moment",
    "second_moment",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(reference_path: Path, candidate_path: Path) -> dict:
    reference = _load(reference_path)
    candidate = _load(candidate_path)
    wanted = reference.get("resident_evolving_audit") or []
    actual = candidate.get("resident_evolving_audit") or []
    steps = min(len(wanted), len(actual))
    exact_counts = {name: 0 for name in _CATEGORIES}
    total_counts = {name: 0 for name in _CATEGORIES}
    per_step_exact: list[bool] = []
    first_mismatches: list[dict] = []
    for index in range(steps):
        step_exact = True
        for category in _CATEGORIES:
            expected_values = wanted[index].get(category)
            observed_values = actual[index].get(category)
            if category == "loss":
                expected_values = {"loss": expected_values}
                observed_values = {"loss": observed_values}
            expected_values = expected_values or {}
            observed_values = observed_values or {}
            total_counts[category] += len(expected_values)
            for name, digest in expected_values.items():
                if observed_values.get(name) == digest:
                    exact_counts[category] += 1
                else:
                    step_exact = False
                    if len(first_mismatches) < 32:
                        first_mismatches.append({
                            "step": index + 1,
                            "category": category,
                            "name": name,
                        })
            if set(observed_values) != set(expected_values):
                step_exact = False
        per_step_exact.append(step_exact)

    reference_final = reference.get("final_training_state") or {}
    candidate_final = candidate.get("final_training_state") or {}
    final_trainable_exact = (
        reference_final.get("trainable", {}).get("sha256")
        == candidate_final.get("trainable", {}).get("sha256")
        and reference_final.get("trainable", {}).get("tensor_count")
        == candidate_final.get("trainable", {}).get("tensor_count")
    )
    final_optimizer_exact = (
        reference_final.get("optimizer_logical", {}).get("sha256")
        == candidate_final.get("optimizer_logical", {}).get("sha256")
        and reference_final.get("optimizer_logical", {}).get("tensor_count")
        == candidate_final.get("optimizer_logical", {}).get("tensor_count")
    )
    execution_exact = (
        reference.get("execution", {}).get("selected_plan") == "resident_exact"
        and candidate.get("execution", {}).get("selected_plan") == "resident_exact"
        and reference.get("execution", {}).get("actual_executor") == "resident_exact"
        and candidate.get("execution", {}).get("actual_executor") == "resident_exact"
        and reference.get("execution", {}).get("fallback") is None
        and candidate.get("execution", {}).get("fallback") is None
    )
    accepted = (
        len(wanted) == len(actual) == 20
        and all(per_step_exact)
        and all(exact_counts[name] == total_counts[name] for name in _CATEGORIES)
        and final_trainable_exact
        and final_optimizer_exact
        and execution_exact
    )
    return {
        "status": "accepted_bit_exact" if accepted else "rejected_nonexact",
        "contract": "isolated_public_resident_evolving_state_v1",
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "completed_evolving_steps": steps,
        "per_step_exact": per_step_exact,
        "exact_counts": {
            name: {
                "exact": exact_counts[name],
                "total": total_counts[name],
            }
            for name in _CATEGORIES
        },
        "final_trainable_exact": final_trainable_exact,
        "final_optimizer_exact": final_optimizer_exact,
        "execution_exact": execution_exact,
        "first_mismatches": first_mismatches,
        "comparisons_inside_performance_timing": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.reference, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "accepted_bit_exact" else 2


if __name__ == "__main__":
    raise SystemExit(main())
