#!/usr/bin/env python3
"""Prove exact public resident checkpoint/resume continuation."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uninterrupted", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    uninterrupted = _load(args.uninterrupted)
    partial = _load(args.partial)
    resumed = _load(args.resumed)
    expected = {
        int(item["step"]): item
        for item in uninterrupted.get("resident_evolving_audit") or []
    }
    observed_items = [
        *(partial.get("resident_evolving_audit") or []),
        *(resumed.get("resident_evolving_audit") or []),
    ]
    observed = {int(item["step"]): item for item in observed_items}
    exact_counts = {name: 0 for name in _CATEGORIES}
    total_counts = {name: 0 for name in _CATEGORIES}
    per_step_exact = []
    first_mismatches = []
    for step in range(1, 21):
        wanted = expected.get(step, {})
        actual = observed.get(step, {})
        step_exact = bool(wanted and actual)
        for category in _CATEGORIES:
            wanted_values = wanted.get(category)
            actual_values = actual.get(category)
            if category == "loss":
                wanted_values = {"loss": wanted_values}
                actual_values = {"loss": actual_values}
            wanted_values = wanted_values or {}
            actual_values = actual_values or {}
            total_counts[category] += len(wanted_values)
            for name, digest in wanted_values.items():
                if actual_values.get(name) == digest:
                    exact_counts[category] += 1
                else:
                    step_exact = False
                    if len(first_mismatches) < 32:
                        first_mismatches.append({
                            "step": step,
                            "category": category,
                            "name": name,
                        })
            if set(actual_values) != set(wanted_values):
                step_exact = False
        per_step_exact.append(step_exact)

    expected_final = uninterrupted.get("final_training_state") or {}
    resumed_final = resumed.get("final_training_state") or {}
    final_trainable_exact = (
        expected_final.get("trainable", {}).get("sha256")
        == resumed_final.get("trainable", {}).get("sha256")
    )
    final_optimizer_exact = (
        expected_final.get("optimizer_logical", {}).get("sha256")
        == resumed_final.get("optimizer_logical", {}).get("sha256")
    )
    data_steps = [int(item["step"]) for item in observed_items]
    accepted = (
        data_steps == list(range(1, 21))
        and all(per_step_exact)
        and all(exact_counts[name] == total_counts[name] for name in _CATEGORIES)
        and final_trainable_exact
        and final_optimizer_exact
        and partial.get("status") == "interrupted"
        and resumed.get("status") == "completed"
    )
    report = {
        "status": "accepted_exact_resume" if accepted else "rejected_resume",
        "contract": "public_resident_checkpoint_resume_v1",
        "split": "10_steps_then_resume_to_20",
        "completed_evolving_steps": len(observed_items),
        "observed_steps": data_steps,
        "per_step_exact": per_step_exact,
        "exact_counts": {
            name: {"exact": exact_counts[name], "total": total_counts[name]}
            for name in _CATEGORIES
        },
        "final_trainable_exact": final_trainable_exact,
        "final_optimizer_exact": final_optimizer_exact,
        "first_mismatches": first_mismatches,
        "comparisons_inside_performance_timing": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
