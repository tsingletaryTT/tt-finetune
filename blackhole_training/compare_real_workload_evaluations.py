#!/usr/bin/env python3
"""Compare deterministic before/after CPU evaluations for one workload."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"evaluation must contain an object: {path}")
    return value


def compare(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    repeat_after: dict[str, Any] | None = None,
    require_nll_improvement: bool = False,
) -> dict[str, Any]:
    same_contract = all((
        before.get("model") == after.get("model"),
        before.get("dataset", {}).get("sha256")
        == after.get("dataset", {}).get("sha256"),
        before.get("configuration") == after.get("configuration"),
        before.get("supervised_tokens") == after.get("supervised_tokens"),
    ))
    before_nll = float(before["mean_negative_log_likelihood"])
    after_nll = float(after["mean_negative_log_likelihood"])
    generations_before = before.get("generations") or []
    generations_after = after.get("generations") or []
    generation_pairs = list(zip(generations_before, generations_after))
    changed = sum(
        left.get("generated_token_ids") != right.get("generated_token_ids")
        for left, right in generation_pairs
    )
    repeat_exact = None
    if repeat_after is not None:
        repeat_exact = all((
            after.get("mean_negative_log_likelihood")
            == repeat_after.get("mean_negative_log_likelihood"),
            after.get("total_negative_log_likelihood")
            == repeat_after.get("total_negative_log_likelihood"),
            after.get("per_example") == repeat_after.get("per_example"),
            after.get("generations") == repeat_after.get("generations"),
        ))
    checks = {
        "same_evaluation_contract": same_contract,
        "finite_before_and_after": all(map(math.isfinite, (
            before_nll, after_nll,
        ))),
        "adapter_changed_at_least_one_generation": changed > 0,
        "repeat_after_is_exact": repeat_exact,
        "heldout_nll_improved": after_nll < before_nll,
    }
    required = [
        checks["same_evaluation_contract"],
        checks["finite_before_and_after"],
        checks["adapter_changed_at_least_one_generation"],
    ]
    if repeat_after is not None:
        required.append(bool(repeat_exact))
    if require_nll_improvement:
        required.append(checks["heldout_nll_improved"])
    return {
        "format": "tt-finetune-real-workload-evaluation-comparison-v1",
        "status": "accepted" if all(required) else "failed",
        "checks": checks,
        "before_mean_nll": before_nll,
        "after_mean_nll": after_nll,
        "absolute_nll_change": after_nll - before_nll,
        "relative_nll_change": (after_nll / before_nll - 1.0),
        "before_perplexity": before.get("perplexity"),
        "after_perplexity": after.get("perplexity"),
        "generation_count": len(generation_pairs),
        "changed_generation_count": changed,
        "require_nll_improvement": require_nll_improvement,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--repeat-after", type=Path)
    parser.add_argument("--require-nll-improvement", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    result = compare(
        _load(arguments.before.expanduser().resolve()),
        _load(arguments.after.expanduser().resolve()),
        repeat_after=(
            _load(arguments.repeat_after.expanduser().resolve())
            if arguments.repeat_after is not None else None
        ),
        require_nll_improvement=arguments.require_nll_improvement,
    )
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
