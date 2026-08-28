#!/usr/bin/env python3
"""Verify standard PEFT reload and continued training from persisted state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from blackhole_training.compare_training_resume import _checkpoint_state


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"metrics must contain an object: {path}")
    return value


def verify(
    metrics: dict[str, Any],
    *,
    source_checkpoint: dict[str, Any],
    final_checkpoint: dict[str, Any],
    minimum_steps: int,
) -> dict[str, Any]:
    initial = (
        (metrics.get("initial_training_state") or {})
        .get("trainable", {})
        .get("sha256")
    )
    final_state = metrics.get("final_training_state") or {}
    final = (final_state.get("trainable") or {}).get("sha256")
    finiteness = final_state.get("trainable_finiteness") or {}
    source_digests = {
        value
        for value in (
            (source_checkpoint.get("trainable") or {}).get("sha256"),
            (
                source_checkpoint.get("trainable_runtime_wrapper") or {}
            ).get("sha256"),
        )
        if value
    }
    final_digests = {
        value
        for value in (
            (final_checkpoint.get("trainable") or {}).get("sha256"),
            (
                final_checkpoint.get("trainable_runtime_wrapper") or {}
            ).get("sha256"),
        )
        if value
    }
    checks = {
        "training_completed": metrics.get("status") == "completed",
        "minimum_steps_completed": int(metrics.get("global_steps") or 0)
        >= minimum_steps,
        "standard_adapter_loaded_exactly": (
            bool(initial) and initial in source_digests
        ),
        "continued_training_changed_state": bool(
            initial and final and initial != final
        ),
        "final_state_finite": bool(finiteness.get("finite"))
        and int(finiteness.get("nonfinite_tensor_count") or 0) == 0,
        "serialized_final_state_matches_runtime": (
            bool(final) and final in final_digests
        ),
        "serialized_final_step_matches_runtime": (
            final_checkpoint.get("step") == metrics.get("global_steps")
        ),
    }
    return {
        "format": "tt-finetune-adapter-reload-continuation-v2",
        "status": "accepted" if all(checks.values()) else "failed",
        "checks": checks,
        "steps": metrics.get("global_steps"),
        "initial_trainable_sha256": initial,
        "final_trainable_sha256": final,
        "serialized_digest_contract": (
            "canonical or runtime-wrapper parameter names; tensor values, "
            "order, dtype and shape remain exact"
        ),
        "source_checkpoint": source_checkpoint,
        "final_checkpoint": final_checkpoint,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--minimum-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    metrics_path = arguments.metrics.expanduser().resolve()
    result = verify(
        _load(metrics_path),
        source_checkpoint=_checkpoint_state(
            arguments.source_checkpoint.expanduser().resolve()
        ),
        final_checkpoint=_checkpoint_state(
            arguments.final_checkpoint.expanduser().resolve()
        ),
        minimum_steps=arguments.minimum_steps,
    )
    result["metrics"] = {
        "path": metrics_path.name,
        "sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
    }
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
