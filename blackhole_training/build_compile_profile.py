#!/usr/bin/env python3
"""Build a factual compile/startup report from completed worker metrics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"metrics must contain an object: {path}")
    if value.get("status") != "completed":
        raise ValueError(f"metrics are not completed: {path}")
    return value


def _public_model_name(value: Any) -> str | None:
    """Return a stable checkpoint identity without publishing host paths."""

    if value is None:
        return None
    raw = str(value)
    for part in Path(raw).parts:
        if part.startswith("models--"):
            encoded = part.removeprefix("models--")
            pieces = encoded.split("--")
            if len(pieces) >= 2:
                return "/".join(pieces)
    return raw if "/" not in raw else Path(raw).name


def _cache_identity(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value)
    return Path(raw).name if "/" in raw else raw


def _public_compile(value: dict[str, Any]) -> dict[str, Any]:
    """Publish compile facts without copying a host cache path."""

    return {
        key: value.get(key)
        for key in (
            "first_step_seconds",
            "estimated_compile_seconds",
            "bucket_first_use",
            "expected_sequence_buckets",
            "observed_sequence_buckets",
            "cross_process_executable_reuse",
            "limitation",
        )
        if key in value
    }


def summarize(name: str, path: Path) -> dict[str, Any]:
    metrics = _load(path)
    events = metrics.get("startup_profile", {}).get("events") or []
    completed_stages = {
        str(event["stage"]): float(event["stage_seconds"])
        for event in events
        if event.get("status") == "completed"
        and event.get("stage_seconds") is not None
    }
    compile_data = metrics.get("compile") or {}
    persistent = metrics.get("persistent_runtime") or {}
    data_pipeline = metrics.get("data_pipeline") or {}
    tokenization = data_pipeline.get("tokenization") or {}
    same_process_reuse_observed = bool(
        persistent.get("client_cache_status") == "hit"
        or persistent.get("cache_status") == "hit"
        or int(persistent.get("prior_jobs_completed") or 0) > 0
    )
    checks = {
        "startup_profile_completed": (
            metrics.get("startup_profile", {}).get("current_stage")
            == "completed"
        ),
        "dataset_stage_measured": (
            "dataset_preparation" in completed_stages
        ),
        "device_compile_stage_measured": (
            "device_compile_and_first_step" in completed_stages
        ),
        "steady_training_stage_measured": (
            "steady_training" in completed_stages
        ),
        "compile_estimate_reported": (
            compile_data.get("estimated_compile_seconds") is not None
        ),
        "cross_process_limit_reported": (
            compile_data.get("cross_process_executable_reuse") is False
            and bool(compile_data.get("limitation"))
        ),
    }
    return {
        "workload": name,
        "status": "accepted" if all(checks.values()) else "failed",
        "checks": checks,
        "model": _public_model_name(metrics.get("model_name")),
        "geometry": {
            "batch_size": metrics.get("batch_size"),
            "sequence_length": metrics.get("max_length"),
            "gradient_accumulation_steps": metrics.get(
                "gradient_accumulation_steps"
            ),
        },
        "metrics": {"artifact": path.name, "sha256": _sha256(path)},
        "profiled_run_total_seconds": metrics.get(
            "startup_profile", {}
        ).get("total_seconds"),
        "startup_stage_seconds": completed_stages,
        "dataset_tokenization_seconds": tokenization.get("seconds"),
        "compile": _public_compile(compile_data),
        "cache": {
            "identity": _cache_identity(metrics.get("compile_cache")),
            "persistent_runtime_status": persistent.get("cache_status"),
            "client_cache_status": persistent.get("client_cache_status"),
            "same_process_executable_reuse_observed": (
                same_process_reuse_observed
            ),
            "cross_process_executable_reuse": bool(
                compile_data.get("cross_process_executable_reuse", False)
            ),
            "limitation": compile_data.get("limitation"),
        },
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Compilation and startup profile",
        "",
        "Compilation is excluded from steady-state training measurements. "
        "Times below come from the public worker's startup-stage events.",
        "",
        "| Workload | Dataset | Device compile + first step | Estimated compile | Cache |",
        "|---|---:|---:|---:|---|",
    ]
    for item in payload["workloads"]:
        stages = item["startup_stage_seconds"]
        compile_data = item["compile"]
        cache = item["cache"]
        lines.append(
            "| {name} | {dataset:.3f}s | {device:.3f}s | {compile:.3f}s | {cache} |".format(
                name=item["workload"],
                dataset=float(stages.get("dataset_preparation", 0.0)),
                device=float(stages.get("device_compile_and_first_step", 0.0)),
                compile=float(compile_data.get("estimated_compile_seconds") or 0.0),
                cache=(
                    cache.get("client_cache_status")
                    or cache.get("persistent_runtime_status")
                    or "isolated"
                ),
            )
        )
    lines.extend([
        "",
        "Current PJRT cannot deserialize a complete executable in a fresh "
        "process. Lower-level compiler artifacts and semantic measured plans "
        "remain reusable; safe complete-executable reuse is same-process only.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        action="append",
        required=True,
        metavar="NAME=METRICS_JSON",
    )
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    arguments = parser.parse_args()
    workloads = []
    for value in arguments.workload:
        if "=" not in value:
            raise ValueError("--workload must be NAME=METRICS_JSON")
        name, raw_path = value.split("=", 1)
        workloads.append(
            summarize(name, Path(raw_path).expanduser().resolve())
        )
    payload = {
        "format": "tt-finetune-compile-startup-profile-v1",
        "status": (
            "accepted"
            if all(item["status"] == "accepted" for item in workloads)
            else "failed"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workloads": workloads,
    }
    json_path = arguments.json_output.expanduser().resolve()
    markdown_path = arguments.markdown_output.expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
