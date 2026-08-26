#!/usr/bin/env python3
"""Build the final real-workload product-hardening acceptance report.

The report is intentionally strict: every workload and product gate is backed
by a named, hashed artifact, and no host development path is published in the
result.  A missing or failed authority makes the aggregate report fail.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"artifact must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {"name": path.name, "sha256": _sha256(path)}


def _state_finite(metrics: dict[str, Any], name: str) -> bool:
    final = metrics.get("final_training_state") or {}
    value = final.get(name) or {}
    return bool(value.get("finite")) and int(
        value.get("nonfinite_tensor_count") or 0
    ) == 0


def _training_summary(
    metrics_path: Path,
    *,
    minimum_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = _load(metrics_path)
    losses = [
        float(value)
        for value in metrics.get("step_losses") or []
        if value is not None
    ]
    initial = (
        (metrics.get("initial_training_state") or {})
        .get("trainable", {})
        .get("sha256")
    )
    final = (
        (metrics.get("final_training_state") or {})
        .get("trainable", {})
        .get("sha256")
    )
    gradient = metrics.get("gradient_audit") or {}
    checks = {
        "completed": metrics.get("status") == "completed",
        "minimum_evolving_steps": int(metrics.get("global_steps") or 0)
        >= minimum_steps,
        "reported_losses_finite": bool(losses)
        and all(math.isfinite(value) for value in losses),
        "gradients_finite": int(
            gradient.get("nonfinite_tensor_count") or 0
        )
        == 0,
        "parameters_finite": _state_finite(
            metrics, "trainable_finiteness"
        ),
        "optimizer_finite": _state_finite(
            metrics, "optimizer_logical_finiteness"
        ),
        "trainable_state_changed": bool(initial and final and initial != final),
    }
    timing = (
        metrics.get("steady_step_e2e_summary")
        or metrics.get("step_e2e_summary")
        or {}
    )
    useful = [int(value) for value in metrics.get("step_useful_tokens") or []]
    seconds = [float(value) for value in metrics.get("step_e2e_seconds") or []]
    batches = metrics.get("batch_identity_history") or []
    recorded_steps = [
        int(item.get("step")) for item in batches
    ] if len(batches) == len(seconds) else list(
        range(
            int(metrics.get("global_steps") or len(seconds))
            - len(seconds) + 1,
            int(metrics.get("global_steps") or len(seconds)) + 1,
        )
    )
    first_uses = {
        int(item["step"])
        for item in (metrics.get("compile") or {}).get(
            "bucket_first_use", []
        )
    }
    warmup_count = int(timing.get("warmup_count") or 0)
    selected_indices = [
        index
        for index, step in enumerate(recorded_steps)
        if index >= warmup_count and step not in first_uses
    ]
    useful_total = sum(useful[index] for index in selected_indices)
    seconds_total = sum(seconds[index] for index in selected_indices)
    useful_per_second = (
        useful_total / seconds_total
        if useful and seconds and seconds_total > 0 else None
    )
    summary = {
        "status": "accepted" if all(checks.values()) else "failed",
        "checks": checks,
        "artifact": _artifact(metrics_path),
        "model": _public_model(metrics.get("model_name")),
        "steps": metrics.get("global_steps"),
        "batch_size": metrics.get("batch_size"),
        "sequence_length": metrics.get("max_length"),
        "step_time": timing,
        "useful_tokens_per_second": useful_per_second,
        "compile": _public_compile(metrics.get("compile") or {}),
        "memory": _public_memory(metrics.get("memory_accounting") or {}),
        "execution_plan": metrics.get("automatic_execution_plan"),
        "numerical_contract": (
            metrics.get("moe_numerical_contract")
            or (
                (metrics.get("memory_accounting") or {})
                .get("weight_memory_mode", {})
                .get("numerical_contract")
            )
        ),
        "initial_trainable_sha256": initial,
        "final_trainable_sha256": final,
    }
    return summary, metrics


def _public_model(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value)
    for part in Path(raw).parts:
        if part.startswith("models--"):
            pieces = part.removeprefix("models--").split("--")
            if len(pieces) >= 2:
                return "/".join(pieces)
    return raw if "/" not in raw else Path(raw).name


def _public_compile(value: dict[str, Any]) -> dict[str, Any]:
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


def _public_memory(value: dict[str, Any]) -> dict[str, Any]:
    mode = value.get("weight_memory_mode") or {}
    return {
        "contract": value.get("contract"),
        "frozen_parameter_logical_bytes": value.get(
            "frozen_parameter_logical_bytes"
        ),
        "trainable_parameter_logical_bytes": value.get(
            "trainable_parameter_logical_bytes"
        ),
        "optimizer_moment_logical_bytes": value.get(
            "optimizer_moment_logical_bytes"
        ),
        "requested_weight_mode": mode.get("requested_mode"),
        "selected_weight_dtype": mode.get("selected_weight_dtype"),
        "numerical_contract": mode.get("numerical_contract"),
        "observed_device_peak_bytes": value.get("observed_device_peak_bytes"),
        "observed_device_peak_contract": value.get(
            "observed_device_peak_contract"
        ),
    }


def _accepted(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _load(path)
    return {
        "status": (
            "accepted" if payload.get("status") == "accepted" else "failed"
        ),
        "artifact": _artifact(path),
    }, payload


def _memory_profile(path: Path) -> dict[str, Any]:
    metrics = _load(path)
    allocator = (
        (metrics.get("memory_accounting") or {}).get("device_allocator")
        or {}
    )
    dram = allocator.get("peak_compile_snapshot_dram") or {}
    checks = {
        "training_completed": metrics.get("status") == "completed",
        "allocator_profile_available": allocator.get("available") is True,
        "measured_dram_snapshot_present": int(
            dram.get("allocated_bytes") or 0
        ) > 0,
        "measurement_contract_reported": bool(
            allocator.get("measurement_contract")
        ),
        "runtime_hwm_limitation_reported": (
            allocator.get("is_exact_runtime_high_water_mark") is False
            and bool(allocator.get("limitation"))
        ),
    }
    return {
        "status": "accepted" if all(checks.values()) else "failed",
        "checks": checks,
        "artifact": _artifact(path),
        "measurement_contract": allocator.get("measurement_contract"),
        "is_exact_runtime_high_water_mark": allocator.get(
            "is_exact_runtime_high_water_mark"
        ),
        "peak_compile_snapshot_dram": {
            key: dram.get(key)
            for key in (
                "allocatable_bytes",
                "allocated_bytes",
                "free_bytes",
            )
        },
        "limitation": allocator.get("limitation"),
    }


def _add_authority(
    target: dict[str, Any], name: str, path: Path
) -> dict[str, Any]:
    summary, payload = _accepted(path)
    target[name] = summary
    return payload


def build(arguments: argparse.Namespace) -> dict[str, Any]:
    small, small_metrics = _training_summary(
        arguments.small_metrics, minimum_steps=5000
    )
    small_resume = _add_authority(
        small, "exact_resume", arguments.small_resume
    )
    _add_authority(small, "behavior_change", arguments.small_behavior)
    _add_authority(
        small, "training_reference", arguments.small_training_reference
    )
    _add_authority(
        small, "adapter_reference", arguments.small_adapter_reference
    )
    small["checks"].update({
        "exact_interrupt_resume": small_resume.get("status") == "accepted",
        "automatic_target_discovery": bool(
            (small_metrics.get("architecture") or {}).get(
                "selected_lora_paths"
            )
        ),
        "data_planner_exercised": bool(small_metrics.get("data_pipeline")),
        "execution_planner_exercised": bool(
            (small_metrics.get("architecture") or {}).get("plan")
            or small_metrics.get("automatic_execution_plan")
        ),
    })

    medium, _ = _training_summary(
        arguments.medium_metrics, minimum_steps=1000
    )
    medium["memory_profile"] = _memory_profile(arguments.medium_memory)
    _add_authority(medium, "heldout_behavior", arguments.medium_behavior)
    _add_authority(
        medium, "training_reference", arguments.medium_training_reference
    )
    _add_authority(
        medium, "adapter_reference", arguments.medium_adapter_reference
    )

    large, large_metrics = _training_summary(
        arguments.large_metrics, minimum_steps=100
    )
    large["memory_profile"] = _memory_profile(arguments.large_memory)
    large_resume = _add_authority(
        large, "exact_resume", arguments.large_resume
    )
    large_reload = _add_authority(
        large, "adapter_reload_continuation", arguments.large_reload
    )
    large_mode = (
        (large_metrics.get("memory_accounting") or {})
        .get("weight_memory_mode", {})
    )
    large["checks"].update({
        "reduced_frozen_weight_path": (
            large_mode.get("selected_weight_dtype") in {"bfp_bf8", "bfp_bf4"}
        ),
        "fused_forward_backward_boundary": (
            (large_metrics.get("automatic_execution_plan") or {}).get(
                "graph_partition"
            )
            == "forward_backward_fused"
        ),
        "deterministic_resume": large_resume.get("status") == "accepted",
        "reload_continued_training": large_reload.get("status") == "accepted",
    })

    moe, moe_metrics = _training_summary(
        arguments.moe_metrics, minimum_steps=100
    )
    moe_resume = _add_authority(moe, "exact_resume", arguments.moe_resume)
    moe_reload = _add_authority(
        moe, "adapter_reload_continuation", arguments.moe_reload
    )
    install = (moe_metrics.get("architecture") or {}).get("moe_install") or {}
    paths = moe_metrics.get("moe_execution_paths") or {}
    moe["checks"].update({
        "dynamic_selected_expert_execution": int(
            paths.get("dynamic_native_expert_bucketed") or 0
        )
        > 0,
        "does_not_evaluate_all_experts": install.get(
            "evaluates_all_experts"
        )
        is False,
        "no_host_token_routing_loop": install.get(
            "host_token_routing_loop"
        )
        is False,
        "numerical_contract_reported": bool(
            moe_metrics.get("moe_numerical_contract")
        ),
        "deterministic_resume": moe_resume.get("status") == "accepted",
        "reload_continued_training": moe_reload.get("status") == "accepted",
    })
    moe["execution"] = {
        "paths": paths,
        "install": install,
        "contract": moe_metrics.get("moe_execution_contract"),
    }

    product: dict[str, Any] = {}
    _add_authority(product, "compile_startup_profile", arguments.compile_profile)
    _add_authority(product, "clean_wheel", arguments.clean_wheel)
    _add_authority(product, "regressions", arguments.regressions)
    wheel = arguments.wheel
    product["release_candidate"] = {
        "name": wheel.name,
        "sha256": _sha256(wheel),
        "bytes": wheel.stat().st_size,
        "source_commit": arguments.source_commit,
        "python": "CPython 3.12",
        "platform": "Linux x86_64 / Tenstorrent Blackhole P150a",
    }
    product["next_direction"] = _artifact(arguments.next_direction)

    workloads = {
        "small_endurance": small,
        "medium_real_workload": medium,
        "large_low_memory": large,
        "sparse_moe": moe,
    }
    for value in workloads.values():
        value["status"] = (
            "accepted"
            if all(value.get("checks", {}).values())
            and all(
                item.get("status") == "accepted"
                for item in value.values()
                if isinstance(item, dict) and "artifact" in item
            )
            else "failed"
        )
    product_status = all(
        value.get("status") == "accepted"
        for value in product.values()
        if isinstance(value, dict) and "status" in value
    )
    return {
        "format": "tt-finetune-real-workload-acceptance-v1",
        "status": (
            "accepted"
            if all(value["status"] == "accepted" for value in workloads.values())
            and product_status else "failed"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "immutable_reference": {
            "version": "0.2.0",
            "source_commit": "4c5f5371fb97d94bf8611bd2f34d0eec6f64e30c",
            "wheel_sha256": (
                "3f8ac41e30af56bef34c52d6be241c64d652ba7a2c5e37eb15070488726d2d5e"
            ),
        },
        "workloads": workloads,
        "product_qualification": product,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Real-workload acceptance",
        "",
        f"Overall status: **{payload['status']}**",
        "",
        "| Workload | Steps | Sequence | Median step | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for name, item in payload["workloads"].items():
        median = item.get("step_time", {}).get("median_seconds")
        rendered = f"{float(median):.6f}s" if median is not None else "n/a"
        lines.append(
            f"| {name.replace('_', ' ')} | {item.get('steps')} | "
            f"{item.get('sequence_length')} | {rendered} | "
            f"{item.get('status')} |"
        )
    release = payload["product_qualification"]["release_candidate"]
    lines.extend([
        "",
        "## Product gate",
        "",
        f"- Wheel: `{release['name']}`",
        f"- Wheel SHA256: `{release['sha256']}`",
        f"- Release source: `{release['source_commit']}`",
        "- Qualification: fresh CPython 3.12 environment, repository paths "
        "unset/unavailable, doctor smoke, training, checkpoint/resume, "
        "adapter export/reload, planner/cache, and runtime relocation.",
        "",
        "Compilation time is reported separately from synchronized steady-state "
        "training. Complete PJRT executable reuse remains same-process only; "
        "fresh processes reuse lower-level artifacts and measured plans.",
        "",
    ])
    return "\n".join(lines)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "small_metrics",
        "small_resume",
        "small_behavior",
        "small_training_reference",
        "small_adapter_reference",
        "medium_metrics",
        "medium_behavior",
        "medium_memory",
        "medium_training_reference",
        "medium_adapter_reference",
        "large_metrics",
        "large_resume",
        "large_memory",
        "large_reload",
        "moe_metrics",
        "moe_resume",
        "moe_reload",
        "compile_profile",
        "clean_wheel",
        "regressions",
        "wheel",
        "next_direction",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    for name, value in vars(arguments).items():
        if isinstance(value, Path):
            setattr(arguments, name, value.expanduser().resolve())
    payload = build(arguments)
    arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    arguments.markdown_output.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
