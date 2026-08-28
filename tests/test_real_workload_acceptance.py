from __future__ import annotations

import json

from blackhole_training.build_real_workload_acceptance import (
    _memory_profile,
    _public_model,
    _training_summary,
)


def test_training_summary_requires_real_evolving_finite_state(tmp_path) -> None:
    path = tmp_path / "training_metrics.json"
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "model_name": "/cache/models--org--model/snapshots/rev",
                "global_steps": 100,
                "batch_size": 1,
                "max_length": 128,
                "step_losses": [1.0, 0.9],
                "step_e2e_seconds": [0.1, 0.1],
                "step_useful_tokens": [100, 100],
                "step_e2e_summary": {"warmup_count": 0},
                "gradient_audit": {"nonfinite_tensor_count": 0},
                "initial_training_state": {
                    "trainable": {"sha256": "before"}
                },
                "final_training_state": {
                    "trainable": {"sha256": "after"},
                    "trainable_finiteness": {
                        "finite": True,
                        "nonfinite_tensor_count": 0,
                    },
                    "optimizer_logical_finiteness": {
                        "finite": True,
                        "nonfinite_tensor_count": 0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    summary, _ = _training_summary(path, minimum_steps=100)

    assert summary["status"] == "accepted"
    assert summary["model"] == "org/model"
    assert summary["useful_tokens_per_second"] == 1000.0
    assert "/cache" not in json.dumps(summary)


def test_training_summary_rejects_unchanged_state(tmp_path) -> None:
    path = tmp_path / "training_metrics.json"
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "global_steps": 100,
                "step_losses": [1.0],
                "gradient_audit": {"nonfinite_tensor_count": 0},
                "initial_training_state": {
                    "trainable": {"sha256": "same"}
                },
                "final_training_state": {
                    "trainable": {"sha256": "same"},
                    "trainable_finiteness": {
                        "finite": True,
                        "nonfinite_tensor_count": 0,
                    },
                    "optimizer_logical_finiteness": {
                        "finite": True,
                        "nonfinite_tensor_count": 0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    summary, _ = _training_summary(path, minimum_steps=100)

    assert summary["status"] == "failed"
    assert not summary["checks"]["trainable_state_changed"]


def test_public_model_identity_strips_snapshot_path() -> None:
    assert (
        _public_model("/tmp/models--HuggingFaceTB--SmolLM2-135M/snapshots/x")
        == "HuggingFaceTB/SmolLM2-135M"
    )


def test_memory_profile_reports_measured_snapshot_and_limitation(tmp_path):
    path = tmp_path / "training_metrics.json"
    path.write_text(
        json.dumps({
            "status": "completed",
            "memory_accounting": {
                "device_allocator": {
                    "available": True,
                    "measurement_contract": "compile-snapshot-v1",
                    "is_exact_runtime_high_water_mark": False,
                    "limitation": "runtime high-water mark is unavailable",
                    "peak_compile_snapshot_dram": {
                        "allocatable_bytes": 100,
                        "allocated_bytes": 60,
                        "free_bytes": 40,
                    },
                },
            },
        }),
        encoding="utf-8",
    )

    result = _memory_profile(path)
    assert result["status"] == "accepted"
    assert result["peak_compile_snapshot_dram"]["allocated_bytes"] == 60
    assert not result["is_exact_runtime_high_water_mark"]
