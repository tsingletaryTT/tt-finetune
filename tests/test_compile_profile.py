from __future__ import annotations

import json

from blackhole_training.build_compile_profile import (
    _cache_identity,
    _public_model_name,
    summarize,
)


def test_public_identities_do_not_publish_development_paths() -> None:
    assert (
        _public_model_name(
            "/tmp/cache/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/"
            "snapshots/abc"
        )
        == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    assert _cache_identity("/tmp/xla_cache/abc123") == "abc123"


def test_compile_profile_requires_complete_stage_evidence(tmp_path) -> None:
    metrics = tmp_path / "training_metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "status": "completed",
                "model_name": "/tmp/models--org--model/snapshots/rev",
                "startup_profile": {
                    "current_stage": "completed",
                    "events": [
                        {
                            "stage": stage,
                            "status": "completed",
                            "stage_seconds": 1.0,
                        }
                        for stage in (
                            "dataset_preparation",
                            "device_compile_and_first_step",
                            "steady_training",
                        )
                    ],
                    "total_seconds": 3.0,
                },
                "compile": {
                    "cache": "/tmp/private/xla-cache",
                    "estimated_compile_seconds": 0.5,
                    "cross_process_executable_reuse": False,
                    "limitation": "not implemented",
                },
                "compile_cache": "/tmp/cache/semantic-key",
                "persistent_runtime": {
                    "client_cache_status": "hit",
                    "prior_jobs_completed": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    result = summarize("medium", metrics)

    assert result["status"] == "accepted"
    assert result["model"] == "org/model"
    assert result["metrics"]["artifact"] == "training_metrics.json"
    assert result["cache"]["identity"] == "semantic-key"
    assert result["cache"]["same_process_executable_reuse_observed"]
    assert "/tmp" not in json.dumps(result)
