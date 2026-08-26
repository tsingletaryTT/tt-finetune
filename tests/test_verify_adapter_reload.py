from __future__ import annotations

from blackhole_training.verify_adapter_reload import verify


def _checkpoint(step: int, digest: str, runtime_digest: str | None = None):
    value = {
        "step": step,
        "trainable": {"sha256": digest},
    }
    if runtime_digest is not None:
        value["trainable_runtime_wrapper"] = {"sha256": runtime_digest}
    return value


def test_adapter_reload_requires_exact_input_and_persisted_output():
    metrics = {
        "status": "completed",
        "global_steps": 2,
        "initial_training_state": {
            "trainable": {"sha256": "source"},
        },
        "final_training_state": {
            "trainable": {"sha256": "updated"},
            "trainable_finiteness": {
                "finite": True,
                "nonfinite_tensor_count": 0,
            },
        },
    }
    result = verify(
        metrics,
        source_checkpoint=_checkpoint(100, "source"),
        final_checkpoint=_checkpoint(2, "updated"),
        minimum_steps=2,
    )
    assert result["status"] == "accepted"

    result = verify(
        metrics,
        source_checkpoint=_checkpoint(100, "different"),
        final_checkpoint=_checkpoint(2, "updated"),
        minimum_steps=2,
    )
    assert result["status"] == "failed"
    assert not result["checks"]["standard_adapter_loaded_exactly"]


def test_adapter_reload_accepts_exact_compiled_wrapper_name_digest():
    metrics = {
        "status": "completed",
        "global_steps": 2,
        "initial_training_state": {
            "trainable": {"sha256": "source-runtime-wrapper"},
        },
        "final_training_state": {
            "trainable": {"sha256": "updated-runtime-wrapper"},
            "trainable_finiteness": {
                "finite": True,
                "nonfinite_tensor_count": 0,
            },
        },
    }

    result = verify(
        metrics,
        source_checkpoint=_checkpoint(
            100, "source-canonical", "source-runtime-wrapper"
        ),
        final_checkpoint=_checkpoint(
            2, "updated-canonical", "updated-runtime-wrapper"
        ),
        minimum_steps=2,
    )

    assert result["status"] == "accepted"
