from __future__ import annotations

from blackhole_training.compare_training_reference import compare


def _metrics(losses, *, initial="initial"):
    return {
        "step_losses": losses,
        "step_learning_rates": [0.1, 0.2, 0.3, 0.4],
        "batch_identity_history": [
            {"step": step, "host": f"batch-{step}"}
            for step in range(1, 5)
        ],
        "initial_training_state": {
            "trainable": {"sha256": initial},
        },
    }


def test_matched_numerical_trajectory_is_accepted():
    result = compare(
        _metrics([None, 2.0, None, 1.0]),
        _metrics([3.0, 2.1, 1.5, 0.9]),
        minimum_correlation=0.9,
        maximum_mean_absolute_error=0.2,
    )
    assert result["status"] == "accepted"
    assert result["observed_loss_pair_count"] == 2


def test_wrong_batch_order_is_rejected():
    candidate = _metrics([3.0, 2.0, 1.5, 1.0])
    candidate["batch_identity_history"][2]["host"] = "wrong"
    result = compare(candidate, _metrics([3.0, 2.0, 1.5, 1.0]))
    assert result["status"] == "failed"
    assert not result["checks"]["same_batch_order"]


def test_serialized_initial_state_authority_accepts_compile_wrapper_name():
    checkpoint = {
        "trainable": {"sha256": "canonical"},
        "trainable_runtime_wrapper": {"sha256": "compiled"},
    }
    result = compare(
        _metrics([3.0, 2.0, 1.5, 1.0], initial="compiled"),
        _metrics([3.0, 2.0, 1.5, 1.0], initial="canonical"),
        initial_checkpoint=checkpoint,
    )
    assert result["status"] == "accepted"
    assert result["checks"][
        "candidate_initial_state_matches_serialized_authority"
    ]
    assert result["checks"][
        "reference_initial_state_matches_serialized_authority"
    ]
