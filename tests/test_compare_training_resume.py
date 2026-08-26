from __future__ import annotations

from blackhole_training.compare_training_resume import compare


def _metrics(*, start: int, end: int, final: str = "final"):
    steps = list(range(start, end + 1))
    return {
        "status": "completed",
        "global_steps": end,
        "step_losses": [None if step % 2 else step / 10 for step in steps],
        "step_learning_rates": [step / 1000 for step in steps],
        "batch_identity_history": [
            {"step": step, "host": f"batch-{step}", "device": None}
            for step in steps
        ],
        "final_training_state": {
            "trainable": {"sha256": final},
            "optimizer_logical": {"sha256": f"optimizer-{final}"},
        },
    }


def test_exact_resume_suffix_is_accepted():
    authority = _metrics(start=1, end=10)
    resumed = _metrics(start=6, end=10)
    result = compare(authority, resumed, split_step=5)
    assert result["status"] == "accepted"
    assert result["suffix_steps_compared"] == 5


def test_resume_state_divergence_is_rejected():
    authority = _metrics(start=1, end=10)
    resumed = _metrics(start=6, end=10, final="different")
    result = compare(authority, resumed, split_step=5)
    assert result["status"] == "failed"
    assert not result["checks"]["trainable_parameters_exact"]
    assert not result["checks"]["optimizer_state_exact"]


def test_serialized_checkpoint_authority_is_enforced():
    authority = _metrics(start=1, end=10)
    resumed = _metrics(start=6, end=10)
    checkpoint = {
        "step": 10,
        "trainable": {"sha256": "serialized-trainable"},
        "optimizer_logical": {"sha256": "serialized-optimizer"},
    }
    accepted = compare(
        authority,
        resumed,
        split_step=5,
        authority_checkpoint=checkpoint,
        resumed_checkpoint=dict(checkpoint),
    )
    assert accepted["status"] == "accepted"
    assert accepted["checks"]["serialized_trainable_parameters_exact"]

    divergent = {
        **checkpoint,
        "optimizer_logical": {"sha256": "different"},
    }
    rejected = compare(
        authority,
        resumed,
        split_step=5,
        authority_checkpoint=checkpoint,
        resumed_checkpoint=divergent,
    )
    assert rejected["status"] == "failed"
    assert not rejected["checks"]["serialized_optimizer_state_exact"]
