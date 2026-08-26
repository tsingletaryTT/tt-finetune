from blackhole_training.compare_real_workload_evaluations import compare


def _evaluation(nll, tokens):
    return {
        "model": "model",
        "dataset": {"sha256": "dataset"},
        "configuration": {"dtype": "bfloat16"},
        "supervised_tokens": 4,
        "mean_negative_log_likelihood": nll,
        "total_negative_log_likelihood": nll * 4,
        "perplexity": 2.0,
        "per_example": [{"nll": nll * 4}],
        "generations": [{"generated_token_ids": tokens}],
    }


def test_improving_deterministic_adapter_evaluation_is_accepted():
    before = _evaluation(2.0, [1, 2])
    after = _evaluation(1.5, [1, 3])

    result = compare(
        before,
        after,
        repeat_after=dict(after),
        require_nll_improvement=True,
    )

    assert result["status"] == "accepted"
    assert result["checks"]["heldout_nll_improved"] is True
    assert result["checks"]["repeat_after_is_exact"] is True


def test_quality_gate_rejects_worse_heldout_nll_when_required():
    result = compare(
        _evaluation(2.0, [1, 2]),
        _evaluation(2.1, [1, 3]),
        require_nll_improvement=True,
    )

    assert result["status"] == "failed"
    assert result["checks"]["heldout_nll_improved"] is False
