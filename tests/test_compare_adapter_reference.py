from __future__ import annotations

from blackhole_training.compare_adapter_reference import compare


def _evaluation(offset: float = 0.0) -> dict:
    return {
        "model": "/tmp/models--org--model/snapshots/revision",
        "dataset": {"sha256": "dataset"},
        "configuration": {"max_length": 128, "dtype": "bfloat16"},
        "supervised_tokens": 30,
        "mean_negative_log_likelihood": 1.0 + offset,
        "per_example": [
            {"index": 0, "mean_nll": 0.5 + offset},
            {"index": 1, "mean_nll": 1.0 + offset},
            {"index": 2, "mean_nll": 1.5 + offset},
        ],
        "generations": [
            {"index": 0, "generated_token_ids": [1, 2, 3]},
            {"index": 1, "generated_token_ids": [4, 5]},
        ],
    }


def test_adapter_reference_accepts_close_correlated_behavior() -> None:
    result = compare(_evaluation(0.1), _evaluation())

    assert result["status"] == "accepted"
    assert result["model"] == "org/model"
    assert result["per_example_nll_correlation"] == 1.0
    assert result["exact_generation_count"] == 2


def test_adapter_reference_rejects_large_nll_error() -> None:
    result = compare(_evaluation(1.0), _evaluation())

    assert result["status"] == "failed"
    assert not result["checks"]["per_example_nll_mae_within_contract"]


def test_adapter_reference_uses_scale_normalized_nll_tolerance() -> None:
    reference = _evaluation()
    candidate = _evaluation()
    reference_values = [1.5, 3.0, 4.5]
    candidate_values = [2.0, 4.0, 6.0]
    for row, value in zip(reference["per_example"], reference_values):
        row["mean_nll"] = value
    for row, value in zip(candidate["per_example"], candidate_values):
        row["mean_nll"] = value
    result = compare(candidate, reference)

    assert result["status"] == "accepted"
    assert result["per_example_nll_mean_absolute_error"] == 1.0
    assert result["per_example_nll_normalized_mean_absolute_error"] < 0.4
