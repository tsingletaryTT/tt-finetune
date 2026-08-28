from tt_finetune.worker import _steady_measurement_indices


def test_late_compile_bucket_is_excluded_from_steady_measurement():
    assert _steady_measurement_indices(
        10,
        warmup_count=4,
        first_use_indices={0, 2, 7},
    ) == [4, 5, 6, 8, 9]
