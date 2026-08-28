import json

from blackhole_training.run_with_tt_telemetry import (
    _write_json_atomic,
    power_limit_sample,
)


def test_telemetry_state_replaces_a_stale_completed_run(tmp_path):
    destination = tmp_path / "telemetry.json"
    destination.write_text('{"status": "completed", "old": true}\n')

    _write_json_atomic(destination, {"status": "running"})

    assert json.loads(destination.read_text()) == {"status": "running"}
    assert not list(tmp_path.glob("*.tmp"))


def test_power_limit_ignores_a_single_transient_excursion():
    streak, violation = power_limit_sample(
        171.0,
        limit_w=160.0,
        required_samples=3,
        prior_streak=0,
    )

    assert streak == 1
    assert violation is None


def test_power_limit_resets_after_a_safe_sample():
    streak, violation = power_limit_sample(
        150.0,
        limit_w=160.0,
        required_samples=3,
        prior_streak=2,
    )

    assert streak == 0
    assert violation is None


def test_power_limit_stops_a_sustained_excursion():
    streak = 0
    violation = None
    for power in (290.0, 291.0, 292.0):
        streak, violation = power_limit_sample(
            power,
            limit_w=285.0,
            required_samples=3,
            prior_streak=streak,
        )

    assert streak == 3
    assert violation == (
        "board power 292.0W >= 285.0W for 3 consecutive samples"
    )
