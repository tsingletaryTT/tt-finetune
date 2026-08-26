from __future__ import annotations

from blackhole_training.record_regressions import summarize


def test_regression_authority_accepts_zero_failure_suite(tmp_path) -> None:
    report = tmp_path / "junit.xml"
    report.write_text(
        '<testsuites><testsuite tests="600" failures="0" errors="0" '
        'skipped="2" time="12.5" /></testsuites>',
        encoding="utf-8",
    )

    result = summarize(
        report,
        source_commit="abc",
        command="pytest",
        minimum_tests=598,
    )

    assert result["status"] == "accepted"
    assert result["counts"] == {
        "tests": 600,
        "failures": 0,
        "errors": 0,
        "skipped": 2,
        "passed": 598,
    }


def test_regression_authority_rejects_failed_or_too_small_suite(
    tmp_path,
) -> None:
    report = tmp_path / "junit.xml"
    report.write_text(
        '<testsuite tests="10" failures="1" errors="0" skipped="0" />',
        encoding="utf-8",
    )

    result = summarize(
        report,
        source_commit="abc",
        command="pytest",
        minimum_tests=598,
    )

    assert result["status"] == "failed"
    assert not result["checks"]["minimum_owned_test_count"]
    assert not result["checks"]["zero_failures"]


def test_regression_authority_aggregates_separate_release_and_engine_suites(
    tmp_path,
) -> None:
    release = tmp_path / "release.xml"
    engine = tmp_path / "engine.xml"
    release.write_text(
        '<testsuite tests="61" failures="0" errors="0" skipped="0" '
        'time="1.5" />',
        encoding="utf-8",
    )
    engine.write_text(
        '<testsuite tests="598" failures="0" errors="0" skipped="0" '
        'time="4.0" />',
        encoding="utf-8",
    )

    result = summarize(
        [("release-source", release), ("frozen-engine", engine)],
        source_commit="abc",
        command="two isolated pytest invocations",
        minimum_tests=650,
    )

    assert result["status"] == "accepted"
    assert result["counts"]["tests"] == 659
    assert [item["scope"] for item in result["suites"]] == [
        "release-source",
        "frozen-engine",
    ]
