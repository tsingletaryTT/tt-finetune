#!/usr/bin/env python3
"""Turn a pytest JUnit report into a hashed release regression authority."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from collections.abc import Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _suite_totals(junit: Path) -> tuple[dict[str, int], float]:
    root = ET.parse(junit).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {
        key: sum(int(float(suite.attrib.get(key, 0))) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    totals["passed"] = (
        totals["tests"]
        - totals["failures"]
        - totals["errors"]
        - totals["skipped"]
    )
    duration = sum(float(suite.attrib.get("time", 0.0)) for suite in suites)
    return totals, duration


def summarize(
    junit: Path | Sequence[tuple[str, Path]],
    *,
    source_commit: str,
    command: str,
    minimum_tests: int,
) -> dict:
    named = (
        [("owned", junit)]
        if isinstance(junit, Path)
        else list(junit)
    )
    if not named:
        raise ValueError("at least one JUnit authority is required")
    suite_results = []
    totals = {
        key: 0
        for key in ("tests", "failures", "errors", "skipped", "passed")
    }
    duration = 0.0
    for name, path in named:
        counts, seconds = _suite_totals(path)
        for key, value in counts.items():
            totals[key] += value
        duration += seconds
        suite_results.append({
            "scope": name,
            "counts": counts,
            "duration_seconds": seconds,
            "junit": {"name": path.name, "sha256": _sha256(path)},
        })
    checks = {
        "minimum_owned_test_count": totals["tests"] >= minimum_tests,
        "zero_failures": totals["failures"] == 0,
        "zero_errors": totals["errors"] == 0,
    }
    return {
        "format": "tt-finetune-regression-qualification-v1",
        "status": "accepted" if all(checks.values()) else "failed",
        "checks": checks,
        "source_commit": source_commit,
        "command": command,
        "counts": totals,
        "duration_seconds": duration,
        "suites": suite_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--junit",
        action="append",
        required=True,
        metavar="SCOPE=PATH",
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--minimum-tests", type=int, default=598)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    junit = []
    for value in arguments.junit:
        if "=" not in value:
            raise ValueError("--junit must be SCOPE=PATH")
        scope, raw_path = value.split("=", 1)
        if not scope:
            raise ValueError("--junit scope must not be empty")
        junit.append((scope, Path(raw_path).expanduser().resolve()))
    result = summarize(
        junit,
        source_commit=arguments.source_commit,
        command=arguments.command,
        minimum_tests=arguments.minimum_tests,
    )
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
