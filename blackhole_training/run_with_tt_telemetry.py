#!/usr/bin/env python3
"""Run a command while sampling Blackhole board telemetry."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time


DEFAULT_MAX_BOARD_POWER_W = 285.0
DEFAULT_POWER_LIMIT_SAMPLES = 3


def power_limit_sample(
    power_w: float,
    *,
    limit_w: float,
    required_samples: int,
    prior_streak: int,
) -> tuple[int, str | None]:
    """Require a sustained board-power excursion before terminating.

    P150a firmware exposes a 300 W board limit and can report legitimate
    sub-second excursions above its nominal TDP.  Treating one 1 Hz sample
    above 160 W as fatal terminated healthy medium-model jobs.  Temperature
    limits remain immediate; only power uses this bounded debounce.
    """

    if power_w < limit_w:
        return 0, None
    streak = prior_streak + 1
    if streak < required_samples:
        return streak, None
    return (
        streak,
        f"board power {power_w:.1f}W >= {limit_w:.1f}W for "
        f"{streak} consecutive samples",
    )


def _blackhole_hwmon() -> Path:
    for path in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            if (path / "name").read_text().strip() == "blackhole":
                return path
        except OSError:
            continue
    raise RuntimeError("Blackhole hwmon telemetry is unavailable")


def _read_number(path: Path, divisor: float = 1.0) -> float:
    return float(path.read_text().strip()) / divisor


def _available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / (1024.0 * 1024.0)
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def _maximum_host_temperature_c() -> float:
    """Return the hottest readable non-Blackhole hwmon temperature."""
    values: list[float] = []
    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            if (hwmon / "name").read_text().strip() == "blackhole":
                continue
        except OSError:
            continue
        for path in hwmon.glob("temp*_input"):
            try:
                value = _read_number(path, 1_000.0)
            except (OSError, ValueError):
                continue
            if -20.0 <= value <= 150.0:
                values.append(value)
    if not values:
        raise RuntimeError("host temperature telemetry is unavailable")
    return max(values)


def sample_blackhole(hwmon: Path) -> dict:
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "power_w": _read_number(hwmon / "power1_input", 1_000_000.0),
        "current_a": _read_number(hwmon / "curr1_input", 1_000.0),
        "voltage_v": _read_number(hwmon / "in0_input", 1_000.0),
        "asic_temperature_c": _read_number(
            hwmon / "temp1_input", 1_000.0,
        ),
        "fan_rpm": int(_read_number(hwmon / "fan1_input")),
        "host_temperature_c": _maximum_host_temperature_c(),
        "host_available_memory_gib": _available_memory_gib(),
    }


def summarize(samples: list[dict]) -> dict:
    numeric_fields = (
        "power_w",
        "current_a",
        "voltage_v",
        "asic_temperature_c",
        "fan_rpm",
        "host_temperature_c",
        "host_available_memory_gib",
    )
    summary: dict[str, object] = {"sample_count": len(samples)}
    for field in numeric_fields:
        values = [sample[field] for sample in samples if field in sample]
        if values:
            summary[field] = {
                "min": min(values),
                "mean": statistics.fmean(values),
                "max": max(values),
            }
    return summary


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _set_parent_death_signal() -> None:
    """Make a detached accelerator child die if its supervisor disappears."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG)")
    except (AttributeError, OSError):
        # The explicit signal forwarding below remains the portable fallback.
        pass


def _stop_process_group(child: subprocess.Popen, first_signal: int) -> None:
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, first_signal)
        child.wait(timeout=20)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _safe_environment(host_threads: int) -> dict[str, str]:
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[variable] = str(host_threads)
    return environment


def _start_sleep_inhibitor() -> subprocess.Popen:
    """Hold a root-owned inhibitor without running the workload as root.

    The helper blocks on stdin, so an unexpected supervisor death closes the
    pipe and releases the inhibitor instead of leaving sleep disabled.
    """
    inhibitor = subprocess.Popen(
        [
            "sudo", "-n", "systemd-inhibit",
            "--what=sleep",
            "--who=Blackhole training qualification",
            "--why=An active accelerator job cannot survive PCIe suspend",
            "--mode=block",
            "--",
            "sh", "-c", "read -r _",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    time.sleep(0.15)
    if inhibitor.poll() is not None:
        error = inhibitor.stderr.read() if inhibitor.stderr else ""
        raise RuntimeError(f"unable to inhibit system sleep: {error.strip()}")
    return inhibitor


def _release_sleep_inhibitor(inhibitor: subprocess.Popen | None) -> None:
    if inhibitor is None:
        return
    if inhibitor.stdin is not None:
        inhibitor.stdin.close()
    try:
        inhibitor.wait(timeout=3)
    except subprocess.TimeoutExpired:
        _stop_process_group(inhibitor, signal.SIGTERM)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--max-temperature-c", type=float, default=88.0)
    parser.add_argument(
        "--max-power-w",
        type=float,
        default=DEFAULT_MAX_BOARD_POWER_W,
    )
    parser.add_argument(
        "--power-limit-samples",
        type=int,
        default=DEFAULT_POWER_LIMIT_SAMPLES,
        help="Consecutive over-limit samples required before termination.",
    )
    parser.add_argument("--max-host-temperature-c", type=float, default=90.0)
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=900.0,
        help="Terminate a stuck workload after this wall-clock duration.",
    )
    parser.add_argument(
        "--min-host-available-memory-gib", type=float, default=4.0,
    )
    parser.add_argument("--host-threads", type=int, default=4)
    parser.add_argument(
        "--allow-suspend",
        action="store_true",
        help="Do not hold a systemd sleep inhibitor while the command runs.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    if arguments.interval <= 0:
        parser.error("--interval must be positive")
    if arguments.host_threads <= 0:
        parser.error("--host-threads must be positive")
    if arguments.max_runtime_seconds <= 0:
        parser.error("--max-runtime-seconds must be positive")
    if arguments.max_power_w <= 0:
        parser.error("--max-power-w must be positive")
    if arguments.power_limit_samples <= 0:
        parser.error("--power-limit-samples must be positive")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    hwmon = _blackhole_hwmon()
    executed_command = list(command)
    started = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat()
    _write_json_atomic(
        arguments.output,
        {
            "format_version": 1,
            "status": "running",
            "command": command,
            "started_at": started_at,
            "message": (
                "telemetry collection is active; final samples are written "
                "when the owned command exits"
            ),
        },
    )
    parent_pid = os.getppid()
    _set_parent_death_signal()
    if os.getppid() != parent_pid:
        raise RuntimeError("supervisor parent exited before launch")
    inhibitor = None
    if not arguments.allow_suspend:
        inhibitor = _start_sleep_inhibitor()
    child = subprocess.Popen(
        executed_command,
        env=_safe_environment(arguments.host_threads),
        start_new_session=True,
    )
    samples: list[dict] = []
    sampling_errors: list[str] = []
    safety_events: list[str] = []
    power_limit_streak = 0
    power_transient_exceedances = 0
    maximum_power_limit_streak = 0
    forwarded_signal: int | None = None

    def forward_signal(signum: int, _frame: object) -> None:
        nonlocal forwarded_signal
        forwarded_signal = signum

    old_handlers = {
        signum: signal.signal(signum, forward_signal)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        while child.poll() is None:
            if forwarded_signal is not None:
                safety_events.append(
                    f"supervisor received signal {forwarded_signal}"
                )
                _stop_process_group(child, forwarded_signal)
                break
            sample_started = time.perf_counter()
            try:
                sample = sample_blackhole(hwmon)
                samples.append(sample)
                violation = None
                if sample["asic_temperature_c"] >= arguments.max_temperature_c:
                    violation = (
                        f"ASIC temperature {sample['asic_temperature_c']:.1f}C "
                        f">= {arguments.max_temperature_c:.1f}C"
                    )
                elif (
                    sample["host_temperature_c"]
                    >= arguments.max_host_temperature_c
                ):
                    violation = (
                        "host temperature "
                        f"{sample['host_temperature_c']:.1f}C >= "
                        f"{arguments.max_host_temperature_c:.1f}C"
                    )
                elif (
                    sample["host_available_memory_gib"]
                    <= arguments.min_host_available_memory_gib
                ):
                    violation = (
                        "host available memory "
                        f"{sample['host_available_memory_gib']:.2f}GiB <= "
                        f"{arguments.min_host_available_memory_gib:.2f}GiB"
                    )
                elif (
                    time.perf_counter() - started
                    >= arguments.max_runtime_seconds
                ):
                    violation = (
                        "runtime "
                        f"{time.perf_counter() - started:.1f}s >= "
                        f"{arguments.max_runtime_seconds:.1f}s"
                    )
                if violation is None:
                    power_limit_streak, power_violation = power_limit_sample(
                        sample["power_w"],
                        limit_w=arguments.max_power_w,
                        required_samples=arguments.power_limit_samples,
                        prior_streak=power_limit_streak,
                    )
                    maximum_power_limit_streak = max(
                        maximum_power_limit_streak,
                        power_limit_streak,
                    )
                    if (
                        sample["power_w"] >= arguments.max_power_w
                        and power_violation is None
                    ):
                        power_transient_exceedances += 1
                    violation = power_violation
                if violation is not None:
                    safety_events.append(violation)
                    _stop_process_group(child, signal.SIGTERM)
                    break
            except Exception as error:
                sampling_errors.append(f"{type(error).__name__}: {error}")
            remaining = arguments.interval - (
                time.perf_counter() - sample_started
            )
            if remaining > 0:
                time.sleep(remaining)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        _release_sleep_inhibitor(inhibitor)
    exit_code = child.wait()
    if safety_events and exit_code == 0:
        exit_code = 125
    duration = time.perf_counter() - started
    payload = {
        "format_version": 1,
        "status": "completed" if exit_code == 0 else "failed",
        "command": command,
        "executed_command": executed_command,
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(),
        "duration_seconds": duration,
        "exit_code": exit_code,
        "samples": samples,
        "sampling_errors": sampling_errors,
        "safety_events": safety_events,
        "limits": {
            "max_temperature_c": arguments.max_temperature_c,
            "max_power_w": arguments.max_power_w,
            "power_limit_samples": arguments.power_limit_samples,
            "max_host_temperature_c": arguments.max_host_temperature_c,
            "max_runtime_seconds": arguments.max_runtime_seconds,
            "min_host_available_memory_gib": (
                arguments.min_host_available_memory_gib
            ),
            "host_threads": arguments.host_threads,
            "sleep_inhibited": inhibitor is not None,
        },
        "power_limit_observation": {
            "transient_exceedance_samples": power_transient_exceedances,
            "maximum_consecutive_exceedance_samples": (
                maximum_power_limit_streak
            ),
        },
        "summary": summarize(samples),
    }
    _write_json_atomic(arguments.output, payload)
    print(f"Telemetry: {arguments.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
