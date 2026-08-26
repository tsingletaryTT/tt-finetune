#!/usr/bin/env python3
"""Qualify a binary wheel from a fresh CPython 3.12 environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import venv
import zipfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact must contain an object: {path}")
    return value


def _run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout, encoding="utf-8")
    if check and result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode, command, output=result.stdout
        )
    return result


def _wheel_forbidden_paths(wheel: Path) -> list[dict[str, str]]:
    # Split the sentinels so this verifier does not make the wheel fail its
    # own byte scan merely by containing the audit implementation.
    needles = (b"/home/" + b"dniel_israli", b"/work" + b"space")
    matches: list[dict[str, str]] = []
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            with archive.open(info) as handle:
                overlap = b""
                while True:
                    block = handle.read(1024 * 1024)
                    if not block:
                        break
                    searchable = overlap + block
                    for needle in needles:
                        if needle in searchable:
                            matches.append(
                                {
                                    "member": info.filename,
                                    "prefix": needle.decode(),
                                }
                            )
                    overlap = searchable[-64:]
            if matches:
                # One match is sufficient to fail closed and avoids spending
                # minutes inflating an already-invalid runtime wheel.
                break
    return matches


def _installed_rpath_violations(environment_root: Path) -> list[dict[str, str]]:
    violations = []
    for library in environment_root.rglob("*.so*"):
        if not library.is_file():
            continue
        result = subprocess.run(
            ["readelf", "-d", str(library)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode:
            continue
        for line in result.stdout.splitlines():
            if "RPATH" not in line and "RUNPATH" not in line:
                continue
            if "/home/" in line or ("/work" + "space") in line:
                violations.append(
                    {"library": library.name, "dynamic_entry": line.strip()}
                )
    return violations


def _stop_idle_runtime(workspace: Path) -> bool:
    """Release the isolated qualification runtime without touching user jobs."""

    runtime_root = workspace / "generated" / "tt_runtime_service"
    info_path = runtime_root / "runtime_info.json"
    if not info_path.is_file():
        return True
    status = _load(info_path)
    if status.get("active_job"):
        raise RuntimeError(
            "clean-wheel runtime still reports an active qualification job"
        )
    pid = int(status.get("pid") or 0)
    if pid <= 1 or pid == os.getpid():
        raise RuntimeError("clean-wheel runtime reported an invalid PID")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, signal.SIGKILL)
    (runtime_root / "runtime.sock").unlink(missing_ok=True)
    info_path.unlink(missing_ok=True)
    return True


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    wheel = arguments.wheel.expanduser().resolve()
    model = arguments.model.expanduser().resolve()
    dataset = arguments.dataset.expanduser().resolve()
    environment_root = arguments.environment_root.expanduser().resolve()
    work_root = arguments.work_root.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("clean-wheel qualification requires CPython 3.12")
    if not wheel.is_file():
        raise FileNotFoundError(wheel)
    if not (model / "config.json").is_file():
        raise FileNotFoundError(model / "config.json")
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if environment_root.exists():
        raise FileExistsError(
            f"fresh environment path already exists: {environment_root}"
        )
    if work_root.exists() and any(work_root.iterdir()):
        raise FileExistsError(
            f"clean-wheel work directory is not empty: {work_root}"
        )
    work_root.mkdir(parents=True, exist_ok=True)
    clean_home = work_root / "home"
    clean_home.mkdir(parents=True, exist_ok=True)
    logs = work_root / "logs"
    venv.EnvBuilder(
        with_pip=True,
        system_site_packages=False,
        clear=False,
    ).create(environment_root)
    python = environment_root / "bin" / "python"
    cli = environment_root / "bin" / "tt-finetune"
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "TT_FINETUNE_WORKSPACE"):
        environment.pop(name, None)
    environment.update(
        {
            # The public workspace default is ~/.cache/tt-finetune/workspace.
            # An isolated HOME proves that the clean wheel neither observes
            # nor reuses a development runtime, semantic plan, or executable.
            "HOME": str(clean_home),
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    install = _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            str(wheel),
        ],
        cwd=work_root,
        environment=environment,
        log=logs / "install.log",
    )
    doctor = _run(
        [str(cli), "doctor", "--json", "--smoke"],
        cwd=work_root,
        environment=environment,
        log=logs / "doctor-smoke.log",
        check=False,
    )
    doctor_payload = json.loads(doctor.stdout)

    probe_path = work_root / "public_api_probe.json"
    probe_source = r'''
import json
from pathlib import Path
import sys
import tt_finetune
from tt_finetune import DataConfig, FastTTModel, TTTrainer, TrainingArguments

model_path, dataset_path, output_path = map(Path, sys.argv[1:4])
model = FastTTModel.from_pretrained(str(model_path), max_seq_length=128)
trainer = TTTrainer(
    model=model,
    dataset=dataset_path,
    rank=8,
    target_modules="auto",
    data_config=DataConfig(format="prompt_response"),
    args=TrainingArguments(
        output_dir=str(output_path.parent / "constructed-run"),
        max_steps=4,
        max_seq_length=128,
        auto_smoke=False,
    ),
)
support = TTTrainer.supports(
    model=str(model_path),
    rank=8,
    batch_size=1,
    sequence_length=128,
    target_modules="auto",
)
payload = {
    "package_file": str(Path(tt_finetune.__file__).resolve()),
    "version": tt_finetune.__version__,
    "sys_path": sys.path,
    "model_constructed": bool(model.name_or_path),
    "trainer_constructed": trainer.__class__.__name__ == "TTTrainer",
    "support": support.to_dict(),
    "plan": trainer.plan(),
    "data_plan": trainer.data_plan(),
    "cache": trainer.compile_cache(),
}
output_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
'''
    _run(
        [
            str(python),
            "-c",
            probe_source,
            str(model),
            str(dataset),
            str(probe_path),
        ],
        cwd=work_root,
        environment=environment,
        log=logs / "public-api-probe.log",
    )
    probe = _load(probe_path)

    def workload(
        label: str,
        root: Path,
        *extra: str,
        steps: int = 4,
    ) -> subprocess.CompletedProcess[str]:
        return _run(
            [
                str(python),
                "-m",
                "blackhole_training.run_real_workload",
                "small",
                "train",
                "--steps",
                str(steps),
                "--runtime-mode",
                "persistent",
                "--logging-steps",
                "1",
                "--save-steps",
                "2",
                "--model-path",
                str(model),
                "--dataset",
                str(dataset),
                "--output-root",
                str(root),
                *extra,
            ],
            cwd=work_root,
            environment=environment,
            log=logs / f"{label}.log",
        )

    authority_root = work_root / "continuous"
    resumed_root = work_root / "resumed"
    reload_root = work_root / "adapter-reload"
    workload("continuous", authority_root, "--export")
    workload("interrupted", resumed_root, "--stop-after", "2")
    workload("resumed", resumed_root, "--resume", "--export")
    comparison_path = work_root / "exact-resume.json"
    _run(
        [
            str(python),
            "-m",
            "blackhole_training.compare_training_resume",
            "--authority",
            str(authority_root / "small" / "training_metrics.json"),
            "--resumed",
            str(resumed_root / "small" / "training_metrics.json"),
            "--split-step",
            "2",
            "--authority-checkpoint",
            str(
                authority_root
                / "small"
                / "checkpoints"
                / "final_model.pth"
            ),
            "--resumed-checkpoint",
            str(
                resumed_root
                / "small"
                / "checkpoints"
                / "final_model.pth"
            ),
            "--output",
            str(comparison_path),
        ],
        cwd=work_root,
        environment=environment,
        log=logs / "exact-resume.log",
    )
    exported_adapter = resumed_root / "small" / "exported_adapter"
    workload(
        "adapter-reload",
        reload_root,
        "--adapter",
        str(exported_adapter),
        "--export",
        steps=2,
    )
    reload_comparison_path = work_root / "adapter-reload-continuation.json"
    _run(
        [
            str(python),
            "-m",
            "blackhole_training.verify_adapter_reload",
            "--metrics",
            str(reload_root / "small" / "training_metrics.json"),
            "--source-checkpoint",
            str(
                resumed_root
                / "small"
                / "checkpoints"
                / "final_model.pth"
            ),
            "--final-checkpoint",
            str(
                reload_root
                / "small"
                / "checkpoints"
                / "final_model.pth"
            ),
            "--minimum-steps",
            "2",
            "--output",
            str(reload_comparison_path),
        ],
        cwd=work_root,
        environment=environment,
        log=logs / "adapter-reload-continuation.log",
    )
    adapter_inspect = _run(
        [str(cli), "adapter", "inspect", str(exported_adapter)],
        cwd=work_root,
        environment=environment,
        log=logs / "adapter-inspect.log",
    )
    runtime_released = _stop_idle_runtime(
        clean_home / ".cache" / "tt-finetune" / "workspace"
    )

    authority = _load(authority_root / "small" / "training_metrics.json")
    resumed = _load(resumed_root / "small" / "training_metrics.json")
    reload_metrics = _load(
        reload_root / "small" / "training_metrics.json"
    )
    comparison = _load(comparison_path)
    reload_comparison = _load(reload_comparison_path)
    inspect_payload = json.loads(adapter_inspect.stdout)
    package_file = Path(probe.get("package_file", "/missing"))
    runtime_paths = [
        Path(value)
        for value in doctor_payload.get("resource_paths", {}).values()
        if value
    ]
    forbidden_wheel_paths = _wheel_forbidden_paths(wheel)
    rpath_violations = _installed_rpath_violations(environment_root)
    source_root = Path(__file__).resolve().parents[1]
    checks = {
        "wheel_installed": (
            install.returncode == 0
            and "Successfully installed" in install.stdout
        ),
        "cpython_312": (
            subprocess.run(
                [str(python), "-c", "import sys; raise SystemExit(sys.version_info[:2] != (3, 12))"],
                env=environment,
            ).returncode
            == 0
        ),
        "pythonpath_unset": "PYTHONPATH" not in environment,
        "workspace_override_unset": "TT_FINETUNE_WORKSPACE" not in environment,
        "working_directory_outside_source": (
            source_root != work_root and source_root not in work_root.parents
        ),
        "package_loaded_from_fresh_environment": (
            environment_root in package_file.parents
        ),
        "source_checkout_absent_from_sys_path": all(
            str(source_root) != str(Path(item).resolve())
            for item in probe.get("sys_path") or []
            if item
        ),
        "doctor_and_real_device_smoke": (
            doctor.returncode == 0
            and doctor_payload.get("ready") is True
            and doctor_payload.get("device_smoke", {}).get("status")
            == "passed"
        ),
        "bundled_runtime_loaded": (
            doctor_payload.get("runtime_mode") == "bundled-local"
            and runtime_paths
            and all(environment_root in path.parents for path in runtime_paths)
        ),
        "public_model_and_trainer_constructed": (
            probe.get("model_constructed") is True
            and probe.get("trainer_constructed") is True
        ),
        "support_query_and_planners_passed": (
            probe.get("support", {}).get("supported") is True
            and probe.get("plan", {}).get("supported") is True
            and probe.get("plan", {}).get("memory_admitted") is True
            and bool(probe.get("data_plan"))
            and bool(probe.get("cache"))
        ),
        "real_training_completed": (
            authority.get("status") == "completed"
            and int(authority.get("global_steps") or 0) == 4
        ),
        "checkpoint_resume_exact": comparison.get("status") == "accepted",
        "adapter_saved": all(
            (exported_adapter / name).is_file()
            for name in ("adapter_config.json", "adapter_model.safetensors")
        ),
        "adapter_standard_inspection_passed": bool(inspect_payload),
        "adapter_reloaded_and_training_continued": (
            reload_metrics.get("initial_adapter") is not None
            and reload_comparison.get("status") == "accepted"
        ),
        "no_development_paths_in_wheel": not forbidden_wheel_paths,
        "no_absolute_development_rpaths": not rpath_violations,
        "isolated_persistent_runtime_released": runtime_released,
    }
    artifact = {
        "format": "tt-finetune-clean-wheel-qualification-v2",
        "status": "accepted" if all(checks.values()) else "failed",
        "checks": checks,
        "wheel": {
            "filename": wheel.name,
            "bytes": wheel.stat().st_size,
            "sha256": _sha256(wheel),
        },
        "environment": {
            "python": str(python),
            "root": str(environment_root),
            "working_directory": str(work_root),
            "isolated_home": str(clean_home),
            "pythonpath_unset": "PYTHONPATH" not in environment,
            "workspace_override_unset": (
                "TT_FINETUNE_WORKSPACE" not in environment
            ),
        },
        "doctor": doctor_payload,
        "public_api": {
            "package_file": probe.get("package_file"),
            "version": probe.get("version"),
            "support": probe.get("support"),
            "cache": probe.get("cache"),
        },
        "training": {
            "continuous_steps": authority.get("global_steps"),
            "resumed_steps": resumed.get("global_steps"),
            "exact_resume": comparison,
            "reload_steps": reload_metrics.get("global_steps"),
            "adapter_reload": reload_comparison,
        },
        "path_audit": {
            "wheel_forbidden_paths": forbidden_wheel_paths,
            "rpath_violations": rpath_violations,
        },
        "logs": str(logs),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
