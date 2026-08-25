"""Runtime discovery and actionable installation diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import json
import subprocess
import sys
from typing import Any

from ._runtime_bootstrap import activate_bundled_runtime


@dataclass(frozen=True)
class RuntimeEnvironment:
    python_compatible: bool
    os_compatible: bool
    architecture_compatible: bool
    device_present: bool
    device_accessible: bool
    hugepages_present: bool
    bundled_runtime_present: bool
    native_libraries_load: bool
    ttnn_importable: bool
    pjrt_plugin_registered: bool
    compiler_available: bool
    runtime_python: str | None
    runtime_mode: str | None
    runtime_versions: dict[str, str]
    resource_paths: dict[str, str]
    ready: bool
    problems: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def discover(cls, workspace: str | Path | None = None) -> "RuntimeEnvironment":
        del workspace  # retained for compatibility with early preview scripts
        runtime = activate_bundled_runtime()
        plugin = runtime / "pjrt_plugin_tt"
        device_path = Path("/dev/tenstorrent")
        hugepage_candidates = (Path("/dev/hugepages-1G"), Path("/dev/hugepages"))
        python_ok = sys.version_info[:2] == (3, 12)
        os_ok = sys.platform.startswith("linux")
        arch_ok = platform.machine() in {"x86_64", "AMD64"}
        runtime_present = all(path.exists() for path in (
            runtime,
            plugin / "pjrt_plugin_tt.so",
            plugin / "lib64" / "libTTMLIRCompiler.so",
            plugin / "lib64" / "libTTMLIRRuntime.so",
            plugin / "tt-metal",
        ))
        device_present = device_path.exists()
        device_nodes = tuple(
            path for path in device_path.glob("[0-9]*") if path.is_char_device()
        ) if device_present else ()
        device_accessible = bool(
            device_nodes
            and all(os.access(path, os.R_OK | os.W_OK) for path in device_nodes)
        )
        hugepages = any(path.exists() for path in hugepage_candidates)
        versions: dict[str, str] = {}
        for distribution in (
            "tt-finetune", "torch", "torch-xla", "pjrt-plugin-tt",
            "transformers", "peft", "datasets", "tokenizers", "safetensors",
        ):
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                versions[distribution] = "not installed"

        native_ok = False
        ttnn_ok = False
        pjrt_ok = False
        compiler_ok = False
        import_errors: list[str] = []
        if runtime_present:
            try:
                ctypes.CDLL(str(plugin / "lib64" / "libTTMLIRRuntime.so"))
                ctypes.CDLL(str(plugin / "lib64" / "libTTMLIRCompiler.so"))
                native_ok = True
                compiler_ok = True
            except OSError as error:
                import_errors.append(f"bundled native library load failed: {error}")
            package_root = Path(__file__).resolve().parents[1]
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        f"import sys; sys.path.insert(0, {str(package_root)!r}); "
                        "import tt_finetune, torch_xla, torch_plugin_tt, ttnn; "
                        "from pjrt_plugin_tt import get_library_path; "
                        "print(get_library_path())"
                    ),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if probe.returncode == 0:
                ttnn_ok = True
                try:
                    pjrt_ok = Path(probe.stdout.strip().splitlines()[-1]).resolve() == (
                        plugin / "pjrt_plugin_tt.so"
                    ).resolve()
                except (IndexError, OSError):
                    pjrt_ok = False
            else:
                import_errors.append(
                    "TTNN/PJRT import probe failed: "
                    + (probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else "unknown error")
                )

        problems: list[str] = []
        if not python_ok:
            problems.append(
                f"Python {platform.python_version()} is unsupported; install CPython 3.12"
            )
        if not os_ok or not arch_ok:
            problems.append(
                f"unsupported host {platform.system()} {platform.machine()}; use Linux x86_64"
            )
        if not runtime_present:
            problems.append("the bundled Tenstorrent user-space runtime is incomplete")
        if not device_present:
            problems.append(
                "/dev/tenstorrent is missing; install/load the Tenstorrent kernel driver"
            )
        elif not device_accessible:
            problems.append(
                "/dev/tenstorrent is not readable/writable by this user; fix device permissions"
            )
        if not hugepages:
            problems.append(
                "no hugepage mount was found; configure /dev/hugepages-1G or /dev/hugepages"
            )
        problems.extend(import_errors)
        return cls(
            python_compatible=python_ok,
            os_compatible=os_ok,
            architecture_compatible=arch_ok,
            device_present=device_present,
            device_accessible=device_accessible,
            hugepages_present=hugepages,
            bundled_runtime_present=runtime_present,
            native_libraries_load=native_ok,
            ttnn_importable=ttnn_ok,
            pjrt_plugin_registered=pjrt_ok,
            compiler_available=compiler_ok,
            runtime_python=sys.executable if runtime_present else None,
            runtime_mode="bundled-local" if runtime_present else None,
            runtime_versions=versions,
            resource_paths={
                "bundled_site_packages": str(runtime),
                "pjrt_plugin": str(plugin / "pjrt_plugin_tt.so"),
                "tt_metal_runtime": str(plugin / "tt-metal"),
                "compiler": str(plugin / "lib64" / "libTTMLIRCompiler.so"),
            },
            ready=not problems,
            problems=tuple(problems),
        )


def run_device_smoke() -> dict[str, Any]:
    """Compile and execute one tiny tensor expression on the TT device."""

    # TT-Metal writes informational messages directly to stdout. Keep the
    # public doctor output machine-readable by containing the device probe in
    # a child process and returning only its explicit result marker.
    package_root = Path(__file__).resolve().parents[1]
    marker = "TT_FINETUNE_SMOKE_RESULT="
    script = f"""
import json, sys, time
sys.path.insert(0, {str(package_root)!r})
import tt_finetune
import torch
import torch_xla.core.xla_model as xm
import torch_plugin_tt
started = time.perf_counter()
device = xm.xla_device()
value = torch.arange(32, dtype=torch.bfloat16, device=device)
result = value + torch.ones_like(value)
xm.mark_step()
xm.wait_device_ops()
host = result.cpu().to(torch.float32)
expected = torch.arange(1, 33, dtype=torch.float32)
exact = bool(torch.equal(host, expected))
print({marker!r} + json.dumps({{
    'status': 'passed' if exact else 'failed',
    'device': str(device),
    'exact': exact,
    'elapsed_seconds': time.perf_counter() - started,
}}))
"""
    probe = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for line in reversed(probe.stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line.removeprefix(marker))
    diagnostic = (probe.stderr or probe.stdout).strip().splitlines()
    return {
        "status": "failed",
        "exact": False,
        "error": diagnostic[-1] if diagnostic else (
            f"device smoke exited with status {probe.returncode}"
        ),
    }


def host_data_dependencies() -> dict[str, bool]:
    return {
        name: importlib.util.find_spec(name) is not None
        for name in ("transformers", "datasets", "peft", "safetensors")
    }


__all__ = ["RuntimeEnvironment", "host_data_dependencies", "run_device_smoke"]
