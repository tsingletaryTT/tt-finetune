from __future__ import annotations

from pathlib import Path
import zipfile

from blackhole_training.qualify_clean_wheel import _wheel_forbidden_paths
from blackhole_training.qualify_clean_wheel import _stop_idle_runtime


def test_wheel_path_audit_rejects_development_prefix(tmp_path: Path) -> None:
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "package/module.py",
            "RESOURCE = " + repr("/home/" + "dniel_israli/build"),
        )

    matches = _wheel_forbidden_paths(wheel)

    assert matches == [
        {
            "member": "package/module.py",
            "prefix": "/home/" + "dniel_israli",
        }
    ]


def test_wheel_path_audit_accepts_package_relative_resources(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("package/module.py", "RESOURCE = '$ORIGIN/runtime'")

    assert _wheel_forbidden_paths(wheel) == []


def test_runtime_cleanup_accepts_an_unused_workspace(tmp_path) -> None:
    assert _stop_idle_runtime(tmp_path / "workspace")
