"""Shape-derived exact AdamW descriptor selection and caching."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from .geometry_planner import GeometryPlan


def _matches(
    path: Path, *, rows: int, prefix_pages: int, gradient_pages: int,
) -> bool:
    try:
        manifest = json.loads((path / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("backend")
        == "blackhole_exact_adamw_descriptor_bundle_v1"
        and manifest.get("rows") == rows
        and manifest.get("gradient_prefix_pages") == prefix_pages
        and manifest.get("gradient_pages") == gradient_pages
        and manifest.get("execution_performed") is False
    )


def descriptor_cache_path(root: str | Path, plan: GeometryPlan) -> Path:
    root = Path(root)
    return root / "generated" / "descriptor_cache" / (
        f"{plan.model.model_family}.pr{plan.physical_rank}."
        f"b{plan.batch_size}.s{plan.sequence_length}."
        f"off{plan.optimizer_gradient_prefix_pages}"
    )


def resolve_descriptor_bundle(
    root: str | Path,
    plan: GeometryPlan,
    *,
    generate: bool = False,
) -> Path:
    """Resolve by physical contract, optionally compiling a cached bundle.

    Logical ranks sharing the same physical tile rank intentionally share one
    descriptor. Batch changes only the scratch-prefix offset and therefore
    receives a distinct cached physical contract without becoming a runtime
    implementation branch.
    """

    if not plan.admitted:
        raise ValueError(f"geometry is not admitted: {plan.admission_reason}")
    root = Path(root).resolve()
    expected = {
        "rows": plan.optimizer_state_rows,
        "prefix_pages": plan.optimizer_gradient_prefix_pages,
        "gradient_pages": plan.optimizer_gradient_pages,
    }
    cache = descriptor_cache_path(root, plan)
    candidates = [cache]
    generated = root / "generated"
    candidates.extend(
        path.parent for path in generated.glob("adamw_descriptors_*/manifest.json")
    )
    candidates.extend(
        path.parent for path in generated.glob(
            "descriptor_cache/*/manifest.json"
        )
    )
    for candidate in candidates:
        if _matches(candidate, **expected):
            return candidate
    if not generate:
        raise FileNotFoundError(
            "no exact AdamW descriptor matches derived geometry: "
            f"rows={expected['rows']}, prefix_pages={expected['prefix_pages']}, "
            f"gradient_pages={expected['gradient_pages']}"
        )

    exporter = root / "blackhole_training" / (
        "export_tt_lang_adamw_descriptors.py"
    )
    if not exporter.is_file():
        raise FileNotFoundError(f"missing descriptor exporter: {exporter}")
    cache.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("PYTHONPATH", str(root))
    command = [
        sys.executable,
        str(exporter),
        "--rows", str(expected["rows"]),
        "--gradient-prefix-pages", str(expected["prefix_pages"]),
        "--output-dir", str(cache),
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode:
        raise RuntimeError(
            "exact AdamW descriptor generation failed:\n"
            + "\n".join(completed.stdout.splitlines()[-80:])
        )
    if not _matches(cache, **expected):
        raise RuntimeError("generated AdamW descriptor contract does not match")
    return cache
