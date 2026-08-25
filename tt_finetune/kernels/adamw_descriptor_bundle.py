"""Strict loader for pre-serialized exact AdamW physical children."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .tt_lang_program_composition import (
    remap_serialized_child,
    serialized_child_tensor_ids,
)


PHASE_ORDER = ("moments", "denominator", "epsilon", "parameter")
PHASE_TENSORS = {
    "moments": ("state_in", "gradient", "moments"),
    "denominator": ("moments", "denominator_scale", "denominator_raw"),
    "epsilon": ("denominator_raw", "scaled_epsilon", "denominator"),
    "parameter": ("moments", "denominator", "decay_factor", "state_out"),
}
PHASE_CBS = {"moments": 17, "denominator": 4, "epsilon": 3, "parameter": 13}
PHASE_MARKERS = {
    "moments": "adamw_exact_moments",
    "denominator": "adamw_exact_denominator",
    "epsilon": "adamw_exact_epsilon",
    "parameter": "adamw_exact_parameter",
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ExactAdamWDescriptorBundle:
    path: Path
    rows: int
    gradient_prefix_pages: int
    gradient_pages: int
    granularity: int
    descriptors: Mapping[str, dict]
    hashes: Mapping[str, str]

    def remap(self, tensors: Mapping[str, int]) -> list[dict]:
        missing = {
            name for names in PHASE_TENSORS.values() for name in names
            if name not in tensors
        }
        if missing:
            raise ValueError(
                f"AdamW descriptor tensor mapping is missing {sorted(missing)}"
            )
        return [
            remap_serialized_child(
                self.descriptors[phase],
                {
                    local: tensors[name]
                    for local, name in enumerate(PHASE_TENSORS[phase])
                },
            )
            for phase in PHASE_ORDER
        ]


def load_exact_adamw_descriptor_bundle(
    path: str | Path,
    *,
    rows: int | None = None,
    gradient_prefix_pages: int | None = None,
    gradient_pages: int | None = None,
    granularity: int | None = None,
) -> ExactAdamWDescriptorBundle:
    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing AdamW descriptor manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "format_version": 1,
        "backend": "blackhole_exact_adamw_descriptor_bundle_v1",
        "architecture": "blackhole",
        "logical_grid": [8, 8],
        "phase_order": list(PHASE_ORDER),
        "state_layout": ["parameter", "exp_avg", "exp_avg_sq"],
        "execution_performed": False,
    }
    wrong = {
        key: (manifest.get(key), expected)
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if wrong:
        raise ValueError(f"invalid AdamW descriptor manifest contract: {wrong}")
    dimensions = {
        "rows": rows,
        "gradient_prefix_pages": gradient_prefix_pages,
        "gradient_pages": gradient_pages,
        "granularity": granularity,
    }
    for key, expected in dimensions.items():
        actual = manifest.get(key)
        if not isinstance(actual, int) or isinstance(actual, bool) or actual < 0:
            raise ValueError(f"invalid AdamW manifest {key}: {actual!r}")
        if expected is not None and actual != expected:
            raise ValueError(
                f"AdamW manifest {key} mismatch: expected {expected}, got {actual}"
            )
    if manifest["rows"] != manifest["gradient_pages"] * 32:
        raise ValueError("AdamW rows do not match gradient_pages * 32")
    if manifest["granularity"] <= 0:
        raise ValueError("AdamW granularity must be positive")

    declared_hashes = manifest.get("phase_sha256")
    if not isinstance(declared_hashes, dict) or set(declared_hashes) != set(
        PHASE_ORDER
    ):
        raise ValueError("AdamW manifest phase_sha256 contract changed")
    descriptors: dict[str, dict] = {}
    for phase in PHASE_ORDER:
        phase_path = root / f"{phase}.json"
        if not phase_path.is_file():
            raise ValueError(f"missing AdamW descriptor: {phase_path}")
        descriptor = json.loads(phase_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(
            _canonical_json(descriptor).encode("utf-8")
        ).hexdigest()
        if digest != declared_hashes[phase]:
            raise ValueError(f"AdamW descriptor hash mismatch for {phase}")
        if descriptor.get("format_version") not in (1, 2, 3):
            raise ValueError(f"unsupported AdamW {phase} artifact format")
        if descriptor.get("core_range") != {
            "start": [0, 0], "end": [7, 7]
        }:
            raise ValueError(f"AdamW {phase} descriptor grid is not 8x8")
        expected_tensors = len(PHASE_TENSORS[phase])
        if descriptor.get("num_tensors") != expected_tensors:
            raise ValueError(f"AdamW {phase} tensor count changed")
        if len(descriptor.get("cb_configs", ())) != PHASE_CBS[phase]:
            raise ValueError(f"AdamW {phase} CB contract changed")
        if serialized_child_tensor_ids(descriptor) != set(
            range(expected_tensors)
        ):
            raise ValueError(f"AdamW {phase} tensor IDs are not canonical")
        compute = [
            kernel for kernel in descriptor.get("kernels", ())
            if kernel.get("thread_type") == "compute"
        ]
        if len(compute) != 1 or PHASE_MARKERS[phase] not in compute[0].get(
            "cpp_source", ""
        ):
            raise ValueError(f"AdamW {phase} native source marker changed")
        descriptors[phase] = descriptor

    return ExactAdamWDescriptorBundle(
        path=root,
        rows=manifest["rows"],
        gradient_prefix_pages=manifest["gradient_prefix_pages"],
        gradient_pages=manifest["gradient_pages"],
        granularity=manifest["granularity"],
        descriptors=descriptors,
        hashes=dict(declared_hashes),
    )
