"""Locate the binary runtime bundled with the preview wheel."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def bundled_site_packages() -> Path:
    return Path(__file__).resolve().parent / "_runtime" / "site-packages"


def activate_bundled_runtime() -> Path:
    root = bundled_site_packages()
    if not root.is_dir():
        # Source-only development and documentation builds intentionally do
        # not carry the multi-hundred-megabyte binary runtime.
        return root
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    plugin = root / "pjrt_plugin_tt"
    os.environ["TT_PJRT_PLUGIN_DIR"] = str(plugin)
    os.environ["TT_METAL_RUNTIME_ROOT"] = str(plugin / "tt-metal")
    # Native resident programs reuse selected TTNN kernel sources at lowering
    # time.  The binary plugin already vendors the matching TT-Metal tree, so
    # expose that package-relative tree through the conventional source-root
    # variable as well.  Respect an explicit developer override.
    os.environ.setdefault("TT_METAL_HOME", str(plugin / "tt-metal"))
    os.environ.setdefault("PJRT_DEVICE", "TT")
    os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")
    return root


RUNTIME_ROOT = activate_bundled_runtime()


__all__ = ["RUNTIME_ROOT", "activate_bundled_runtime", "bundled_site_packages"]
