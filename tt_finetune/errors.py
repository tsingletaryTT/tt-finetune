"""Actionable public exceptions for the Blackhole training product."""

from __future__ import annotations

import re


class TTFinetuneError(RuntimeError):
    """Base class for user-facing tt-finetune failures."""


class TTDependencyError(TTFinetuneError):
    pass


class TTDeviceError(TTFinetuneError):
    pass


class TTUnsupportedModelError(TTFinetuneError):
    pass


class TTUnsupportedGeometryError(TTFinetuneError):
    pass


class TTOutOfMemoryError(TTFinetuneError):
    pass


class TTInvalidLoRATargetError(TTFinetuneError):
    pass


class TTCompileError(TTFinetuneError):
    pass


class TTRuntimeError(TTFinetuneError):
    pass


def classify_runtime_error(error: BaseException | str) -> TTFinetuneError:
    """Translate an internal failure into one actionable public exception."""

    text = str(error)
    lowered = text.lower()
    if re.search(r"out of memory|\boom\b|dram capacity|cannot allocate", lowered):
        return TTOutOfMemoryError(
            "The requested model/profile exceeded this P150a's device memory. "
            "Reduce model size, sequence length, batch size, or LoRA rank. "
            f"Runtime detail: {text}"
        )
    if "unsupported" in lowered and any(
        token in lowered for token in ("architecture", "semantic", "model_type")
    ):
        return TTUnsupportedModelError(
            f"This model architecture is outside the current TT contract: {text}"
        )
    if "lora" in lowered and any(
        token in lowered for token in ("target", "module", "topology")
    ):
        return TTInvalidLoRATargetError(
            f"The requested LoRA target does not match the model: {text}"
        )
    if any(
        token in lowered
        for token in (
            "circular buffer",
            "circular buffers",
            "exceeds max l1 size",
            "exceeds l1",
            "unsupported geometry",
            "unsupported shape",
        )
    ):
        return TTUnsupportedGeometryError(
            "The requested tensor geometry does not fit the selected "
            "Blackhole program's per-core L1 budget. Reduce sequence length, "
            "batch size, or LoRA rank, or choose a qualified geometry. "
            f"Runtime detail: {text}"
        )
    if any(token in lowered for token in ("stablehlo", "ttmlir", "lowering", "compile")):
        return TTCompileError(
            "TT compilation failed. Re-run with verbose=True and inspect the "
            f"saved runtime log. Compiler detail: {text}"
        )
    if any(token in lowered for token in ("/dev/tenstorrent", "pjrt_device", "hugepage")):
        return TTDeviceError(
            "The Blackhole runtime/device is unavailable. Run `tt-finetune "
            f"doctor` for the exact missing prerequisite. Detail: {text}"
        )
    return TTRuntimeError(
        f"Blackhole training failed. Re-run with verbose=True. Detail: {text}"
    )


__all__ = [
    "TTCompileError", "TTDependencyError", "TTDeviceError",
    "TTFinetuneError", "TTInvalidLoRATargetError", "TTOutOfMemoryError",
    "TTRuntimeError", "TTUnsupportedGeometryError",
    "TTUnsupportedModelError", "classify_runtime_error",
]
