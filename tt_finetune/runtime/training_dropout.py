"""Explicit deterministic dropout buffers for TT-XLA training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def config_has_training_dropout(config: Any) -> bool:
    """Detect positive training-dropout fields without family allowlists."""

    for name, value in vars(config).items():
        lowered = name.lower()
        if lowered.startswith("summary_"):
            continue
        if not (
            "dropout" in lowered
            or "layerdrop" in lowered
            or lowered.endswith("_pdrop")
        ):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if float(value) > 0.0:
            return True
    return False


class ExplicitMaskDropout(nn.Module):
    """Inverted dropout whose mask is an ordinary lifted module buffer."""

    def __init__(self, probability: float, shape: tuple[int, ...]):
        super().__init__()
        if not 0.0 < probability < 1.0:
            raise ValueError("explicit dropout probability must be in (0, 1)")
        self.p = float(probability)
        self.shape = tuple(int(value) for value in shape)
        self.register_buffer(
            "mask",
            torch.ones(self.shape, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return value
        return value * self.mask


@dataclass(frozen=True)
class DropoutInstallReport:
    installed_paths: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    probabilities: tuple[float, ...]

    @property
    def installed_count(self) -> int:
        return len(self.installed_paths)


def install_explicit_training_dropout(
    model: nn.Module,
    *,
    batch_size: int,
    sequence_length: int,
) -> DropoutInstallReport:
    """Discover static dropout shapes on CPU and install buffer-backed ops."""

    candidates = [
        (path, module)
        for path, module in model.named_modules()
        if path
        and isinstance(module, nn.Dropout)
        and float(module.p) > 0.0
    ]
    if not candidates:
        return DropoutInstallReport((), (), ())
    observed: dict[str, list[tuple[int, ...]]] = {
        path: [] for path, _module in candidates
    }
    hooks = []
    for path, module in candidates:
        def record(_module, inputs, *, path=path):
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError(
                    f"dropout {path} did not receive a tensor input"
                )
            observed[path].append(tuple(inputs[0].shape))
        hooks.append(module.register_forward_pre_hook(record))
    was_training = model.training
    model.eval()
    try:
        parameter = next(model.parameters())
        input_ids = torch.zeros(
            (batch_size, sequence_length),
            dtype=torch.long,
            device=parameter.device,
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    replacements = []
    for path, module in candidates:
        calls = observed[path]
        if len(calls) != 1:
            raise RuntimeError(
                f"dropout {path} is called {len(calls)} times per static "
                "forward; shared dropout modules are not yet supported"
            )
        replacement = ExplicitMaskDropout(float(module.p), calls[0])
        replacement.train(was_training)
        replacements.append((path, replacement, calls[0], float(module.p)))
    for path, replacement, _shape, _probability in replacements:
        parent_path, _, child_name = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, child_name, replacement)
    report = DropoutInstallReport(
        installed_paths=tuple(item[0] for item in replacements),
        shapes=tuple(item[2] for item in replacements),
        probabilities=tuple(item[3] for item in replacements),
    )
    model._tt_dropout_install_report = report
    return report


class DropoutMaskController:
    """Populate fresh explicit masks while retaining no mask history."""

    def __init__(
        self,
        model: nn.Module,
        seed: int,
        *,
        initial_micro_step: int = 0,
    ):
        self.seed = int(seed)
        self.micro_step = int(initial_micro_step)
        self.modules = tuple(
            (path, module)
            for path, module in model.named_modules()
            if isinstance(module, ExplicitMaskDropout)
        )
        self.packed_mask: torch.Tensor | None = None

    def _seed_for(self, index: int) -> int:
        value = (
            self.seed
            + 0x1E3779B97F4A7C15 * (self.micro_step + 1)
            + 0x0BF58476D1CE4E5B9 * (index + 1)
        )
        return value & ((1 << 63) - 1)

    def prepare(self, *, training: bool) -> None:
        if not training:
            return
        host_masks = []
        for index, (_path, module) in enumerate(self.modules):
            generator = torch.Generator().manual_seed(self._seed_for(index))
            keep = torch.rand(module.shape, generator=generator) >= module.p
            mask = keep.to(dtype=module.mask.dtype) / (1.0 - module.p)
            host_masks.append(mask.reshape(-1))
        if not host_masks:
            return
        device = self.modules[0][1].mask.device
        self.packed_mask = torch.cat(host_masks).to(device)
        offset = 0
        for _path, module in self.modules:
            length = module.mask.numel()
            # A compiled module lifts registered buffers by object identity.
            # Rebinding ``module.mask`` would leave the executable reading the
            # first mask it captured.  Preserve the buffer and mutate its
            # contents so every forward consumes the current micro-step mask.
            module.mask.copy_(
                self.packed_mask[
                    offset : offset + length
                ].reshape(module.shape)
            )
            offset += length

    def finish(self, *, training: bool) -> None:
        if training:
            self.micro_step += 1


__all__ = [
    "config_has_training_dropout",
    "DropoutInstallReport",
    "DropoutMaskController",
    "ExplicitMaskDropout",
    "install_explicit_training_dropout",
]
