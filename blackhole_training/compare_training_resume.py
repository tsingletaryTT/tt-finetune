#!/usr/bin/env python3
"""Prove deterministic continuation across an interrupted training run.

This verifier compares the suffix of an uninterrupted authority with a run
that resumed from the same optimizer-step boundary.  Wall-clock and compile
fields are intentionally excluded; the authority is the evolving data order,
loss observations, learning-rate schedule, trainable parameters, and logical
optimizer state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"training metrics must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_hash(metrics: dict[str, Any], name: str) -> str | None:
    state = metrics.get("final_training_state") or {}
    value = state.get(name) or {}
    return value.get("sha256")


def _checkpoint_state(path: Path) -> dict[str, Any]:
    """Hash the serialized host checkpoint independently of XLA tensors.

    Runtime state audits intentionally inspect live device tensors.  Exact
    resume qualification also needs a persistence authority, so this hashes
    the trainable model payload and logical optimizer tensors as they were
    actually written to disk.  Metadata such as timestamps is excluded.
    """

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_state = payload.get("model_state_dict") or {}
    serialized_optimizer = payload.get("optimizer_state_dict") or {}
    optimizer_state = serialized_optimizer.get("state") or {}
    model_tensors = [
        (str(name).removeprefix("_orig_mod."), value)
        for name, value in model_state.items()
        if torch.is_tensor(value)
    ]
    runtime_model_tensors = [
        (str(name), value)
        for name, value in model_state.items()
        if torch.is_tensor(value)
    ]
    optimizer_tensors = []
    for parameter, values in sorted(
        optimizer_state.items(), key=lambda item: str(item[0])
    ):
        if not isinstance(values, dict):
            continue
        for name, value in sorted(values.items()):
            if torch.is_tensor(value):
                optimizer_tensors.append(
                    (f"{parameter}.{name}", value)
                )
    resident_state = serialized_optimizer.get("optimizer_state")
    if torch.is_tensor(resident_state):
        optimizer_tensors.append(("resident.optimizer_state", resident_state))
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "step": payload.get("step"),
        "trainable": _tensor_digest(torch, model_tensors),
        "trainable_runtime_wrapper": _tensor_digest(
            torch, runtime_model_tensors
        ),
        "optimizer_logical": _tensor_digest(torch, optimizer_tensors),
    }


def _tensor_digest(torch, named_tensors) -> dict[str, Any]:
    digest = hashlib.sha256()
    tensor_count = 0
    elements = 0
    for name, value in sorted(named_tensors, key=lambda item: item[0]):
        host = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(host.dtype).encode("ascii"))
        digest.update(str(tuple(host.shape)).encode("ascii"))
        digest.update(host.reshape(-1).view(torch.uint8).numpy().tobytes())
        tensor_count += 1
        elements += host.numel()
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": tensor_count,
        "elements": elements,
    }


def _suffix(
    metrics: dict[str, Any], key: str, split_step: int
) -> list[Any]:
    values = metrics.get(key) or []
    if not isinstance(values, list):
        raise TypeError(f"{key} must be a list")
    global_steps = int(metrics.get("global_steps") or 0)
    start_step = global_steps - len(values) + 1
    first_index = max(0, split_step + 1 - start_step)
    return values[first_index:]


def compare(
    authority: dict[str, Any],
    resumed: dict[str, Any],
    *,
    split_step: int,
    authority_checkpoint: dict[str, Any] | None = None,
    resumed_checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    authority_steps = int(authority.get("global_steps") or 0)
    resumed_steps = int(resumed.get("global_steps") or 0)
    authority_losses = _suffix(authority, "step_losses", split_step)
    resumed_losses = _suffix(resumed, "step_losses", split_step)
    authority_rates = _suffix(
        authority, "step_learning_rates", split_step
    )
    resumed_rates = _suffix(resumed, "step_learning_rates", split_step)
    authority_batches = _suffix(
        authority, "batch_identity_history", split_step
    )
    resumed_batches = _suffix(
        resumed, "batch_identity_history", split_step
    )
    checks = {
        "both_completed": (
            authority.get("status") == "completed"
            and resumed.get("status") == "completed"
        ),
        "same_final_step": authority_steps == resumed_steps,
        "split_is_internal": 0 < split_step < authority_steps,
        "same_suffix_length": (
            len(authority_losses) == len(resumed_losses)
            and len(authority_rates) == len(resumed_rates)
            and len(authority_batches) == len(resumed_batches)
            and len(authority_losses) == authority_steps - split_step
        ),
        "loss_observations_exact": authority_losses == resumed_losses,
        "learning_rates_exact": authority_rates == resumed_rates,
        "batch_order_exact": authority_batches == resumed_batches,
        "trainable_parameters_exact": (
            _state_hash(authority, "trainable")
            == _state_hash(resumed, "trainable")
            and _state_hash(authority, "trainable") is not None
        ),
        "optimizer_state_exact": (
            _state_hash(authority, "optimizer_logical")
            == _state_hash(resumed, "optimizer_logical")
            and _state_hash(authority, "optimizer_logical") is not None
        ),
    }
    if (authority_checkpoint is None) != (resumed_checkpoint is None):
        raise ValueError(
            "authority and resumed checkpoints must be supplied together"
        )
    if authority_checkpoint is not None and resumed_checkpoint is not None:
        checks.update({
            "serialized_checkpoint_step_exact": (
                authority_checkpoint.get("step")
                == resumed_checkpoint.get("step")
                == authority_steps
            ),
            "serialized_trainable_parameters_exact": (
                authority_checkpoint["trainable"]["sha256"]
                == resumed_checkpoint["trainable"]["sha256"]
            ),
            "serialized_optimizer_state_exact": (
                authority_checkpoint["optimizer_logical"]["sha256"]
                == resumed_checkpoint["optimizer_logical"]["sha256"]
            ),
        })
    result = {
        "format": "tt-finetune-resume-comparison-v1",
        "status": "accepted" if all(checks.values()) else "failed",
        "contract": (
            "exact deterministic continuation; timing and compilation "
            "metadata excluded"
        ),
        "split_step": split_step,
        "final_step": authority_steps,
        "suffix_steps_compared": len(authority_losses),
        "checks": checks,
        "authority_final": {
            "trainable_sha256": _state_hash(authority, "trainable"),
            "optimizer_logical_sha256": _state_hash(
                authority, "optimizer_logical"
            ),
        },
        "resumed_final": {
            "trainable_sha256": _state_hash(resumed, "trainable"),
            "optimizer_logical_sha256": _state_hash(
                resumed, "optimizer_logical"
            ),
        },
    }
    if authority_checkpoint is not None and resumed_checkpoint is not None:
        result["serialized_checkpoint_authority"] = {
            "authority": authority_checkpoint,
            "resumed": resumed_checkpoint,
            "contract": (
                "host-serialized trainable and logical optimizer tensors; "
                "timestamps and other incidental metadata excluded"
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--split-step", type=int, required=True)
    parser.add_argument("--authority-checkpoint", type=Path)
    parser.add_argument("--resumed-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    authority_path = arguments.authority.expanduser().resolve()
    resumed_path = arguments.resumed.expanduser().resolve()
    result = compare(
        _load(authority_path),
        _load(resumed_path),
        split_step=arguments.split_step,
        authority_checkpoint=(
            _checkpoint_state(
                arguments.authority_checkpoint.expanduser().resolve()
            )
            if arguments.authority_checkpoint is not None else None
        ),
        resumed_checkpoint=(
            _checkpoint_state(
                arguments.resumed_checkpoint.expanduser().resolve()
            )
            if arguments.resumed_checkpoint is not None else None
        ),
    )
    result["authority"] = {
        "path": str(authority_path),
        "sha256": _sha256(authority_path),
    }
    result["resumed"] = {
        "path": str(resumed_path),
        "sha256": _sha256(resumed_path),
    }
    destination = arguments.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
