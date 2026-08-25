"""Lossless reload of the explicit tt-finetune sparse-MoE PEFT extension."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class SparseAdapterLoadReport:
    loaded: bool
    format: str | None
    tensor_count: int
    parameter_count: int
    weights_sha256: str | None
    reason: str

    def to_dict(self):
        return vars(self)


def load_sparse_moe_adapter_extension(model, adapter_path: str | Path):
    """Load every TT sparse-expert tensor or fail without partial state."""

    root = Path(adapter_path)
    config_path = root / "tt_sparse_moe_config.json"
    if not config_path.is_file():
        return SparseAdapterLoadReport(
            False, None, 0, 0, None, "adapter has no TT sparse-MoE extension"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format") != "tt-finetune-sparse-moe-v1":
        raise ValueError(
            "unsupported TT sparse-MoE adapter format: "
            f"{config.get('format')!r}"
        )
    weights_path = root / config.get(
        "extension_weights", "tt_sparse_moe_adapter.safetensors"
    )
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    from safetensors.torch import load_file

    state = load_file(str(weights_path))
    expected_count = int(config.get("tensor_count", len(state)))
    expected_parameters = int(
        config.get(
            "parameter_count",
            sum(value.numel() for value in state.values()),
        )
    )
    parameters = sum(value.numel() for value in state.values())
    if len(state) != expected_count or parameters != expected_parameters:
        raise RuntimeError(
            "TT sparse-MoE extension metadata does not match its weights"
        )
    model_keys = set(model.state_dict())
    unexpected = sorted(set(state) - model_keys)
    if unexpected:
        raise RuntimeError(
            "TT sparse-MoE tensors do not match the installed canonical "
            f"expert topology: {unexpected[:10]}"
        )
    sparse_suffixes = (
        ".lora_A_gate_up",
        ".lora_B_gate_up",
        ".lora_A_down",
        ".lora_B_down",
    )
    generic_namespaces = (
        ".routed_lora_A.",
        ".routed_lora_B.",
        ".shared_lora_A.",
        ".shared_lora_B.",
    )
    expected_sparse = {
        name for name in model_keys
        if name.endswith(sparse_suffixes)
        or any(namespace in name for namespace in generic_namespaces)
    }
    missing = sorted(expected_sparse - set(state))
    if missing:
        raise RuntimeError(
            "TT sparse-MoE extension is incomplete for the installed "
            f"canonical expert topology: {missing[:10]}"
        )
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            "unexpected TT sparse-MoE keys after validation: "
            f"{result.unexpected_keys[:10]}"
        )
    loaded_state = model.state_dict()
    mismatched = [
        name
        for name, value in state.items()
        if not value.equal(loaded_state[name].detach().cpu())
    ]
    if mismatched:
        raise RuntimeError(
            "TT sparse-MoE extension did not reload exactly: "
            f"{mismatched[:10]}"
        )
    return SparseAdapterLoadReport(
        True,
        config["format"],
        len(state),
        parameters,
        hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "all explicit expert tensors loaded exactly",
    )
