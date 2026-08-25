"""Public standard-PEFT adapter inspection and inference merge utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdapterInfo:
    path: Path
    peft_type: str
    base_model_name_or_path: str | None
    rank: int | None
    alpha: int | None
    target_modules: tuple[str, ...]
    use_rslora: bool
    use_dora: bool
    lora_plus_lr_ratio: float | None
    layers_to_transform: tuple[int, ...] | None
    layers_pattern: str | None
    adapter_features: tuple[str, ...]
    feedforward_modules: tuple[str, ...]
    tensor_count: int
    parameter_count: int
    weights_sha256: str
    tokenizer_metadata_present: bool
    sparse_moe_extension: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            **vars(self),
            "path": str(self.path),
            "target_modules": list(self.target_modules),
        }


def inspect_adapter(path: str | Path) -> AdapterInfo:
    from safetensors.torch import load_file

    root = Path(path).expanduser().resolve()
    config_path = root / "adapter_config.json"
    weights_path = root / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"{root} is not a standard safetensors PEFT adapter"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training_path = root / "tt_finetune_training.json"
    training = (
        json.loads(training_path.read_text(encoding="utf-8"))
        if training_path.is_file()
        else {}
    )
    tensors = load_file(str(weights_path))
    sparse_path = root / "tt_sparse_moe_config.json"
    sparse = (
        json.loads(sparse_path.read_text(encoding="utf-8"))
        if sparse_path.is_file()
        else None
    )
    return AdapterInfo(
        path=root,
        peft_type=str(config.get("peft_type", "UNKNOWN")),
        base_model_name_or_path=config.get("base_model_name_or_path"),
        rank=(int(config["r"]) if config.get("r") is not None else None),
        alpha=(
            int(config["lora_alpha"])
            if config.get("lora_alpha") is not None
            else None
        ),
        target_modules=tuple(sorted(config.get("target_modules") or [])),
        use_rslora=bool(config.get("use_rslora", False)),
        use_dora=bool(config.get("use_dora", False)),
        lora_plus_lr_ratio=(
            float(training["lora_plus_lr_ratio"])
            if training.get("lora_plus_lr_ratio") is not None
            else None
        ),
        layers_to_transform=(
            tuple(
                config.get("layers_to_transform")
                or training.get("layers_to_transform")
            )
            if (
                config.get("layers_to_transform") is not None
                or training.get("layers_to_transform") is not None
            )
            else None
        ),
        layers_pattern=(
            config.get("layers_pattern") or training.get("layers_pattern")
        ),
        adapter_features=tuple(training.get("adapter_features") or ()),
        feedforward_modules=tuple(
            sorted(config.get("feedforward_modules") or [])
        ),
        tensor_count=len(tensors),
        parameter_count=sum(value.numel() for value in tensors.values()),
        weights_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        tokenizer_metadata_present=(
            (root / "tokenizer_config.json").is_file()
            and any(
                (root / name).is_file()
                for name in ("tokenizer.json", "tokenizer.model", "vocab.json")
            )
        ),
        sparse_moe_extension=sparse,
    )


def merge_adapter(
    *,
    base_model: str,
    adapter: str | Path,
    output_dir: str | Path,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> Path:
    """Merge a standard LoRA into a host inference model and save it."""

    info = inspect_adapter(adapter)
    if info.sparse_moe_extension is not None:
        raise NotImplementedError(
            "the explicit TT sparse-MoE extension cannot be silently merged "
            "by standard PEFT; use the TT sparse adapter loader"
        )
    if info.peft_type.upper() != "LORA":
        raise NotImplementedError(
            f"merge currently supports PEFT LoRA, got {info.peft_type}"
        )
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"merge destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=revision,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base, str(info.path))
    merged = peft_model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(destination, safe_serialization=True)
    AutoTokenizer.from_pretrained(
        base_model,
        revision=revision,
        trust_remote_code=trust_remote_code,
    ).save_pretrained(destination)
    (destination / "tt_finetune_merge.json").write_text(
        json.dumps(
            {
                "format": "tt-finetune-merged-inference-v1",
                "base_model": base_model,
                "source_adapter": str(info.path),
                "source_adapter_sha256": info.weights_sha256,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination
