"""Lazy model and tokenizer objects used by the notebook API."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any

from .config import IA3Config, LoRAConfig, PEFTAdapterConfig


def _adapter_config(path_or_id: str | Path) -> dict[str, Any]:
    """Read standard PEFT metadata without loading the base model weights."""

    value = str(path_or_id)
    local = Path(value).expanduser()
    config_path = (
        local / "adapter_config.json" if local.is_dir() else local
    )
    if config_path.is_file():
        if config_path.name != "adapter_config.json":
            raise ValueError(
                "adapter_path must be a PEFT adapter directory or Hub ID"
            )
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError(
                "huggingface_hub is required to load an adapter by Hub ID"
            ) from error
        config_path = Path(
            hf_hub_download(value, filename="adapter_config.json")
        )
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    peft_type = str(metadata.get("peft_type", "LORA")).upper()
    if peft_type not in {"LORA", "IA3"}:
        raise NotImplementedError(
            "tt-finetune adapter import accepts standard PEFT LoRA/IA3 "
            "adapters; "
            f"got peft_type={metadata.get('peft_type')!r}"
        )
    return metadata


def _config_from_adapter(path_or_id: str | Path) -> PEFTAdapterConfig:
    metadata = _adapter_config(path_or_id)
    training_metadata = {}
    local = Path(str(path_or_id)).expanduser()
    training_path = local / "tt_finetune_training.json"
    if training_path.is_file():
        training_metadata = json.loads(
            training_path.read_text(encoding="utf-8")
        )
    peft_type = str(metadata.get("peft_type", "LORA")).upper()
    if peft_type == "IA3":
        if metadata.get("target_modules") is None:
            raise ValueError(
                "PEFT IA3 metadata is incomplete: missing target_modules"
            )
        return IA3Config(
            target_modules=metadata["target_modules"],
            feedforward_modules=(
                metadata.get("feedforward_modules") or ()
            ),
            modules_to_save=metadata.get("modules_to_save"),
            # Imported tensors own their values; initialization is bypassed.
            init_ia3_weights=False,
            layers_to_transform=training_metadata.get(
                "layers_to_transform"
            ),
            layers_pattern=training_metadata.get("layers_pattern"),
        )

    required = ("r", "lora_alpha", "target_modules")
    missing = [name for name in required if metadata.get(name) is None]
    if missing:
        raise ValueError(
            "PEFT adapter metadata is incomplete: missing " + ", ".join(missing)
        )
    return LoRAConfig(
        r=int(metadata["r"]),
        lora_alpha=int(metadata["lora_alpha"]),
        target_modules=metadata["target_modules"],
        lora_dropout=float(metadata.get("lora_dropout", 0.0)),
        bias=str(metadata.get("bias", "none")),
        modules_to_save=metadata.get("modules_to_save"),
        use_rslora=bool(metadata.get("use_rslora", False)),
        use_dora=bool(metadata.get("use_dora", False)),
        lora_plus_lr_ratio=training_metadata.get("lora_plus_lr_ratio"),
        layers_to_transform=metadata.get("layers_to_transform"),
        layers_pattern=metadata.get("layers_pattern"),
        # The tensors are loaded from the adapter, so initialization is not
        # part of the continued-training contract.
        init_lora_weights=False,
    )


def _lora_from_adapter(path_or_id: str | Path) -> LoRAConfig:
    """Compatibility helper retained for callers requiring LoRA specifically."""

    config = _config_from_adapter(path_or_id)
    if not isinstance(config, LoRAConfig):
        raise TypeError("the supplied PEFT adapter is not LoRA")
    return config


@dataclass
class TokenizerHandle:
    """Lazy tokenizer that only loads Transformers when notebook code needs it."""

    name_or_path: str
    revision: str | None = None
    trust_remote_code: bool = False
    use_fast: bool = True
    _loaded: Any = field(default=None, init=False, repr=False)

    def load(self) -> Any:
        if self._loaded is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as error:
                raise RuntimeError(
                    "Transformers is available in the TT runtime environment. "
                    "Install it locally only if tokenizer access is needed in "
                    "the host notebook process."
                ) from error
            self._loaded = AutoTokenizer.from_pretrained(
                self.name_or_path,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
                use_fast=self.use_fast,
            )
        return self._loaded

    def __getattr__(self, name: str) -> Any:
        if name == "_loaded":
            raise AttributeError(name)
        return getattr(self.load(), name)


@dataclass(frozen=True)
class AdapterPlan:
    """One named adapter in the host-side lifecycle registry."""

    name: str
    adapter: PEFTAdapterConfig
    path: str | None = None

    @property
    def lora(self) -> PEFTAdapterConfig:
        """Legacy alias; new code should use ``adapter``."""

        return self.adapter


@dataclass(frozen=True)
class BlackholeModel:
    """A lazy, serializable model plan rather than a host-resident weight copy."""

    name_or_path: str
    max_seq_length: int = 256
    dtype: str = "bfloat16"
    revision: str | None = None
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    use_fast_tokenizer: bool = True
    # ``lora`` is retained as a serialized compatibility field.  It now owns
    # any supported PEFT adapter config; ``adapter_config`` is the generic API.
    lora: PEFTAdapterConfig | None = None
    adapter_path: str | None = None
    adapter_name: str = "default"
    adapters: tuple[AdapterPlan, ...] = ()
    training: bool = True

    @property
    def is_peft_model(self) -> bool:
        return self.lora is not None

    @property
    def adapter_config(self) -> PEFTAdapterConfig | None:
        return self.lora

    def load_adapter(
        self,
        path_or_id: str | Path,
        *,
        adapter_name: str = "default",
    ) -> "BlackholeModel":
        """Return a plan that continues training a standard PEFT adapter."""

        return self._add_adapter_plan(
            AdapterPlan(
                name=adapter_name,
                adapter=_config_from_adapter(path_or_id),
                path=str(path_or_id),
            ),
            activate=True,
        )

    def add_adapter(
        self,
        name: str,
        lora_config: LoRAConfig | None = None,
        *,
        adapter_config: PEFTAdapterConfig | None = None,
        activate: bool = False,
        **lora_kwargs: Any,
    ) -> "BlackholeModel":
        supplied = sum(
            value is not None for value in (lora_config, adapter_config)
        )
        if supplied > 1 or (supplied and lora_kwargs):
            raise ValueError(
                "pass adapter_config, lora_config, or LoRA keyword arguments"
            )
        config: PEFTAdapterConfig = (
            adapter_config or lora_config or LoRAConfig(**lora_kwargs)
        )
        return self._add_adapter_plan(
            AdapterPlan(name=name, adapter=config),
            activate=activate or not self.adapters,
        )

    def _add_adapter_plan(
        self,
        plan: AdapterPlan,
        *,
        activate: bool,
    ) -> "BlackholeModel":
        if not plan.name or "." in plan.name:
            raise ValueError(
                "adapter names must be non-empty and cannot contain '.'"
            )
        if plan.name in {item.name for item in self.adapters}:
            raise ValueError(f"adapter already exists: {plan.name}")
        adapters = (*self.adapters, plan)
        if not activate:
            return replace(self, adapters=adapters)
        return replace(
            self,
            lora=plan.adapter,
            adapter_path=plan.path,
            adapter_name=plan.name,
            adapters=adapters,
        )

    def set_adapter(self, name: str) -> "BlackholeModel":
        try:
            plan = next(item for item in self.adapters if item.name == name)
        except StopIteration as error:
            raise KeyError(
                f"unknown adapter {name!r}; available: {self.adapter_names}"
            ) from error
        return replace(
            self,
            lora=plan.adapter,
            adapter_path=plan.path,
            adapter_name=plan.name,
        )

    @property
    def adapter_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.adapters)

    def inspect_adapters(self) -> dict[str, dict[str, Any]]:
        return {
            item.name: {
                "active": item.name == self.adapter_name,
                "path": item.path,
                "adapter_type": item.adapter.adapter_type,
                "adapter": item.adapter,
                "lora": (
                    item.adapter
                    if isinstance(item.adapter, LoRAConfig)
                    else None
                ),
                "trainable_when_selected": True,
                "inactive_state": (
                    "not materialized in the training graph"
                    if item.name != self.adapter_name
                    else "active"
                ),
            }
            for item in self.adapters
        }

    def inspect_adapter(self, path_or_id: str | Path | None = None):
        from .adapters import inspect_adapter

        selected = path_or_id or self.adapter_path
        if selected is None:
            raise ValueError("no adapter path was supplied or loaded")
        return inspect_adapter(selected)

    def merge_adapter(
        self,
        output_dir: str | Path,
        *,
        path_or_id: str | Path | None = None,
    ) -> Path:
        from .adapters import merge_adapter

        selected = path_or_id or self.adapter_path
        if selected is None:
            raise ValueError("no adapter path was supplied or loaded")
        return merge_adapter(
            base_model=self.name_or_path,
            adapter=selected,
            output_dir=output_dir,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
        )


class FastLanguageModel:
    """Unsloth-style entry point backed by a Blackhole execution plan."""

    @staticmethod
    def from_pretrained(
        model_name: str,
        *,
        max_seq_length: int = 256,
        dtype: str = "bfloat16",
        revision: str | None = None,
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
        use_fast_tokenizer: bool = True,
        load_in_4bit: bool = False,
        adapter: str | Path | None = None,
        adapter_path: str | Path | None = None,
        **_: Any,
    ) -> tuple[BlackholeModel, TokenizerHandle]:
        if load_in_4bit:
            raise NotImplementedError(
                "CUDA-style 4-bit QLoRA is not implemented for Blackhole. "
                "Frozen-weight TT precision overrides are a separate feature."
            )
        model = BlackholeModel(
            name_or_path=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            revision=revision,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            use_fast_tokenizer=use_fast_tokenizer,
        )
        if adapter is not None and adapter_path is not None:
            raise ValueError("pass adapter or adapter_path, not both")
        selected_adapter = adapter_path if adapter_path is not None else adapter
        if selected_adapter is not None:
            model = model.load_adapter(selected_adapter)
        tokenizer = TokenizerHandle(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast_tokenizer,
        )
        return model, tokenizer

    @staticmethod
    def get_peft_model(
        model: BlackholeModel,
        *,
        r: int = 8,
        lora_alpha: int = 16,
        target_modules: str | list[str] = "auto",
        lora_dropout: float = 0.0,
        bias: str = "none",
        modules_to_save: list[str] | None = None,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_plus_lr_ratio: float | None = None,
        layers_to_transform: list[int] | None = None,
        layers_pattern: str | None = None,
        init_lora_weights: bool | str = True,
        **_: Any,
    ) -> BlackholeModel:
        if not isinstance(model, BlackholeModel):
            raise TypeError("get_peft_model expects a BlackholeModel")
        lora = LoRAConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            modules_to_save=modules_to_save,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_plus_lr_ratio=lora_plus_lr_ratio,
            layers_to_transform=layers_to_transform,
            layers_pattern=layers_pattern,
            init_lora_weights=init_lora_weights,
        )
        return model.add_adapter(
            "default",
            lora_config=lora,
            activate=True,
        )

    @staticmethod
    def get_ia3_model(
        model: BlackholeModel,
        *,
        target_modules: str | list[str] = "ia3",
        feedforward_modules: str | list[str] = "ia3_feedforward",
        modules_to_save: list[str] | None = None,
        layers_to_transform: list[int] | None = None,
        layers_pattern: str | None = None,
        init_ia3_weights: bool = True,
        backend: str = "generic",
    ) -> BlackholeModel:
        if not isinstance(model, BlackholeModel):
            raise TypeError("get_ia3_model expects a BlackholeModel")
        return model.add_adapter(
            "default",
            adapter_config=IA3Config(
                target_modules=target_modules,
                feedforward_modules=feedforward_modules,
                modules_to_save=modules_to_save,
                layers_to_transform=layers_to_transform,
                layers_pattern=layers_pattern,
                init_ia3_weights=init_ia3_weights,
                backend=backend,
            ),
            activate=True,
        )

    @staticmethod
    def for_training(model: BlackholeModel) -> BlackholeModel:
        return replace(model, training=True)

    @staticmethod
    def for_inference(model: BlackholeModel) -> BlackholeModel:
        return replace(model, training=False)


class FastTTModel(FastLanguageModel):
    """Single-object model loader for the concise public trainer API.

    `FastLanguageModel` keeps the exact Unsloth tuple contract.  This loader
    returns only the lazy model plan; `TTTrainer` creates the matching lazy
    tokenizer automatically.
    """

    @staticmethod
    def from_pretrained(model_name: str, **kwargs: Any) -> BlackholeModel:
        model, _ = FastLanguageModel.from_pretrained(model_name, **kwargs)
        return model

    @staticmethod
    def with_lora(model: BlackholeModel, **kwargs: Any) -> BlackholeModel:
        return FastLanguageModel.get_peft_model(model, **kwargs)

    @staticmethod
    def with_ia3(model: BlackholeModel, **kwargs: Any) -> BlackholeModel:
        return FastLanguageModel.get_ia3_model(model, **kwargs)
