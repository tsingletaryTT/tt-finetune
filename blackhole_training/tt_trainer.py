#!/usr/bin/env python3
"""Notebook-friendly, model-agnostic LoRA interface for Blackhole."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .compatibility_registry import CompatibilityRegistry
from .pipeline_engine import Pipeline


class TTPreflightError(RuntimeError):
    """A model, dataset, or requested profile failed preflight validation."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        detail = self.report.get("error") or ""
        message = (
            next(
                (
                    line
                    for line in reversed(detail.splitlines())
                    if line.strip()
                ),
                "",
            )
            or "Blackhole LoRA preflight failed"
        )
        super().__init__(message)


def _positive(name: str, value: int | float | None):
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass
class TTModelConfig:
    """Hugging Face model-loading settings."""

    name_or_path: str
    revision: str | None = None
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    use_fast_tokenizer: bool = True
    initial_adapter_path: str | None = None
    initial_adapter_name: str = "default"

    @classmethod
    def from_object(
        cls,
        model: str | Path | Any,
        overrides: Mapping[str, Any] | "TTModelConfig" | None = None,
    ) -> "TTModelConfig":
        if isinstance(model, cls):
            base = asdict(model)
        elif isinstance(model, Mapping):
            base = dict(model)
        else:
            base = {"name_or_path": TTLoRATrainer._model_name(model)}
        if "name" in base and "name_or_path" not in base:
            base["name_or_path"] = base.pop("name")
        if overrides:
            base.update(
                asdict(overrides)
                if isinstance(overrides, cls)
                else dict(overrides)
            )
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: item for key, item in base.items() if key in allowed})


@dataclass
class TTTrainingArguments:
    """Training arguments using familiar Transformers-style field names."""

    output_dir: str
    run_name: str | None = None
    num_train_epochs: int = 1
    max_steps: int | None = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float | None = None
    max_seq_length: int = 256
    dtype: str = "torch.bfloat16"
    optim: str = "adamw_torch"
    optimizer_layout: str = "auto"
    execution_graph_layout: str = "auto"
    lr_scheduler_type: str = "constant"
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    logging_steps: int = 10
    eval_steps: int = 250
    save_strategy: str = "none"
    save_steps: int = 250
    save_optim: bool = False
    keep_last_n: int = 3
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    run_final_validation: bool = True
    seed: int = 3407
    group_compatible_adapters: bool = True
    weight_dtype_overrides: str | None = None
    memory_mode: str = "auto"
    profile_device_memory: bool = False
    auto_smoke: bool = True
    smoke_steps: int = 2

    def __post_init__(self):
        for name in (
            "num_train_epochs",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "max_seq_length",
        ):
            _positive(name, getattr(self, name))
        _positive("max_steps", self.max_steps)
        _positive("max_grad_norm", self.max_grad_norm)
        _positive("smoke_steps", self.smoke_steps)
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.optim not in {"adamw", "adamw_torch"}:
            raise ValueError("The generic TT trainer currently supports AdamW")
        if self.optimizer_layout not in {
            "auto",
            "tile_packed",
            "per_parameter",
        }:
            raise ValueError(
                "optimizer_layout must be auto, tile_packed, or per_parameter"
            )
        if self.execution_graph_layout not in {
            "auto",
            "phase_split",
            "monolithic",
        }:
            raise ValueError(
                "execution_graph_layout must be auto, phase_split, or "
                "monolithic"
            )
        if self.lr_scheduler_type not in {
            "constant",
            "constant_with_warmup",
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
        }:
            raise ValueError(f"Unsupported lr_scheduler_type: {self.lr_scheduler_type}")
        if self.warmup_steps < 0 or not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_steps/ratio are outside their valid range")
        if self.memory_mode not in {
            "auto", "bf16", "bf8_frozen", "bf4_frozen"
        }:
            raise ValueError(
                "memory_mode must be auto, bf16, bf8_frozen, or bf4_frozen"
            )

    @classmethod
    def from_object(cls, value: Any) -> "TTTrainingArguments":
        if isinstance(value, cls):
            return value
        aliases = {
            "batch_size": "per_device_train_batch_size",
            "max_length": "max_seq_length",
            "num_epochs": "num_train_epochs",
            "steps_freq": "logging_steps",
            "val_steps_freq": "eval_steps",
        }
        if isinstance(value, Mapping):
            raw = dict(value)
        else:
            raw = {
                field_name: getattr(value, field_name)
                for field_name in cls.__dataclass_fields__
                if hasattr(value, field_name)
            }
        for source, target in aliases.items():
            if source in raw and target not in raw:
                raw[target] = raw.pop(source)
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: item for key, item in raw.items() if key in allowed})


@dataclass
class TTLoRAConfig:
    """Internal PEFT settings (legacy name retained for YAML compatibility)."""

    adapter_type: str = "lora"
    r: int = 8
    lora_alpha: int = 16
    target_modules: str | Sequence[str] = "auto"
    lora_dropout: float = 0.0
    bias: str = "none"
    modules_to_save: Sequence[str] | None = None
    use_rslora: bool = False
    use_dora: bool = False
    lora_plus_lr_ratio: float | None = None
    layers_to_transform: Sequence[int] | None = None
    layers_pattern: str | None = None
    init_lora_weights: bool | str = True
    task_type: str = "CAUSAL_LM"
    backend: str = "auto"
    resident_full_mlp_backward: bool = False
    resident_output_attention_backward: bool = False
    ia3_feedforward_modules: str | Sequence[str] = "ia3_feedforward"
    init_ia3_weights: bool = True

    def __post_init__(self):
        self.adapter_type = self.adapter_type.lower().replace("-", "_")
        if self.adapter_type not in {"lora", "ia3"}:
            raise ValueError("adapter_type must be lora or ia3")
        _positive("r", self.r)
        _positive("lora_alpha", self.lora_alpha)
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.bias not in {"none", "all", "lora_only"}:
            raise ValueError("bias must be one of: none, all, lora_only")
        if isinstance(self.target_modules, str) and not self.target_modules:
            raise ValueError("target_modules cannot be empty")
        if self.backend not in {
            "auto",
            "generic",
            "native_experimental",
        }:
            raise ValueError(
                "backend must be auto, generic, or native_experimental"
            )
        if (
            self.resident_full_mlp_backward
            or self.resident_output_attention_backward
        ) and self.backend != "native_experimental":
            raise ValueError(
                "resident LoRA blocks require backend='native_experimental'"
            )
        if self.adapter_type == "ia3":
            if self.backend not in {"auto", "generic"}:
                raise ValueError(
                    "IA3 does not use the native LoRA backend; choose auto or generic"
                )
            if self.use_rslora or self.use_dora or self.lora_plus_lr_ratio is not None:
                raise ValueError("rsLoRA, DoRA and LoRA+ are LoRA-only options")
        if self.lora_plus_lr_ratio is not None:
            if self.lora_plus_lr_ratio <= 0:
                raise ValueError("lora_plus_lr_ratio must be positive")
            if self.lora_plus_lr_ratio == 1:
                raise ValueError(
                    "lora_plus_lr_ratio=1 is ordinary LoRA; use None"
                )
        if self.layers_to_transform is not None:
            layers = tuple(self.layers_to_transform)
            if not layers or any(
                not isinstance(index, int) or index < 0 for index in layers
            ):
                raise ValueError(
                    "layers_to_transform must contain non-negative integers"
                )
            if len(set(layers)) != len(layers):
                raise ValueError("layers_to_transform contains duplicates")
        if self.layers_pattern is not None and not self.layers_pattern:
            raise ValueError("layers_pattern cannot be empty")

    @classmethod
    def from_object(cls, value: Any | None) -> "TTLoRAConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            raw = dict(value)
        else:
            raw = {
                name: getattr(value, name)
                for name in cls.__dataclass_fields__
                if hasattr(value, name)
            }
        if "alpha" in raw and "lora_alpha" not in raw:
            raw["lora_alpha"] = raw.pop("alpha")
        if "dropout" in raw and "lora_dropout" not in raw:
            raw["lora_dropout"] = raw.pop("dropout")
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: item for key, item in raw.items() if key in allowed})


@dataclass
class TTDataConfig:
    """JSONL schema and prompt-formatting settings."""

    format: str = "chat"
    train_file: str | None = None
    eval_file: str | None = None
    messages_field: str = "messages"
    prompt_field: str = "prompt"
    response_field: str = "response"
    text_field: str = "text"
    system_prompt: str | None = None
    assistant_only_loss: bool = True
    chat_template_enable_thinking: bool = False
    fallback_role_separator: str = "\n"
    fallback_assistant_prefix: str = "Assistant:"
    text_transform: str = "none"
    truncation: bool = True
    packing: bool | str = "auto"
    append_eos_token: bool = True
    sequence_bucketing: bool | str = "auto"
    sequence_buckets: list[int] | tuple[int, ...] | None = None
    tokenization_workers: int | str = "auto"
    dataloader_workers: int | str = "auto"
    prefetch_batches: int = 2

    def __post_init__(self):
        if self.format not in {"chat", "prompt_response", "text"}:
            raise ValueError("format must be chat, prompt_response, or text")
        for name in ("packing", "sequence_bucketing"):
            if getattr(self, name) not in {True, False, "auto"}:
                raise ValueError(f"{name} must be True, False, or auto")
        for name in ("tokenization_workers", "dataloader_workers"):
            value = getattr(self, name)
            if value != "auto" and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be auto or a non-negative integer")
        if self.prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be positive")

    @classmethod
    def from_object(cls, value: Any | None) -> "TTDataConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        raw = dict(value) if isinstance(value, Mapping) else asdict(value)
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: item for key, item in raw.items() if key in allowed})


@dataclass
class TTTrainResult:
    run_name: str
    base_model: str
    output_dir: Path
    adapter_dir: Path
    metrics_path: Path
    artifact_report: Path
    run_log: Path
    preflight_report: Path
    artifacts: dict[str, Any] = field(default_factory=dict)


class TTLoRATrainer:
    """Plan, validate, and train a standard PEFT LoRA on Blackhole.

    The model is loaded inside the official TT-XLA container. Passing a model
    object is supported when it exposes ``name_or_path`` or
    ``config._name_or_path``; the live Python weights themselves are not copied
    into the container.
    """

    def __init__(
        self,
        model: str | Path | Any,
        args: TTTrainingArguments | Mapping[str, Any] | Any,
        train_dataset: str | Path | Iterable[Mapping[str, Any]] | Any,
        eval_dataset: str | Path | Iterable[Mapping[str, Any]] | Any | None = None,
        lora_config: TTLoRAConfig | Mapping[str, Any] | Any | None = None,
        data_config: TTDataConfig | Mapping[str, Any] | None = None,
        model_config: TTModelConfig | Mapping[str, Any] | None = None,
        *,
        workspace: str | Path | None = None,
        container_workspace: str | None = None,
    ):
        self.workspace = self._resolve_workspace(workspace)
        # The preview package executes directly from its installed Python
        # environment.  Keeping the historical path translator with the host
        # workspace as its root preserves the qualified config contract while
        # avoiding containers and source-tree mounts.
        self.container_workspace = (
            str(self.workspace)
            if container_workspace is None
            else container_workspace.rstrip("/")
        )
        self.local_execution = container_workspace is None
        self.args = TTTrainingArguments.from_object(args)
        self.lora_config = TTLoRAConfig.from_object(lora_config)
        self.data_config = TTDataConfig.from_object(data_config)
        self.model_config = TTModelConfig.from_object(model, model_config)
        self.model_name = self.model_config.name_or_path
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.run_name = self._run_name()
        self.generated_dir = (
            self.workspace / "blackhole_training" / "generated" / self.run_name
        )
        self._public_output_dir = self._resolve_public_output_path(
            self.args.output_dir
        )
        try:
            self._public_output_dir.relative_to(self.workspace)
            self._staged_output_dir: Path | None = None
        except ValueError:
            # A remote execution backend may only expose its owned workspace.
            # Stage arbitrary output there and publish it afterward. Local
            # wheel execution writes directly to the requested destination.
            self._staged_output_dir = (
                None if self.local_execution
                else self.generated_dir / "external_output"
            )
        self.manifest_path = self.generated_dir / "tt_trainer_pipeline.yaml"
        self.registry = CompatibilityRegistry(
            self.workspace
            / "blackhole_training"
            / "generated"
            / "COMPATIBILITY_REGISTRY.json"
        )
        self._last_profile_id = None

    @staticmethod
    def _resolve_workspace(workspace: str | Path | None) -> Path:
        """Find the checkout/data root without a development PYTHONPATH.

        An explicit argument wins, followed by ``TT_FINETUNE_WORKSPACE``.  A
        wheel installed from a checkout then works naturally when invoked from
        that checkout; source-tree use retains the historical fallback.
        """

        if workspace is not None:
            return Path(workspace).expanduser().resolve()
        configured = os.environ.get("TT_FINETUNE_WORKSPACE")
        if configured:
            return Path(configured).expanduser().resolve()
        root = Path.home() / ".cache" / "tt-finetune" / "workspace"
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()

    def _engine_fingerprint(self) -> str:
        relative_paths = (
            "blackhole_training/tt_trainer.py",
            "blackhole_training/pipeline_engine.py",
            "blackhole_training/compatibility_registry.py",
            "blackhole_training/preflight_lora.py",
            "blackhole_training/export_peft_adapter.py",
            "tt_finetune/config.py",
            "tt_finetune/optim.py",
            "tt_finetune/worker.py",
            "tt_finetune/runtime/loop.py",
            "tt_finetune/runtime/torch_xla_ops.py",
            "tt_finetune/runtime/training_dropout.py",
            "tt_finetune/kernels/streamed_causal_attention.py",
            "blacksmith/tools/checkpoints_manager.py",
            "blacksmith/tools/workaround_utils.py",
            "blacksmith/experiments/torch/qwen/configs.py",
            "blacksmith/experiments/torch/qwen/train.py",
            "blacksmith/datasets/torch/jsonl_causal/jsonl_causal_dataset.py",
            "blacksmith/models/torch/huggingface/hf_models.py",
        )
        digest = hashlib.sha256()
        source_root = Path(__file__).resolve().parents[1]
        for relative in relative_paths:
            path = self.workspace / relative
            if not path.is_file():
                path = source_root / relative
            if not path.is_file():
                raise FileNotFoundError(
                    f"Cannot fingerprint required trainer source: {relative}"
                )
            digest.update(relative.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()[:20]

    def _registry_config(self, path: Path) -> dict[str, Any]:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["engine_fingerprint"] = self._engine_fingerprint()
        return config

    @staticmethod
    def _model_name(model: Any) -> str:
        if isinstance(model, (str, Path)):
            return str(model)
        if getattr(model, "name_or_path", None):
            return str(model.name_or_path)
        config = getattr(model, "config", None)
        if config is not None and getattr(config, "_name_or_path", None):
            return str(config._name_or_path)
        raise TypeError(
            "model must be a Hugging Face id/local path or expose "
            "name_or_path/config._name_or_path"
        )

    def _run_name(self) -> str:
        raw = self.args.run_name or Path(self.args.output_dir).name
        safe = re.sub(r"[^A-Za-z0-9_-]+", "-", raw).strip("-_")
        if not safe:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            safe = f"tt-lora-{stamp}"
        return safe

    def _resolve_public_output_path(self, value: str | Path) -> Path:
        text = str(value)
        if text == self.container_workspace:
            return self.workspace
        if text.startswith(self.container_workspace + "/"):
            return (
                self.workspace / text[len(self.container_workspace) + 1 :]
            ).resolve()
        path = Path(text).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (Path.cwd() / path).resolve()

    @property
    def public_output_dir(self) -> Path:
        return self._public_output_dir

    @property
    def execution_output_dir(self) -> Path:
        return self._staged_output_dir or self._public_output_dir

    @property
    def stages_external_output(self) -> bool:
        return self._staged_output_dir is not None

    def _prepare_execution_output(self, *, resume: bool) -> None:
        """Create a clean workspace-owned execution view of public output."""

        if self._staged_output_dir is None:
            return
        stage = self._staged_output_dir
        if stage.exists():
            shutil.rmtree(stage)
        stage.parent.mkdir(parents=True, exist_ok=True)
        if resume:
            if not self._public_output_dir.is_dir():
                raise FileNotFoundError(
                    "cannot resume because the external output directory does "
                    f"not exist: {self._public_output_dir}"
                )
            shutil.copytree(self._public_output_dir, stage)

    def _publish_execution_output(self) -> None:
        """Publish the completed workspace staging tree to the user's path."""

        if self._staged_output_dir is None:
            return
        if not self._staged_output_dir.is_dir():
            raise FileNotFoundError(
                f"staged training output is missing: {self._staged_output_dir}"
            )
        self._public_output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            self._staged_output_dir,
            self._public_output_dir,
            dirs_exist_ok=True,
        )

    def _to_container_path(self, value: str | Path, *, output: bool = False) -> str:
        text = str(value)
        if text == self.container_workspace or text.startswith(
            self.container_workspace + "/"
        ):
            return text
        path = (
            self._resolve_public_output_path(value)
            if output
            else Path(text).expanduser()
        )
        if not path.is_absolute():
            path = self.workspace / path
        path = path.resolve()
        if self.local_execution:
            return str(path)
        if self._staged_output_dir is not None:
            try:
                relative = path.relative_to(self._public_output_dir)
            except ValueError:
                pass
            else:
                staged = self._staged_output_dir / relative
                staged_relative = staged.relative_to(self.workspace)
                return (
                    f"{self.container_workspace}/"
                    f"{staged_relative.as_posix()}"
                )
        try:
            relative = path.relative_to(self.workspace)
        except ValueError as error:
            kind = "Output" if output else "Local input"
            raise ValueError(
                f"{kind} paths must be inside workspace {self.workspace}: {path}"
            ) from error
        return f"{self.container_workspace}/{relative.as_posix()}"

    def _container_model_name(self) -> str:
        candidate = Path(self.model_name).expanduser()
        if candidate.is_absolute():
            return self._to_container_path(candidate)
        if candidate.exists():
            return self._to_container_path(candidate.resolve())
        return self.model_name

    def _metadata_model_name(self) -> str:
        """Keep Hub IDs portable and make local checkpoint identity stable."""

        candidate = Path(self.model_name).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
        return self.model_name

    def _container_adapter_path(self) -> str | None:
        value = self.model_config.initial_adapter_path
        if value is None:
            return None
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return self._to_container_path(candidate)
        if candidate.exists():
            return self._to_container_path(candidate.resolve())
        return value

    def _host_path(self, value: str | Path) -> Path:
        text = str(value)
        if self.local_execution:
            return self._resolve_public_output_path(value)
        if text == self.container_workspace:
            return self.workspace
        if text.startswith(self.container_workspace + "/"):
            return self.workspace / text[len(self.container_workspace) + 1 :]
        candidate = self._resolve_public_output_path(value)
        if candidate == self._public_output_dir:
            return self._public_output_dir
        path = Path(text).expanduser()
        return path.resolve() if path.is_absolute() else (self.workspace / path).resolve()

    @staticmethod
    def _rows(dataset: Any) -> Iterable[Mapping[str, Any]]:
        if hasattr(dataset, "to_list"):
            return dataset.to_list()
        return dataset

    def _dataset_path(self, dataset: Any, split: str) -> str | None:
        configured = getattr(self.data_config, f"{split}_file")
        source = configured or dataset
        if source is None:
            return None
        if isinstance(source, (str, Path)):
            return self._to_container_path(source)
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        output = self.generated_dir / "inputs" / f"{split}.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with output.open("w", encoding="utf-8") as destination:
            for count, row in enumerate(self._rows(source), start=1):
                if not isinstance(row, Mapping):
                    raise TypeError(
                        f"{split}_dataset row {count} is not a mapping: "
                        f"{type(row).__name__}"
                    )
                destination.write(
                    json.dumps(dict(row), ensure_ascii=False) + "\n"
                )
        if count == 0:
            raise ValueError(f"{split}_dataset is empty")
        return self._to_container_path(output)

    def to_pipeline_manifest(self, *, smoke_steps: int | None = None) -> dict[str, Any]:
        train_file = self._dataset_path(self.train_dataset, "train")
        eval_file = self._dataset_path(self.eval_dataset, "eval")
        if train_file is None:
            raise ValueError("train_dataset or data_config.train_file is required")
        do_eval = eval_file is not None
        max_steps = smoke_steps if smoke_steps is not None else self.args.max_steps
        targets = self.lora_config.target_modules
        if not isinstance(targets, str):
            targets = list(targets)
        output_root = self._to_container_path(self.args.output_dir, output=True)
        stage = {
            "name": "adapter",
            "output_subdir": "",
            "train_file": train_file,
            "eval_file": eval_file,
            "max_length": self.args.max_seq_length,
        }
        manifest = {
            "name": self.run_name,
            "workspace": str(self.workspace),
            "container_workspace": self.container_workspace,
            "seed": self.args.seed,
            "execution": {
                "group_compatible_adapters": (
                    self.args.group_compatible_adapters
                ),
                "require_distinct_adapter_weights": False,
            },
            "foundation": {"enabled": False},
            "model": {
                "name": self._container_model_name(),
                "metadata_name": self._metadata_model_name(),
                "revision": self.model_config.revision,
                "trust_remote_code": self.model_config.trust_remote_code,
                "attn_implementation": self.model_config.attn_implementation,
                "use_fast_tokenizer": self.model_config.use_fast_tokenizer,
                "initial_adapter_path": self._container_adapter_path(),
                "initial_adapter_name": self.model_config.initial_adapter_name,
            },
            "data": {
                "format": self.data_config.format,
                "messages_field": self.data_config.messages_field,
                "prompt_field": self.data_config.prompt_field,
                "response_field": self.data_config.response_field,
                "text_field": self.data_config.text_field,
                "system_prompt": self.data_config.system_prompt,
                "assistant_only_loss": self.data_config.assistant_only_loss,
                "chat_template_enable_thinking": (
                    self.data_config.chat_template_enable_thinking
                ),
                "fallback_role_separator": (
                    self.data_config.fallback_role_separator
                ),
                "fallback_assistant_prefix": (
                    self.data_config.fallback_assistant_prefix
                ),
                "text_transform": self.data_config.text_transform,
                "truncation": self.data_config.truncation,
                "packing": self.data_config.packing,
                "append_eos_token": self.data_config.append_eos_token,
                "sequence_bucketing": self.data_config.sequence_bucketing,
                "sequence_buckets": (
                    list(self.data_config.sequence_buckets)
                    if self.data_config.sequence_buckets is not None
                    else None
                ),
                "tokenization_workers": self.data_config.tokenization_workers,
                "dataloader_workers": self.data_config.dataloader_workers,
                "prefetch_batches": self.data_config.prefetch_batches,
                "do_eval": do_eval,
                "stages": [stage],
            },
            "adapters": {
                "adapter_type": self.lora_config.adapter_type,
                "dtype": self.args.dtype,
                "learning_rate": self.args.learning_rate,
                "batch_size": self.args.per_device_train_batch_size,
                "gradient_accumulation_steps": (
                    self.args.gradient_accumulation_steps
                ),
                "gradient_checkpointing": self.args.gradient_checkpointing,
                "num_epochs": self.args.num_train_epochs,
                "max_steps_per_stage": max_steps,
                "train_samples_per_stage": self.args.max_train_samples,
                "eval_samples_per_stage": self.args.max_eval_samples,
                "optim": self.args.optim,
                "optimizer_layout": self.args.optimizer_layout,
                "execution_graph_layout": (
                    self.args.execution_graph_layout
                ),
                "weight_decay": self.args.weight_decay,
                "adam_beta1": self.args.adam_beta1,
                "adam_beta2": self.args.adam_beta2,
                "adam_epsilon": self.args.adam_epsilon,
                "lr_scheduler_type": self.args.lr_scheduler_type,
                "warmup_steps": self.args.warmup_steps,
                "warmup_ratio": self.args.warmup_ratio,
                "max_grad_norm": self.args.max_grad_norm,
                "run_final_validation": (
                    self.args.run_final_validation and do_eval
                ),
                "rank": self.lora_config.r,
                "alpha": self.lora_config.lora_alpha,
                "targets": targets,
                "dropout": self.lora_config.lora_dropout,
                "bias": self.lora_config.bias,
                "modules_to_save": (
                    list(self.lora_config.modules_to_save)
                    if self.lora_config.modules_to_save
                    else None
                ),
                "use_rslora": self.lora_config.use_rslora,
                "use_dora": self.lora_config.use_dora,
                "lora_plus_lr_ratio": (
                    self.lora_config.lora_plus_lr_ratio
                ),
                "layers_to_transform": (
                    list(self.lora_config.layers_to_transform)
                    if self.lora_config.layers_to_transform is not None
                    else None
                ),
                "layers_pattern": self.lora_config.layers_pattern,
                "init_weights": self.lora_config.init_lora_weights,
                "task_type": self.lora_config.task_type,
                "backend": self.lora_config.backend,
                "resident_full_mlp_backward": (
                    self.lora_config.resident_full_mlp_backward
                ),
                "resident_output_attention_backward": (
                    self.lora_config.resident_output_attention_backward
                ),
                "ia3_feedforward_modules": (
                    self.lora_config.ia3_feedforward_modules
                ),
                "init_ia3_weights": self.lora_config.init_ia3_weights,
                "output_root": output_root,
                "steps_freq": self.args.logging_steps,
                "val_steps_freq": self.args.eval_steps,
                "save_strategy": self.args.save_strategy,
                "save_steps": self.args.save_steps,
                "save_optim": self.args.save_optim,
                "keep_last_n": self.args.keep_last_n,
                "weight_dtype_overrides": self.args.weight_dtype_overrides,
                "memory_mode": self.args.memory_mode,
                "profile_device_memory": self.args.profile_device_memory,
            },
        }
        return manifest

    def save_pipeline_manifest(
        self,
        path: str | Path | None = None,
        *,
        smoke_steps: int | None = None,
    ) -> Path:
        destination = Path(path).resolve() if path else self.manifest_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(
                self.to_pipeline_manifest(smoke_steps=smoke_steps),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        return destination

    def save_config(self, path: str | Path) -> Path:
        """Write the concise public YAML schema, suitable for ``tt_train.py``."""
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        public = {
            "model": asdict(self.model_config),
            "data": asdict(self.data_config),
            "lora": asdict(self.lora_config),
            "training": asdict(self.args),
        }
        destination.write_text(
            yaml.safe_dump(public, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        workspace: str | Path | None = None,
    ) -> "TTLoRATrainer":
        source = Path(path).resolve()
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        model = raw.get("model", {})
        model_name = model.get("name_or_path") or model.get("name")
        if not model_name:
            raise ValueError("model.name_or_path is required")
        data = TTDataConfig.from_object(raw.get("data"))
        return cls(
            model=model_name,
            args=raw.get("training", {}),
            train_dataset=data.train_file,
            eval_dataset=data.eval_file,
            lora_config=raw.get("lora"),
            data_config=data,
            model_config=model,
            workspace=workspace,
        )

    def _prepare(self, smoke_steps: int | None = None):
        manifest = self.save_pipeline_manifest(smoke_steps=smoke_steps)
        pipeline = Pipeline(manifest)
        warmup_path, adapter_paths = pipeline.render()
        try:
            resolved_paths = pipeline.preflight(adapter_paths)
        except subprocess.CalledProcessError as error:
            report_path = pipeline.generated_dir / "preflight_adapter.json"
            if report_path.is_file():
                raise TTPreflightError(
                    json.loads(report_path.read_text(encoding="utf-8"))
                ) from None
            raise TTPreflightError(
                {
                    "status": "failed",
                    "exit_code": error.returncode,
                    "error": str(error),
                }
            ) from None
        report_path = pipeline.generated_dir / "preflight_adapter.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        resolved_config = self._registry_config(resolved_paths["adapter"])
        self._last_profile_id = self.registry.record_preflight(
            report,
            resolved_config,
        )
        return pipeline, warmup_path, resolved_paths

    def preflight(self, *, smoke_steps: int | None = None) -> dict[str, Any]:
        pipeline, _, _ = self._prepare(smoke_steps=smoke_steps)
        report_path = pipeline.generated_dir / "preflight_adapter.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        registry = self.registry.load()
        model = registry["models"].get(
            self.registry.model_key(report),
            {},
        )
        profile = model.get("profiles", {}).get(self._last_profile_id, {})
        report["compatibility_profile_id"] = self._last_profile_id
        report["hardware_status"] = profile.get("hardware_status", "not_tested")
        if profile.get("metrics"):
            report["known_hardware_metrics"] = profile["metrics"]
            steady = profile["metrics"].get("steady_step_seconds")
            planned_steps = (
                smoke_steps
                if smoke_steps is not None
                else self.args.max_steps
            )
            if steady is not None and planned_steps is not None:
                report["projected_steady_training_seconds"] = (
                    steady * planned_steps
                )
        summary_path = pipeline.generated_dir / "PREFLIGHT_SUMMARY.json"
        summary_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return report

    def train(
        self,
        *,
        resume: bool = False,
        smoke_steps: int | None = None,
    ) -> TTTrainResult:
        pipeline, warmup_path, adapter_paths = self._prepare(
            smoke_steps=smoke_steps
        )
        pipeline.allow_state_migration = resume
        preflight_path = pipeline.generated_dir / "preflight_adapter.json"
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        resolved_config = self._registry_config(adapter_paths["adapter"])
        registry = self.registry.load()
        profile = (
            registry["models"]
            .get(self.registry.model_key(preflight), {})
            .get("profiles", {})
            .get(self._last_profile_id, {})
        )
        if (
            smoke_steps is None
            and self.args.auto_smoke
            and profile.get("hardware_status") != "passed"
        ):
            pipeline.write_log(
                "Automatic hardware smoke",
                "started",
                {
                    "profile": self._last_profile_id,
                    "steps": self.args.smoke_steps,
                },
            )
            smoke_result = self.smoke(steps=self.args.smoke_steps)
            pipeline.write_log(
                "Automatic hardware smoke",
                "completed",
                {"adapter": smoke_result.adapter_dir},
            )
        output_dir = self._host_path(self.args.output_dir)
        if (
            self.stages_external_output
            and output_dir.exists()
            and any(output_dir.iterdir())
            and not resume
        ):
            raise FileExistsError(f"Output already exists: {output_dir}")
        self._prepare_execution_output(resume=resume)
        try:
            pipeline.run_foundation(warmup_path, resume=resume)
            pipeline.run_adapters(adapter_paths, resume=resume)
            artifacts = pipeline.verify_artifacts()
        except Exception as error:
            self.registry.record_hardware(
                preflight,
                resolved_config,
                status="failed",
                error=error,
            )
            raise
        self._publish_execution_output()
        metrics_path = output_dir / "training_metrics.json"
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.is_file()
            else None
        )
        self.registry.record_hardware(
            preflight,
            resolved_config,
            status="passed",
            metrics=metrics,
            adapter_path=output_dir / "peft_adapter",
        )
        pipeline.write_log(
            "TTLoRATrainer",
            "completed",
            {"resume": resume, "smoke_steps": smoke_steps},
        )
        return TTTrainResult(
            run_name=self.run_name,
            base_model=pipeline.adapter_base_model,
            output_dir=output_dir,
            adapter_dir=output_dir / "peft_adapter",
            metrics_path=output_dir / "training_metrics.json",
            artifact_report=pipeline.generated_dir / "ARTIFACTS.md",
            run_log=pipeline.run_log,
            preflight_report=pipeline.generated_dir / "preflight_adapter.json",
            artifacts=artifacts,
        )

    def smoke(self, *, steps: int = 2, resume: bool = False) -> TTTrainResult:
        """Run a compatibility gate in a sibling output, preserving full output."""
        _positive("steps", steps)
        suffix = "-smoke"
        output = str(self.args.output_dir)
        smoke_output = output if output.endswith(suffix) else output + suffix
        run_name = self.run_name if self.run_name.endswith(suffix) else self.run_name + suffix
        smoke_args = replace(
            self.args,
            output_dir=smoke_output,
            run_name=run_name,
            max_steps=steps,
            save_strategy="none",
        )
        trainer = type(self)(
            model=self.model_config,
            args=smoke_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            lora_config=self.lora_config,
            data_config=self.data_config,
            workspace=self.workspace,
            container_workspace=self.container_workspace,
        )
        return trainer.train(resume=resume, smoke_steps=steps)


TTTrainer = TTLoRATrainer
