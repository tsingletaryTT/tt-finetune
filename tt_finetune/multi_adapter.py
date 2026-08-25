"""One-call shared-foundation and multi-LoRA notebook training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import yaml

from blackhole_training.pipeline_engine import Pipeline

from .config import DataConfig, LoRAConfig, TrainingArguments


@dataclass(frozen=True)
class AdapterStage:
    name: str
    train_file: str
    eval_file: str | None = None
    max_seq_length: int = 256
    output_subdir: str | None = None
    train_samples: int | None = None
    eval_samples: int | None = None


@dataclass(frozen=True)
class TokenWarmupConfig:
    initialized_model: str
    warmed_model: str
    token_contract: str
    output_dir: str
    learning_rate: float = 5e-5
    batch_size: int = 8
    max_seq_length: int = 256
    train_samples_per_stage: int | None = None
    eval_samples_per_stage: int | None = None
    max_steps: int | None = None
    dtype: str = "torch.bfloat16"
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    num_train_epochs: int = 1
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float | None = None
    lr_scheduler_type: str = "constant"
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    run_final_validation: bool = True
    token_row_mode: str = "joint"

    def __post_init__(self):
        if self.token_row_mode not in {"joint", "output_tied"}:
            raise ValueError(
                "token_row_mode must be 'joint' or 'output_tied'"
            )


@dataclass(frozen=True)
class MultiAdapterTrainResult:
    run_name: str
    foundation_dir: Path
    adapter_dirs: dict[str, Path]
    manifest_path: Path
    run_log: Path
    artifacts: dict


class MultiAdapterSFTTrainer:
    """Train a shared token foundation and independent LoRAs on Blackhole.

    Compatible stages are grouped into one TT process so they reuse the
    compiled static graph. Outputs remain ordinary Hugging Face/PEFT folders.
    """

    def __init__(
        self,
        *,
        run_name: str,
        stages: Sequence[AdapterStage],
        lora_config: LoRAConfig,
        training_args: TrainingArguments,
        foundation: TokenWarmupConfig,
        data_config: DataConfig | None = None,
        workspace: str | Path | None = None,
        output_root: str | Path | None = None,
        container_workspace: str | None = None,
    ):
        if not stages:
            raise ValueError("at least one adapter stage is required")
        self.workspace = (
            Path(workspace).expanduser().resolve()
            if workspace
            else Path(__file__).resolve().parents[1]
        )
        self.run_name = run_name
        self.stages = list(stages)
        self.lora_config = lora_config
        self.training_args = training_args
        self.foundation = foundation
        self.data_config = data_config or DataConfig(format="chat")
        self.container_workspace = (
            str(self.workspace)
            if container_workspace is None
            else container_workspace.rstrip("/")
        )
        self.output_root = Path(
            output_root or training_args.output_dir
        ).expanduser().resolve()
        self.generated_dir = (
            self.workspace / "blackhole_training" / "generated" / run_name
        )
        self.manifest_path = self.generated_dir / "tt_multi_adapter.yaml"

    def _container_path(self, value: str | Path) -> str:
        text = str(value)
        if text == self.container_workspace or text.startswith(
            self.container_workspace + "/"
        ):
            return text
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.workspace / path
        path = path.resolve()
        try:
            relative = path.relative_to(self.workspace)
        except ValueError as error:
            raise ValueError(
                f"All local paths must be inside workspace {self.workspace}: {path}"
            ) from error
        return f"{self.container_workspace}/{relative.as_posix()}"

    def manifest(self) -> dict:
        targets = self.lora_config.target_modules
        if not isinstance(targets, str):
            targets = list(targets)
        data = asdict(self.data_config)
        data.pop("text_field", None)
        data["stages"] = [
            {
                "name": stage.name,
                "train_file": self._container_path(stage.train_file),
                "eval_file": (
                    self._container_path(stage.eval_file)
                    if stage.eval_file
                    else None
                ),
                "max_length": stage.max_seq_length,
                **(
                    {"output_subdir": stage.output_subdir}
                    if stage.output_subdir is not None
                    else {}
                ),
                **(
                    {"train_samples": stage.train_samples}
                    if stage.train_samples is not None
                    else {}
                ),
                **(
                    {"eval_samples": stage.eval_samples}
                    if stage.eval_samples is not None
                    else {}
                ),
            }
            for stage in self.stages
        ]
        args = self.training_args
        lora = self.lora_config
        return {
            "name": self.run_name,
            "workspace": str(self.workspace),
            "container_workspace": self.container_workspace,
            "seed": args.seed,
            "execution": {
                "group_compatible_adapters": True,
                "require_distinct_adapter_weights": True,
            },
            "foundation": {
                "initialized_path": self._container_path(
                    self.foundation.initialized_model
                ),
                "warmed_path": self._container_path(
                    self.foundation.warmed_model
                ),
                "token_contract": self._container_path(
                    self.foundation.token_contract
                ),
                "warmup_project_dir": self._container_path(
                    self.foundation.output_dir
                ),
                "warmup": {
                    "max_length": self.foundation.max_seq_length,
                    "dtype": self.foundation.dtype,
                    "learning_rate": self.foundation.learning_rate,
                    "batch_size": self.foundation.batch_size,
                    "train_samples_per_stage": (
                        self.foundation.train_samples_per_stage
                    ),
                    "eval_samples_per_stage": (
                        self.foundation.eval_samples_per_stage
                    ),
                    "max_steps": self.foundation.max_steps,
                    "gradient_accumulation_steps": (
                        self.foundation.gradient_accumulation_steps
                    ),
                    "gradient_checkpointing": (
                        self.foundation.gradient_checkpointing
                    ),
                    "num_epochs": self.foundation.num_train_epochs,
                    "weight_decay": self.foundation.weight_decay,
                    "adam_beta1": self.foundation.adam_beta1,
                    "adam_beta2": self.foundation.adam_beta2,
                    "adam_epsilon": self.foundation.adam_epsilon,
                    "max_grad_norm": self.foundation.max_grad_norm,
                    "lr_scheduler_type": (
                        self.foundation.lr_scheduler_type
                    ),
                    "warmup_steps": self.foundation.warmup_steps,
                    "warmup_ratio": self.foundation.warmup_ratio,
                    "run_final_validation": (
                        self.foundation.run_final_validation
                    ),
                    "token_row_mode": self.foundation.token_row_mode,
                },
            },
            "data": data,
            "adapters": {
                "dtype": (
                    args.dtype
                    if args.dtype.startswith("torch.")
                    else f"torch.{args.dtype}"
                ),
                "learning_rate": args.learning_rate,
                "batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": (
                    args.gradient_accumulation_steps
                ),
                "gradient_checkpointing": args.gradient_checkpointing,
                "num_epochs": args.num_train_epochs,
                "max_steps_per_stage": args.max_steps,
                "optim": "adamw_torch",
                "optimizer_layout": args.optimizer_layout,
                "optimizer_pack_bytes": args.optimizer_pack_bytes,
                "optimizer_packing_strategy": (
                    args.optimizer_packing_strategy
                ),
                "execution_graph_layout": args.execution_graph_layout,
                "weight_decay": args.weight_decay,
                "adam_beta1": args.adam_beta1,
                "adam_beta2": args.adam_beta2,
                "adam_epsilon": args.adam_epsilon,
                "lr_scheduler_type": args.lr_scheduler_type,
                "warmup_steps": args.warmup_steps,
                "warmup_ratio": args.warmup_ratio,
                "max_grad_norm": args.max_grad_norm,
                "run_final_validation": True,
                "rank": lora.r,
                "alpha": lora.lora_alpha,
                "targets": targets,
                "dropout": lora.lora_dropout,
                "bias": lora.bias,
                "modules_to_save": (
                    list(lora.modules_to_save)
                    if lora.modules_to_save
                    else None
                ),
                "use_rslora": lora.use_rslora,
                "use_dora": lora.use_dora,
                "lora_plus_lr_ratio": lora.lora_plus_lr_ratio,
                "layers_to_transform": (
                    list(lora.layers_to_transform)
                    if lora.layers_to_transform is not None
                    else None
                ),
                "layers_pattern": lora.layers_pattern,
                "init_weights": lora.init_lora_weights,
                "backend": lora.backend,
                "resident_full_mlp_backward": (
                    lora.resident_full_mlp_backward
                ),
                "resident_output_attention_backward": (
                    lora.resident_output_attention_backward
                ),
                "output_root": self._container_path(self.output_root),
                "steps_freq": args.logging_steps,
                "val_steps_freq": args.eval_steps,
                "save_strategy": args.save_strategy,
                "save_optim": False,
            },
        }

    def save_manifest(self) -> Path:
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            yaml.safe_dump(
                self.manifest(),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        return self.manifest_path

    def preflight(self) -> dict[str, dict]:
        pipeline = Pipeline(self.save_manifest())
        _, adapter_paths = pipeline.render()
        resolved = pipeline.preflight(adapter_paths)
        return {
            name: yaml.safe_load(path.read_text(encoding="utf-8"))
            for name, path in resolved.items()
        }

    def train(self, *, resume: bool = False) -> MultiAdapterTrainResult:
        pipeline = Pipeline(self.save_manifest())
        pipeline.allow_state_migration = resume
        warmup_path, adapter_paths = pipeline.render()
        resolved = pipeline.preflight(adapter_paths)
        pipeline.run_foundation(warmup_path, resume=resume)
        pipeline.run_adapters(resolved, resume=resume)
        artifacts = pipeline.verify_artifacts()
        pipeline.write_log(
            "tt_finetune.MultiAdapterSFTTrainer",
            "completed",
            {"resume": resume},
        )
        foundation = pipeline.container_to_host(
            self.manifest()["foundation"]["warmed_path"]
        )
        adapter_dirs = {
            stage.name: self.output_root
            / (stage.output_subdir or stage.name)
            / "peft_adapter"
            for stage in self.stages
        }
        return MultiAdapterTrainResult(
            run_name=self.run_name,
            foundation_dir=foundation,
            adapter_dirs=adapter_dirs,
            manifest_path=self.manifest_path,
            run_log=pipeline.run_log,
            artifacts=artifacts,
        )
