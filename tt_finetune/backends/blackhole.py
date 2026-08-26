"""Blackhole backend using the currently verified TT-XLA execution engine."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ..config import DataConfig, IA3Config, TrainingArguments
from ..model import BlackholeModel
from .base import TrainingBackend


_DTYPE_MAP = {
    "bfloat16": "torch.bfloat16",
    "bf16": "torch.bfloat16",
    "float16": "torch.float16",
    "fp16": "torch.float16",
    "float32": "torch.float32",
    "fp32": "torch.float32",
}


class BlackholeXlaBackend(TrainingBackend):
    """Adapter from the stable library contract to the proven TT-XLA engine.

    This boundary is intentionally private. The execution implementation can
    move in-process without changing notebook code.
    """

    name = "blackhole_xla"

    def __init__(
        self,
        *,
        model: BlackholeModel,
        args: TrainingArguments,
        train_dataset: Any,
        eval_dataset: Any = None,
        data_config: DataConfig | None = None,
        workspace: str | Path | None = None,
    ):
        if model.lora is None:
            raise ValueError(
                "No adapter is attached. Call "
                "FastLanguageModel.get_peft_model(model, ...) first."
            )
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_config = data_config or DataConfig()
        self.workspace = workspace
        if args.auto_tune and type(self) is BlackholeXlaBackend:
            raise NotImplementedError(
                "auto_tune requires the direct profile-search backend"
            )
        self._engine = self._build_engine()

    def _build_engine(self):
        from blackhole_training import (
            TTDataConfig,
            TTLoRAConfig,
            TTLoRATrainer,
            TTModelConfig,
            TTTrainingArguments,
        )

        max_length = self.args.max_seq_length or self.model.max_seq_length
        dtype = _DTYPE_MAP.get(self.args.dtype.lower(), self.args.dtype)
        legacy_args = TTTrainingArguments(
            output_dir=self.args.output_dir,
            run_name=self.args.run_name,
            num_train_epochs=self.args.num_train_epochs,
            max_steps=self.args.max_steps,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            gradient_checkpointing=self.args.gradient_checkpointing,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            adam_beta1=self.args.adam_beta1,
            adam_beta2=self.args.adam_beta2,
            adam_epsilon=self.args.adam_epsilon,
            max_grad_norm=self.args.max_grad_norm,
            max_seq_length=max_length,
            dtype=dtype,
            optimizer_layout=self.args.optimizer_layout,
            execution_graph_layout=self.args.execution_graph_layout,
            lr_scheduler_type=self.args.lr_scheduler_type,
            warmup_steps=self.args.warmup_steps,
            warmup_ratio=self.args.warmup_ratio,
            logging_steps=self.args.logging_steps,
            eval_steps=self.args.eval_steps,
            save_strategy=self.args.save_strategy,
            save_steps=self.args.save_steps,
            save_optim=self.args.save_optimizer_state,
            keep_last_n=self.args.keep_last_checkpoints,
            memory_mode=self.args.memory_mode,
            profile_device_memory=self.args.profile_device_memory,
            seed=self.args.seed,
            auto_smoke=self.args.auto_smoke,
            smoke_steps=self.args.smoke_steps,
        )
        adapter = self.model.adapter_config
        assert adapter is not None
        if isinstance(adapter, IA3Config):
            lora = TTLoRAConfig(
                adapter_type="ia3",
                r=1,
                lora_alpha=1,
                target_modules=adapter.target_modules,
                modules_to_save=adapter.modules_to_save,
                layers_to_transform=adapter.layers_to_transform,
                layers_pattern=adapter.layers_pattern,
                backend=adapter.backend,
                ia3_feedforward_modules=adapter.feedforward_modules,
                init_ia3_weights=adapter.init_ia3_weights,
            )
        else:
            lora = TTLoRAConfig(**asdict(adapter))
        data = TTDataConfig(**asdict(self.data_config))
        model_config = TTModelConfig(
            name_or_path=self.model.name_or_path,
            revision=self.model.revision,
            trust_remote_code=self.model.trust_remote_code,
            attn_implementation=self.model.attn_implementation,
            use_fast_tokenizer=self.model.use_fast_tokenizer,
            initial_adapter_path=self.model.adapter_path,
            # The compiled graph uses one canonical internal adapter slot.
            # Public names select which host-side adapter is materialized;
            # inactive named adapters never enter the optimizer graph.
            initial_adapter_name="default",
        )
        return TTLoRATrainer(
            model=self.model.name_or_path,
            args=legacy_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            lora_config=lora,
            data_config=data,
            model_config=model_config,
            workspace=self.workspace,
        )

    def preflight(self) -> dict[str, Any]:
        return self._engine.preflight()

    def smoke(self, steps: int = 2) -> Any:
        return self._engine.smoke(steps=steps)

    def train(
        self,
        resume: bool | str = False,
        stop_after: int | None = None,
    ) -> Any:
        if stop_after is not None:
            raise NotImplementedError(
                "stop_after is supported by the direct public backend"
            )
        if isinstance(resume, str):
            raise NotImplementedError(
                "named checkpoint resume is supported by the direct backend"
            )
        return self._engine.train(resume=resume)

    def save_config(self, path: str | Path) -> Path:
        return self._engine.save_config(path)
