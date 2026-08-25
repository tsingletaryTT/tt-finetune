"""Notebook-facing supervised fine-tuning trainer."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable, Mapping, Sequence

from .backends import BlackholeXlaBackend, DirectBlackholeBackend
from .config import (
    CompileBucket,
    DataConfig,
    IA3Config,
    LoRAConfig,
    PEFTAdapterConfig,
    TrainingArguments,
)
from .data import resolve_dataset
from .model import BlackholeModel, FastLanguageModel, TokenizerHandle


class SFTTrainer:
    """Train a causal-language-model LoRA on Blackhole.

    The public object is intentionally backend-neutral. Today ``auto`` selects
    the verified TT-XLA backend; a direct in-process backend can replace it
    without changing this API.
    """

    def __init__(
        self,
        *,
        model: BlackholeModel,
        args: TrainingArguments | Mapping[str, Any] | Any,
        train_dataset: Any,
        eval_dataset: Any = None,
        tokenizer: TokenizerHandle | Any | None = None,
        data_config: DataConfig | None = None,
        dataset_text_field: str | None = None,
        formatting_func: Callable[[Mapping[str, Any]], str] | None = None,
        backend: str = "auto",
        workspace: str | Path | None = None,
    ):
        if not isinstance(model, BlackholeModel):
            raise TypeError(
                "model must come from FastLanguageModel.from_pretrained"
            )
        self.model = model
        self.args = TrainingArguments.from_object(args)
        self.tokenizer = tokenizer or TokenizerHandle(
            model.name_or_path,
            revision=model.revision,
            trust_remote_code=model.trust_remote_code,
            use_fast=model.use_fast_tokenizer,
        )
        self.data_config = data_config or DataConfig()
        train_dataset = resolve_dataset(train_dataset, default_split="train")
        eval_dataset = resolve_dataset(eval_dataset, default_split="validation")
        if dataset_text_field:
            self.data_config = replace(
                self.data_config,
                format="text",
                text_field=dataset_text_field,
                assistant_only_loss=False,
            )
        if formatting_func is not None:
            train_dataset = self._format_rows(train_dataset, formatting_func)
            if eval_dataset is not None:
                eval_dataset = self._format_rows(eval_dataset, formatting_func)
            self.data_config = replace(
                self.data_config,
                format="text",
                text_field="text",
                assistant_only_loss=False,
            )
        train_dataset, self.data_config = self._resolve_data_contract(
            train_dataset, self.data_config
        )
        if backend not in {"auto", "blackhole_xla", "blackhole_direct"}:
            raise ValueError(f"Unknown backend: {backend}")
        backend_class = (
            DirectBlackholeBackend
            if backend in {"auto", "blackhole_direct"}
            else BlackholeXlaBackend
        )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._backend_name = (
            "blackhole_direct"
            if backend_class is DirectBlackholeBackend
            else "blackhole_xla"
        )
        self.backend = backend_class(
            model=model,
            args=self.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_config=self.data_config,
            workspace=workspace,
        )
        self._last_result = None

    @classmethod
    def _resolve_data_contract(
        cls,
        dataset: Any,
        config: DataConfig,
    ) -> tuple[Any, DataConfig]:
        if config.format != "auto":
            return dataset, config
        dataset, row = cls._peek_row(dataset)
        keys = set(row)
        if config.messages_field in keys:
            detected = "chat"
        elif {config.prompt_field, config.response_field} <= keys:
            detected = "prompt_response"
        elif config.text_field in keys:
            detected = "text"
        else:
            raise ValueError(
                "Could not infer dataset format. Expected a messages field, "
                "prompt+response fields, or a text field; pass DataConfig "
                "with explicit field names for a custom schema."
            )
        return dataset, replace(
            config,
            format=detected,
            assistant_only_loss=(
                config.assistant_only_loss if detected != "text" else False
            ),
        )

    @staticmethod
    def _peek_row(dataset: Any) -> tuple[Any, Mapping[str, Any]]:
        if isinstance(dataset, (str, Path)):
            source = Path(dataset)
            if not source.is_file():
                raise FileNotFoundError(source)
            with source.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        break
                else:
                    raise ValueError(f"dataset is empty: {source}")
            if not isinstance(row, Mapping):
                raise TypeError("the first JSONL row is not an object")
            return dataset, row
        if hasattr(dataset, "to_list"):
            rows = dataset.to_list()
            if not rows:
                raise ValueError("dataset is empty")
            row = rows[0]
            return dataset, row
        if isinstance(dataset, Sequence):
            if not dataset:
                raise ValueError("dataset is empty")
            row = dataset[0]
            if not isinstance(row, Mapping):
                raise TypeError("dataset row 0 is not a mapping")
            return dataset, row
        rows = list(dataset)
        if not rows:
            raise ValueError("dataset is empty")
        row = rows[0]
        if not isinstance(row, Mapping):
            raise TypeError("dataset row 0 is not a mapping")
        return rows, row

    @staticmethod
    def _format_rows(
        dataset: Any,
        formatting_func: Callable[[Mapping[str, Any]], str],
    ) -> list[dict[str, str]]:
        rows: Iterable[Any] = (
            dataset.to_list() if hasattr(dataset, "to_list") else dataset
        )
        output = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TypeError(f"dataset row {index} is not a mapping")
            text = formatting_func(row)
            if not isinstance(text, str) or not text:
                raise ValueError(
                    f"formatting_func returned invalid text for row {index}"
                )
            output.append({"text": text})
        if not output:
            raise ValueError("dataset is empty")
        return output

    def preflight(self) -> dict[str, Any]:
        return self.backend.preflight()

    def smoke(self, steps: int | None = None) -> Any:
        return self.backend.smoke(steps or self.args.smoke_steps)

    def train(
        self,
        *,
        resume_from_checkpoint: bool | str = False,
        stop_after: int | None = None,
    ) -> Any:
        self._last_result = self.backend.train(
            resume=resume_from_checkpoint,
            stop_after=stop_after,
        )
        return self._last_result

    def compile(
        self,
        *,
        buckets: Sequence[CompileBucket | Mapping[str, Any] | Any]
        | None = None,
    ) -> dict[str, Any]:
        """Precompile/profile this exact semantic training geometry.

        Candidate jobs use isolated model and optimizer state. The resulting
        measured plan may be reused by ``train()``; candidate training state is
        never reused.
        """

        if buckets is not None:
            normalized: list[CompileBucket] = []
            seen: set[tuple[int, int, int, int]] = set()
            for value in buckets:
                bucket = CompileBucket.from_object(value)
                identity = (
                    bucket.rank,
                    bucket.batch_size,
                    bucket.sequence_length,
                    bucket.gradient_accumulation_steps,
                )
                if identity not in seen:
                    normalized.append(bucket)
                    seen.add(identity)
            if not normalized:
                raise ValueError("compile buckets cannot be empty")
            profiles = [self._compile_bucket(item) for item in normalized]
            return {
                "status": "compiled_bucket_profiles_ready",
                "bucket_count": len(profiles),
                "profiles": profiles,
                "scheduling": {
                    "device_profile_jobs": "serialized",
                    "maximum_concurrent_device_jobs": 1,
                    "reason": (
                        "one P150a cannot safely admit independent PJRT "
                        "profiling processes concurrently"
                    ),
                    "parallel_host_discovery": "allowed",
                },
                "training_state_reused": False,
            }

        # A newly measured profile contains dataclass-derived tuples while a
        # cache hit has passed through JSON and therefore contains lists.
        # Normalize at the public boundary so first compile and cache-hit
        # results have one stable, JSON-compatible value contract.
        return json.loads(json.dumps(
            self.backend.compile_profile(), sort_keys=True, default=str
        ))

    def _compile_bucket(self, bucket: CompileBucket) -> dict[str, Any]:
        adapter = self.model.adapter_config
        assert adapter is not None
        if isinstance(adapter, IA3Config):
            if bucket.rank != 1:
                raise ValueError("IA3 compile buckets use rank=1")
            planned_adapter: PEFTAdapterConfig = adapter
        else:
            assert isinstance(adapter, LoRAConfig)
            alpha = (
                adapter.lora_alpha
                if bucket.rank == adapter.r
                else max(
                    1,
                    round(
                        bucket.rank * adapter.lora_alpha / adapter.r
                    ),
                )
            )
            planned_adapter = replace(
                adapter,
                r=bucket.rank,
                lora_alpha=alpha,
            )
        plans = tuple(
            replace(item, adapter=planned_adapter)
            if item.name == self.model.adapter_name else item
            for item in self.model.adapters
        )
        planned_model = replace(
            self.model,
            max_seq_length=bucket.sequence_length,
            lora=planned_adapter,
            adapters=plans,
        )
        suffix = (
            f"compile-r{bucket.rank}-b{bucket.batch_size}-"
            f"s{bucket.sequence_length}-g{bucket.gradient_accumulation_steps}"
        )
        planned_args = replace(
            self.args,
            output_dir=f"{self.args.output_dir}.{suffix}",
            per_device_train_batch_size=bucket.batch_size,
            max_seq_length=bucket.sequence_length,
            gradient_accumulation_steps=(
                bucket.gradient_accumulation_steps
            ),
        )
        trainer = SFTTrainer(
            model=planned_model,
            args=planned_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_config=self.data_config,
            backend=self._backend_name,
            workspace=self.backend._engine.workspace,
        )
        result = trainer.compile()
        return {"bucket": bucket.to_dict(), **result}

    def compile_cache(self) -> dict[str, Any]:
        """Return measured-plan cache inventory and persistence limits."""

        return self.backend.profile_cache_inventory()

    def data_plan(self) -> dict[str, Any]:
        """Inspect packing/bucketing/prefetch and B/S/accumulation candidates."""

        return self.backend.data_plan()

    def save_pretrained(self, path: str | Path) -> Path:
        if self._last_result is None:
            raise RuntimeError("train() must complete before save_pretrained()")
        source = Path(self._last_result.adapter_dir)
        destination = Path(path).expanduser().resolve()
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(
                f"adapter destination is not empty: {destination}"
            )
        destination.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            target = destination / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        return destination

    def save_adapter(
        self,
        path: str | Path,
        *,
        name: str | None = None,
    ) -> Path:
        """Save the active result or copy one inactive named source adapter."""

        selected = name or self.model.adapter_name
        if selected == self.model.adapter_name:
            return self.save_pretrained(path)
        try:
            plan = next(
                item for item in self.model.adapters if item.name == selected
            )
        except StopIteration as error:
            raise KeyError(
                f"unknown adapter {selected!r}; available: "
                f"{self.model.adapter_names}"
            ) from error
        if plan.path is None:
            raise RuntimeError(
                f"inactive adapter {selected!r} has not been trained or loaded"
            )
        source = Path(plan.path).expanduser().resolve()
        destination = Path(path).expanduser().resolve()
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(
                f"adapter destination is not empty: {destination}"
            )
        destination.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            target = destination / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        return destination

    def save_config(self, path: str | Path) -> Path:
        return self.backend.save_config(path)


class TTTrainer(SFTTrainer):
    """Concise Hugging Face-style trainer backed by the proven TT engine."""

    @classmethod
    def supports(
        cls,
        *,
        model: Any,
        rank: int = 8,
        batch_size: int = 1,
        sequence_length: int = 128,
        target_modules: str | Sequence[str] = "auto",
        adapter_type: str = "lora",
        feedforward_modules: str | Sequence[str] = "ia3_feedforward",
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_dropout: float = 0.0,
        lora_bias: str = "none",
        lora_plus_lr_ratio: float | None = None,
        layers_to_transform: Sequence[int] | None = None,
        precision: str = "bfloat16",
        revision: str | None = None,
        trust_remote_code: bool = False,
        memory_mode: str = "auto",
    ):
        """Return a structured static support/admission prediction."""

        from .capabilities import query_support

        return query_support(
            model=model,
            rank=rank,
            batch_size=batch_size,
            sequence_length=sequence_length,
            target_modules=target_modules,
            adapter_type=adapter_type,
            feedforward_modules=feedforward_modules,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_dropout=lora_dropout,
            lora_bias=lora_bias,
            lora_plus_lr_ratio=lora_plus_lr_ratio,
            layers_to_transform=layers_to_transform,
            precision=precision,
            revision=revision,
            trust_remote_code=trust_remote_code,
            memory_mode=memory_mode,
        )

    def plan(self) -> dict[str, Any]:
        """Inspect automatic architecture, memory and execution planning.

        This is compilation-free.  ``selection_status`` inside
        ``automatic_plan`` distinguishes an analytical candidate from a
        measured silicon winner or compatible cache hit.
        """

        adapter = self.model.adapter_config
        assert adapter is not None
        if hasattr(adapter, "r"):
            rank = int(adapter.r)
            use_rslora = bool(getattr(adapter, "use_rslora", False))
            use_dora = bool(getattr(adapter, "use_dora", False))
            lora_plus_lr_ratio = getattr(adapter, "lora_plus_lr_ratio", None)
            feedforward = "ia3_feedforward"
        else:
            rank = 1
            use_rslora = False
            use_dora = False
            lora_plus_lr_ratio = None
            feedforward = adapter.feedforward_modules
        support = self.supports(
            model=self.model,
            rank=rank,
            batch_size=self.args.per_device_train_batch_size,
            sequence_length=(
                self.args.max_seq_length or self.model.max_seq_length
            ),
            target_modules=adapter.target_modules,
            adapter_type=adapter.adapter_type,
            feedforward_modules=feedforward,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_plus_lr_ratio=lora_plus_lr_ratio,
            layers_to_transform=adapter.layers_to_transform,
            precision=self.args.dtype,
            revision=self.model.revision,
            trust_remote_code=self.model.trust_remote_code,
            memory_mode=self.args.memory_mode,
        )
        return support.to_dict()

    def __init__(
        self,
        *,
        model: BlackholeModel,
        dataset: Any = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        args: TrainingArguments | Mapping[str, Any] | Any | None = None,
        tokenizer: TokenizerHandle | Any | None = None,
        adapter_config: PEFTAdapterConfig | None = None,
        rank: int = 8,
        alpha: int | None = None,
        dropout: float = 0.0,
        target_modules: str | Sequence[str] = "auto",
        bias: str = "none",
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_plus_lr_ratio: float | None = None,
        layers_to_transform: Sequence[int] | None = None,
        layers_pattern: str | None = None,
        output_dir: str = "./tt-output",
        backend: str = "auto",
        data_config: DataConfig | None = None,
        workspace: str | Path | None = None,
        **training_kwargs: Any,
    ):
        if train_dataset is not None and dataset is not None:
            raise ValueError("pass dataset or train_dataset, not both")
        train_dataset = train_dataset if train_dataset is not None else dataset
        if train_dataset is None:
            raise ValueError("dataset or train_dataset is required")
        if model.lora is None:
            if adapter_config is not None:
                model = model.add_adapter(
                    "default",
                    adapter_config=adapter_config,
                    activate=True,
                )
            else:
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=rank,
                    lora_alpha=alpha if alpha is not None else 2 * rank,
                    target_modules=target_modules,
                    lora_dropout=dropout,
                    bias=bias,
                    use_rslora=use_rslora,
                    use_dora=use_dora,
                    lora_plus_lr_ratio=lora_plus_lr_ratio,
                    layers_to_transform=(
                        list(layers_to_transform)
                        if layers_to_transform is not None
                        else None
                    ),
                    layers_pattern=layers_pattern,
                )
        elif adapter_config is not None:
            raise ValueError(
                "model already has an active adapter; do not also pass "
                "adapter_config"
            )
        if args is None:
            args = TrainingArguments(output_dir=output_dir, **training_kwargs)
        elif training_kwargs:
            raise ValueError(
                "pass either args or individual training keyword arguments"
            )
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_config=data_config,
            backend=backend,
            workspace=workspace,
        )
