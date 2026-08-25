"""Stable public configuration objects for tt-finetune."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence, TypeAlias, runtime_checkable


def _positive(name: str, value: int | float | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True)
class LoRAConfig:
    """Portable LoRA topology, independent of the execution backend."""

    r: int = 8
    lora_alpha: int = 16
    target_modules: str | Sequence[str] = "auto"
    lora_dropout: float = 0.0
    bias: str = "none"
    modules_to_save: Sequence[str] | None = None
    use_rslora: bool = False
    use_dora: bool = False
    # LoRA+ applies a higher optimizer LR to B (and DoRA magnitude) while A
    # keeps TrainingArguments.learning_rate.  None preserves ordinary LoRA.
    lora_plus_lr_ratio: float | None = None
    # PEFT-native layer selection.  The public helpers normalize ranges to a
    # stable tuple; the model-family layer container is discovered during
    # preflight and no checkpoint name selects execution code.
    layers_to_transform: Sequence[int] | None = None
    layers_pattern: str | None = None
    init_lora_weights: bool | str = True
    backend: str = "auto"
    resident_full_mlp_backward: bool = False
    resident_output_attention_backward: bool = False
    adapter_type: str = field(default="lora", init=False)

    def __post_init__(self) -> None:
        _positive("r", self.r)
        _positive("lora_alpha", self.lora_alpha)
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.bias not in {"none", "all", "lora_only"}:
            raise ValueError("bias must be none, all, or lora_only")
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
    def for_layer_range(
        cls,
        start: int,
        stop: int,
        **kwargs: Any,
    ) -> "LoRAConfig":
        """Create a half-open canonical decoder-layer selection."""

        if start < 0 or stop <= start:
            raise ValueError("layer range must satisfy 0 <= start < stop")
        return cls(
            layers_to_transform=tuple(range(start, stop)),
            **kwargs,
        )


@runtime_checkable
class AdapterConfig(Protocol):
    """Backend-independent contract shared by public PEFT methods."""

    adapter_type: str
    target_modules: str | Sequence[str]
    modules_to_save: Sequence[str] | None
    layers_to_transform: Sequence[int] | None
    layers_pattern: str | None
    backend: str


@dataclass(frozen=True)
class IA3Config:
    """Portable (IA)³ topology using canonical projection roles.

    The default ``ia3`` target policy resolves to separate K/V projections
    plus the architecture's MLP down projection.  Fused-QKV architectures are
    rejected because scaling the whole fused output would also alter Q and is
    not equivalent to the published IA3 semantics.
    """

    target_modules: str | Sequence[str] = "ia3"
    feedforward_modules: str | Sequence[str] = "ia3_feedforward"
    modules_to_save: Sequence[str] | None = None
    init_ia3_weights: bool = True
    layers_to_transform: Sequence[int] | None = None
    layers_pattern: str | None = None
    backend: str = "generic"
    adapter_type: str = field(default="ia3", init=False)

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "generic"}:
            raise ValueError(
                "IA3 currently supports backend='auto' or 'generic'; the "
                "native LoRA kernels do not implement IA3 scaling"
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
    def for_layer_range(
        cls,
        start: int,
        stop: int,
        **kwargs: Any,
    ) -> "IA3Config":
        if start < 0 or stop <= start:
            raise ValueError("layer range must satisfy 0 <= start < stop")
        return cls(
            layers_to_transform=tuple(range(start, stop)),
            **kwargs,
        )


PEFTAdapterConfig: TypeAlias = LoRAConfig | IA3Config


@dataclass(frozen=True)
class CompileBucket:
    """One semantic R/B/S geometry to pre-profile on the attached P150a."""

    rank: int
    batch_size: int
    sequence_length: int
    gradient_accumulation_steps: int = 1

    def __post_init__(self) -> None:
        _positive("rank", self.rank)
        _positive("batch_size", self.batch_size)
        _positive("sequence_length", self.sequence_length)
        _positive(
            "gradient_accumulation_steps",
            self.gradient_accumulation_steps,
        )

    @classmethod
    def from_object(
        cls, value: "CompileBucket" | Mapping[str, Any] | Any
    ) -> "CompileBucket":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        return cls(**{
            name: getattr(value, name)
            for name in cls.__dataclass_fields__
            if hasattr(value, name)
        })

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingArguments:
    """Transformers-like SFT arguments with Blackhole-specific safety controls."""

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
    max_seq_length: int | None = None
    dtype: str = "bfloat16"
    optimizer_layout: str = "auto"
    optimizer_pack_bytes: int = 1024 * 1024
    optimizer_packing_strategy: str = "contiguous"
    execution_graph_layout: str = "auto"
    lr_scheduler_type: str = "constant"
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    logging_steps: int = 10
    eval_steps: int = 250
    save_strategy: str = "none"
    save_steps: int = 250
    save_optimizer_state: bool = True
    keep_last_checkpoints: int = 3
    compile_cache_dir: str | None = None
    runtime_mode: str = "isolated"
    audit_training_state: bool = False
    profile_product_boundaries: bool = False
    profile_device_memory: bool = False
    memory_mode: str = "auto"
    experimental_fx_fusion: bool = False
    verbose: bool = False
    seed: int = 3407
    auto_smoke: bool = True
    smoke_steps: int = 2
    auto_tune: bool = False
    auto_tune_warmup_steps: int = 4
    auto_tune_measurement_steps: int = 20
    auto_tune_max_candidates: int = 4

    def __post_init__(self) -> None:
        for name in (
            "num_train_epochs",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
        ):
            _positive(name, getattr(self, name))
        _positive("max_steps", self.max_steps)
        _positive("max_seq_length", self.max_seq_length)
        _positive("max_grad_norm", self.max_grad_norm)
        _positive("smoke_steps", self.smoke_steps)
        if self.auto_tune_warmup_steps < 0:
            raise ValueError("auto_tune_warmup_steps must be non-negative")
        _positive("auto_tune_measurement_steps", self.auto_tune_measurement_steps)
        _positive("auto_tune_max_candidates", self.auto_tune_max_candidates)
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.optimizer_layout not in {
            "auto",
            "tile_packed",
            "per_parameter",
        }:
            raise ValueError(
                "optimizer_layout must be auto, tile_packed, or per_parameter"
            )
        _positive("optimizer_pack_bytes", self.optimizer_pack_bytes)
        if self.optimizer_packing_strategy not in {
            "contiguous",
            "shape_grouped",
        }:
            raise ValueError(
                "optimizer_packing_strategy must be contiguous or "
                "shape_grouped"
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
        if self.warmup_steps < 0 or not 0 <= self.warmup_ratio <= 1:
            raise ValueError("invalid warmup configuration")
        if self.logging_steps <= 0 or self.eval_steps <= 0:
            raise ValueError("logging_steps and eval_steps must be positive")
        if self.save_strategy not in {"none", "steps", "epoch"}:
            raise ValueError("save_strategy must be none, steps, or epoch")
        _positive("save_steps", self.save_steps)
        _positive("keep_last_checkpoints", self.keep_last_checkpoints)
        if self.runtime_mode not in {"isolated", "persistent"}:
            raise ValueError(
                "runtime_mode must be isolated or persistent"
            )
        if self.memory_mode not in {
            "auto",
            "bf16",
            "bf8_frozen",
            "bf4_frozen",
        }:
            raise ValueError(
                "memory_mode must be auto, bf16, bf8_frozen, or bf4_frozen"
            )
        if self.profile_device_memory and self.runtime_mode != "isolated":
            raise ValueError(
                "profile_device_memory requires runtime_mode='isolated' so "
                "allocator reports cannot include another job's state"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_object(
        cls, value: "TrainingArguments" | Mapping[str, Any] | Any
    ) -> "TrainingArguments":
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
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: item for key, item in raw.items() if key in allowed})


SFTConfig = TrainingArguments


@dataclass(frozen=True)
class DataConfig:
    """Dataset schema and completion-loss behavior."""

    format: str = "auto"
    messages_field: str = "messages"
    prompt_field: str = "prompt"
    response_field: str = "response"
    text_field: str = "text"
    system_prompt: str | None = None
    assistant_only_loss: bool = True
    chat_template_enable_thinking: bool = False
    text_transform: str = "none"
    truncation: bool = True
    packing: bool | str = "auto"
    append_eos_token: bool = True
    sequence_bucketing: bool | str = "auto"
    sequence_buckets: Sequence[int] | None = None
    tokenization_workers: int | str = "auto"
    dataloader_workers: int | str = "auto"
    prefetch_batches: int = 2

    def __post_init__(self) -> None:
        if self.format not in {"auto", "chat", "prompt_response", "text"}:
            raise ValueError(
                "format must be auto, chat, prompt_response, or text"
            )
        if self.text_transform not in {"none", "router_moa_tokens"}:
            raise ValueError(
                "text_transform must be none or router_moa_tokens"
            )
        for name, value in (
            ("packing", self.packing),
            ("sequence_bucketing", self.sequence_bucketing),
        ):
            if value not in {True, False, "auto"}:
                raise ValueError(f"{name} must be True, False, or auto")
        for name, value in (
            ("tokenization_workers", self.tokenization_workers),
            ("dataloader_workers", self.dataloader_workers),
        ):
            if value != "auto" and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be auto or a non-negative integer")
        if self.prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be positive")
        if self.sequence_buckets is not None:
            normalized = tuple(int(value) for value in self.sequence_buckets)
            if not normalized or any(value <= 0 or value % 32 for value in normalized):
                raise ValueError(
                    "sequence_buckets must contain positive 32-token multiples"
                )
