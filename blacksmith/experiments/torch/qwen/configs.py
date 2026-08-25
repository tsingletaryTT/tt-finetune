# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple

from pydantic import Field

from blacksmith.tools.templates.configs import TrainingConfig as BaseTrainingConfig
from blacksmith.tools.test_config import TestConfig


class TrainingConfig(BaseTrainingConfig):
    # Dataset settings
    dataset_id: str = Field(default="text2sql")
    train_file: Optional[str] = Field(default=None)
    eval_file: Optional[str] = Field(default=None)
    train_files: Optional[List[str]] = Field(default=None)
    eval_files: Optional[List[str]] = Field(default=None)
    max_train_samples: Optional[int] = Field(default=None, gt=0)
    max_eval_samples: Optional[int] = Field(default=None, gt=0)
    max_samples_per_file: Optional[int] = Field(default=None, gt=0)
    max_train_samples_per_file: Optional[int] = Field(default=None, gt=0)
    max_eval_samples_per_file: Optional[int] = Field(default=None, gt=0)
    dataset_format: str = Field(default="chat")  # [chat, prompt_response, text]
    messages_field: str = Field(default="messages")
    prompt_field: str = Field(default="prompt")
    response_field: str = Field(default="response")
    text_field: str = Field(default="text")
    dataset_system_prompt: Optional[str] = Field(default=None)
    assistant_only_loss: bool = Field(default=True)
    text_transform: str = Field(default="none")  # [none, router_moa_tokens]
    dataset_truncation: bool = Field(default=True)
    dataset_packing: bool | str = Field(default="auto")
    dataset_append_eos_token: bool = Field(default=True)
    dataset_sequence_bucketing: bool | str = Field(default="auto")
    dataset_sequence_buckets: Optional[List[int]] = Field(default=None)
    dataset_tokenization_workers: int | str = Field(default="auto")
    dataset_dataloader_workers: int | str = Field(default="auto")
    dataset_prefetch_batches: int = Field(default=2, gt=0)
    fallback_role_separator: str = Field(default="\n")
    fallback_assistant_prefix: str = Field(default="Assistant:")
    chat_template_enable_thinking: bool = Field(default=False)
    do_eval: bool = Field(default=True)

    # Model settings
    model_name: str = Field(default="Qwen/Qwen2.5-0.5B")
    model_revision: Optional[str] = Field(default=None)
    trust_remote_code: bool = Field(default=False)
    attn_implementation: Optional[str] = Field(default=None)
    use_fast_tokenizer: bool = Field(default=True)
    max_length: int = Field(default=128, gt=0)
    dtype: str = Field(default="torch.bfloat16")
    memory_mode: str = Field(default="auto")
    profile_device_memory: bool = Field(default=False)

    # Training hyperparameters
    training_model_type: str = Field(default="lora")  # [lora/peft, adapters]
    adapter_type: str = Field(default="lora")  # [lora, ia3]
    learning_rate: float = Field(default=2e-5, gt=0)
    batch_size: int = Field(default=32, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    gradient_checkpointing: bool = Field(default=False)
    num_epochs: int = Field(default=1, gt=0)
    optim: str = Field(default="adamw_torch")
    optimizer_layout: str = Field(default="auto")
    execution_graph_layout: str = Field(default="auto")
    weight_decay: float = Field(default=0.0, ge=0)
    adam_beta1: float = Field(default=0.9, gt=0, lt=1)
    adam_beta2: float = Field(default=0.999, gt=0, lt=1)
    adam_epsilon: float = Field(default=1e-8, gt=0)
    max_grad_norm: Optional[float] = Field(default=None, gt=0)
    max_steps: Optional[int] = Field(default=None, gt=0)
    run_final_validation: bool = Field(default=False)
    lr_scheduler_type: str = Field(default="constant")
    warmup_steps: int = Field(default=0, ge=0)
    warmup_ratio: float = Field(default=0.0, ge=0, le=1)

    # Logging settings
    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="qwen-finetuning")
    wandb_run_name: str = Field(default="tt-qwen-test")
    wandb_tags: list[str] = Field(default_factory=lambda: ["test"])
    wandb_watch_mode: str = Field(default="all")
    wandb_log_freq: int = Field(default=1000)
    model_to_wandb: bool = Field(default=False)
    steps_freq: int = Field(default=25)
    val_steps_freq: int = Field(default=25)
    epoch_freq: int = Field(default=1)
    measure_e2e_time: bool = Field(default=False)
    print_examples: bool = Field(default=False)
    ignored_index: int = Field(default=-100)

    # Checkpoint settings
    resume_from_checkpoint: bool = Field(default=False)
    resume_option: str = Field(default="last")  # [last, best, path]
    checkpoint_path: str = Field(default="")  # path to checkpoint if resume_option is "path"
    checkpoint_metric: str = Field(default="eval/loss")
    checkpoint_metric_mode: str = Field(default="min")  # [min, max]
    keep_last_n: int = Field(default=3, ge=0)
    keep_best_n: int = Field(default=3, ge=0)
    save_strategy: str = Field(default="epoch")
    save_steps: int = Field(default=250, gt=0)
    project_dir: str = Field(default="blacksmith/experiments/torch/qwen")
    save_optim: bool = Field(default=False)
    storage_backend: str = Field(default="local")
    sync_to_storage: bool = Field(default=False)
    load_from_storage: bool = Field(default=False)
    remote_path: str = Field(default="")

    # Reproducibility settings
    seed: int = Field(default=23)
    deterministic: bool = Field(default=False)

    # LoRA setup
    lora_r: int = Field(default=4, ge=0)
    lora_alpha: int = Field(default=8, gt=0)
    lora_target_modules: list[str] = Field(default_factory=lambda: ["all-linear"])
    lora_task_type: str = Field(default="CAUSAL_LM")
    lora_dropout: float = Field(default=0.0, ge=0, lt=1)
    lora_bias: str = Field(default="none")
    lora_modules_to_save: Optional[list[str]] = Field(default=None)
    lora_use_rslora: bool = Field(default=False)
    lora_use_dora: bool = Field(default=False)
    lora_plus_lr_ratio: Optional[float] = Field(default=None, gt=0)
    lora_layers_to_transform: Optional[List[int]] = Field(default=None)
    lora_layers_pattern: Optional[str] = Field(default=None)
    lora_init_weights: bool | str = Field(default=True)
    lora_backend: str = Field(default="auto")
    lora_resident_full_mlp_backward: bool = Field(default=False)
    lora_resident_output_attention_backward: bool = Field(default=False)
    ia3_feedforward_modules: str | List[str] = Field(
        default="ia3_feedforward"
    )
    init_ia3_weights: bool = Field(default=True)
    token_contract_path: Optional[str] = Field(default=None)

    # Device settings
    mesh_shape: Optional[List[int]] = Field(default=None)  # Note that currently only 2D meshes are supported.
    mesh_axis_names: Optional[List[str]] = Field(default=None)  # e.g., ["data", "model"]
    input_sharding_dim: Optional[str] = Field(
        default=None
    )  # If defined, we will shard inputs along this mesh axis dimension.
    # Tensor parallelism sharding patterns (regex pattern based - matches module names).
    # Format: List of tuples (regex_pattern, sharding_spec_tuple).
    model_sharding_patterns: Optional[List[Tuple[str, Tuple[Optional[str], ...]]]] = Field(default=None)

    # Other settings
    framework: str = Field(default="pytorch")
    use_tt: bool = Field(default=True)
    test_config: Optional[TestConfig] = Field(default=None)
