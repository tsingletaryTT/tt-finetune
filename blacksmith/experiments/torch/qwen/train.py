# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import math
import time
import traceback
from pathlib import Path

import torch
import torch_xla
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.qwen.configs import TrainingConfig
from blacksmith.models.torch.huggingface.hf_models import get_model
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.torch_helpers import (
    collate_fn_for_causal_lm,
    collect_examples,
    show_examples,
)


def validate(
    model: torch.nn.Module,
    val_data_loader: DataLoader,
    loss_fn: torch.nn.Module,
    logger: TrainingLogger,
    device_manager: DeviceManager,
    config: TrainingConfig,
    tokenizer: AutoTokenizer = None,
) -> float:
    logger.info("Starting validation...")

    collected_examples = []
    max_examples = 5
    total_val_loss = 0.0
    num_val_batches = 0
    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Validation"):
            batch = device_manager.prepare_batch(batch)

            # Shard model if tensor parallelism is used.
            device_manager.shard_model(model)

            # Forward pass.
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits

            # Shift logits for causal LM: predict next token
            # logits[:, :-1] predicts tokens at positions 1:
            shift_logits = logits[:, :-1, :].contiguous()

            # Loss
            loss = loss_fn(shift_logits.view(-1, model.model.config.vocab_size), batch["labels"].view(-1))
            total_val_loss += loss.item()

            # Predictions
            predictions = shift_logits.argmax(dim=-1)
            if config.use_tt:
                torch_xla.sync(wait=True)

            num_val_batches += 1

            if config.print_examples:
                collected_examples = collect_examples(
                    batch_size=batch["labels"].shape[0],
                    collected_examples=collected_examples,
                    max_examples=max_examples,
                    input_ids=batch["input_ids"],
                    expected_output=batch["labels"],
                    predictions=predictions,
                    num_val_batches=num_val_batches,
                )

    if config.print_examples and tokenizer is not None:
        logger.info("Printing validation examples...")
        show_examples(collected_examples, tokenizer, config, logger)

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    logger.info(f"Average validation loss: {avg_val_loss}")

    return avg_val_loss


def train(
    config: TrainingConfig, device_manager: DeviceManager, logger: TrainingLogger, checkpoint_manager: CheckpointManager
):
    logger.info("Starting training...")

    # Load model.
    model = get_model(config, device_manager.device)

    model_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Loaded {config.model_name} model.")
    logger.info(f"Model parameters: {model_parameters}")
    logger.info(f"Trainable parameters: {trainable_parameters}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if config.optim not in {"adamw_torch", "adamw"}:
        raise ValueError(
            f"Unsupported optimizer {config.optim!r}; expected 'adamw_torch' or 'adamw'"
        )
    optimizer = torch.optim.AdamW(
        trainable_params,
        capturable=config.use_tt,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.ignored_index)

    resumed_step = 0
    # Load checkpoint if needed.
    if config.resume_from_checkpoint:
        checkpoint_state = checkpoint_manager.load_checkpoint(model, optimizer)
        if checkpoint_state:
            resumed_step = int(checkpoint_state.get("step", 0))

    # Load dataset.
    train_dataset = get_dataset(config=config, split="train", collate_fn=collate_fn_for_causal_lm)
    train_dataloader = train_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Train dataset size: {len(train_dataloader) * config.batch_size}")

    eval_dataset = None
    eval_dataloader = None
    if config.do_eval:
        eval_dataset = get_dataset(
            config=config,
            split="validation",
            collate_fn=collate_fn_for_causal_lm,
        )
        eval_dataloader = eval_dataset.get_dataloader()
        logger.info(
            f"Loaded {config.dataset_id} dataset. Eval dataset size: "
            f"{len(eval_dataloader) * config.batch_size}"
        )

    optimizer_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    planned_optimizer_steps = (
        config.max_steps
        if config.max_steps is not None
        else optimizer_steps_per_epoch * config.num_epochs
    )
    epochs_to_run = (
        config.num_epochs
        if config.max_steps is None
        else math.ceil(
            max(config.max_steps - resumed_step, 0)
            / optimizer_steps_per_epoch
        )
    )
    warmup_steps = (
        config.warmup_steps
        if config.warmup_steps > 0
        else round(planned_optimizer_steps * config.warmup_ratio)
    )
    scheduler = None
    if config.lr_scheduler_type != "constant" or warmup_steps > 0:
        scheduler_name = (
            "constant_with_warmup"
            if config.lr_scheduler_type == "constant" and warmup_steps > 0
            else config.lr_scheduler_type
        )
        scheduler = get_scheduler(
            scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=planned_optimizer_steps,
        )
        if resumed_step:
            scheduler.last_epoch = resumed_step

    global_step = resumed_step
    running_loss = 0.0
    step_times = []
    step_losses = []
    step_learning_rates = []
    metrics_path = Path(config.project_dir) / "training_metrics.json"
    durable_metrics = {
        "status": "running",
        "model_name": config.model_name,
        "training_model_type": config.training_model_type,
        "model_parameters": model_parameters,
        "trainable_parameters": trainable_parameters,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": (
            config.batch_size * config.gradient_accumulation_steps
        ),
        "max_length": config.max_length,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "max_steps": config.max_steps,
        "planned_optimizer_steps": planned_optimizer_steps,
        "epochs_to_run": epochs_to_run,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_steps": warmup_steps,
        "resumed_step": resumed_step,
    }
    try:
        # Initial validation
        if config.do_eval:
            model.eval()
            valid_loss = validate(
                model,
                eval_dataloader,
                loss_fn,
                logger,
                device_manager,
                config,
                eval_dataset.tokenizer,
            )
            logger.log_metrics({"val/loss": valid_loss}, commit=True, step=global_step)
            durable_metrics["initial_validation_loss"] = valid_loss
            model.train()

        # TODO: Refactor when https://github.com/tenstorrent/tt-blacksmith/issues/602#issue-4596214372 is resolved.
        train_start = None
        step_start = None
        if config.measure_e2e_time:
            train_start = time.perf_counter()

        stop_training = (
            config.max_steps is not None and global_step >= config.max_steps
        )
        epoch = 0
        for epoch in range(epochs_to_run):
            if stop_training:
                break
            accumulation_step = 0
            accumulated_loss = 0.0
            optimizer.zero_grad()
            batches_in_epoch = len(train_dataloader)
            for batch_index, batch in enumerate(tqdm(train_dataloader)):
                if accumulation_step == 0:
                    remaining_batches = batches_in_epoch - batch_index
                    accumulation_target = min(
                        config.gradient_accumulation_steps,
                        remaining_batches,
                    )
                    accumulated_loss = 0.0
                    if config.measure_e2e_time:
                        step_start = time.perf_counter()

                # Shard batch if data parallelism is used.
                batch = device_manager.prepare_batch(batch)

                # Shard model if tensor parallelism is used.
                device_manager.shard_model(model)

                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

                shift_logits = outputs.logits[..., :-1, :].contiguous()
                loss = loss_fn(shift_logits.view(-1, model.model.config.vocab_size), batch["labels"].view(-1))

                (loss / accumulation_target).backward()
                if config.use_tt:
                    torch_xla.sync(wait=True)

                loss_value = loss.item()
                accumulated_loss += loss_value
                accumulation_step += 1
                if accumulation_step < accumulation_target:
                    continue

                if config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        max_norm=config.max_grad_norm,
                    )
                device_manager.optimizer_step(optimizer)
                if scheduler is not None:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                step_learning_rates.append(
                    current_lr.item()
                    if isinstance(current_lr, torch.Tensor)
                    else float(current_lr)
                )
                optimizer.zero_grad()
                global_step += 1
                accumulation_step = 0
                optimizer_step_loss = accumulated_loss / accumulation_target
                running_loss += optimizer_step_loss
                step_losses.append(optimizer_step_loss)

                if config.measure_e2e_time:
                    step_elapsed = time.perf_counter() - step_start
                    step_times.append(step_elapsed)
                    logger.info(f"Step {global_step} e2e time: {step_elapsed:.3f}s")

                if config.steps_freq > 0 and global_step % config.steps_freq == 0:
                    avg_loss = running_loss / config.steps_freq
                    logger.log_metrics({"train/loss": avg_loss}, commit=False, step=global_step)
                    running_loss = 0.0

                # Validation
                if (
                    config.do_eval
                    and config.val_steps_freq > 0
                    and global_step % config.val_steps_freq == 0
                ):
                    model.eval()
                    valid_loss = validate(
                        model,
                        eval_dataloader,
                        loss_fn,
                        logger,
                        device_manager,
                        config,
                        eval_dataset.tokenizer,
                    )
                    logger.log_metrics({"val/loss": valid_loss}, commit=False, step=global_step)
                    model.train()

                # Commit metrics to W&B.
                logger.log_metrics({}, commit=True, step=global_step)

                # Save step checkpoint
                if checkpoint_manager.should_save_checkpoint(global_step):
                    checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

                if config.max_steps is not None and global_step >= config.max_steps:
                    stop_training = True
                    break

            # Save epoch checkpoint.
            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)
            if stop_training:
                break

        if config.measure_e2e_time:
            train_elapsed = time.perf_counter() - train_start
            logger.info(f"Training e2e time: {train_elapsed:.3f}s ({global_step} steps)")

        if config.do_eval and config.run_final_validation:
            model.eval()
            final_valid_loss = validate(
                model,
                eval_dataloader,
                loss_fn,
                logger,
                device_manager,
                config,
                eval_dataset.tokenizer,
            )
            durable_metrics["final_validation_loss"] = final_valid_loss
            logger.log_metrics(
                {"val/final_loss": final_valid_loss},
                commit=True,
                step=global_step,
            )
            model.train()

        # Save final model.
        final_model_path = checkpoint_manager.save_checkpoint(
            model, global_step, epoch, optimizer, checkpoint_name="final_model.pth"
        )
        logger.log_artifact(final_model_path, artifact_type="model", name="final_model.pth")
        durable_metrics.update(
            {
                "status": "completed",
                "global_steps": global_step,
                "step_losses": step_losses,
                "step_learning_rates": step_learning_rates,
                "step_e2e_seconds": step_times,
                "training_e2e_seconds": (
                    train_elapsed if config.measure_e2e_time else None
                ),
                "steady_step_seconds": (
                    sum(step_times[1:]) / len(step_times[1:])
                    if len(step_times) > 1
                    else None
                ),
                "steady_seconds_per_example": (
                    sum(step_times[1:])
                    / len(step_times[1:])
                    / config.batch_size
                    / config.gradient_accumulation_steps
                    if len(step_times) > 1
                    else None
                ),
                "checkpoint": str(final_model_path),
            }
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(durable_metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(f"Saved durable training metrics: {metrics_path}")

    except Exception as e:
        durable_metrics.update(
            {
                "status": "failed",
                "global_steps": global_step,
                "step_losses": step_losses,
                "step_learning_rates": step_learning_rates,
                "step_e2e_seconds": step_times,
                "error": str(e),
            }
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(durable_metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        traceback_str = traceback.format_exc()
        logger.error(f"Training failed with error: {str(e)}", traceback_str)
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    # Config setup.
    default_config = Path(__file__).parent / "single_chip" / "qwen_text2sql.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config)

    # Reproducibility setup.
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Logger setup.
    logger = TrainingLogger(config, args.test_log_filename_prefix)

    # Device setup.
    device_manager = DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")

    # Checkpoint manager setup.
    checkpoint_manager = CheckpointManager(config, logger, device_manager.device)

    # Start training.
    train(config, device_manager, logger, checkpoint_manager)
