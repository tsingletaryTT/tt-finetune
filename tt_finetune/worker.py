"""Direct TT-XLA training worker driven by the tt-finetune runtime loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

import yaml

from .runtime import LoopConfig, LoopState, TrainingLoop


AUTO_FORWARD_SPLIT_MAX_LOGITS_BYTES = 384 * 1024 * 1024
_INITIALIZED_XLA_CACHE: Path | None = None


def _initialize_xla_cache(path: Path) -> None:
    """Initialize torch-xla's process-global cache exactly once."""

    global _INITIALIZED_XLA_CACHE
    resolved = path.resolve()
    if _INITIALIZED_XLA_CACHE is not None:
        if _INITIALIZED_XLA_CACHE != resolved:
            raise RuntimeError(
                "the persistent TT runtime cannot switch XLA cache roots: "
                f"{_INITIALIZED_XLA_CACHE} != {resolved}"
            )
        return
    import torch_xla.runtime as xr

    resolved.mkdir(parents=True, exist_ok=True)
    xr.initialize_cache(str(resolved), readonly=False)
    _INITIALIZED_XLA_CACHE = resolved


def _tensor_digest(torch, named_tensors) -> dict:
    """Hash a small audited state without depending on NumPy BF16 support."""

    digest = hashlib.sha256()
    tensor_count = 0
    elements = 0
    for name, value in sorted(named_tensors, key=lambda item: item[0]):
        host = value.detach().cpu().contiguous()
        # torch.compile wraps parameters under ``_orig_mod``.  That wrapper
        # is an execution detail, not part of persisted training state; using
        # the canonical name makes CPU/reference and TT audits directly
        # comparable when tensor values are identical.
        canonical_name = str(name).removeprefix("_orig_mod.")
        digest.update(canonical_name.encode("utf-8"))
        digest.update(str(host.dtype).encode("ascii"))
        digest.update(str(tuple(host.shape)).encode("ascii"))
        digest.update(
            host.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
        tensor_count += 1
        elements += host.numel()
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": tensor_count,
        "elements": elements,
    }


def _tensor_finiteness(torch, named_tensors) -> dict:
    nonfinite = []
    tensor_count = 0
    for name, value in sorted(named_tensors, key=lambda item: item[0]):
        tensor_count += 1
        host = value.detach().cpu()
        if not bool(torch.isfinite(host).all().item()):
            nonfinite.append(name)
    return {
        "finite": not nonfinite,
        "tensor_count": tensor_count,
        "nonfinite_tensor_count": len(nonfinite),
        "nonfinite_tensor_names": nonfinite[:32],
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_summary(
    values: list[float],
    *,
    warmup_count: int,
) -> dict:
    measured = values[warmup_count:]
    return {
        "warmup_count": min(warmup_count, len(values)),
        "sample_count": len(measured),
        "median_seconds": (
            statistics.median(measured) if measured else None
        ),
        "p10_seconds": _percentile(measured, 0.10),
        "p90_seconds": _percentile(measured, 0.90),
        "min_seconds": min(measured) if measured else None,
        "max_seconds": max(measured) if measured else None,
        "mean_seconds": (
            statistics.fmean(measured) if measured else None
        ),
    }


def _steady_measurement_indices(
    record_count: int,
    *,
    warmup_count: int,
    first_use_indices: set[int],
) -> list[int]:
    """Exclude both declared warmups and every late compile-bucket use."""

    return [
        index
        for index in range(record_count)
        if index >= warmup_count and index not in first_use_indices
    ]


def _lora_plus_parameter_groups(named_trainable, *, base_lr, ratio):
    """Build stable LoRA+ AdamW groups without depending on model family.

    PEFT owns the parameter naming contract.  A remains at the base learning
    rate; B and DoRA's magnitude vector use the configured higher rate.  Any
    explicitly trainable bias/modules-to-save stay at the base rate.
    """

    a_parameters = [
        parameter
        for name, parameter in named_trainable
        if ".lora_A." in name
    ]
    high_lr_parameters = [
        parameter
        for name, parameter in named_trainable
        if ".lora_B." in name or "lora_magnitude_vector" in name
    ]
    assigned = {id(value) for value in (*a_parameters, *high_lr_parameters)}
    other_parameters = [
        parameter
        for _, parameter in named_trainable
        if id(parameter) not in assigned
    ]
    if not a_parameters or not high_lr_parameters:
        raise RuntimeError(
            "LoRA+ requested, but trainable LoRA A/B parameters were not "
            "both found after adapter installation"
        )
    groups = [
        {"params": a_parameters, "lr": base_lr},
        {"params": high_lr_parameters, "lr": base_lr * ratio},
    ]
    if other_parameters:
        groups.append({"params": other_parameters, "lr": base_lr})
    report = {
        "enabled": True,
        "lr_ratio": ratio,
        "a_learning_rate": base_lr,
        "b_learning_rate": base_lr * ratio,
        "a_tensor_count": len(a_parameters),
        "b_or_magnitude_tensor_count": len(high_lr_parameters),
        "other_tensor_count": len(other_parameters),
        "scheduler_contract": "shared_multiplier_preserves_ratio",
    }
    return groups, report


def _logical_memory_accounting(model, config, architecture_report) -> dict:
    """Describe logical persistent and training state without claiming peak."""

    named = list(model.named_parameters())
    frozen = [parameter for _, parameter in named if not parameter.requires_grad]
    trainable = [parameter for _, parameter in named if parameter.requires_grad]
    frozen_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in frozen
    )
    trainable_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in trainable
    )
    descriptor = (
        architecture_report.architecture if architecture_report is not None else None
    )
    activation_estimate = None
    attention_estimate = None
    if descriptor is not None:
        token_hidden = (
            int(config.batch_size)
            * int(config.max_length)
            * int(descriptor.hidden_size)
            * 2
        )
        # Conservative framework path: residual/norm/attention/MLP checkpoint
        # equivalents per layer. Resident/streamed paths reuse these arenas,
        # so this remains a labelled logical upper estimate.
        activation_estimate = token_hidden * int(descriptor.layer_count) * 6
        attention_estimate = (
            int(config.batch_size)
            * int(descriptor.attention_heads)
            * int(config.max_length)
            * int(config.max_length)
            * 4
        )
    memory_mode = getattr(model, "_tt_memory_mode_report", None)
    return {
        "contract": "logical_tensor_state_plus_labelled_workspace_estimates_v1",
        "frozen_parameter_logical_bytes": frozen_bytes,
        "trainable_parameter_logical_bytes": trainable_bytes,
        "gradient_logical_bytes": trainable_bytes,
        # Stock/packed AdamW moments follow trainable dtype in the accepted
        # TT path; scalar step tensors are reported separately after creation.
        "optimizer_moment_logical_bytes": trainable_bytes * 2,
        "activation_checkpoint_upper_estimate_bytes": activation_estimate,
        "dense_attention_scratch_upper_estimate_bytes": attention_estimate,
        "parameter_tensor_count": len(named),
        "frozen_tensor_count": len(frozen),
        "trainable_tensor_count": len(trainable),
        "weight_memory_mode": memory_mode,
        "not_directly_observable": [
            "layout_conversion_temporaries",
            "untilize_tilize_buffers",
            "compiler_runtime_owned_buffers",
            "transient_workspaces_allocated_after_program_compile",
        ],
    }


def _automatic_forward_split(
    *,
    batch_size: int,
    max_length: int,
    vocab_size: int,
    element_size: int,
) -> tuple[bool, int]:
    estimated_logits_bytes = (
        batch_size * max_length * vocab_size * element_size
    )
    return (
        estimated_logits_bytes
        <= AUTO_FORWARD_SPLIT_MAX_LOGITS_BYTES,
        estimated_logits_bytes,
    )


def _automatic_tile_packed_optimizer(
    architecture_report,
    *,
    requested: bool,
    foreach: bool,
    use_tt: bool,
) -> tuple[bool, bool]:
    """Honor explicit choices, otherwise consume the architecture optimizer plan."""

    if requested:
        return True, False
    if not use_tt or foreach or architecture_report is None:
        return False, False
    selection = architecture_report.plan.selection("optimizer")
    if selection.implementation == "tile_packed_adamw":
        return True, True
    return False, False


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="last",
        help="Resume from the latest checkpoint or from the supplied path.",
    )
    parser.add_argument("--xla-cache", type=Path)
    parser.add_argument(
        "--audit-training-state",
        action="store_true",
        help="Record exact initial/final trainable and optimizer state hashes.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help="Diagnostic static sequence-length override.",
    )
    parser.add_argument(
        "--dtype",
        choices=("torch.bfloat16", "torch.float32"),
        help="Diagnostic/runtime precision override.",
    )
    parser.add_argument(
        "--token-row-mode",
        choices=("joint", "output_tied"),
        help=(
            "Use joint input/output token-row backward or train output rows "
            "and tie their learned delta into input rows at checkpoint time."
        ),
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        help=(
            "Stop this invocation after N optimizer steps without changing "
            "the configured scheduler horizon."
        ),
    )
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument(
        "--disable-grad-clip",
        action="store_true",
        help="Diagnostic override that sets max_grad_norm to null.",
    )
    parser.add_argument(
        "--audit-device-inputs",
        action="store_true",
        help=(
            "Round-trip input_ids/labels through the TT device and record "
            "per-step batch signatures."
        ),
    )
    parser.add_argument(
        "--backward-loss-scale",
        type=float,
        help=(
            "Divide the loss by this factor during backward, then exactly "
            "unscale during host-controlled clipping."
        ),
    )
    parser.add_argument(
        "--fuse-step",
        action="store_true",
        help=(
            "Defer the loss read until optimizer_step so TT-XLA may compile "
            "backward, gradient clipping, and AdamW as one graph."
        ),
    )
    parser.add_argument(
        "--profile-phases",
        action="store_true",
        help=(
            "Insert diagnostic barriers and time batch transfer, forward, "
            "cross-entropy, backward, clipping, and optimizer phases."
        ),
    )
    parser.add_argument(
        "--profile-warmup-steps",
        type=int,
        default=4,
        help=(
            "Exclude this many initial optimizer steps from phase timing "
            "summaries. Raw timings remain in training_metrics.json."
        ),
    )
    parser.add_argument(
        "--loss-read-interval",
        type=int,
        default=1,
        help=(
            "Read loss to the host every N optimizer steps. The final "
            "configured step is always read; default 1 preserves compatibility."
        ),
    )
    parser.add_argument(
        "--attention-query-block",
        type=int,
        default=None,
        help=(
            "Internal streamed-attention query chunk. This is "
            "normally selected by the geometry autotuner."
        ),
    )
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "safe", "native"),
        default=None,
        help="Internal long-sequence attention schedule.",
    )
    parser.add_argument(
        "--peft-execution-strategy",
        choices=(
            "framework_safe",
            "native_grouped",
            "native_resident_mlp",
            "native_resident_mlp_attention",
        ),
        default="framework_safe",
        help=(
            "Internal PEFT ownership boundary selected by the "
            "geometry autotuner."
        ),
    )
    parser.add_argument(
        "--moe-execution-strategy",
        choices=("auto", "safe", "sparse", "dynamic"),
        default="auto",
        help="Internal routed-expert schedule.",
    )
    parser.add_argument(
        "--capture-profile-ir",
        action="store_true",
        help=(
            "Capture first-step StableHLO for forward, loss, and backward "
            "under the output directory."
        ),
    )
    parser.add_argument(
        "--capture-ttnn-forward-graph",
        action="store_true",
        help=(
            "Capture one steady-state TTNN runtime graph after the configured "
            "profile warmup steps. This does not enable per-op blocking."
        ),
    )
    parser.add_argument(
        "--torch-fx-fusion",
        action="store_true",
        help=(
            "Enable TT-Torch's experimental FX fusion pass for a measured "
            "correctness/performance trial."
        ),
    )
    parser.add_argument(
        "--split-clip-optimizer",
        action="store_true",
        help=(
            "Materialize clipped gradients before AdamW instead of compiling "
            "clipping and optimizer update as one TT-XLA graph."
        ),
    )
    parser.add_argument(
        "--packed-gradient-clip",
        action="store_true",
        help=(
            "Experimental exact-norm clipping trial: retain the native "
            "96-tensor norm order but scale pre-packed optimizer gradients."
        ),
    )
    parser.add_argument(
        "--split-optimizer-phases",
        action="store_true",
        help=(
            "Materialize packed gradients, AdamW updates, and split-parameter "
            "copyback as separate TT-XLA graphs."
        ),
    )
    parser.add_argument(
        "--split-after-forward",
        action="store_true",
        help="Materialize model logits before constructing the loss graph.",
    )
    parser.add_argument(
        "--auto-split-after-forward",
        action="store_true",
        help=(
            "Use the post-forward boundary only when the estimated logits "
            "buffer stays within the established Blackhole safety limit."
        ),
    )
    parser.add_argument(
        "--split-after-loss",
        action="store_true",
        help="Materialize cross-entropy before constructing backward.",
    )
    parser.add_argument(
        "--optimizer-foreach",
        action="store_true",
        help=(
            "Use PyTorch's multi-tensor AdamW path. Experimental on TT-XLA."
        ),
    )
    parser.add_argument(
        "--optimizer-tile-packed",
        action="store_true",
        help=(
            "Pack LoRA parameters and AdamW states into bounded "
            "32x32-aligned buffers."
        ),
    )
    parser.add_argument(
        "--optimizer-pack-bytes",
        type=int,
        default=1024 * 1024,
        help=(
            "Maximum tile-packed AdamW buffer size in bytes. This is an "
            "architecture-tuning option; the established safe default is "
            "1 MiB."
        ),
    )
    parser.add_argument(
        "--optimizer-packing-strategy",
        choices=("contiguous", "shape_grouped"),
        default="contiguous",
        help=(
            "Pack across shape boundaries, or reproduce the historical "
            "shape-grouped optimizer/checkpoint layout."
        ),
    )
    return parser.parse_args()


def _load_config(arguments: argparse.Namespace):
    from blacksmith.experiments.torch.qwen.configs import TrainingConfig

    raw = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    if arguments.output_dir:
        raw["project_dir"] = arguments.output_dir
        raw["wandb_project"] = Path(arguments.output_dir).name
        raw["wandb_run_name"] = Path(arguments.output_dir).name
    if arguments.dtype is not None:
        raw["dtype"] = arguments.dtype
    if arguments.token_row_mode == "output_tied":
        if raw.get("training_model_type") != "token_rows":
            raise ValueError(
                "--token-row-mode applies only to training_model_type=token_rows"
            )
        raw["training_model_type"] = "token_rows_output_tied"
    if arguments.max_steps is not None:
        raw["max_steps"] = arguments.max_steps
    if arguments.max_length is not None:
        if arguments.max_length <= 0:
            raise ValueError("--max-length must be positive")
        raw["max_length"] = arguments.max_length
    if arguments.no_eval:
        raw["do_eval"] = False
        raw["run_final_validation"] = False
        raw["eval_files"] = []
    if arguments.resume_from_checkpoint:
        raw["resume_from_checkpoint"] = True
        if arguments.resume_from_checkpoint == "last":
            raw["resume_option"] = "last"
        else:
            raw["resume_option"] = "path"
            raw["checkpoint_path"] = arguments.resume_from_checkpoint
    if arguments.disable_grad_clip:
        raw["max_grad_norm"] = None
    if arguments.torch_fx_fusion:
        raw["tt_enable_torch_fx_fusion_pass"] = True
    raw["measure_e2e_time"] = True
    return TrainingConfig(**raw)


def _scheduler(config, optimizer, planned_steps):
    from transformers import get_scheduler

    warmup_steps = (
        config.warmup_steps
        if config.warmup_steps > 0
        else round(planned_steps * config.warmup_ratio)
    )
    if config.lr_scheduler_type == "constant" and warmup_steps == 0:
        return None, warmup_steps
    scheduler_name = (
        "constant_with_warmup"
        if config.lr_scheduler_type == "constant"
        else config.lr_scheduler_type
    )
    return (
        get_scheduler(
            scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=planned_steps,
        ),
        warmup_steps,
    )


def _scheduler_values(config, planned_steps):
    """Reproduce Transformers scheduler values without mutating TT AdamW."""

    import torch

    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=config.learning_rate,
    )
    scheduler, warmup_steps = _scheduler(
        config,
        optimizer,
        planned_steps,
    )
    values = [float(optimizer.param_groups[0]["lr"])]
    for _ in range(planned_steps):
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        values.append(float(optimizer.param_groups[0]["lr"]))
    return values, warmup_steps


def _validate(model, dataloader, operations) -> float:
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            loss = operations.forward_loss(batch)
            operations.sync()
            losses.append(operations.loss_value(loss))
    model.train()
    if not losses:
        raise ValueError("validation dataloader has zero batches")
    return sum(losses) / len(losses)


def _tie_output_delta_into_input_checkpoint(
    torch,
    checkpoint: str,
    parameter_names: list[str],
) -> dict:
    input_names = [
        name
        for name in parameter_names
        if name.endswith(".embed_tokens.new_delta")
    ]
    output_names = [
        name
        for name in parameter_names
        if name.endswith(".lm_head.new_delta")
    ]
    if len(input_names) != 1 or len(output_names) != 1:
        raise RuntimeError(
            "Expected one input and one output token delta, got "
            f"input={input_names}, output={output_names}"
        )
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload["model_state_dict"]
    output_name = output_names[0]
    if output_name not in state:
        raise RuntimeError(
            f"Output token delta is missing from checkpoint: {output_name}"
        )
    state[input_names[0]] = state[output_name].clone()
    torch.save(payload, checkpoint)
    return {
        "mode": "output_tied",
        "input_parameter": input_names[0],
        "output_parameter": output_name,
        "copied_elements": state[output_name].numel(),
    }


def _run_public_resident_training(
    *,
    torch,
    config,
    model,
    architecture_report,
    device_manager,
    checkpoint_manager,
    logger,
    metrics,
    publish_metrics,
    record_stage,
    get_dataset,
    collate_fn_for_causal_lm,
    stop_after: int | None,
    profile_phases: bool,
    profile_warmup_steps: int,
    loss_read_interval: int,
    audit_device_inputs: bool,
    audit_training_state: bool,
    compile_sequence_buckets,
    compile_bucket_repeats: int,
) -> dict:
    """Run the public loop directly against the admitted resident ABI."""

    from .runtime import ResidentTrainingOps, build_resident_executor

    admission = getattr(model, "_tt_public_resident_admission", None)
    if admission is None or not admission.admitted:
        raise RuntimeError("public resident worker called without resident admission")
    if architecture_report.executor_backend != "resident_exact":
        raise RuntimeError(
            "emitted resident plan and public executor admission disagree"
        )
    if compile_sequence_buckets is not None:
        unsupported = sorted(
            value
            for value in set(map(int, compile_sequence_buckets))
            if value <= 0
            or value % 32
            or value > int(config.max_length)
        )
        if unsupported:
            raise ValueError(
                "resident logical sequence buckets must be positive 32-token "
                f"multiples no larger than {config.max_length}; got {unsupported}"
            )

    record_stage("optimizer_setup", "started")
    resume_payload = None
    if config.resume_from_checkpoint:
        resume_payload = checkpoint_manager.read_selected_checkpoint()
        if resume_payload is None:
            raise FileNotFoundError(
                "resume was requested but no matching checkpoint exists"
            )
        checkpoint_manager.restore_model_payload(resume_payload, model)
    resumed = (
        checkpoint_manager.checkpoint_metadata(resume_payload)
        if resume_payload is not None else {}
    )
    resumed_step = int(resumed.get("step", 0))
    resumed_epoch = int(resumed.get("epoch", 0))
    resumed_trainer_state = resumed.get("metrics", {}).get(
        "trainer_state", {}
    )
    resumed_micro_step = int(resumed_trainer_state.get("micro_step", 0))
    resumed_batch_in_epoch = int(
        resumed_trainer_state.get("batch_in_epoch", 0)
    )

    executor = build_resident_executor(
        model,
        plan=architecture_report.plan,
        decoder_path=architecture_report.weight_map.decoder_path,
        device=device_manager.device,
        runtime_source_hash=os.environ.get(
            "TT_FINETUNE_RUNTIME_SOURCE_HASH"
        ),
    )
    metrics["architecture"]["executor_backend"] = "resident_exact"
    metrics["architecture"]["executor_fallback_reason"] = None
    metrics["execution"] = {
        "selected_plan": architecture_report.plan.selected_stack_backend,
        "actual_executor": "resident_exact",
        "fallback": None,
        "resident": executor.report.to_dict(),
        "persistent_physical_state": True,
        "logical_parameter_refresh_per_step": False,
    }
    metrics["peft_execution_strategy"] = "resident_full_stack"
    metrics["optimizer_tile_packed"] = True
    metrics["optimizer_tile_packed_selected_automatically"] = True
    if audit_training_state:
        initial_rows = executor.packed.trainable_pages.shape[0]
        initial_state = executor.engine.optimizer_state.detach().cpu()
        initial_logical = executor.packed.logical_state(
            initial_state[:initial_rows]
        )
        metrics["initial_training_state"] = {
            "trainable": _tensor_digest(torch, list(initial_logical.items())),
            "trainable_finiteness": _tensor_finiteness(
                torch, list(initial_logical.items())
            ),
        }
    record_stage("optimizer_setup", "completed")

    record_stage("dataset_preparation", "started")
    dataset = get_dataset(
        config=config,
        split="train",
        collate_fn=collate_fn_for_causal_lm,
    )
    data_pipeline_report = (
        dataset.pipeline_report(epoch=resumed_epoch)
        if hasattr(dataset, "pipeline_report") else None
    )
    saved_data_pipeline = resumed_trainer_state.get("data_pipeline", {})
    if (
        saved_data_pipeline
        and data_pipeline_report is not None
        and saved_data_pipeline.get("contract_hash")
        != data_pipeline_report.get("contract_hash")
    ):
        raise ValueError(
            "resume dataset contract differs from the checkpoint: "
            f"saved={saved_data_pipeline.get('contract_hash')} "
            f"current={data_pipeline_report.get('contract_hash')}"
        )
    if data_pipeline_report is not None:
        metrics["data_pipeline"] = data_pipeline_report
    if compile_sequence_buckets is not None:
        if not hasattr(dataset, "get_compile_warmup_dataloader"):
            raise ValueError(
                "the selected dataset does not support deterministic "
                "sequence-bucket compile warmup"
            )
        dataloader = dataset.get_compile_warmup_dataloader(
            compile_sequence_buckets,
            repeats=compile_bucket_repeats,
            epoch=resumed_epoch,
        )
        metrics["compile_warmup_schedule"] = dict(
            dataset.compile_warmup_schedule
        )
        dataloader_factory = lambda _epoch: dataloader
    else:
        dataloader = dataset.get_dataloader()
        dataloader_factory = (
            dataset.get_dataloader_for_epoch
            if hasattr(dataset, "get_dataloader_for_epoch")
            else lambda _epoch: dataloader
        )
    steps_per_epoch = math.ceil(
        len(dataloader) / config.gradient_accumulation_steps
    )
    planned_steps = (
        config.max_steps
        if config.max_steps is not None
        else steps_per_epoch * config.num_epochs
    )
    learning_rates, warmup_steps = _scheduler_values(config, planned_steps)
    operations = ResidentTrainingOps(
        executor=executor,
        device_manager=device_manager,
        learning_rates=learning_rates,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        epsilon=config.adam_epsilon,
        weight_decay=config.weight_decay,
        initial_step=resumed_step,
        profile_phases=profile_phases,
        audit_gradients=audit_training_state,
        audit_device_inputs=audit_device_inputs,
    )
    optimizer = operations.optimizer
    if resume_payload is not None:
        optimizer_state = resume_payload.get("optimizer_state_dict")
        if optimizer_state is None:
            raise ValueError(
                "exact resident resume requires a checkpoint saved with optimizer state"
            )
        operations.load_checkpoint_state(optimizer_state)
    record_stage("dataset_preparation", "completed")

    record_stage("training_schedule_setup", "started")
    running_losses = []
    running_seconds = []
    running_learning_rates = []
    checkpoint_events = []
    batch_identity_history = []
    compile_stage_completed = False

    def trainer_checkpoint_state(state, *, batch_in_epoch=None):
        value = {
            "micro_step": state.micro_step,
            "batch_in_epoch": (
                state.batch_in_epoch
                if batch_in_epoch is None else batch_in_epoch
            ),
            "resident_optimizer_step": operations.optimizer_steps,
        }
        if data_pipeline_report is not None:
            value["data_pipeline"] = {
                "algorithm": data_pipeline_report["algorithm"],
                "contract_hash": data_pipeline_report["contract_hash"],
                "epoch": state.epoch,
                "batch_offset": value["batch_in_epoch"],
                "shuffle_seed": config.seed,
                "prefetch_state": "deterministically_reconstructed",
            }
        return value

    def save_checkpoint(record, state, *, final=False):
        operations.sync_model_for_checkpoint()
        final_epoch = (
            state.epoch
            if state.batch_in_epoch == 0 else max(0, state.epoch - 1)
        )
        return checkpoint_manager.save_checkpoint(
            model,
            record if isinstance(record, int) else record.step,
            final_epoch,
            optimizer,
            metrics={"trainer_state": trainer_checkpoint_state(state)},
            checkpoint_name="final_model.pth" if final else None,
        )

    def on_step(record, state):
        nonlocal compile_stage_completed
        if not compile_stage_completed:
            record_stage("device_compile_and_first_step", "completed")
            record_stage("steady_training", "started")
            compile_stage_completed = True
        running_losses.append(record.loss)
        running_seconds.append(record.seconds)
        running_learning_rates.append(record.learning_rate)
        batch_identity_history.append({
            "step": record.step,
            "host": operations.last_host_batch_signature,
            "device": operations.last_device_batch_signature,
        })
        if (
            record.step == 1
            or record.step % config.steps_freq == 0
            or record.step == planned_steps
        ):
            rendered = "not-read" if record.loss is None else f"{record.loss:.4f}"
            logger.info(
                f"tt-finetune step {record.step}/{planned_steps} | loss {rendered} | "
                f"{record.seconds:.3f}s | {record.useful_tokens / record.seconds:.1f} tokens/s"
            )
        if checkpoint_manager.should_save_checkpoint(record.step):
            path = save_checkpoint(record, state)
            checkpoint_events.append({"step": record.step, "path": path})
        metrics.update({
            "global_steps": record.step,
            "step_losses": running_losses,
            "step_learning_rates": running_learning_rates,
            "step_e2e_seconds": running_seconds,
            "checkpoint_events": checkpoint_events,
            "batch_identity_history": batch_identity_history,
        })
        if record.step == 1 or record.step % config.steps_freq == 0:
            publish_metrics()

    loop = TrainingLoop(
        LoopConfig(
            num_epochs=config.num_epochs,
            gradient_accumulation_steps=1,
            max_steps=stop_after if stop_after is not None else config.max_steps,
            max_grad_norm=1.0,
            # Keep forward, backward, clipping and AdamW under one device
            # execution boundary.  Loss is read only after optimizer sync.
            fuse_backward_optimizer=True,
            loss_read_interval=loss_read_interval,
        ),
        operations,
        on_step=on_step,
    )
    record_stage("training_schedule_setup", "completed")
    record_stage("device_compile_and_first_step", "started")
    state, records = loop.run(
        dataloader_factory,
        state=LoopState(
            epoch=resumed_epoch,
            optimizer_step=resumed_step,
            micro_step=resumed_micro_step,
            batch_in_epoch=resumed_batch_in_epoch,
        ),
    )
    if not compile_stage_completed:
        record_stage("device_compile_and_first_step", "completed")
        record_stage("steady_training", "started")
    record_stage("steady_training", "completed")

    record_stage("checkpoint_and_export", "started")
    checkpoint = save_checkpoint(state.optimizer_step, state, final=True)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    packed_checkpoint = checkpoint_payload.get("optimizer_state_dict", {})
    packed_tensor = packed_checkpoint.get("optimizer_state")
    if not isinstance(packed_tensor, torch.Tensor):
        # Saving only the adapter is a valid user choice.  The live packed
        # state remains available for final finiteness/audit evidence, while
        # exact resume is intentionally unavailable unless save_optim=True.
        packed_tensor = executor.engine.optimizer_state.detach().cpu()
        metrics["execution"]["exact_resume_state_saved"] = False
    else:
        metrics["execution"]["exact_resume_state_saved"] = True
    rows = executor.packed.trainable_pages.shape[0]
    optimizer_sections = {
        "parameters": executor.packed.logical_state(packed_tensor[:rows]),
        "first_moment": executor.packed.logical_state(packed_tensor[rows:2 * rows]),
        "second_moment": executor.packed.logical_state(packed_tensor[2 * rows:]),
    }
    model_state = checkpoint_payload["model_state_dict"]
    if audit_training_state:
        metrics["final_training_state"] = {
            "trainable": _tensor_digest(torch, list(model_state.items())),
            "trainable_finiteness": _tensor_finiteness(
                torch, list(model_state.items())
            ),
            "optimizer_logical": _tensor_digest(
                torch,
                [
                    (f"{section}.{name}", value)
                    for section, values in optimizer_sections.items()
                    for name, value in values.items()
                ],
            ),
            "optimizer_logical_finiteness": _tensor_finiteness(
                torch,
                [
                    (f"{section}.{name}", value)
                    for section, values in optimizer_sections.items()
                    for name, value in values.items()
                ],
            ),
            "checkpoint_sha256": hashlib.sha256(
                Path(checkpoint).read_bytes()
            ).hexdigest(),
        }
    phase_seconds = {
        name: list(values) for name, values in operations.phase_seconds.items()
    }
    step_seconds = [record.seconds for record in records]
    step_summary = _timing_summary(
        step_seconds, warmup_count=profile_warmup_steps
    )
    metrics.update({
        "status": (
            "completed" if state.optimizer_step >= planned_steps else "interrupted"
        ),
        "planned_optimizer_steps": planned_steps,
        "warmup_steps": warmup_steps,
        "global_steps": state.optimizer_step,
        "epochs_completed": state.epoch,
        "step_losses": [record.loss for record in records],
        "step_learning_rates": [record.learning_rate for record in records],
        "step_e2e_seconds": step_seconds,
        "step_tokens_per_second": [
            record.useful_tokens / record.seconds for record in records
        ],
        "step_useful_tokens": [record.useful_tokens for record in records],
        "step_padded_tokens": [record.padded_tokens for record in records],
        "step_sequence_lengths": [
            list(record.sequence_lengths) for record in records
        ],
        "step_data_wait_seconds": [record.data_wait_seconds for record in records],
        "checkpoint_events": checkpoint_events,
        "batch_identity_history": batch_identity_history,
        "gradient_clipping": {
            "max_grad_norm": 1.0,
            "implementation": operations.tt_clip_mode,
            "history": [],
        },
        "gradient_audit": operations.gradient_audit,
        "resident_evolving_audit": operations.evolving_audit,
        "checkpoint": checkpoint,
        "optimizer_placement": "resident_packed_device",
        "phase_seconds": phase_seconds,
        "phase_summaries": {
            name: _timing_summary(
                values,
                warmup_count=0 if name == "zero_grad" else profile_warmup_steps,
            )
            for name, values in phase_seconds.items()
        },
        "step_e2e_summary": step_summary,
        "steady_step_e2e_summary": step_summary,
        "steady_step_seconds": step_summary["mean_seconds"],
        "steady_measurement_steps": [
            record.step for record in records[profile_warmup_steps:]
        ],
        "compile": {
            "first_step_seconds": step_seconds[0] if step_seconds else None,
            "compilation_excluded_from_steady_state": True,
            "resident_executor_reused_across_steps": True,
            "physical_sequence_bucket": int(config.max_length),
            "logical_sequence_buckets": sorted(
                operations.logical_sequence_lengths_seen
            ),
            "resident_executor_reused_across_compatible_buckets": True,
            "per_bucket_parameter_repacking": False,
        },
    })
    record_stage("checkpoint_and_export", "completed")
    metrics["startup_profile"]["current_stage"] = "completed"
    publish_metrics()
    return metrics


def run(
    config,
    *,
    audit_device_inputs: bool = False,
    stop_after: int | None = None,
    backward_loss_scale: float | None = None,
    fuse_step: bool = False,
    profile_phases: bool = False,
    profile_warmup_steps: int = 4,
    capture_profile_ir: bool = False,
    capture_ttnn_forward_graph: bool = False,
    optimizer_foreach: bool = False,
    optimizer_tile_packed: bool = False,
    optimizer_pack_bytes: int = 1024 * 1024,
    optimizer_packing_strategy: str = "contiguous",
    split_clip_optimizer: bool = False,
    split_optimizer_phases: bool = False,
    split_after_forward: bool = False,
    auto_split_after_forward: bool = False,
    split_after_loss: bool = False,
    packed_gradient_clip: bool = False,
    attention_query_block: int | None = None,
    attention_backend: str | None = None,
    peft_execution_strategy: str = "framework_safe",
    moe_execution_strategy: str = "auto",
    loss_read_interval: int = 1,
    xla_cache: Path | None = None,
    audit_training_state: bool = False,
    service_context: dict | None = None,
    compile_sequence_buckets: tuple[int, ...] | list[int] | None = None,
    compile_bucket_repeats: int = 4,
) -> dict:
    startup_started = time.perf_counter()
    optimizer_tile_packed_requested = optimizer_tile_packed
    if stop_after is not None and stop_after <= 0:
        raise ValueError("stop_after must be positive")
    if profile_warmup_steps < 0:
        raise ValueError("profile_warmup_steps must be non-negative")
    if loss_read_interval <= 0:
        raise ValueError("loss_read_interval must be positive")
    if compile_bucket_repeats <= 0:
        raise ValueError("compile_bucket_repeats must be positive")
    if attention_query_block is not None and (
        attention_query_block <= 0 or attention_query_block % 32
    ):
        raise ValueError("attention_query_block must be a positive tile multiple")
    if attention_query_block is not None:
        config.tt_attention_query_chunk_tokens = int(attention_query_block)
    if attention_backend is not None:
        if attention_backend not in {"auto", "safe", "native"}:
            raise ValueError("unsupported attention_backend")
        config.tt_attention_backend = attention_backend
    if peft_execution_strategy not in {
        "framework_safe",
        "native_grouped",
        "native_resident_mlp",
        "native_resident_mlp_attention",
    }:
        raise ValueError("unsupported peft_execution_strategy")
    if peft_execution_strategy != "framework_safe":
        config.lora_backend = "native_experimental"
        config.lora_resident_full_mlp_backward = peft_execution_strategy in {
            "native_resident_mlp",
            "native_resident_mlp_attention",
        }
        config.lora_resident_output_attention_backward = (
            peft_execution_strategy == "native_resident_mlp_attention"
        )
    if moe_execution_strategy not in {"auto", "safe", "sparse", "dynamic"}:
        raise ValueError("unsupported moe_execution_strategy")
    config.tt_moe_backend = moe_execution_strategy
    if backward_loss_scale is None:
        backward_loss_scale = (
            128.0
            if (
                config.use_tt
                and config.training_model_type == "token_rows"
                and config.max_grad_norm is not None
            )
            else 1.0
        )
    if backward_loss_scale < 1.0:
        raise ValueError("backward_loss_scale must be at least 1")
    if optimizer_foreach and optimizer_tile_packed:
        raise ValueError(
            "optimizer_foreach and optimizer_tile_packed are mutually exclusive"
        )
    if packed_gradient_clip and not optimizer_tile_packed:
        raise ValueError(
            "packed_gradient_clip requires optimizer_tile_packed"
        )
    if optimizer_pack_bytes <= 0:
        raise ValueError("optimizer_pack_bytes must be positive")
    if optimizer_packing_strategy not in {
        "contiguous",
        "shape_grouped",
    }:
        raise ValueError(
            "optimizer_packing_strategy must be contiguous or shape_grouped"
        )
    if split_after_forward and auto_split_after_forward:
        raise ValueError(
            "split_after_forward and auto_split_after_forward are mutually "
            "exclusive"
        )

    output = Path(config.project_dir)
    output.mkdir(parents=True, exist_ok=True)
    if capture_profile_ir:
        profile_ir_dir = output / "profile_ir"
        profile_ir_dir.mkdir(parents=True, exist_ok=True)
        os.environ["XLA_SAVE_TENSORS_FILE"] = str(
            profile_ir_dir / "compiled_graphs.stablehlo.txt"
        )
        os.environ["XLA_SAVE_TENSORS_FMT"] = "stablehlo"

    import torch

    if bool(getattr(config, "profile_device_memory", False)):
        # Must be enabled before DeviceManager opens the PJRT-owned device.
        # TT_METAL_LOGS_PATH is supplied by the isolated public backend.
        import ttnn

        ttnn.device.EnableMemoryReports()

    if xla_cache is not None:
        _initialize_xla_cache(xla_cache)

    from blacksmith.datasets.torch.dataset_utils import get_dataset
    from blacksmith.models.torch.huggingface.hf_models import get_model
    from blacksmith.tools.checkpoints_manager import CheckpointManager
    from blacksmith.tools.device_manager import DeviceManager
    from blacksmith.tools.logging_manager import TrainingLogger
    from blacksmith.tools.reproducibility_manager import ReproducibilityManager
    from blacksmith.tools.torch_helpers import collate_fn_for_causal_lm

    from .runtime.torch_xla_ops import TorchXlaTrainingOps
    from .runtime.training_dropout import (
        DropoutMaskController,
    )

    metrics_path = output / "training_metrics.json"
    logger = TrainingLogger(config)
    reproducibility = ReproducibilityManager(config)
    reproducibility.setup()
    device_manager = DeviceManager(config)
    checkpoint_manager = CheckpointManager(
        config,
        logger,
        device_manager.device,
    )
    metrics = {
        "status": "running",
        "engine": "tt_finetune.runtime.TrainingLoop",
        "model_name": config.model_name,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": (
            config.batch_size * config.gradient_accumulation_steps
        ),
        "max_length": config.max_length,
        "max_steps": config.max_steps,
        "execution_stop_after": stop_after,
        "tt_input_materialization_barrier": bool(config.use_tt),
        "tt_input_materialization_contract": (
            "batch_only_xla_sync_multi"
            if getattr(
                config, "tt_selective_input_materialization", False
            )
            else "global_xla_step_marker"
            if config.use_tt
            else "host_or_cuda_default"
        ),
        "backward_loss_scale": backward_loss_scale,
        "right_padding_safe_attention": bool(config.use_tt),
        "fuse_backward_optimizer": fuse_step,
        "profile_phases": profile_phases,
        "profile_warmup_steps": profile_warmup_steps,
        "loss_read_interval": loss_read_interval,
        "compile_cache": str(xla_cache) if xla_cache is not None else None,
        "capture_profile_ir": capture_profile_ir,
        "capture_ttnn_forward_graph": capture_ttnn_forward_graph,
        "torch_fx_fusion": config.tt_enable_torch_fx_fusion_pass,
        "optimizer_foreach": optimizer_foreach,
        "optimizer_tile_packed_requested": optimizer_tile_packed_requested,
        "optimizer_tile_packed": optimizer_tile_packed,
        "optimizer_pack_bytes": optimizer_pack_bytes,
        "optimizer_packing_strategy": optimizer_packing_strategy,
        "split_clip_optimizer": split_clip_optimizer,
        "packed_gradient_clip": packed_gradient_clip,
        "attention_query_block": int(getattr(
            config, "tt_attention_query_chunk_tokens", 128
        )),
        "attention_backend": str(getattr(
            config, "tt_attention_backend", "auto"
        )),
        "peft_execution_strategy": peft_execution_strategy,
        "moe_execution_strategy": moe_execution_strategy,
        "split_optimizer_phases": split_optimizer_phases,
        "split_after_forward_requested": split_after_forward,
        "auto_split_after_forward": auto_split_after_forward,
        "split_after_loss": split_after_loss,
        "lora_backend_requested": getattr(
            config, "lora_backend", "auto"
        ),
        "persistent_runtime": service_context,
        "audit_training_state": audit_training_state,
        "startup_profile": {
            "contract": "public_startup_stages_v1",
            "current_stage": "runtime_initialization",
            "events": [],
        },
    }
    stage_started = startup_started

    def publish_metrics() -> None:
        temporary = metrics_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(metrics_path)

    def record_stage(stage: str, status: str) -> None:
        nonlocal stage_started
        now = time.perf_counter()
        event = {
            "stage": stage,
            "status": status,
            "elapsed_seconds": now - startup_started,
        }
        if status == "completed":
            event["stage_seconds"] = now - stage_started
        else:
            stage_started = now
        profile = metrics["startup_profile"]
        profile["current_stage"] = (
            stage if status == "started" else f"{stage}_completed"
        )
        profile["events"].append(event)
        publish_metrics()
        logger.info(
            f"tt-finetune stage {stage}: {status} "
            f"({event['elapsed_seconds']:.1f}s elapsed)"
        )

    publish_metrics()
    record_stage("runtime_initialization", "completed")

    try:
        record_stage("model_loading", "started")
        logger.info("Loading model through tt-finetune direct worker")
        model = get_model(config, device_manager.device, compile_model=True)
        model.train()
        record_stage("model_loading", "completed")
        record_stage("optimizer_setup", "started")
        uncompiled_model = getattr(model, "_orig_mod", model)
        native_lora_report = getattr(
            uncompiled_model,
            "_tt_native_lora_install_report",
            None,
        )
        resident_attention_report = getattr(
            uncompiled_model,
            "_tt_resident_attention_install_report",
            None,
        )
        architecture_report = getattr(
            uncompiled_model,
            "_tt_architecture_report",
            None,
        )
        initial_adapter_report = getattr(
            uncompiled_model,
            "_tt_initial_adapter_report",
            None,
        )
        sparse_adapter_report = getattr(
            uncompiled_model,
            "_tt_sparse_adapter_report",
            None,
        )
        metrics["initial_adapter"] = initial_adapter_report
        metrics["sparse_adapter_reload"] = (
            sparse_adapter_report.to_dict()
            if sparse_adapter_report is not None else None
        )
        metrics["architecture"] = (
            architecture_report.to_dict()
            if architecture_report is not None else None
        )
        metrics["memory_accounting"] = _logical_memory_accounting(
            uncompiled_model,
            config,
            architecture_report,
        )
        resident_admission = getattr(
            uncompiled_model,
            "_tt_public_resident_admission",
            None,
        )
        metrics["resident_admission"] = (
            resident_admission.to_dict()
            if resident_admission is not None else None
        )
        if resident_admission is not None and resident_admission.admitted:
            # The ordinary model intentionally remains on the host.  The
            # resident worker packs it once, keeps the physical state
            # authoritative across steps, and refreshes logical PEFT tensors
            # only at checkpoint/export boundaries.
            return _run_public_resident_training(
                torch=torch,
                config=config,
                model=uncompiled_model,
                architecture_report=architecture_report,
                device_manager=device_manager,
                checkpoint_manager=checkpoint_manager,
                logger=logger,
                metrics=metrics,
                publish_metrics=publish_metrics,
                record_stage=record_stage,
                get_dataset=get_dataset,
                collate_fn_for_causal_lm=collate_fn_for_causal_lm,
                stop_after=stop_after,
                profile_phases=profile_phases,
                profile_warmup_steps=profile_warmup_steps,
                loss_read_interval=loss_read_interval,
                audit_device_inputs=audit_device_inputs,
                audit_training_state=audit_training_state,
                compile_sequence_buckets=compile_sequence_buckets,
                compile_bucket_repeats=compile_bucket_repeats,
            )
        # The architecture report's executor_backend is the admitted public
        # plan, after all training-semantic checks.  Record that boundary for
        # the safe path too: users must never have to infer whether a resident
        # candidate was silently replaced by ordinary autograd.
        admitted_backend = (
            architecture_report.executor_backend
            if architecture_report is not None
            else "safe_torch_xla_autograd"
        )
        if admitted_backend != "safe_torch_xla_autograd":
            raise RuntimeError(
                "public execution plan is neither admitted resident nor the "
                f"safe executor: {admitted_backend}"
            )
        fallback_reason = (
            architecture_report.executor_fallback_reason
            if architecture_report is not None else
            "explicit generic adapter/runtime selection"
        )
        metrics["execution"] = {
            "selected_plan": "safe_torch_xla_autograd",
            "actual_executor": "safe_torch_xla_autograd",
            "fallback": fallback_reason,
            "resident_candidate": (
                architecture_report.plan.selected_stack_backend
                if architecture_report is not None else None
            ),
            "persistent_physical_state": False,
            "logical_parameter_refresh_per_step": True,
        }
        (
            optimizer_tile_packed,
            optimizer_tile_packed_selected_automatically,
        ) = _automatic_tile_packed_optimizer(
            architecture_report,
            requested=optimizer_tile_packed_requested,
            foreach=optimizer_foreach,
            use_tt=bool(config.use_tt),
        )
        lora_plus_lr_ratio = getattr(config, "lora_plus_lr_ratio", None)
        if lora_plus_lr_ratio is not None and config.use_tt:
            # Stock capturable AdamW with independently scheduled Python LR
            # groups is not numerically safe on the installed TT-XLA stack.
            # The packed optimizer owns group multipliers as device data and
            # preserves the proven finite AdamW equation.
            optimizer_tile_packed = True
            optimizer_tile_packed_selected_automatically = True
        metrics["optimizer_tile_packed"] = optimizer_tile_packed
        metrics["optimizer_tile_packed_selected_automatically"] = (
            optimizer_tile_packed_selected_automatically
        )
        metrics["native_lora"] = (
            {
                "backend": native_lora_report.backend,
                "installed_count": native_lora_report.installed_count,
                "installed_paths": list(
                    native_lora_report.installed_paths
                ),
                "skipped": native_lora_report.skipped,
            }
            if native_lora_report is not None
            else {
                "backend": "generic",
                "installed_count": 0,
                "installed_paths": [],
                "skipped": {},
            }
        )
        metrics["resident_attention"] = (
            {
                "installed_count": resident_attention_report.installed_count,
                "installed_paths": list(
                    resident_attention_report.installed_paths
                ),
                "skipped": resident_attention_report.skipped,
                "fallbacks": resident_attention_report.fallbacks,
            }
            if resident_attention_report is not None
            else {
                "installed_count": 0,
                "installed_paths": [],
                "skipped": {},
                "fallbacks": {},
            }
        )
        effective_split_after_forward = split_after_forward
        estimated_logits_bytes = None
        if auto_split_after_forward:
            (
                effective_split_after_forward,
                estimated_logits_bytes,
            ) = _automatic_forward_split(
                batch_size=config.batch_size,
                max_length=config.max_length,
                vocab_size=model.config.vocab_size,
                element_size=torch.empty(
                    (),
                    dtype=config.torch_dtype(),
                ).element_size(),
            )
            if not effective_split_after_forward:
                logger.info(
                    "Automatic post-forward boundary disabled: estimated "
                    f"logits buffer {estimated_logits_bytes:,} bytes exceeds "
                    f"{AUTO_FORWARD_SPLIT_MAX_LOGITS_BYTES:,}-byte limit"
                )
        metrics.update(
            {
                "split_after_forward": effective_split_after_forward,
                "estimated_logits_bytes": estimated_logits_bytes,
                "auto_forward_split_max_logits_bytes": (
                    AUTO_FORWARD_SPLIT_MAX_LOGITS_BYTES
                    if auto_split_after_forward
                    else None
                ),
            }
        )
        publish_metrics()
        model_parameters = sum(
            parameter.numel() for parameter in model.parameters()
        )
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        named_trainable = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        trainable = [parameter for _, parameter in named_trainable]
        resume_payload = None
        if config.resume_from_checkpoint:
            resume_payload = checkpoint_manager.read_selected_checkpoint()
            if resume_payload is None:
                raise FileNotFoundError(
                    "resume was requested but no matching checkpoint exists"
                )
            # Restore logical model parameters before TilePackedAdamW aliases
            # them into packed buffers. Loading into an already-aliased rank
            # slice asks TTNN to pad a view whose MeshBuffer only owns the
            # slice, which fails runtime admission.
            checkpoint_manager.restore_model_payload(resume_payload, model)
            if config.use_tt:
                import torch_xla

                torch_xla.sync(wait=True)
        use_host_master_optimizer = (
            config.use_tt
            and config.training_model_type == "token_rows_output_tied"
        )
        host_master_parameters = None
        optimizer_parameters = trainable
        if lora_plus_lr_ratio is not None:
            optimizer_parameters, metrics["lora_plus"] = (
                _lora_plus_parameter_groups(
                    named_trainable,
                    base_lr=config.learning_rate,
                    ratio=lora_plus_lr_ratio,
                )
            )
        else:
            metrics["lora_plus"] = {"enabled": False}
        if use_host_master_optimizer:
            host_master_parameters = [
                torch.nn.Parameter(
                    parameter.detach().cpu().float(),
                    requires_grad=True,
                )
                for parameter in trainable
            ]
            optimizer_parameters = host_master_parameters
        model_parameter_names = [
            name for name, _ in model.named_parameters()
        ]
        optimizer_kwargs = {
            "capturable": (
                config.use_tt
                and not use_host_master_optimizer
                and not optimizer_foreach
            ),
            "foreach": optimizer_foreach,
            "lr": config.learning_rate,
            "betas": (config.adam_beta1, config.adam_beta2),
            "eps": config.adam_epsilon,
            "weight_decay": config.weight_decay,
        }
        if optimizer_tile_packed:
            if use_host_master_optimizer:
                raise ValueError(
                    "Tile-packed AdamW does not support host-master parameters"
                )
            from .optim import TilePackedAdamW

            optimizer = TilePackedAdamW(
                optimizer_parameters,
                max_pack_bytes=optimizer_pack_bytes,
                packing_strategy=optimizer_packing_strategy,
                profile_phases=profile_phases,
                split_phases=split_optimizer_phases,
                **optimizer_kwargs,
            )
            metrics["optimizer_packing"] = {
                "strategy": optimizer.packing_strategy,
                "logical_elements": optimizer.logical_elements,
                "padded_elements": optimizer.padded_elements,
                "tile_occupancy": (
                    optimizer.logical_elements / optimizer.padded_elements
                ),
                "original_tensor_count": len(optimizer.model_parameters),
                "packed_tensor_count": len(optimizer.packed_parameters),
                "shared_view_parameter_count": len(
                    optimizer.shared_view_indices
                ),
                "copyback_parameter_count": (
                    len(optimizer.model_parameters)
                    - len(optimizer.shared_view_indices)
                ),
                "max_pack_bytes": optimizer.max_pack_bytes,
            }
            parameter_bytes = sum(
                value.numel() * value.element_size()
                for value in optimizer.packed_parameters
            )
            metrics["memory_accounting"]["packed_optimizer"] = {
                "packed_parameter_bytes": parameter_bytes,
                "packed_gradient_bytes": parameter_bytes,
                "packed_moment_bytes": parameter_bytes * 2,
                "logical_elements": optimizer.logical_elements,
                "padded_elements": optimizer.padded_elements,
                "shares_model_parameter_storage_count": len(
                    optimizer.shared_view_indices
                ),
                "copyback_parameter_count": (
                    len(optimizer.model_parameters)
                    - len(optimizer.shared_view_indices)
                ),
            }
        else:
            # PyTorch explicitly rejects XLA from the capturable foreach
            # allow-list. The foreach experiment therefore keeps its scalar
            # step counters on the host while tensor updates stay on device.
            optimizer = torch.optim.AdamW(
                optimizer_parameters,
                **optimizer_kwargs,
            )

        resumed = {}
        if resume_payload is not None:
            checkpoint_manager.restore_optimizer_payload(
                resume_payload, optimizer
            )
            resumed = checkpoint_manager.checkpoint_metadata(resume_payload)
        resumed_step = int(resumed.get("step", 0))
        resumed_epoch = int(resumed.get("epoch", 0))
        resumed_trainer_state = resumed.get("metrics", {}).get(
            "trainer_state", {}
        )
        resumed_micro_step = int(resumed_trainer_state.get("micro_step", 0))
        resumed_batch_in_epoch = int(
            resumed_trainer_state.get("batch_in_epoch", 0)
        )
        if audit_training_state:
            metrics["initial_training_state"] = {
                "trainable": _tensor_digest(
                    torch,
                    [
                        (name, parameter)
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                    ],
                ),
                "optimizer_state_entries": len(optimizer.state),
                "resumed_step": resumed_step,
                "resumed_epoch": resumed_epoch,
                "resumed_micro_step": resumed_micro_step,
                "resumed_batch_in_epoch": resumed_batch_in_epoch,
            }

        record_stage("optimizer_setup", "completed")
        record_stage("dataset_preparation", "started")
        dataset = get_dataset(
            config=config,
            split="train",
            collate_fn=collate_fn_for_causal_lm,
        )
        data_pipeline_report = (
            dataset.pipeline_report(epoch=resumed_epoch)
            if hasattr(dataset, "pipeline_report")
            else None
        )
        saved_data_pipeline = resumed_trainer_state.get("data_pipeline", {})
        if (
            saved_data_pipeline
            and data_pipeline_report is not None
            and saved_data_pipeline.get("contract_hash")
            != data_pipeline_report.get("contract_hash")
        ):
            raise ValueError(
                "resume dataset contract differs from the checkpoint: "
                f"saved={saved_data_pipeline.get('contract_hash')} "
                f"current={data_pipeline_report.get('contract_hash')}. "
                "Use the same tokenization, packing, bucketing, seed, and "
                "source data for exact continuation."
            )
        if data_pipeline_report is not None:
            metrics["data_pipeline"] = data_pipeline_report
        if compile_sequence_buckets is not None:
            if not hasattr(dataset, "get_compile_warmup_dataloader"):
                raise ValueError(
                    "the selected dataset does not support deterministic "
                    "sequence-bucket compile warmup"
                )
            dataloader = dataset.get_compile_warmup_dataloader(
                compile_sequence_buckets,
                repeats=compile_bucket_repeats,
                epoch=resumed_epoch,
            )
            metrics["compile_warmup_schedule"] = dict(
                dataset.compile_warmup_schedule
            )
            dataloader_factory = lambda _epoch: dataloader
        else:
            dataloader = dataset.get_dataloader()
            dataloader_factory = (
                dataset.get_dataloader_for_epoch
                if hasattr(dataset, "get_dataloader_for_epoch")
                else lambda _epoch: dataloader
            )
        optimizer_steps_per_epoch = math.ceil(
            len(dataloader) / config.gradient_accumulation_steps
        )
        planned_steps = (
            config.max_steps
            if config.max_steps is not None
            else optimizer_steps_per_epoch * config.num_epochs
        )
        record_stage("dataset_preparation", "completed")
        record_stage("training_schedule_setup", "started")
        use_preloaded_learning_rates = (
            config.use_tt
            and optimizer_tile_packed
            and (
                config.lr_scheduler_type != "constant"
                or config.warmup_steps > 0
                or config.warmup_ratio > 0
            )
        )
        if use_preloaded_learning_rates:
            scheduler_values, warmup_steps = _scheduler_values(
                config,
                planned_steps,
            )
            device_schedule_buffer = torch.full(
                (32, 32),
                scheduler_values[resumed_step],
                dtype=torch.float32,
                device=trainable[0].device,
            )
            # Materialize one stable LR input tile.  TilePackedAdamW copies
            # the current host schedule value into this same buffer before
            # each update, so compile setup and device memory stay O(1) in
            # the number of training steps.
            import torch_xla

            torch_xla.sync(wait=True)
            optimizer.configure_learning_rate_schedule(
                [device_schedule_buffer],
                host_values=scheduler_values,
                initial_step=resumed_step,
            )
            scheduler = None
            metrics["optimizer_learning_rate_schedule"] = {
                "mode": "single_mutable_device_tile",
                "update_count": len(scheduler_values) - 1,
                "avoids_value_specialized_adamw_graphs": True,
                "device_buffer_count": 1,
            }
        else:
            scheduler, warmup_steps = _scheduler(
                config,
                optimizer,
                planned_steps,
            )
            if scheduler is not None and resumed_step:
                scheduler.last_epoch = resumed_step

        operations = TorchXlaTrainingOps(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device_manager=device_manager,
            ignored_index=config.ignored_index,
            tt_clip_mode=(
                "host"
                if (
                    config.use_tt
                    and config.training_model_type.startswith("token_rows")
                )
                else "packed_exact_norm"
                if packed_gradient_clip
                else "native"
            ),
            audit_device_inputs=audit_device_inputs,
            backward_loss_scale=backward_loss_scale,
            right_padding_safe_attention=bool(config.use_tt),
            host_master_parameters=host_master_parameters,
            profile_phases=profile_phases,
            profile_ir_dir=(
                output / "profile_ir" if capture_profile_ir else None
            ),
            ttnn_forward_graph_path=(
                output / "profile_ir" / "forward.ttnn_graph.json"
                if capture_ttnn_forward_graph
                else None
            ),
            ttnn_forward_graph_capture_after=profile_warmup_steps,
            split_clip_optimizer=split_clip_optimizer,
            split_after_forward=effective_split_after_forward,
            split_after_loss=split_after_loss,
            audit_gradients=audit_training_state,
            training_dropout=(
                DropoutMaskController(
                    uncompiled_model,
                    config.seed + 0xD09,
                    initial_micro_step=resumed_micro_step,
                )
                if config.use_tt and getattr(
                    uncompiled_model, "_tt_dropout_install_report", None
                ) is not None
                else None
            ),
        )
        eval_dataloader = None
        if config.do_eval:
            eval_dataset = get_dataset(
                config=config,
                split="validation",
                collate_fn=collate_fn_for_causal_lm,
            )
            eval_dataloader = eval_dataset.get_dataloader()
            metrics["initial_validation_loss"] = _validate(
                model,
                eval_dataloader,
                operations,
            )
        validation_history = []
        gradient_norm_history = []
        batch_identity_history = []
        running_step_losses = []
        running_step_seconds = []
        running_step_learning_rates = []
        checkpoint_events = []
        compile_stage_completed = False
        expected_compile_buckets = set(
            int(value)
            for value in (
                (data_pipeline_report or {}).get("bucketing", {})
                .get("boundaries", [config.max_length])
            )
        )
        observed_compile_buckets: set[int] = set()

        def trainer_checkpoint_state(_state, *, batch_in_epoch=None):
            value = {
                "micro_step": _state.micro_step,
                "batch_in_epoch": (
                    _state.batch_in_epoch
                    if batch_in_epoch is None
                    else batch_in_epoch
                ),
            }
            if data_pipeline_report is not None:
                value["data_pipeline"] = {
                    "algorithm": data_pipeline_report["algorithm"],
                    "contract_hash": data_pipeline_report["contract_hash"],
                    "epoch": _state.epoch,
                    "batch_offset": value["batch_in_epoch"],
                    "shuffle_seed": config.seed,
                    "prefetch_state": "deterministically_reconstructed",
                }
            return value

        def on_step(record, _state):
            nonlocal compile_stage_completed
            observed_compile_buckets.update(record.sequence_lengths)
            if not compile_stage_completed:
                # The startup compile boundary is the first complete training
                # step. Rare later sequence shapes are independent first-use
                # bucket events, not evidence that all intervening training
                # was still "compiling".
                record_stage("device_compile_and_first_step", "completed")
                record_stage("steady_training", "started")
                compile_stage_completed = True
            rendered_loss = (
                f"{record.loss:.4f}"
                if record.loss is not None
                else "not-read"
            )
            if (
                record.step == 1
                or record.step % config.steps_freq == 0
                or record.step == planned_steps
            ):
                tokens_per_second = record.useful_tokens / record.seconds
                logger.info(
                    f"tt-finetune step {record.step}/{planned_steps} | "
                    f"loss {rendered_loss} | {record.seconds:.3f}s | "
                    f"{tokens_per_second:.1f} tokens/s"
                )
            running_step_losses.append(record.loss)
            running_step_seconds.append(record.seconds)
            running_step_learning_rates.append(record.learning_rate)
            save_for_step = checkpoint_manager.should_save_checkpoint(
                record.step
            )
            current_epoch_batches = (
                len(dataset.epoch_plan(_state.epoch).batches)
                if hasattr(dataset, "epoch_plan")
                else len(dataloader)
            )
            save_for_epoch = (
                _state.batch_in_epoch >= current_epoch_batches
                and checkpoint_manager.should_save_checkpoint(
                    record.step,
                    _state.epoch + 1,
                )
            )
            if save_for_step or save_for_epoch:
                checkpoint_at_epoch_boundary = (
                    _state.batch_in_epoch >= current_epoch_batches
                )
                checkpoint_path = checkpoint_manager.save_checkpoint(
                    model,
                    record.step,
                    (
                        _state.epoch + 1
                        if checkpoint_at_epoch_boundary
                        else _state.epoch
                    ),
                    optimizer,
                    metrics={"trainer_state": trainer_checkpoint_state(
                        _state,
                        batch_in_epoch=(
                            0
                            if checkpoint_at_epoch_boundary
                            else _state.batch_in_epoch
                        ),
                    )},
                )
                checkpoint_events.append({
                    "step": record.step,
                    "path": checkpoint_path,
                })
            if operations.last_host_batch_signature is not None:
                batch_identity_history.append(
                    {
                        "step": record.step,
                        "host": operations.last_host_batch_signature,
                        "device": operations.last_device_batch_signature,
                    }
                )
            if (
                eval_dataloader is not None
                and config.val_steps_freq > 0
                and record.step % config.val_steps_freq == 0
            ):
                validation_loss = _validate(
                    model,
                    eval_dataloader,
                    operations,
                )
                validation_history.append(
                    {
                        "step": record.step,
                        "loss": validation_loss,
                    }
                )
            if operations.last_gradient_norm is not None:
                gradient_norm_history.append(
                    {
                        "step": record.step,
                        "pre_clip_l2_norm": (
                            operations.last_gradient_norm
                        ),
                        "clip_coefficient": (
                            operations.last_clip_coefficient
                        ),
                        "applied_scaled_gradient_multiplier": (
                            operations.last_gradient_multiplier
                        ),
                        "tensors": operations.last_gradient_diagnostics,
                    }
                )
            if (
                record.step == 1
                or record.step % config.steps_freq == 0
                or record.step == planned_steps
                or save_for_step
                or save_for_epoch
            ):
                metrics.update(
                    {
                        "global_steps": record.step,
                        "step_losses": running_step_losses,
                        "step_learning_rates": running_step_learning_rates,
                        "step_e2e_seconds": running_step_seconds,
                        "step_tokens_per_second": [
                            (
                                config.batch_size
                                * config.max_length
                                * config.gradient_accumulation_steps
                                / seconds
                            )
                            for seconds in running_step_seconds
                        ],
                        "checkpoint_events": checkpoint_events,
                        "batch_identity_history": batch_identity_history,
                        "gradient_clipping": {
                            "max_grad_norm": config.max_grad_norm,
                            "implementation": operations.tt_clip_mode,
                            "history": gradient_norm_history,
                        },
                    }
                )
                publish_metrics()

        loop = TrainingLoop(
            LoopConfig(
                num_epochs=config.num_epochs,
                gradient_accumulation_steps=(
                    config.gradient_accumulation_steps
                ),
                max_steps=(
                    stop_after
                    if stop_after is not None
                    else config.max_steps
                ),
                max_grad_norm=config.max_grad_norm,
                fuse_backward_optimizer=fuse_step,
                loss_read_interval=loss_read_interval,
            ),
            operations,
            on_step=on_step,
        )
        record_stage("training_schedule_setup", "completed")
        record_stage("device_compile_and_first_step", "started")
        state, records = loop.run(
            dataloader_factory,
            state=LoopState(
                epoch=resumed_epoch,
                optimizer_step=resumed_step,
                micro_step=resumed_micro_step,
                batch_in_epoch=resumed_batch_in_epoch,
            ),
        )
        if not compile_stage_completed:
            record_stage("device_compile_and_first_step", "completed")
            record_stage("steady_training", "started")
            compile_stage_completed = True
        if compile_stage_completed:
            record_stage("steady_training", "completed")
        record_stage("checkpoint_and_export", "started")
        if eval_dataloader is not None and config.run_final_validation:
            metrics["final_validation_loss"] = _validate(
                model,
                eval_dataloader,
                operations,
            )
        final_resume_epoch = (
            state.epoch
            if state.batch_in_epoch == 0
            else max(0, state.epoch - 1)
        )
        checkpoint = checkpoint_manager.save_checkpoint(
            model,
            state.optimizer_step,
            final_resume_epoch,
            optimizer,
            metrics={"trainer_state": trainer_checkpoint_state(state)},
            checkpoint_name="final_model.pth",
        )
        if audit_training_state:
            checkpoint_payload = torch.load(checkpoint, map_location="cpu")
            model_state = checkpoint_payload.get("model_state_dict", {})
            optimizer_state = checkpoint_payload.get(
                "optimizer_state_dict", {}
            ).get("state", {})
            optimizer_tensors = []
            for index, values in optimizer_state.items():
                for name, value in values.items():
                    if torch.is_tensor(value):
                        optimizer_tensors.append(
                            (f"{index}.{name}", value)
                        )
            logical_optimizer_tensors = (
                optimizer.logical_state_tensors_from_state_dict(
                    checkpoint_payload.get("optimizer_state_dict", {})
                )
                if optimizer_tile_packed
                else optimizer_tensors
            )
            metrics["final_training_state"] = {
                "trainable": _tensor_digest(
                    torch, list(model_state.items())
                ),
                "trainable_finiteness": _tensor_finiteness(
                    torch, list(model_state.items())
                ),
                "optimizer": _tensor_digest(torch, optimizer_tensors),
                "optimizer_logical": _tensor_digest(
                    torch, logical_optimizer_tensors
                ),
                "optimizer_finiteness": _tensor_finiteness(
                    torch, optimizer_tensors
                ),
                "optimizer_logical_finiteness": _tensor_finiteness(
                    torch, logical_optimizer_tensors
                ),
                "checkpoint_sha256": hashlib.sha256(
                    Path(checkpoint).read_bytes()
                ).hexdigest(),
            }
            if not metrics["final_training_state"][
                "trainable_finiteness"
            ]["finite"]:
                raise FloatingPointError(
                    "non-finite trainable parameters detected in the final "
                    "checkpoint: "
                    + ", ".join(
                        metrics["final_training_state"][
                            "trainable_finiteness"
                        ]["nonfinite_tensor_names"][:8]
                    )
                )
            if not metrics["final_training_state"][
                "optimizer_finiteness"
            ]["finite"]:
                raise FloatingPointError(
                    "non-finite optimizer state detected in the final checkpoint"
                )
            if not metrics["final_training_state"][
                "optimizer_logical_finiteness"
            ]["finite"]:
                raise FloatingPointError(
                    "non-finite logical optimizer state detected in the final "
                    "checkpoint"
                )
        token_row_coupling = None
        if config.training_model_type == "token_rows_output_tied":
            token_row_coupling = _tie_output_delta_into_input_checkpoint(
                torch,
                checkpoint,
                model_parameter_names,
            )
        phase_seconds = {
            name: list(values)
            for name, values in operations.phase_seconds.items()
        }
        if optimizer_tile_packed:
            phase_seconds.update(
                {
                    name: list(values)
                    for name, values in optimizer.phase_seconds.items()
                }
            )
        phase_summaries = {
            name: _timing_summary(
                values,
                # zero_grad is also called at epoch boundaries, so it is a
                # call-level diagnostic rather than a one-sample-per-step
                # phase.
                warmup_count=(
                    0 if name == "zero_grad" else profile_warmup_steps
                ),
            )
            for name, values in phase_seconds.items()
        }
        step_e2e_seconds = [record.seconds for record in records]
        step_e2e_summary = _timing_summary(
            step_e2e_seconds,
            warmup_count=profile_warmup_steps,
        )
        first_use_indices = []
        seen_sequence_lengths: set[int] = set()
        for index, record in enumerate(records):
            new_lengths = set(record.sequence_lengths) - seen_sequence_lengths
            if new_lengths:
                first_use_indices.append((index, tuple(sorted(new_lengths))))
                seen_sequence_lengths.update(new_lengths)
        first_use_index_set = {index for index, _ in first_use_indices}
        steady_samples = [
            record.seconds
            for index, record in enumerate(records)
            if index not in first_use_index_set
        ]
        steady_measurement_indices = _steady_measurement_indices(
            len(records),
            warmup_count=profile_warmup_steps,
            first_use_indices=first_use_index_set,
        )
        steady_measurement_samples = [
            records[index].seconds for index in steady_measurement_indices
        ]
        steady_step_e2e_summary = _timing_summary(
            steady_measurement_samples,
            warmup_count=0,
        )
        steady_reference = (
            statistics.median(steady_samples)
            if steady_samples else None
        )
        estimated_compile_seconds = (
            sum(
                max(0.0, records[index].seconds - steady_reference)
                for index, _ in first_use_indices
            )
            if first_use_indices and steady_reference is not None else None
        )
        bucket_first_use = [
            {
                "step": records[index].step,
                "sequence_lengths": list(lengths),
                "seconds": records[index].seconds,
                "estimated_compile_seconds": (
                    max(0.0, records[index].seconds - steady_reference)
                    if steady_reference is not None else None
                ),
            }
            for index, lengths in first_use_indices
        ]
        peft_execution_paths: dict[str, int] = {}
        moe_execution_paths: dict[str, int] = {}
        moe_owner_runtime: list[dict[str, Any]] = []
        for module in uncompiled_model.modules():
            path = getattr(module, "last_execution_path", None)
            if isinstance(path, str):
                peft_execution_paths[path] = peft_execution_paths.get(path, 0) + 1
            moe_path = getattr(module, "last_execution_schedule", None)
            if isinstance(moe_path, str):
                moe_execution_paths[moe_path] = (
                    moe_execution_paths.get(moe_path, 0) + 1
                )
                selected_count = getattr(
                    module, "last_selected_assignment_count", None
                )
                packed_count = getattr(
                    module, "last_packed_slot_count", None
                )
                expert_count = getattr(module, "expert_count", None)
                top_k = getattr(module, "top_k", None)
                moe_owner_runtime.append({
                    "schedule": moe_path,
                    "expert_count": expert_count,
                    "top_k": top_k,
                    "selected_assignment_count": selected_count,
                    "physical_packed_slot_count": packed_count,
                    "ideal_selected_expert_fraction": (
                        float(top_k) / float(expert_count)
                        if isinstance(top_k, int)
                        and isinstance(expert_count, int)
                        and expert_count > 0 else None
                    ),
                })
        moe_numerical_contract = (
            "deterministic_no_drop_selected_block_bf16_hifi4_fp32_v1"
            if moe_execution_paths.get("dynamic_native_expert_bucketed", 0) > 0
            else "safe_bf16_reference"
        )
        metrics.update(
            {
                "status": (
                    "completed"
                    if state.optimizer_step >= planned_steps
                    else "interrupted"
                ),
                "model_parameters": model_parameters,
                "trainable_parameters": trainable_parameters,
                "planned_optimizer_steps": planned_steps,
                "warmup_steps": warmup_steps,
                "global_steps": state.optimizer_step,
                "epochs_completed": state.epoch,
                "step_losses": [record.loss for record in records],
                "step_learning_rates": [
                    record.learning_rate for record in records
                ],
                "step_e2e_seconds": step_e2e_seconds,
                "step_tokens_per_second": [
                    record.useful_tokens / record.seconds
                    for record in records
                ],
                "step_useful_tokens": [
                    record.useful_tokens for record in records
                ],
                "step_padded_tokens": [
                    record.padded_tokens for record in records
                ],
                "step_sequence_lengths": [
                    list(record.sequence_lengths) for record in records
                ],
                "step_data_wait_seconds": [
                    record.data_wait_seconds for record in records
                ],
                "checkpoint_events": checkpoint_events,
                "validation_history": validation_history,
                "batch_identity_history": batch_identity_history,
                "unique_host_batches": len(
                    {
                        item["host"]
                        for item in batch_identity_history
                    }
                ),
                "gradient_clipping": {
                    "max_grad_norm": config.max_grad_norm,
                    "implementation": operations.tt_clip_mode,
                    "history": gradient_norm_history,
                },
                "gradient_audit": operations.gradient_audit,
                "safe_evolving_audit": operations.evolving_audit,
                "peft_execution_paths": peft_execution_paths,
                "moe_execution_paths": moe_execution_paths,
                "moe_execution_contract": {
                    "owners": moe_owner_runtime,
                    "host_token_routing_loop": (
                        False if moe_execution_paths else None
                    ),
                    "evaluates_all_experts": (
                        False if moe_execution_paths else None
                    ),
                    "capacity_semantics": (
                        "exact_no_drop_per_expert_token_bound"
                        if moe_execution_paths.get(
                            "dynamic_native_expert_bucketed", 0
                        ) > 0 else None
                    ),
                    "routing_location": "device",
                    "expert_block_rows": (
                        32
                        if moe_execution_paths.get(
                            "dynamic_native_expert_bucketed", 0
                        ) > 0 else None
                    ),
                },
                "moe_numerical_contract": moe_numerical_contract,
                "ttnn_forward_graph": {
                    "requested": capture_ttnn_forward_graph,
                    "captured": operations.ttnn_forward_graph_captured,
                    "path": (
                        str(operations.ttnn_forward_graph_path)
                        if operations.ttnn_forward_graph_path is not None
                        else None
                    ),
                    "capture_after_warmup_steps": profile_warmup_steps,
                },
                "dropout_runtime": {
                    "contract": (
                        "explicit_stateless_inverted_dropout_v1"
                        if operations.training_dropout is not None
                        else "model_has_no_training_dropout"
                    ),
                    "initial_micro_step": resumed_micro_step,
                    "calls_per_forward": (
                        len(operations.training_dropout.modules)
                        if operations.training_dropout is not None
                        else 0
                    ),
                },
                "steady_step_seconds": (
                    steady_step_e2e_summary["mean_seconds"]
                ),
                "steady_measurement_steps": [
                    records[index].step
                    for index in steady_measurement_indices
                ],
                "checkpoint": checkpoint,
                "token_row_coupling": token_row_coupling,
                "optimizer_placement": (
                    "host_fp32_master"
                    if use_host_master_optimizer
                    else "device"
                ),
                "phase_seconds": phase_seconds,
                "phase_summaries": phase_summaries,
                "step_e2e_summary": step_e2e_summary,
                "steady_step_e2e_summary": steady_step_e2e_summary,
                "compile": {
                    "cache": str(xla_cache) if xla_cache is not None else None,
                    "first_step_seconds": (
                        step_e2e_seconds[0] if step_e2e_seconds else None
                    ),
                    "estimated_compile_seconds": estimated_compile_seconds,
                    "bucket_first_use": bucket_first_use,
                    "expected_sequence_buckets": sorted(
                        expected_compile_buckets
                    ),
                    "observed_sequence_buckets": sorted(
                        seen_sequence_lengths
                    ),
                    "cross_process_executable_reuse": False,
                    "limitation": (
                        "PJRT executable deserialization is not implemented; "
                        "the cache still reuses lower-level build artifacts"
                    ),
                },
            }
        )
        record_stage("checkpoint_and_export", "completed")
        metrics["startup_profile"]["current_stage"] = "completed"
        metrics["startup_profile"]["total_seconds"] = (
            time.perf_counter() - startup_started
        )
        publish_metrics()
        logger.info(f"Saved direct-worker metrics: {metrics_path}")
        return metrics
    except Exception as error:
        metrics.update(
            {
                "status": "failed",
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        metrics.setdefault("startup_profile", {})["current_stage"] = "failed"
        metrics["startup_profile"]["total_seconds"] = (
            time.perf_counter() - startup_started
        )
        publish_metrics()
        raise
    finally:
        logger.finish()


def main() -> None:
    arguments = _arguments()
    config = _load_config(arguments)
    result = run(
        config,
        audit_device_inputs=arguments.audit_device_inputs,
        stop_after=arguments.stop_after,
        backward_loss_scale=arguments.backward_loss_scale,
        fuse_step=arguments.fuse_step,
        profile_phases=arguments.profile_phases,
        profile_warmup_steps=arguments.profile_warmup_steps,
        capture_profile_ir=arguments.capture_profile_ir,
        capture_ttnn_forward_graph=arguments.capture_ttnn_forward_graph,
        optimizer_foreach=arguments.optimizer_foreach,
        optimizer_tile_packed=arguments.optimizer_tile_packed,
        optimizer_pack_bytes=arguments.optimizer_pack_bytes,
        optimizer_packing_strategy=arguments.optimizer_packing_strategy,
        split_clip_optimizer=arguments.split_clip_optimizer,
        split_optimizer_phases=arguments.split_optimizer_phases,
        split_after_forward=arguments.split_after_forward,
        auto_split_after_forward=arguments.auto_split_after_forward,
        split_after_loss=arguments.split_after_loss,
        packed_gradient_clip=arguments.packed_gradient_clip,
        attention_query_block=arguments.attention_query_block,
        attention_backend=arguments.attention_backend,
        peft_execution_strategy=arguments.peft_execution_strategy,
        moe_execution_strategy=arguments.moe_execution_strategy,
        loss_read_interval=arguments.loss_read_interval,
        xla_cache=arguments.xla_cache,
        audit_training_state=arguments.audit_training_state,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
