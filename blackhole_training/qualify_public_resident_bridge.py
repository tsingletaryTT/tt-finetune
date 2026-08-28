#!/usr/bin/env python3
"""Qualify the normal TTTrainer API against the frozen resident authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument(
        "--target-modules",
        choices=("auto", "auto_all_linear"),
        default="auto_all_linear",
    )
    parser.add_argument(
        "--adapter-backend",
        choices=("auto", "generic"),
        default="auto",
        help="generic is an internal matched safe-reference authority",
    )
    parser.add_argument(
        "--expected-executor",
        choices=("resident_exact", "safe_torch_xla_autograd"),
        default="resident_exact",
    )
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=1000)
    parser.add_argument(
        "--profile-phases",
        action="store_true",
        help="capture synchronized public resident phase boundaries",
    )
    parser.add_argument(
        "--sequence-buckets",
        type=int,
        nargs="+",
        help=(
            "enable deterministic logical sequence bucketing using these "
            "32-token boundaries (for example: 32 64 128)"
        ),
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="capture every evolving logical tensor hash (not for timing)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()

    from tt_finetune import (
        DataConfig,
        FastTTModel,
        LoRAConfig,
        TTTrainer,
        TrainingArguments,
    )

    model = FastTTModel.from_pretrained(
        args.model,
        max_seq_length=args.sequence_length,
    )
    if args.adapter is not None:
        model = model.load_adapter(args.adapter)
    training = TrainingArguments(
        output_dir=str(args.output),
        run_name="public-resident-qualification",
        max_steps=args.steps,
        num_train_epochs=1000,
        max_seq_length=args.sequence_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2.0e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="constant",
        # Logging is a separately measured product boundary.  The matched
        # resident authority does not copy the scalar loss to the host inside
        # every steady-state step.
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=max(1, args.steps // 2),
        save_optimizer_state=True,
        runtime_mode="isolated",
        memory_mode="bf16",
        audit_training_state=args.audit,
        profile_product_boundaries=args.profile_phases,
        auto_smoke=False,
        auto_tune=False,
        seed=3407,
    )
    adapter_arguments = (
        {}
        if args.adapter is not None
        else {
            "adapter_config": LoRAConfig(
                r=args.rank,
                lora_alpha=2 * args.rank,
                target_modules=args.target_modules,
                backend=args.adapter_backend,
            )
        }
    )
    trainer = TTTrainer(
        model=model,
        dataset=args.dataset,
        **adapter_arguments,
        # The accepted resident state contains Q/K/V/O and gate/up/down LoRA
        # for every decoder layer.  This is a training-semantic contract, not
        # a model-name selector.
        data_config=DataConfig(
            format="text",
            assistant_only_loss=False,
            packing=False,
            sequence_bucketing=args.sequence_buckets is not None,
            sequence_buckets=(
                tuple(args.sequence_buckets)
                if args.sequence_buckets is not None else None
            ),
            tokenization_workers=1,
            dataloader_workers=0,
            prefetch_batches=1,
        ),
        args=training,
    )
    result = trainer.train(
        resume_from_checkpoint=args.resume,
        stop_after=args.stop_after,
    )
    metrics = json.loads(Path(result.metrics_path).read_text())
    execution = metrics.get("execution", {})
    if args.adapter_backend == "generic" and not execution:
        execution = {
            "selected_plan": "safe_torch_xla_autograd",
            "actual_executor": "safe_torch_xla_autograd",
            "fallback": "explicit qualification-only generic adapter backend",
        }
    expected_status = "interrupted" if args.stop_after else "completed"
    accepted = (
        metrics.get("status") == expected_status
        and execution.get("actual_executor") == args.expected_executor
        and (
            execution.get("fallback") is None
            if args.expected_executor == "resident_exact"
            else True
        )
    )
    report = {
        "status": "accepted" if accepted else "rejected",
        "model": args.model,
        "rank": args.rank,
        "target_modules": args.target_modules,
        "adapter_backend": args.adapter_backend,
        "expected_executor": args.expected_executor,
        "batch_size": 1,
        "sequence_length": args.sequence_length,
        "steps": args.steps,
        "stop_after": args.stop_after,
        "warmups": args.warmups,
        "execution": execution,
        "step_e2e_summary": metrics.get("step_e2e_summary"),
        "step_sequence_lengths": metrics.get("step_sequence_lengths"),
        "step_useful_tokens": metrics.get("step_useful_tokens"),
        "step_padded_tokens": metrics.get("step_padded_tokens"),
        "data_pipeline": metrics.get("data_pipeline"),
        "compile": metrics.get("compile"),
        "checkpoint": metrics.get("checkpoint"),
        "gradient_audit": metrics.get("gradient_audit"),
        # The full audit contains one digest per evolving tensor and belongs in
        # training_metrics.json.  Keep the human-facing qualification report
        # bounded so an exactness run does not print several megabytes or copy
        # that evidence a second time.
        "resident_evolving_audit": {
            "captured": bool(metrics.get("resident_evolving_audit")),
            "step_count": len(metrics.get("resident_evolving_audit") or []),
            "authoritative_path": str(result.metrics_path),
        },
        "safe_evolving_audit": {
            "captured": bool(metrics.get("safe_evolving_audit")),
            "step_count": len(metrics.get("safe_evolving_audit") or []),
            "authoritative_path": str(result.metrics_path),
        },
        "final_training_state": metrics.get("final_training_state"),
        "metrics_path": str(result.metrics_path),
    }
    report_path = args.output / "PUBLIC_RESIDENT_BRIDGE.report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
