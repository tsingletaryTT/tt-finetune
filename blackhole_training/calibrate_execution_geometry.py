#!/usr/bin/env python3
"""Offline whole-step calibration for deterministic compiler policies.

This is deliberately not a normal training entry point.  It asks the public
trainer to measure a small macro-diverse candidate portfolio in fresh worker
processes, preserving the safe candidate as the evolving-state oracle.  The
result is development evidence used to improve deterministic, name-free
compiler rules; ordinary ``TTTrainer.train()`` never performs this search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument(
        "--target-modules",
        choices=("auto", "auto_all_linear"),
        default="auto_all_linear",
    )
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=3)
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
    trainer = TTTrainer(
        model=model,
        dataset=args.dataset,
        adapter_config=LoRAConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            target_modules=args.target_modules,
        ),
        data_config=DataConfig(
            format="text",
            assistant_only_loss=False,
            packing=False,
            sequence_bucketing=False,
            tokenization_workers=1,
            dataloader_workers=0,
            prefetch_batches=1,
        ),
        args=TrainingArguments(
            output_dir=str(args.output),
            run_name="offline-execution-calibration",
            max_steps=args.warmups + args.steps,
            num_train_epochs=1000,
            max_seq_length=args.sequence_length,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=2.0e-4,
            weight_decay=0.01,
            max_grad_norm=1.0,
            lr_scheduler_type="constant",
            logging_steps=1000,
            save_strategy="none",
            save_optimizer_state=True,
            runtime_mode="isolated",
            memory_mode="bf16",
            audit_training_state=False,
            auto_smoke=False,
            auto_tune=True,
            auto_tune_warmup_steps=args.warmups,
            auto_tune_measurement_steps=args.steps,
            auto_tune_max_candidates=args.candidates,
            compile_cache_dir=str(args.output / "profile_cache"),
            seed=3407,
        ),
    )
    result = trainer.compile()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "OFFLINE_EXECUTION_CALIBRATION.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
