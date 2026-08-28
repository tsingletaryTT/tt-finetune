#!/usr/bin/env python3
"""Run one semantic holdout through the normal FastTTModel/TTTrainer API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tt_finetune import FastTTModel, TTTrainer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--adapter")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    model = FastTTModel.from_pretrained(
        args.model,
        max_seq_length=args.sequence_length,
        adapter=args.adapter,
    )
    trainer = TTTrainer(
        model=model,
        dataset=args.dataset,
        rank=args.rank,
        max_steps=args.steps,
        max_seq_length=args.sequence_length,
        per_device_train_batch_size=1,
        learning_rate=2e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=args.steps,
        save_strategy="none",
        audit_training_state=True,
        auto_smoke=False,
        output_dir=str(args.output / "run"),
    )
    preflight = trainer.preflight()
    plan = trainer.plan()
    result = trainer.train()
    adapter = trainer.save_pretrained(args.output / "adapter")
    evidence = {
        "model": args.model,
        "rank": args.rank,
        "batch": 1,
        "sequence": args.sequence_length,
        "steps": args.steps,
        "preflight": preflight,
        "plan": plan,
        "result": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(result).items()
        },
        "adapter": str(adapter),
    }
    (args.output / "PUBLIC_HOLDOUT_QUALIFICATION.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
