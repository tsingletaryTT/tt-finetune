#!/usr/bin/env python3
"""Run pinned real-data tt-finetune qualification workloads.

This command deliberately uses only the public ``FastTTModel``/``TTTrainer``
surface for model construction, planning, compilation, training, resume, and
adapter export.  It may be launched with an installed wheel from any working
directory; the source checkout is not added to ``sys.path``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any


MODELS = {
    "small": {
        "id": "HuggingFaceTB/SmolLM2-135M",
        "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        "steps": 5000,
        "sequence": 128,
        "rank": 8,
        "memory_mode": "bf16",
        "save_steps": 500,
        "warmup_steps": 50,
    },
    "medium": {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "steps": 1000,
        "sequence": 256,
        "rank": 16,
        "memory_mode": "bf16",
        "save_steps": 100,
        "warmup_steps": 20,
    },
    "large": {
        "id": "Qwen/Qwen2.5-7B",
        "revision": "d149729398750b98c0af14eb82c78cfe92750796",
        "steps": 100,
        "sequence": 32,
        "rank": 8,
        "memory_mode": "bf4_frozen",
        "save_steps": 25,
        "warmup_steps": 5,
    },
    "moe": {
        "id": "ibm-granite/granite-3.1-1b-a400m-instruct",
        "revision": "0da7a48b0276d500ce5922fd2b33944091fc6c09",
        "steps": 100,
        "sequence": 128,
        "rank": 8,
        "memory_mode": "bf16",
        "save_steps": 25,
        "warmup_steps": 5,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: list[float], warmups: int = 4) -> dict[str, Any]:
    samples = [float(value) for value in values[warmups:]]
    if not samples:
        return {"warmup_count": warmups, "sample_count": 0}
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        index = fraction * (len(ordered) - 1)
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        weight = index - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "warmup_count": warmups,
        "sample_count": len(samples),
        "median_seconds": statistics.median(samples),
        "p10_seconds": percentile(0.10),
        "p90_seconds": percentile(0.90),
        "mean_seconds": statistics.fmean(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", choices=sorted(MODELS))
    parser.add_argument(
        "action", choices=("inspect", "smoke", "compile", "train")
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--auto-tune", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--runtime-mode", choices=("isolated", "persistent"), default="persistent")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Continue from a previously exported adapter through the public API.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path.cwd() / "train.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.cwd() / "tt-finetune-real-workloads",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    from tt_finetune import (
        DataConfig,
        FastTTModel,
        TTTrainer,
        TrainingArguments,
    )

    spec = dict(MODELS[arguments.workload])
    if arguments.steps is not None:
        spec["steps"] = arguments.steps
    if arguments.logging_steps <= 0:
        raise ValueError("--logging-steps must be positive")
    if arguments.save_steps is not None:
        if arguments.save_steps <= 0:
            raise ValueError("--save-steps must be positive")
        spec["save_steps"] = arguments.save_steps
    model_path = (
        arguments.model_path.expanduser().resolve()
        if arguments.model_path is not None else None
    )
    model_source = str(model_path) if model_path is not None else spec["id"]
    dataset = arguments.dataset.expanduser().resolve()
    adapter = (
        arguments.adapter.expanduser().resolve()
        if arguments.adapter is not None
        else None
    )
    output = arguments.output_root.expanduser().resolve() / arguments.workload
    if arguments.action == "smoke":
        output = arguments.output_root.expanduser().resolve() / f"{arguments.workload}_smoke"
    if model_path is not None and not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path)
    if not dataset.is_file():
        raise FileNotFoundError(dataset)

    model = FastTTModel.from_pretrained(
        model_source,
        max_seq_length=spec["sequence"],
        revision=(None if model_path is not None else spec["revision"]),
        adapter=adapter,
    )
    training_arguments = TrainingArguments(
        output_dir=str(output),
        run_name=f"real-workload-{arguments.workload}",
        max_steps=spec["steps"],
        num_train_epochs=10000,
        max_seq_length=spec["sequence"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_steps=spec["warmup_steps"],
        logging_steps=arguments.logging_steps,
        save_strategy="steps",
        save_steps=spec["save_steps"],
        save_optimizer_state=True,
        keep_last_checkpoints=3,
        runtime_mode=("isolated" if arguments.profile_memory else arguments.runtime_mode),
        memory_mode=spec["memory_mode"],
        audit_training_state=True,
        profile_device_memory=arguments.profile_memory,
        auto_smoke=False,
        auto_tune=arguments.auto_tune,
        auto_tune_warmup_steps=4,
        auto_tune_measurement_steps=20,
        # Real-workload hardening exercises the safe plan and both materially
        # different graph partitions. Minor optimizer-pack variants are
        # already covered by the planner qualification matrix and can each
        # trigger a full cold graph compile in this installed PJRT build.
        auto_tune_max_candidates=3,
        seed=arguments.seed,
    )
    trainer = TTTrainer(
        model=model,
        dataset=dataset,
        rank=spec["rank"],
        target_modules="auto",
        data_config=DataConfig(
            format="prompt_response",
            assistant_only_loss=True,
            packing="auto",
            sequence_bucketing="auto",
            tokenization_workers="auto",
            dataloader_workers="auto",
            prefetch_batches=2,
        ),
        args=training_arguments,
    )

    common = {
        "format": "tt-finetune-real-workload-run-v1",
        "workload": arguments.workload,
        "action": arguments.action,
        "model": {
            "id": spec["id"],
            "revision": spec["revision"],
            "local_snapshot": str(model_path) if model_path is not None else None,
        },
        "dataset": {
            "path": str(dataset),
            "sha256": _sha256(dataset),
        },
        "configuration": {
            "steps": spec["steps"],
            "sequence_length": spec["sequence"],
            "rank": spec["rank"],
            "memory_mode": spec["memory_mode"],
            "target_modules": "auto",
            "seed": arguments.seed,
            "initial_adapter": str(adapter) if adapter is not None else None,
            "auto_tune": arguments.auto_tune,
            "runtime_mode": training_arguments.runtime_mode,
            "logging_steps": arguments.logging_steps,
            "save_steps": spec["save_steps"],
        },
        "installed_package": str(Path(sys.modules["tt_finetune"].__file__).resolve()),
    }
    started = time.perf_counter()
    if arguments.action == "inspect":
        payload = {
            **common,
            "status": "completed",
            "support": TTTrainer.supports(
                model=model_source,
                rank=spec["rank"],
                batch_size=1,
                sequence_length=spec["sequence"],
                target_modules="auto",
                memory_mode=spec["memory_mode"],
            ).to_dict(),
            "plan": trainer.plan(),
            "data_plan": trainer.data_plan(),
        }
    elif arguments.action == "smoke":
        result = trainer.smoke(2)
        payload = {
            **common,
            "status": "completed",
            "elapsed_seconds": time.perf_counter() - started,
            "smoke_result": result.to_dict() if hasattr(result, "to_dict") else vars(result),
        }
    elif arguments.action == "compile":
        compile_result = trainer.compile()
        payload = {
            **common,
            "status": "completed",
            "elapsed_seconds": time.perf_counter() - started,
            "compile": compile_result,
            "compile_cache": trainer.compile_cache(),
        }
    else:
        result = trainer.train(
            resume_from_checkpoint=arguments.resume,
            stop_after=arguments.stop_after,
        )
        metrics_path = Path(result.metrics_path).resolve()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        losses = [
            float(value) for value in metrics.get("step_losses", [])
            if value is not None
        ]
        step_times = [float(value) for value in metrics.get("step_e2e_seconds", [])]
        useful_tokens = [int(value) for value in metrics.get("step_useful_tokens", [])]
        padded_tokens = [int(value) for value in metrics.get("step_padded_tokens", [])]
        first_recorded_step = int(metrics.get("global_steps") or 0) - len(
            step_times
        ) + 1
        steady_steps = set(metrics.get("steady_measurement_steps") or [])
        steady_indices = [
            index
            for index in range(len(step_times))
            if first_recorded_step + index in steady_steps
        ]
        if not steady_indices:
            steady_indices = list(range(min(4, len(step_times)), len(step_times)))
        steady_time_total = sum(step_times[index] for index in steady_indices)
        export_path = None
        if arguments.export and metrics.get("status") == "completed":
            export_path = trainer.save_pretrained(output / "exported_adapter")
        checks = {
            "metrics_completed_or_intentionally_interrupted": metrics.get("status") in {"completed", "interrupted"},
            "requested_boundary_reached": metrics.get("global_steps") == (arguments.stop_after or spec["steps"]),
            "reported_losses_present_and_finite": bool(losses) and all(math.isfinite(value) for value in losses),
            "audited_gradients_finite": (
                metrics.get("gradient_audit", {}).get(
                    "nonfinite_tensor_count"
                )
                == 0
            ),
            "final_parameters_finite": bool(metrics.get("final_training_state", {}).get("trainable_finiteness", {}).get("finite")),
            "final_optimizer_finite": bool(metrics.get("final_training_state", {}).get("optimizer_finiteness", {}).get("finite")),
            "trainable_state_changed": metrics.get("initial_training_state", {}).get("trainable", {}).get("sha256") != metrics.get("final_training_state", {}).get("trainable", {}).get("sha256"),
        }
        payload = {
            **common,
            "status": "accepted" if all(checks.values()) else "failed",
            "elapsed_seconds": time.perf_counter() - started,
            "resume_requested": arguments.resume,
            "stop_after": arguments.stop_after,
            "checks": checks,
            "metrics_path": str(metrics_path),
            "metrics_sha256": _sha256(metrics_path),
            "adapter_dir": str(Path(result.adapter_dir).resolve()),
            "exported_adapter": str(export_path) if export_path else None,
            "global_steps": metrics.get("global_steps"),
            "compile": metrics.get("compile"),
            "persistent_runtime": metrics.get("persistent_runtime"),
            "selected_execution": metrics.get("automatic_execution_plan"),
            "architecture_execution": {
                "backend": (metrics.get("architecture") or {}).get(
                    "executor_backend"
                ),
                "fallback_reason": (metrics.get("architecture") or {}).get(
                    "executor_fallback_reason"
                ),
            },
            "step_time": (
                metrics.get("steady_step_e2e_summary")
                or _summary(step_times)
            ),
            "throughput": {
                "useful_tokens_per_second": (
                    sum(useful_tokens[index] for index in steady_indices)
                    / steady_time_total
                    if steady_time_total else None
                ),
                "padded_tokens_per_second": (
                    sum(padded_tokens[index] for index in steady_indices)
                    / steady_time_total
                    if steady_time_total else None
                ),
                "useful_token_fraction": (sum(useful_tokens) / sum(padded_tokens) if padded_tokens and sum(padded_tokens) else None),
            },
            "memory_accounting": metrics.get("memory_accounting"),
            "initial_training_state": metrics.get("initial_training_state"),
            "final_training_state": metrics.get("final_training_state"),
            "checkpoint_events": metrics.get("checkpoint_events"),
            "peft_execution_paths": metrics.get("peft_execution_paths"),
            "moe_execution_paths": metrics.get("moe_execution_paths"),
            "moe_execution_contract": metrics.get(
                "moe_execution_contract"
            ),
            "moe_install": (
                metrics.get("architecture", {}).get("moe_install")
            ),
            "moe_numerical_contract": metrics.get("moe_numerical_contract"),
            "gradient_audit": metrics.get("gradient_audit"),
            "loss_samples": losses,
        }

    evidence_dir = arguments.output_root.expanduser().resolve() / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    suffix = "resume" if arguments.resume else "initial"
    if arguments.stop_after is not None:
        suffix += f"-stop{arguments.stop_after}"
    destination = evidence_dir / f"{arguments.workload}-{arguments.action}-{suffix}.json"
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload.get("status") in {"completed", "accepted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
