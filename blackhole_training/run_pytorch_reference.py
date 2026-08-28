#!/usr/bin/env python3
"""Run a matched CPU PyTorch/PEFT authority from a resolved TT config.

The reference deliberately reuses the same model loader, PEFT target list,
tokenizer, deterministic epoch planner, collator, optimizer settings, learning
rate schedule, and checkpoint format as the TT worker.  Only device execution
and TT-specific optimizer/graph options are disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--stop-after",
        type=int,
        help=(
            "Stop after this optimizer step while retaining the configured "
            "full-run learning-rate schedule."
        ),
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument(
        "--moe-execution-strategy",
        choices=("safe", "dynamic"),
        default="safe",
        help=(
            "Use the ordinary Transformers MoE path or the same no-drop "
            "dynamic expert owner as the TT candidate."
        ),
    )
    parser.add_argument(
        "--initial-adapter",
        type=Path,
        help="PEFT adapter containing the exact TT initial trainable state",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="last",
        help=(
            "Resume the matched reference from the latest checkpoint, or "
            "from the explicit checkpoint path when supplied."
        ),
    )
    return parser.parse_args()


def _latest_checkpoint(output: Path) -> Path:
    history_path = output / "checkpoints" / "checkpoint_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    checkpoints = history.get("checkpoints") or []
    if not checkpoints:
        raise RuntimeError("matched reference produced no training checkpoint")
    entry = checkpoints[-1]
    path = Path(entry["path"]).expanduser()
    if not path.is_file():
        path = history_path.parent / str(entry["name"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _export_adapter(payload: dict, output: Path) -> Path:
    destination = output / "peft_adapter"
    targets = payload["lora_target_modules"]
    if isinstance(targets, str):
        targets = [targets]
    command = [
        sys.executable,
        "-m",
        "blackhole_training.export_peft_adapter",
        "--base-model",
        str(payload["model_name"]),
        "--metadata-base-model",
        str(payload.get("model_metadata_name") or payload["model_name"]),
        "--checkpoint",
        str(_latest_checkpoint(output)),
        "--output",
        str(destination),
        "--adapter-type",
        str(payload.get("adapter_type") or "lora"),
        "--targets",
        *[str(value) for value in targets],
        "--dropout",
        str(payload.get("lora_dropout") or 0.0),
        "--bias",
        str(payload.get("lora_bias") or "none"),
        "--task-type",
        str(payload.get("lora_task_type") or "CAUSAL_LM"),
        "--init-weights",
        json.dumps(payload.get("lora_init_weights", True)),
    ]
    if payload.get("model_revision"):
        command.extend(["--model-revision", str(payload["model_revision"])])
    if payload.get("trust_remote_code"):
        command.append("--trust-remote-code")
    if payload.get("attn_implementation"):
        command.extend(
            ["--attn-implementation", str(payload["attn_implementation"])]
        )
    if not payload.get("use_fast_tokenizer", True):
        command.append("--slow-tokenizer")
    if payload.get("adapter_type", "lora") == "ia3":
        modules = payload.get("ia3_feedforward_modules") or []
        if isinstance(modules, str):
            modules = [modules]
        command.extend(
            ["--feedforward-modules", *[str(value) for value in modules]]
        )
        command.extend(
            [
                "--init-ia3-weights",
                json.dumps(payload.get("init_ia3_weights", True)),
            ]
        )
    else:
        command.extend(
            [
                "--rank",
                str(payload["lora_r"]),
                "--alpha",
                str(payload["lora_alpha"]),
            ]
        )
        if payload.get("lora_use_rslora"):
            command.append("--use-rslora")
        if payload.get("lora_use_dora"):
            command.append("--use-dora")
        if payload.get("lora_plus_lr_ratio") is not None:
            command.extend(
                [
                    "--lora-plus-lr-ratio",
                    str(payload["lora_plus_lr_ratio"]),
                ]
            )
        if payload.get("lora_layers_to_transform"):
            command.extend(
                [
                    "--layers-to-transform",
                    *[
                        str(value)
                        for value in payload["lora_layers_to_transform"]
                    ],
                ]
            )
        if payload.get("lora_layers_pattern"):
            command.extend(
                ["--layers-pattern", str(payload["lora_layers_pattern"])]
            )
    if payload.get("lora_modules_to_save"):
        command.extend(
            [
                "--modules-to-save",
                *[str(value) for value in payload["lora_modules_to_save"]],
            ]
        )
    subprocess.run(command, check=True)
    required = (
        destination / "adapter_config.json",
        destination / "adapter_model.safetensors",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError("matched reference adapter export is incomplete")
    return destination.resolve()


def main() -> int:
    arguments = _arguments()
    source = arguments.resolved_config.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("resolved config must contain a YAML mapping")

    payload.update(
        {
            "use_tt": False,
            "profile_device_memory": False,
            "deterministic": True,
            "project_dir": str(output),
            "wandb_project": "tt-finetune-reference",
            "wandb_run_name": "pytorch-peft-reference",
            "steps_freq": arguments.logging_steps,
            "save_steps": arguments.save_steps,
            "keep_last_n": 3,
            "resume_from_checkpoint": bool(
                arguments.resume_from_checkpoint
            ),
        }
    )
    if arguments.resume_from_checkpoint:
        if arguments.resume_from_checkpoint == "last":
            payload["resume_option"] = "last"
        else:
            payload["resume_option"] = "path"
            payload["checkpoint_path"] = str(
                Path(arguments.resume_from_checkpoint)
                .expanduser()
                .resolve()
            )
    if arguments.steps is not None:
        if arguments.steps <= 0:
            raise ValueError("steps must be positive")
        payload["max_steps"] = arguments.steps
    if arguments.stop_after is not None:
        if arguments.stop_after <= 0:
            raise ValueError("stop-after must be positive")
        if arguments.stop_after > int(payload["max_steps"]):
            raise ValueError("stop-after cannot exceed configured max_steps")
    if arguments.initial_adapter is not None:
        initial_adapter = arguments.initial_adapter.expanduser().resolve()
        if not (initial_adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(initial_adapter)
        payload["initial_adapter_path"] = str(initial_adapter)
        payload["initial_adapter_name"] = "default"

    # TT-only graph/runtime controls must not leak into the reference path.
    payload["optimizer_layout"] = "per_parameter"
    payload["execution_graph_layout"] = "monolithic"
    payload["lora_backend"] = (
        "auto"
        if arguments.moe_execution_strategy == "dynamic"
        else "generic"
    )
    payload["lora_resident_full_mlp_backward"] = False
    payload["lora_resident_output_attention_backward"] = False

    output.mkdir(parents=True, exist_ok=True)
    reference_config_path = output / "resolved_reference_config.yaml"
    reference_config_path.write_text(
        yaml.safe_dump(payload, sort_keys=True),
        encoding="utf-8",
    )

    # The direct worker uses this superset config for every supported causal
    # LM family; architecture discovery still comes from the loaded model.
    if str(payload.get("dtype")) in {"torch.bfloat16", "bfloat16", "bf16"}:
        # Some x86 oneDNN builds advertise BF16 forward support but reject
        # GELU backward at execution time.  The ordinary PyTorch kernels are
        # a valid deterministic CPU authority, so probe once and disable only
        # the unsupported acceleration path.  This does not affect TT runs.
        probe = torch.ones(4, dtype=torch.bfloat16, requires_grad=True)
        try:
            torch.nn.functional.gelu(probe).sum().backward()
        except RuntimeError as error:
            if "DNNL does not support bf16/f16 backward" not in str(error):
                raise
            torch.backends.mkldnn.enabled = False

    from blacksmith.experiments.torch.qwen.configs import TrainingConfig
    from tt_finetune.worker import run

    config = TrainingConfig(**payload)
    started = time.perf_counter()
    metrics = run(
        config,
        stop_after=arguments.stop_after,
        audit_training_state=True,
        loss_read_interval=1,
        optimizer_foreach=False,
        optimizer_tile_packed=False,
        packed_gradient_clip=False,
        peft_execution_strategy="framework_safe",
        moe_execution_strategy=arguments.moe_execution_strategy,
    )
    adapter_dir = _export_adapter(payload, output)
    metrics_path = output / "training_metrics.json"
    expected_boundary = arguments.stop_after or int(payload["max_steps"])
    accepted = (
        metrics.get("global_steps") == expected_boundary
        and metrics.get("status")
        == ("interrupted" if arguments.stop_after is not None else "completed")
    )
    result = {
        "format": "tt-finetune-pytorch-reference-v1",
        "status": "accepted" if accepted else "failed",
        "training_status": metrics.get("status"),
        "authority": "matched PyTorch/Transformers/PEFT CPU training",
        "moe_execution_strategy": arguments.moe_execution_strategy,
        "source_config": str(reference_config_path),
        "source_config_sha256": _sha256(reference_config_path),
        "initial_adapter": payload.get("initial_adapter_path"),
        "resume_from_checkpoint": arguments.resume_from_checkpoint,
        "output": str(output),
        "elapsed_seconds": time.perf_counter() - started,
        "steps": metrics.get("global_steps"),
        "configured_steps": payload.get("max_steps"),
        "stop_after": arguments.stop_after,
        "model": metrics.get("model_name"),
        "data_pipeline": metrics.get("data_pipeline"),
        "initial_training_state": metrics.get("initial_training_state"),
        "final_training_state": metrics.get("final_training_state"),
        "step_losses": metrics.get("step_losses"),
        "step_learning_rates": metrics.get("step_learning_rates"),
        "batch_identity_history": metrics.get("batch_identity_history"),
        "metrics_path": str(metrics_path),
        "metrics_sha256": _sha256(metrics_path),
        "adapter_dir": str(adapter_dir),
        "adapter": {
            "config_sha256": _sha256(adapter_dir / "adapter_config.json"),
            "weights_sha256": _sha256(
                adapter_dir / "adapter_model.safetensors"
            ),
        },
    }
    evidence = output / "pytorch_reference.json"
    evidence.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
