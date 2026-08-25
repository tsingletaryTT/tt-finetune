#!/usr/bin/env python3
"""Render and execute a documented foundation + multi-LoRA Blackhole pipeline."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Pipeline:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path.resolve()
        self.spec = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))
        self.name = self.spec["name"]
        default_workspace = Path.home() / ".cache" / "tt-finetune" / "workspace"
        self.workspace = Path(self.spec.get("workspace", default_workspace)).resolve()
        self.container_workspace = self.spec.get(
            "container_workspace", str(self.workspace)
        ).rstrip("/")
        self.engine_dir = Path(__file__).resolve().parent
        self.generated_dir = (
            self.workspace / "blackhole_training" / "generated" / self.name
        )
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.run_log = self.generated_dir / "RUN_LOG.md"
        self.state_path = self.generated_dir / "state.json"
        self.manifest_fingerprint = hashlib.sha256(
            json.dumps(self.spec, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.allow_state_migration = False
        self.image = self.spec.get(
            "image",
            "ghcr.io/tenstorrent/tt-xla/tt-xla-base-ubuntu-24-04:latest",
        )
        self._validate()

    @property
    def foundation_enabled(self):
        return self.spec.get("foundation", {}).get("enabled", True)

    @property
    def adapter_base_model(self):
        if self.foundation_enabled:
            return self.spec["foundation"]["warmed_path"]
        return self.spec["model"]["name"]

    def _validate(self):
        if not self.name.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Pipeline name may contain only letters, numbers, _ and -")
        if not isinstance(self.spec.get("data"), dict):
            raise ValueError("Pipeline requires a data mapping")
        stages = self.spec["data"].get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("Pipeline requires at least one data.stages entry")
        if not isinstance(self.spec.get("adapters"), dict):
            raise ValueError("Pipeline requires an adapters mapping")
        for key in ("learning_rate", "batch_size", "rank", "alpha", "targets", "output_root"):
            if key not in self.spec["adapters"]:
                raise ValueError(f"Pipeline adapters.{key} is required")
        stage_names = [stage.get("name") for stage in stages]
        invalid_names = [
            name
            for name in stage_names
            if not isinstance(name, str)
            or not name.replace("_", "").replace("-", "").isalnum()
        ]
        if invalid_names:
            raise ValueError(
                f"Stage names may contain only letters, numbers, _ and -: {invalid_names}"
            )
        if len(stage_names) != len(set(stage_names)):
            raise ValueError(f"Duplicate stage names: {stage_names}")
        for stage in stages:
            for key in ("train_file", "max_length"):
                if not stage.get(key):
                    raise ValueError(
                        f"Stage {stage['name']!r} requires {key}"
                    )
            output_subdir = stage.get("output_subdir", stage["name"])
            if output_subdir not in (None, "", "."):
                subdir = Path(str(output_subdir))
                if subdir.is_absolute() or ".." in subdir.parts:
                    raise ValueError(
                        f"Stage {stage['name']!r} has unsafe output_subdir: "
                        f"{output_subdir!r}"
                    )
        if self.foundation_enabled:
            for key in ("initialized_path", "warmed_path", "token_contract"):
                self.container_to_host(self.spec["foundation"][key])
        elif not self.spec.get("model", {}).get("name"):
            raise ValueError(
                "A pipeline without foundation warmup requires model.name"
            )

    def load_state(self):
        if not self.state_path.exists():
            return {
                "pipeline": self.name,
                "manifest_fingerprint": self.manifest_fingerprint,
                "phases": {},
            }
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("manifest_fingerprint") != self.manifest_fingerprint:
            if self.allow_state_migration:
                history = self.generated_dir / "state_history"
                history.mkdir(parents=True, exist_ok=True)
                old_fingerprint = state.get("manifest_fingerprint", "unknown")[:16]
                archived = history / (
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                    f"{old_fingerprint}.json"
                )
                self.state_path.replace(archived)
                self.write_log(
                    "Resume state migration",
                    "completed",
                    {
                        "archived_state": archived,
                        "reason": "manifest changed; artifacts require exact topology verification",
                    },
                )
                return {
                    "pipeline": self.name,
                    "manifest_fingerprint": self.manifest_fingerprint,
                    "phases": {},
                }
            raise ValueError(
                f"Resume state belongs to a different manifest: {self.state_path}"
            )
        return state

    def mark_state(self, phase, status, **details):
        state = self.load_state()
        state["updated_at"] = now()
        state["phases"][phase] = {
            "status": status,
            "updated_at": state["updated_at"],
            **details,
        }
        self.state_path.write_text(
            json.dumps(state, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def valid_hf_model(path):
        return path.is_dir() and (path / "config.json").is_file()

    @staticmethod
    def valid_peft_adapter(path):
        return (
            path.is_dir()
            and (path / "adapter_config.json").is_file()
            and (
                (path / "adapter_model.safetensors").is_file()
                or (path / "adapter_model.bin").is_file()
            )
        )

    @staticmethod
    def peft_config_mismatches(path, training_config, base_model):
        adapter_config = json.loads(
            (path / "adapter_config.json").read_text(encoding="utf-8")
        )
        expected_targets = sorted(training_config["lora_target_modules"])
        actual_targets = sorted(adapter_config.get("target_modules") or [])
        checks = {
            "base_model_name_or_path": (
                adapter_config.get("base_model_name_or_path"),
                base_model,
            ),
            "r": (adapter_config.get("r"), training_config["lora_r"]),
            "lora_alpha": (
                adapter_config.get("lora_alpha"),
                training_config["lora_alpha"],
            ),
            "target_modules": (actual_targets, expected_targets),
            "lora_dropout": (
                adapter_config.get("lora_dropout", 0.0),
                training_config.get("lora_dropout", 0.0),
            ),
            "bias": (
                adapter_config.get("bias", "none"),
                training_config.get("lora_bias", "none"),
            ),
            "use_rslora": (
                adapter_config.get("use_rslora", False),
                training_config.get("lora_use_rslora", False),
            ),
            "use_dora": (
                adapter_config.get("use_dora", False),
                training_config.get("lora_use_dora", False),
            ),
        }
        return {
            key: {"actual": actual, "expected": expected}
            for key, (actual, expected) in checks.items()
            if actual != expected
        }

    @staticmethod
    def sha256_file(path):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def base_fingerprint(self):
        base = self.adapter_base_model
        if not str(base).startswith(self.container_workspace + "/"):
            return hashlib.sha256(f"hub:{base}".encode("utf-8")).hexdigest()
        root = self.container_to_host(base)
        digest = hashlib.sha256()
        for path in sorted(root.iterdir()):
            if not path.is_file():
                continue
            digest.update(path.name.encode("utf-8"))
            digest.update(str(path.stat().st_size).encode("ascii"))
            if path.name in {
                "config.json",
                "generation_config.json",
                "special_tokens_map.json",
                "token_contract.json",
                "tokenizer_config.json",
            }:
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def stage_project_dir(self, stage):
        output_root = self.spec["adapters"]["output_root"].rstrip("/")
        output_subdir = stage.get("output_subdir", stage["name"])
        if output_subdir in (None, "", "."):
            return output_root
        return f"{output_root}/{str(output_subdir).strip('/')}"

    def verify_artifacts(self):
        adapter = self.spec["adapters"]
        base_fingerprint = self.base_fingerprint()
        records = []
        failures = []
        warnings = []
        seen_hashes = {}
        require_distinct = self.spec.get("execution", {}).get(
            "require_distinct_adapter_weights",
            False,
        )
        for stage in self.spec["data"]["stages"]:
            name = stage["name"]
            root = self.container_to_host(
                f"{self.stage_project_dir(stage)}/peft_adapter"
            )
            if not self.valid_peft_adapter(root):
                failures.append(f"{name}: incomplete PEFT directory")
                continue
            config = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
            configured_base = config.get("base_model_name_or_path")
            if configured_base != self.adapter_base_model:
                failures.append(
                    f"{name}: PEFT base is {configured_base!r}, expected "
                    f"{self.adapter_base_model!r}"
                )
            weights = (
                root / "adapter_model.safetensors"
                if (root / "adapter_model.safetensors").is_file()
                else root / "adapter_model.bin"
            )
            weight_hash = self.sha256_file(weights)
            if weight_hash in seen_hashes:
                message = f"{name}: adapter weights equal {seen_hashes[weight_hash]}"
                if require_distinct:
                    failures.append(message)
                else:
                    warnings.append(message)
            seen_hashes[weight_hash] = name
            records.append(
                {
                    "adapter_type": str(
                        config.get("peft_type", "LORA")
                    ).lower(),
                    "stage": name,
                    "path": str(root),
                    "base_model": configured_base,
                    "base_fingerprint": base_fingerprint,
                    "weight_sha256": weight_hash,
                    "weight_bytes": weights.stat().st_size,
                    "rank": config.get("r"),
                    "alpha": config.get("lora_alpha"),
                    "targets": config.get("target_modules"),
                }
            )
        status = "verified" if not failures and len(records) == len(self.spec["data"]["stages"]) else "failed"
        report = {
            "pipeline": self.name,
            "verified_at": now(),
            "status": status,
            "base_model": self.adapter_base_model,
            "base_fingerprint": base_fingerprint,
            "adapters": records,
            "failures": failures,
            "warnings": warnings,
        }
        json_path = self.generated_dir / "ARTIFACTS.json"
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        markdown_path = self.generated_dir / "ARTIFACTS.md"
        lines = [
            f"# Verified Artifacts: {self.name}",
            "",
            f"- Status: **{status}**",
            f"- Base model: `{self.adapter_base_model}`",
            f"- Base fingerprint: `{base_fingerprint}`",
            "",
            "| Adapter | Rank | Alpha | Weight SHA-256 |",
            "|---|---:|---:|---|",
        ]
        lines.extend(
            f"| {item['stage']} | {item['rank']} | {item['alpha']} | "
            f"`{item['weight_sha256']}` |"
            for item in records
        )
        if failures:
            lines.extend(["", "## Failures", ""])
            lines.extend(f"- {failure}" for failure in failures)
        if warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in warnings)
        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.write_log(
            "Artifact verification",
            status,
            {"report": markdown_path, "adapter_count": len(records)},
        )
        if failures:
            raise RuntimeError("Artifact verification failed: " + "; ".join(failures))
        return report

    def container_to_host(self, value):
        candidate = Path(str(value)).expanduser()
        if candidate.is_absolute() and not str(value).startswith(
            self.container_workspace + "/"
        ):
            return candidate.resolve()
        prefix = self.container_workspace + "/"
        if not str(value).startswith(prefix):
            raise ValueError(f"Container path must start with {prefix}: {value}")
        return self.workspace / str(value)[len(prefix) :]

    def write_log(self, title, status, details=None):
        if not self.run_log.exists():
            self.run_log.write_text(
                f"# Pipeline Run Log: {self.name}\n\n"
                f"Manifest: `{self.manifest_path}`\n\n",
                encoding="utf-8",
            )
        with self.run_log.open("a", encoding="utf-8") as output:
            output.write(f"## {now()} — {title}\n\n")
            output.write(f"- Status: **{status}**\n")
            for key, value in (details or {}).items():
                output.write(f"- {key}: `{value}`\n")
            output.write("\n")

    def base_training_fields(self, run_name, project_dir):
        return {
            "log_level": "INFO",
            "use_wandb": False,
            "wandb_project": self.spec.get("wandb_project", self.name),
            "wandb_run_name": run_name,
            "wandb_tags": ["blackhole", self.name],
            "steps_freq": 1,
            "val_steps_freq": 1000,
            "epoch_freq": 1,
            "print_examples": False,
            "ignored_index": -100,
            "resume_from_checkpoint": False,
            "save_strategy": "none",
            "project_dir": project_dir,
            "save_optim": False,
            "storage_backend": "local",
            "sync_to_storage": False,
            "load_from_storage": False,
            "remote_path": "",
            "seed": self.spec.get("seed", 3407),
            "deterministic": False,
            "framework": "pytorch",
            "use_tt": True,
        }

    def dataset_fields(self, train_files, eval_files, max_length):
        data = self.spec["data"]
        model = self.spec.get("model", {})
        fields = {
            "dataset_id": "jsonl_causal",
            "dataset_format": data.get("format", "chat"),
            "train_files": train_files,
            "eval_files": eval_files,
            "messages_field": data.get("messages_field", "messages"),
            "prompt_field": data.get("prompt_field", "prompt"),
            "response_field": data.get("response_field", "response"),
            "text_field": data.get("text_field", "text"),
            "dataset_system_prompt": data.get("system_prompt"),
            "assistant_only_loss": data.get("assistant_only_loss", True),
            "text_transform": data.get("text_transform", "none"),
            "dataset_truncation": data.get("truncation", True),
            "dataset_packing": data.get("packing", False),
            "dataset_append_eos_token": data.get("append_eos_token", True),
            "dataset_sequence_bucketing": data.get(
                "sequence_bucketing", "auto"
            ),
            "dataset_sequence_buckets": data.get("sequence_buckets"),
            "dataset_tokenization_workers": data.get(
                "tokenization_workers", "auto"
            ),
            "dataset_dataloader_workers": data.get(
                "dataloader_workers", "auto"
            ),
            "dataset_prefetch_batches": data.get("prefetch_batches", 2),
            "fallback_role_separator": data.get("fallback_role_separator", "\n"),
            "fallback_assistant_prefix": data.get(
                "fallback_assistant_prefix",
                "Assistant:",
            ),
            "chat_template_enable_thinking": data.get(
                "chat_template_enable_thinking",
                False,
            ),
            "do_eval": data.get("do_eval", True),
            "model_revision": model.get("revision"),
            "trust_remote_code": model.get("trust_remote_code", False),
            "attn_implementation": model.get("attn_implementation"),
            "use_fast_tokenizer": model.get("use_fast_tokenizer", True),
            "max_length": max_length,
        }
        return fields

    def render(self):
        stages = self.spec["data"]["stages"]
        warmup_path = None
        if self.foundation_enabled:
            foundation = self.spec["foundation"]
            warmup = foundation["warmup"]
            train_files = [stage["train_file"] for stage in stages]
            eval_files = [stage["eval_file"] for stage in stages]

            warmup_config = self.dataset_fields(
                train_files,
                eval_files,
                warmup["max_length"],
            )
            warmup_config.update(
                {
                    "max_train_samples_per_file": warmup["train_samples_per_stage"],
                    "max_eval_samples_per_file": warmup["eval_samples_per_stage"],
                    "model_name": foundation["initialized_path"],
                    "token_contract_path": foundation["token_contract"],
                    "dtype": warmup.get("dtype", "torch.bfloat16"),
                    "training_model_type": (
                        "token_rows_output_tied"
                        if warmup.get("token_row_mode") == "output_tied"
                        else "token_rows"
                    ),
                    "learning_rate": warmup["learning_rate"],
                    "batch_size": warmup["batch_size"],
                    "gradient_accumulation_steps": warmup.get(
                        "gradient_accumulation_steps",
                        1,
                    ),
                    "gradient_checkpointing": warmup.get(
                        "gradient_checkpointing",
                        False,
                    ),
                    "num_epochs": warmup.get("num_epochs", 1),
                    "max_steps": warmup.get("max_steps"),
                    "optim": warmup.get("optim", "adamw_torch"),
                    "weight_decay": warmup.get("weight_decay", 0.0),
                    "adam_beta1": warmup.get("adam_beta1", 0.9),
                    "adam_beta2": warmup.get("adam_beta2", 0.999),
                    "adam_epsilon": warmup.get("adam_epsilon", 1e-8),
                    "max_grad_norm": warmup.get("max_grad_norm"),
                    "lr_scheduler_type": warmup.get(
                        "lr_scheduler_type",
                        "constant",
                    ),
                    "warmup_steps": warmup.get("warmup_steps", 0),
                    "warmup_ratio": warmup.get("warmup_ratio", 0.0),
                    "run_final_validation": warmup.get(
                        "run_final_validation",
                        False,
                    ),
                    "test_config": {"max_steps_per_epoch": warmup.get("max_steps")},
                }
            )
            warmup_config.update(
                self.base_training_fields(
                    f"{self.name}-foundation-warmup",
                    foundation["warmup_project_dir"],
                )
            )
            warmup_config["measure_e2e_time"] = True
            warmup_path = self.generated_dir / "foundation_warmup.yaml"
            warmup_path.write_text(
                yaml.safe_dump(warmup_config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )

        adapter = self.spec["adapters"]
        adapter_paths = {}
        for stage in stages:
            stage_config = self.dataset_fields(
                [stage["train_file"]],
                [stage["eval_file"]] if stage.get("eval_file") else [],
                stage["max_length"],
            )
            max_train_samples = stage.get(
                "train_samples",
                adapter.get("train_samples_per_stage"),
            )
            max_eval_samples = stage.get(
                "eval_samples",
                adapter.get("eval_samples_per_stage"),
            )
            max_steps = stage.get(
                "max_steps",
                adapter.get("max_steps_per_stage"),
            )
            stage_config.update(
                {
                    "model_name": self.adapter_base_model,
                    "model_metadata_name": self.spec.get("model", {}).get(
                        "metadata_name", self.adapter_base_model
                    ),
                    "dtype": stage.get(
                        "dtype",
                        adapter.get("dtype", "torch.bfloat16"),
                    ),
                    "training_model_type": "lora",
                    "learning_rate": stage.get(
                        "learning_rate",
                        adapter["learning_rate"],
                    ),
                    "batch_size": stage.get("batch_size", adapter["batch_size"]),
                    "gradient_accumulation_steps": stage.get(
                        "gradient_accumulation_steps",
                        adapter.get("gradient_accumulation_steps", 1),
                    ),
                    "gradient_checkpointing": stage.get(
                        "gradient_checkpointing",
                        adapter.get("gradient_checkpointing", False),
                    ),
                    "num_epochs": stage.get(
                        "num_epochs",
                        adapter.get("num_epochs", 1),
                    ),
                    "max_steps": max_steps,
                    "optim": stage.get(
                        "optim",
                        adapter.get("optim", "adamw_torch"),
                    ),
                    "optimizer_layout": stage.get(
                        "optimizer_layout",
                        adapter.get("optimizer_layout", "auto"),
                    ),
                    "optimizer_pack_bytes": stage.get(
                        "optimizer_pack_bytes",
                        adapter.get("optimizer_pack_bytes", 1024 * 1024),
                    ),
                    "optimizer_packing_strategy": stage.get(
                        "optimizer_packing_strategy",
                        adapter.get(
                            "optimizer_packing_strategy", "contiguous"
                        ),
                    ),
                    "execution_graph_layout": stage.get(
                        "execution_graph_layout",
                        adapter.get("execution_graph_layout", "auto"),
                    ),
                    "memory_mode": stage.get(
                        "memory_mode", adapter.get("memory_mode", "auto")
                    ),
                    "profile_device_memory": stage.get(
                        "profile_device_memory",
                        adapter.get("profile_device_memory", False),
                    ),
                    "weight_decay": stage.get(
                        "weight_decay",
                        adapter.get("weight_decay", 0.0),
                    ),
                    "adam_beta1": stage.get(
                        "adam_beta1",
                        adapter.get("adam_beta1", 0.9),
                    ),
                    "adam_beta2": stage.get(
                        "adam_beta2",
                        adapter.get("adam_beta2", 0.999),
                    ),
                    "adam_epsilon": stage.get(
                        "adam_epsilon",
                        adapter.get("adam_epsilon", 1e-8),
                    ),
                    "max_grad_norm": stage.get(
                        "max_grad_norm",
                        adapter.get("max_grad_norm"),
                    ),
                    "run_final_validation": stage.get(
                        "run_final_validation",
                        adapter.get("run_final_validation", False),
                    ),
                    "lr_scheduler_type": stage.get(
                        "lr_scheduler_type",
                        adapter.get("lr_scheduler_type", "constant"),
                    ),
                    "warmup_steps": stage.get(
                        "warmup_steps",
                        adapter.get("warmup_steps", 0),
                    ),
                    "warmup_ratio": stage.get(
                        "warmup_ratio",
                        adapter.get("warmup_ratio", 0.0),
                    ),
                    "adapter_type": stage.get(
                        "adapter_type",
                        adapter.get("adapter_type", "lora"),
                    ),
                    "lora_r": stage.get("rank", adapter["rank"]),
                    "lora_alpha": stage.get("alpha", adapter["alpha"]),
                    "lora_target_modules": (
                        [
                            stage.get("targets", adapter["targets"])
                        ]
                        if isinstance(
                            stage.get("targets", adapter["targets"]),
                            str,
                        )
                        else stage.get("targets", adapter["targets"])
                    ),
                    "lora_task_type": stage.get(
                        "task_type",
                        adapter.get("task_type", "CAUSAL_LM"),
                    ),
                    "lora_dropout": stage.get(
                        "dropout",
                        adapter.get("dropout", 0.0),
                    ),
                    "lora_bias": stage.get(
                        "bias",
                        adapter.get("bias", "none"),
                    ),
                    "lora_modules_to_save": stage.get(
                        "modules_to_save",
                        adapter.get("modules_to_save"),
                    ),
                    "lora_use_rslora": stage.get(
                        "use_rslora",
                        adapter.get("use_rslora", False),
                    ),
                    "lora_use_dora": stage.get(
                        "use_dora",
                        adapter.get("use_dora", False),
                    ),
                    "lora_plus_lr_ratio": stage.get(
                        "lora_plus_lr_ratio",
                        adapter.get("lora_plus_lr_ratio"),
                    ),
                    "lora_layers_to_transform": stage.get(
                        "layers_to_transform",
                        adapter.get("layers_to_transform"),
                    ),
                    "lora_layers_pattern": stage.get(
                        "layers_pattern",
                        adapter.get("layers_pattern"),
                    ),
                    "lora_init_weights": stage.get(
                        "init_weights",
                        adapter.get("init_weights", True),
                    ),
                    "ia3_feedforward_modules": stage.get(
                        "ia3_feedforward_modules",
                        adapter.get(
                            "ia3_feedforward_modules",
                            "ia3_feedforward",
                        ),
                    ),
                    "init_ia3_weights": stage.get(
                        "init_ia3_weights",
                        adapter.get("init_ia3_weights", True),
                    ),
                    "initial_adapter_path": self.spec.get("model", {}).get(
                        "initial_adapter_path"
                    ),
                    "initial_adapter_name": self.spec.get("model", {}).get(
                        "initial_adapter_name", "default"
                    ),
                    "lora_backend": stage.get(
                        "backend",
                        adapter.get("backend", "auto"),
                    ),
                    "lora_resident_full_mlp_backward": stage.get(
                        "resident_full_mlp_backward",
                        adapter.get("resident_full_mlp_backward", False),
                    ),
                    "lora_resident_output_attention_backward": stage.get(
                        "resident_output_attention_backward",
                        adapter.get(
                            "resident_output_attention_backward", False
                        ),
                    ),
                }
            )
            if max_train_samples is not None:
                stage_config["max_train_samples"] = max_train_samples
            if max_eval_samples is not None:
                stage_config["max_eval_samples"] = max_eval_samples
            weight_overrides = stage.get(
                "weight_dtype_overrides",
                adapter.get("weight_dtype_overrides"),
            )
            if weight_overrides:
                stage_config["weight_dtype_overrides"] = weight_overrides
            project_dir = self.stage_project_dir(stage)
            stage_config.update(
                self.base_training_fields(
                    f"{self.name}-{stage['name']}",
                    project_dir,
                )
            )
            for key in (
                "steps_freq",
                "val_steps_freq",
                "save_strategy",
                "save_steps",
                "save_optim",
                "keep_last_n",
                "keep_best_n",
            ):
                if key in adapter:
                    stage_config[key] = adapter[key]
            stage_config["measure_e2e_time"] = True
            path = self.generated_dir / f"adapter_{stage['name']}.yaml"
            path.write_text(
                yaml.safe_dump(stage_config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            adapter_paths[stage["name"]] = path

        resolved = {
            "manifest": str(self.manifest_path),
            "generated_at": now(),
            "warmup_config": str(warmup_path),
            "adapter_configs": {name: str(path) for name, path in adapter_paths.items()},
        }
        (self.generated_dir / "resolved.json").write_text(
            json.dumps(resolved, indent=2) + "\n",
            encoding="utf-8",
        )
        self.write_log(
            "Configuration render",
            "completed",
            {
                "warmup": warmup_path or "disabled",
                "adapter_count": len(adapter_paths),
            },
        )
        return warmup_path, adapter_paths

    def run_command(self, title, command, *, capture_output=False):
        self.write_log(title, "started", {"command": shlex.join(map(str, command))})
        environment = os.environ.copy()
        package_root = str(Path(__file__).resolve().parents[1])
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            package_root if not existing else package_root + os.pathsep + existing
        )
        try:
            result = subprocess.run(
                command,
                check=True,
                cwd=self.workspace,
                env=environment,
                text=capture_output,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
            )
        except subprocess.CalledProcessError as error:
            self.write_log(title, "failed", {"exit_code": error.returncode})
            if capture_output:
                if error.stdout:
                    print(error.stdout, file=sys.stderr, end="")
                if error.stderr:
                    print(error.stderr, file=sys.stderr, end="")
            raise
        self.write_log(title, "completed")
        return result.stdout if capture_output else None

    def run_packaged_python(self, title, script, arguments):
        self.run_command(
            title,
            [sys.executable, str(self.engine_dir / script), *map(str, arguments)],
        )

    def preflight(self, adapter_paths):
        preflight_script = self.engine_dir / "preflight_lora.py"
        preflight_base = self.adapter_base_model
        if self.foundation_enabled:
            foundation = self.spec["foundation"]
            warmed = self.container_to_host(foundation["warmed_path"])
            if not warmed.is_dir():
                preflight_base = foundation["initialized_path"]
        resolved_paths = {}
        for stage_name, config_path in adapter_paths.items():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            preflight_config = config_path
            if config["model_name"] != preflight_base:
                config["model_name"] = preflight_base
                preflight_config = self.generated_dir / f"preflight_config_{stage_name}.yaml"
                preflight_config.write_text(
                    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
            report = self.generated_dir / f"preflight_{stage_name}.json"
            resolved = self.generated_dir / f"adapter_{stage_name}_resolved.yaml"
            try:
                self.run_command(
                    f"Preflight: {stage_name}",
                    [
                        sys.executable,
                        preflight_script,
                        "--config",
                        preflight_config,
                        "--report",
                        report,
                        "--resolved-config",
                        resolved,
                    ],
                    capture_output=True,
                )
            except subprocess.CalledProcessError as error:
                failure = {
                    "status": "failed",
                    "stage": stage_name,
                    "config": str(preflight_config),
                    "exit_code": error.returncode,
                    "error": (error.stderr or error.stdout or "").strip(),
                    "failed_at": now(),
                }
                report.write_text(
                    json.dumps(failure, indent=2) + "\n",
                    encoding="utf-8",
                )
                raise
            if preflight_base != self.adapter_base_model:
                resolved_config = yaml.safe_load(
                    resolved.read_text(encoding="utf-8")
                )
                resolved_config["model_name"] = self.adapter_base_model
                resolved.write_text(
                    yaml.safe_dump(
                        resolved_config,
                        sort_keys=False,
                        allow_unicode=True,
                    ),
                    encoding="utf-8",
                )
            resolved_paths[stage_name] = resolved
        return resolved_paths

    def run_foundation(self, warmup_path, resume=False):
        if not self.foundation_enabled:
            self.write_log("Foundation token warmup", "skipped", {"reason": "disabled"})
            return
        foundation = self.spec["foundation"]
        initialized = self.container_to_host(foundation["initialized_path"])
        warmed = self.container_to_host(foundation["warmed_path"])
        if not initialized.is_dir():
            raise FileNotFoundError(f"Initialized foundation is missing: {initialized}")
        if self.valid_hf_model(warmed):
            if resume:
                self.write_log(
                    "Foundation token warmup",
                    "skipped",
                    {"reason": "verified warmed foundation already exists"},
                )
                self.mark_state(
                    "foundation",
                    "completed",
                    artifact=str(warmed),
                )
                return
            raise FileExistsError(
                f"Refusing to overwrite warmed foundation: {warmed}. "
                "Choose a new warmed_path or use --resume."
            )
        checkpoint = (
            self.container_to_host(foundation["warmup_project_dir"])
            / "checkpoints"
            / "final_model.pth"
        )
        if not (resume and checkpoint.is_file()):
            self.run_command(
                "Foundation token warmup",
                [
                    self.engine_dir / "run_tt_finetune_worker.sh",
                    warmup_path,
                ],
            )
        else:
            self.write_log(
                "Foundation token warmup",
                "skipped",
                {"reason": "verified warmup checkpoint already exists"},
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Warmup checkpoint was not created: {checkpoint}")
        if warmed.exists():
            raise FileExistsError(
                f"Incomplete warmed foundation path exists and will not be overwritten: {warmed}"
            )
        self.run_packaged_python(
            "Merge warmed foundation",
            "merge_token_warmup.py",
            [
                "--foundation",
                foundation["initialized_path"],
                "--checkpoint",
                f"{foundation['warmup_project_dir']}/checkpoints/final_model.pth",
                "--output",
                foundation["warmed_path"],
            ],
        )
        if not self.valid_hf_model(warmed):
            raise RuntimeError(f"Merged foundation failed integrity check: {warmed}")
        self.mark_state("foundation", "completed", artifact=str(warmed))

    def run_adapters(self, adapter_paths, resume=False):
        if self.foundation_enabled:
            warmed = self.container_to_host(self.adapter_base_model)
            if not self.valid_hf_model(warmed):
                raise FileNotFoundError(f"Warmed foundation is missing: {warmed}")
        adapter = self.spec["adapters"]
        tasks = []
        stages_by_name = {
            stage["name"]: stage for stage in self.spec["data"]["stages"]
        }
        for stage_name, config_path in adapter_paths.items():
            project_dir = self.stage_project_dir(stages_by_name[stage_name])
            project_host = self.container_to_host(project_dir)
            checkpoint_host = project_host / "checkpoints" / "final_model.pth"
            output_host = project_host / "peft_adapter"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if self.valid_peft_adapter(output_host):
                if not resume:
                    raise FileExistsError(
                        f"Adapter already exists: {output_host}. Use --resume to verify and skip it."
                    )
                mismatches = self.peft_config_mismatches(
                    output_host,
                    config,
                    self.adapter_base_model,
                )
                if mismatches:
                    raise ValueError(
                        f"Existing adapter does not match resolved training "
                        f"topology for {stage_name}: {mismatches}"
                    )
                self.write_log(
                    f"Adapter: {stage_name}",
                    "skipped",
                    {"reason": "verified PEFT adapter already exists"},
                )
                self.mark_state(
                    f"adapter:{stage_name}",
                    "completed",
                    artifact=str(output_host),
                )
                continue
            if project_host.exists() and not resume:
                raise FileExistsError(
                    f"Adapter output already exists: {project_host}. "
                    "Choose a new output_root or use --resume."
                )
            tasks.append(
                {
                    "name": stage_name,
                    "config": config_path,
                    "project_dir": project_dir,
                    "checkpoint_host": checkpoint_host,
                    "output_host": output_host,
                    "needs_training": not (resume and checkpoint_host.is_file()),
                }
            )

        training_tasks = [task for task in tasks if task["needs_training"]]
        group_enabled = self.spec.get("execution", {}).get(
            "group_compatible_adapters",
            True,
        )
        if group_enabled:
            groups = {}
            for task in training_tasks:
                config = yaml.safe_load(task["config"].read_text(encoding="utf-8"))
                key = (
                    config["model_name"],
                    config["dtype"],
                    config["max_length"],
                    config["batch_size"],
                    config["lora_r"],
                    config["lora_alpha"],
                    tuple(config["lora_target_modules"]),
                    config.get("lora_dropout", 0.0),
                    config.get("lora_bias", "none"),
                    tuple(config.get("lora_modules_to_save") or []),
                    config.get("lora_use_rslora", False),
                    config.get("lora_use_dora", False),
                    config.get("gradient_accumulation_steps", 1),
                    config.get("optim", "adamw_torch"),
                    config.get("optimizer_layout", "auto"),
                    config.get("optimizer_pack_bytes", 1024 * 1024),
                    config.get(
                        "optimizer_packing_strategy", "contiguous"
                    ),
                    config.get("execution_graph_layout", "auto"),
                    config.get("weight_decay", 0.0),
                    config.get("max_grad_norm"),
                    config.get("weight_dtype_overrides"),
                    config.get("memory_mode", "auto"),
                    config.get("profile_device_memory", False),
                )
                groups.setdefault(key, []).append(task)
            for index, group in enumerate(groups.values(), start=1):
                group_configs = [
                    yaml.safe_load(
                        task["config"].read_text(encoding="utf-8")
                    )
                    for task in group
                ]
                layouts = {
                    config.get("optimizer_layout", "auto")
                    for config in group_configs
                }
                invalid_layouts = layouts - {
                    "auto",
                    "tile_packed",
                    "per_parameter",
                }
                if invalid_layouts:
                    raise ValueError(
                        f"Unsupported optimizer layouts: {invalid_layouts}"
                    )
                packing_strategies = {
                    config.get(
                        "optimizer_packing_strategy", "contiguous"
                    )
                    for config in group_configs
                }
                invalid_packing_strategies = packing_strategies - {
                    "contiguous",
                    "shape_grouped",
                }
                if invalid_packing_strategies:
                    raise ValueError(
                        "Unsupported optimizer packing strategies: "
                        f"{invalid_packing_strategies}"
                    )
                graph_layouts = {
                    config.get("execution_graph_layout", "auto")
                    for config in group_configs
                }
                invalid_graph_layouts = graph_layouts - {
                    "auto",
                    "phase_split",
                    "monolithic",
                }
                if invalid_graph_layouts:
                    raise ValueError(
                        "Unsupported execution graph layouts: "
                        f"{invalid_graph_layouts}"
                    )
                if (
                    len(group) > 1
                    and layouts == {"per_parameter"}
                    and graph_layouts == {"monolithic"}
                ):
                    self.run_command(
                        f"Grouped adapter training: shape group {index}",
                        [
                            self.engine_dir / "run_lora_group.sh",
                            *[task["config"] for task in group],
                        ],
                    )
                else:
                    for task, config in zip(group, group_configs):
                        command = [
                            self.engine_dir / "run_tt_finetune_worker.sh",
                            task["config"],
                        ]
                        if config.get(
                            "optimizer_layout", "auto"
                        ) in {"auto", "tile_packed"}:
                            command.append("--optimizer-tile-packed")
                            command.extend(
                                [
                                    "--optimizer-pack-bytes",
                                    str(
                                        config.get(
                                            "optimizer_pack_bytes",
                                            1024 * 1024,
                                        )
                                    ),
                                    "--optimizer-packing-strategy",
                                    config.get(
                                        "optimizer_packing_strategy",
                                        "contiguous",
                                    ),
                                ]
                            )
                        graph_layout = config.get(
                            "execution_graph_layout", "auto"
                        )
                        if graph_layout in {"auto", "phase_split"}:
                            command.append("--split-clip-optimizer")
                            command.append("--split-after-loss")
                            if config.get(
                                "optimizer_layout", "auto"
                            ) in {"auto", "tile_packed"}:
                                command.append("--split-optimizer-phases")
                            command.append(
                                "--auto-split-after-forward"
                                if graph_layout == "auto"
                                else "--split-after-forward"
                            )
                        self.run_command(
                            f"Adapter training: {task['name']}",
                            command,
                        )
        else:
            for task in training_tasks:
                config = yaml.safe_load(
                    task["config"].read_text(encoding="utf-8")
                )
                layout = config.get("optimizer_layout", "auto")
                if layout not in {
                    "auto",
                    "tile_packed",
                    "per_parameter",
                }:
                    raise ValueError(
                        f"Unsupported optimizer layout: {layout}"
                    )
                graph_layout = config.get(
                    "execution_graph_layout", "auto"
                )
                if graph_layout not in {
                    "auto",
                    "phase_split",
                    "monolithic",
                }:
                    raise ValueError(
                        "Unsupported execution graph layout: "
                        f"{graph_layout}"
                    )
                packing_strategy = config.get(
                    "optimizer_packing_strategy", "contiguous"
                )
                if packing_strategy not in {
                    "contiguous",
                    "shape_grouped",
                }:
                    raise ValueError(
                        "Unsupported optimizer packing strategy: "
                        f"{packing_strategy}"
                    )
                command = [
                    self.engine_dir / "run_tt_finetune_worker.sh",
                    task["config"],
                ]
                if layout in {"auto", "tile_packed"}:
                    command.append("--optimizer-tile-packed")
                    command.extend(
                        [
                            "--optimizer-pack-bytes",
                            str(
                                config.get(
                                    "optimizer_pack_bytes", 1024 * 1024
                                )
                            ),
                            "--optimizer-packing-strategy",
                            packing_strategy,
                        ]
                    )
                if graph_layout in {"auto", "phase_split"}:
                    command.append("--split-clip-optimizer")
                    command.append("--split-after-loss")
                    if layout in {"auto", "tile_packed"}:
                        command.append("--split-optimizer-phases")
                    command.append(
                        "--auto-split-after-forward"
                        if graph_layout == "auto"
                        else "--split-after-forward"
                    )
                self.run_command(
                    f"Adapter training: {task['name']}",
                    command,
                )

        for task in tasks:
            stage_name = task["name"]
            config_path = task["config"]
            project_dir = task["project_dir"]
            checkpoint_host = task["checkpoint_host"]
            output_host = task["output_host"]
            if not task["needs_training"]:
                self.write_log(
                    f"Adapter training: {stage_name}",
                    "skipped",
                    {"reason": "verified final checkpoint already exists"},
                )
            if not checkpoint_host.is_file():
                raise FileNotFoundError(
                    f"Adapter checkpoint was not created: {checkpoint_host}"
                )
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            checkpoint = f"{project_dir}/checkpoints/final_model.pth"
            output = f"{project_dir}/peft_adapter"
            export_arguments = [
                "--base-model",
                self.adapter_base_model,
                "--metadata-base-model",
                config.get("model_metadata_name")
                or self.adapter_base_model,
                "--checkpoint",
                checkpoint,
                "--output",
                output,
                "--targets",
                *config["lora_target_modules"],
                "--task-type",
                config.get("lora_task_type", "CAUSAL_LM"),
                "--adapter-type",
                config.get("adapter_type", "lora"),
            ]
            if config.get("adapter_type", "lora") == "ia3":
                feedforward = config.get("ia3_feedforward_modules") or []
                if isinstance(feedforward, str):
                    feedforward = [feedforward]
                export_arguments.extend(
                    [
                        "--feedforward-modules", *feedforward,
                        "--init-ia3-weights",
                        str(config.get("init_ia3_weights", True)).lower(),
                    ]
                )
            else:
                export_arguments.extend(
                    [
                        "--rank", config["lora_r"],
                        "--alpha", config["lora_alpha"],
                        "--dropout", config.get("lora_dropout", 0.0),
                        "--bias", config.get("lora_bias", "none"),
                        "--init-weights",
                        str(config.get("lora_init_weights", True)).lower(),
                    ]
                )
            if config.get("model_revision"):
                export_arguments.extend(
                    ["--model-revision", config["model_revision"]]
                )
            if config.get("trust_remote_code", False):
                export_arguments.append("--trust-remote-code")
            if config.get("attn_implementation"):
                export_arguments.extend(
                    ["--attn-implementation", config["attn_implementation"]]
                )
            if not config.get("use_fast_tokenizer", True):
                export_arguments.append("--slow-tokenizer")
            modules_to_save = config.get("lora_modules_to_save")
            if modules_to_save:
                export_arguments.extend(["--modules-to-save", *modules_to_save])
            if config.get("lora_use_rslora", False):
                export_arguments.append("--use-rslora")
            if config.get("lora_use_dora", False):
                export_arguments.append("--use-dora")
            if config.get("lora_plus_lr_ratio") is not None:
                export_arguments.extend(
                    [
                        "--lora-plus-lr-ratio",
                        str(config["lora_plus_lr_ratio"]),
                    ]
                )
            if config.get("lora_layers_to_transform") is not None:
                export_arguments.extend(
                    [
                        "--layers-to-transform",
                        *map(str, config["lora_layers_to_transform"]),
                    ]
                )
            if config.get("lora_layers_pattern"):
                export_arguments.extend(
                    ["--layers-pattern", config["lora_layers_pattern"]]
                )
            self.run_packaged_python(
                f"PEFT export: {stage_name}",
                "export_peft_adapter.py",
                export_arguments,
            )
            if not self.valid_peft_adapter(output_host):
                raise RuntimeError(
                    f"PEFT export failed integrity check: {output_host}"
                )
            self.mark_state(
                f"adapter:{stage_name}",
                "completed",
                artifact=str(output_host),
                base_model=self.adapter_base_model,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("all", "foundation", "adapters"),
        default="all",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and skip completed artifacts; continue from final checkpoints.",
    )
    args = parser.parse_args()

    pipeline = Pipeline(args.config)
    warmup_path, adapter_paths = pipeline.render()
    adapter_paths = pipeline.preflight(adapter_paths)
    if args.command == "plan":
        print(f"Plan ready: {pipeline.generated_dir}")
        return
    if args.phase in ("all", "foundation"):
        pipeline.run_foundation(warmup_path, resume=args.resume)
    if args.phase in ("all", "adapters"):
        pipeline.run_adapters(adapter_paths, resume=args.resume)
        pipeline.verify_artifacts()
    pipeline.write_log("Pipeline command", "completed", {"phase": args.phase})
    print(f"Pipeline completed: {pipeline.run_log}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"pipeline_engine.py: {error}", file=sys.stderr)
        raise
