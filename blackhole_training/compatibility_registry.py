"""Durable compatibility records for model/profile hardware gates."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


class CompatibilityRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self):
        if not self.path.exists():
            return {"schema_version": 1, "models": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, registry):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    @staticmethod
    def model_key(report: Mapping[str, Any]):
        revision = report.get("model_revision") or "default"
        return f"{report['model']}@{revision}"

    @staticmethod
    def profile(report: Mapping[str, Any], config: Mapping[str, Any]):
        values = {
            "engine_fingerprint": config.get("engine_fingerprint"),
            "dtype": config.get("dtype"),
            "max_length": report.get("max_length"),
            "batch_size": report.get("batch_size"),
            "gradient_accumulation_steps": config.get(
                "gradient_accumulation_steps",
                1,
            ),
            "gradient_checkpointing": config.get(
                "gradient_checkpointing",
                False,
            ),
            "rank": report.get("lora_rank"),
            "alpha": report.get("lora_alpha"),
            "targets": report.get("lora_targets"),
            "dropout": config.get("lora_dropout", 0.0),
            "bias": config.get("lora_bias", "none"),
            "modules_to_save": config.get("lora_modules_to_save"),
            "use_rslora": config.get("lora_use_rslora", False),
            "use_dora": config.get("lora_use_dora", False),
            "init_weights": config.get("lora_init_weights", True),
            "optimizer": config.get("optim", "adamw_torch"),
            "optimizer_layout": config.get("optimizer_layout", "auto"),
            "optimizer_pack_bytes": config.get(
                "optimizer_pack_bytes", 1024 * 1024
            ),
            "optimizer_packing_strategy": config.get(
                "optimizer_packing_strategy", "contiguous"
            ),
            "execution_graph_layout": config.get(
                "execution_graph_layout",
                "auto",
            ),
            "weight_decay": config.get("weight_decay", 0.0),
            "max_grad_norm": config.get("max_grad_norm"),
            "lr_scheduler_type": config.get("lr_scheduler_type", "constant"),
            "warmup_steps": config.get("warmup_steps", 0),
            "warmup_ratio": config.get("warmup_ratio", 0.0),
            "attn_implementation": config.get("attn_implementation"),
            "weight_dtype_overrides": config.get("weight_dtype_overrides"),
        }
        fingerprint = hashlib.sha256(
            json.dumps(values, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return fingerprint, values

    def record_preflight(self, report, config):
        registry = self.load()
        key = self.model_key(report)
        profile_id, profile = self.profile(report, config)
        model = registry["models"].setdefault(
            key,
            {
                "model": report["model"],
                "revision": report.get("model_revision"),
                "architecture": report.get("architecture"),
                "model_type": report.get("model_type"),
                "profiles": {},
            },
        )
        model["last_updated"] = now()
        record = model["profiles"].setdefault(profile_id, profile)
        record.update(
            {
                "preflight_status": report.get("status"),
                "preflight_at": now(),
                "matched_linear_modules": report.get(
                    "matched_linear_modules"
                ),
                "matched_module_types": report.get("matched_module_types"),
                "estimated_lora_parameters": report.get(
                    "estimated_lora_parameters"
                ),
                "exact_trainable_parameters": report.get(
                    "exact_trainable_parameters"
                ),
                "dataset": {
                    "format": report.get("dataset_format"),
                    "train": report.get("train"),
                    "validation": report.get("validation"),
                },
            }
        )
        self.save(registry)
        return profile_id

    def record_hardware(
        self,
        report,
        config,
        *,
        status,
        metrics=None,
        adapter_path=None,
        error=None,
    ):
        registry = self.load()
        key = self.model_key(report)
        profile_id, profile = self.profile(report, config)
        model = registry["models"].setdefault(
            key,
            {
                "model": report["model"],
                "revision": report.get("model_revision"),
                "architecture": report.get("architecture"),
                "model_type": report.get("model_type"),
                "profiles": {},
            },
        )
        record = model["profiles"].setdefault(profile_id, profile)
        record["hardware_status"] = status
        record["hardware_checked_at"] = now()
        if adapter_path:
            record["adapter_path"] = str(adapter_path)
        if metrics:
            record["metrics"] = {
                key: metrics.get(key)
                for key in (
                    "global_steps",
                    "initial_validation_loss",
                    "final_validation_loss",
                    "steady_step_seconds",
                    "steady_seconds_per_example",
                    "trainable_parameters",
                )
            }
        if error:
            record["last_error"] = str(error)
        else:
            record.pop("last_error", None)
        model["last_updated"] = now()
        self.save(registry)
        return profile_id
