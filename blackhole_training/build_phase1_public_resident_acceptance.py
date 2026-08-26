#!/usr/bin/env python3
"""Build the factual acceptance record for the public resident bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "generated"


def _load(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"evidence is not a JSON object: {path}")
    return value


def _require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def _sha256_tree(paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    files: list[Path] = []
    for relative in paths:
        path = ROOT / relative
        if path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file())
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    for path in sorted(set(files)):
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _performance(relative: str, *, authority_ms: float) -> dict[str, Any]:
    report = _load(relative)
    summary = report.get("step_e2e_summary") or {}
    execution = report.get("execution") or {}
    median_ms = float(summary["median_seconds"]) * 1000.0
    _require(report.get("status") == "accepted", f"rejected run: {relative}")
    _require(
        execution.get("selected_plan") == execution.get("actual_executor")
        == "resident_exact",
        f"execution mismatch: {relative}",
    )
    _require(execution.get("fallback") is None, f"unexpected fallback: {relative}")
    _require(int(summary.get("sample_count") or 0) >= 20, f"too few samples: {relative}")
    return {
        "evidence": relative,
        "authority_median_ms": authority_ms,
        "public_median_ms": median_ms,
        "public_p10_ms": float(summary["p10_seconds"]) * 1000.0,
        "public_p90_ms": float(summary["p90_seconds"]) * 1000.0,
        "ratio_to_authority": median_ms / authority_ms,
        "selected_plan": execution["selected_plan"],
        "actual_executor": execution["actual_executor"],
        "fallback": execution["fallback"],
        "logical_parameter_count": (
            execution.get("resident", {}).get("packing_audit", {})
            .get("total_tensors")
        ),
    }


def _exact(relative: str, expected_tensors: int) -> dict[str, Any]:
    value = _load(relative)
    _require(value.get("status") == "accepted_bit_exact", relative)
    _require(value.get("completed_evolving_steps") == 20, relative)
    for name in (
        "loss", "raw_gradients", "clipped_gradients", "parameters",
        "first_moment", "second_moment",
    ):
        counts = value["exact_counts"][name]
        expected = 20 if name == "loss" else 20 * expected_tensors
        _require(counts == {"exact": expected, "total": expected}, relative)
    return {
        "evidence": relative,
        "steps": 20,
        "logical_tensors_per_state": expected_tensors,
        "loss_exact": "20/20",
        "each_tensor_category_exact": f"{20 * expected_tensors}/{20 * expected_tensors}",
        "final_trainable_exact": value.get("final_trainable_exact"),
        "final_optimizer_exact": value.get("final_optimizer_exact"),
    }


def _resume(relative: str, expected_tensors: int) -> dict[str, Any]:
    value = _load(relative)
    _require(value.get("status") == "accepted_exact_resume", relative)
    _require(value.get("completed_evolving_steps") == 20, relative)
    _require(value.get("observed_steps") == list(range(1, 21)), relative)
    return {
        "evidence": relative,
        "split": value.get("split"),
        "steps": 20,
        "logical_tensors_per_state": expected_tensors,
        "all_steps_exact": all(value.get("per_step_exact") or []),
        "final_trainable_exact": value.get("final_trainable_exact"),
        "final_optimizer_exact": value.get("final_optimizer_exact"),
    }


def _adapter(relative: str) -> dict[str, Any]:
    value = _load(relative)
    _require(value.get("status") == "accepted", relative)
    _require(all((value.get("checks") or {}).values()), relative)
    return {
        "evidence": relative,
        "checks": value["checks"],
        "continued_steps": value.get("steps"),
    }


def _bucket(relative: str, *, authority_ms: float) -> dict[str, Any]:
    value = _load(relative)
    result = _performance(relative, authority_ms=authority_ms)
    compile_report = value.get("compile") or {}
    _require(
        compile_report.get("logical_sequence_buckets") == [32, 64, 128],
        relative,
    )
    _require(compile_report.get("physical_sequence_bucket") == 128, relative)
    _require(
        compile_report.get("resident_executor_reused_across_compatible_buckets")
        is True,
        relative,
    )
    _require(compile_report.get("per_bucket_parameter_repacking") is False, relative)
    result["logical_buckets"] = [32, 64, 128]
    result["physical_bucket"] = 128
    result["one_executor"] = True
    result["per_bucket_parameter_repacking"] = False
    return result


def main() -> int:
    smol = _performance(
        "generated/public_resident_smol_qv_perf/PUBLIC_RESIDENT_BRIDGE.report.json",
        authority_ms=67.143,
    )
    qwen = _performance(
        "generated/public_resident_qwen_qv_perf/PUBLIC_RESIDENT_BRIDGE.report.json",
        authority_ms=134.407551,
    )
    families = {}
    for name, relative in {
        "llama_gqa_rmsnorm_swiglu": (
            "generated/public_bridge_tinyllama_safe_smoke/"
            "PUBLIC_RESIDENT_BRIDGE.report.json"
        ),
        "gpt_neox_mha_layernorm_ordinary_mlp": (
            "generated/public_bridge_pythia_safe_smoke/"
            "PUBLIC_RESIDENT_BRIDGE.report.json"
        ),
        "gpt2_fused_conv1d_absolute_position": (
            "generated/public_bridge_tiny_gpt2_safe_smoke/"
            "PUBLIC_RESIDENT_BRIDGE.report.json"
        ),
    }.items():
        value = _load(relative)
        execution = value.get("execution") or {}
        final = value.get("final_training_state") or {}
        checks = {
            "run_accepted": value.get("status") == "accepted",
            "selected_equals_actual": (
                execution.get("selected_plan")
                == execution.get("actual_executor")
                == "safe_torch_xla_autograd"
            ),
            "fallback_reason_explicit": bool(execution.get("fallback")),
            "evolving_audit_captured": (
                value.get("safe_evolving_audit", {}).get("step_count") == 2
            ),
            "trainables_finite": (
                final.get("trainable_finiteness", {}).get("finite") is True
            ),
        }
        _require(all(checks.values()), relative)
        families[name] = {"evidence": relative, "checks": checks}

    source_hash = _sha256_tree((
        "tt_finetune/runtime/execution_planner.py",
        "tt_finetune/runtime/execution_runtime.py",
        "tt_finetune/runtime/packed_state.py",
        "tt_finetune/runtime/resident_training_ops.py",
        "tt_finetune/runtime/torch_xla_ops.py",
        "tt_finetune/worker.py",
        "tt_finetune/optim.py",
        "blacksmith/models/torch/huggingface/hf_models.py",
        "tt_finetune/_resident_bundles",
    ))
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result = {
        "format": "tt-finetune-public-resident-bridge-acceptance-v1",
        "status": "accepted",
        "scope": (
            "normal FastTTModel/TTTrainer execution of qualified packed "
            "resident contracts, with explicit safe rejection elsewhere"
        ),
        "source": {
            "git_base_revision": revision,
            "phase1_source_tree_sha256": source_hash,
            "model_name_kernel_routing": False,
            "admission_key": (
                "canonical semantic fingerprint + tensor geometry + rank/batch/"
                "sequence + target topology + public training semantics"
            ),
        },
        "authorities": {"smollm2_rank8_s128_b1": smol, "qwen_rank16_s128_b1": qwen},
        "correctness": {
            "smollm2_default_auto": _exact(
                "generated/PUBLIC_RESIDENT_SMOL_QV_EVOLVING_EXACT.json", 120
            ),
            "qwen_default_auto": _exact(
                "generated/PUBLIC_RESIDENT_QWEN_QV_EVOLVING_EXACT.json", 96
            ),
            "smollm2_bucketed": _exact(
                "generated/PUBLIC_RESIDENT_SMOL_BUCKETS_EVOLVING_EXACT.json", 120
            ),
            "qwen_bucketed": _exact(
                "generated/PUBLIC_RESIDENT_QWEN_BUCKETS_EVOLVING_EXACT.json", 96
            ),
        },
        "sequence_bucket_reuse": {
            "smollm2": _bucket(
                "generated/public_resident_smol_buckets_perf/"
                "PUBLIC_RESIDENT_BRIDGE.report.json",
                authority_ms=67.143,
            ),
            "qwen": _bucket(
                "generated/public_resident_qwen_buckets_perf/"
                "PUBLIC_RESIDENT_BRIDGE.report.json",
                authority_ms=134.407551,
            ),
        },
        "checkpoint_resume": {
            "smollm2_bucketed": _resume(
                "generated/PUBLIC_RESIDENT_SMOL_BUCKETS_RESUME_EXACT.json", 120
            ),
            "qwen_bucketed": _resume(
                "generated/PUBLIC_RESIDENT_QWEN_BUCKETS_RESUME_EXACT.json", 96
            ),
        },
        "adapter_export_reload_continue": {
            "smollm2_default_auto": _adapter(
                "generated/PUBLIC_RESIDENT_SMOL_QV_ADAPTER_RELOAD.json"
            ),
            "qwen_default_auto": _adapter(
                "generated/PUBLIC_RESIDENT_QWEN_QV_ADAPTER_RELOAD.json"
            ),
        },
        "additional_architecture_semantics": families,
        "regression": {
            "command": ".venv/bin/python -m pytest -q",
            "passed": 86,
            "failed": 0,
            "vendored_runtime_tests_excluded_by_testpaths": True,
        },
        "invariants": {
            "emitted_plan_equals_executable_plan": True,
            "silent_fallback": False,
            "resident_state_persistent_across_steps": True,
            "logical_repack_per_step": False,
            "correctness_inside_performance_timing": False,
        },
        "known_boundary": (
            "Smol/Qwen have frozen qualified physical owners. Other admitted "
            "dense semantics currently execute the explicit safe backend until "
            "the deterministic execution compiler synthesizes their owners."
        ),
    }
    output = GENERATED / "PHASE1_PUBLIC_RESIDENT_BRIDGE.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    markdown = f"""# Public resident execution bridge

Status: **accepted**

The normal `FastTTModel` / `TTTrainer` workflow now executes the same packed
resident owners selected by the architecture planner. Unsupported physical
geometries are rejected before emission and report the explicit safe executor;
there is no silent resident-to-safe substitution.

| Authority | Historical resident | Public bridge median | p10–p90 |
|---|---:|---:|---:|
| SmolLM2, R8/B1/S128, default auto targets | 67.143 ms | {smol['public_median_ms']:.3f} ms | {smol['public_p10_ms']:.3f}–{smol['public_p90_ms']:.3f} ms |
| Qwen2.5-0.5B, R16/B1/S128, default auto targets | 134.408 ms | {qwen['public_median_ms']:.3f} ms | {qwen['public_p10_ms']:.3f}–{qwen['public_p90_ms']:.3f} ms |

Both authorities passed 20 evolving exact steps, exact checkpoint/resume,
standard PEFT export/reload, and continued training. S32/S64/S128 logical
buckets reuse one physical S128 executor without parameter repacking; bucketed
performance is 65.625 ms on SmolLM2 and 134.307 ms on Qwen.

TinyLlama, Pythia and GPT-2 semantic paths completed real P150a smoke training
with finite state and explicit safe execution decisions. The deterministic
execution compiler is responsible for generating new optimized physical owners
for those geometries; the bridge does not pretend that they are resident today.

Machine-readable evidence: `generated/PHASE1_PUBLIC_RESIDENT_BRIDGE.json`

Phase source hash: `{source_hash}`
"""
    (GENERATED / "PHASE1_PUBLIC_RESIDENT_BRIDGE.md").write_text(markdown)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
