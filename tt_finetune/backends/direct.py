"""Direct Blackhole backend driven by the tt-finetune runtime worker."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import statistics
import subprocess
import sys
import time
from typing import Any
import uuid

import yaml

from blackhole_training import TTTrainResult

from .blackhole import BlackholeXlaBackend
from ..config import IA3Config, LoRAConfig
from ..errors import TTDependencyError, classify_runtime_error


_UNIX_SOCKET_PATH_LIMIT = 107


def _short_runtime_socket_path(workspace: Path) -> Path:
    """Return a private, workspace-stable AF_UNIX path below Linux's limit.

    The public workspace is intentionally allowed to live below an arbitrary
    user-selected cache/home directory.  AF_UNIX does not share that freedom:
    Linux limits ``sun_path`` to 108 bytes including its terminator.  Keep
    durable service metadata in the workspace, but address the local service
    through a short per-user runtime directory keyed by the resolved workspace.
    """

    uid = os.getuid()
    identity = hashlib.sha256(
        os.fsencode(str(workspace.expanduser().resolve()))
    ).hexdigest()[:20]
    candidates = []
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime:
        candidates.append(Path(xdg_runtime) / "tt-finetune")
    candidates.append(Path("/tmp") / f"tt-finetune-{uid}")
    for root in candidates:
        socket_path = root / f"rt-{identity}.sock"
        if len(os.fsencode(str(socket_path))) > _UNIX_SOCKET_PATH_LIMIT:
            continue
        try:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            metadata = root.lstat()
        except OSError:
            continue
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != uid:
            continue
        os.chmod(root, 0o700)
        return socket_path
    raise RuntimeError(
        "no safe AF_UNIX path is short enough for the persistent TT runtime"
    )


def _percentile_local(values: tuple[float, ...], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _native_runtime_failure_diagnostic(
    log_path: Path,
    *,
    start_offset: int,
) -> dict[str, Any] | None:
    """Extract a structured failure emitted outside PJRT's StatusOr.

    TT-Metal writes allocator failures to the persistent service's stderr,
    while current PJRT sometimes returns only ``INTERNAL: Error code: 13``.
    Reading only bytes appended by this job prevents a historical failure in
    the shared log from being attributed to a later request.
    """

    try:
        with log_path.open("rb") as source:
            source.seek(max(0, int(start_offset)))
            text = source.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    from ..memory import parse_tt_oom_boundary

    oom = parse_tt_oom_boundary(text)
    if oom is None:
        return None
    return {
        "kind": "device_dram_out_of_memory",
        "message": (
            "TT-Metal device DRAM is out of memory at an allocation "
            "boundary; reduce model size, sequence length, batch size, or "
            "adapter rank. Fragmentation can make the largest free block "
            "smaller than aggregate free memory."
        ),
        "allocator": oom,
    }


class DirectBlackholeBackend(BlackholeXlaBackend):
    """Execute training with ``tt_finetune.runtime.TrainingLoop`` on TT-XLA."""

    name = "blackhole_direct"

    def _memory_support(self):
        from ..capabilities import query_support

        adapter = self.model.adapter_config
        assert adapter is not None
        if isinstance(adapter, IA3Config):
            return query_support(
                model=self.model,
                rank=1,
                batch_size=self.args.per_device_train_batch_size,
                sequence_length=(
                    self.args.max_seq_length or self.model.max_seq_length
                ),
                target_modules=adapter.target_modules,
                feedforward_modules=adapter.feedforward_modules,
                adapter_type="ia3",
                layers_to_transform=adapter.layers_to_transform,
                precision=self.args.dtype,
                revision=self.model.revision,
                trust_remote_code=self.model.trust_remote_code,
                memory_mode=self.args.memory_mode,
            )
        lora = adapter
        assert isinstance(lora, LoRAConfig)
        return query_support(
            model=self.model,
            rank=lora.r,
            batch_size=self.args.per_device_train_batch_size,
            sequence_length=(
                self.args.max_seq_length or self.model.max_seq_length
            ),
            target_modules=lora.target_modules,
            use_rslora=lora.use_rslora,
            use_dora=lora.use_dora,
            lora_dropout=lora.lora_dropout,
            lora_bias=lora.bias,
            lora_plus_lr_ratio=lora.lora_plus_lr_ratio,
            layers_to_transform=lora.layers_to_transform,
            precision=self.args.dtype,
            revision=self.model.revision,
            trust_remote_code=self.model.trust_remote_code,
            memory_mode=self.args.memory_mode,
        )

    def _persistent_cache_id(self, report: dict[str, Any]) -> str:
        plan = report.get("execution_plan", {})
        selected_worker_arguments = self._selected_worker_arguments()
        data_config = getattr(self, "data_config", None)
        sequence_buckets = getattr(data_config, "sequence_buckets", None)
        contract = {
            "compile_cache_id": self._compile_cache_id(),
            "architecture": report.get("architecture_descriptor"),
            "canonical_targets": report.get("canonical_lora_target_paths"),
            "execution_plan": plan,
            "graph_arguments": self._graph_worker_arguments(report),
            "selected_worker_arguments": selected_worker_arguments,
            "data_compile_shapes": {
                "sequence_bucketing": getattr(
                    data_config, "sequence_bucketing", "auto"
                ),
                "sequence_buckets": (
                    list(sequence_buckets)
                    if sequence_buckets is not None else None
                ),
            },
            "optimizer": {
                "layout": self.args.optimizer_layout,
                "pack_bytes": self.args.optimizer_pack_bytes,
                "packing_strategy": self.args.optimizer_packing_strategy,
                # TT-XLA may value-specialize device scalar inputs.  These
                # values therefore belong to the executable compatibility
                # contract even though they do not change tensor shapes.
                "learning_rate": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
                "adam_beta1": self.args.adam_beta1,
                "adam_beta2": self.args.adam_beta2,
                "adam_epsilon": self.args.adam_epsilon,
                "max_grad_norm": self.args.max_grad_norm,
                "gradient_accumulation_steps": (
                    self.args.gradient_accumulation_steps
                ),
                "scheduler": self.args.lr_scheduler_type,
                "warmup_steps": self.args.warmup_steps,
                "warmup_ratio": self.args.warmup_ratio,
                "planned_max_steps": self.args.max_steps,
                "planned_epochs": self.args.num_train_epochs,
                "experimental_fx_fusion": self.args.experimental_fx_fusion,
                "memory_mode": self.args.memory_mode,
            },
        }
        return hashlib.sha256(
            json.dumps(contract, sort_keys=True, default=str).encode()
        ).hexdigest()[:32]

    def _persistent_root(self) -> Path:
        root = self._engine.workspace / "generated" / "tt_runtime_service"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _persistent_socket_path(self) -> Path:
        return _short_runtime_socket_path(self._engine.workspace)

    def _persistent_container_name(self) -> str:
        identity = hashlib.sha256(
            str(self._engine.workspace).encode("utf-8")
        ).hexdigest()[:12]
        return f"tt-finetune-{os.getuid()}-{identity}"

    def _persistent_container_path(self, path: Path) -> str:
        return self._engine._to_container_path(path)

    def _service_status(
        self,
        socket_path: Path,
        *,
        info_path: Path | None = None,
    ) -> dict[str, Any] | None:
        if not socket_path.exists():
            return None
        try:
            from ..service_client import request

            return request(
                socket_path,
                {"action": "status"},
                timeout_seconds=1.0,
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            # The runtime server intentionally serializes accelerator jobs.
            # A synchronous train handler cannot answer status until the job
            # ends, so use its atomically published state instead of blocking
            # a second public API call indefinitely.
            info_path = info_path or socket_path.parent / "runtime_info.json"
            try:
                status = json.loads(info_path.read_text(encoding="utf-8"))
                pid = int(status.get("pid") or 0)
                if pid <= 1:
                    return None
                os.kill(pid, 0)
                return status
            except (
                OSError,
                RuntimeError,
                ValueError,
                json.JSONDecodeError,
            ):
                return None

    def _stop_persistent_service(
        self,
        socket_path: Path,
        status: dict[str, Any] | None = None,
        *,
        expected_active_job: str | None = None,
        info_path: Path | None = None,
    ) -> bool:
        """Release an idle service or the exact abandoned job we own."""

        info_path = info_path or socket_path.parent / "runtime_info.json"
        status = status or self._service_status(
            socket_path, info_path=info_path
        )
        if status is None:
            return False
        active_job = status.get("active_job")
        if active_job and active_job != expected_active_job:
            raise RuntimeError(
                "the persistent TT runtime is executing another training job; "
                "finish or interrupt that job before starting an isolated "
                "device run"
            )
        pid = int(status.get("pid") or 0)
        if pid <= 1 or pid == os.getpid():
            raise RuntimeError("persistent TT runtime reported an invalid PID")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            os.kill(pid, signal.SIGKILL)
        socket_path.unlink(missing_ok=True)
        info_path.unlink(missing_ok=True)
        return True

    def _cancel_owned_persistent_job(
        self,
        socket_path: Path,
        job_id: str,
    ) -> bool:
        """Cancel only the package service still running this exact job."""

        info_path = self._persistent_root() / "runtime_info.json"
        status = self._service_status(socket_path, info_path=info_path)
        if status is None or status.get("active_job") != job_id:
            return False
        return self._stop_persistent_service(
            socket_path,
            status,
            expected_active_job=job_id,
            info_path=info_path,
        )

    def _ensure_persistent_service(self) -> tuple[Path, dict[str, Any]]:
        root = self._persistent_root()
        socket_path = self._persistent_socket_path()
        info_path = root / "runtime_info.json"
        fingerprint = self._direct_fingerprint()
        status = self._service_status(socket_path, info_path=info_path)
        if (
            status is not None
            and status.get("fingerprint") == fingerprint
        ):
            return socket_path, status
        if status is not None:
            self._stop_persistent_service(
                socket_path, status, info_path=info_path
            )
        socket_path.unlink(missing_ok=True)
        info_path.unlink(missing_ok=True)
        log_path = root / "runtime_service.log"
        environment = os.environ.copy()
        environment.update({
            "PJRT_DEVICE": "TT",
            "XLA_STABLEHLO_COMPILE": "1",
            "WANDB_MODE": "disabled",
        })
        command = [
            sys.executable, "-m", "tt_finetune.runtime_service",
            "--socket", str(socket_path),
            "--fingerprint", fingerprint,
            "--info", str(info_path),
        ]
        log = log_path.open("a", encoding="utf-8")
        subprocess.Popen(
            command,
            cwd=self._engine.workspace,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            status = self._service_status(socket_path, info_path=info_path)
            if status is not None:
                return socket_path, status
            time.sleep(0.25)
        raise RuntimeError(
            "persistent TT runtime did not become ready:\n"
            + (log_path.read_text(errors="replace")[-8000:] if log_path.exists() else "")
        )

    def _compile_cache_id(self) -> str:
        contract = {
            "engine": self._direct_fingerprint(),
            "model": self.model.name_or_path,
            "revision": self.model.revision,
            "sequence_length": self.args.max_seq_length or self.model.max_seq_length,
            "batch_size": self.args.per_device_train_batch_size,
            "dtype": self.args.dtype,
            "lora": asdict(self.model.lora) if self.model.lora else None,
            "memory_mode": self.args.memory_mode,
        }
        return hashlib.sha256(
            json.dumps(contract, sort_keys=True, default=str).encode()
        ).hexdigest()[:24]

    def _direct_fingerprint(self) -> str:
        root = Path(__file__).resolve().parents[2]
        paths = (
            "tt_finetune/runtime/loop.py",
            "tt_finetune/runtime/torch_xla_ops.py",
            "tt_finetune/runtime/training_dropout.py",
            "tt_finetune/runtime/memory_planner.py",
            "tt_finetune/runtime/execution_planner.py",
            "tt_finetune/runtime/autotuner.py",
            "tt_finetune/data_pipeline.py",
            "tt_finetune/capabilities.py",
            "tt_finetune/kernels/streamed_causal_attention.py",
            "tt_finetune/kernels/dynamic_sparse_expert.py",
            "tt_finetune/optim.py",
            "tt_finetune/worker.py",
            "tt_finetune/config.py",
            "tt_finetune/backends/direct.py",
            "tt_finetune/runtime_service.py",
            "tt_finetune/service_client.py",
            "blacksmith/datasets/torch/jsonl_causal/jsonl_causal_dataset.py",
            "blacksmith/tools/checkpoints_manager.py",
            "blacksmith/tools/templates/configs.py",
            "blacksmith/experiments/torch/qwen/configs.py",
            "blacksmith/models/torch/huggingface/hf_models.py",
            "blacksmith/tools/logging_manager.py",
            "blacksmith/tools/workaround_utils.py",
            "tt_finetune/integration/architecture_runtime.py",
            "tt_finetune/integration/grouped_peft.py",
            "tt_finetune/integration/resident_attention.py",
            "tt_finetune/integration/sparse_moe.py",
            "tt_finetune/integration/streamed_attention.py",
            "tt_finetune/kernels/native_grouped_lora_autograd.py",
            "tt_finetune/kernels/native_full_attention_autograd.py",
            "tt_finetune/kernels/tt_lang_output_attention_resident_backward.py",
        )
        digest = hashlib.sha256(
            self._engine._engine_fingerprint().encode("utf-8")
        )
        for relative in paths:
            path = root / relative
            if not path.is_file():
                path = self._engine.workspace / relative
            if not path.is_file():
                raise TTDependencyError(
                    "Cannot fingerprint required direct-runtime source: "
                    f"{relative}. The tt-finetune wheel requires a compatible "
                    "TT runtime workspace for external TT-XLA/TT-Metal sources."
                )
            digest.update(relative.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()[:20]

    def _direct_registry_config(self, path: Path) -> dict[str, Any]:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["engine_fingerprint"] = self._direct_fingerprint()
        config["execution_engine"] = "tt_finetune.runtime.TrainingLoop"
        config["optimizer_layout"] = self.args.optimizer_layout
        config["optimizer_pack_bytes"] = self.args.optimizer_pack_bytes
        config["optimizer_packing_strategy"] = (
            self.args.optimizer_packing_strategy
        )
        config["execution_graph_layout"] = (
            self.args.execution_graph_layout
        )
        selected = getattr(self, "_selected_execution_candidate", None)
        if selected is not None:
            config["selected_execution_candidate"] = selected.to_dict()
        return config

    def _use_monolithic_step(self, report: dict[str, Any]) -> bool:
        """Choose the one-barrier step when the admitted graph is bounded.

        The architecture exact qualifiers use this schedule.  Long-sequence and
        very large admitted graphs retain the safer phase boundaries.  This
        decision consumes geometry and memory facts only, never model names.
        """

        if isinstance(self.model.adapter_config, IA3Config):
            # resident runtime's one-call gradient arena is LoRA-layout-specific. IA3
            # stays on the proven generic phase-split graph until a native IA3
            # owner exists.
            return False
        if self.args.execution_graph_layout == "monolithic":
            return True
        if self.args.execution_graph_layout == "phase_split":
            return False
        plan = report.get("execution_plan", {})
        estimated = int(plan.get("estimated_device_bytes") or 0)
        sequence = int(plan.get("sequence_length") or 0)
        expert_weights = int(plan.get("moe_expert_weight_bytes") or 0)
        return bool(
            plan.get("memory_admitted")
            and 0 < estimated <= 8 * 1024**3
            and 0 < sequence <= 128
            and expert_weights == 0
        )

    def _graph_worker_arguments(
        self, report: dict[str, Any]
    ) -> list[str]:
        selected = getattr(self, "_selected_execution_candidate", None)
        if selected is not None:
            if selected.graph_partition == "monolithic":
                return ["--fuse-step"]
            if selected.graph_partition == "forward_backward_fused":
                return [
                    "--split-clip-optimizer",
                    "--split-optimizer-phases",
                ]
            arguments = [
                "--split-clip-optimizer",
                "--split-after-loss",
                "--split-optimizer-phases",
            ]
            arguments.append(
                "--auto-split-after-forward"
                if selected.graph_partition == "phase_split_auto"
                else "--split-after-forward"
            )
            return arguments
        if self._use_monolithic_step(report):
            return ["--fuse-step"]
        arguments = ["--split-clip-optimizer", "--split-after-loss"]
        if self.args.optimizer_layout in {"auto", "tile_packed"}:
            arguments.append("--split-optimizer-phases")
        arguments.append(
            "--auto-split-after-forward"
            if self.args.execution_graph_layout == "auto"
            else "--split-after-forward"
        )
        return arguments

    def _selected_worker_arguments(self) -> tuple[str, ...]:
        """Return the complete measured runtime ABI, if one was selected."""

        selected = getattr(self, "_selected_execution_candidate", None)
        if selected is None:
            return ()
        from ..runtime import candidate_worker_arguments

        return candidate_worker_arguments(selected)

    def _execution_profile_contract(
        self, automatic_plan: dict[str, Any]
    ) -> dict[str, Any]:
        contract = json.loads(json.dumps(
            automatic_plan["request_contract"], sort_keys=True
        ))
        contract["compiler_hash"] = self._engine._engine_fingerprint()
        contract["runtime_hash"] = self._direct_fingerprint()
        contract["source_hash"] = self._direct_fingerprint()
        return contract

    @staticmethod
    def _state_contract(metrics: dict[str, Any]) -> dict[str, Any]:
        state = metrics.get("final_training_state") or {}
        optimizer = state.get("optimizer_logical") or state.get("optimizer") or {}
        return {
            "trainable": (state.get("trainable") or {}).get("sha256"),
            "optimizer": optimizer.get("sha256"),
            "losses": metrics.get("step_losses"),
        }

    def _autotune(
        self,
        *,
        report: dict[str, Any],
        resolved: Path,
        pipeline: Any,
        force: bool = False,
    ) -> dict[str, Any] | None:
        """Measure a bounded set of currently executable whole-step plans."""

        support = self._memory_support()
        automatic = support.automatic_plan
        if not automatic:
            if not (self.args.auto_tune or force):
                return None
            raise RuntimeError(
                "geometry-aware autotuning has no analytically admitted candidate; "
                "inspect trainer.plan() for the safe memory/capability reason"
            )
        from ..runtime import (
            ExecutionProfileCache,
            adaptive_candidate_portfolio,
            execution_candidate_from_dict,
            candidate_worker_arguments,
            select_non_regressing_winner,
            semantic_profile_key,
        )

        contract = self._execution_profile_contract(automatic)
        profile_key = semantic_profile_key(contract)
        cache_root = self._execution_profile_cache_root()
        cache = ExecutionProfileCache(cache_root / "profiles.json")
        cached_key, cached, cache_reason = cache.lookup_contract_explain(
            contract
        )
        assert cached_key == profile_key
        if cached is not None:
            selected = execution_candidate_from_dict(cached["selected"])
            self._selected_execution_candidate = selected
            return {
                **cached,
                "selection_status": "cache_hit_measured",
                "cache_status": "hit",
                "cache_reason": cache_reason,
            }
        if not (self.args.auto_tune or force):
            # A cache miss must still consume the planner's admitted static
            # execution contract.  Falling back to the legacy CLI defaults is
            # not equivalent: in particular, a large reduced-weight graph may
            # export the complete autograd-saved forward boundary and fragment
            # P150a DRAM even though the geometry planner selected the proven
            # forward/backward-fused schedule.  This is an analytical/default
            # selection, never mislabeled as a measured autotuning result.
            selected_document = automatic.get("selected") or {}
            selected = execution_candidate_from_dict(selected_document)
            if not selected.admitted:
                raise RuntimeError(
                    "the geometry-aware planner selected a rejected static "
                    f"candidate {selected.candidate_id}: "
                    + "; ".join(selected.rejection_reasons)
                )
            self._selected_execution_candidate = selected
            return {
                **automatic,
                "profile_key": profile_key,
                "selection_status": "analytical_unmeasured",
                "cache_status": "miss",
                "cache_reason": cache_reason,
                "selected": selected.to_dict(),
                "request_contract": contract,
                "measurements": [],
                "qualification": {
                    "measured": False,
                    "training_state_reused": False,
                    "selection_authority": (
                        "admitted static geometry plan"
                    ),
                    "note": (
                        "Call trainer.compile() or set auto_tune=True to "
                        "measure alternative plans."
                    ),
                },
            }

        if self.args.runtime_mode == "persistent":
            # Candidate authorities run in fresh PJRT processes. An idle
            # long-lived service still owns device mappings and can make the
            # first candidate fail with a misleading TLB/NOC admission error.
            # Release it before profiling; the selected plan is materialized
            # into a new service after the search completes.
            runtime_root = self._persistent_root()
            socket_path = self._persistent_socket_path()
            self._stop_persistent_service(
                socket_path,
                info_path=runtime_root / "runtime_info.json",
            )

        candidates = [
            execution_candidate_from_dict(item)
            for item in automatic.get("candidate_audit", [])
            if item.get("admitted")
        ]
        candidates.sort(key=lambda item: (
            item.analytical_score, item.candidate_id
        ))
        if not candidates:
            raise RuntimeError("geometry-aware planner returned no executable candidate")

        # Deduplicate analytical candidates that lower to the same worker ABI.
        by_runtime: dict[tuple[str, ...], Any] = {}
        for candidate in candidates:
            signature = candidate_worker_arguments(candidate)
            by_runtime.setdefault(signature, candidate)
        candidates = list(by_runtime.values())
        # The analytical planner's selected candidate is the static authority.
        # Reconstructing it from CLI defaults was subtly wrong for large
        # reduced-weight graphs: the planner selected the capacity-safe fused
        # boundary while this path silently put the materializing phase split
        # back first. Candidate identity is semantic/runtime-keyed, so consume
        # the planner selection verbatim.
        selected_id = str(
            (automatic.get("selected") or {}).get("candidate_id") or ""
        )
        reference = next(
            (item for item in candidates if item.candidate_id == selected_id),
            candidates[0],
        )
        candidates = list(adaptive_candidate_portfolio(
            candidates,
            static_candidate_id=reference.candidate_id,
            budget=self.args.auto_tune_max_candidates,
        ))

        warmups = self.args.auto_tune_warmup_steps
        measured_steps = self.args.auto_tune_measurement_steps
        total_steps = warmups + measured_steps
        run_root = (
            pipeline.generated_dir / "autotune" / profile_key
            / str(time.time_ns())
        )
        measurements: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        oracle: dict[str, Any] | None = None

        def run_candidate(candidate: Any, output_host: Path) -> tuple[
            dict[str, Any], tuple[float, ...], dict[str, Any]
        ]:
            output_host.mkdir(parents=True, exist_ok=False)
            output_container = self._engine._to_container_path(output_host)
            cache_host = (
                cache_root / "xla" / profile_key / output_host.name
            )
            cache_host.mkdir(parents=True, exist_ok=True)
            arguments = [
                "python", "-m", "tt_finetune.worker",
                "--config", self._engine._to_container_path(resolved),
                "--output-dir", output_container,
                "--max-steps", str(total_steps),
                "--loss-read-interval", "1",
                "--audit-training-state",
                "--no-eval",
                "--xla-cache", self._engine._to_container_path(cache_host),
                *candidate_worker_arguments(candidate),
            ]
            telemetry = output_host / "telemetry.json"
            self._run_local(
                device=True,
                arguments=arguments,
                telemetry_path=telemetry,
                device_environment=self._device_environment(report),
            )
            metrics = json.loads(
                (output_host / "training_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            if candidate.peft_strategy != "framework_safe":
                native = metrics.get("native_lora") or {}
                if int(native.get("installed_count") or 0) <= 0:
                    raise RuntimeError(
                        f"candidate {candidate.candidate_id} requested "
                        f"{candidate.peft_strategy}, but installed no native "
                        "LoRA owners"
                    )
                paths = metrics.get("peft_execution_paths") or {}
                expected_path = {
                    "native_grouped": "native_grouped2",
                    "native_resident_mlp": "native_resident_full_mlp_owner",
                    "native_resident_mlp_attention": (
                        "native_resident_full_mlp_owner"
                    ),
                }[candidate.peft_strategy]
                if not any(
                    name.startswith(expected_path) and int(count) > 0
                    for name, count in paths.items()
                ):
                    raise RuntimeError(
                        f"candidate {candidate.candidate_id} requested "
                        f"{candidate.peft_strategy}, but its measured worker "
                        f"reported execution paths {paths}"
                    )
                if candidate.peft_strategy == "native_resident_mlp_attention":
                    attention = metrics.get("resident_attention") or {}
                    if int(attention.get("installed_count") or 0) <= 0:
                        raise RuntimeError(
                            f"candidate {candidate.candidate_id} installed no "
                            "resident attention owner"
                        )
            if candidate.moe_schedule is not None:
                paths = metrics.get("moe_execution_paths") or {}
                if int(paths.get(candidate.moe_schedule) or 0) <= 0:
                    raise RuntimeError(
                        f"candidate {candidate.candidate_id} requested "
                        f"{candidate.moe_schedule}, but its measured worker "
                        f"reported MoE paths {paths}"
                    )
                install = (
                    (metrics.get("architecture") or {}).get("moe_install")
                    or {}
                )
                expected_backend = {
                    "assignment_weight_gather": "safe",
                    "expert_capacity_bucketed": "sparse",
                    "dynamic_native_expert_bucketed": "dynamic",
                }[candidate.moe_schedule]
                if (
                    install.get("backend") != expected_backend
                    or install.get("host_token_routing_loop") is not False
                    or install.get("evaluates_all_experts") is not False
                ):
                    raise RuntimeError(
                        f"candidate {candidate.candidate_id} did not realize "
                        "the requested selected-only device MoE contract: "
                        f"{install}"
                    )
            samples = tuple(
                float(value) * 1000.0
                for value in metrics.get("step_e2e_seconds", [])[warmups:]
            )
            if len(samples) != measured_steps:
                raise RuntimeError(
                    f"candidate {candidate.candidate_id} produced "
                    f"{len(samples)} measured steps, expected {measured_steps}"
                )
            state = self._state_contract(metrics)
            return metrics, samples, state

        def reduced_contract(metrics: dict[str, Any]) -> str | None:
            return (
                (metrics.get("memory_accounting") or {})
                .get("weight_memory_mode", {})
                .get("numerical_contract")
            )

        def finite_training_state(metrics: dict[str, Any]) -> bool:
            state = metrics.get("final_training_state") or {}
            losses = [
                value for value in metrics.get("step_losses", [])
                if value is not None
            ]
            return bool(
                losses
                and all(math.isfinite(float(value)) for value in losses)
                and (state.get("trainable_finiteness") or {}).get("finite")
                and (
                    state.get("optimizer_logical_finiteness")
                    or state.get("optimizer_finiteness")
                    or {}
                ).get("finite")
            )

        for candidate in candidates:
            output_host = run_root / candidate.candidate_id
            try:
                metrics, samples, state = run_candidate(candidate, output_host)
            except Exception as error:
                if candidate.candidate_id == reference.candidate_id:
                    raise RuntimeError(
                        "the static geometry-aware authority failed and tuning cannot "
                        "make a safe selection"
                    ) from error
                rejected.append({
                    "candidate_id": candidate.candidate_id,
                    "reason": "candidate compile/runtime/realization failure",
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                continue
            state_matches_oracle = oracle is None or state == oracle
            if oracle is None:
                oracle = state
            numerical_contract = reduced_contract(metrics)
            deterministic_replay_required = bool(
                numerical_contract
                and numerical_contract.startswith("deterministic_native_")
                and finite_training_state(metrics)
                and (
                    not state_matches_oracle
                    or candidate.graph_partition == "forward_backward_fused"
                )
            )
            if not state_matches_oracle and not deterministic_replay_required:
                rejected.append({
                    "candidate_id": candidate.candidate_id,
                    "reason": "evolving training state differs from safe oracle",
                    "state": state,
                    "numerical_contract": numerical_contract,
                })
                continue
            phase_medians = {
                name: float(summary["median_seconds"]) * 1000.0
                for name, summary in (metrics.get("phase_summaries") or {}).items()
                if summary.get("median_seconds") is not None
            }
            measurements.append({
                "candidate_id": candidate.candidate_id,
                "sample_count": len(samples),
                "median_ms": statistics.median(samples),
                "p10_ms": _percentile_local(samples, 0.10),
                "p90_ms": _percentile_local(samples, 0.90),
                "samples_ms": list(samples),
                "useful_tokens_per_step": list(
                    (metrics.get("step_useful_tokens") or [])[warmups:]
                ),
                "median_useful_tokens_per_second": statistics.median([
                    float(tokens) / (milliseconds / 1000.0)
                    for tokens, milliseconds in zip(
                        (metrics.get("step_useful_tokens") or [])[warmups:],
                        samples,
                    )
                ]) if len(
                    (metrics.get("step_useful_tokens") or [])[warmups:]
                ) == len(samples) else None,
                "median_physical_tokens_per_second": statistics.median([
                    float(tokens) / (milliseconds / 1000.0)
                    for tokens, milliseconds in zip(
                        (metrics.get("step_padded_tokens") or [])[warmups:],
                        samples,
                    )
                ]) if len(
                    (metrics.get("step_padded_tokens") or [])[warmups:]
                ) == len(samples) else None,
                "median_data_wait_ms": statistics.median([
                    float(value) * 1000.0
                    for value in (
                        metrics.get("step_data_wait_seconds") or []
                    )[warmups:]
                ]) if len(
                    (metrics.get("step_data_wait_seconds") or [])[warmups:]
                ) == len(samples) else None,
                "region_ms": phase_medians,
                "estimated_compile_seconds": metrics.get(
                    "estimated_compile_seconds",
                    (metrics.get("compile") or {}).get(
                        "estimated_compile_seconds"
                    ),
                ),
                "state": state,
                "losses": metrics.get("step_losses"),
                "state_matches_safe_oracle": state_matches_oracle,
                "numerical_contract": numerical_contract,
                "deterministic_replay_required": (
                    deterministic_replay_required
                ),
            })
        if not measurements:
            raise RuntimeError("every geometry-aware measured candidate was rejected")
        # BF16 candidates must match the safe oracle. Reduced frozen-weight
        # schedules have a separate deterministic contract, so a changed
        # arithmetic grouping is accepted only after a fresh-process replay
        # reproduces every evolving loss and final logical state exactly.
        remaining_measurements = list(measurements)
        winner = None
        selection_safety = None
        while remaining_measurements:
            proposed, selection_safety = select_non_regressing_winner(
                remaining_measurements,
                static_candidate_id=reference.candidate_id,
            )
            if not proposed["deterministic_replay_required"]:
                winner = proposed
                break
            candidate = next(
                item for item in candidates
                if item.candidate_id == proposed["candidate_id"]
            )
            replay_host = run_root / (
                candidate.candidate_id + "-determinism-replay"
            )
            replay_metrics, _, replay_state = run_candidate(
                candidate, replay_host
            )
            replay_exact = bool(
                replay_state == proposed["state"]
                and replay_metrics.get("step_losses") == proposed["losses"]
                and finite_training_state(replay_metrics)
            )
            proposed["deterministic_replay"] = {
                "fresh_worker_process": True,
                "exact": replay_exact,
                "state": replay_state,
            }
            if replay_exact:
                winner = proposed
                break
            rejected.append({
                "candidate_id": candidate.candidate_id,
                "reason": (
                    "reduced-weight schedule failed fresh-process "
                    "determinism replay"
                ),
                "state": proposed["state"],
                "replay_state": replay_state,
            })
            remaining_measurements.remove(proposed)
        if winner is None:
            raise RuntimeError(
                "every measured geometry-aware candidate failed its numerical "
                "contract"
            )
        selected = next(
            item for item in candidates
            if item.candidate_id == winner["candidate_id"]
        )
        profile = {
            "profile_key": profile_key,
            "selection_status": "measured",
            "cache_status": "miss_measured",
            "cache_reason": cache_reason,
            "selected": selected.to_dict(),
            "candidates_generated": automatic["candidates_generated"],
            "candidates_admitted": automatic["candidates_admitted"],
            "candidates_profiled": len(measurements),
            "candidate_audit": automatic["candidate_audit"],
            "measurements": measurements,
            "rejected_measurements": rejected,
            "safe_fallback": automatic["safe_fallback"],
            "request_contract": contract,
            "qualification": {
                "warmup_steps": warmups,
                "measured_steps": measured_steps,
                "fresh_worker_process_per_candidate": True,
                "safe_oracle_required_for_bf16": True,
                "reduced_weight_requires_exact_self_replay": True,
                "winner_state_matches_safe_oracle": winner[
                    "state_matches_safe_oracle"
                ],
                "winner_deterministic_replay": winner.get(
                    "deterministic_replay"
                ),
                "selection_metric": (
                    "median_useful_tokens_per_second"
                    if winner.get("median_useful_tokens_per_second") is not None
                    else "median_synchronized_step_latency"
                ),
                "adaptive_macro_first_search": True,
                "static_regression_guard": selection_safety,
            },
        }
        cache.store_document(profile_key=profile_key, profile=profile)
        self._selected_execution_candidate = selected
        return profile

    def _execution_profile_cache_root(self) -> Path:
        """Return the shared semantic-profile cache root.

        The key already includes compiler/runtime/source and full semantic
        geometry. Keeping the inventory outside a run-specific generated
        directory lets different output paths reuse the same measured plan
        without reusing any model, optimizer, RNG, or dataset state.
        """

        return Path(
            self.args.compile_cache_dir
            or (
                self._engine.workspace
                / "generated/profiles"
            )
        )

    def profile_cache_inventory(self) -> dict[str, Any]:
        from ..runtime import ExecutionProfileCache

        root = self._execution_profile_cache_root()
        summary = ExecutionProfileCache(root / "profiles.json").describe()
        return {
            **summary,
            "cache_root": str(root),
            "executable_persistence": {
                "cross_process_deserialization": False,
                "reason": (
                    "the installed PJRT runtime does not expose safe "
                    "executable deserialization"
                ),
                "same_process_reuse": True,
            },
        }

    def data_plan(self) -> dict[str, Any]:
        """Inspect and jointly plan host data geometry without PJRT/device use."""

        from ..data_pipeline import plan_data_geometries
        from ..runtime.memory_planner import P150A_ALLOCATABLE_DRAM_BYTES

        pipeline, _, resolved_paths = self._engine._prepare()
        resolved = resolved_paths["adapter"]
        output_host = pipeline.generated_dir / "data_plan.json"
        self._run_local(
            device=False,
            arguments=[
                "python", "-m", "tt_finetune.data_inspector",
                "--config", self._engine._to_container_path(resolved),
                "--output", self._engine._to_container_path(output_host),
            ],
        )
        report = json.loads(output_host.read_text(encoding="utf-8"))
        support = self._memory_support()
        measured_step_seconds = None
        automatic = support.automatic_plan
        if automatic:
            contract = self._execution_profile_contract(automatic)
            from ..runtime import ExecutionProfileCache

            _key, cached, _reason = ExecutionProfileCache(
                self._execution_profile_cache_root() / "profiles.json"
            ).lookup_contract_explain(contract)
            if cached is not None:
                selected_id = (cached.get("selected") or {}).get(
                    "candidate_id"
                )
                winner = next((
                    item for item in cached.get("measurements", [])
                    if item.get("candidate_id") == selected_id
                ), None)
                if winner and winner.get("median_ms") is not None:
                    measured_step_seconds = float(winner["median_ms"]) / 1000.0

        histogram = (
            report.get("shape_distribution", {})
            .get("tile_length_histogram", {})
        )
        sequence_length = (
            self.args.max_seq_length or self.model.max_seq_length
        )
        approximate_lengths = [
            int(boundary)
            for boundary, count in histogram.items()
            for _ in range(int(count))
        ]
        if not approximate_lengths:
            approximate_lengths = [sequence_length]
        report["automatic_data_geometry_plan"] = plan_data_geometries(
            approximate_lengths,
            batch_size=self.args.per_device_train_batch_size,
            sequence_length=sequence_length,
            gradient_accumulation_steps=(
                self.args.gradient_accumulation_steps
            ),
            seed=self.args.seed,
            packing_enabled=bool(report["packing"]["enabled"]),
            packing_semantically_safe=bool(
                self.data_config.format == "text"
                and self.data_config.append_eos_token
            ),
            estimated_device_bytes=support.estimated_device_bytes,
            device_capacity_bytes=P150A_ALLOCATABLE_DRAM_BYTES,
            measured_step_seconds=measured_step_seconds,
            preparation_seconds_per_batch=(
                float(report["tokenization"]["seconds"])
                / max(1, int(report["accepted_rows"]))
            ),
        )
        report["planner_integration"] = {
            "same_semantic_contract": True,
            "hardware_step_measurement_available": (
                measured_step_seconds is not None
            ),
            "selection_is_production_accepted": False,
            "reason": (
                "data candidates are planner-visible; each changed B/S/G "
                "portfolio still requires synchronized whole-step profiling"
            ),
        }
        output_host.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return report

    def compile_profile(self) -> dict[str, Any]:
        """Measure and cache a plan without consuming the user's train run."""

        started = time.perf_counter()
        pipeline, _, resolved_paths = self._engine._prepare()
        resolved = resolved_paths["adapter"]
        report_path = pipeline.generated_dir / "preflight_adapter.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        memory_support = self._memory_support()
        report["memory_plan"] = memory_support.memory_breakdown
        report["memory_mode_requested"] = (
            memory_support.memory_mode_requested
        )
        report["memory_mode_selected"] = (
            memory_support.memory_mode_selected
        )
        if not memory_support.memory_admitted:
            from ..errors import TTOutOfMemoryError

            raise TTOutOfMemoryError(
                "The requested single-P150a memory contract is not admitted: "
                + memory_support.reason
            )
        profile = self._autotune(
            report=report,
            resolved=resolved,
            pipeline=pipeline,
            force=True,
        )
        assert profile is not None
        report["execution_profile"] = profile
        report_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        persistent_runtime = None
        if getattr(
            getattr(self, "args", None), "runtime_mode", "isolated"
        ) == "persistent":
            persistent_runtime = self._warm_persistent_profile(
                pipeline=pipeline,
                resolved=resolved,
                report=report,
            )
        return {
            "status": "compiled_profile_ready",
            "elapsed_seconds": time.perf_counter() - started,
            "selection_status": profile["selection_status"],
            "cache_status": profile["cache_status"],
            "cache_reason": profile["cache_reason"],
            "profile_key": profile["profile_key"],
            "selected": profile["selected"],
            "measurements": profile["measurements"],
            "qualification": profile["qualification"],
            "cache": self.profile_cache_inventory(),
            "persistent_runtime": persistent_runtime,
            "training_state_reused": False,
            "note": (
                "Candidate profiling uses fresh model/optimizer worker "
                "processes. A later training run reuses the measured plan, "
                "never candidate training state."
            ),
        }

    def _warm_persistent_profile(
        self,
        *,
        pipeline: Any,
        resolved: Path,
        report: dict[str, Any],
    ) -> dict[str, Any]:
        """Materialize selected bucket executables in the long-lived service.

        The bounded warmup job owns fresh temporary model/optimizer/RNG state
        and discards it when complete. Only process-owned compiler/executable
        caches survive, which lets the later public train job start from a
        clean state while avoiding cold per-bucket PJRT compiles in the same
        service.
        """

        _, status = self._ensure_persistent_service()
        cache_key = self._persistent_cache_id(report)
        if cache_key in set(status.get("cache_keys") or ()):
            return {
                "status": "executable_ready",
                "cache_status": "hit",
                "cache_key": cache_key,
                "warmup_job_executed": False,
                "same_process_reuse": True,
                "training_state_reused": False,
            }
        output_host = (
            pipeline.generated_dir / "persistent_compile_warmup"
            / f"{cache_key}-{uuid.uuid4().hex[:10]}"
        )
        output_host.mkdir(parents=True, exist_ok=False)
        from ..data_pipeline import normalize_sequence_buckets

        bucketed = self.data_config.sequence_bucketing is not False
        boundaries = (
            normalize_sequence_buckets(
                self.args.max_seq_length or self.model.max_seq_length,
                self.data_config.sequence_buckets,
            )
            if bucketed else (
                self.args.max_seq_length or self.model.max_seq_length,
            )
        )
        # Use a real batch from each populated bucket in explicit round-robin
        # order. Four passes provide a bounded executable-reuse check without
        # relying on a rare shape appearing early in the shuffled epoch.
        compile_bucket_repeats = 4
        warmup_steps = max(1, compile_bucket_repeats * len(boundaries))
        metrics = self._execute_persistent(
            resolved=resolved,
            report=report,
            output=self._engine._to_container_path(output_host),
            host_output=output_host,
            resume=False,
            stop_after=None,
            telemetry_path=output_host / "telemetry.json",
            max_steps_override=warmup_steps,
            compile_sequence_buckets=boundaries,
            compile_bucket_repeats=compile_bucket_repeats,
        )
        runtime = metrics.get("persistent_runtime") or {}
        observed_sequence_lengths = sorted({
            int(value)
            for step in metrics.get("step_sequence_lengths", [])
            for value in step
        })
        if not observed_sequence_lengths:
            observed_sequence_lengths = sorted(set(
                int(value)
                for value in metrics.get("step_padded_tokens", [])
            ))
        return {
            "status": "executable_ready",
            "cache_status": runtime.get("client_cache_status", "miss"),
            "cache_key": cache_key,
            "warmup_job_executed": True,
            "warmup_metrics": str(output_host / "training_metrics.json"),
            "warmup_steps": warmup_steps,
            "planned_sequence_buckets": list(boundaries),
            "observed_sequence_buckets": observed_sequence_lengths,
            "empty_sequence_buckets": (
                metrics.get("compile_warmup_schedule", {})
                .get("empty_buckets", [])
            ),
            "schedule": metrics.get("compile_warmup_schedule"),
            "same_process_reuse": True,
            "training_state_reused": False,
        }

    def _device_environment(
        self,
        report: dict[str, Any],
        *,
        memory_log_container: str | None = None,
    ) -> dict[str, str] | None:
        """Return capability-driven runtime source paths for one device job.

        TTML's streamed SDPA runtime creates its reader/compute/writer kernels
        from the external ``tt-train`` source tree.  The PJRT wheel contains
        the runtime library but not those sources.  Expose the matching
        workspace checkout only when the planner selected that implementation;
        ordinary TTNN programs continue using the wheel's system kernel tree.
        """

        environment: dict[str, str] = {}
        if memory_log_container is not None:
            environment["TT_METAL_LOGS_PATH"] = memory_log_container

        selections = report.get("execution_plan", {}).get(
            "selections", []
        )
        needs_ttml_sources = any(
            item.get("primitive") == "attention"
            and item.get("implementation") == "native_streamed_causal"
            for item in selections
            if isinstance(item, dict)
        )
        if needs_ttml_sources:
            relative = Path(
                "tt-train/sources/ttml/metal/ops/sdpa_fw/device/"
                "kernels/dataflow/sdpa_fw_reader_kernel.cpp"
            )
            source_root = (
                Path(__file__).resolve().parents[1]
                / "_resources" / "ttml"
            )
            if not (source_root / relative).is_file():
                raise TTDependencyError(
                    "The selected streamed-attention runtime requires the "
                    "matching TT-Metal tt-train kernel sources. Expected "
                    f"{source_root / relative}. Install the project runtime "
                    "bundle or choose a shorter qualified sequence geometry."
                )
            overlay = self._ttml_kernel_overlay(source_root)
            environment["TT_METAL_KERNEL_PATH"] = (
                self._engine._to_container_path(overlay)
            )
        return environment or None

    def _ttml_kernel_overlay(self, source_root: Path) -> Path:
        """Create a content-addressed include-safe view of TTML kernels.

        TT-Metal adds the selected entry kernel's directory to the JIT include
        path, while TTML headers use repository-rooted ``tt-train/...``
        includes.  A small generated overlay adds a relative ``tt-train`` link
        beside each source directory.  It neither edits the vendor checkout nor
        shadows ordinary TTNN kernels from the runtime wheel.
        """

        ttml_root = source_root / "tt-train/sources/ttml/metal"
        source_files = sorted(
            path for path in ttml_root.rglob("*") if path.is_file()
        )
        digest = hashlib.sha256()
        for path in source_files:
            digest.update(str(path.relative_to(source_root)).encode("utf-8"))
            digest.update(path.read_bytes())
        overlay = (
            self._engine.workspace / "generated/ttml_kernel_overlay"
            / digest.hexdigest()[:20]
        )
        overlay_tt_train = overlay / "tt-train"
        for source in source_files:
            relative = source.relative_to(source_root)
            destination = overlay / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() and not destination.is_symlink():
                link = os.path.relpath(source, destination.parent)
                try:
                    destination.symlink_to(link)
                except FileExistsError:
                    pass
            include_link = destination.parent / "tt-train"
            if not include_link.exists() and not include_link.is_symlink():
                link = os.path.relpath(overlay_tt_train, destination.parent)
                try:
                    include_link.symlink_to(link, target_is_directory=True)
                except FileExistsError:
                    pass
        return overlay

    def preflight(self) -> dict[str, Any]:
        report = self._engine.preflight()
        report.pop("known_hardware_metrics", None)
        report.pop("projected_steady_training_seconds", None)
        resolved = (
            self._engine.generated_dir / "adapter_adapter_resolved.yaml"
        )
        config = self._direct_registry_config(resolved)
        profile_id = self._engine.registry.record_preflight(report, config)
        registry = self._engine.registry.load()
        profile = (
            registry["models"]
            .get(self._engine.registry.model_key(report), {})
            .get("profiles", {})
            .get(profile_id, {})
        )
        report["compatibility_profile_id"] = profile_id
        report["hardware_status"] = profile.get(
            "hardware_status", "not_tested"
        )
        report["execution_engine"] = config["execution_engine"]
        report["engine_fingerprint"] = config["engine_fingerprint"]
        memory_support = self._memory_support()
        report["memory_plan"] = memory_support.memory_breakdown
        report["memory_mode_requested"] = (
            memory_support.memory_mode_requested
        )
        report["memory_mode_selected"] = (
            memory_support.memory_mode_selected
        )
        report["memory_admitted"] = memory_support.memory_admitted
        if not memory_support.memory_admitted:
            report["compatible"] = False
            report["reason"] = memory_support.reason
        if profile.get("metrics"):
            report["known_hardware_metrics"] = profile["metrics"]
        return report

    def _run_local(
        self,
        *,
        device: bool,
        arguments: list[str],
        telemetry_path: Path | None = None,
        device_environment: dict[str, str] | None = None,
    ) -> None:
        workspace = self._engine.workspace
        container = self._engine.container_workspace
        if arguments and arguments[0] == "python":
            arguments = [sys.executable, *arguments[1:]]
        command = list(arguments)
        environment = os.environ.copy()
        environment["WANDB_MODE"] = "disabled"
        if device:
            environment.update({
                "PJRT_DEVICE": "TT",
                "XLA_STABLEHLO_COMPILE": "1",
            })
            environment.update(device_environment or {})
        if device:
            telemetry = telemetry_path or (
                self._engine.generated_dir / "telemetry_public_train.json"
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve().parents[2] / "blackhole_training" / "run_with_tt_telemetry.py"),
                "--output", str(telemetry),
                "--interval", "1.0",
                "--max-runtime-seconds", "86400",
                "--", *command,
            ]
        if not device:
            subprocess.run(command, check=True, cwd=workspace, env=environment)
            return
        log_path = (telemetry_path or telemetry).with_suffix(".runtime.log")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=workspace,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                # The telemetry supervisor and accelerator worker must form
                # one owned process tree. Without a separate group, an
                # interrupted API call could leave the worker alive and retain
                # the device TLB mapping needed by the next job.
                start_new_session=True,
            )
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    log.write(line)
                    if self.args.verbose or any(marker in line for marker in (
                        "tt-finetune step", "Loading model", "Saved ",
                        "Traceback", "TT_FATAL", "ERROR", "Error",
                    )):
                        print(line, end="", file=sys.stderr)
                return_code = process.wait()
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
        if return_code:
            tail = "\n".join(
                log_path.read_text(errors="replace").splitlines()[-40:]
            )
            raise subprocess.CalledProcessError(
                return_code, command, output=tail
            )

    def _export(self, resolved: Path, output: str) -> None:
        config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        script = Path(__file__).resolve().parents[2] / "blackhole_training" / "export_peft_adapter.py"
        arguments = [
            "python",
            str(script),
            "--base-model",
            config["model_name"],
            "--metadata-base-model",
            self.model.name_or_path,
            "--checkpoint",
            f"{output}/checkpoints/final_model.pth",
            "--output",
            f"{output}/peft_adapter",
            "--targets",
            *config["lora_target_modules"],
            "--task-type",
            config.get("lora_task_type", "CAUSAL_LM"),
        ]
        adapter_type = config.get("adapter_type", "lora")
        arguments.extend(["--adapter-type", adapter_type])
        if adapter_type == "ia3":
            feedforward = config.get("ia3_feedforward_modules") or []
            if isinstance(feedforward, str):
                feedforward = [feedforward]
            arguments.extend(
                [
                    "--feedforward-modules",
                    *feedforward,
                    "--init-ia3-weights",
                    str(config.get("init_ia3_weights", True)).lower(),
                ]
            )
        else:
            arguments.extend(
                [
                    "--rank", str(config["lora_r"]),
                    "--alpha", str(config["lora_alpha"]),
                    "--dropout", str(config.get("lora_dropout", 0.0)),
                    "--bias", config.get("lora_bias", "none"),
                    "--init-weights",
                    str(config.get("lora_init_weights", True)).lower(),
                ]
            )
        if config.get("model_revision"):
            arguments.extend(["--model-revision", config["model_revision"]])
        if config.get("trust_remote_code"):
            arguments.append("--trust-remote-code")
        if config.get("attn_implementation"):
            arguments.extend(
                ["--attn-implementation", config["attn_implementation"]]
            )
        if not config.get("use_fast_tokenizer", True):
            arguments.append("--slow-tokenizer")
        if config.get("lora_modules_to_save"):
            arguments.extend(
                ["--modules-to-save", *config["lora_modules_to_save"]]
            )
        if config.get("lora_use_rslora"):
            arguments.append("--use-rslora")
        if config.get("lora_use_dora"):
            arguments.append("--use-dora")
        if config.get("lora_plus_lr_ratio") is not None:
            arguments.extend(
                [
                    "--lora-plus-lr-ratio",
                    str(config["lora_plus_lr_ratio"]),
                ]
            )
        if config.get("lora_layers_to_transform") is not None:
            arguments.extend(
                [
                    "--layers-to-transform",
                    *map(str, config["lora_layers_to_transform"]),
                ]
            )
        if config.get("lora_layers_pattern"):
            arguments.extend(
                ["--layers-pattern", config["lora_layers_pattern"]]
            )
        self._run_local(device=False, arguments=arguments)

    def _persistent_run_options(
        self,
        report: dict[str, Any],
        cache_path: Path,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "loss_read_interval": self.args.logging_steps,
            "xla_cache": self._persistent_container_path(cache_path),
            "optimizer_tile_packed": self.args.optimizer_layout in {
                "auto", "tile_packed"
            },
            "optimizer_pack_bytes": self.args.optimizer_pack_bytes,
            "optimizer_packing_strategy": (
                self.args.optimizer_packing_strategy
            ),
            "audit_training_state": self.args.audit_training_state,
            "profile_phases": self.args.profile_product_boundaries,
        }
        selected = getattr(self, "_selected_execution_candidate", None)
        if selected is not None:
            options.update({
                "optimizer_tile_packed": True,
                "optimizer_pack_bytes": selected.optimizer_pack_bytes,
                "optimizer_packing_strategy": (
                    selected.optimizer_packing_strategy
                ),
                "attention_query_block": selected.attention_query_block,
                "attention_backend": (
                    "safe"
                    if selected.attention_strategy == "safe_streamed"
                    else "native"
                    if selected.attention_strategy == "native_streamed"
                    else "auto"
                ),
                "peft_execution_strategy": selected.peft_strategy,
                "moe_execution_strategy": (
                    "dynamic"
                    if selected.moe_schedule
                    == "dynamic_native_expert_bucketed"
                    else "sparse"
                    if selected.moe_schedule == "expert_capacity_bucketed"
                    else "safe"
                    if selected.moe_schedule == "assignment_weight_gather"
                    else "auto"
                ),
            })
        mapping = {
            "--fuse-step": "fuse_step",
            "--split-clip-optimizer": "split_clip_optimizer",
            "--split-after-loss": "split_after_loss",
            "--split-optimizer-phases": "split_optimizer_phases",
            "--auto-split-after-forward": "auto_split_after_forward",
            "--split-after-forward": "split_after_forward",
        }
        for argument in self._graph_worker_arguments(report):
            if argument in mapping:
                options[mapping[argument]] = True
        return options

    def _execute_persistent(
        self,
        *,
        resolved: Path,
        report: dict[str, Any],
        output: str,
        host_output: Path,
        resume: bool | str,
        stop_after: int | None,
        telemetry_path: Path,
        max_steps_override: int | None = None,
        compile_sequence_buckets: tuple[int, ...] | None = None,
        compile_bucket_repeats: int = 4,
    ) -> dict[str, Any]:
        socket_path, service_status = self._ensure_persistent_service()
        if service_status.get("active_job"):
            raise RuntimeError(
                "the persistent TT runtime is already executing training job "
                f"{service_status['active_job']}; wait for it to finish or "
                "interrupt its owning TTTrainer call before starting another"
            )
        runtime_root = self._persistent_root()
        service_log_path = runtime_root / "runtime_service.log"
        try:
            service_log_start = service_log_path.stat().st_size
        except OSError:
            service_log_start = 0
        cache_path = runtime_root / "xla_cache" / self._direct_fingerprint()
        cache_path.mkdir(parents=True, exist_ok=True)
        job_id = f"{int(time.time())}-{uuid.uuid4().hex[:10]}"
        request_path = runtime_root / f"request-{job_id}.json"
        response_path = runtime_root / f"response-{job_id}.json"
        resume_value: bool | str = resume
        if isinstance(resume, str):
            resume_value = self._engine._to_container_path(resume)
        payload = {
            "action": "train",
            "job_id": job_id,
            "engine_fingerprint": self._direct_fingerprint(),
            "cache_key": self._persistent_cache_id(report),
            "config": self._engine._to_container_path(resolved),
            "output_dir": output,
            "max_steps": (
                self.args.max_steps
                if max_steps_override is None else max_steps_override
            ),
            "stop_after": stop_after,
            "resume_from_checkpoint": resume_value,
            "no_eval": self.eval_dataset is None,
            "run_options": self._persistent_run_options(
                report, cache_path
            ),
            "config_overrides": {
                "tt_enable_torch_fx_fusion_pass": (
                    self.args.experimental_fx_fusion
                )
            },
        }
        if stop_after is not None:
            payload["run_options"]["stop_after"] = stop_after
        if compile_sequence_buckets is not None:
            payload["run_options"]["compile_sequence_buckets"] = list(
                compile_sequence_buckets
            )
            payload["run_options"]["compile_bucket_repeats"] = int(
                compile_bucket_repeats
            )
        request_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        client_command = [
            sys.executable,
            "-m", "tt_finetune.service_client",
            "--socket", str(socket_path),
            "--request", str(request_path),
            "--response", str(response_path),
        ]
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[2] / "blackhole_training" / "run_with_tt_telemetry.py"),
            "--output", str(telemetry_path),
            "--interval", "1.0",
            "--max-runtime-seconds", "86400",
            "--", *client_command,
        ]
        log_path = telemetry_path.with_suffix(".runtime.log")
        client_started = time.perf_counter()
        last_stage = None
        last_heartbeat = client_started
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=self._engine.workspace,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                while process.poll() is None:
                    metrics_path = host_output / "training_metrics.json"
                    stage = None
                    if metrics_path.is_file():
                        try:
                            current = json.loads(
                                metrics_path.read_text(encoding="utf-8")
                            )
                            stage = (
                                current.get("startup_profile", {})
                                .get("current_stage")
                            )
                        except (json.JSONDecodeError, OSError):
                            # A concurrently published progress file can be
                            # observed between filesystem operations on exotic
                            # filesystems. The worker uses atomic replace, but
                            # retrying here keeps old release workers safe too.
                            stage = None
                    now = time.perf_counter()
                    if stage and stage != last_stage:
                        print(
                            f"tt-finetune | {stage.replace('_', ' ')} | "
                            f"{now - client_started:.1f}s elapsed",
                            file=sys.stderr,
                            flush=True,
                        )
                        last_stage = stage
                        last_heartbeat = now
                    elif now - last_heartbeat >= 60:
                        label = (
                            stage.replace("_", " ")
                            if stage else "starting runtime job"
                        )
                        print(
                            f"tt-finetune | {label} | "
                            f"{now - client_started:.1f}s elapsed",
                            file=sys.stderr,
                            flush=True,
                        )
                        last_heartbeat = now
                    time.sleep(1)
                return_code = process.wait()
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                self._cancel_owned_persistent_job(socket_path, job_id)
                raise
        client_elapsed = time.perf_counter() - client_started
        response = (
            json.loads(response_path.read_text(encoding="utf-8"))
            if response_path.is_file()
            else None
        )
        if return_code or not response or response.get("status") != "completed":
            self._cancel_owned_persistent_job(socket_path, job_id)
            detail = response or {
                "client_log": log_path.read_text(errors="replace")[-8000:]
            }
            native_failure = _native_runtime_failure_diagnostic(
                service_log_path,
                start_offset=service_log_start,
            )
            if native_failure is not None:
                detail = dict(detail)
                detail["error"] = native_failure["message"]
                detail["device_failure"] = native_failure
            raise RuntimeError(
                "persistent TT runtime job failed:\n"
                + json.dumps(detail, indent=2)
            )
        metrics_path = host_output / "training_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.setdefault("persistent_runtime", {}).update(
            {
                "client_cache_status": response["cache_status"],
                "service_elapsed_seconds": response[
                    "service_elapsed_seconds"
                ],
            }
        )
        service_elapsed = float(response["service_elapsed_seconds"])
        boundary_seconds = max(0.0, client_elapsed - service_elapsed)
        step_count = max(1, int(metrics.get("global_steps") or 1))
        steady = float(metrics.get("steady_step_seconds") or 0.0)
        per_step = boundary_seconds / step_count
        metrics["public_product_boundary"] = {
            "client_elapsed_seconds": client_elapsed,
            "service_elapsed_seconds": service_elapsed,
            "fixed_boundary_seconds": boundary_seconds,
            "amortized_seconds_per_step": per_step,
            "steady_step_overhead_fraction": (
                per_step / steady if steady > 0 else None
            ),
            "scope": (
                "persistent client, telemetry wrapper and IPC only; "
                "the matched internal TrainingLoop step is measured by "
                "step_e2e_seconds"
            ),
        }
        metrics_path.write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        return metrics

    def train(
        self,
        resume: bool | str = False,
        stop_after: int | None = None,
    ) -> TTTrainResult:
        pipeline, _, resolved_paths = self._engine._prepare()
        resolved = resolved_paths["adapter"]
        report_path = pipeline.generated_dir / "preflight_adapter.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        memory_support = self._memory_support()
        report["memory_plan"] = memory_support.memory_breakdown
        report["memory_mode_requested"] = (
            memory_support.memory_mode_requested
        )
        report["memory_mode_selected"] = (
            memory_support.memory_mode_selected
        )
        if not memory_support.memory_admitted:
            from ..errors import TTOutOfMemoryError

            raise TTOutOfMemoryError(
                "The requested single-P150a memory contract is not admitted: "
                + memory_support.reason
                + ". Use memory_mode='auto' or reduce rank/batch/sequence."
            )
        output = self._engine._to_container_path(
            self.args.output_dir,
            output=True,
        )
        host_output = self._engine._host_path(self.args.output_dir)
        execution_output = self._engine.execution_output_dir
        if host_output.exists() and any(host_output.iterdir()) and not resume:
            raise FileExistsError(
                f"Direct output already exists: {host_output}"
            )
        report_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        execution_profile = self._autotune(
            report=report,
            resolved=resolved,
            pipeline=pipeline,
        )
        if execution_profile is not None:
            report["execution_profile"] = execution_profile
            report_path.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
        registry_config = self._direct_registry_config(resolved)
        profile_id = self._engine.registry.record_preflight(
            report,
            registry_config,
        )
        registry = self._engine.registry.load()
        profile = (
            registry["models"]
            .get(self._engine.registry.model_key(report), {})
            .get("profiles", {})
            .get(profile_id, {})
        )
        if (
            self.args.auto_smoke
            and not self.args.auto_tune
            and profile.get("hardware_status") != "passed"
            and (
                self.args.max_steps is None
                or self.args.max_steps > self.args.smoke_steps
            )
        ):
            self.smoke(self.args.smoke_steps)
        self._engine._prepare_execution_output(resume=bool(resume))
        worker_arguments = [
            "python",
            "-m",
            "tt_finetune.worker",
            "--config",
            self._engine._to_container_path(resolved),
            "--output-dir",
            output,
            "--loss-read-interval",
            str(self.args.logging_steps),
        ]
        if self.args.max_steps is not None:
            worker_arguments.extend(
                ["--max-steps", str(self.args.max_steps)]
            )
        if stop_after is not None:
            worker_arguments.extend(["--stop-after", str(stop_after)])
        if resume:
            worker_arguments.append("--resume-from-checkpoint")
            if isinstance(resume, str):
                worker_arguments.append(
                    self._engine._to_container_path(resume)
                )
        cache_host = Path(
            self.args.compile_cache_dir
            or (pipeline.generated_dir / "xla_cache" / self._compile_cache_id())
        )
        cache_host.mkdir(parents=True, exist_ok=True)
        worker_arguments.extend([
            "--xla-cache", self._engine._to_container_path(cache_host)
        ])
        if self.args.audit_training_state:
            worker_arguments.append("--audit-training-state")
        if self.args.profile_product_boundaries:
            worker_arguments.append("--profile-phases")
        if self.args.experimental_fx_fusion:
            worker_arguments.append("--torch-fx-fusion")
        if self.eval_dataset is None:
            worker_arguments.append("--no-eval")
        selected = getattr(self, "_selected_execution_candidate", None)
        if (
            self.args.optimizer_layout in {"auto", "tile_packed"}
            and selected is None
        ):
            worker_arguments.append("--optimizer-tile-packed")
            worker_arguments.extend(
                [
                    "--optimizer-pack-bytes",
                    str(self.args.optimizer_pack_bytes),
                    "--optimizer-packing-strategy",
                    self.args.optimizer_packing_strategy,
                ]
            )
        if selected is not None:
            # The measured winner is an executable runtime contract, not only
            # a graph-partition hint.  Consume every qualified attention,
            # PEFT, MoE, optimizer, and partition choice exactly as it was
            # measured; otherwise public train() silently falls back to a
            # different and potentially much slower schedule.
            worker_arguments.extend(self._selected_worker_arguments())
        else:
            worker_arguments.extend(self._graph_worker_arguments(report))
        try:
            telemetry_path = (
                pipeline.generated_dir / "telemetry_public_train.json"
            )
            if self.args.runtime_mode == "persistent":
                metrics = self._execute_persistent(
                    resolved=resolved,
                    report=report,
                    output=output,
                    host_output=execution_output,
                    resume=resume,
                    stop_after=stop_after,
                    telemetry_path=telemetry_path,
                )
            else:
                # A long-lived service retains PJRT/TT-Metal device mappings
                # even while it owns no training state and is otherwise idle.
                # Starting a fresh-process run beside it can fail before model
                # loading with ``tt_tlb_alloc``/TLB-window exhaustion.  The
                # isolated contract requires exclusive process ownership, so
                # release an idle package service before spawning the worker;
                # _stop_persistent_service deliberately refuses to terminate
                # an unrelated active training job.
                runtime_root = self._persistent_root()
                self._stop_persistent_service(
                    self._persistent_socket_path(),
                    info_path=runtime_root / "runtime_info.json",
                )
                memory_log_host = execution_output / "tt_metal_logs"
                memory_log_container = f"{output}/tt_metal_logs"
                self._run_local(
                    device=True,
                    arguments=worker_arguments,
                    telemetry_path=telemetry_path,
                    device_environment=self._device_environment(
                        report,
                        memory_log_container=(
                            memory_log_container
                            if self.args.profile_device_memory else None
                        ),
                    ),
                )
                metrics_path = execution_output / "training_metrics.json"
                metrics = json.loads(
                    metrics_path.read_text(encoding="utf-8")
                )
                if self.args.profile_device_memory:
                    from ..memory import parse_tt_metal_memory_reports

                    device_memory = parse_tt_metal_memory_reports(
                        memory_log_host
                    )
                    metrics.setdefault("memory_accounting", {})[
                        "device_allocator"
                    ] = device_memory
                    logical = metrics.get("memory_accounting", {})
                    peak = device_memory.get(
                        "peak_compile_snapshot_dram"
                    ) or {}
                    peak_bytes = peak.get("allocated_bytes")
                    known = sum(
                        int(logical.get(name) or 0)
                        for name in (
                            "frozen_parameter_logical_bytes",
                            "trainable_parameter_logical_bytes",
                            "gradient_logical_bytes",
                            "optimizer_moment_logical_bytes",
                        )
                    )
                    logical["compile_snapshot_unclassified_bytes"] = (
                        max(0, int(peak_bytes) - known)
                        if peak_bytes is not None else None
                    )
                    metrics_path.write_text(
                        json.dumps(metrics, indent=2) + "\n",
                        encoding="utf-8",
                    )
            if execution_profile is not None:
                selected_plan = execution_profile.get("selected") or {}
                metrics["automatic_execution_plan"] = selected_plan
                metrics["execution_profile"] = {
                    "profile_key": execution_profile.get("profile_key"),
                    "selection_status": execution_profile.get(
                        "selection_status"
                    ),
                    "cache_status": execution_profile.get("cache_status"),
                    "selected_candidate_id": selected_plan.get(
                        "candidate_id"
                    ),
                }
                (execution_output / "training_metrics.json").write_text(
                    json.dumps(metrics, indent=2) + "\n",
                    encoding="utf-8",
                )
            self._export(resolved, output)
            metrics_path = execution_output / "training_metrics.json"
            adapter = execution_output / "peft_adapter"
            if not pipeline.valid_peft_adapter(adapter):
                raise RuntimeError(f"Invalid PEFT adapter: {adapter}")
            memory_report = metrics.get("memory_accounting", {}).get(
                "weight_memory_mode", {}
            )
            adapter_config = self.model.adapter_config
            assert adapter_config is not None
            if isinstance(adapter_config, IA3Config):
                adapter_features = ["ia3"]
                lora_plus_ratio = None
            else:
                adapter_features = [
                    name
                    for enabled, name in (
                        (adapter_config.use_rslora, "rslora"),
                        (adapter_config.use_dora, "dora"),
                        (
                            adapter_config.lora_plus_lr_ratio is not None,
                            "lora_plus",
                        ),
                    )
                    if enabled
                ]
                lora_plus_ratio = adapter_config.lora_plus_lr_ratio
            (adapter / "tt_finetune_training.json").write_text(
                json.dumps(
                    {
                        "format": "tt-finetune-training-metadata-v1",
                        "base_model_name_or_path": self.model.name_or_path,
                        "base_model_revision": self.model.revision,
                        "requested_memory_mode": self.args.memory_mode,
                        "selected_frozen_weight_dtype": memory_report.get(
                            "selected_weight_dtype", "bf16"
                        ),
                        "numerical_contract": memory_report.get(
                            "numerical_contract", "bf16_exact"
                        ),
                        "training_dtype": self.args.dtype,
                        "adapter_method": "+".join(adapter_features) or "lora",
                        "adapter_features": adapter_features,
                        "lora_plus_lr_ratio": lora_plus_ratio,
                        "layers_to_transform": (
                            list(adapter_config.layers_to_transform)
                            if adapter_config.layers_to_transform is not None
                            else None
                        ),
                        "layers_pattern": adapter_config.layers_pattern,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            self._engine._publish_execution_output()
            metrics_path = host_output / "training_metrics.json"
            adapter = host_output / "peft_adapter"
            self._engine.registry.record_hardware(
                report,
                registry_config,
                status="passed",
                metrics=metrics,
                adapter_path=adapter,
            )
        except Exception as error:
            self._engine.registry.record_hardware(
                report,
                registry_config,
                status="failed",
                error=error,
            )
            raise classify_runtime_error(error) from error
        artifact_report = pipeline.generated_dir / "DIRECT_ARTIFACTS.md"
        artifact_report.write_text(
            "# Direct tt-finetune artifacts\n\n"
            f"- engine fingerprint: `{self._direct_fingerprint()}`\n"
            f"- compatibility profile: `{profile_id}`\n"
            f"- compile cache ID: `{self._compile_cache_id()}`\n"
            f"- runtime mode: `{self.args.runtime_mode}`\n"
            f"- checkpoint: `{host_output / 'checkpoints/final_model.pth'}`\n"
            f"- PEFT adapter: `{adapter}`\n"
            f"- metrics: `{metrics_path}`\n",
            encoding="utf-8",
        )
        return TTTrainResult(
            run_name=self._engine.run_name,
            base_model=self.model.name_or_path,
            output_dir=host_output,
            adapter_dir=adapter,
            metrics_path=metrics_path,
            artifact_report=artifact_report,
            run_log=pipeline.run_log,
            preflight_report=report_path,
            artifacts={
                "status": "verified",
                "execution_engine": (
                    "tt_finetune.runtime.TrainingLoop"
                ),
                "engine_fingerprint": self._direct_fingerprint(),
                "profile_id": profile_id,
                "compile_cache_id": self._compile_cache_id(),
                "runtime_mode": self.args.runtime_mode,
            },
        )

    def smoke(self, steps: int = 2) -> TTTrainResult:
        suffix = "-smoke"
        output = self.args.output_dir
        smoke_output = output if output.endswith(suffix) else output + suffix
        smoke_model = self.model
        smoke_args = replace(
            self.args,
            output_dir=smoke_output,
            run_name=(
                self.args.run_name + suffix
                if self.args.run_name
                and not self.args.run_name.endswith(suffix)
                else self.args.run_name
            ),
            max_steps=steps,
            auto_smoke=False,
        )
        backend = type(self)(
            model=smoke_model,
            args=smoke_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_config=self.data_config,
            workspace=self.workspace,
        )
        return backend.train()
