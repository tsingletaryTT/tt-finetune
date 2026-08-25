"""Long-lived, single-card TT runtime service.

The service retains compiler/runtime caches while deliberately constructing a
fresh model, optimizer, dataset, RNG schedule, and trainer state for every job.
It is an execution host, not a training-state server.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import socketserver
import time
import traceback
from typing import Any

import yaml


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--info", type=Path, required=True)
    return parser.parse_args()


def _load_job_config(request: dict[str, Any]):
    from blacksmith.experiments.torch.qwen.configs import TrainingConfig

    raw = yaml.safe_load(
        Path(request["config"]).read_text(encoding="utf-8")
    )
    raw.update(request.get("config_overrides") or {})
    raw["project_dir"] = request["output_dir"]
    raw["wandb_project"] = Path(request["output_dir"]).name
    raw["wandb_run_name"] = Path(request["output_dir"]).name
    if request.get("max_steps") is not None:
        raw["max_steps"] = int(request["max_steps"])
    if request.get("no_eval", False):
        raw["do_eval"] = False
        raw["run_final_validation"] = False
        raw["eval_files"] = []
    resume = request.get("resume_from_checkpoint")
    if resume:
        raw["resume_from_checkpoint"] = True
        if resume is True or resume == "last":
            raw["resume_option"] = "last"
        else:
            raw["resume_option"] = "path"
            raw["checkpoint_path"] = str(resume)
    raw["measure_e2e_time"] = True
    return TrainingConfig(**raw)


class RuntimeState:
    def __init__(self, fingerprint: str, info_path: Path):
        self.fingerprint = fingerprint
        self.info_path = info_path
        self.started_at = time.time()
        self.jobs_started = 0
        self.jobs_completed = 0
        self.cache_keys: set[str] = set()
        self.active_job: str | None = None
        self._write_info()

    def _write_info(self) -> None:
        self.info_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "pid": os.getpid(),
            "fingerprint": self.fingerprint,
            "started_at_unix": self.started_at,
            "jobs_started": self.jobs_started,
            "jobs_completed": self.jobs_completed,
            "cache_keys": sorted(self.cache_keys),
            "active_job": self.active_job,
            "training_state_retained": False,
        }
        temporary = self.info_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(self.info_path)

    def status(self) -> dict[str, Any]:
        return json.loads(self.info_path.read_text(encoding="utf-8"))

    def run_job(self, request: dict[str, Any]) -> dict[str, Any]:
        from .worker import run

        if request.get("engine_fingerprint") != self.fingerprint:
            raise RuntimeError(
                "persistent runtime source fingerprint mismatch"
            )
        job_id = str(request["job_id"])
        cache_key = str(request["cache_key"])
        cache_status = "hit" if cache_key in self.cache_keys else "miss"
        self.jobs_started += 1
        self.active_job = job_id
        self._write_info()
        output = Path(request["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "persistent_runtime.log"
        started = time.perf_counter()
        try:
            config = _load_job_config(request)
            run_options = dict(request.get("run_options") or {})
            if run_options.get("xla_cache") is not None:
                run_options["xla_cache"] = Path(run_options["xla_cache"])
            run_options["service_context"] = {
                "job_id": job_id,
                "cache_key": cache_key,
                "cache_status": cache_status,
                "service_pid": os.getpid(),
                "service_started_at_unix": self.started_at,
                "fresh_training_state": True,
                "prior_jobs_completed": self.jobs_completed,
            }
            with log_path.open("w", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    metrics = run(config, **run_options)
            self.cache_keys.add(cache_key)
            self.jobs_completed += 1
            return {
                "status": "completed",
                "job_id": job_id,
                "cache_key": cache_key,
                "cache_status": cache_status,
                "service_pid": os.getpid(),
                "service_elapsed_seconds": time.perf_counter() - started,
                "metrics_path": str(output / "training_metrics.json"),
                "global_steps": metrics.get("global_steps"),
            }
        finally:
            # Every reference to job-owned model/optimizer/RNG/data state is
            # local to worker.run.  Collection here is explicit; executable
            # caches remain process-owned inside torch/torch-xla.
            self.active_job = None
            gc.collect()
            self._write_info()


class RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            request = json.loads(self.rfile.readline())
            action = request.get("action", "train")
            state: RuntimeState = self.server.runtime_state  # type: ignore[attr-defined]
            if action == "status":
                response = {"status": "ready", **state.status()}
            elif action == "train":
                response = state.run_job(request)
            else:
                raise ValueError(f"unsupported service action: {action!r}")
        except Exception as error:
            response = {
                "status": "failed",
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))


class UnixRuntimeServer(socketserver.UnixStreamServer):
    allow_reuse_address = True


def main() -> None:
    arguments = _arguments()
    arguments.socket.parent.mkdir(parents=True, exist_ok=True)
    arguments.info.parent.mkdir(parents=True, exist_ok=True)
    arguments.socket.unlink(missing_ok=True)
    state = RuntimeState(arguments.fingerprint, arguments.info)
    with UnixRuntimeServer(str(arguments.socket), RequestHandler) as server:
        server.runtime_state = state  # type: ignore[attr-defined]
        os.chmod(arguments.socket, 0o666)
        server.serve_forever(poll_interval=0.25)


if __name__ == "__main__":
    main()
