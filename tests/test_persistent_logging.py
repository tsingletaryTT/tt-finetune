import io
import json
import os
from pathlib import Path
import signal
import socket
import threading
from types import SimpleNamespace

import pytest

from blacksmith.tools.logging_manager import TrainingLogger
from tt_finetune.backends.direct import (
    DirectBlackholeBackend,
    _native_runtime_failure_diagnostic,
)
from tt_finetune.runtime_service import (
    _client_connection_is_open,
    _watch_training_client,
)


def _config():
    return SimpleNamespace(
        wandb_run_name="persistent-runtime-regression",
        log_level="INFO",
        use_wandb=False,
    )


def test_repeated_job_logger_does_not_propagate_to_stale_root_stream():
    first_stream = io.StringIO()
    first = TrainingLogger(_config())
    first.std_logger.handlers[0].setStream(first_stream)
    first.info("first job")
    first_stream.close()

    second_stream = io.StringIO()
    second = TrainingLogger(_config())
    second.std_logger.handlers[0].setStream(second_stream)
    second.info("second job")

    assert second.std_logger.propagate is False
    assert "second job" in second_stream.getvalue()


def test_idle_persistent_service_is_released_before_isolated_profile(
    monkeypatch, tmp_path
):
    socket_path = Path(tmp_path) / "runtime.sock"
    info_path = Path(tmp_path) / "runtime_info.json"
    socket_path.touch()
    info_path.touch()
    backend = object.__new__(DirectBlackholeBackend)
    calls = []

    def kill(pid, sig):
        calls.append((pid, sig))
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(os, "kill", kill)
    stopped = backend._stop_persistent_service(
        socket_path,
        {"pid": 4242, "active_job": None},
    )

    assert stopped is True
    assert calls[0] == (4242, signal.SIGTERM)
    assert not socket_path.exists()
    assert not info_path.exists()


def test_active_persistent_service_is_not_terminated(tmp_path):
    backend = object.__new__(DirectBlackholeBackend)
    with pytest.raises(RuntimeError, match="isolated device run"):
        backend._stop_persistent_service(
            Path(tmp_path) / "runtime.sock",
            {"pid": 4242, "active_job": "job-in-progress"},
        )


def test_exact_owned_active_job_can_be_terminated(monkeypatch, tmp_path):
    socket_path = Path(tmp_path) / "runtime.sock"
    info_path = Path(tmp_path) / "runtime_info.json"
    socket_path.touch()
    info_path.touch()
    backend = object.__new__(DirectBlackholeBackend)
    calls = []

    def kill(pid, sig):
        calls.append((pid, sig))
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(os, "kill", kill)
    stopped = backend._stop_persistent_service(
        socket_path,
        {"pid": 4242, "active_job": "owned-job"},
        expected_active_job="owned-job",
    )

    assert stopped is True
    assert calls[0] == (4242, signal.SIGTERM)


def test_service_status_falls_back_to_atomic_info_during_active_job(
    monkeypatch, tmp_path
):
    socket_path = Path(tmp_path) / "runtime.sock"
    info_path = Path(tmp_path) / "runtime_info.json"
    socket_path.touch()
    info_path.write_text(json.dumps({
        "pid": 4242,
        "fingerprint": "runtime",
        "active_job": "running",
    }))
    backend = object.__new__(DirectBlackholeBackend)

    def unavailable(*_args, **_kwargs):
        raise TimeoutError

    monkeypatch.setattr("tt_finetune.service_client.request", unavailable)
    monkeypatch.setattr(os, "kill", lambda _pid, _sig: None)

    status = backend._service_status(socket_path)

    assert status is not None
    assert status["active_job"] == "running"


def test_native_oom_diagnostic_reads_only_current_job_log_segment(tmp_path):
    log = tmp_path / "runtime_service.log"
    log.write_text(
        "old Out of Memory: Not enough space to allocate 8 B across 8 banks, "
        "where each bank needs to store 1 B, but bank size is 100 B "
        "(allocated: 90 B, free: 10 B, largest free block: 5 B)\n",
        encoding="utf-8",
    )
    offset = log.stat().st_size
    with log.open("a", encoding="utf-8") as destination:
        destination.write("current job completed without allocator failure\n")

    assert _native_runtime_failure_diagnostic(
        log, start_offset=offset
    ) is None

    offset = log.stat().st_size
    with log.open("a", encoding="utf-8") as destination:
        destination.write(
            "Out of Memory: Not enough space to allocate 1089994752 B "
            "across 8 banks, where each bank needs to store 136249344 B, "
            "but bank size is 4277141376 B (allocated: 3733921856 B, "
            "free: 543219520 B, largest free block: 135715712 B)\n"
        )

    diagnostic = _native_runtime_failure_diagnostic(
        log, start_offset=offset
    )
    assert diagnostic is not None
    assert diagnostic["kind"] == "device_dram_out_of_memory"
    assert diagnostic["allocator"]["request"] == 1089994752
    assert diagnostic["allocator"]["largest"] == 135715712


def test_training_client_disconnect_is_observable():
    service, client = socket.socketpair()
    try:
        assert _client_connection_is_open(service) is True
        client.close()
        assert _client_connection_is_open(service) is False
    finally:
        service.close()


def test_training_client_disconnect_terminates_orphaned_service(monkeypatch):
    service, client = socket.socketpair()
    stopped = threading.Event()
    terminated = threading.Event()

    monkeypatch.setattr(
        os,
        "kill",
        lambda pid, sig: terminated.set()
        if pid == os.getpid() and sig == signal.SIGTERM
        else None,
    )
    watcher = threading.Thread(
        target=_watch_training_client,
        args=(service, stopped),
        kwargs={"interval_seconds": 0.01},
    )
    watcher.start()
    client.close()
    try:
        assert terminated.wait(timeout=1.0)
    finally:
        stopped.set()
        watcher.join(timeout=1.0)
        service.close()
