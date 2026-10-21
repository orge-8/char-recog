"""Inference lifecycle: cold start, breaker, manual stop, thread negotiation.

Every test drives a real child process, so these run without PyTorch installed.
"""

from pathlib import Path
import subprocess
import sys
import time

import pytest

from recog.service import Service


def worker(service: Service, code: str):
    with (service.store.root / "inference.log").open("a", encoding="utf-8") as log:
        return subprocess.Popen(
            [sys.executable, "-u", "-c", code], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8",
        )


ECHO = 'import sys\nfor line in sys.stdin: print(\'{"status":"ready"}\', flush=True)\n'


def echo_service(tmp_path: Path, monkeypatch):
    """A worker that answers every line, so warm-up reaches `ready` instantly."""
    service = Service(tmp_path, sys.executable)
    processes = []

    def spawn(*args, **kwargs):
        process = worker(service, ECHO)
        processes.append(process)
        return process

    monkeypatch.setattr(service, "_spawn", spawn)
    return service, processes


def test_worker_exit_reports_current_error_and_reaps_process(tmp_path: Path, monkeypatch):
    service = Service(tmp_path, sys.executable)
    (tmp_path / "inference.log").write_text("OLD ERROR\n", encoding="utf-8")
    processes = []

    def spawn(*args, **kwargs):
        process = worker(service, "import sys;print(\"ModuleNotFoundError: No module named 'torch'\",file=sys.stderr);sys.exit(1)")
        processes.append(process)
        return process

    monkeypatch.setattr(service, "_spawn", spawn)
    try:
        result = service.request({"operation": "load"}, timeout=2)
        # A dead worker now opens the breaker instead of disabling recognition.
        assert result["status"] == "cooling"
        assert result["error"] == "未安装 PyTorch"
        assert service.inference_state() == "cooling"
        assert service.inference_enabled is True
        assert "OLD ERROR" not in result["error"]
        assert processes[0].poll() == 1
        assert service.inference is None
    finally:
        service.close()


def test_timeout_opens_breaker_instead_of_stopping(tmp_path: Path, monkeypatch):
    service = Service(tmp_path, sys.executable)
    processes = []

    def spawn(*args, **kwargs):
        # Answers the warm-up ping and the first real request, then goes silent.
        process = worker(
            service,
            "import sys,time\n"
            "for _ in range(2):\n"
            "    sys.stdin.readline()\n"
            "    print('{\"status\":\"ready\"}', flush=True)\n"
            "time.sleep(10)\n",
        )
        processes.append(process)
        return process

    monkeypatch.setattr(service, "_spawn", spawn)
    try:
        assert service.request({"operation": "load"}, timeout=3)["status"] == "ready"
        result = service.request({"operation": "load"}, timeout=0.2)
        assert result["status"] == "model_error"
        assert "推理响应超时" in result["error"]
        assert service.inference_state() == "cooling"
        assert service.inference_enabled is True
        assert processes[0].poll() is not None
        assert service.inference is None
        # Requests during the cooldown are distinguishable from a manual stop.
        assert service.request({"operation": "load"}, timeout=0.2)["status"] == "cooling"
    finally:
        service.close()


def test_breaker_expiry_respawns_on_next_request(tmp_path: Path, monkeypatch):
    service, processes = echo_service(tmp_path, monkeypatch)
    try:
        assert service.request({"operation": "load"}, timeout=3)["status"] == "ready"
        service._record_failure("测试用故障")
        assert service.inference_state() == "cooling"
        service._reap(service.inference)
        with service._state:
            service.breaker_until = 0.0
            service.retry_at = 0.0
        result = service.request({"operation": "load"}, timeout=3)
        assert result["status"] == "ready"
        assert service.inference_state() == "ready"
        assert len(processes) == 2
    finally:
        service.close()


def test_hook_path_never_blocks_on_cold_start(tmp_path: Path, monkeypatch):
    service = Service(tmp_path, sys.executable)
    processes = []

    def spawn(*args, **kwargs):
        process = worker(service, "import sys,time\ntime.sleep(5)\n")
        processes.append(process)
        return process

    monkeypatch.setattr(service, "_spawn", spawn)
    try:
        started = time.monotonic()
        result = service.request({"operation": "load"}, timeout=6, wait=False)
        elapsed = time.monotonic() - started
        assert elapsed < 0.5, "the chat path must not wait for a cold worker"
        assert result["status"] == "warming"
        assert service.breaker_until == 0.0
        for _ in range(100):
            if processes:
                break
            time.sleep(0.02)
        assert len(processes) == 1, "the lazy path still kicks the warm-up"
    finally:
        service.close()


def test_busy_does_not_trip_breaker(tmp_path: Path, monkeypatch):
    service, processes = echo_service(tmp_path, monkeypatch)
    try:
        assert service.request({"operation": "load"}, timeout=3)["status"] == "ready"
        assert service.infer_lock.acquire(timeout=1)
        try:
            result = service.request({"operation": "load"}, timeout=1)
        finally:
            service.infer_lock.release()
        assert result["status"] == "busy"
        assert service.breaker_until == 0.0
        assert service.inference_state() == "ready"
        assert processes[0].poll() is None
    finally:
        service.close()


def test_manual_stop_is_sticky_against_tick_and_time(tmp_path: Path, monkeypatch):
    service, processes = echo_service(tmp_path, monkeypatch)
    try:
        service.enable_auto()
        assert service.request({"operation": "load"}, timeout=3)["status"] == "ready"
        service.stop_inference()
        assert service.inference_state() == "stopped"
        for _ in range(3):
            service._tick_inference()
            time.sleep(0.2)
        assert len(processes) == 1, "an explicit stop must not be auto-recovered"
        assert service.request({"operation": "load"}, timeout=1)["status"] == "inference_stopped"
        assert service.request({"operation": "load"}, timeout=5, resume=True)["status"] == "ready"
        assert len(processes) == 2
    finally:
        service.close()
    reopened = Service(tmp_path, sys.executable)
    try:
        assert reopened.inference_enabled
    finally:
        reopened.close()


def test_warmup_failure_backs_off(tmp_path: Path, monkeypatch):
    service = Service(tmp_path, sys.executable)

    def spawn(*args, **kwargs):
        return worker(service, "import sys;sys.exit(1)")

    monkeypatch.setattr(service, "_spawn", spawn)

    def until_cooling(limit: float = 15.0) -> str:
        """失败是后台线程记的，机器忙时 2 秒未必够 —— 等状态落到 cooling，而不是等固定时长。"""
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            status = service.request({"operation": "load"}, timeout=2)["status"]
            if status != "warming":
                return status
            time.sleep(0.2)
        return status

    try:
        assert until_cooling() == "cooling"
        with service._state:
            service.breaker_until = 0.0
            service.retry_at = 0.0
        assert until_cooling() == "cooling"
        assert service.warm_failures == 2
        remaining = service.retry_at - time.monotonic()
        assert 60 <= remaining <= 110, remaining
    finally:
        service.close()


def test_threads_control_frame_follows_training(tmp_path: Path, monkeypatch):
    service = Service(tmp_path, sys.executable)
    trace = tmp_path / "frames.log"
    code = (
        "import json, sys\n"
        f"log = open({str(trace)!r}, 'a', encoding='utf-8')\n"
        "for line in sys.stdin:\n"
        "    log.write(line)\n"
        "    log.flush()\n"
        "    op = json.loads(line).get('operation')\n"
        "    print(json.dumps({'status': 'ok', 'threads': 1} if op == 'threads' else {'status': 'ready'}), flush=True)\n"
    )

    def spawn(*args, **kwargs):
        return worker(service, code)

    monkeypatch.setattr(service, "_spawn", spawn)
    try:
        service.enable_auto()
        assert service.request({"operation": "load"}, timeout=3)["status"] == "ready"
        service.mark_training(True)
        service._tick_inference()
        assert '"value": 1' in trace.read_text(encoding="utf-8")
        service.mark_training(False)
        service._tick_inference()
        assert '"value": 0' in trace.read_text(encoding="utf-8")
    finally:
        service.close()
