"""Real default 15+10 epoch CPU workers and atomic activation/rollback.

需要 PyTorch（训练与推理 worker 的真实环境）：缺少 torch/torchvision 时整体跳过，
不阻塞 `python -m pytest tests`。安装方式见 README「运行环境」。
"""

from pathlib import Path
import json
import sys
import time

import pytest

pytest.importorskip("torch", reason="CPU 端到端验收需要 PyTorch")
pytest.importorskip("torchvision", reason="CPU 端到端验收需要 torchvision")

from recog.service import Service  # noqa: E402
from recog.store import atomic_json  # noqa: E402
from test_catalog import CHAR, corpus, upload  # noqa: E402


def wait(service, job, timeout=240):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        row = service.store.job(job)
        if row["state"] in {"complete", "failed", "cancelled"}:
            assert row["state"] == "complete", row
            return service.store.report(job)
        time.sleep(0.5)
    raise AssertionError("CPU training timeout")


def served(service, raw):
    """单请求多模型协议：一次推理返回全部已启用角色，取本次用例的角色。"""
    result = service.infer(raw)
    assert result["status"] == "ok", result
    return next(item for item in result["results"] if item["character"] == CHAR)


def test_cpu_acceptance(tmp_path):
    store = corpus(tmp_path)
    service = Service(tmp_path, sys.executable)
    try:
        first = service.submit("initial", "fresh", character=CHAR, input_size=224)
        assert service.submit("initial", character=CHAR)["id"] == first["id"]
        service.pause(first["id"])
        deadline = time.monotonic() + 60
        while service.store.job(first["id"])["state"] == "pausing" and time.monotonic() < deadline:
            time.sleep(.25)
        assert service.store.job(first["id"])["state"] == "paused"
        assert (tmp_path / "jobs" / first["id"] / "resume.pt").exists()
        while any(p.poll() is None for p in service.processes.values()) and time.monotonic() < deadline:
            time.sleep(.25)
        assert service.resume(first["id"])["id"] == first["id"]
        report = wait(service, first["id"])
        assert len(report["history"]) == 15 and report["audit"]["passed"]
        assert report["counts"] == dict(train=28, val=6, test=6)
        assert report["input_size"] == 224
        assert report["peak_rss_bytes"] > 0
        assert service.activate(CHAR, first["id"])["status"] == "ready"
        path = store.root / store.images(character=CHAR)["rows"][0]["path"]
        hit = served(service, path.read_bytes())
        assert hit["version"] == first["id"] and 0 <= hit["score"] <= 1
        assert hit["score_is_calibrated_probability"] is False
        assert hit["input_size"] == 224
        for i in range(40, 46):
            upload(store, i, "self" if i < 43 else "other")
        second = service.submit("continuation", character=CHAR, input_size=448)
        snapshot = json.loads((tmp_path / "jobs" / second["id"] / "snapshot.json").read_text("utf-8"))
        assert snapshot["parent"] == first["id"] and snapshot["mode"] == "incremental"
        assert {r["id"] for r in snapshot["images"] if r["split"] == "train"}.isdisjoint({
            r["id"]
            for r in json.loads((tmp_path / "jobs" / first["id"] / "snapshot.json").read_text("utf-8"))["images"]
            if r["split"] == "train"
        })
        assert len(snapshot["trained_sources"]) == 34
        report2 = wait(service, second["id"])
        assert len(report2["history"]) == 10 and report2["historical_test"]
        assert report2["counts"]["train"] == 6
        assert service.activate(CHAR, second["id"])["status"] == "ready"
        prediction = served(service, path.read_bytes())
        assert prediction["version"] == second["id"] and prediction["input_size"] == 448
        assert service.rollback(CHAR)["version"] == first["id"]
        assert served(service, path.read_bytes())["input_size"] == 224
        assert served(service, path.read_bytes())["version"] == first["id"]
        # Tampering rejects activation and leaves old serving model intact.
        weights = tmp_path / "models" / second["id"] / "weights.pt"
        with weights.open("ab") as stream:
            stream.write(b"tamper")
        with pytest.raises(ValueError):
            service.activate(CHAR, second["id"])
        assert store.active(CHAR) == first["id"]
        results = {
            "fixture": "40 synthetic images; validates mechanics, not identity accuracy",
            "fresh": report,
            "incremental": report2,
        }
        atomic_json(Path(__file__).parents[1] / "test-results" / "cpu-acceptance.json", results)
    finally:
        service.close()
