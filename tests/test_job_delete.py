"""Manual task-record cleanup keeps the model's report and snapshot."""

import json
import sys
import time

import pytest

from recog.service import Service
from recog.store import Store

CHAR = "诗岸"
JOB = "b" * 32
SNAPSHOT = {
    "character": CHAR,
    "device": "cpu",
    "input_size": 320,
    "images": [{"id": "x" * 32, "split": "train"}],
    "trained_sources": [{"id": "x" * 32, "label": "self", "sha": "s", "decoded": "d", "group_id": "g"}],
}
REPORT = {"character": CHAR, "test": {"f1": 0.9}, "historical_test": True}


def add_job(service, state="complete", model=False):
    service.store.root.joinpath("jobs", JOB).mkdir(parents=True)
    directory = service.store.root / "jobs" / JOB
    (directory / "snapshot.json").write_text(json.dumps(SNAPSHOT), encoding="utf-8")
    (directory / "report.json").write_text(json.dumps(REPORT), encoding="utf-8")
    (directory / "resume.pt").write_bytes(b"checkpoint")
    with service.store.db() as db:
        db.execute(
            "INSERT INTO jobs (id,request_key,state,pid,error,created,snapshot_sha,character,device,input_size) "
            "VALUES (?,?,?,NULL,'',?,?,?,?,?)",
            (JOB, "k" * 8, state, time.time(), "sha", CHAR, "cpu", 320),
        )
        if model:
            (service.store.root / "models" / JOB).mkdir(parents=True)
            (service.store.root / "models" / JOB / "weights.pt").write_bytes(b"weights")
            db.execute(
                "INSERT INTO models VALUES (?,?,?,?,?,?)",
                (JOB, JOB, "w", json.dumps({"test": {}}), time.time(), CHAR),
            )


@pytest.fixture
def service(tmp_path):
    instance = Service(tmp_path, python=sys.executable)
    try:
        yield instance
    finally:
        instance.close()


def test_completed_job_is_removed_but_model_report_survives(service):
    add_job(service, model=True)
    result = service.delete_jobs([JOB])
    assert result["deleted"] == 1
    assert result["archived"] == 2
    store = service.store
    with store.db() as db:
        assert db.execute("SELECT 1 FROM jobs WHERE id=?", (JOB,)).fetchone() is None
        assert db.execute("SELECT 1 FROM models WHERE id=?", (JOB,)).fetchone() is not None
    assert not (store.root / "jobs" / JOB).exists()
    assert (store.root / "models" / JOB / "weights.pt").exists()
    assert json.loads((store.root / "models" / JOB / "report.json").read_text("utf-8")) == REPORT
    assert store.report(JOB) == REPORT
    with store.db() as db:
        assert store.trained_sources(db, JOB) == SNAPSHOT["trained_sources"]
    row = store.job_or_model(JOB)
    assert row["state"] == "complete" and row["character"] == CHAR


def test_failed_job_cleans_up_without_a_model(service):
    add_job(service, state="failed")
    assert service.delete_jobs([JOB])["deleted"] == 1
    assert not (service.store.root / "jobs" / JOB).exists()
    assert service.store.models() == []


def test_active_job_is_refused(service):
    add_job(service, state="running")
    with pytest.raises(ValueError, match="运行中"):
        service.delete_jobs([JOB])
    assert (service.store.root / "jobs" / JOB / "snapshot.json").exists()
    assert service.store.job(JOB)["state"] == "running"


def test_paused_job_can_be_deleted(service):
    add_job(service, state="paused")
    assert service.delete_jobs([JOB])["deleted"] == 1
    assert not (service.store.root / "jobs" / JOB).exists()


def test_unknown_and_malformed_ids_are_refused(service):
    add_job(service, model=True)
    with pytest.raises(ValueError, match="不存在"):
        service.delete_jobs(["c" * 32])
    with pytest.raises(ValueError, match="无效"):
        service.delete_jobs(["../etc"])
    with pytest.raises(ValueError, match="最多"):
        service.delete_jobs([f"{index:032x}" for index in range(51)])
    assert (service.store.root / "jobs" / JOB / "snapshot.json").exists()


def test_delete_action_is_declared_on_the_training_page():
    from pathlib import Path

    data = json.loads((Path(__file__).resolve().parents[1] / "webui.json").read_text(encoding="utf-8"))
    training = next(page for page in data["pages"] if page["id"] == "training")
    action = training["actions"]["delete_jobs"]
    assert action["api"] == "self_training_delete"
    assert action["parameters"]["jobs"]["required"] is True


def test_ui_filters_are_validated(tmp_path):
    store = Store(tmp_path)
    assert store.set_ui_state({"job_scope": "all", "job_state": "failed"}) == {
        "job_scope": "all",
        "job_state": "failed",
    }
    assert store.set_ui_state({"job_scope": "some", "job_state": "running"}) == {
        "job_scope": "all",
        "job_state": "failed",
    }
