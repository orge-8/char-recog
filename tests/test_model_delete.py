"""Model version cleanup: hard delete with guards for serving and training."""

import json
import sys
import time

import pytest

from recog.service import Service

CHAR = "诗岸"
VERSION = "a" * 32
OTHER = "b" * 32


def add_model(service, version, *, active=False, job_state=None, parent=""):
    directory = service.store.root / "models" / version
    directory.mkdir(parents=True)
    (directory / "weights.pt").write_bytes(b"w" * 6)
    (directory / "audit.json").write_text('{"sha":"x"}', encoding="utf-8")
    (directory / "report.json").write_text('{"archived":true}', encoding="utf-8")
    snapshot = {"character": CHAR, "parent": parent, "device": "cpu", "input_size": 320, "images": []}
    if job_state:
        jobs = service.store.root / "jobs" / version
        jobs.mkdir(parents=True)
        (jobs / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
        (jobs / "report.json").write_text("{}", encoding="utf-8")
    with service.store.db() as db:
        db.execute(
            "INSERT INTO models VALUES (?,?,?,?,?,?)",
            (version, version, "sha", json.dumps({"validation": {}, "test": {}, "input_size": 320}), time.time(), CHAR),
        )
        if job_state:
            db.execute(
                "INSERT INTO jobs (id,request_key,state,pid,error,created,snapshot_sha,character,device,input_size) "
                "VALUES (?,?,?,NULL,'',?,?,?,?,?)",
                (version, version, job_state, time.time(), "sha", CHAR, "cpu", 320),
            )
    if active:
        service.store.set_active(CHAR, version)


@pytest.fixture
def service(tmp_path):
    instance = Service(tmp_path, python=sys.executable)
    try:
        yield instance
    finally:
        instance.close()


def test_version_files_and_row_are_removed_but_job_record_survives(service):
    add_model(service, VERSION, job_state="complete")
    expected = service.store.model_size(VERSION)
    assert expected > 0
    assert service.delete_versions([VERSION]) == {"deleted": 1, "failed": [], "released": expected}
    assert not (service.store.root / "models" / VERSION).exists()
    assert service.store.models() == []
    with service.store.db() as db:
        assert db.execute("SELECT state FROM jobs WHERE id=?", (VERSION,)).fetchone()[0] == "complete"


def test_active_version_refuses_the_whole_batch(service):
    add_model(service, VERSION, active=True)
    add_model(service, OTHER)
    with pytest.raises(ValueError, match="使用中"):
        service.delete_versions([VERSION, OTHER])
    for version in (VERSION, OTHER):
        assert (service.store.root / "models" / version / "weights.pt").exists()
    assert len(service.store.models()) == 2
    assert service.store.active(CHAR) == VERSION


def test_parent_of_a_paused_job_is_refused(service):
    add_model(service, VERSION)
    add_model(service, OTHER, job_state="paused", parent=VERSION)
    with pytest.raises(ValueError, match="父模型"):
        service.delete_versions([VERSION])
    assert (service.store.root / "models" / VERSION / "weights.pt").exists()


def test_completed_job_does_not_block_its_parent(service):
    add_model(service, VERSION)
    add_model(service, OTHER, job_state="complete", parent=VERSION)
    assert service.delete_versions([VERSION])["deleted"] == 1
    assert not (service.store.root / "models" / VERSION).exists()


def test_deleted_rollback_target_is_forgotten(service):
    add_model(service, VERSION, active=True)
    add_model(service, OTHER)
    service.store.set_setting(f"previous:{CHAR}", OTHER)
    assert service.delete_versions([OTHER])["deleted"] == 1
    assert service.store.previous(CHAR) == ""
    with pytest.raises(ValueError, match="无可回退版本"):
        service.rollback(CHAR)


def test_unknown_and_malformed_ids_are_refused(service):
    add_model(service, VERSION)
    with pytest.raises(ValueError, match="不存在"):
        service.delete_versions(["c" * 32])
    with pytest.raises(ValueError, match="无效"):
        service.delete_versions(["../etc"])
    with pytest.raises(ValueError, match="最多"):
        service.delete_versions([f"{index:032x}" for index in range(51)])
    assert (service.store.root / "models" / VERSION / "weights.pt").exists()


def test_delete_action_is_declared_on_the_models_page():
    from pathlib import Path

    data = json.loads((Path(__file__).resolve().parents[1] / "webui.json").read_text(encoding="utf-8"))
    models = next(page for page in data["pages"] if page["id"] == "models")
    action = models["actions"]["delete_versions"]
    assert action["api"] == "self_models_delete"
    assert action["parameters"]["versions"]["required"] is True
