"""jobs table carries device/input_size so polling stops reading snapshots."""

import json
import sqlite3
import sys
import time

import pytest

from recog.service import Service
from recog.store import Store

CHAR = "洛天依"

# Schema as shipped in 0.3.4: jobs has no device/input_size columns.
OLD_DDL = """
CREATE TABLE images(id TEXT PRIMARY KEY, path TEXT, sha TEXT, decoded TEXT, phash TEXT,
  aspect REAL, width INTEGER, height INTEGER, group_id TEXT, label TEXT,
  removed INTEGER DEFAULT 0, reason TEXT DEFAULT '', created REAL,
  character TEXT NOT NULL DEFAULT '');
CREATE TABLE groups(id TEXT PRIMARY KEY, split TEXT);
CREATE TABLE jobs(id TEXT PRIMARY KEY, request_key TEXT UNIQUE, state TEXT, pid INTEGER,
  error TEXT, created REAL, snapshot_sha TEXT, character TEXT NOT NULL DEFAULT '');
CREATE TABLE models(id TEXT PRIMARY KEY, job_id TEXT, sha TEXT, metrics TEXT, created REAL,
  character TEXT NOT NULL DEFAULT '');
CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT);
"""

JOB = "a" * 32


def old_database(root, snapshot):
    connection = sqlite3.connect(root / "catalog.sqlite3")
    try:
        connection.executescript(OLD_DDL)
        connection.execute(
            "INSERT INTO jobs VALUES (?,?, 'complete', NULL, '', ?, '', ?)",
            (JOB, "old-key", time.time(), CHAR),
        )
        connection.execute("INSERT INTO settings VALUES ('characters', '[]')")
        connection.commit()
    finally:
        connection.close()
    directory = root / "jobs" / JOB
    directory.mkdir(parents=True)
    (directory / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    (directory / "progress.json").write_text(json.dumps({"phase": "complete"}), encoding="utf-8")


def snapshot(device="cuda", size=448):
    return {"device": device, "input_size": size, "images": [{"id": "x" * 32}], "character": CHAR}


def test_schema_migrates_from_034_without_losing_rows(tmp_path):
    old_database(tmp_path, snapshot())
    store = Store(tmp_path)
    with store.db() as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
        count = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    assert {"device", "input_size"} <= columns
    assert count == 1
    assert store.job(JOB)["device"] == ""


def test_first_poll_backfills_then_stops_reading_snapshots(tmp_path, monkeypatch):
    old_database(tmp_path, snapshot())
    service = Service(tmp_path, python=sys.executable)
    try:
        rows = service.jobs()["rows"]
        assert [(row["device"], row["input_size"]) for row in rows] == [("cuda", 448)]
        assert rows[0]["device_text"] == "CUDA"
        # If the hot path still parsed snapshots this would explode.
        monkeypatch.setattr(
            Service, "_snapshot_meta", staticmethod(lambda path: (_ for _ in ()).throw(AssertionError("read snapshot")))
        )
        again = service.jobs()["rows"]
        assert [(row["device"], row["input_size"]) for row in again] == [("cuda", 448)]
    finally:
        service.close()


def test_backfill_is_persisted_even_after_the_snapshot_disappears(tmp_path):
    old_database(tmp_path, snapshot(device="cpu", size=320))
    service = Service(tmp_path, python=sys.executable)
    try:
        assert service.jobs()["rows"][0]["device"] == "cpu"
    finally:
        service.close()
    (tmp_path / "jobs" / JOB / "snapshot.json").unlink()
    store = Store(tmp_path)
    assert store.job(JOB)["device"] == "cpu"
    assert store.job(JOB)["input_size"] == 320
    service = Service(tmp_path, python=sys.executable)
    try:
        assert service.jobs()["rows"][0]["device"] == "cpu"
    finally:
        service.close()


def test_unreadable_snapshot_leaves_the_row_unknown(tmp_path):
    root = tmp_path
    old_database(root, snapshot())
    (root / "jobs" / JOB / "snapshot.json").unlink()
    service = Service(root, python=sys.executable)
    try:
        row = service.jobs()["rows"][0]
        assert row["device"] is None
        assert row["device_text"] == "设备未知"
    finally:
        service.close()


def test_submit_writes_device_and_size(tmp_path, monkeypatch):
    from test_catalog import corpus

    corpus(tmp_path)
    service = Service(tmp_path, python=str(tmp_path / "not-installed"))
    try:
        result = service.submit("key-1", device="cpu", character=CHAR, input_size=448)
        row = service.store.job(result["id"])
        assert row["device"] == "cpu"
        assert row["input_size"] == 448
        listed = service.jobs()["rows"][0]
        assert (listed["device"], listed["input_size"]) == ("cpu", 448)
    finally:
        service.close()


def test_snapshot_meta_rejects_unexpected_values(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"device": "tpu", "input_size": 320}), encoding="utf-8")
    assert Service._snapshot_meta(path) is None
    path.write_text(json.dumps({"device": "cpu", "input_size": 999}), encoding="utf-8")
    assert Service._snapshot_meta(path) == ("cpu", 320)
    path.write_text("not json", encoding="utf-8")
    assert Service._snapshot_meta(path) is None
