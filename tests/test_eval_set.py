"""Fixed evaluation set: whole batches, stable membership, per-label budget.

0.6.0 removed the medium / text / negative-identity axes, so the eval budget is
per label instead of per bucket. The invariant that still matters is unchanged:
a (width, height) batch is a proxy for one source, and a batch never straddles
eval and the training side.
"""

import time

import numpy as np
from PIL import Image
import pytest

from recog.store import EVAL_PER_LABEL, Store

CHAR = "洛天依"


def add(store, seed, label, size=(120, 96)):
    """Upload one noise image. Size is explicit because batch identity is (w, h)."""
    width, height = size
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    path = store.root / "uploads" / f"{seed}-{time.time_ns()}.png"
    Image.fromarray(pixels).save(path)
    return store.ingest(path, label, CHAR)


def batch_key(row):
    return (row["label"], row["width"], row["height"])


def rows(store):
    with store.db() as db:
        return [
            dict(row)
            for row in db.execute(
                "SELECT i.*,g.split FROM images i JOIN groups g ON g.id=i.group_id "
                "WHERE i.character=? AND i.removed=0 AND i.reason='' ORDER BY i.id",
                (CHAR,),
            )
        ]


def freeze(store, job="job"):
    with store.db() as db:
        return store.freeze(db, job, "fresh", "cpu", CHAR)


def seed_corpus(store):
    """Three batches per label, ten images each: one whole batch fits the budget."""
    seed = 1000
    for size in ((100, 100), (110, 110), (120, 120)):
        for _ in range(10):
            add(store, seed, "self", size)
            seed += 1
    for size in ((140, 90), (150, 100), (160, 110)):
        for _ in range(10):
            add(store, seed, "other", size)
            seed += 1


def test_whole_batch_never_crosses_train_and_eval(tmp_path):
    store = Store(tmp_path)
    seed_corpus(store)
    freeze(store)
    batches: dict = {}
    for row in rows(store):
        batches.setdefault(batch_key(row), set()).add(row["split"])
    # train/val/test mixing inside a batch is the normal random split; the
    # invariant that matters is that no batch is split between eval and anything.
    assert all("eval" not in splits or splits == {"eval"} for splits in batches.values()), batches
    assert "eval" in {row["split"] for row in rows(store)}


def test_eval_takes_at_most_the_per_label_budget(tmp_path):
    store = Store(tmp_path)
    seed_corpus(store)
    freeze(store)
    splits = [(row["label"], row["split"]) for row in rows(store)]
    # 30 per label, 20 reserved for training => 10 in eval, one whole batch.
    assert splits.count(("self", "eval")) == 10
    assert splits.count(("other", "eval")) == 10
    assert store.eval_groups(CHAR) == 20
    assert splits.count(("self", "train")) + splits.count(("self", "val")) + splits.count(("self", "test")) == 20


def test_budget_never_eats_the_training_reserve(tmp_path):
    """Exactly 20 groups per label: the reserve wins, so no eval set is built."""
    store = Store(tmp_path)
    for index in range(20):
        add(store, 1000 + index, "self", (100, 100))
    for index in range(20):
        add(store, 2000 + index, "other", (120, 90))
    snapshot = freeze(store)
    assert store.eval_groups(CHAR) == 0
    assert snapshot["eval_build"]["built"] is False
    assert {row["split"] for row in snapshot["images"]} <= {"train", "val", "test"}


def test_single_dominant_batch_is_sampled_and_reported(tmp_path):
    """One batch bigger than the budget cannot be isolated; say so instead of lying."""
    store = Store(tmp_path)
    for index in range(60):
        add(store, 1000 + index, "self", (100, 100))
    for index in range(40):
        add(store, 2000 + index, "other", (120, 90))
    snapshot = freeze(store)
    unisolated = snapshot["eval_build"]["unisolated"]
    assert {entry["label"] for entry in unisolated} == {"self", "other"}
    assert all(entry["size"] == 60 or entry["size"] == 40 for entry in unisolated)
    splits = [(row["label"], row["split"]) for row in rows(store)]
    # self: 60 - 20 reserve = 40 available, capped by EVAL_PER_LABEL = 25.
    assert splits.count(("self", "eval")) == EVAL_PER_LABEL
    # other: 40 - 20 reserve = 20 available, below the cap.
    assert splits.count(("other", "eval")) == 20


def test_eval_membership_is_stable_across_runs(tmp_path):
    store = Store(tmp_path)
    seed_corpus(store)
    freeze(store, "first")
    first = {row["id"]: row["split"] for row in rows(store) if row["split"] == "eval"}
    freeze(store, "second")
    second = {row["id"]: row["split"] for row in rows(store) if row["split"] == "eval"}
    assert first and first == second


def test_eval_set_builds_without_any_annotation(tmp_path):
    """0.6.0: nothing has to be tagged any more, so the eval set always builds."""
    store = Store(tmp_path)
    seed_corpus(store)
    snapshot = freeze(store)
    assert snapshot["eval_build"]["built"] is True
    assert store.eval_groups(CHAR) == 20


def test_removed_annotation_columns_are_gone(tmp_path):
    store = Store(tmp_path)
    add(store, 1, "self")
    with store.db() as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(images)")}
    assert "style" not in columns and "neg_type" not in columns and "has_text" not in columns


OLD_SCHEMA = """
CREATE TABLE images(id TEXT PRIMARY KEY, path TEXT, sha TEXT, decoded TEXT, phash TEXT, aspect REAL,
  width INTEGER, height INTEGER, group_id TEXT, label TEXT, removed INTEGER DEFAULT 0,
  reason TEXT DEFAULT '', created REAL, character TEXT NOT NULL DEFAULT '',
  style TEXT NOT NULL DEFAULT '', neg_type TEXT NOT NULL DEFAULT '', has_text TEXT NOT NULL DEFAULT '',
  source TEXT NOT NULL DEFAULT '', model_score REAL);
CREATE TABLE groups(id TEXT PRIMARY KEY, split TEXT);
CREATE TABLE jobs(id TEXT PRIMARY KEY, request_key TEXT UNIQUE, state TEXT, pid INTEGER, error TEXT,
  created REAL, snapshot_sha TEXT, character TEXT NOT NULL DEFAULT '', device TEXT NOT NULL DEFAULT '',
  input_size INTEGER);
CREATE TABLE models(id TEXT PRIMARY KEY, job_id TEXT, sha TEXT, metrics TEXT, created REAL, character TEXT NOT NULL DEFAULT '');
CREATE TABLE settings(key TEXT PRIMARY KEY,value TEXT);
"""


def test_upgrade_drops_the_columns_and_keeps_every_row(tmp_path):
    """0.5.x → 0.6.0 is a destructive migration on real libraries; prove the data survives."""
    import sqlite3

    root = tmp_path / "old"
    root.mkdir()
    with sqlite3.connect(root / "catalog.sqlite3") as db:
        db.executescript(OLD_SCHEMA)
        db.execute(
            "INSERT INTO images(id,path,sha,decoded,phash,aspect,width,height,group_id,label,removed,reason,"
            "created,character,style,neg_type,has_text,source,model_score) VALUES "
            "('a','images/a.png','s','d',1.0,1.25,100,80,'g','self',0,'',1.0,?, 'flat','','yes','webui',0.5)",
            (CHAR,),
        )
        db.execute("INSERT INTO groups VALUES ('g','train')")
        db.execute("INSERT INTO settings VALUES ('keep','me')")

    # Opening the plugin's store runs the migration.
    store = Store(root)
    with store.db() as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(images)")}
        saved = dict(db.execute("SELECT * FROM images").fetchone())
        setting = db.execute("SELECT value FROM settings WHERE key='keep'").fetchone()[0]
    assert "style" not in columns and "neg_type" not in columns and "has_text" not in columns
    assert saved["label"] == "self" and saved["character"] == CHAR
    assert saved["source"] == "webui" and saved["model_score"] == 0.5
    assert setting == "me"
    # Idempotent: the second open must not try to rebuild again.
    again = Store(root)
    with again.db() as db:
        assert "style" not in {row[1] for row in db.execute("PRAGMA table_info(images)")}
        assert db.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1


def test_rebuild_eval_is_idempotent_and_matches_the_plan(tmp_path):
    """The confirmation dialog promises an exact cost, so the pick must be repeatable."""
    store = Store(tmp_path)
    seed_corpus(store)
    freeze(store)
    plan = store.eval_plan(CHAR)
    assert plan["groups"] > 0
    # Unchanged data => the plan re-picks the current eval set, so it costs no
    # training images at all. That is the property the dialog relies on.
    assert plan["from_train"] == 0
    assert plan["from_other"] == plan["groups"]
    store.rebuild_eval(CHAR)
    first = {row["id"] for row in rows(store) if row["split"] == "eval"}
    store.rebuild_eval(CHAR)
    second = {row["id"] for row in rows(store) if row["split"] == "eval"}
    assert first and first == second
    assert plan["groups"] == len(first)


def test_rebuild_rejects_an_absurd_budget(tmp_path):
    store = Store(tmp_path)
    seed_corpus(store)
    with pytest.raises(ValueError):
        store.rebuild_eval(CHAR, per_label=4)
    with pytest.raises(ValueError):
        store.rebuild_eval(CHAR, per_label=201)


def test_eval_plan_does_not_touch_splits(tmp_path):
    store = Store(tmp_path)
    seed_corpus(store)
    freeze(store)
    before = {row["id"]: row["split"] for row in rows(store)}
    store.eval_plan(CHAR)
    after = {row["id"]: row["split"] for row in rows(store)}
    assert before == after


def test_legacy_library_without_eval_offers_a_plan(tmp_path):
    """A library whose groups are all assigned gets no eval set; the plan must say
    what rebuilding would cost instead of asking the user to tag more images."""
    store = Store(tmp_path)
    seed_corpus(store)
    freeze(store)
    with store.db() as db:
        db.execute("UPDATE groups SET split='train' WHERE split='eval'")
    readiness = store.training_readiness(CHAR)
    assert readiness["eval_groups"] == 0
    plan = readiness["eval_plan"]
    assert plan["groups"] > 0
    assert plan["from_train"] > 0


def test_readiness_reports_the_eval_group_count(tmp_path):
    store = Store(tmp_path)
    seed_corpus(store)
    readiness = store.training_readiness(CHAR)
    assert readiness["eval_groups"] == 0
    assert "coverage" not in readiness and "coverage_text" not in readiness
    freeze(store)
    assert store.training_readiness(CHAR)["eval_groups"] == 20
