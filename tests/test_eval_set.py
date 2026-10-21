"""Fixed evaluation set: whole batches, stable membership, legacy compatibility."""

import time

import numpy as np
from PIL import Image
import pytest

from recog.store import Store

CHAR = "洛天依"


def add(store, seed, label, style="", size=(120, 96), neg_type=""):
    """Upload one noise image. Size is explicit because batch identity is (w, h)."""
    width, height = size
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    path = store.root / "uploads" / f"{seed}-{time.time_ns()}.png"
    Image.fromarray(pixels).save(path)
    return store.ingest(path, label, CHAR, style, neg_type)


def batch_key(row):
    return (row["label"], row["style"] if row["label"] == "self" else row["neg_type"], row["width"], row["height"])


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
    """Batches small enough to fit the per-style budget, so isolation is exact."""
    seed = 1000
    for size in ((100, 100), (110, 110), (120, 120)):
        for _ in range(12):
            add(store, seed, "self", "flat", size)
            seed += 1
    for size in ((300, 400), (310, 410), (320, 420)):
        for _ in range(10):
            add(store, seed, "self", "paint", size)
            seed += 1
    for size in ((140, 90), (150, 100), (160, 110)):
        for _ in range(12):
            add(store, seed, "other", "photo", size, neg_type="non_character")
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


def test_eval_takes_at_most_the_per_style_budget(tmp_path):
    store = Store(tmp_path)
    seed_corpus(store)
    freeze(store)
    coverage = {row["style"]: row for row in store.style_coverage(CHAR)}
    assert coverage["flat"]["eval"] == 24  # 12 + 12, the third batch would exceed 25
    assert coverage["paint"]["eval"] == 20  # 10 + 10
    # Leftovers are then randomly split into train/val/test, so 12 - 4 = 8 train.
    assert coverage["flat"]["train"] + coverage["flat"]["other"] == 12
    assert coverage["paint"]["train"] + coverage["paint"]["other"] == 10


def test_bucket_smaller_than_target_goes_entirely_to_eval(tmp_path):
    store = Store(tmp_path)
    for index in range(20):
        add(store, 1000 + index, "self", "flat", (100, 100))
    for index in range(20):
        add(store, 2000 + index, "other", "photo", (120, 90), neg_type="non_character")
    for index in range(3):
        add(store, 3000 + index, "self", "paint", (300, 400))
    freeze(store)
    coverage = {row["style"]: row for row in store.style_coverage(CHAR)}
    assert coverage["paint"]["eval"] == 3
    assert coverage["paint"]["train"] == 0


def test_single_dominant_batch_is_sampled_and_reported(tmp_path):
    """One batch bigger than the budget cannot be isolated; say so instead of lying."""
    store = Store(tmp_path)
    for index in range(60):
        add(store, 1000 + index, "self", "flat", (100, 100))
    for index in range(40):
        add(store, 2000 + index, "other", "photo", (120, 90), neg_type="non_character")
    snapshot = freeze(store)
    unisolated = snapshot["eval_build"]["unisolated"]
    assert {entry["label"] for entry in unisolated} == {"self", "other"}
    coverage = {row["style"]: row for row in store.style_coverage(CHAR)}
    assert coverage["flat"]["eval"] == 25
    # The other 35 went to train/val/test, so only 25 of them stay in "train".
    assert coverage["flat"]["train"] + coverage["flat"]["other"] == 35


def test_eval_membership_is_stable_across_runs(tmp_path):
    store = Store(tmp_path)
    seed_corpus(store)
    freeze(store, "first")
    first = {row["id"]: row["split"] for row in rows(store) if row["split"] == "eval"}
    freeze(store, "second")
    second = {row["id"]: row["split"] for row in rows(store) if row["split"] == "eval"}
    assert first and first == second


def test_untagged_library_gets_no_eval_set(tmp_path):
    """A 0.5.x library has no style tags, so nothing is moved into eval."""
    store = Store(tmp_path)
    for index in range(20):
        add(store, 1000 + index, "self")
    for index in range(20):
        add(store, 2000 + index, "other")
    snapshot = freeze(store)
    assert {row["split"] for row in snapshot["images"]} <= {"train", "val", "test"}
    assert snapshot["eval_build"]["built"] is False


def test_legacy_rows_keep_empty_style_after_migration(tmp_path):
    store = Store(tmp_path)
    add(store, 1, "self", "flat")
    with store.db() as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(images)")}
        assert {"style", "neg_type"} <= columns
        # Simulate a pre-0.5.12 row by clearing both new columns.
        db.execute("UPDATE images SET style='',neg_type=''")
    listed = store.images(character=CHAR)["rows"]
    assert listed[0]["style"] == ""
    assert listed[0]["style_text"] == "未标注类型"


def test_style_and_neg_type_validation(tmp_path):
    store = Store(tmp_path)
    result = add(store, 1, "self", "flat")
    with pytest.raises(ValueError):
        store.set_style(result["id"], "official")
    with pytest.raises(ValueError):
        store.set_style(result["id"], "flat", "non_character")  # positives cannot carry a neg bucket
    negative = add(store, 2, "other", "photo", neg_type="non_character")
    store.set_style(negative["id"], "photo", "sticker")
    listed = {row["id"]: row for row in store.images(character=CHAR)["rows"]}
    assert listed[negative["id"]]["neg_type"] == "sticker"


def test_batch_style_rejects_mixed_labels_for_negative_bucket(tmp_path):
    store = Store(tmp_path)
    positive = add(store, 1, "self", "flat")
    negative = add(store, 2, "other", "photo", neg_type="non_character")
    with pytest.raises(ValueError):
        store.batch_style(f"{positive['id']},{negative['id']}", "flat", "sticker")
    store.batch_style(f"{positive['id']},{negative['id']}", "paint")
    listed = {row["id"]: row for row in store.images(character=CHAR)["rows"]}
    assert listed[positive["id"]]["style"] == "paint"
    assert listed[negative["id"]]["style"] == "paint"


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
    # Production calls both with defaults; a custom budget would describe a
    # different set than the one the dialog promised.
    store.rebuild_eval(CHAR)
    first = {row["id"] for row in rows(store) if row["split"] == "eval"}
    store.rebuild_eval(CHAR)
    second = {row["id"] for row in rows(store) if row["split"] == "eval"}
    assert first and first == second
    # Same seed and same data => the plan already described this set.
    assert plan["groups"] == len(first)


def test_eval_plan_does_not_touch_splits(tmp_path):
    store = Store(tmp_path)
    seed_corpus(store)
    freeze(store)
    before = {row["id"]: row["split"] for row in rows(store)}
    store.eval_plan(CHAR)
    after = {row["id"]: row["split"] for row in rows(store)}
    assert before == after


def test_legacy_library_without_eval_offers_a_plan(tmp_path):
    """A 0.5.x library has every group assigned, so no eval set; the plan must say
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
    assert sum(plan["by_style"].values()) + sum(plan["by_neg"].values()) == plan["groups"]


def test_readiness_reports_coverage(tmp_path):
    store = Store(tmp_path)
    seed_corpus(store)
    readiness = store.training_readiness(CHAR)
    assert {row["style"] for row in readiness["coverage"]} == {"flat", "paint", "photo", "composite", ""}
    # 实拍 only exists as negatives here, so it is missing as a positive medium.
    assert readiness["coverage_missing"] == ["实拍", "图文/梗图"]
    assert "平涂/Q版" in readiness["coverage_text"]
