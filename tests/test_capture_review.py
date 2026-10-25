"""The 0.5.22 capture review queue: ingest gate, shared quota, review, cleanup.

Two invariants run through the file:
- a chat capture only enters the library if it can become a group's usable
  member, so junk never reaches the queue nor occupies disk;
- the review queue, the 100-slot quota and the one-off cleanup all use the same
  "reviewable group" definition, so the number the admin sees is the number the
  cap counts.
"""

import time

import numpy as np
from PIL import Image
import pytest

from recog.store import CAPTURE_LIMIT, Store

CHAR = "洛天依"


def picture(seed, size=(120, 96), color=None):
    if color is not None:
        return Image.new("RGB", size, color)
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (*size[::-1], 3), dtype=np.uint8))


def save(store, image, name=None):
    path = store.root / "uploads" / (name or f"{time.time_ns()}.png")
    image.save(path)
    return path


def capture(store, seed, score=0.7, size=(120, 96), color=None, character=CHAR):
    return store.ingest(save(store, picture(seed, size, color)), None, character, "chat", score)


def upload(store, seed, label=None, size=(120, 96), color=None):
    return store.ingest(save(store, picture(seed, size, color)), label, CHAR, "webui")["id"]


def row(store, image_id):
    with store.db() as db:
        return dict(db.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone())


def count(store, sql, parameters=()):
    with store.db() as db:
        return db.execute(sql, parameters).fetchone()[0]


def varied(store, image_id, size=(240, 192)):
    """Re-encode an existing image larger: same pHash, no exact-duplicate match."""
    with Image.open(store.root / row(store, image_id)["path"]) as image:
        return save(store, image.resize(size))


def legacy(store, image_id):
    """Mark a webui fixture as a chat row without passing the ingest gate."""
    with store.db() as db:
        db.execute("UPDATE images SET source='chat' WHERE id=?", (image_id,))


def test_solid_capture_is_dropped_with_its_temp_file(tmp_path):
    store = Store(tmp_path)
    path = save(store, picture(0, color="white"))
    result = store.ingest(path, None, CHAR, "chat", 0.7)
    assert result["skipped"] and result["reason"] == "solid"
    assert not path.exists()
    assert count(store, "SELECT COUNT(*) FROM images") == 0
    assert count(store, "SELECT COUNT(*) FROM groups") == 0
    assert store.capture_pending(CHAR) == 0


def test_capture_is_dropped_when_the_group_already_has_a_usable_member(tmp_path, monkeypatch):
    store = Store(tmp_path)
    upload(store, 1, "self")
    monkeypatch.setattr("recog.store.similar", lambda a, b: True)
    path = save(store, picture(2))
    result = store.ingest(path, None, CHAR, "chat", 0.7)
    assert result["skipped"] and result["reason"] == "group_covered"
    assert not path.exists()
    assert store.capture_pending(CHAR) == 0


def test_capture_joins_a_group_whose_only_member_was_removed(tmp_path, monkeypatch):
    store = Store(tmp_path)
    first = upload(store, 1, "self")
    original = row(store, first)
    store.set_removed(first, True)
    monkeypatch.setattr("recog.store.similar", lambda a, b: True)
    created = capture(store, 2)
    assert not created.get("skipped")
    kept = row(store, created["id"])
    assert kept["group_id"] == original["group_id"]
    assert kept["reason"] == ""
    assert store.capture_pending(CHAR) == 1


def test_capture_is_allowed_when_only_solid_members_would_match(tmp_path, monkeypatch):
    store = Store(tmp_path)
    solid = store.ingest(save(store, picture(0, color="white")), None, CHAR, "webui")
    assert solid["reason"] == "solid"
    monkeypatch.setattr("recog.store.similar", lambda a, b: True)
    created = capture(store, 2)
    assert not created.get("skipped")
    assert row(store, created["id"])["reason"] == ""
    assert store.capture_pending(CHAR) == 1


def test_capture_that_bridges_two_splits_is_dropped(tmp_path, monkeypatch):
    store = Store(tmp_path)
    train = upload(store, 1, "self")
    holdout = upload(store, 2, "other")
    with store.db() as db:
        db.execute("UPDATE groups SET split='train' WHERE id=(SELECT group_id FROM images WHERE id=?)", (train,))
        db.execute("UPDATE groups SET split='test' WHERE id=(SELECT group_id FROM images WHERE id=?)", (holdout,))
    monkeypatch.setattr("recog.store.similar", lambda a, b: True)
    path = save(store, picture(3))
    result = store.ingest(path, None, CHAR, "chat", 0.7)
    assert result["skipped"] and result["reason"] == "split_conflict"
    assert not path.exists()
    assert store.capture_pending(CHAR) == 0


def test_exact_duplicate_still_returns_the_existing_row(tmp_path):
    store = Store(tmp_path)
    first = capture(store, 1)
    path = store.root / "uploads" / "again.png"
    path.write_bytes((store.root / row(store, first["id"])["path"]).read_bytes())
    result = store.ingest(path, None, CHAR, "chat", 0.7)
    assert result["already_exists"] and result["id"] == first["id"]
    assert not path.exists()
    assert count(store, "SELECT COUNT(*) FROM images") == 1


def test_pending_counts_reviewable_groups_and_ignores_legacy_junk(tmp_path):
    store = Store(tmp_path)
    capture(store, 1)
    capture(store, 2)
    junk = store.ingest(save(store, picture(3, color="white")), None, CHAR, "webui")
    legacy(store, junk["id"])
    assert row(store, junk["id"])["reason"] == "solid"
    assert store.capture_pending(CHAR) == 2
    assert store.capture_pending(CHAR) < CAPTURE_LIMIT


def test_captured_view_lists_the_group_and_reports_the_queue(tmp_path):
    store = Store(tmp_path)
    created = capture(store, 1, score=0.81)
    junk = store.ingest(save(store, picture(2, color="white")), None, CHAR, "webui")
    legacy(store, junk["id"])
    data = store.images(character=CHAR, view="captured")
    assert [r["id"] for r in data["rows"]] == [created["id"]]
    assert data["total"] == 1
    assert data["capture"] == {"pending_groups": 1, "auto_rows": 1, "limit": CAPTURE_LIMIT}


def test_captured_view_orders_by_capture_score(tmp_path):
    store = Store(tmp_path)
    low = capture(store, 1, score=0.55)
    high = capture(store, 2, score=0.85)
    tier = capture(store, 3, score=0.85)
    listed = [r["id"] for r in store.images(character=CHAR, view="captured")["rows"]]
    assert listed[:2] == [high["id"], tier["id"]] or listed[:2] == [tier["id"], high["id"]]
    assert listed[2] == low["id"]


def test_review_queue_lists_every_character_together(tmp_path):
    store = Store(tmp_path)
    low = capture(store, 1, score=0.55)
    high = capture(store, 2, score=0.85, character="言和")
    data = store.images(view="captured")
    assert [r["id"] for r in data["rows"]] == [high["id"], low["id"]]
    assert {r["character"] for r in data["rows"]} == {CHAR, "言和"}
    assert data["total"] == 2
    assert data["capture"]["pending_groups"] == 2
    assert store.capture_pending() == 2
    assert store.capture_pending(CHAR) == 1
    scoped = store.images(character=CHAR, view="captured")
    assert [r["id"] for r in scoped["rows"]] == [low["id"]]
    assert scoped["capture"]["pending_groups"] == 1


def test_only_the_review_queue_runs_without_a_character(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(ValueError, match="角色"):
        store.images(view="all")


def test_confirm_and_undo_stay_scoped_to_the_rows_character(tmp_path):
    store = Store(tmp_path)
    created = capture(store, 1, character="言和")
    store.confirm_capture(created["id"], "言和", "self")
    assert row(store, created["id"])["label"] == "self"
    with pytest.raises(ValueError):
        store.undo_capture(created["id"], CHAR)


def test_cleanup_can_cover_every_character(tmp_path):
    store = Store(tmp_path)
    first = store.ingest(save(store, picture(1, color="white")), None, CHAR, "webui")
    second = store.ingest(save(store, picture(2, color="white")), None, "言和", "webui")
    legacy(store, first["id"])
    legacy(store, second["id"])
    assert store.purge_auto_captures()["deleted"] == 2
    assert count(store, "SELECT COUNT(*) FROM images") == 0


def test_confirm_writes_the_label_in_one_step(tmp_path):
    store = Store(tmp_path)
    created = capture(store, 1)
    store.confirm_capture(created["id"], CHAR, "self")
    kept = row(store, created["id"])
    assert kept["label"] == "self"
    assert store.capture_pending(CHAR) == 0
    assert store.images(character=CHAR, view="captured")["total"] == 0


def test_confirm_does_not_ask_for_a_medium(tmp_path):
    # 0.6.0: the medium pick was the only hard annotation gate in the plugin.
    store = Store(tmp_path)
    created = capture(store, 2)
    store.confirm_capture(created["id"], CHAR, "self")
    assert row(store, created["id"])["label"] == "self"


def test_confirm_rejects_webui_rows_and_second_confirmations(tmp_path):
    store = Store(tmp_path)
    uploaded = upload(store, 1, "self")
    with pytest.raises(ValueError):
        store.confirm_capture(uploaded, CHAR, "self")
    created = capture(store, 2)
    store.confirm_capture(created["id"], CHAR, "self")
    with pytest.raises(ValueError):
        store.confirm_capture(created["id"], CHAR, "other")


def test_undo_puts_the_group_back_in_the_queue(tmp_path):
    store = Store(tmp_path)
    created = capture(store, 1)
    store.confirm_capture(created["id"], CHAR, "self")
    store.undo_capture(created["id"], CHAR)
    kept = row(store, created["id"])
    assert kept["label"] is None
    assert store.capture_pending(CHAR) == 1
    with pytest.raises(ValueError):
        store.undo_capture(created["id"], CHAR)


def test_undo_refuses_when_the_group_has_another_labelled_member(tmp_path, monkeypatch):
    store = Store(tmp_path)
    first = upload(store, 1, "self")
    monkeypatch.setattr("recog.store.similar", lambda a, b: True)
    second = store.ingest(save(store, picture(2)), None, CHAR, "webui")
    store.label(second["id"], "self")
    legacy(store, second["id"])
    assert row(store, first)["group_id"] == row(store, second["id"])["group_id"]
    # _reconcile already excludes a duplicate, so put the row back in the
    # "usable" state this guard is meant to catch.
    with store.db() as db:
        db.execute("UPDATE images SET reason='' WHERE id=?", (second["id"],))
    with pytest.raises(ValueError, match="同组"):
        store.undo_capture(second["id"], CHAR)


def test_purge_removes_the_row_its_file_and_its_thumbnail(tmp_path):
    store = Store(tmp_path)
    created = capture(store, 1)
    store.images(character=CHAR, view="captured")  # builds the thumbnail cache
    kept = row(store, created["id"])
    path = store.root / kept["path"]
    cache = store.root / "thumbnails" / (kept["sha"] + "-512.jpg")
    assert path.exists() and cache.exists()
    store.purge_capture(created["id"], CHAR)
    assert not path.exists() and not cache.exists()
    assert count(store, "SELECT COUNT(*) FROM images WHERE id=?", (created["id"],)) == 0
    assert count(store, "SELECT COUNT(*) FROM groups") == 0


def test_purge_refuses_confirmed_rows(tmp_path):
    store = Store(tmp_path)
    created = capture(store, 1)
    store.confirm_capture(created["id"], CHAR, "self")
    with pytest.raises(ValueError):
        store.purge_capture(created["id"], CHAR)


def test_cleanup_deletes_only_untouched_legacy_junk(tmp_path):
    store = Store(tmp_path)
    # Two labelled variants form a conflict: the human labels must survive.
    conflict_a = upload(store, 1, "self")
    conflict_b = store.ingest(varied(store, conflict_a), "other", CHAR)["id"]
    # One labelled keeper plus the variant the catalog marked as duplicate.
    keeper = upload(store, 3, "self")
    duplicate = store.ingest(varied(store, keeper), None, CHAR, "webui")["id"]
    legacy(store, conflict_a)
    legacy(store, conflict_b)
    legacy(store, duplicate)
    solid = store.ingest(save(store, picture(5, color="white")), None, CHAR, "webui")["id"]
    legacy(store, solid)
    assert row(store, duplicate)["reason"] == "duplicate"
    assert row(store, conflict_a)["reason"] == "label_conflict"

    kept_paths = [store.root / row(store, i)["path"] for i in (keeper, conflict_a, conflict_b)]
    junk_path = store.root / row(store, duplicate)["path"]
    result = store.purge_auto_captures(CHAR)
    assert result["deleted"] == 2  # duplicate + solid
    assert not junk_path.exists()
    assert all(path.exists() for path in kept_paths)
    assert count(store, "SELECT COUNT(*) FROM images WHERE id=?", (keeper,)) == 1
    assert count(store, "SELECT COUNT(*) FROM images WHERE id=?", (conflict_a,)) == 1
    assert count(store, "SELECT COUNT(*) FROM images WHERE id=?", (conflict_b,)) == 1
    assert count(store, "SELECT COUNT(*) FROM images WHERE id=?", (duplicate,)) == 0
    assert count(store, "SELECT COUNT(*) FROM images WHERE id=?", (solid,)) == 0
    # Only the solid row lived alone, so only its group disappears.
    assert count(store, "SELECT COUNT(*) FROM groups") == 2
