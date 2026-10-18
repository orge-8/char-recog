"""SourceIndex must be bit-for-bit equivalent to the old linear scan."""

import json
import random

import pytest

from recog.store import SourceIndex, Store

CHAR = "洛天依"
KEYS = ["k0", "k1", "k2", ""]  # deliberately tiny pool to force collisions


def reference(row, sources):
    """The 0.3.4 implementation, verbatim."""
    for source in reversed(sources):
        if source["group_id"] == row["group_id"] or source["sha"] == row["sha"] or source["decoded"] == row["decoded"]:
            return source["label"] == row["label"]
    return False


def make_row(rng):
    return {
        "group_id": rng.choice(KEYS),
        "sha": rng.choice(KEYS),
        "decoded": rng.choice(KEYS),
        "label": rng.choice(["self", "other"]),
    }


def test_randomised_equivalence_with_the_reference_scan():
    rng = random.Random(20261018)
    for _ in range(500):
        sources = [make_row(rng) for _ in range(rng.randint(0, 6))]
        index = SourceIndex(sources)
        for _ in range(4):
            row = make_row(rng)
            assert index.seen(row) == reference(row, sources), (row, sources)


def test_latest_match_decides_not_the_most_recent_same_label():
    sources = [
        {"group_id": "g1", "sha": "s9", "decoded": "d9", "label": "other"},
        {"group_id": "g2", "sha": "s1", "decoded": "d8", "label": "self"},
    ]
    index = SourceIndex(sources)
    # Matches source 0 by group_id and source 1 by sha; the later index wins.
    assert index.seen({"group_id": "g1", "sha": "s1", "decoded": "dz", "label": "self"}) is True
    assert index.seen({"group_id": "g1", "sha": "s1", "decoded": "dz", "label": "other"}) is False


def test_duplicate_keys_use_the_last_occurrence():
    sources = [
        {"group_id": "g1", "sha": "s1", "decoded": "d1", "label": "self"},
        {"group_id": "g2", "sha": "s1", "decoded": "d2", "label": "other"},
    ]
    index = SourceIndex(sources)
    # setdefault-style indexes would answer True here; the scan answers False.
    assert index.seen({"group_id": "gx", "sha": "s1", "decoded": "dx", "label": "self"}) is False


def test_empty_strings_are_real_keys():
    sources = [{"group_id": "", "sha": "", "decoded": "", "label": "self"}]
    assert SourceIndex(sources).seen({"group_id": "", "sha": "", "decoded": "", "label": "self"}) is True


def test_empty_source_list_never_matches():
    assert SourceIndex([]).seen({"group_id": "g", "sha": "s", "decoded": "d", "label": "self"}) is False


def _seed_active_model(store, root, sources):
    """Minimal models row + snapshot so trained_sources() returns real entries."""
    with store.db() as db:
        db.execute("INSERT INTO models VALUES (?,?,?,?,?,?)", ("v1", "j1", "sha", "{}", 0.0, CHAR))
        db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (f"active:{CHAR}", "v1"))
    directory = root / "jobs" / "j1"
    directory.mkdir(parents=True)
    (directory / "snapshot.json").write_text(
        json.dumps({"trained_sources": sources, "images": []}), encoding="utf-8"
    )


def test_training_readiness_does_not_fall_back_to_the_linear_scan(tmp_path, monkeypatch):
    from PIL import Image
    import numpy as np

    store = Store(tmp_path)
    group = "g" * 32
    image = Image.fromarray(np.random.default_rng(7).integers(0, 256, (48, 48, 3), dtype=np.uint8))
    path = tmp_path / "uploads" / "one.png"
    image.save(path)
    ident = store.ingest(path, "self", CHAR)["id"]
    with store.db() as db:
        db.execute("UPDATE groups SET split='train' WHERE id=?", (group,))
        row = db.execute("SELECT * FROM images WHERE id=?", (ident,)).fetchone()
        group = row["group_id"]
        sources = [{"group_id": group, "sha": row["sha"], "decoded": row["decoded"], "label": "self", "id": ident}]
    _seed_active_model(store, tmp_path, sources)

    monkeypatch.setattr(Store, "source_seen", staticmethod(lambda row, sources: (_ for _ in ()).throw(AssertionError("linear scan"))))
    readiness = store.training_readiness(CHAR)
    assert readiness["pending_count"] == 0
    assert readiness["groups"]["self"] == 1
