"""The negative bucket is the identity axis; medium stays on `style`.

「表情包」used to be a third value and overlapped both remaining ones: a sticker of
another character was simultaneously "other character" and "sticker". It is a
medium, already carried by ``style=composite``, so it was retired in 0.5.17.
"""

import json
import time

import numpy as np
from PIL import Image
import pytest

from recog.scores import by_neg
from recog.store import Store

CHAR = "洛天依"


def add(store, seed, label, style="", size=(120, 96), neg_type=""):
    width, height = size
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    path = store.root / "uploads" / f"{seed}-{time.time_ns()}.png"
    Image.fromarray(pixels).save(path)
    return store.ingest(path, label, CHAR, style, neg_type)


def rows(store):
    with store.db() as db:
        return [
            dict(row)
            for row in db.execute(
                "SELECT i.*,g.split FROM images i JOIN groups g ON g.id=i.group_id "
                "WHERE i.character=? ORDER BY i.id",
                (CHAR,),
            )
        ]


def test_sticker_is_rejected_on_write(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(ValueError):
        add(store, 1, "other", "photo", neg_type="sticker")
    assert rows(store) == []


def test_sticker_cannot_be_set_after_the_fact(tmp_path):
    store = Store(tmp_path)
    row = add(store, 1, "other", "photo", neg_type="non_character")
    with pytest.raises(ValueError):
        store.set_style(row["id"], "photo", "sticker")
    with pytest.raises(ValueError):
        store.batch_style(row["id"], neg_type="sticker")


def test_legacy_values_are_cleared_once_and_counted(tmp_path):
    """Clearing is destructive, so it must leave a trace instead of vanishing."""
    store = Store(tmp_path)
    row = add(store, 1, "other", "photo", neg_type="non_character")
    with store.db() as db:
        db.execute("UPDATE images SET neg_type='sticker' WHERE id=?", (row["id"],))

    reopened = Store(tmp_path)  # schema init runs the one-time cleanup
    with reopened.db() as db:
        assert db.execute("SELECT neg_type FROM images WHERE id=?", (row["id"],)).fetchone()[0] == ""
        recorded = db.execute("SELECT value FROM settings WHERE key='neg_type_reset'").fetchone()
    assert recorded is not None
    assert json.loads(recorded[0])["cleared"] == 1

    Store(tmp_path)  # idempotent: nothing left to clear, no second record
    with reopened.db() as db:
        assert db.execute("SELECT value FROM settings WHERE key='neg_type_reset'").fetchone()[0]


def test_untagged_negatives_still_enter_the_eval_set(tmp_path):
    """Tagging only the medium used to leave the false-positive column empty."""
    store = Store(tmp_path)
    for index in range(30):
        add(store, 1000 + index, "self", "flat", (100, 100))
    for index in range(30):
        add(store, 2000 + index, "other", "photo", (120, 90))  # no neg_type at all

    with store.db() as db:
        store.freeze(db, "job", "fresh", "cpu", CHAR)

    evaluated = [row for row in rows(store) if row["split"] == "eval"]
    assert [row for row in evaluated if row["label"] == "other"], evaluated


def test_by_neg_splits_identity_and_marks_thin_buckets():
    predictions = [
        {"label": "other", "neg_type": "other_character", "score": 0.9},
        {"label": "other", "neg_type": "other_character", "score": 0.8},
        {"label": "other", "neg_type": "other_character", "score": 0.3},
        {"label": "other", "neg_type": "other_character", "score": 0.2},
        {"label": "other", "neg_type": "other_character", "score": 0.1},
        {"label": "other", "neg_type": "other_character", "score": 0.05},
        {"label": "other", "neg_type": "non_character", "score": 0.95},
        {"label": "other", "neg_type": "", "score": 0.7},
        {"label": "self", "neg_type": "", "score": 0.99},  # positives never counted
    ]
    table = {row["neg_type"]: row for row in by_neg(predictions, threshold=0.6, min_count=5)}
    assert set(table) == {"other_character", "non_character", ""}
    assert table["other_character"]["n"] == 6
    assert table["other_character"]["flagged"] == 2
    assert table["other_character"]["fpr"] == pytest.approx(2 / 6)
    assert table["other_character"]["unreliable"] is False
    assert table["non_character"]["unreliable"] is True
    assert table[""]["label"] == "未标注"
