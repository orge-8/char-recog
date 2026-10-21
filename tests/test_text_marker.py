"""Text overlay is a marker, not a fourth medium.

「图文/梗图」used to sit in `style` next to 平涂/厚涂/实拍, which meant one field
answered two questions: how the image is painted, and whether text was pasted on
top. It is now `has_text` (yes/no), and the old value migrates its information
(「图文/梗图」means there is text) instead of being dropped.
"""

import json
import time

import numpy as np
from PIL import Image
import pytest

from recog.scores import by_text
from recog.store import Store

CHAR = "洛天依"


def add(store, seed, label, style="", size=(120, 96), neg_type="", has_text=""):
    width, height = size
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    path = store.root / "uploads" / f"{seed}-{time.time_ns()}.png"
    Image.fromarray(pixels).save(path)
    return store.ingest(path, label, CHAR, style, neg_type, has_text)


def rows(store):
    with store.db() as db:
        return [dict(r) for r in db.execute("SELECT * FROM images WHERE character=? ORDER BY id", (CHAR,))]


def test_composite_is_no_longer_a_medium(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(ValueError):
        add(store, 1, "self", "composite")
    assert rows(store) == []


def test_overlay_marker_is_validated(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(ValueError):
        add(store, 1, "self", "flat", has_text="maybe")
    result = add(store, 2, "self", "flat", has_text="yes")
    assert rows(store)[0]["has_text"] == "yes"
    with pytest.raises(ValueError):
        store.set_style(result["id"], "flat", has_text="text")


def test_legacy_composite_moves_its_information_to_the_marker(tmp_path):
    """The old value said "there is text"; that fact must survive, the medium may not."""
    store = Store(tmp_path)
    row = add(store, 1, "self", "flat", has_text="no")
    with store.db() as db:
        db.execute("UPDATE images SET style='composite' WHERE id=?", (row["id"],))

    reopened = Store(tmp_path)
    with reopened.db() as db:
        migrated = dict(db.execute("SELECT * FROM images WHERE id=?", (row["id"],)).fetchone())
        recorded = db.execute("SELECT value FROM settings WHERE key='composite_migrated'").fetchone()
    assert migrated["style"] == ""
    # has_text was already 'no', so the migration must not overwrite a real answer.
    assert migrated["has_text"] == "no"
    assert json.loads(recorded[0])["moved"] == 1


def test_legacy_composite_sets_the_marker_when_unknown(tmp_path):
    store = Store(tmp_path)
    row = add(store, 1, "self", "flat", has_text="")
    with store.db() as db:
        db.execute("UPDATE images SET style='composite' WHERE id=?", (row["id"],))
    Store(tmp_path)
    with store.db() as db:
        migrated = dict(db.execute("SELECT * FROM images WHERE id=?", (row["id"],)).fetchone())
    assert migrated["style"] == ""
    assert migrated["has_text"] == "yes"


def test_batch_style_accepts_the_marker_alone(tmp_path):
    store = Store(tmp_path)
    first = add(store, 1, "self", "flat")
    second = add(store, 2, "self", "paint")
    result = store.batch_style(f"{first['id']},{second['id']}", has_text="yes")
    assert result["has_text"] == "yes"
    assert result["style"] is None  # untouched
    with store.db() as db:
        assert {r[0] for r in db.execute("SELECT style FROM images")} == {"flat", "paint"}


def test_by_text_reports_recall_on_the_marker():
    predictions = [
        {"label": "self", "has_text": "yes", "score": 0.9},
        {"label": "self", "has_text": "yes", "score": 0.2},
        {"label": "self", "has_text": "no", "score": 0.95},
        {"label": "self", "has_text": "no", "score": 0.9},
        {"label": "self", "has_text": "no", "score": 0.85},
        {"label": "self", "has_text": "no", "score": 0.8},
        {"label": "self", "has_text": "no", "score": 0.75},
    ]
    table = {row["has_text"]: row for row in by_text(predictions, threshold=0.6, min_count=2)}
    assert table["yes"]["n"] == 2
    assert table["yes"]["recall"] == pytest.approx(0.5)
    assert table["yes"]["unreliable"] is False
    assert table["no"]["n"] == 5
    assert table["no"]["recall"] == 1.0
