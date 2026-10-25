"""Picture library filters and promoting a test result into the library.

0.6.0 removed the medium / text filters along with the tags they filtered on, so
this file now covers the views that survived plus the promote path — a
recognition-test image was once thrown away, which meant "the model got this
wrong" could never become training data.
"""

import io
import time

import numpy as np
from PIL import Image
import pytest

from recog.store import Store

CHAR = "洛天依"


def add(store, seed, label="self", size=(120, 96)):
    width, height = size
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    path = store.root / "uploads" / f"{seed}-{time.time_ns()}.png"
    Image.fromarray(pixels).save(path)
    return store.ingest(path, label, CHAR)


def ids(result):
    return {row["id"] for row in result["rows"]}


def test_eligible_view_excludes_unlabelled_and_removed(tmp_path):
    store = Store(tmp_path)
    labelled = add(store, 1, "self")
    unlabelled = add(store, 2, None)
    removed = add(store, 3, "self")
    with store.db() as db:
        db.execute("UPDATE images SET removed=1 WHERE id=?", (removed["id"],))

    listed = ids(store.images(character=CHAR, view="eligible"))
    assert listed == {labelled["id"]}
    assert unlabelled["id"] not in listed


def test_unlabelled_view_lists_only_rows_without_a_label(tmp_path):
    store = Store(tmp_path)
    add(store, 1, "self")
    waiting = add(store, 2, None)
    assert ids(store.images(character=CHAR, view="unlabelled")) == {waiting["id"]}


def test_removed_view_lists_what_can_be_restored(tmp_path):
    store = Store(tmp_path)
    add(store, 1, "self")
    removed = add(store, 2, "self")
    with store.db() as db:
        db.execute("UPDATE images SET removed=1 WHERE id=?", (removed["id"],))
    assert ids(store.images(character=CHAR, view="removed")) == {removed["id"]}


def test_unknown_views_are_rejected(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(ValueError):
        store.images(character=CHAR, view="untyped")


def test_listing_carries_no_annotation_text(tmp_path):
    store = Store(tmp_path)
    add(store, 1, "self")
    row = store.images(character=CHAR)["rows"][0]
    for field in ("style_text", "neg_type_text", "has_text_text"):
        assert field not in row


def test_a_test_result_can_be_ingested_from_raw_bytes(tmp_path):
    """The promote path writes bytes with an .img suffix; decoding must not care."""
    store = Store(tmp_path)
    buffer = io.BytesIO()
    Image.fromarray(np.random.default_rng(9).integers(0, 256, (96, 120, 3), dtype=np.uint8)).save(
        buffer, format="PNG"
    )
    target = store.root / "uploads" / (f"promote-{time.time_ns()}.img")
    target.write_bytes(buffer.getvalue())

    first = store.ingest(target, "other", CHAR)
    assert first.get("id")

    again = store.root / "uploads" / (f"promote-{time.time_ns()}.img")
    again.write_bytes(buffer.getvalue())
    repeat = store.ingest(again, "other", CHAR)
    assert repeat.get("already_exists") is True
    assert repeat["id"] == first["id"]
