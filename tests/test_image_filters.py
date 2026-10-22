"""Picture library filters and promoting a test result into the library.

Two gaps this covers:
- the medium / text tags could be set but never filtered on, so the list stayed
  one undifferentiated 24-per-page stream;
- a recognition-test image was thrown away, so "the model got this wrong" could
  never become training data.
"""

import io
import time

import numpy as np
from PIL import Image
import pytest

from recog.store import Store

CHAR = "洛天依"


def add(store, seed, label="self", style="", has_text="", size=(120, 96)):
    width, height = size
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    path = store.root / "uploads" / f"{seed}-{time.time_ns()}.png"
    Image.fromarray(pixels).save(path)
    return store.ingest(path, label, CHAR, style, "", has_text)


def ids(result):
    return {row["id"] for row in result["rows"]}


def test_untyped_view_lists_only_eligible_images_without_a_medium(tmp_path):
    store = Store(tmp_path)
    flat = add(store, 1, style="flat")
    plain = add(store, 2, style="")
    removed = add(store, 3, style="")
    with store.db() as db:
        db.execute("UPDATE images SET removed=1 WHERE id=?", (removed["id"],))

    listed = ids(store.images(character=CHAR, view="untyped"))
    assert listed == {plain["id"]}
    assert flat["id"] not in listed


def test_style_filter_narrows_the_list(tmp_path):
    store = Store(tmp_path)
    flat = add(store, 1, style="flat")
    paint = add(store, 2, style="paint")
    assert ids(store.images(character=CHAR, style="flat")) == {flat["id"]}
    assert ids(store.images(character=CHAR, style="paint")) == {paint["id"]}


def test_text_filter_narrows_the_list(tmp_path):
    store = Store(tmp_path)
    with_text = add(store, 1, has_text="yes")
    without = add(store, 2, has_text="no")
    assert ids(store.images(character=CHAR, has_text="yes")) == {with_text["id"]}
    assert ids(store.images(character=CHAR, has_text="no")) == {without["id"]}


def test_filters_combine(tmp_path):
    store = Store(tmp_path)
    wanted = add(store, 1, style="flat", has_text="yes")
    add(store, 2, style="flat", has_text="no")
    add(store, 3, style="paint", has_text="yes")
    assert ids(store.images(character=CHAR, style="flat", has_text="yes")) == {wanted["id"]}


def test_unknown_filter_values_are_rejected(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(ValueError):
        store.images(character=CHAR, style="official")
    with pytest.raises(ValueError):
        store.images(character=CHAR, has_text="maybe")


def test_a_test_result_can_be_ingested_from_raw_bytes(tmp_path):
    """The promote path writes bytes with an .img suffix; decoding must not care."""
    store = Store(tmp_path)
    buffer = io.BytesIO()
    Image.fromarray(np.random.default_rng(9).integers(0, 256, (96, 120, 3), dtype=np.uint8)).save(
        buffer, format="PNG"
    )
    target = store.root / "uploads" / (f"promote-{time.time_ns()}.img")
    target.write_bytes(buffer.getvalue())

    first = store.ingest(target, "other", CHAR, "", "", "")
    assert first.get("id")

    again = store.root / "uploads" / (f"promote-{time.time_ns()}.img")
    again.write_bytes(buffer.getvalue())
    repeat = store.ingest(again, "other", CHAR, "", "", "")
    assert repeat.get("already_exists") is True
    assert repeat["id"] == first["id"]
