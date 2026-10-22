"""Chat capture: keep the images the model hesitates about, as *unlabelled* rows.

Three rules under test:
- only the band 0.4–0.9 is captured (confident answers teach nothing, rejected
  ones are the ones a noisy group would flood us with);
- the captured row has no label — the model's guess is not ground truth, and
  training on it would teach the next model its own mistake;
- the queue is capped per character so it cannot become a second library.
"""

import time

import numpy as np
from PIL import Image
import pytest

from recog.scores import CAPTURE_BAND, capture_candidate
from recog.store import CAPTURE_LIMIT, Store

CHAR = "洛天依"


def result(score, character=CHAR, status="ok"):
    return {"character": character, "score": score, "status": status}


def add(store, seed, label=None, source="", score=None):
    pixels = np.random.default_rng(seed).integers(0, 256, (96, 120, 3), dtype=np.uint8)
    path = store.root / "uploads" / f"{seed}-{time.time_ns()}.png"
    Image.fromarray(pixels).save(path)
    return store.ingest(path, label, CHAR, "", "", "", source, score)


def test_band_picks_the_highest_hesitant_score():
    picked = capture_candidate([result(0.35), result(0.55), result(0.72), result(0.95)])
    assert picked["score"] == 0.72


def test_confident_or_rejected_scores_are_not_captured():
    assert capture_candidate([result(0.05), result(0.3)]) is None
    assert capture_candidate([result(0.9), result(0.99)]) is None
    assert capture_candidate([]) is None


def test_failed_results_are_ignored():
    assert capture_candidate([result(0.7, status="error")]) is None


def test_band_boundaries_are_half_open():
    low, high = CAPTURE_BAND
    assert capture_candidate([result(low)]) is not None
    assert capture_candidate([result(high)]) is None


def test_captured_image_is_stored_unlabelled_with_its_score(tmp_path):
    store = Store(tmp_path)
    created = add(store, 1, source="chat", score=0.71)
    with store.db() as db:
        row = dict(db.execute("SELECT * FROM images WHERE id=?", (created["id"],)).fetchone())
    assert row["label"] is None
    assert row["source"] == "chat"
    assert row["model_score"] == pytest.approx(0.71)


def test_pending_counter_only_counts_unlabelled_chat_rows(tmp_path):
    store = Store(tmp_path)
    add(store, 1, source="chat", score=0.7)
    add(store, 2, source="chat", score=0.6)
    add(store, 3, label="self", source="chat", score=0.8)  # already confirmed
    add(store, 4, label=None, source="webui")  # uploaded, not captured
    assert store.capture_pending(CHAR) == 2


def test_capture_rows_show_up_in_the_unlabelled_view(tmp_path):
    store = Store(tmp_path)
    captured = add(store, 1, source="chat", score=0.66)
    listed = {row["id"] for row in store.images(character=CHAR, view="unlabelled")["rows"]}
    assert captured["id"] in listed
    row = next(row for row in store.images(character=CHAR, view="unlabelled")["rows"] if row["id"] == captured["id"])
    assert row["source_text"] == "聊天捕获"
    assert "0.660" in row["captured_text"]


def test_queue_cap_is_a_hundred_per_character():
    assert CAPTURE_LIMIT == 100
