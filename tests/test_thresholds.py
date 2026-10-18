"""Per-character online gate: storage, defaults and corruption handling."""

import pytest

from recog.scores import DEFAULT_THRESHOLD
from recog.store import Store

CHAR = "洛天依"


def test_threshold_defaults_and_roundtrip(tmp_path):
    store = Store(tmp_path)
    assert store.threshold(CHAR) == DEFAULT_THRESHOLD
    result = store.set_threshold(CHAR, 0.72)
    assert result == {"character": CHAR, "threshold": 0.72, "default": DEFAULT_THRESHOLD}
    assert store.threshold(CHAR) == 0.72
    assert Store(tmp_path).threshold(CHAR) == 0.72
    assert store.setting(f"threshold:{CHAR}") == "0.720"


@pytest.mark.parametrize("value", [0.0, 1.0, -0.5, 1.5, float("nan"), "abc", None, ""])
def test_threshold_rejects_invalid_values(tmp_path, value):
    store = Store(tmp_path)
    with pytest.raises(ValueError):
        store.set_threshold(CHAR, value)
    assert store.threshold(CHAR) == DEFAULT_THRESHOLD


def test_threshold_corrupt_setting_falls_back(tmp_path):
    store = Store(tmp_path)
    for corrupt in ("oops", "2.0", "-1", "nan"):
        store.set_setting(f"threshold:{CHAR}", corrupt)
        assert store.threshold(CHAR) == DEFAULT_THRESHOLD
    assert store.thresholds() == {}


def test_thresholds_returns_every_override_in_one_call(tmp_path):
    store = Store(tmp_path)
    store.set_threshold(CHAR, 0.55)
    store.set_threshold("言和", 0.8)
    assert store.thresholds() == {CHAR: 0.55, "言和": 0.8}
    # A broken row must be skipped, not crash the worker's bulk read.
    store.set_setting("threshold:坏数据", "not-a-number")
    assert store.thresholds() == {CHAR: 0.55, "言和": 0.8}


def test_threshold_namespace_does_not_leak_into_inference_targets(tmp_path):
    store = Store(tmp_path)
    store.set_threshold(CHAR, 0.7)
    assert store.active_characters() == []
    store.set_active(CHAR, "v1")
    assert store.active_characters() == [CHAR]


def test_clear_threshold_removes_the_override(tmp_path):
    store = Store(tmp_path)
    store.set_threshold(CHAR, 0.9)
    result = store.clear_threshold(CHAR)
    assert result == {"character": CHAR, "threshold": DEFAULT_THRESHOLD, "default": DEFAULT_THRESHOLD}
    assert store.threshold(CHAR) == DEFAULT_THRESHOLD
    assert store.setting(f"threshold:{CHAR}") == ""


def test_threshold_requires_a_character(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(ValueError, match="缺少角色名"):
        store.threshold("")
    with pytest.raises(ValueError, match="缺少角色名"):
        store.set_threshold("   ", 0.6)
