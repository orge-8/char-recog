"""Remembered WebUI choices: the host's sandboxed iframe has no localStorage."""

import json
from pathlib import Path

from recog.store import Store

ROOT = Path(__file__).resolve().parents[1]
STATE = {
    "character": "诗岸",
    "filter": "excluded",
    "mode": "fresh",
    "device": "cuda",
    "input_size": 448,
    "device_chosen": True,
    "capture_character": "",
    "job_scope": "all",
    "job_state": "failed",
}


def test_ui_state_round_trips(tmp_path):
    store = Store(tmp_path)
    assert store.ui_state() == {}
    assert store.set_ui_state(STATE) == STATE
    assert store.ui_state() == STATE


def test_partial_saves_merge(tmp_path):
    store = Store(tmp_path)
    store.set_ui_state({"filter": "unlabelled"})
    store.set_ui_state({"mode": "fresh"})
    assert store.ui_state() == {"filter": "unlabelled", "mode": "fresh"}


def test_explicit_null_input_size_means_use_the_form_default(tmp_path):
    store = Store(tmp_path)
    store.set_ui_state({"input_size": 224})
    store.set_ui_state({"input_size": None})
    state = store.ui_state()
    assert "input_size" in state and state["input_size"] is None


def test_invalid_values_are_dropped_instead_of_failing(tmp_path):
    store = Store(tmp_path)
    store.set_ui_state(
        {
            "character": "  ",
            "capture_character": "言和",
            "filter": "bogus",
            "mode": "fresh",
            "input_size": 999,
            "device_chosen": "yes",
        }
    )
    assert store.ui_state() == {"capture_character": "言和", "mode": "fresh"}


def test_stale_or_broken_stored_json_is_ignored(tmp_path):
    store = Store(tmp_path)
    store.set_setting("ui_state", json.dumps({"filter": "nope", "device": "tpu", "character": "x" * 33}))
    assert store.ui_state() == {}
    store.set_setting("ui_state", "{not json")
    assert store.ui_state() == {}


def test_ui_save_is_declared_on_every_page():
    data = json.loads((ROOT / "webui.json").read_text(encoding="utf-8"))
    for page in data["pages"]:
        assert (page.get("queries") or {}).get("ui", {}).get("api") == "self_ui", page["id"]
        action = page["actions"]["ui_save"]
        assert action["api"] == "self_ui_save", page["id"]
        assert action["parameters"]["state"]["required"] is True, page["id"]
