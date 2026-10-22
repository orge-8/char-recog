"""Prompt policy: one character per image, close calls suppressed, gates respected."""

import asyncio
import base64
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

# QA 修复（gstack-qa-lead, 2026-10-22）：maibot_sdk 缺失时原先在收集阶段直接 ImportError
# 中断整个 pytest 运行，改为显式 importorskip（与 test_tool.py 同一问题）。
pytest.importorskip("maibot_sdk", reason="plugin.py 依赖宿主 SDK maibot_sdk，缺失时整模块跳过")

spec = importlib.util.spec_from_file_location(
    "annotation_plugin_test",
    Path(__file__).parents[1] / "plugin.py",
    submodule_search_locations=[str(Path(__file__).parents[1])],
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

CHAR = "洛天依"


def build(results, limits=None, seen=None):
    """A plugin whose worker always returns the same scores."""
    seen = seen if seen is not None else []
    characters = sorted({item["character"] for item in results}) or [CHAR]
    state = {"limits": dict(limits or {})}

    def infer(raw, timeout=8, character="", *, wait=True):
        seen.append(raw)
        return {"status": "ok", "results": [dict(item) for item in results]}

    plugin = module.create_plugin()
    plugin._plugin_config_instance = module.Config(characters=characters)
    plugin._image_cache = {}
    plugin.service = SimpleNamespace(
        store=SimpleNamespace(
            style_coverage=lambda character: [],
            active_characters=lambda: characters,
            active=lambda character: "v1",
            thresholds=lambda: dict(state["limits"]),
        ),
        infer=infer,
        inference_enabled=True,
    )
    return plugin, seen, state


def annotate(plugin, data=b"picture"):
    message = {"raw_message": [{"type": "image", "binary_data_base64": base64.b64encode(data).decode()}]}
    parts = asyncio.run(plugin.annotate_received_images(message))["modified_kwargs"]["message"]["raw_message"]
    return [part["data"] for part in parts if part["type"] == "text"]


def hit(character, score):
    return {"status": "ok", "character": character, "score": score, "version": "v1"}


def test_prompts_only_the_highest_accepted_character():
    plugin, _, _ = build([hit("A", 0.90), hit("B", 0.60)], {"A": 0.6, "B": 0.6})
    texts = annotate(plugin)
    assert len(texts) == 1
    assert "A" in texts[0] and "B" not in texts[0]


def test_close_margin_prompts_nobody():
    plugin, _, _ = build([hit("A", 0.85), hit("B", 0.84)], {"A": 0.6, "B": 0.6})
    assert annotate(plugin) == []


def test_five_point_lead_still_prompts_the_winner():
    plugin, _, _ = build([hit("A", 0.85), hit("B", 0.80)], {"A": 0.6, "B": 0.6})
    texts = annotate(plugin)
    assert len(texts) == 1 and "A" in texts[0]


def test_tie_prompts_nobody():
    plugin, _, _ = build([hit("A", 0.8), hit("B", 0.8)], {"A": 0.6, "B": 0.6})
    assert annotate(plugin) == []


def test_single_character_hit_and_miss():
    plugin, _, _ = build([hit(CHAR, 0.7)], {CHAR: 0.6})
    assert len(annotate(plugin)) == 1
    plugin, _, _ = build([hit(CHAR, 0.59)], {CHAR: 0.6})
    assert annotate(plugin) == []


def test_all_below_their_gate_prompts_nobody():
    plugin, _, _ = build([hit("A", 0.59), hit("B", 0.3)], {"A": 0.6, "B": 0.6})
    assert annotate(plugin) == []


def test_a_gate_above_reach_does_not_shadow_an_accepted_character():
    # A is the raw top-1 but its admin gate is out of reach; B still qualifies.
    plugin, _, _ = build([hit("A", 0.75), hit("B", 0.70)], {"A": 0.8, "B": 0.6})
    texts = annotate(plugin)
    assert len(texts) == 1 and "B" in texts[0]


def test_changing_a_gate_invalidates_the_prediction_cache():
    plugin, seen, state = build([hit(CHAR, 0.7)], {CHAR: 0.6})
    assert len(annotate(plugin)) == 1
    assert len(seen) == 1
    # Same image, same model version, but a stricter gate: the cached verdict
    # must not be reused, otherwise up to 512 images keep the old decision.
    state["limits"][CHAR] = 0.9
    assert annotate(plugin) == []
    assert len(seen) == 2


def test_tool_returns_every_score_plus_the_aggregation():
    results = [hit("A", 0.9), hit("B", 0.7)]
    plugin, _, _ = build(results, {"A": 0.6, "B": 0.6})
    message = {"session_id": "chat", "raw_message": [{"type": "image", "binary_data_base64": base64.b64encode(b"x").decode()}]}
    plugin._ctx = SimpleNamespace(message=SimpleNamespace(get_by_id=AsyncMock(return_value=message)))
    result = asyncio.run(plugin.recognize("m1", 0, stream_id="chat"))
    assert [item["character"] for item in result["results"]] == ["A", "B"]
    assert result["top1"]["character"] == "A"
    assert result["margin"] == pytest.approx(0.2)
    assert result["ambiguous"] is False
    assert "content" in result


def test_tool_reports_no_margin_when_only_one_character_qualifies():
    results = [hit("A", 0.9), hit("B", 0.4)]
    plugin, _, _ = build(results, {"A": 0.6, "B": 0.6})
    message = {"session_id": "chat", "raw_message": [{"type": "image", "binary_data_base64": base64.b64encode(b"x").decode()}]}
    plugin._ctx = SimpleNamespace(message=SimpleNamespace(get_by_id=AsyncMock(return_value=message)))
    result = asyncio.run(plugin.recognize("m1", 0, stream_id="chat"))
    assert result["top1"]["character"] == "A"
    assert result["margin"] is None
    assert result["ambiguous"] is False


def test_report_metrics_follow_the_current_gate():
    from recog.scores import EVALUATION_THRESHOLD

    def metric(precision, recall, numerator=1, denominator=1, count=4):
        return {
            "precision": precision,
            "precision_numerator": numerator,
            "precision_denominator": denominator,
            "recall": recall,
            "other_recall": 1.0,
            "f1": 0.8,
            "accuracy": 0.9,
            "balanced_accuracy": 0.85,
            "confusion_matrix": [[2, 0], [1, 1]],
            "count": count,
        }

    validation = [
        {"id": "v1", "label": "self", "score": 0.9},
        {"id": "v2", "label": "self", "score": 0.55},
        {"id": "v3", "label": "other", "score": 0.3},
        {"id": "v4", "label": "other", "score": 0.2},
    ]
    report = {
        "selection": {"epoch": 3, "validation": metric(0.5, 0.5), "threshold": EVALUATION_THRESHOLD},
        "test": metric(0.5, 0.5),
        "validation_predictions": validation,
        "test_predictions": validation,
        "historical_test": False,
        "comparable_to_previous": True,
        "total_seconds": 12.0,
        "training_seconds_including_validation": 10.0,
        "audit": {"passed": True},
        "character": CHAR,
        "history": [],
    }
    plugin = module.create_plugin()
    plugin.service = SimpleNamespace(
        store=SimpleNamespace(
            style_coverage=lambda character: [],
            job=lambda job_id: {"character": CHAR},
            report=lambda job_id: report,
            threshold=lambda character: 0.6,
        )
    )
    result = asyncio.run(plugin.report("job"))
    test_row = next(row for row in result["metrics"] if row["split"] == "测试")
    # At 0.6 only the 0.9 self image fires: recall 1/2, precision 1/1.
    assert test_row["recall"] == "50.0%"
    assert test_row["threshold"] == "0.60"
    assert test_row["threshold_value"] == 0.6
    threshold = result["threshold"]
    assert threshold["current"] == 0.6
    assert threshold["selection"] == EVALUATION_THRESHOLD
    assert threshold["suggested"]["value"] >= 0.5
    assert len(threshold["sweep"]) == 19
    assert "当前判定阈值 0.60" in result["summary"]


def test_report_without_validation_predictions_still_renders():
    report = {
        "selection": {"epoch": 1, "validation": {"precision": 1.0, "precision_numerator": 1, "precision_denominator": 1,
                                                  "recall": 1.0, "other_recall": 1.0, "f1": 1.0, "accuracy": 1.0,
                                                  "balanced_accuracy": 1.0, "confusion_matrix": [[1, 0], [0, 1]], "count": 2}},
        "test": {"precision": 1.0, "precision_numerator": 1, "precision_denominator": 1,
                 "recall": 1.0, "other_recall": 1.0, "f1": 1.0, "accuracy": 1.0,
                 "balanced_accuracy": 1.0, "confusion_matrix": [[1, 0], [0, 1]], "count": 2},
        "historical_test": True,
        "comparable_to_previous": False,
        "total_seconds": 1.0,
        "training_seconds_including_validation": 1.0,
        "audit": {"passed": True},
        "character": CHAR,
    }
    plugin = module.create_plugin()
    plugin.service = SimpleNamespace(
        store=SimpleNamespace(
            style_coverage=lambda character: [],
            job=lambda job_id: {"character": CHAR},
            report=lambda job_id: report,
            threshold=lambda character: 0.6,
        )
    )
    result = asyncio.run(plugin.report("job"))
    assert result["threshold"]["suggested"] is None
    assert result["threshold"]["sweep"] == []
    assert len(result["metrics"]) == 2
