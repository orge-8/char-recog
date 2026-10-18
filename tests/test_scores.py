"""Threshold math, suggestions and aggregation — all torch-free."""

import subprocess
import sys
from pathlib import Path

import pytest

from recog import scores


def legacy_metrics(labels, scores_list):
    """0.3.4 implementation, kept here as the reference for the 0.5 comparison."""
    predicted = [int(s >= 0.5) for s in scores_list]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predicted))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "precision_numerator": tp,
        "precision_denominator": tp + fp,
        "recall": recall,
        "other_recall": tn / (tn + fp) if tn + fp else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "balanced_accuracy": (recall + (tn / (tn + fp) if tn + fp else 0.0)) / 2,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "count": len(labels),
    }


def test_metrics_at_half_matches_the_legacy_function_exactly():
    labels = [1, 1, 0, 1, 0, 0, 1, 0]
    scores_list = [0.9, 0.4, 0.7, 0.6, 0.55, 0.2, 0.5, 0.49]
    current = scores.metrics(labels, scores_list)
    reference = legacy_metrics(labels, scores_list)
    assert {k: v for k, v in current.items() if k != "threshold"} == reference
    assert current["threshold"] == 0.5
    assert current == scores.metrics(labels, scores_list, 0.5)


def test_metrics_threshold_moves_the_confusion_matrix():
    labels = [1, 1, 0]
    scores_list = [0.9, 0.7, 0.3]
    at_half = scores.metrics(labels, scores_list, 0.5)
    at_eighty = scores.metrics(labels, scores_list, 0.8)
    assert at_half["confusion_matrix"] == [[1, 0], [0, 2]]
    assert at_eighty["confusion_matrix"] == [[1, 0], [1, 1]]
    assert at_eighty["recall"] == 0.5
    assert at_eighty["precision"] == 1.0


def test_scores_module_never_imports_torch():
    code = (
        "import sys; import recog.scores; "
        "assert 'torch' not in sys.modules, 'scores pulled torch'; "
        "assert 'PIL' not in sys.modules, 'scores pulled PIL'"
    )
    subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parents[1], check=True)


def test_sweep_is_nineteen_rows_and_half_matches_metrics():
    labels = [1, 0, 1, 0]
    scores_list = [0.9, 0.2, 0.55, 0.45]
    rows = scores.sweep(labels, scores_list)
    assert len(rows) == 19
    assert rows[0]["threshold"] == 0.05 and rows[-1]["threshold"] == 0.95
    half = next(row for row in rows if row["threshold"] == 0.5)
    assert half == scores.metrics(labels, scores_list, 0.5)
    for row in rows:
        assert {"threshold", "precision", "recall", "f1", "precision_denominator"} <= set(row)


def test_sweep_handles_empty_and_single_class():
    assert scores.metrics([], [])["count"] == 0
    assert scores.metrics([], [])["accuracy"] == 0.0
    for labels in ([1, 1], [0, 0]):
        rows = scores.sweep(labels, [0.9, 0.1])
        assert len(rows) == 19
        assert all(row["count"] == 2 for row in rows)


def test_suggest_picks_the_lowest_threshold_that_reaches_target_precision():
    # Negatives top out at 0.6, so 0.65 is the first grid point with zero false
    # positives - and the lowest such point keeps recall at 1.0.
    labels = [1, 1, 1, 1, 0, 0]
    scores_list = [0.9, 0.85, 0.8, 0.75, 0.6, 0.5]
    suggestion = scores.suggest_threshold(labels, scores_list)
    assert suggestion["value"] == 0.65
    assert suggestion["target_met"] is True
    assert suggestion["precision"] == 1.0
    assert suggestion["recall"] == 1.0
    assert suggestion["precision_denominator"] == 4


def test_suggest_never_goes_below_the_model_decision_boundary():
    labels = [1, 1, 0, 0]
    scores_list = [0.9, 0.8, 0.1, 0.05]
    suggestion = scores.suggest_threshold(labels, scores_list)
    assert suggestion["value"] >= scores.SUGGEST_MIN
    assert suggestion["value"] == 0.5


def test_suggest_marks_small_validation_as_unreliable():
    labels = [1, 1, 1, 0, 0, 0]
    scores_list = [0.9, 0.8, 0.75, 0.3, 0.2, 0.4]
    suggestion = scores.suggest_threshold(labels, scores_list)
    assert suggestion["reliable"] is False
    assert suggestion["count"] == 6
    assert (suggestion["positive"], suggestion["negative"]) == (3, 3)
    assert "参考" in suggestion["reason"]
    assert suggestion["confidence_lower"] < 0.5


def test_suggest_does_not_claim_a_target_it_cannot_reach():
    labels = [1, 1, 0, 0]
    scores_list = [0.9, 0.8, 0.95, 0.85]
    suggestion = scores.suggest_threshold(labels, scores_list)
    assert suggestion["target_met"] is False
    assert suggestion["value"] >= scores.SUGGEST_MIN
    assert suggestion["precision"] == 0.5


def test_wilson_lower_known_values():
    assert scores.wilson_lower(3, 3) == pytest.approx(0.4385, abs=1e-3)
    assert scores.wilson_lower(0, 3) == 0.0
    assert scores.wilson_lower(0, 0) == 0.0


@pytest.mark.parametrize(
    "score,threshold,expected",
    [
        (0.6, 0.6, "命中"),
        (0.5999, 0.6, "不确定"),
        (0.4, 0.6, "未命中"),
        (0.79, 0.8, "不确定"),
        (0.6, 0.8, "未命中"),
        (0.9, 0.8, "命中"),
        (0.0, 0.05, "未命中"),
    ],
)
def test_judgment_band_derives_from_a_single_threshold(score, threshold, expected):
    assert scores.judgment(score, threshold) == expected


def test_judgment_default_keeps_the_034_behaviour():
    for score, expected in [(0, "未命中"), (0.4, "未命中"), (0.5, "不确定"), (0.6, "命中"), (1, "命中")]:
        assert scores.judgment(score) == expected


def test_judgment_falls_back_on_an_invalid_threshold():
    assert scores.judgment(0.65, 0) == "命中"


def test_select_top_keeps_only_the_best_accepted_character():
    results = [
        {"status": "ok", "character": "A", "score": 0.9},
        {"status": "ok", "character": "B", "score": 0.6},
    ]
    policy = scores.select_top(results, {"A": 0.6, "B": 0.6})
    assert policy["top1"]["character"] == "A"
    assert policy["margin"] == pytest.approx(0.3)
    assert policy["ambiguous"] is False


def test_select_top_suppresses_a_close_call():
    results = [
        {"status": "ok", "character": "A", "score": 0.85},
        {"status": "ok", "character": "B", "score": 0.80},
    ]
    policy = scores.select_top(results, {})
    assert policy["margin"] == pytest.approx(0.05)
    assert policy["ambiguous"] is True


def test_select_top_suppresses_a_tie():
    results = [
        {"status": "ok", "character": "A", "score": 0.8},
        {"status": "ok", "character": "B", "score": 0.8},
    ]
    assert scores.select_top(results, {})["ambiguous"] is True


def test_select_top_single_result_has_no_margin():
    policy = scores.select_top([{"status": "ok", "character": "A", "score": 0.7}], {})
    assert policy["top1"]["character"] == "A"
    assert policy["margin"] is None
    assert policy["ambiguous"] is False


def test_select_top_returns_nothing_when_all_are_below_their_gate():
    results = [
        {"status": "ok", "character": "A", "score": 0.59},
        {"status": "ok", "character": "B", "score": 0.3},
    ]
    policy = scores.select_top(results, {"A": 0.6, "B": 0.6})
    assert policy == {"top1": None, "margin": None, "ambiguous": False}


def test_select_top_does_not_let_an_unreachable_gate_shadow_another_character():
    # A is the raw top-1 but its admin gate (0.8) is out of reach. B still clears
    # its own gate, so B is the only acceptable answer - not "ambiguous".
    results = [
        {"status": "ok", "character": "A", "score": 0.75},
        {"status": "ok", "character": "B", "score": 0.70},
    ]
    policy = scores.select_top(results, {"A": 0.8, "B": 0.6})
    assert policy["top1"]["character"] == "B"
    assert policy["margin"] is None
    assert policy["ambiguous"] is False


def test_select_top_prefers_the_threshold_reported_by_the_worker():
    results = [{"status": "ok", "character": "A", "score": 0.7, "threshold": 0.8}]
    assert scores.select_top(results, {"A": 0.6})["top1"] is None


def test_select_top_ignores_failed_models():
    results = [
        {"status": "model_error", "character": "A", "error": "boom"},
        {"status": "ok", "character": "B", "score": 0.9},
    ]
    assert scores.select_top(results, {})["top1"]["character"] == "B"
    assert scores.select_top([], {})["top1"] is None
