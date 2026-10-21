"""Per-medium metrics and the advice sentence that names the weak bucket."""

import pytest

from recog.scores import by_style, style_advice


def pred(style, label, score):
    return {"id": f"{style}-{label}-{score}", "label": label, "score": score, "style": style, "neg_type": ""}


def test_groups_by_style_and_computes_recall():
    rows = [pred("flat", "self", 0.9) for _ in range(8)] + [pred("flat", "self", 0.2) for _ in range(2)]
    result = by_style(rows, 0.6)
    assert len(result) == 1
    assert result[0]["style"] == "flat"
    assert result[0]["n"] == 10
    assert result[0]["positives"] == 10
    assert result[0]["recall"] == pytest.approx(0.8)
    assert result[0]["unreliable"] is False


def test_false_positive_rate_needs_negatives():
    rows = [pred("paint", "self", 0.9) for _ in range(6)] + [
        pred("paint", "other", 0.9),
        pred("paint", "other", 0.1),
    ]
    result = by_style(rows, 0.6)[0]
    assert result["negatives"] == 2
    assert result["fpr"] == pytest.approx(0.5)
    # Two negatives cannot support an FPR number, even if the positives could.
    assert result["unreliable"] is True


def test_small_bucket_is_unreliable_but_still_listed():
    rows = [pred("photo", "self", 0.9) for _ in range(3)]
    result = by_style(rows, 0.6)
    assert result[0]["n"] == 3
    assert result[0]["unreliable"] is True


def test_untagged_bucket_sorts_last_and_is_labelled():
    rows = [pred("", "self", 0.9)] + [pred("flat", "self", 0.9)]
    result = by_style(rows, 0.6)
    assert result[-1]["style"] == ""
    assert result[-1]["label"] == "未标注"


def test_advice_names_missing_medium_first():
    coverage = [
        {"style": "flat", "label": "平涂/Q版", "train": 100, "eval": 20, "verdict": "covered"},
        {"style": "photo", "label": "实拍", "train": 0, "eval": 0, "verdict": "missing"},
    ]
    styles = [{"style": "flat", "label": "平涂/Q版", "n": 20, "positives": 20, "negatives": 0, "recall": 0.9, "fpr": 0.0, "precision": 0.9, "unreliable": False}]
    advice = style_advice(coverage, styles)
    assert "实拍" in advice
    assert "没有" in advice


def test_advice_points_at_the_weakest_measured_medium():
    coverage = [
        {"style": "flat", "label": "平涂/Q版", "train": 100, "eval": 20, "verdict": "covered"},
        {"style": "paint", "label": "厚涂插画", "train": 60, "eval": 20, "verdict": "covered"},
    ]
    styles = [
        {"style": "flat", "label": "平涂/Q版", "n": 20, "positives": 20, "negatives": 0, "recall": 0.92, "fpr": 0.0, "precision": 0.9, "unreliable": False},
        {"style": "paint", "label": "厚涂插画", "n": 20, "positives": 20, "negatives": 0, "recall": 0.35, "fpr": 0.0, "precision": 0.8, "unreliable": False},
    ]
    advice = style_advice(coverage, styles)
    assert "厚涂插画" in advice
    assert "35%" in advice
    assert "30 张" in advice


def test_advice_without_eval_set_but_with_tags_points_at_rebuild():
    """264 tagged images plus "tag 20-25 per type" is advice for work already done."""
    advice = style_advice([], [], evaluated=False, labelled=264, buildable=44)
    assert "重建评估集" in advice
    assert "264" in advice
    assert "20–25" not in advice


def test_advice_with_too_few_tags_does_not_promise_an_eval_set():
    advice = style_advice([], [], evaluated=False, labelled=12, buildable=0)
    assert "太少" in advice
    assert "重建评估集" not in advice


def test_advice_without_any_tags_asks_for_tags():
    advice = style_advice([], [], evaluated=False, labelled=0, buildable=0)
    assert "标注内容类型" in advice


def test_advice_with_only_small_buckets_asks_for_labels():
    coverage = [{"style": "flat", "label": "平涂/Q版", "train": 30, "eval": 3, "verdict": "covered"}]
    styles = [{"style": "flat", "label": "平涂/Q版", "n": 3, "positives": 3, "negatives": 0, "recall": 0.0, "fpr": 0.0, "precision": 0.0, "unreliable": True}]
    advice = style_advice(coverage, styles)
    assert "不足 5 张" in advice
    assert "0%" not in advice
