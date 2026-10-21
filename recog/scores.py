"""Threshold math shared by the training worker and the host process.

This module is imported by plugin.py, which must never import torch. Keep it
dependency-free: no torch, numpy, PIL, or package-local imports.
"""

from math import sqrt

# Two different thresholds live in this plugin and must never be mixed up:
#   EVALUATION_THRESHOLD  how the trainer selects an epoch and audits a report
#   DEFAULT_THRESHOLD     the online gate for prompts and tool judgments
EVALUATION_THRESHOLD = 0.5
DEFAULT_THRESHOLD = 0.6

MIN_THRESHOLD = 0.05
MAX_THRESHOLD = 0.95
# 0.6/0.4 (the 0.3.x behaviour) is exactly threshold / threshold - band.
JUDGMENT_BAND = 0.2

# Content type (medium) is a single axis on purpose: only this axis has a measured
# effect on recall (thick-paint fan art 2% vs flat chibi sticker 91%). The names are
# subject-agnostic so the same four buckets work for game characters, idols or pets.
STYLE_LABELS = {"flat": "平涂/Q版", "paint": "厚涂插画", "photo": "实拍", "composite": "图文/梗图"}
NEG_LABELS = {"other_character": "其他角色", "non_character": "非角色内容", "sticker": "表情包"}
# Below this many eval images a bucket gets no percentage: a 3-image "0%" is noise.
STYLE_MIN_COUNT = 5
# Raw scores from different per-character models are not calibrated against each
# other, so an exact tie is reported as no call at all. Measured on two real
# models (洛天依 + 言和) over 241 confident solo images: at 0.08 this dropped 4.6%
# of correct hits whose top-1 was 0.93–1.00 (the runner-up was always the other
# character's false positive). Top-1 already resolves that conflict, so the
# margin only needs to catch genuine ties.
AMBIGUITY_MARGIN = 0.02
# A threshold below 0.5 asks the model to fire while it prefers "not this one".
SUGGEST_MIN = 0.5
TARGET_PRECISION = 0.90
MIN_SUPPORT = 20
MIN_PER_CLASS = 10


def metrics(labels: list, scores: list, threshold: float = EVALUATION_THRESHOLD) -> dict:
    predicted = [int(score >= threshold) for score in scores]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predicted))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    other_recall = tn / (tn + fp) if tn + fp else 0.0
    return {
        "precision": precision,
        "precision_numerator": tp,
        "precision_denominator": tp + fp,
        "recall": recall,
        "other_recall": other_recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "balanced_accuracy": (recall + other_recall) / 2,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "count": len(labels),
        "threshold": threshold,
    }


def by_style(predictions: list, threshold: float = EVALUATION_THRESHOLD, min_count: int = STYLE_MIN_COUNT) -> list:
    """Per-medium recall / false-positive rate on the fixed evaluation set.

    Recall needs positives, FPR needs negatives; a bucket that only has one side
    reports the other as 0.0 and is marked unreliable, so the UI never renders a
    percentage that no image supports.
    """
    buckets: dict = {}
    for row in predictions:
        buckets.setdefault(row.get("style") or "", []).append(row)
    rows = []
    for style, items in sorted(buckets.items(), key=lambda kv: (kv[0] == "", kv[0])):
        labels = [int(row["label"] == "self") for row in items]
        scores = [row["score"] for row in items]
        summary = metrics(labels, scores, threshold)
        positives = sum(labels)
        negatives = len(labels) - positives
        rows.append(
            {
                "style": style,
                "label": STYLE_LABELS.get(style, "未标注"),
                "n": len(items),
                "positives": positives,
                "negatives": negatives,
                "recall": summary["recall"],
                "fpr": 1 - summary["other_recall"] if negatives else 0.0,
                "precision": summary["precision"],
                "unreliable": bool(
                    len(items) < min_count
                    or (positives and positives < min_count)
                    or (negatives and negatives < min_count)
                ),
            }
        )
    return rows


def style_advice(coverage: list, styles: list, evaluated: bool = True, labelled: int = 0, buildable: int = 0) -> str:
    """One actionable sentence: which medium is missing or weakest, and how many to add.

    "No eval set" has two very different causes and they need different actions:
    nothing is tagged yet (tag more), or the tags exist but the fixed eval set was
    never built (click rebuild). Telling a library with 264 tagged images to "tag
    20-25 per type" sends the user to do work they already did.
    """
    if not evaluated:
        if buildable > 0:
            return (f"已标类型的图有 {labelled} 张，但还没有评估集：到训练页点「重建评估集」，"
                    "训练后就能看到分档结果。")
        if labelled > 0:
            return "已标类型的图还太少，暂时建不出评估集（每类要留 20 组给训练）；先补图或补标。"
        return "还没有分类型评估：先给图标注内容类型，训练后就能看到哪一类认不出。"
    missing = [row["label"] for row in coverage if row.get("verdict") == "missing"]
    usable = [row for row in styles if row["n"] >= STYLE_MIN_COUNT and row["positives"] >= STYLE_MIN_COUNT]
    parts = []
    if missing:
        parts.append("训练集里没有" + "、".join(f"「{name}」" for name in missing) + "类图，遇到这类图基本认不出")
    if usable:
        worst = min(usable, key=lambda row: row["recall"])
        best = max(usable, key=lambda row: row["recall"])
        if worst is not best and best["recall"] - worst["recall"] >= 0.15:
            parts.append(
                f"「{worst['label']}」召回 {worst['recall']:.0%}，"
                f"比「{best['label']}」低 {(best['recall'] - worst['recall']) * 100:.0f} 个百分点，"
                "建议再补 30 张这一类"
            )
        elif worst["recall"] < 0.6:
            parts.append(f"「{worst['label']}」召回只有 {worst['recall']:.0%}，建议再补 30 张这一类")
    if parts:
        return "；".join(parts) + "。"
    if usable:
        return "各类型召回都在同一档，暂时不需要补图。"
    return "评估集里各类型的样本都不足 5 张，先给每类补到 20–25 张再评估。"


def judgment(score: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Three-way label derived from a single gate.

    The band below the threshold keeps the 0.3.x semantics: 0.6 fires, 0.4 or
    lower is a clear miss, and the grey zone in between is "uncertain".
    """
    if not 0.0 < threshold < 1.0:
        threshold = DEFAULT_THRESHOLD
    if score >= threshold:
        return "命中"
    # round() because 0.6 - 0.2 is 0.39999999999999997 in binary floating point,
    # which would push an exact 0.4 out of the "clear miss" band.
    if score <= round(max(0.0, threshold - JUDGMENT_BAND), 6):
        return "未命中"
    return "不确定"


def grid(step: float = 0.05) -> list:
    """Evenly spaced candidate thresholds, ascending."""
    count = int(round((MAX_THRESHOLD - MIN_THRESHOLD) / step))
    return [round(MIN_THRESHOLD + step * index, 2) for index in range(count + 1)]


def sweep(labels: list, scores: list, thresholds=None) -> list:
    """Metrics at every candidate threshold.

    Only needs the persisted {label, score} pairs, so it never re-runs a model.
    """
    return [metrics(labels, scores, value) for value in (thresholds or grid())]


def wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    """Lower bound of the 95% Wilson interval.

    A 3/3 validation result is not a 100% precise model; showing only the point
    estimate would be a lie, so the UI reports this bound next to it.
    """
    if total <= 0:
        return 0.0
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = z * sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - spread)


def suggest_threshold(
    labels: list,
    scores: list,
    target: float = TARGET_PRECISION,
    min_samples: int = MIN_SUPPORT,
    min_per_class: int = MIN_PER_CLASS,
) -> dict:
    """Lowest threshold that still reaches the target precision.

    Taking the lowest such threshold maximises recall among the acceptable
    operating points. Validation data only: tuning on the sealed test split
    would destroy its value as evidence.
    """
    labels = list(labels)
    scores = list(scores)
    positives = sum(1 for value in labels if value == 1)
    negatives = len(labels) - positives
    rows = [row for row in sweep(labels, scores) if row["threshold"] >= SUGGEST_MIN]
    eligible = [row for row in rows if row["precision_denominator"] > 0 and row["precision"] >= target]
    if eligible:
        chosen = eligible[0]
    else:
        candidates = [row for row in rows if row["precision_denominator"] > 0] or rows or [metrics(labels, scores)]
        chosen = max(candidates, key=lambda row: (row["precision"], row["recall"], row["threshold"]))
    reliable = len(labels) >= min_samples and min(positives, negatives) >= min_per_class
    return {
        "value": chosen["threshold"],
        "target_precision": target,
        "target_met": bool(eligible),
        "precision": chosen["precision"],
        "precision_numerator": chosen["precision_numerator"],
        "precision_denominator": chosen["precision_denominator"],
        "recall": chosen["recall"],
        "f1": chosen["f1"],
        "count": len(labels),
        "positive": positives,
        "negative": negatives,
        "reliable": reliable,
        "confidence_lower": wilson_lower(chosen["precision_numerator"], chosen["precision_denominator"]),
        "reason": ""
        if reliable
        else f"验证集仅{len(labels)}张（正例{positives}/负例{negatives}），建议值仅作参考",
    }


def _limit(result: dict, thresholds: dict) -> float:
    """The gate actually applied to this result, preferring the worker's value."""
    value = result.get("threshold")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return float(thresholds.get(result.get("character"), DEFAULT_THRESHOLD))


def select_top(results: list, thresholds: dict, margin: float = AMBIGUITY_MARGIN) -> dict:
    """Pick the single character worth printing, if any.

    Candidates are only the characters that clear their own gate. A character
    whose threshold the admin raised out of reach must not permanently shadow a
    different character that does clear its gate.
    """
    scored = [
        result
        for result in results or []
        if result.get("status") == "ok" and isinstance(result.get("score"), (int, float))
    ]
    ranked = sorted(scored, key=lambda result: result["score"], reverse=True)
    hits = [result for result in ranked if result["score"] >= _limit(result, thresholds)]
    if not hits:
        return {"top1": None, "margin": None, "ambiguous": False}
    top = hits[0]
    gap = top["score"] - hits[1]["score"] if len(hits) > 1 else None
    limit = _limit(top, thresholds)
    return {
        "top1": {
            "character": top.get("character", ""),
            "score": top["score"],
            "threshold": limit,
            "judgment": judgment(top["score"], limit),
        },
        "margin": gap,
        "ambiguous": bool(gap is not None and gap < margin),
    }
