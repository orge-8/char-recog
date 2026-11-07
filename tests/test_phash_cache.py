# -*- coding: utf-8 -*-
"""感知哈希去重（0.7.2）回归测试。

上游 mohobot 用 pHash 复用「同一张图换 URL / 转码」的视觉结果；char-recog
0.7.1 只有 sha256(raw) 精确键，换字节就重跑一次推理。本文件守住三条边界：

1. 感知命中必须复用推理结果，且只在 (模型版本, 阈值, 解码版本) 完全相同的作用域内；
2. 可选依赖缺失时静默退回精确缓存，行为与 0.7.1 一致；
3. 缓存淘汰时 pHash 索引同步清理，不留悬空条目。
"""
import asyncio
import base64
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("maibot_sdk", reason="plugin.py 依赖宿主 SDK maibot_sdk，缺失时整模块跳过")

_PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _load(name, path, package=False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


phash_mod = _load("char_recog_phash_test", _PLUGIN_DIR / "recog" / "phash.py")
module = _load("phash_plugin_test", _PLUGIN_DIR / "plugin.py", package=True)

CHAR = "洛天依"


def hit(score=0.7):
    return {"status": "ok", "character": CHAR, "score": score, "version": "v1"}


def build(results=None, limits=None, seen=None):
    seen = seen if seen is not None else []
    results = results if results is not None else [hit()]
    state = {"limits": dict(limits or {CHAR: 0.6})}

    def infer(raw, timeout=8, character="", *, wait=True):
        seen.append(raw)
        return {"status": "ok", "results": [dict(item) for item in results]}

    plugin = module.create_plugin()
    plugin._plugin_config_instance = module.Config(characters=[CHAR])
    plugin._image_cache = {}
    plugin._image_phash = {}
    plugin.service = SimpleNamespace(
        store=SimpleNamespace(
            active_characters=lambda: [CHAR],
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


# ── 纯函数 ────────────────────────────────────────────────────────

def test_hamming_distance_and_incomparable_inputs():
    assert phash_mod.hamming_distance("00ff", "00ff") == 0
    assert phash_mod.hamming_distance("0f", "f0") == 8
    assert phash_mod.hamming_distance("00", "0000") == 999
    assert phash_mod.hamming_distance("zz", "00") == 999
    assert phash_mod.hamming_distance("", "00") == 999


def test_similar_key_rejects_a_different_cache_scope():
    scope = ((CHAR, "v1"), ((CHAR, 0.6),), 3)
    in_scope = (("sha-a",) + scope, "00" * 8)
    other_model = (("sha-b", (CHAR, "v2"), ((CHAR, 0.6),), 3), "00" * 8)
    assert phash_mod.similar_key("00" * 8, [other_model, in_scope], scope) == in_scope[0]
    assert phash_mod.similar_key("00" * 8, [other_model], scope) is None
    assert phash_mod.similar_key(None, [in_scope], scope) is None


def test_similar_key_threshold_is_strictly_less_than_five():
    scope = ((), (), 1)
    # 4 bits differ -> hit; 5 bits differ -> miss (upstream uses < 5).
    four = (("a",) + scope, "0f" + "00" * 7)
    five = (("b",) + scope, "1f" + "00" * 7)
    assert phash_mod.similar_key("00" * 8, [four], scope) == four[0]
    assert phash_mod.similar_key("00" * 8, [five], scope) is None


def test_compute_phash_is_fail_open_without_optional_dependency(monkeypatch):
    monkeypatch.setattr(phash_mod, "PHASH_AVAILABLE", False)
    assert phash_mod.compute_phash(b"not-an-image") is None
    assert phash_mod.compute_phash(b"") is None


# ── 插件层复用 ────────────────────────────────────────────────────

def test_same_picture_different_bytes_reuses_inference(monkeypatch):
    monkeypatch.setattr(module, "compute_phash", lambda raw: "00" * 8)
    plugin, seen, _ = build()
    assert len(annotate(plugin, b"picture-v1")) == 1
    assert len(annotate(plugin, b"picture-v2")) == 1
    assert len(seen) == 1, "感知相同但字节不同的图必须复用第一次推理结果"
    assert len(plugin._image_cache) == 2
    assert len(plugin._image_phash) == 2


def test_phash_hit_does_not_cross_a_changed_gate(monkeypatch):
    monkeypatch.setattr(module, "compute_phash", lambda raw: "00" * 8)
    plugin, seen, state = build()
    assert len(annotate(plugin, b"picture-v1")) == 1
    state["limits"][CHAR] = 0.9  # 同一张图，判定门槛变了
    assert annotate(plugin, b"picture-v2") == []
    assert len(seen) == 2, "旧门槛下的判定不能被感知命中复用"


def test_phash_hit_does_not_cross_a_changed_decode_version(monkeypatch):
    monkeypatch.setattr(module, "compute_phash", lambda raw: "00" * 8)
    plugin, seen, _ = build()
    assert len(annotate(plugin, b"picture-v1")) == 1
    monkeypatch.setattr(module, "DECODE_VERSION", module.DECODE_VERSION + 1)
    assert len(annotate(plugin, b"picture-v2")) == 1
    assert len(seen) == 2


def test_missing_optional_dependency_falls_back_to_exact_cache(monkeypatch):
    monkeypatch.setattr(module, "compute_phash", lambda raw: None)
    plugin, seen, _ = build()
    assert len(annotate(plugin, b"same")) == 1
    assert len(annotate(plugin, b"same")) == 1
    assert len(seen) == 1, "完全相同的字节仍走精确缓存"
    assert len(annotate(plugin, b"other")) == 1
    assert len(seen) == 2, "降级后换字节必须重新推理，与 0.7.1 一致"
    assert plugin._image_phash == {}


def test_eviction_keeps_phash_index_in_sync(monkeypatch):
    monkeypatch.setattr(module, "compute_phash", lambda raw: "00" * 8)
    plugin, _, _ = build()
    # 塞满 512 条，再进一张新图：FIFO 淘汰时索引必须同步移除。
    scope = (((CHAR, "v1"),), ((CHAR, 0.6),), module.DECODE_VERSION)
    oldest = ("old-0",) + scope
    for i in range(512):
        key = (f"old-{i}",) + scope
        plugin._image_cache[key] = {"status": "ok", "results": [hit()]}
        plugin._image_phash[key] = "11" * 8
    assert len(annotate(plugin, b"fresh")) == 1
    assert len(plugin._image_cache) == 512
    assert len(plugin._image_phash) == 512
    assert oldest not in plugin._image_cache
    assert oldest not in plugin._image_phash
