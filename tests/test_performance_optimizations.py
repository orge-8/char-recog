"""性能与体积优化的行为门禁。

These tests deliberately assert observable invariants (one wire thread, no eager
model loads, one thumbnail scan per page, bounded cache, one physical copy) rather
than wall-clock thresholds. Absolute timings differ between hosts; the invariants
are what prevent a regression from silently reintroducing the old work.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("torch", reason="预取等价性需要真实 Dataset/张量")
import torch  # noqa: E402

from recog.model import Dataset, _prefetched_batches  # noqa: E402
from recog.service import Service  # noqa: E402
from recog.store import Store  # noqa: E402


def _image(seed: int, size=(96, 120)) -> Image.Image:
    return Image.fromarray(
        np.random.default_rng(seed).integers(0, 256, (*size[::-1], 3), dtype=np.uint8)
    )


def _service(tmp_path: Path, monkeypatch, code: str) -> Service:
    service = Service(tmp_path, sys.executable)

    def spawn(*args, **kwargs):
        with (tmp_path / "inference.log").open("a", encoding="utf-8") as log:
            return subprocess.Popen(
                [sys.executable, "-u", "-c", code],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                encoding="utf-8",
            )

    monkeypatch.setattr(service, "_spawn", spawn)
    return service


def test_inference_uses_one_persistent_wire_thread(tmp_path, monkeypatch):
    code = (
        "import sys\n"
        "for line in sys.stdin:\n"
        "    print('{\"status\":\"ready\"}', flush=True)\n"
    )
    service = _service(tmp_path, monkeypatch, code)
    try:
        assert service.request({"operation": "load"}, timeout=3)["status"] == "ready"
        wire = service._wire
        assert wire is not None and wire.thread.is_alive()
        for _ in range(5):
            assert service.request({"operation": "load"}, timeout=3)["status"] == "ready"
        assert service._wire is wire, "请求之间重建了 I/O 通道"
        readers = [t for t in threading.enumerate() if t.name == "infer-wire"]
        assert len(readers) == 1, f"常驻通道线程数量不为 1：{readers}"
    finally:
        service.close()


def test_warmup_does_not_preload_every_active_model(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.set_active("角色甲", "a" * 32)
    store.set_active("角色乙", "b" * 32)
    trace = tmp_path / "frames.log"
    code = (
        "import sys\n"
        f"log = open({str(trace)!r}, 'a', encoding='utf-8')\n"
        "for line in sys.stdin:\n"
        "    log.write(line)\n"
        "    log.flush()\n"
        "    print('{\"status\":\"ready\"}', flush=True)\n"
    )
    service = _service(tmp_path, monkeypatch, code)
    try:
        assert service.request({"operation": "load", "character": "角色甲"}, timeout=3)["status"] == "ready"
        import json

        lines = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 2, f"预热阶段发送了多余请求：{lines}"
        assert lines[0]["operation"] == "ping"
        assert lines[1]["character"] == "角色甲"
    finally:
        service.close()


def test_thumbnail_trim_runs_once_per_gallery_page(tmp_path, monkeypatch):
    store = Store(tmp_path)
    for seed in range(25):
        path = store.root / "uploads" / f"{seed}.png"
        _image(seed).save(path)
        store.ingest(path, "self", "洛天依")
    calls = []
    monkeypatch.setattr(store, "_trim_thumbnails", lambda: calls.append(1) or 0)
    rows = store.images(page=1, character="洛天依")["rows"]
    assert len(rows) == 24
    assert len(calls) == 1, f"一页触发了 {len(calls)} 次缩略图清理"


def test_thumbnail_cache_is_bounded_by_bytes_and_count(tmp_path, monkeypatch):
    store = Store(tmp_path)
    cache = store.root / "thumbnails"
    cache.mkdir()
    for index in range(5):
        path = cache / f"{index}.jpg"
        path.write_bytes(b"x" * 100)
        os.utime(path, (time.time() - 100 + index, time.time() - 100 + index))
    monkeypatch.setattr(Store, "THUMBNAIL_LIMIT", 100)
    monkeypatch.setattr(Store, "THUMBNAIL_BYTES", 250)
    removed = store._trim_thumbnails()
    remaining = list(cache.iterdir())
    assert removed == 3
    assert len(remaining) == 2
    assert sum(path.stat().st_size for path in remaining) <= 250
    # The survivors are the newest entries, i.e. LRU rather than arbitrary.
    assert {path.name for path in remaining} == {"3.jpg", "4.jpg"}


def test_duplicate_content_is_one_physical_file(tmp_path):
    store = Store(tmp_path)
    payload = tmp_path / "same.png"
    _image(7).save(payload)
    raw = payload.read_bytes()
    first = store.root / "uploads" / "a.png"
    second = store.root / "uploads" / "b.png"
    first.write_bytes(raw)
    second.write_bytes(raw)
    store.ingest(first, "self", "角色甲")
    store.ingest(second, "self", "角色乙")
    paths = [store.root / row["path"] for row in store.images(character="角色甲")["rows"]]
    paths += [store.root / row["path"] for row in store.images(character="角色乙")["rows"]]
    assert len(paths) == 2
    assert os.path.samefile(paths[0], paths[1]), "同一内容仍占两份物理文件"
    stats = store.storage_stats()
    assert stats["images_saved"] >= len(raw)


def test_existing_duplicate_files_are_consolidated_without_row_changes(tmp_path):
    store = Store(tmp_path)
    first = store.root / "uploads" / "one.png"
    _image(11).save(first)
    image_id = store.ingest(first, "self", "角色甲")["id"]
    with store.db() as db:
        row = dict(db.execute("SELECT path,sha FROM images WHERE id=?", (image_id,)).fetchone())
    path = store.root / row["path"]
    # Simulate a pre-0.8 row with an independent copy and no blob.
    path.unlink()
    _image(11).save(path)
    blob = store.blob_path(row["sha"])
    if blob.exists():
        blob.unlink()
    assert store.ensure_blob_links()["linked"] == 1
    assert blob.is_file()
    assert os.path.samefile(path, blob)
    with store.db() as db:
        assert db.execute("SELECT COUNT(*) FROM images WHERE id=?", (image_id,)).fetchone()[0] == 1


def test_cache_cleanup_never_deletes_referenced_or_recent_data(tmp_path):
    store = Store(tmp_path)
    upload = store.root / "uploads" / "kept.png"
    _image(21).save(upload)
    store.ingest(upload, "self", "洛天依")
    with store.db() as db:
        sha = db.execute("SELECT sha FROM images LIMIT 1").fetchone()[0]
    referenced = store.blob_path(sha)
    assert referenced.is_file()

    downloads = store.root / "dependencies" / "downloads"
    downloads.mkdir(parents=True)
    old_wheel = downloads / "torch.whl"
    old_wheel.write_bytes(b"old")
    recent_part = downloads / "torchvision.whl.part"
    recent_part.write_bytes(b"recent")

    orphan_old = store.root / "blobs" / "ff" / ("f" * 64)
    orphan_recent = store.root / "blobs" / "ee" / ("e" * 64)
    orphan_old.parent.mkdir(parents=True)
    orphan_recent.parent.mkdir(parents=True)
    orphan_old.write_bytes(b"orphan")
    orphan_recent.write_bytes(b"orphan")
    now = time.time()
    old = now - 40 * 86400
    os.utime(old_wheel, (old, old))
    os.utime(orphan_old, (old, old))

    result = store.cleanup_caches(now=now)
    assert result["files"] >= 2
    assert not old_wheel.exists()
    assert recent_part.exists()
    assert referenced.exists(), "被图片记录引用的 blob 被误删"
    assert not orphan_old.exists()
    assert orphan_recent.exists(), "刚写入、可能仍在入库中的 blob 被误删"


def test_prefetched_batches_preserve_serial_output(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    rows = []
    for index in range(7):
        _image(100 + index).save(root / f"{index}.png")
        rows.append({"path": f"{index}.png", "label": "self" if index % 2 else "other", "id": str(index)})
    dataset = Dataset(root, rows, augment=False, input_size=224)
    serial = [dataset[index] for index in range(7)]
    prefetched = []
    batches = _prefetched_batches(dataset, list(range(7)), batch=3, offset=0, steps=3)
    try:
        for step, loaded in batches:
            prefetched.extend(dataset.render(item) for item in loaded)
    finally:
        batches.close()
    assert len(prefetched) == len(serial) == 7
    for (left, left_label), (right, right_label) in zip(serial, prefetched):
        assert left_label == right_label
        assert torch.equal(left, right)


def test_gallery_polling_is_activity_driven():
    source = (Path(__file__).parents[1] / "webui" / "index.html").read_text(encoding="utf-8")
    assert "schedulePoll();" in source
    assert "setInterval(() =>" not in source, "固定 5 秒轮询又被加回来了"
    assert "function pollingNeeded()" in source
