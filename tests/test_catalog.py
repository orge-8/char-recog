import io
import time

import numpy as np
from PIL import Image
import pytest

from recog.images import decode, whole_image, judgment, thumbnail
from recog.store import Store, digest
from recog.service import Service

CHAR = "洛天依"


def upload(store, seed, label=None, variant=False):
    image = Image.fromarray(np.random.default_rng(seed).integers(0, 256, (96, 120, 3), dtype=np.uint8))
    if variant:
        image = image.resize((240, 192))
    path = store.root / "uploads" / f"{seed}-{time.time_ns()}.png"
    image.save(path)
    result = store.ingest(path, character=CHAR)
    if label:
        store.label(result["id"], label)
    return result["id"]


def corpus(root):
    store = Store(root)
    for i in range(40):
        upload(store, i, "self" if i < 20 else "other")
    return store


def test_thumbnail_keeps_detail_within_page_budget(tmp_path):
    import base64

    path = tmp_path / "large.jpg"
    Image.new("RGB", (1280, 720), "red").save(path)
    raw = base64.b64decode(thumbnail(path).split(",", 1)[1])
    assert len(raw) <= 12_000
    with Image.open(io.BytesIO(raw)) as image:
        assert image.size == (512, 288)


def test_upload_label_and_cached_annotation_refresh(tmp_path, monkeypatch):
    import recog.images as images

    store = Store(tmp_path)
    path = store.root / "uploads" / "negative.png"
    image = Image.fromarray(np.random.default_rng(321).integers(0, 256, (96, 120, 3), dtype=np.uint8))
    image.save(path)
    result = store.ingest(path, "other", CHAR)
    rows = store.images(character=CHAR)["rows"]
    assert rows[0]["label"] == "other"
    cached = rows[0]["thumbnail"]
    monkeypatch.setattr(images, "decode", lambda _: (_ for _ in ()).throw(AssertionError("Cache miss")))
    store.label(result["id"], "self")
    assert store.images(character=CHAR)["rows"][0]["thumbnail"] == cached
    assert store.images(character=CHAR)["rows"][0]["label"] == "self"
    store.set_removed(result["id"], True)
    assert store.images(character=CHAR)["rows"] == []
    assert store.images(view="removed", character=CHAR)["rows"][0]["id"] == result["id"]


def test_reupload_reuses_existing_image_and_keeps_label(tmp_path):
    store = Store(tmp_path)
    ident = upload(store, 95, "self")
    with store.db() as db:
        original = dict(db.execute("SELECT * FROM images WHERE id=?", (ident,)).fetchone())
    for recompressed in (False, True):
        incoming = store.root / "uploads" / "renamed.png"
        if recompressed:
            with Image.open(store.root / original["path"]) as image:
                image.save(incoming, compress_level=0)
        else:
            incoming.write_bytes((store.root / original["path"]).read_bytes())
        result = store.ingest(incoming, character=CHAR)
        assert result["already_exists"] and result["id"] == ident
        assert not incoming.exists()
    with store.db() as db:
        assert db.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1
        assert db.execute("SELECT label FROM images WHERE id=?", (ident,)).fetchone()[0] == "self"
    assert len(list((store.root / "images").iterdir())) == 1


@pytest.mark.parametrize("format", ["JPEG", "PNG", "WEBP"])
def test_supported_static_formats(format):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 24), "red").save(buffer, format=format)
    assert decode(buffer.getvalue()).size == (16, 24)


def test_decode_and_letterbox():
    image = Image.new("RGB", (20, 100), "red")
    out = whole_image(image)
    assert out.size == (320, 320) and out.getpixel((0, 0)) == (124, 116, 104)
    buffer = io.BytesIO()
    image.save(buffer, format="GIF")
    with pytest.raises(ValueError):
        decode(buffer.getvalue())


def test_decode_draft_uses_a_reduced_scale_but_keeps_geometry():
    import recog.images as images

    buffer = io.BytesIO()
    Image.new("RGB", (1200, 900), "red").save(buffer, "JPEG")
    raw = buffer.getvalue()
    assert decode(raw).size == (1200, 900)
    # 1200/2 = 600 still covers 320, so libjpeg decodes at 1/2 scale.
    assert decode(raw, 320).size == (600, 450)
    assert decode(raw, 448).size == (600, 450)

    png = io.BytesIO()
    Image.new("RGB", (1200, 900), "red").save(png, "PNG")
    assert decode(png.getvalue(), 320).size == (1200, 900), "非 JPEG 忽略 draft"

    assert images.MAX_EDGE == 448


def test_pixel_guard_runs_before_draft(monkeypatch):
    """draft() rewrites size immediately; the guard must use the original."""
    import recog.images as images

    buffer = io.BytesIO()
    Image.new("RGB", (1200, 900), "red").save(buffer, "JPEG")
    raw = buffer.getvalue()
    monkeypatch.setattr(images, "MAX_PIXELS", 1_000_000)
    with pytest.raises(ValueError, match="4000万像素"):
        decode(raw, 320)


def test_ingest_keeps_the_full_resolution_for_fingerprints(tmp_path):
    """The dedupe hash covers full-size pixels, so ingest must never draft."""
    store = Store(tmp_path)
    path = store.root / "uploads" / "large.png"
    Image.new("RGB", (1200, 900), "orange").save(path)
    ident = store.ingest(path, "self", CHAR)["id"]
    with store.db() as db:
        row = db.execute("SELECT width, height FROM images WHERE id=?", (ident,)).fetchone()
    assert (row["width"], row["height"]) == (1200, 900)


def test_decode_names_the_detected_format():
    """A real library had 32% rejected with a message that did not say what it found."""
    unsupported = io.BytesIO()
    Image.new("RGB", (16, 24), "red").save(unsupported, format="BMP")
    with pytest.raises(ValueError, match="BMP"):
        decode(unsupported.getvalue())

    animated = io.BytesIO()
    frames = [Image.new("RGB", (8, 8), "red"), Image.new("RGB", (8, 8), "blue")]
    frames[0].save(animated, format="GIF", save_all=True, append_images=frames[1:])
    with pytest.raises(ValueError, match="动图"):
        decode(animated.getvalue())


@pytest.mark.parametrize(
    "score,expected",
    [
        (0, "未命中"),
        (0.4, "未命中"),
        (0.4001, "不确定"),
        (0.5, "不确定"),
        (0.5999, "不确定"),
        (0.6, "命中"),
        (1, "命中"),
    ],
)
def test_three_state_thresholds(score, expected):
    assert judgment(score) == expected


def test_solid_duplicate_conflict_and_restore(tmp_path):
    store = Store(tmp_path)
    solid = tmp_path / "uploads" / "solid.png"
    Image.new("RGB", (64, 64), "white").save(solid)
    assert store.ingest(solid, character=CHAR)["reason"] == "solid"
    a = upload(store, 80, "self")
    b = upload(store, 80, "other", variant=True)
    assert {r["reason"] for r in store.images(character=CHAR)["rows"] if r["id"] in {a, b}} == {"label_conflict"}
    store.label(b, "self")
    assert {r["reason"] for r in store.images(character=CHAR)["rows"] if r["id"] in {a, b}} == {"", "duplicate"}
    store.label(a, "self", True)
    assert next(r for r in store.images(character=CHAR)["rows"] if r["id"] == b)["reason"] == ""


def test_snapshot_preserves_split_and_frozen_truth(tmp_path):
    store = corpus(tmp_path)
    with store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        first = store.freeze(db, "first", "fresh", "cpu", CHAR)
    splits = {r["id"]: r["split"] for r in first["images"]}
    assert {s: sum(r["split"] == s for r in first["images"]) for s in ["train", "val", "test"]} == {
        "train": 28,
        "val": 6,
        "test": 6,
    }
    for i in range(40, 46):
        upload(store, i, "self" if i < 43 else "other")
    with store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        second = store.freeze(db, "second", "fresh", "cpu", CHAR)
    assert all(splits[r["id"]] == r["split"] for r in second["images"] if r["id"] in splits)
    target = next(r for r in second["images"] if r["split"] == "test")
    frozen = digest(second)
    store.set_setting(f"last_held_truth:{CHAR}", second["held_truth"])
    store.label(target["id"], "other" if target["label"] == "self" else "self")
    with store.db() as db:
        third = store.freeze(db, "third", "fresh", "cpu", CHAR)
    assert not third["comparable_to_previous"]
    assert digest(second) == frozen
    assert store.images(page=999, character=CHAR)["pagination"]["page"] == 2


def test_bridging_train_and_holdout_is_excluded(tmp_path, monkeypatch):
    store = Store(tmp_path)
    a = upload(store, 1, "self")
    b = upload(store, 2, "other")
    with store.db() as db:
        db.execute("UPDATE groups SET split='train' WHERE id=(SELECT group_id FROM images WHERE id=?)", (a,))
        db.execute("UPDATE groups SET split='test' WHERE id=(SELECT group_id FROM images WHERE id=?)", (b,))
    monkeypatch.setattr("recog.store.similar", lambda a, b: True)
    c = upload(store, 3, "self")
    assert next(r for r in store.images(character=CHAR)["rows"] if r["id"] == c)["reason"] == "split_conflict"
    assert next(r for r in store.images(character=CHAR)["rows"] if r["id"] == a)["split"] == "train"


def test_submission_idempotent_and_missing_runtime(tmp_path):
    corpus(tmp_path)
    service = Service(tmp_path, python=str(tmp_path / "not-installed"))
    try:
        result = service.submit("same", character=CHAR)
        assert result["state"] == "failed" and "环境" in result["error"]
        assert service.submit("same", character=CHAR)["id"] == result["id"]
        assert service.infer(b"anything")["status"] == "no_model"
    finally:
        service.close()


def test_batch_across_pages_atomic_and_remove_all(tmp_path):
    store = corpus(tmp_path)
    first = store.images(1, character=CHAR)["rows"][0]["id"]
    second = store.images(2, character=CHAR)["rows"][0]["id"]
    result = store.batch_update(f"{first},{second}", "other")
    assert result["updated"] == 2
    with store.db() as db:
        assert all(db.execute("SELECT label FROM images WHERE id=?", (i,)).fetchone()[0] == "other" for i in (first, second))
    with pytest.raises(ValueError, match="不存在"):
        store.batch_update(f"{first},{'f' * 32}", "self")
    with store.db() as db:
        assert db.execute("SELECT label FROM images WHERE id=?", (first,)).fetchone()[0] == "other"
    store.batch_update(f"{first},{second}", "remove")
    assert store.images(character=CHAR)["total"] == 38
    assert store.batch_update("", "remove_all", CHAR)["updated"] == 38
    assert store.images(character=CHAR)["total"] == 0
    assert store.images(view="removed", character=CHAR)["total"] == 40
    store.set_removed(first, False)
    assert store.images(character=CHAR)["total"] == 1


def test_batch_accepts_csv_and_json_array(tmp_path):
    """0.3.3 pages sent JSON arrays; the backend must accept both shapes."""
    store = corpus(tmp_path)
    first = store.images(1, character=CHAR)["rows"][0]["id"]
    second = store.images(1, character=CHAR)["rows"][1]["id"]
    assert store.batch_update(f"{first},{second}", "other")["updated"] == 2
    assert store.batch_update(f'["{first}", "{second}"]', "self")["updated"] == 2
    with store.db() as db:
        assert db.execute("SELECT label FROM images WHERE id=?", (first,)).fetchone()[0] == "self"
    with pytest.raises(ValueError, match="请先选择图片"):
        store.batch_update("", "self")
    with pytest.raises(ValueError, match="无效图片ID"):
        store.batch_update("[bad", "self")
    with pytest.raises(ValueError, match="无效图片ID"):
        store.batch_update(f"{first},nothex", "self")


def test_remove_all_is_scoped_to_character(tmp_path):
    store = corpus(tmp_path)
    other = "言和"
    for seed in range(100, 103):
        image = Image.fromarray(np.random.default_rng(seed).integers(0, 256, (96, 120, 3), dtype=np.uint8))
        path = store.root / "uploads" / f"other-{seed}.png"
        image.save(path)
        store.ingest(path, "self", other)
    assert store.images(character=other)["total"] == 3
    assert store.batch_update("", "remove_all", CHAR)["updated"] == 40
    assert store.images(character=CHAR)["total"] == 0
    assert store.images(character=other)["total"] == 3, "其他角色的图库不能被牵连"
    with pytest.raises(ValueError, match="缺少角色名"):
        store.batch_update("", "remove_all", "")
