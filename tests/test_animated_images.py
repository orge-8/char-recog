"""动图支持（0.7.0）：容器放行、首帧语义、原始字节去重、聊天捕获。

口径来自 `docs/decisions-animated-images.md`：
- D7 格式边界 = GIF + 动态 WebP + APNG（MP4/WebM 不在范围）。
- D19 在线只取首帧，不做灰区补帧；`decode()` 不得读 `n_frames`。
- D20 原始动图存一份；首帧负责指纹/缩略图/训练读取；精确去重认原始字节，
  避免「首帧相同、后续帧不同」的变体被误删。

夹具全部现写（64×48、5 帧、100ms），不从 `fabiocsa/maibot-read-gif` 逐字复制，
因此 `NOTICE.md` 不需要新增声明。
"""

import base64
import io

import numpy as np
from PIL import Image
import pytest

from recog import images as images_mod
from recog.images import DECODE_VERSION, decode, decode_with_animation
from recog.store import Store

CHAR = "洛天依"
SIZE = (64, 48)


def animation(format="GIF", frames=5, colors=None):
    """A real multi-frame file in the given container.

    WebP 默认走有损（首帧 255,0,0 会变成 255,0,2），断言首帧颜色时要求无损；
    与插件无关，只是让「是不是第一帧」这个判断不被编码噪声干扰。
    """
    colors = colors or [(i * 40, 60, 120) for i in range(frames)]
    images = [Image.new("RGB", SIZE, color) for color in colors]
    buffer = io.BytesIO()
    options = {"lossless": True, "quality": 100} if format == "WEBP" else {}
    images[0].save(
        buffer,
        format=format,
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
        **options,
    )
    return buffer.getvalue()


def textured(seeds=(0, 1, 2, 3, 4), format="GIF"):
    """多帧随机纹理。纯色帧会被 ingest 判成 `solid`，聊天路径直接跳过，
    所以「聊天动图能不能进审核队列」只能用有纹理的帧验证。"""
    images = [
        Image.fromarray(np.random.default_rng(seed).integers(0, 256, (SIZE[1], SIZE[0], 3), dtype=np.uint8))
        for seed in seeds
    ]
    buffer = io.BytesIO()
    options = {"lossless": True, "quality": 100} if format == "WEBP" else {}
    images[0].save(
        buffer, format=format, save_all=True, append_images=images[1:], duration=100, loop=0, **options
    )
    return buffer.getvalue()


def upload(store, raw, suffix=".gif", name=None):
    """Store requires the file to sit under its own uploads/ directory."""
    path = store.root / "uploads" / (name or f"{abs(hash(raw))}-{len(raw)}{suffix}")
    path.write_bytes(raw)
    return path


@pytest.mark.parametrize("format", ["GIF", "PNG", "WEBP"])
def test_first_frame_is_what_every_container_decodes_to(format):
    """APNG 容器仍是 PNG、动态 WebP 仍是 WEBP，所以三者都靠 seek(0) 落到首帧。"""
    raw = animation(format, colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (9, 9, 9), (8, 8, 8)])
    image, animated = decode_with_animation(raw)
    assert animated is True
    assert image.size == SIZE
    assert image.getpixel((0, 0)) == (255, 0, 0), "首帧必须是第一帧，不是最后一帧或合成帧"


def test_is_animated_agrees_with_n_frames_on_every_container():
    """`is_animated` 是入库唯一依据；它必须与 `n_frames > 1` 等价，否则标识会错。"""
    for format in ("GIF", "PNG", "WEBP"):
        raw = animation(format, frames=6)
        with Image.open(io.BytesIO(raw)) as source:
            assert source.is_animated == (source.n_frames > 1), format
            assert source.n_frames == 6
        assert decode_with_animation(raw)[1] is True, format


def test_decode_does_not_report_animation_but_returns_the_same_pixels():
    """`decode()` 是给 worker/训练用的：返回首帧，不暴露动画语义。"""
    raw = animation("GIF")
    plain = decode(raw)
    flagged, animated = decode_with_animation(raw)
    assert animated is True
    assert plain.tobytes() == flagged.tobytes()


def test_single_frame_gif_is_not_treated_as_animation():
    """与 read-gif 相反：单帧 GIF 不算动图，不加 `animated:` 前缀。"""
    buffer = io.BytesIO()
    Image.new("RGB", SIZE, (10, 20, 30)).save(buffer, format="GIF")
    image, animated = decode_with_animation(buffer.getvalue())
    assert animated is False
    assert image.size == SIZE


def test_extension_does_not_decide_the_format(tmp_path):
    """真实库里 551/583 个 GIF 顶着 .jpg 扩展名，解码必须只看内容。"""
    store = Store(tmp_path)
    raw = animation("GIF")
    result = store.ingest(upload(store, raw, suffix=".jpg"), "self", CHAR)
    with store.db() as db:
        row = db.execute("SELECT sha, decoded, width, height FROM images WHERE id=?", (result["id"],)).fetchone()
    assert row["decoded"] == "animated:" + row["sha"]
    assert (row["width"], row["height"]) == SIZE


def test_oversized_animation_is_rejected_before_decoding_pixels(monkeypatch):
    monkeypatch.setattr(images_mod, "MAX_PIXELS", 100)
    with pytest.raises(ValueError, match="4000万像素"):
        decode(animation("GIF"))


def test_ingest_keeps_the_original_bytes_and_a_static_thumbnail(tmp_path):
    """D20/D21：原始动图原样落盘，宽高与缩略图取首帧。"""
    store = Store(tmp_path)
    raw = animation("GIF", colors=[(255, 0, 0), (0, 0, 255), (0, 255, 0), (1, 2, 3), (4, 5, 6)])
    ident = store.ingest(upload(store, raw), "self", CHAR)["id"]

    stored = store.root / "images"
    files = [path for path in stored.iterdir() if path.stem == ident]
    assert len(files) == 1 and files[0].read_bytes() == raw, "原始动图必须逐字节保留"

    row = store.images(character=CHAR)["rows"][0]
    preview = base64.b64decode(row["thumbnail"].split(",", 1)[1])
    with Image.open(io.BytesIO(preview)) as thumbnail:
        assert thumbnail.format == "JPEG", "审核页只显示静态首帧缩略图"
        # JPEG 有损：允许每通道 ±3 的编码噪声，只判断「是首帧」而不是「是最后一帧」
        assert max(abs(a - b) for a, b in zip(thumbnail.getpixel((0, 0)), (255, 0, 0))) <= 3


def test_two_animations_sharing_a_first_frame_are_both_kept(tmp_path):
    """首帧相同、后续帧不同 = 两个真变体，不能因为像素指纹相同被删掉。"""
    store = Store(tmp_path)
    first = textured(seeds=(0, 1, 2, 3, 4))
    second = textured(seeds=(0, 9, 8, 7, 6))
    assert first != second

    one = store.ingest(upload(store, first, name="one.gif"), "self", CHAR)
    two = store.ingest(upload(store, second, name="two.gif"), "self", CHAR)
    assert "already_exists" not in two
    with store.db() as db:
        rows = db.execute("SELECT id, sha, decoded FROM images ORDER BY id").fetchall()
    assert {row["id"] for row in rows} == {one["id"], two["id"]}
    assert len({row["decoded"] for row in rows}) == 2
    assert all(row["decoded"].startswith("animated:") for row in rows)


def test_identical_bytes_are_still_a_duplicate(tmp_path):
    store = Store(tmp_path)
    raw = animation("GIF")
    first = store.ingest(upload(store, raw, name="a.gif"), "self", CHAR)
    second = store.ingest(upload(store, raw, name="b.gif"), "self", CHAR)
    assert second.get("already_exists") is True
    assert second["id"] == first["id"]


def test_animated_capture_enters_the_review_queue_unlabelled(tmp_path):
    """D17：聊天动图以 label=NULL 进审核队列，绝不自动成为训练真值。"""
    store = Store(tmp_path)
    # 纯色帧会被 ingest 判成 solid 并在聊天路径直接跳过，这里用有纹理的帧。
    created = store.ingest(upload(store, textured()), None, CHAR, "chat", 0.71)
    assert created.get("skipped") is None, f"聊天动图被跳过：{created}"
    with store.db() as db:
        row = db.execute("SELECT label, source, model_score FROM images WHERE id=?", (created["id"],)).fetchone()
    assert row["label"] is None
    assert row["source"] == "chat"
    assert row["model_score"] == pytest.approx(0.71)
    assert store.capture_pending(CHAR) == 1


def test_corrupt_animation_never_reaches_the_catalog(tmp_path):
    store = Store(tmp_path)
    payload = b"GIF89a" + b"\x00" * 40
    path = upload(store, payload, name="broken.gif")
    with pytest.raises(Exception):
        store.ingest(path, "self", CHAR)
    with store.db() as db:
        assert db.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 0


def test_decode_version_is_exposed_for_cache_invalidation():
    """plugin.py 的识别缓存把 DECODE_VERSION 并进 key；常量必须存在且为正整数。"""
    assert isinstance(DECODE_VERSION, int) and DECODE_VERSION >= 1
