"""Whole-image preprocessing and duplicate fingerprints, adapted from Mai_recog_self."""

from pathlib import Path
import base64
import hashlib
import io
import math
import warnings
import uuid

from PIL import Image, ImageOps, ImageStat

from .scores import judgment  # re-export: per-character thresholds live in scores.py

MAX_BYTES = 20 * 1024 * 1024
# base64 会把体积放大 4/3，所以「二进制闸门 × 4/3」才是 base64 字符串的上限。
# 0.6.1 之前 plugin.py 里硬编码 28 MiB，换算成原始字节是 22.1 MiB，已经越过
# MAX_BYTES —— 两个口径不一致，改任何一边都会让另一边变成错的。统一从这里推导。
MAX_BASE64_BYTES = MAX_BYTES * 4 // 3
MAX_PIXELS = 40_000_000
# 与 store.freeze 允许的输入尺寸保持一致；解码时按它 draft，避免二次放大。
MAX_EDGE = 448
# 图像语义版本。只要「同一份字节解出来的张量」发生变化（例如从「动图直接拒收」
# 改成「取首帧」），就必须递增：plugin.py 的识别缓存与库里的 decoded 指纹都以它
# 为前提，不递增会让旧语义下的判定被继续复用。
DECODE_VERSION = 1
# APNG 的容器格式仍是 PNG，动态 WebP 的仍是 WEBP，所以这里是容器白名单而不是动画白名单。
SUPPORTED_FORMATS = ("JPEG", "PNG", "WEBP", "GIF")


def _decode(raw: bytes, draft_edge: int | None = None) -> tuple[Image.Image, bool]:
    """Decode to RGB and report whether the source is animated.

    `draft_edge` lets libjpeg decode a JPEG at a reduced scale. Drafting skips the
    full-size IDCT and upsampling (12MP to 320 input drops from ~302 ms to ~25 ms
    end to end) but keeps the original pixels' geometry. Only JPEG honours it;
    other formats decode normally.

    Animation is handled by `seek(0)`: every caller gets the first frame. This is a
    deliberate semantic, not a limitation — 509 animated illustrations measured on
    0.6.x showed the first frame and a 5-frame median differ in verdict on only
    11/509 (2.2%), while "any frame above the gate" raised false positives from 32
    to 52. Reading `is_animated` costs ~0.14 ms on a real GIF and does not scan
    frames, so the online path never pays for frames it will not use.
    """
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("文件为空或超过20MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                # Name what was actually detected: a survey of a real library found
                # 32% of files rejected here, mostly animated GIFs renamed to .jpg,
                # and "仅支持JPEG/PNG/静态WebP" left the admin unable to tell why.
                if source.format not in SUPPORTED_FORMATS:
                    raise ValueError(
                        f"实际格式为 {source.format or '未知'}，仅支持 JPEG/PNG/WebP/GIF"
                    )
                # Original dimensions, read before draft: draft() rewrites size
                # immediately, so checking afterwards would let an oversized JPEG
                # through the pixel guard.
                width, height = source.size
                if width * height > MAX_PIXELS:
                    raise ValueError("解码超过4000万像素")
                animated = bool(getattr(source, "is_animated", False))
                if draft_edge and source.format == "JPEG":
                    source.draft("RGB", (draft_edge, draft_edge))
                source.seek(0)
                source.load()
                return ImageOps.exif_transpose(source).convert("RGB"), animated
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ValueError("图片解码尺寸超限") from exc


def decode(raw: bytes, draft_edge: int | None = None) -> Image.Image:
    """First frame only — the online and training path both use this."""
    return _decode(raw, draft_edge)[0]


def decode_with_animation(raw: bytes, draft_edge: int | None = None) -> tuple[Image.Image, bool]:
    """Same decode, plus the animation flag. Only `Store.ingest` needs the flag:
    it decides whether the row's exact-duplicate fingerprint may key on pixels."""
    return _decode(raw, draft_edge)


def whole_image(image: Image.Image, edge: int = 320) -> Image.Image:
    image = image.copy()
    image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (edge, edge), (124, 116, 104))
    canvas.paste(image, ((edge - image.width) // 2, (edge - image.height) // 2))
    return canvas


def fingerprint(image: Image.Image) -> dict:
    # DCT pHash is invariant to re-encoding and resizing; no extra SciPy dependency.
    import numpy as np

    pixels = np.asarray(image.convert("L").resize((32, 32)), dtype=float)
    n = np.arange(32)
    basis = np.cos(np.pi / 32 * (n + 0.5)[None, :] * n[:, None])
    dct = (basis @ pixels @ basis.T)[:8, :8].flatten()
    bits = dct > np.median(dct[1:])
    return {
        "decoded": hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest(),
        "phash": f"{sum(int(bit) << i for i, bit in enumerate(bits)):016x}",
        "aspect": image.width / image.height,
        "width": image.width,
        "height": image.height,
        "solid": max(ImageStat.Stat(image.resize((64, 64))).stddev) < 2,
    }


def similar(a: dict, b: dict) -> bool:
    return a["decoded"] == b["decoded"] or (
        (int(a["phash"], 16) ^ int(b["phash"], 16)).bit_count() <= 4 and abs(math.log(a["aspect"] / b["aspect"])) < 0.12
    )


def thumbnail(path: Path, cache: Path | None = None) -> str:
    if cache is not None and cache.is_file():
        return "data:image/jpeg;base64," + base64.b64encode(cache.read_bytes()).decode()
    image = decode(path.read_bytes(), 512)
    # Keep a 24-image page below the gateway's 512KiB response budget.
    for edge in (512, 384, 320, 256, 192, 128):
        preview = image.copy()
        preview.thumbnail((edge, edge))
        for quality in (80, 65, 50):
            buffer = io.BytesIO()
            preview.save(buffer, format="JPEG", quality=quality, optimize=True)
            if buffer.tell() <= 12_000:
                if cache is not None:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    temp = cache.with_suffix("." + uuid.uuid4().hex + ".tmp")
                    temp.write_bytes(buffer.getvalue())
                    temp.replace(cache)
                return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    raise ValueError("缩略图超过响应大小预算")
