"""Repeatable local benchmark for the optimisations shipped in 0.8.0.

Absolute milliseconds are host-specific. Compare old/new on the same machine,
same interpreter and same temporary corpus; ratios are the meaningful output.
The script never touches plugin data or the repository.
"""

from pathlib import Path
import argparse
import json
import statistics
import tempfile
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def timed(function, repeats: int = 5) -> float:
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        values.append((time.perf_counter() - started) * 1000)
    return round(statistics.median(values), 3)


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from recog.images import thumbnail
    from recog.store import Store

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=40)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="char-recog-bench-") as temporary:
        root = Path(temporary)
        store = Store(root)
        for index in range(max(8, args.images)):
            array = np.random.default_rng(index).integers(0, 256, (960, 720, 3), dtype=np.uint8)
            path = store.root / "uploads" / f"{index}.jpg"
            Image.fromarray(array).save(path, quality=90)
            store.ingest(path, "self" if index % 2 else "other", "洛天依")

        first = store.images(character="洛天依")["rows"][0]
        image_path = store.root / first["path"]
        thumb_path = store.root / "thumbnails" / (first["sha"] + "-512.jpg")
        thumbnail(image_path, thumb_path)  # warm the cache before timing hits
        report = {
            "images": store.images(character="洛天依")["total"],
            "gallery_page_ms": timed(lambda: store.images(page=1, character="洛天依"), 5),
            "thumbnail_hit_ms": timed(lambda: thumbnail(image_path, thumb_path), 5),
            "storage_stats_ms": timed(store.storage_stats, 3),
            "model_sizes_ms": timed(store.model_sizes, 5),
            "cleanup_ms": timed(store.cleanup_caches, 3),
            "storage": store.storage_stats(),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
