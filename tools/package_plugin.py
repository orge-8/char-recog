"""Build the single-file plugin release without development artifacts.

The release intentionally keeps the MobileNetV3 base weights: the project ships
one usable package, so the size win comes from excluding Git metadata, tests,
experiment records, tools and caches rather than from moving the base online.
"""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import argparse

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = (
    "plugin.py",
    "_manifest.json",
    "webui.json",
    "_locales",
    "assets",
    "recog",
    "webui",
    "LICENSE",
    "NOTICE.md",
    "README.md",
    "CHANGELOG.md",
)
# Generated pages are derived from index.html, but the host loads each page as an
# independent document, so both the source and generated pages are shipped.
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", "test-results"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log"}


def _iter_files():
    for name in INCLUDE:
        base = ROOT / name
        if base.is_file():
            yield base
            continue
        if not base.is_dir():
            raise FileNotFoundError(f"发布清单缺少 {name}")
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            yield path


def build(destination: Path) -> Path:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for path in _iter_files():
            archive.write(path, path.relative_to(ROOT).as_posix())
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 char-recog 单包发布 ZIP")
    parser.add_argument("destination", nargs="?", default="dist/char-recog.zip")
    args = parser.parse_args()
    result = build(Path(args.destination))
    print(f"{result} ({result.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
