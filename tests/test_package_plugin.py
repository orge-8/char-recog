"""发布包清单门禁：运行必需文件在、开发产物不在。"""

import importlib.util
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).parents[1]


def _load_packager():
    path = ROOT / "tools" / "package_plugin.py"
    spec = importlib.util.spec_from_file_location("char_recog_packager", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_contains_runtime_and_excludes_development_artifacts(tmp_path):
    packager = _load_packager()
    destination = packager.build(tmp_path / "char-recog.zip")
    with ZipFile(destination) as archive:
        names = set(archive.namelist())
    assert "plugin.py" in names
    assert "_manifest.json" in names
    assert "webui/index.html" in names
    assert "assets/mobilenet_v3_small-047dcff4.pth" in names
    assert not any(name.startswith(("tests/", "docs/", "tools/", ".git/", ".github/")) for name in names)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
    assert not any(name.endswith((".log", "config.toml")) for name in names)
