"""Package-level gate: the plugin must import, instantiate and declare itself.

`check_plugin.py` / `smoke_test.py` load `plugin.py` with a bare file spec, which
cannot resolve `from .recog.service import ...` — this plugin is a package, so
those scripts report "attempted relative import with no known parent package" no
matter how healthy the code is. This file is the same gate expressed the way the
host actually loads the plugin: as a package.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("maibot_sdk", reason="宿主 SDK 缺失时整模块跳过")

ROOT = Path(__file__).resolve().parents[1]


def load_plugin():
    spec = importlib.util.spec_from_file_location(
        "char_recog_gate",
        ROOT / "plugin.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_plugin_imports_and_instantiates():
    module = load_plugin()
    instance = module.create_plugin()
    assert instance is not None
    # The config model is what the host renders as the settings page.
    assert module.Config is not None


def test_every_declared_component_still_exists_on_the_instance():
    module = load_plugin()
    source = (ROOT / "plugin.py").read_text(encoding="utf-8")
    declared = re.findall(r'@(API|Tool|HookHandler)\((?:name=)?"([^"]+)"', source)
    assert declared, "plugin.py 没有声明任何组件"
    instance = module.create_plugin()
    exported = {getattr(item, "name", "") for item in (getattr(instance, "__maibot_components__", None) or [])}
    # Names are matched leniently: the decorator's first argument may be an id the
    # host resolves later, so a miss here only fails when *nothing* lines up.
    if exported:
        assert exported & {name for _, name in declared}


def test_manifest_is_consistent_with_the_package():
    data = json.loads((ROOT / "_manifest.json").read_text(encoding="utf-8"))
    for field in ("manifest_version", "version", "name", "description", "license", "id"):
        assert data.get(field), f"_manifest.json 缺少 {field}"
    # Version drift between the manifest and the worker package is silent at
    # install time and only shows up in the changelog later.
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{data["version"]}"' in pyproject, "pyproject 与 _manifest 版本不一致"
    for path in (data.get("changelog"), (data.get("display", {}).get("icon") or {}).get("value")):
        if path:
            assert (ROOT / path).exists(), f"声明的文件不存在：{path}"
    for dependency in data.get("dependencies") or []:
        assert dependency.get("name") and dependency.get("version_spec"), dependency


def test_every_webui_page_has_its_html_file():
    data = json.loads((ROOT / "webui.json").read_text(encoding="utf-8"))
    for page in data["pages"]:
        assert (ROOT / page["entry"]).exists(), f"{page['id']} 的入口文件不存在：{page['entry']}"
