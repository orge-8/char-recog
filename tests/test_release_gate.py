"""上线门禁补充审计（不依赖宿主 SDK）。

`tests/` 下所有触及 `plugin.py` 的既有文件都写了 `importorskip("maibot_sdk")`，
在本地与无宿主 CI 上恒为 skipped——也就是说插件层目前**没有任何一条会真正执行
的测试**。本文件用最小 SDK 桩把 `plugin.py` 按宿主的方式（包形态）加载，补齐四类
既有覆盖碰不到的门禁：

1. **接缝回归**：用真实图片驱动「插件 `_capture` → `store.ingest`」这条真实链路，
   断言图片真的落库。既有 `test_chat_capture.py` 只直接调 `store.ingest`，接缝本身
   零覆盖，0.6.0 删除 `ingest` 三个标注参数时漏改调用点也不会有人发现。
2. **边界输入配对**：每条退化用例（空串 / 纯空白 / 全占位符 `[图片]` / 缺失字段 /
   类型错误 / 只有无关字段）都配一条同 API 的合法正用例，且正用例的正数据由
   `prepare` 现场造真（真图片、真图片 ID、真任务行）。正用例必须仍然工作——
   否则「fail-closed」可以靠把所有输入都判错来蒙混过关。
3. **副作用守卫**：整轮边界调用不得派生子进程、不得改动源码树；空 upload_id /
   空角色名不得触达宿主 API。
4. **一致性 / 生命周期 / 事件循环占用**：清单与代码、文档与代码（AST 判定，不做
   文本匹配，避免被文档字符串误伤），`on_unload` 可重复调用，重图路径不阻塞事件
   循环（并配一条同步直调对照，证明 ticker 确实能测到阻塞）。

判定纪律：FAIL 先甄别归属。产品缺陷写进报告、本轮不改源码；属于本文件自身写错的
才改本文件。
"""

from __future__ import annotations

import ast
import asyncio
import base64
import importlib.util
import io
import json
import re
import subprocess
import sys
import time
import types
import uuid
from pathlib import Path

import numpy as np
import pytest
import pytest
from PIL import Image

from recog.store import Store

ROOT = Path(__file__).resolve().parents[1]
CHAR = "洛天依"
HEAVY_EDGE = 2400  # 重图边长：实测 ingest ~190ms，足够让 5ms 心跳在异步窗口内走满 5 拍以上


# ---------------------------------------------------------------------------
# 最小 maibot_sdk 桩
# ---------------------------------------------------------------------------


class _Unset:
    pass


_UNSET = _Unset()


class _Field:
    """`Field(default=...)` / `Field(default_factory=...)` 的占位与求值。"""

    def __init__(self, default=_UNSET, default_factory=None, description=""):
        self.default = default
        self.default_factory = default_factory
        self.description = description

    def resolve(self):
        if self.default_factory is not None:
            return self.default_factory()
        if self.default is not _UNSET:
            return self.default
        return None


def Field(default=_UNSET, *, default_factory=None, description="", **_extra):  # noqa: N802
    return _Field(default, default_factory, description)


def _config_fields(cls) -> dict:
    out: dict = {}
    for klass in reversed(cls.__mro__):
        for name, value in vars(klass).items():
            if isinstance(value, _Field):
                out[name] = value.resolve()
    return out


class PluginConfigBase:
    def __init__(self, **overrides):
        for name, value in _config_fields(type(self)).items():
            setattr(self, name, value)
        for key, value in overrides.items():
            setattr(self, key, value)


class MaiBotPlugin:
    config_model = None

    def __init__(self):
        self.ctx = None
        model = type(self).config_model
        self.config = model() if model is not None else None


def API(name, version="1"):  # noqa: N802
    def wrap(func):
        func.__gate_component__ = ("api", name, version)
        return func

    return wrap


def Tool(name, description="", parameters=()):  # noqa: N802
    def wrap(func):
        func.__gate_component__ = ("tool", name, description)
        return func

    return wrap


def HookHandler(name, timeout_ms=0):  # noqa: N802
    def wrap(func):
        func.__gate_component__ = ("hook", name, timeout_ms)
        return func

    return wrap


class _ToolParamType:
    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    NUMBER = "number"


class ToolParameterInfo:  # noqa: N801
    def __init__(self, name="", param_type="", description="", required=False):
        self.name = name
        self.param_type = param_type
        self.description = description
        self.required = required


def _install_sdk_stub():
    sdk = types.ModuleType("maibot_sdk")
    sdk.API = API
    sdk.Tool = Tool
    sdk.HookHandler = HookHandler
    sdk.MaiBotPlugin = MaiBotPlugin
    sdk.PluginConfigBase = PluginConfigBase
    sdk.Field = Field
    types_module = types.ModuleType("maibot_sdk.types")
    types_module.ToolParameterInfo = ToolParameterInfo
    types_module.ToolParamType = _ToolParamType
    sdk.types = types_module
    return sdk, types_module


_STUB_MODULES = ("maibot_sdk", "maibot_sdk.types")
_LOADED_PREFIX = "char_recog_gate"


@pytest.fixture(scope="module")
def plugin_module():
    """按宿主方式（包形态）加载 plugin.py，用完摘掉桩与已加载子模块。"""
    saved = {name: sys.modules.get(name) for name in _STUB_MODULES}
    sdk, types_module = _install_sdk_stub()
    sys.modules["maibot_sdk"] = sdk
    sys.modules["maibot_sdk.types"] = types_module
    package = types.ModuleType(_LOADED_PREFIX)
    package.__path__ = [str(ROOT)]
    sys.modules[_LOADED_PREFIX] = package
    spec = importlib.util.spec_from_file_location(f"{_LOADED_PREFIX}.plugin", ROOT / "plugin.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        for name in [n for n in sys.modules if n == _LOADED_PREFIX or n.startswith(f"{_LOADED_PREFIX}.")]:
            del sys.modules[name]
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


# ---------------------------------------------------------------------------
# 宿主上下文桩：只记录外部触达，绝不真的连宿主
# ---------------------------------------------------------------------------


class _Webui:
    def __init__(self):
        self.claimed: list[str] = []
        self.handlers: dict[str, Path] = {}

    def register(self, upload_id: str, path: Path) -> None:
        self.handlers[upload_id] = path

    async def claim_upload(self, upload_id: str) -> dict:
        self.claimed.append(upload_id)
        return {"path": str(self.handlers.get(upload_id, Path("/nonexistent-host-upload")))}


class _Message:
    def __init__(self):
        self.requested: list[str] = []
        self.handlers: dict[str, dict] = {}

    def register(self, message_id: str, payload: dict) -> None:
        self.handlers[message_id] = payload

    async def get_by_id(self, message_id: str, **_kwargs) -> dict:
        self.requested.append(message_id)
        return self.handlers.get(message_id, {"message_id": message_id, "content": []})


class _Paths:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir


class _Ctx:
    def __init__(self, data_dir: Path):
        self.webui = _Webui()
        self.message = _Message()
        self.paths = _Paths(data_dir)


class _ServiceShim:
    """各 API 只碰 `service.store`，用真 Store 即可跑通真实链路。

    `overlay_path=None` 让 `environment.submit` 在派生子进程之前就 fail-closed；
    `infer` 是模型边界，本门禁不测模型，返回一个合法结果即可。
    """

    overlay_path = None

    def __init__(self, store: Store):
        self.store = store

    def infer(self, raw: bytes, timeout: float = 8, character: str = "", **_kwargs) -> dict:
        # 字段按 run_test/_tool_result 真实读取的形状给全，否则链路会在与本门禁
        # 无关的地方 KeyError，给出误导性红。
        return {
            "status": "ok",
            "results": [
                {
                    "character": CHAR,
                    "score": 0.8,
                    "status": "ok",
                    "judgment": "达标",
                    "threshold": 0.5,
                    "version": "v1",
                }
            ],
        }

    def close(self) -> None:
        """on_unload 走的是 `offload(self.service.close)`，桩上必须可调用。"""
        self.closed = True


def _png_bytes(edge: int, seed: int) -> bytes:
    pixels = np.random.default_rng(seed).integers(0, 256, (edge, edge, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def env(plugin_module, tmp_path):
    """一个加载好的插件 + 真 Store + 可观测的宿主桩。"""
    plugin = plugin_module.create_plugin()
    store = Store(tmp_path / "data")
    plugin.service = _ServiceShim(store)
    plugin.ctx = _Ctx(tmp_path / "data")
    return plugin, store


def _run(coro):
    return asyncio.run(coro)


def _is_error_contract(result) -> bool:
    """API 层的统一失败结构：success=False，或工具层的 status 错误码。"""
    if not isinstance(result, dict):
        return False
    if result.get("success") is False:
        return True
    return isinstance(result.get("status"), str) and result["status"] in {
        "missing_chat_context",
        "invalid_image_index",
        "invalid_character",
        "message_not_found",
        "image_not_found",
        "image_expired",
        "image_too_large",
        "invalid_image",
        "error",
        "no_model",
    }


# ---------------------------------------------------------------------------
# 1. 清单 / 代码 / 文档一致性
# ---------------------------------------------------------------------------


def _parse(path: Path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _imported_modules(tree) -> set[str]:
    """AST 收集 import 的顶层模块名（含 `from x import y` 的 x）。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def _host_call_sites(tree) -> set[str]:
    """`self.ctx.<namespace>.<attr>` 的全部「命名空间.属性」。"""
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        value = node.func.value
        if not isinstance(value, ast.Attribute) or not isinstance(value.value, ast.Attribute):
            continue
        if not isinstance(value.value.value, ast.Name) or value.value.value.id != "self":
            continue
        if value.value.attr != "ctx":
            continue
        found.add(f"{value.attr}.{node.func.attr}")
    return found


# 跑在 MaiBot 宿主进程里的模块（消息路径 + 入库路径）
HOST_PROCESS_GRAPH = (
    "plugin.py",
    "recog/service.py",
    "recog/store.py",
    "recog/images.py",
    "recog/scores.py",
    "recog/processes.py",
    "recog/environment.py",
)
# 以 `python -m` 独立子进程运行的模块，不在宿主进程内
SUBPROCESS_GRAPH = ("recog/probes.py", "recog/bootstrap.py", "recog/dependency_download.py")
# 依赖声明核验用全图：子进程同样需要 Pillow/numpy
ALL_IMPORT_GRAPH = HOST_PROCESS_GRAPH + SUBPROCESS_GRAPH

# 宿主侧（不派生子进程）import 图里出现的第三方包 -> 清单里的发行名
THIRD_PARTY_TO_DISTRIBUTION = {
    "PIL": "Pillow",
    "numpy": "numpy",
    "psutil": "psutil",
}


def _components(module) -> list[tuple[str, str, object]]:
    out = []
    for name in dir(module.CharacterRecognition):
        marker = getattr(getattr(module.CharacterRecognition, name), "__gate_component__", None)
        if marker:
            out.append((marker[0], marker[1], marker[2]))
    return out


def test_manifest_and_webui_json_are_valid_and_wired(plugin_module):
    manifest = json.loads((ROOT / "_manifest.json").read_text(encoding="utf-8"))
    for field in ("manifest_version", "version", "name", "description", "license", "id"):
        assert manifest.get(field), f"_manifest.json 缺少 {field}"
    webui = json.loads((ROOT / "webui.json").read_text(encoding="utf-8"))
    assert webui.get("pages"), "webui.json 没有声明任何页面"
    for page in webui["pages"]:
        assert (ROOT / page["entry"]).exists(), f"{page['id']} 的入口文件不存在：{page['entry']}"
    # 每个 action 都必须有对应 API，否则前端点了没反应
    declared = {name for _, name, _ in _components(plugin_module)}
    for action in webui.get("actions") or []:
        assert action.get("id") in declared, f"webui.json 声明了未实现的 action：{action.get('id')}"


def test_manifest_version_matches_pyproject():
    manifest = json.loads((ROOT / "_manifest.json").read_text(encoding="utf-8"))
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{manifest["version"]}"' in pyproject, (
        f"pyproject 版本与 _manifest.json({manifest['version']}) 不一致"
    )


def test_changelog_top_version_matches_the_manifest():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    manifest = json.loads((ROOT / "_manifest.json").read_text(encoding="utf-8"))
    headings = [line for line in changelog.splitlines() if line.startswith("## ")]
    assert headings, "CHANGELOG.md 没有版本条目"
    assert manifest["version"] in headings[0], (
        f"CHANGELOG 最新条目是 {headings[0]!r}，但 _manifest.json 版本是 {manifest['version']}"
    )


def test_declared_host_capabilities_match_the_code():
    """清单声明的宿主能力必须正好等于代码触达的能力，双向收口。"""
    manifest = json.loads((ROOT / "_manifest.json").read_text(encoding="utf-8"))
    # capabilities 是 "namespace.method" 字符串列表
    declared = {entry for entry in manifest.get("capabilities") or [] if isinstance(entry, str)}
    assert declared, "_manifest.json 没有声明任何宿主能力"
    used = _host_call_sites(_parse(ROOT / "plugin.py"))
    assert used, "没有在 plugin.py 里找到任何 self.ctx.* 调用，AST 判定可能失效"
    missing = used - declared
    assert not missing, f"代码触达了清单未声明的宿主能力：{sorted(missing)}"
    unused = declared - used
    assert not unused, f"清单声明了代码从不使用的宿主能力：{sorted(unused)}"


def test_declared_dependencies_cover_the_import_graph():
    """import 图用到的第三方包必须已声明；声明了的必须真的被用到。"""
    manifest = json.loads((ROOT / "_manifest.json").read_text(encoding="utf-8"))
    declared = {
        dependency.get("name") for dependency in manifest.get("dependencies") or [] if dependency.get("name")
    }
    used: set[str] = set()
    for relative in ALL_IMPORT_GRAPH:
        path = ROOT / relative
        if path.exists():
            used |= _imported_modules(_parse(path))
    expected = {THIRD_PARTY_TO_DISTRIBUTION[name] for name in used if name in THIRD_PARTY_TO_DISTRIBUTION}
    missing = expected - declared
    assert not missing, f"import 图用到但清单未声明：{sorted(missing)}（真机上必然 ImportError）"
    unused = declared - expected
    assert not unused, f"清单声明了从不 import 的依赖：{sorted(unused)}"


def test_readme_referenced_paths_exist():
    """README 反引号里点名的仓库内路径都必须真实存在。

    只认「反引号包裹 + 含目录分隔 + 不像通配」的 token：散文里提到的裸文件名
    （`bot.py`、`snapshot.json` 之类是别处的产物）与 `webui/{a,b}.html` 这种
    shell 花括号展开都不算引用。
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    referenced: set[str] = set()
    for token in readme.split("`"):
        token = token.strip()
        if not token or "/" not in token:
            continue
        # 围栏代码块的语言标签（```powershell）会连着整段代码一起被切出来，
        # 真正的路径引用是单个无空白 token。
        if any(character.isspace() for character in token):
            continue
        if any(character in token for character in "{}*?<>|"):
            continue
        if token.startswith(("http://", "https://", "#", "/")):
            continue
        if token.endswith((".py", ".md", ".json", ".html", ".toml", ".svg", ".sqlite3")):
            referenced.add(token)
    missing = sorted(path for path in referenced if not (ROOT / path).exists())
    assert not missing, f"README 引用了不存在的文件：{missing}"


def test_message_path_makes_no_network_calls():
    """消息/入库路径（跑在宿主进程里的模块）不得发起网络请求。

    `recog/dependency_download.py` 确实会用 urllib 拉 PyTorch wheel，但它是安装
    依赖时以子进程运行的，不在消息路径上——那条通道单独在报告里说明。
    """
    forbidden = {"requests", "httpx", "aiohttp", "urllib3", "socket", "ftplib", "smtplib", "paramiko"}
    for relative in HOST_PROCESS_GRAPH:
        path = ROOT / relative
        if not path.exists():
            continue
        tree = _parse(path)
        found = _imported_modules(tree) & forbidden
        assert not found, f"{relative} import 了网络库 {sorted(found)}"
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("urllib.request"):
                raise AssertionError(f"{relative}:{node.lineno} 使用了 urllib.request")


def test_host_process_never_uses_dynamic_import_or_exec():
    """宿主进程内的模块不得调用 eval/exec/compile/__import__。

    范围限定宿主进程：`recog/probes.py` 的 `__import__` 只接受插件自己拼的
    spawn 参数，是覆盖层优先级探针，不在宿主进程内执行。
    """
    forbidden = {"eval", "exec", "compile", "__import__"}
    for relative in HOST_PROCESS_GRAPH:
        path = ROOT / relative
        if not path.exists():
            continue
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden:
                raise AssertionError(f"{relative}:{node.lineno} 调用了 {node.func.id}()")


def test_changelog_test_count_matches_the_shipped_test_suite():
    """CHANGELOG 自述的 collected 数必须与当前可收集数一致。

    排除两个审计期新增文件：`tests/test_security_audit.py`（安全同事在写，未跟踪）
    与 `tests/test_release_gate.py`（本文件）。统计口径 = 当前版本变更集自带的测试。

    解析纪律：只认**第一个版本条目**里的**第一个** `N collected`。
    0.6.1 条目同时写了「全量 412 collected」和「0.6.0 口径 235 collected」两行，
    旧写法把一行里所有数字无条件拼接，取到了 60235 这种数 —— 那是测试自己写错，
    不是产品缺陷。锚定首个条目 + 首个匹配，避免文档多写几个数字就误报。

    计数口径以装了 torch 的完整环境为准；仓库 CI 刻意不装 torch，此时少收集的
    正是 torch 用例，不能据此判文档过期。
    """
    pytest.importorskip("torch", reason="计数口径以含 torch 的完整环境为准")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    # 第一个 "## x.y.z" 标题之后的内容，即当前版本条目
    match = re.search(r"^## (\d+\.\d+\.\d+)\b", changelog, re.MULTILINE)
    assert match, "CHANGELOG 里找不到任何版本条目"
    section = changelog[match.start():]
    claimed = None
    for line in section.splitlines():
        if "collected" in line:
            found = re.search(r"(\d+)\s+collected", line)
            if found:
                claimed = int(found.group(1))
                break
    assert claimed is not None, f"{match.group(1)} 条目里找不到 'N collected' 自述"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        "--ignore=tests/test_security_audit.py",
        "--ignore=tests/test_release_gate.py",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=300)
    counted = sum(1 for line in completed.stdout.splitlines() if "::" in line)
    assert counted == claimed, (
        f"CHANGELOG 自述 {claimed} collected，实际（0.6.0 变更集口径）为 {counted} collected，"
        f"相差 {counted - claimed} 条"
    )


# ---------------------------------------------------------------------------
# 2. 接缝回归：聊天捕获必须真的落库
# ---------------------------------------------------------------------------


def _chat_results(score: float = 0.72) -> list[dict]:
    return [{"character": CHAR, "score": score, "status": "ok"}]


def _find_function(tree, name: str, within: str | None = None):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            if within is None:
                return node
            if any(isinstance(child, ast.FunctionDef) and child.name == within for child in ast.walk(node)):
                return node
    return None


def test_chat_capture_call_site_matches_store_ingest_arity():
    """`_capture` 里 `store.ingest(...)` 的位置参数个数必须等于当前签名允许的个数。

    0.6.0 从 `Store.ingest` 删掉 style/neg_type/has_text 三个参数，`result_promote`
    的调用点跟着改了，`_capture` 的没改——这条用例把那个漏改钉住。
    """
    ingest = _find_function(_parse(ROOT / "recog" / "store.py"), "ingest")
    assert ingest is not None, "recog/store.py 里找不到 Store.ingest"
    allowed = len(ingest.args.args) - 1  # 去掉 self

    capture = _find_function(_parse(ROOT / "plugin.py"), "_capture")
    assert capture is not None, "plugin.py 里找不到 _capture"
    calls = [
        node
        for node in ast.walk(capture)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "ingest"
    ]
    assert calls, "_capture 里没有 store.ingest 调用"
    passed = len(calls[0].args)
    assert passed <= allowed, (
        f"plugin.py:{calls[0].lineno} 的 store.ingest 传了 {passed} 个位置参数，"
        f"而 recog/store.py:{ingest.lineno} 的签名只接受 {allowed} 个"
        "（0.6.0 删除了 style/neg_type/has_text，此调用点漏改）"
    )


def test_chat_capture_lands_the_image_in_the_library(env, caplog):
    """真图穿过 `_capture` 必须落库：label=None、source=chat、得分入库。

    这是既有 `test_chat_capture.py` 唯一没覆盖的一段——它直接调 `store.ingest`，
    从未走过插件这一侧。
    """
    import logging

    plugin, store = env
    raw = _png_bytes(96, 11)
    with caplog.at_level(logging.WARNING, logger="char-recog"):
        plugin._capture(raw, _chat_results())
    with store.db() as db:
        rows = db.execute("SELECT * FROM images WHERE source='chat'").fetchall()
    assert len(rows) == 1, (
        "聊天捕获没有落库。store.ingest 抛的异常被 plugin.py:199 的 except Exception 吞掉、"
        f"只打了 warning 日志：{[record.getMessage() for record in caplog.records]}"
    )
    row = dict(rows[0])
    assert row["label"] is None, "聊天捕获的图必须保持未标注——模型的猜测不是真值"
    assert row["character"] == CHAR
    assert row["model_score"] == pytest.approx(0.72, abs=1e-6)


def test_chat_capture_leaves_no_orphan_upload_file(env):
    """入库失败也不得在 uploads/ 留下孤儿文件（磁盘会静静涨满）。"""
    plugin, store = env
    plugin._capture(_png_bytes(96, 12), _chat_results())
    leftovers = sorted(path.name for path in (store.root / "uploads").iterdir())
    assert leftovers == [], f"uploads/ 残留孤儿文件：{leftovers}"


def test_out_of_band_scores_are_not_captured(env):
    """正用例：0.4–0.9 之外的得分不该被捕获（防上面几条是「全捕获」假绿）。"""
    plugin, store = env
    plugin._capture(_png_bytes(96, 13), _chat_results(0.99))
    with store.db() as db:
        assert db.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# 3. 边界输入：每条退化用例配对一条同 API 的合法正用例
# ---------------------------------------------------------------------------


def _prepare_upload(plugin, store) -> dict:
    source = store.root / "uploads" / "gate-valid.png"
    source.write_bytes(_png_bytes(48, 21))
    plugin.ctx.webui.register("gate-valid", source)
    return {"upload_id": "gate-valid"}


def _prepare_test_image(plugin, store) -> dict:
    result_id = uuid.uuid4().hex
    directory = store.root / "test-images"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (result_id + ".image")).write_bytes(_png_bytes(48, 22))
    store.set_setting("test_history", json.dumps([{"id": result_id}], ensure_ascii=False))
    return {"result_id": result_id}


def _prepare_complete_job(plugin, store) -> dict:
    job_id = uuid.uuid4().hex
    with store.db() as db:
        db.execute(
            "INSERT INTO jobs (id,request_key,state,pid,error,created,snapshot_sha,character,device,input_size) "
            "VALUES (?,?,'complete',NULL,'',?,?,?,?,?)",
            (job_id, "gate", time.time(), "sha", CHAR, "cpu", 320),
        )
    return {"job_id": job_id}


def _prepare_library_image_for_batch(plugin, store) -> dict:
    path = store.root / "uploads" / "gate-library.png"
    path.write_bytes(_png_bytes(48, 24))
    created = store.ingest(path, None, CHAR, "webui", None)
    return {"image_ids": created["id"]}


def _prepare_library_image_for_label(plugin, store) -> dict:
    path = store.root / "uploads" / "gate-library-label.png"
    path.write_bytes(_png_bytes(48, 25))
    created = store.ingest(path, None, CHAR, "webui", None)
    return {"image_id": created["id"]}


def _prepare_message(plugin, store) -> dict:
    message_id = "gate-message"
    plugin.ctx.message.register(
        message_id,
        {
            "message_id": message_id,
            "session_id": "gate-stream",
            "raw_message": [
                {
                    "type": "image",
                    "binary_data_base64": base64.b64encode(_png_bytes(48, 23)).decode(),
                }
            ],
        },
    )
    return {"message_id": message_id, "stream_id": "gate-stream"}


# (api, 退化 kwargs, 期望, prepare|None, 正用例 kwargs, id)
# 期望 "error" = 必须返回错误契约；"ok" = 纯占位符但格式合法，只要求不崩且返回 dict。
BOUNDARY_ROWS = [
    ("images", {"page": 0, "view": "captured"}, "error", None, {"page": 1, "view": "captured"}, "images_page_zero"),
    ("images", {"page": 1, "view": "不存在的视图", "character": CHAR}, "error", None,
     {"page": 1, "view": "all", "character": CHAR}, "images_unknown_view"),
    ("images", {"page": 1, "view": "all", "character": ""}, "error", None,
     {"page": 1, "view": "all", "character": CHAR}, "images_missing_character"),
    ("images", {"page": 1, "view": "all", "character": "   "}, "error", None,
     {"page": 1, "view": "all", "character": CHAR}, "images_blank_character"),
    ("images", {"page": 1, "view": "all", "character": "[图片]"}, "ok", None,
     {"page": 1, "view": "all", "character": CHAR}, "images_placeholder_character"),
    ("ui_save", {"state": ""}, "error", None, {"state": json.dumps({"view": "all"})}, "ui_save_empty_state"),
    ("ui_save", {"state": "   "}, "error", None, {"state": json.dumps({"view": "unlabelled"})}, "ui_save_blank_state"),
    ("ui_save", {"state": "[图片]"}, "error", None, {"state": json.dumps({"view": "eligible"})},
     "ui_save_placeholder_state"),
    ("ui_save", {"state": "[1, 2, 3]"}, "error", None, {"state": json.dumps({"view": "removed"})},
     "ui_save_non_object_json"),
    ("ui_save", {"state": None}, "error", None, {"state": json.dumps({"view": "errors"})}, "ui_save_none_state"),
    ("apply_threshold", {"character": "", "value": ""}, "error", None, {"character": CHAR, "value": "0.5"},
     "apply_threshold_all_empty"),
    ("apply_threshold", {"character": CHAR, "value": "不是数字"}, "error", None,
     {"character": CHAR, "value": "0.5"}, "apply_threshold_bad_value"),
    ("apply_threshold", {"character": CHAR, "value": None}, "error", None,
     {"character": CHAR, "value": "0.5"}, "apply_threshold_none_value"),
    ("batch_images", {"image_ids": "", "operation": "self"}, "error", _prepare_library_image_for_batch,
     {"image_ids": "a" * 32, "operation": "self"}, "batch_images_empty_ids"),
    ("batch_images", {"image_ids": "不是ID", "operation": "self"}, "error", _prepare_library_image_for_batch,
     {"image_ids": "a" * 32, "operation": "self"}, "batch_images_garbage_ids"),
    ("batch_images", {"image_ids": "a" * 32, "operation": "不存在的操作"}, "error",
     _prepare_library_image_for_batch, {"image_ids": "a" * 32, "operation": "self"},
     "batch_images_unknown_operation"),
    ("batch_images", {"image_ids": "a" * 32, "operation": "self"}, "error", _prepare_library_image_for_batch,
     {"image_ids": "a" * 32, "operation": "self"}, "batch_images_unknown_ids"),
    ("remove_all_images", {"character": ""}, "error", None, {"character": CHAR}, "remove_all_images_no_character"),
    ("result_promote", {"result_id": "", "character": CHAR, "label": "self"}, "error", _prepare_test_image,
     {"result_id": "", "character": CHAR, "label": "self"}, "result_promote_empty_id"),
    ("label", {"image_id": "", "label": "self"}, "error", _prepare_library_image_for_label,
     {"image_id": "a" * 32, "label": "self"}, "label_empty_id"),
    ("select_report", {"job_id": ""}, "error", _prepare_complete_job, {"job_id": ""}, "select_report_empty_job"),
    ("select_errors", {"job_id": ""}, "error", _prepare_complete_job, {"job_id": ""}, "select_errors_empty_job"),
    ("upload", {"upload_id": "gate-valid", "character": "", "upload_label": "self"}, "error", _prepare_upload,
     {"upload_id": "gate-valid", "character": CHAR, "upload_label": "self"}, "upload_empty_character"),
    ("upload", {"upload_id": "gate-valid", "character": CHAR, "upload_label": "不是身份"}, "error", _prepare_upload,
     {"upload_id": "gate-valid", "character": CHAR, "upload_label": "self"}, "upload_bad_label"),
    ("manage_environment", {"kind": ""}, "error", None, "__none__", "manage_environment_empty_kind"),
    ("recognize", {"message_id": "", "image_index": 0, "character": "", "stream_id": ""}, "error", _prepare_message,
     {"message_id": "gate-message", "image_index": 0, "character": "", "stream_id": "gate-stream"},
     "recognize_missing_stream_id"),
    ("recognize", {"message_id": "gate-message", "image_index": -1, "character": "", "stream_id": "gate-stream"},
     "error", _prepare_message, {"message_id": "gate-message", "image_index": 0, "character": "",
                                 "stream_id": "gate-stream"}, "recognize_negative_index"),
    ("recognize", {"message_id": "gate-message", "image_index": "0", "character": "", "stream_id": "gate-stream"},
     "error", _prepare_message, {"message_id": "gate-message", "image_index": 0, "character": "",
                                 "stream_id": "gate-stream"}, "recognize_string_index"),
    ("recognize", {"message_id": "gate-message", "image_index": 0, "character": "[图片]",
                   "stream_id": "gate-stream"}, "ok", _prepare_message,
     {"message_id": "gate-message", "image_index": 0, "character": "", "stream_id": "gate-stream"},
     "recognize_placeholder_character"),
]

BOUNDARY_PAIRS = [pytest.param(*row[:5], id=row[5]) for row in BOUNDARY_ROWS]


@pytest.mark.parametrize("api,bad,expect,prepare,good", BOUNDARY_PAIRS)
def test_boundary_input_fails_closed_without_raising(env, api, bad, expect, prepare, good):
    """退化输入必须返回结构化错误，而不是把裸异常抛回前端。

    `expect="ok"` 用于「纯占位符但格式合法」的输入：插件允许任意角色名，
    此时要求只是不崩、且仍然返回 dict。
    """
    plugin, _store = env
    result = _run(getattr(plugin, api)(**bad))
    assert isinstance(result, dict), f"{api}({bad}) 没有返回 dict：{result!r}"
    if expect == "error":
        assert _is_error_contract(result), f"{api} 收到退化输入 {bad} 时既没抛异常、也没返回错误契约：{result!r}"


@pytest.mark.parametrize("api,bad,expect,prepare,good", BOUNDARY_PAIRS)
def test_paired_valid_input_still_works(env, api, bad, expect, prepare, good):
    """配对正用例：同一 API 的合法输入必须真的工作。

    没有这一条，「fail-closed」可以靠无条件返回 success:false 来全绿。
    `good == "__none__"` 表示该 API 的成功路径需要真实依赖环境（装/卸 PyTorch），
    超出单测范围，此时不配正用例，只保留退化用例与副作用守卫。
    """
    if good == "__none__":
        pytest.skip(f"{api} 的成功路径需要真实依赖环境，单测内无可配正用例")
    plugin, store = env
    if prepare is not None:
        good = {**good, **prepare(plugin, store)}
    result = _run(getattr(plugin, api)(**good))
    assert isinstance(result, dict), f"{api}({good}) 没有返回 dict：{result!r}"
    assert not _is_error_contract(result), f"{api} 的合法输入 {good} 被误判为失败：{result!r}"


# ---------------------------------------------------------------------------
# 4. 副作用守卫
# ---------------------------------------------------------------------------


def test_empty_upload_id_never_reaches_the_host(env):
    """空 upload_id 必须在触达宿主 claim_upload 之前就被挡掉。"""
    plugin, _store = env
    result = _run(plugin.upload("", CHAR, "self"))
    assert plugin.ctx.webui.claimed == [], f"空 upload_id 被透传给了宿主 claim_upload：{plugin.ctx.webui.claimed}"
    assert _is_error_contract(result), f"空 upload_id 未返回错误契约：{result!r}"


def test_invalid_character_never_reaches_the_host_upload(env):
    """空角色名不得触达宿主，也不得写出上传目录之外的文件。"""
    plugin, store = env
    result = _run(plugin.upload("gate-valid", "", "self"))
    assert plugin.ctx.webui.claimed == [], "角色名为空时仍然调用了宿主 claim_upload"
    assert _is_error_contract(result), f"空角色名未返回错误契约：{result!r}"


def test_boundary_suite_has_no_external_side_effects(plugin_module, tmp_path, monkeypatch):
    """整轮边界调用：不派生子进程、不改动源码树。"""
    ignored = ("__pycache__", ".pytest_cache", ".ruff_cache", "test-results")
    # 只快照源码子树：data/ dependencies/ 是运行期产物，与本用例无关且可能很大。
    watched = [ROOT / "plugin.py", ROOT / "_manifest.json", ROOT / "webui.json", ROOT / "pyproject.toml",
               ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "recog", ROOT / "webui", ROOT / "tests"]

    def snapshot() -> dict[str, tuple[int, int]]:
        state = {}
        for base in watched:
            paths = [base] if base.is_file() else sorted(base.rglob("*"))
            for path in paths:
                if not path.is_file():
                    continue
                relative = path.relative_to(ROOT)
                if any(part in ignored for part in relative.parts):
                    continue
                stat = path.stat()
                state[str(relative)] = (stat.st_size, stat.st_mtime_ns)
        return state

    spawned: list[object] = []
    real_popen = subprocess.Popen

    def spy(command, *args, **kwargs):
        spawned.append(command)
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", spy)

    before = snapshot()
    plugin = plugin_module.create_plugin()
    store = Store(tmp_path / "guard")
    plugin.service = _ServiceShim(store)
    plugin.ctx = _Ctx(tmp_path / "guard")
    for api, bad, _expect, _prepare, _good, _row_id in BOUNDARY_ROWS:
        try:
            _run(getattr(plugin, api)(**bad))
        except Exception:  # noqa: BLE001 - 本用例只关心磁盘/进程副作用
            pass
    after = snapshot()

    assert spawned == [], f"边界输入竟然派生了子进程：{spawned}"
    changed = sorted(set(before) ^ set(after)) + sorted(
        key for key in set(before) & set(after) if before[key] != after[key]
    )
    assert not changed, f"边界调用改动了源码树：{changed}"


# ---------------------------------------------------------------------------
# 5. 生命周期
# ---------------------------------------------------------------------------


def test_on_unload_is_idempotent(env):
    """卸载可重复调用：宿主重载插件时不会因为二次 close 抛异常。"""
    plugin, _store = env
    _run(plugin.on_unload())
    _run(plugin.on_unload())


def test_on_unload_before_on_load_is_a_no_op(plugin_module):
    """注册失败（on_load 从未跑过）时 on_unload 不得炸。"""
    plugin = plugin_module.create_plugin()
    assert plugin.service is None
    _run(plugin.on_unload())


def test_on_unload_after_service_close_does_not_raise(env):
    """service 已被别处 close 过后再卸载，仍然不得抛异常。"""
    plugin, _store = env
    plugin.service.close()
    _run(plugin.on_unload())
    _run(plugin.on_unload())


# ---------------------------------------------------------------------------
# 6. 事件循环占用
# ---------------------------------------------------------------------------


class _Ticker:
    """事件循环心跳：阻塞期间 ticker 停摆，ticks 即为被卡住的节拍数。"""

    def __init__(self, interval: float = 0.005):
        self.interval = interval
        self.ticks = 0
        self.stop = False

    async def run(self):
        while not self.stop:
            self.ticks += 1
            await asyncio.sleep(self.interval)


@pytest.fixture(scope="module")
def heavy_png(tmp_path_factory) -> bytes:
    return _png_bytes(HEAVY_EDGE, 31)


def test_image_path_does_not_block_the_event_loop(env, heavy_png):
    """上传入库这条重图路径必须让出事件循环（阻塞窗口内心跳 >= 5 拍）。"""
    plugin, store = env
    source = store.root / "uploads" / "heavy.png"
    source.write_bytes(heavy_png)
    plugin.ctx.webui.register("heavy", source)
    ticker = _Ticker()

    async def scenario():
        task = asyncio.create_task(ticker.run())
        await asyncio.sleep(0.05)  # 先让心跳建立节奏
        before = ticker.ticks
        result = await plugin.upload("heavy", CHAR, "self")
        during = ticker.ticks - before
        ticker.stop = True
        await task
        return result, during

    result, during = _run(scenario())
    assert not _is_error_contract(result), f"重图上传失败：{result!r}"
    assert during >= 5, (
        f"事件循环被阻塞：{ticker.interval}s 心跳在入库期间只走了 {during} 拍"
        "（<5 说明 ingest 的解码/指纹/落库占住了循环）"
    )


def test_the_same_work_inline_blocks_the_event_loop(env, heavy_png):
    """同步直调对照：同一份重图直接在循环里 ingest，阻塞窗口内心跳必须为 0。

    没有这条对照，上面的阈值可能只是「sleep 太多」的假象。
    """
    _plugin, store = env
    source = store.root / "uploads" / "heavy-inline.png"
    source.write_bytes(heavy_png)
    ticker = _Ticker()

    async def scenario():
        task = asyncio.create_task(ticker.run())
        await asyncio.sleep(0.05)
        before = ticker.ticks
        store.ingest(source, None, CHAR, "webui", None)  # 同步直调，刻意不让出循环
        during = ticker.ticks - before
        ticker.stop = True
        await task
        return during

    during = _run(scenario())
    assert during == 0, (
        f"同步直调竟然没有阻塞事件循环（阻塞窗口内走了 {during} 拍），"
        "说明本文件的 ticker 测不到阻塞，上一条用例的阈值失去意义"
    )


# ---------------------------------------------------------------------------
# 版本约束断流：声明区间必须覆盖实际安装版本
#
# 0.6.0 漂过一次：manifest / pyproject / requirements / environment.py 四处都写
# `Pillow>=11,<13`，而覆盖层实装 13.0.0 —— 约束从未被强制执行，也没人发现。
# 抬区间是补救，断流测试才是防止再漂的东西。
# ---------------------------------------------------------------------------


def _declared_ranges():
    """从四个声明处解析 (包名, min, max_exclusive)。"""
    import re

    manifest = json.loads((ROOT / "_manifest.json").read_text(encoding="utf-8"))
    ranges = {}
    for dep in manifest.get("dependencies", []):
        if dep.get("type") != "python_package":
            continue
        spec = dep.get("version_spec", "")
        found = re.findall(r"(>=|>)\s*([\d.]+)|(<=|<)\s*([\d.]+)", spec)
        low = next((float(m[1]) for m in found if m[0]), None)
        high = next((float(m[3]) for m in found if m[2]), None)
        ranges[dep["name"].lower()] = (low, high)
    for source, pattern in (
        ("pyproject.toml", r'"([A-Za-z0-9_.-]+)>=([\d.]+),<([\d.]+)"'),
        ("requirements-worker.txt", r"^([A-Za-z0-9_.-]+)>=([\d.]+),<([\d.]+)$"),
    ):
        text = (ROOT / source).read_text(encoding="utf-8")
        for name, low, high in re.findall(pattern, text, re.MULTILINE):
            ranges.setdefault(name.lower(), (float(low), float(high)))
    return ranges


def test_declared_dependency_ranges_cover_the_installed_versions():
    ranges = _declared_ranges()
    assert ranges, "一个依赖声明都没解析到，_declared_ranges 写错了"
    for module, package in (("PIL", "pillow"), ("numpy", "numpy"), ("psutil", "psutil")):
        spec = importlib.util.find_spec(module)
        if spec is None:
            pytest.skip(f"{module} 未安装，跳过版本比对")
        import importlib.metadata as metadata

        installed = metadata.version(package if package != "pillow" else "Pillow")
        major_minor = float(".".join(installed.split(".")[:2]))
        low, high = ranges.get(package, (None, None))
        assert low is not None and high is not None, f"{package} 缺少版本区间声明：{ranges.get(package)}"
        assert low <= major_minor < high, (
            f"{package} 声明 [{low}, {high}) 不覆盖实装 {installed}；"
            "请同步 _manifest.json / pyproject.toml / requirements-worker.txt"
        )


def test_pinned_packages_match_the_declared_pillow_ceiling():
    """environment.py 的钉死版本不能越过 manifest 声明的大版本上限。"""
    from recog import environment as environment_mod

    ranges = _declared_ranges()
    low, high = ranges["pillow"]
    pinned = [p for p in environment_mod.PINNED_PACKAGES if p.lower().startswith("pillow==")]
    assert pinned, "PINNED_PACKAGES 里没有钉死 Pillow"
    version = pinned[0].split("==", 1)[1]
    major_minor = float(".".join(version.split(".")[:2]))
    assert low <= major_minor < high, f"钉死的 Pillow {version} 越过声明区间 [{low}, {high})"


# ---------------------------------------------------------------------------
# 发布打包：WebUI 生成页必须被 git 跟踪
#
# `webui.json` 声明 `entry: "webui/capture.html"`，但该文件是 `webui/build.py`
# 的产物、从未 `git add`。按 commit / `git archive` 出包就会缺这个入口，宿主会因
# 整份 WebUI 声明作废而拒绝加载插件 —— 0.5.13 踩过同类的坑。
# ---------------------------------------------------------------------------


def test_every_declared_webui_page_is_tracked_by_git():
    manifest = json.loads((ROOT / "_manifest.json").read_text(encoding="utf-8"))
    pages = []
    for page in manifest.get("webui", {}).get("pages", []) if isinstance(manifest.get("webui"), dict) else []:
        entry = page.get("entry")
        if entry:
            pages.append(entry)
    # webui.json 的另一种声明形态：页面清单在 pages 数组里
    contract = json.loads((ROOT / "webui.json").read_text(encoding="utf-8"))
    for page in contract.get("pages", []) if isinstance(contract.get("pages"), list) else []:
        if isinstance(page, dict) and page.get("entry"):
            pages.append(page["entry"])
    if not pages:
        pytest.skip("未能从 manifest/webui.json 解析出页面清单，跳过")

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60
    ).stdout.splitlines()
    tracked = {line.strip() for line in tracked}
    missing = [p for p in pages if p not in tracked]
    assert not missing, (
        f"以下 WebUI 页面被声明但未被 git 跟踪，按 commit 打包会缺失：{missing}；"
        "请 `git add` 它们"
    )
