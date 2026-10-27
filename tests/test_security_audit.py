"""上线前安全取证用例（OWASP Top 10 + STRIDE）。

这一份是**证据文件**，不是回归守卫。约定：

* 名字以 ``test_*`` 开头的用例断言的是**代码今天真实的行为**。
* 若断言的是一段已知缺陷，用例标题/文档串里标注 ``[已知缺陷 F-0xx]``，
  这样它在 CI 里是绿的，但任何人都能一眼看出"这是被确认存在、尚未修的问题"，
  而不是"这是被认可的行为"。修好之后再回来把断言翻过来。
* 每条取证用例都配**反向对照**：只写正例会拿到"永远绿"的装饰性断言。

插件只做审计，本文件不修改任何被测源码。
"""

from pathlib import Path
from urllib.parse import urlsplit
import ast
import importlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import types

import pytest

np = pytest.importorskip("numpy", reason="numpy 缺失")

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# 无 SDK 环境下的加载脚手架
#
# `plugin.py` 用相对导入（from .recog.service import ...），必须作为包加载。
# 合成包用 `__path__` 指到插件根目录，不必在插件目录里新建 `__init__.py`。
# ---------------------------------------------------------------------------
_PKG = "charrecog_audit_pkg"


class _FieldInstance:
    """stub 的 Field 必须返回**实例**而不是描述符，否则类属性拿到的是描述符对象。"""

    def __init__(self, default=None, **kwargs):
        self.default = default
        self.kwargs = kwargs

    def __repr__(self):
        return f"Field(default={self.default!r})"


def _install_sdk_stub():
    sdk = types.ModuleType("maibot_sdk")

    def api(name, version="1"):
        def decorate(function):
            function.api_name = name
            function.api_version = version
            return function

        return decorate

    def tool(name, description="", parameters=None):
        def decorate(function):
            function.tool_name = name
            function.tool_description = description
            function.tool_parameters = parameters or []
            return function

        return decorate

    def hook(name, timeout_ms=0):
        def decorate(function):
            function.hook_name = name
            function.hook_timeout_ms = timeout_ms
            return function

        return decorate

    class ToolParamType:
        STRING = "string"
        INTEGER = "integer"

    class ToolParameterInfo:
        def __init__(self, name, param_type, description="", required=False):
            self.name = name
            self.param_type = param_type
            self.description = description
            self.required = required

    sdk.API = api
    sdk.Tool = tool
    sdk.HookHandler = hook
    sdk.Field = _FieldInstance
    sdk.PluginConfigBase = type("PluginConfigBase", (), {})
    sdk.MaiBotPlugin = type("MaiBotPlugin", (), {})
    types_mod = types.ModuleType("maibot_sdk.types")
    types_mod.ToolParameterInfo = ToolParameterInfo
    types_mod.ToolParamType = ToolParamType
    sys.modules["maibot_sdk"] = sdk
    sys.modules["maibot_sdk.types"] = types_mod


def _remove_sdk_stub():
    """还原 sys.modules 到本文件加载之前的状态。

    历史背景（0.6.1 起写法已变）：这里原本是无条件 `sys.modules.pop(...)`，
    理由是「不留着自己的桩，免得把 test_tool.py 的 importorskip 变成不跳过」。

    但 `tests/conftest.py` 现在**整个会话**都装着桩，目的正是让
    `test_tool.py` / `test_plugin_loads.py` / `test_annotation_policy.py`
    真正跑起来。而无条件 pop 发生在**模块导入期**，也就是 pytest 的收集阶段 ——
    收集完桩就没了，等那些文件真正执行时 `maibot_sdk` 已消失，
    报的是 `ModuleNotFoundError`（全量回归里实测复现）。

    所以改成保存再还原：不动别人（含 conftest）装好的桩，只把自己盖上去的
    那层撤掉。conftest 不在时才回到「没有桩」的原状。
    """
    for name in ("maibot_sdk", "maibot_sdk.types"):
        previous = _SDK_STUB_PREVIOUS.get(name, _MISSING)
        if previous is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class _Missing:
    """哨兵：表示加载本文件之前 sys.modules 里没有这一项。"""


_MISSING = _Missing()
_SDK_STUB_PREVIOUS = {
    name: sys.modules.get(name, _MISSING) for name in ("maibot_sdk", "maibot_sdk.types")
}


_install_sdk_stub()
try:
    package = types.ModuleType(_PKG)
    package.__path__ = [str(ROOT)]
    sys.modules[_PKG] = package
    PLUGIN = importlib.import_module(f"{_PKG}.plugin")
finally:
    _remove_sdk_stub()

# recog 走常规顶层导入（与既有测试一致），这样共用同一个模块实例
from recog import dependency_download as downloads  # noqa: E402
from recog import images as images_mod  # noqa: E402
from recog import store as store_mod  # noqa: E402
from recog.service import Service  # noqa: E402

CharacterRecognition = PLUGIN.CharacterRecognition


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _api_names(source: str) -> set:
    return set(re.findall(r'@API\("([a-z_]+)"', source))


# ===========================================================================
# A. 反序列化 RCE —— 实战清单第 3 条（本插件重点）
# ===========================================================================

PAYLOAD_SOURCE = "__import__('pathlib').Path(r'{marker}').write_text('PWNED')"


def _build_evil_checkpoint(path: Path, marker: Path) -> None:
    """构造一个在 `torch.load` 时会执行任意代码的 .pth。"""
    import torch

    class Evil:
        def __reduce__(self):
            return (eval, (PAYLOAD_SOURCE.format(marker=str(marker)),))

    # 形状完全仿照 model.py save_pause() 写出的断点字典
    payload = {
        "model": {"classifier.3.weight": Evil()},
        "optimizer": {},
        "initial": {},
        "history": [],
        "coverage": [],
        "best": None,
        "generator": torch.Generator().get_state(),
        "epoch": 1,
        "finished": False,
        "step": 0,
        "order": [],
        "loss": 0.0,
        "seen": 0,
        "random": None,
        "numpy": None,
        "torch": None,
        "cuda": [],
        "seconds": 0.0,
        "peak_rss": 0,
    }
    torch.save(payload, path)


@pytest.mark.parametrize("label", ["torch"])
def test_A1_weights_only_false_executes_arbitrary_code(tmp_path, label):
    """[已知缺陷 F-001] recog/model.py:193 显式传 weights_only=False。

    电磁炮弹已上膛：同一个文件，weights_only=False 会执行 __reduce__。
    """
    torch = pytest.importorskip("torch", reason=f"{label} 缺失")
    marker = tmp_path / "pwned.txt"
    checkpoint = tmp_path / "evil_resume.pt"
    _build_evil_checkpoint(checkpoint, marker)
    assert not marker.exists(), "前置条件失败：标记文件本不该存在"

    torch.load(str(checkpoint), map_location="cpu", weights_only=False)

    assert marker.exists(), (
        "预期 torch.load(weights_only=False) 应执行 payload，但标记文件未出现"
    )
    assert marker.read_text() == "PWNED"


def test_A2_weights_only_true_refuses_the_same_file(tmp_path):
    """反向对照：model.py:50 用的 weights_only=True 挡住了同一个文件。"""
    torch = pytest.importorskip("torch", reason="torch 缺失")
    marker = tmp_path / "pwned.txt"
    checkpoint = tmp_path / "evil_resume.pt"
    _build_evil_checkpoint(checkpoint, marker)

    with pytest.raises(Exception) as caught:
        torch.load(str(checkpoint), map_location="cpu", weights_only=True)

    assert not marker.exists(), "weights_only=True 竟然也执行了 payload"
    assert "Weights only load failed" in str(caught.value), (
        f"异常类型与预期不符：{type(caught.value).__name__}: {caught.value}"
    )


def test_A3_torch_default_is_also_safe(tmp_path):
    """反向对照 2：torch>=2.6 之后缺省就是 weights_only=True。"""
    torch = pytest.importorskip("torch", reason="torch 缺失")
    marker = tmp_path / "pwned.txt"
    checkpoint = tmp_path / "evil_resume.pt"
    _build_evil_checkpoint(checkpoint, marker)

    with pytest.raises(Exception):
        torch.load(str(checkpoint), map_location="cpu")

    assert not marker.exists(), "torch 缺省行为竟然执行了 payload"


def test_A4_forged_pair_no_longer_passes_the_guard(tmp_path):
    """[F-002 已修复] 伪造的「resume.pt + resume-audit.json」一对不再被接受。

    0.6.0 的校验读同目录的 resume-audit.json，同写者即可伪造一对通过
    （原版测试 `test_A4_resume_guard_accepts_a_forged_pair` 逐字复制了那套
    逻辑并断言它确实会放行）。0.6.1 起基准改存数据库，与 resume.pt 分处两个
    文件 —— 现在同样的一对伪造文件，应当被**拒绝**，且不得发生任何代码执行。
    """
    torch = pytest.importorskip("torch", reason="torch 缺失")
    from recog.store import Store

    job = "a" * 32
    root = tmp_path / "root"
    (root / "jobs" / job).mkdir(parents=True)
    store = Store(root)
    snapshot = {"job_id": job, "device": "cpu", "input_size": 320, "images": [], "mode": "fresh"}
    marker = tmp_path / "PWNED"

    job_dir = root / "jobs" / job
    _build_evil_checkpoint(job_dir / "resume.pt", marker)
    forged_sha = __import__("hashlib").sha256((job_dir / "resume.pt").read_bytes()).hexdigest()
    # 攻击者能写目录，所以这一对照样造得出来
    (job_dir / "resume-audit.json").write_text(
        json.dumps({"snapshot_sha": store_mod.digest(snapshot), "sha": forged_sha}), encoding="utf-8"
    )

    # 数据库里没有这个 job 的断点记录 —— 服务端从未认可过这个文件
    assert store.checkpoint_sha(job) is None, "前置条件失败：服务端不该有记录"

    guard_raised = None
    try:
        expected = store.checkpoint_sha(job)
        if expected is None:
            raise ValueError("断点缺少服务端哈希基准，拒绝恢复")
        if expected != forged_sha:
            raise ValueError("checkpoint integrity mismatch")
        torch.load(job_dir / "resume.pt", map_location="cpu", weights_only=True)
    except ValueError as exc:
        guard_raised = str(exc)

    assert guard_raised is not None, "伪造的一对文件本应被拒绝"
    assert not marker.exists(), "伪造断点被加载并执行了 —— RCE 仍然可达"


def test_A5_the_other_three_torch_load_sinks_are_safe():
    """四个 torch.load 落点逐一归类：三个 weights_only=True，一个 False。

    只报"有个 False"是不完整的，必须确认另外三个真的是 True。
    """
    source = _source("recog/model.py")
    loads = re.findall(r"torch\.load\([^\n]*\)", source)
    assert loads, "recog/model.py 里没找到任何 torch.load"

    explicit_true = [line for line in loads if "weights_only=True" in line]
    explicit_false = [line for line in loads if "weights_only=False" in line]

    # 0.6.1：resume.pt 断点那一处已从 False 改成 True（F-001 修复），
    # 所以现在三个落点全部显式 True、零 False。这条断言从「记录缺陷」
    # 变成「防止缺陷回来」—— 谁再把 weights_only 关掉，这里立刻红。
    assert len(loads) == 3, f"torch.load 落点数应为 3，实际 {len(loads)}：{loads}"
    assert len(explicit_true) == 3, f"weights_only=True 应为 3 处，实际：{explicit_true}"
    assert not explicit_false, f"不允许出现 weights_only=False，实际：{explicit_false}"
    # 三个落点分别对应底座 / 线上模型 / 训练断点，都要点名确认
    assert any("BASE_FILE" in line for line in explicit_true), "底座加载未使用 weights_only=True"
    assert any("path," in line or "path)" in line for line in explicit_true), "线上模型加载未使用 weights_only=True"
    assert any("checkpoint" in line for line in explicit_true), "训练断点未使用 weights_only=True"


def test_A5b_legacy_numpy_rng_state_still_loads():
    """0.6.1 起断点里的 numpy RNG state 存成原生 dict，老断点也要能读。"""
    torch = pytest.importorskip("torch", reason="torch 缺失")
    from recog.model import _dump_numpy_state, _load_numpy_state  # noqa: F401

    state = np.random.get_state()
    dumped = _dump_numpy_state(state)
    assert isinstance(dumped, dict) and isinstance(dumped["keys"], list)
    assert all(isinstance(k, int) for k in dumped["keys"])
    restored = _load_numpy_state(dumped)
    assert restored[0] == state[0]
    assert np.array_equal(restored[1], state[1])
    assert restored[2:] == state[2:]
    # 旧格式（直接存 tuple）原样透传，由 set_state 自己处理
    assert _load_numpy_state(state) is state


def test_A6_inference_loads_models_with_weights_only_true():
    """worker.py 加载线上模型时走 loaded_model()，它内部是 weights_only=True。

    这是最要紧的一条：别人塞进来的 weights.pt 不会被 pickled 打穿。
    """
    worker = _source("recog/worker.py")
    assert "loaded_model(path)" in worker, "worker 推理路径不再走 loaded_model"
    loaded = re.findall(r"torch\.load\([^\n]*\)", _source("recog/model.py"))
    sink = [line for line in loaded if "loaded_model" in _source("recog/model.py").split(line)[0][-400:]]
    # loaded_model() 体里那一行
    body = _source("recog/model.py").split("def loaded_model")[1].split("def ")[0]
    assert "weights_only=True" in body, "loaded_model() 不再强制 weights_only=True"
    assert sink is not None


def test_A7_pinned_base_asset_matches_its_sha():
    """assets 里 10MB 的底座权重有硬编码 SHA256，且实测对得上。"""
    import hashlib

    pytest.importorskip("torch", reason="torch 缺失：常量在 recog.model 里，导入即需要 torch")
    from recog import model as model_mod

    actual = hashlib.sha256((ROOT / "assets" / "mobilenet_v3_small-047dcff4.pth").read_bytes()).hexdigest()
    assert actual == model_mod.BASE_SHA, (
        f"实测 {actual}\n常量 {model_mod.BASE_SHA}"
    )


def test_A8_no_api_accepts_a_model_path(tmp_path):
    """可达性边界：47 个 API 里没有一个接受"路径"或"文件名"参数。

    这是 F-001 不可被远程触发的结构性原因 —— 不是靠校验挡住的，
    而是根本没有把路径暴露出来。
    """
    source = _source("plugin.py")
    names = _api_names(source)
    assert len(names) == 47, f"API 数量应为 47，实际 {len(names)}：{sorted(names)}"

    # 每个 API 的参数名都扫一遍
    params = set(re.findall(r"async def \w+\(\s*self,?([^)]*)\)", source))
    flat = " ".join(params)
    for forbidden in ("path", "file", "filename", "file_name", "model_path", "pth", "url", "uri"):
        assert not re.search(rf"\b{forbidden}\b", flat), (
            f"发现接受路径/URL 形态的 API 参数：{forbidden}"
        )


def test_A9_uploads_never_become_pickle_inputs():
    """上传链路终点是 store.ingest()，它对上传文件先 decode() 成 PIL 图像。

    也就是说：即使有人上传一个改名成 .jpg 的 .pth，也到不了 torch.load。
    """
    store = _source("recog/store.py")
    ingest = store.split("def ingest(")[1].split("\n    def ")[0]
    assert "decode_with_animation(raw)" in ingest, "ingest() 不再先解码图像"
    # 入库路径是 images/<uuid><suffix>，与 models/ 的 weights.pt 不同目录
    assert '"images"' in ingest and "uuid.uuid4().hex" in ingest, (
        f"ingest() 的落盘位置与预期不符：{ingest[:200]}"
    )


# ===========================================================================
# B. 凭据外发 / SSRF / 依赖下载投毒 —— 实战清单第 1、2、6 条
# ===========================================================================

WHEEL_NAMES = {
    "torch": "torch-2.11.0+cpu-cp313-cp313-win_amd64.whl",
    "torchvision": "torchvision-0.26.0+cpu-cp313-cp313-win_amd64.whl",
}
GOOD_SHA = "a" * 64

HOSTILE_INDEXES = {
    "正常相对链接": '<a href="@N@#sha256=@S@">w</a>',
    "绝对链接指向恶意主机": '<a href="https://evil.example.com/x/@N@#sha256=@S@">w</a>',
    "协议相对链接 //evil": '<a href="//evil.example.com/x/@N@#sha256=@S@">w</a>',
    "云元数据 169.254.169.254": '<a href="http://169.254.169.254/latest/@N@#sha256=@S@">w</a>',
    "file:// 本地文件": '<a href="file:///C:/Windows/win.ini#sha256=@S@">w</a>',
    "gopher:// 老式 SSRF": '<a href="gopher://evil.example.com/_@N@#sha256=@S@">w</a>',
    "同主机恶意路径": '<a href="/payload/@N@#sha256=@S@">w</a>',
    "HTML 实体混淆 hxxps": '<a href="&#104;ttps://evil.example.com/@N@#sha256=@S@">w</a>',
}


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def fake_wheel_index(monkeypatch):
    """把 urlopen / subprocess.run 换成受控替身，并记录每一次出站请求。"""
    seen = []

    def make(index_bodies):
        def urlopen(request, timeout=None):
            url = request if isinstance(request, str) else request.full_url
            seen.append(url)
            name = "torchvision" if "torchvision" in url else "torch"
            return _Resp(index_bodies[name].encode())

        return urlopen

    class _Probe:
        stdout = json.dumps([[3, 13], "win32", "AMD64"])
        returncode = 0

    monkeypatch.setattr(downloads, "subprocess", types.SimpleNamespace(run=lambda *a, **k: _Probe()))
    return seen, make


@pytest.mark.parametrize("label", list(HOSTILE_INDEXES))
def test_B1_wheel_urls_pins_the_host_no_matter_what_the_index_says(fake_wheel_index, label):
    """dependency_download.py:44 把主机钉死了。

    坑 B 的反面教材是"白名单决定能不能取图"；这里做的是正确的一半：
    索引页的 href 只贡献 **path**，主机永远被覆写成 download.pytorch.org。
    所以即使索引页被中间人改写，也不会打到 evil.example.com / 169.254.169.254。
    """
    seen, make = fake_wheel_index
    template = HOSTILE_INDEXES[label]
    bodies = {
        package: template.replace("@N@", name).replace("@S@", GOOD_SHA)
        for package, name in WHEEL_NAMES.items()
    }
    downloads.urlopen = make(bodies)

    if label in {"file:// 本地文件", "gopher:// 老式 SSRF"}:
        # 这两种连文件名都对不上，直接在遍历里被 continue 掉
        with pytest.raises(ValueError, match="未找到匹配的官方安装包"):
            downloads.wheel_urls("python", "cpu")
        return

    wheels = downloads.wheel_urls("python", "cpu")
    hosts = sorted({urlsplit(url).hostname for url, _ in wheels})
    assert hosts == ["download.pytorch.org"], (
        f"[{label}] 最终下载主机被索引页带跑了：{hosts}"
    )
    # 索引页请求本身也必须只打官方主机
    index_hosts = sorted({urlsplit(url).hostname for url in seen})
    assert index_hosts == ["download.pytorch.org"], (
        f"[{label}] 出站索引请求打到了别处：{index_hosts}"
    )


@pytest.mark.parametrize(
    "template,expected",
    [
        ('<a href="@N@">w</a>', "官方安装包缺少 SHA256"),
        ('<a href="@N@#sha256=deadbeef">w</a>', "官方安装包缺少 SHA256"),
    ],
)
def test_B2_wheel_urls_requires_a_full_sha256_fragment(fake_wheel_index, template, expected):
    """反向对照：缺 sha256 或长度不对 —— 直接拒，而不是"照下不误"。"""
    seen, make = fake_wheel_index
    bodies = {
        package: template.replace("@N@", name) for package, name in WHEEL_NAMES.items()
    }
    downloads.urlopen = make(bodies)
    with pytest.raises(ValueError, match=expected):
        downloads.wheel_urls("python", "cpu")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("../../evil.whl", "安装包 SHA256 校验失败"),  # 穿越被 rsplit 削掉，只剩文件名
        ("a/b.whl", "安装包 SHA256 校验失败"),
        ("evil.txt", "无效安装包文件名"),
        ("evil.whl", "安装包 SHA256 校验失败"),
    ],
)
def test_B3_download_rejects_traversal_and_non_wheel_names(tmp_path, monkeypatch, name, expected):
    """下载落盘的文件名：先卡后缀，再被 rsplit 削掉 ../ 段，最后卡 SHA256。"""
    monkeypatch.setattr(downloads, "urlopen", _always_empty_response)
    with pytest.raises(ValueError, match=expected):
        downloads.download(
            "https://download.pytorch.org/whl/cpu/" + name, "0" * 64, tmp_path, lambda **k: None
        )
    # 无论怎样都不该在目标目录外留下东西
    assert not (tmp_path.parent / "evil.whl").exists()


def _always_empty_response(request, timeout=None):
    class _R:
        status = 200
        headers = {"Content-Length": "0"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            return b""

    return _R()


def test_B4_download_rejects_a_hash_mismatch(tmp_path, monkeypatch):
    """正向对照：内容对不上 hash 就拒收，而不是留下一个坏轮子。"""
    monkeypatch.setattr(downloads, "urlopen", _always_empty_response)
    good = WHEEL_NAMES["torch"]
    with pytest.raises(ValueError, match="安装包 SHA256 校验失败"):
        downloads.download(
            "https://download.pytorch.org/whl/cpu/" + good, "0" * 64, tmp_path, lambda **k: None
        )


def test_B5_tls_verification_is_never_disabled():
    """清单第 6 条点名的 `verify=False` 临时写法：这份代码里没有。"""
    source = _source("recog/dependency_download.py") + _source("recog/environment.py")
    for bad in ("verify=False", "_create_unverified", "ssl._create", "check_hostname=False"):
        assert bad not in source, f"发现被关闭的 TLS 校验：{bad}"
    # 出站用的是裸 urlopen + Request，继承 urllib 的默认证书校验
    assert "urlopen(Request(url, headers=headers), timeout=30)" in source


def test_B6_no_http_client_library_is_used_at_all():
    """清单 1/2 的前提检查：插件根本没装 httpx / requests / aiohttp。

    所以「httpx 重定向时无条件 pop Cookie」「client.cookies 无域名限定」
    这两条坑在这里**不成立** —— 前提不存在。写成"未命中"而不是"通过"。
    """
    sources = "".join(
        _source(name)
        for name in (
            "plugin.py",
            "recog/store.py",
            "recog/service.py",
            "recog/model.py",
            "recog/worker.py",
            "recog/environment.py",
            "recog/dependency_download.py",
            "recog/images.py",
            "recog/scores.py",
            "recog/bootstrap.py",
            "recog/launcher.py",
            "recog/probes.py",
            "recog/processes.py",
        )
    )
    for client in ("import httpx", "import requests", "import aiohttp", "httpx.", "requests.get", "urllib3"):
        assert client not in sources, f"发现了 HTTP 客户端库：{client}"
    # 唯一的出站是标准库 urllib
    assert "from urllib.request import Request, urlopen" in sources


def test_B7_download_sends_no_credentials():
    """清单第 1 条的实质：即使被重定向，也没有凭据可漏。"""
    source = _source("recog/dependency_download.py")
    headers = re.search(r"headers = \{([^}]*)\}", source).group(1)
    for secret in ("Cookie", "Authorization", "Authorization", "X-API-Key", "token"):
        assert secret not in headers, f"下载请求头里带了凭据：{secret}"
    assert "User-Agent" in headers
    # 没有 cookie jar / 会话对象
    for bad in ("HTTPCookieProcessor", "CookieJar", "session", "cookies"):
        assert bad not in source, f"发现了凭据容器：{bad}"


def test_B8_index_url_is_hardcoded_not_user_supplied():
    """清单第 2 条：没有任何用户可控 URL 进入取数路径。

    SSRF 的结论是"不适用"，而不是"已缓解" —— 因为没有那条数据通路。
    """
    source = _source("recog/dependency_download.py")
    urls = re.findall(r'f?"(https?://[^"]+)"', source)
    assert urls, "没找到任何硬编码 URL"
    for url in urls:
        assert url.startswith("https://download.pytorch.org"), f"硬编码 URL 指向了别处：{url}"
    # 用户唯一能影响的只有 build 参数，且被白名单收口
    assert 'build = "cpu" if kind == "cpu" else "cu128"' in _source("recog/environment.py")


# ===========================================================================
# C. 路径穿越 —— 实战清单第 4 条
# ===========================================================================

TRAVERSAL = [
    "../../../../Windows/System32/calc.exe",
    "..%2f..%2fetc/passwd",
    "..\\..\\..\\evil",
    "....//....//evil",
    "/etc/passwd",
    "a" * 32 + "/..",
    "A" * 32,  # 大写不是合法 32 位 hex
    "g" * 32,
]


@pytest.mark.parametrize("payload", TRAVERSAL)
def test_C1_clean_ids_rejects_every_traversal_shape(payload):
    """_clean_ids 是 job/version 编号的唯一入口，卡死 32 位小写 hex。

    所有 ID 类的路径拼接（jobs/<id>、models/<id>）都靠它收口。
    """
    with pytest.raises(ValueError, match="无效任务编号"):
        Service._clean_ids(json.dumps([payload]), "任务")


def test_C1b_clean_ids_accepts_a_real_id():
    """反向对照：合法 32 位 hex 必须照常放行，否则上面的拒绝毫无意义。"""
    good = "0123456789abcdef" * 2
    assert Service._clean_ids(json.dumps([good]), "任务") == [good]
    assert Service._clean_ids([good], "任务") == [good]
    # 去重与上限也要生效，否则批量删除会变成放大倍数
    assert Service._clean_ids([good, good], "任务") == [good]
    with pytest.raises(ValueError, match="每次最多删除"):
        Service._clean_ids([f"{i:032x}" for i in range(51)], "任务")
    # 裸字符串不是合法形态（单独一个 id 也得装在数组里）
    with pytest.raises(ValueError, match="编号格式不正确"):
        Service._clean_ids(good, "任务")


@pytest.mark.parametrize("result_id", ["../../evil", "..", "a" * 31, "a" * 33, "A" * 32, ""])
def test_C2_read_test_image_rejects_traversal(tmp_path, result_id):
    """read_test_image 在拼路径之前先卡 32 位小写 hex。

    合法形态走不到这里（会被引擎转发到"已过期"分支），所以反面用例只测
    非法形态。read_test_image 是 staticmethod，参数需要 .store 属性。
    """
    plugin = _stub_plugin(store_mod.Store(tmp_path))
    with pytest.raises(ValueError, match="测试图片不存在"):
        CharacterRecognition.read_test_image(plugin, result_id)


def _stub_plugin(store):
    """只提供 read_test_image 需要的那一段属性链。"""
    return types.SimpleNamespace(service=types.SimpleNamespace(store=store))


def test_C2b_a_well_formed_id_still_does_not_escape_the_directory(tmp_path):
    """反向对照：连合法的 32 位 hex 也拿不到目录外的东西。

    它先在 test_history 白名单里查 id，查不到就拒 —— 不是"拼出来再看在不在"。
    """
    plugin = _stub_plugin(store_mod.Store(tmp_path))
    with pytest.raises(ValueError, match="测试图片已过期"):
        CharacterRecognition.read_test_image(plugin, "a" * 32)
    # 目录外没有任何文件被读到，也没有文件被创建
    assert not (tmp_path.parent / ("a" * 32 + ".image")).exists()


def test_C3_ingest_refuses_files_outside_the_upload_dir(tmp_path):
    """上传落点是 root/uploads，外面的路径一律拒。"""
    store = store_mod.Store(tmp_path)
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    with pytest.raises(ValueError, match="只能领取本插件上传文件"):
        store.ingest(outside, "other", "角色A")


def test_C3b_ingest_accepts_files_inside_the_upload_dir(tmp_path):
    """反向对照：uploads 里的正常文件要能进去，否则上面是"啥都拒"。"""
    import struct
    import zlib

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    # 每行前面要有 1 字节 filter，这是手写 PNG 最容易漏的一处
    scanlines = b"".join(b"\x00" + b"\x40\x60\x80" * 8 for _ in range(8))
    raw = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )
    store = store_mod.Store(tmp_path)
    target = tmp_path / "uploads" / "ok.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    result = store.ingest(target, "other", "角色A")
    assert result["id"], f"uploads 内的合法图片未能入库：{result}"


def test_C4_ingest_refuses_a_symlinked_claim(tmp_path):
    """符号链接指向外面的文件：resolve() 之后就不再属于 uploads，拒。

    注意：这台 Windows 机器上 os.symlink 会"成功"但落下来的是一个 0 字节普通文件
    （无开发者模式、无管理员），is_symlink() 返回 False。所以这里必须先确认
    链接真的是链接，否则测的就不是同一回事。
    """
    store = store_mod.Store(tmp_path)
    outside = tmp_path.parent / "symtarget.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    link = tmp_path / "uploads" / "link.png"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("当前环境无法创建符号链接（Windows 需要开发者模式/管理员）")
    if not link.is_symlink() or not link.exists():
        pytest.skip(
            "os.symlink 未返回真实链接（Windows 无 SeCreateSymbolicLinkPrivilege），"
            "is_symlink()=False —— 本机无法行为化复现，见 test_C4b"
        )
    with pytest.raises(ValueError, match="只能领取本插件上传文件"):
        store.ingest(link, "other", "角色A")


def test_C4b_the_symlink_guard_exists_and_is_reachable(tmp_path):
    """本机建不出真链接时，退化成结构断言：守卫在、且守卫的是同一个根。

    `or path.is_symlink()` 是 `is_relative_to` 之外的**第二道**保险：
    前者挡"resolve 之前的原路径就是链接"，后者挡"resolve 之后跑出根"。
    """
    source = _source("recog/store.py")
    ingest = source.split("        character: str = \"\",\n        source: str = \"\",")[1].split(
        "\n        return {\"id\": ident"
    )[0]
    assert "path = path.resolve()" in ingest, "ingest 不再 resolve 传入路径"
    assert "or path.is_symlink()" in ingest, "符号链接这道守卫消失了"
    assert "is_relative_to(self.root / \"uploads\")" in ingest, "目录逃逸这道守卫消失了"


def test_C5_validate_character_accepts_traversal_shapes_but_is_never_a_path():
    """[观察项 F-003] 角色名只卡长度 32，不卡字符集。

    '../../evil'、含 NUL 的名字都能过。但**它从来不参与路径拼接**：
    所有含 character 的 f-string 都是 settings 的 key 或展示文案。
    所以这是纵深防御缺口，不是已复现的漏洞。
    """
    for name in ("../../evil", "a/../b", "x\x00y", "x" * 32):
        accepted = store_mod.Store.validate_character(name)
        assert accepted, f"角色名被拒：{name!r}"
    with pytest.raises(ValueError, match="角色名过长"):
        store_mod.Store.validate_character("x" * 33)

    # 关键：character 从不与路径拼接相遇
    offenders = []
    for relative in ("recog/store.py", "recog/service.py", "recog/worker.py", "plugin.py"):
        for number, line in enumerate(_source(relative).splitlines(), 1):
            if "character" not in line:
                continue
            if re.search(r"/\s*\"(jobs|models|images|uploads|thumbnails|test-images)\"", line):
                offenders.append(f"{relative}:{number}")
            if re.search(r"Path\([^)]*character", line):
                offenders.append(f"{relative}:{number}")
    assert not offenders, f"角色名流进了路径拼接：{offenders}"


def test_C6_delete_jobs_and_versions_reject_traversal(tmp_path):
    """删除动作走 _clean_ids，穿越形态进不来。"""
    service = Service(tmp_path)
    for payload in ('["../../evil"]', '["..%2f..%2fevil"]', '["' + "A" * 32 + '"]'):
        with pytest.raises(ValueError, match="无效(任务|版本)编号"):
            service.delete_jobs(payload)
        with pytest.raises(ValueError, match="无效(任务|版本)编号"):
            service.delete_versions(payload)


# ===========================================================================
# D. 注入类（SQL / 命令 / 代码）—— 清单第 5、8 条
# ===========================================================================


@pytest.mark.parametrize(
    "view",
    [
        "all",
        "eligible",
        "excluded",
        "unlabelled",
        "errors",
        "removed",
        "captured",
    ],
)
def test_D1_images_view_is_allowlisted_positive(tmp_path, view):
    """正向对照：7 个合法视图都要能出结果。"""
    store = store_mod.Store(tmp_path)
    result = store.images(1, view, "", "洛天依")
    assert result["rows"] == [], f"合法视图 {view} 反而没返回结果"


@pytest.mark.parametrize(
    "view",
    [
        "all; DROP TABLE images--",
        "1=1",
        "all' OR '1'='1",
        "all UNION SELECT value FROM settings--",
        "captured; ATTACH DATABASE 'x' AS y--",
    ],
)
def test_D1b_images_view_is_allowlisted_negative(tmp_path, view):
    """store.py:552 的 view 白名单是这里唯一的防线，实测挡得住。"""
    store = store_mod.Store(tmp_path)
    with pytest.raises(ValueError, match="无效分页"):
        store.images(1, view, "", "洛天依")


def test_D1c_injection_attempts_leave_the_schema_intact(tmp_path):
    """打完之后表还在 —— 白名单生效的硬证据。"""
    store = store_mod.Store(tmp_path)
    for view in ("all; DROP TABLE images--", "all' OR '1'='1", "1 UNION SELECT 1--"):
        try:
            store.images(1, view, "", "洛天依")
        except ValueError:
            pass
    with store.db() as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        rows = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        settings = db.execute("SELECT COUNT(*) FROM settings").fetchone()[0]
    expected = {"images", "groups", "jobs", "models", "settings"}
    assert expected <= tables, f"表被删了：{tables}"
    assert rows == 0 and settings == 0, "注入竟然写出了数据"


@pytest.mark.parametrize(
    "character",
    ["洛天依", "洛天依' OR 1=1--", "x'; DROP TABLE images; --", "x UNION SELECT value FROM settings--"],
)
def test_D2_character_is_bound_as_data_not_sql(tmp_path, character):
    """角色名能带引号，但它是 ? 绑定的值，不是拼进 SQL 的片段。

    「能存进去」≠「能注入」。这里验证的是后半句。
    """
    store = store_mod.Store(tmp_path)
    store.images(1, "all", character, "洛天依")
    with store.db() as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "images" in tables, "images 表被 DROP 了"
    # 真的被当成字符串存了下来
    with store.db() as db:
        db.execute("INSERT INTO images(id,path,character) VALUES ('x','x',?)", (character,))
        stored = db.execute("SELECT character FROM images WHERE id='x'").fetchone()[0]
    assert stored == character, f"字符名被 SQL 片段化了：{stored!r}"


def test_D3_every_fstring_sql_identifier_comes_from_a_fixed_set():
    """store.py 里 6 处 f-string SQL 逐个点名，确认标识符不是用户输入。

    f-string 本身不是罪，罪是 f-string 里插了外部输入。这里插的是：
    固定元组里的表名、",".join("?") 生成的占位符、以及字典查表得到的常量 where。
    """
    source = _source("recog/store.py")
    fstrings = re.findall(r'execute\(f"([^"]+)"', source) + re.findall(r"execute\(\s*f\"([^\"]+)\"", source)
    assert fstrings, "没找到 f-string SQL"

    placeholders = [s for s in fstrings if "({placeholders})" in s or "({marks})" in s]
    assert placeholders, "占位符拼接用法消失了，需重新评估"
    # 每一处拼接都必须由 "?".join 生成：动态部分只能是问号的个数，不能有内容
    for statement in placeholders:
        for dynamic in re.findall(r"\{(\w+)\}", statement):
            assert dynamic in {"placeholders", "marks"}, f"f-string SQL 里插了别的变量：{dynamic}"
    for name in ("placeholders", "marks"):
        assert re.search(rf'{name} = ","\.join\("\?" for _ in \w+\)', source), (
            f"{name} 不再是纯问号拼接：需人工复核 SQL"
        )

    # where/order 来自字典查表，键集合被 view 白名单收口
    assert '"captured": CAPTURE_REVIEWABLE' in source
    assert '}[view]' in source
    # 表名来自固定元组
    assert 'for table in ("images", "jobs", "models"):' in source


@pytest.mark.parametrize("bad", ["x", "x' OR 1=1--", "a" * 31, "A" * 32, "a" * 32 + ";DROP"])
def test_D4_batch_update_rejects_non_hex_ids(tmp_path, bad):
    """批量标注的 ID 也走 32 位 hex 校验。"""
    store = store_mod.Store(tmp_path)
    with pytest.raises(ValueError):
        store.batch_update([bad], "self")


def test_D5_no_shell_true_anywhere():
    """清单第 5 条：全仓库没有 shell=True / os.system。"""
    for relative in ("plugin.py", "recog/service.py", "recog/environment.py", "recog/dependency_download.py"):
        source = _source(relative)
        for bad in ("shell=True", "os.system(", "os.popen(", "subprocess.getoutput"):
            assert bad not in source, f"{relative} 里发现了 {bad}"


# 已知"一定返回 list"的构造者：runtime_command 用 `[*command, *args]` 收尾。
_LIST_RETURNING_CALLS = {"runtime_command", "self.runtime_command"}


def _subprocess_sites(relative):
    """收集一个模块里所有 subprocess.run/Popen/call/check_output 调用。"""
    tree = ast.parse(_source(relative))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"run", "Popen", "call", "check_output"}:
            continue
        if ast.unparse(node.func.value) != "subprocess":
            continue
        found.append((tree, node))
    return found


def _is_list_expression(node):
    """节点是否**结构上**就是一个 list（含 *a 解包的列表显示）。"""
    if isinstance(node, ast.List):
        return True
    if isinstance(node, ast.Call) and ast.unparse(node.func) in _LIST_RETURNING_CALLS:
        return True
    return False


def test_D6_subprocess_argv_are_lists_not_strings():
    """命令注入的结构性前提不存在：所有 subprocess 首参最终都是 list。

    用 AST + 一层过程内解析，而不是正则 —— 这些调用都是跨行书写的，
    正则会把 `subprocess.run(\\n  [a, b], ...)` 误判成"传了字符串"。
    """
    offenders = []
    resolved = []
    sites = 0
    for relative in (
        "plugin.py",
        "recog/service.py",
        "recog/environment.py",
        "recog/dependency_download.py",
        "recog/bootstrap.py",
        "recog/launcher.py",
    ):
        for tree, node in _subprocess_sites(relative):
            sites += 1
            if not node.args:
                offenders.append(f"{relative}:{node.lineno} 无位置参数")
                continue
            first = node.args[0]
            if _is_list_expression(first):
                # list 里不许出现 ["sh","-c",<f-string>] 这种把字符串又拼回去的写法
                if isinstance(first, ast.List):
                    for element in first.elts:
                        if isinstance(element, ast.JoinedStr):
                            offenders.append(f"{relative}:{node.lineno} list 内含 f-string 元素")
                        if isinstance(element, ast.List):
                            offenders.append(f"{relative}:{node.lineno} 首参是嵌套 list")
                resolved.append(f"{relative}:{node.lineno} 字面 list")
                continue
            if not isinstance(first, ast.Name):
                offenders.append(f"{relative}:{node.lineno} 首参类型 {type(first).__name__}")
                continue

            # 变量形态：往上找它到底被赋成什么
            name = first.id
            owner = next(
                (
                    candidate
                    for candidate in ast.walk(tree)
                    if isinstance(candidate, ast.FunctionDef) and node in ast.walk(candidate)
                ),
                None,
            )
            origin = None
            if owner is not None:
                # 形参：去查这个函数的**所有**调用方
                if name in {arg.arg for arg in owner.args.args}:
                    index = [arg.arg for arg in owner.args.args].index(name)
                    for caller in ast.walk(tree):
                        if not isinstance(caller, ast.Call):
                            continue
                        if ast.unparse(caller.func) != owner.name:
                            continue
                        if len(caller.args) <= index:
                            origin = "缺少实参"
                            break
                        if _is_list_expression(caller.args[index]):
                            origin = f"调用方传 list（{owner.name} 第 {index} 参）"
                        else:
                            origin = f"调用方传 {type(caller.args[index]).__name__}"
                            break
                else:
                    for assign in ast.walk(owner):
                        if not isinstance(assign, ast.Assign):
                            continue
                        if any(
                            isinstance(target, ast.Name) and target.id == name
                            for target in assign.targets
                        ):
                            origin = (
                                "赋成 list"
                                if _is_list_expression(assign.value)
                                else f"赋成 {type(assign.value).__name__}"
                            )
            if origin is None:
                # 全模块范围内没有任何调用方 => 这条路径不可达
                origin = "无调用方（死代码，不可达）"
            if origin.startswith(("赋成 list", "调用方传 list", "无调用方")):
                resolved.append(f"{relative}:{node.lineno} 变量 {name} -> {origin}")
            else:
                offenders.append(f"{relative}:{node.lineno} 变量 {name} -> {origin}")

    assert sites == 9, f"subprocess 站点数 {sites}（预期 9），代码可能已重构"
    assert not offenders, f"subprocess 调用没有用 list 传参：{offenders}"
    assert len(resolved) == 9, f"只有 {len(resolved)}/{sites} 个站点被解析：{resolved}"


def test_D7_worker_argv_never_interpolates_a_command_string():
    """训练/推理进程的参数是拼接的 list，job id 只作为独立 argv 元素。

    所以即使 job id 带了空格或分号，也不会被 shell 解释。
    """
    source = _source("recog/service.py")
    spawn = source.split("def _spawn(")[1].split("\n    def ")[0]
    assert 'command.append(job)' in spawn, "job id 被拼进命令字符串而不是作为独立 argv"
    assert "self.runtime_command(" in spawn
    runtime = source.split("def runtime_command(")[1].split("\n    def ")[0]
    assert "return [*command, *args]" in runtime, "runtime_command 不再返回 list"


def test_D8_probes_replaced_dynamic_code_execution():
    """历史坑：`-c <snippet>` 动态执行已被命名探针取代。"""
    probes = _source("recog/probes.py")
    assert "Replaces the former `-c <snippet>` execution mode" in probes
    # 探针名查表，不是任意代码
    assert "if name not in PROBES:" in probes
    assert 'raise SystemExit(f"unknown probe' in probes


# ===========================================================================
# E. 日志泄露与硬编码凭据 —— 清单第 9、3 条
# ===========================================================================


def test_E1_no_hardcoded_secrets():
    """硬编码密钥：静态扫描零命中（与 team-lead 的既有结论一致，这里保留证据）。"""
    sources = {
        name: _source(name)
        for name in (
            "plugin.py",
            "recog/store.py",
            "recog/service.py",
            "recog/model.py",
            "recog/worker.py",
            "recog/environment.py",
            "recog/dependency_download.py",
            "recog/scores.py",
            "recog/images.py",
            "recog/bootstrap.py",
            "recog/launcher.py",
        )
    }
    offenders = []
    for name, source in sources.items():
        for pattern in (
            r"(api[_-]?key|secret|password|passwd|token|credential)\s*[:=]\s*[\"'][^\"'\s{}]{8,}[\"']",
            r"BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY",
            r"sk-[A-Za-z0-9]{20,}",
            r"ghp_[A-Za-z0-9]{20,}",
        ):
            for match in re.finditer(pattern, source, re.IGNORECASE):
                line = source[: match.start()].count("\n") + 1
                offenders.append(f"{name}:{line} {pattern[:24]}")
    assert not offenders, f"发现硬编码凭据：{offenders}"


def test_E2_no_response_body_is_logged():
    """清单第 9 条点名的 `logger.error(res.text[:500])`：这里没有。

    前提也不存在 —— 插件没有任何 HTTP 响应对象可打。
    """
    for relative in ("plugin.py", "recog/store.py", "recog/service.py", "recog/worker.py"):
        source = _source(relative)
        for bad in (".text[:", "res.text", "response.text", "r.text", ".content[:", ".body"):
            assert bad not in source, f"{relative} 里发现了响应体打印：{bad}"


def test_E3_logger_calls_use_lazy_percent_formatting():
    """logging 用惰性 %s，异常对象才不会在构造消息时被拼进敏感串。

    而且这些异常来源全是本地（sqlite3 / OSError / 本地 torch），
    栈里没有 HTTP 适配器，也就不会夹带 cookie。
    """
    offenders = []
    for relative in ("plugin.py", "recog/store.py", "recog/service.py", "recog/worker.py"):
        source = _source(relative)
        for number, line in enumerate(source.splitlines(), 1):
            if "_LOG." not in line and "logger." not in line:
                continue
            # 允许 exc_info=True 的惰性调用，禁止 f-string 插值
            if re.search(r'_LOG\.\w+\(f"', line) or re.search(r"logger\.\w+\(f\"", line):
                offenders.append(f"{relative}:{number} {line.strip()[:70]}")
    assert not offenders, f"日志用了 f-string 插值：{offenders}"


def test_E4_logged_exception_sources_carry_no_credentials():
    """清单第 9 条第二条：异常对象夹带凭据。这里的前提不存在。"""
    sources = _source("plugin.py") + _source("recog/service.py") + _source("recog/store.py")
    # 会进日志的异常类型只有这几种，全部来自本地调用
    for bad in ("httpx", "requests.", "aiohttp", "urllib.error.HTTPError"):
        assert bad not in sources, f"存在会夹带凭据的异常来源：{bad}"
    # 唯一的远端异常来源是下载，它只进 ValueError("安装包 SHA256 校验失败") 不进日志
    downloads_source = _source("recog/dependency_download.py")
    assert "_LOG" not in downloads_source, "下载模块不该直接打日志"


# ===========================================================================
# F. 资源耗尽 DoS —— 清单第 11 条
# ===========================================================================


def test_F1_decode_rejects_oversized_images():
    """解码前先量尺寸，40M 像素上限；且 DecompressionBombWarning 被提为错误。"""
    source = _source("recog/images.py")
    assert "MAX_BYTES = 20 * 1024 * 1024" in source
    assert "MAX_PIXELS = 40_000_000" in source
    assert 'simplefilter("error", Image.DecompressionBombWarning)' in source
    assert "width * height > MAX_PIXELS" in source
    # 尺寸必须在 draft 之前读，否则会被 draft 改小蒙混过关
    body = source.split("def _decode(")[1].split("\ndef ")[0]
    assert body.index("width, height = source.size") < body.index("source.draft(")


def test_F2_decode_restricts_containers_and_always_takes_the_first_frame():
    """0.7.0 起：容器白名单是 JPEG/PNG/WebP/GIF，动图不再被拒，统一 seek(0) 取首帧。

    这条测试守两件事：白名单不能被悄悄放宽到「任何 Pillow 认识的东西」；
    「动图请走静态」这条旧语义不能以任何形式残留（否则会静默回到拒收行为）。
    """
    source = _source("recog/images.py")
    body = source.split("def _decode(")[1].split("\ndef ")[0]
    assert 'SUPPORTED_FORMATS = ("JPEG", "PNG", "WEBP", "GIF")' in source
    assert "source.format not in SUPPORTED_FORMATS" in body
    # 旧语义的痕量：帧数硬拒与其文案都必须彻底消失
    assert "n_frames" not in body
    assert "请上传静态图片" not in body
    # 首帧必须在 load() 之前定位，否则动图会解到当前帧
    assert body.index("source.seek(0)") < body.index("source.load()")


def test_F3_foreign_and_corrupt_formats_are_still_rejected():
    """反向对照：GIF 现在能进，BMP/SVG/截断文件仍然进不来。"""
    from PIL import Image  # 本文件其余部分只读源码，不需要 PIL

    valid = io.BytesIO()
    Image.new("RGB", (8, 6), "red").save(valid, format="GIF")
    assert images_mod.decode(valid.getvalue()).size == (8, 6)

    for label, data in (
        ("bmp", b"BM" + b"\x00" * 64),
        ("svg", b"<svg xmlns='x'/>"),
        ("truncated-gif", b"GIF89a" + b"\x00" * 32),
    ):
        with pytest.raises(Exception) as caught:
            images_mod.decode(data)
        assert type(caught.value).__name__ in {"UnidentifiedImageError", "OSError", "ValueError"}, (
            f"[{label}] 异常类型 {type(caught.value).__name__}"
        )


def test_F4_capture_uses_an_unbounded_executor_queue():
    """[已知缺陷 F-004] plugin.py:177 用 loop.run_in_executor(None, ...)。

    None 走默认 ThreadPoolExecutor，它的 _work_queue 是 SimpleQueue —— **无界**。
    每张入队图片都握着最多 20MiB 的原始字节，队列不设上限意味着内存随
    聊天图片洪流线性增长。CAPTURE_LIMIT 只卡"入库数量"，不卡"在途工作量"。
    """
    import concurrent.futures
    import queue

    executor = concurrent.futures.ThreadPoolExecutor()
    try:
        assert isinstance(executor._work_queue, queue.SimpleQueue), (
            "默认执行器的队列不再是 SimpleQueue，结论需重估"
        )
    finally:
        executor.shutdown(wait=False)

    source = _source("plugin.py")
    assert "loop.run_in_executor(None, self._capture, raw, results)" in source
    # CAPTURE_LIMIT 的检查发生在"入库"之前，而不是在"入队"之前
    capture = source.split("def _capture(")[1].split("\n    async def ")[0]
    assert capture.index("capture_pending(character) >= CAPTURE_LIMIT") < capture.index(
        "target.write_bytes(raw)"
    ), "CAPTURE_LIMIT 的检查位置变了"
    assert capture.index("target.write_bytes(raw)") < capture.index("store.ingest(")


def test_F5_capture_has_a_storage_limit_and_a_size_cap():
    """缓解措施确实存在：100 条待确认上限 + base64 入口上限。

    0.6.1 起上限不再硬编码 28 MiB，而是从 `images.MAX_BYTES` 推导的
    `MAX_BASE64_BYTES`（base64 放大 4/3）。断言改成核对常量本身与推导关系，
    而不是某一行字面量 —— 否则统一口径的改动会把这条测试撞红。
    """
    source = _source("plugin.py")
    assert "MAX_BASE64_BYTES" in source
    assert "len(binary) > MAX_BASE64_BYTES" in source
    assert "if self.service.store.capture_pending(character) >= CAPTURE_LIMIT:" in source
    assert store_mod.CAPTURE_LIMIT == 100
    # 推导关系必须成立：base64 闸门换算回原始字节，不能越过 MAX_BYTES
    from recog.images import MAX_BASE64_BYTES, MAX_BYTES

    assert MAX_BASE64_BYTES * 3 // 4 <= MAX_BYTES
    assert MAX_BASE64_BYTES * 3 // 4 >= MAX_BYTES - 3


def test_F6_one_training_job_at_a_time_is_enforced_by_the_database():
    """并发训练上限：除了应用层的检查，DB 里还有部分唯一索引兜底。"""
    source = _source("recog/store.py")
    assert "CREATE UNIQUE INDEX IF NOT EXISTS one_training ON jobs((1))" in source
    assert "CREATE UNIQUE INDEX IF NOT EXISTS one_training_v2 ON jobs((1))" in source
    service = _source("recog/service.py")
    assert "上一个训练进程尚未退出" in service
    assert "已有训练任务" in service


def test_F7_batch_actions_are_capped():
    """批量操作每批最多 1000 张，删除每条最多 50 条。"""
    assert "len(ids) > 1000" in _source("recog/store.py")
    assert "DELETE_LIMIT = 50" in _source("recog/service.py")


# ===========================================================================
# G. 架构 / 访问控制 / 提示词注入面
# ===========================================================================


def test_G1_capabilities_match_actual_usage():
    """声明的 2 个能力，代码里正好 3 处调用，不多不少。"""
    manifest = json.loads((ROOT / "_manifest.json").read_text(encoding="utf-8"))
    declared = set(manifest["capabilities"])
    assert declared == {"webui.claim_upload", "message.get_by_id"}, f"清单声明变了：{declared}"

    source = _source("plugin.py")
    assert source.count("claim_upload(") == 2, "claim_upload 调用数与声明不一致"
    assert source.count("get_by_id(") == 1, "get_by_id 调用数与声明不一致"

    # webui.json 用的是宿主的能力词汇（custom_pages / image_upload），
    # 与 _manifest.json 的 SDK 能力词汇不同，两者都要在。
    webui = json.loads((ROOT / "webui.json").read_text(encoding="utf-8"))
    page_capabilities = set(webui.get("required_capabilities") or [])
    assert {"custom_pages", "image_upload"} <= page_capabilities, (
        f"webui.json 缺少页面/上传能力声明：{page_capabilities}"
    )
    assert len(webui["pages"]) == 6, f"页面数变了：{[p['id'] for p in webui['pages']]}"


def test_G2_every_html_action_is_declared_in_the_webui_contract():
    """页面里用到的 action 名必须都在 webui.json 里，否则宿主不认。"""
    webui = json.loads((ROOT / "webui.json").read_text(encoding="utf-8"))
    declared = set()
    for page in webui["pages"]:
        declared.update((page.get("actions") or {}).keys())
        declared.update((page.get("queries") or {}).keys())

    used = set()
    for html in sorted((ROOT / "webui").glob("*.html")):
        used.update(re.findall(r'action="([a-z_]+)"', html.read_text(encoding="utf-8")))
    assert used, "页面里没找到任何 action"
    missing = sorted(used - declared)
    assert not missing, f"页面用了未在 webui.json 声明的 action：{missing}"


def test_G3_eight_apis_are_not_reachable_through_the_webui_contract():
    """有 8 个 API 声明了但没接进 webui.json —— 攻击面反而更小。

    记在这里是为了让"45 个 API"这个数字不被误读成"45 个可达端点"。
    """
    source = _source("plugin.py")
    webui = json.loads((ROOT / "webui.json").read_text(encoding="utf-8"))
    wired = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "api" and isinstance(value, str):
                    wired.add(value)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(webui)
    declared = _api_names(source)
    assert len(declared) == 47, f"API 总数变了：{len(declared)}"
    assert len(wired) == 39, f"webui 接线数变了：{len(wired)}"
    unreachable = sorted(declared - wired)
    assert unreachable == [
        "self_clear_errors",
        "self_images_batch",
        "self_label_clear",
        "self_label_negative",
        "self_label_positive",
        "self_select_errors",
        "self_select_report",
        "self_training_new_request",
    ], f"不可达 API 集合变了：{unreachable}"
    # 反向：不能有"接了线但没声明"的
    assert not (wired - declared), f"webui.json 引用了不存在的 API：{sorted(wired - declared)}"


def test_G4_prompt_injection_surface_is_a_fixed_template():
    """清单第 7 条（间接提示词注入 / fail-open 拒答）：**不适用**。

    插件唯一的产出是「[提示：此图中的角色看起来很可能是 "X"]」这个定长模板，
    X 来自管理员配置的角色名，**没有任何远端内容**进入它：
    没有 OCR、没有图注、不读别人发的消息文本、不调用 LLM、没有 LLM 输出要净化。
    因此"模型拒答被当成正常内容发布"这条坑的前提不存在。
    """
    source = _source("plugin.py")
    template = 'f"\\n[提示：此图中的角色看起来很可能是 “{top[\'character\']}”]\\n"'
    assert template in source, "注入到消息里的模板变了，需重新评估提示词注入面"

    # 模板里只有一个插值位，且它的来源链完全在管理员手里：
    #   settings.active:<角色>  <- 管理员点"启用"  <- DB 的 models 行
    # self_model_activate 从 DB 取角色名，根本不接受客户端传值。
    # 0.6.1 起那次读取被挪进 `_model_character`（为了走 offload，避免同步
    # SQLite 挂住事件循环），所以断言跟着改成「读取必须发生在 DB 查询函数里」，
    # 而不是钉死某一行赋值语句的长相。
    assert "def _model_character(" in source, "activate 的 DB 读取应集中在 _model_character"
    assert "await offload(self._model_character" in source, "activate 必须经 offload 读 DB"
    assert len(re.findall(r"\{[^{}]+\}", template)) == 1, "模板出现了第二个插值位"

    # 钩子体内不存在任何远端文本来源（OCR / 图注 / 发送者昵称 / 别人发的正文）
    hook = source.split("async def annotate_received_images(")[1].split("\n    async def on_unload")[0]
    for bad in ("ocr", "caption", "alt_text", "nickname", "sender", "text_content", "content_text"):
        assert not re.search(rf"\b{bad}\b", hook, re.IGNORECASE), f"钩子内出现了远端文本来源：{bad}"
    assert "get_by_id" not in hook, "消息钩子不该再去取消息正文"


def test_G5_remote_images_only_contribute_a_float():
    """收到的图片只贡献一个 float 分数，不贡献任何字符串。

    这是 G4 那条"不适用"结论的正面证据：唯一跨信任边界进入 LLM 上下文的
    东西是管理员写的角色名。
    """
    source = _source("plugin.py")
    hook = source.split("async def annotate_received_images(")[1].split("\n    async def on_unload")[0]
    # 注入的那段文本里出现的变量只有 character，不含 raw / binary / 任何图片派生值
    injected = hook.split('"data": f"\\n[提示')[1].split('\n')[0]
    for forbidden in ("raw", "binary", "base64", "b64", "bytes", "score"):
        assert forbidden not in injected, f"图片内容进了注入文本：{injected}"
    # 分数只用于阈值比较，不进模板
    assert "score" not in injected


def test_G6_sensitive_writes_are_atomic():
    """清单第 10 条：JSON/断点都走 tmp + os.replace 原子替换。

    凭据本身没有（settings 里存的是界面状态和环境路径，无密钥），
    所以 0600 那条不适用；但原子写这条是实打实做到了。
    """
    store = _source("recog/store.py")
    atomic = store.split("def atomic_json(")[1].split("\n\ndef ")[0]
    assert "temp = path.with_suffix(\".tmp\")" in atomic
    assert "temp.replace(path)" in atomic

    model = _source("recog/model.py")
    save_pause = model.split("def save_pause(")[1].split("\n        # Explicit")[0]
    assert "temporary = checkpoint.with_suffix(\".tmp\")" in save_pause
    assert "temporary.replace(checkpoint)" in save_pause


def test_G7_no_secret_is_written_to_the_database():
    """落地数据里没有凭据，所以「chmod 0600」这条不适用而非缺失。"""
    keys = set()
    for relative in ("plugin.py", "recog/store.py", "recog/service.py", "recog/worker.py"):
        for match in re.finditer(r'set_setting\(\s*f?"([^"]*)"', _source(relative)):
            keys.add(match.group(1))
        for match in re.finditer(r'set_setting\("([^"]*)"', _source(relative)):
            keys.add(match.group(1))
    for key in keys:
        lowered = key.lower()
        for bad in ("token", "secret", "password", "api_key", "credential", "cookie"):
            assert bad not in lowered, f"settings 里存了凭据类键值：{key}"


def test_G8_uv_install_uses_a_pinned_index():
    """uv 安装时 --index-url 指到官方源，且父项目索引不会盖过它。"""
    source = _source("recog/environment.py")
    assert '--index-url", index' in source
    assert '"--no-config"' in source
    assert 'index = "https://download.pytorch.org/whl/" + ("cpu" if kind == "cpu" else "cu128")' in source


def test_G9_overlay_path_is_confined_to_the_plugin_directory():
    """依赖覆盖层只能落在插件自己的 dependencies 下，且拒绝符号链接。"""
    source = _source("recog/environment.py")
    remove_owned = source.split("def remove_owned(")[1].split("\n\ndef ")[0]
    assert "is_relative_to(parent)" in remove_owned
    assert "path.is_symlink()" in remove_owned
    assert 'raise ValueError("依赖目录必须位于插件自己的dependencies内")' in remove_owned
    # 禁止改动底层 Python 环境
    assert 'raise ValueError("禁止修改底层Python环境，请使用插件覆盖层")' in source
