"""WebUI 前端门禁：脚本必须能被解析，派生页面必须与 index.html 同步。

0.6.1 把 `renderTraining` 里的就绪度模板改坏（删除「内容类型」提示时多留了
`</p>` 和 `` ` : ""} `` 两处残片），六个自定义页面的 `<script type="module">`
因此整块解析失败 —— 浏览器一行 JS 都不执行，页面停在静态空壳上：下拉框空白、
徽章停在「加载中」、图库一直转圈，而服务端日志里没有任何记录。既有测试全部只看
Python 侧，没有任何门禁读过 HTML 里的脚本。

这两条门禁针对的正是这个盲区：
1. 语法：每个页面的 module 脚本都要过 `node --check`（宿主只做字符串透传，
   插件交付的必须是自洽的 JS）。
2. 同步：webui/*.html 必须等于用当前 index.html 重新跑 build.py 的结果，
   否则「改了 index 忘了 build」会以同样静默的方式上线。
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
WEBUI = ROOT / "webui"
PAGES = ["images", "capture", "training", "models", "test", "environment"]
DOCUMENTS = ["index", *PAGES]

SCRIPT_PATTERN_OPEN = '<script type="module">'
SCRIPT_PATTERN_CLOSE = "</script>"


def find_node() -> str | None:
    """定位 node 解释器；找不到就明确报告，不静默降级。"""
    for name in ("node", "node.exe"):
        found = shutil.which(name)
        if found:
            return found
    # Windows 便携环境不保证 node 在 PATH 上，按托管路径兜底。
    for candidate in sorted(Path.home().glob(".workbuddy/binaries/node/versions/*/node.exe")):
        return str(candidate)
    return None


def extract_script(html: str) -> str:
    start = html.find(SCRIPT_PATTERN_OPEN)
    assert start != -1, "页面里没有 module 脚本"
    start += len(SCRIPT_PATTERN_OPEN)
    end = html.find(SCRIPT_PATTERN_CLOSE, start)
    assert end != -1, "module 脚本没有闭合"
    return html[start:end]


@pytest.mark.parametrize("name", DOCUMENTS)
def test_webui_script_parses(tmp_path, name):
    node = find_node()
    if node is None:
        pytest.skip("未找到 node 解释器：JS 语法门禁未执行（安装 node 后本用例才会真正校验）")
    document = (WEBUI / f"{name}.html").read_text(encoding="utf-8")
    script = tmp_path / f"{name}.mjs"
    script.write_text(extract_script(document), encoding="utf-8")
    result = subprocess.run([node, "--check", str(script)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0, f"webui/{name}.html 的脚本无法解析：\n{result.stderr.strip()[:2000]}"


def test_generated_pages_match_build_output(tmp_path):
    """真跑一遍 build.py，避免测试里复刻规则而与其漂移。"""
    work = tmp_path / "webui"
    work.mkdir()
    shutil.copy2(WEBUI / "index.html", work / "index.html")
    shutil.copy2(WEBUI / "build.py", work / "build.py")
    rebuilt = subprocess.run(
        [sys.executable, str(work / "build.py")], cwd=str(tmp_path), capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    assert rebuilt.returncode == 0, f"webui/build.py 执行失败：\n{rebuilt.stderr.strip()[:2000]}"
    drifted = [
        name
        for name in DOCUMENTS
        if (work / f"{name}.html").read_text(encoding="utf-8") != (WEBUI / f"{name}.html").read_text(encoding="utf-8")
    ]
    assert not drifted, f"这些页面与 index.html 不同步（先跑 webui/build.py）：{', '.join(drifted)}"
