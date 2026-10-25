"""子进程必须能在不依赖 cwd 的前提下 import 到 `recog`。

真机实测：点「安装 PyTorch (CPU)」后，安装控制器以
`python.exe: Error while finding module specification for 'recog.environment'
(ModuleNotFoundError: No module named 'recog')` 立刻退出，前端永远停在「检测中…」。

`-m recog.<模块>` 能不能解析，只取决于 `sys.path` 里有没有插件根。原实现把这件事
单独托付给 `Popen(cwd=插件根)` —— 一个隐含前提，任何一处 cwd 偏差都会让子进程在
启动阶段就死，而且死得没有堆栈、父进程也不知道。现在三条启动链都显式带上
`env=PYTHONPATH=插件根`，本文件把这三处钉住。
"""

import ast
import os
from pathlib import Path

ROOT = Path(__file__).parents[1]

# 文件 → 必须带 env= 的函数名（这些函数负责拉起 recog 自己的子进程）。
GUARDED = {
    "recog/environment.py": {"launch", "probe_environment"},
    "recog/service.py": {"_spawn"},
    "recog/dependency_download.py": {"wheel_urls"},
}


def _calls(path: Path, function_names: set[str]):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in function_names:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if isinstance(func, ast.Attribute) and func.attr in {"Popen", "run"}:
                yield node.name, child


def test_every_self_spawn_passes_an_explicit_env():
    missing = []
    for relative, functions in GUARDED.items():
        path = ROOT / relative
        assert path.is_file(), f"{relative} 不存在"
        for function, call in _calls(path, functions):
            keywords = {keyword.arg for keyword in call.keywords}
            if "env" not in keywords:
                missing.append(f"{relative}:{call.lineno}（{function}）")
    assert not missing, "这些子进程启动没有显式 env=，`-m recog.*` 会重新依赖 cwd：" + "、".join(missing)


def test_child_env_puts_the_plugin_root_first():
    from recog import environment

    assert environment.PLUGIN_ROOT == ROOT
    entries = environment.child_env()["PYTHONPATH"].split(os.pathsep)
    assert entries[0] == str(ROOT), "插件根必须是 PYTHONPATH 的第一项"
    inherited = os.environ.get("PYTHONPATH")
    if inherited:
        assert inherited in entries, "原有 PYTHONPATH 必须保留（叠加而不是覆盖）"


def test_log_tail_turns_a_silent_exit_into_a_reason(tmp_path):
    from recog.environment import log_tail

    log = tmp_path / "environment.log"
    assert log_tail(log) == "", "日志还不存在时不应抛异常"
    log.write_text("第一行\n第二行\n最后一行\n", encoding="utf-8")
    assert log_tail(log, lines=2) == "第二行 / 最后一行"


def test_no_child_process_starts_a_recog_module_with_dash_m():
    """`python -m recog.x` 依赖解释器自己把 cwd/PYTHONPATH 放进 sys.path。

    embeddable 发行版带 `python*._pth` 时这两条全被忽略，命令在启动阶段就失败，
    而父进程拿不到任何 traceback —— 真机「永远检测中」就是这么来的。所有子进程
    统一走 `recog/launcher.py`（按文件路径启动 + 显式插入插件根）。
    """
    offenders = []
    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or path.parts[-2] == "tests":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.List, ast.Tuple)):
                continue
            values = [item.value for item in node.elts if isinstance(item, ast.Constant)]
            if "-m" in values and any(isinstance(v, str) and v.startswith("recog.") for v in values):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, "这些命令仍在用 `-m recog.*` 启动子进程：" + "、".join(offenders)


def test_launcher_exists_next_to_the_package():
    launcher = ROOT / "recog" / "launcher.py"
    assert launcher.is_file(), "recog/launcher.py 缺失：所有子进程都会启动失败"
    source = launcher.read_text(encoding="utf-8")
    assert "sys.path.insert(0, str(ROOT))" in source, "launcher 必须显式插入插件根"
