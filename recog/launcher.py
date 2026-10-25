"""按文件路径自举：子进程不依赖 PYTHONPATH、也不依赖 cwd。

真机（OneKey 整合包）的 `.onekey/environments/<uuid>/python.exe` 是 Windows
embeddable 发行版：目录里只要存在 `python*._pth`，Python 就**忽略 PYTHONPATH**，
也不把 cwd 放进 `sys.path`。此时 `python -m recog.environment` 在解释器启动阶段就
报 `No module named 'recog'` —— cwd 设对、PYTHONPATH 设对都没用，日志只留一行，
父进程完全无感（`state` 卡在 running，前端永远「检测中」）。

改成按文件路径启动本模块：`sys.path[0]` 由解释器置为 `recog/` 目录，再把插件根显式
插进去，自举就与任何环境变量配置彻底解耦。历史安全审计禁止 `-c <snippet>` 动态执行，
所以这里用真实文件而不是内联代码。

用法：
    python -X utf8 <插件根>/recog/launcher.py [--overlay <目录>] <模块> [参数...]
"""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = sys.argv[1:]
    overlay = None
    if args[:1] == ["--overlay"]:
        if len(args) < 2:
            raise SystemExit("--overlay 需要一个目录参数")
        overlay = Path(args[1])
        args = args[2:]
    if not args:
        raise SystemExit("用法: launcher.py [--overlay <目录>] <模块> [参数...]")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if overlay is not None:
        from recog.bootstrap import activate

        activate(overlay)
    module, *rest = args
    sys.argv = [module, *rest]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
