"""FakeHost lifecycle smoke for the devkit gate.

This plugin is a package, so loading `plugin.py` as a bare file cannot resolve
relative imports. Import it with package search locations — the same way the host
does — and drive on_load/on_unload with the real SDK context in a temporary data
directory. No worker is started because there are no active models.
"""

import asyncio
import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        from maibot_sdk.context import PluginContext, PluginPaths
    except Exception as exc:  # pragma: no cover - devkit without SDK
        print(f"SKIP: maibot_sdk 不可用（{exc}）")
        return 0

    spec = importlib.util.spec_from_file_location(
        "char_recog_smoke", ROOT / "plugin.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    async def run() -> None:
        with tempfile.TemporaryDirectory(prefix="char-recog-smoke-") as temporary:
            data = Path(temporary) / "data"
            runtime = Path(temporary) / "runtime"
            data.mkdir()
            runtime.mkdir()
            ctx = PluginContext("org.civetc.char-recog", None, PluginPaths(data, runtime))
            plugin = module.create_plugin()
            plugin._set_context(ctx)
            plugin.set_plugin_config(plugin.get_default_config())
            await plugin.on_load()
            try:
                components = plugin.get_components()
                assert components, "生命周期加载后没有任何组件"
                await plugin.on_unload()
                await plugin.on_unload()
            finally:
                if plugin.service is not None and not plugin.service.closed.is_set():
                    await asyncio.to_thread(plugin.service.close)

    asyncio.run(run())
    print("PASS: 生命周期与组件注册冒烟通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
