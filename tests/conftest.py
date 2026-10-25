"""让 `plugin.py` 在没装宿主 SDK 的机器上也能被真正执行。

## 这个文件解决什么问题

`tests/test_plugin_loads.py`、`tests/test_tool.py`、`tests/test_annotation_policy.py`
三个模块顶部都写着 `pytest.importorskip("maibot_sdk")`。开发机没装宿主 SDK 时，
**这三个模块整模块跳过，pytest 连收集都不做** —— 一共 28 个测试函数对门禁完全
不可见，`plugin.py` 897 行零执行覆盖。

这不是理论问题：0.6.0 把 `Store.ingest` 从 9 个参数砍到 6 个时漏改了 `_capture`
的调用点，整条聊天捕获链路静默失效，而**没有任何一条测试报出来**，正是因为唯一
会执行插件层的三个文件全都是 0 收集。

## 为什么装在 conftest 而不是各测试文件里

`pytest.importorskip` 在**模块导入期**执行，比任何 fixture 都早。stub 必须在
conftest 被导入时就已落在 `sys.modules` 里，所以只能写在模块顶层。

## 为什么这个 stub 是安全的

它只提供 `plugin.py` 实际 import 的那 8 个符号（见 plugin.py:17-18）：
`API` / `Tool` / `HookHandler` / `MaiBotPlugin` / `PluginConfigBase` / `Field` /
`ToolParameterInfo` / `ToolParamType`。`recog/` 全目录零 `maibot_sdk` 引用，
所以桩不会渗进算法层。

每个装饰器都把元数据挂到函数对象上（`api_name` / `tool_name` / `hook_name`），
测试因此可以像真 SDK 一样按名字检索命令与工具。`Field` 必须返回**实例**而不是
描述符 —— 否则 `Config` 类属性拿到的是描述符对象，`create_plugin()` 直接崩。

## 已知边界（不要在这里补真 SDK 语义）

这个桩**不实现**权限模型、不校验调用方身份、不提供 `ctx`。它只让插件能被导入、
能实例化、能跑无宿主交互的逻辑。凡是需要宿主行为的测试，仍应像
`tests/test_release_gate.py` 那样自带 fake ctx，不要把这个文件扩成半个 SDK ——
那会让「测的是插件还是测的是桩」变得说不清。
"""

import sys
import types

_STUB_INSTALLED = "maibot_sdk" in sys.modules


class _FieldInstance:
    """`Field(...)` 的返回值会被直接当作默认值用，必须返回实例而非描述符。"""

    def __init__(self, default=None, **kwargs):
        self.default = default
        self.kwargs = kwargs

    def __repr__(self):
        return f"Field(default={self.default!r})"


class _PluginConfigBase:
    """配置基类：吃 kwargs，类上的 `Field(...)` 声明提供默认值。

    真 SDK 的配置模型是 pydantic 风格；这里只复刻测试用到的两条语义 ——
    ① `Config(characters=[...])` 按名覆盖；② 没给的字段走 Field 的
    `default` / `default_factory`。类型校验、 Literal 约束等一概不实现。
    """

    def __init__(self, **kwargs):
        for klass in reversed(type(self).__mro__):
            for name, value in vars(klass).items():
                if isinstance(value, _FieldInstance) and name not in kwargs:
                    factory = value.kwargs.get("default_factory")
                    setattr(self, name, factory() if factory else value.default)
        for name, value in kwargs.items():
            setattr(self, name, value)


class _MaiBotPlugin:
    """只提供插件自身与测试真正会碰到的成员。

    刻意不实现：权限模型、调用方身份、事件分发、ctx 注入。需要宿主行为的
    测试请自带 fake ctx（见 `tests/test_release_gate.py`）。
    """

    config_model = None

    def __init__(self):
        self._plugin_config_instance = None
        self._ctx = None

    @property
    def ctx(self):
        """宿主上下文。插件只读不写（plugin.py:67/245/805/901），测试统一设 `_ctx`。"""

        return self._ctx

    @property
    def config(self):
        if self._plugin_config_instance is None:
            self._plugin_config_instance = type(self).config_model()
        return self._plugin_config_instance

    async def on_config_update(self, scope: str, config_data, version: str) -> None:
        """宿主默认行为是忽略。插件必须重写它，否则改配置不生效。

        `tests/test_tool.py::test_required_config_update_and_scope` 断言
        `type(plugin).on_config_update is not MaiBotPlugin.on_config_update`，
        所以基类里必须有这个方法可比。
        """

    async def on_load(self) -> None:
        raise NotImplementedError

    async def on_unload(self) -> None:
        raise NotImplementedError

    def get_default_config(self) -> dict:
        model = type(self).config_model
        out = {}
        for name, value in vars(model).items():
            if isinstance(value, _FieldInstance):
                factory = value.kwargs.get("default_factory")
                out[name] = factory() if factory else value.default
        return {"plugin": out}

    def set_plugin_config(self, data) -> None:
        if isinstance(data, dict) and "plugin" in data and isinstance(data["plugin"], dict):
            data = data["plugin"]
        model = type(self).config_model
        known = {
            name
            for klass in reversed(model.__mro__)
            for name, value in vars(klass).items()
            if isinstance(value, _FieldInstance)
        }
        self._plugin_config_instance = model(**{k: v for k, v in (data or {}).items() if k in known})


def _extract_plugin_config_version(config_data) -> str:
    """复刻真实契约：只读 `config_data["plugin"]["config_version"]`，缺节即抛。"""

    if not isinstance(config_data, dict) or "plugin" not in config_data:
        raise ValueError("配置缺少 plugin 节")
    section = config_data.get("plugin")
    if not isinstance(section, dict) or "config_version" not in section:
        raise ValueError("配置缺少 plugin.config_version")
    return section["config_version"]


def _install_sdk_stub() -> None:
    if _STUB_INSTALLED:
        # 真机或别的 conftest 已经装过：绝不覆盖，否则会把自己变成唯一的真相来源。
        return

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
        BOOLEAN = "boolean"
        NUMBER = "number"
        ARRAY = "array"
        OBJECT = "object"

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
    sdk.PluginConfigBase = _PluginConfigBase
    sdk.MaiBotPlugin = _MaiBotPlugin

    config_mod = types.ModuleType("maibot_sdk.config")
    config_mod.extract_plugin_config_version = _extract_plugin_config_version

    types_mod = types.ModuleType("maibot_sdk.types")
    types_mod.ToolParameterInfo = ToolParameterInfo
    types_mod.ToolParamType = ToolParamType

    sys.modules["maibot_sdk"] = sdk
    sys.modules["maibot_sdk.types"] = types_mod
    sys.modules["maibot_sdk.config"] = config_mod


_install_sdk_stub()
