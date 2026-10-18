"""Static contract checks between webui.json, index.html and plugin.py.

These exist because the batch actions silently broke for three releases: the page
sent a JSON array string while the API split on commas, and nothing verified the
two sides against each other.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
PAGES = ["images", "training", "models", "test", "environment"]

# `operate(node.dataset.<group>, ...)` dispatches to a group of aliases.
GROUPS = {
    "node.dataset.batch": ["batch_positive", "batch_negative", "batch_remove"],
    "node.dataset.jobAction": ["pause", "resume", "cancel"],
    "node.dataset.imageAction": ["remove", "restore"],
    "node.dataset.action": ["stop", "start"],
}

QUOTES = "'\"`"


def declarations():
    data = json.loads((ROOT / "webui.json").read_text(encoding="utf-8"))
    return {page["id"]: page["actions"] for page in data["pages"]}


def read_source():
    return (ROOT / "webui" / "index.html").read_text(encoding="utf-8")


def slice_call(html: str, start: int) -> str:
    """Return the argument text of the call starting right after `operate(`."""
    depth, index = 1, start
    while index < len(html) and depth:
        char = html[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char in QUOTES:
            quote = char
            index += 1
            while index < len(html) and html[index] != quote:
                index += 2 if html[index] == "\\" else 1
        index += 1
    return html[start : index - 1]


def object_keys(args: str):
    """Names of the properties of the first top-level object literal, if any."""
    depth, index = 0, 0
    while index < len(args):
        char = args[index]
        if char in QUOTES:
            quote = char
            index += 1
            while index < len(args) and args[index] != quote:
                index += 2 if args[index] == "\\" else 1
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "{":
            if depth:
                index += 1
                continue
            body, level = [], 1
            index += 1
            while index < len(args) and level:
                current = args[index]
                if current in QUOTES:
                    quote = current
                    body.append(current)
                    index += 1
                    while index < len(args) and args[index] != quote:
                        body.append(args[index])
                        index += 1
                elif current == "{":
                    level += 1
                elif current == "}":
                    level -= 1
                    if not level:
                        break
                body.append(current)
                index += 1
            return parse_keys("".join(body))
        index += 1
    return None


def parse_keys(body: str):
    parts, depth, current, index = [], 0, [], 0
    while index < len(body):
        char = body[index]
        if char in QUOTES:
            quote = char
            current.append(char)
            index += 1
            while index < len(body) and body[index] != quote:
                current.append(body[index])
                index += 1
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and not depth:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    parts.append("".join(current))
    keys = set()
    for part in parts:
        part = part.strip()
        if not part:
            continue
        name = part.split(":", 1)[0].strip() if ":" in part else part
        if re.fullmatch(r"[A-Za-z_$][\w$]*", name):
            keys.add(name)
    return keys


def calls(html: str):
    """[(alias or dataset group, {argument names} or None)] for every operate()."""
    found = []
    for match in re.finditer(r"operate\(", html):
        args = slice_call(html, match.end()).strip()
        alias = re.match(r"'([a-z_]+)'", args)
        group = re.match(r"(node\.dataset\.\w+)", args)
        if alias:
            target = alias.group(1)
        elif group:
            target = group.group(1)
        else:
            continue  # ternary/computed alias; covered by its own schema test
        found.append((target, object_keys(args)))
    return found


def test_operate_arguments_match_declared_parameters():
    html = read_source()
    declared = declarations()
    found = calls(html)
    assert len(found) >= 10, "解析器失效：operate 调用点数量异常"
    for page in PAGES:
        actions = declared[page]
        for target, names in found:
            for alias in GROUPS.get(target, [target]):
                assert alias in actions, f"{page}: 前端调用了未声明的 action {alias}"
                parameters = actions[alias].get("parameters") or {}
                if names is None:
                    assert all(not spec.get("required") for spec in parameters.values()), (
                        f"{page}.{alias}: 无参调用但存在必填参数"
                    )
                else:
                    # Optional parameters may be omitted by a given call site,
                    # but everything required must be sent and nothing unknown.
                    assert names <= set(parameters), f"{page}.{alias}: 前端多传了 {names - set(parameters)}"
                    required = {name for name, spec in parameters.items() if spec.get("required")}
                    assert required <= names, f"{page}.{alias}: 缺少必填参数 {required - names}"


def test_declared_apis_exist_in_plugin():
    apis = set(re.findall(r'@API\("([^"]+)"', (ROOT / "plugin.py").read_text(encoding="utf-8")))
    for page, actions in declarations().items():
        for alias, spec in actions.items():
            assert spec["api"] in apis, f"{page}.{alias} → {spec['api']} 不存在"


def test_pages_declare_identical_actions():
    declared = declarations()
    reference = declared[PAGES[0]]
    for page in PAGES[1:]:
        assert set(declared[page]) == set(reference), f"{page} 的 action 集合与 {PAGES[0]} 不一致"
        for alias in reference:
            assert declared[page][alias].get("parameters") == reference[alias].get("parameters"), (
                f"{page}.{alias} 的参数声明与 {PAGES[0]} 不一致"
            )


def test_batch_uses_csv_and_remove_all_is_scoped():
    actions = declarations()[PAGES[0]]
    for alias in ("batch_positive", "batch_negative", "batch_remove"):
        spec = actions[alias]["parameters"]["image_ids"]
        assert spec["type"] == "string", "image_ids 必须是逗号分隔的字符串"
        assert spec.get("max_length", 0) >= 33_000, "1000 张 ID 需要足够的长度上限"
    character = actions["remove_all"]["parameters"]["character"]
    assert character["required"] is True
    assert character["max_length"] == 32
    assert "仅当前角色" in actions["remove_all"]["confirmation"]


def test_frontend_sends_csv_and_clears_character_state():
    html = read_source()
    assert "[...selected].join(',')" in html, "批量参数必须与后端 CSV 契约一致"
    assert "JSON.stringify([...selected])" not in html
    assert "operate('remove_all', { character })" in html
    assert "selected.clear();" in html
    assert "20 * 1024 * 1024" in html and "64 * 1024 * 1024" not in html
    assert "${key}_value" in html, "报告对比必须消费后端数值字段"
    assert "parseFloat(a[key])" not in html
    assert "个百分点" in html


def test_threshold_controls_are_wired():
    html = read_source()
    actions = declarations()[PAGES[0]]
    for alias in ("apply_threshold", "reset_threshold"):
        assert alias in actions, f"webui.json 缺少 {alias}"
    assert actions["apply_threshold"]["parameters"]["value"]["type"] == "string"
    assert actions["reset_threshold"]["parameters"]["character"]["required"] is True
    assert "data-threshold-apply" in html
    assert "data-threshold-reset" in html
    assert "operate('apply_threshold'" in html
    assert "operate('reset_threshold'" in html
    assert "function thresholdPanel(" in html


def test_generated_pages_are_in_sync():
    source = read_source()
    markers = ("[...selected].join(',')", "20 * 1024 * 1024", "${key}_value", "selected.clear();")
    for page in PAGES:
        html = (ROOT / "webui" / f"{page}.html").read_text(encoding="utf-8")
        for marker in markers:
            assert marker in html, f"{page}.html 缺少 {marker}，请重新运行 webui/build.py"
        assert len(html) >= len(source) - 200, f"{page}.html 与共享源明显不同步"
