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
    # Pages no longer declare the same set: the host caps actions at 20 per page,
    # so each page carries only what its own section can trigger. The union still
    # has to cover every call site, and the spec used is the declaring page's.
    union = set().union(*(set(actions) for actions in declared.values()))
    for target, names in found:
        for alias in GROUPS.get(target, [target]):
            assert alias in union, f"前端调用了未声明的 action {alias}"
            page = next(name for name, actions in declared.items() if alias in actions)
            parameters = declared[page][alias].get("parameters") or {}
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


def test_pages_bind_upload_actions_they_declare():
    """Host rule: uploads.<kind>.action must exist on that page and require upload_id.

    Missing this is what made 0.5.13 fail on the real host: the pages kept their
    `uploads` blocks while only two of them still declared `upload`/`test_upload`.
    """
    data = json.loads((ROOT / "webui.json").read_text(encoding="utf-8"))
    for page in data["pages"]:
        actions = page["actions"]
        for kind, spec in (page.get("uploads") or {}).items():
            alias = spec["action"]
            assert alias in actions, f"{page['id']}.{kind}: uploads 绑定了未声明的 action {alias}"
            parameters = actions[alias].get("parameters") or {}
            assert parameters.get("upload_id", {}).get("required") is True, (
                f"{page['id']}.{kind}: {alias} 必须接受必填 upload_id"
            )


def test_declared_queries_exist_in_plugin():
    apis = set(re.findall(r'@API\("([^"]+)"', (ROOT / "plugin.py").read_text(encoding="utf-8")))
    data = json.loads((ROOT / "webui.json").read_text(encoding="utf-8"))
    for page in data["pages"]:
        for name, spec in (page.get("queries") or {}).items():
            assert spec["api"] in apis, f"{page['id']}.{name} → {spec['api']} 不存在"


def test_declared_apis_exist_in_plugin():
    apis = set(re.findall(r'@API\("([^"]+)"', (ROOT / "plugin.py").read_text(encoding="utf-8")))
    for page, actions in declarations().items():
        for alias, spec in actions.items():
            assert spec["api"] in apis, f"{page}.{alias} → {spec['api']} 不存在"


def test_pages_stay_within_host_action_limit():
    """The host rejects a page declaring more than 20 actions (WebUIExtension)."""
    for page, actions in declarations().items():
        assert len(actions) <= 20, f"{page}: 声明了 {len(actions)} 个 action，宿主上限 20"


def visible_section(page: str) -> str:
    html = (ROOT / "webui" / f"{page}.html").read_text(encoding="utf-8")
    matches = list(re.finditer(r'<section class="section" id="([^"]+)"([^>]*)>', html))
    for match in matches:
        if "hidden" in match.group(2):
            continue
        # Cut at this section's own closing tag: everything after the last section
        # (dialogs, the shared script) belongs to every page, not to this one.
        end = html.find("</section>", match.start())
        return html[match.start(): end + len("</section>")]
    raise AssertionError(f"{page}: 没有可见区块")


def test_page_declares_the_actions_its_visible_section_triggers():
    """A page must declare every action a control in its own section can fire."""
    for page in PAGES:
        body = visible_section(page)
        # Two batch buttons reuse one alias; the rest map 1:1.
        aliases = {
            "batch_style": "images_batch_style",
            "batch_neg": "images_batch_style",
            "batch_text": "images_batch_style",
        }
        needed = {aliases.get(value, value) for value in re.findall(r'data-batch="([a-z_]+)"', body)}
        if "data-job-action" in body:
            needed |= {"pause", "resume", "cancel"}
        if "data-image-action" in body:
            needed |= {"remove", "restore"}
        if "data-image-label" in body:
            needed.add("label")
        if "data-env-mode" in body:
            needed.add("environment_mode")
        if "data-open-upload" in body:
            needed.add("upload")
        needed |= set(re.findall(r'data-action="([a-z_]+)"', body))
        missing = needed - set(declarations()[page])
        assert not missing, f"{page}: 区块可触发但未声明 {sorted(missing)}"


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
    declared = declarations()
    owner = next(page for page, actions in declared.items() if "apply_threshold" in actions)
    actions = declared[owner]
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


def test_upload_action_declares_the_tags_the_page_sends():
    """startUpload sends upload_style / upload_neg_type through maibot.upload.

    They were added to the page in 0.5.12 but never declared, and because upload
    goes through the upload binding rather than operate(), no existing test looked
    at the argument list. An undeclared parameter is dropped or rejected by the
    host, which would make batch tagging silently do nothing.
    """
    html = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    for name in ("upload_style", "upload_neg_type"):
        assert name in html, f"页面不再发送 {name}，这条测试的前提已变"
    data = json.loads((ROOT / "webui.json").read_text(encoding="utf-8"))
    for page in data["pages"]:
        spec = page["actions"].get("upload")
        if not spec:
            continue
        for name in ("upload_style", "upload_neg_type"):
            assert name in spec["parameters"], f"{page['id']}: upload 未声明 {name}"


def test_retired_negative_bucket_is_gone_everywhere():
    """「表情包」is a medium (style=composite), not an identity — it must not return."""
    html = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    assert 'value="sticker"' not in html
    data = json.loads((ROOT / "webui.json").read_text(encoding="utf-8"))
    for page in data["pages"]:
        for alias, spec in page["actions"].items():
            choices = ((spec.get("parameters") or {}).get("neg_type") or {}).get("choices") or []
            assert "sticker" not in choices, f"{page['id']}.{alias} 仍允许 sticker"


def test_upload_declares_the_overlay_marker():
    """文字叠加 is a marker on its own field, so upload must declare it too."""
    data = json.loads((ROOT / "webui.json").read_text(encoding="utf-8"))
    for page in data["pages"]:
        spec = page["actions"].get("upload")
        if not spec:
            continue
        param = spec["parameters"].get("upload_has_text")
        assert param is not None, f"{page['id']}: upload 未声明 upload_has_text"
        assert set(param["choices"]) == {"yes", "no", ""}


def test_medium_choices_no_longer_mix_an_overlay():
    """A fourth medium「图文/梗图」mixed 画法 with 叠加层; it must stay gone."""
    data = json.loads((ROOT / "webui.json").read_text(encoding="utf-8"))
    for page in data["pages"]:
        for alias, spec in page["actions"].items():
            for name in ("style", "upload_style"):
                declared = ((spec.get("parameters") or {}).get(name) or {}).get("choices")
                if not declared:  # this action does not take a medium at all
                    continue
                assert "composite" not in declared, f"{page['id']}.{alias} 仍允许 composite"
                assert len([c for c in declared if c]) == 3, f"{page['id']}.{alias} 画法档数不是 3"


def test_picture_query_declares_the_tag_filters():
    """Tagging without filtering was the gap: the fields existed, the list ignored them."""
    data = json.loads((ROOT / "webui.json").read_text(encoding="utf-8"))
    html = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    for page in data["pages"]:
        query = (page.get("queries") or {}).get("pictures")
        if not query:
            continue
        params = query["parameters"]
        assert "untyped" in params["view"]["choices"], "../pictures 缺少「未标注类型」视图"
        assert set(params["style"]["choices"]) == {"flat", "paint", "photo"}
        assert set(params["has_text"]["choices"]) == {"yes", "no"}
    assert "style-filter" in html and "text-filter" in html


def test_recognition_results_can_be_promoted_into_the_library():
    data = json.loads((ROOT / "webui.json").read_text(encoding="utf-8"))
    html = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    page = next(item for item in data["pages"] if item["id"] == "test")
    action = page["actions"].get("promote")
    assert action is not None, "测试页没有声明 promote"
    assert {"result_id", "character", "label"} <= set(action["parameters"])
    assert "data-promote" in html

