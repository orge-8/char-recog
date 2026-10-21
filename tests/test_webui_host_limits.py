"""Host limits for webui.json, straight from the WebUI Pages documentation.

These exist because two release cycles were spent discovering them one error at a
time on a machine we cannot reach from here:

  每个插件最多 20 个页面；组件页每页最多 200 个组件、8 层嵌套、10 个查询、20 个操作
  每个绑定最多 30 个参数
  上传 action 必须接收必填字符串 upload_id，且不能声明 confirmation
  单页 HTML 最大 2 MiB，全部自定义页合计最大 8 MiB；HTML 必须是文件，不能是符号链接

The host validates the whole declaration at once, so a violation on any page makes
the entire plugin fail to load — including pages that were fine before.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
MAX_PAGES = 20
MAX_ACTIONS_PER_PAGE = 20
MAX_QUERIES_PER_PAGE = 10
MAX_PARAMETERS_PER_BINDING = 30
MAX_PAGE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024


def document() -> dict:
    return json.loads((ROOT / "webui.json").read_text(encoding="utf-8"))


def test_page_count_within_limit():
    pages = document()["pages"]
    assert len(pages) <= MAX_PAGES, f"{len(pages)} 个页面，宿主上限 {MAX_PAGES}"


def test_actions_and_queries_within_limits():
    for page in document()["pages"]:
        actions = page.get("actions") or {}
        queries = page.get("queries") or {}
        assert len(actions) <= MAX_ACTIONS_PER_PAGE, (
            f"{page['id']}: {len(actions)} 个 action，宿主上限 {MAX_ACTIONS_PER_PAGE}"
        )
        assert len(queries) <= MAX_QUERIES_PER_PAGE, (
            f"{page['id']}: {len(queries)} 个 query，宿主上限 {MAX_QUERIES_PER_PAGE}"
        )


def test_parameters_per_binding_within_limit():
    for page in document()["pages"]:
        for section in ("actions", "queries"):
            for name, spec in (page.get(section) or {}).items():
                parameters = spec.get("parameters") or {}
                assert len(parameters) <= MAX_PARAMETERS_PER_BINDING, (
                    f"{page['id']}.{section}.{name}: {len(parameters)} 个参数，上限 {MAX_PARAMETERS_PER_BINDING}"
                )


def test_upload_bindings_point_at_declared_actions_with_upload_id():
    for page in document()["pages"]:
        actions = page.get("actions") or {}
        for kind, spec in (page.get("uploads") or {}).items():
            alias = spec.get("action")
            assert alias in actions, f"{page['id']}.uploads.{kind}: 未声明 action {alias}"
            parameters = actions[alias].get("parameters") or {}
            assert parameters.get("upload_id", {}).get("required") is True, (
                f"{page['id']}.uploads.{kind}: {alias} 必须接收必填 upload_id"
            )
            assert "confirmation" not in actions[alias], (
                f"{page['id']}.uploads.{kind}: 上传 action 不能声明 confirmation"
            )


def test_entry_files_are_regular_files_inside_the_plugin():
    for page in document()["pages"]:
        path = (ROOT / page["entry"]).resolve()
        assert path.is_relative_to(ROOT.resolve()), f"{page['id']}: entry 跳出插件目录"
        assert path.is_file(), f"{page['id']}: entry 不存在 {page['entry']}"
        assert not path.is_symlink(), f"{page['id']}: entry 不能是符号链接"


def test_custom_page_size_budget():
    total = 0
    for page in document()["pages"]:
        path = ROOT / page["entry"]
        size = path.stat().st_size
        total += size
        assert size <= MAX_PAGE_BYTES, f"{page['id']}: {size} 字节，单页上限 {MAX_PAGE_BYTES}"
    assert total <= MAX_TOTAL_BYTES, f"自定义页合计 {total} 字节，上限 {MAX_TOTAL_BYTES}"


def test_webui_json_size_within_limit():
    size = (ROOT / "webui.json").stat().st_size
    assert size <= 128 * 1024, f"webui.json 为 {size} 字节，上限 128 KiB"
