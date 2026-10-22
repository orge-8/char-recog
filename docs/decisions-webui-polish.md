# WebUI 优化决策记录（char-recog）

## 已确定

| # | 结论 | 理由 |
|---|------|------|
| D1 | 工作基线 = 0.3.2（`e8e1743`），工作目录为本地插件源码目录 | 0.3.2 已推送 GitHub（插件中心提交前整改），在旧基线上改会产生合并成本 |
| D2 | 优化目标 = A 视觉/交互打磨 + B 功能补全 | 用户指定；C（性能/体积）与 D（宿主适配故障）本次不作为目标 |
| D3 | 拷打强度 = 常规，留决策记录（本文件） | 泛化改造后首次 WebUI 打磨，方向需对齐但不必挑战前提 |
| D4 | webui.json 结构审计现状 0 error / 0 warning | `audit_webui_json.py` 实测（2026-10-17） |
| D5 | 训练曲线（per-epoch loss/acc）属于高成本项：`worker.py` 只落盘最终 `selected_epoch` 与 `total_seconds`，无逐轮指标 | 读 `recog/worker.py` 确认；要做需改 worker 落盘 + service 透传 + 前端图表 |
| D6 | 模型版本删除无后端 API（`service.py` 无 delete 路径） | 读 `recog/service.py` 确认；要做 = 新 @API + webui.json 声明 + 确认交互 |
| D7 | 范围允许动后端；B 类本期只做 B3（报告对比），B1/B2/B4/B5 排后续版本 | 用户指定；B3 复用现有 `report` 查询零后端改动 |
| D8 | ~~「删除全部零确认」疑点~~ → 经 dashboard v1.3.8 源码核实**不成立**：custom 页面 action 由父侧 `plugin-custom-page.tsx` 拦截，binding 有 `confirmation` 时弹宿主确认框（45s 超时自动取消），接受后以 `confirmed=true` 调后端（`webui_extensions.py:222` 的 409 只是二次防线） | 链路：iframe `maibot.action` → 父侧确认对话框 → `invokePluginWebUI(confirmed=true)` → 后端放行；无修复必要 |

## 校订（2026-10-17，0.3.4 复核）

- **D5 的前提是错的**：`recog/model.py:306` 已逐轮 append `{epoch, loss, validation}`，`recog/worker.py:50` 已把含 `history` 的完整 `report.json` 落盘。缺的只是 `plugin.py` 的 `self_training_report` 把 `history` 从 summary 里剔除且未返回给前端。也就是说训练曲线**不是**高成本项，透传 + 一个内联 SVG 折线即可。0.3.4 未做此项（不在止血范围内），但成本判断应更正。
- **D4 不可复现**：`audit_webui_json.py` 不在本仓库（`tools/` 只有 `tools/migrate_from_maiself.py`），「0 error / 0 warning」无留在版本控制里的复现方式。
- **D8 结论仍成立但只覆盖次要风险**：确认框确实由宿主父侧提供，但它拦不住语义错误——0.3.4 修复的正是「删除全部」跨角色清库，属于 binding 语义问题，与确认框是否存在无关。另外 D8 依据的 dashboard 1.3.8 与 `_manifest.json` 声明的 `host_application ≥1.3.6` 存在版本差，1.3.6/1.3.7 上的确认行为未核实。
- B1/B2/B4/B5 在本文件里**从未定义**，后续无法追踪；0.3.4 也未补定义。

## 待定

- ~~T1 范围~~ → D7 允许动后端；B 类本期只做 B3（报告对比），B1/B2/B4/B5 排后续版本
- ~~T2 B 类优先级~~ → D7
- ~~T3 风险项~~ → D8 先核实 dashboard 源码中 custom 页面 action 的 confirmation 行为，再决定是否前端兜底
- ~~T4 版本策略~~ → 0.3.2 已发布，bump **0.3.3**（已执行：`_manifest.json` / `pyproject.toml` / CHANGELOG）
