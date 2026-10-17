# 更新日志

## 0.3.3 · 2026-10-17（WebUI 交互打磨）

仅改 `webui/index.html` 共享源（`webui/build.py` 重新生成 5 个页面），训练与推理链路零改动。

- 新增：评测报告支持两任务对比。训练任务页的报告卡片新增「对比」下拉，选中另一份已完成报告后
  并排展示双方指标表格与混淆矩阵，并给出测试集精确率/召回率/F1/准确率的差值行（绿升红降）。
  复用现有 `report` 查询，无后端与 `webui.json` 改动。
- 优化：图片库翻页、切换筛选、切换角色、跳转错判列表时不再整格闪空态 —— 保留当前内容并叠加
  半透明加载遮罩（`pointer-events:none` 防误点），新数据到达后原位替换。
- 优化：图片库分页新增「跳至 N 页」输入框（Enter 或点击跳转，自动夹取到有效范围）。
- 优化：顶部提示条（notice）的成功/普通消息 4 秒后自动消退，错误消息保持常驻直到被替换。
- 优化：上传对话框新增批次总进度条与「已完成 X / 共 Y 张 · 失败 Z 张」文本，百张分批上传时
  不再只有逐文件进度。
- 优化：运行环境页「安装任务与日志」卡片新增「复制日志」按钮（剪贴板 API，失败时回退
  `execCommand`，再失败提示手动复制）。
- 文档：新增 `docs/decisions-webui-polish.md` 决策记录（含对 custom 页面 action 确认弹窗链路
  的源码核实结论：宿主父侧负责弹确认框，45 秒超时自动取消）。

## 0.3.2 · 2026-10-14（插件中心提交前整改）

提交官方插件中心前的整备，全部为文档/测试/健壮性改动，训练与推理链路（`recog/model.py`）零改动。

- 修复：Hook 在 `on_load` 完成前被触发时的 `AttributeError`。宿主先注册组件再调用 `on_load`，
  而 `on_load` 内部会 await 线程化 IO，这段窗口里消息钩子可能被派发 —— 原实现读
  `self.service.inference_enabled` 与 `self._image_cache` 都会抛异常。现在先备好钩子状态、
  并在钩子首行判定 `self.service is None`。
- 修复：`tests/test_cpu_acceptance.py` 仍是单角色时代的用例（`service.submit/activate/rollback/infer`
  都按旧签名调用），导致 `python -m pytest tests` 必红。已按当前签名重写，并让 3 个需要 PyTorch 的
  模块在缺依赖时自动跳过（原本 `test_input_size`/`test_resume` 会在收集阶段报错）。
- 修正：`_manifest.json` 的 `version` 提升到 `0.3.2`，与本次发布对应。
- 文档：README 全面对齐当前实现（插件目录、`characters` 配置、`recognize_character_image` 签名、
  `ctx.paths.data_dir` 数据位置、上传体积上限），新增「依赖模型」「数据流向与网络访问」两节，
  并把原「上游 README 原文」改为「与上游的差异」对照表，避免原文中的旧配置名/旧路径继续误导。
- 文档：`VALIDATION.md` 删除开发者本机临时目录路径，更正「SDK 2.11.0 尚未发布」这一过期结论
  （该版本已于 2026-10-09 发布）。
- 文档：`pyproject.toml` 的 worker 工程名去残留（`mai-recog-self-worker` → `char-recog-worker`），版本对齐 `0.3.2`。
- 文档：`webui/build.py` 与 `tools/migrate_from_maiself.py` 的注释/帮助里残留的旧目录名与旧插件 ID
  （`mai_recog_self`、`org.civetc.cv-char-recog`）改为当前值；`--character` 改为必填，去掉个人角色名默认值。

## 0.3.1 · 2026-10-14（上线前全检整改）

- **消除动态执行面**：worker 探针从 `-c <代码片段>` 改为具名模块 `-m recog.probes <name>`
  （interpreter_info / environment_status），bootstrap 删除 `-c` 分支，运行面不再有 exec。
- 审计清单 12：Hook 对无法解码图片的静默跳过改为 debug 日志留痕（含字节数与原因）。
- 审计其余项结论：网络面（依赖下载）主机硬编码 + SHA256 强校验 + 防穿越，通过；
  资源上限（20MiB/40MP/28MiB base64/512 缓存）齐备；README 引用真实存在；on_unload 清理完整。

## 0.3.0 · 2026-10-13（char-recog 泛化线起点）

基于 SengokuCola/MaiSelfPlugin 0.2.0（GPL-3.0）泛化为「每角色一个二分类模型」的通用角色识别插件（原名 cv-char-recog /「中V角色识别」，0.3.0 起更名泛化；中V名单仅为默认预置）。

- 角色成为一等维度：images/jobs/models 表新增 `character` 列（旧库自动 ALTER 迁移），
  启用/回退/留真值记录按 `active:<角色>` 等 namespaced key 存储。
- 图库、查重、近重复分组、训练快照、评测报告全部按角色隔离；批量标注禁止跨角色。
- `serve` worker 支持多模型常驻：按角色惰性加载、单请求循环推理全部已启用角色，
  返回逐角色得分；单个角色模型加载失败不影响其他角色。
- 自动识别 Hook 对每张图只做一次全角色推理（6 秒预算不变），命中 ≥0.6 逐角色追加提示。
- Tool 更名 `recognize_character_image`，可选 `character` 参数查询单角色判定。
- WebUI 相关 API 增加 `character` 参数（upload/images/training_form/models 等），
  `self_model_activate` 从模型记录反解角色；`self_characters` 返回角色名单与已启用列表。
- config：`character_name`（单角色）→ `characters`（角色名单，默认预置中V主要角色）。
- 训练/审计/指标链路（`model.py`）零改动——"self/other" 语义复用为「是/不是该角色」。
- WebUI 全量适配：头部新增角色选择器，5 页查询/操作声明 `character` 参数与 `self_characters` 查询；
  统计卡、批量按钮、上传弹窗、训练就绪度文案随选中角色动态变化；判断文案去"自己"化（命中/未命中/不确定）。

## 0.2.0 · 2026-10-11（上游 MaiSelfPlugin）

- 训练页可选 224、320、448 输入尺寸，模型保存对应尺寸，识别和回退自动读取。
- 先缩放图片再做颜色增强，减少大图处理耗时。
- 模型保存累计训练来源，增量使用新增或标签已修改的图片，图片库显示训练状态。
- GPU 版 PyTorch 默认选择 CUDA，任务记录显示训练设备和输入尺寸。
- 测试图片支持使用当前模型重新识别，图片库采用紧凑卡片布局。
- 移除每批原图总量 128MiB 的上传限制。
- 下载显示字节进度，插件重载和 MaiBot 重启后恢复任务与进度。
- 自定义 Python 环境由管理员管理，依赖安装仅修改插件覆盖层。
- 停止推理后保持关闭，通过手动开启或重载插件恢复。
- 精简页面和 README 文案，补充插件分类、作者、仓库链接与 SVG 头像。

## 0.1.0 · 2026-10-11

- 提供图片上传、标注、查重、批量管理和训练任务。
- 支持 CPU、CUDA 训练，任务暂停、继续和取消。
- 提供评测报告、错判图片、模型启用与版本回退。
- 自动识别聊天图片和表情，提供 `recognize_self_image` 查询工具。
- 提供识别测试与运行环境管理页面。
