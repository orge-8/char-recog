# 工程验收记录

2026-10-09，本地 Windows、Python 3.13，CPU 限制为 4 线程。测试使用临时合成图片，不使用个人图库或既有角色模型。

## 完整 CPU 流程

真实执行 ImageNet 初始化、15轮从头训练、保存重载、启用、增加6张标注后的10轮增量训练、启用新版本及回退。首次40个独立图组，训练28/验证6/测试6；增量训练34/验证6/测试6。任务中的身份真值和划分冻结，新增数据保持历史留出。

| 模式 | 训练含轮间验证 | 训练引擎总时间 | 任务含进程准备 | 进程峰值内存 |
|---|---:|---:|---:|---:|
| 从头15轮 | 6.62秒 | 6.85秒 | 11.73秒 | 441MiB |
| 增量10轮 | 5.80秒 | 6.10秒 | 10.91秒 | 437MiB |

这些是小型合成素材的流程测量，运行时有其他测试并行，不代表大图库训练耗时或真实角色识别准确率。实际时间取决于图片解码、数据量和CPU。

## 校验范围

- 素材格式/尺寸、纯色、完全重复、近重复、标签冲突、跨划分变体冲突、原始图片哈希。
- 历史划分保持、训练快照、增量回放、验证选择记录先于测试、指标重算、有限性与权重实际变化、保存后CPU重载。
- 单训练任务、幂等提交、立即取消与启动之间的竞争、卸载、重启不自动重训、损坏模型拒绝、版本回退及常驻进程缓存更新。
- 工具聊天流检查、多图序号、失效素材、无模型、三态判断。插件向LLM只声明message_id和image_index，会话参数由既有宿主上下文注入。
- 登录保护的实际multipart路由、SDK上传凭证领取、人工标注、CPU训练、报告、启用、工具调用、增量训练、回退通过集成测试。未启动真实Bot或调用外部LLM服务。
- 宿主/SDK/工具与集成测试共104项通过。插件数据/CPU/生命周期15项通过，包含修复后的立即取消竞争。前端相关19项通过。

## CUDA 补充

RTX 5060 Ti 16GB 使用独立既有CUDA环境完成1轮训练、验证、测试与保存后CPU重载，全部模型审计通过。引擎总时间1.89秒，CUDA已分配内存峰值约150MiB。此轮仅验证CUDA流程与分类头预热，不代表15轮完整视觉微调性能；默认工作环境仍为CPU。

## 部署限制

SDK 2.11.0 已于 2026-10-09 发布（当前最新 2.13.0），宿主最新为 1.3.7；本插件声明 `host_application ≥1.3.6`、`sdk ≥2.11.0`，用已发布版本即可，无需本地 SDK 路径。插件已按宿主 loader 语义在 SDK 2.11.0 与 2.13.0 下验证导入与 `create_plugin()`。安装及启动方式见 README。合成测试结果不能作为产品识别效果的承诺；管理员应提供真实正负例并检查独立新图片。

## 2026-10-10 交互与暂停更新核验

本轮宿主/插件轻量测试80项、真实CPU流程5项、前端交互14项通过。CPU流程包含15轮从头训练（先暂停并恢复）、10轮增量、启用与回退、取消和卸载。额外在批次及验证边界分别暂停，恢复后的所选权重与不中断训练逐项完全一致，验证历史也一致。

40张合成素材的从头引擎6.69秒（含验证6.40秒），任务总16.88秒（含两次进程准备及暂停操作）；增量引擎6.03秒（含验证5.74秒），任务总10.59秒。峰值RSS分别468MiB、465MiB。此为流程测试，不能外推真实图库识别效果。

自动接收图片提示已验证正负例、多图、表情、转发、缓存复用和模型切换；依赖管理验证共享环境只读、安装时拒绝新工作进程与环境选择持久化。没有实际卸载/重装用户当前PyTorch环境，也没有为本轮改UI再启动真实用户数据训练。本轮未重复CUDA训练，沿用上述既有CUDA流程验证。前端生产构建完成；运行中的Bot需要重启/重新加载插件和刷新新版前端后使用。

## 上传和页面状态整理

2026-10-10：上传弹窗改为 button.detail + dialog，身份选择改为 select.presentation=buttons。报告、错判筛选使用当前页面URL参数，不再写共享设置。补测取消与失败重试、冻结批参数、后台轮询时操作、同页导航参数和禁用原因；本轮80个Python测试、18个前端测试通过。前端生产构建及中英文文档站构建通过。此次未重新训练模型，也未重启正在运行的Bot。

## 依赖覆盖层

2026-10-10：默认继承MaiBot解释器，重依赖仅安装到插件数据目录 dependencies/packages，子进程在 .pth 处理前优先加载覆盖层。安装先暂存并核验真实导入路径再切换，失败保留旧层；卸载只删除覆盖层，底层包不卸载。

在临时数据目录实装 torch 2.11.0+cpu / torchvision 0.26.0+cpu，使用MaiBot主环境Python完成15轮合成样本CPU训练、权重/隔离/重载核验、启用与推理，再卸载覆盖层。安装和卸载前后主环境全部发行包的 METADATA/RECORD/INSTALLER SHA清单一致。核验记录保存在该轮的临时数据目录（`.../mai-self-overlay-*/verification.json`，已随临时目录清理，不再记录本机绝对路径）。训练含验证 7.18秒，总训练流程 7.48秒，峰值RSS 459.8MiB。此轮未修改用户图库或启用用户模型，未重复GPU流程。

## 2026-10-14 提交插件中心前复核

- 静态：`check_submission.py` 逐条复核关卡 1；`_manifest.json` 通过宿主 `PluginManifest` 校验；`webui.json` 通过宿主（1.3.7）`load_webui_extension` 校验；`plugin.py` 在 SDK 2.11.0 与 2.13.0 下按宿主 loader 语义导入成功，`capabilities` 与代码实际调用逐项一致。
- 测试：`python -m pytest tests` → 52 passed / 4 skipped（本机无 PyTorch，3 个训练相关模块自动跳过，1 项需 MaiBot `src.plugin_runtime`）。本轮**未**执行需要 PyTorch 的 CPU/CUDA 端到端流程，上述历史数据仍以原环境为准。

  该节的「4 skipped」计数方式是正确的（`importorskip` 在模块级调用时，整个模块只记 1 项跳过），但当时的口径与 0.3.4 的用例总数不同，见下一节。

- 修复：Hook 在 `on_load` 完成前被派发时的 `AttributeError`（先备状态 + 钩子首行判空）；`tests/test_cpu_acceptance.py` 按当前 `submit/activate/rollback/infer` 签名重写。

## 2026-10-17 · 0.3.4 止血复核

环境：Windows 10、Python 3.12.10、Pillow 12.3.0、numpy 2.5.4、pytest 9.1.1，隔离 venv（`--system-site-packages`）**未安装 PyTorch**。

```text
python -m pytest tests -q
68 passed, 4 skipped in 306.86s
```

四个跳过项：`test_cpu_acceptance.py`（1，模块级）、`test_input_size.py`（1）、`test_resume.py`（1）均因缺少 PyTorch；`test_tool.py::test_default_config_has_required_version`（1）需要 MaiBot 的 `src.plugin_runtime`。

本轮覆盖的回归点：

| 修复 | 对应用例 |
|---|---|
| 批量参数契约（CSV ≡ JSON 数组） | `test_catalog::test_batch_accepts_csv_and_json_array`、`test_webui_contract::test_operate_arguments_match_declared_parameters` |
| `remove_all` 仅当前角色 | `test_catalog::test_remove_all_is_scoped_to_character`、`test_tool::test_remove_all_forwards_character` |
| 超时开熔断而非永久关停 | `test_inference_transport::test_timeout_opens_breaker_instead_of_stopping`、`::test_worker_exit_reports_current_error_and_reaps_process` |
| 冷启动不占聊天预算 | `::test_hook_path_never_blocks_on_cold_start` |
| `busy` 不触发熔断 | `::test_busy_does_not_trip_breaker` |
| 手动停止抗自动恢复 | `::test_manual_stop_is_sticky_against_tick_and_time` |
| 预热失败退避 | `::test_warmup_failure_backs_off` |
| 训练期线程数协商 | `::test_threads_control_frame_follows_training` |
| 监控线程抗异常 | `test_service_monitor::test_monitor_survives_a_failing_tick` |
| 报告对比数值 | `test_tool::test_report_metrics_expose_numeric_values`、`test_webui_contract::test_frontend_sends_csv_and_clears_character_state` |
| 生成产物同步 | `test_webui_contract::test_generated_pages_are_in_sync` |

本轮**未**执行需要 PyTorch 的 CPU/CUDA 端到端训练流程，因此以下行为仍未在真实模型上验证，部署后需真机确认：

1. 冷启动预热的真实耗时（`import torch` + 逐角色载入权重）是否落在 45 秒预算内；
2. 训练期间推理降到 1 线程后，实际识别延迟与 6 秒预算的关系；
3. 「每请求只解码一次」在 8 个启用角色、4000 万像素图下的实测收益；
4. 熔断 30 秒是否适合真机的推理耗时分布。

另需更正上文两处不可复现的描述：`check_submission.py` 与 `audit_webui_json.py` 并不在本仓库中（`tools/` 只有 `tools/migrate_from_maiself.py`），当时引用的「0 error / 0 warning」无法用仓库内容重跑；「前端 19/14/18 项测试」同样没有对应的测试代码留在仓库里。这些结论如需保留，应把脚本一并纳入版本控制。

## 2026-10-18 · 0.4.0 阈值与性能

环境：Windows 10、Python 3.12.10、Pillow 12.3.0、numpy 2.5.4、pytest 9.1.1，隔离 venv（`--system-site-packages`）**未安装 PyTorch**。

```text
python -m pytest tests -q
133 passed, 4 skipped in 159.93s
```

跳过项与 0.3.4 相同：`test_cpu_acceptance.py`（1，模块级）、`test_input_size.py`（1）、`test_resume.py`（1）因缺少 PyTorch；`test_tool.py::test_default_config_has_required_version`（1）需要 `src.plugin_runtime`。

同一台机器上 0.3.4 为 `68 passed, 4 skipped in 306.86s`。用例增至 133 项而总耗时降到 160 秒，主要来自下面两项改造（既有用例本身就在反复调用 `jobs()` 与 `freeze()`）。

### 性能实测（同机、同口径）

| 项 | 改前 | 改后 | 方法 |
|---|---:|---:|---|
| `jobs()` 50 个任务 × 1000 图 | **907 ms/次**，读 **25.1 MiB** | **0.04 ms/次**，读 0 字节 | 造 50 份快照后对比「逐个 `json.loads` 取 device/size」与「一条 SELECT」 |
| `training_readiness()` N=2000 | **167.2 ms/次** | 建索引 0.9 ms + 查询 1.2 ms | 与旧 `source_seen` 在同一 `cands/sources` 上对照，结果逐条相等 |
| `training_readiness()` N=1000 / 500 | 42.3 / 10.3 ms | 0.8 / 1.1 ms | 同上 |

训练页每 5 秒轮询一次，因此第一项的收益是**每分钟约 11 秒的 CPU 与 300 MiB 磁盘读**降到接近零。

### 新增覆盖

| 领域 | 用例 |
|---|---|
| 阈值数学 | `metrics` 在 0.5 下与旧实现逐字段相等；换阈值改变混淆矩阵；`sweep` 19 档且 0.5 档等于 `metrics(...,0.5)`；空集/单类不除零 |
| 建议阈值 | 取「满足 90% 精确率的最低阈值」、不低于 0.50、小样本 `reliable=False` 且不伪造达标、目标不可达时 `target_met=False`、Wilson 下限已知值 |
| 三态判定 | 参数化 7 组边界，含 `0.6-0.2` 的二进制浮点边界（0.4 必须落在「未命中」） |
| 角色级阈值 | 默认 0.60、写入读回、非法/损坏值回退、批量映射跳过坏行、命名空间不污染推理目标、恢复默认是删键 |
| 聚合策略 | 只提示最高分、边际 0.05 抑制双方、平分抑制、单角色命中/未命中、门槛不可达不遮蔽其他角色、**改阈值后缓存失效并重新推理** |
| 报告重算 | 当前阈值 0.60 下召回变为 50.0%、行内标注阈值、`selection` 仍为 0.50、无验证集预测时不显示建议值 |
| `jobs` 迁移 | 0.3.4 schema 迁移后新列存在且行不丢；首次回填后 `_snapshot_meta` 抛错仍正常（证明热路径不读快照）；删掉快照后列值仍在；`submit` 写入两列 |
| 索引等价性 | 500 组随机 `sources/rows`（键池仅 4 个值以制造碰撞）与旧实现完全一致；三个反例钉住「最后一次匹配说了算」 |

### 仍未验证

1. 需要 PyTorch 的 4 项仍跳过，**未跑真实模型训练与推理**；
2. 阈值建议值在真实图库上的合理性（合成素材的分数分布不代表动漫角色）；
3. 跨角色 raw score 不可比，0.08 边际是启发式，需真机日志调参；
4. `jobs` 迁移只在合成的 0.3.4 库上验证，未在有真实历史任务的实例上升级过。

## 2026-10-18 · 0.4.1 训练解码缓存评估与 torch 全量复核

环境：Windows 10，Python 3.12.10，**torch 2.14.1+cpu / torchvision 0.29.1+cpu**（隔离 venv，4 线程），Pillow 12.3.0，numpy 2.5.4。素材为真实角色图库 `Ori`（1711 张 jpg/png/webp/gif，826 MB，按角色分目录）。

### 安装 PyTorch 后暴露的 8 个失败（已修）

0.3.0 起 `test_cpu_acceptance` / `test_input_size` / `test_resume` 三个模块一直因缺少 PyTorch 整体跳过，因此**从未真正运行过**。装上 torch 后 8 项失败：

| 数量 | 性质 | 原因 |
|---:|---|---|
| 7 | 测试自身的 bug | `store.freeze(db, job, mode, device)` 少传 `character`；`test_input_size` 把 `input_size` 当成 `character` 传；`test_legacy_weights_use_320` 用了未导入的 `torch`。参数化错误自 0.3.0 引入 `character` 维度时就存在 |
| 1 | **0.4.0 的真实缺陷** | `activate()` 只等 8 秒冷启动。`import torch` + 逐角色载入在此机器上可超过 8 秒（尤其刚跑完训练），预热仍在继续但代码先报错。改为等 25 秒，超时且仍在启动时给「正在启动，请稍后重试」而不是把锅甩给运行环境 |

修复后（含新增的格式文案用例）：

```text
python -m pytest tests -q
143 passed, 1 skipped in 714.82s
```

（中途那次 142 passed / 472s 是加入新用例之前的数字；714s 这一轮与本文其它基准测量并行运行，机器更忙。）

唯一跳过项仍是需要 `src.plugin_runtime` 的默认配置契约用例。

### 真实素材的尺寸分布（决定优化是否值得）

采样 400 张：

| 分位 | P10 | P25 | 中位 | P75 | P90 | 最大 |
|---|---:|---:|---:|---:|---:|---:|
| 像素 | 0.04MP | 0.09MP | 0.28MP | 1.05MP | 1.91MP | 16.2MP |

| 区间 | <0.5MP | 0.5–1MP | 1–2MP | 2–5MP | 5–10MP | >10MP |
|---|---:|---:|---:|---:|---:|---:|
| 占比 | 60.0% | 11.5% | 19.5% | 7.8% | 0.5% | 0.8% |

**≥2MP 只占 9.0%**，这与「训练解码是瓶颈」的假设相反。

### 三条路径的实测对比

采样 60 张真实图的真实 `Dataset`（`augment=True`，`input_size=320`）：

| 路径 | ms/张 | 2000 张 × 15 轮 | 相对现状 |
|---|---:|---:|---:|
| 现状（原图直解） | 48.6 | 24.3 分钟 | 1.00x |
| 磁盘缓存（320 q92） | 22.9 | 11.4 分钟 | 2.12x |
| `draft()` 解码 | 20.8 | 10.4 分钟 | **2.33x** |

12MP 合成素材（最坏情况，10 张）：现状 184.5 ms/张（92 分钟）→ 缓存 22.5（11 分钟）→ draft 46.7（23 分钟）。

保真度（20 张真实插画，ImageNet 底座 576 维特征）：

| 路径 | 余弦最小 | 余弦中位 | top-1 一致 |
|---|---:|---:|---:|
| 磁盘缓存 320 q92 | 0.994634 | 0.998361 | 19/20 |
| `draft()` | 0.999155 | **1.000000** | **20/20** |

分数层面的影响（用真实角色图训练二分类器：洛天依 53 正 / 其他角色 84 负，训练 104 张、评估 33 张、4 轮仅训练分类头）：

| 路径 | \|Δscore\| 均值 | P95 | 最大 | 命中/不确定/未命中三态翻转 |
|---|---:|---:|---:|---:|
| 磁盘缓存 320 q92 | 0.0213 | 0.0603 | 0.1327 | 1/33 |
| `draft()` | **0.0030** | **0.0177** | **0.0253** | 1/33 |

补充：PNG 缓存到精确 `input_size` 可做到**逐位相等**（余弦 1.000000、张量 MAE 0），但体积 81–166 KB/张（JPEG q92 只有 11–27 KB）。

### 结论：不采纳训练解码缓存

1. 真实库 91% 的图 <2MP，而缓存的固定开销（读文件 + 解码小图 + letterbox）约 9–13 ms，对 0.4MP 以下的图是**净损失**（1.4x 变慢）。
2. 比 `draft()` 更慢（2.12x vs 2.33x）且保真更差：余弦 0.9946 vs 0.9992，Δscore 均值 0.021 vs 0.003，最大 0.133 vs 0.025。
3. 磁盘成本 11–27 KB/张（要逐位相等则需 81–166 KB），还要引入 sha 键、清理策略与「小图不缓存」的分支。
4. **最要命的是训练/推理输入不一致**：训练读缓存（JPEG 往返），线上推理读原图。0.4.0 刚把阈值与判定对齐，再引入一个系统性输入偏差不值当。

建议的替代方向（**本次未实现**，需单独决策）：`decode(raw, draft_edge=...)` 走 JPEG 的 `draft()` 缩尺解码 —— 无磁盘、无失效、训练与推理走同一条路径，实测 2.33x 且中位逐位相等。

### 顺带发现：真实图库 32.3% 无法入库

1711 张里 553 张被 `decode()` 拒绝，而旧错误信息只说「仅支持JPEG/PNG/静态WebP」，不告诉管理员实际是什么：

| 数量 | 真实格式 | 说明 |
|---:|---|---|
| 509 | 动图 GIF | 扩展名多为 `.jpg`，实为表情/动画多帧 |
| 44 | 静态 GIF（单帧） | 技术上可以直接支持 |

0.4.1 已改为报出探测到的实际格式与帧数。**是否允许单帧 GIF**、或取动图首帧，属于产品决策，本次未改。

### 未验证

1. 上述分数漂移来自 33 张评估集与 4 轮仅分类头的训练，不是生产模型；真实模型的绝对漂移可能不同，但两条路径的相对关系应稳定。
2. 未在真机上对比「训练期间」的端到端耗时（含 DataLoader 与验证），只测了单张预处理。
3. 真实素材没有图库外的负例，二分类的判别力不代表线上效果。

### 0.4.2 · draft 实装后的复测

按上面的结论只实装了 `draft()`，没有做磁盘缓存。实装后用同一批真实素材复测生产代码路径：

| 项 | 改前 | 改后 | 说明 |
|---|---:|---:|---|
| `Dataset` 每张（augment=True，320） | 128.6 ms | 80.8 ms | **1.59x**，每张省 47.8 ms → 2000 张 × 15 轮约省 24 分钟 |
| 12MP 单图推理预处理 | 202 ms | 100 ms | 同一会话对比 |

同一份脚本在另一次会话中得到 48.6 → 20.8 ms（2.33x，每张省 27.8 ms）。**绝对耗时跨会话不可比**（机器负载不同，同会话内对比才有效）；比值落在 1.6–2.3 之间，取决于「解码 + 缩放」在单张成本里的占比，而该占比随素材尺寸分布变化。

分数漂移（按生产路径名取值，33 张真实角色图，同一二分类模型）：

| 路径 | \|Δscore\| 均值 | P95 | 最大 | 命中/不确定/未命中翻转 |
|---|---:|---:|---:|---:|
| 训练路径 `decode(320)` | 0.0048 | 0.0217 | 0.0347 | **0/33** |
| 推理路径 `decode(448)` | 0.0007 | 0.0048 | 0.0104 | **0/33** |

推理路径几乎无漂移：它按 448 解、再缩到 320，而 448 是 libjpeg 缩尺后的实际边长，与全尺寸解码再缩放只差一点点重采样。

安全性复验：`draft()` 会**立即改写 `source.size`**，因此 4000 万像素护栏已被移到 draft 之前（`tests/test_catalog.py::test_pixel_guard_runs_before_draft` 钉住）；入库指纹仍按原始尺寸计算，`test_ingest_keeps_the_full_resolution_for_fingerprints` 钉住；旧的暂停任务（`resize_first=False`）不解 draft，`test_resize_precedes_augmentation` 的 legacy 分支继续通过。

```text
python -m pytest tests -q
146 passed, 1 skipped in 381.80s
```

仍未验证：

1. 未在真机上跑完整训练对比（本机是 CPU + 合成快照，只有单张预处理与 `Dataset` 迭代层面）。
2. 已有模型的分数会因输入分布变化轻微漂移（上表），**升级后建议重训一次**。
3. draft 对 PNG/WebP 无效（Pillow 忽略），这部分图没有提速。

## 2026-10-18 · 0.4.3 兼容范围整改

`_manifest.json` 的 `sdk.max_version` 由 `2.99.99` 改为 `2.13.99`，纯元数据改动，代码与运行时不涉及。

### 声明范围的依据

| 字段 | 声明值 | 实际验证过的 | 依据 |
|---|---|---|---|
| `host_application` | `1.3.6 – 1.3.99` | 1.3.6、1.3.7 | 宿主 loader 语义导入 + `load_webui_extension` 校验（见 2026-10-14 小节） |
| `sdk` | `2.11.0 – 2.13.99` | 2.11.0、2.13.0 | 同小节：按宿主 loader 语义在 SDK 2.11.0 与 2.13.0 下验证导入与 `create_plugin()` |

`1.3.99` / `2.13.99` 表示「该次版本线的补丁级向后兼容」，不是「永远兼容」。被插件中心拦下的是 `X.99.99` / `99.99.99` 这类末两段连续 99 的写法（2026-10 起的 AI 审核硬阻断项）。

### 未验证的区间（如实标注）

- SDK **2.12.x** 未单独验证。它落在声明区间内，按语义化版本视为兼容，但没有实机记录。
- `2.14.0` 及以上、宿主 `1.4.0` 及以上**不在**声明范围内，宿主会拒绝加载。若在那些版本上跑通，需要再发一版放宽 `max_version`。

### 自查脚本的两处需人工判读（外部脚本，本仓库不含）

> 下面的 `scripts/check_submission.py` 是插件中心的外部自查脚本，不在本仓库，无法直接复跑；这段保留的是当时的输出与判断依据。

```text
python scripts/check_submission.py . --plugin-id org.civetc.char-recog
合计: PASS 21 / FAIL 2 / WARN 4
```

- `[FAIL] eval/exec 用户输入 —— 命中: recog/model.py` 是**误报**。三处命中全是 `model.eval()`（`recog/model.py:62/93/198`），即 PyTorch 切换推理模式；脚本用的 `\b(eval|exec)\s*\(` 无法区分。无动态执行、无 `eval()` 调用，不需要改动。
- `[WARN] 依赖声明完整性` 指出 `torch` / `torchvision` / `psutil` 未列入 `dependencies`：这是刻意的，插件不随包安装 PyTorch，而是由环境页在管理员确认后装入独立覆盖层（见「依赖覆盖层」小节）。把这几个写进 `dependencies` 会让宿主在安装插件时强制拉取数 GB 的 torch。

## 2026-10-18 · 0.5.0 样本量受控实验与默认增广

完整方法、全部表格与失效边界见 [docs/experiments-sample-size.md](docs/experiments-sample-size.md)。这一节只记录结论与可复核的入口。

### 实验规模

真实素材（`Ori` 洛天依 264 张去重后正例、相似角色 866、异画风 525、非角色 24），固定 128 张测试集（40 正 / 94 负）在全部 20 个配置中逐张相同，配对比较。**20 次训练**，累计约 100 分钟 CPU，每次走插件真实 `Store.freeze()` + `model.train()`，超参不改。

### 结论摘要

| 维度 | 结论 |
|---|---|
| 数量 | 每类 20 组正例：精确率 0.52；50 组 0.64；100 组 0.66；200 组 0.84 |
| 正负比 | 50 正 + 200 负（1:4）→ 0.799，超过 100 正 + 100 负（0.661）；召回不降 |
| 负例来源 | 只用相似角色 vs 混入异画风：AUC 0.901 vs 0.906，无可分辨差异 |
| 增广 | 裁剪+翻转：F1 0.659→0.723（N=50）、0.716→0.794（N=100） |
| 硬负例 | 训练负例最高分仅 0.016，无素材可挖；F1 +0.027 与「多 5 张重复样本」不可区分 |
| 验证集 | 3 张/类时自动建议阈值从 0.54 漂到 0.40，校准后 F1 基本不变 |

### 实装与复核

- `snapshot["augmentation"] = "strong"` 由 `Store.freeze()` 写入，`model.train()` 透传给 `Dataset`；缺少该字段的旧快照保持只有颜色抖动。
- 生产路径重测 N=100 增广档：AUC 0.934 / 精确率 0.685 / 召回 0.925 / F1 0.787，与实验夹具有 0.02 以内差异。
- 实装时发现并修掉一个真实缺陷：torchvision 的 `RandomResizedCrop` 对 PIL 输入返回 PIL，直接送 `Normalize` 会 `TypeError`。`tests/test_input_size.py::test_strong_augmentation_keeps_shape_and_trains` 同时覆盖该回归与整轮训练审计。

### 仍未验证

1. 只有一个正类角色，外观高度相近的角色（举例：同为蓝发双马尾的另一角色）难度会更高，绝对数字不可直接搬运。
2. 测试集 40 张正例，精确率 95% 区间约 ±0.15；差值小于 0.10 的条目不作结论。
3. 非角色切片仅 14 张测试图，只能当探针。
4. 全部在 CPU 上跑，未验证 GPU 与增量训练是否继承相同结论。

5. 异画风负例池（`Thumb`）经模型复检含 22 张目标角色图（≥0.60），因此上表精确率被低估约 0.02–0.04；详见 `docs/experiments-sample-size.md` 的「已知缺陷」。

## 2026-10-20 复测（0.5.14）

同一开发环境，完整依赖（含 CPU 版 PyTorch）：`pytest tests` → **179 passed / 1 skipped（588s）**。

本轮新增的覆盖点：

- 内容类型（`style` / `neg_type`）迁移与校验，老库迁移后划分逐行不变（`tests/test_eval_set.py`）。
- 固定评估集：批次整批归边、稀有类型优先、每类预留 20 组训练、重跑成员不变。
- 分档指标与建议文案：`n < 5` 标记样本不足，不输出百分比（`tests/test_style_metrics.py`）。
- 宿主 WebUI 限额：页面数、每页 action/query 上限、参数数、`uploads` 绑定、页面体积（`tests/test_webui_host_limits.py`）。
