# 动图识别决策记录（char-recog）

> 2026-10-25 起。每定一条追加「结论 + 理由」；未定问题集中列在文末。

## 已确定

| # | 结论 | 理由 / 依据 |
|---|------|-------------|
| D1 | 拷打强度 = 细问（3），保留本决策记录 | 用户指定（2026-10-25） |
| D2 | 目标范围 = C：在线聊天识别 + 图库/训练两条链路都要；共用抽帧，聚合与去重策略分开 | 用户指定 |
| D3 | 现状：动图在解码层被硬拒（先数帧再判格式）；真实图库 1711 张中 32.3% 因动图被拒，其中 509 张是扩展名伪装成 `.jpg` 的 GIF | `recog/images.py:42`；CHANGELOG 0.5.x；`docs/experiments-sample-size.md:15` |
| D4 | **宿主侧也会拦动图**：MaiBot 1.3.14 的 WebUI 上传 `validate_image()` 明确只收静态图片，动图在到达插件前就被拒 | 读 `MaiBot-1.3.14/src/plugin_runtime/upload_store.py:63-84`；插件 `webui/capture.html` 走 `maibot.upload` 同一通道 |
| D5 | Pillow 12.3 可解 GIF / WebP / APNG 动画；主要成本是顺序 seek：640²×30 帧全扫 135ms，800²×300 帧约 2.0s；640² GIF 单次 seek 到第 29 帧 85ms | 本地实测（2026-10-25）；随机访问后段帧必须顺序解码，不能按帧号 O(1) 取 |
| D6 | 在线路径硬预算 6s；worker 单请求单图、串行处理；冷 worker 直接放弃该图 | `plugin.py:107`、`recog/worker.py:167-196` |
| D7 | **格式边界 = GIF + 动态 WebP + APNG**；MP4 / WebM 短视频不在本期范围 | 用户选择 A（2026-10-25）；Pillow 一套接口全覆盖，视频要另加 ffmpeg/pyav 与另一条消息链路 |
| D8 | 宿主 **1.3.15 仍拒绝动图上传**，与 1.3.14 相同；插件前端 `maibot.upload` 走的就是这条通道，所以“插件能在 WebUI 上传图片”对静态图成立、对动图不成立 | 经 GitHub API 读取 `v1.3.15/src/plugin_runtime/upload_store.py:63-84`：格式白名单后 `seek(1)` 成功即抛“仅支持静态图片” |
| D9 | 插件自建 base64 上传通道不可行：宿主 WebUI 非上传接口的参数上限是 **64 KiB**，装不下 20 MiB 图片 | `MaiBot-1.3.14/src/plugin_runtime/webui_schema.py:19`；上传专属路由上限才是 `MAX_BYTES + MAX_ARGUMENT_BYTES`，而它受 D8 校验 |
| D10 | 本机真实库已定位：`Desktop/Ori` + `Desktop/Thumb` 共 2353 张，**509 张真动图，全部 GIF**（正例 157 / 相似角色负例 352 / 异画风负例 0）；另有 73 张单帧 GIF | Pillow 全量扫描（2026-10-25）；动图帧数中位 20、p90 68、max 398；单帧像素中位 10.6 万、p90 40 万、max 400 万 |
| D11 | 真实动图全扫描成本中位约 53ms/张；抽样 60 张里只有 4 张首/中/尾帧 pHash 都 ≤4 bit，46/54 最大帧间距离 >8；37/60 带透明 | 本地实测（2026-10-25）；结论：首帧不能代表整段动图，但全扫描在真实尺寸下通常负担得起 |
| D12 | 本机有可用模型：`char-exp/runs/goodmodel` 的 `selected.pt`，角色=洛天依，input_size=320，训练 500 张（100 正 / 400 负），验证 F1 0.857 @0.5 | 读 `snapshot.json` / `selection.json`；已用临时 venv 复现验证集指标 |
| D13 | **MaiBot 自身对聊天动图基本只看首帧**：OpenAI 兼容客户端（含 Responses）把 GIF 转成首帧 PNG；Gemini 直接传 `image/gif`，但其声明支持列表不含 GIF；插件 Provider 不收 GIF。唯一多帧路径是表情包描述：`gif_2_static_image` 抽最多 15 个差异帧拼成横图 | `openai_client.py:354-363`、`openai_responses_client.py:154`、`gemini_client.py:1378-1384`、`plugin_client.py:137-143`、`emoji_manager.py:1023`；宿主没有本地角色识别能力 |
| D14 | 现有模型对动图并不“瞎”：509 张实测，首帧 AUC 0.871 / 5 帧中位 AUC 0.872；@0.6 首帧 F1 0.742、5 帧中位 F1 0.744；首帧与 5 帧中位判定只差 11/509（2.2%）；单段动图内分数标准差中位 0.001 | 临时 venv + `selected.pt` 实测（2026-10-25）；解码中位 31ms、p90 118ms，单帧推理中位 17ms、p90 24ms |
| D15 | 由 D14 得：在线识别**默认只取首帧**即可，多帧只用于首帧分数落在灰区时的兜底；训练侧动图的主要价值是补 157 张正例、352 张硬负例，而不是“模型完全不会” | D14 的 AUC/F1 与判定翻转率；多帧“任一帧过线”会把负例误报从 32 抬到 52，精度换召回不划算 |
| D16 | QQ 聊天的动图主要走表情包通道，且插件已经收得到：NapCat 适配器把 `image` 段的 `sub_type=1`（以及非 0/4/9 的子类型）映射为 `emoji`，`image`/`emoji` 都下载原文件后以 `binary_data_base64` 透传，不转码；插件 `plugin.py:119` 同时处理这两类段。因此不需要新增消息段类型，动图解码支持一开，聊天表情包就自动覆盖 | 本机 NapCat 适配器源码 `repo/codecs/inbound/message_codec.py:398-410`；`sub_type=0/4/9` 仍是普通图，普通图同样可能是 GIF/动态 WebP |
| D17 | “QQ 聊天动图基本都是表情包”成立，但表情包域 ≠ D14 的动画插画域：D14 只证明机制可行，不能外推表情包上的精度。表情包含文字、拼贴、真人、多角色、低分辨率，主要风险是误报（precision）而不是漏报；聊天捕获只收 0.4–0.9 灰区且以 `label=NULL` 进审核队列，不会自动进训练，因此可以开捕获积累真实表情包样本，但阈值和训练集要按表情包单独校准 | `scores.py:167` 的 `CAPTURE_BAND`；`store.py:33` 的 `label IS NULL AND source='chat'` 审核条件 |
| D18 | 训练/图库侧 = 方案 A：先做在线 + 聊天捕获；509 张动画只作离线评估集，不做本地批量导入；等审核队列积累到真实表情包样本、且增量训练验证有效后，再决定是否补批量导入工具 | 用户选择（2026-10-25）。理由见 D16/D17：真实表情包才是主分布，D14 的 509 张是插画域；聊天捕获不自动训练、只进审核队列 |
| D19 | 在线动图判定 = 只取首帧；灰区不补帧，不区分 `image`/`emoji`。多帧聚合留到真实表情包样本证明首帧确实漏判后再做 | 用户选择 A（2026-10-25）。D14：5 帧中位相对首帧 F1 0.742→0.744、判定差 2.2%；“任一帧过线”把负例误报 32→52。6s 预算 + 单 worker 串行；宿主普通 `image` 只看首帧 |
| D20 | 图库/训练侧 = 原始动图存一份；首帧负责指纹、缩略图、审核展示和训练读取；审核队列一张动图一行；原始文件保留，未来可扩帧。动图的精确去重不再只用首帧 `decoded` 哈希，避免“首帧相同、后续不同”的变体被误删 | 用户选择 A（2026-10-25）。改动最小，`decode()` 首帧化后 worker / thumbnail / Dataset 全部复用；`CAPTURE_LIMIT=100` 不会被多帧行迅速填满；D14 未证明多帧训练有明确收益。本机 583 个 GIF 实测：`is_animated` 与 `n_frames > 1` 判断完全一致，平均 0.139ms vs 2.144ms，入库标识用它即可，不需要全帧扫描 |
| D21 | 审核页动图预览 = 只显示首帧静态缩略图，不播原动图，也不生成多帧拼图。原始动图仍保留在库中，以后可升级为播放或拼图预览 | 用户选择 A（2026-10-25）。捕获条件是首帧 0.4–0.9 灰区，首帧已有目标角色；现有缩略图预算约 12KB，原动图可达 20MiB，新增预览接口成本和 DoS 面不划算 |
| D22 | **宿主动图输入的真实边界**：主程序在 `chat.receive.before_process` 之前就会把超过 `visual.max_image_size_mb`（默认 16 MB）的 image/emoji 压成首帧 JPEG（透明区域拍白底）；RPC 帧上限 16 MB 且不分片，单图 raw 约 12 MB 是实际送达上限。因此 D20 的“原始动图存一份”只对 ≤ 阈值且能进帧的动图成立；>16 MB 压缩成功时按静态图处理，压缩失败或 `discard` 时插件看不到；12–16 MB 可能被 Hook 跳过。≤12 MB 的在线识别仍按首帧，判定语义不变 | 1.3.14/1.3.15 源码实测：`bot.py:778` 先压缩、`:797` 才调 Hook；`official_configs.py` 默认 16 MB / `compress`；`utils_image.py:154-197` 输出首帧 JPEG；`transport/base.py:18` 的 16 MB 帧上限。外部参考 `fabiocsa/maibot-read-gif` README 也记录了同一限制 |
| D23 | 工具描述去掉「（中V）」，同时补上覆盖范围：`识别消息中的图片包含哪些角色（仅覆盖已启用模型的角色）；可指定单个角色，缺省返回全部已启用角色的得分。` | 用户选择 B（2026-10-27，外部评论提出）。理由：插件只对有启用模型的角色打分，LLM 不知道边界时会把「这个角色没配模型」误报成「不是这个角色」；用词与 `plugin.py` 的 `characters` 配置说明「识别只覆盖已启用模型的角色」保持一致。README 里「默认预置中V主要角色」是对默认名单的**事实陈述**，保留不改 |
| D24 | 本次改动与动图首帧支持合并为一个 0.7.0 发布，不单独发 0.6.5 | 用户选择（2026-10-27）。文案修正属零风险附带改动，合并只需一次真机验证与一次整目录替换；分两次发会让「识别格式 / 审核页」两批文档改动互相牵扯 |
| D25 | 0.7.0 的验收仍按原口径做 509 张离线回放，资源已重新定位：模型在 `char-exp/runs/goodmodel/jobs/02623341426c4da6baa3fad9b884b1d7/selected.pt`（比文档原写的路径多两层 `jobs/<job_id>/`），语料是 `Desktop/Ori` + `Desktop/Thumb` 的 2355 个文件，按字节判定含 583 个 GIF（551 个扩展名伪装） | 用户选择「先找回再回放」（2026-10-27）。实测确认资源完好：`selected.pt` 6.2 MB / 洛天依 / 2026-10-18；语料计数与 D10 的 583 一致。原文档写的 `goodmodel/selected.pt` 是少写了两级目录，不是资源丢失 |

> 外部参考（2026-10-27）：`fabiocsa/maibot-read-gif`（MIT，0.1.2）本机 40 个测试通过。只借鉴其宿主压缩边界、动画测试夹具矩阵、缓存指纹版本化、forward 深度上限；不借鉴帧板 + VLM、全帧扫描、持久化描述缓存。若逐字复用其测试夹具，按 MIT 在 `NOTICE.md` 保留声明。

## 待定

- ~~T1 宿主拦截后，图库/训练链路怎么进动图~~ → D18：聊天捕获自然积累；509 张只作离线评估；批量导入延后，触发条件是审核队列出现足够真实表情包且增量训练验证有效
- ~~T2 格式边界~~ → D7
- ~~T3 在线聚合~~ → D19：只取首帧，不做灰区补帧
- ~~T4 训练样本~~ → D20：原始动图保留、首帧训练、同一条目一个 group；扩帧留到以后
- ~~T5 验证数据~~ → D14：已用真实库 509 张动图 + `goodmodel/selected.pt` 完成离线实测，结论不是机制推演
- ~~T6 成本上限~~ → D19/D21：在线只解首帧、不读 `n_frames`、不扫全帧；沿用 `MAX_BYTES=20MiB` / `MAX_PIXELS=4000万`；入库只做有界动画标识探测；缩略图仍受约 12KB 预算；审核页不播原动图
- ~~T7 在线提示与宿主视觉输入的一致性~~ → D19：提示只基于首帧，与宿主普通 `image` 的首帧输入一致；表情包虽是宿主多帧拼图描述，但首帧提示仍落在同一画面内

## 风险与待核实

- 宿主动图上传拦截已在 1.3.14/1.3.15 核实（D8）；其他 1.3.x 未验
- D22 的宿主压缩/RPC 上限：>16 MB 动图入库时已是首帧 JPEG；12–16 MB 可能整条 Hook 被跳过；这两类都不算回归，但必须写进 README 限制
- ~~QQ / NapCat 是否原样下发 GIF 字节~~ → D16：适配器下载原文件并 base64 透传、不转码；但尚未实采一批真机表情包验证格式分布（GIF / 动态 WebP / APNG 各占多少）
- 表情包误报率未测：D14 的 509 张全是动画插画，不含 QQ 表情包；现有阈值 0.6 在表情包上的 precision 未知
- GIF 透明区域转 RGB 会变黑，与现有静态 PNG 处理一致；宿主压缩路径（>16 MB）拍白底，同一动图可能因体积不同得到不同背景；是否统一是另一个会影响存量模型的决策
- `tests/test_security_audit.py:1153-1165` 与 README「仅支持静态」的断言需同步修改
- 同一动图抽多帧时必须共享 `group_id`，否则训练/验证/测试集隔离会失效
- 多帧推理会拉长 worker 串行占用，可能降低同一条消息里多图/多角色的标注覆盖率
- forward 嵌套目前无插件侧深度上限；计划加 `MAX_FORWARD_DEPTH=3`。宿主序列化/反序列化也会递归，插件侧只能防自身栈，不能保证宿主侧不先失败

## 实施计划（0.7.0）· 已执行，2026-10-27

> 状态：**已完成**。全量门禁 443 passed / 3 skipped（含 torch），口径内 266 collected；
> 509 张离线回放已跑（见 `VALIDATION.md`「2026-10-27 · 0.7.0」）。唯一与计划不同的地方：
> 回放语料因目录整理比建档时多一张负例（510 张），逐张剔除验证后基线可复现。
> 工具描述改动按 D23 执行；版本合并按 D24；资源定位按 D25。

### 1. 代码改动

| 文件 | 改动 | 备注 |
|------|------|------|
| `recog/images.py` | 格式白名单加入 GIF（APNG 继续走 PNG、动态 WebP 走 WEBP）；删除 `n_frames != 1` 硬拒；`seek(0)` 后取首帧；错误文案改为 JPEG/PNG/WebP/GIF；新增 `DECODE_VERSION = 1` | `decode()` 保持在线语义：不读 `n_frames`、不扫后续帧。新增 `decode_with_animation()` 返回 `(image, animated)`，只给入库用，内部用 `is_animated`；本机 583 个 GIF 实测与 `n_frames > 1` 完全一致，平均 0.139ms。`DECODE_VERSION` 是图像语义版本，未来从首帧扩到多帧时递增，用于让缓存失效 |
| `recog/store.py` | `ingest()` 改用 `decode_with_animation()`；动图写入 `decoded = "animated:<原始sha>"`；原始文件原样保存 | 精确去重只认原始字节，避免“首帧相同、后续不同”的变体被误删；schema、pHash、分组、solid、缩略图逻辑不变 |
| `recog/worker.py` | 不改逻辑 | 推理仍调用 `decode(raw, MAX_EDGE)`，自动变成首帧语义；无多帧协议 |
| `recog/model.py` | 不改逻辑 | 训练 Dataset 仍调用 `decode()`，首帧读取；原始动图保留，未来可扩帧 |
| `plugin.py` | 两处小改：`_image_cache` 的 key 并入 `DECODE_VERSION`；`visit()` 增加 `MAX_FORWARD_DEPTH=3` 深度上限 | `image`/`emoji` 共用 `visit()`；聊天捕获继续走 `store.ingest()`，动图自动进审核队列；缓存键变更让未来改判定语义时旧缓存自动失效 |
| `webui/*.html` | 不改功能 | 上传仍只收 JPEG/PNG/静态 WebP；审核页只显示首帧静态缩略图，不播动图（D21） |

### 2. 测试改动

- `tests/test_catalog.py`
  - 静态格式参数化加入 `GIF`。
  - 删除“GIF 必须报错”的断言；新增动图首帧解码断言。
  - `test_decode_names_the_detected_format` 保留 BMP 拒收，改为验证 GIF 不再被拒。
- `tests/test_security_audit.py`
  - F2 改为“格式白名单 + 首帧语义”：断言允许集合含 GIF、源码不再出现“动图拒收”、`seek(0)` 在 `load()` 前。
  - F3 改为“有效 GIF 可解，BMP/SVG/损坏文件仍拒绝”。
  - A9 的 `decode(raw)` 断言跟随改名，改为 `decode_with_animation(raw)`。
- 新增 `tests/test_animated_images.py`
  - 夹具参考 read-gif 的 `tests/image_fixtures.py`：64×48、5–8 帧、80–120ms 的 GIF / WebP / APNG 生成器；char-recog 不嗅探容器，不需要它的 `acTL` 跳过逻辑。
  - GIF / APNG / 动态 WebP 均取第一帧（首帧颜色、尺寸断言）。
  - `.jpg` 扩展名伪装 GIF 仍按内容识别。
  - 单帧 GIF 可解，且 `animated=False`、不加 `animated:` 前缀（与 read-gif 的“忽略单帧 GIF”相反）。
  - 超过 `MAX_PIXELS` 的动图在 `load()` 前被拒。
  - 入库保留原始字节，宽高取首帧，缩略图为静态 JPEG。
  - 同首帧、不同后续帧的两张动图都保留；同字节重复仍判重。
  - `source="chat"` 的动图以 `label=NULL` 进审核队列。
  - 损坏/截断动图抛出既有异常，不进入库。
- `tests/test_annotation_policy.py`
  - 增加 `emoji` 段携带 GIF 字节的用例：仍产生提示，并触发捕获。
  - 缓存键：同一张图第二次调用应命中缓存；把 `plugin.DECODE_VERSION` 改成其他值后应重新 `infer`，证明该常量确实在 key 里。
  - forward 深度：构造 4 层嵌套，最内层图片不再送 worker，且不抛 `RecursionError`。
- `tests/test_chat_capture.py` / `tests/test_capture_review.py`
  - 按需补动图捕获行与 `capture_pending` 断言，避免只测静态 PNG。

### 3. 文档与版本

- `README.md`：识别格式改为 JPEG/PNG/WebP/GIF；说明动图统一取首帧；WebUI 上传仍只收静态图的宿主限制；补 D22 的宿主压缩/RPC 上限（在线实际上限约 12 MB；>16 MB 压缩成功时按静态首帧处理；12–16 MB 可能无标注），并说明“原始动图保留”只对 ≤ 阈值成立。
- `VALIDATION.md`：补本次门禁结果与 509 张回放数据。
- `CHANGELOG.md`：新增 0.7.0 条目（首帧支持 GIF/APNG/动态 WebP；图库保留原图；测试与安全边界）。
- `_manifest.json` / `pyproject.toml`：版本同步 0.7.0；`test_release_gate.py` 会校验两者与 CHANGELOG 顶部一致。
- `NOTICE.md`：若测试夹具逐字复用 read-gif（MIT），保留其版权与许可声明；只借鉴结构或自写夹具则不需要。

### 4. 验收口径

- 开发环境：`python -m pytest tests -q` 全量通过；无 torch 时按既有规则跳过 3 个训练/推理模块。
- 带 CPU PyTorch 的隔离环境再跑一次全量，覆盖 `test_cpu_acceptance.py`、`test_input_size.py`、`test_resume.py`。
- 离线回放：用 `goodmodel/selected.pt` 对 509 张真实动图跑首帧，AUC 0.871、F1@0.6 0.742 与 D14 基线一致，判定差异应为 0。
- 安全边界：>20 MiB、>4000 万像素、BMP/SVG/损坏文件仍被拒；`DecompressionBombWarning` 仍升级为错误。>20 MiB 是 decode 层闸门，在线动图实际可见上限见 D22（约 12 MB）。
- 大图边界（D22，人工/文档项，不纳入自动门禁）：≤12 MB 动图按动图识别；12–16 MB 可能无标注；>16 MB 压缩成功时按静态首帧入库，压缩失败时跳过；以上都不得导致消息处理失败。
- 行为边界：聊天动图能产生提示并进审核队列；审核行 `label=NULL`；缩略图为静态首帧；同首帧不同后续帧不被误删；forward 4 层嵌套不抛 `RecursionError` 且超过 3 层不再识别；同图重复调用命中缓存、改 `DECODE_VERSION` 后重新 `infer`。
- 真机冒烟（人工，不计入自动门禁）：在宿主 1.3.15 发一张 ≤12 MB 的动图表情包，确认提示出现；若方便再发一张 >16 MB 的 GIF，确认按静态图处理且不崩；WebUI 上传 GIF 仍被宿主拒绝（预期）。

### 5. 明确不做

- WebUI 上传动图、审核页播动图/多帧拼图。
- 多帧在线聚合、多帧训练抽样、MP4/WebM。
- 509 张批量导入工具（D18 延后，触发条件见 T1）。
- 透明帧改灰底、`mface` 专项适配。
- read-gif 的帧板 + VLM 描述、全帧扫描、持久化描述缓存、魔数嗅探闸门。
- 在线展示帧数/时长/fps（需要全帧扫描；`is_animated` 只回答“是否动图”）。
- 对 >16 MB 动图做原始动画恢复（宿主已压成 JPEG，插件拿不到原件，见 D22）。

### 6. 执行顺序

1. `recog/images.py`（含 `DECODE_VERSION`）+ `recog/store.py` + 新增/更新核心测试；先跑定向测试。
2. `plugin.py`（缓存键 + forward 深度）/ `worker.py` / `model.py` 回归，加 `emoji` GIF、缓存版本、4 层 forward 用例；跑应用层测试。
3. README / VALIDATION / CHANGELOG / 版本号，补 D22 的宿主边界与 NOTICE 归属；跑 `test_release_gate.py`。
4. 全量 pytest（开发环境 + 带 torch 环境）与 509 张离线回放；记录结果并更新 VALIDATION。
