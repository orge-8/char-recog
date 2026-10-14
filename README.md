# 角色识别（char-recog）

管理员上传并标注角色图片，为**每个角色训练一个二分类小模型**（"是/不是该角色"）。插件自动识别收到的图片和表情，命中时在对应图片后添加上下文提示；同时提供只读工具 `recognize_character_image`。角色名单在插件配置 `characters` 中维护（默认预置中V主要角色，可任意增删）。

本插件是 [SengokuCola/MaiSelfPlugin](https://github.com/SengokuCola/MaiSelfPlugin)（GPL-3.0）的泛化改造版：把"是否包含 Bot 自己"的单角色二分类，泛化为**每角色一个二分类模型**的通用角色识别。训练、版本管理、依赖覆盖层等工程链路与上游一致；与上游的具体差异见文末「与上游的差异」。

- 插件 ID：`org.civetc.char-recog`
- 数据目录：由宿主 SDK 授予（`ctx.paths.data_dir`，默认 `data/plugins/org.civetc.char-recog/`）
- 插件目录名可以自取（宿主按 `_manifest.json` 识别），本文档示例用 `plugins/char-recog`

---

## 安装

需要 MaiBot **1.3.6+**、SDK **2.11.0+**，以及支持 `custom_pages`、`image_upload` 的 WebUI。

1. 把插件目录放入 `plugins/`（例如 `plugins/char-recog`）。
2. 完整重启 MaiBot（热重载对含 manifest 的插件不生效）。
3. 在 WebUI 插件管理页启用，出现「**角色识别**」工作区。
4. 首次加载由宿主生成 `config.toml`；角色名单通过 `characters` 配置（默认预置中V主要角色）。
5. 插件附带通用预训练底座，角色模型需要管理员自行训练（见「训练与模型」）。

用本地 SDK 开发源码调试主程序时：

```powershell
$env:MAIBOT_PLUGIN_SDK_PATH = (Resolve-Path ../maibot-plugin-sdk).Path
$env:MAIBOT_WEBUI_USE_LOCAL_DASHBOARD = '1'
uv run bot.py
```

（使用本地 WebUI 前先在 `dashboard` 目录执行 `npm run build`。已发布版本无需这两步。）

## 运行环境

默认使用 MaiBot 的 Python，插件依赖安装在自己的覆盖层 `dependencies/packages`（位于插件数据目录内）。训练和推理优先加载覆盖层，其余包继承主环境。安装和卸载仅修改覆盖层，**不改动 MaiBot 主环境**。

自定义环境使用已有的 Python 解释器或环境目录，依赖由管理员管理。页面选择会持久保存，并优先于配置中的 `worker_python`；该配置留空时使用 MaiBot Python。更换底层 Python 大版本后需重新安装覆盖层。**被指定的解释器会以该路径执行插件的训练与推理代码**，请只指向可信环境。

GPU 版 PyTorch 默认选择 CUDA，也可选择 CPU；CPU 版使用 CPU；识别推理固定用 CPU。训练或推理运行期间锁定依赖操作，安装期间锁定工作进程启动。

安装先在临时目录完成，再核验导入路径、版本和 CUDA 运算，成功后切换到覆盖层，失败时保留原层。GPU 主包约 2.7GB，torchvision、配套依赖及解压文件需要额外空间。

下载进度、安装阶段和日志保存在插件数据目录。重载插件或重启 MaiBot 后，继续读取存活安装进程的进度，或恢复中断任务。下载支持断点续传；服务器拒绝续传或文件校验失败时重新下载，残留安装目录清理后重新安装。失败任务可手动重试。

停止推理后，通过手动开启或重新加载插件恢复。重载页面仅更新页面文件。

## 依赖模型

这台机器上依赖分三层，出问题时先确认自己在哪一层：

| 层 | 内容 | 谁负责 | 位置 |
|---|---|---|---|
| 宿主环境 | `Pillow`、`numpy` | MaiBot 自身依赖（`_manifest.json` 的 `dependencies` 会校验版本） | MaiBot 的 Python |
| 插件覆盖层 | `torch`、`torchvision`、`Pillow`、`numpy`、`psutil` | 在 WebUI「运行环境」页点安装，插件自己下载并装入 | `<数据目录>/dependencies/packages/` |
| 自定义环境 | 同上，全部 | 管理员自行准备 | 管理员指定的解释器/环境 |

- **宿主环境**只装插件加载链路真正需要的东西：`recog/images.py` 顶层 `from PIL import Image`，`numpy` 在指纹计算时懒加载。这两个包在 `_manifest.json` 的 `dependencies` 里声明为 `Pillow >=11,<13`、`numpy >=2,<3`；宿主版本不满足时会拒绝加载。
- **插件覆盖层**里的 `torch`/`torchvision`/`psutil` 不写进 `dependencies`：体积太大（GPU 版约 2.7GB），不该让宿主预装。`psutil` 只在训练 worker 里用于采样峰值内存。
- **自定义环境**模式下插件不会安装任何东西，`torch`、`torchvision`、`psutil`、`Pillow`、`numpy` 必须已经存在；缺包时页面会提示「未安装 xxx」。
- 安装依赖需要 `uv` 在 PATH 中，并且能访问 `download.pytorch.org`（见「数据流向与网络访问」）。没有 `uv` 会明确报错，不会静默降级。

## 图片与标注

上传时为本批图片选择"是/不是该角色"。图片库支持修改标签、跨页批量操作、删除和恢复；删除全部覆盖整个图片库。历史训练任务引用的原图会保留。

支持 JPEG、PNG 和静态 WebP（GIF/动图和其它格式会被拒绝）。宿主按页面声明的 `image_max_edge: 4000` 把上传副本等比缩放到 4000 像素以内；插件侧再校验单张 **20 MiB / 4000 万像素**上限。

- 每批最多选择 100 个文件。
- 每页最多 24 张，单次批量操作最多 1000 张。

相同文件或像素完全一致的图片复用已有记录和标签。近重复图片按变体分组，每组保留一个训练代表。查重在上传后进行，管理员可核对排除原因和标签冲突。

## 训练与模型

每类至少需要 **20 个独立图组**。图片自动分为训练、验证和测试集。后续图片保持已有划分；同图变体跟随原组，连接不同划分的图片暂时排除。

默认增量训练：加载当前启用模型，仅训练新增或标签已修改的图片。模型保存累计训练来源，图片库显示当前模型是否训练过该图片。首次训练和从头训练使用通用预训练底座（ImageNet MobileNetV3-Small，SHA256 见 `recog/model.py` 的 `BASE_SHA`）。

每个实例同时运行一个训练任务。任务开始时冻结数据和标签，训练期间的标注修改用于下一次任务。暂停保存训练状态，继续使用同一快照；最终核验阶段可取消。中断任务由管理员手动重试。

报告包含精确率及其分子分母、召回率、F1、准确率、混淆矩阵、错判图片和耗时。混淆矩阵行表示真值、列表示预测，顺序为"不是该角色／是该角色"。重复使用的测试集标为历史留出评测；留出标签或成员变化后，报告间的分数需要分别评估。

模型通过核验后由管理员启用，可回退上一启用版本。训练期间当前模型继续服务，版本切换前先完成加载和校验。

训练页可选择 224、320 或 448；插件配置 `input_size` 设置默认尺寸，默认为 320。设置用于新训练任务；识别和版本回退使用各模型保存的尺寸。

## 自动识别与测试

收到图片、表情和转发中的图片时，模型得分达到 0.6 会在对应图片后添加角色提示。单条消息处理预算为 6 秒，超时保留原消息（fail-open）。`auto_recognize=false` 可关闭自动识别。

识别测试保留最近 24 张图片及结果，可使用当前模型重新识别，显示判断、得分、版本和耗时。聊天图片与测试图片均独立于训练库。

工具接口：

```text
recognize_character_image(message_id, image_index=0, character="")
```

工具读取当前聊天流中指定消息的图片或表情，`image_index` 为零基序号，`character` 留空则返回全部已启用角色的得分。未启用模型、消息或图片失效、索引错误及加载失败时返回具体状态。

判断门槛：得分 ≥0.6 为"命中"，≤0.4 为"未命中"，其余为"不确定"。得分表示模型输出，身份仍需结合上下文核对。

## 数据存储

持久数据全部写在宿主 SDK 授予的插件数据目录（默认 `data/plugins/org.civetc.char-recog/`），插件源码目录只放只读素材：

- `catalog.sqlite3`：图片、标签、任务和版本记录。
- `uploads/`：宿主上传凭证落点（领取后即入库，不长期保留）。
- `images/`、`thumbnails/`：原图与缩略图缓存。
- `test-images/`：识别测试用图（只保留最近 24 张，超期即删）。
- `jobs/<任务>/`：训练快照、断点、进度与报告。
- `models/<版本>/weights.pt`：角色模型。
- `dependencies/packages/`：依赖覆盖层。
- `dependencies/downloads/`：下载缓存与续传文件。
- `environment-status.json`、`environment.log`、`inference.log`：安装与推理状态、日志。

发布插件时包含源码、页面、声明文件和通用预训练底座。运行数据、环境、配置和日志由各实例保管。

`tools/migrate_from_maiself.py` 可以把上游 `maibot.mai-recog-self` 数据目录整体迁移过来（整库归属到一个角色、保留划分与模型血统，幂等，支持 `--dry-run`）。运行它需要宿主 Python 或插件覆盖层里已有 Pillow + numpy。

## 数据流向与网络访问

- **不外发任何用户数据**：不使用 `httpx`/`requests`/`aiohttp`，不向第三方上传聊天记录、用户 ID、群号、图片、配置或 token；没有用户可控 URL，因此也没有 SSRF 面。
- **唯一的出站访问**是安装 PyTorch 依赖时的 `download.pytorch.org`（官方索引与官方 CDN），域名在代码里硬编码，不跟随索引返回值取主机。
- 每个 wheel 都用索引提供的 **SHA256 强校验**，校验失败即删除重下；安装包缓存拒绝符号链接、拒绝越出插件目录的路径。
- 安装动作由管理员在 WebUI「运行环境」页显式点击触发，**不会在插件加载时自动联网**。
- 训练与推理的模型权重只读本地文件：ImageNet 底座与已训练模型都做 SHA256 校验后才加载。
- 训练断点（`resume.pt`）用 `torch.load` 恢复，仅由训练 worker 写入，并与 `resume-audit.json` 里的快照摘要 + 文件 SHA256 双向核对；用户上传的图片永远不会成为反序列化输入。

## 验证

测试命令：

```shell
python -m pytest tests
```

缺少 PyTorch 时，3 个需要真实训练/推理的测试模块会自动跳过（`test_cpu_acceptance.py`、`test_input_size.py`、`test_resume.py`），其余用例仍会执行。工具测试需要本地 SDK 的 `src.plugin_runtime` 可导入（在 MaiBot 主程序目录下运行时满足）。

实测结果见 [VALIDATION.md](VALIDATION.md)，合成素材测试用于验证工程流程，实际识别效果由训练数据决定。

## 页面开发

共享源文件为 `webui/index.html`。修改后在插件目录执行：

```shell
uv run --no-sync python webui/build.py
```

生成 `webui/{images,training,models,test,environment}.html` 后点击"重载页面"。修改 `webui.json` 则需要重新加载插件。

## 与上游的差异

相对 SengokuCola/MaiSelfPlugin，本插件做了这些改动（上游原文不再保留，避免与实现不一致）：

| 项 | 上游 | 本插件 |
|---|---|---|
| 识别目标 | 单角色（"是不是 Bot 自己"） | 每角色一个二分类模型，`characters` 可增删 |
| 配置键 | `character_name`（单角色） | `characters`（角色名单）+ `input_size` + `worker_python` |
| 工具名 | `recognize_self_image(message_id, image_index=0)` | `recognize_character_image(message_id, image_index=0, character="")` |
| 判定文案 | "是自己／不是自己" | "命中／未命中／不确定" |
| 数据库 | 无角色维度 | `images`/`jobs`/`models` 增加 `character` 列，旧库自动 ALTER 迁移 |
| WebUI | 单角色页面 | 头部角色选择器，5 页按角色隔离查询与操作 |
| 依赖管理 | 插件目录内 `.venv` | 共享覆盖层（数据目录内）+ 可选自定义环境 |
| 数据目录 | `data/plugins/maibot.mai-recog-self/` | 宿主授予的 `ctx.paths.data_dir` |

未复制上游的任何个人图片、标签库、实验结果或已训练的身份模型；`assets/mobilenet_v3_small-047dcff4.pth` 是 PyTorch 官方 ImageNet 权重。

## 许可与来源

- 本仓库：GPL-3.0-or-later（见 [LICENSE](LICENSE)、[NOTICE.md](NOTICE.md)）。
- 改编自 [SengokuCola/MaiSelfPlugin](https://github.com/SengokuCola/MaiSelfPlugin)（GPL-3.0）与独立的 `Mai_recog_self` 工程；整图补边、MobileNetV3-Small 二分类及训练选轮逻辑来自后者。
- MobileNetV3 实现由 torchvision 提供。
