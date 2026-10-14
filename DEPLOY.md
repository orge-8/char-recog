# 真机部署指南 · cv-char-recog v0.3.0

> 交付版本：commit `9e9e790`（工作区 `plugins/cv-char-recog`）
> 定位：替换真机上已装的 MaiSelfPlugin（`maibot.mai-recog-self`），并迁移其 1000+ 张标注图库与已训模型。
> 部署方式：**整目录替换 + 完整重启 MaiBot**（热重载对插件不生效）。

---

## 0. 路径约定（按真机实际情况替换）

| 变量 | 含义 | 示例/说明 |
|---|---|---|
| `$MAIBOT` | 真机 MaiBot 根目录（含 `bot.py`） | 例如 `C:\...\MaiBot` |
| `$PLUGINS` | `$MAIBOT/plugins` | 插件目录 |
| `$DATA` | `$MAIBOT/data/plugins` | 插件数据目录 |
| `$SRC_DATA` | MaiSelfPlugin 数据目录 | `$DATA/maibot.mai-recog-self` |

部署后的插件目录：`$PLUGINS/cv_char_recog`
部署后的数据目录：`$DATA/org.civetc.cv-char-recog`

---

## 1. 前置核验（**必做，缺一不可**）

```bash
# 1.1 host 版本 ≥1.3.6 且 SDK ≥2.11.0
cd $MAIBOT && python -c "import maibot_sdk,sys;print('sdk',maibot_sdk.__version__)"
grep -m1 version $MAIBOT/pyproject.toml          # host 版本
# ⚠ 若 host 是 1.2.x：不满足门槛，必须先升级 MaiBot 再部署。

# 1.2 WebUI 能力（manifest 声明 custom_pages + image_upload）
#   在 WebUI 插件管理页确认能看到本插件的「中V角色识别」工作区入口

# 1.3 磁盘空间：预留 ≥3GB（torch 覆盖层下载 ~200MB + 安装 ~1GB，另有解压缓存）
```

**如发现 host 仍是 1.2.x**：停止部署，先升级。cv-char-recog 的 `_manifest.json` 要求 `host_application ≥1.3.6`、`sdk ≥2.11.0`，低于门槛会被宿主拒绝加载。

---

## 2. 备份（不可跳过）

```bash
# 旧插件与其数据（迁移完成前不要删）
cp -r $SRC_DATA $SRC_DATA.bak-$(date +%Y%m%d)
# 旧插件源码目录（若与数据目录同名，一并备份）
cp -r $PLUGINS/mai_recog_self $PLUGINS/mai_recog_self.bak 2>/dev/null
# plugin 配置（含 characters / auto_recognize）
cp -r $MAIBOT/config $MAIBOT/config.bak-$(date +%Y%m%d) 2>/dev/null
```

---

## 3. 桌面副本同步（工作区 → 桌面，逐字节一致）

```bash
SRC="C:/Users/38160/WorkBuddy/2026-10-13-21-14-57/plugins/cv-char-recog"
DST="C:/Users/38160/Desktop/cv-char-recog"
rm -rf "$DST" && mkdir -p "$DST"
cp -r "$SRC"/. "$DST"/
rm -rf "$DST/.git" "$DST/.pytest_cache" "$DST/__pycache__" "$DST/tests"   # 真机不需要
```

---

## 4. 停用 MaiSelfPlugin

在 WebUI 插件管理页停用（`enabled=false`）**或**直接移除 `$PLUGINS/mai_recog_self`。
目的：避免两个插件同时注册 `chat.receive.before_process` 造成**双推理 + 双提示**。

---

## 5. 部署插件目录（整目录替换）

把步骤 3 的桌面副本整目录拷到 `$PLUGINS/cv_char_recog`（目录名用下划线，与上游风格一致）。

**关键文件清单（37 个文件，10.4 MB，已剔除 .git/tests）：**

| 文件 | 大小 | sha256(前16位) |
|---|---|---|
| `_manifest.json` | 1,518 | `3be3e187791e04ea` |
| `plugin.py` | 29,409 | `c968b3885e08e52c` |
| `webui.json` | 48,676 | `7796dfff6ed6b266` |
| `recog/store.py` | 23,947 | `bc31e42a787d90e7` |
| `recog/service.py` | 20,823 | `8fa8c97b9dc0066b` |
| `recog/environment.py` | 20,566 | `cf1b474081f291db` |
| `recog/model.py` | 18,251 | `4a6d6a4b8955ca7f` |
| `recog/worker.py` | 7,184 | `8a3ad8071df279d6` |
| `recog/images.py` | 3,632 | `ff929acb23d5c1e2` |
| `recog/dependency_download.py` | 5,816 | `645730f4963a91fc` |
| `recog/bootstrap.py` | 999 | `f7b0a6957d117fdd` |
| `recog/processes.py` | 1,055 | `d29d8295799b94c1` |
| `tools/migrate_from_maiself.py` | 8,034 | `5e958902be5289a7` |
| `assets/mobilenet_v3_small-047dcff4.pth` | 10,306,551 | `047dcff4addef86e`（**必须完整，缺则训练/首次加载报“底座缺失或SHA256校验失败”**） |
| `webui/index.html` | 53,221 | `b32ce18c8e9498d0` |
| `webui/images.html` | 53,283 | `69adb5a6e4830c37` |
| `webui/training.html` | 53,285 | `3730e49e802f4054` |
| `webui/models.html` | 53,283 | `c7aa49c396ed16da` |
| `webui/test.html` | 53,281 | `d1382f55d1e1b0cd` |
| `webui/environment.html` | 53,288 | `0d012d2cec9bc6e7` |
| `webui/build.py` | 1,357 | `c86db88dcb0be78d` |
| `requirements-worker.txt` | 227 | `34ace9a89f382d67` |
| `pyproject.toml` | 291 | `b267d3a9d54b0814` |
| `CHANGELOG.md` / `README.md` / `LICENSE` / `NOTICE.md` / `VALIDATION.md` / `assets/icon.svg` / `_locales/zh-CN.json` | — | 见仓库 |

---

## 6. 完整重启 MaiBot

热重载对插件不生效，**必须完整重启进程**。重启后确认启动日志出现插件加载成功（无 `ModuleNotFoundError`、无 manifest 校验失败）。

---

## 7. 安装依赖覆盖层

WebUI → 插件工作区 → **运行环境**页 → 点击 **CPU 版本**：

- 下载 ~200MB，安装后约 1GB，装在 `$DATA/org.civetc.cv-char-recog/dependencies/packages`（**不污染 MaiBot 主环境**）
- 支持断点续传；插件重载/重启后可恢复安装进度
- 安装完成前训练与推理均被锁住（页面有状态提示）
- 无 GPU 也能用：推理路径为 CPU

```bash
# 校验：日志应显示 torch/torchvision 导入与版本核验通过
tail -f $DATA/org.civetc.cv-char-recog/environment.log
```

---

### 8.0 执行环境（Windows / PowerShell）

迁移脚本依赖 Pillow + numpy（导入 `recog.store` 链路），**必须用 MaiBot 宿主 Python 或插件依赖覆盖层**，裸系统 Python 通常缺包会在 `from PIL import Image` 处失败。

**先停止 MaiBot 再迁移**（避免与插件进程并发写 `catalog.sqlite3`），迁移完成后再启动。

> ⚠ 踩坑记录（真机 `E:\mai\maibot`，2026-10-14）：
> 1. `$MAIBOT` / `$DATA` 是**占位变量，必须先赋值**；未赋值会被展开成空串，路径塌成 `E:\...`。
> 2. 插件部署目录名是 **连字符 `cv-char-recog`**（不是 `cv_char_recog`）。
> 3. 不要在插件目录里执行 `uv run`——它会在插件目录新建一个空的 `.venv`（系统 Python，无 Pillow/numpy）。
>    应 `cd` 到 MaiBot 根目录再 `uv run`，才能用到 MaiBot 自己的环境。误建的话删掉：
>    `Remove-Item -Recurse -Force "$MAIBOT\plugins\cv-char-recog\.venv"`

```powershell
# 0. 切 UTF-8，避免中文角色名（--character）乱码
chcp 65001 | Out-Null
$env:PYTHONUTF8 = "1"

# 1. 变量赋值（真机示例：E:\mai\maibot）
$MAIBOT = "E:\mai\maibot"
$DATA   = "$MAIBOT\data\plugins"

# 2. 勘察：确认源/目标数据目录名与 Python 可用性
Get-ChildItem "$DATA" | Select-Object Name
Test-Path "$MAIBOT\.venv\Scripts\python.exe"      # MaiBot 是否有自带 venv

# 3. 停 MaiBot（略）后，从 MaiBot 根目录用宿主 Python 执行
cd $MAIBOT
uv run --no-sync python -c "import PIL, numpy; print('deps ok')"
uv run --no-sync python "$MAIBOT\plugins\cv-char-recog\tools\migrate_from_maiself.py" `
    --src "$DATA\maibot.mai-recog-self" `
    --dst "$DATA\org.civetc.cv-char-recog" `
    --character 鸣澜 --dry-run

# 4. 确认计划无误，去掉 --dry-run 再执行一次
```

退化路径（真机未用 uv）：用插件依赖覆盖层挂载后运行（在插件目录下执行）：

```powershell
cd "$MAIBOT\plugins\cv_char_recog"
& "$MAIBOT\.venv\Scripts\python.exe" -X utf8 `
    -m recog.bootstrap "$DATA\org.civetc.cv-char-recog\dependencies\packages" `
    tools\migrate_from_maiself.py --src "$DATA\maibot.mai-recog-self" `
    --dst "$DATA\org.civetc.cv-char-recog" --character 鸣澜
```

Git Bash 等价写法（路径用 `/c/...`，空格加引号）：

```bash
cd "$MAIBOT" && uv run --no-sync python "$MAIBOT/plugins/cv_char_recog/tools/migrate_from_maiself.py" \
  --src "$DATA/maibot.mai-recog-self" --dst "$DATA/org.civetc.cv-char-recog" --character 鸣澜 --dry-run
```

#### 8.0.1 真机 MaiBot 没有 `.venv` 时怎么办

`Test-Path "$MAIBOT\.venv\Scripts\python.exe"` 返回 `False` 时，`uv run` 在根目录也只会新建空环境。改用下面任一方式：

**A. 直接用 MaiBot 在跑的 Python（推荐，无需联网）**

```powershell
# MaiBot 运行时执行，拿到实际 python.exe 路径
Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
    Select-Object -ExpandProperty ExecutablePath -Unique

# 停服后迁移
& $PY -X utf8 "$MAIBOT\plugins\cv-char-recog\tools\migrate_from_maiself.py" `
    --src "$DATA\maibot.mai-recog-self" --dst "$DATA\org.civetc.cv-char-recog" `
    --character 鸣澜 --dry-run
```

**B. uv 临时环境（需联网拉 Pillow/numpy，不污染插件目录）**

```powershell
cd "$MAIBOT\plugins\cv-char-recog"
uv run --no-project --with Pillow --with numpy python tools\migrate_from_maiself.py `
    --src "$DATA\maibot.mai-recog-self" --dst "$DATA\org.civetc.cv-char-recog" `
    --character 鸣澜 --dry-run
```

> 已确认的真机目录名（`Get-ChildItem $DATA`）：源 `maibot.mai-recog-self`、目标 `org.civetc.cv-char-recog`。

### 8.1 迁移命令本体

```bash
cd $PLUGINS/cv_char_recog

# 8.1 先看计划（不写入任何数据）
python tools/migrate_from_maiself.py \
  --src $DATA/maibot.mai-recog-self \
  --dst $DATA/org.civetc.cv-char-recog \
  --character 鸣澜 --dry-run

# 8.2 执行迁移
python tools/migrate_from_maiself.py \
  --src $DATA/maibot.mai-recog-self \
  --dst $DATA/org.civetc.cv-char-recog \
  --character 鸣澜
```

迁移内容：图片文件 + 标注 + 划分（train/val/test）+ 已训模型（登记为「鸣澜」并设为使用中）+ 训练血统（`trained_sources`，保证后续是**增量训练**而非从头重训）+ 回退链与留出真值。**幂等**，重复运行不会重复入库。

迁移后旧数据目录先保留，验证通过后（步骤 10）再清理。

---

## 9. 配置角色名单

在宿主生成的 `config.toml` 中找到本插件配置段，确认 `characters`：

```toml
[plugin.org.civetc.cv-char-recog]
characters = ["鸣澜", "洛天依", "言和", "乐正绫", "乐正龙牙", "徵羽摩柯", "墨清弦", "星尘", "心华"]
auto_recognize = true
input_size = 320
```

- 「鸣澜」= 迁移来的 Bot 自身形象；其余为中V角色（可任意增删）
- 只有**已启用模型的角色**参与识别，未训练的角色自动跳过
- 想先只验证迁移、不打扰聊天，把 `auto_recognize` 设为 `false`

改完配置需**重载插件**生效（角色名单写入插件数据目录供 WebUI 读取）。

---

## 10. 验证清单

| # | 验证项 | 预期 |
|---|---|---|
| 1 | 插件加载 | 启动日志无报错；WebUI 出现「中V角色识别」工作区 |
| 2 | 角色选择器 | 头部下拉列出 `characters` 里的角色，切换后统计卡/按钮文案随之变化 |
| 3 | 图库 | 选「鸣澜」应看到迁移的 1000+ 张图，标注与划分正常 |
| 4 | 模型版本 | 迁移模型显示「使用中」；可查看报告/错判；可回退 |
| 5 | 识别测试 | 上传一张 Bot 形象图 → 命中「鸣澜」，得分 ≥0.6；耗时应在 0.1–0.3s 量级（首次含模型加载） |
| 6 | 聊天自动识别 | 群里发图 → 命中时消息追加 `[提示：此图中的角色看起来很可能是 "鸣澜"]` |
| 7 | 增量训练 | 新标若干图后提交训练：任务快照的 `parent` 应为迁移模型，只训新增图 |
| 8 | 性能 | 每张图全角色推理 0.1–0.3s CPU；单条消息 6s 预算；重复图片走缓存（0 推理） |
| 9 | 环境（可选） | 有 torch 时补跑 3 个 torch 测试模块：`pytest tests -q` |

验证通过后：

```bash
# 清理旧插件资源（省 ~1GB torch 覆盖层 + 图库磁盘）
rm -rf $PLUGINS/mai_recog_self $SRC_DATA.bak-* $SRC_DATA
```

---

## 11. 回滚预案

| 场景 | 处置 |
|---|---|
| 插件加载失败 | 删除 `$PLUGINS/cv_char_recog` → 恢复 MaiSelfPlugin → 重启；插件数据目录独立，不影响其他插件 |
| 迁移出问题 | 删除 `$DATA/org.civetc.cv-char-recog` 后重跑迁移（源目录在第 2 步已备份） |
| 识别误报多 | 模型版本页回退上一版本；或调高判定门槛（当前命中阈值 0.6，在 `plugin.py` 与 `recog/images.py` 的 `judgment` 中） |
| 依赖装不上 | 运行环境页切「自定义环境」，指向已有 Python/环境目录，由管理员自行管理依赖 |

---

## 13. 0.3.0 更名迁移（cv-char-recog → char-recog，真机已部署旧 id 时执行）

插件 id 从 `org.civetc.cv-char-recog` 改为 `org.civetc.char-recog`，数据目录随 id 变化。**迁移数据已验证可用，改名不影响数据内容**：

1. 停 MaiBot
2. 数据目录改名（数据兼容，直接改）：`Rename-Item "$DATA\org.civetc.cv-char-recog" "org.civetc.char-recog"`
3. 插件目录替换为新版（更名后的构建产物）
4. 若宿主在 config 目录按 id 存了插件配置，把旧 id 的配置目录/文件同步改名为新 id（找不到就以「新 id + WebUI 配置页重新填 characters」为准）
5. 启动 MaiBot → WebUI 应显示「角色识别」→ 确认鸣澜模型仍「使用中」、图库 1008 张

---

## 12. 共存提示

- **MaiSelfPlugin**：已停用/替换，勿再启用（双 Hook 双推理）。
- **character-recognizer（v1.0.14）**：按决策 D12，验证通过后下线。
- **character-recognizer-zhongv（org.mai-mai.zhongv-recognizer v0.2.1）**：走的是「百炼多模态向量 + VLM」路线，与本地训练路线机制不同；若同时启用，同一张图可能被**两条提示**命中。建议二选一，或保留中V识别走 zhongv、Bot 自身形象走 cv-char-recog（把 `characters` 只留「鸣澜」）。
