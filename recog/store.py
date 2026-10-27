"""SQLite catalog, persistent variant groups and immutable task snapshots."""

from pathlib import Path
from typing import Any
from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import json
import random
import shutil
import sqlite3
import time
import uuid
import logging

from .images import decode_with_animation, fingerprint, similar, thumbnail
from .scores import (
    DEFAULT_THRESHOLD,
    EVALUATION_THRESHOLD,
    MAX_THRESHOLD,
    MIN_THRESHOLD,
)

# 固定评估集：每类默认 25 组。0.5.x 按内容类型分档累加预算（每档 25 / 每个负例
# 分档 15），0.6.0 撤掉标注轴后扁平为每类一个上限，挑选规则（整批隔离）不变。
EVAL_PER_LABEL = 25
# 聊天捕获上限：待确认队列不能变成第二个图库。
CAPTURE_LIMIT = 100
# 人工审核口径，也是 100 张配额的口径：未标注、未被自动排除、同组还没有真值。
# 0.5.22 的入库闸门保证新的聊天图不会带着 reason 进库，所以 reason='' 与队列一致；
# 老库里 reason!='' 的聊天图不再吃配额，由一次性清理动作处理。
CAPTURE_REVIEWABLE = (
    "i.removed=0 AND i.reason='' AND i.label IS NULL AND i.source='chat' "
    "AND NOT EXISTS (SELECT 1 FROM images o WHERE o.group_id=i.group_id "
    "AND o.removed=0 AND o.label IN ('self','other'))"
)
# 一组只出一条代表，按 id 取最小，保证翻页和撤销后位置稳定。
CAPTURE_REPRESENTATIVE = (
    "i.id=(SELECT MIN(c.id) FROM images c WHERE c.group_id=i.group_id AND c.removed=0 "
    "AND c.label IS NULL AND c.source='chat' AND c.reason='')"
)
# WebUI 的「上次选择」。宿主把 custom 页放进 sandbox="allow-scripts" 的 iframe
# （不透明源），localStorage 会抛 SecurityError，所以只能存在插件库里。
UI_CHOICES = {
    "filter": {"all", "eligible", "excluded", "unlabelled", "errors", "removed"},
    "mode": {"incremental", "fresh"},
    "device": {"cpu", "cuda"},
    "job_scope": {"current", "all"},
    "job_state": {"all", "complete", "failed", "cancelled", "paused"},
}

_LOG = logging.getLogger("char-recog")
# 每类留至少这么多组给训练：这是插件能训练的下限，评估集不能把它吃掉。
EVAL_MIN_TRAIN = 20


def dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(dump(value).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(dump(value), encoding="utf-8")
    for attempt in range(6):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.02)


class SourceIndex:
    """O(1) equivalent of scanning ``reversed(sources)`` for the first match.

    The reference implementation returns at the first source (scanning from the
    end) whose group_id, sha, or decoded hash matches, so the answer is decided
    by the highest matching index. Three separate maps keep the three key spaces
    apart: a flat map would let ``row["sha"]`` collide with some other source's
    ``decoded`` value and produce a match the reference never makes.
    """

    __slots__ = ("sources", "groups", "shas", "decoded")

    def __init__(self, sources):
        self.sources = sources
        self.groups: dict = {}
        self.shas: dict = {}
        self.decoded: dict = {}
        for index, source in enumerate(sources):
            # Later writes overwrite earlier ones: "last occurrence wins".
            self.groups[source["group_id"]] = index
            self.shas[source["sha"]] = index
            self.decoded[source["decoded"]] = index

    def seen(self, row) -> bool:
        index = max(
            self.groups.get(row["group_id"], -1),
            self.shas.get(row["sha"], -1),
            self.decoded.get(row["decoded"], -1),
        )
        return index >= 0 and self.sources[index]["label"] == row["label"]


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        for name in ("images", "jobs", "models", "uploads"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript("""
              PRAGMA journal_mode=WAL;
              CREATE TABLE IF NOT EXISTS images(id TEXT PRIMARY KEY, path TEXT, sha TEXT, decoded TEXT, phash TEXT, aspect REAL, width INTEGER, height INTEGER, group_id TEXT, label TEXT, removed INTEGER DEFAULT 0, reason TEXT DEFAULT '', created REAL, character TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT '', model_score REAL);
              CREATE TABLE IF NOT EXISTS groups(id TEXT PRIMARY KEY, split TEXT);
              CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, request_key TEXT UNIQUE, state TEXT, pid INTEGER, error TEXT, created REAL, snapshot_sha TEXT, character TEXT NOT NULL DEFAULT '', device TEXT NOT NULL DEFAULT '', input_size INTEGER);
              CREATE UNIQUE INDEX IF NOT EXISTS one_training ON jobs((1)) WHERE state IN ('queued','running','cancelling');
              CREATE UNIQUE INDEX IF NOT EXISTS one_training_v2 ON jobs((1)) WHERE state IN ('queued','running','pausing','cancelling');
              CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY, job_id TEXT, sha TEXT, metrics TEXT, created REAL, character TEXT NOT NULL DEFAULT '');
              CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT);
              CREATE TABLE IF NOT EXISTS job_checkpoints(job_id TEXT PRIMARY KEY, sha TEXT NOT NULL);
              CREATE INDEX IF NOT EXISTS images_group ON images(group_id);
              CREATE INDEX IF NOT EXISTS images_character ON images(character);
            """)
            # Legacy single-character databases upgrade in place.
            for table in ("images", "jobs", "models"):
                columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
                if "character" not in columns:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN character TEXT NOT NULL DEFAULT ''")
            # 0.4.0: device/size live on the row so listing jobs stops reading
            # every snapshot.json (25 MiB per poll on a busy install).
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            if "device" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN device TEXT NOT NULL DEFAULT ''")
            if "input_size" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN input_size INTEGER")
            # 0.5.20: where an image came from, and what the model said when it was
            # captured. A chat-captured image is *not* ground truth, so the score is
            # kept next to it instead of the label.
            columns = {row[1] for row in db.execute("PRAGMA table_info(images)")}
            if "source" not in columns:
                db.execute("ALTER TABLE images ADD COLUMN source TEXT NOT NULL DEFAULT ''")
            if "model_score" not in columns:
                db.execute("ALTER TABLE images ADD COLUMN model_score REAL")
            # 0.6.0: the annotation axes (medium / text overlay / negative identity)
            # are removed, physically — a column left behind empty is an invitation
            # for the next feature to start filling it again. The table is rebuilt
            # instead of `ALTER TABLE ... DROP COLUMN`, which needs SQLite 3.35+ and
            # the host's interpreter decides that version, not us. Labels, splits,
            # sources and model scores carry over unchanged; only the three tags go.
            #
            # 0.6.1 起这段重建有三重保证，因为 0.6.0 的版本有两个实测出来的崩溃窗口：
            #   ① 整个重建包在一个 SAVEPOINT 里 —— SQLite 的 DDL 是事务性的，进程在
            #      中途死掉会让整段回滚，不会留下「images 没了、数据在 images_v6 里」
            #      这种既加载不了又救不回来的状态；
            #   ② 开头先 DROP TABLE IF EXISTS images_v6 —— 上一次尝试若死在
            #      CREATE 之后、DROP images 之前，会留下半份 images_v6；不先清掉，
            #      下一次 CREATE 就会撞 "table images_v6 already exists"，
            #      整个插件再也加载不起来（这个窗口实测可复现）；
            #   ③ 检测到「有 images_v6 却没有 images」时自动补完 RENAME ——
            #      那是 0.6.0 已发布版本留下的孤儿状态，老库首次加载要能自愈。
            tables = {
                row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            columns = {row[1] for row in db.execute("PRAGMA table_info(images)")} if "images" in tables else set()
            legacy = {"style", "neg_type", "has_text"} & columns
            orphaned = "images" not in tables and "images_v6" in tables
            if legacy or orphaned:
                db.execute("SAVEPOINT migrate_v6")
                try:
                    db.execute("DROP TABLE IF EXISTS images_v6")
                    if orphaned:
                        # 0.6.0 的迁移死在 DROP TABLE images 之后、RENAME 之前。
                        # images_v6 里就是全部数据，补完重命名即恢复。
                        db.execute("ALTER TABLE images_v6 RENAME TO images")
                    else:
                        db.execute(
                            "CREATE TABLE images_v6(id TEXT PRIMARY KEY, path TEXT, sha TEXT, decoded TEXT, phash TEXT, "
                            "aspect REAL, width INTEGER, height INTEGER, group_id TEXT, label TEXT, removed INTEGER DEFAULT 0, "
                            "reason TEXT DEFAULT '', created REAL, character TEXT NOT NULL DEFAULT '', "
                            "source TEXT NOT NULL DEFAULT '', model_score REAL)"
                        )
                        db.execute(
                            "INSERT INTO images_v6(id,path,sha,decoded,phash,aspect,width,height,group_id,label,removed,reason,"
                            "created,character,source,model_score) SELECT id,path,sha,decoded,phash,aspect,width,height,group_id,"
                            "label,removed,reason,created,character,source,model_score FROM images"
                        )
                        db.execute("DROP TABLE images")
                        db.execute("ALTER TABLE images_v6 RENAME TO images")
                    db.execute("CREATE INDEX IF NOT EXISTS images_group ON images(group_id)")
                    db.execute("CREATE INDEX IF NOT EXISTS images_character ON images(character)")
                    if legacy:
                        db.execute(
                            "INSERT OR REPLACE INTO settings VALUES (?,?)",
                            (
                                "annotation_removed",
                                dump({"columns": sorted(legacy), "at": time.time()}),
                            ),
                        )
                    db.execute("RELEASE SAVEPOINT migrate_v6")
                except BaseException:
                    # 连 CancelledError / KeyboardInterrupt 一起回滚：迁移做一半
                    # 比不做更危险，绝不能留半成品给下一次加载。
                    db.execute("ROLLBACK TO SAVEPOINT migrate_v6")
                    db.execute("RELEASE SAVEPOINT migrate_v6")
                    raise

    @contextmanager
    def db(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.root / "catalog.sqlite3", timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def setting(self, key: str, default: str = "") -> str:
        with self.db() as db:
            row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self.db() as db:
            db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, value))

    THUMBNAIL_LIMIT = 4096

    def _trim_thumbnails(self) -> None:
        """缩略图缓存按 sha 命名、只增不减，与图片库同寿命增长。

        没有上限时一个装了几万张图的库会顺手攒下几万个 512px JPEG。
        这里按条目数设上限，超了删最旧的；删缓存只是下次多算一次缩略图，
        不影响正确性，所以失败也只在 debug 级留痕。
        """
        cache = self.root / "thumbnails"
        try:
            entries = [p for p in cache.iterdir() if p.is_file()]
        except OSError:
            return
        if len(entries) <= self.THUMBNAIL_LIMIT:
            return
        entries.sort(key=lambda p: p.stat().st_mtime)
        for entry in entries[: len(entries) - self.THUMBNAIL_LIMIT]:
            try:
                entry.unlink()
            except OSError as exc:
                _LOG.debug("淘汰缩略图缓存失败（%s）：%s", entry, exc)

    def sweep_uploads(self, max_age_seconds: float = 86400.0) -> int:
        """清掉 uploads/ 里超过时限的孤儿文件，返回删除个数。

        `ingest` 的每条退出路径都会移走或删掉自己的文件，所以正常情况下
        uploads/ 是空的。但只要 `ingest` 抛异常（库锁、IOERR、解码失败、
        或历史上 0.6.0 的参数个数 bug），那个 `.img` 就永久留在磁盘上，
        而且**没有任何日志**——用户只看到盘在慢慢涨。

        挂在 on_load 上而不是定时任务：插件重启一次清一次，成本是一次
        readdir，不需要额外的线程或计时器。`max_age_seconds` 用作保险，
        避免删掉正在写入的文件（`_capture` 先 write_bytes 再 ingest，
        两者之间只隔几毫秒，但网络盘上不好说）。
        """
        uploads = self.root / "uploads"
        try:
            entries = list(uploads.iterdir())
        except OSError:
            return 0
        cutoff = time.time() - max_age_seconds
        removed = 0
        for entry in entries:
            try:
                if not entry.is_file() or entry.stat().st_mtime > cutoff:
                    continue
                entry.unlink()
                removed += 1
            except OSError as exc:
                _LOG.debug("清理上传孤儿文件失败（%s）：%s", entry, exc)
        if removed:
            _LOG.info("清理 uploads/ 孤儿文件 %d 个（超过 %d 秒未处理）", removed, int(max_age_seconds))
        return removed

    def ui_state(self) -> dict:
        try:
            stored = json.loads(self.setting("ui_state", "{}"))
        except (TypeError, ValueError):
            stored = {}
        return self._clean_ui(stored if isinstance(stored, dict) else {})

    def set_ui_state(self, patch: dict) -> dict:
        if not isinstance(patch, dict):
            raise ValueError("界面状态格式不正确")
        state = self.ui_state()
        state.update(self._clean_ui(patch))
        self.set_setting("ui_state", json.dumps(state, ensure_ascii=False))
        return state

    @staticmethod
    def _clean_ui(state: dict) -> dict:
        """Keep only known options so a stale client cannot smuggle arbitrary values."""
        clean = {}
        for key, allowed in UI_CHOICES.items():
            value = state.get(key)
            if isinstance(value, str) and value in allowed:
                clean[key] = value
        for key in ("character", "capture_character"):
            if key not in state or not isinstance(state[key], str):
                continue
            value = state[key].strip()
            if (key == "capture_character" and not value) or (value and len(value) <= 32):
                clean[key] = value
        if "input_size" in state:
            value = state["input_size"]
            if value is None or (type(value) is int and value in (224, 320, 448)):
                clean["input_size"] = value
        if isinstance(state.get("device_chosen"), bool):
            clean["device_chosen"] = state["device_chosen"]
        return clean

    # 角色名最终会经 plugin.py 的 f-string 原样进入 LLM 上下文提示，所以这里
    # 不只是「非空 + 限长」：换行、引号、方括号、Unicode 双向控制字符都能改变
    # 提示词的结构或视觉顺序（自伤型提示注入，提交训练的人可控 32 字符）。
    # 做法是**黑名单控字符**而不是白名单字符类——角色名要支持中日韩汉字与
    # 各种 P 主起的怪名字，白名单必然误伤；黑名单只挡「能改变提示词结构」的
    # 那一小类，正常名字不会碰线。
    _CHARACTER_RE = None

    @staticmethod
    def validate_character(character: str) -> str:
        import re

        if not isinstance(character, str) or not character.strip():
            raise ValueError("缺少角色名")
        character = character.strip()
        if len(character) > 32:
            raise ValueError("角色名过长（最多32字）")
        if Store._CHARACTER_RE is None:
            # CR/LF/TAB 能另起一行伪造成新指令；引号与方括号能闭合模板里的
            # 「」；U+200E/200F/202A-202E/2066-2069 是双向控制符，能视觉上
            # 重排文字顺序。反斜杠与竖线防转义混淆。
            forbidden = "\n\r\t\"'`[]{}<>\\|\u200b\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
            Store._CHARACTER_RE = re.compile(rf"[{re.escape(forbidden)}]")
        if Store._CHARACTER_RE.search(character):
            raise ValueError("角色名包含不允许的字符（换行、引号、括号或方向控制符）")
        return character

    def active(self, character: str) -> str:
        return self.setting(f"active:{character}")

    def set_active(self, character: str, version: str) -> None:
        """Swap in the new active version, keeping the old one for rollback."""
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            key = f"active:{character}"
            old = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            if old and old[0] != version:
                db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (f"previous:{character}", old[0]))
            db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, version))

    def previous(self, character: str) -> str:
        return self.setting(f"previous:{character}")

    def active_characters(self) -> list:
        """Characters that currently have an active model (inference targets)."""
        with self.db() as db:
            return sorted(row[0][len("active:"):] for row in db.execute("SELECT key FROM settings WHERE key LIKE 'active:%'") if row[0].startswith("active:"))

    @staticmethod
    def _parse_threshold(raw) -> float | None:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        # `value != value` rejects NaN, which float() happily accepts.
        if value != value or not MIN_THRESHOLD <= value <= MAX_THRESHOLD:
            return None
        return value

    def _threshold(self, db: sqlite3.Connection, character: str) -> float:
        row = db.execute("SELECT value FROM settings WHERE key=?", (f"threshold:{character}",)).fetchone()
        return self._parse_threshold(row[0] if row else None) or DEFAULT_THRESHOLD

    def threshold(self, character: str) -> float:
        """Per-character online gate; falls back to the default on corrupt data."""
        character = self.validate_character(character)
        with self.db() as db:
            return self._threshold(db, character)

    def thresholds(self) -> dict:
        """Every configured gate in one query, for the inference worker."""
        with self.db() as db:
            rows = db.execute("SELECT key,value FROM settings WHERE key LIKE 'threshold:%'").fetchall()
        result = {}
        for key, value in rows:
            parsed = self._parse_threshold(value)
            if parsed is not None:
                result[key[len("threshold:"):]] = parsed
        return result

    def set_threshold(self, character: str, value) -> dict:
        character = self.validate_character(character)
        parsed = self._parse_threshold(value)
        if parsed is None:
            raise ValueError(f"阈值必须是 {MIN_THRESHOLD} 到 {MAX_THRESHOLD} 之间的数字")
        self.set_setting(f"threshold:{character}", f"{parsed:.3f}")
        return {"character": character, "threshold": parsed, "default": DEFAULT_THRESHOLD}

    def clear_threshold(self, character: str) -> dict:
        """Delete the override instead of writing 0.6: "default" stays a moving target."""
        character = self.validate_character(character)
        with self.db() as db:
            db.execute("DELETE FROM settings WHERE key=?", (f"threshold:{character}",))
        return {"character": character, "threshold": DEFAULT_THRESHOLD, "default": DEFAULT_THRESHOLD}

    def _reconcile(self, db: sqlite3.Connection, group: str) -> None:
        rows = db.execute(
            "SELECT * FROM images WHERE group_id=? AND removed=0 AND reason NOT IN ('solid','split_conflict') ORDER BY created,id",
            (group,),
        ).fetchall()
        labels = {r["label"] for r in rows if r["label"] in {"self", "other"}}
        rows = sorted(rows, key=lambda r: (r["label"] not in {"self", "other"}, r["created"], r["id"]))
        for index, row in enumerate(rows):
            reason = "label_conflict" if len(labels) > 1 else ("duplicate" if index else "")
            db.execute("UPDATE images SET reason=? WHERE id=?", (reason, row["id"]))

    @staticmethod
    def _skip_capture(path: Path, reason: str) -> dict:
        """Drop the temp file without touching the catalog.

        Runs in the capture worker thread; failures are logged at debug and never
        reach the chat path.
        """
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            _LOG.debug("聊天捕获临时文件未能删除（%s）：%s", path, exc)
        _LOG.debug("聊天捕获未入库（%s）：%s", reason, path)
        return {"id": "", "reason": reason, "skipped": True}

    def ingest(
        self,
        path: Path,
        label: str | None = None,
        character: str = "",
        source: str = "",
        model_score: float | None = None,
    ) -> dict:
        if label not in {None, "self", "other"}:
            raise ValueError("上传身份必须为是Bot或不是Bot")
        character = self.validate_character(character)
        path = path.resolve()
        if not path.is_relative_to(self.root / "uploads") or path.is_symlink():
            raise ValueError("只能领取本插件上传文件")
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        image, animated = decode_with_animation(raw)
        fp = fingerprint(image)
        if animated:
            # 动图的像素指纹只覆盖首帧，两张「首帧相同、后续帧不同」的动图会算出
            # 同一个 decoded —— 直接用它做精确去重会把真变体当成重复删掉。改认原始
            # 字节：同一份文件仍然判重，不同文件各自留档。pHash 不动，模糊合并照旧
            # 走 similar()。
            fp["decoded"] = "animated:" + sha
        ident = uuid.uuid4().hex
        destination = self.root / "images" / (ident + path.suffix)
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT id,reason FROM images WHERE character=? AND (sha=? OR decoded=?) ORDER BY created,id LIMIT 1",
                (character, sha, fp["decoded"]),
            ).fetchone()
            if existing is not None:
                path.unlink()
                return {"id": existing["id"], "reason": existing["reason"], "already_exists": True}
            matches = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM images WHERE character=? AND reason NOT IN ('solid','split_conflict')",
                    (character,),
                )
                if similar(fp, dict(r))
            ]
            groups = {r["group_id"] for r in matches}
            splits = {r[0] for g in groups for r in db.execute("SELECT split FROM groups WHERE id=?", (g,)) if r[0]}
            # 0.5.22: a chat capture is only worth keeping if it can become the
            # group's usable member. Anything else would be stored as duplicate /
            # automatic garbage, invisible in the review queue and pure disk cost.
            # Uploads keep the old behaviour: a variant there is an explicit choice.
            if source == "chat":
                if fp["solid"]:
                    return self._skip_capture(path, "solid")
                if len(splits) > 1:
                    return self._skip_capture(path, "split_conflict")
                if groups:
                    marks = ",".join("?" for _ in groups)
                    usable = db.execute(
                        f"SELECT 1 FROM images WHERE group_id IN ({marks}) AND removed=0 AND reason='' LIMIT 1",
                        tuple(groups),
                    ).fetchone()
                    if usable is not None:
                        return self._skip_capture(path, "group_covered")
            reason = "solid" if fp["solid"] else ("split_conflict" if len(splits) > 1 else "")
            group = min(groups) if groups and reason != "split_conflict" else ident
            db.execute(
                "INSERT OR IGNORE INTO groups VALUES (?,?)",
                (group, next(iter(splits), None) if reason != "split_conflict" else None),
            )
            if not reason:
                for old in groups - {group}:
                    db.execute("UPDATE images SET group_id=? WHERE group_id=?", (group, old))
                    db.execute("DELETE FROM groups WHERE id=?", (old,))
                if splits:
                    db.execute("UPDATE groups SET split=? WHERE id=?", (next(iter(splits)), group))
            shutil.move(path, destination)
            db.execute(
                "INSERT INTO images(id,path,sha,decoded,phash,aspect,width,height,group_id,label,removed,reason,created,character,source,model_score) VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?,?,?,?)",
                (
                    ident,
                    str(destination.relative_to(self.root)),
                    sha,
                    fp["decoded"],
                    fp["phash"],
                    fp["aspect"],
                    fp["width"],
                    fp["height"],
                    group,
                    label,
                    reason,
                    time.time(),
                    character,
                    source,
                    model_score,
                ),
            )
            self._reconcile(db, group)
        return {"id": ident, "reason": reason, "character": character}

    def label(self, image_id: str, label: str, remove: bool | None = None) -> dict:
        if label not in {"self", "other", "skip", "unlabelled"}:
            raise ValueError("无效标注")
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT group_id FROM images WHERE id=?", (image_id,)).fetchone()
            if row is None:
                raise ValueError("图片不存在")
            db.execute(
                "UPDATE images SET label=?,removed=COALESCE(?,removed) WHERE id=?",
                (None if label == "unlabelled" else label, int(remove) if remove is not None else None, image_id),
            )
            self._reconcile(db, row[0])
        return {"id": image_id, "label": label, "removed": remove}

    def set_removed(self, image_id: str, removed: bool) -> dict:
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT group_id FROM images WHERE id=?", (image_id,)).fetchone()
            if row is None:
                raise ValueError("图片不存在")
            db.execute("UPDATE images SET removed=? WHERE id=?", (int(removed), image_id))
            self._reconcile(db, row[0])
        return {"id": image_id, "removed": removed}

    @staticmethod
    def split_ids(image_ids) -> list:
        """Accept both CSV and a JSON array string.

        0.3.3 and earlier pages sent `JSON.stringify([...selected])`; the host may
        also cache page files, so both shapes stay valid instead of failing late.
        """
        if isinstance(image_ids, (list, tuple)):
            values = [str(value).strip() for value in image_ids]
        elif isinstance(image_ids, str) and image_ids.strip().startswith("["):
            try:
                parsed = json.loads(image_ids)
            except ValueError as exc:
                raise ValueError("无效图片ID") from exc
            if not isinstance(parsed, list):
                raise ValueError("无效图片ID")
            values = [str(value).strip() for value in parsed]
        else:
            values = [value.strip() for value in str(image_ids or "").split(",")]
        return list(dict.fromkeys(value for value in values if value))

    def batch_update(self, image_ids, operation: str, character: str = "") -> dict:
        """Apply the entire selection in one transaction, without partial updates."""
        if operation not in {"self", "other", "remove", "remove_all"}:
            raise ValueError("无效批量操作")
        ids = self.split_ids(image_ids)
        if operation != "remove_all" and (not ids or len(ids) > 1000):
            raise ValueError("请先选择图片，每批最多1000张")
        if any(len(i) != 32 or any(c not in "0123456789abcdef" for c in i) for i in ids):
            raise ValueError("无效图片ID")
        if operation == "remove_all":
            # Scoped on purpose: this button lives on a single character's page.
            character = self.validate_character(character)
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if operation == "remove_all":
                rows = db.execute(
                    "SELECT id,group_id FROM images WHERE removed=0 AND character=?", (character,)
                ).fetchall()
                db.execute("UPDATE images SET removed=1 WHERE removed=0 AND character=?", (character,))
            else:
                placeholders = ",".join("?" for _ in ids)
                rows = db.execute(f"SELECT id,group_id FROM images WHERE id IN ({placeholders})", ids).fetchall()
                if len(rows) != len(ids):
                    raise ValueError("选择中有不存在的图片，请清空选择后重新选择")
                if operation in {"self", "other", "remove"}:
                    characters = {
                        row[0]
                        for row in db.execute(
                            f"SELECT DISTINCT character FROM images WHERE id IN ({placeholders})", ids
                        )
                    }
                    if len(characters) > 1:
                        raise ValueError("选择跨多个角色，请按角色分别标注")
                if operation == "remove":
                    db.execute(f"UPDATE images SET removed=1 WHERE id IN ({placeholders})", ids)
                else:
                    if db.execute(f"SELECT 1 FROM images WHERE removed=1 AND id IN ({placeholders}) LIMIT 1", ids).fetchone():
                        raise ValueError("选择中有已删除图片，请先恢复或取消选择")
                    db.execute(f"UPDATE images SET label=? WHERE id IN ({placeholders})", [operation, *ids])
            for group in {row["group_id"] for row in rows}:
                self._reconcile(db, group)
        return {"updated": len(rows), "operation": operation}

    def trained_sources(self, db, version: str) -> list:
        if not version:
            return []
        record = db.execute("SELECT job_id FROM models WHERE id=?", (version,)).fetchone()
        if record is None:
            raise ValueError("模型版本不存在")
        snapshot = self.trained_snapshot(record[0])
        if snapshot is None:
            raise ValueError("模型训练记录不完整")
        return snapshot.get("trained_sources", [r for r in snapshot["images"] if r["split"] == "train"])

    def trained_snapshot(self, job_id: str) -> dict | None:
        """Task snapshots move to models/<job>/ when the task record is cleaned up."""
        for base in ("jobs", "models"):
            path = self.root / base / job_id / "snapshot.json"
            try:
                return json.loads(path.read_text("utf-8"))
            except (OSError, ValueError):
                continue
        return None

    @staticmethod
    def source_seen(row, sources) -> bool:
        """Compatibility wrapper; hot paths build a SourceIndex instead."""
        return SourceIndex(sources).seen(row)

    def images(
        self,
        page: int = 1,
        view: str = "all",
        job_id: str = "",
        character: str = "",
    ) -> dict:
        if page < 1 or view not in {"all", "eligible", "excluded", "unlabelled", "errors", "removed", "captured"}:
            raise ValueError("无效分页")
        # The review queue is cross-character: one list, one pass. Everything else
        # is scoped, so an empty character means "all" only for that view.
        if not character and view != "captured":
            raise ValueError("缺少角色名")
        if character:
            character = self.validate_character(character)
        threshold = self.threshold(character) if character else 0.0
        where = {
            "all": "i.removed=0",
            "eligible": "i.removed=0 AND i.reason='' AND i.label IN ('self','other')",
            "excluded": "(i.removed=1 OR i.reason!='' OR i.label='skip')",
            "unlabelled": "i.label IS NULL",
            "errors": "1",
            "removed": "i.removed=1",
            # 0.5.22: the chat-capture review queue. One row per group so a variant
            # is not the same question twice.
            "captured": CAPTURE_REVIEWABLE + " AND " + CAPTURE_REPRESENTATIVE,
        }[view]
        error_ids = set()
        predictions = {}
        if view == "errors":
            if not job_id:
                return {
                    "rows": [],
                    "gallery": [],
                    "pages": 1,
                    "total": 0,
                    "pagination": {"page": 1, "pages": 1, "total": 0},
                }
            report = self.report(job_id)
            pairs = report.get("test_predictions", []) + report.get("validation_predictions", [])
            predictions = {r["id"]: r for r in pairs}
            # "Misjudged" is relative to the gate in force now, not the 0.5 the
            # training report was generated with.
            error_ids = {
                r["id"]
                for r in pairs
                if ("self" if r["score"] >= threshold else "other") != r["label"]
            }
        parameters: list = []
        if character:
            where += " AND i.character=?"
            parameters.append(character)
        if view == "errors":
            if error_ids:
                where += " AND i.id IN (" + ",".join("?" for _ in error_ids) + ")"
                parameters = [*parameters, *sorted(error_ids)]
            else:
                where = "0"
                parameters = []
        with self.db() as db:
            sources = self.trained_sources(db, self.active(character)) if character else []
            total = db.execute(f"SELECT COUNT(*) FROM images i WHERE {where}", parameters).fetchone()[0]
            page = min(page, max(1, (total + 23) // 24))
            # Capture review is a queue: the most confident grey-zone guesses first,
            # so early decisions also teach the model the most.
            order = "i.model_score DESC, i.created ASC" if view == "captured" else "i.created DESC"
            rows = [dict(r) for r in db.execute(
                f"SELECT i.*,g.split FROM images i JOIN groups g ON g.id=i.group_id WHERE {where} ORDER BY {order} LIMIT 24 OFFSET ?",
                [*parameters, (page - 1) * 24],
            )]
            capture = None
            if view == "captured":
                scope = " AND i.character=?" if character else ""
                scope_args = (character,) if character else ()
                # pending_groups == total: one reason='' row per group is exactly
                # what the queue lists, and it is also the quota unit.
                capture = {
                    # Counted without the tag filters: the page shows one global
                    # number, not "how many match the filter".
                    "pending_groups": db.execute(
                        f"SELECT COUNT(*) FROM images i WHERE {CAPTURE_REVIEWABLE}{scope}",
                        scope_args,
                    ).fetchone()[0],
                    "auto_rows": db.execute(
                        "SELECT COUNT(*) FROM images i WHERE i.source='chat' "
                        f"AND i.label IS NULL AND i.removed=0 AND NOT ({CAPTURE_REVIEWABLE}){scope}",
                        scope_args,
                    ).fetchone()[0],
                    "limit": CAPTURE_LIMIT,
                }
        index = SourceIndex(sources)
        gallery = []
        for r in rows:
            reasons = {"solid": "纯色/近纯色", "duplicate": "同图变体，仅保留一个代表", "label_conflict": "同图身份标签冲突", "split_conflict": "连接了不同历史划分"}
            name = r.get("character") or character
            r["status"] = "已移出训练库" if r["removed"] else reasons.get(r["reason"], "跳过" if r["label"] == "skip" else "可用")
            r["label_text"] = {"self": f"是{name}", "other": f"不是{name}", "skip": "跳过"}.get(r["label"], "未标注")
            r["source_text"] = {"chat": "聊天捕获", "webui": "网页上传", "test": "识别测试"}.get(
                r.get("source") or "", ""
            )
            r["captured_text"] = (
                f"捕获时得分 {r['model_score']:.3f}（不是真值，待确认）"
                if r.get("model_score") is not None
                else ""
            )
            r["trained"] = index.seen(r)
            r["training_text"] = "当前模型已训练" if r["trained"] else "评估集" if r["split"] == "eval" else "验证集" if r["split"] == "val" else "测试集" if r["split"] == "test" else "待训练"
            prediction = predictions.get(r["id"])
            if prediction:
                predicted = "self" if prediction["score"] >= threshold else "other"
                r["model_text"] = (
                    f"模型：{'是' + name if predicted == 'self' else '不是' + name}"
                    f" · 得分 {prediction['score']:.3f}（阈值 {threshold:.2f}）"
                    f" · 冻结真值：{'是' + name if prediction['label'] == 'self' else '不是' + name}"
                )
            else:
                r["model_text"] = ""
            gallery.append(
                {
                    "thumbnail": thumbnail(self.root / r["path"], self.root / "thumbnails" / (r["sha"] + "-512.jpg")),
                    "title": r["id"],
                    "description": f"{r['label_text']} / {r['status']} / {r['split'] or '未划分'}",
                    "badge": r["label_text"],
                    "subtitle": f"{r['width']}×{r['height']}",
                }
            )
            self._trim_thumbnails()
        for row, preview in zip(rows, gallery):
            row["thumbnail"] = preview["thumbnail"]
        result = {
            "rows": rows,
            "pages": max(1, (total + 23) // 24),
            "total": total,
            "pagination": {"page": page, "pages": max(1, (total + 23) // 24), "total": total},
        }
        if capture is not None:
            result["capture"] = capture
        return result

    def capture_pending(self, character: str = "") -> int:
        """Reviewable capture groups: the queue length and the quota unit.

        One group is one decision, so variants of one picture must not consume
        several slots. The 0.5.22 ingest gate keeps new captures out of the
        auto-excluded states, so the remaining ``reason!=''`` rows are old data
        and cannot inflate the queue. ``character=""`` counts every character,
        which is what the review page shows.
        """
        if character:
            character = self.validate_character(character)
        scope, args = (" AND i.character=?", (character,)) if character else ("", ())
        with self.db() as db:
            return db.execute(
                f"SELECT COUNT(DISTINCT i.group_id) FROM images i WHERE {CAPTURE_REVIEWABLE}{scope}",
                args,
            ).fetchone()[0]

    def confirm_capture(self, image_id: str, character: str, label: str) -> dict:
        """Turn one grey-zone capture into ground truth.

        0.6.0 dropped the medium pick from this path: it was the plugin's only
        hard annotation gate, and asking it here made every review answer cost
        two decisions instead of one.
        """
        character = self.validate_character(character)
        if label not in {"self", "other"}:
            raise ValueError("确认结果只能是该角色或不是该角色")
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
            if row is None:
                raise ValueError("图片不存在")
            if row["source"] != "chat":
                raise ValueError("只能确认聊天捕获的图片")
            if row["character"] != character:
                raise ValueError("图片不属于当前角色")
            if row["reason"] or row["removed"]:
                raise ValueError("这张图已被自动排除，不能确认")
            if row["label"] is not None:
                raise ValueError("这张图已经确认过了")
            covered = db.execute(
                "SELECT 1 FROM images WHERE group_id=? AND id<>? AND removed=0 AND label IN ('self','other') LIMIT 1",
                (row["group_id"], image_id),
            ).fetchone()
            if covered:
                raise ValueError("同组图片已有标注，无需重复确认")
            db.execute("UPDATE images SET label=? WHERE id=?", (label, image_id))
            self._reconcile(db, row["group_id"])
        return {"id": image_id, "label": label}

    def undo_capture(self, image_id: str, character: str) -> dict:
        """Put one confirmed capture back into the queue (one-step undo)."""
        character = self.validate_character(character)
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
            if row is None:
                raise ValueError("图片不存在")
            if row["source"] != "chat" or row["character"] != character:
                raise ValueError("只能撤销当前角色的聊天捕获")
            if row["label"] not in {"self", "other"}:
                raise ValueError("这张图还没有确认，无法撤销")
            if row["reason"] or row["removed"]:
                raise ValueError("这张图已被自动排除或移出训练库，无法撤销")
            other = db.execute(
                "SELECT 1 FROM images WHERE group_id=? AND id<>? AND removed=0 AND label IN ('self','other') LIMIT 1",
                (row["group_id"], image_id),
            ).fetchone()
            if other:
                raise ValueError("同组还有已标注图片，撤销会破坏同组真值")
            db.execute("UPDATE images SET label=NULL WHERE id=?", (image_id,))
            self._reconcile(db, row["group_id"])
        return {"id": image_id, "label": None}

    def _delete_captures(self, db, rows) -> list:
        """Drop rows and empty groups; return files to remove after commit.

        File removal is deferred because the caller still owns the transaction:
        deleting a file inside it would leave a record pointing at nothing after
        a rollback. The shared-sha check runs after all rows are gone so a batch
        of duplicates can still free its thumbnail.
        """
        doomed = list(rows)
        for row in doomed:
            db.execute("DELETE FROM images WHERE id=?", (row["id"],))
        files = []
        for row in doomed:
            if not db.execute("SELECT 1 FROM images WHERE group_id=? LIMIT 1", (row["group_id"],)).fetchone():
                db.execute("DELETE FROM groups WHERE id=?", (row["group_id"],))
            shared = db.execute("SELECT 1 FROM images WHERE sha=? LIMIT 1", (row["sha"],)).fetchone()
            files.append((row["path"], row["sha"], bool(shared)))
        return files

    def _drop_capture_files(self, files) -> None:
        for path_text, sha, shared in files:
            path = (self.root / path_text).resolve()
            if not path.is_relative_to(self.root / "images"):
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                # The record is already gone; a stuck file must not report failure
                # and invite a second click on a row that no longer exists.
                _LOG.warning("聊天捕获文件未能删除（%s）：%s", path, exc)
            if not shared:
                try:
                    (self.root / "thumbnails" / (sha + "-512.jpg")).unlink(missing_ok=True)
                except OSError:
                    pass

    def purge_capture(self, image_id: str, character: str) -> dict:
        """Delete an unconfirmed chat capture for good: the row *and* the file."""
        character = self.validate_character(character)
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
            if row is None:
                raise ValueError("图片不存在")
            if row["source"] != "chat" or row["character"] != character:
                raise ValueError("只能删除当前角色的聊天捕获")
            if row["label"] is not None:
                raise ValueError("已经确认的图片不能在这里删除；请到图片标注页移出训练库")
            files = self._delete_captures(db, [row])
        self._drop_capture_files(files)
        return {"id": image_id, "deleted": True}

    def purge_auto_captures(self, character: str = "") -> dict:
        """One-off cleanup for chat captures the catalog already excluded.

        The ingest gate stops new rows from landing in these states, so what is
        left is historical junk that occupies disk without ever reaching review.
        ``label IS NULL`` is deliberate: a ``label_conflict`` row carries a
        human label, and deleting it would destroy ground truth. The page is
        cross-character, so ``character=""`` cleans every character.
        """
        if character:
            character = self.validate_character(character)
        scope, args = (" AND character=?", (character,)) if character else ("", ())
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                f"SELECT * FROM images WHERE source='chat' AND reason!='' AND label IS NULL{scope}",
                args,
            ).fetchall()
            files = self._delete_captures(db, rows)
        self._drop_capture_files(files)
        return {"deleted": len(rows)}

    def training_readiness(self, character: str) -> dict:
        character = self.validate_character(character)
        counts = {"self": 0, "other": 0}
        with self.db() as db:
            for row in db.execute(
                "SELECT label,COUNT(DISTINCT group_id) AS count FROM images WHERE character=? AND removed=0 AND reason='' AND label IN ('self','other') GROUP BY label",
                (character,),
            ):
                counts[row["label"]] = row["count"]
            active = self.active(character)
            sources = self.trained_sources(db, active)
            candidates = db.execute("SELECT i.*,g.split FROM images i JOIN groups g ON g.id=i.group_id WHERE i.character=? AND i.removed=0 AND i.reason='' AND i.label IN ('self','other') AND (g.split='train' OR g.split IS NULL)", (character,)).fetchall()
        index = SourceIndex(sources)
        pending = sum(not index.seen(r) for r in candidates)
        smallest = min(counts.values())
        # 档位来自受控实验（docs/experiments-sample-size.md）：20–49 组正例时
        # 0.60 阈值下的精确率只有 0.52–0.64，即约三分之一到一半的提示是错的。
        level = (
            "insufficient" if smallest < 20
            else "thin" if smallest < 50
            else "usable" if smallest < 100
            else "good"
        )
        advice = {
            "insufficient": "每类至少 20 个可训练独立图组才能开始训练。",
            "thin": "样本偏少：这个量级的误报较多。补到 100 组，并优先补你实际会遇到的图（群聊截图、画师同人），只用官方立绘会在同人图上几乎认不出。",
            "usable": "可用，但误报仍偏高。再补一些群里常见的正例图，比只加正例数量更有效；负例也要覆盖表情包和其他角色。",
            "good": "样本量足够。继续增加仍是净收益，但边际变小。",
        }[level]
        negative_thin = counts["self"] >= 20 and counts["other"] < 2 * counts["self"]
        eval_groups = self.eval_groups(character)
        return {
            "can_train": min(counts.values()) >= 20,
            "has_model": bool(active),
            "pending_count": pending,
            "level": level,
            "advice": advice,
            "negative_thin": negative_thin,
            "eval_groups": eval_groups,
            # Groups, not rows: the training page shows decisions to make.
            "capture_pending": self.capture_pending(character),
            # Only computed when it can answer something: "重建会付出什么代价".
            # Skipped when an eval set already exists (nothing to plan).
            "eval_plan": None if eval_groups else self.eval_plan(character),
            "summary": f"「{character}」可训练独立图组：是 {counts['self']} / 不是 {counts['other']}。每类至少20组，重复变体只算一组。",
            "groups": counts,
        }

    def eval_groups(self, character: str) -> int:
        """How many groups the fixed evaluation set currently holds.

        0.6.0 replaced the per-medium coverage table with this single number: the
        eval set still exists (it is what makes reports comparable), but it is no
        longer sliced by an annotation nobody could apply consistently.
        """
        character = self.validate_character(character)
        with self.db() as db:
            return db.execute(
                "SELECT COUNT(DISTINCT i.group_id) FROM images i JOIN groups g ON g.id=i.group_id "
                "WHERE i.character=? AND i.removed=0 AND i.reason='' AND g.split='eval'",
                (character,),
            ).fetchone()[0]

    def eval_plan(self, character: str) -> dict:
        """What `rebuild_eval` would take, without writing anything.

        Used by the training page so the confirmation can state the real cost
        (how many images and from which split) instead of a vague warning.
        """
        character = self.validate_character(character)
        with self.db() as db:
            rows = [
                dict(r)
                for r in db.execute(
                    "SELECT i.*,g.split FROM images i JOIN groups g ON g.id=i.group_id "
                    "WHERE i.character=? AND i.removed=0 AND i.reason='' AND i.label IN ('self','other') ORDER BY i.id",
                    (character,),
                )
            ]
        if not rows:
            return {"groups": 0, "from_train": 0, "from_test": 0, "from_val": 0, "from_other": 0}
        picked, unisolated = self._pick_eval_groups(rows)
        first: dict = {}
        for row in rows:
            first.setdefault(row["group_id"], row)
        counts = {"from_train": 0, "from_test": 0, "from_val": 0, "from_other": 0}
        for group in picked:
            row = first[group]
            key = {"train": "from_train", "test": "from_test", "val": "from_val"}.get(row["split"], "from_other")
            counts[key] += 1
        return {"groups": len(picked), "unisolated": unisolated, **counts}

    def _ensure_eval(self, db: sqlite3.Connection, character: str) -> dict:
        """Build the fixed evaluation set once, using only unassigned groups.

        Only `split IS NULL` rows participate: an install that already trained on
        0.5.x keeps every historical assignment byte-for-byte, which is what makes
        old reports comparable. Those installs simply get no eval set until an
        admin explicitly rebuilds one.
        """
        exists = db.execute(
            "SELECT 1 FROM groups g JOIN images i ON i.group_id=g.id WHERE i.character=? AND g.split='eval' LIMIT 1",
            (character,),
        ).fetchone()
        if exists:
            return {"built": False, "reason": "exists"}
        rows = [
            dict(r)
            for r in db.execute(
                "SELECT i.*,g.split FROM images i JOIN groups g ON g.id=i.group_id "
                "WHERE i.character=? AND i.removed=0 AND i.reason='' AND i.label IN ('self','other') "
                "AND g.split IS NULL ORDER BY i.id",
                (character,),
            )
        ]
        if not rows:
            return {"built": False, "reason": "nothing_free"}
        picked, unisolated = self._pick_eval_groups(rows)
        for group in picked:
            db.execute("UPDATE groups SET split='eval' WHERE id=?", (group,))
        if picked:
            db.execute(
                "INSERT OR REPLACE INTO settings VALUES (?,?)",
                (f"eval_built:{character}", dump({"groups": len(picked), "created": time.time(), "unisolated": unisolated})),
            )
        return {
            "built": bool(picked),
            "reason": "created" if picked else "nothing_picked",
            "groups": len(picked),
            "unisolated": unisolated,
        }

    @staticmethod
    def _pick_eval_groups(
        rows: list,
        per_label: int = EVAL_PER_LABEL,
        seed: int = 2026,
    ) -> tuple:
        """Pick whole (width, height) batches; a batch never crosses sides.

        Batch identity is a proxy for source: `洛天依4` had 47/151 images at
        1242×1863, and splitting that batch randomly is what made 0.5.8's 82%
        look better than it is. 0.6.0 dropped the medium axis, so the budget is
        per label instead of per bucket; the isolation rule is unchanged.

        A batch is taken whole unless it is the only batch for that label and
        would break the reserve; then it is sampled inside and reported as
        unisolated rather than silently pretending the isolation held.
        """
        rng = random.Random(seed)
        representatives: dict = {}
        for row in rows:
            representatives.setdefault(row["group_id"], row)
        picked: list = []
        unisolated: list = []
        for label in ("self", "other"):
            cohort = [r for r in representatives.values() if r["label"] == label]
            # The reserve is per label: a library can spare images for evaluation
            # only if it keeps the 20 groups that training requires.
            budget = max(0, len(cohort) - EVAL_MIN_TRAIN)
            wanted = min(per_label, budget)
            if wanted <= 0:
                continue
            batches: dict = {}
            for row in cohort:
                batches.setdefault((row["width"], row["height"]), []).append(row)
            # Small batches first: they are the ones that would otherwise be
            # crowded out, and they are the cheapest whole batches to take.
            keys = sorted(batches, key=lambda key: (len(batches[key]), key))
            rng.shuffle(keys)
            keys.sort(key=lambda key: len(batches[key]))
            taken = 0
            for key in keys:
                if taken >= wanted:
                    break
                batch = batches[key]
                if len(batch) <= wanted - taken:
                    picked.extend(row["group_id"] for row in batch)
                    taken += len(batch)
                    continue
                if taken == 0:
                    sample = rng.sample(batch, wanted)
                    picked.extend(row["group_id"] for row in sample)
                    unisolated.append({"label": label, "batch": "×".join(str(part) for part in key), "size": len(batch), "taken": wanted})
                    taken = wanted
        return list(dict.fromkeys(picked)), unisolated

    def rebuild_eval(
        self,
        character: str,
        per_label: int = EVAL_PER_LABEL,
        seed: int | None = None,
    ) -> dict:
        """Explicit admin action: re-pick the eval set from all eligible groups.

        Unlike the automatic build this may take groups that were already assigned,
        so reports before/after are not comparable — the report says so.
        """
        character = self.validate_character(character)
        if type(per_label) is not int or not 5 <= per_label <= 200:
            raise ValueError("每类评估张数必须在 5 到 200 之间")
        with self.db() as db:
            rows = [
                dict(r)
                for r in db.execute(
                    "SELECT i.*,g.split FROM images i JOIN groups g ON g.id=i.group_id "
                    "WHERE i.character=? AND i.removed=0 AND i.reason='' AND i.label IN ('self','other') ORDER BY i.id",
                    (character,),
                )
            ]
        if not rows:
            raise ValueError("该角色还没有可用于评估的图片")
        # Deterministic by default: `eval_plan` promises the user an exact cost in the
        # confirmation dialog, so the pick that follows must be the same pick.
        picked, unisolated = self._pick_eval_groups(rows, per_label, 2026 if seed is None else seed)
        if not picked:
            raise ValueError("可用于评估的图组不足，无法建立评估集")
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            # Release the old eval set first: otherwise a rebuild only ever grows it.
            db.execute(
                "UPDATE groups SET split=NULL WHERE split='eval' AND id IN "
                "(SELECT group_id FROM images WHERE character=?)",
                (character,),
            )
            db.executemany("UPDATE groups SET split=? WHERE id=?", [("eval", group) for group in picked])
            db.execute(
                "INSERT OR REPLACE INTO settings VALUES (?,?)",
                (
                    f"eval_built:{character}",
                    dump({"groups": len(picked), "created": time.time(), "rebuilt": True, "unisolated": unisolated}),
                ),
            )
        return {"groups": len(picked), "rebuilt": True, "unisolated": unisolated}

    def freeze(self, db: sqlite3.Connection, job_id: str, mode: str, device: str, character: str, input_size: int = 320) -> dict:
        character = self.validate_character(character)
        if type(input_size) is not int or input_size not in {224, 320, 448}:
            raise ValueError("图片尺寸必须为224、320或448")
        # The fixed eval set is picked before splitting: it must never be a source
        # of training rows, and it must stay identical across training runs.
        eval_info = self._ensure_eval(db, character)
        rows = [
            dict(r)
            for r in db.execute(
                "SELECT i.*,g.split FROM images i JOIN groups g ON g.id=i.group_id WHERE i.character=? AND i.removed=0 AND i.reason='' AND i.label IN ('self','other') ORDER BY i.id",
                (character,),
            )
        ]
        # A labelled variant can supply the group truth to its unlabelled representative.
        # 0.6.1 之前这里对每个未标注行跑两次查询（N+1），且 split 用
        # `.fetchone()[0]` 直接取值 —— groups 缺行时是 TypeError 而不是
        # 业务上该抛的 ValueError。改成一次查两张表在内存里 JOIN。
        representatives = {r["group_id"]: r for r in rows}
        truth_by_group = {}
        for r in db.execute(
            "SELECT group_id,label FROM images WHERE character=? AND removed=0 AND reason='' "
            "AND label IN ('self','other') ORDER BY id",
            (character,),
        ):
            truth_by_group.setdefault(r["group_id"], r["label"])
        split_by_group = {
            r["id"]: r["split"] for r in db.execute("SELECT id,split FROM groups")
        }
        for r in db.execute(
            "SELECT * FROM images WHERE character=? AND removed=0 AND reason='' AND label IS NULL",
            (character,),
        ):
            truth = truth_by_group.get(r["group_id"])
            if not truth:
                continue
            value = dict(r)
            value["label"] = truth
            split = split_by_group.get(r["group_id"])
            if split is None:
                raise ValueError(f"图片组 {r['group_id']} 缺少划分信息，请重新划分")
            value["split"] = split
            representatives[r["group_id"]] = value
        rows = list(representatives.values())
        rng = random.Random(42)
        for label in ("self", "other"):
            cohort = [r for r in rows if r["label"] == label]
            # Eval rows are never trained, so they must not inflate the targets.
            trainable = [r for r in cohort if r["split"] != "eval"]
            if len(trainable) < 20:
                name = ("是" if label == "self" else "不是") + character
                raise ValueError(f"「{character}」的“{name}”至少需要20个可训练独立图片组，当前{len(trainable)}；请上传或补标该类图片")
            new = sorted((r for r in trainable if not r["split"]), key=lambda r: r["group_id"])
            rng.shuffle(new)
            # Fill cumulative targets while preserving every existing assignment.
            targets = {"test": max(2, round(len(trainable) * 0.15)), "val": max(2, round(len(trainable) * 0.15))}
            targets["train"] = len(trainable) - targets["test"] - targets["val"]
            counts = {s: sum(r["split"] == s for r in trainable) for s in targets}
            for r in new:
                split = max(targets, key=lambda s: targets[s] - counts[s])
                r["split"] = split
                counts[split] += 1
                db.execute("UPDATE groups SET split=? WHERE id=?", (split, r["group_id"]))
            if any(counts[s] < 2 for s in targets):
                raise ValueError(f"{label} 每个划分至少需要2张")
        parent = self.active(character) if mode == "incremental" else ""
        parent_record = db.execute("SELECT sha FROM models WHERE id=?", (parent,)).fetchone() if parent else None
        if parent and parent_record is None:
            raise ValueError("当前启用模型的记录不存在")
        if mode == "incremental" and not parent:
            mode = "fresh"
        inherited_sources = self.trained_sources(db, parent)
        if parent:
            inherited = SourceIndex(inherited_sources)
            rows = [r for r in rows if r["split"] != "train" or not inherited.seen(r)]
            if not any(r["split"] == "train" for r in rows):
                raise ValueError("没有待训练图片，请新增图片、修改标签或选择从头训练")
        learned = [r for r in rows if r["split"] == "train"]
        trained_sources = [{k: r[k] for k in ("id", "label", "sha", "decoded", "group_id")} for r in [*inherited_sources, *learned]]
        held_truth = digest(sorted((r["id"], r["label"]) for r in rows if r["split"] != "train"))
        previous = db.execute(
            "SELECT snapshot_sha FROM jobs WHERE state='complete' ORDER BY created DESC LIMIT 1"
        ).fetchone()
        old_truth = self.setting(f"last_held_truth:{character}")
        return {
            "job_id": job_id,
            "mode": mode,
            "device": device,
            "input_size": input_size,
            "preprocessing": "resize_before_augment",
            "augmentation": "strong",
            "training_policy": "new_only" if parent else "all",
            "trained_sources": trained_sources,
            "seed": 42,
            "parent": parent,
            "parent_sha": parent_record[0] if parent_record else "",
            "character": character,
            "eval_build": eval_info,
            "images": rows,
            "held_truth": held_truth,
            "comparable_to_previous": not old_truth or held_truth == old_truth,
            "historical_test": bool(previous),
            # Evaluation gate only; the online gate is per character (settings).
            "threshold": EVALUATION_THRESHOLD,
            "created": time.time(),
        }

    def report(self, job_id: str) -> dict:
        with self.db() as db:
            known = db.execute(
                "SELECT 1 FROM jobs WHERE id=? UNION ALL SELECT 1 FROM models WHERE id=?", (job_id, job_id)
            ).fetchone()
        if known is None:
            raise ValueError("任务不存在")
        for base in ("jobs", "models"):
            path = self.root / base / job_id / "report.json"
            try:
                text = path.read_text("utf-8")
            except OSError:
                continue
            return json.loads(text)
        return {}

    def job_or_model(self, job_id: str) -> dict:
        """The live job row, or the model row once the task record was cleaned up."""
        with self.db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is not None:
                return dict(row)
            row = db.execute("SELECT * FROM models WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("任务不存在")
        return {**dict(row), "state": "complete"}

    def job(self, job_id: str) -> dict:
        with self.db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("任务不存在")
        return dict(row)

    def checkpoint_sha(self, job_id: str) -> str | None:
        """服务端持有的断点哈希，与被校验的 resume.pt 分处两个文件。

        0.6.1 之前这个值写在 resume.pt 同目录的 resume-audit.json 里，由同一个
        worker 紧跟着 torch.save 写入 —— 同目录同写者，伪造一对即可通过校验。
        改存数据库后，「能写 resume.pt」不再等于「能让校验通过」。
        """
        with self.db() as db:
            row = db.execute("SELECT sha FROM job_checkpoints WHERE job_id=?", (job_id,)).fetchone()
        return row["sha"] if row else None

    def set_checkpoint_sha(self, job_id: str, sha: str) -> None:
        with self.db() as db:
            db.execute(
                "INSERT INTO job_checkpoints(job_id,sha) VALUES (?,?) "
                "ON CONFLICT(job_id) DO UPDATE SET sha=excluded.sha",
                (job_id, sha),
            )

    def forget_checkpoint_sha(self, job_id: str) -> None:
        with self.db() as db:
            db.execute("DELETE FROM job_checkpoints WHERE job_id=?", (job_id,))

    def backfill_job_meta(self, rows: list, meta) -> None:
        """Persist device/input_size for rows written before 0.4.0.

        `meta(job_id)` returns ``(device, input_size)`` or ``None``. The UPDATE is
        limited to rows that are still empty so concurrent polls cannot clash.
        """
        updates = []
        for row in rows:
            if row.get("device"):
                continue
            found = meta(row["id"])
            if found is None:
                continue
            row["device"], row["input_size"] = found
            updates.append((found[0], found[1], row["id"]))
        if not updates:
            return
        with self.db() as db:
            db.executemany("UPDATE jobs SET device=?,input_size=? WHERE id=? AND device=''", updates)

    def models(self) -> list:
        with self.db() as db:
            return [dict(r) for r in db.execute("SELECT * FROM models ORDER BY created DESC")]

    def model_size(self, version: str) -> int:
        """Bytes the version occupies, including reports archived next to it."""
        total = 0
        for item in (self.root / "models" / version).rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                continue
        return total
