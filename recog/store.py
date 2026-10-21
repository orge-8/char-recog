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

from .images import decode, fingerprint, similar, thumbnail
from .scores import (
    DEFAULT_THRESHOLD,
    EVALUATION_THRESHOLD,
    MAX_THRESHOLD,
    MIN_THRESHOLD,
    NEG_LABELS,
    STYLE_LABELS,
    STYLE_MIN_COUNT,
    TEXT_LABELS,
)

# 内容类型（媒材）单轴。只有这一轴有实测效应：同人厚涂插画召回 2%，平涂 Q 版贴纸 91%。
# 名称刻意与「官方/二创」无关 —— 那对差别不是召回差异的来源（docs/p0-change-list.md）。
STYLES = STYLE_LABELS
NEG_TYPES = NEG_LABELS
# 文字叠加：与三档正交的标记，不是第四种画法。
TEXT_MARKS = TEXT_LABELS
# 固定评估集：每档默认 25 张。低于 5 张不给百分比（小样本不给点估计）。
EVAL_PER_STYLE = 25
EVAL_PER_NEG_KIND = 15
EVAL_MIN_COUNT = STYLE_MIN_COUNT
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
              CREATE TABLE IF NOT EXISTS images(id TEXT PRIMARY KEY, path TEXT, sha TEXT, decoded TEXT, phash TEXT, aspect REAL, width INTEGER, height INTEGER, group_id TEXT, label TEXT, removed INTEGER DEFAULT 0, reason TEXT DEFAULT '', created REAL, character TEXT NOT NULL DEFAULT '', style TEXT NOT NULL DEFAULT '', neg_type TEXT NOT NULL DEFAULT '', has_text TEXT NOT NULL DEFAULT '');
              CREATE TABLE IF NOT EXISTS groups(id TEXT PRIMARY KEY, split TEXT);
              CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, request_key TEXT UNIQUE, state TEXT, pid INTEGER, error TEXT, created REAL, snapshot_sha TEXT, character TEXT NOT NULL DEFAULT '', device TEXT NOT NULL DEFAULT '', input_size INTEGER);
              CREATE UNIQUE INDEX IF NOT EXISTS one_training ON jobs((1)) WHERE state IN ('queued','running','cancelling');
              CREATE UNIQUE INDEX IF NOT EXISTS one_training_v2 ON jobs((1)) WHERE state IN ('queued','running','pausing','cancelling');
              CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY, job_id TEXT, sha TEXT, metrics TEXT, created REAL, character TEXT NOT NULL DEFAULT '');
              CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT);
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
            # 0.5.12: content type (medium) and negative bucket. Additive only —
            # every existing row keeps style='' and neg_type='', which reproduces
            # the 0.5.x behaviour exactly (one '' stratum, no eval set).
            columns = {row[1] for row in db.execute("PRAGMA table_info(images)")}
            if "style" not in columns:
                db.execute("ALTER TABLE images ADD COLUMN style TEXT NOT NULL DEFAULT ''")
            if "neg_type" not in columns:
                db.execute("ALTER TABLE images ADD COLUMN neg_type TEXT NOT NULL DEFAULT ''")
            if "has_text" not in columns:
                db.execute("ALTER TABLE images ADD COLUMN has_text TEXT NOT NULL DEFAULT ''")
            # 0.5.18: 「图文/梗图」was a fourth medium that mixed画法 with叠加层. The
            # information it carried (text on the image) moves to has_text; the
            # medium stays empty because a meme is usually平涂 but we cannot tell.
            moved = db.execute("SELECT COUNT(*) FROM images WHERE style='composite'").fetchone()[0]
            if moved:
                db.execute(
                    "UPDATE images SET has_text=CASE WHEN has_text='' THEN 'yes' ELSE has_text END, "
                    "style='' WHERE style='composite'"
                )
                db.execute(
                    "INSERT OR REPLACE INTO settings VALUES (?,?)",
                    ("composite_migrated", dump({"moved": moved, "at": time.time()})),
                )
            # 0.5.17: 「表情包」left the negative bucket because it is a medium, not an
            # identity. Leaving the old values in place would make them invisible
            # everywhere (no label, excluded from eval sampling) instead of merely
            # wrong, so they are cleared once and the count is kept for diagnosis.
            stale = db.execute(
                "SELECT COUNT(*) FROM images WHERE neg_type NOT IN ('', 'other_character', 'non_character')"
            ).fetchone()[0]
            if stale:
                db.execute(
                    "UPDATE images SET neg_type='' WHERE neg_type NOT IN ('', 'other_character', 'non_character')"
                )
                db.execute(
                    "INSERT OR REPLACE INTO settings VALUES (?,?)",
                    ("neg_type_reset", dump({"cleared": stale, "at": time.time()})),
                )

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

    @staticmethod
    def validate_character(character: str) -> str:
        if not isinstance(character, str) or not character.strip():
            raise ValueError("缺少角色名")
        character = character.strip()
        if len(character) > 32:
            raise ValueError("角色名过长（最多32字）")
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

    def ingest(
        self,
        path: Path,
        label: str | None = None,
        character: str = "",
        style: str = "",
        neg_type: str = "",
        has_text: str = "",
    ) -> dict:
        if label not in {None, "self", "other"}:
            raise ValueError("上传身份必须为是Bot或不是Bot")
        style = (style or "").strip()
        neg_type = (neg_type or "").strip()
        has_text = (has_text or "").strip()
        if style and style not in STYLES:
            raise ValueError("内容类型必须为平涂/Q版、厚涂插画或实拍")
        if has_text and has_text not in TEXT_MARKS:
            raise ValueError("文字叠加必须为有文字或无文字")
        if neg_type and neg_type not in NEG_TYPES:
            raise ValueError("负例类型必须为其他角色或非角色内容")
        if neg_type and label != "other":
            raise ValueError("只有「不是该角色」的图片才能选负例类型")
        character = self.validate_character(character)
        path = path.resolve()
        if not path.is_relative_to(self.root / "uploads") or path.is_symlink():
            raise ValueError("只能领取本插件上传文件")
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        image = decode(raw)
        fp = fingerprint(image)
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
                "INSERT INTO images(id,path,sha,decoded,phash,aspect,width,height,group_id,label,removed,reason,created,character,style,neg_type,has_text) VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?,?,?,?,?)",
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
                    style,
                    neg_type,
                    has_text,
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
    def validate_style(style: str) -> str:
        style = (style or "").strip()
        if style and style not in STYLES:
            raise ValueError("内容类型必须为平涂/Q版、厚涂插画、实拍或图文/梗图")
        return style

    @staticmethod
    def validate_has_text(has_text: str) -> str:
        has_text = (has_text or "").strip()
        if has_text and has_text not in TEXT_MARKS:
            raise ValueError("文字叠加必须为有文字或无文字")
        return has_text

    @staticmethod
    def validate_neg_type(neg_type: str) -> str:
        neg_type = (neg_type or "").strip()
        if neg_type and neg_type not in NEG_TYPES:
            raise ValueError("负例类型必须为其他角色、非角色内容或表情包")
        return neg_type

    def set_style(
        self, image_id: str, style: str = "", neg_type: str | None = None, has_text: str | None = None
    ) -> dict:
        """Tag one image. `None` leaves that field untouched, "" clears it."""
        style = self.validate_style(style)
        if neg_type is not None:
            neg_type = self.validate_neg_type(neg_type)
        if has_text is not None:
            has_text = self.validate_has_text(has_text)
        with self.db() as db:
            row = db.execute("SELECT label FROM images WHERE id=?", (image_id,)).fetchone()
            if row is None:
                raise ValueError("图片不存在")
            if neg_type and row["label"] != "other":
                raise ValueError("只有「不是该角色」的图片才能选负例类型")
            fields, values = ["style=?"], [style]
            if neg_type is not None:
                fields.append("neg_type=?")
                values.append(neg_type)
            if has_text is not None:
                fields.append("has_text=?")
                values.append(has_text)
            db.execute(f"UPDATE images SET {', '.join(fields)} WHERE id=?", (*values, image_id))
        return {"id": image_id, "style": style, "neg_type": neg_type, "has_text": has_text}

    def batch_style(
        self, image_ids, style: str | None = None, neg_type: str | None = None, has_text: str | None = None
    ) -> dict:
        """Tag a whole selection in one transaction.

        ``None`` means "leave unchanged" and ``""`` means "clear", so the page can
        set a medium without silently wiping the negative bucket and vice versa.
        """
        if style is None and neg_type is None and has_text is None:
            raise ValueError("请选择要设置的内容类型、负例类型或文字叠加")
        if style is not None:
            style = self.validate_style(style)
        if neg_type is not None:
            neg_type = self.validate_neg_type(neg_type)
        if has_text is not None:
            has_text = self.validate_has_text(has_text)
        ids = self.split_ids(image_ids)
        if not ids or len(ids) > 1000:
            raise ValueError("请先选择图片，每批最多1000张")
        if any(len(i) != 32 or any(c not in "0123456789abcdef" for c in i) for i in ids):
            raise ValueError("无效图片ID")
        placeholders = ",".join("?" for _ in ids)
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(f"SELECT id,label FROM images WHERE id IN ({placeholders})", ids).fetchall()
            if len(rows) != len(ids):
                raise ValueError("选择中有不存在的图片，请清空选择后重新选择")
            if neg_type and any(row["label"] != "other" for row in rows):
                raise ValueError("选择中包含「是该角色」的图片，不能设负例类型")
            fields, values = [], []
            if style is not None:
                fields.append("style=?")
                values.append(style)
            if neg_type is not None:
                fields.append("neg_type=?")
                values.append(neg_type)
            if has_text is not None:
                fields.append("has_text=?")
                values.append(has_text)
            db.execute(f"UPDATE images SET {', '.join(fields)} WHERE id IN ({placeholders})", [*values, *ids])
        return {"updated": len(ids), "style": style, "neg_type": neg_type, "has_text": has_text}

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
        snapshot = json.loads((self.root / "jobs" / record[0] / "snapshot.json").read_text("utf-8"))
        return snapshot.get("trained_sources", [r for r in snapshot["images"] if r["split"] == "train"])

    @staticmethod
    def source_seen(row, sources) -> bool:
        """Compatibility wrapper; hot paths build a SourceIndex instead."""
        return SourceIndex(sources).seen(row)

    def images(self, page: int = 1, view: str = "all", job_id: str = "", character: str = "") -> dict:
        character = self.validate_character(character)
        threshold = self.threshold(character)
        if page < 1 or view not in {"all", "eligible", "excluded", "unlabelled", "errors", "removed"}:
            raise ValueError("无效分页")
        where = {
            "all": "i.removed=0",
            "eligible": "i.removed=0 AND i.reason='' AND i.label IN ('self','other')",
            "excluded": "(i.removed=1 OR i.reason!='' OR i.label='skip')",
            "unlabelled": "i.label IS NULL",
            "errors": "1",
            "removed": "i.removed=1",
        }[view] + " AND i.character=?"
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
        parameters: list = [character]
        if view == "errors":
            if error_ids:
                where += " AND i.id IN (" + ",".join("?" for _ in error_ids) + ")"
                parameters = [character, *sorted(error_ids)]
            else:
                where = "0"
                parameters = []
        with self.db() as db:
            sources = self.trained_sources(db, self.active(character))
            total = db.execute(f"SELECT COUNT(*) FROM images i WHERE {where}", parameters).fetchone()[0]
            page = min(page, max(1, (total + 23) // 24))
            rows = [dict(r) for r in db.execute(
                f"SELECT i.*,g.split FROM images i JOIN groups g ON g.id=i.group_id WHERE {where} ORDER BY i.created DESC LIMIT 24 OFFSET ?",
                [*parameters, (page - 1) * 24],
            )]
        index = SourceIndex(sources)
        gallery = []
        for r in rows:
            reasons = {"solid": "纯色/近纯色", "duplicate": "同图变体，仅保留一个代表", "label_conflict": "同图身份标签冲突", "split_conflict": "连接了不同历史划分"}
            r["status"] = "已移出训练库" if r["removed"] else reasons.get(r["reason"], "跳过" if r["label"] == "skip" else "可用")
            r["label_text"] = {"self": f"是{character}", "other": f"不是{character}", "skip": "跳过"}.get(r["label"], "未标注")
            r["style_text"] = STYLES.get(r.get("style") or "", "未标注类型")
            r["neg_type_text"] = NEG_TYPES.get(r.get("neg_type") or "", "")
            r["has_text_text"] = TEXT_MARKS.get(r.get("has_text") or "", "")
            r["trained"] = index.seen(r)
            r["training_text"] = "当前模型已训练" if r["trained"] else "评估集" if r["split"] == "eval" else "验证集" if r["split"] == "val" else "测试集" if r["split"] == "test" else "待训练"
            prediction = predictions.get(r["id"])
            if prediction:
                predicted = "self" if prediction["score"] >= threshold else "other"
                r["model_text"] = (
                    f"模型：{'是' + character if predicted == 'self' else '不是' + character}"
                    f" · 得分 {prediction['score']:.3f}（阈值 {threshold:.2f}）"
                    f" · 冻结真值：{'是' + character if prediction['label'] == 'self' else '不是' + character}"
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
        for row, preview in zip(rows, gallery):
            row["thumbnail"] = preview["thumbnail"]
        return {
            "rows": rows,
            "pages": max(1, (total + 23) // 24),
            "total": total,
            "pagination": {"page": page, "pages": max(1, (total + 23) // 24), "total": total},
        }

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
        coverage = self.style_coverage(character)
        missing = [row["label"] for row in coverage if row["verdict"] == "missing"]
        eval_groups = sum(row["eval"] for row in coverage)
        labelled = sum(row["train"] + row["eval"] + row["other"] for row in coverage if row["style"])
        coverage_text = " · ".join(f"{row['label']} {row['train']}" for row in coverage if row["style"])
        return {
            "can_train": min(counts.values()) >= 20,
            "has_model": bool(active),
            "pending_count": pending,
            "level": level,
            "advice": advice,
            "negative_thin": negative_thin,
            "coverage": coverage,
            "coverage_text": coverage_text,
            "coverage_missing": missing,
            "eval_groups": eval_groups,
            # Only computed when it can answer something: "重建会付出什么代价".
            # Skipped when an eval set exists or nothing is tagged (no plan possible).
            "eval_plan": None if eval_groups or not labelled else self.eval_plan(character),
            "summary": f"「{character}」可训练独立图组：是 {counts['self']} / 不是 {counts['other']}。每类至少20组，重复变体只算一组。",
            "groups": counts,
        }

    def style_coverage(self, character: str) -> list:
        """训练集/评估集各档的实际组数。空档返回 verdict=missing，永远可输出。"""
        character = self.validate_character(character)
        with self.db() as db:
            rows = db.execute(
                "SELECT i.style,g.split,COUNT(DISTINCT i.group_id) AS count FROM images i "
                "JOIN groups g ON g.id=i.group_id WHERE i.character=? AND i.removed=0 AND i.reason='' "
                "AND i.label='self' GROUP BY i.style,g.split",
                (character,),
            ).fetchall()
        counts: dict = {}
        for row in rows:
            bucket = counts.setdefault(row["style"] or "", {"train": 0, "eval": 0, "other": 0})
            if row["split"] == "eval":
                bucket["eval"] += row["count"]
            elif row["split"] in ("train", None, ""):
                # Unassigned rows are trainable, not "other": coverage is asked
                # before the first freeze as often as after it.
                bucket["train"] += row["count"]
            else:
                bucket["other"] += row["count"]
        result = []
        for style, label in STYLES.items():
            bucket = counts.get(style, {"train": 0, "eval": 0, "other": 0})
            verdict = "missing" if bucket["train"] + bucket["eval"] == 0 else (
                "eval_only" if bucket["train"] == 0 else "covered"
            )
            result.append(
                {
                    "style": style,
                    "label": label,
                    "train": bucket["train"],
                    "eval": bucket["eval"],
                    "other": bucket["other"],
                    "verdict": verdict,
                }
            )
        untagged = counts.get("", {"train": 0, "eval": 0, "other": 0})
        result.append(
            {
                "style": "",
                "label": "未标注",
                "train": untagged["train"],
                "eval": untagged["eval"],
                "other": untagged["other"],
                "verdict": "untagged" if sum(untagged.values()) else "empty",
            }
        )
        return result

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
            return {"groups": 0, "by_style": {}, "by_neg": {}, "from_train": 0, "from_test": 0,
                    "from_val": 0, "from_other": 0}
        picked, unisolated = self._pick_eval_groups(rows)
        first: dict = {}
        for row in rows:
            first.setdefault(row["group_id"], row)
        by_style: dict = {}
        by_neg: dict = {}
        counts = {"from_train": 0, "from_test": 0, "from_val": 0, "from_other": 0}
        for group in picked:
            row = first[group]
            if row["label"] == "self" and row["style"]:
                by_style[row["style"]] = by_style.get(row["style"], 0) + 1
            elif row["label"] == "other" and row["neg_type"]:
                by_neg[row["neg_type"]] = by_neg.get(row["neg_type"], 0) + 1
            key = {"train": "from_train", "test": "from_test", "val": "from_val"}.get(row["split"], "from_other")
            counts[key] += 1
        return {"groups": len(picked), "by_style": by_style, "by_neg": by_neg, "unisolated": unisolated, **counts}

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
            "reason": "created" if picked else "no_style",
            "groups": len(picked),
            "unisolated": unisolated,
        }

    @staticmethod
    def _pick_eval_groups(
        rows: list,
        per_style: int = EVAL_PER_STYLE,
        neg_per_kind: int = EVAL_PER_NEG_KIND,
        seed: int = 2026,
    ) -> tuple:
        """Pick whole (style, width, height) batches; a batch never crosses sides.

        Batch identity is a proxy for source: `洛天依4` had 47/151 images at
        1242×1863, and splitting that batch randomly is what made 0.5.8's 82%
        look better than it is. The budget is per medium, but the reserve is per
        label — a library can spare 25 images for evaluation only if it keeps the
        20 groups that training requires.

        A batch is taken whole unless it is the only batch for that medium and
        would break the reserve; then it is sampled inside and reported as
        unisolated rather than silently pretending the isolation held.
        """
        rng = random.Random(seed)
        representatives: dict = {}
        for row in rows:
            representatives.setdefault(row["group_id"], row)
        picked: list = []
        unisolated: list = []
        for label, field, wanted, values, allow_blank in (
            ("self", "style", per_style, tuple(STYLES), False),
            # "" is a real bucket for negatives: a negative whose identity was never
            # bucketed still has to be evaluated, otherwise the FPR column is empty
            # for every user who only tagged the medium.
            ("other", "neg_type", neg_per_kind, (*NEG_TYPES, ""), True),
        ):
            cohort = [
                r
                for r in representatives.values()
                if r["label"] == label and (allow_blank or r[field])
            ]
            budget = max(0, len(cohort) - EVAL_MIN_TRAIN)
            taken = 0
            # Rarest medium first: a 3-image bucket must not be crowded out by the
            # dominant one, because it is exactly the bucket most likely to be the
            # weak spot the report is supposed to find.
            order = sorted(values, key=lambda value: sum(1 for r in cohort if r[field] == value))
            for value in order:
                if budget - taken <= 0:
                    break
                subset = [r for r in cohort if r[field] == value]
                if not subset:
                    continue
                style_budget = min(wanted, budget - taken)
                batches: dict = {}
                for row in subset:
                    batches.setdefault((row["width"], row["height"]), []).append(row)
                keys = sorted(batches, key=lambda key: (len(batches[key]), key))
                rng.shuffle(keys)
                keys.sort(key=lambda key: len(batches[key]))
                style_taken = 0
                for key in keys:
                    if style_taken >= style_budget:
                        break
                    batch = batches[key]
                    if len(batch) <= style_budget - style_taken:
                        picked.extend(row["group_id"] for row in batch)
                        style_taken += len(batch)
                        continue
                    if style_taken == 0:
                        sample = rng.sample(batch, style_budget)
                        picked.extend(row["group_id"] for row in sample)
                        unisolated.append(
                            {"label": label, "style": value, "batch": len(batch), "taken": style_budget}
                        )
                        style_taken = style_budget
                taken += style_taken
        return list(dict.fromkeys(picked)), unisolated

    def rebuild_eval(
        self,
        character: str,
        per_style: int = EVAL_PER_STYLE,
        neg_per_kind: int = EVAL_PER_NEG_KIND,
        seed: int | None = None,
    ) -> dict:
        """Explicit admin action: re-pick the eval set from all eligible groups.

        Unlike the automatic build this may take groups that were already assigned,
        so reports before/after are not comparable — the report says so.
        """
        character = self.validate_character(character)
        if type(per_style) is not int or not 5 <= per_style <= 200:
            raise ValueError("每档评估张数必须在 5 到 200 之间")
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
        picked, unisolated = self._pick_eval_groups(rows, per_style, neg_per_kind, 2026 if seed is None else seed)
        if not picked:
            raise ValueError("没有已标注内容类型的图片，无法建立评估集")
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
        representatives = {r["group_id"]: r for r in rows}
        for r in db.execute("SELECT * FROM images WHERE character=? AND removed=0 AND reason='' AND label IS NULL", (character,)):
            truth = db.execute(
                "SELECT label FROM images WHERE group_id=? AND label IN ('self','other') AND removed=0 LIMIT 1",
                (r["group_id"],),
            ).fetchone()
            if truth:
                value = dict(r)
                value["label"] = truth[0]
                value["split"] = db.execute("SELECT split FROM groups WHERE id=?", (r["group_id"],)).fetchone()[0]
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
        self.job(job_id)  # reject arbitrary paths
        path = self.root / "jobs" / job_id / "report.json"
        return json.loads(path.read_text("utf-8")) if path.exists() else {}

    def job(self, job_id: str) -> dict:
        with self.db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("任务不存在")
        return dict(row)

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
