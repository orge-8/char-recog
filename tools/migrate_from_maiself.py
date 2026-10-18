"""Migrate MaiSelfPlugin (maibot.mai-recog-self) data into char-recog.

Source: the upstream plugin's data directory, e.g.
  MaiBot/data/plugins/maibot.mai-recog-self/
    catalog.sqlite3  images/  jobs/<job>/  models/<job>/weights.pt

Target: this plugin's data directory (org.civetc.char-recog). The whole
source catalog is attributed to ONE character (--character): "self" means
"is that character", "other" means "is not". Train/val/test splits, labels,
removal reasons and the trained model lineage are preserved, so incremental
training continues from the migrated model.

Usage (run from the plugin directory):
  python tools/migrate_from_maiself.py \
      --src  ../maibot.mai-recog-self \
      --dst  ../org.civetc.char-recog \
      --character 洛天依 [--dry-run]

The target may already contain data for other characters; migration skips
source images whose sha already exists under the same character.
"""

from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recog.store import Store, atomic_json, digest  # noqa: E402

IMAGE_KEYS = ("id", "path", "sha", "decoded", "phash", "aspect", "width", "height", "group_id", "label", "removed", "reason", "created")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def migrate(source: Path, target: Path, character: str, dry_run: bool = False) -> None:
    character = Store.validate_character(character)
    source_catalog = source / "catalog.sqlite3"
    if not source_catalog.is_file():
        raise SystemExit(f"源目录缺少 catalog.sqlite3：{source_catalog}")
    if (target / "catalog.sqlite3").resolve() == source_catalog.resolve():
        raise SystemExit("源与目标是同一个目录")
    store = Store(target)

    src = sqlite3.connect(source_catalog)
    src.row_factory = sqlite3.Row
    try:
        settings = {row["key"]: row["value"] for row in src.execute("SELECT key,value FROM settings")}
        jobs = {row["id"]: dict(row) for row in src.execute("SELECT * FROM jobs")}
        models = [dict(r) for r in src.execute("SELECT * FROM models ORDER BY created")]
        groups = [(r["id"], r["split"]) for r in src.execute("SELECT id,split FROM groups")]
        images = [dict(r) for r in src.execute("SELECT * FROM images ORDER BY created,id")]
    finally:
        src.close()

    complete_models = [m for m in models if jobs.get(m["job_id"], {}).get("state") == "complete"]
    print(f"源数据：图片 {len(images)} 张 · 图组 {len(groups)} 个 · 完成模型 {len(complete_models)} 个")
    existing = set()
    with store.db() as db:
        for row in db.execute("SELECT sha FROM images WHERE character=?", (character,)):
            existing.add(row[0])

    plan_images = [r for r in images if r["sha"] not in existing]
    print(f"迁移计划：图片 {len(plan_images)} 张（跳过目标已有 {len(images) - len(plan_images)} 张）· 角色「{character}」")
    if dry_run:
        print("dry-run 结束，未写入任何数据。")
        return

    # 1. Image files + rows, groups (splits preserved).
    copied = 0
    for row in plan_images:
        source_file = source / row["path"]
        if not source_file.is_file():
            print(f"  跳过缺失文件：{row['path']}")
            continue
        destination = target / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source_file, destination)
            copied += 1
    print(f"图片文件复制：{copied} 个")

    with store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        for group, split in groups:
            db.execute("INSERT OR IGNORE INTO groups VALUES (?,?)", (group, split))
        for row in plan_images:
            values = [row[key] for key in IMAGE_KEYS] + [character]
            placeholders = ",".join("?" for _ in values)
            db.execute(f"INSERT OR IGNORE INTO images({','.join(IMAGE_KEYS)},character) VALUES ({placeholders})", values)

        # 2. Completed models: weights + audit, models rows, jobs rows, patched snapshots.
        for record in complete_models:
            job = record["job_id"]
            source_model_dir = source / "models" / job
            source_job_dir = source / "jobs" / job
            weights = source_model_dir / "weights.pt"
            if not weights.is_file():
                print(f"  跳过缺权重模型：{job}")
                continue
            target_model_dir = target / "models" / job
            target_model_dir.mkdir(parents=True, exist_ok=True)
            target_weights = target_model_dir / "weights.pt"
            if not target_weights.exists():
                shutil.copy2(weights, target_weights)
            for name in ("audit.json",):
                if (source_model_dir / name).is_file():
                    shutil.copy2(source_model_dir / name, target_model_dir / name)
            actual_sha = sha256_file(target_weights)
            if actual_sha != record["sha"]:
                print(f"  警告：模型 {job} 权重 sha 不符（记录 {record['sha'][:8]}，实际 {actual_sha[:8]}），按实际值登记")

            target_job_dir = target / "jobs" / job
            target_job_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = source_job_dir / "snapshot.json"
            snapshot = json.loads(snapshot_path.read_text("utf-8")) if snapshot_path.is_file() else {}
            snapshot["character"] = character
            atomic_json(target_job_dir / "snapshot.json", snapshot)
            snapshot_sha = digest(snapshot)
            for name in ("report.json", "selection.json", "progress.json"):
                if (source_job_dir / name).is_file():
                    shutil.copy2(source_job_dir / name, target_job_dir / name)

            metrics = record["metrics"] or "{}"
            created = record["created"] or jobs.get(job, {}).get("created", 0.0)
            db.execute(
                "INSERT OR REPLACE INTO models VALUES (?,?,?,?,?,?)",
                (job, job, actual_sha, metrics, created, character),
            )
            db.execute(
                "INSERT OR REPLACE INTO jobs "
                "(id,request_key,state,pid,error,created,snapshot_sha,character,device,input_size) "
                "VALUES (?,?,COALESCE((SELECT state FROM jobs WHERE id=?),'complete'),NULL,'',?,?,?,?,?)",
                (
                    job,
                    f"migrated-{job[:12]}",
                    job,
                    created,
                    snapshot_sha,
                    character,
                    (snapshot or {}).get("device", ""),
                    (snapshot or {}).get("input_size", 320),
                ),
            )

        # 3. Settings: active / previous / held-truth per character.
        active = settings.get("active", "")
        previous = settings.get("previous", "")
        if active and any(m["job_id"] == active for m in complete_models):
            db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (f"active:{character}", active))
        if previous and any(m["job_id"] == previous for m in complete_models):
            db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (f"previous:{character}", previous))
        held = settings.get("last_held_truth", "")
        if held:
            db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (f"last_held_truth:{character}", held))

    print("迁移完成。请在 WebUI 重新加载插件后检查：模型版本页应显示迁移的模型为「使用中」。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, type=Path, help="MaiSelfPlugin 数据目录（含 catalog.sqlite3）")
    parser.add_argument("--dst", required=True, type=Path, help="char-recog 数据目录")
    parser.add_argument("--character", required=True, help="源图库整体归属的角色名（必须显式指定）")
    parser.add_argument("--dry-run", action="store_true", help="只打印迁移计划，不写入")
    args = parser.parse_args()
    migrate(args.src, args.dst, args.character, args.dry_run)


if __name__ == "__main__":
    main()
