"""Dedicated train / persistent CPU inference processes, stdio JSON protocol."""

from pathlib import Path
import argparse
import base64
import hashlib
import json
import os
import sys
import time
import traceback

from .store import Store, atomic_json, digest
from .images import MAX_EDGE, decode
from .scores import DEFAULT_THRESHOLD, judgment


def train_job(root: Path, job_id: str) -> None:
    store = Store(root)
    store.job(job_id)
    directory = root / "jobs" / job_id
    snapshot = json.loads((directory / "snapshot.json").read_text("utf-8"))
    if digest(snapshot) != store.job(job_id)["snapshot_sha"]:
        raise ValueError("快照SHA不符")
    from .model import train, TrainingPaused

    def progress(value):
        atomic_json(directory / "progress.json", value)

    def cancelled():
        lease = directory / "lease"
        return (directory / "cancel").exists() or not lease.exists() or time.time() - lease.stat().st_mtime > 120

    try:
        with store.db() as db:
            db.execute(
                "UPDATE jobs SET state=CASE WHEN state='queued' THEN 'running' ELSE state END,pid=?,"
                " device=COALESCE(NULLIF(device,''),?), input_size=COALESCE(input_size,?) WHERE id=?",
                (os.getpid(), snapshot.get("device", ""), snapshot.get("input_size", 320), job_id),
            )
        if cancelled():
            raise InterruptedError("训练已取消")
        progress({"phase": "preparing"})
        report = train(root, snapshot, progress, cancelled, paused=lambda: (directory / "pause").exists())
        report["total_job_seconds"] = time.time() - snapshot["created"]
        if cancelled():
            raise InterruptedError("训练已取消")
        destination = root / "models" / job_id
        destination.mkdir(exist_ok=True)
        (directory / "selected.pt").replace(destination / "weights.pt")
        sha = hashlib.sha256((destination / "weights.pt").read_bytes()).hexdigest()
        atomic_json(directory / "report.json", report)
        atomic_json(destination / "audit.json", {**report["audit"], "sha256": sha, "snapshot_sha": digest(snapshot)})
        with store.db() as db:
            db.execute(
                "INSERT INTO models VALUES (?,?,?,?,?,?)",
                (
                    job_id,
                    job_id,
                    sha,
                    json.dumps({"validation": report["selection"]["validation"], "test": report["test"], "input_size": report["input_size"]}),
                    time.time(),
                    snapshot["character"],
                ),
            )
            db.execute("UPDATE jobs SET state='complete' WHERE id=?", (job_id,))
            db.execute(
                "INSERT OR REPLACE INTO settings VALUES (?,?)",
                (f"last_held_truth:{snapshot['character']}", snapshot["held_truth"]),
            )
        progress(
            {"phase": "complete", "selected_epoch": report["selection"]["epoch"], "seconds": report["total_seconds"]}
        )
        (directory / "resume.pt").unlink(missing_ok=True)
        (directory / "resume-audit.json").unlink(missing_ok=True)
    except TrainingPaused:
        with store.db() as db:
            db.execute("UPDATE jobs SET state='paused',pid=NULL WHERE id=?", (job_id,))
        value = json.loads((directory / "progress.json").read_text("utf-8"))
        progress({**value, "phase": "paused"})
    except BaseException as exc:
        state = "cancelled" if isinstance(exc, InterruptedError) else "failed"
        with store.db() as db:
            db.execute("UPDATE jobs SET state=?,error=? WHERE id=?", (state, str(exc), job_id))
        progress({"phase": state, "error": str(exc)})
        traceback.print_exc(file=sys.stderr)
        raise


def serve(root: Path) -> None:
    import torch
    from .model import loaded_model, score_tensor, tensor_for_image

    default_threads = min(4, torch.get_num_threads())
    torch.set_num_threads(default_threads)
    current_threads = default_threads
    store = Store(root)
    loaded = {}  # character -> (version, model)

    def load(character: str, version: str):
        record = next((r for r in store.models() if r["id"] == version), None)
        if record is None:
            raise ValueError("模型版本不存在")
        path = root / "models" / version / "weights.pt"
        audit = json.loads((path.parent / "audit.json").read_text("utf-8"))
        if not audit["passed"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha"]:
            raise ValueError("模型核验失败")
        model = loaded_model(path)  # assign only after successful load
        loaded[character] = (version, model)
        return model

    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request.get("operation")
            if operation == "ping":
                # Readiness handshake; the host uses it to bound cold start.
                result = {"status": "ready", "threads": current_threads, "default_threads": default_threads}
            elif operation == "threads":
                # Training is CPU hungry: lend threads back while it runs.
                value = int(request.get("value") or 0)
                target = default_threads if value <= 0 else max(1, min(default_threads, value))
                if target != current_threads:
                    torch.set_num_threads(target)
                    current_threads = target
                result = {"status": "ok", "threads": current_threads}
            elif operation == "load":
                character = Store.validate_character(request.get("character", ""))
                version = request.get("version") or store.active(character)
                if not version:
                    result = {"status": "no_model"}
                else:
                    model = load(character, version)
                    result = {
                        "status": "ready",
                        "character": character,
                        "version": version,
                        "input_size": model.input_size,
                    }
            else:
                raw = base64.b64decode(request["image"], validate=True)
                requested = request.get("character", "")
                targets = [Store.validate_character(requested)] if requested else store.active_characters()
                try:
                    # Read the gates per request: an admin change must apply to the
                    # very next image, and a read failure must not kill inference.
                    limits = store.thresholds()
                except Exception:
                    limits = {}
                # Decode once per request, then letterbox once per input size:
                # N characters used to pay N full PIL decodes of the same image.
                # Draft at the largest supported input size: models may mix 224/
                # 320/448, and `draft` returns at least the requested edge.
                image = decode(raw, MAX_EDGE)
                tensors = {}
                results = []
                for character in targets:
                    version = store.active(character)
                    if not version:
                        continue
                    try:
                        if loaded.get(character, ("", None))[0] != version:
                            load(character, version)
                        _, model = loaded[character]
                        size = model.input_size
                        tensor = tensors.get(size)
                        if tensor is None:
                            tensor = tensors[size] = tensor_for_image(image, size)
                        score = score_tensor(model, tensor)
                        limit = limits.get(character, DEFAULT_THRESHOLD)
                        results.append(
                            {
                                "status": "ok",
                                "character": character,
                                "version": version,
                                "input_size": size,
                                "score": score,
                                "threshold": limit,
                                "judgment": judgment(score, limit),
                                "score_is_calibrated_probability": False,
                            }
                        )
                    except Exception as exc:
                        loaded.pop(character, None)
                        results.append({"status": "model_error", "character": character, "error": str(exc)})
                result = {"status": "ok" if results else "no_model", "results": results}
        except Exception as exc:
            result = {"status": "model_error", "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["train", "serve"])
    parser.add_argument("root", type=Path)
    parser.add_argument("job", nargs="?")
    args = parser.parse_args()
    if args.operation == "train":
        train_job(args.root, args.job)
    else:
        serve(args.root)


if __name__ == "__main__":
    main()
