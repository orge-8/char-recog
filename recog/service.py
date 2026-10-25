"""Fast task submission and isolated, cached inference worker management."""

from pathlib import Path
import asyncio
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

from .store import Store, atomic_json, digest
from .processes import process_alive

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# 子进程一律按文件路径自举：embeddable 解释器有 `python*._pth` 时会忽略
# PYTHONPATH 与 cwd，`-m recog.*` 在启动阶段就失败（详见 recog/launcher.py）。
LAUNCHER = PACKAGE_ROOT / "recog" / "launcher.py"
_LOG = logging.getLogger("char-recog")

# Inference lifecycle tuning. Long budgets belong to the background warm-up thread
# only: a chat request is never dragged past the caller's own deadline.
BREAKER_SECONDS = 30.0
WARMUP_BACKOFF = (0.0, 30.0, 60.0, 120.0, 300.0)
WARMUP_PING_TIMEOUT = 45.0
WARMUP_LOAD_TIMEOUT = 20.0
WARMUP_DEADLINE = 120.0
WIRE_LOCK_WAIT = 0.3

# Manual cleanup only ever removes finished work. A paused job still owns a
# resumable checkpoint, so the page warns about that before asking.
DELETABLE_STATES = {"complete", "failed", "cancelled", "interrupted", "paused"}
DELETE_LIMIT = 50


def inference_error_message(detail: str) -> str:
    """Keep diagnostic tracebacks in logs; translate missing packages for the UI."""
    missing = re.findall(r"No module named ['\"]([^'\"]+)['\"]", detail)
    if missing:
        names = {"torch": "PyTorch", "torchvision": "torchvision", "PIL": "Pillow", "numpy": "NumPy", "psutil": "psutil"}
        packages = dict.fromkeys(names.get(name.split('.')[0], name.split('.')[0]) for name in missing)
        return "；".join(f"未安装 {name}" for name in packages)
    return "推理服务启动或运行失败，请查看 inference.log"


class Service:
    def __init__(self, root: Path, python: str = ""):
        self.store = Store(root)
        self.configure_python(python)
        self.processes = {}
        self.inference = None
        self.inference_log_start = 0
        # Intent and fault are separate facts. `user_stopped` is the only sticky
        # bit: it records an explicit admin action and only an explicit start may
        # clear it. Machine faults use the breaker + retry_at pair instead, so a
        # single slow image can no longer disable automatic recognition for good.
        self.user_stopped = False
        self.breaker_until = 0.0
        self.retry_at = 0.0
        self.warm_failures = 0
        self.ready = False
        self.warming = False
        self.last_error = ""
        self.last_error_at = 0.0
        # Thread negotiation: 0 keeps the worker default, 1 is the training mode.
        self.desired_threads = 0
        self.threads_applied = 0
        self.auto_warm = False
        self.train_lock = threading.Lock()
        # Activation and deletion must not interleave: a version loaded by the
        # worker may not be removed from disk before it becomes the active one.
        self.version_lock = threading.Lock()
        self.infer_lock = threading.Lock()
        # Spawn/warm-up are serialized here; interactive requests never take it.
        self._start_lock = threading.Lock()
        # Leaf lock: guards the fields above and must never be held across IO.
        self._state = threading.RLock()
        self._ready_event = threading.Event()
        self._warm_thread = None
        self.closed = threading.Event()
        self.environment_process = None
        self.environment_cache = None
        self.environment_checked = 0.0
        with self.store.db() as db:
            for row in db.execute("SELECT id FROM jobs WHERE state IN ('queued','running','pausing','cancelling')"):
                (root / "jobs" / row[0] / "cancel").touch()
            db.execute(
                "UPDATE jobs SET state='interrupted',error='控制器重启；可手动重试' WHERE state IN ('queued','running','pausing','cancelling')"
            )
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self.monitor.start()

    def update_config(self, python: str) -> None:
        # Existing training and inference processes keep their environment; new
        # processes use the updated executable without cancelling active work.
        with self.train_lock:
            self.configure_python(python)
            self.environment_cache = None

    def configure_python(self, python: str = "") -> None:
        override = self.store.setting("worker_override")
        custom = override == "private" or self.store.setting("worker_mode") == "custom"
        self.overlay_path = None if custom else self.store.root / "dependencies" / "packages"
        self.python = (
            str(PACKAGE_ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
            if override == "private" else override or python or sys.executable
        )

    def runtime_command(self, *args: str, overlay: Path | None = None) -> list[str]:
        target = self.overlay_path if overlay is None else overlay
        command = [self.python, "-X", "utf8", "-u", str(LAUNCHER)]
        if target is not None:
            command.extend(["--overlay", str(target)])
        return [*command, *args]

    def child_env(self) -> dict:
        """Explicit plugin root on PYTHONPATH: `-m recog.*` must not depend on cwd alone."""
        env = dict(os.environ)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join([str(PACKAGE_ROOT), *([existing] if existing else [])])
        return env

    def _spawn(self, operation: str, job: str = "", pipe: bool = False):
        self.check_environment_idle()
        if not Path(self.python).is_file():
            raise ValueError(
                "插件训练环境尚未安装，请在插件目录执行 uv venv .venv 与 uv pip install --python .venv -r requirements-worker.txt"
            )
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        command = self.runtime_command("recog.worker", operation, str(self.store.root))
        if job:
            command.append(job)
        if pipe:
            stderr = open(self.store.root / "inference.log", "a", encoding="utf-8")
            try:
                return subprocess.Popen(
                    command,
                    cwd=PACKAGE_ROOT,
                    env=self.child_env(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    text=True,
                    encoding="utf-8",
                    creationflags=flags,
                )
            finally:
                stderr.close()
        log = open(self.store.root / "jobs" / job / "worker.log", "a", encoding="utf-8")
        try:
            return subprocess.Popen(command, cwd=PACKAGE_ROOT, env=self.child_env(), stdout=log, stderr=log, creationflags=flags)
        finally:
            log.close()

    def submit(
        self, request_key: str, mode: str = "incremental", device: str = "cpu", character: str = "", input_size: int = 320
    ) -> dict:
        character = Store.validate_character(character)
        if mode not in {"fresh", "incremental"} or device not in {"cpu", "cuda"}:
            raise ValueError("无效训练模式或设备")
        if not request_key or len(request_key) > 128:
            raise ValueError("需要长度1到128的幂等请求编号")
        with self.train_lock:
            self.check_environment_idle()
            with self.store.db() as db:
                db.execute("BEGIN IMMEDIATE")
                existing = db.execute("SELECT * FROM jobs WHERE request_key=?", (request_key,)).fetchone()
                if existing:
                    return dict(existing)
                if any(process.poll() is None for process in self.processes.values()):
                    raise ValueError("上一个训练进程尚未退出")
                active = db.execute(
                    "SELECT id FROM jobs WHERE state IN ('queued','running','pausing','cancelling')"
                ).fetchone()
                if active:
                    raise ValueError("已有训练任务")
                # Previous workers acknowledge cancel; never overlap an orphan after restart.
                for row in db.execute("SELECT id,pid FROM jobs WHERE state='interrupted' AND pid IS NOT NULL"):
                    if not process_alive(row[1]):
                        continue
                    raise ValueError("旧训练进程尚未退出，请稍后手动重试")
                job = uuid.uuid4().hex
                snapshot = self.store.freeze(db, job, mode, device, character, input_size)
                directory = self.store.root / "jobs" / job
                directory.mkdir()
                atomic_json(directory / "snapshot.json", snapshot)
                (directory / "lease").touch()
                db.execute(
                    "INSERT INTO jobs (id,request_key,state,pid,error,created,snapshot_sha,character,device,input_size) "
                    "VALUES (?,?,'queued',NULL,'',?,?,?,?,?)",
                    (job, request_key, time.time(), digest(snapshot), character, device, input_size),
                )
            try:
                process = self._spawn("train", job)
                self.processes[job] = process
                with self.store.db() as db:
                    db.execute("UPDATE jobs SET pid=? WHERE id=?", (process.pid, job))
            except Exception as exc:
                with self.store.db() as db:
                    db.execute("UPDATE jobs SET state='failed',error=? WHERE id=?", (str(exc), job))
        return self.store.job(job)

    def _monitor(self):
        # The thread must survive: if it dies, the lease stops refreshing and the
        # training worker treats the host as gone and cancels itself after 120s.
        while not self.closed.wait(1):
            try:
                self._refresh_jobs()
            except Exception:
                _LOG.exception("训练监控本轮异常，已跳过（下一轮继续刷新租约）")
            try:
                self._tick_inference()
            except Exception:
                _LOG.exception("推理状态检查异常，已跳过")

    def _refresh_jobs(self):
        with self.train_lock:
            for job, process in list(self.processes.items()):
                directory = self.store.root / "jobs" / job
                if process.poll() is None:
                    (directory / "lease").touch()
                    if (directory / "cancel").exists() and time.time() - (
                        directory / "cancel"
                    ).stat().st_mtime > 10:
                        process.terminate()
                else:
                    with self.store.db() as db:
                        state = "cancelled" if (directory / "cancel").exists() else "failed"
                        db.execute(
                            "UPDATE jobs SET state=?,error=? WHERE id=? AND state IN ('queued','running','pausing','cancelling')",
                            (state, f"训练进程退出，code={process.returncode}", job),
                        )
                    self.processes.pop(job)
            # 训练是否在跑，直接看 processes 即可；`training_active` 曾是第二份
            # 真相，由两把不同的锁分别写（这里是 train_lock，mark_training 是
            # _state），全文只有下面那一次「读自己刚写的值」，没有别的消费者。
            # 0.6.1 删掉它，`desired_threads` 已承载全部语义。
            training_running = any(process.poll() is None for process in self.processes.values())
        if not training_running and self.desired_threads:
            self.mark_training(False)

    def cancel(self, job: str) -> dict:
        self.store.job(job)
        with self.store.db() as db:
            db.execute("UPDATE jobs SET state='cancelled' WHERE id=? AND state='paused'", (job,))
            db.execute(
                "UPDATE jobs SET state='cancelling' WHERE id=? AND state IN ('queued','running','pausing')", (job,)
            )
        (self.store.root / "jobs" / job / "cancel").touch()
        return self.store.job(job)

    def pause(self, job: str) -> dict:
        with self.train_lock, self.store.db() as db:
            row = db.execute("SELECT state FROM jobs WHERE id=?", (job,)).fetchone()
            if row is None or row[0] not in {"queued", "running"}:
                raise ValueError("仅运行中的任务可暂停")
            path = self.store.root / "jobs" / job / "progress.json"
            if path.exists() and json.loads(path.read_text("utf-8")).get("phase") == "test":
                raise ValueError("已进入最终测试核验，请等待完成或取消")
            (self.store.root / "jobs" / job / "pause").touch()
            db.execute("UPDATE jobs SET state='pausing' WHERE id=?", (job,))
        return self.store.job(job)

    def resume(self, job: str) -> dict:
        with self.train_lock:
            self.check_environment_idle()
            if any(p.poll() is None for p in self.processes.values()):
                raise ValueError("训练进程尚未退出，请稍后继续")
            with self.store.db() as db:
                row = db.execute("SELECT state FROM jobs WHERE id=?", (job,)).fetchone()
                if row is None or row[0] != "paused":
                    raise ValueError("仅暂停的任务可继续")
                if db.execute(
                    "SELECT 1 FROM jobs WHERE state IN ('queued','running','pausing','cancelling')"
                ).fetchone():
                    raise ValueError("已有其他训练任务")
                directory = self.store.root / "jobs" / job
                if not (directory / "resume.pt").exists():
                    raise ValueError("恢复断点不存在")
                (directory / "pause").unlink(missing_ok=True)
                (directory / "lease").touch()
                db.execute("UPDATE jobs SET state='queued',error='' WHERE id=?", (job,))
            try:
                process = self._spawn("train", job)
                self.processes[job] = process
                with self.store.db() as db:
                    db.execute("UPDATE jobs SET pid=? WHERE id=?", (process.pid, job))
            except Exception as exc:
                with self.store.db() as db:
                    db.execute("UPDATE jobs SET state='paused',error=? WHERE id=?", (str(exc), job))
                raise
        return self.store.job(job)

    def delete_jobs(self, job_ids) -> dict:
        ids = self._clean_ids(job_ids, "任务")
        with self.train_lock:
            with self.store.db() as db:
                placeholders = ",".join("?" for _ in ids)
                rows = {
                    row["id"]: row["state"]
                    for row in db.execute(f"SELECT id,state FROM jobs WHERE id IN ({placeholders})", ids)
                }
            if len(rows) != len(ids):
                raise ValueError("选择中有不存在的任务，请刷新后重试")
            if any(state not in DELETABLE_STATES for state in rows.values()):
                raise ValueError("运行中的任务不能删除，请先取消并等待结束")
            archived = 0
            removed: list[str] = []
            failed: list[str] = []
            for job in ids:
                directory = self.store.root / "jobs" / job
                models_dir = self.store.root / "models" / job
                try:
                    if models_dir.exists():
                        archived += self._archive_job_files(directory, models_dir)
                    if directory.exists():
                        shutil.rmtree(directory)
                except OSError:
                    _LOG.warning("无法删除任务目录 %s", directory, exc_info=True)
                    failed.append(job)
                    continue
                removed.append(job)
            if removed:
                with self.store.db() as db:
                    placeholders = ",".join("?" for _ in removed)
                    db.execute(f"DELETE FROM jobs WHERE id IN ({placeholders})", removed)
            return {"deleted": len(removed), "archived": archived, "failed": failed}

    def delete_versions(self, version_ids) -> dict:
        ids = self._clean_ids(version_ids, "版本")
        with self.train_lock, self.version_lock:
            with self.store.db() as db:
                placeholders = ",".join("?" for _ in ids)
                rows = {
                    row["id"]: dict(row)
                    for row in db.execute(f"SELECT * FROM models WHERE id IN ({placeholders})", ids)
                }
                if len(rows) != len(ids):
                    raise ValueError("选择中有不存在的版本，请刷新后重试")
                active = db.execute(
                    f"SELECT value FROM settings WHERE key LIKE 'active:%' AND value IN ({placeholders})", ids
                ).fetchone()
                if active is not None:
                    raise ValueError(f"版本 {active[0][:8]} 正在使用中，请先启用其他版本")
                states = {"queued": "排队中", "running": "运行中", "pausing": "正在暂停", "paused": "已暂停", "cancelling": "正在取消"}
                for job, state in db.execute(
                    "SELECT id,state FROM jobs WHERE state NOT IN ('complete','failed','cancelled','interrupted')"
                ):
                    try:
                        snapshot = json.loads((self.store.root / "jobs" / job / "snapshot.json").read_text("utf-8"))
                    except (OSError, ValueError):
                        continue
                    parent = snapshot.get("parent")
                    if parent in ids:
                        raise ValueError(
                            f"版本 {parent[:8]} 正被{states.get(state, state)}任务 {job[:8]} 用作父模型，请先取消该任务"
                        )
            released = 0
            removed: list[str] = []
            failed: list[str] = []
            for version in ids:
                directory = self.store.root / "models" / version
                try:
                    size = self.store.model_size(version)
                    if directory.exists():
                        shutil.rmtree(directory)
                except OSError:
                    _LOG.warning("无法删除模型目录 %s", directory, exc_info=True)
                    failed.append(version)
                    continue
                released += size
                removed.append(version)
            if removed:
                with self.store.db() as db:
                    placeholders = ",".join("?" for _ in removed)
                    db.execute(f"DELETE FROM models WHERE id IN ({placeholders})", removed)
                    # A deleted rollback target must not leave the button enabled.
                    db.execute(f"DELETE FROM settings WHERE key LIKE 'previous:%' AND value IN ({placeholders})", removed)
            return {"deleted": len(removed), "failed": failed, "released": released}

    @staticmethod
    def _clean_ids(values, noun: str) -> list[str]:
        parsed = values
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except ValueError as exc:
                raise ValueError(f"{noun}编号格式不正确") from exc
        if not isinstance(parsed, list):
            raise ValueError(f"{noun}编号格式不正确")
        values = [str(value).strip() for value in parsed]
        if any(len(value) != 32 or any(c not in "0123456789abcdef" for c in value) for value in values):
            raise ValueError(f"无效{noun}编号")
        values = list(dict.fromkeys(values))
        if not values:
            raise ValueError(f"请先选择{noun}")
        if len(values) > DELETE_LIMIT:
            raise ValueError(f"每次最多删除 {DELETE_LIMIT} 条{noun}记录")
        return values

    @staticmethod
    def _archive_job_files(directory: Path, models_dir: Path) -> int:
        """Keep the report and snapshot with the model; they are what its page reads."""
        copied = 0
        for name in ("report.json", "snapshot.json"):
            source = directory / name
            destination = models_dir / name
            if not source.exists() or destination.exists():
                continue
            try:
                atomic_json(destination, json.loads(source.read_text("utf-8")))
            except (OSError, ValueError) as exc:
                raise OSError(f"无法留档 {name}") from exc
            copied += 1
        return copied

    @staticmethod
    def _snapshot_meta(path: Path):
        """(device, input_size) from a snapshot, or None when unreadable."""
        try:
            snapshot = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            return None
        device = snapshot.get("device")
        if device not in {"cpu", "cuda"}:
            return None
        size = snapshot.get("input_size", 320)
        if type(size) is not int or size not in {224, 320, 448}:
            size = 320
        return device, size

    def jobs(self) -> dict:
        with self.store.db() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT 50")]
        # Rows written before 0.4.0 have no device column: read each snapshot once
        # and persist the answer, so later polls touch only SQLite.
        self.store.backfill_job_meta(
            rows, lambda job: self._snapshot_meta(self.store.root / "jobs" / job / "snapshot.json")
        )
        for row in rows:
            path = self.store.root / "jobs" / row["id"] / "progress.json"
            try:
                row["progress"] = json.loads(path.read_text("utf-8"))
            except (OSError, ValueError):
                # Windows readers can race an atomic replace; the next poll recovers.
                row["progress"] = {"phase": row["state"]}
            device = row.get("device") or None
            row["device"] = device
            row["input_size"] = row.get("input_size") or (320 if device else None)
            row["device_text"] = {"cpu": "CPU", "cuda": "CUDA"}.get(device, "设备未知")
            p = row["progress"]
            labels = {
                "queued": "排队中",
                "preparing": "准备模型和数据",
                "running": "运行中",
                "training": "训练中",
                "validation": "验证中",
                "test": "测试和核验",
                "complete": "完成",
                "failed": "失败",
                "paused": "已暂停",
                "pausing": "正在保存断点",
                "cancelled": "已取消",
                "cancelling": "取消中",
                "interrupted": "已中断",
            }
            row["name"] = time.strftime("训练 %m-%d %H:%M", time.localtime(row["created"])) + (
                f" · {row['character']}" if row["character"] else ""
            )
            row["state_text"] = labels.get(row["state"], row["state"])
            fraction = ((p.get("epoch", 1) - 1) + p.get("step", 0) / max(1, p.get("steps", 1))) / max(
                1, p.get("epochs", 1)
            )
            row["percent"] = (
                100 if row["state"] == "complete" else min(99, round(fraction * 100, 1)) if p.get("epochs") else 0
            )
            seconds = p.get("seconds", 0)
            row["time_text"] = f"已用 {seconds:.0f} 秒"
            if p.get("phase") == "training" and fraction > 0.05 and seconds > 5:
                row["time_text"] += f" · 预计还需约 {seconds * (1 / fraction - 1):.0f} 秒（不含最终核验）"
            row["progress_text"] = labels.get(p.get("phase"), "准备中") + (
                f" · 第{p['epoch']}/{p['epochs']}轮 · {p.get('seen', 0)}/{p.get('total_images', 0)}张 · {p.get('step', 0)}/{p.get('steps', 0)}批"
                if p.get("epochs")
                else ""
            )
            row["can_pause"] = row["state"] in {"queued", "running"} and p.get("phase") != "test"
            row["can_resume"] = row["state"] == "paused"
            row["can_cancel"] = row["state"] in {"queued", "running", "pausing", "paused"}
            row["has_report"] = row["state"] == "complete"
        return {"rows": rows}

    # ------------------------------------------------------------ inference lifecycle
    #
    # Four facts are tracked independently instead of one boolean:
    #   user_stopped      explicit admin stop (sticky, only start clears it)
    #   breaker_until     machine fault cooldown
    #   retry_at          earliest automatic warm-up attempt (exponential backoff)
    #   ready / warming   worker readiness and background warm-up progress
    # Derived state is exposed through inference_state().

    @property
    def inference_enabled(self) -> bool:
        """False only after an explicit stop; faults use the breaker instead."""
        return not self.user_stopped

    def inference_state(self) -> str:
        with self._state:
            if self.closed.is_set():
                return "closed"
            if self.user_stopped:
                return "stopped"
            if self.warming:
                return "starting"
            # Backoff after repeated failures is a cooldown as much as the breaker.
            if time.monotonic() < max(self.breaker_until, self.retry_at):
                return "cooling"
            process = self.inference
            if self.ready and process is not None and process.poll() is None:
                return "ready"
            return "cold"

    def inference_retry_in(self) -> int:
        with self._state:
            return max(0, int(max(self.breaker_until, self.retry_at) - time.monotonic()))

    def enable_auto(self) -> None:
        """Allow the background warm-up; called once from plugin.on_load."""
        with self._state:
            self.auto_warm = True

    def mark_training(self, active: bool) -> None:
        """Lend CPU back to training; the next tick applies it to the worker."""
        with self._state:
            self.desired_threads = 1 if active else 0

    @property
    def start_lock(self):
        """Serializes worker spawn/warm-up against dependency installation."""
        return self._start_lock

    def _active_versions(self) -> dict:
        return {character: self.store.active(character) for character in self.store.active_characters()}

    def _not_ready_result(self) -> dict:
        state = self.inference_state()
        if state == "stopped":
            return {
                "status": "inference_stopped",
                "state": "stopped",
                "error": "推理服务已停止，请手动开启或重新加载插件",
            }
        if state == "cooling":
            remaining = self.inference_retry_in()
            return {
                "status": "cooling",
                "state": "cooling",
                "error": self.last_error or f"推理服务冷却中，约 {remaining} 秒后自动重试",
                "retry_after": remaining,
            }
        return {"status": "warming", "state": state, "error": "推理服务正在启动"}

    def request_warmup(self, *, proactive: bool) -> None:
        """Start (at most) one background warm-up thread."""
        with self._state:
            if self.closed.is_set() or self.user_stopped or self.warming or self.ready:
                return
            now = time.monotonic()
            # Backoff applies to every automatic path, so a broken install cannot
            # spawn a torch import every second. Explicit starts clear it first.
            if now < self.breaker_until or (self.warm_failures and now < self.retry_at):
                return
            if proactive and not self.auto_warm:
                return
            self.warming = True
            self._ready_event.clear()
            self._warm_thread = threading.Thread(target=self._warm, name="infer-warm", daemon=True)
            self._warm_thread.start()

    def _abort_warm(self) -> bool:
        return self.closed.is_set() or self.user_stopped

    def _record_failure(self, error: str) -> None:
        with self._state:
            self.ready = False
            self.last_error = error
            self.last_error_at = time.time()
            if self.user_stopped:
                # An explicit stop must never be resurrected by an automatic retry.
                self._ready_event.set()
                return
            self.warm_failures += 1
            index = min(self.warm_failures, len(WARMUP_BACKOFF) - 1)
            self.breaker_until = time.monotonic() + BREAKER_SECONDS
            self.retry_at = self.breaker_until + WARMUP_BACKOFF[index]
            # Wake blocked callers so they report the failure instead of waiting
            # out their whole budget for a worker that already gave up.
            self._ready_event.set()

    def _terminate(self, process, thread=None) -> None:
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        if thread is not None:
            thread.join(0.5)
            if thread.is_alive():
                # A live reader still owns the pipe; closing it under the thread
                # raises on Windows, and the process is already terminating.
                return
        # Closing the pipes while an exchange is in flight would race its reader.
        if not self.infer_lock.acquire(blocking=False):
            return
        try:
            for stream in (process.stdin, process.stdout):
                try:
                    if stream is not None and not stream.closed:
                        stream.close()
                except OSError:
                    pass
        finally:
            self.infer_lock.release()

    def _reap(self, process, thread=None) -> None:
        self._terminate(process, thread)
        with self._state:
            self.inference = None
            self.ready = False
            self.warming = False
            self._ready_event.clear()

    @staticmethod
    def _round_trip(process, payload: dict, timeout: float):
        """Write one request and read one line, bounded by `timeout`."""
        output = []

        def read_response():
            try:
                process.stdin.write(json.dumps(payload) + "\n")
                process.stdin.flush()
                output.append(process.stdout.readline())
            except (OSError, UnicodeError):
                output.append("")

        thread = threading.Thread(target=read_response, daemon=True)
        thread.start()
        thread.join(max(0.0, timeout))
        return (output[0] if output else ""), thread.is_alive(), thread

    def _exchange_on(self, process, payload: dict, timeout: float) -> dict:
        """Round trip used by the warm-up thread; takes the wire lock itself."""
        if not self.infer_lock.acquire(timeout=max(0.05, timeout)):
            raise TimeoutError("推理通道繁忙")
        try:
            line, timed_out, thread = self._round_trip(process, payload, timeout)
        finally:
            self.infer_lock.release()
        if timed_out:
            thread.join(0.5)
            raise TimeoutError("推理服务响应超时")
        if not line:
            raise RuntimeError("推理服务无响应")
        return json.loads(line)

    def _log_tail(self) -> str:
        try:
            with (self.store.root / "inference.log").open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(self.inference_log_start, stream.tell() - 2000))
                return stream.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _warm(self) -> None:
        started = time.monotonic()
        process = None
        try:
            with self._start_lock:
                if self._abort_warm():
                    return
                log = self.store.root / "inference.log"
                self.inference_log_start = log.stat().st_size if log.exists() else 0
                process = self._spawn("serve", pipe=True)
                with self._state:
                    self.inference = process
                # Cold start imports torch/torchvision; this budget never reaches chat.
                ping = self._exchange_on(process, {"operation": "ping"}, WARMUP_PING_TIMEOUT)
                if ping.get("status") != "ready":
                    raise RuntimeError(ping.get("error") or "推理服务无响应")
                with self._state:
                    threads = self.desired_threads
                if threads:
                    try:
                        self._exchange_on(process, {"operation": "threads", "value": threads}, 5.0)
                        with self._state:
                            self.threads_applied = threads
                    except Exception as exc:
                        _LOG.warning("推理线程数调整失败：%s", exc)
                for character, version in sorted(self._active_versions().items()):
                    if not version:
                        continue
                    if self._abort_warm():
                        break
                    try:
                        result = self._exchange_on(
                            process,
                            {"operation": "load", "character": character, "version": version},
                            WARMUP_LOAD_TIMEOUT,
                        )
                    except Exception as exc:
                        # One bad character must not invalidate the whole worker.
                        _LOG.warning("预热跳过「%s」：%s", character, exc)
                    else:
                        if result.get("status") != "ready":
                            _LOG.warning("预热跳过「%s」：%s", character, result.get("error") or result.get("status"))
                    if time.monotonic() - started > WARMUP_DEADLINE:
                        raise RuntimeError("推理预热超时")
            with self._state:
                aborted = self.user_stopped or self.closed.is_set()
                self.warming = False
                if not aborted:
                    self.ready = True
                    self.breaker_until = 0.0
                    self.retry_at = 0.0
                    self.warm_failures = 0
                    self.last_error = ""
                self._ready_event.set()
            if aborted:
                self._reap(process)
                return
        except BaseException as exc:
            self._reap(process)
            # Keep the diagnostic in inference.log; the log tail is what tells
            # "missing torch" apart from a generic launch failure.
            self._record_failure(inference_error_message(f"{exc}\n{self._log_tail()}"))
            _LOG.warning("推理预热失败（第 %d 次）：%s", self.warm_failures, exc)

    def _tick_inference(self) -> None:
        """Background self-healing; runs on the monitor thread."""
        if not self.auto_warm or self.user_stopped or self.closed.is_set():
            return
        state = self.inference_state()
        process = self.inference
        if state == "ready":
            with self._state:
                wanted, applied = self.desired_threads, self.threads_applied
            if wanted != applied and process is not None:
                try:
                    result = self._exchange_on(process, {"operation": "threads", "value": wanted}, 1.0)
                except Exception as exc:
                    _LOG.debug("调整推理线程数失败：%s", exc)
                else:
                    if result.get("status") == "ok":
                        with self._state:
                            self.threads_applied = wanted
            elif process is not None and process.poll() is not None:
                self._reap(process)
                self._record_failure(f"推理进程意外退出，code={process.returncode}")
            return
        if state in {"cold", "cooling"} and time.monotonic() >= max(self.breaker_until, self.retry_at):
            cache = self.environment_cache
            if cache and not cache.get("detecting") and not cache.get("installed"):
                # Nothing to start yet: the admin has not installed PyTorch.
                return
            try:
                self.check_environment_idle()
            except ValueError:
                return
            if self._active_versions():
                self.request_warmup(proactive=True)

    def wait_ready(self, budget: float) -> str:
        """Block until ready/stopped/closed or the budget runs out."""
        state = self.inference_state()
        if state == "cooling":
            return state
        if state == "cold":
            self.request_warmup(proactive=False)
        deadline = time.monotonic() + max(0.0, budget)
        while True:
            state = self.inference_state()
            if state in {"ready", "stopped", "closed", "cooling"}:
                return state
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return state
            self._ready_event.wait(min(remaining, 0.25))

    def start_inference(self, wait: float = 6.0, *, require_models: bool = True) -> dict:
        """Explicit admin start: clears both the stop flag and the breaker."""
        with self._state:
            self.user_stopped = False
            self.breaker_until = 0.0
            self.retry_at = 0.0
            self.warm_failures = 0
            self.last_error = ""
        versions = self._active_versions()
        if require_models and not versions:
            self.request_warmup(proactive=False)
            return {"state": "cold", "error": "尚未启用任何角色模型", "ready_characters": []}
        self.request_warmup(proactive=False)
        state = self.wait_ready(wait)
        return {
            "state": state,
            "ready_characters": sorted(versions) if state == "ready" else [],
            "error": "" if state == "ready" else (self.last_error or "推理服务正在启动"),
        }

    def restart_inference(self) -> None:
        """Drop the worker (environment changed) without recording an admin stop."""
        with self._start_lock:
            self._reap(self.inference)
            with self._state:
                self.breaker_until = 0.0
                self.retry_at = 0.0
                self.warm_failures = 0
        if not self.user_stopped:
            self.request_warmup(proactive=False)

    def stop_worker(self) -> None:
        """Drop the worker only; keeps the admin intent untouched."""
        with self._start_lock:
            self._reap(self.inference)

    def stop_inference(self) -> None:
        """Explicit admin stop. Sticky until start_inference/request(resume=True)."""
        with self._state:
            self.user_stopped = True
            self.breaker_until = 0.0
            self.retry_at = 0.0
            self.warm_failures = 0
        if self._start_lock.acquire(timeout=2.0):
            try:
                self._reap(self.inference)
            finally:
                self._start_lock.release()
        else:
            self._reap(self.inference)
        with self._state:
            self.warming = False
            self._ready_event.set()

    def request(self, payload: dict, timeout: float = 60, *, resume: bool = False, wait: bool = True) -> dict:
        """One request/response round trip.

        `wait=False` is the chat path: it never blocks on a cold worker, it only
        kicks the background warm-up and reports the current state, so the caller
        always keeps its own deadline.
        """
        budget = max(0.05, timeout)
        deadline = time.monotonic() + budget
        if resume:
            result = self.start_inference(wait=min(budget, 8.0), require_models=False)
            if result["state"] != "ready":
                return {"status": "model_error", "error": result["error"], "state": result["state"]}
        elif not wait:
            if self.inference_state() != "ready":
                self.request_warmup(proactive=False)
                return self._not_ready_result()
        else:
            state = self.wait_ready(max(0.0, deadline - time.monotonic()))
            if state != "ready":
                return self._not_ready_result()
        return self.exchange(payload, max(0.0, deadline - time.monotonic()))

    def exchange(self, payload: dict, timeout: float) -> dict:
        """Send one payload to a ready worker. `busy` never trips the breaker."""
        if not self.infer_lock.acquire(timeout=min(WIRE_LOCK_WAIT, max(0.02, timeout))):
            return {"status": "busy", "error": "推理繁忙"}
        thread = None
        try:
            with self._state:
                process = self.inference
                if self.user_stopped:
                    return self._not_ready_result()
                if not self.ready or process is None or process.poll() is not None:
                    return self._not_ready_result()
            line, timed_out, thread = self._round_trip(process, payload, timeout)
            if timed_out or not line:
                self._reap(process, thread)
                if timed_out:
                    return self._fail(f"推理响应超时（{timeout:g} 秒），请查看运行环境和 inference.log")
                return self._fail(inference_error_message(self._log_tail()))
            try:
                result = json.loads(line)
            except ValueError:
                self._reap(process, thread)
                return self._fail("推理响应格式错误，请查看 inference.log")
            self._ok()
            return result
        finally:
            self.infer_lock.release()

    def _fail(self, error: str) -> dict:
        self._record_failure(error)
        with self._state:
            self.inference = None
            self.ready = False
        return {"status": "model_error", "error": error}

    def _ok(self) -> None:
        with self._state:
            self.breaker_until = 0.0
            self.retry_at = 0.0
            self.warm_failures = 0
            self.last_error = ""

    def check_environment_idle(self) -> None:
        """Never start a worker while dependencies are being replaced."""
        if self.environment_process and self.environment_process.poll() is None:
            raise ValueError("依赖安装/卸载正在运行，请等待完成")
        path = self.store.root / "environment-status.json"
        if path.exists():
            job = json.loads(path.read_text("utf-8"))
            if job.get("state") == "running":
                if not process_alive(job["pid"]):
                    return
                raise ValueError("依赖安装/卸载正在运行，请等待完成")

    def infer(self, raw: bytes, timeout: float = 8, character: str = "", *, wait: bool = True) -> dict:
        """Single-character score (character empty = every character with an active model)."""
        if character:
            if not self.store.active(character):
                return {"status": "no_model"}
        elif not self.store.active_characters():
            return {"status": "no_model"}
        payload = {"image": base64.b64encode(raw).decode()}
        if character:
            payload["character"] = character
        try:
            return self.request(payload, timeout, wait=wait)
        except Exception as exc:
            return {"status": "model_error", "error": str(exc)}

    def activate(self, character: str, version: str) -> dict:
        with self.store.db() as db:
            row = db.execute("SELECT character FROM models WHERE id=?", (version,)).fetchone()
        if row is None:
            raise ValueError("版本不存在")
        if row[0] and row[0] != character:
            raise ValueError("版本属于其他角色")
        state = self.inference_state()
        if state == "stopped":
            raise ValueError("推理服务已手动停止，请先在运行环境页开启推理")
        if state in {"cooling", "starting"}:
            raise ValueError("推理服务正在启动或冷却，请稍后重试")
        if state != "ready":
            self.request_warmup(proactive=False)
            # An explicit admin action may wait longer than a chat request: a cold
            # `import torch` takes 10-20s on a machine that just finished training.
            state = self.wait_ready(25)
            if state == "starting":
                raise ValueError("推理服务正在启动（首次加载 PyTorch 较慢），请稍后重试")
            if state != "ready":
                raise ValueError(self.last_error or "推理服务未能启动，请检查运行环境")
        with self.version_lock:
            with self.store.db() as db:
                row = db.execute("SELECT character FROM models WHERE id=?", (version,)).fetchone()
            if row is None:
                raise ValueError("版本不存在")
            if row[0] and row[0] != character:
                raise ValueError("版本属于其他角色")
            payload = {"operation": "load", "character": character, "version": version}
            result = self.exchange(payload, 8)
            if result.get("status") != "ready":
                raise ValueError(result.get("error", "模型加载失败"))
            self.store.set_active(character, version)
        return result

    def rollback(self, character: str) -> dict:
        previous = self.store.previous(character)
        if not previous:
            raise ValueError("无可回退版本")
        return self.activate(character, previous)

    def close(self) -> None:
        self.closed.set()
        with self._state:
            self._ready_event.set()
        self.monitor.join(3)
        with self.train_lock:
            for job, process in self.processes.items():
                self.cancel(job)
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                with self.store.db() as db:
                    db.execute(
                        "UPDATE jobs SET state='cancelled',error='插件卸载' WHERE id=? AND state!='complete'", (job,)
                    )
        self._reap(self.inference)
        warm = self._warm_thread
        if warm is not None and warm.is_alive():
            warm.join(5)


async def offload(function, *args, **kwargs):
    return await asyncio.to_thread(function, *args, **kwargs)
