"""Admin-trained per-character identity recognition for MaiBot.

Generalized from SengokuCola/MaiSelfPlugin (GPL-3.0): one binary model per
character; "self" means "is this character", "other" means "is not".
"""

import logging
from pathlib import Path
from typing import Any, Literal, Optional
import base64
import json
import uuid
import hashlib
import time

from maibot_sdk import API, Tool, HookHandler, MaiBotPlugin, PluginConfigBase, Field
from maibot_sdk.types import ToolParameterInfo, ToolParamType

from .recog.service import Service, offload
from .recog.scores import (
    DEFAULT_THRESHOLD,
    EVALUATION_THRESHOLD,
    JUDGMENT_BAND,
    metrics,
    select_top,
    suggest_threshold,
    sweep,
)

DEFAULT_CHARACTERS = ["洛天依", "言和", "乐正绫", "乐正龙牙", "徵羽摩柯", "墨清弦", "星尘", "心华"]
_LOG = logging.getLogger("char-recog")


class PluginSectionConfig(PluginConfigBase):
    enabled: bool = Field(default=True, description="是否启用角色识别插件")
    config_version: str = Field(default="0.3.0", description="配置版本")


class Config(PluginConfigBase):
    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    characters: list[str] = Field(
        default_factory=lambda: list(DEFAULT_CHARACTERS),
        description="可训练/识别的角色名单（任意角色，中V名单仅为默认预置）；识别只覆盖已启用模型的角色",
    )
    worker_python: str = Field(default="", description="留空继承MaiBot Python并使用插件依赖覆盖层；可指定其他底层Python")
    input_size: Literal[224, 320, 448] = Field(default=320, description="新训练任务的图片尺寸；已训练模型使用自己的尺寸")
    auto_recognize: bool = Field(default=True, description="自动识别收到的图片，命中结果作为上下文提示")


class CharacterRecognition(MaiBotPlugin):
    config_model = Config
    service: Optional[Service] = None

    async def on_load(self) -> None:
        # 组件注册先于 on_load（runner 先 _register_plugin 再 _invoke_plugin_on_load），
        # 因此 on_load 让出事件循环期间消息钩子就可能被触发。先备好钩子要用的状态，
        # 再把 service 挂上去，避免这段窗口里抛 AttributeError。
        self._image_cache = {}
        self.service = await offload(Service, self.ctx.paths.data_dir, self.config.worker_python)
        from .recog.environment import recover
        await offload(recover, self.service)
        await self.sync_characters()
        # Let the monitor warm the worker in the background so the first image
        # does not have to pay the torch import inside a 6 second chat budget.
        self.service.enable_auto()

    async def sync_characters(self) -> None:
        """Persist the configured character list for the WebUI selectors."""
        characters = [c.strip() for c in self.config.characters if isinstance(c, str) and c.strip()]
        await offload(self.service.store.set_setting, "characters", json.dumps(characters, ensure_ascii=False))

    def active_versions(self) -> dict:
        """character -> active model version."""
        return {c: self.service.store.active(c) for c in self.service.store.active_characters()}

    def active_thresholds(self) -> dict:
        """character -> online gate, for every character with an active model."""
        limits = self.service.store.thresholds()
        return {c: limits.get(c, DEFAULT_THRESHOLD) for c in self.service.store.active_characters()}

    @HookHandler("chat.receive.before_process", timeout_ms=8000)
    async def annotate_received_images(self, message: dict, **kwargs: Any) -> dict:
        """Annotate individual received media without collecting training examples."""
        if self.service is None or not self.config.auto_recognize or not self.service.inference_enabled:
            return {"action": "continue"}
        versions = await offload(self.active_versions)
        if not versions:
            return {"action": "continue"}
        limits = await offload(self.active_thresholds)
        versions_key = tuple(sorted(versions.items()))
        # Thresholds belong in the cache key: changing one must not reuse the up
        # to 512 images judged under the previous gate.
        limits_key = tuple(sorted(limits.items()))
        deadline = time.monotonic() + 6

        async def visit(parts):
            result = []
            for part in parts:
                if part.get("type") == "forward":
                    nodes = []
                    for node in part.get("data", []):
                        nodes.append({**node, "content": await visit(node.get("content", []))})
                    result.append({**part, "data": nodes})
                    continue
                result.append(part)
                if part.get("type") not in {"image", "emoji"} or time.monotonic() >= deadline:
                    continue
                binary = part.get("binary_data_base64")
                if not isinstance(binary, str) or len(binary) > 28 * 1024 * 1024:
                    continue
                try:
                    raw = base64.b64decode(binary, validate=True)
                    key = (hashlib.sha256(raw).hexdigest(), versions_key, limits_key)
                    prediction = self._image_cache.get(key)
                    if prediction is None:
                        # wait=False: a cold or cooling worker never blocks the chat
                        # path; this image simply goes unannotated and the next one
                        # is picked up once the background warm-up finishes.
                        prediction = await offload(
                            self.service.infer, raw, max(0.1, deadline - time.monotonic()), "", wait=False
                        )
                        if prediction.get("status") == "ok" and prediction.get("results"):
                            if len(self._image_cache) >= 512:
                                self._image_cache.pop(next(iter(self._image_cache)))
                            self._image_cache[key] = prediction
                    policy = select_top(prediction.get("results", []), limits)
                    top = policy["top1"]
                    if top and not policy["ambiguous"]:
                        result.append(
                            {
                                "type": "text",
                                "data": f"\n[提示：此图中的角色看起来很可能是 “{top['character']}”]\n",
                            }
                        )
                except (ValueError, OSError) as exc:
                    # 保留消息原文（fail-open），但留一条线索：哪张图、为什么跳过。
                    _LOG.debug("跳过无法识别的图片：%s 字节 · %s", len(binary) if isinstance(binary, str) else -1, exc)
                    continue
            return result

        updated = dict(message)
        updated["raw_message"] = await visit(message.get("raw_message", []))
        return {"action": "continue", "modified_kwargs": {**kwargs, "message": updated}}

    async def on_unload(self) -> None:
        # Registration may fail before on_load creates the service.
        if self.service is not None:
            await offload(self.service.close)

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        """The runner applies typed self configuration before invoking this hook."""
        if scope != "self":
            return
        await offload(self.service.update_config, self.config.worker_python)
        await self.sync_characters()

    async def _action(self, function, *args, **kwargs) -> dict:
        try:
            return await offload(function, *args, **kwargs)
        except (ValueError, OSError) as exc:
            return {"success": False, "error": str(exc)}

    def _validate_character(self, character: str) -> str:
        return self.service.store.validate_character(character)

    @API("self_upload", version="1")
    async def upload(self, upload_id: str, character: str, upload_label: str = "self") -> dict:
        if upload_label not in {"self", "other"}:
            return {"success": False, "error": "上传身份必须为是角色或不是角色"}
        try:
            character = self._validate_character(character)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        claimed = await self.ctx.webui.claim_upload(upload_id)
        result = await self._action(self.service.store.ingest, Path(claimed["path"]), upload_label, character)
        if result.get("already_exists"):
            result["message"] = "图片已存在，复用原标注"
        return result

    @API("self_characters", version="1")
    async def characters(self) -> dict:
        rows = json.loads(await offload(self.service.store.setting, "characters", "[]"))
        return {"rows": rows, "active": sorted(await offload(self.active_versions))}

    @API("self_images", version="1")
    async def images(self, page: int = 1, view: str = "all", character: str = "", job_id: str = "") -> dict:
        errors = job_id if view == "errors" else ""
        try:
            character = self._validate_character(character)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        result = await offload(self.service.store.images, page, view, job_id, character)
        result.update(
            errors_active=bool(errors),
            character=character,
            source_text=f"错判图片 · 任务 {errors[:8]}（人工标签与模型判断不一致）" if errors else f"图片库 · {character}",
        )
        return result

    @API("self_label", version="1")
    async def label(self, image_id: str, label: str, remove: bool | None = None) -> dict:
        return await self._action(self.service.store.label, image_id, label, remove)

    @API("self_label_positive", version="1")
    async def label_positive(self, image_id: str) -> dict:
        return await self.label(image_id, "self")

    @API("self_label_negative", version="1")
    async def label_negative(self, image_id: str) -> dict:
        return await self.label(image_id, "other")

    @API("self_label_clear", version="1")
    async def label_clear(self, image_id: str) -> dict:
        return await self.label(image_id, "unlabelled")

    @API("self_image_remove", version="1")
    async def remove_image(self, image_id: str) -> dict:
        return await self._action(self.service.store.set_removed, image_id, True)

    @API("self_image_restore", version="1")
    async def restore_image(self, image_id: str) -> dict:
        return await self._action(self.service.store.set_removed, image_id, False)

    @API("self_images_batch", version="1")
    async def batch_images(self, image_ids: str, operation: str) -> dict:
        if operation not in {"self", "other", "remove"}:
            return {"success": False, "error": "无效批量操作"}
        return await self._action(self.service.store.batch_update, image_ids, operation)

    @API("self_images_remove_all", version="1")
    async def remove_all_images(self, character: str = "") -> dict:
        """Only the selected character is affected; an empty name fails closed."""
        try:
            character = self._validate_character(character)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        return await self._action(self.service.store.batch_update, "", "remove_all", character)

    @API("self_images_batch_positive", version="1")
    async def batch_positive(self, image_ids: str) -> dict:
        return await self.batch_images(image_ids, "self")

    @API("self_images_batch_negative", version="1")
    async def batch_negative(self, image_ids: str) -> dict:
        return await self.batch_images(image_ids, "other")

    @API("self_images_batch_remove", version="1")
    async def batch_remove(self, image_ids: str) -> dict:
        return await self.batch_images(image_ids, "remove")

    @API("self_training_submit", version="1")
    async def submit(
        self, request_key: str, character: str = "", mode: str = "incremental", device: str = "cpu", input_size: int = 0
    ) -> dict:
        from .recog.environment import status

        environment = await offload(status, self.service)
        if environment.get("detecting"):
            return {"success": False, "error": "正在检测运行环境，请稍后再开始训练"}
        if not environment.get("installed") or device == "cuda" and not environment.get("cuda"):
            return {"success": False, "error": "所选设备不可用，请先检查运行环境或安装匹配的PyTorch"}
        result = await self._action(
            self.service.submit, request_key, mode, device, character, input_size or self.config.input_size
        )
        if result.get("id") and await offload(self.service.store.setting, "form_request_key") == request_key:
            await offload(self.service.store.set_setting, "form_request_key", uuid.uuid4().hex)
        if result.get("state") in {"failed", "cancelled", "interrupted"}:
            return {
                **result,
                "success": False,
                "error": f"请求已对应任务 {result['id']}（{result['state']}），未启动新训练。"
                f"请检查设备和环境后再次点击“开始训练”。原失败原因：{result.get('error') or '无'}",
            }
        return result

    @API("self_training_pause", version="1")
    async def pause_training(self, job_id: str) -> dict:
        return await self._action(self.service.pause, job_id)

    @API("self_training_resume", version="1")
    async def resume_training(self, job_id: str) -> dict:
        return await self._action(self.service.resume, job_id)

    @API("self_select_report", version="1")
    async def select_report(self, job_id: str) -> dict:
        await offload(self.service.store.job, job_id)
        return {"navigate_page": "training", "navigate_params": {"job_id": job_id}}

    @API("self_select_errors", version="1")
    async def select_errors(self, job_id: str) -> dict:
        row = await offload(self.service.store.job, job_id)
        if row["state"] != "complete":
            return {"success": False, "error": "任务未完成，错判报告尚未生成"}
        return {
            "navigate_page": "images",
            "navigate_params": {"job_id": job_id, "view": "errors", "character": row.get("character", "")},
        }

    @API("self_clear_errors", version="1")
    async def clear_errors(self) -> dict:
        return {"navigate_page": "images", "navigate_params": {"view": "all"}}

    @API("self_environment", version="1")
    async def environment(self) -> dict:
        from .recog.environment import status

        return await offload(status, self.service)

    @API("self_environment_manage", version="1")
    async def manage_environment(self, kind: str) -> dict:
        from .recog.environment import submit

        return await self._action(submit, self.service, kind)

    @API("self_environment_mode", version="1")
    async def environment_mode(self, mode: str, python_path: str = "") -> dict:
        return await self._action(self.change_environment, mode, python_path)

    def change_environment(self, mode: str, python_path: str = "") -> dict:
        import sys

        if mode not in {"host", "custom"}:
            raise ValueError("无效环境")
        python = sys.executable
        if mode == "custom":
            candidate = Path(python_path.strip()).expanduser()
            if not candidate.is_absolute():
                raise ValueError("请填写已有 Python 解释器或环境目录的绝对路径")
            if candidate.is_dir():
                names = ("Scripts/python.exe", "python.exe") if sys.platform == "win32" else ("bin/python", "bin/python3")
                candidate = next((candidate / name for name in names if (candidate / name).is_file()), candidate / names[0])
            if not candidate.is_file():
                raise ValueError("Python 解释器不存在；插件不会创建环境")
            python = str(candidate.resolve())
        with self.service.train_lock:
            self.service.check_environment_idle()
            if any(p.poll() is None for p in self.service.processes.values()):
                raise ValueError("训练中不能切换环境")
            # Drop the worker without recording an admin stop: after the switch the
            # background warm-up brings inference back on the new interpreter.
            self.service.stop_worker()
            self.service.store.set_setting("worker_override", python)
            self.service.store.set_setting("worker_mode", mode)
            self.service.configure_python()
            self.service.environment_cache = None
        self.service.restart_inference()
        return {"success": True}

    def stop_inference_sync(self):
        self.service.stop_inference()

    @API("self_stop_inference", version="1")
    async def stop_inference(self) -> dict:
        await offload(self.stop_inference_sync)
        return {"success": True}

    @API("self_start_inference", version="1")
    async def start_inference(self) -> dict:
        cache = self.service.environment_cache or {}
        if cache.get("missing"):
            names = {"torch": "PyTorch", "torchvision": "torchvision"}
            return {"success": False, "error": "；".join(f"未安装 {names.get(name, name)}" for name in cache["missing"])}
        versions = await offload(self.active_versions)
        if not versions:
            return {"success": False, "error": "尚未启用任何角色模型"}
        # One character failing to load must not take the whole worker down.
        result = await offload(self.service.start_inference, 6.0)
        if result["state"] == "ready":
            return {"success": True, "result": {"ready_characters": result["ready_characters"]}}
        if result["state"] in {"starting", "cold", "warming"}:
            # Cold start continued in the background; this is not a failure.
            return {"success": True, "result": result, "message": "推理服务正在后台启动，完成后自动开始识别"}
        return {"success": False, "error": result.get("error") or "推理服务未能启动", "result": result}

    @API("self_training_form", version="1")
    async def training_form(self, character: str = "") -> dict:
        try:
            character = self._validate_character(character)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        key = await offload(self.form_key)
        if not key:
            key = uuid.uuid4().hex
            await offload(self.service.store.set_setting, "form_request_key", key)
        return {
            "request_key": key,
            "character": character,
            "input_size": self.config.input_size,
            "threshold": await offload(self.service.store.threshold, character),
            "threshold_default": DEFAULT_THRESHOLD,
            **await offload(self.service.store.training_readiness, character),
        }

    def form_key(self):
        with self.service.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            r = db.execute("SELECT value FROM settings WHERE key='form_request_key'").fetchone()
            key = r[0] if r else uuid.uuid4().hex
            existing = db.execute("SELECT state FROM jobs WHERE request_key=?", (key,)).fetchone()
            if existing and existing[0] in {"failed", "complete", "cancelled", "interrupted"}:
                key = uuid.uuid4().hex
            db.execute("INSERT OR REPLACE INTO settings VALUES ('form_request_key',?)", (key,))
        return key

    @API("self_training_new_request", version="1")
    async def new_request(self) -> dict:
        key = uuid.uuid4().hex
        await offload(self.service.store.set_setting, "form_request_key", key)
        return {"request_key": key}

    @API("self_training_jobs", version="1")
    async def jobs(self) -> dict:
        return await offload(self.service.jobs)

    @API("self_training_cancel", version="1")
    async def cancel(self, job_id: str) -> dict:
        return await self._action(self.service.cancel, job_id)

    @API("self_training_report", version="1")
    async def report(self, job_id: str = "") -> dict:
        if not job_id:
            return {"summary": "选择任务以查看报告", "metrics": [], "gallery": [], "pages": 1, "technical": ""}
        try:
            row = await offload(self.service.store.job, job_id)
            report = await offload(self.service.store.report, job_id)
        except ValueError as exc:
            return {"summary": str(exc), "metrics": [], "technical": ""}
        if not report:
            return {"summary": "报告尚未完成", "metrics": [], "gallery": [], "pages": 1, "technical": ""}
        character = row.get("character") or report.get("character") or "该角色"
        current = await offload(self.service.store.threshold, character)
        stored_selection = report.get("selection", {})
        # The trainer always evaluates at EVALUATION_THRESHOLD; reports written by
        # older versions may not carry the field, so fall back to the constant.
        selection_threshold = stored_selection.get("threshold", EVALUATION_THRESHOLD)
        validation_predictions = report.get("validation_predictions") or []
        test_predictions = report.get("test_predictions") or []

        def at_gate(predictions, fallback):
            """Recompute at the gate in force now, not the one used at training time."""
            if not predictions:
                return fallback
            return metrics(
                [int(r["label"] == "self") for r in predictions],
                [r["score"] for r in predictions],
                current,
            )

        validation_metrics = at_gate(validation_predictions, stored_selection.get("validation") or report["test"])
        test_metrics = at_gate(test_predictions, report["test"])
        suggestion = None
        sweep_rows = []
        if validation_predictions:
            labels = [int(r["label"] == "self") for r in validation_predictions]
            scores = [r["score"] for r in validation_predictions]
            suggestion = suggest_threshold(labels, scores)
            sweep_rows = sweep(labels, scores)
        summary = {
            k: v for k, v in report.items() if k not in {"history", "test_predictions", "validation_predictions"}
        }
        summary["评测性质"] = "历史留出评测" if report["historical_test"] else "首次封存测试"
        summary["比较提示"] = (
            "留出真值或成员改变，与旧报告不可直接比较"
            if not report["comparable_to_previous"]
            else "留出真值及成员未改变"
        )
        summary["当前判定阈值"] = current
        summary["训练时评测阈值"] = selection_threshold
        rows = []
        for name, m in (("验证", validation_metrics), ("测试", test_metrics)):
            rows.append(
                {
                    "split": name,
                    "precision": f"{m['precision_numerator']}/{m['precision_denominator']} = {m['precision']:.1%}",
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "accuracy": m["accuracy"],
                    "count": m["count"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "other_recall": m["other_recall"],
                    # Numeric copies for the WebUI comparison view: the formatted
                    # strings above cannot be diffed (parseFloat("12/15 = 80.0%") = 12).
                    "precision_value": round(m["precision"], 6),
                    "recall_value": round(m["recall"], 6),
                    "f1_value": round(m["f1"], 6),
                    "accuracy_value": round(m["accuracy"], 6),
                    "other_recall_value": round(m["other_recall"], 6),
                    "balanced_accuracy_value": round(m["balanced_accuracy"], 6),
                    "threshold": f"{current:.2f}",
                    "threshold_value": current,
                    "matrix": f"真“非{character}” → 非{character}/是{character}：{m['confusion_matrix'][0][0]}/{m['confusion_matrix'][0][1]}；真“{character}” → 非{character}/是{character}：{m['confusion_matrix'][1][0]}/{m['confusion_matrix'][1][1]}",
                }
            )
        readable = (
            f"角色「{character}」 · 所选第{stored_selection['epoch']}轮 · 总用时{report['total_seconds']:.1f}秒"
            f" · 训练（含验证）{report['training_seconds_including_validation']:.1f}秒\n{summary['评测性质']} · {summary['比较提示']}\n"
            f"指标按当前判定阈值 {current:.2f} 重算（训练时评测口径 {selection_threshold:.2f}）\n"
            f"核验：{'通过' if report['audit']['passed'] else '未通过'}"
        )
        for row_ in rows:
            for field in ("recall", "f1", "accuracy", "other_recall", "balanced_accuracy"):
                row_[field] = f"{row_[field]:.1%}"
        return {
            "summary": readable,
            "metrics": rows,
            "technical": json.dumps(summary, ensure_ascii=False),
            "threshold": {
                "current": current,
                "default": DEFAULT_THRESHOLD,
                "selection": selection_threshold,
                "band": JUDGMENT_BAND,
                "suggested": suggestion,
                "sweep": sweep_rows,
            },
        }

    @API("self_models", version="1")
    async def versions(self, character: str = "") -> dict:
        try:
            character = self._validate_character(character)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        rows = [r for r in await offload(self.service.store.models) if not character or r.get("character") in {"", character}]
        active = await offload(self.service.store.active, character)
        previous = await offload(self.service.store.previous, character)
        for r in rows:
            m = json.loads(r["metrics"])
            r.update(
                name=time.strftime("模型 %m-%d %H:%M", time.localtime(r["created"])),
                status="使用中" if r["id"] == active else "未启用",
                can_activate=r["id"] != active,
                input_size=m.get("input_size", 320),
            )
            for split in ("validation", "test"):
                metric = m[split]
                r[split + "_text"] = (
                    f"精确率 {metric['precision']:.1%}（{metric['precision_numerator']}/{metric['precision_denominator']}） · 召回 {metric['recall']:.1%} · F1 {metric['f1']:.1%} · 准确率 {metric['accuracy']:.1%} · {metric['count']}张"
                )
        return {
            "rows": rows,
            "character": character,
            "active": next((r["name"] for r in rows if r["id"] == active), "未启用"),
            "can_rollback": bool(previous),
            "rollback_text": next(("回退目标：" + r["name"] for r in rows if r["id"] == previous), "无可回退版本"),
            "threshold": await offload(self.service.store.threshold, character),
            "threshold_default": DEFAULT_THRESHOLD,
        }

    @API("self_threshold_apply", version="1")
    async def apply_threshold(self, character: str = "", value: str = "") -> dict:
        """Set the per-character online gate. Model weights are untouched."""
        try:
            character = self._validate_character(character)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        try:
            number = float(str(value).strip())
        except (TypeError, ValueError):
            return {"success": False, "error": "阈值必须是 0.05 到 0.95 之间的数字"}
        try:
            result = await offload(self.service.store.set_threshold, character, number)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        return {**result, "success": True}

    @API("self_threshold_reset", version="1")
    async def reset_threshold(self, character: str = "") -> dict:
        try:
            character = self._validate_character(character)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        return {**await offload(self.service.store.clear_threshold, character), "success": True}

    @API("self_model_activate", version="1")
    async def activate(self, version: str) -> dict:
        """Derive the owning character from the model record."""
        with self.service.store.db() as db:
            row = db.execute("SELECT character FROM models WHERE id=?", (version,)).fetchone()
        if row is None:
            return {"success": False, "error": "版本不存在"}
        character = row[0]
        if not character:
            return {"success": False, "error": "模型记录缺少角色信息，无法启用"}
        return await self._action(self.service.activate, character, version)

    @API("self_model_rollback", version="1")
    async def rollback(self, character: str = "") -> dict:
        return await self._action(self.service.rollback, character)

    @API("self_test_upload", version="1")
    async def test_upload(self, upload_id: str = "", result_id: str = "") -> dict:
        if result_id:
            raw = await offload(self.read_test_image, result_id)
            result = await offload(self.run_test, raw, result_id)
        else:
            claimed = await self.ctx.webui.claim_upload(upload_id)
            path = Path(claimed["path"])
            try:
                result = await offload(self.run_test, await offload(path.read_bytes), uuid.uuid4().hex)
            finally:
                await offload(path.unlink, missing_ok=True)
        return result

    def read_test_image(self, result_id: str) -> bytes:
        if len(result_id) != 32 or any(c not in "0123456789abcdef" for c in result_id):
            raise ValueError("测试图片不存在")
        history = json.loads(self.service.store.setting("test_history", "[]"))
        if not any(r.get("id") == result_id for r in history):
            raise ValueError("测试图片已过期")
        path = self.service.store.root / "test-images" / (result_id + ".image")
        if not path.is_file():
            raise ValueError("测试图片已过期，请重新上传")
        return path.read_bytes()

    def run_test(self, raw: bytes, result_id: str) -> dict:
        from .recog.images import thumbnail
        started = time.monotonic()
        result = self.service.infer(raw)
        directory = self.service.store.root / "test-images"
        directory.mkdir(exist_ok=True)
        path = directory / (result_id + ".image")
        path.write_bytes(raw)
        result.update(id=result_id, thumbnail=thumbnail(path), elapsed=f"耗时 {time.monotonic() - started:.2f} 秒")
        hits = [r for r in result.get("results", []) if r.get("status") == "ok"]
        if hits:
            result["summary"] = "；".join(
                f"{r['character']} · 得分 {r['score']:.3f}（{r['judgment']} · 阈值 {r.get('threshold', DEFAULT_THRESHOLD):.2f}）"
                for r in hits
            )
            result.update(select_top(hits, {}))
        else:
            result["summary"] = f"识别未完成：{result.get('error') or result.get('status')}"
        result["version_text"] = "模型版本：" + "；".join(f"{r['character']}:{r.get('version', '')[:8]}" for r in hits)
        self.save_test_history(result)
        self.service.store.set_setting("last_test", json.dumps(result, ensure_ascii=False))
        return result

    @API("self_test_result", version="1")
    async def test_result(self) -> dict:
        return {"rows": json.loads(await offload(self.service.store.setting, "test_history", "[]"))}

    def save_test_history(self, result):
        # Serialize history changes and retain the files needed for re-recognition.
        with self.service.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT value FROM settings WHERE key='test_history'").fetchone()
            history = json.loads(row[0]) if row else []
            history = [r for r in history if not result.get("id") or r.get("id") != result["id"]]
            combined = [result, *history]
            db.execute("INSERT OR REPLACE INTO settings VALUES ('test_history',?)", (json.dumps(combined[:24]),))
            for expired in combined[24:]:
                image_id = expired.get("id", "")
                if len(image_id) == 32 and all(c in "0123456789abcdef" for c in image_id):
                    (self.service.store.root / "test-images" / (image_id + ".image")).unlink(missing_ok=True)

    @staticmethod
    def _tool_result(result: dict) -> dict:
        return {**result, "name": "recognize_character_image", "content": json.dumps(result, ensure_ascii=False)}

    @Tool(
        "recognize_character_image",
        description="识别消息中的图片包含哪些（中V）角色；可指定单个角色，缺省返回全部已启用角色的得分。",
        parameters=[
            ToolParameterInfo(
                name="message_id", param_type=ToolParamType.STRING, description="当前聊天中的真实消息ID", required=True
            ),
            ToolParameterInfo(
                name="image_index",
                param_type=ToolParamType.INTEGER,
                description="消息图片/表情零基序号",
                required=False,
            ),
            ToolParameterInfo(
                name="character",
                param_type=ToolParamType.STRING,
                description="可选：只查询该角色的判定与得分",
                required=False,
            ),
        ],
    )
    async def recognize(self, message_id: str, image_index: int = 0, character: str = "", stream_id: str = "", **kwargs: Any) -> dict:
        del kwargs
        if not stream_id:
            return self._tool_result({"status": "missing_chat_context"})
        if type(image_index) is not int or image_index < 0:
            return self._tool_result({"status": "invalid_image_index"})
        if character:
            try:
                character = self._validate_character(character)
            except ValueError as exc:
                return self._tool_result({"status": "invalid_character", "error": str(exc)})
        message = await self.ctx.message.get_by_id(message_id, stream_id=stream_id, include_binary_data=True)
        if not message or message.get("session_id") != stream_id:
            return self._tool_result({"status": "message_not_found"})
        media = [part for part in message.get("raw_message", []) if part.get("type") in {"image", "emoji"}]
        if image_index >= len(media):
            return self._tool_result({"status": "image_not_found"})
        binary = media[image_index].get("binary_data_base64")
        if not binary:
            return self._tool_result({"status": "image_expired"})
        if len(binary) > 28 * 1024 * 1024:
            return self._tool_result({"status": "image_too_large"})
        try:
            raw = base64.b64decode(binary, validate=True)
        except ValueError:
            return self._tool_result({"status": "invalid_image"})
        result = await offload(self.service.infer, raw, 8, character)
        if character and result.get("results"):
            result = {**result, "results": [r for r in result["results"] if r.get("character") == character]}
        # Keep every score for debugging; add the same aggregation the prompt path
        # uses so the model can tell "one clear answer" from "too close to call".
        return self._tool_result({**result, **select_top(result.get("results", []), {})})


def create_plugin() -> CharacterRecognition:
    return CharacterRecognition()
