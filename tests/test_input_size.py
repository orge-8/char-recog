"""Input size belongs to each snapshot and saved model."""
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="输入尺寸用例需要 PyTorch")
pytest.importorskip("torchvision", reason="输入尺寸用例需要 torchvision")

from PIL import Image  # noqa: E402
import torch  # noqa: E402
from recog.model import Dataset, fresh_model, infer, loaded_model, train  # noqa: E402
from test_catalog import CHAR, corpus  # noqa: E402


def test_resize_precedes_augmentation(tmp_path):
    Image.new("RGB", (1600, 900), "orange").save(tmp_path / "image.png")
    dataset = Dataset(tmp_path, [{"path": "image.png", "label": "self"}], True, 448)
    sizes = []
    def jitter(image):
        sizes.append(image.size)
        return image
    dataset.jitter = jitter
    tensor, label = dataset[0]
    assert sizes == [(448, 448)]
    assert tensor.shape == (3, 448, 448) and label == 1
    legacy = Dataset(tmp_path, dataset.rows, True, 320, False)
    legacy.jitter = jitter
    legacy[0]
    assert sizes[-1] == (1600, 900)


@pytest.mark.parametrize("size", [224, 448])
def test_size_survives_training_reload_and_inference(tmp_path, size):
    store = corpus(tmp_path)
    with store.db() as db:
        snapshot = store.freeze(db, "size", "fresh", "cpu", CHAR, size)
    assert snapshot["input_size"] == size
    assert snapshot["preprocessing"] == "resize_before_augment"
    directory = tmp_path / "jobs" / "size"
    directory.mkdir()
    report = train(tmp_path, snapshot, lambda p: None, lambda: False, max_epochs=1)
    assert report["audit"]["passed"] and report["input_size"] == size
    model = loaded_model(directory / "selected.pt")
    assert model.input_size == size
    seen = []
    hook = model.register_forward_pre_hook(lambda model, args: seen.append(tuple(args[0].shape)))
    raw = (tmp_path / snapshot["images"][0]["path"]).read_bytes()
    assert 0 <= infer(model, raw) <= 1
    hook.remove()
    assert seen == [(1, 3, size, size)]


def test_legacy_weights_use_320(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save(fresh_model().state_dict(), path)
    assert loaded_model(path).input_size == 320


def test_invalid_size_rejected(tmp_path):
    store = corpus(tmp_path)
    with store.db() as db, pytest.raises(ValueError, match="224"):
        store.freeze(db, "invalid", "fresh", "cpu", CHAR, 999)


def test_training_batch_may_contain_only_one_class(tmp_path):
    store = corpus(tmp_path)
    with store.db() as db:
        snapshot = store.freeze(db, "oneclass", "fresh", "cpu", CHAR, 224)
    snapshot["images"] = [r for r in snapshot["images"] if r["split"] != "train" or r["label"] == "self"]
    (tmp_path / "jobs" / "oneclass").mkdir()
    report = train(tmp_path, snapshot, lambda p: None, lambda: False, max_epochs=1)
    assert report["audit"]["passed"]
    assert report["counts"]["train"] == 14


def test_strong_augmentation_keeps_shape_and_trains(tmp_path):
    """随机裁剪 + 翻转必须产出张量：torchvision 的 RandomResizedCrop 对 PIL 输入
    返回的仍是 PIL，直接送 Normalize 会报 TypeError（0.5.0 开发时踩过）。"""
    Image.new("RGB", (900, 600), "orange").save(tmp_path / "image.png")
    rows = [{"path": "image.png", "label": "self"}]
    strong = Dataset(tmp_path, rows, True, 320, True, "strong")
    tensor, label = strong[0]
    assert tensor.shape == (3, 320, 320) and label == 1
    assert strong.augmentation == "strong"
    assert strong.strong.transforms[0].scale == (0.6, 1.0)

    legacy = Dataset(tmp_path, rows, True, 320, True, "jitter")
    assert legacy[0][0].shape == (3, 320, 320)

    # 端到端：整轮训练 + 审计在 strong 档位下必须通过。
    store = corpus(tmp_path)
    with store.db() as db:
        snapshot = store.freeze(db, "aug", "fresh", "cpu", CHAR, 320)
    assert snapshot["augmentation"] == "strong"
    directory = tmp_path / "jobs" / "aug"
    directory.mkdir()
    report = train(tmp_path, snapshot, lambda p: None, lambda: False, max_epochs=1)
    assert report["audit"]["passed"]


def test_legacy_snapshot_without_augmentation_key_stays_on_jitter(tmp_path):
    Image.new("RGB", (640, 480), "orange").save(tmp_path / "image.png")
    dataset = Dataset(tmp_path, [{"path": "image.png", "label": "other"}], True, 224, True,
                      {}.get("augmentation", "jitter"))
    assert dataset.augmentation == "jitter"
    assert dataset[0][0].shape == (3, 224, 224)
