"""Prueba de humo del entrenamiento con un dataset sintetico chico (sin descargar pesos)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from lib.evaluation.classification import classification_metrics, confusion_matrix
from lib.models.base import load_checkpoint
from lib.models.cnn_custom import CustomCNN
from lib.models.resnet_classifier import ResNetClassifier
from lib.services.classifier_service import ClassifierService
from lib.training.trainer import TrainConfig, train

CLASSES = ["Afghan", "Beagle", "Boxer"]
# Cada raza tiene un color distinto: un problema que hasta una CNN chica aprende.
COLORS = {"Afghan": (30, 30, 220), "Beagle": (30, 220, 30), "Boxer": (220, 30, 30)}
SIZE = 32


def _make_dataset(root: Path, per_split: dict[str, int]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for split, n in per_split.items():
        for breed in CLASSES:
            for i in range(n):
                image = np.clip(
                    np.array(COLORS[breed], dtype=np.int16) + rng.integers(-20, 20, (40, 48, 3)), 0, 255
                ).astype(np.uint8)
                rel = f"{split}/{breed}/{i:03d}.jpg"
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                ok, buf = cv2.imencode(".jpg", image)
                assert ok
                buf.tofile(str(root / rel))
                rows.append((rel, breed, split))
    return pd.DataFrame(rows, columns=["path", "breed", "split"])


def _config(**kwargs) -> TrainConfig:
    base = dict(
        model="cnn_custom", epochs=3, batch_size=8, image_size=SIZE, cnn_channels=(8, 16),
        cnn_embedding_dim=16, lr=5e-3, patience=5, device="cpu",
    )
    base.update(kwargs)
    return TrainConfig(**base)


def test_classification_metrics_known_values() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    m = classification_metrics(y_true, y_pred, 3)
    assert m["accuracy"] == pytest.approx(4 / 6)
    # precision por clase: 1/2, 2/3, 1/1; recall: 1/2, 1, 1/2
    assert m["precision"] == pytest.approx((0.5 + 2 / 3 + 1.0) / 3)
    assert m["recall"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    # especificidad por clase: 3/4, 3/4, 4/4
    assert m["specificity"] == pytest.approx((0.75 + 0.75 + 1.0) / 3)


def test_models_follow_contract() -> None:
    x = torch.randn(2, 3, SIZE, SIZE)
    cnn = CustomCNN(CLASSES, embedding_dim=16, channels=(8, 16)).eval()
    assert cnn.embed(x).shape == (2, 16) and cnn(x).shape == (2, 3)
    resnet = ResNetClassifier(CLASSES, pretrained=False).eval()
    assert resnet.embed(x).shape == (2, 512) and resnet(x).shape == (2, 3)


def test_feature_extraction_freezes_backbone_and_its_batchnorm() -> None:
    model = ResNetClassifier(CLASSES, pretrained=False)
    model.freeze_backbone()
    model.train()
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    assert not model.backbone.training and model.head.training
    assert [g["name"] for g in model.param_groups(1e-3, 1e-4)] == ["head"]


def test_train_saves_best_checkpoint_and_history(tmp_path: Path) -> None:
    manifest = _make_dataset(tmp_path / "data", {"train": 8, "valid": 3, "test": 3})
    result = train(_config(), manifest, tmp_path / "data", tmp_path / "m.pth", tmp_path / "h.json")
    model = load_checkpoint(result.checkpoint_path)
    assert model.classes == CLASSES and model.embedding_dim == 16
    history = json.loads(result.history_path.read_text(encoding="utf-8"))
    assert len(history["history"]) == result.epochs_run == 3
    assert history["best_epoch"] == result.best_epoch
    assert history["best_val_accuracy"] == max(e["val_accuracy"] for e in history["history"])


def test_early_stopping(tmp_path: Path) -> None:
    manifest = _make_dataset(tmp_path / "data", {"train": 4, "valid": 2, "test": 2})
    result = train(_config(epochs=20, patience=1, lr=0.0), manifest, tmp_path / "data", tmp_path / "m.pth", tmp_path / "h.json")
    assert result.epochs_run < 20


def test_service_train_evaluate_embed(tmp_path: Path) -> None:
    data = tmp_path / "data"
    manifest = _make_dataset(data / "dataset", {"train": 10, "valid": 3, "test": 3})
    (data / "splits").mkdir(parents=True)
    manifest.to_csv(data / "splits" / "splits_v1.csv", index=False)
    service = ClassifierService(
        checkpoints={"resnet18_finetuned": tmp_path / "models" / "a.pth", "cnn_custom": tmp_path / "models" / "b.pth"},
        image_size=SIZE,
        dataset_path=data / "dataset",
        output_path=tmp_path / "out",
    )
    result = service.train_classifier(_config(epochs=6))
    assert service.active_model_name == "cnn_custom"
    assert service.history_path().exists() and result.checkpoint_path == tmp_path / "models" / "b.pth"

    metrics = service.evaluate_classifier()
    assert set(metrics) == {"accuracy", "precision", "recall", "specificity", "f1"}
    assert metrics["accuracy"] > 1 / 3  # aprende algo con colores separables

    image = np.full((40, 40, 3), COLORS["Beagle"], dtype=np.uint8)
    vector = np.array(service.extract_custom_embedding(image))
    assert vector.shape == (16,)
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_invalid_config_raises() -> None:
    with pytest.raises(ValueError):
        _config(model="otro").validate()
    with pytest.raises(ValueError):
        _config(model="resnet18_finetuned", strategy="otra").validate()
    with pytest.raises(ValueError):
        _config(augmentation={"no_existe": 1}).validate()


def test_cnn_input_stats_default_is_identity() -> None:
    x = torch.randn(2, 3, SIZE, SIZE)
    base = CustomCNN(CLASSES, embedding_dim=16, channels=(8, 16)).eval()
    torch.manual_seed(0)
    shifted = CustomCNN(CLASSES, embedding_dim=16, channels=(8, 16), input_mean=[1.0, 2.0, 3.0], input_std=[2.0, 2.0, 2.0]).eval()
    shifted.load_state_dict(base.state_dict() | {"input_mean": shifted.input_mean, "input_std": shifted.input_std})
    expected = base.embed((x - torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1)) / 2.0)
    assert torch.allclose(shifted.embed(x), expected, atol=1e-5)
    with pytest.raises(ValueError):
        CustomCNN(CLASSES, input_mean=[0.0, 0.0], input_std=[1.0, 1.0])


def test_train_stats_and_custom_augmentation(tmp_path: Path) -> None:
    manifest = _make_dataset(tmp_path / "data", {"train": 4, "valid": 2, "test": 2})
    config = _config(epochs=1, cnn_train_stats=True, augmentation={"crop_border": 0.25, "hflip_p": 1.0})
    result = train(config, manifest, tmp_path / "data", tmp_path / "m.pth", tmp_path / "h.json")
    history = json.loads(result.history_path.read_text(encoding="utf-8"))
    mean, std = history["input_stats"]
    assert len(mean) == 3 and all(s > 0 for s in std)
    model = load_checkpoint(result.checkpoint_path)
    assert torch.allclose(model.input_mean.flatten(), torch.tensor(mean, dtype=torch.float32))
    assert history["config"]["augmentation"] == {"crop_border": 0.25, "hflip_p": 1.0}


def test_confused_pairs_mcnemar_bootstrap() -> None:
    from lib.evaluation.classification import bootstrap_ci, mcnemar_test, top_confused_pairs

    cm = np.array([[5, 3, 0], [1, 6, 0], [0, 2, 4]])
    pairs = top_confused_pairs(cm, CLASSES, k=5)
    assert pairs[0] == {"breed_a": "Afghan", "breed_b": "Beagle", "a_as_b": 3, "b_as_a": 1, "errors": 4}
    assert [p["errors"] for p in pairs] == [4, 2]

    a = np.array([1] * 10 + [0] * 0, dtype=bool)
    b = np.array([0] * 10, dtype=bool)
    result = mcnemar_test(a, b)
    assert result["only_a"] == 10 and result["only_b"] == 0
    assert result["p_value"] == pytest.approx(2 / 2**10)
    assert mcnemar_test(a, a)["p_value"] == 1.0

    mean, low, high = bootstrap_ci(np.array([1, 0] * 50))
    assert mean == 0.5 and low < 0.5 < high
