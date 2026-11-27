"""Pruebas de las herramientas de analisis de modelos: clasificacion selectiva, Grad-CAM,
costo, embeddings por lotes y conteo de detecciones."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lib.evaluation.classification import apply_confidence_threshold, confidence_threshold, risk_coverage
from lib.evaluation.cost import count_parameters, time_calls, torch_cpu_latency
from lib.evaluation.detection import count_table, detect_confidences
from lib.evaluation.gradcam import GradCAM, default_target_layer
from lib.models.cnn_custom import CustomCNN
from lib.models.inference import embed_batch_fn
from lib.models.resnet_classifier import ResNetClassifier

CLASSES = ["Afghan", "Beagle", "Boxer"]


def test_risk_coverage_and_threshold() -> None:
    # confianzas 0.9 (acierto), 0.8 (acierto), 0.6 (error), 0.5 (acierto)
    probs = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.6, 0.3, 0.1], [0.25, 0.25, 0.5]])
    y = np.array([0, 1, 1, 2])
    curve = risk_coverage(probs, y)
    assert curve["threshold"].tolist() == pytest.approx([0.5, 0.6, 0.8, 0.9])
    assert curve["coverage"].tolist() == pytest.approx([1.0, 0.75, 0.5, 0.25])
    assert curve["accuracy"].tolist() == pytest.approx([0.75, 2 / 3, 1.0, 1.0])
    best = confidence_threshold(probs, y, target_accuracy=0.95)
    assert best == {"threshold": pytest.approx(0.8), "coverage": 0.5, "accuracy": 1.0}
    assert confidence_threshold(probs, y, target_accuracy=0.7)["threshold"] == 0.0
    applied = apply_confidence_threshold(probs, y, 0.7)
    assert applied["coverage"] == 0.5 and applied["accuracy"] == 1.0 and applied["errors_rejected"] == 1.0


def test_risk_coverage_ties_accept_together() -> None:
    probs = np.array([[0.7, 0.3], [0.7, 0.3], [0.6, 0.4]])
    curve = risk_coverage(probs, np.array([0, 1, 0]))
    assert curve["accepted"].tolist() == [3, 2]  # las dos de 0.7 entran juntas


@pytest.mark.parametrize("model", [CustomCNN(CLASSES, channels=(8, 16), embedding_dim=16), ResNetClassifier(CLASSES, pretrained=False)])
def test_gradcam_map_and_hooks(model) -> None:
    model.train()
    x = torch.randn(1, 3, 64, 64)
    with GradCAM(model) as cam:
        heatmap, index, prob = cam(x)
    assert heatmap.shape == (64, 64) and 0.0 <= heatmap.min() and heatmap.max() <= 1.0
    assert 0 <= index < 3 and 0.0 < prob <= 1.0
    assert model.training  # restaura el modo
    layer = default_target_layer(model)
    assert not layer._forward_hooks and not layer._backward_hooks  # quita los hooks
    with pytest.raises(ValueError):
        GradCAM(model)(torch.randn(2, 3, 64, 64))


def test_gradcam_works_with_frozen_backbone() -> None:
    model = ResNetClassifier(CLASSES, pretrained=False)
    model.freeze_backbone()
    with GradCAM(model) as cam:
        heatmap, _, _ = cam(torch.randn(1, 3, 64, 64), class_index=2)
    assert heatmap.shape == (64, 64)


def test_cost_helpers() -> None:
    model = CustomCNN(CLASSES, channels=(8, 16), embedding_dim=16)
    assert count_parameters(model) == sum(p.numel() for p in model.parameters())
    stats = torch_cpu_latency(model, 32, warmup=1, repeats=3)
    assert stats["ms_median"] > 0 and stats["ms_p90"] >= stats["ms_median"] * 0.5
    with pytest.raises(ValueError):
        time_calls(lambda: None, repeats=0)


def test_embed_batch_fn_is_normalized() -> None:
    model = CustomCNN(CLASSES, channels=(8, 16), embedding_dim=16)
    images = [np.random.default_rng(i).integers(0, 255, (40, 60, 3), dtype=np.uint8) for i in range(3)]
    emb = embed_batch_fn(model, 32)(images)
    assert emb.shape == (3, 16) and emb.dtype == np.float32
    assert np.linalg.norm(emb, axis=1) == pytest.approx(np.ones(3), abs=1e-5)


class _FakeBoxes:
    def __init__(self, conf, cls):
        self.conf, self.cls = torch.tensor(conf), torch.tensor(cls)

    def __len__(self):
        return len(self.conf)


class _FakeYolo:
    def __init__(self, outputs):
        self.outputs, self.calls = iter(outputs), []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return [type("R", (), {"boxes": next(self.outputs)})()]


def test_detection_counts() -> None:
    yolo = _FakeYolo([_FakeBoxes([0.9, 0.3], [16, 16]), _FakeBoxes([0.2], [16]), None])
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    confs, ms = detect_confidences(yolo, [image] * 3, imgsz=320)
    assert yolo.calls[0]["imgsz"] == 320 and yolo.calls[0]["classes"] == [16]
    assert [len(c) for c in confs] == [2, 1, 0] and ms >= 0
    table = count_table(confs, n_dogs=[2, 0, 1], groups=["varios", "sin perros", "uno"], thresholds=[0.25, 0.5])
    assert table.loc[(0.25, "varios"), "exact_count"] == 1.0
    assert table.loc[(0.5, "varios"), "count_mae"] == 1.0
    assert table.loc[(0.25, "sin perros"), "false_alarm"] == 0.0
    assert table.loc[(0.25, "uno"), "detection_rate"] == 0.0
