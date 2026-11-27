from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lib.models.base import save_checkpoint
from lib.models.dummy import DummyEmbeddingClassifier
from lib.services.classifier_service import ClassifierService
from lib.services.detection_service import UNKNOWN_BREED, DetectionService

CLASSES = ["Afghan", "Beagle", "Boxer"]
SIZE = 32


class _FakeYolo:
    """Doble de ultralytics.YOLO: devuelve cajas fijas y registra los argumentos."""

    def __init__(self, xyxy: list[list[float]], conf: list[float], cls: list[int]) -> None:
        self.boxes = SimpleNamespace(
            xyxy=torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
            conf=torch.tensor(conf, dtype=torch.float32),
            cls=torch.tensor(cls, dtype=torch.float32),
        )
        self.calls: list[dict] = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(boxes=_Boxes(self.boxes))]


class _Boxes:
    def __init__(self, ns) -> None:
        self.xyxy, self.conf, self.cls = ns.xyxy, ns.conf, ns.cls

    def __len__(self) -> int:
        return int(self.conf.shape[0])


def _service(tmp_path: Path, fmt: str = "pth", min_confidence: float = 0.0) -> DetectionService:
    model = DummyEmbeddingClassifier(CLASSES, embedding_dim=8, seed=0)
    if fmt == "pth":
        path = tmp_path / "m.pth"
        save_checkpoint(model, path)
    else:
        from lib.models.export import export_onnx

        path = export_onnx(model, tmp_path / "m.onnx", image_size=SIZE)
    classifier = ClassifierService(
        checkpoints={"resnet18_finetuned": path},
        image_size=SIZE,
        dataset_path=tmp_path,
        output_path=tmp_path,
    )
    return DetectionService(
        classifier=classifier,
        yolo_model="no-se-usa.pt",
        conf_threshold=0.25,
        dog_class_id=16,
        iou_threshold=0.5,
        min_confidence=min_confidence,
    )


def _image(h: int = 100, w: int = 200) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_detect_dogs_ints_clipped_sorted_and_filtered(tmp_path: Path) -> None:
    service = _service(tmp_path)
    fake = _FakeYolo(
        xyxy=[[10.4, 20.6, 50.2, 80.0], [-5.0, -3.0, 250.0, 130.0], [1, 1, 5, 5]],
        conf=[0.6, 0.9, 0.8],
        cls=[16, 16, 15],  # la tercera es un gato: se descarta
    )
    service._yolo = fake
    detections = service.detect_dogs(_image())
    assert [d[1] for d in detections] == pytest.approx([0.9, 0.6])
    (box_a, _), (box_b, _) = detections
    assert box_a == (0, 0, 200, 100)  # recortada a la imagen
    assert box_b == (10, 21, 50, 80)
    assert all(isinstance(v, int) for box, _ in detections for v in box)
    assert fake.calls[0]["iou"] == 0.5 and fake.calls[0]["conf"] == 0.25
    assert fake.calls[0]["classes"] == [16]


def test_detect_dogs_imgsz(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service._yolo = _FakeYolo(xyxy=[], conf=[], cls=[])
    service.detect_dogs(_image())
    assert "imgsz" not in service._yolo.calls[0]  # sin imgsz: el valor por defecto de ultralytics
    service.imgsz = 320
    service.detect_dogs(_image())
    assert service._yolo.calls[1]["imgsz"] == 320
    with pytest.raises(ValueError):
        DetectionService(service.classifier, "x.pt", 0.25, 16, 0.5, 0.0, imgsz=0)


def test_detect_dogs_empty(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service._yolo = _FakeYolo(xyxy=[], conf=[], cls=[])
    assert service.detect_dogs(_image()) == []


def test_detect_dogs_invalid_image(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(ValueError):
        service.detect_dogs(np.zeros((10, 10), dtype=np.uint8))


def test_classify_pth(tmp_path: Path) -> None:
    service = _service(tmp_path)
    breed, score = service.classify_detected_dog(_image(40, 30))
    assert breed in CLASSES
    assert 1 / len(CLASSES) <= score <= 1.0


def test_classify_unknown_below_threshold(tmp_path: Path) -> None:
    service = _service(tmp_path, min_confidence=1.01)
    breed, score = service.classify_detected_dog(_image(40, 30))
    assert breed == UNKNOWN_BREED
    assert 0.0 < score <= 1.0


def test_classify_onnx_matches_pth(tmp_path: Path) -> None:
    crop = _image(40, 30)
    pth = _service(tmp_path / "a").classify_detected_dog(crop)
    onnx_result = _service(tmp_path / "b", fmt="onnx").classify_detected_dog(crop)
    assert onnx_result[0] == pth[0]
    assert onnx_result[1] == pytest.approx(pth[1], abs=1e-4)
