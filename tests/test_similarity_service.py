"""Tests de la Etapa 1 (similarity_service.py).

extract_embedding usa la ResNet50 real: la primera vez descarga ~100 MB de pesos
en models/torch (ignorado por git).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lib.services.similarity_service import SimilarityService

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "dataset"
IMAGE_SIZE = 224


class _EmptyStore:
    def all(self) -> list:
        return []

    def append(self, record) -> None:
        pass


def _service(metric: str = "cosine", threshold: float = 0.55) -> SimilarityService:
    return SimilarityService(
        store=_EmptyStore(),
        similarity_metric=metric,
        similarity_threshold=threshold,
        top_k=10,
        image_size=IMAGE_SIZE,
        model_name="baseline",
    )


@pytest.fixture(scope="module")
def service() -> SimilarityService:
    # TORCH_HOME absoluto: el de src/.env es relativo a src/ y desde la raiz apuntaria afuera del repo.
    mp = pytest.MonkeyPatch()
    mp.setenv("TORCH_HOME", str(ROOT / "models" / "torch"))
    yield _service()
    mp.undo()


def _random_image(seed: int, height: int = 180, width: int = 240) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def _dataset_image(relative: str) -> np.ndarray:
    import cv2

    path = DATASET / relative
    if not path.exists():
        pytest.skip(f"Dataset no descargado: falta {path}")
    image = cv2.imread(str(path))
    assert image is not None
    return image


# --- extract_embedding (E1.1) -------------------------------------------------


def test_embedding_dimension_type_and_norm(service: SimilarityService) -> None:
    embedding = service.extract_embedding(_random_image(0))
    assert isinstance(embedding, list)
    assert len(embedding) == 2048
    assert all(type(v) is float for v in embedding)
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)


def test_embedding_is_deterministic_and_model_loaded_once(service: SimilarityService) -> None:
    image = _random_image(1)
    first = service.extract_embedding(image)
    embedder = service._get_baseline_embedder()
    second = service.extract_embedding(image)
    assert service._get_baseline_embedder() is embedder
    np.testing.assert_allclose(first, second, atol=1e-6)


def test_input_image_is_not_modified(service: SimilarityService) -> None:
    image = _random_image(2)
    copy = image.copy()
    service.extract_embedding(image)
    np.testing.assert_array_equal(image, copy)


def test_service_matches_batch_extraction(service: SimilarityService) -> None:
    # La notebook extrae en lotes con el mismo BaselineEmbedder: debe dar lo mismo que la app.
    images = [_random_image(3), _random_image(4, 120, 300)]
    batch = service._get_baseline_embedder().embed_batch(images)
    assert batch.shape == (2, 2048)
    for image, row in zip(images, batch):
        np.testing.assert_allclose(service.extract_embedding(image), row, atol=1e-4)


def test_same_breed_is_closer_than_other_breed(service: SimilarityService) -> None:
    a1 = service.extract_embedding(_dataset_image("train/Afghan/001.jpg"))
    a2 = service.extract_embedding(_dataset_image("train/Afghan/002.jpg"))
    other = service.extract_embedding(_dataset_image("train/Yorkie/01.jpg"))
    assert service.similarity(a1, a2) > service.similarity(a1, other)


def test_invalid_image_raises(service: SimilarityService) -> None:
    with pytest.raises(ValueError):
        service.extract_embedding(np.zeros((10, 10), dtype=np.uint8))
