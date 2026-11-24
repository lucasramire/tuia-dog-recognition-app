"""Tests de la Etapa 1 (similarity_service.py).

extract_embedding usa la ResNet50 real: la primera vez descarga ~100 MB de pesos
en models/torch (ignorado por git).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lib.schemas import EmbeddingRecord, Neighbor
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


# --- search_similar_images (E1.2) ---------------------------------------------


def _record(path: str, breed: str, vector: list[float]) -> EmbeddingRecord:
    return EmbeddingRecord(id_imagen=path, embedding=vector, path=path, breed=breed)


# Consulta [1, 0]. Por coseno: a (1.0) > b (0.8) > c (0.0) > d (-1.0).
# d esta lejos en angulo pero, sin normalizar, a distinta distancia euclidea que c.
RECORDS = [
    _record("c.jpg", "Boxer", [0.0, 1.0]),
    _record("a.jpg", "Afghan", [2.0, 0.0]),
    _record("d.jpg", "Pug", [-1.0, 0.0]),
    _record("b.jpg", "Afghan", [0.8, 0.6]),
]
QUERY = [1.0, 0.0]


class _JsonLikeStore:
    """Como EmbeddingStore: solo all()."""

    def __init__(self, records: list[EmbeddingRecord]) -> None:
        self.records = records

    def all(self) -> list[EmbeddingRecord]:
        return list(self.records)

    def append(self, record: EmbeddingRecord) -> None:
        self.records.append(record)


class _PgLikeStore(_JsonLikeStore):
    """Como PgVectorEmbeddingStore: search() ordena por coseno en la base; all() no se usa."""

    def __init__(self, records: list[EmbeddingRecord]) -> None:
        super().__init__(records)
        self.search_calls: list[int] = []

    def all(self) -> list[EmbeddingRecord]:
        raise AssertionError("con pgvector no se debe traer toda la base")

    def search(self, query: list[float], k: int = 10) -> list[EmbeddingRecord]:
        self.search_calls.append(k)
        q = np.asarray(query)

        def cos(r: EmbeddingRecord) -> float:
            v = np.asarray(r.embedding)
            return float(q @ v / (np.linalg.norm(q) * np.linalg.norm(v)))

        return sorted(self.records, key=cos, reverse=True)[:k]


def _with_store(store, metric: str = "cosine", threshold: float = 0.55) -> SimilarityService:
    service = _service(metric, threshold)
    service.store = store
    return service


@pytest.mark.parametrize("store_cls", [_JsonLikeStore, _PgLikeStore])
def test_search_cosine_order_and_scores(store_cls) -> None:
    service = _with_store(store_cls(RECORDS))
    neighbors = service.search_similar_images(QUERY, 3)
    assert [n.path for n in neighbors] == ["a.jpg", "b.jpg", "c.jpg"]
    assert [n.score for n in neighbors] == pytest.approx([1.0, 0.8, 0.0], abs=1e-6)
    assert all(isinstance(n, Neighbor) and n.url is None for n in neighbors)


def test_search_l2_uses_l2_score_and_order() -> None:
    service = _with_store(_JsonLikeStore(RECORDS), metric="l2")
    neighbors = service.search_similar_images(QUERY, 4)
    # distancias: b 0.632, a 1.0, c 1.414, d 2.0 -> score 1 / (1 + d)
    assert [n.path for n in neighbors] == ["b.jpg", "a.jpg", "c.jpg", "d.jpg"]
    expected = [1 / (1 + np.linalg.norm(np.subtract(QUERY, r))) for r in ([0.8, 0.6], [2, 0], [0, 1], [-1, 0])]
    assert [n.score for n in neighbors] == pytest.approx(expected, abs=1e-6)


def test_search_with_normalized_vectors_ranks_equal_for_cosine_and_l2() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(30, 8))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    records = [_record(f"{i}.jpg", "X", v.tolist()) for i, v in enumerate(vectors)]
    query = vectors[0] + 0.1 * rng.normal(size=8)
    query = (query / np.linalg.norm(query)).tolist()
    cosine = _with_store(_JsonLikeStore(records)).search_similar_images(query, 10)
    l2 = _with_store(_JsonLikeStore(records), metric="l2").search_similar_images(query, 10)
    assert [n.path for n in cosine] == [n.path for n in l2]


def test_search_pgvector_uses_store_search_with_top_k() -> None:
    store = _PgLikeStore(RECORDS)
    _with_store(store).search_similar_images(QUERY, 2)
    assert store.search_calls == [2]


def test_search_top_k_larger_than_base_and_edge_cases() -> None:
    service = _with_store(_JsonLikeStore(RECORDS))
    assert len(service.search_similar_images(QUERY, 50)) == len(RECORDS)
    assert service.search_similar_images(QUERY, 0) == []
    assert _with_store(_JsonLikeStore([])).search_similar_images(QUERY, 5) == []
    with pytest.raises(ValueError):
        service.search_similar_images([], 5)


def test_search_dimension_mismatch_raises_clear_error() -> None:
    service = _with_store(_JsonLikeStore(RECORDS))
    with pytest.raises(ValueError, match="dimensiones"):
        service.search_similar_images([1.0, 0.0, 0.0], 3)


# --- predict_breed_from_neighbors (E1.3) --------------------------------------


def _n(breed: str, score: float) -> Neighbor:
    return Neighbor(path=f"{breed}-{score}.jpg", breed=breed, score=score)


def test_predict_empty_is_unknown() -> None:
    assert _service().predict_breed_from_neighbors([]) == ("unknown", 0.0)


def test_predict_below_threshold_is_unknown() -> None:
    breed, score = _service(threshold=0.6).predict_breed_from_neighbors([_n("Pug", 0.59), _n("Pug", 0.5)])
    assert breed == "unknown"
    assert score == pytest.approx(0.59)


def test_predict_threshold_is_inclusive() -> None:
    assert _service(threshold=0.6).predict_breed_from_neighbors([_n("Pug", 0.6)])[0] == "Pug"


def test_predict_weighted_vote_beats_simple_majority() -> None:
    # Por cantidad gana Boxer (2 contra 1); por peso gana Afghan (0.95 contra 0.80).
    neighbors = [_n("Afghan", 0.95), _n("Boxer", 0.40), _n("Boxer", 0.40)]
    breed, score = _service(threshold=0.3).predict_breed_from_neighbors(neighbors)
    assert breed == "Afghan"
    assert score == pytest.approx(0.95)


def test_predict_returns_mean_score_of_winner() -> None:
    neighbors = [_n("Beagle", 0.9), _n("Beagle", 0.7), _n("Pug", 0.8)]
    breed, score = _service().predict_breed_from_neighbors(neighbors)
    assert breed == "Beagle"
    assert score == pytest.approx(0.8)


def test_predict_tie_is_deterministic() -> None:
    a = [_n("Pug", 0.8), _n("Beagle", 0.8)]
    assert _service().predict_breed_from_neighbors(a) == _service().predict_breed_from_neighbors(a[::-1])


def test_predict_negative_scores_do_not_vote() -> None:
    neighbors = [_n("Pug", 0.6), _n("Boxer", -0.9), _n("Boxer", -0.9)]
    assert _service(threshold=0.5).predict_breed_from_neighbors(neighbors)[0] == "Pug"
