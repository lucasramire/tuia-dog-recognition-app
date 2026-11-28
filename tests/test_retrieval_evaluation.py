"""Tests de la evaluacion de la Etapa 1 (lib/evaluation/retrieval.py) y de la regla de voto.

Vectores sinteticos: no hacen falta pesos ni el dataset.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from lib.evaluation.metrics import ndcg_at_k
from lib.evaluation.retrieval import (
    calibration_masks,
    compare_metrics,
    evaluate_queries,
    l2_normalize,
    load_or_compute_embeddings,
    operating_point,
    per_breed_table,
    rank_gallery,
    rates_at,
    similarity_matrix,
    threshold_sweep,
    top_k_sweep,
)
from lib.schemas import Neighbor
from lib.search.voting import UNKNOWN_BREED, predict_breed
from lib.services.similarity_service import SimilarityService


class _EmptyStore:
    def all(self) -> list:
        return []

    def append(self, record) -> None:
        pass


def _service(metric: str = "cosine", threshold: float = 0.55) -> SimilarityService:
    return SimilarityService(
        store=_EmptyStore(), similarity_metric=metric, similarity_threshold=threshold,
        top_k=10, image_size=224, model_name="baseline",
    )


def _random(n: int, d: int = 16, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


# --- voto -------------------------------------------------------------------


def test_predict_breed_empty_and_threshold():
    assert predict_breed([], []) == (UNKNOWN_BREED, 0.0)
    assert predict_breed(["Pug"], [0.4], threshold=0.5) == (UNKNOWN_BREED, 0.4)
    assert predict_breed(["Pug"], [0.4], threshold=None)[0] == "Pug"


def test_predict_breed_weighted_vote_and_mean_score():
    breed, score = predict_breed(["Pug", "Beagle", "Beagle"], [0.9, 0.5, 0.35])
    assert breed == "Pug"  # 0,9 pesa mas que 0,5 + 0,35
    assert score == pytest.approx(0.9)
    breed, score = predict_breed(["Pug", "Beagle", "Beagle"], [0.9, 0.5, 0.5])
    assert breed == "Beagle"
    assert score == pytest.approx(0.5)


def test_predict_breed_lengths_must_match():
    with pytest.raises(ValueError):
        predict_breed(["Pug"], [0.1, 0.2])


# --- similitud y ranking ------------------------------------------------------


@pytest.mark.parametrize("metric", ["cosine", "l2"])
def test_similarity_matrix_matches_service(metric):
    q, g = _random(4, seed=1) * 3.0, _random(6, seed=2) * 0.5  # crudos, normas distintas
    sim = similarity_matrix(q, g, metric)
    service = _service(metric)
    expected = np.array([[service.similarity(a.tolist(), b.tolist()) for b in g] for a in q])
    np.testing.assert_allclose(sim, expected, rtol=1e-5, atol=1e-6)


def test_similarity_matrix_rejects_bad_input():
    with pytest.raises(ValueError):
        similarity_matrix(_random(2, 4), _random(2, 5))
    with pytest.raises(ValueError):
        similarity_matrix(_random(2), _random(2), "manhattan")


def test_l2_normalize_keeps_zero_rows():
    x = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(l2_normalize(x), [[0.6, 0.8], [0.0, 0.0]])


def test_rank_gallery_descending_and_stable():
    sim = np.array([[0.1, 0.9, 0.5, 0.9]])
    assert rank_gallery(sim, 3).tolist() == [[1, 3, 2]]  # empate 0,9: primero el indice menor
    assert rank_gallery(sim, 10).shape == (1, 4)


def test_normalized_cosine_and_l2_rank_equal_raw_l2_can_differ():
    rng = np.random.default_rng(3)
    q, g = _random(20, seed=4), _random(50, seed=5) * rng.uniform(0.2, 5.0, size=(50, 1)).astype(np.float32)
    qn, gn = l2_normalize(q), l2_normalize(g)
    assert np.array_equal(rank_gallery(similarity_matrix(qn, gn, "cosine"), 10), rank_gallery(similarity_matrix(qn, gn, "l2"), 10))
    assert not np.array_equal(rank_gallery(similarity_matrix(q, g, "cosine"), 10), rank_gallery(similarity_matrix(q, g, "l2"), 10))


# --- metricas por consulta ----------------------------------------------------


def _toy():
    gallery_breeds = ["A", "A", "A", "B", "B", "C"]
    query_breeds = ["A", "B"]
    sim = np.array(
        [
            [0.9, 0.8, 0.1, 0.7, 0.2, 0.3],  # A: vecinos A, A, B, C, B, A
            [0.1, 0.2, 0.3, 0.4, 0.95, 0.5],  # B: vecinos B, C, B, A, A, A
        ],
        dtype=np.float32,
    )
    return sim, query_breeds, gallery_breeds


def test_evaluate_queries_ndcg_precision_and_prediction():
    sim, qb, gb = _toy()
    ev = evaluate_queries(sim, qb, gb, ndcg_k=3, vote_k=3)
    row_a, row_b = ev.per_query.iloc[0], ev.per_query.iloc[1]
    assert ev.neighbors[0].tolist() == [0, 1, 3]
    assert row_a["ndcg"] == pytest.approx(ndcg_at_k([1, 1, 0], k=3, n_relevant=3))
    assert row_a["precision"] == pytest.approx(2 / 3)
    assert row_a["predicted_breed"] == "A" and row_a["correct"]
    assert row_b["n_relevant"] == 2
    assert row_b["ndcg"] == pytest.approx(ndcg_at_k([1, 0, 1], k=3, n_relevant=2))
    assert row_b["best_score"] == pytest.approx(0.95)


def test_evaluate_queries_matches_service_prediction():
    """La raza que mide la notebook es la que devolveria la app con los mismos vecinos."""
    q, g = _random(30, seed=6), _random(80, seed=7)
    gb = [f"r{i % 7}" for i in range(80)]
    qb = [f"r{i % 7}" for i in range(30)]
    sim = similarity_matrix(q, g, "cosine")
    ev = evaluate_queries(sim, qb, gb, ndcg_k=10, vote_k=10, threshold=0.2)
    service = _service(threshold=0.2)
    for i in range(len(qb)):
        neighbors = [Neighbor(path=str(j), breed=gb[j], score=float(sim[i, j])) for j in ev.neighbors[i]]
        breed, score = service.predict_breed_from_neighbors(neighbors)
        assert ev.per_query.loc[i, "predicted_breed"] == breed
        assert ev.per_query.loc[i, "score"] == pytest.approx(score)


def test_threshold_rejects_and_counts_as_error():
    sim, qb, gb = _toy()
    ev = evaluate_queries(sim, qb, gb, ndcg_k=3, threshold=0.92)
    assert ev.per_query["predicted_breed"].tolist() == [UNKNOWN_BREED, "B"]
    summary = ev.summary(ci=False)
    assert summary["rejected"] == pytest.approx(0.5)
    assert summary["accuracy"] == pytest.approx(0.5)


def test_summary_with_ci_brackets_mean():
    sim, qb, gb = _toy()
    s = evaluate_queries(sim, qb, gb, ndcg_k=3).summary()
    for name in ("ndcg@10", "precision@10", "accuracy"):
        assert s[f"{name}_low"] <= s[name] <= s[f"{name}_high"]


def test_evaluate_queries_shape_mismatch():
    sim, qb, gb = _toy()
    with pytest.raises(ValueError):
        evaluate_queries(sim, qb[:1], gb)


def test_per_breed_and_top_k_sweep():
    sim, qb, gb = _toy()
    ev = evaluate_queries(sim, qb, gb, ndcg_k=3)
    table = per_breed_table(ev.per_query)
    assert set(table.index) == {"A", "B"} and table["n"].sum() == 2
    sweep = top_k_sweep(sim, qb, gb, ks=(1, 3))
    assert sweep.loc[1, "precision@k"] == pytest.approx(1.0)
    assert sweep.loc[3, "precision@k"] == pytest.approx((2 / 3 + 2 / 3) / 2)


def test_compare_metrics_table():
    q, g = _random(10, seed=8), _random(40, seed=9)
    table = compare_metrics(q, g, [f"r{i % 4}" for i in range(10)], [f"r{i % 4}" for i in range(40)])
    assert table.loc[("normalizado", "l2"), "same_top10"] == pytest.approx(1.0)
    assert table.loc[("normalizado", "cosine"), "ndcg@10"] == pytest.approx(table.loc[("crudo", "cosine"), "ndcg@10"])


# --- umbral -------------------------------------------------------------------


def test_threshold_sweep_rates():
    curve = threshold_sweep(np.array([0.6, 0.8, 0.9]), np.array([0.3, 0.7]))
    at_07 = curve[np.isclose(curve["threshold"], 0.7)].iloc[0]
    assert at_07["far"] == pytest.approx(0.5)  # 0,7 >= 0,7: aceptada
    assert at_07["tar"] == pytest.approx(2 / 3)
    assert curve["far"].iloc[-1] == 0.0 and curve["tar"].iloc[-1] == 0.0
    assert (np.diff(curve["far"]) <= 0).all() and (np.diff(curve["tar"]) <= 0).all()


def test_operating_point_smallest_threshold_meeting_far():
    rng = np.random.default_rng(10)
    known = rng.normal(0.8, 0.05, 500)
    unknown = rng.normal(0.6, 0.05, 400)
    op = operating_point(known, unknown, target_far=0.05, known_correct=np.ones(500, dtype=bool))
    assert op["far"] <= 0.05
    # Un umbral apenas menor ya no cumple la politica.
    below = op["curve"][op["curve"]["threshold"] < op["threshold"]].iloc[-1]
    assert below["far"] > 0.05
    assert op["threshold_env"] >= op["threshold"]
    assert op["far_env"] <= 0.05
    assert op["threshold_env"] == pytest.approx(math.ceil(op["threshold"] * 1e4) / 1e4)
    assert op["dir"] == pytest.approx(op["tar"])
    assert 0.9 < op["auc"] <= 1.0
    assert op["n_unknown"] == 400 and op["far_resolution"] == pytest.approx(1 / 400)
    assert rates_at(known, unknown, op["threshold"])["tar"] == pytest.approx(op["tar"])


def test_operating_point_validates_inputs():
    with pytest.raises(ValueError):
        operating_point(np.array([0.5]), np.array([0.4]), target_far=0.0)
    with pytest.raises(ValueError):
        operating_point(np.array([0.5]), np.array([]))


def test_calibration_masks_disjoint_and_without_test():
    manifest = pd.DataFrame(
        {
            "path": [f"{s}/{b}/{i}.jpg" for i, (s, b) in enumerate([("train", "A"), ("train", "Z"), ("valid", "A"), ("valid", "Z"), ("test", "A"), ("test", "Z")])],
            "breed": ["A", "Z", "A", "Z", "A", "Z"],
            "split": ["train", "train", "valid", "valid", "test", "test"],
        }
    )
    masks = calibration_masks(manifest, ["Z"])
    assert masks["gallery"].tolist() == [True, False, False, False, False, False]
    assert masks["known"].tolist() == [False, False, True, False, False, False]
    assert masks["unknown"].tolist() == [False, True, False, True, False, False]
    with pytest.raises(ValueError):
        calibration_masks(manifest, ["NoExiste"])


# --- cache de embeddings --------------------------------------------------------


def test_load_or_compute_embeddings_uses_cache(tmp_path, monkeypatch):
    import lib.evaluation.retrieval as retrieval

    calls = []

    def fake_compute(paths, embed_batch, batch_size=64):
        calls.append(len(paths))
        return np.arange(len(paths) * 2, dtype=np.float32).reshape(len(paths), 2)

    monkeypatch.setattr(retrieval, "compute_embeddings", fake_compute)
    cache = tmp_path / "emb.npz"
    first = load_or_compute_embeddings(cache, ["a", "b"], ["a", "b"], embed_batch=None)
    again = load_or_compute_embeddings(cache, ["a", "b"], ["a", "b"], embed_batch=None)
    np.testing.assert_array_equal(first, again)
    assert calls == [2]
    load_or_compute_embeddings(cache, ["a", "c"], ["a", "c"], embed_batch=None)
    assert calls == [2, 2]  # claves distintas: recalcula
