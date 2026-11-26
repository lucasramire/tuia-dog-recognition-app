"""Evaluacion de la busqueda por similitud (Etapa 1, enunciado 4.3; N2.1 del plan).

Sin base de datos (sugerencia del enunciado): los embeddings de la galeria y de las
consultas son matrices de numpy y la busqueda es un producto matricial. Las metricas
reproducen lo que hace la app:

  - `similarity_matrix` calcula los mismos scores que `SimilarityService.similarity`
    (coseno, o 1 / (1 + distancia euclidea) con `l2`).
  - La raza predicha sale de `lib.search.voting.predict_breed`, la misma funcion que usa
    `predict_breed_from_neighbors`.
  - NDCG@10 con `ndcg_at_k` provista y `n_relevant` = imagenes de la raza en la galeria.

Base en la catedra (class_4): la busqueda 1:N como producto matricial contra la galeria
(7.9, `F.linear(emb, gallery_embeddings)`), los embeddings juntados en una matriz para
evaluar (6.14.2, `collect_embeddings`) y el rechazo por el mejor candidato bajo el umbral
(`identify_face`, 6.14.1).

Calibracion del umbral (D-12, D-26): criterio de class_4 6.14.7 (umbral medido, nunca
tipeado; FAR objetivo, curva ROC, TAR@FAR y EER, informando cuantos negativos se usaron).
Desvio justificado: class_4 calibra sobre PARES genuino/impostor, que da un umbral de
verificacion 1:1. La app no compara pares: compara el MEJOR VECINO de una consulta contra
el umbral (identificacion 1:N, como `identify_face`). Por eso aca se calibra sobre el mejor
score de cada consulta: una consulta "desconocida" es una imagen de una raza que no esta en
la galeria de calibracion, y es un falso aceptado si su mejor vecino supera el umbral.
Calibrar con pares daria un umbral para una decision que la app no toma.

Un umbral por modelo de embeddings (class_4 6.18 y 6.19): cada modelo tiene su propia
escala de scores, asi que el umbral calibrado vale solo para el modelo con que se midio.
`operating_point` es generica: se corre una vez por modelo (baseline, Modelo A, Modelo B).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lib.evaluation.classification import bootstrap_ci
from lib.evaluation.metrics import ndcg_at_k
from lib.search.voting import UNKNOWN_BREED, predict_breed

logger = logging.getLogger(__name__)

METRICS = ("cosine", "l2")


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Filas con norma 1; las filas nulas quedan en cero (similitud coseno 0, como la app)."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def compute_embeddings(
    paths: Sequence[Path],
    embed_batch: Callable[[list[np.ndarray]], np.ndarray],
    batch_size: int = 64,
) -> np.ndarray:
    """Embeddings (N, D) de imagenes en disco con cualquier extractor por lotes.

    `embed_batch` recibe una lista de imagenes BGR uint8 y devuelve (lote, D), por
    ejemplo `BaselineEmbedder.embed_batch`. Una imagen ilegible es un error (los splits
    ya estan limpios; no se inventa un vector).
    """
    from lib.dataset.torch_dataset import read_bgr  # import diferido: carga torch

    if batch_size <= 0:
        raise ValueError(f"batch_size debe ser positivo, llego {batch_size}")
    chunks = []
    for start in range(0, len(paths), batch_size):
        images = [read_bgr(Path(p)) for p in paths[start : start + batch_size]]
        chunks.append(np.asarray(embed_batch(images), dtype=np.float32))
        done = min(start + batch_size, len(paths))
        if (start // batch_size) % 20 == 19 or done == len(paths):
            logger.info("Embeddings: %d / %d imagenes", done, len(paths))
    if not chunks:
        raise ValueError("compute_embeddings recibio una lista vacia")
    return np.concatenate(chunks)


def load_or_compute_embeddings(
    cache_path: Path,
    keys: Sequence[str],
    paths: Sequence[Path],
    embed_batch: Callable[[list[np.ndarray]], np.ndarray],
    batch_size: int = 64,
) -> np.ndarray:
    """Como `compute_embeddings`, pero reusa `cache_path` (.npz) si tiene exactamente las mismas `keys`.

    `keys` identifica cada imagen (por ejemplo la ruta relativa del manifiesto); si cambian
    los splits, el cache no coincide y se recalcula.
    """
    cache_path = Path(cache_path)
    keys = np.asarray([str(k) for k in keys])
    if len(keys) != len(paths):
        raise ValueError(f"Hay {len(keys)} claves y {len(paths)} rutas")
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            if np.array_equal(data["keys"], keys):
                logger.info("Embeddings leidos de %s", cache_path)
                return data["embeddings"]
        logger.info("El cache %s no coincide con las imagenes pedidas: se recalcula", cache_path)
    embeddings = compute_embeddings(paths, embed_batch, batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings, keys=keys)
    return embeddings


# --------------------------------------------------------------------------
# Busqueda
# --------------------------------------------------------------------------


def similarity_matrix(queries: np.ndarray, gallery: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Scores (consultas, galeria) con la misma formula que `SimilarityService.similarity`.

    cosine: a.b / (|a| |b|). l2: 1 / (1 + |a - b|), calculada en float64 para no perder
    precision en la resta de normas con vectores crudos.
    """
    q = np.asarray(queries)
    g = np.asarray(gallery)
    if q.ndim != 2 or g.ndim != 2 or q.shape[1] != g.shape[1]:
        raise ValueError(f"Formas incompatibles: consultas {q.shape}, galeria {g.shape}")
    metric = metric.lower()
    if metric == "cosine":
        return l2_normalize(q) @ l2_normalize(g).T
    if metric == "l2":
        q64, g64 = q.astype(np.float64), g.astype(np.float64)
        sq = (q64**2).sum(axis=1)[:, None] + (g64**2).sum(axis=1)[None, :] - 2.0 * q64 @ g64.T
        return (1.0 / (1.0 + np.sqrt(np.maximum(sq, 0.0)))).astype(np.float32)
    raise ValueError(f"Metrica desconocida '{metric}'; validas: {list(METRICS)}")


def rank_gallery(sim: np.ndarray, k: int) -> np.ndarray:
    """Indices (consultas, k) de la galeria ordenados por score descendente.

    Orden estable: ante empates gana el indice menor, asi el resultado es reproducible.
    """
    if k <= 0:
        raise ValueError(f"k debe ser positivo, llego {k}")
    k = min(int(k), sim.shape[1])
    return np.argsort(-sim, axis=1, kind="stable")[:, :k]


@dataclass
class RetrievalEvaluation:
    """Resultado por consulta y los vecinos recuperados (para mostrar ejemplos)."""

    per_query: pd.DataFrame
    neighbors: np.ndarray  # (consultas, k) indices de la galeria
    neighbor_scores: np.ndarray  # (consultas, k)

    def summary(self, ci: bool = True, n_resamples: int = 2000, seed: int = 42) -> dict[str, float]:
        """NDCG@10 medio, precision@10 y accuracy, con IC bootstrap del 95 % (D-24)."""
        table = self.per_query
        out: dict[str, float] = {"n": int(len(table))}
        for name, column in (("ndcg@10", "ndcg"), ("precision@10", "precision"), ("accuracy", "correct")):
            values = table[column].to_numpy(dtype=np.float64)
            if ci:
                mean, low, high = bootstrap_ci(values, n_resamples=n_resamples, seed=seed)
                out.update({name: mean, f"{name}_low": low, f"{name}_high": high})
            else:
                out[name] = float(values.mean())
        out["rejected"] = float((table["predicted_breed"] == UNKNOWN_BREED).mean())
        return out


def evaluate_queries(
    sim: np.ndarray,
    query_breeds: Sequence[str],
    gallery_breeds: Sequence[str],
    ndcg_k: int = 10,
    vote_k: int | None = None,
    threshold: float | None = None,
) -> RetrievalEvaluation:
    """NDCG@k, precision@k y raza predicha para cada consulta.

    - NDCG y precision usan los `ndcg_k` primeros vecinos (relevante = misma raza).
    - La raza se predice con los `vote_k` primeros (TOP_K de la app; por defecto `ndcg_k`)
      y el umbral de la app (`threshold=None`: sin rechazo, conjunto cerrado).
    """
    query_breeds = np.asarray(query_breeds, dtype=object)
    gallery_breeds = np.asarray(gallery_breeds, dtype=object)
    if sim.shape != (len(query_breeds), len(gallery_breeds)):
        raise ValueError(
            f"sim tiene forma {sim.shape} y hay {len(query_breeds)} consultas y {len(gallery_breeds)} imagenes de galeria"
        )
    vote_k = ndcg_k if vote_k is None else int(vote_k)
    order = rank_gallery(sim, max(ndcg_k, vote_k))
    scores = np.take_along_axis(sim, order, axis=1)
    in_gallery = pd.Series(gallery_breeds).value_counts()

    rows = []
    for i, breed in enumerate(query_breeds):
        neighbor_breeds = gallery_breeds[order[i]]
        relevant = (neighbor_breeds[:ndcg_k] == breed).astype(np.float64)
        n_relevant = int(in_gallery.get(breed, 0))
        predicted, score = predict_breed(
            list(neighbor_breeds[:vote_k]), scores[i, :vote_k].tolist(), threshold
        )
        rows.append(
            {
                "query_breed": breed,
                "predicted_breed": predicted,
                "score": float(score),
                "best_score": float(scores[i, 0]),
                "ndcg": ndcg_at_k(relevant, k=ndcg_k, n_relevant=n_relevant),
                "precision": float(relevant.sum() / ndcg_k),
                "n_relevant": n_relevant,
                "correct": predicted == breed,
            }
        )
    return RetrievalEvaluation(pd.DataFrame(rows), order, scores)


def per_breed_table(per_query: pd.DataFrame) -> pd.DataFrame:
    """NDCG@10, precision@10 y accuracy por raza, de peor a mejor NDCG (descriptivo: n chico)."""
    table = per_query.groupby("query_breed").agg(
        n=("ndcg", "size"),
        ndcg=("ndcg", "mean"),
        precision=("precision", "mean"),
        accuracy=("correct", "mean"),
    )
    return table.sort_values(["ndcg", "accuracy"]).rename_axis("raza")


def top_k_sweep(
    sim: np.ndarray,
    query_breeds: Sequence[str],
    gallery_breeds: Sequence[str],
    ks: Sequence[int] = (1, 3, 5, 10, 20),
) -> pd.DataFrame:
    """Efecto de TOP_K: accuracy del voto con k vecinos y precision@k, sin umbral."""
    rows = []
    for k in ks:
        evaluation = evaluate_queries(sim, query_breeds, gallery_breeds, ndcg_k=10, vote_k=int(k))
        hits = np.asarray(gallery_breeds, dtype=object)[evaluation.neighbors[:, : int(k)]] == np.asarray(
            query_breeds, dtype=object
        )[:, None]
        rows.append(
            {
                "top_k": int(k),
                "accuracy": float(evaluation.per_query["correct"].mean()),
                "precision@k": float(hits.mean()),
            }
        )
    return pd.DataFrame(rows).set_index("top_k")


def compare_metrics(
    query_raw: np.ndarray,
    gallery_raw: np.ndarray,
    query_breeds: Sequence[str],
    gallery_breeds: Sequence[str],
    k: int = 10,
) -> pd.DataFrame:
    """cosine contra l2, con embeddings normalizados (lo que guarda la app) y crudos (D-11).

    Con norma 1, |a - b|^2 = 2 - 2 cos(a, b): ambas metricas ordenan igual y solo cambia la
    escala del score (y por eso el umbral). Con vectores crudos l2 tambien depende de la
    norma y el orden puede cambiar; el coseno no depende de la norma. `same_top10` es la
    fraccion de consultas con el mismo top-k (mismo orden) que el coseno.
    """
    variants = {"normalizado": (l2_normalize(query_raw), l2_normalize(gallery_raw)), "crudo": (query_raw, gallery_raw)}
    rows = []
    for variant, (q, g) in variants.items():
        reference = None
        for metric in METRICS:
            sim = similarity_matrix(q, g, metric)
            evaluation = evaluate_queries(sim, query_breeds, gallery_breeds, ndcg_k=k)
            if reference is None:
                reference = evaluation.neighbors
            summary = evaluation.summary(ci=False)
            rows.append(
                {
                    "embeddings": variant,
                    "metrica": metric,
                    "ndcg@10": summary["ndcg@10"],
                    "precision@10": summary["precision@10"],
                    "accuracy": summary["accuracy"],
                    "same_top10": float((evaluation.neighbors == reference).all(axis=1).mean()),
                    "score_min": float(sim.min()),
                    "score_max": float(sim.max()),
                }
            )
    return pd.DataFrame(rows).set_index(["embeddings", "metrica"])


# --------------------------------------------------------------------------
# Umbral de "unknown" (D-12, D-26)
# --------------------------------------------------------------------------


def calibration_masks(manifest: pd.DataFrame, calibration_breeds: Sequence[str]) -> dict[str, np.ndarray]:
    """Mascaras sobre las filas del manifiesto para calibrar el umbral.

    - gallery: `train` sin las razas apartadas.
    - known: `valid` de las razas de la galeria (deben aceptarse).
    - unknown: `valid` y `train` de las razas apartadas (no estan en la galeria de
      calibracion, asi que son desconocidas; usar tambien `train` da ~10 veces mas
      negativos que `valid` solo, y la FAR se estima con menos ruido).
    `test` no se usa: ahi se reporta.
    """
    calibration = set(calibration_breeds)
    missing = calibration - set(manifest["breed"])
    if missing:
        raise ValueError(f"Razas de calibracion que no estan en el manifiesto: {sorted(missing)}")
    held_out = manifest["breed"].isin(calibration).to_numpy()
    split = manifest["split"].to_numpy()
    masks = {
        "gallery": (split == "train") & ~held_out,
        "known": (split == "valid") & ~held_out,
        "unknown": np.isin(split, ["train", "valid"]) & held_out,
    }
    if (masks["gallery"] & masks["unknown"]).any() or (masks["gallery"] & masks["known"]).any():
        raise AssertionError("Una consulta de calibracion quedo en la galeria")
    return masks


def rates_at(known_scores: np.ndarray, unknown_scores: np.ndarray, threshold: float) -> dict[str, float]:
    """FAR y TAR a un umbral: se acepta una consulta si su mejor vecino es >= umbral (como la app)."""
    known = np.asarray(known_scores, dtype=np.float64)
    unknown = np.asarray(unknown_scores, dtype=np.float64)
    tar = float((known >= threshold).mean())
    return {"threshold": float(threshold), "far": float((unknown >= threshold).mean()), "tar": tar, "frr": 1.0 - tar}


def threshold_sweep(known_scores: np.ndarray, unknown_scores: np.ndarray) -> pd.DataFrame:
    """FAR, TAR y FRR en cada umbral posible (los scores observados), de menor a mayor.

    Se barren los scores observados en lugar de una grilla fija (class_4 usa linspace):
    entre dos scores consecutivos las tasas no cambian, asi se obtienen todos los puntos
    de la curva sin aproximar. El ultimo umbral rechaza todo (FAR = TAR = 0).
    """
    known = np.sort(np.asarray(known_scores, dtype=np.float64))
    unknown = np.sort(np.asarray(unknown_scores, dtype=np.float64))
    if known.size == 0 or unknown.size == 0:
        raise ValueError("Hacen falta consultas conocidas y desconocidas para calibrar")
    candidates = np.unique(np.concatenate([known, unknown]))
    candidates = np.append(candidates, np.nextafter(candidates[-1], np.inf))
    # Aceptadas: score >= umbral -> cantidad = n - (scores < umbral).
    far = 1.0 - np.searchsorted(unknown, candidates, side="left") / unknown.size
    tar = 1.0 - np.searchsorted(known, candidates, side="left") / known.size
    return pd.DataFrame({"threshold": candidates, "far": far, "tar": tar, "frr": 1.0 - tar})


def operating_point(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    target_far: float = 0.05,
    known_correct: np.ndarray | None = None,
    decimals: int = 4,
) -> dict[str, object]:
    """Umbral en la FAR objetivo, mas EER, J de Youden y AUC (class_4 6.14.7).

    - threshold: el menor umbral con FAR <= objetivo (la FAR baja al subir el umbral, asi
      que es el de mayor TAR compatible con la politica).
    - threshold_env: el mismo redondeado hacia arriba a `decimals` (lo que va a
      SIMILARITY_THRESHOLD; redondear hacia abajo podria superar la FAR objetivo), con sus
      tasas recalculadas.
    - dir: si se pasa `known_correct` (la raza predicha sin umbral es correcta), fraccion de
      conocidas aceptadas Y bien clasificadas (tasa de deteccion e identificacion).
    - far_resolution: 1 / negativos; una FAR por debajo de eso no se puede medir.
    """
    if not 0.0 < target_far < 1.0:
        raise ValueError(f"target_far debe estar en (0, 1), llego {target_far}")
    known = np.asarray(known_scores, dtype=np.float64)
    unknown = np.asarray(unknown_scores, dtype=np.float64)
    curve = threshold_sweep(known, unknown)
    i = int(np.flatnonzero(curve["far"].to_numpy() <= target_far)[0])
    point = curve.iloc[i]

    gap = np.abs(curve["far"] - curve["frr"]).to_numpy()
    j = int(np.argmin(gap))
    youden = (curve["tar"] - curve["far"]).to_numpy()
    y = int(np.argmax(youden))
    roc = curve.sort_values(["far", "tar"])
    auc = float(np.trapezoid(roc["tar"], roc["far"]))

    scale = 10**decimals
    env = rates_at(known, unknown, math.ceil(float(point["threshold"]) * scale) / scale)

    result: dict[str, object] = {
        "target_far": float(target_far),
        "threshold": float(point["threshold"]),
        "far": float(point["far"]),
        "tar": float(point["tar"]),
        "frr": float(point["frr"]),
        "threshold_env": env["threshold"],
        "far_env": env["far"],
        "tar_env": env["tar"],
        "eer": float((curve["far"].iloc[j] + curve["frr"].iloc[j]) / 2),
        "eer_threshold": float(curve["threshold"].iloc[j]),
        "youden_threshold": float(curve["threshold"].iloc[y]),
        "youden_j": float(youden[y]),
        "auc": auc,
        "n_known": int(known.size),
        "n_unknown": int(unknown.size),
        "far_resolution": 1.0 / unknown.size,
        "curve": curve,
    }
    if known_correct is not None:
        correct = np.asarray(known_correct, dtype=bool)
        if correct.shape != known.shape:
            raise ValueError("known_correct debe tener una entrada por consulta conocida")
        result["dir"] = float((correct & (known >= result["threshold"])).mean())
    return result
