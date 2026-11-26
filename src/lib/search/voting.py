"""Prediccion de raza a partir de los vecinos recuperados (Etapa 1, D-05 y D-06 del plan).

Una sola implementacion de la regla, usada por `SimilarityService.predict_breed_from_neighbors`
(la app) y por `lib.evaluation.retrieval` (la notebook): asi lo que se mide es exactamente
lo que hace la app.

Regla:
  - Sin vecinos, o con el mejor vecino por debajo del umbral: "unknown" (rechazo por
    similitud con el vecino mas cercano, criterio de `identify_face` en class_4).
  - Si no, voto ponderado por similitud (kNN de Dudani, 1976): cada vecino suma su
    score a su raza; las similitudes negativas no suman.
  - Score devuelto: similitud media de los vecinos de la raza ganadora.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

UNKNOWN_BREED = "unknown"


def predict_breed(
    breeds: Sequence[str], scores: Sequence[float], threshold: float | None = None
) -> tuple[str, float]:
    """(raza, score) a partir de las razas y scores de los vecinos, en cualquier orden.

    `threshold=None` desactiva el rechazo (prediccion de conjunto cerrado).
    """
    if len(breeds) != len(scores):
        raise ValueError(f"Hay {len(breeds)} razas y {len(scores)} scores")
    if not breeds:
        return UNKNOWN_BREED, 0.0

    scores = [float(s) for s in scores]
    best_score = max(scores)
    if threshold is not None and best_score < threshold:
        return UNKNOWN_BREED, best_score

    weights: dict[str, float] = defaultdict(float)
    by_breed: dict[str, list[float]] = defaultdict(list)
    for breed, score in zip(breeds, scores):
        weights[breed] += max(score, 0.0)
        by_breed[breed].append(score)

    # Desempate: peso total, luego cantidad de votos, luego mejor vecino, luego nombre
    # (para que el resultado no dependa del orden de entrada).
    winner = min(
        by_breed,
        key=lambda b: (-weights[b], -len(by_breed[b]), -max(by_breed[b]), b),
    )
    return winner, sum(by_breed[winner]) / len(by_breed[winner])
