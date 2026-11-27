"""Metricas de clasificacion multiclase (Etapa 2), macro-promediadas.

Cada raza se trata como uno-contra-todos y se usan las funciones provistas en
`metrics.py` (precision_recall_f1, specificity); luego se promedia sin pesos
(macro), asi cada raza pesa igual aunque train este desbalanceado.
"""

from __future__ import annotations

import math

import numpy as np

from lib.evaluation.metrics import precision_recall_f1, specificity


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Matriz (verdadera, predicha) de conteos."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    return cm


def per_class_metrics(cm: np.ndarray) -> dict[str, np.ndarray]:
    total = cm.sum()
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = total - tp - fp - fn
    prec, rec, f1, spec = [], [], [], []
    for k in range(cm.shape[0]):
        p, r, f = precision_recall_f1(int(tp[k]), int(fp[k]), int(fn[k]))
        prec.append(p)
        rec.append(r)
        f1.append(f)
        spec.append(specificity(int(tn[k]), int(fp[k])))
    return {
        "precision": np.array(prec),
        "recall": np.array(rec),
        "f1": np.array(f1),
        "specificity": np.array(spec),
        "support": cm.sum(axis=1),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, float]:
    """accuracy y macro precision, recall, specificity y F1.

    Solo se promedian las razas presentes en y_true (support > 0).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        raise ValueError("No hay predicciones para evaluar")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} y y_pred {y_pred.shape} tienen formas distintas")
    cm = confusion_matrix(y_true, y_pred, num_classes)
    per = per_class_metrics(cm)
    present = per["support"] > 0
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "precision": float(per["precision"][present].mean()),
        "recall": float(per["recall"][present].mean()),
        "specificity": float(per["specificity"][present].mean()),
        "f1": float(per["f1"][present].mean()),
    }


def top_confused_pairs(cm: np.ndarray, classes: list[str], k: int = 10) -> list[dict]:
    """Pares de razas con mas errores (en ambos sentidos), de mayor a menor."""
    errors = cm + cm.T
    np.fill_diagonal(errors, 0)
    rows, cols = np.triu_indices(len(classes), k=1)
    counts = errors[rows, cols]
    order = np.argsort(-counts, kind="stable")[:k]
    return [
        {
            "breed_a": classes[rows[i]],
            "breed_b": classes[cols[i]],
            "a_as_b": int(cm[rows[i], cols[i]]),
            "b_as_a": int(cm[cols[i], rows[i]]),
            "errors": int(counts[i]),
        }
        for i in order
        if counts[i] > 0
    ]


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    """McNemar exacto (binomial, dos colas) sobre aciertos pareados de dos modelos.

    Solo cuentan los casos discordantes: b = A acierta y B no, c = B acierta y A no.
    Bajo H0 (mismo error) b ~ Binomial(b + c, 0.5).
    """
    a = np.asarray(correct_a, dtype=bool)
    b_ = np.asarray(correct_b, dtype=bool)
    if a.shape != b_.shape:
        raise ValueError("Los vectores de aciertos deben tener la misma forma")
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    n = b + c
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p_value": 1.0}
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return {"only_a": b, "only_b": c, "p_value": float(min(1.0, 2 * tail))}


def risk_coverage(probs: np.ndarray, y_true: np.ndarray) -> "pd.DataFrame":
    """Curva de clasificacion selectiva (D-27): un umbral por cada confianza observada.

    Para cada umbral t se aceptan las predicciones con probabilidad maxima >= t. `coverage` es la
    proporcion aceptada y `accuracy` la accuracy de lo aceptado. Ordenada de menor a mayor umbral.
    """
    import pandas as pd

    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    if probs.ndim != 2 or probs.shape[0] != y_true.shape[0] or y_true.size == 0:
        raise ValueError(f"probs {probs.shape} e y_true {y_true.shape} no son compatibles")
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    order = np.argsort(-confidence, kind="stable")
    conf_sorted, correct_sorted = confidence[order], correct[order]
    accepted = np.arange(1, len(order) + 1)
    hits = np.cumsum(correct_sorted)
    # Con empates de confianza solo vale el ultimo indice de cada valor (se aceptan todos juntos).
    last = np.r_[conf_sorted[1:] != conf_sorted[:-1], True]
    table = pd.DataFrame(
        {
            "threshold": conf_sorted[last],
            "coverage": accepted[last] / len(order),
            "accuracy": hits[last] / accepted[last],
            "accepted": accepted[last],
        }
    )
    return table.iloc[::-1].reset_index(drop=True)


def confidence_threshold(probs: np.ndarray, y_true: np.ndarray, target_accuracy: float = 0.95) -> dict[str, float]:
    """Menor umbral cuya accuracy aceptada alcanza `target_accuracy` (criterio de D-27).

    El menor umbral es el que mas cobertura conserva. Si ya sin umbral se cumple el objetivo,
    devuelve 0 (no hace falta rechazar nada).
    """
    if not 0.0 < target_accuracy <= 1.0:
        raise ValueError(f"target_accuracy debe estar en (0, 1], llego {target_accuracy}")
    curve = risk_coverage(probs, y_true)
    full = curve.iloc[0]
    if full["accuracy"] >= target_accuracy:
        return {"threshold": 0.0, "coverage": 1.0, "accuracy": float(full["accuracy"])}
    ok = curve[curve["accuracy"] >= target_accuracy]
    if ok.empty:
        raise ValueError(f"Ningun umbral alcanza accuracy {target_accuracy}")
    best = ok.iloc[0]
    return {"threshold": float(best["threshold"]), "coverage": float(best["coverage"]), "accuracy": float(best["accuracy"])}


def apply_confidence_threshold(probs: np.ndarray, y_true: np.ndarray, threshold: float) -> dict[str, float]:
    """Cobertura y accuracy de lo aceptado con un umbral fijo (para medir en `test` el umbral de `valid`)."""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    accepted = probs.max(axis=1) >= threshold
    correct = probs.argmax(axis=1) == y_true
    return {
        "threshold": float(threshold),
        "coverage": float(accepted.mean()),
        "accuracy": float(correct[accepted].mean()) if accepted.any() else float("nan"),
        "errors_rejected": float((~correct & ~accepted).sum() / max((~correct).sum(), 1)),
    }


def bootstrap_ci(
    values: np.ndarray, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float, float]:
    """Media e intervalo bootstrap percentil (1 - alpha) de valores por muestra (ej. aciertos 0/1)."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("values esta vacio")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n_resamples, values.size))].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(low), float(high)
