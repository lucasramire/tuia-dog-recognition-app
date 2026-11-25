"""Metricas de clasificacion multiclase (Etapa 2), macro-promediadas.

Cada raza se trata como uno-contra-todos y se usan las funciones provistas en
`metrics.py` (precision_recall_f1, specificity); luego se promedia sin pesos
(macro), asi cada raza pesa igual aunque train este desbalanceado.
"""

from __future__ import annotations

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
