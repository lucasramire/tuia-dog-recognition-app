"""Graficos de la Etapa 2: curvas de entrenamiento y matriz de confusion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_histories(histories: Mapping[str, dict], title: str | None = None) -> plt.Figure:
    """Loss y accuracy de train (linea punteada) y valid (linea llena) por epoca.

    `histories` mapea nombre -> JSON de historial que escribe el trainer. Se marca
    la mejor epoca (la del checkpoint guardado) de cada corrida.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for k, (name, payload) in enumerate(histories.items()):
        color = f"C{k % 10}"
        epochs = [e["epoch"] for e in payload["history"]]
        for ax, key in zip(axes, ("loss", "accuracy")):
            ax.plot(epochs, [e[f"train_{key}"] for e in payload["history"]], linestyle=":", color=color)
            ax.plot(epochs, [e[f"val_{key}"] for e in payload["history"]], color=color, label=name)
        best = payload.get("best_epoch")
        if best:
            axes[1].scatter([best], [payload["best_val_accuracy"]], color=color, zorder=3)
    axes[0].set(title="Loss (punteada: train, llena: valid)", xlabel="epoca", ylabel="loss")
    axes[1].set(title="Accuracy (punto: mejor epoca en valid)", xlabel="epoca", ylabel="accuracy")
    axes[1].legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_confusion(
    cm: np.ndarray, classes: Sequence[str], title: str = "Matriz de confusion normalizada por fila"
) -> plt.Figure:
    """Matriz normalizada por fila (cada fila suma 1: recall por raza en la diagonal)."""
    cm = np.asarray(cm, dtype=np.float64)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(13, 12))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), labels=[c[:14] for c in classes], rotation=90, fontsize=6)
    ax.set_yticks(range(n), labels=[c[:14] for c in classes], fontsize=6)
    ax.set(xlabel="predicha", ylabel="verdadera", title=title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    return fig
