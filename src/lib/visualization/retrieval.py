"""Figuras de la evaluacion de la Etapa 1: consultas con su top-k y calibracion del umbral."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lib.dataset.eda import load_rgb

HIT_COLOR = "tab:green"
MISS_COLOR = "tab:red"


def show_query_results(
    queries: Sequence[dict],
    size: int = 112,
    cell_inches: float = 1.25,
    suptitle: str | None = None,
) -> plt.Figure:
    """Una fila por consulta: la imagen consultada y sus vecinos con raza y score.

    Cada dict tiene `query_path`, `query_breed`, `predicted_breed`, `neighbor_paths`,
    `neighbor_breeds` y `neighbor_scores`. El titulo del vecino va en verde si es de la
    raza de la consulta y en rojo si no.
    """
    if not queries:
        raise ValueError("show_query_results recibio una lista vacia")
    cols = 1 + max(len(q["neighbor_paths"]) for q in queries)
    fig, axes = plt.subplots(
        len(queries), cols, figsize=(cols * cell_inches, len(queries) * (cell_inches + 0.3)), squeeze=False
    )
    for row, query in zip(axes, queries):
        for ax in row:
            ax.axis("off")
        row[0].imshow(load_rgb(Path(query["query_path"]), size))
        ok = query["predicted_breed"] == query["query_breed"]
        row[0].set_title(
            f"{query['query_breed']}\npred: {query['predicted_breed']}",
            fontsize=7,
            color=HIT_COLOR if ok else MISS_COLOR,
            fontweight="bold",
        )
        for ax, path, breed, score in zip(
            row[1:], query["neighbor_paths"], query["neighbor_breeds"], query["neighbor_scores"]
        ):
            ax.imshow(load_rgb(Path(path), size))
            ax.set_title(
                f"{breed}\n{score:.3f}", fontsize=6.5, color=HIT_COLOR if breed == query["query_breed"] else MISS_COLOR
            )
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    return fig


def plot_threshold_calibration(
    op: dict, known_scores: np.ndarray, unknown_scores: np.ndarray, title: str = ""
) -> plt.Figure:
    """Curva ROC (TAR contra FAR) y distribuciones del mejor score de conocidas y desconocidas.

    Mismo par de figuras que class_4 6.14.7: la ROC describe al modelo, el umbral es un
    punto elegido sobre ella; los histogramas muestran donde se solapan las distribuciones.
    """
    curve = op["curve"].sort_values(["far", "tar"])
    fig, (ax_roc, ax_hist) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_roc.plot(curve["far"], curve["tar"], lw=1.5, label=f"ROC (AUC {op['auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], ls=":", color="grey", lw=1)
    ax_roc.axvline(op["target_far"], ls="--", color="grey", lw=1)
    ax_roc.scatter(
        [op["far"]], [op["tar"]], color="black", zorder=3,
        label=f"FAR objetivo {op['target_far']:.0%}: TAR {op['tar']:.3f} (umbral {op['threshold']:.4f})",
    )
    ax_roc.set_xlabel("FAR (desconocidas aceptadas)")
    ax_roc.set_ylabel("TAR (conocidas aceptadas)")
    ax_roc.set_title(f"ROC, EER {op['eer']:.3f}")
    ax_roc.legend(fontsize=8, loc="lower right")

    bins = np.linspace(
        min(np.min(known_scores), np.min(unknown_scores)), max(np.max(known_scores), np.max(unknown_scores)), 50
    )
    ax_hist.hist(unknown_scores, bins=bins, alpha=0.6, density=True, label=f"desconocidas (n={len(unknown_scores)})")
    ax_hist.hist(known_scores, bins=bins, alpha=0.6, density=True, label=f"conocidas (n={len(known_scores)})")
    ax_hist.axvline(op["threshold"], ls="--", color="black", lw=1, label="umbral (FAR objetivo)")
    ax_hist.axvline(op["eer_threshold"], ls=":", color="grey", lw=1, label="umbral (EER)")
    ax_hist.set_xlabel("similitud con el mejor vecino")
    ax_hist.set_ylabel("densidad")
    ax_hist.set_title("Distribucion del mejor score")
    ax_hist.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
