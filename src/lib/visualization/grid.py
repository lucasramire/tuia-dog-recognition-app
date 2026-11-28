"""Grillas de imagenes para las notebooks (EDA, filtrado, augmentation, errores)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.dataset.eda import load_rgb


def show_grid(
    images: Sequence[np.ndarray | Path | str],
    titles: Sequence[str] | None = None,
    cols: int = 8,
    size: int = 112,
    cell_inches: float = 1.3,
    title_fontsize: int = 7,
    suptitle: str | None = None,
) -> plt.Figure:
    """Muestra imagenes en una grilla. Acepta rutas (se leen y ajustan con padding) o arreglos RGB."""
    n = len(images)
    if n == 0:
        raise ValueError("show_grid recibio una lista vacia")
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell_inches, rows * (cell_inches + 0.15)), squeeze=False)
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k >= n:
            continue
        item = images[k]
        rgb = load_rgb(Path(item), size) if isinstance(item, (str, Path)) else item
        ax.imshow(rgb)
        if titles is not None:
            ax.set_title(titles[k], fontsize=title_fontsize)
    if suptitle:
        fig.suptitle(suptitle, fontsize=title_fontsize + 3)
    fig.tight_layout()
    return fig


def show_pairs(
    pairs: pd.DataFrame,
    dataset_path: Path,
    n: int = 6,
    pairs_per_row: int = 3,
    size: int = 112,
    suptitle: str | None = None,
) -> plt.Figure:
    """Pares de imagenes lado a lado (los `n` primeros de `pairs`), con raza, split y scores del par.

    `pairs` tiene `path_i`, `path_j`, `breed_i`, `breed_j`, `split_i`, `split_j`, `phash_distance`
    y `cosine` (salida de `lib.dataset.quality.describe_pairs`).
    """
    rows = pairs.head(n)
    if rows.empty:
        raise ValueError("show_pairs recibio una tabla vacia")
    images, titles = [], []
    for r in rows.itertuples(index=False):
        scores = f"pHash {int(r.phash_distance)} | cos {r.cosine:.3f}"
        images += [dataset_path / r.path_i, dataset_path / r.path_j]
        titles += [f"{r.breed_i} ({r.split_i})\n{scores}", f"{r.breed_j} ({r.split_j})"]
    return show_grid(images, titles, cols=2 * pairs_per_row, size=size, suptitle=suptitle)
