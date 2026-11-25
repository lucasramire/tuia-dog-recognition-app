"""Grillas de imagenes para las notebooks (EDA, filtrado, augmentation, errores)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

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
