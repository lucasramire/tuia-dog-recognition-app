"""Dataset de PyTorch sobre el manifiesto de splits.

Lee las imagenes en BGR (OpenCV) y aplica un transform BGR uint8 -> float32
(3, S, S): `make_transform(train=True)` en train (augmentation + preprocesamiento
compartido) y `make_transform(train=False)` en valid/test (solo preprocesamiento).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lib.dataset.manifest import SPLITS


def read_bgr(path: Path) -> np.ndarray:
    """Lee una imagen BGR uint8. Usa imdecode para tolerar rutas con caracteres no ASCII."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    return image


def subset_per_class(rows: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    """Hasta `max_per_class` filas por raza, al azar con semilla (pruebas de humo)."""
    shuffled = rows.sample(frac=1.0, random_state=seed)
    return shuffled.groupby("breed", sort=False).head(max_per_class).sort_index().reset_index(drop=True)


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        dataset_path: Path,
        classes: list[str],
        transform: Callable[[np.ndarray], np.ndarray],
        max_per_class: int | None = None,
        seed: int = 42,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split invalido '{split}'; validos: {list(SPLITS)}")
        rows = manifest[manifest["split"] == split].reset_index(drop=True)
        if max_per_class is not None:
            rows = subset_per_class(rows, int(max_per_class), seed)
        unknown = sorted(set(rows["breed"]) - set(classes))
        if unknown:
            raise ValueError(f"Razas de '{split}' que no estan en classes: {unknown}")
        if rows.empty:
            raise ValueError(f"El split '{split}' no tiene imagenes")
        self.rows = rows
        self.split = split
        self.dataset_path = Path(dataset_path)
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform
        self.targets = [self.class_to_idx[b] for b in rows["breed"]]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = read_bgr(self.dataset_path / self.rows.at[index, "path"])
        return torch.from_numpy(self.transform(image)), self.targets[index]

    def class_counts(self) -> np.ndarray:
        return np.bincount(np.asarray(self.targets), minlength=len(self.classes))
