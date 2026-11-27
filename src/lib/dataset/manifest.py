"""Manifiesto de splits: contrato entre la preparacion de datos y el entrenamiento.

Un CSV con tres columnas:
  - path:  ruta de la imagen relativa a DATASET_PATH (ej. train/Afghan/001.jpg)
  - breed: nombre de la raza, normalizado (sin espacios repetidos ni en los bordes)
  - split: train | valid | test

El entrenamiento y la evaluacion leen solo este archivo. La notebook de
preprocesamiento lo reemplaza con los splits limpios sin cambiar el formato.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["path", "breed", "split"]
SPLITS = ("train", "valid", "test")
DEFAULT_MANIFEST_NAME = "splits_v0.csv"


def normalize_breed(name: str) -> str:
    """Quita espacios en los bordes y colapsa los repetidos ('American  Spaniel')."""
    return " ".join(str(name).split())


def build_manifest_from_dogs_csv(dogs_csv: Path) -> pd.DataFrame:
    """Arma el manifiesto inicial desde dogs.csv (filepaths, labels, data set)."""
    if not dogs_csv.exists():
        raise ValueError(f"No existe {dogs_csv}. Descarga el dataset primero (scripts/download_dataset.py).")
    raw = pd.read_csv(dogs_csv)
    missing = {"filepaths", "labels", "data set"} - set(raw.columns)
    if missing:
        raise ValueError(f"{dogs_csv} no tiene las columnas esperadas: faltan {sorted(missing)}")
    manifest = pd.DataFrame(
        {
            "path": raw["filepaths"].astype(str).str.replace("\\", "/", regex=False),
            "breed": raw["labels"].map(normalize_breed),
            "split": raw["data set"].astype(str).str.strip().str.lower(),
        }
    )
    return manifest[MANIFEST_COLUMNS]


def validate_manifest(
    manifest: pd.DataFrame, dataset_path: Path | None = None, check_files: bool = False
) -> None:
    """Valida el contrato. Lanza ValueError con un mensaje claro si algo no cumple."""
    if list(manifest.columns) != MANIFEST_COLUMNS:
        raise ValueError(f"Columnas invalidas {list(manifest.columns)}; se esperaba {MANIFEST_COLUMNS}")
    if manifest.empty:
        raise ValueError("El manifiesto esta vacio")
    if manifest.isna().any().any():
        raise ValueError("El manifiesto tiene valores vacios")

    bad_splits = sorted(set(manifest["split"]) - set(SPLITS))
    if bad_splits:
        raise ValueError(f"Splits invalidos {bad_splits}; validos: {list(SPLITS)}")
    for split in SPLITS:
        if not (manifest["split"] == split).any():
            raise ValueError(f"El manifiesto no tiene imagenes de '{split}'")

    if manifest["path"].duplicated().any():
        dup = manifest.loc[manifest["path"].duplicated(), "path"].iloc[0]
        raise ValueError(f"Ruta repetida en el manifiesto (una imagen en dos filas): {dup}")

    not_normalized = manifest.loc[manifest["breed"] != manifest["breed"].map(normalize_breed), "breed"]
    if not not_normalized.empty:
        raise ValueError(f"Raza sin normalizar: '{not_normalized.iloc[0]}'")

    train_breeds = set(manifest.loc[manifest["split"] == "train", "breed"])
    outside = sorted(set(manifest["breed"]) - train_breeds)
    if outside:
        raise ValueError(f"Razas en valid/test que no estan en train: {outside}")

    if check_files:
        if dataset_path is None:
            raise ValueError("check_files requiere dataset_path")
        missing = [p for p in manifest["path"] if not (dataset_path / p).exists()]
        if missing:
            raise ValueError(f"{len(missing)} archivos del manifiesto no existen; primero: {missing[0]}")


def save_manifest(manifest: pd.DataFrame, path: Path) -> None:
    validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False, encoding="utf-8")
    logger.info("Manifiesto guardado en %s (%d filas)", path, len(manifest))


def load_manifest(
    path: Path, dataset_path: Path | None = None, check_files: bool = False
) -> pd.DataFrame:
    """Lee y valida un manifiesto."""
    if not path.exists():
        raise ValueError(f"No existe el manifiesto {path}")
    manifest = pd.read_csv(path, encoding="utf-8")
    validate_manifest(manifest, dataset_path=dataset_path, check_files=check_files)
    return manifest


def split_images(
    manifest: pd.DataFrame, dataset_path: Path, split: str, limit: int = 0
) -> list[tuple[Path, str]]:
    """(ruta absoluta, raza) de las imagenes de un split, ordenadas por raza y ruta.

    Es lo que indexa scripts/build_index.py: con los splits limpios la galeria queda igual a la
    medida en 2-train.ipynb (sin las imagenes de valid/test ni las descartadas en la limpieza).
    `limit` > 0 toma como maximo esa cantidad por raza (indexado de prueba).
    """
    if split not in SPLITS:
        raise ValueError(f"Split invalido '{split}'; validos: {list(SPLITS)}")
    rows = manifest.loc[manifest["split"] == split].sort_values(["breed", "path"])
    if limit > 0:
        rows = rows.groupby("breed", sort=True).head(limit)
    return [(dataset_path / path, breed) for path, breed in zip(rows["path"], rows["breed"])]


def class_names(manifest: pd.DataFrame) -> list[str]:
    """Lista ordenada de razas de train: el indice de cada raza es su posicion.

    Es el mapeo indice -> raza que se guarda en el checkpoint (atributo `classes`).
    """
    return sorted(manifest.loc[manifest["split"] == "train", "breed"].unique().tolist())
