"""Registro del conjunto independiente (P1.4, enunciado 3.4): `data/splits/independent_v1.csv`.

Una fila por imagen con su fuente y licencia. Las imagenes viven en
`data/dataset/independent/<raza>/` (fuera de git); el CSV si se versiona.

Incluye la validacion del registro, los resumenes para la notebook 1 y el control de fuga
contra el dataset (una imagen del independiente que ya esta en el dataset inflaria las
metricas): mismo criterio de duplicado que el filtrado (D-15, `CleaningConfig.duplicates`).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from lib.dataset.quality import DuplicateCriterion

logger = logging.getLogger(__name__)

SCENES = ("single", "multiple", "no_dog")
SOURCES = ("own", "wikimedia", "unsplash", "pexels")
PEOPLE = ("none", "not_identifiable")
MULTIPLE_BREED = "multiple"
NO_DOG_BREED = "none"
# Licencias aceptadas (RECOLECCION_INDEPENDIENTE.md): foto propia, dominio publico, CC0, CC BY,
# CC BY-SA, Unsplash y Pexels. NC, ND y GFDL sola no.
_LICENSE = re.compile(
    r"^(own|public domain|cc0( 1\.0)?|cc by(-sa)? \d\.\d.*|unsplash license|pexels license)$", re.IGNORECASE
)

COLUMNS = [
    "path", "breed", "scene", "n_dogs", "breeds_in_scene", "source", "url", "author",
    "license", "license_url", "retrieved", "collector", "people", "notes",
]
INDEPENDENT_DIR = "independent"
# Numeracion por integrante para que dos personas no generen el mismo nombre de archivo.
COLLECTOR_RANGES: dict[str, range] = {"lucas": range(1, 50), "valentin": range(51, 100)}


def slug(name: str) -> str:
    """'Boston Terrier' -> 'boston_terrier', 'Shih-Tzu' -> 'shih_tzu'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_registry(path: Path) -> pd.DataFrame:
    """Lee el registro (vacio si solo tiene encabezado) y verifica las columnas."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    if list(table.columns) != COLUMNS:
        raise ValueError(f"Columnas de {path}: {list(table.columns)}; se esperaban {COLUMNS}")
    return table


def next_numbers(folder: Path, prefix: str, collector: str, count: int) -> list[int]:
    """Los proximos `count` numeros libres de `<prefix>_NN.*` dentro del rango del integrante."""
    if collector not in COLLECTOR_RANGES:
        raise ValueError(f"collector desconocido '{collector}'; validos: {list(COLLECTOR_RANGES)}")
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.[a-z]+$", re.IGNORECASE)
    used = {int(m.group(1)) for p in folder.glob(f"{prefix}_*") if (m := pattern.match(p.name))} if folder.exists() else set()
    free = [n for n in COLLECTOR_RANGES[collector] if n not in used]
    if len(free) < count:
        raise ValueError(f"No quedan numeros libres para {collector} en {folder} (pedidos {count}, libres {len(free)})")
    return free[:count]


# --------------------------------------------------------------------------
# Validacion y resumen
# --------------------------------------------------------------------------


def validate_registry(
    table: pd.DataFrame,
    dataset_path: Path,
    classes: Sequence[str],
    check_images: bool = True,
    min_side: int = 224,
) -> pd.DataFrame:
    """Problemas del registro, una fila por problema (`path`, `problem`); vacio si esta todo bien.

    Revisa: rutas unicas y existentes dentro de `independent/`, coherencia entre escena, raza y
    cantidad de perros, raza dentro de las del dataset, fuente, licencia aceptada (y `own` solo
    con fuente propia), url y autor presentes, personas y quien la junto. Con `check_images`
    tambien que la imagen se pueda leer y tenga lado menor >= `min_side`.
    """
    known = set(classes)
    problems: list[tuple[str, str]] = []

    def add(path: str, problem: str) -> None:
        problems.append((path, problem))

    for path in table.loc[table["path"].duplicated(keep=False), "path"].unique():
        add(path, "ruta repetida")
    web = table[table["source"] != "own"]
    for url in web.loc[web["url"].duplicated(keep=False), "url"].unique():
        for path in web.loc[web["url"] == url, "path"]:
            add(path, f"misma url que otra fila: {url}")

    for row in table.itertuples(index=False):
        path = row.path
        if not path.startswith(f"{INDEPENDENT_DIR}/"):
            add(path, f"la ruta no esta dentro de {INDEPENDENT_DIR}/")
        if row.scene not in SCENES:
            add(path, f"escena invalida '{row.scene}'")
        try:
            n_dogs = int(row.n_dogs)
        except ValueError:
            add(path, f"n_dogs no es un entero: '{row.n_dogs}'")
            n_dogs = -1
        if row.scene == "single" and (row.breed not in known or n_dogs != 1):
            add(path, "escena single: la raza debe ser una del dataset y n_dogs = 1")
        if row.scene == "multiple" and (row.breed != MULTIPLE_BREED or n_dogs < 2):
            add(path, f"escena multiple: breed = '{MULTIPLE_BREED}' y n_dogs >= 2")
        if row.scene == "no_dog" and (row.breed != NO_DOG_BREED or n_dogs != 0):
            add(path, f"escena no_dog: breed = '{NO_DOG_BREED}' y n_dogs = 0")
        for breed in filter(None, (b.strip() for b in row.breeds_in_scene.split(";"))):
            if breed not in known:
                add(path, f"raza desconocida en breeds_in_scene: '{breed}'")
        if row.source not in SOURCES:
            add(path, f"fuente invalida '{row.source}'")
        if not _LICENSE.match(row.license.strip()):
            add(path, f"licencia no aceptada o mal escrita '{row.license}'")
        if (row.source == "own") != (row.license.strip().lower() == "own"):
            add(path, "license = own va solo con source = own")
        if row.source != "own" and not row.url.strip():
            add(path, "falta la url de la pagina de la imagen")
        if not row.author.strip():
            add(path, "falta el autor")
        if row.people not in PEOPLE:
            add(path, f"people invalido '{row.people}'")
        if row.collector not in COLLECTOR_RANGES:
            add(path, f"collector invalido '{row.collector}'")
        file = Path(dataset_path) / path
        if not file.exists():
            add(path, "no existe el archivo")
        elif check_images:
            image = cv2.imread(str(file))
            if image is None:
                add(path, "no se puede leer la imagen")
            elif min(image.shape[:2]) < min_side:
                add(path, f"lado menor {min(image.shape[:2])} < {min_side}")
    return pd.DataFrame(problems, columns=["path", "problem"])


def summarize_registry(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tablas para la notebook: imagenes por raza (limpias), por escena, por licencia y por fuente."""
    single = table[table["scene"] == "single"]
    return {
        "por_raza": single.groupby("breed").size().rename("imagenes").sort_values(ascending=False).to_frame(),
        "por_escena": table.groupby("scene").size().reindex(SCENES, fill_value=0).rename("imagenes").to_frame(),
        "por_licencia": table.groupby("license").size().sort_values(ascending=False).rename("imagenes").to_frame(),
        "por_fuente": table.groupby("source").size().rename("imagenes").to_frame(),
    }


# --------------------------------------------------------------------------
# Control de fuga contra el dataset
# --------------------------------------------------------------------------


def measure_images(paths: Sequence[Path], image_size: int) -> tuple[np.ndarray, np.ndarray]:
    """pHash (uint64) y embeddings L2 del baseline: las mismas mediciones que el filtrado (P1.3)."""
    from lib.dataset.quality import compute_baseline_embeddings, phash_int  # import diferido: carga torch

    hashes = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"No se pudo leer {path}")
        hashes.append(phash_int(image))
    return np.array(hashes, dtype=np.uint64), compute_baseline_embeddings(list(paths), image_size)


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def dataset_leaks(
    hashes: np.ndarray,
    embeddings: np.ndarray,
    dataset_hashes: np.ndarray,
    dataset_embeddings: np.ndarray,
    dataset_paths: Sequence[str],
    criterion: DuplicateCriterion,
) -> pd.DataFrame:
    """Por imagen del independiente: su vecina mas parecida del dataset y si es la misma foto.

    `leak` usa el criterio de duplicado del filtrado: (pHash <= max y coseno >= min_con_phash)
    o coseno >= min_solo. Las columnas `max_cosine` y `min_phash` permiten ver que tan cerca
    quedan las que no son fuga.
    """
    if len(dataset_paths) != len(dataset_embeddings) or len(dataset_hashes) != len(dataset_embeddings):
        raise ValueError("dataset_hashes, dataset_embeddings y dataset_paths deben tener el mismo largo")
    cosine = _l2(embeddings) @ _l2(dataset_embeddings).T
    hamming = np.bitwise_count(
        np.asarray(hashes, dtype=np.uint64)[:, None] ^ np.asarray(dataset_hashes, dtype=np.uint64)[None, :]
    ).astype(int)
    same = ((hamming <= criterion.max_phash_distance) & (cosine >= criterion.min_cosine_with_phash)) | (
        cosine >= criterion.min_cosine_alone
    )
    nearest = cosine.argmax(axis=1)
    paths = np.asarray(dataset_paths)
    return pd.DataFrame(
        {
            "max_cosine": cosine.max(axis=1),
            "nearest": paths[nearest],
            "min_phash": hamming.min(axis=1),
            "leak": same.any(axis=1),
            "leak_with": [", ".join(paths[row]) for row in same],
        }
    )


def internal_near_duplicates(embeddings: np.ndarray, min_cosine: float = 0.95) -> pd.DataFrame:
    """Pares (i, j), i < j, del propio conjunto con coseno >= `min_cosine` (fotos casi iguales)."""
    e = _l2(embeddings)
    sims = e @ e.T
    i, j = np.nonzero(np.triu(sims >= min_cosine, k=1))
    return pd.DataFrame({"i": i, "j": j, "cosine": sims[i, j]})
