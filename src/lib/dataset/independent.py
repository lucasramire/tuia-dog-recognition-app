"""Registro del conjunto independiente (P1.4, enunciado 3.4): `data/splits/independent_v1.csv`.

Una fila por imagen con su fuente y licencia. Las imagenes viven en
`data/dataset/independent/<raza>/` (fuera de git); el CSV si se versiona.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

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
