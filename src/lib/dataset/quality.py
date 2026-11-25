"""Control de calidad y deteccion de duplicados del dataset (P1.3, enunciado 3.3).

Reglas (D-15 del plan):
  - Archivo corrupto o ilegible.
  - Resolucion minima (lado menor).
  - Borrosidad: varianza del Laplaciano sobre la imagen en grises. El Laplaciano
    es un filtro de bordes (Class 1, celda 53): una imagen nitida tiene muchos
    bordes y una respuesta muy variable; una borrosa, poca varianza. La escala
    depende de la resolucion y del contenido, por eso el umbral se elige por
    percentiles mirando ejemplos, no con un valor fijo.
  - Exposicion extrema: brillo medio y fraccion de pixeles saturados.
  - Duplicados y casi-duplicados: pHash (rapido, robusto a reescalado y
    compresion) y similitud coseno de embeddings del baseline (captura recortes y
    flips, que el pHash no ve).

Las reglas de calidad se aplican solo a `train` (no se cambia la dificultad de
la evaluacion). Los duplicados entre splits son fuga: se quitan de valid/test y
la imagen queda en train. Este modulo solo mide y marca; la decision de que
filas se quitan la toma `build_drop_table` con umbrales explicitos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

SPLIT_PRIORITY = {"train": 0, "valid": 1, "test": 2}


# --------------------------------------------------------------------------
# Metricas por imagen
# --------------------------------------------------------------------------


def laplacian_variance(image_bgr: np.ndarray) -> float:
    """Varianza de la respuesta del Laplaciano sobre la imagen en grises."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_stats(image_bgr: np.ndarray, dark_level: int = 10, bright_level: int = 245) -> dict[str, float]:
    """Brillo medio (0 a 255) y fraccion de pixeles casi negros o casi blancos."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "dark_fraction": float((gray <= dark_level).mean()),
        "bright_fraction": float((gray >= bright_level).mean()),
    }


def phash_int(image_bgr: np.ndarray) -> int:
    """pHash de 64 bits como entero (DCT de baja frecuencia comparada con la mediana)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return int(str(imagehash.phash(Image.fromarray(rgb))), 16)


def image_quality_row(path: Path) -> dict[str, object]:
    """Mide una imagen. Si no se puede leer, marca `readable=False` y deja el resto vacio."""
    row: dict[str, object] = {"readable": False, "error": ""}
    try:
        image = cv2.imread(str(path))
        if image is None:
            row["error"] = "cv2.imread devolvio None"
            return row
        with Image.open(path) as pil:
            pil.verify()  # detecta archivos truncados que OpenCV a veces decodifica igual
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de lectura es "corrupto"
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    height, width = image.shape[:2]
    row.update(
        readable=True,
        height=int(height),
        width=int(width),
        channels=int(image.shape[2]) if image.ndim == 3 else 1,
        blur_var=laplacian_variance(image),
        phash=phash_int(image),
        **exposure_stats(image),
    )
    return row


def compute_quality_table(manifest: pd.DataFrame, dataset_path: Path) -> pd.DataFrame:
    """Una fila por imagen del manifiesto con sus metricas de calidad y su pHash."""
    rows = []
    for i, rel in enumerate(manifest["path"]):
        rows.append(image_quality_row(dataset_path / rel))
        if (i + 1) % 2000 == 0:
            logger.info("Calidad: %d / %d imagenes", i + 1, len(manifest))
    table = pd.concat([manifest.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    unreadable = int((~table["readable"]).sum())
    logger.info("Calidad: %d imagenes, %d ilegibles", len(table), unreadable)
    return table


def compute_baseline_embeddings(
    paths: list[Path], image_size: int, batch_size: int = 64
) -> np.ndarray:
    """Embeddings L2 del baseline (ResNet50, el mismo extractor de la Etapa 1), forma (N, 2048).

    Las imagenes ilegibles quedan como filas de ceros (similitud 0 con todo).
    """
    from lib.models.baseline import BaselineEmbedder  # import diferido: carga torch

    embedder = BaselineEmbedder(image_size)
    out = np.zeros((len(paths), embedder.embedding_dim), dtype=np.float32)
    batch, idx = [], []
    for k, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is not None:
            batch.append(image)
            idx.append(k)
        if len(batch) == batch_size or (k == len(paths) - 1 and batch):
            out[idx] = embedder.embed_batch(batch)
            batch, idx = [], []
        if (k + 1) % 2000 == 0:
            logger.info("Embeddings: %d / %d imagenes", k + 1, len(paths))
    return out


# --------------------------------------------------------------------------
# Pares de duplicados
# --------------------------------------------------------------------------


def _l2(embeddings: np.ndarray) -> np.ndarray:
    """Normaliza filas a norma 1; las filas nulas (imagenes ilegibles) quedan en cero."""
    e = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    return np.divide(e, norms, out=np.zeros_like(e), where=norms > 0)


def phash_pairs(hashes: np.ndarray, max_distance: int, chunk: int = 1024) -> pd.DataFrame:
    """Pares (i, j), i < j, con distancia de Hamming entre pHash <= max_distance."""
    h = np.asarray(hashes, dtype=np.uint64)
    found_i, found_j, found_d = [], [], []
    for start in range(0, len(h), chunk):
        block = h[start : start + chunk]
        dist = np.bitwise_count(block[:, None] ^ h[None, :])
        ii, jj = np.nonzero(dist <= max_distance)
        ii = ii + start
        keep = ii < jj
        found_i.append(ii[keep])
        found_j.append(jj[keep])
        found_d.append(dist[ii[keep] - start, jj[keep]])
    return pd.DataFrame(
        {
            "i": np.concatenate(found_i).astype(int),
            "j": np.concatenate(found_j).astype(int),
            "phash_distance": np.concatenate(found_d).astype(int),
        }
    )


def embedding_pairs(embeddings: np.ndarray, min_similarity: float, chunk: int = 1024) -> pd.DataFrame:
    """Pares (i, j), i < j, con similitud coseno >= min_similarity."""
    e = _l2(embeddings)
    found_i, found_j, found_s = [], [], []
    for start in range(0, len(e), chunk):
        sims = e[start : start + chunk] @ e.T
        ii, jj = np.nonzero(sims >= min_similarity)
        ii = ii + start
        keep = ii < jj
        found_i.append(ii[keep])
        found_j.append(jj[keep])
        found_s.append(sims[ii[keep] - start, jj[keep]])
    return pd.DataFrame(
        {
            "i": np.concatenate(found_i).astype(int),
            "j": np.concatenate(found_j).astype(int),
            "cosine": np.concatenate(found_s).astype(float),
        }
    )


def merge_pairs(phash: pd.DataFrame, embedding: pd.DataFrame) -> pd.DataFrame:
    """Une los candidatos de ambos metodos en una tabla (i, j, phash_distance, cosine).

    Un par que solo aparece en un metodo tiene NaN en la columna del otro; se
    completa despues con `fill_pair_scores` para poder compararlos.
    """
    return phash.merge(embedding, on=["i", "j"], how="outer").sort_values(["i", "j"]).reset_index(drop=True)


def fill_pair_scores(pairs: pd.DataFrame, hashes: np.ndarray, embeddings: np.ndarray) -> pd.DataFrame:
    """Calcula la distancia pHash y el coseno de cada par (tambien los que faltaban)."""
    out = pairs.copy()
    i = out["i"].to_numpy()
    j = out["j"].to_numpy()
    h = np.asarray(hashes, dtype=np.uint64)
    out["phash_distance"] = np.bitwise_count(h[i] ^ h[j]).astype(int)
    e = _l2(embeddings)
    out["cosine"] = np.einsum("nd,nd->n", e[i], e[j]).astype(float)
    return out


@dataclass(frozen=True)
class DuplicateCriterion:
    """Cuando un par candidato se considera la misma foto.

    pHash cercano solo cuenta si los embeddings lo confirman (fondos lisos o
    composiciones simples pueden dar pHash parecidos sin ser la misma foto); un
    coseno muy alto cuenta solo, porque captura recortes, zoom y cambios de color
    que el pHash no ve.
    """

    max_phash_distance: int
    min_cosine_with_phash: float
    min_cosine_alone: float


def confirm_duplicates(pairs: pd.DataFrame, c: DuplicateCriterion) -> pd.DataFrame:
    """Filtra los pares (con phash_distance y cosine completos) que cumplen el criterio."""
    by_phash = (pairs["phash_distance"] <= c.max_phash_distance) & (pairs["cosine"] >= c.min_cosine_with_phash)
    by_cosine = pairs["cosine"] >= c.min_cosine_alone
    return pairs[by_phash | by_cosine].reset_index(drop=True)


def describe_pairs(pairs: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Agrega a cada par rutas, razas y splits de ambas imagenes y el tipo de par."""
    out = pairs.copy()
    for side in ("i", "j"):
        for col in ("path", "breed", "split"):
            out[f"{col}_{side}"] = table[col].to_numpy()[out[side].to_numpy()]
    out["same_breed"] = out["breed_i"] == out["breed_j"]
    out["kind"] = np.where(out["split_i"] == out["split_j"], "dentro de " + out["split_i"], "entre splits")
    return out


def duplicate_groups(n: int, pairs: pd.DataFrame) -> np.ndarray:
    """Componentes conexas de los pares (union-find): id de grupo por imagen."""
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(pairs["i"], pairs["j"]):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(x) for x in range(n)])


# --------------------------------------------------------------------------
# Decision de descarte
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityThresholds:
    """Umbrales de las reglas. Se eligen en la notebook mirando distribuciones y ejemplos."""

    min_side: int
    min_blur_var: float
    min_brightness: float
    max_brightness: float


def quality_flags(table: pd.DataFrame, t: QualityThresholds) -> pd.DataFrame:
    """Una columna booleana por regla de calidad (True = la imagen la incumple)."""
    readable = table["readable"].astype(bool)
    side = table[["height", "width"]].min(axis=1)
    return pd.DataFrame(
        {
            "corrupt": ~readable,
            "low_resolution": readable & (side < t.min_side),
            "blurry": readable & (table["blur_var"] < t.min_blur_var),
            "bad_exposure": readable
            & ((table["brightness"] < t.min_brightness) | (table["brightness"] > t.max_brightness)),
        }
    )


def duplicate_drops(table: pd.DataFrame, pairs: pd.DataFrame) -> pd.Series:
    """Motivo de descarte por duplicado ('' si se conserva).

    Si un grupo de copias tiene mas de una raza, la misma foto esta etiquetada de
    dos formas y no se puede saber cual es la correcta: se descartan todas.
    En los demas grupos se conserva una sola imagen: la de train si hay (asi la
    fuga se quita de valid/test y la imagen sigue sirviendo para entrenar), si no
    la de valid antes que la de test (test queda lo mas intacto posible), y a
    igual split la primera por ruta.
    """
    reasons = pd.Series("", index=table.index, dtype=object)
    if pairs.empty:
        return reasons
    frame = pd.DataFrame(
        {
            "group": duplicate_groups(len(table), pairs),
            "prio": table["split"].map(SPLIT_PRIORITY).to_numpy(),
            "path": table["path"].to_numpy(),
            "breed": table["breed"].to_numpy(),
        },
        index=table.index,
    )
    sizes = frame["group"].map(frame["group"].value_counts())
    breeds_in_group = frame["group"].map(frame.groupby("group")["breed"].nunique())
    conflict = (sizes > 1) & (breeds_in_group > 1)
    reasons[conflict] = "etiqueta contradictoria"

    in_group = frame[(sizes > 1) & ~conflict].sort_values(["group", "prio", "path"])
    keeper_prio = in_group.groupby("group")["prio"].transform("first")
    dropped = in_group.index[in_group.groupby("group").cumcount() > 0]
    crosses = in_group.loc[dropped, "prio"] != keeper_prio.loc[dropped]
    reasons.loc[dropped] = np.where(crosses, "fuga entre splits", "duplicado dentro del split")
    return reasons


def build_drop_table(table: pd.DataFrame, pairs: pd.DataFrame, t: QualityThresholds) -> pd.DataFrame:
    """Tabla con el motivo de descarte de cada imagen ('' = se conserva).

    `pairs` son los duplicados ya confirmados (columnas i, j); el criterio de
    confirmacion (pHash y/o coseno) se decide y documenta en la notebook.
    Prioridad del motivo: corrupta, luego duplicado/fuga, luego calidad. Las
    reglas de calidad solo descartan en train; corruptas y duplicados en
    cualquier split.
    """
    flags = quality_flags(table, t)
    dup = duplicate_drops(table, pairs)
    is_train = table["split"] == "train"

    reason = pd.Series("", index=table.index, dtype=object)
    for rule, label in (
        ("bad_exposure", "exposicion extrema"),
        ("blurry", "borrosa"),
        ("low_resolution", "resolucion baja"),
    ):
        reason = reason.mask(flags[rule] & is_train, label)
    reason = reason.mask(dup != "", dup)
    reason = reason.mask(flags["corrupt"], "corrupta")

    out = table[["path", "breed", "split"]].copy()
    out = pd.concat([out, flags], axis=1)
    out["drop_reason"] = reason
    return out
