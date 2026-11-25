"""Calculos del analisis exploratorio (P1.1 y P1.2, enunciado 3.1 y 3.2).

La notebook 1 llama a estas funciones y muestra tablas y graficos; aca solo
hay calculos, sin dibujar.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Clases de COCO que interesan para describir las escenas.
COCO_PERSON, COCO_CAT, COCO_DOG = 0, 15, 16


# --------------------------------------------------------------------------
# Distribucion de clases
# --------------------------------------------------------------------------


def class_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    """Imagenes por raza (filas) y split (columnas train, valid, test)."""
    counts = manifest.groupby(["breed", "split"]).size().unstack(fill_value=0)
    return counts.reindex(columns=[c for c in ("train", "valid", "test") if c in counts.columns])


def split_statistics(counts: pd.DataFrame) -> pd.DataFrame:
    """Media, mediana, minimo, maximo, desvio y razon max/min de imagenes por raza en cada split."""
    stats = counts.agg(["sum", "mean", "median", "min", "max", "std"]).T
    stats["max_over_min"] = stats["max"] / stats["min"]
    return stats.round(2)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de Wilson para una proporcion (accuracy por raza con n imagenes)."""
    if n <= 0:
        raise ValueError("n debe ser positivo")
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


# --------------------------------------------------------------------------
# Razas faciles de confundir (con embeddings del baseline)
# --------------------------------------------------------------------------


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def nearest_neighbors(embeddings: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Indice del vecino mas cercano (coseno) de cada fila, sin contarse a si misma."""
    e = _l2(embeddings)
    nearest = np.empty(len(e), dtype=int)
    for start in range(0, len(e), chunk):
        sims = e[start : start + chunk] @ e.T
        rows = np.arange(sims.shape[0])
        sims[rows, rows + start] = -np.inf
        nearest[start : start + chunk] = sims.argmax(axis=1)
    return nearest


def nearest_neighbor_confusions(embeddings: np.ndarray, breeds: np.ndarray) -> pd.DataFrame:
    """Pares de razas cuyo vecino mas cercano (1-NN coseno, sin contarse a si misma) es de la otra.

    Mide que tan mezcladas estan dos razas en el espacio del baseline: es la
    confusion que va a sufrir la busqueda por similitud de la Etapa 1. Devuelve
    una fila por par no ordenado con la cantidad de imagenes confundidas en
    ambos sentidos y la fraccion respecto del total de imagenes del par.
    """
    breeds = np.asarray(breeds)
    frame = pd.DataFrame({"a": breeds, "b": breeds[nearest_neighbors(embeddings)]})
    wrong = frame[frame["a"] != frame["b"]].copy()
    wrong["pair"] = [tuple(sorted(p)) for p in zip(wrong["a"], wrong["b"])]
    sizes = pd.Series(breeds).value_counts()
    out = wrong.groupby("pair").size().rename("confused").reset_index()
    out["breed_a"] = out["pair"].str[0]
    out["breed_b"] = out["pair"].str[1]
    out["images_in_pair"] = out["breed_a"].map(sizes) + out["breed_b"].map(sizes)
    out["fraction"] = out["confused"] / out["images_in_pair"]
    return out.drop(columns="pair").sort_values("confused", ascending=False).reset_index(drop=True)


def nearest_neighbor_accuracy(embeddings: np.ndarray, breeds: np.ndarray) -> pd.Series:
    """Por raza, fraccion de imagenes cuyo 1-NN es de la misma raza (dificultad relativa)."""
    breeds = np.asarray(breeds)
    hits = breeds[nearest_neighbors(embeddings)] == breeds
    return pd.Series(hits, index=breeds).groupby(level=0).mean().sort_values()


# --------------------------------------------------------------------------
# Encuadre y contenido de las escenas (YOLO preentrenado en COCO)
# --------------------------------------------------------------------------


def summarize_detections(
    classes: np.ndarray, confidences: np.ndarray, boxes_xyxy: np.ndarray, height: int, width: int
) -> dict[str, float]:
    """Resume las detecciones de una imagen: cantidad por clase y encuadre del perro mas grande."""
    classes = np.asarray(classes, dtype=int)
    dogs = classes == COCO_DOG
    row: dict[str, float] = {
        "n_dogs": int(dogs.sum()),
        "n_persons": int((classes == COCO_PERSON).sum()),
        "n_cats": int((classes == COCO_CAT).sum()),
        "dog_conf": float(np.max(confidences[dogs])) if dogs.any() else 0.0,
        "dog_area_frac": 0.0,
        "dog_box_aspect": float("nan"),
    }
    if dogs.any():
        b = np.asarray(boxes_xyxy, dtype=float)[dogs]
        areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        k = int(areas.argmax())
        row["dog_area_frac"] = float(areas[k] / (height * width))
        row["dog_box_aspect"] = float((b[k, 2] - b[k, 0]) / max(b[k, 3] - b[k, 1], 1e-6))
    return row


def run_yolo_framing(
    paths: list[Path],
    model_path: Path,
    conf: float,
    iou: float,
    imgsz: int,
    batch_size: int = 32,
    device: str | int | None = None,
) -> pd.DataFrame:
    """Pasa YOLO por cada imagen y devuelve una fila por imagen con `summarize_detections`.

    Se usa el modelo preentrenado tal cual (clases de COCO) para describir el
    dataset: cuantos perros, personas o gatos hay y que fraccion de la imagen
    ocupa el perro. No es la funcion detect_dogs de la app (Etapa 3).

    `imgsz` importa: con imagenes de 224 px, el valor por defecto de ultralytics
    (640) las agranda casi 3 veces y la deteccion empeora mucho (medido en 600
    imagenes del dataset: perro detectado en 54,5% a 640 contra 84,5% a 320).
    """
    from ultralytics import YOLO  # import diferido: solo para el EDA

    model = YOLO(str(model_path))
    rows = []
    for start in range(0, len(paths), batch_size):
        chunk = [str(p) for p in paths[start : start + batch_size]]
        results = model.predict(chunk, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)
        for result in results:
            h, w = result.orig_shape
            boxes = result.boxes
            rows.append(
                summarize_detections(
                    boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.xyxy.cpu().numpy(), h, w
                )
            )
        if (start // batch_size) % 50 == 0:
            logger.info("YOLO: %d / %d imagenes", min(start + batch_size, len(paths)), len(paths))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Imagenes para grillas
# --------------------------------------------------------------------------


def load_rgb(path: Path, size: int | None = None) -> np.ndarray:
    """Lee una imagen para mostrarla (RGB); si `size`, la ajusta a un cuadrado con padding blanco."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"No se pudo leer {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if size is None:
        return rgb
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    top, left = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas
