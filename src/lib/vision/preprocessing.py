"""Preprocesamiento compartido entre notebooks, entrenamiento y servicios.

Este es el unico lugar donde se decide como una imagen BGR (convencion de
OpenCV) se convierte en la entrada de los modelos. Entrenamiento, evaluacion,
busqueda por similitud y deteccion deben importar de aqui: si dos caminos
preprocesan distinto, los embeddings y las predicciones dejan de ser comparables.

Pasos: BGR a RGB, resize conservando la proporcion, normalizacion con
estadisticas de ImageNet, relleno con ceros (la media de ImageNet una vez
normalizada) hasta un cuadrado de IMAGE_SIZE y orden CHW.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _validate_bgr(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Se esperaba un np.ndarray, llego {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ValueError(f"Se esperaba una imagen uint8, llego dtype={image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Se esperaba una imagen BGR (alto, ancho, 3), llego shape={image.shape}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"La imagen esta vacia: shape={image.shape}")


def _resolve_size(image_size: int | None) -> int:
    if image_size is not None:
        size = int(image_size)
    else:
        # Import diferido: evita cargar la configuracion al importar el modulo.
        from lib.config import settings

        size = int(settings.image_size)
    if size <= 0:
        raise ValueError(f"image_size debe ser positivo, llego {size}")
    return size


def preprocess_image(image_bgr: np.ndarray, image_size: int | None = None) -> np.ndarray:
    """Convierte una imagen BGR uint8 en un arreglo float32 de forma (3, S, S).

    La imagen de entrada no se modifica. Si `image_size` es None se usa
    `settings.image_size`. Los bordes de relleno valen 0 tras normalizar.
    """
    _validate_bgr(image_bgr)
    size = _resolve_size(image_size)

    height, width = image_bgr.shape[:2]
    scale = size / max(height, width)
    new_w = max(1, min(size, round(width * scale)))
    new_h = max(1, min(size, round(height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=interpolation)

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

    canvas = np.zeros((size, size, 3), dtype=np.float32)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = normalized

    return np.ascontiguousarray(canvas.transpose(2, 0, 1))


def preprocess_batch(images_bgr: list[np.ndarray], image_size: int | None = None) -> np.ndarray:
    """Apila varias imagenes BGR en un arreglo float32 de forma (N, 3, S, S)."""
    if not images_bgr:
        raise ValueError("preprocess_batch recibio una lista vacia")
    return np.stack([preprocess_image(img, image_size) for img in images_bgr])


def preprocess_to_tensor(image_bgr: np.ndarray, image_size: int | None = None):
    """Igual que preprocess_image pero devuelve un torch.Tensor (3, S, S)."""
    import torch

    return torch.from_numpy(preprocess_image(image_bgr, image_size))
