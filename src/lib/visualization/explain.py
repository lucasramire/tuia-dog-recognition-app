"""Imagenes para el analisis de errores y la deteccion: Grad-CAM superpuesto y cajas de YOLO."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from lib.vision.preprocessing import denormalize_image, preprocess_image


def model_input_rgb(image_bgr: np.ndarray, image_size: int) -> np.ndarray:
    """Lo que ve el modelo: la imagen redimensionada con padding, en RGB uint8 (S, S, 3)."""
    return denormalize_image(preprocess_image(image_bgr, image_size))


def overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superpone un mapa en [0, 1] (mismo alto y ancho que la imagen) con la paleta JET."""
    if rgb.shape[:2] != heatmap.shape:
        raise ValueError(f"La imagen {rgb.shape[:2]} y el mapa {heatmap.shape} tienen tamanos distintos")
    colored = cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET)[:, :, ::-1]
    return np.uint8(np.clip((1 - alpha) * rgb + alpha * colored, 0, 255))


def draw_boxes(
    image_bgr: np.ndarray, boxes: Sequence[tuple[tuple[int, int, int, int], float]], max_side: int = 640
) -> np.ndarray:
    """Cajas y confianzas sobre la imagen, reducida a `max_side` para mostrarla. Devuelve RGB."""
    canvas = image_bgr.copy()
    thickness = max(2, round(max(canvas.shape[:2]) / 300))
    for (x1, y1, x2, y2), conf in boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), thickness)
        cv2.putText(canvas, f"{conf:.2f}", (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * thickness, (0, 220, 0), thickness)
    scale = max_side / max(canvas.shape[:2])
    if scale < 1:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return canvas[:, :, ::-1].copy()
