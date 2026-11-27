"""Validacion de la deteccion de perros (Etapa 3, E3.2).

No hay cajas anotadas, pero si la cantidad de perros de cada imagen (1 en los splits y en las
imagenes de una raza del conjunto independiente, `n_dogs` en las escenas, 0 en las escenas sin
perros). Por eso se evalua el CONTEO, que es lo que la app necesita: un recorte por perro.

  - detection_rate: imagenes con perros donde se detecta al menos uno.
  - exact_count: imagenes donde se detectan exactamente `n_dogs` perros.
  - count_mae: error absoluto medio del conteo.
  - false_alarm: imagenes SIN perros donde se detecta alguno.

Para comparar umbrales se corre YOLO una sola vez con un umbral bajo (`conf_floor`) y despues se
filtran las cajas. Es equivalente a correrlo con cada umbral: el NMS solo suprime una caja con
otra de mayor confianza, asi que las cajas que superan un umbral no dependen de las que no.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import pandas as pd

DOG_CLASS_ID = 16  # COCO


def detect_confidences(
    yolo: object,
    images_bgr: Sequence[np.ndarray],
    imgsz: int = 640,
    conf_floor: float = 0.05,
    iou: float = 0.7,
    dog_class_id: int = DOG_CLASS_ID,
    device: str | None = None,
) -> tuple[list[np.ndarray], float]:
    """Confianzas de los perros detectados por imagen y la latencia mediana en ms por imagen.

    `device="cpu"` mide la latencia donde corre la app; la primera imagen incluye la carga
    del modelo, por eso se informa la mediana.
    """
    confidences, times = [], []
    extra = {} if device is None else {"device": device}
    for image in images_bgr:
        start = time.perf_counter()
        result = yolo.predict(
            image, conf=conf_floor, iou=iou, classes=[dog_class_id], imgsz=imgsz, verbose=False, **extra
        )[0]
        times.append((time.perf_counter() - start) * 1000.0)
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            confidences.append(np.zeros(0, dtype=np.float32))
            continue
        cls = boxes.cls.cpu().numpy().astype(int)
        confidences.append(boxes.conf.cpu().numpy()[cls == dog_class_id].astype(np.float32))
    return confidences, float(np.median(times)) if times else float("nan")


def count_table(
    confidences: Sequence[np.ndarray],
    n_dogs: Sequence[int],
    groups: Sequence[str],
    thresholds: Sequence[float],
) -> pd.DataFrame:
    """Metricas de conteo por grupo de imagenes y por umbral de confianza."""
    n_dogs = np.asarray(n_dogs, dtype=int)
    groups = np.asarray(groups, dtype=object)
    if not (len(confidences) == len(n_dogs) == len(groups)):
        raise ValueError("confidences, n_dogs y groups deben tener el mismo largo")
    rows = []
    for threshold in thresholds:
        counts = np.array([int((c >= threshold).sum()) for c in confidences])
        for group in pd.unique(groups):
            mask = groups == group
            truth, found = n_dogs[mask], counts[mask]
            with_dogs = truth > 0
            row = {"umbral": float(threshold), "grupo": group, "imagenes": int(mask.sum())}
            if with_dogs.any():
                row.update(
                    detection_rate=float((found[with_dogs] > 0).mean()),
                    exact_count=float((found[with_dogs] == truth[with_dogs]).mean()),
                    count_mae=float(np.abs(found[with_dogs] - truth[with_dogs]).mean()),
                )
            if (~with_dogs).any():
                row["false_alarm"] = float((found[~with_dogs] > 0).mean())
            rows.append(row)
    return pd.DataFrame(rows).set_index(["umbral", "grupo"])
