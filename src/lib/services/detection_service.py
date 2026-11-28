from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import torch

from lib.models.export import ONNX_CLASSES_KEY
from lib.schemas import ClassifyResult, DetectResult, DogDetection
from lib.services.classifier_service import ClassifierService
from lib.vision.preprocessing import preprocess_batch

logger = logging.getLogger(__name__)

UNKNOWN_BREED = "unknown"

# Tamano de entrada de YOLO adaptativo (E3.2): YOLO escala la imagen para que su lado mayor mida
# `imgsz`. Agrandar una imagen chica (224 px a 640) deja al perro gigante y borroso, a una escala
# que la red casi no vio en COCO, y deja de detectarlo; achicar una foto grande a 320 pierde los
# perros lejanos. Por eso se usa el lado mayor real, sin agrandar por debajo de 320 ni pasar de 640.
YOLO_IMGSZ_MIN = 320
YOLO_IMGSZ_MAX = 640
YOLO_STRIDE = 32  # YOLO necesita lados multiplos del stride maximo de la red


def adaptive_imgsz(height: int, width: int, low: int = YOLO_IMGSZ_MIN, high: int = YOLO_IMGSZ_MAX) -> int:
    """Lado mayor de la imagen redondeado hacia arriba a multiplo de 32 y limitado a [low, high]."""
    side = -(-max(int(height), int(width)) // YOLO_STRIDE) * YOLO_STRIDE
    return int(min(max(side, low), high))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class DetectionService:
    """Etapa 3: pipeline de deteccion y clasificacion.

    Funciones a implementar por el estudiante:
      - detect_dogs(image)
      - classify_detected_dog(crop)

    La orquestacion (predict: deteccion -> recorte -> clasificacion -> JSON)
    ya esta provista.
    """

    def __init__(
        self,
        classifier: ClassifierService,
        yolo_model: str,
        conf_threshold: float,
        dog_class_id: int,
        iou_threshold: float | None = None,
        min_confidence: float | None = None,
        imgsz: int | None = None,
    ) -> None:
        self.classifier = classifier
        self.yolo_model_name = yolo_model
        self.conf_threshold = conf_threshold
        self.dog_class_id = dog_class_id
        # Parametros agregados (justificados en el PR): si no se pasan, salen del .env.
        if iou_threshold is None or min_confidence is None:
            from lib.config import settings

            iou_threshold = settings.yolo_iou_threshold if iou_threshold is None else iou_threshold
            min_confidence = settings.classifier_min_confidence if min_confidence is None else min_confidence
        self.iou_threshold = float(iou_threshold)
        self.min_confidence = float(min_confidence)
        # Lado mayor al que YOLO redimensiona la imagen. None = adaptativo por imagen (adaptive_imgsz);
        # un entero lo fija (por ejemplo 640, el valor por defecto de ultralytics).
        if imgsz is not None and int(imgsz) <= 0:
            raise ValueError(f"imgsz debe ser positivo, llego {imgsz}")
        self.imgsz = None if imgsz is None else int(imgsz)
        self._yolo: Any = None
        self._yolo_lock = threading.Lock()

    def _get_yolo(self) -> Any:
        """Carga el modelo YOLO una sola vez (no en cada request)."""
        if self._yolo is None:
            with self._yolo_lock:
                if self._yolo is None:
                    from ultralytics import YOLO

                    logger.info("Cargando YOLO desde %s", self.yolo_model_name)
                    self._yolo = YOLO(self.yolo_model_name)
        return self._yolo

    @staticmethod
    def _clip_xyxy(
        x1: int, y1: int, x2: int, y2: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1:
            x2 = min(x1 + 1, width)
        if y2 <= y1:
            y2 = min(y1 + 1, height)
        return x1, y1, x2, y2

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV / ultralytics)
        return image

    # ------------------------------------------------------------------
    # Etapa 3: funciones a implementar
    # ------------------------------------------------------------------

    def detect_dogs(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        """
        Detecta todos los perros presentes en la imagen usando un modelo YOLO
        pre-entrenado (ej: YOLOv8n via ultralytics). No es necesario entrenar
        el detector.

        Sugerencias:
          - self.yolo_model_name, self.conf_threshold y self.dog_class_id
            (clase 'dog' = 16 en COCO) vienen de la configuracion (.env).
          - Debe funcionar con un perro, multiples perros y escenas complejas.

        Retorna una lista de ((x1, y1, x2, y2), confidence) en pixeles.
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError(f"detect_dogs espera una imagen BGR (alto, ancho, 3); llego {getattr(image, 'shape', type(image))}")
        height, width = image.shape[:2]

        # ultralytics interpreta los np.ndarray como BGR, igual que OpenCV.
        imgsz = self.imgsz if self.imgsz is not None else adaptive_imgsz(height, width)
        results = self._get_yolo().predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.dog_class_id],
            imgsz=imgsz,
            verbose=False,
        )
        detections: list[tuple[tuple[int, int, int, int], float]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), conf, c in zip(xyxy, confs, cls):
                if c != self.dog_class_id:
                    continue
                box = self._clip_xyxy(
                    int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), height, width
                )
                detections.append((box, float(conf)))
        detections.sort(key=lambda item: item[1], reverse=True)
        return detections

    def classify_detected_dog(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Clasifica la raza del recorte de un perro detectado usando el modelo
        entrenado en la Etapa 2 (self.classifier.load_model()).

        El recorte llega en BGR (OpenCV). Retorna (raza, score).
        """
        probs, classes = self._predict_probabilities(crop)
        best = int(np.argmax(probs))
        score = float(probs[best])
        if score < self.min_confidence:
            return UNKNOWN_BREED, score
        return classes[best], score

    def _predict_probabilities(self, crop: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Probabilidades softmax del modelo activo (.pth o .onnx) y la lista de razas."""
        # Mismo preprocesamiento que en el entrenamiento (valida la entrada).
        batch = preprocess_batch([crop], self.classifier.image_size)
        model = self.classifier.load_model()

        if isinstance(model, torch.nn.Module):
            classes = getattr(model, "classes", None)
            if not classes:
                raise ValueError("El checkpoint no tiene el atributo 'classes' (lista de razas)")
            device = next(model.parameters()).device
            model.eval()
            with torch.inference_mode():
                logits = model(torch.from_numpy(batch).to(device))
            probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
            return probs, list(classes)

        # onnxruntime.InferenceSession
        meta = model.get_modelmeta().custom_metadata_map
        if ONNX_CLASSES_KEY not in meta:
            raise ValueError(
                f"El modelo ONNX no tiene la lista de razas en sus metadatos ('{ONNX_CLASSES_KEY}'). "
                "Exportalo con lib.models.export."
            )
        classes = json.loads(meta[ONNX_CLASSES_KEY])
        input_name = model.get_inputs()[0].name
        logits = model.run(None, {input_name: batch})[0]
        probs = _softmax(np.asarray(logits, dtype=np.float32))[0]
        if probs.shape[0] != len(classes):
            raise ValueError(f"El ONNX devuelve {probs.shape[0]} clases y sus metadatos listan {len(classes)}")
        return probs, list(classes)

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def classify_image(
        self, source_path: str, output_path: Path, model_name: str | None = None
    ) -> str:
        """Clasifica la imagen completa con el modelo entrenado (pestaña Etapa 2).

        Reutiliza classify_detected_dog tratando la imagen entera como recorte,
        por lo que requiere la Etapa 2 (modelo entrenado) y classify_detected_dog.
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        # El modelo elegido vale solo para esta clasificacion: se restaura el activo para
        # que /detect siga usando el modelo configurado.
        previous = self.classifier.active_model_name
        if model_name:
            self.classifier.set_active_model(model_name)
        try:
            breed, score = self.classify_detected_dog(image)
        finally:
            self.classifier.set_active_model(previous)
        payload = ClassifyResult(
            source_path=source_path,
            model=model_name or self.classifier.active_model_name,
            breed=breed,
            score=round(float(score), 4),
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)

    def predict(self, source_path: str, output_path: Path) -> str:
        """Flujo completo: deteccion -> bounding boxes -> recortes -> clasificacion.

        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        height, width = image.shape[:2]

        detections: list[DogDetection] = []
        for (box, det_score) in self.detect_dogs(image):
            x1, y1, x2, y2 = self._clip_xyxy(*[int(v) for v in box], height, width)
            crop = image[y1:y2, x1:x2]
            breed, breed_score = self.classify_detected_dog(crop)
            detections.append(
                DogDetection(
                    bbox=[x1, y1, x2, y2],
                    det_score=round(float(det_score), 4),
                    breed=breed,
                    breed_score=round(float(breed_score), 4),
                )
            )

        detected_breeds = sorted({item.breed for item in detections if item.breed != "unknown"})
        payload = DetectResult(
            source_path=source_path,
            detections=detections,
            detected_breeds=detected_breeds,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
