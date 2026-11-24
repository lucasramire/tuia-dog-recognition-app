"""Extractor de embeddings del baseline de la Etapa 1: ResNet50 preentrenada en ImageNet.

El embedding es la salida del global average pooling, previa a la capa `fc`
(se reemplaza por `Identity`), normalizada en L2. Con vectores de norma 1 el
producto interno es la similitud coseno (criterio de class_4, seccion 6.12).

Pesos: `IMAGENET1K_V1`, explicitos, como en Class 1 (seccion 4.5). `DEFAULT`
hoy resuelve a `IMAGENET1K_V2`, que es otro checkpoint.

Lo usan `SimilarityService.extract_embedding` (una imagen) y las notebooks
(lotes), asi ambos caminos comparten modelo y preprocesamiento.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from lib.vision.preprocessing import IMAGENET_MEAN, IMAGENET_STD, preprocess_batch

logger = logging.getLogger(__name__)

BASELINE_WEIGHTS = ResNet50_Weights.IMAGENET1K_V1
BASELINE_EMBEDDING_DIM = 2048


def _check_normalization(weights: ResNet50_Weights) -> None:
    """Verifica que el preprocesamiento compartido use las estadisticas de los pesos.

    La catedra pide no fijar media y desvio sin contrastarlos con el modelo
    (class_4 6.8.2, Class_2_lab): una diferencia no da error, desplaza las
    activaciones en silencio.
    """
    transforms = weights.transforms()
    mean = np.asarray(transforms.mean, dtype=np.float32)
    std = np.asarray(transforms.std, dtype=np.float32)
    if not (np.allclose(mean, IMAGENET_MEAN) and np.allclose(std, IMAGENET_STD)):
        raise ValueError(
            f"La normalizacion de lib/vision/preprocessing.py (mean={IMAGENET_MEAN.tolist()}, "
            f"std={IMAGENET_STD.tolist()}) no coincide con la de {weights} "
            f"(mean={mean.tolist()}, std={std.tolist()})"
        )


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaselineEmbedder:
    """ResNet50 sin `fc`, cargada una sola vez, en modo evaluacion."""

    def __init__(self, image_size: int, device: torch.device | str | None = None) -> None:
        if int(image_size) <= 0:
            raise ValueError(f"image_size debe ser positivo, llego {image_size}")
        self.image_size = int(image_size)
        self.device = torch.device(device) if device is not None else default_device()
        self.embedding_dim = BASELINE_EMBEDDING_DIM

        _check_normalization(BASELINE_WEIGHTS)
        model = resnet50(weights=BASELINE_WEIGHTS)
        if model.fc.in_features != self.embedding_dim:
            raise ValueError(
                f"ResNet50 devuelve {model.fc.in_features} dimensiones, se esperaban {self.embedding_dim}"
            )
        model.fc = nn.Identity()
        self.model = model.to(self.device).eval()
        logger.info("Baseline ResNet50 (%s) cargada en %s", BASELINE_WEIGHTS, self.device)

    @torch.inference_mode()
    def embed_batch(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        """Embeddings L2-normalizados de imagenes BGR uint8, forma (N, 2048), float32."""
        batch = torch.from_numpy(preprocess_batch(images_bgr, self.image_size)).to(self.device)
        features = self.model(batch)
        return F.normalize(features, p=2, dim=1).cpu().numpy().astype(np.float32)

    def embed(self, image_bgr: np.ndarray) -> list[float]:
        """Embedding L2-normalizado de una imagen BGR uint8, como lista de floats de Python."""
        return self.embed_batch([image_bgr])[0].tolist()
