"""Inferencia por lotes con un modelo entrenado (contrato EmbeddingClassifier).

Lo usan las notebooks para calcular embeddings y probabilidades de muchas imagenes con el
mismo preprocesamiento que la app (`preprocess_batch`), sin pasar de a una por el servicio.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from lib.models.base import EmbeddingClassifier
from lib.vision.preprocessing import preprocess_batch


def embed_batch_fn(
    model: EmbeddingClassifier, image_size: int, normalize: bool = True
) -> Callable[[list[np.ndarray]], np.ndarray]:
    """Funcion lista de imagenes BGR uint8 -> embeddings (N, D) float32.

    Normalizados en L2 por defecto, igual que `extract_custom_embedding` y el baseline, asi la
    busqueda de la notebook mide lo mismo que la app. Sirve como `embed_batch` de
    `lib.evaluation.retrieval.compute_embeddings`.
    """
    model.eval()
    device = next(model.parameters()).device

    @torch.inference_mode()
    def embed(images_bgr: list[np.ndarray]) -> np.ndarray:
        batch = torch.from_numpy(preprocess_batch(images_bgr, image_size)).to(device)
        features = model.embed(batch)
        if normalize:
            features = F.normalize(features, p=2, dim=1)
        return features.float().cpu().numpy()

    return embed


def predict_proba_fn(model: EmbeddingClassifier, image_size: int) -> Callable[[list[np.ndarray]], np.ndarray]:
    """Funcion lista de imagenes BGR uint8 -> probabilidades softmax (N, razas) float32."""
    model.eval()
    device = next(model.parameters()).device

    @torch.inference_mode()
    def predict(images_bgr: list[np.ndarray]) -> np.ndarray:
        batch = torch.from_numpy(preprocess_batch(images_bgr, image_size)).to(device)
        return torch.softmax(model(batch).float(), dim=1).cpu().numpy()

    return predict
