"""Modelo diminuto que cumple el contrato, para probar sin entrenar.

Sirve para escribir y probar `extract_custom_embedding`, `classify_detected_dog`,
el servicio de clasificacion y las notebooks antes de que exista un modelo real.
No es un modelo de verdad: sus predicciones no significan nada.
"""

from __future__ import annotations

import torch
from torch import nn

from lib.models.base import EmbeddingClassifier


class DummyEmbeddingClassifier(EmbeddingClassifier):
    def __init__(self, classes: list[str], embedding_dim: int = 16, seed: int = 0) -> None:
        super().__init__(classes, embedding_dim)
        generator = torch.get_rng_state()
        torch.manual_seed(seed)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.to_embedding = nn.Linear(3, embedding_dim)
        self.head = nn.Linear(embedding_dim, len(classes))
        torch.set_rng_state(generator)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_embedding(self.pool(x).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))
