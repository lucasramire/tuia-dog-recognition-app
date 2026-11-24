"""Contrato de los modelos entrenados y de sus checkpoints.

Todo modelo que se entrene (Modelo A y Modelo B) hereda de `EmbeddingClassifier`
y se guarda con `save_checkpoint`, que usa `torch.save(model)` (modelo completo,
no `state_dict`). Asi `ClassifierService.load_model` lo recupera con
`torch.load(..., weights_only=False)` y encuentra:

  - `classes`: lista de razas; el indice i del logit corresponde a `classes[i]`.
  - `embedding_dim`: dimension del vector devuelto por `embed`.
  - `embed(x)`: embedding previo al clasificador, forma (N, embedding_dim),
    sin normalizar (la normalizacion L2 la aplica quien lo consume).
  - `forward(x)`: logits de forma (N, len(classes)).

La clase concreta debe ser importable desde `src/lib/` (por ejemplo
`lib/models/cnn_custom.py`), si no `torch.load` falla. `torch.load` con
`weights_only=False` ejecuta pickle: cargar solo checkpoints propios.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


class EmbeddingClassifier(nn.Module, ABC):
    """Clasificador de razas que tambien expone su embedding."""

    def __init__(self, classes: list[str], embedding_dim: int) -> None:
        super().__init__()
        classes = list(classes)
        if not classes:
            raise ValueError("classes no puede estar vacia")
        if len(set(classes)) != len(classes):
            raise ValueError("classes tiene razas repetidas")
        if int(embedding_dim) <= 0:
            raise ValueError(f"embedding_dim debe ser positivo, llego {embedding_dim}")
        self.classes: list[str] = classes
        self.embedding_dim: int = int(embedding_dim)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @abstractmethod
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding previo al clasificador, forma (N, embedding_dim), sin normalizar."""


def validate_model_contract(model: object) -> None:
    """Lanza ValueError si el modelo no cumple el contrato (atributos, no ejecucion)."""
    if not isinstance(model, EmbeddingClassifier):
        raise ValueError(
            f"El modelo debe heredar de EmbeddingClassifier, es {type(model).__name__}"
        )
    classes = getattr(model, "classes", None)
    if not isinstance(classes, list) or not classes or not all(isinstance(c, str) for c in classes):
        raise ValueError("model.classes debe ser una lista no vacia de nombres de raza")
    if len(set(classes)) != len(classes):
        raise ValueError("model.classes tiene razas repetidas")
    dim = getattr(model, "embedding_dim", None)
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("model.embedding_dim debe ser un entero positivo")


def save_checkpoint(model: EmbeddingClassifier, path: Path) -> None:
    """Guarda el modelo completo (no el state_dict) validando el contrato."""
    validate_model_contract(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    torch.save(model, path)
    logger.info("Checkpoint guardado en %s (%d razas, embedding %d)", path, model.num_classes, model.embedding_dim)


def load_checkpoint(path: Path) -> EmbeddingClassifier:
    """Carga un checkpoint propio y valida el contrato. Solo para archivos de confianza."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Checkpoint not found: {path}")
    model = torch.load(path, map_location="cpu", weights_only=False)
    validate_model_contract(model)
    return model.eval()
