"""Modelo A: ResNet18 preentrenada en ImageNet con cabeza nueva (D-21 del plan).

Estrategias de transfer learning:
  - feature_extraction: backbone congelado (y sus BatchNorm en modo eval),
    solo se entrena la cabeza lineal.
  - finetune: todo entrenable, con learning rate menor en el backbone que en
    la cabeza para no borrar lo preentrenado (olvido catastrofico).

El embedding es la salida del global average pooling (512), previa a la cabeza.
Pesos IMAGENET1K_V1 explicitos, como el baseline (Class 1, 4.5).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from lib.models.base import EmbeddingClassifier
from lib.vision.preprocessing import IMAGENET_MEAN, IMAGENET_STD

RESNET18_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
RESNET18_EMBEDDING_DIM = 512
STRATEGIES = ("feature_extraction", "finetune")


def _check_normalization() -> None:
    """La normalizacion compartida debe ser la de los pesos (si no, desplaza activaciones en silencio)."""
    t = RESNET18_WEIGHTS.transforms()
    if not (np.allclose(t.mean, IMAGENET_MEAN) and np.allclose(t.std, IMAGENET_STD)):
        raise ValueError(
            f"La normalizacion de lib/vision/preprocessing.py no coincide con la de {RESNET18_WEIGHTS} "
            f"(mean={list(t.mean)}, std={list(t.std)})"
        )


class ResNetClassifier(EmbeddingClassifier):
    def __init__(self, classes: list[str], pretrained: bool = True, dropout: float = 0.0) -> None:
        super().__init__(classes, RESNET18_EMBEDDING_DIM)
        if pretrained:
            _check_normalization()
        backbone = resnet18(weights=RESNET18_WEIGHTS if pretrained else None)
        if backbone.fc.in_features != RESNET18_EMBEDDING_DIM:
            raise ValueError(f"ResNet18 devuelve {backbone.fc.in_features} dimensiones, se esperaban 512")
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(RESNET18_EMBEDDING_DIM, self.num_classes)
        self.backbone_frozen = False

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone_frozen = True
        self.train(self.training)

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.backbone_frozen = False
        self.train(self.training)

    def train(self, mode: bool = True) -> "ResNetClassifier":
        super().train(mode)
        # Con el backbone congelado, sus BatchNorm no deben actualizar estadisticas.
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def param_groups(self, lr_head: float, lr_backbone: float) -> list[dict]:
        groups = [{"params": list(self.head.parameters()), "lr": lr_head, "name": "head"}]
        if not self.backbone_frozen:
            groups.append({"params": list(self.backbone.parameters()), "lr": lr_backbone, "name": "backbone"})
        return groups

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.embed(x)))
