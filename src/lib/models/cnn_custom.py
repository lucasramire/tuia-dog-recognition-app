"""Modelo B: CNN propia entrenada desde cero (D-22 del plan).

Arquitectura base: 5 bloques conv3x3 - BatchNorm - ReLU - maxpool 2x2 con
canales 32, 64, 128, 256, 256; global average pooling; capa lineal de
embedding (256) con ReLU y dropout; clasificador lineal.

- BatchNorm estabiliza y acelera el entrenamiento desde cero.
- GAP en lugar de aplanar: muchos menos parametros y algo de invariancia a la
  posicion, importante con ~100 imagenes por raza.
- La capa de embedding expone la representacion para la busqueda por
  similitud (Etapa 1 con EMBEDDING_MODEL=cnn_custom).

La clase vive en un modulo importable para que `torch.load` del modelo
completo funcione desde los servicios.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from lib.models.base import EmbeddingClassifier

DEFAULT_CHANNELS: tuple[int, ...] = (32, 64, 128, 256, 256)


def _conv_block(in_ch: int, out_ch: int, batchnorm: bool) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not batchnorm)]
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
    return nn.Sequential(*layers)


class CustomCNN(EmbeddingClassifier):
    def __init__(
        self,
        classes: list[str],
        embedding_dim: int = 256,
        channels: Sequence[int] = DEFAULT_CHANNELS,
        dropout: float = 0.3,
        batchnorm: bool = True,
    ) -> None:
        super().__init__(classes, embedding_dim)
        if not channels:
            raise ValueError("channels no puede estar vacio")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout debe estar en [0, 1), llego {dropout}")
        self.channels = tuple(int(c) for c in channels)
        self.dropout_p = float(dropout)
        self.batchnorm = bool(batchnorm)

        blocks = []
        in_ch = 3
        for out_ch in self.channels:
            blocks.append(_conv_block(in_ch, out_ch, self.batchnorm))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(nn.Linear(in_ch, self.embedding_dim), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(self.dropout_p)
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.pool(self.features(x)).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.embed(x)))
