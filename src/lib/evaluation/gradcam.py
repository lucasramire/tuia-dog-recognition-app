"""Grad-CAM propio con hooks de PyTorch (D-10).

Grad-CAM (Selvaraju et al., 2017) explica una prediccion con un mapa de calor sobre la imagen:

  1. Se toman las activaciones A_k (k = canal) de la ultima capa convolucional, que todavia
     conservan la posicion espacial (en la ResNet18 son 512 mapas de 7x7 para 224 px).
  2. Se calcula el gradiente del score de la clase (el logit, antes del softmax) respecto de
     cada A_k. Su promedio espacial alpha_k mide cuanto le importa ese canal a la clase.
  3. El mapa es ReLU(sum_k alpha_k * A_k): solo quedan las zonas que empujan a favor de la
     clase. Se reescala al tamano de la entrada y se normaliza a [0, 1] para mostrarlo.

Se usa la ultima capa convolucional porque es la de mayor nivel semantico que todavia tiene
resolucion espacial; despues viene el pooling global, que la pierde.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lib.models.base import EmbeddingClassifier


def default_target_layer(model: EmbeddingClassifier) -> nn.Module:
    """Ultima capa convolucional de los modelos del TP (ResNet18 y CNN propia)."""
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "layer4"):
        return backbone.layer4  # ResNet: salida del ultimo bloque residual (512 x 7 x 7)
    features = getattr(model, "features", None)
    if isinstance(features, nn.Sequential) and len(features) > 0:
        return features[-1]  # CNN propia: salida del ultimo bloque conv (256 x 7 x 7)
    raise ValueError(f"No se sabe cual es la ultima capa convolucional de {type(model).__name__}")


class GradCAM:
    """Mapa Grad-CAM de un modelo y una capa. Usar con `with GradCAM(...) as cam:` para quitar los hooks."""

    def __init__(self, model: EmbeddingClassifier, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.layer = target_layer if target_layer is not None else default_target_layer(model)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            self.layer.register_forward_hook(self._save_activations),
            self.layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradients(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        self._gradients = grad_output[0].detach()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc: object) -> None:
        self.remove()

    def __call__(self, x: torch.Tensor, class_index: int | None = None) -> tuple[np.ndarray, int, float]:
        """Mapa (H, W) en [0, 1] para una imagen (1, 3, H, W) ya preprocesada.

        `class_index=None` explica la clase predicha. Devuelve (mapa, clase, probabilidad).
        """
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError(f"GradCAM espera un tensor (1, 3, H, W); llego {tuple(x.shape)}")
        was_training = self.model.training
        self.model.eval()
        device = next(self.model.parameters()).device
        x = x.to(device).requires_grad_(True)  # hace falta para el hook de backward con capas congeladas
        try:
            with torch.enable_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                index = int(probs.argmax(dim=1)) if class_index is None else int(class_index)
                self.model.zero_grad(set_to_none=True)
                logits[0, index].backward()
        finally:
            self.model.train(was_training)
        if self._activations is None or self._gradients is None:
            raise RuntimeError("Los hooks no capturaron activaciones o gradientes")
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # alpha_k: (1, K, 1, 1)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        peak = float(cam.max())
        cam = cam / peak if peak > 0 else cam
        return cam.cpu().numpy(), index, float(probs[0, index].detach())
