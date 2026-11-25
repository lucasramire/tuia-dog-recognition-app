"""Data augmentation online para entrenamiento (P1.5, enunciado 3.6; D-16 del plan).

Se aplica solo a train, en cada epoca y sobre la imagen BGR uint8 original;
despues pasa por el preprocesamiento compartido (`lib/vision/preprocessing.py`),
asi entrenamiento e inferencia comparten un unico camino. valid, test y la app
usan solo el preprocesamiento.

Transformaciones (todas conservan la raza y ocurren en fotos reales):
  - Recorte desde los bordes con proporcion variable: simula los recortes de
    YOLO de la Etapa 3, que suelen ser mas ajustados que el encuadre del
    dataset y cortan partes del perro o cambian la relacion de aspecto.
  - Flip horizontal: un perro mirando a la izquierda sigue siendo de la misma raza.
  - Rotacion suave: camara inclinada. El borde se rellena con la media de
    ImageNet, igual que el padding del preprocesamiento (vale 0 al normalizar).
  - Brillo y contraste: iluminacion distinta.
  - Desenfoque y ruido suaves: fotos de celular, movimiento, poca luz.

Excluidas: flip vertical y rotaciones de 90 grados (los perros no aparecen
cabeza abajo; Class_2_lab los usa porque las imagenes satelitales no tienen
"arriba"), y cambios de tono o saturacion fuertes (el color del pelaje
distingue razas).

La implementacion usa Albumentations, que ya esta en el requirements.txt provisto.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import albumentations as A
import cv2
import numpy as np

from lib.vision.preprocessing import IMAGENET_MEAN, preprocess_image

logger = logging.getLogger(__name__)

# Media de ImageNet en BGR uint8 (la imagen llega en BGR): el relleno de la rotacion
# queda en 0 despues de normalizar, igual que el padding del preprocesamiento.
IMAGENET_MEAN_BGR = tuple(int(round(v * 255)) for v in IMAGENET_MEAN[::-1])


@dataclass(frozen=True)
class AugmentationConfig:
    """Intensidad y probabilidad de cada transformacion. Los valores salen de D-16."""

    crop_border: float = 0.15  # fraccion maxima que se recorta de cada lado
    crop_p: float = 0.5
    hflip_p: float = 0.5
    rotate_degrees: float = 12.0
    rotate_p: float = 0.4
    brightness: float = 0.2
    contrast: float = 0.2
    color_p: float = 0.5
    blur_sigma: tuple[float, float] = (0.3, 1.2)
    blur_p: float = 0.15
    noise_std: tuple[float, float] = (0.02, 0.06)  # fraccion del rango 0-255
    noise_p: float = 0.15


def seed_pipeline(pipeline: A.Compose, seed: int | None) -> None:
    """Siembra cada transformacion con una semilla distinta derivada de `seed`.

    `A.Compose(seed=s)` (y `Compose.set_random_seed`) le pasa la MISMA semilla a
    todas las transformaciones: transformaciones con igual probabilidad quedan
    perfectamente correlacionadas (por ejemplo flip y recorte siempre juntos) y
    sus parametros aleatorios tambien. Medido en Albumentations 2.0.8: con dos
    flips de p=0,5 y semilla, nunca se aplica uno solo. Con SeedSequence cada
    transformacion recibe una semilla independiente y el resultado sigue siendo
    reproducible.
    """
    if seed is None:
        return
    pipeline.set_random_seed(seed)
    children = np.random.SeedSequence(seed).spawn(len(pipeline.transforms))
    for transform, child in zip(pipeline.transforms, children):
        transform.set_random_seed(int(child.generate_state(1)[0]))


def build_train_augmentation(config: AugmentationConfig = AugmentationConfig(), seed: int | None = None) -> A.Compose:
    """Pipeline de Albumentations para train (entrada y salida: BGR uint8, tamano variable)."""
    pipeline = A.Compose(
        [
            A.RandomCropFromBorders(
                crop_left=config.crop_border,
                crop_right=config.crop_border,
                crop_top=config.crop_border,
                crop_bottom=config.crop_border,
                p=config.crop_p,
            ),
            A.HorizontalFlip(p=config.hflip_p),
            A.Affine(
                rotate=(-config.rotate_degrees, config.rotate_degrees),
                border_mode=cv2.BORDER_CONSTANT,
                fill=IMAGENET_MEAN_BGR,
                p=config.rotate_p,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=config.brightness, contrast_limit=config.contrast, p=config.color_p
            ),
            A.GaussianBlur(blur_limit=0, sigma_limit=config.blur_sigma, p=config.blur_p),
            A.GaussNoise(std_range=config.noise_std, p=config.noise_p),
        ]
    )
    seed_pipeline(pipeline, seed)
    return pipeline


ImageTransform = Callable[[np.ndarray], np.ndarray]


class EvalTransform:
    """Solo el preprocesamiento compartido (valid, test, independiente, app)."""

    def __init__(self, image_size: int) -> None:
        self.image_size = int(image_size)

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        return preprocess_image(image_bgr, self.image_size)


class TrainTransform:
    """Augmentation sobre la imagen BGR y despues el preprocesamiento compartido.

    Es una clase (no una funcion anidada) para que el DataLoader pueda copiarla
    a sus workers con pickle; en Windows los workers se crean con spawn.

    Con DataLoader(num_workers > 0) cada worker recibe una copia con la misma
    semilla: sin re-sembrar, todos los workers (y todas las epocas, porque los
    workers se recrean) repetirian la misma secuencia de augmentations. Dentro
    de un worker se re-siembra con la semilla que PyTorch le asigna, distinta
    por worker y por epoca y derivada de torch.manual_seed (reproducible).
    """

    def __init__(self, image_size: int, config: AugmentationConfig = AugmentationConfig(), seed: int | None = None) -> None:
        self.image_size = int(image_size)
        self.augmentation = build_train_augmentation(config, seed)
        self._worker_seed: int | None = None

    def _reseed_if_in_new_worker(self) -> None:
        from torch.utils.data import get_worker_info  # import diferido: el modulo no requiere torch

        info = get_worker_info()
        if info is not None and info.seed != self._worker_seed:
            self._worker_seed = info.seed
            seed_pipeline(self.augmentation, int(info.seed % 2**32))

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        self._reseed_if_in_new_worker()
        augmented = self.augmentation(image=image_bgr)["image"]
        return preprocess_image(np.ascontiguousarray(augmented), self.image_size)


def make_transform(
    train: bool,
    image_size: int,
    config: AugmentationConfig = AugmentationConfig(),
    seed: int | None = None,
) -> ImageTransform:
    """Imagen BGR uint8 -> arreglo float32 (3, S, S) listo para el modelo.

    train=True aplica la augmentation y despues el preprocesamiento compartido;
    train=False (valid, test, independiente, app) aplica solo el preprocesamiento.
    """
    return TrainTransform(image_size, config, seed) if train else EvalTransform(image_size)


def augmentation_examples(
    image_bgr: np.ndarray, n: int, config: AugmentationConfig = AugmentationConfig(), seed: int = 0
) -> list[np.ndarray]:
    """`n` versiones aumentadas de una misma imagen (BGR uint8), para la grilla de la notebook."""
    augmentation = build_train_augmentation(config, seed)
    return [augmentation(image=image_bgr)["image"] for _ in range(n)]
