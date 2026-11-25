from __future__ import annotations

import pickle

import albumentations as A
import cv2
import numpy as np
import pytest

from lib.dataset.augmentation import (
    IMAGENET_MEAN_BGR,
    AugmentationConfig,
    EvalTransform,
    TrainTransform,
    augmentation_examples,
    build_train_augmentation,
    make_transform,
    seed_pipeline,
)
from lib.vision.preprocessing import preprocess_image

SIZE = 64
ONLY_NOTHING = AugmentationConfig(crop_p=0, hflip_p=0, rotate_p=0, color_p=0, blur_p=0, noise_p=0)


def _image(seed: int = 0, height: int = 90, width: int = 120) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def test_eval_transform_is_only_the_shared_preprocessing() -> None:
    image = _image()
    np.testing.assert_array_equal(make_transform(False, SIZE)(image), preprocess_image(image, SIZE))
    assert isinstance(make_transform(False, SIZE), EvalTransform)


def test_train_transform_output_and_input_untouched() -> None:
    image = _image()
    copy = image.copy()
    out = make_transform(True, SIZE, seed=0)(image)
    assert out.shape == (3, SIZE, SIZE) and out.dtype == np.float32
    np.testing.assert_array_equal(image, copy)


def test_train_transform_without_augmentations_equals_eval() -> None:
    image = _image(1)
    np.testing.assert_array_equal(TrainTransform(SIZE, ONLY_NOTHING, seed=0)(image), preprocess_image(image, SIZE))


def test_seed_reproducible_and_samples_vary() -> None:
    image = _image(2)
    t1, t2 = TrainTransform(SIZE, seed=5), TrainTransform(SIZE, seed=5)
    seq1 = [t1(image) for _ in range(6)]
    seq2 = [t2(image) for _ in range(6)]
    for x, y in zip(seq1, seq2):
        np.testing.assert_array_equal(x, y)
    assert len({x.tobytes() for x in seq1}) >= 5
    other = TrainTransform(SIZE, seed=6)
    assert any(not np.array_equal(x, other(image)) for x in seq1)


def test_seeded_transforms_decide_independently() -> None:
    # Con Compose(seed=...) de Albumentations todas las transformaciones comparten semilla y
    # dos con igual p se aplican siempre juntas. seed_pipeline les da semillas distintas.
    pipeline = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)])
    seed_pipeline(pipeline, 5)
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    only_one = 0
    for _ in range(200):
        out = pipeline(image=image)["image"]
        h = np.array_equal(out, image[:, ::-1])
        v = np.array_equal(out, image[::-1])
        only_one += int(h or v)
    assert only_one > 50  # independientes: ~100 de 200


def test_no_vertical_flip_or_90_degree_rotation() -> None:
    names = {type(t).__name__ for t in build_train_augmentation().transforms}
    assert not names & {"VerticalFlip", "RandomRotate90", "Transpose", "D4", "HueSaturationValue", "ColorJitter"}
    assert "HorizontalFlip" in names


def test_rotation_border_is_imagenet_mean_so_it_normalizes_to_zero() -> None:
    affine = next(t for t in build_train_augmentation().transforms if isinstance(t, A.Affine))
    assert affine.border_mode == cv2.BORDER_CONSTANT
    assert tuple(affine.fill) == IMAGENET_MEAN_BGR
    # la media de ImageNet en BGR uint8, normalizada, queda casi en 0 (igual que el padding)
    filled = np.full((SIZE, SIZE, 3), IMAGENET_MEAN_BGR, np.uint8)
    assert np.abs(preprocess_image(filled, SIZE)).max() < 0.01
    # y aparece en las esquinas de una imagen rotada lo suficiente
    config = AugmentationConfig(crop_p=0, hflip_p=0, rotate_p=1, rotate_degrees=30, color_p=0, blur_p=0, noise_p=0)
    white = np.full((SIZE, SIZE, 3), 255, np.uint8)
    corners = [tuple(img[0, 0]) for img in augmentation_examples(white, 10, config, seed=0)]
    assert IMAGENET_MEAN_BGR in corners


def test_crop_from_borders_keeps_most_of_the_image() -> None:
    config = AugmentationConfig(crop_p=1, crop_border=0.15, hflip_p=0, rotate_p=0, color_p=0, blur_p=0, noise_p=0)
    image = _image(3, 200, 300)
    for cropped in augmentation_examples(image, 10, config, seed=1):
        h, w = cropped.shape[:2]
        assert 0.7 * 200 - 1 <= h <= 200 and 0.7 * 300 - 1 <= w <= 300


def test_examples_count_and_type() -> None:
    examples = augmentation_examples(_image(4), 5, seed=0)
    assert len(examples) == 5 and all(e.dtype == np.uint8 and e.ndim == 3 for e in examples)


@pytest.mark.parametrize("train", [True, False])
def test_transforms_are_picklable_for_dataloader_workers(train: bool) -> None:
    transform = make_transform(train, SIZE, seed=0)
    restored = pickle.loads(pickle.dumps(transform))
    image = _image(5)
    np.testing.assert_array_equal(restored(image), pickle.loads(pickle.dumps(transform))(image))


def test_pipeline_is_albumentations_compose() -> None:
    assert isinstance(build_train_augmentation(seed=0), A.Compose)
