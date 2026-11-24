from __future__ import annotations

import numpy as np
import pytest

from lib.vision.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess_batch,
    preprocess_image,
    preprocess_to_tensor,
)

SIZE = 64


def _solid(height: int, width: int, bgr: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = bgr
    return image


def test_shape_and_dtype() -> None:
    out = preprocess_image(_solid(100, 100, (10, 20, 30)), SIZE)
    assert out.shape == (3, SIZE, SIZE)
    assert out.dtype == np.float32


def test_bgr_is_converted_to_rgb() -> None:
    # Pixel azul puro en BGR: el canal R de salida debe ser el de menor valor.
    out = preprocess_image(_solid(SIZE, SIZE, (255, 0, 0)), SIZE)
    center = out[:, SIZE // 2, SIZE // 2]
    expected = (np.array([0.0, 0.0, 1.0], dtype=np.float32) - IMAGENET_MEAN) / IMAGENET_STD
    np.testing.assert_allclose(center, expected, atol=1e-5)


def test_imagenet_normalization() -> None:
    out = preprocess_image(_solid(SIZE, SIZE, (128, 128, 128)), SIZE)
    expected = (np.float32(128 / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    np.testing.assert_allclose(out[:, SIZE // 2, SIZE // 2], expected, atol=1e-5)


def test_wide_image_is_padded_with_zeros_not_stretched() -> None:
    # 20 x 100 (alto x ancho): se escala a 13 x 64 y se centra con relleno 0.
    out = preprocess_image(_solid(20, 100, (200, 100, 50)), SIZE)
    assert np.all(out[:, 0, :] == 0.0)
    assert np.all(out[:, -1, :] == 0.0)
    assert not np.all(out[:, SIZE // 2, :] == 0.0)
    rows_with_content = np.where(np.any(out != 0.0, axis=(0, 2)))[0]
    assert rows_with_content.max() - rows_with_content.min() + 1 == 13


def test_tall_image_is_padded_on_the_sides() -> None:
    out = preprocess_image(_solid(100, 20, (200, 100, 50)), SIZE)
    assert np.all(out[:, :, 0] == 0.0)
    assert np.all(out[:, :, -1] == 0.0)
    assert not np.all(out[:, :, SIZE // 2] == 0.0)


def test_input_is_not_modified() -> None:
    image = np.random.default_rng(0).integers(0, 256, (40, 70, 3), dtype=np.uint8)
    copy = image.copy()
    preprocess_image(image, SIZE)
    np.testing.assert_array_equal(image, copy)


def test_deterministic() -> None:
    image = np.random.default_rng(1).integers(0, 256, (50, 30, 3), dtype=np.uint8)
    np.testing.assert_array_equal(preprocess_image(image, SIZE), preprocess_image(image, SIZE))


def test_batch_and_tensor() -> None:
    images = [_solid(30, 40, (1, 2, 3)), _solid(80, 20, (4, 5, 6))]
    batch = preprocess_batch(images, SIZE)
    assert batch.shape == (2, 3, SIZE, SIZE)
    tensor = preprocess_to_tensor(images[0], SIZE)
    assert tuple(tensor.shape) == (3, SIZE, SIZE)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.zeros((0, 10, 3), dtype=np.uint8),
        "no es una imagen",
    ],
)
def test_invalid_input_raises(bad) -> None:
    with pytest.raises(ValueError):
        preprocess_image(bad, SIZE)


def test_invalid_size_raises() -> None:
    with pytest.raises(ValueError):
        preprocess_image(_solid(10, 10, (0, 0, 0)), 0)


def test_empty_batch_raises() -> None:
    with pytest.raises(ValueError):
        preprocess_batch([], SIZE)
