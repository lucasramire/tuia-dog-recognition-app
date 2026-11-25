from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from lib.dataset.eda import (
    COCO_CAT,
    COCO_DOG,
    COCO_PERSON,
    class_counts,
    load_rgb,
    nearest_neighbor_accuracy,
    nearest_neighbor_confusions,
    nearest_neighbors,
    split_statistics,
    summarize_detections,
    wilson_interval,
)


def _manifest() -> pd.DataFrame:
    rows = [("A", "train")] * 4 + [("B", "train")] * 2 + [("A", "valid"), ("B", "valid"), ("A", "test"), ("B", "test")]
    return pd.DataFrame([(f"{s}/{b}/{i}.jpg", b, s) for i, (b, s) in enumerate(rows)], columns=["path", "breed", "split"])


def test_class_counts_and_statistics() -> None:
    counts = class_counts(_manifest())
    assert list(counts.columns) == ["train", "valid", "test"]
    assert counts.loc["A", "train"] == 4 and counts.loc["B", "train"] == 2
    stats = split_statistics(counts)
    assert stats.loc["train", "sum"] == 6 and stats.loc["train", "max_over_min"] == 2.0
    assert stats.loc["test", "mean"] == 1.0


def test_wilson_interval() -> None:
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.49, abs=0.01) and high == pytest.approx(0.94, abs=0.01)
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-9) and 0.25 < high < 0.35
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_nearest_neighbors_excludes_self() -> None:
    e = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    assert nearest_neighbors(e, chunk=3).tolist() == [1, 0, 3, 2]


def test_nearest_neighbor_confusions_and_accuracy() -> None:
    # A y B mezcladas (vecinos cruzados), C separada.
    e = np.array([[1, 0, 0], [0.99, 0.1, 0], [0.98, 0.0, 0.1], [0, 0, 1], [0, 0.1, 0.99]], dtype=np.float32)
    breeds = np.array(["A", "B", "A", "C", "C"])
    conf = nearest_neighbor_confusions(e, breeds)
    assert len(conf) == 1
    row = conf.iloc[0]
    assert (row["breed_a"], row["breed_b"]) == ("A", "B")
    assert row["confused"] >= 2 and row["images_in_pair"] == 3
    acc = nearest_neighbor_accuracy(e, breeds)
    assert acc["C"] == 1.0 and acc["B"] == 0.0


def test_summarize_detections() -> None:
    classes = np.array([COCO_DOG, COCO_DOG, COCO_PERSON, COCO_CAT])
    confs = np.array([0.9, 0.6, 0.8, 0.5])
    boxes = np.array([[0, 0, 50, 100], [0, 0, 10, 10], [0, 0, 5, 5], [0, 0, 5, 5]], dtype=float)
    row = summarize_detections(classes, confs, boxes, height=100, width=100)
    assert (row["n_dogs"], row["n_persons"], row["n_cats"]) == (2, 1, 1)
    assert row["dog_conf"] == pytest.approx(0.9)
    assert row["dog_area_frac"] == pytest.approx(0.5)
    assert row["dog_box_aspect"] == pytest.approx(0.5)


def test_summarize_detections_without_dogs() -> None:
    row = summarize_detections(np.array([COCO_PERSON]), np.array([0.7]), np.array([[0, 0, 5, 5]]), 10, 10)
    assert row["n_dogs"] == 0 and row["dog_area_frac"] == 0.0 and np.isnan(row["dog_box_aspect"])


def test_load_rgb_pads_to_square(tmp_path: Path) -> None:
    path = tmp_path / "x.png"
    image = np.zeros((20, 40, 3), np.uint8)
    image[:, :, 0] = 255  # azul en BGR
    cv2.imwrite(str(path), image)
    rgb = load_rgb(path, size=40)
    assert rgb.shape == (40, 40, 3)
    assert tuple(rgb[20, 20]) == (0, 0, 255)  # azul en RGB
    assert tuple(rgb[0, 0]) == (255, 255, 255)  # padding blanco
    with pytest.raises(ValueError):
        load_rgb(tmp_path / "no_existe.png")
