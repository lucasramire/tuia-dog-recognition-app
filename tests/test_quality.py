from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from lib.dataset.quality import (
    DuplicateCriterion,
    QualityThresholds,
    build_drop_table,
    confirm_duplicates,
    compute_quality_table,
    duplicate_drops,
    duplicate_groups,
    embedding_pairs,
    exposure_stats,
    fill_pair_scores,
    laplacian_variance,
    merge_pairs,
    phash_int,
    phash_pairs,
)

T = QualityThresholds(min_side=64, min_blur_var=50.0, min_brightness=20.0, max_brightness=235.0)


def _textured(seed: int, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)


# --- metricas por imagen --------------------------------------------------------


def test_blur_lowers_laplacian_variance() -> None:
    sharp = _textured(0)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 5)
    assert laplacian_variance(blurred) < laplacian_variance(sharp) / 5
    assert laplacian_variance(np.full((64, 64, 3), 128, np.uint8)) == 0.0


def test_exposure_stats() -> None:
    dark = exposure_stats(np.zeros((32, 32, 3), np.uint8))
    assert dark["brightness"] == 0.0 and dark["dark_fraction"] == 1.0 and dark["bright_fraction"] == 0.0
    white = exposure_stats(np.full((32, 32, 3), 255, np.uint8))
    assert white["brightness"] == 255.0 and white["bright_fraction"] == 1.0


def test_phash_robust_to_resize_and_jpeg_but_not_to_other_image() -> None:
    image = _textured(1, 256)
    resized = cv2.resize(image, (180, 180), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 60])
    recompressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    h = phash_int(image)
    near = [phash_int(resized), phash_int(recompressed)]
    far = phash_int(_textured(2, 256))
    assert all(bin(h ^ x).count("1") <= 6 for x in near)
    assert bin(h ^ far).count("1") > 16


# --- tabla de calidad -----------------------------------------------------------


def test_compute_quality_table_reads_and_flags_corrupt(tmp_path: Path) -> None:
    (tmp_path / "train/A").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "train/A/ok.jpg"), _textured(3))
    (tmp_path / "train/A/broken.jpg").write_bytes(b"no es un jpg")
    manifest = pd.DataFrame(
        {"path": ["train/A/ok.jpg", "train/A/broken.jpg"], "breed": ["A", "A"], "split": ["train", "train"]}
    )
    table = compute_quality_table(manifest, tmp_path)
    assert table["readable"].tolist() == [True, False]
    assert table.loc[0, "height"] == 128 and table.loc[0, "blur_var"] > 0
    assert table.loc[1, "error"]


# --- pares de duplicados --------------------------------------------------------


def test_phash_pairs_hamming() -> None:
    hashes = np.array([0b0000, 0b0001, 0b0111, 0xFFFF], dtype=np.uint64)
    pairs = phash_pairs(hashes, max_distance=1, chunk=2)
    assert list(zip(pairs["i"], pairs["j"], pairs["phash_distance"])) == [(0, 1, 1)]
    assert len(phash_pairs(hashes, max_distance=3)) == 3  # (0,1), (0,2), (1,2)


def test_embedding_pairs_cosine_and_zero_rows() -> None:
    e = np.array([[1, 0], [0.99, 0.141], [0, 1], [0, 0]], dtype=np.float32)
    pairs = embedding_pairs(e, min_similarity=0.95, chunk=3)
    assert list(zip(pairs["i"], pairs["j"])) == [(0, 1)]
    assert pairs["cosine"].iloc[0] == pytest.approx(0.99, abs=1e-3)


def test_merge_and_fill_pair_scores() -> None:
    hashes = np.array([0, 1, 0xFF], dtype=np.uint64)
    emb = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    merged = merge_pairs(phash_pairs(hashes, 1), embedding_pairs(emb, 0.9))
    filled = fill_pair_scores(pd.DataFrame({"i": [0, 0], "j": [1, 2]}), hashes, emb)
    assert list(zip(merged["i"], merged["j"])) == [(0, 1)]
    assert filled["phash_distance"].tolist() == [1, 8]
    assert filled["cosine"].tolist() == pytest.approx([1.0, 0.0])


def test_duplicate_groups_are_transitive() -> None:
    groups = duplicate_groups(5, pd.DataFrame({"i": [0, 1], "j": [1, 3]}))
    assert groups[0] == groups[1] == groups[3]
    assert len({groups[0], groups[2], groups[4]}) == 3


# --- decision de descarte -------------------------------------------------------


def _table(rows: list[tuple[str, str, bool, int, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["path", "split", "readable", "height", "blur_var", "brightness"]
    ).assign(breed="A", width=lambda d: d["height"])


def test_duplicate_keeps_train_and_drops_leak_from_test() -> None:
    table = _table(
        [
            ("test/A/1.jpg", "test", True, 224, 100, 120),
            ("train/A/1.jpg", "train", True, 224, 100, 120),
            ("valid/A/1.jpg", "valid", True, 224, 100, 120),
            ("train/A/2.jpg", "train", True, 224, 100, 120),
        ]
    )
    reasons = duplicate_drops(table, pd.DataFrame({"i": [0, 1, 3], "j": [1, 2, 1]}))
    assert reasons.tolist() == ["fuga entre splits", "", "fuga entre splits", "duplicado dentro del split"]


def test_duplicate_between_valid_and_test_keeps_valid() -> None:
    table = _table([("test/A/1.jpg", "test", True, 224, 100, 120), ("valid/A/1.jpg", "valid", True, 224, 100, 120)])
    assert duplicate_drops(table, pd.DataFrame({"i": [0], "j": [1]})).tolist() == ["fuga entre splits", ""]


def test_confirm_duplicates_criterion() -> None:
    pairs = pd.DataFrame(
        {
            "i": [0, 1, 2, 3],
            "j": [9, 9, 9, 9],
            "phash_distance": [2, 2, 30, 30],
            "cosine": [0.95, 0.60, 0.98, 0.94],
        }
    )
    c = DuplicateCriterion(max_phash_distance=8, min_cosine_with_phash=0.9, min_cosine_alone=0.97)
    # pHash cercano confirmado; pHash cercano sin confirmar no; coseno alto solo si; coseno medio no.
    assert confirm_duplicates(pairs, c)["i"].tolist() == [0, 2]


def test_same_photo_with_two_breeds_drops_all_copies() -> None:
    table = _table(
        [
            ("train/A/1.jpg", "train", True, 224, 100, 120),
            ("train/B/1.jpg", "train", True, 224, 100, 120),
            ("test/B/2.jpg", "test", True, 224, 100, 120),
        ]
    )
    table.loc[1:, "breed"] = "B"
    reasons = duplicate_drops(table, pd.DataFrame({"i": [0, 1], "j": [1, 2]}))
    assert reasons.tolist() == ["etiqueta contradictoria"] * 3


def test_quality_rules_only_drop_in_train() -> None:
    table = _table(
        [
            ("train/A/blur.jpg", "train", True, 224, 10, 120),
            ("test/A/blur.jpg", "test", True, 224, 10, 120),
            ("train/A/dark.jpg", "train", True, 224, 100, 5),
            ("train/A/small.jpg", "train", True, 32, 100, 120),
            ("valid/A/broken.jpg", "valid", False, np.nan, np.nan, np.nan),
            ("train/A/ok.jpg", "train", True, 224, 100, 120),
        ]
    )
    drops = build_drop_table(table, pd.DataFrame({"i": [], "j": []}, dtype=int), T)
    assert drops["drop_reason"].tolist() == [
        "borrosa",
        "",
        "exposicion extrema",
        "resolucion baja",
        "corrupta",
        "",
    ]
    assert bool(drops.loc[1, "blurry"])  # se marca aunque no se descarte (test)
