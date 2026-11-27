from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lib.dataset.manifest import (
    MANIFEST_COLUMNS,
    build_manifest_from_dogs_csv,
    class_names,
    load_manifest,
    normalize_breed,
    save_manifest,
    split_images,
    validate_manifest,
)


def _dogs_csv(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    path = tmp_path / "dogs.csv"
    pd.DataFrame(rows, columns=["filepaths", "labels", "data set"]).to_csv(path, index=False)
    return path


ROWS = [
    ("train/Afghan/001.jpg", "Afghan", "train"),
    ("train/Beagle/001.jpg", "Beagle", "train"),
    ("valid/Afghan/01.jpg", "Afghan", "valid"),
    ("valid/Beagle/01.jpg", "Beagle", "valid"),
    ("test/Afghan/1.jpg", "Afghan", "test"),
    ("test/Beagle/1.jpg", "Beagle", "test"),
]


def test_normalize_breed() -> None:
    assert normalize_breed("American  Spaniel") == "American Spaniel"
    assert normalize_breed("  Beagle ") == "Beagle"


def test_build_manifest_columns_and_normalization(tmp_path: Path) -> None:
    rows = ROWS + [("valid/American  Spaniel/01.jpg", "American  Spaniel", "valid")]
    manifest = build_manifest_from_dogs_csv(_dogs_csv(tmp_path, rows))
    assert list(manifest.columns) == MANIFEST_COLUMNS
    # la raza se normaliza, la ruta queda tal cual para poder abrir el archivo
    row = manifest[manifest["path"] == "valid/American  Spaniel/01.jpg"].iloc[0]
    assert row["breed"] == "American Spaniel"


def test_roundtrip_and_class_names(tmp_path: Path) -> None:
    manifest = build_manifest_from_dogs_csv(_dogs_csv(tmp_path, ROWS))
    out = tmp_path / "splits" / "m.csv"
    save_manifest(manifest, out)
    loaded = load_manifest(out)
    pd.testing.assert_frame_equal(manifest, loaded)
    assert class_names(loaded) == ["Afghan", "Beagle"]


def test_check_files(tmp_path: Path) -> None:
    manifest = build_manifest_from_dogs_csv(_dogs_csv(tmp_path, ROWS))
    with pytest.raises(ValueError):
        validate_manifest(manifest, dataset_path=tmp_path, check_files=True)
    for p in manifest["path"]:
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_bytes(b"x")
    validate_manifest(manifest, dataset_path=tmp_path, check_files=True)


def test_invalid_manifests_raise(tmp_path: Path) -> None:
    good = build_manifest_from_dogs_csv(_dogs_csv(tmp_path, ROWS))

    bad_split = good.copy()
    bad_split.loc[0, "split"] = "val"
    with pytest.raises(ValueError):
        validate_manifest(bad_split)

    duplicated = pd.concat([good, good.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        validate_manifest(duplicated)

    unknown_breed = good.copy()
    unknown_breed.loc[unknown_breed["split"] == "test", "breed"] = "Poodle"
    with pytest.raises(ValueError):
        validate_manifest(unknown_breed)

    missing_split = good[good["split"] != "test"]
    with pytest.raises(ValueError):
        validate_manifest(missing_split)

    wrong_columns = good.rename(columns={"breed": "label"})
    with pytest.raises(ValueError):
        validate_manifest(wrong_columns)


def test_missing_dogs_csv_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_manifest_from_dogs_csv(tmp_path / "no_existe.csv")


def _manifest(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def test_split_images_only_requested_split_sorted(tmp_path: Path) -> None:
    manifest = _manifest(
        [
            ("train/Beagle/002.jpg", "Beagle", "train"),
            ("test/Afghan/1.jpg", "Afghan", "test"),
            ("train/Afghan/001.jpg", "Afghan", "train"),
            ("valid/Afghan/01.jpg", "Afghan", "valid"),
            ("train/Beagle/001.jpg", "Beagle", "train"),
        ]
    )
    items = split_images(manifest, tmp_path, "train")
    assert items == [
        (tmp_path / "train/Afghan/001.jpg", "Afghan"),
        (tmp_path / "train/Beagle/001.jpg", "Beagle"),
        (tmp_path / "train/Beagle/002.jpg", "Beagle"),
    ]


def test_split_images_excludes_images_moved_to_test(tmp_path: Path) -> None:
    # En los splits v1 una imagen de la carpeta train/ puede ser de test: no se indexa.
    manifest = _manifest(
        [
            ("train/Afghan/001.jpg", "Afghan", "train"),
            ("train/Afghan/002.jpg", "Afghan", "test"),
        ]
    )
    assert split_images(manifest, tmp_path, "train") == [(tmp_path / "train/Afghan/001.jpg", "Afghan")]


def test_split_images_limit_per_breed(tmp_path: Path) -> None:
    rows = [(f"train/{b}/{i:03d}.jpg", b, "train") for b in ("Afghan", "Beagle") for i in range(5)]
    items = split_images(_manifest(rows), tmp_path, "train", limit=2)
    assert [(p.name, b) for p, b in items] == [
        ("000.jpg", "Afghan"), ("001.jpg", "Afghan"), ("000.jpg", "Beagle"), ("001.jpg", "Beagle"),
    ]


def test_split_images_invalid_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Split invalido"):
        split_images(_manifest(ROWS), tmp_path, "gallery")
