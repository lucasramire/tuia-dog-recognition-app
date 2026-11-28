"""Tests del registro del conjunto independiente y de los filtros de Wikimedia (sin red)."""

from __future__ import annotations

import pytest

from lib.dataset.independent import COLUMNS, load_registry, next_numbers, slug
from lib.dataset.wikimedia import Candidate, clean_html, is_accepted_license, parse_candidate, pick_diverse


@pytest.mark.parametrize(
    "name, ok",
    [
        ("CC BY-SA 4.0", True), ("CC BY 2.0", True), ("CC0", True), ("Public domain", True),
        ("CC BY-NC 2.0", False), ("CC BY-NC-SA 3.0", False), ("CC BY-ND 4.0", False),
        ("GFDL", False), ("", False), ("Fair use", False),
    ],
)
def test_is_accepted_license(name, ok):
    assert is_accepted_license(name) is ok


def test_clean_html():
    assert clean_html('<a href="//x">Juan&nbsp;P&eacute;rez</a>') == "Juan Pérez"


def _page(mime="image/jpeg", w=800, h=600, lic="CC BY-SA 4.0"):
    return {
        "title": "File:Perro.jpg",
        "imageinfo": [{
            "mime": mime, "width": w, "height": h, "descriptionurl": "https://commons.wikimedia.org/wiki/File:Perro.jpg",
            "thumburl": "https://upload.wikimedia.org/thumb/Perro.jpg",
            "extmetadata": {"LicenseShortName": {"value": lic}, "Artist": {"value": "<b>Ana</b>"},
                            "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0"}},
        }],
    }


def test_parse_candidate_filters():
    c = parse_candidate("Pug", _page(), min_side=224)
    assert c is not None and c.author == "Ana" and c.license == "CC BY-SA 4.0"
    assert parse_candidate("Pug", _page(mime="image/svg+xml"), 224) is None
    assert parse_candidate("Pug", _page(w=200), 224) is None
    assert parse_candidate("Pug", _page(lic="CC BY-NC 2.0"), 224) is None
    assert parse_candidate("Pug", {"title": "File:x.jpg"}, 224) is None


def _cand(i: int, author: str) -> Candidate:
    return Candidate("Pug", f"File:{i}.jpg", f"u{i}", f"t{i}", author, "CC0", "", 800, 600)


def test_pick_diverse_limits_per_author_and_is_reproducible():
    pool = [_cand(i, "a") for i in range(5)] + [_cand(10 + i, f"b{i}") for i in range(5)]
    chosen = pick_diverse(pool, n=6, max_per_author=2, seed=1)
    assert len(chosen) == 6
    assert sum(c.author == "a" for c in chosen) <= 2
    assert [c.title for c in chosen] == [c.title for c in pick_diverse(pool, 6, 2, seed=1)]


def test_slug():
    assert slug("Boston Terrier") == "boston_terrier"
    assert slug("Shih-Tzu") == "shih_tzu"


def test_next_numbers_respects_ranges_and_existing(tmp_path):
    (tmp_path / "pug_01.jpg").write_bytes(b"x")
    (tmp_path / "pug_03.jpg").write_bytes(b"x")
    assert next_numbers(tmp_path, "pug", "lucas", 3) == [2, 4, 5]
    assert next_numbers(tmp_path, "pug", "valentin", 2) == [51, 52]
    assert next_numbers(tmp_path / "no_existe", "pug", "lucas", 1) == [1]
    with pytest.raises(ValueError):
        next_numbers(tmp_path, "pug", "otro", 1)


def test_registry_template_has_expected_columns():
    from pathlib import Path

    template = Path(__file__).resolve().parents[1] / "data" / "splits" / "independent_v1.csv"
    assert list(load_registry(template).columns) == COLUMNS


# --- validacion, resumen y fuga ---------------------------------------------------

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib.dataset.independent import (  # noqa: E402
    dataset_leaks,
    internal_near_duplicates,
    summarize_registry,
    validate_registry,
)
from lib.dataset.quality import DuplicateCriterion  # noqa: E402


def _row(path, breed="Pug", scene="single", n_dogs="1", source="wikimedia", url="https://x/1", license="CC0", **kw):
    row = dict.fromkeys(COLUMNS, "")
    row.update(path=path, breed=breed, scene=scene, n_dogs=n_dogs, source=source, url=url, author="Ana",
               license=license, collector="lucas", people="none")
    row.update(kw)
    return row


def _registry(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def test_validate_registry_ok_and_problems(tmp_path):
    import cv2

    (tmp_path / "independent" / "Pug").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "independent/Pug/pug_01.jpg"), np.zeros((300, 300, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "independent/Pug/pug_02.jpg"), np.zeros((100, 300, 3), np.uint8))
    ok = _registry([_row("independent/Pug/pug_01.jpg")])
    assert validate_registry(ok, tmp_path, ["Pug"]).empty

    bad = _registry([
        _row("independent/Pug/pug_02.jpg", url="https://x/2"),  # lado menor 100
        _row("independent/Pug/pug_03.jpg", url="https://x/3", license="CC BY-NC 2.0"),  # no existe, NC
        _row("independent/_multi/m_01.jpg", breed="multiple", scene="multiple", n_dogs="1", url="https://x/4"),
        _row("independent/Pug/pug_04.jpg", source="own", url="", license="CC0"),  # own sin license own
        _row("independent/Pug/pug_05.jpg", url="https://x/1", people="si"),
        _row("independent/Pug/pug_06.jpg", url="https://x/1"),  # url repetida con la anterior
    ])
    problems = validate_registry(bad, tmp_path, ["Pug"])
    text = " | ".join(problems["problem"])
    for expected in ["lado menor 100", "licencia no aceptada", "no existe el archivo", "escena multiple",
                     "license = own", "people invalido", "misma url"]:
        assert expected in text, expected


def test_summarize_registry_counts():
    table = _registry([
        _row("a", url="1"), _row("b", url="2"), _row("c", breed="Beagle", url="3"),
        _row("d", breed="none", scene="no_dog", n_dogs="0", url="4"),
    ])
    s = summarize_registry(table)
    assert s["por_raza"].loc["Pug", "imagenes"] == 2
    assert s["por_escena"].loc["no_dog", "imagenes"] == 1
    assert s["por_escena"].loc["multiple", "imagenes"] == 0


def test_dataset_leaks_uses_criterion():
    criterion = DuplicateCriterion(max_phash_distance=8, min_cosine_with_phash=0.90, min_cosine_alone=0.97)
    ds_emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
    ds_hash = np.array([0, 0xFFFF, 0xFFFFFFFF], dtype=np.uint64)
    ind_emb = np.array([[0.93, 0.37], [0.7, 0.71], [0.0, 1.0]], dtype=np.float32)
    ind_hash = np.array([0b111, 0xFFFFFFFFFFFF, 0xFFFFFFFFFFFF], dtype=np.uint64)
    out = dataset_leaks(ind_hash, ind_emb, ds_hash, ds_emb, ["a", "b", "c"], criterion)
    # 0: pHash a distancia 3 de "a" y coseno ~0,93 -> fuga por pHash + coseno.
    # 1: coseno con "c" ~0,98 -> fuga por coseno solo.
    # 2: igual a "b" en embedding (coseno 1) -> fuga.
    assert out["leak"].tolist() == [True, True, True]
    assert out.loc[0, "leak_with"] == "a"
    ind_far = np.array([[0.9, -0.44]], dtype=np.float32)
    far = dataset_leaks(np.array([0xFFFFFFFFFFFFFFFF], dtype=np.uint64), ind_far, ds_hash, ds_emb, ["a", "b", "c"], criterion)
    assert not far.loc[0, "leak"] and far.loc[0, "nearest"] == "a"


def test_internal_near_duplicates():
    emb = np.array([[1, 0], [0.99, 0.14], [0, 1]], dtype=np.float32)
    pairs = internal_near_duplicates(emb, 0.95)
    assert pairs[["i", "j"]].values.tolist() == [[0, 1]]
