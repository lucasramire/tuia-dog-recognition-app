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
