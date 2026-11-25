from __future__ import annotations

import pandas as pd
import pytest

from lib.dataset.splits import cross_split_pairs, images_with_neighbors, rebalance_eval_splits, to_manifest


def _drops() -> pd.DataFrame:
    rows = []
    for breed in ("A", "B"):
        rows += [(f"train/{breed}/{k}.jpg", breed, "train", "") for k in range(6)]
        rows += [(f"valid/{breed}/0.jpg", breed, "valid", "")]
        rows += [(f"test/{breed}/0.jpg", breed, "test", "")]
    rows += [("test/A/leak.jpg", "A", "test", "fuga entre splits")]
    return pd.DataFrame(rows, columns=["path", "breed", "split", "drop_reason"])


def _no_pairs() -> pd.DataFrame:
    return pd.DataFrame({"i": pd.Series(dtype=int), "j": pd.Series(dtype=int), "cosine": pd.Series(dtype=float)})


def test_rebalance_fills_each_breed_and_drops_discarded() -> None:
    kept = rebalance_eval_splits(_drops(), _no_pairs(), per_class=2, seed=0, isolation_cosine=0.95)
    counts = kept.groupby(["split", "breed"]).size()
    assert counts[("valid", "A")] == counts[("valid", "B")] == 2
    assert counts[("test", "A")] == counts[("test", "B")] == 2
    assert counts[("train", "A")] == counts[("train", "B")] == 4
    assert "test/A/leak.jpg" not in set(kept["path"])
    assert int(kept["moved_from_train"].sum()) == 4


def test_rebalance_is_reproducible_with_seed() -> None:
    a = rebalance_eval_splits(_drops(), _no_pairs(), 2, seed=7, isolation_cosine=0.95)
    b = rebalance_eval_splits(_drops(), _no_pairs(), 2, seed=7, isolation_cosine=0.95)
    pd.testing.assert_frame_equal(a, b)


def test_rebalance_only_moves_isolated_images() -> None:
    drops = _drops()
    # train/A/0..3 tienen vecinos parecidos (coseno >= 0.95); solo 4 y 5 estan aisladas.
    pairs = pd.DataFrame({"i": [0, 2, 1], "j": [1, 3, 3], "cosine": [0.99, 0.96, 0.90]})
    kept = rebalance_eval_splits(drops, pairs, per_class=2, seed=0, isolation_cosine=0.95)
    moved_a = sorted(kept[(kept["breed"] == "A") & kept["moved_from_train"]]["path"])
    assert moved_a == ["train/A/4.jpg", "train/A/5.jpg"]


def test_near_copies_in_eval_move_to_train_and_get_replaced() -> None:
    drops = _drops()
    valid_a = int(drops.index[drops["path"] == "valid/A/0.jpg"][0])
    test_a = int(drops.index[drops["path"] == "test/A/0.jpg"][0])
    # valid/A/0 se parece a train/A/0; test/A/0 se parece a valid/A/0 (cadena: al mover valid,
    # el par valid-test pasa a cruzar train-test y test tambien tiene que moverse).
    pairs = pd.DataFrame({"i": [0, valid_a], "j": [valid_a, test_a], "cosine": [0.96, 0.955]})
    kept = rebalance_eval_splits(drops, pairs, per_class=1, seed=0, isolation_cosine=0.95)
    assert kept.loc[valid_a, "split"] == "train" and kept.loc[test_a, "split"] == "train"
    assert bool(kept.loc[valid_a, "moved_to_train"]) and bool(kept.loc[test_a, "moved_to_train"])
    counts = kept.groupby(["split", "breed"]).size()
    assert counts[("valid", "A")] == counts[("test", "A")] == 1
    close = pairs[pairs["cosine"] >= 0.95]
    assert cross_split_pairs(kept.index, kept["split"], close).empty


def test_rebalance_fails_clearly_when_pool_is_short() -> None:
    pairs = pd.DataFrame({"i": [0, 2, 3], "j": [1, 4, 4], "cosine": [0.99, 0.96, 0.95]})
    with pytest.raises(ValueError, match="aisladas"):
        rebalance_eval_splits(_drops(), pairs, per_class=3, seed=0, isolation_cosine=0.95)


def test_images_with_neighbors() -> None:
    pairs = pd.DataFrame({"i": [0, 1], "j": [2, 3], "cosine": [0.96, 0.90]})
    assert images_with_neighbors(5, pairs, 0.95).tolist() == [True, False, True, False, False]


def test_to_manifest_rejects_leaks_and_validates() -> None:
    kept = rebalance_eval_splits(_drops(), _no_pairs(), 1, seed=0, isolation_cosine=0.95)
    manifest = to_manifest(kept, pd.DataFrame({"i": [0], "j": [1]}))  # dos de train: no es fuga
    assert list(manifest.columns) == ["path", "breed", "split"]
    valid_idx = kept.index[kept["split"] == "valid"][0]
    with pytest.raises(ValueError, match="entre splits"):
        to_manifest(kept, pd.DataFrame({"i": [0], "j": [valid_idx]}))


def test_cross_split_pairs_ignores_dropped_images() -> None:
    kept = rebalance_eval_splits(_drops(), _no_pairs(), 1, seed=0, isolation_cosine=0.95)
    dropped_idx = 16  # test/A/leak.jpg, no esta en kept
    assert cross_split_pairs(kept.index, kept["split"], pd.DataFrame({"i": [0], "j": [dropped_idx]})).empty
