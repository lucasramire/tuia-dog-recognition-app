"""Splits limpios a partir del filtrado (P1.6, enunciado 3.7).

Partiendo de la tabla de descartes de `quality.build_drop_table`:
  1. Se quitan las imagenes descartadas (corruptas, duplicadas, fugas, calidad).
  2. Las imagenes de valid/test con una casi-copia en otro split (coseno >=
     `isolation_cosine`, por debajo del criterio de duplicado) pasan a train.
  3. Las fugas se quitaron de valid/test, que quedan desbalanceados. Se completan
     hasta `per_class` imagenes por raza con imagenes de train elegidas al azar
     (semilla fija) entre las que no tienen ninguna imagen parecida en el
     dataset (coseno < `isolation_cosine` con todas las demas). Asi una imagen
     movida a valid/test no deja una casi-copia en train (D-24, opcion elegida:
     completar desde train).
  4. Se verifica que ningun par de duplicados confirmados cruce splits.

El resultado tiene el formato del manifiesto (`lib/dataset/manifest.py`).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from lib.dataset.manifest import MANIFEST_COLUMNS, validate_manifest

logger = logging.getLogger(__name__)

EVAL_SPLITS = ("valid", "test")


def images_with_neighbors(n: int, pairs: pd.DataFrame, min_cosine: float) -> np.ndarray:
    """Mascara booleana: True si la imagen tiene algun par con coseno >= min_cosine."""
    close = pairs[pairs["cosine"] >= min_cosine]
    mask = np.zeros(n, dtype=bool)
    mask[close["i"].to_numpy()] = True
    mask[close["j"].to_numpy()] = True
    return mask


def rebalance_eval_splits(
    drops: pd.DataFrame,
    pairs: pd.DataFrame,
    per_class: int,
    seed: int,
    isolation_cosine: float,
) -> pd.DataFrame:
    """Manifiesto limpio con valid y test completados hasta `per_class` por raza.

    `drops` es la salida de `build_drop_table` (mismo orden de filas que `pairs`
    usa en i, j). `pairs` son los pares candidatos con su coseno, no solo los
    confirmados: sirven para elegir imagenes sin vecinos parecidos.
    """
    kept = drops[drops["drop_reason"] == ""].copy()
    kept["moved_from_train"] = False
    kept["moved_to_train"] = False
    isolated = ~images_with_neighbors(len(drops), pairs, isolation_cosine)
    rng = np.random.default_rng(seed)

    # Casi-copias por debajo del criterio de duplicado (coseno entre isolation_cosine y el
    # criterio): la imagen de valid/test pasa a train, no se pierde. Se repite porque mover
    # una imagen puede dejar a su vecina de valid/test con un par que ahora cruza splits.
    close = pairs[pairs["cosine"] >= isolation_cosine]
    while True:
        crossing = cross_split_pairs(kept.index, kept["split"], close)
        involved = pd.Index(np.concatenate([crossing["i"].to_numpy(), crossing["j"].to_numpy()]))
        to_move = involved.unique()
        to_move = to_move[kept.loc[to_move, "split"].isin(EVAL_SPLITS).to_numpy()]
        if len(to_move) == 0:
            break
        kept.loc[to_move, "split"] = "train"
        kept.loc[to_move, "moved_to_train"] = True

    for split in EVAL_SPLITS:
        counts = kept[kept["split"] == split].groupby("breed").size()
        for breed in sorted(kept["breed"].unique()):
            have = int(counts.get(breed, 0))
            if have > per_class:
                raise ValueError(f"{split}/{breed} tiene {have} imagenes, mas que per_class={per_class}")
            need = per_class - have
            if need == 0:
                continue
            pool = kept.index[
                (kept["split"] == "train") & (kept["breed"] == breed) & isolated[kept.index]
            ].to_numpy()
            if len(pool) < need:
                raise ValueError(
                    f"No hay suficientes imagenes de train aisladas de {breed} para completar "
                    f"{split} ({len(pool)} disponibles, faltan {need})"
                )
            chosen = rng.choice(np.sort(pool), size=need, replace=False)
            kept.loc[chosen, "split"] = split
            kept.loc[chosen, "moved_from_train"] = True
    logger.info(
        "Splits limpios: %s; de valid/test a train: %d; de train a valid/test: %d",
        kept["split"].value_counts().to_dict(),
        int(kept["moved_to_train"].sum()),
        int(kept["moved_from_train"].sum()),
    )
    return kept


def cross_split_pairs(manifest_index: pd.Index, splits: pd.Series, pairs: pd.DataFrame) -> pd.DataFrame:
    """Pares cuyas dos imagenes estan en el manifiesto y en splits distintos."""
    present = pairs[pairs["i"].isin(manifest_index) & pairs["j"].isin(manifest_index)]
    si = splits.loc[present["i"]].to_numpy()
    sj = splits.loc[present["j"]].to_numpy()
    return present[si != sj]


def to_manifest(kept: pd.DataFrame, confirmed_pairs: pd.DataFrame) -> pd.DataFrame:
    """Valida que no haya fugas confirmadas y devuelve el manifiesto (path, breed, split)."""
    leaks = cross_split_pairs(kept.index, kept["split"], confirmed_pairs)
    if not leaks.empty:
        raise ValueError(f"Quedan {len(leaks)} pares de duplicados entre splits; primero: {leaks.iloc[0].to_dict()}")
    manifest = kept[MANIFEST_COLUMNS].sort_values(["split", "breed", "path"]).reset_index(drop=True)
    validate_manifest(manifest)
    return manifest
