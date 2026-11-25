"""Proceso completo de limpieza: manifiesto original -> splits limpios (P1.3 + P1.6).

Lo usan `scripts/build_splits.py` y `1-pre-processing.ipynb`, asi el archivo
entregado sale siempre del mismo codigo. Los valores por defecto de
`CleaningConfig` son los elegidos mirando distribuciones y ejemplos (ver la
notebook y la seccion D-15 / D-24 del plan); cambiarlos regenera los splits.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lib.dataset.quality import (
    DuplicateCriterion,
    QualityThresholds,
    build_drop_table,
    compute_baseline_embeddings,
    compute_quality_table,
    confirm_duplicates,
    describe_pairs,
    embedding_pairs,
    fill_pair_scores,
    merge_pairs,
    phash_pairs,
)
from lib.dataset.splits import cross_split_pairs, rebalance_eval_splits, to_manifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleaningConfig:
    # Calidad (solo train). Borrosidad < 30: por debajo hay fotos borrosas reales; entre 30 y 50
    # la mayoria son perros blancos y peludos nitidos (poca textura, Laplaciano bajo).
    # Exposicion: los extremos observados (brillo medio 27,7 a 243) son perros negros o blancos
    # sobre fondos del mismo color, no fotos mal expuestas; los limites no descartan ninguna.
    quality: QualityThresholds = field(
        default_factory=lambda: QualityThresholds(
            min_side=64, min_blur_var=30.0, min_brightness=15.0, max_brightness=250.0
        )
    )
    # Duplicados: pHash <= 8 fue siempre la misma foto; coseno >= 0,97 tambien (recortes, zoom,
    # color); entre 0,95 y 0,97 mezcla copias con fotos distintas de la misma raza.
    duplicates: DuplicateCriterion = field(
        default_factory=lambda: DuplicateCriterion(
            max_phash_distance=8, min_cosine_with_phash=0.90, min_cosine_alone=0.97
        )
    )
    # Candidatos que se miden (mas amplio que el criterio, para poder revisar la frontera).
    candidate_max_phash: int = 12
    candidate_min_cosine: float = 0.93
    # Splits de evaluacion: 10 por raza como el dataset original, completando desde train
    # con imagenes sin vecinos de coseno >= 0,95 (mas estricto que el criterio de duplicado).
    per_class: int = 10
    isolation_cosine: float = 0.95
    # Razas apartadas para calibrar el umbral de "unknown" de la Etapa 1 (D-12).
    n_calibration_breeds: int = 7
    seed: int = 42


@dataclass
class CleaningResult:
    manifest: pd.DataFrame
    drops: pd.DataFrame
    pairs: pd.DataFrame
    confirmed: pd.DataFrame
    calibration_breeds: list[str]
    summary: dict[str, object]


def load_or_compute_measurements(
    manifest: pd.DataFrame, dataset_path: Path, cache_dir: Path, image_size: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Tabla de calidad y embeddings del baseline, cacheados en `cache_dir` (fuera de git)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    table_path = cache_dir / "quality.csv"
    emb_path = cache_dir / "embeddings_baseline.npz"

    if table_path.exists():
        # phash como texto: un hash hexadecimal sin letras se leeria como numero y perderia ceros.
        table = pd.read_csv(table_path, dtype={"phash": str, "error": str})
    else:
        table = compute_quality_table(manifest, dataset_path)
        table["phash"] = table["phash"].map(lambda v: "" if pd.isna(v) else format(int(v), "016x"))
        table.to_csv(table_path, index=False)
    if emb_path.exists():
        cached = np.load(emb_path, allow_pickle=False)
        embeddings = cached["embeddings"]
        cached_paths = cached["path"].tolist()
    else:
        embeddings = compute_baseline_embeddings([dataset_path / p for p in manifest["path"]], image_size)
        cached_paths = manifest["path"].tolist()
        np.savez_compressed(emb_path, embeddings=embeddings, path=np.array(cached_paths))

    if table["path"].tolist() != manifest["path"].tolist() or cached_paths != manifest["path"].tolist():
        raise ValueError(f"El cache de {cache_dir} no corresponde a este manifiesto: borrarlo y recalcular")
    return table, embeddings


def phash_array(table: pd.DataFrame) -> np.ndarray:
    return np.array([int(h, 16) if isinstance(h, str) and h else 0 for h in table["phash"]], dtype=np.uint64)


def run_cleaning(
    manifest: pd.DataFrame,
    table: pd.DataFrame,
    embeddings: np.ndarray,
    config: CleaningConfig = CleaningConfig(),
) -> CleaningResult:
    hashes = phash_array(table)
    candidates = merge_pairs(
        phash_pairs(hashes, config.candidate_max_phash),
        embedding_pairs(embeddings, config.candidate_min_cosine),
    )
    pairs = describe_pairs(fill_pair_scores(candidates, hashes, embeddings), table)
    confirmed = confirm_duplicates(pairs, config.duplicates)
    drops = build_drop_table(table, confirmed, config.quality)

    kept = rebalance_eval_splits(drops, pairs, config.per_class, config.seed, config.isolation_cosine)
    final = to_manifest(kept, confirmed)

    breeds = sorted(final["breed"].unique())
    rng = np.random.default_rng(config.seed)
    calibration = sorted(rng.choice(breeds, size=config.n_calibration_breeds, replace=False).tolist())

    residual = cross_split_pairs(kept.index, kept["split"], pairs[pairs["cosine"] >= config.isolation_cosine])
    if not residual.empty:
        raise ValueError(f"Quedan {len(residual)} pares con coseno >= {config.isolation_cosine} entre splits")
    summary = {
        "original": manifest["split"].value_counts().to_dict(),
        "final": final["split"].value_counts().to_dict(),
        "dropped_by_reason_and_split": drops[drops["drop_reason"] != ""]
        .groupby(["drop_reason", "split"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index"),
        "moved_from_train": kept[kept["moved_from_train"]]["split"].value_counts().to_dict(),
        "moved_to_train_near_copies": drops.loc[kept.index[kept["moved_to_train"]], "split"]
        .value_counts()
        .to_dict(),
        "confirmed_duplicate_pairs": int(len(confirmed)),
        "cross_split_pairs_above_isolation_cosine": int(len(residual)),
        "calibration_breeds": calibration,
        "config": asdict(config),
    }
    logger.info("Limpieza: %s", {k: summary[k] for k in ("original", "final", "moved_from_train")})
    return CleaningResult(final, drops, pairs, confirmed, calibration, summary)


def save_outputs(result: CleaningResult, splits_dir: Path, version: str) -> dict[str, Path]:
    """Guarda manifiesto, descartes, razas de calibracion y resumen en `splits_dir`."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": splits_dir / f"splits_{version}.csv",
        "dropped": splits_dir / f"dropped_{version}.csv",
        "calibration": splits_dir / f"calibration_breeds_{version}.json",
        "summary": splits_dir / f"cleaning_summary_{version}.json",
    }
    result.manifest.to_csv(paths["manifest"], index=False, encoding="utf-8")
    dropped = result.drops[result.drops["drop_reason"] != ""][["path", "breed", "split", "drop_reason"]]
    dropped.sort_values(["drop_reason", "split", "path"]).to_csv(paths["dropped"], index=False, encoding="utf-8")
    paths["calibration"].write_text(json.dumps(result.calibration_breeds, indent=2), encoding="utf-8")
    paths["summary"].write_text(json.dumps(result.summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for name, path in paths.items():
        logger.info("Guardado %s: %s", name, path)
    return paths
