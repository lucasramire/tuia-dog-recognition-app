"""Genera los splits limpios (data/splits/splits_<version>.csv) desde el manifiesto original.

Mide calidad, pHash y embeddings del baseline (cacheados en output/), detecta
duplicados y fugas, descarta segun las reglas de lib/dataset/cleaning.py y
completa valid/test hasta 10 por raza desde train. Es lo mismo que ejecuta
1-pre-processing.ipynb; sirve para regenerar los splits sin abrir la notebook.

Uso (desde la raiz del repo):
    python scripts/build_splits.py [--version v1]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lib.dataset.cleaning import (  # noqa: E402
    CleaningConfig,
    load_or_compute_measurements,
    run_cleaning,
    save_outputs,
)
from lib.dataset.manifest import load_manifest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("build_splits")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "dataset")
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "splits" / "splits_v0.csv")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "output" / "p1")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    manifest = load_manifest(args.source, dataset_path=args.dataset, check_files=True)
    table, embeddings = load_or_compute_measurements(manifest, args.dataset, args.cache_dir, args.image_size)
    result = run_cleaning(manifest, table, embeddings, CleaningConfig())
    save_outputs(result, args.splits_dir, args.version)

    logger.info("Original: %s", result.summary["original"])
    logger.info("Final: %s", result.summary["final"])
    logger.info("Razas de calibracion: %s", result.calibration_breeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
