"""Genera data/splits/splits_v0.csv desde data/dataset/dogs.csv.

Es el manifiesto inicial (splits originales del dataset, sin filtrar). La
notebook 1-pre-processing.ipynb produce despues los splits limpios con el
mismo formato.

Uso (desde la raiz del repo):
    python scripts/build_initial_manifest.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lib.dataset.manifest import (  # noqa: E402
    DEFAULT_MANIFEST_NAME,
    build_manifest_from_dogs_csv,
    class_names,
    save_manifest,
    validate_manifest,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("build_initial_manifest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "splits" / DEFAULT_MANIFEST_NAME)
    parser.add_argument("--skip-file-check", action="store_true", help="No verificar que existan las imagenes")
    args = parser.parse_args()

    manifest = build_manifest_from_dogs_csv(args.dataset / "dogs.csv")
    validate_manifest(manifest, dataset_path=args.dataset, check_files=not args.skip_file_check)
    save_manifest(manifest, args.output)

    counts = manifest["split"].value_counts().to_dict()
    logger.info("Imagenes por split: %s", counts)
    logger.info("Razas: %d", len(class_names(manifest)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
