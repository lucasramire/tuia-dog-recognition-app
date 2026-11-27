"""Construye la base vectorial indexando el dataset con extract_embedding (Etapa 1).

Requiere haber implementado SimilarityService.extract_embedding (o, si
EMBEDDING_MODEL != baseline, ClassifierService.extract_custom_embedding).
Si USE_PGVECTOR=true, la base de datos debe estar corriendo
(`docker compose up postgres -d`).

Uso:
    python scripts/build_index.py [--split train] [--limit 0] [--model baseline]
                                  [--manifest data/splits/splits_v1.csv | --no-manifest]

Las imagenes salen del manifiesto de splits limpios (por defecto `data/splits/splits_v1.csv`,
de 1-pre-processing.ipynb): asi la galeria es la misma que se evaluo en 2-train.ipynb y no
incluye imagenes de valid/test (consultas) ni las descartadas en la limpieza. Si ese archivo
no existe, o con --no-manifest, se recorre la carpeta `DATASET_PATH/<split>/` como antes.

Cada modelo de embeddings tiene su propia base (tabla `embeddings` para el baseline y
`embeddings_<modelo>` para los otros, o un JSON por modelo): `--model` indexa la del modelo
indicado (por defecto, EMBEDDING_MODEL) sin tocar las demas.

Dentro de Docker (con `docker compose up -d` corriendo):
    docker compose exec backend python /scripts/build_index.py [--split train]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if not (SRC / "lib").is_dir():
    # Dentro del contenedor del backend: src/ esta montado en /app (WORKDIR) y scripts/ en /scripts.
    ROOT = SRC = Path.cwd()
sys.path.insert(0, str(SRC))

# Usa la misma configuracion que el backend local (src/.env, con paths relativos a src/).
# Si src/.env no existe, se usan los defaults con paths relativos a la raiz del repo.
CALLER_CWD = Path.cwd()  # para resolver --manifest relativo a donde se invoco el script
os.chdir(SRC if (SRC / ".env").is_file() else ROOT)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_MANIFEST = Path("splits") / "splits_v1.csv"  # relativo a DATA_PATH


def folder_images(root: Path, limit: int) -> list[tuple[Path, str]]:
    """(ruta, raza) recorriendo DATASET_PATH/<split>/<raza>/ (sin manifiesto)."""
    items: list[tuple[Path, str]] = []
    for breed_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(p for p in breed_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        items.extend((p, breed_dir.name) for p in (images[:limit] if limit else images))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", default="train", help="Subcarpeta del dataset a indexar (train/valid/test)."
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Maximo de imagenes por raza (0 = todas)."
    )
    parser.add_argument(
        "--model", choices=["baseline", "resnet18_finetuned", "cnn_custom"], default=None,
        help="Modelo de embeddings cuya base se indexa (por defecto, EMBEDDING_MODEL del .env).",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--manifest", type=Path, default=None,
        help="Manifiesto de splits (CSV path,breed,split). Por defecto DATA_PATH/splits/splits_v1.csv.",
    )
    source.add_argument(
        "--no-manifest", action="store_true",
        help="Recorrer la carpeta DATASET_PATH/<split>/ en lugar del manifiesto.",
    )
    args = parser.parse_args()

    from lib.bootstrap import build_classifier, build_similarity, build_store, store_size
    from lib.config import settings
    from lib.dataset.manifest import load_manifest, split_images

    if args.no_manifest:
        manifest_path = None
    elif args.manifest is not None:
        manifest_path = CALLER_CWD / args.manifest  # si es absoluta, queda igual
        if not manifest_path.is_file():
            sys.exit(f"No existe el manifiesto {manifest_path}")
    else:
        manifest_path = settings.data_path / DEFAULT_MANIFEST
        if not manifest_path.is_file():
            print(f"Aviso: no existe {manifest_path}; se recorre la carpeta del split.")
            manifest_path = None

    if manifest_path is not None:
        try:
            manifest = load_manifest(manifest_path)
            items = split_images(manifest, settings.dataset_path, args.split, args.limit)
        except ValueError as exc:
            sys.exit(f"Manifiesto invalido: {exc}")
        missing = [path for path, _ in items if not path.is_file()]
        if missing:
            sys.exit(
                f"{len(missing)} imagenes del manifiesto no estan en {settings.dataset_path}; "
                f"primera: {missing[0]}. Descarga el dataset con scripts/download_dataset.py"
            )
        print(f"Imagenes de '{args.split}' segun {manifest_path}: {len(items)}")
    else:
        root = settings.dataset_path / args.split
        if not root.is_dir():
            sys.exit(f"No existe {root}. Descarga el dataset con scripts/download_dataset.py")
        items = folder_images(root, args.limit)
        print(f"Imagenes de la carpeta {root}: {len(items)}")

    model = args.model or settings.embedding_model
    store = build_store(settings, model)
    similarity = build_similarity(settings, store)
    already = store_size(store)
    if already:
        print(
            f"Aviso: la base de '{model}' ya tiene {already} embeddings; se agregan los nuevos "
            "(si es un reindexado, vaciarla antes para no duplicar vecinos)."
        )

    if model == "baseline":
        extractor = similarity.extract_embedding
    else:
        classifier = build_classifier(settings)
        classifier.set_active_model(model)
        extractor = classifier.extract_custom_embedding
    similarity.extract_embedding = extractor  # type: ignore[method-assign]

    counts: dict[str, int] = {}
    for image_path, breed in items:
        similarity.index_image(str(image_path), breed, {"split": args.split, "model": model})
        counts[breed] = counts.get(breed, 0) + 1
    for breed, count in counts.items():
        print(f"{breed}: {count} imagenes indexadas")

    print(f"Total: {sum(counts.values())} embeddings almacenados (modelo: {model})")


if __name__ == "__main__":
    main()
