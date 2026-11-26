"""Junta candidatas de Wikimedia Commons para el conjunto independiente (P1.4, enunciado 3.4).

Flujo (desde la raiz del repo):

  1. Verificar que existan las categorias de Commons de tus razas:
       python scripts/collect_independent.py check --collector lucas
  2. Bajar candidatas a una carpeta de revision (fuera de git):
       python scripts/collect_independent.py fetch --collector lucas
     Quedan en data/dataset/independent_candidates/<raza>/ junto con candidates.csv
     (pagina, autor y licencia de cada una).
  3. Revisar a mano y BORRAR las que no sirven: raza dudosa, mas de un perro, personas
     reconocibles, dibujos, cachorros muy chicos, fotos casi iguales. Dejar 5 o 6 por raza.
  4. Pasar las que quedaron a data/dataset/independent/<raza>/ y registrarlas en
     data/splits/independent_v1.csv:
       python scripts/collect_independent.py accept --collector lucas

Nada se acepta sin revision: la raza y las personas las confirma quien revisa. Las fotos
con una persona no reconocible se registran con people=none; corregir a mano a
`not_identifiable` en el CSV si corresponde.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from lib.dataset.independent import COLUMNS, INDEPENDENT_DIR, load_registry, next_numbers, slug  # noqa: E402
from lib.dataset.wikimedia import COMMONS_CATEGORIES, CommonsClient, pick_diverse  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("collect_independent")

# Reparto de RECOLECCION_INDEPENDIENTE.md (cada par confundible, a la misma persona).
COLLECTOR_BREEDS = {
    "lucas": [
        "Lhasa", "Shih-Tzu", "Boston Terrier", "Bulldog", "French Bulldog", "Cockapoo", "Labradoodle",
        "American Hairless", "Mex Hairless", "Labrador", "Beagle", "Pug", "Chihuahua",
    ],
    "valentin": [
        "Border Collie", "Collie", "German Sheperd", "Malinois", "Shiba Inu", "Dingo", "Rhodesian",
        "Vizsla", "Golden Retriever", "Siberian Husky", "Poodle", "Yorkie",
    ],
}
CANDIDATE_COLUMNS = ["file", "breed", "url", "author", "license", "license_url", "retrieved", "width", "height"]


def resolve_breeds(args: argparse.Namespace) -> list[str]:
    breeds = args.breeds or COLLECTOR_BREEDS[args.collector]
    unknown = [b for b in breeds if b not in COMMONS_CATEGORIES]
    if unknown:
        raise SystemExit(f"Razas sin categoria de Commons configurada: {unknown}")
    return breeds


def cmd_check(args: argparse.Namespace) -> int:
    breeds = resolve_breeds(args)
    infos = CommonsClient().category_info([COMMONS_CATEGORIES[b] for b in breeds])
    problems = 0
    for breed in breeds:
        category = COMMONS_CATEGORIES[breed]
        info = infos[category]
        if not info["exists"]:
            status = "NO EXISTE"
        elif info["redirect_to"]:
            status = f"REDIRIGE a {info['redirect_to']}"
        elif info["files"] + info["subcats"] == 0:
            status = "VACIA"
        else:
            status = "ok"
        problems += status != "ok"
        logger.info("%-18s Category:%-28s archivos %4d, subcategorias %3d  %s",
                    breed, category, info["files"], info["subcats"], status)
    if problems:
        logger.info("Corregir COMMONS_CATEGORIES en src/lib/dataset/wikimedia.py para las que no dicen ok.")
    return 1 if problems else 0


def _safe_name(title: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", title.removeprefix("File:").rsplit(".", 1)[0]).strip("_")
    return stem[:60] or "image"


def cmd_fetch(args: argparse.Namespace) -> int:
    client = CommonsClient()
    out: Path = args.out
    candidates_csv = out / "candidates.csv"
    previous = pd.read_csv(candidates_csv, dtype=str, keep_default_na=False) if candidates_csv.exists() else pd.DataFrame(columns=CANDIDATE_COLUMNS)
    registered = set(load_registry(args.registry)["url"]) if args.registry.exists() else set()
    known_urls = registered | set(previous["url"])

    rows = []
    for breed in resolve_breeds(args):
        found = client.candidates(breed, COMMONS_CATEGORIES[breed], min_side=args.min_side)
        fresh = [c for c in found if c.page_url not in known_urls]
        chosen = pick_diverse(fresh, args.per_breed, args.max_per_author, args.seed)
        for k, c in enumerate(chosen, start=1):
            suffix = Path(urlparse(c.thumb_url).path).suffix.lower() or ".jpg"
            path = out / breed / f"{k:02d}_{_safe_name(c.title)}{suffix}"
            try:
                client.download(c.thumb_url, path)
            except Exception as exc:  # una descarga fallida no frena al resto
                logger.warning("No se pudo bajar %s: %s", c.page_url, exc)
                continue
            rows.append({
                "file": path.relative_to(out).as_posix(), "breed": breed, "url": c.page_url,
                "author": c.author, "license": c.license, "license_url": c.license_url,
                "retrieved": date.today().isoformat(), "width": c.width, "height": c.height,
            })
        logger.info("%s: %d bajadas en %s", breed, sum(r["breed"] == breed for r in rows), out / breed)

    table = pd.concat([previous, pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)], ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(candidates_csv, index=False, encoding="utf-8")
    logger.info("Listo: revisar %s, borrar las que no sirven y correr `accept`.", out)
    return 0


def cmd_accept(args: argparse.Namespace) -> int:
    out: Path = args.out
    candidates_csv = out / "candidates.csv"
    if not candidates_csv.exists():
        raise SystemExit(f"No existe {candidates_csv}: correr `fetch` primero")
    candidates = pd.read_csv(candidates_csv, dtype=str, keep_default_na=False)
    registry = load_registry(args.registry)
    breeds = set(resolve_breeds(args))

    kept = candidates[candidates["breed"].isin(breeds) & candidates["file"].map(lambda f: (out / f).exists())]
    kept = kept[~kept["url"].isin(set(registry["url"]))]
    if kept.empty:
        logger.info("No hay candidatas nuevas para aceptar.")
        return 0

    new_rows = []
    for breed, group in kept.groupby("breed", sort=True):
        folder = args.dataset / INDEPENDENT_DIR / breed
        numbers = next_numbers(folder, slug(breed), args.collector, len(group))
        for number, (_, row) in zip(numbers, group.iterrows()):
            source = out / row["file"]
            target = folder / f"{slug(breed)}_{number:02d}{source.suffix.lower()}"
            folder.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), target)
            new_rows.append({
                "path": target.relative_to(args.dataset).as_posix(), "breed": breed, "scene": "single",
                "n_dogs": "1", "breeds_in_scene": "", "source": "wikimedia", "url": row["url"],
                "author": row["author"], "license": row["license"], "license_url": row["license_url"],
                "retrieved": row["retrieved"], "collector": args.collector, "people": "none", "notes": "",
            })
        logger.info("%s: %d aceptadas", breed, len(group))

    registry = pd.concat([registry, pd.DataFrame(new_rows, columns=COLUMNS)], ignore_index=True)
    registry.to_csv(args.registry, index=False, encoding="utf-8")
    remaining = candidates[~candidates["url"].isin({r["url"] for r in new_rows})]
    remaining.to_csv(candidates_csv, index=False, encoding="utf-8")
    logger.info("%d filas nuevas en %s", len(new_rows), args.registry)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check", "fetch", "accept"])
    parser.add_argument("--collector", choices=sorted(COLLECTOR_BREEDS), required=True)
    parser.add_argument("--breeds", nargs="+", help="razas (nombre de carpeta); por defecto, las del integrante")
    parser.add_argument("--per-breed", type=int, default=15, help="candidatas a bajar por raza")
    parser.add_argument("--max-per-author", type=int, default=2, help="evita varias fotos de una misma sesion")
    parser.add_argument("--min-side", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "dataset")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "dataset" / "independent_candidates")
    parser.add_argument("--registry", type=Path, default=ROOT / "data" / "splits" / "independent_v1.csv")
    args = parser.parse_args()
    return {"check": cmd_check, "fetch": cmd_fetch, "accept": cmd_accept}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
