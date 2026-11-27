"""Candidatas para el conjunto independiente desde Wikimedia Commons (P1.4, enunciado 3.4).

Consulta la API publica de Commons (sin credenciales) y devuelve, por raza, imagenes con
licencia libre y su metadata (pagina, autor, licencia), para que una persona las revise y
elija. Nada se acepta automaticamente: la raza y la ausencia de personas reconocibles las
confirma quien revisa.

Criterios:
  - Solo licencias que permiten el uso academico y la redistribucion por link: dominio
    publico, CC0, CC BY y CC BY-SA. Se excluyen NC, ND, GFDL sola y "fair use".
  - JPG o PNG con lado menor >= `min_side` (por defecto 224, IMAGE_SIZE).
  - Diversidad: como mucho `max_per_author` imagenes por autor, porque las fotos de un
    mismo autor suelen ser de la misma sesion (mismo perro, mismo fondo).
  - Se recorren las subcategorias de un nivel, salteando las que no son fotos del perro
    (arte, dibujos, estampillas, juguetes, cruzas, cachorros).
"""

from __future__ import annotations

import html
import logging
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

API_URL = "https://commons.wikimedia.org/w/api.php"
# Identificacion generica del cliente (politica de User-Agent de Wikimedia), sin datos personales.
USER_AGENT = "tuia-dog-recognition-tp/1.0 (academic project; python-requests)"

# Carpeta del dataset -> categoria de Commons. Verificar con `collect_independent.py check`.
COMMONS_CATEGORIES: dict[str, str] = {
    "Lhasa": "Lhasa Apso",
    "Shih-Tzu": "Shih Tzu",
    "Boston Terrier": "Boston Terrier",
    "Bulldog": "Bulldog",
    "French Bulldog": "French Bulldog",
    "Cockapoo": "Cockapoo",
    "Labradoodle": "Labradoodle",
    "American Hairless": "American Hairless Terrier",
    "Mex Hairless": "Xoloitzquintle",
    "Labrador": "Labrador Retriever",
    "Beagle": "Beagle (dog)",
    "Pug": "Pug",
    "Chihuahua": "Chihuahua (dog)",
    "Border Collie": "Border Collie",
    "Collie": "Rough Collie",
    "German Sheperd": "German Shepherd Dog",
    "Malinois": "Belgian Shepherd Malinois",
    "Shiba Inu": "Shiba Inu",
    "Dingo": "Dingoes",
    "Rhodesian": "Rhodesian Ridgeback",
    "Vizsla": "Hungarian Vizsla",
    "Golden Retriever": "Golden Retriever",
    "Siberian Husky": "Siberian Husky",
    "Poodle": "Poodles",
    "Yorkie": "Yorkshire Terrier",
}

# Negativos para calibrar el umbral de "unknown" de la Etapa 1 (D-30): imagenes sin perro.
# Grupo -> categoria de Commons. Sin Coyote, Dingo, Dhole ni African Wild Dog: son clases del
# dataset. Lobo: la subespecie euroasiatica, porque "Canis lupus" incluye al perro domestico.
NEGATIVE_CATEGORIES: dict[str, str] = {
    "cat": "Felis silvestris catus",
    "fox": "Vulpes vulpes",
    "wolf": "Canis lupus lupus",
    "bear": "Ursus arctos",
    "sheep": "Ovis aries",
    "horse": "Horses",
    "cattle": "Bos taurus",
    "rabbit": "Oryctolagus cuniculus",
    "living_room": "Living rooms",
    "kitchen": "Kitchens",
    "bedroom": "Bedrooms",
    "empty_street": "Empty streets",
    "garden": "Gardens",
}

_ACCEPTED_LICENSE = re.compile(r"^(public domain|pd\b.*|cc0.*|cc[ -]by( |-)\d.*|cc[ -]by-sa( |-)\d.*)$", re.IGNORECASE)
_REJECTED_LICENSE = re.compile(r"\b(nc|nd)\b|non-?commercial|no ?deriv|fair use", re.IGNORECASE)
_SKIP_SUBCATEGORY = re.compile(
    r"art\b|arts\b|drawing|painting|illustration|statue|sculpture|stamp|logo|toy|cartoon|heraldry|"
    r"postcard|in culture|mix|cross|puppies|puppy|skull|skeleton|taxidermy|anatomy|video|audio|"
    r"sound|map|ancestor",
    re.IGNORECASE,
)
_IMAGE_MIMES = {"image/jpeg", "image/png"}
# Wikimedia limita (429) las miniaturas de anchos no estandar; 960 es uno de los estandar.
THUMB_WIDTH = 960
MAX_RETRY_WAIT = 120.0


@dataclass
class Candidate:
    breed: str
    title: str  # "File:..."
    page_url: str  # pagina de la imagen en Commons (lo que va en `url`)
    thumb_url: str  # copia reescalada para descargar
    author: str
    license: str
    license_url: str
    width: int
    height: int


def is_accepted_license(short_name: str) -> bool:
    """True si la licencia permite el uso del TP (PD, CC0, CC BY, CC BY-SA)."""
    name = (short_name or "").strip()
    return bool(name) and not _REJECTED_LICENSE.search(name) and bool(_ACCEPTED_LICENSE.match(name))


def clean_html(value: str) -> str:
    """Texto plano desde los campos HTML de extmetadata (Artist suele venir como link)."""
    text = re.sub(r"<[^>]+>", " ", value or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def parse_candidate(breed: str, page: dict, min_side: int) -> Candidate | None:
    """Candidate desde una pagina de la API, o None si no cumple formato, tamano o licencia."""
    infos = page.get("imageinfo") or []
    if not infos:
        return None
    info = infos[0]
    if info.get("mime") not in _IMAGE_MIMES:
        return None
    width, height = int(info.get("width", 0)), int(info.get("height", 0))
    if min(width, height) < min_side:
        return None
    meta = info.get("extmetadata") or {}
    license_name = clean_html((meta.get("LicenseShortName") or {}).get("value", ""))
    if not is_accepted_license(license_name):
        return None
    author = clean_html((meta.get("Artist") or {}).get("value", "")) or "unknown"
    return Candidate(
        breed=breed,
        title=page["title"],
        page_url=info.get("descriptionurl", ""),
        thumb_url=info.get("thumburl") or info.get("url", ""),
        author=author,
        license=license_name,
        license_url=clean_html((meta.get("LicenseUrl") or {}).get("value", "")),
        width=width,
        height=height,
    )


def pick_diverse(candidates: list[Candidate], n: int, max_per_author: int, seed: int) -> list[Candidate]:
    """Hasta `n` candidatas al azar (semilla fija) con como mucho `max_per_author` por autor."""
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    per_author: dict[str, int] = defaultdict(int)
    chosen = []
    for c in shuffled:
        if per_author[c.author] >= max_per_author:
            continue
        per_author[c.author] += 1
        chosen.append(c)
        if len(chosen) == n:
            break
    return chosen


class CommonsClient:
    """Cliente minimo de la API de Commons con pausa entre pedidos."""

    def __init__(self, pause: float = 1.0, timeout: float = 30.0, retries: int = 4) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.pause = pause
        self.timeout = timeout
        self.retries = retries

    def _request(self, url: str, params: dict | None = None) -> requests.Response:
        """GET con pausa entre pedidos; ante 429 o 5xx espera (Retry-After o backoff) y reintenta."""
        for attempt in range(self.retries + 1):
            time.sleep(self.pause)
            response = self.session.get(url, params=params, timeout=self.timeout)
            if response.status_code != 429 and response.status_code < 500:
                response.raise_for_status()
                return response
            if attempt == self.retries:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.isdigit() else 5.0 * 2**attempt
            if wait > MAX_RETRY_WAIT:
                raise RuntimeError(
                    f"Commons pide esperar {wait:.0f} s ({response.status_code}) para {url}. "
                    "Volver a correr mas tarde: lo ya bajado queda registrado."
                )
            logger.warning("Commons respondio %d; reintento en %.0f s", response.status_code, wait)
            time.sleep(wait)
        raise AssertionError("no se alcanza")

    def _get(self, params: dict) -> dict:
        response = self._request(API_URL, {**params, "format": "json", "formatversion": 2})
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"Error de la API de Commons: {data['error']}")
        return data

    def category_info(self, categories: list[str]) -> dict[str, dict]:
        """Por categoria: si existe, cantidad de archivos y subcategorias, y a donde redirige.

        Una sola consulta para todas (hasta 50). Una categoria "redirect" de Commons existe
        pero esta vacia: `redirect_to` dice cual usar.
        """
        titles = [f"Category:{c}" for c in categories]
        data = self._get({
            "action": "query", "prop": "categoryinfo|templates|links", "titles": "|".join(titles),
            "tltemplates": "Template:Category redirect", "plnamespace": 14, "pllimit": "max",
        })
        query = data["query"]
        normalized = {n["from"]: n["to"] for n in query.get("normalized", [])}
        pages = {p["title"]: p for p in query["pages"]}
        out = {}
        for category, title in zip(categories, titles):
            page = pages.get(normalized.get(title, title), {})
            info = page.get("categoryinfo")
            if page.get("missing") or info is None:
                out[category] = {"exists": False, "files": 0, "subcats": 0, "redirect_to": ""}
                continue
            redirect = ""
            if page.get("templates"):
                links = [l["title"].removeprefix("Category:") for l in page.get("links", [])]
                redirect = links[0] if len(links) == 1 else ", ".join(links)
            out[category] = {
                "exists": True, "files": int(info.get("files", 0)), "subcats": int(info.get("subcats", 0)),
                "redirect_to": redirect,
            }
        return out

    def subcategories(self, category: str) -> list[str]:
        names, params = [], {
            "action": "query", "list": "categorymembers", "cmtitle": f"Category:{category}",
            "cmtype": "subcat", "cmlimit": 200,
        }
        while True:
            data = self._get(params)
            names += [m["title"].removeprefix("Category:") for m in data["query"]["categorymembers"]]
            if "continue" not in data:
                return names
            params.update(data["continue"])

    def files(self, category: str, breed: str, min_side: int, limit: int, thumb_width: int) -> list[Candidate]:
        """Archivos de una categoria (sin recursion) que pasan los filtros, hasta `limit` revisados."""
        params = {
            "action": "query", "generator": "categorymembers", "gcmtitle": f"Category:{category}",
            "gcmtype": "file", "gcmlimit": 50, "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata", "iiurlwidth": thumb_width,
            "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist",
        }
        found, seen = [], 0
        while seen < limit:
            data = self._get(params)
            pages = (data.get("query") or {}).get("pages", [])
            seen += len(pages)
            found += [c for c in (parse_candidate(breed, p, min_side) for p in pages) if c is not None]
            if "continue" not in data:
                break
            params.update(data["continue"])
        return found

    def candidates(
        self, breed: str, category: str, min_side: int = 224, max_files: int = 400, thumb_width: int = THUMB_WIDTH
    ) -> list[Candidate]:
        """Candidatas de la categoria y de sus subcategorias directas (salteando arte, cruzas, etc.)."""
        categories = [category] + [s for s in self.subcategories(category) if not _SKIP_SUBCATEGORY.search(s)]
        found: dict[str, Candidate] = {}
        for cat in categories:
            if len(found) >= max_files:
                break
            for c in self.files(cat, breed, min_side, max_files, thumb_width):
                found.setdefault(c.title, c)
        logger.info("%s: %d candidatas con licencia libre en %d categorias", breed, len(found), len(categories))
        return list(found.values())

    def download(self, url: str, path: Path) -> None:
        response = self._request(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
