"""Construccion de los servicios del sistema (wiring compartido entre la API y los scripts)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from lib.config import Settings
from lib.services.classifier_service import ClassifierService
from lib.services.detection_service import DetectionService
from lib.services.similarity_service import SimilarityService
from lib.storage.base import EmbeddingStoreProtocol
from lib.storage.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)

StoreType = Union[EmbeddingStoreProtocol, EmbeddingStore]
UrlResolver = Callable[[Path], Optional[str]]


@dataclass
class ServiceContainer:
    store: StoreType
    similarity: SimilarityService
    classifier: ClassifierService
    detection: DetectionService


BASELINE_MODEL = "baseline"


def store_table_for(model: str) -> str:
    """Tabla de pgvector de cada modelo de embeddings (mejora opcional 7.3, una tabla por modelo).

    El baseline conserva la tabla original `embeddings`, asi una base ya indexada sigue sirviendo.
    """
    return "embeddings" if model == BASELINE_MODEL else f"embeddings_{model}"


def store_json_for(settings: Settings, model: str) -> Path:
    """Archivo JSON de cada modelo cuando USE_PGVECTOR=false (el baseline usa EMBEDDINGS_PATH)."""
    path = Path(settings.embeddings_path)
    return path if model == BASELINE_MODEL else path.with_name(f"{path.stem}_{model}{path.suffix}")


def build_store(settings: Settings, model: Optional[str] = None) -> StoreType:
    """Base vectorial de un modelo de embeddings (por defecto, el de EMBEDDING_MODEL).

    Cada modelo tiene su propia tabla (o archivo JSON) con su dimension, asi cambiar de modelo
    en la Etapa 1 no obliga a reindexar ni borra la base de otro modelo.
    """
    model = model or settings.embedding_model
    dim = settings.embedding_dim_for(model)
    if settings.use_pgvector:
        logger.info("Using PostgreSQL vector store (modelo %s, tabla %s)", model, store_table_for(model))
        # Import lazy: psycopg/pgvector solo se requieren con USE_PGVECTOR=true
        # (permite usar el resto de los servicios en entornos sin postgres, ej: Colab).
        from lib.storage.pgvector_store import PgVectorEmbeddingStore

        return PgVectorEmbeddingStore(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            embedding_dim=dim,
            table=store_table_for(model),
        )
    logger.info("Using JSON file vector store (modelo %s)", model)
    return EmbeddingStore(store_json_for(settings, model))


def store_size(store: StoreType) -> int:
    """Cantidad de embeddings indexados (pgvector cuenta en la base; JSON lee el archivo)."""
    count = getattr(store, "count", None)
    return int(count()) if callable(count) else len(store.all())


def build_classifier(settings: Settings) -> ClassifierService:
    return ClassifierService(
        checkpoints={
            "resnet18_finetuned": settings.model_path / settings.resnet18_model_name,
            "cnn_custom": settings.model_path / settings.cnn_custom_model_name,
        },
        image_size=settings.image_size,
        dataset_path=settings.dataset_path,
        output_path=settings.output_path,
    )


def build_similarity(
    settings: Settings,
    store: StoreType,
    url_resolver: Optional[UrlResolver] = None,
) -> SimilarityService:
    return SimilarityService(
        store=store,
        similarity_metric=settings.similarity_metric,
        similarity_threshold=settings.similarity_threshold,
        top_k=settings.top_k,
        image_size=settings.image_size,
        model_name=settings.embedding_model,
        url_resolver=url_resolver,
        data_root=settings.data_path,
    )


def build_detection(settings: Settings, classifier: ClassifierService) -> DetectionService:
    return DetectionService(
        classifier=classifier,
        yolo_model=settings.yolo_model,
        conf_threshold=settings.yolo_conf_threshold,
        dog_class_id=settings.yolo_dog_class_id,
    )


def build_services(
    settings: Settings,
    url_resolver: Optional[UrlResolver] = None,
) -> ServiceContainer:
    # EMBEDDING_DIM tiene que coincidir con el modelo de EMBEDDING_MODEL: si no, se falla al arrancar
    # con un mensaje claro, en lugar de al primer insert en pgvector (enunciado, seccion 11).
    settings.check_embedding_dim()
    store = build_store(settings)
    classifier = build_classifier(settings)
    similarity = build_similarity(settings, store, url_resolver)
    detection = build_detection(settings, classifier)
    return ServiceContainer(
        store=store,
        similarity=similarity,
        classifier=classifier,
        detection=detection,
    )
