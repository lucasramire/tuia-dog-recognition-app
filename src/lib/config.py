from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv
import logging
import os
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    # extra="ignore": el .env puede tener variables que Settings no declara (ej. TORCH_HOME,
    # que lee torch). Para usar una variable nueva desde el codigo, declararla aca.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    cors_origins: str = Field(
        default="*",
        description="Comma-separated allowed origins, or * for any.",
    )
    app_name: str = "Dog Breed Recognition TP2"

    # Modelo de embeddings seleccionado: baseline | resnet18_finetuned | cnn_custom
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "baseline")
    model_path: Path = Path(os.getenv("MODEL_PATH", "models"))
    resnet18_model_name: str = os.getenv("RESNET18_MODEL_NAME", "resnet18_finetuned.pth")
    cnn_custom_model_name: str = os.getenv("CNN_CUSTOM_MODEL_NAME", "cnn_custom.pth")

    # Busqueda por similitud
    similarity_metric: str = os.getenv("SIMILARITY_METRIC", "cosine")
    similarity_threshold: float = os.getenv("SIMILARITY_THRESHOLD", 0.55)
    top_k: int = os.getenv("TOP_K", 10)
    image_size: int = os.getenv("IMAGE_SIZE", 224)
    # Dimension del embedding del modelo elegido en EMBEDDING_MODEL (ver expected_embedding_dim)
    embedding_dim: int = os.getenv("EMBEDDING_DIM", 2048)
    # Dimension de cada modelo: una columna vector(dim) de pgvector tiene dimension fija
    baseline_embedding_dim: int = os.getenv("BASELINE_EMBEDDING_DIM", 2048)
    resnet18_embedding_dim: int = os.getenv("RESNET18_EMBEDDING_DIM", 512)
    cnn_custom_embedding_dim: int = os.getenv("CNN_CUSTOM_EMBEDDING_DIM", 256)

    # Configuracion de YOLO (Etapa 3)
    yolo_model: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
    yolo_conf_threshold: float = os.getenv("YOLO_CONF_THRESHOLD", 0.25)
    # IoU de la supresion de no maximos (NMS) de YOLO; 0.7 es el valor por defecto de ultralytics
    yolo_iou_threshold: float = os.getenv("YOLO_IOU_THRESHOLD", 0.7)
    yolo_dog_class_id: int = os.getenv("YOLO_DOG_CLASS_ID", 16)

    # Probabilidad minima para aceptar la raza de un recorte; por debajo devuelve "unknown".
    # 0 desactiva el rechazo. Se calibra sobre valid en 2-train.ipynb.
    classifier_min_confidence: float = os.getenv("CLASSIFIER_MIN_CONFIDENCE", 0.0)

    # Paths
    embeddings_path: Path = Path(os.getenv("EMBEDDINGS_PATH", "data/embeddings.json"))
    data_path: Path = Path(os.getenv("DATA_PATH", "data"))
    dataset_path: Path = Path(os.getenv("DATASET_PATH", "data/dataset"))
    output_path: Path = Path(os.getenv("OUTPUT_PATH", "output"))
    max_workers: int = os.getenv("MAX_WORKERS", 2)

    # PostgreSQL / pgvector
    use_pgvector: bool = os.getenv("USE_PGVECTOR", "True").lower() == "true"
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = os.getenv("POSTGRES_PORT", 5432)
    postgres_db: str = os.getenv("POSTGRES_DB", "dogs")
    postgres_user: str = os.getenv("POSTGRES_USER", "dogs_user")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "dogs_pass")

    @property
    def expected_embedding_dim(self) -> int:
        """Dimension esperada para el modelo elegido en EMBEDDING_MODEL."""
        dims = {
            "baseline": self.baseline_embedding_dim,
            "resnet18_finetuned": self.resnet18_embedding_dim,
            "cnn_custom": self.cnn_custom_embedding_dim,
        }
        if self.embedding_model not in dims:
            raise ValueError(
                f"EMBEDDING_MODEL desconocido: '{self.embedding_model}'. Valores validos: {sorted(dims)}"
            )
        return dims[self.embedding_model]

    def check_embedding_dim(self) -> None:
        """Falla con un mensaje claro si EMBEDDING_DIM no coincide con el modelo elegido."""
        expected = self.expected_embedding_dim
        if int(self.embedding_dim) != int(expected):
            raise ValueError(
                f"EMBEDDING_DIM={self.embedding_dim} no coincide con la dimension de "
                f"'{self.embedding_model}' ({expected}). Corregi EMBEDDING_DIM en el .env."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
