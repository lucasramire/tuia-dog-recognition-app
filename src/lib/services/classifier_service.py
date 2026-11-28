from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import onnxruntime

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_NAME = "splits_v1.csv"


class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Funciones a implementar por el estudiante:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)

    La carga de checkpoints (.pth / .onnx) y la seleccion del modelo activo
    ya estan provistas.
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
    ) -> None:
        # checkpoints: nombre logico -> ruta del archivo (ej. resnet18_finetuned -> models/resnet18_finetuned.pth)
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self._loaded: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion.

        Valores esperados: resnet18_finetuned | cnn_custom.
        """
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo.

        Soporta modelos PyTorch (.pth) y exportados a ONNX (.onnx).
        """
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    def train_classifier(self, config: Any = None, manifest_path: Path | None = None) -> Any:
        """
        Entrena el clasificador de razas sobre el dataset (self.dataset_path).

        Modelo A (obligatorio): fine-tuning de ResNet18 (o ResNet50) pre-entrenado.
        Modelo B (obligatorio): CNN propia.

        Debe:
          - Usar los splits train/valid definidos en 1-pre-processing.ipynb.
          - Aplicar el preprocesamiento y data augmentation justificados.
          - Guardar el checkpoint resultante en self.active_checkpoint
            (ej: models/resnet18_finetuned.pth).

        `config` (lib.training.trainer.TrainConfig) es opcional: sin el, se entrena
        el modelo activo con la receta por defecto del plan. Si `config.model`
        difiere del activo, pasa a ser el activo. El historial por epoca queda en
        JSON junto al checkpoint (<nombre>_history.json). Retorna un TrainResult.
        """
        from lib.config import settings
        from lib.dataset.manifest import load_manifest
        from lib.training.trainer import TrainConfig, train

        if config is None:
            config = TrainConfig(
                model=self.active_model_name,
                image_size=self.image_size,
                cnn_embedding_dim=int(settings.cnn_custom_embedding_dim),
            )
        if config.model != self.active_model_name:
            self.set_active_model(config.model)
        checkpoint = self.active_checkpoint
        if checkpoint.suffix.lower() != ".pth":
            raise ValueError(f"El entrenamiento guarda .pth; el checkpoint activo es {checkpoint}")
        manifest_path = Path(manifest_path) if manifest_path else self.default_manifest_path
        manifest = load_manifest(manifest_path)
        logger.info("Entrenando %s con %s", config.model, manifest_path)

        result = train(
            config,
            manifest,
            dataset_path=self.dataset_path,
            checkpoint_path=checkpoint,
            history_path=self.history_path(config.model),
        )
        self._loaded.pop(self.active_model_name, None)  # el proximo load_model lee el checkpoint nuevo
        return result

    @property
    def default_manifest_path(self) -> Path:
        """Splits limpios de 1-pre-processing.ipynb: data/splits/splits_v1.csv."""
        return self.dataset_path.parent / "splits" / DEFAULT_MANIFEST_NAME

    def history_path(self, name: str | None = None) -> Path:
        checkpoint = self.checkpoints[name or self.active_model_name]
        return checkpoint.with_name(f"{checkpoint.stem}_history.json")

    def _torch_model(self, name: str | None = None) -> torch.nn.Module:
        model = self.load_model(name)
        if not isinstance(model, torch.nn.Module):
            raise ValueError(
                f"'{name or self.active_model_name}' esta cargado como ONNX; esta operacion necesita el .pth"
            )
        if not getattr(model, "classes", None):
            raise ValueError("El checkpoint no tiene el atributo 'classes' (lista de razas)")
        return model.eval()

    def predict_split(
        self, split: str = "test", manifest_path: Path | None = None, batch_size: int = 64
    ) -> dict[str, Any]:
        """Predicciones del modelo activo sobre un split (para metricas, matriz de confusion y errores).

        Retorna y_true, y_pred, probs (N, razas), classes y las filas del manifiesto.
        """
        from lib.dataset.augmentation import make_transform
        from lib.dataset.manifest import load_manifest
        from lib.dataset.torch_dataset import ManifestDataset
        from lib.training.trainer import predict

        model = self._torch_model()
        manifest_path = Path(manifest_path) if manifest_path else self.default_manifest_path
        dataset = ManifestDataset(
            load_manifest(manifest_path), split, self.dataset_path, list(model.classes),
            make_transform(train=False, image_size=self.image_size),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        device = next(model.parameters()).device
        y_true, y_pred, probs = predict(model, loader, device)
        return {"y_true": y_true, "y_pred": y_pred, "probs": probs, "classes": list(model.classes), "rows": dataset.rows}

    def evaluate_classifier(self, split: str = "test", manifest_path: Path | None = None) -> dict[str, float]:
        """
        Evalua el modelo activo sobre el conjunto de prueba.

        Debe reportar: accuracy, precision, recall (sensibilidad),
        specificity (especificidad) y F1-Score. La matriz de confusion y las
        curvas de entrenamiento se documentan en 2-train.ipynb.

        Retorna un dict con las metricas, ej:
          {"accuracy": 0.91, "precision": 0.90, "recall": 0.89,
           "specificity": 0.99, "f1": 0.90}

        Precision, recall, specificity y F1 son macro (promedio sin pesos por raza).
        """
        from lib.evaluation.classification import classification_metrics

        pred = self.predict_split(split, manifest_path)
        metrics = classification_metrics(pred["y_true"], pred["y_pred"], len(pred["classes"]))
        logger.info("Metricas de %s sobre %s: %s", self.active_model_name, split, metrics)
        return metrics

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando el modelo propio activo
        (penultima capa del ResNet18 fine-tuned o de la CNN custom).

        Se usa cuando EMBEDDING_MODEL != baseline para que la busqueda por
        similitud (Etapa 1) funcione con los modelos entrenados.
        La imagen llega en BGR (OpenCV). Retorna una lista de floats de
        dimension EMBEDDING_DIM.

        El vector se normaliza en L2, igual que el baseline: con norma 1 el
        producto interno es la similitud coseno.
        """
        from lib.vision.preprocessing import preprocess_to_tensor

        model = self._torch_model()
        x = preprocess_to_tensor(image, self.image_size).unsqueeze(0)
        device = next(model.parameters()).device
        with torch.inference_mode():
            vector = model.embed(x.to(device)).float()
            vector = torch.nn.functional.normalize(vector, p=2, dim=1)[0].cpu().numpy()
        if vector.shape[0] != model.embedding_dim:
            raise ValueError(f"El embedding mide {vector.shape[0]} y el modelo declara {model.embedding_dim}")
        return vector.astype(np.float32).tolist()
