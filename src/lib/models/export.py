"""Exportacion a ONNX de un modelo que cumple el contrato (EmbeddingClassifier).

El .onnx devuelve logits (N, num_classes) y guarda la lista de razas como JSON
en los metadatos (clave `classes`), porque ONNX no conserva atributos de Python.
`DetectionService.classify_detected_dog` la lee de ahi.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import onnx
import torch

from lib.models.base import EmbeddingClassifier, validate_model_contract

logger = logging.getLogger(__name__)

ONNX_CLASSES_KEY = "classes"
ONNX_EMBEDDING_DIM_KEY = "embedding_dim"


def export_onnx(model: EmbeddingClassifier, path: Path, image_size: int, opset: int = 17) -> Path:
    validate_model_contract(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = copy.deepcopy(model).cpu().eval()
    dummy = torch.zeros(1, 3, int(image_size), int(image_size), dtype=torch.float32)
    torch.onnx.export(
        snapshot,
        (dummy,),
        str(path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    proto = onnx.load(str(path))
    for key, value in (
        (ONNX_CLASSES_KEY, json.dumps(snapshot.classes, ensure_ascii=True)),
        (ONNX_EMBEDDING_DIM_KEY, str(snapshot.embedding_dim)),
    ):
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(proto, str(path))
    logger.info("Modelo exportado a ONNX en %s", path)
    return path
