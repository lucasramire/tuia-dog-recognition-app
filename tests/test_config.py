from __future__ import annotations

import pytest

from lib.config import Settings


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_new_variables_have_defaults() -> None:
    # config.py carga src/.env al importarse y los defaults de Settings salen de esas variables:
    # se compara contra el entorno (si la variable no esta, el default del codigo).
    import os

    s = _settings()
    assert s.yolo_iou_threshold == float(os.getenv("YOLO_IOU_THRESHOLD", 0.7))
    assert s.classifier_min_confidence == float(os.getenv("CLASSIFIER_MIN_CONFIDENCE", 0.0))
    assert (s.baseline_embedding_dim, s.resnet18_embedding_dim, s.cnn_custom_embedding_dim) == (2048, 512, 256)


@pytest.mark.parametrize(
    "model,dim",
    [("baseline", 2048), ("resnet18_finetuned", 512), ("cnn_custom", 256)],
)
def test_embedding_dim_matches_model(model: str, dim: int) -> None:
    s = _settings(embedding_model=model, embedding_dim=dim)
    assert s.expected_embedding_dim == dim
    s.check_embedding_dim()


def test_embedding_dim_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        _settings(embedding_model="baseline", embedding_dim=512).check_embedding_dim()


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError):
        _settings(embedding_model="otro").check_embedding_dim()
