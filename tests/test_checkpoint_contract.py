from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lib.models.base import (
    EmbeddingClassifier,
    load_checkpoint,
    save_checkpoint,
    validate_model_contract,
)
from lib.models.dummy import DummyEmbeddingClassifier

CLASSES = ["Afghan", "Beagle", "Boxer"]


def test_shapes() -> None:
    model = DummyEmbeddingClassifier(CLASSES, embedding_dim=16).eval()
    x = torch.randn(4, 3, 32, 32)
    assert model.embed(x).shape == (4, 16)
    assert model(x).shape == (4, 3)
    assert model.num_classes == 3


def test_roundtrip_keeps_classes_dim_and_outputs(tmp_path: Path) -> None:
    model = DummyEmbeddingClassifier(CLASSES, embedding_dim=16, seed=3).eval()
    path = tmp_path / "models" / "dummy.pth"
    save_checkpoint(model, path)
    loaded = load_checkpoint(path)
    assert loaded.classes == CLASSES
    assert loaded.embedding_dim == 16
    x = torch.randn(2, 3, 32, 32)
    torch.testing.assert_close(loaded(x), model(x))
    torch.testing.assert_close(loaded.embed(x), model.embed(x))


def test_save_does_not_change_original_device_or_mode(tmp_path: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DummyEmbeddingClassifier(CLASSES).to(device).train()
    save_checkpoint(model, tmp_path / "m.pth")
    assert model.training
    assert next(model.parameters()).device.type == device
    loaded = load_checkpoint(tmp_path / "m.pth")
    assert not loaded.training
    assert next(loaded.parameters()).device.type == "cpu"


def test_seed_is_reproducible() -> None:
    a = DummyEmbeddingClassifier(CLASSES, seed=1)
    b = DummyEmbeddingClassifier(CLASSES, seed=1)
    x = torch.randn(1, 3, 8, 8)
    torch.testing.assert_close(a(x), b(x))


def test_invalid_construction_raises() -> None:
    with pytest.raises(ValueError):
        DummyEmbeddingClassifier([])
    with pytest.raises(ValueError):
        DummyEmbeddingClassifier(["A", "A"])
    with pytest.raises(ValueError):
        DummyEmbeddingClassifier(CLASSES, embedding_dim=0)


def test_validate_rejects_non_contract_models() -> None:
    with pytest.raises(ValueError):
        validate_model_contract(torch.nn.Linear(2, 2))
    assert issubclass(DummyEmbeddingClassifier, EmbeddingClassifier)


def test_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "no_existe.pth")
