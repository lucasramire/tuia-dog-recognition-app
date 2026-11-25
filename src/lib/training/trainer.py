"""Entrenamiento de los clasificadores de la Etapa 2 (D-21, D-22, D-23, D-25 del plan).

Receta: AdamW, scheduler coseno por epoca, label smoothing, precision mixta
(AMP, solo en GPU), early stopping sobre la accuracy de validacion y semillas
fijas. Se guarda el checkpoint de la MEJOR epoca segun validacion (no el de la
ultima, advertencia de Class_2_lab) y el historial por epoca en JSON.

Nunca se usa test: la seleccion de epoca, estrategia e hiperparametros sale de valid.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from lib.dataset.augmentation import make_transform
from lib.dataset.manifest import class_names
from lib.dataset.torch_dataset import ManifestDataset
from lib.models.base import EmbeddingClassifier, save_checkpoint
from lib.models.cnn_custom import CustomCNN
from lib.models.resnet_classifier import STRATEGIES, ResNetClassifier

logger = logging.getLogger(__name__)

MODEL_NAMES = ("resnet18_finetuned", "cnn_custom")
DEFAULT_EPOCHS = {"resnet18_finetuned": 30, "cnn_custom": 60}


@dataclass
class TrainConfig:
    model: str = "resnet18_finetuned"
    strategy: str = "finetune"  # solo Modelo A: feature_extraction | finetune
    epochs: int | None = None  # None: 30 para el Modelo A, 60 para el B
    batch_size: int = 64
    lr_head: float = 1e-3  # Modelo A, cabeza
    lr_backbone: float = 1e-4  # Modelo A, backbone (solo finetune)
    lr: float = 1e-3  # Modelo B
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    patience: int = 5
    seed: int = 42
    num_workers: int = 0
    amp: bool = True
    augment: bool = True
    class_weights: bool = False  # D-25: pesos inversos a la frecuencia
    image_size: int = 224
    cnn_embedding_dim: int = 256
    cnn_dropout: float = 0.3
    cnn_batchnorm: bool = True
    cnn_channels: tuple[int, ...] = (32, 64, 128, 256, 256)
    pretrained: bool = True  # Modelo A; False solo para tests sin descargar pesos
    max_per_class_train: int | None = None  # pruebas de humo
    max_per_class_valid: int | None = None
    device: str | None = None

    def resolved_epochs(self) -> int:
        return int(self.epochs) if self.epochs is not None else DEFAULT_EPOCHS[self.model]

    def validate(self) -> None:
        if self.model not in MODEL_NAMES:
            raise ValueError(f"Modelo desconocido '{self.model}'; validos: {list(MODEL_NAMES)}")
        if self.model == "resnet18_finetuned" and self.strategy not in STRATEGIES:
            raise ValueError(f"Estrategia desconocida '{self.strategy}'; validas: {list(STRATEGIES)}")
        if self.resolved_epochs() <= 0 or self.batch_size <= 0 or self.patience <= 0:
            raise ValueError("epochs, batch_size y patience deben ser positivos")


@dataclass
class TrainResult:
    checkpoint_path: Path
    history_path: Path
    best_epoch: int
    best_val_accuracy: float
    epochs_run: int
    history: list[dict] = field(default_factory=list)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None) -> torch.device:
    return torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(config: TrainConfig, classes: list[str]) -> EmbeddingClassifier:
    if config.model == "cnn_custom":
        return CustomCNN(
            classes,
            embedding_dim=config.cnn_embedding_dim,
            channels=config.cnn_channels,
            dropout=config.cnn_dropout,
            batchnorm=config.cnn_batchnorm,
        )
    model = ResNetClassifier(classes, pretrained=config.pretrained)
    if config.strategy == "feature_extraction":
        model.freeze_backbone()
    return model


def build_optimizer(model: EmbeddingClassifier, config: TrainConfig) -> torch.optim.Optimizer:
    if isinstance(model, ResNetClassifier):
        groups = model.param_groups(config.lr_head, config.lr_backbone)
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": config.lr, "name": "all"}]
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def make_loader(dataset: ManifestDataset, config: TrainConfig, shuffle: bool, device: torch.device) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


@torch.inference_mode()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y_true, y_pred, probabilidades) sobre un loader."""
    model.eval()
    trues, probs = [], []
    for images, targets in loader:
        with _autocast(device, amp):
            logits = model(images.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        trues.append(targets.numpy())
    p = np.concatenate(probs)
    return np.concatenate(trues), p.argmax(axis=1), p


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with _autocast(device, amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss.detach()) * targets.size(0)
            correct += int((logits.argmax(dim=1) == targets).sum())
            seen += targets.size(0)
    return total_loss / seen, correct / seen


def _write_history(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def train(
    config: TrainConfig,
    manifest: pd.DataFrame,
    dataset_path: Path,
    checkpoint_path: Path,
    history_path: Path,
) -> TrainResult:
    """Entrena con train, elige la mejor epoca con valid y guarda checkpoint e historial."""
    config.validate()
    set_seed(config.seed)
    device = resolve_device(config.device)
    epochs = config.resolved_epochs()
    classes = class_names(manifest)

    train_ds = ManifestDataset(
        manifest, "train", dataset_path, classes,
        make_transform(train=config.augment, image_size=config.image_size, seed=config.seed),
        max_per_class=config.max_per_class_train, seed=config.seed,
    )
    valid_ds = ManifestDataset(
        manifest, "valid", dataset_path, classes,
        make_transform(train=False, image_size=config.image_size),
        max_per_class=config.max_per_class_valid, seed=config.seed,
    )
    train_loader = make_loader(train_ds, config, shuffle=True, device=device)
    valid_loader = make_loader(valid_ds, config, shuffle=False, device=device)

    model = build_model(config, classes).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    weight = None
    if config.class_weights:
        counts = train_ds.class_counts().astype(np.float64)
        weight = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    train_criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=config.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Entrenando %s (%s) en %s: %d train, %d valid, %d razas, %d parametros (%d entrenables)",
        config.model, config.strategy if config.model == "resnet18_finetuned" else "-", device,
        len(train_ds), len(valid_ds), len(classes), n_params, n_trainable,
    )

    payload = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(config).items()},
        "classes": classes,
        "device": str(device),
        "train_size": len(train_ds),
        "valid_size": len(valid_ds),
        "parameters": n_params,
        "trainable_parameters": n_trainable,
        "history": [],
        "best_epoch": None,
        "best_val_accuracy": None,
    }
    history: list[dict] = payload["history"]
    best_acc, best_epoch, stale = -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        lrs = {g.get("name", str(i)): g["lr"] for i, g in enumerate(optimizer.param_groups)}
        train_loss, train_acc = _run_epoch(model, train_loader, train_criterion, device, use_amp, optimizer, scaler)
        val_loss, val_acc = _run_epoch(model, valid_loader, eval_criterion, device, use_amp)
        scheduler.step()
        improved = val_acc > best_acc
        if improved:
            best_acc, best_epoch, stale = val_acc, epoch, 0
            save_checkpoint(model, checkpoint_path)
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "lr": lrs,
                "seconds": time.perf_counter() - start,
                "best": improved,
            }
        )
        payload["best_epoch"], payload["best_val_accuracy"] = best_epoch, best_acc
        _write_history(history_path, payload)
        logger.info(
            "Epoca %d/%d: train loss %.4f acc %.4f | valid loss %.4f acc %.4f%s (%.1fs)",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc, " *" if improved else "", history[-1]["seconds"],
        )
        if stale >= config.patience:
            logger.info("Early stopping: %d epocas sin mejorar (mejor epoca %d)", stale, best_epoch)
            break

    return TrainResult(
        checkpoint_path=Path(checkpoint_path),
        history_path=Path(history_path),
        best_epoch=best_epoch,
        best_val_accuracy=best_acc,
        epochs_run=len(history),
        history=history,
    )
