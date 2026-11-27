"""Costo computacional de los modelos: parametros, tamano y latencia en CPU (N2.6, D-09).

La latencia se mide en CPU porque es donde corre la app (Docker sin GPU). Se mide una imagen
por llamada (batch 1), que es el caso de un request, con unas pasadas de calentamiento antes
(la primera llamada paga la reserva de memoria y la eleccion de kernels) y se informa la
mediana, que es robusta a interrupciones del sistema operativo.
"""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def file_size_mb(path: Path) -> float:
    return Path(path).stat().st_size / 2**20


def time_calls(fn: Callable[[], object], warmup: int = 5, repeats: int = 30) -> dict[str, float]:
    """Mediana y percentil 90 en milisegundos de `repeats` llamadas, despues de `warmup`."""
    if repeats <= 0:
        raise ValueError(f"repeats debe ser positivo, llego {repeats}")
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return {"ms_median": statistics.median(times), "ms_p90": float(np.percentile(times, 90))}


def torch_cpu_latency(
    model: nn.Module, image_size: int, call: str = "forward", warmup: int = 5, repeats: int = 30
) -> dict[str, float]:
    """Latencia por imagen del modelo de PyTorch en CPU (`call` = "forward" o "embed").

    Trabaja sobre una copia en CPU para no mover el modelo original de dispositivo.
    """
    import copy

    cpu_model = copy.deepcopy(model).cpu().eval()
    fn = cpu_model.embed if call == "embed" else cpu_model
    x = torch.randn(1, 3, image_size, image_size)

    def run() -> None:
        with torch.inference_mode():
            fn(x)

    return time_calls(run, warmup, repeats)


def onnx_cpu_latency(onnx_path: Path, image_size: int, warmup: int = 5, repeats: int = 30) -> dict[str, float]:
    """Latencia por imagen del modelo exportado a ONNX con onnxruntime en CPU."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    x = np.random.default_rng(0).standard_normal((1, 3, image_size, image_size)).astype(np.float32)
    return time_calls(lambda: session.run(None, {name: x}), warmup, repeats)


def cpu_description() -> str:
    """Procesador y cantidad de hilos de PyTorch, para dejar asentado donde se midio."""
    name = platform.processor() or platform.machine()
    return f"{name} | {torch.get_num_threads()} hilos de PyTorch | {platform.system()}"
