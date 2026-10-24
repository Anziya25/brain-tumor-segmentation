"""Repeatable inference-only latency measurement.

Usage from Python:
    from app.benchmarking import benchmark_inference
    result = benchmark_inference(model, example_tensor, device="cpu")
    print(result)

Preprocessing, model loading, and NIfTI I/O are excluded from these timings.
"""

import statistics
import time

import torch

from app.device_utils import synchronize_device


def benchmark_inference(model, sample, device, warmup=2, repetitions=5):
    """Return inference-only latency in milliseconds, excluding warm-up."""
    if warmup < 0 or repetitions < 1:
        raise ValueError("warmup must be >= 0 and repetitions must be >= 1")

    device = torch.device(device)
    model = model.to(device).eval()
    sample = sample.to(device)
    timings_ms = []

    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        synchronize_device(device)

        for _ in range(repetitions):
            synchronize_device(device)
            started = time.perf_counter()
            model(sample)
            synchronize_device(device)
            timings_ms.append((time.perf_counter() - started) * 1000)

    return {
        "device": str(device),
        "warmup": warmup,
        "repetitions": repetitions,
        "median_ms": statistics.median(timings_ms),
        "mean_ms": statistics.mean(timings_ms),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "timings_ms": timings_ms,
        "scope": "model_forward_only_excludes_preprocessing_and_io",
    }
