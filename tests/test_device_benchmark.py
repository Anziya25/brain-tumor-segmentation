import pytest
import torch

from app.benchmarking import benchmark_inference
from app.device_utils import autocast_context, make_grad_scaler, synchronize_device


def test_cpu_autocast_does_not_change_precision():
    with autocast_context("cpu"):
        result = torch.ones(2, dtype=torch.float32) * 2
    assert result.dtype == torch.float32


def test_cpu_grad_scaler_disabled():
    assert not make_grad_scaler("cpu").is_enabled()


def test_cpu_synchronization_noop():
    assert synchronize_device("cpu") is None


def test_benchmark_returns_repeated_forward_timings():
    model = torch.nn.Identity()
    sample = torch.ones(1, 2)
    result = benchmark_inference(model, sample, "cpu", warmup=1, repetitions=3)
    assert result["device"] == "cpu"
    assert result["repetitions"] == 3
    assert len(result["timings_ms"]) == 3
    assert all(ms >= 0 for ms in result["timings_ms"])
    assert result["scope"] == "model_forward_only_excludes_preprocessing_and_io"


@pytest.mark.parametrize("warmup,repetitions", [(-1, 3), (1, 0)])
def test_benchmark_rejects_invalid_counts(warmup, repetitions):
    with pytest.raises(ValueError):
        benchmark_inference(torch.nn.Identity(), torch.ones(1), "cpu", warmup, repetitions)
