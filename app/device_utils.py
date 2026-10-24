"""Device-aware precision helpers for CPU and CUDA training."""

from contextlib import nullcontext

import torch


def autocast_context(device):
    """Use CUDA mixed precision only when running on CUDA."""
    if torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(device):
    """Return a disabled scaler on CPU, enabled scaler on CUDA."""
    return torch.amp.GradScaler("cuda", enabled=torch.device(device).type == "cuda")


def synchronize_device(device):
    """Synchronize asynchronous CUDA work before timing; no-op on CPU."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
