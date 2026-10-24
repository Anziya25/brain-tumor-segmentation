"""Reproducibility helpers for fine-tuning runs.

Seeding improves repeatability but does not guarantee bitwise-identical results
across devices, CUDA kernels, dependency versions, or hardware.
"""

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from monai.utils import set_determinism


def seed_training(seed: int) -> None:
    """Seed Python, NumPy, PyTorch and MONAI random transforms."""
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root, check=True, capture_output=True, text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def build_run_metadata(
    *, seed: int, project_root: Path, split_manifest: Path,
    pretrained_weights: Path, config: dict, device: torch.device,
) -> dict:
    """Collect safe, non-patient-specific run metadata."""
    import monai

    return {
        "seed": seed,
        "git_commit": git_commit(project_root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "monai": monai.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "split_manifest_sha256": sha256_file(split_manifest),
        "pretrained_weights_sha256": sha256_file(pretrained_weights),
        "config": config,
        "determinism_note": (
            "Seeds and MONAI deterministic settings improve repeatability; "
            "identical results across hardware/software are not guaranteed."
        ),
    }


def save_run_metadata(metadata: dict, path: Path) -> None:
    """Atomically persist a JSON metadata record beside the checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temp_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
