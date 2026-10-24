import json
import random

import numpy as np
import pytest
import torch

from app.reproducibility import save_run_metadata, seed_training, sha256_file


def test_seed_training_repeats_python_numpy_and_torch_draws():
    seed_training(42)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_training(42)
    second = (random.random(), np.random.rand(), torch.rand(1).item())
    assert first == second


@pytest.mark.parametrize("invalid", [-1, 2**32, 1.5, True])
def test_seed_training_rejects_invalid_seed(invalid):
    with pytest.raises(ValueError, match="seed"):
        seed_training(invalid)


def test_metadata_saved_as_json(tmp_path):
    target = tmp_path / "run.json"
    save_run_metadata({"seed": 42, "config": {"epochs": 15}}, target)
    assert json.loads(target.read_text())["config"]["epochs"] == 15
    assert not list(tmp_path.glob("*.tmp"))


def test_file_sha256(tmp_path):
    path = tmp_path / "sample"
    path.write_bytes(b"abc")
    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
