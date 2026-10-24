"""Integration tests for real NIfTI I/O and inference postprocessing.

The pretrained UNETR is intentionally replaced by a deterministic predictor;
real MONAI preprocessing, NIfTI saving, and volume calculations still run.
"""

import nibabel as nib
import numpy as np
import pytest
import torch

from app import inference_service
from app.preprocessing import get_inference_transforms


def write_modalities(tmp_path, affine):
    paths = {}
    for name in ("t1", "t1ce", "t2", "flair"):
        path = tmp_path / f"{name}.nii.gz"
        values = np.ones((8, 8, 8), dtype=np.float32)
        values[2:6, 2:6, 2:6] = 10.0
        nib.save(nib.Nifti1Image(values, affine), str(path))
        paths[name] = path
    return paths


def make_service():
    service = inference_service.SegmentationService.__new__(
        inference_service.SegmentationService
    )
    service.device = torch.device("cpu")
    service.model = torch.nn.Identity()
    service.transforms = get_inference_transforms()
    return service


def test_real_nifti_output_and_volumes(tmp_path, monkeypatch):
    affine = np.eye(4)
    paths = write_modalities(tmp_path, affine)

    def deterministic_inference(inputs, roi_size, sw_batch_size, predictor, overlap):
        assert inputs.shape == (1, 4, 8, 8, 8)
        probs = torch.zeros((1, 3, 8, 8, 8), dtype=torch.float32)
        probs[0, 0, 1, 1, 1] = 1.0  # ET without independently predicted TC/WT
        probs[0, 1, 2, 2, 2] = 1.0  # TC without independently predicted WT
        probs[0, 2, 3, 3, 3] = 1.0  # WT only
        return probs

    monkeypatch.setattr(inference_service, "sliding_window_inference", deterministic_inference)
    output = tmp_path / "prediction.nii.gz"
    result = make_service().predict(
        paths["t1"], paths["t1ce"], paths["t2"], paths["flair"], output
    )

    saved = nib.load(str(output))
    labels = np.asarray(saved.dataobj)
    assert labels.shape == (8, 8, 8)
    assert set(np.unique(labels)) == {0, 1, 2, 4}
    assert labels[1, 1, 1] == 4
    assert labels[2, 2, 2] == 1
    assert labels[3, 3, 3] == 2
    np.testing.assert_allclose(saved.affine, affine, atol=1e-5)
    assert result["enhancing_tumor_volume_ml"] == 0.001
    assert result["tumor_core_volume_ml"] == 0.002
    assert result["whole_tumor_volume_ml"] == 0.003
    assert result["mask_path"] == str(output)


def test_misaligned_nifti_rejected_before_inference(tmp_path, monkeypatch):
    paths = write_modalities(tmp_path, np.eye(4))
    shifted = np.eye(4)
    shifted[0, 3] = 3.0
    nib.save(
        nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), shifted),
        str(paths["flair"]),
    )

    def must_not_run(*args, **kwargs):
        pytest.fail("Inference should not run for misaligned inputs")

    monkeypatch.setattr(inference_service, "sliding_window_inference", must_not_run)
    with pytest.raises(ValueError, match="grid differs"):
        make_service().predict(
            paths["t1"], paths["t1ce"], paths["t2"], paths["flair"],
            tmp_path / "prediction.nii.gz",
        )
