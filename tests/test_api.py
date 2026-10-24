
import io

import nibabel as nib
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from app.main import app, get_segmentation_service


class FakeSegmentationService:
    device = "cpu"
    model_val_dice = 0.9
    metadata = {"val_dice_mean": 0.9}

    def predict(
        self,
        t1_path,
        t1ce_path,
        t2_path,
        flair_path,
        output_mask_path,
    ):
        output_path = Path(output_mask_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()

        return {
            "enhancing_tumor_volume_ml": 1.0,
            "tumor_core_volume_ml": 2.0,
            "whole_tumor_volume_ml": 3.0,
            "mask_path": str(output_path),
        }


app.dependency_overrides[get_segmentation_service] = (
    lambda: FakeSegmentationService()
)

client = TestClient(app)


def test_health():
    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_val_dice"] == 0.9


def test_predict_rejects_bad_file_type():
    files = {
        "t1": ("t1.txt", io.BytesIO(b"x"), "text/plain"),
        "t1ce": ("t1ce.nii", io.BytesIO(b"x"), "application/octet-stream"),
        "t2": ("t2.nii", io.BytesIO(b"x"), "application/octet-stream"),
        "flair": ("flair.nii", io.BytesIO(b"x"), "application/octet-stream"),
    }

    resp = client.post("/predict", files=files)

    assert resp.status_code == 400


def test_predict_missing_file():
    files = {
        "t1": ("t1.nii", io.BytesIO(b"x"), "application/octet-stream")
    }

    resp = client.post("/predict", files=files)

    assert resp.status_code == 422


def test_predict_success():
    files = valid_mri_uploads()

    resp = client.post("/predict", files=files)

    assert resp.status_code == 200

    body = resp.json()
    assert body["whole_tumor_volume_ml"] == 3.0
    assert body["enhancing_tumor_volume_ml"] == 1.0
    assert body["tumor_core_volume_ml"] == 2.0
    assert body["mask_download_path"].startswith("/predictions/")


def valid_mri_uploads():
    data = nib.Nifti1Image(
        np.ones((4, 4, 4), dtype=np.float32), np.eye(4)
    ).to_bytes()
    return {
        name: (f"{name}.nii", io.BytesIO(data), "application/octet-stream")
        for name in ("t1", "t1ce", "t2", "flair")
    }


def test_predict_rejects_fake_nifti_bytes():
    files = {
        name: (f"{name}.nii", io.BytesIO(b"not a nifti"), "application/octet-stream")
        for name in ("t1", "t1ce", "t2", "flair")
    }
    resp = client.post("/predict", files=files)
    assert resp.status_code == 400


def test_predict_rejects_misaligned_modalities():
    files = valid_mri_uploads()
    affine = np.eye(4)
    affine[0, 3] = 5
    shifted = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), affine)
    files["flair"] = ("flair.nii", io.BytesIO(shifted.to_bytes()), "application/octet-stream")
    resp = client.post("/predict", files=files)
    assert resp.status_code == 400


def test_inference_slot_rejects_second_request():
    from fastapi import HTTPException
    from app.main import inference_slot

    with inference_slot():
        try:
            with inference_slot():
                raise AssertionError("Second inference slot must not be acquired")
        except HTTPException as exc:
            assert exc.status_code == 429
    # The slot is released after the first request exits.
    with inference_slot():
        pass


def test_expired_predictions_removed(tmp_path, monkeypatch):
    import os
    from app import main

    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    expired = tmp_path / ("a" * 32 + "_mask.nii.gz")
    recent = tmp_path / ("b" * 32 + "_mask.nii.gz")
    unrelated = tmp_path / "notes.txt"
    for path in (expired, recent, unrelated):
        path.write_bytes(b"test")
    os.utime(expired, (100, 100))
    os.utime(recent, (100000, 100000))
    os.utime(unrelated, (100, 100))

    main.cleanup_expired_predictions(now=100001)
    assert not expired.exists()
    assert recent.exists()
    assert unrelated.exists()
