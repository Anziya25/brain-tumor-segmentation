
import io
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
    files = {
        name: (f"{name}.nii", io.BytesIO(b"x"), "application/octet-stream")
        for name in ["t1", "t1ce", "t2", "flair"]
    }

    resp = client.post("/predict", files=files)

    assert resp.status_code == 200

    body = resp.json()
    assert body["whole_tumor_volume_ml"] == 3.0
    assert body["enhancing_tumor_volume_ml"] == 1.0
    assert body["tumor_core_volume_ml"] == 2.0
    assert body["mask_download_path"].startswith("/predictions/")
