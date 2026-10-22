import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.inference_service import SegmentationService, get_segmentation_service

app = FastAPI(title="Brain Tumor MRI Segmentation API")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "predictions"
OUTPUT_DIR.mkdir(exist_ok=True)

class PredictionResponse(BaseModel):
    enhancing_tumor_volume_ml: float
    tumor_core_volume_ml: float
    whole_tumor_volume_ml: float
    mask_download_path: str

class HealthResponse(BaseModel):
    status: str
    device: str
    model_val_dice: float

@app.get("/health", response_model=HealthResponse)
def health(service: SegmentationService = Depends(get_segmentation_service)):
    return HealthResponse(
        status="ok",
        device=str(service.device),
        model_val_dice=service.metadata.get("val_dice_mean", 0.0),
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...),
    service: SegmentationService = Depends(get_segmentation_service),
):
    for f in (t1, t1ce, t2, flair):
        if not f.filename.endswith((".nii", ".nii.gz")):
            raise HTTPException(status_code=400, detail=f"'{f.filename}' is not a .nii/.nii.gz file")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        paths = {}
        for name, upload in [("t1", t1), ("t1ce", t1ce), ("t2", t2), ("flair", flair)]:
            dest = tmpdir / upload.filename
            with open(dest, "wb") as out_file:
                shutil.copyfileobj(upload.file, out_file)
            paths[name] = str(dest)

        output_mask_path = OUTPUT_DIR / f"{uuid.uuid4().hex}_mask.nii.gz"

        try:
            result = service.predict(
                paths["t1"], paths["t1ce"], paths["t2"], paths["flair"], str(output_mask_path)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {e!s}")

    return PredictionResponse(
        enhancing_tumor_volume_ml=result["enhancing_tumor_volume_ml"],
        tumor_core_volume_ml=result["tumor_core_volume_ml"],
        whole_tumor_volume_ml=result["whole_tumor_volume_ml"],
        mask_download_path=result["mask_path"],
    )