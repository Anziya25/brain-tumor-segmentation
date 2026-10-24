
import logging
import re
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.inference_service import SegmentationService, get_segmentation_service

logger = logging.getLogger(__name__)

app = FastAPI(title="Brain Tumor MRI Segmentation API")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE_BYTES = 512 * 1024 * 1024  # 512 MiB per MRI file
UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB


class PredictionResponse(BaseModel):
    enhancing_tumor_volume_ml: float
    tumor_core_volume_ml: float
    whole_tumor_volume_ml: float
    mask_download_path: str


class HealthResponse(BaseModel):
    status: str
    device: str
    model_val_dice: float


def get_nifti_suffix(filename: str | None) -> str | None:
    if not filename:
        return None

    lower_name = filename.lower()

    if lower_name.endswith(".nii.gz"):
        return ".nii.gz"
    if lower_name.endswith(".nii"):
        return ".nii"

    return None


def save_upload(upload: UploadFile, destination: Path) -> None:
    total_bytes = 0

    with destination.open("wb") as output_file:
        while True:
            chunk = upload.file.read(UPLOAD_CHUNK_SIZE)

            if not chunk:
                break

            total_bytes += len(chunk)

            if total_bytes > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Each MRI file must be no larger than 512 MiB.",
                )

            output_file.write(chunk)

    if total_bytes == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Uploaded MRI file is empty.",
        )


@app.get("/health", response_model=HealthResponse)
def health(
    service: SegmentationService = Depends(get_segmentation_service),
):
    return HealthResponse(
        status="ok",
        device=str(service.device),
        model_val_dice=float(service.model_val_dice),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...),
    service: SegmentationService = Depends(get_segmentation_service),
):
    uploads = {
        "t1": t1,
        "t1ce": t1ce,
        "t2": t2,
        "flair": flair,
    }

    # Validate all filenames before writing any files.
    suffixes = {}

    for modality, upload in uploads.items():
        suffix = get_nifti_suffix(upload.filename)

        if suffix is None:
            raise HTTPException(
                status_code=400,
                detail=f"{modality} must be a .nii or .nii.gz file.",
            )

        suffixes[modality] = suffix

    try:
        with tempfile.TemporaryDirectory(prefix="brain_tumor_") as tmp:
            temp_dir = Path(tmp)
            paths = {}

            for modality, upload in uploads.items():
                # Never use the client-provided filename as a filesystem path.
                destination = temp_dir / f"{modality}{suffixes[modality]}"
                save_upload(upload, destination)
                paths[modality] = str(destination)

            output_name = f"{uuid.uuid4().hex}_mask.nii.gz"
            output_path = OUTPUT_DIR / output_name

            result = service.predict(
                t1_path=paths["t1"],
                t1ce_path=paths["t1ce"],
                t2_path=paths["t2"],
                flair_path=paths["flair"],
                output_mask_path=str(output_path),
            )

            if not output_path.is_file():
                raise RuntimeError("Inference did not produce a segmentation mask.")

            return PredictionResponse(
                enhancing_tumor_volume_ml=float(
                    result["enhancing_tumor_volume_ml"]
                ),
                tumor_core_volume_ml=float(
                    result["tumor_core_volume_ml"]
                ),
                whole_tumor_volume_ml=float(
                    result["whole_tumor_volume_ml"]
                ),
                mask_download_path=f"/predictions/{output_name}",
            )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Brain tumor MRI inference failed")
        raise HTTPException(
            status_code=500,
            detail="Inference failed. Check the server logs for details.",
        )


@app.get("/predictions/{mask_name}", response_class=FileResponse)
def download_prediction(mask_name: str):
    # Only serve files created using the API's expected naming convention.
    if not re.fullmatch(r"[0-9a-f]{32}_mask\.nii\.gz", mask_name):
        raise HTTPException(status_code=400, detail="Invalid mask filename.")

    mask_path = OUTPUT_DIR / mask_name

    if not mask_path.is_file():
        raise HTTPException(status_code=404, detail="Segmentation mask not found.")

    return FileResponse(
        path=mask_path,
        media_type="application/gzip",
        filename=mask_name,
    )
