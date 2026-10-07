import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from monai.inferers import sliding_window_inference

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "unetr-brats"
FT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "finetuned_unetr.pth"
sys.path.insert(0, str(MODEL_DIR))

from inference import load_model
from app.preprocessing import get_inference_transforms

PATCH_SIZE = (128, 128, 128)
VOXEL_VOLUME_ML = 1.0 / 1000  # 1mm^3 voxel after resampling = 0.001 mL

class SegmentationService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.metadata = load_model(str(MODEL_DIR / "pytorch_model.bin"))
        checkpoint = torch.load(FT_CHECKPOINT_PATH, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.transforms = get_inference_transforms()
        print(f"Model loaded on {self.device}. Fine-tuned val_dice: {checkpoint.get('val_dice_mean', 'n/a')}")

    def predict(self, t1_path, t1ce_path, t2_path, flair_path, output_mask_path):
        sample = {"t1": t1_path, "t1ce": t1ce_path, "t2": t2_path, "flair": flair_path}
        processed = self.transforms(sample)
        image = processed["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = sliding_window_inference(
                image, PATCH_SIZE, 1, lambda x: torch.sigmoid(self.model(x)), overlap=0.25
            )
        pred_binary = (probs > 0.5).cpu().numpy()[0]  # [ET, TC, WT]

        # Combine into one label map for visualization: 4=ET, 1=TC-only, 2=WT-only
        et, tc, wt = pred_binary[0], pred_binary[1], pred_binary[2]
        combined = np.zeros_like(et, dtype=np.uint8)
        combined[wt > 0] = 2
        combined[tc > 0] = 1
        combined[et > 0] = 4

        # Save as NIfTI using the original T1 scan's affine, so it's spatially correct
        original_t1 = nib.load(t1_path)
        out_img = nib.Nifti1Image(combined, affine=original_t1.affine)
        nib.save(out_img, output_mask_path)

        return {
            "enhancing_tumor_volume_ml": round(float(et.sum()) * VOXEL_VOLUME_ML, 2),
            "tumor_core_volume_ml": round(float(tc.sum()) * VOXEL_VOLUME_ML, 2),
            "whole_tumor_volume_ml": round(float(wt.sum()) * VOXEL_VOLUME_ML, 2),
            "mask_path": str(output_mask_path),
        }

# Lazy singleton: the model loads on first real use, not at import time.
# This matters for testing — CI has no GPU and no model weights committed to
# the repo, so tests need to be able to swap this out entirely without ever
# triggering a real load.
import functools

@functools.lru_cache(maxsize=1)
def get_segmentation_service():
    return SegmentationService()