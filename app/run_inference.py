import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "unetr-brats"
sys.path.insert(0, str(MODEL_DIR))  # lets us import their inference.py

from inference import load_model, predict  # from the downloaded repo
from datalist import build_datalist
from preprocessing import get_preprocessing_transforms

# 1. Load the pretrained model — pass the exact weights file path,
# matching the repo's own tested usage (not just the folder)
model, metadata = load_model(str(MODEL_DIR / "pytorch_model.bin"))
print(f"Model's own reported validation Dice: {metadata.get('val_dice_mean', 'n/a')}")

# 2. Preprocess one patient
datalist = build_datalist(limit=1)
transforms = get_preprocessing_transforms()
sample = transforms(datalist[0])

image = sample["image"].unsqueeze(0)  # add batch dim -> (1, 4, D, H, W)
seg = sample["seg"]                   # (1, D, H, W), raw labels 0/1/2/4

# 3. Run inference (sliding window handled internally by predict())
probabilities = predict(model, image)          # (1, 3, D, H, W)
pred_binary = (probabilities > 0.5).cpu().numpy()[0]  # order: [ET, TC, WT]

# 4. Build ground-truth binary masks matching the model's output convention
seg_np = seg.squeeze(0).numpy()
gt_et = (seg_np == 4).astype(np.uint8)
gt_tc = np.isin(seg_np, [1, 4]).astype(np.uint8)
gt_wt = np.isin(seg_np, [1, 2, 4]).astype(np.uint8)

def dice_score(pred, gt, eps=1e-6):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    intersection = (pred * gt).sum()
    return (2. * intersection + eps) / (pred.sum() + gt.sum() + eps)

print("ET Dice:", dice_score(pred_binary[0], gt_et))
print("TC Dice:", dice_score(pred_binary[1], gt_tc))
print("WT Dice:", dice_score(pred_binary[2], gt_wt))