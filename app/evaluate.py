import sys
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Compose

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "unetr-brats"
FT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "finetuned_unetr.pth"
sys.path.insert(0, str(MODEL_DIR))

from custom_transforms import ConvertToBratsMultiChanneld
from datalist import build_datalist
from inference import load_model
from preprocessing import get_preprocessing_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (128, 128, 128)

# --- Held-out patients: NOT used in Phase 4 training or validation (patients 1-40) ---
full_list = build_datalist(limit=60)
held_out_files = full_list[-20:]  # patients 41-60
print(f"Held-out evaluation set: {len(held_out_files)} patients")

eval_transforms = Compose([
    get_preprocessing_transforms(),
    ConvertToBratsMultiChanneld(keys=["seg"]),
])

eval_ds = Dataset(data=held_out_files, transform=eval_transforms)
eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=0)

# --- Load both models on the same architecture ---
baseline_model, metadata = load_model(str(MODEL_DIR / "pytorch_model.bin"))
baseline_model.to(device)
baseline_model.eval()

finetuned_model, _ = load_model(str(MODEL_DIR / "pytorch_model.bin"))
ft_checkpoint = torch.load(FT_CHECKPOINT_PATH, map_location=device)
finetuned_model.load_state_dict(ft_checkpoint["model_state_dict"])
finetuned_model.to(device)
finetuned_model.eval()

def evaluate(model, loader):
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")
    dice_metric.reset()
    hd_metric.reset()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["seg"].to(device)
            preds = sliding_window_inference(
                images, PATCH_SIZE, 1, lambda x: torch.sigmoid(model(x)), overlap=0.25
            )
            preds_bin = (preds > 0.5).float()
            dice_metric(y_pred=preds_bin, y=targets)
            hd_metric(y_pred=preds_bin, y=targets)

    return dice_metric.aggregate(), hd_metric.aggregate()

print("\nEvaluating pretrained baseline...")
baseline_dice, baseline_hd = evaluate(baseline_model, eval_loader)

print("Evaluating fine-tuned model...")
finetuned_dice, finetuned_hd = evaluate(finetuned_model, eval_loader)

regions = ["ET", "TC", "WT"]
print("\n" + "="*70)
print(f"{'Region':<8}{'Baseline Dice':<16}{'Fine-tuned Dice':<18}{'Baseline HD95':<16}{'Fine-tuned HD95':<16}")
print("-"*70)
for i, r in enumerate(regions):
    print(f"{r:<8}{baseline_dice[i].item():<16.4f}{finetuned_dice[i].item():<18.4f}{baseline_hd[i].item():<16.2f}{finetuned_hd[i].item():<16.2f}")
print("="*70)
print(f"{'Mean':<8}{baseline_dice.mean().item():<16.4f}{finetuned_dice.mean().item():<18.4f}{baseline_hd.mean().item():<16.2f}{finetuned_hd.mean().item():<16.2f}")