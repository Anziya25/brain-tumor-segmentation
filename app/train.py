import sys
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandShiftIntensityd,
)
from torch.optim import AdamW

from app.checkpoint_selection import should_save_checkpoint
from app.data_splits import create_or_load_splits
from app.device_utils import autocast_context, make_grad_scaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "unetr-brats"
sys.path.insert(0, str(MODEL_DIR))

from app.custom_transforms import ConvertToBratsMultiChanneld
from app.datalist import build_datalist
from inference import load_model
from app.preprocessing import get_preprocessing_transforms

# --- Config: START SMALL for the first run ---
NUM_PATIENTS = 60       # 32 train + 8 validation + 20 held-out test
VAL_SPLIT = 8
PATCH_SIZE = (128, 128, 128)
EPOCHS = 15              # smoke test; raise to 15-20 once this works
LR = 1e-5
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "finetuned_unetr.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data ---
full_list = build_datalist(limit=NUM_PATIENTS)
splits = create_or_load_splits(
    full_list, PROJECT_ROOT / "models" / "patient_splits.json",
    train_count=NUM_PATIENTS - VAL_SPLIT - 20, val_count=VAL_SPLIT, test_count=20,
)
train_files = splits["train"]
val_files = splits["val"]
print(f"Train patients: {len(train_files)}, Val patients: {len(val_files)}")

base_transforms = get_preprocessing_transforms()

train_transforms = Compose([
    base_transforms,
    RandCropByPosNegLabeld(
        keys=["image", "seg"], label_key="seg",
        spatial_size=PATCH_SIZE, pos=1, neg=1, num_samples=1,
    ),
    ConvertToBratsMultiChanneld(keys=["seg"]),
    RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=0),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
])

val_transforms = Compose([
    base_transforms,
    ConvertToBratsMultiChanneld(keys=["seg"]),
])

train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

# --- Model ---
model, metadata = load_model(str(MODEL_DIR / "pytorch_model.bin"))
model.to(device)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
loss_fn = DiceLoss(sigmoid=True, smooth_nr=0, smooth_dr=1e-5, squared_pred=True)
scaler = make_grad_scaler(device)
dice_metric = DiceMetric(include_background=True, reduction="mean_batch")

best_val_dice = float("-inf")
print("Starting fine-tune; checkpoint selection uses this run's validation Dice.")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        images = batch["image"].to(device)
        targets = batch["seg"].to(device)

        optimizer.zero_grad()
        with autocast_context(device):
            outputs = model(images)
            loss = loss_fn(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / max(len(train_loader), 1)

    # --- Validation ---
    model.eval()
    dice_metric.reset()
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            targets = batch["seg"].to(device)
            preds = sliding_window_inference(
                images, PATCH_SIZE, 1, lambda x: torch.sigmoid(model(x)), overlap=0.25
            )
            preds_bin = (preds > 0.5).float()
            dice_metric(y_pred=preds_bin, y=targets)

    val_dice = dice_metric.aggregate()
    mean_val_dice = val_dice.mean().item()
    print(f"Epoch {epoch+1}/{EPOCHS} - train loss: {avg_loss:.4f} - "
          f"val Dice (ET,TC,WT): {val_dice.tolist()} - mean: {mean_val_dice:.4f}")

    if should_save_checkpoint(mean_val_dice, best_val_dice):
        best_val_dice = mean_val_dice
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "val_dice_mean": best_val_dice}, CHECKPOINT_PATH)
        print(f"  -> New best model saved (val_dice={best_val_dice:.4f})")

print("Fine-tuning complete. Best val Dice:", best_val_dice)