import json
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
from app.data_splits import patient_id
from app.device_utils import autocast_context, make_grad_scaler
from app.custom_transforms import ConvertToBratsMultiChanneld
from app.datalist import build_datalist
from app.preprocessing import get_preprocessing_transforms
from app.reproducibility import build_run_metadata, save_run_metadata, seed_training

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "unetr-brats"
sys.path.insert(0, str(MODEL_DIR))

from inference import load_model  # noqa: E402 - model package path set above

# --- Config: START SMALL for the first run ---
NUM_PATIENTS = 369      # 259 train + 55 validation + 55 held-out test
VAL_SPLIT = 55
PATCH_SIZE = (128, 128, 128)
EPOCHS = 15              # smoke test; raise to 15-20 once this works
LR = 1e-5
SEED = 42
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "finetuned_unetr_fresh.pth"
SPLIT_MANIFEST = PROJECT_ROOT / "models" / "patient_splits_fresh.json"

seed_training(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data ---
if not SPLIT_MANIFEST.is_file():
    raise FileNotFoundError(f"Run the split-creation step first: {SPLIT_MANIFEST}")
if CHECKPOINT_PATH.exists() or CHECKPOINT_PATH.with_suffix(".run.json").exists():
    raise FileExistsError(
        f"Fresh-run output already exists: {CHECKPOINT_PATH}; "
        "move it aside explicitly before starting another run"
    )

full_list = build_datalist()
by_id = {patient_id(entry): entry for entry in full_list}
if len(by_id) != len(full_list):
    raise ValueError("Duplicate patient IDs in dataset")
splits_ids = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
expected_counts = {"train": 259, "val": 55, "test": 55}
if set(splits_ids) != set(expected_counts):
    raise ValueError("Split manifest must have exactly train, val and test keys")
for group, count in expected_counts.items():
    if not isinstance(splits_ids[group], list) or len(splits_ids[group]) != count:
        raise ValueError(f"Invalid {group} split: expected {count} patient IDs")
all_ids = [pid for group in expected_counts for pid in splits_ids[group]]
if len(all_ids) != len(set(all_ids)):
    raise ValueError("Duplicate or overlapping patient IDs across splits")
if len(full_list) != NUM_PATIENTS or set(all_ids) != set(by_id):
    raise ValueError("Split manifest does not match the complete available dataset")
train_files = [by_id[pid] for pid in splits_ids["train"]]
val_files = [by_id[pid] for pid in splits_ids["val"]]
# The test cohort is intentionally excluded from training and checkpoint selection.
print(f"Train patients: {len(train_files)}, Val patients: {len(val_files)}")

run_metadata = build_run_metadata(
    seed=SEED,
    project_root=PROJECT_ROOT,
    split_manifest=SPLIT_MANIFEST,
    pretrained_weights=MODEL_DIR / "pytorch_model.bin",
    config={
        "num_patients": NUM_PATIENTS,
        "train_count": len(train_files),
        "val_count": len(val_files),
        "test_count": len(splits_ids["test"]),
        "patch_size": list(PATCH_SIZE),
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": 1e-5,
        "train_batch_size": 1,
        "validation_overlap": 0.25,
        "prediction_threshold": 0.5,
    },
    device=device,
)
save_run_metadata(run_metadata, CHECKPOINT_PATH.with_suffix(".run.json"))
print(f"Training seed: {SEED}; run metadata saved beside checkpoint.")

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
        torch.save({"model_state_dict": model.state_dict(), "val_dice_mean": best_val_dice, "run_metadata": run_metadata}, CHECKPOINT_PATH)
        print(f"  -> New best model saved (val_dice={best_val_dice:.4f})")

print("Fine-tuning complete. Best val Dice:", best_val_dice)