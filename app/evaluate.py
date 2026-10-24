
import gc
import sys
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Compose

from app.evaluation_reporting import write_evaluation_report
from app.inference_service import enforce_region_nesting
from app.reproducibility import git_commit, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "unetr-brats"
BASELINE_WEIGHTS = MODEL_DIR / "pytorch_model.bin"
FT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "finetuned_unetr.pth"

PATCH_SIZE = (128, 128, 128)
SW_BATCH_SIZE = 1
REGIONS = ["ET", "TC", "WT"]


def evaluate(model, loader, device):
    dice_metric = DiceMetric(
        include_background=True,
        reduction="mean_batch",
        get_not_nans=True,
    )
    hd_metric = HausdorffDistanceMetric(
        include_background=True,
        percentile=95,
        reduction="mean_batch",
        get_not_nans=True,
    )

    model.eval()

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device)
            targets = batch["seg"].to(device)

            outputs = sliding_window_inference(
                inputs=images,
                roi_size=PATCH_SIZE,
                sw_batch_size=SW_BATCH_SIZE,
                predictor=lambda x: torch.sigmoid(model(x)),
                overlap=0.25,
            )

            predictions = enforce_region_nesting((outputs > 0.5).float())

            dice_metric(y_pred=predictions, y=targets)
            hd_metric(y_pred=predictions, y=targets)

            print(
                f"\rEvaluated {batch_index}/{len(loader)} patients",
                end="",
                flush=True,
            )

    print()

    dice_values, dice_counts = dice_metric.aggregate()
    hd_values, hd_counts = hd_metric.aggregate()

    return {
        "dice": dice_values.detach().cpu(),
        "dice_counts": dice_counts.detach().cpu(),
        "hd95": hd_values.detach().cpu(),
        "hd95_counts": hd_counts.detach().cpu(),
    }


def print_results(baseline, finetuned):
    print("\n" + "=" * 100)
    print(
        f"{'Region':<8}"
        f"{'Baseline Dice':<17}"
        f"{'Fine-tuned Dice':<19}"
        f"{'Baseline HD95':<18}"
        f"{'Fine-tuned HD95':<18}"
    )
    print("-" * 100)

    for i, region in enumerate(REGIONS):
        print(
            f"{region:<8}"
            f"{baseline['dice'][i].item():<17.4f}"
            f"{finetuned['dice'][i].item():<19.4f}"
            f"{baseline['hd95'][i].item():<18.2f}"
            f"{finetuned['hd95'][i].item():<18.2f}"
        )

    print("-" * 100)

    for label, results in [
        ("Baseline", baseline),
        ("Fine-tuned", finetuned),
    ]:
        dice = results["dice"]
        dice_counts = results["dice_counts"]
        hd95 = results["hd95"]
        hd95_counts = results["hd95_counts"]

        valid_dice = torch.isfinite(dice) & (dice_counts > 0)
        valid_hd95 = torch.isfinite(hd95) & (hd95_counts > 0)

        mean_dice = (
            dice[valid_dice].mean().item()
            if valid_dice.any()
            else float("nan")
        )
        mean_hd95 = (
            hd95[valid_hd95].mean().item()
            if valid_hd95.any()
            else float("nan")
        )

        print(f"{label} mean Dice: {mean_dice:.4f}")
        print(f"{label} mean HD95: {mean_hd95:.2f} mm")
        print(f"{label} valid Dice region counts: {dice_counts.tolist()}")
        print(f"{label} valid HD95 region counts: {hd95_counts.tolist()}")

    print("=" * 100)
    print(
        "Interpretation note: inspect valid counts and NaN values. "
        "Empty-region handling can affect metric interpretation."
    )


def load_finetuned_model(load_model, device):
    model, _ = load_model(str(BASELINE_WEIGHTS))

    # Only load checkpoints produced by your trusted training pipeline.
    checkpoint = torch.load(
        FT_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint must contain a 'model_state_dict' entry."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint

    model.to(device)
    model.eval()
    return model


def main():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    if not BASELINE_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"Baseline weights not found: {BASELINE_WEIGHTS}"
        )

    if not FT_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found: {FT_CHECKPOINT_PATH}"
        )

    # Import model utilities only when evaluation actually runs.
    sys.path.insert(0, str(MODEL_DIR))

    from inference import load_model

    from app.custom_transforms import ConvertToBratsMultiChanneld
    from app.datalist import build_datalist
    from app.data_splits import create_or_load_splits
    from app.preprocessing import get_preprocessing_transforms

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Evaluation device: {device}")

    full_list = build_datalist(limit=60)

    if len(full_list) != 60:
        raise RuntimeError(
            f"Expected 60 patients, but build_datalist returned "
            f"{len(full_list)}."
        )

    splits = create_or_load_splits(
        full_list, PROJECT_ROOT / "models" / "patient_splits.json",
        train_count=32, val_count=8, test_count=20,
    )
    held_out_files = splits["test"]

    print(f"Held-out evaluation set: {len(held_out_files)} patients")
    print("Verify patient IDs against the training and validation splits.")

    eval_transforms = Compose(
        [
            get_preprocessing_transforms(),
            ConvertToBratsMultiChanneld(keys=["seg"]),
        ]
    )

    eval_ds = Dataset(
        data=held_out_files,
        transform=eval_transforms,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print("\nLoading pretrained baseline...")
    baseline_model, _ = load_model(str(BASELINE_WEIGHTS))
    baseline_model.to(device)
    baseline_model.eval()

    print("Evaluating pretrained baseline...")
    baseline_results = evaluate(
        baseline_model,
        eval_loader,
        device,
    )

    del baseline_model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\nLoading fine-tuned model...")
    finetuned_model = load_finetuned_model(load_model, device)

    print("Evaluating fine-tuned model...")
    finetuned_results = evaluate(
        finetuned_model,
        eval_loader,
        device,
    )

    print_results(baseline_results, finetuned_results)

    report_path = PROJECT_ROOT / "models" / "evaluation_report.json"
    write_evaluation_report(
        report_path,
        baseline=baseline_results,
        finetuned=finetuned_results,
        provenance={
            "git_commit": git_commit(PROJECT_ROOT),
            "device": str(device),
            "held_out_patients": len(held_out_files),
            "split_manifest_sha256": sha256_file(PROJECT_ROOT / "models" / "patient_splits.json"),
            "baseline_weights_sha256": sha256_file(BASELINE_WEIGHTS),
            "finetuned_checkpoint_sha256": sha256_file(FT_CHECKPOINT_PATH),
            "threshold": 0.5,
            "sliding_window_overlap": 0.25,
            "patch_size": list(PATCH_SIZE),
            "postprocessing": "threshold_0.5_then_union_nesting_ET_subset_TC_subset_WT",
        },
    )
    print(f"Machine-readable evaluation report saved: {report_path}")


if __name__ == "__main__":
    main()
