"""Evaluate the fresh UNETR checkpoint on its reserved 55-patient test cohort.

Run: python -m app.evaluate_fresh
Does not modify training checkpoints or the legacy evaluation report.
"""
import gc
import json
import sys
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose

from app.data_splits import patient_id
from app.datalist import build_datalist
from app.evaluate import evaluate, print_results
from app.evaluation_reporting import write_evaluation_report
from app.reproducibility import git_commit, sha256_file
from app.custom_transforms import ConvertToBratsMultiChanneld
from app.preprocessing import get_preprocessing_transforms

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "unetr-brats"
BASELINE = MODEL_DIR / "pytorch_model.bin"
CHECKPOINT = ROOT / "models" / "finetuned_unetr_fresh.pth"
MANIFEST = ROOT / "models" / "patient_splits_fresh.json"
REPORT = ROOT / "models" / "evaluation_report_fresh.json"


def main():
    for path in (BASELINE, CHECKPOINT, MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    if REPORT.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {REPORT}")

    ids = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {"train": 258, "val": 55, "test": 55}
    if set(ids) != set(expected):
        raise ValueError("Manifest must contain train, val and test groups")
    for group, count in expected.items():
        if not isinstance(ids[group], list) or len(ids[group]) != count:
            raise ValueError(f"Expected {count} patient IDs in {group}")
    all_ids = [pid for group in expected for pid in ids[group]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Patient IDs overlap across groups")

    entries = build_datalist()
    by_id = {patient_id(entry): entry for entry in entries}
    if len(entries) != 368 or len(by_id) != 368 or set(by_id) != set(all_ids):
        raise ValueError("Current dataset differs from the saved 368-patient manifest")

    # Training metadata contains torch.torch_version.TorchVersion; allow only this known type.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Unexpected fine-tuned checkpoint structure")
    metadata = checkpoint.get("run_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Fresh checkpoint is missing run metadata")
    if metadata.get("split_manifest_sha256") != sha256_file(MANIFEST):
        raise ValueError("Checkpoint was trained with a different patient split manifest")
    if metadata.get("pretrained_weights_sha256") != sha256_file(BASELINE):
        raise ValueError("Pretrained baseline weights differ from training provenance")

    test_files = [by_id[pid] for pid in ids["test"]]
    loader = DataLoader(
        Dataset(
            data=test_files,
            transform=Compose([
                get_preprocessing_transforms(),
                ConvertToBratsMultiChanneld(keys=["seg"]),
            ]),
        ),
        batch_size=1, shuffle=False, num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {len(test_files)} held-out test patients on {device}", flush=True)

    sys.path.insert(0, str(MODEL_DIR))
    from inference import load_model

    print("Evaluating pretrained baseline...", flush=True)
    baseline_model, _ = load_model(str(BASELINE))
    baseline_model.to(device)
    baseline_results = evaluate(baseline_model, loader, device)
    del baseline_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Evaluating freshly fine-tuned model...", flush=True)
    model, _ = load_model(str(BASELINE))
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    model.to(device)
    finetuned_results = evaluate(model, loader, device)
    del model
    print_results(baseline_results, finetuned_results)

    write_evaluation_report(
        REPORT,
        baseline=baseline_results,
        finetuned=finetuned_results,
        provenance={
            "git_commit": git_commit(ROOT),
            "device": str(device),
            "held_out_patients": len(test_files),
            "split_manifest_sha256": sha256_file(MANIFEST),
            "baseline_weights_sha256": sha256_file(BASELINE),
            "finetuned_checkpoint_sha256": sha256_file(CHECKPOINT),
            "threshold": 0.5,
            "sliding_window_overlap": 0.25,
            "patch_size": [128, 128, 128],
            "postprocessing": "threshold_0.5_then_union_nesting_ET_subset_TC_subset_WT",
            "note": "Test patients excluded from this fine-tuning run; upstream pretrained training overlap is unknown.",
        },
    )
    print(f"Fresh held-out evaluation report saved: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
