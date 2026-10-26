"""Audit fresh training/evaluation provenance without modifying experiment artifacts.

Run: python -m app.audit_fresh
Writes models/reproducibility_audit_fresh.json (no patient identifiers).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from app.reproducibility import sha256_file

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
CHECKPOINT = MODELS / "finetuned_unetr_fresh.pth"
RUN = MODELS / "finetuned_unetr_fresh.run.json"
SPLITS = MODELS / "patient_splits_fresh.json"
BASELINE = MODELS / "unetr-brats" / "pytorch_model.bin"
EVALUATION = MODELS / "evaluation_report_fresh.json"
OUTPUT = MODELS / "reproducibility_audit_fresh.json"


def main():
    for path in (CHECKPOINT, RUN, SPLITS, BASELINE, EVALUATION):
        if not path.is_file():
            raise FileNotFoundError(f"Missing experiment artifact: {path}")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {OUTPUT}")

    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    embedded = checkpoint.get("run_metadata")
    run = json.loads(RUN.read_text(encoding="utf-8"))
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    if not isinstance(embedded, dict) or not isinstance(run, dict):
        raise ValueError("Missing run metadata")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("Missing model weights in checkpoint")

    split_counts = {key: len(splits[key]) for key in ("train", "val", "test")}
    ids = [pid for key in ("train", "val", "test") for pid in splits[key]]
    hashes = {
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "split_manifest_sha256": sha256_file(SPLITS),
        "pretrained_weights_sha256": sha256_file(BASELINE),
        "evaluation_report_sha256": sha256_file(EVALUATION),
    }
    provenance = evaluation.get("provenance", {})
    checks = {
        "expected_split_counts": split_counts == {"train": 258, "val": 55, "test": 55},
        "unique_patient_ids": len(ids) == len(set(ids)) == 368,
        "run_metadata_matches_checkpoint": json.loads(json.dumps(embedded, default=str)) == run,
        "checkpoint_split_hash_matches": embedded.get("split_manifest_sha256") == hashes["split_manifest_sha256"],
        "checkpoint_pretrained_hash_matches": embedded.get("pretrained_weights_sha256") == hashes["pretrained_weights_sha256"],
        "evaluation_split_hash_matches": provenance.get("split_manifest_sha256") == hashes["split_manifest_sha256"],
        "evaluation_pretrained_hash_matches": provenance.get("baseline_weights_sha256") == hashes["pretrained_weights_sha256"],
        "evaluation_checkpoint_hash_matches": provenance.get("finetuned_checkpoint_sha256") == hashes["checkpoint_sha256"],
        "evaluation_test_count_matches": provenance.get("held_out_patients") == 55,
        "seed_is_42": embedded.get("seed") == 42,
        "epochs_configured_15": embedded.get("config", {}).get("epochs") == 15,
        "best_epoch_is_15": checkpoint.get("epoch") == 15,
    }
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "split_counts": split_counts,
        "best_epoch": checkpoint.get("epoch"),
        "best_validation_dice": float(checkpoint["val_dice_mean"]),
        "training_metadata": run,
        "artifact_hashes": hashes,
        "limitations": [
            "Training stdout/epoch-by-epoch logs were not persisted by the original training script.",
            "Seeds and deterministic settings do not guarantee bitwise-identical results across hardware or dependencies.",
            "Potential patient overlap with upstream pretraining is not independently established.",
            "Hashes verify consistency of available artifacts, not authenticity of the underlying data.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print("Fresh experiment reproducibility audit:", result["status"])
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"Checkpoint SHA-256: {hashes['checkpoint_sha256']}")
    print(f"Audit report: {OUTPUT}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
