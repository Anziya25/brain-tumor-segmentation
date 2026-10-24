"""Deterministic patient-level splits with an auditable JSON manifest."""

import json
from pathlib import Path


def patient_id(entry):
    """BraTS patient identity from the modality's parent directory."""
    return Path(entry["t1"]).parent.name


def create_or_load_splits(entries, manifest_path, train_count=32, val_count=8, test_count=20):
    """Create a manifest once; reject dataset drift and overlapping patient IDs."""
    required = train_count + val_count + test_count
    ids = [patient_id(entry) for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate patient IDs in dataset")
    if len(ids) < required:
        raise ValueError(f"Need at least {required} patients, found {len(ids)}")

    # Select a fixed, sorted cohort so adding patients after the first 60
    # requires explicit manifest regeneration rather than silent reassignment.
    selected = sorted(ids)[:required]
    expected = {
        "train": selected[:train_count],
        "val": selected[train_count:train_count + val_count],
        "test": selected[train_count + val_count:],
    }
    path = Path(manifest_path)
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(
                "Split manifest differs from current patient cohort; "
                "review dataset changes before regenerating it"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")

    groups = [set(expected[name]) for name in ("train", "val", "test")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Patient leakage across dataset splits")

    by_id = {patient_id(entry): entry for entry in entries}
    return {name: [by_id[pid] for pid in expected[name]] for name in expected}
