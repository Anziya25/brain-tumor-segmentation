import json

import pytest

from app.data_splits import create_or_load_splits, patient_id


def patients(n=60):
    return [{"t1": f"/dataset/BraTS20_Training_{i:03d}/t1.nii"} for i in range(n)]


def test_splits_are_disjoint_and_have_expected_sizes(tmp_path):
    splits = create_or_load_splits(patients(), tmp_path / "splits.json")
    ids = {name: {patient_id(x) for x in entries} for name, entries in splits.items()}
    assert [len(ids[name]) for name in ("train", "val", "test")] == [32, 8, 20]
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]


def test_manifest_is_stable_across_input_order(tmp_path):
    path = tmp_path / "splits.json"
    first = create_or_load_splits(patients(), path)
    second = create_or_load_splits(list(reversed(patients())), path)
    assert {k: [patient_id(x) for x in v] for k, v in first.items()} == {
        k: [patient_id(x) for x in v] for k, v in second.items()
    }


def test_duplicate_patient_is_rejected(tmp_path):
    rows = patients() + [patients()[0]]
    with pytest.raises(ValueError, match="Duplicate"):
        create_or_load_splits(rows, tmp_path / "splits.json")


def test_insufficient_patients_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="at least"):
        create_or_load_splits(patients(59), tmp_path / "splits.json")


def test_manifest_tampering_is_rejected(tmp_path):
    path = tmp_path / "splits.json"
    create_or_load_splits(patients(), path)
    data = json.loads(path.read_text())
    data["test"][0] = data["train"][0]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest differs"):
        create_or_load_splits(patients(), path)
