# Experiment reproducibility record

This document distinguishes **verified code behavior** from **experiment evidence not yet supplied**. It is not a claim that the model's original training run has been reproduced.

## 1. Experiment identity

| Field | Recorded value / status |
| --- | --- |
| Repository | `Anziya25/brain-tumor-segmentation` |
| Working branch | `fix/segmentation-consistency` |
| Model family | Pretrained UNETR, fine-tuned using PyTorch/MONAI |
| Dataset family | BraTS 2020 training cases |
| Pretrained model source and revision | **TODO — verify source URL, revision, and license** |
| Exact data release and provenance | **TODO — record original archive, version, and permitted usage** |
| Original experiment commit SHA | **TODO — record SHA used for the actual training run** |
| Trained checkpoint hash | **TODO — SHA-256 of `finetuned_unetr.pth`** |
| Split manifest hash | **TODO — SHA-256 of `patient_splits.json`** |
| Training hardware and CUDA/driver | **TODO — capture from actual training environment** |
| Actual random seed(s) | **Not explicitly set in current `app/train.py`** |
| Model accuracy results | **TODO — provide evaluated outputs, not estimated values** |

## 2. Dataset and leakage prevention

`app/datalist.py` discovers complete BraTS-style cases with T1, T1ce, T2, FLAIR, and segmentation labels. Its default path is:

```text
data/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/
```

`app/train.py` requests up to 60 cases. `app/data_splits.py` sorts patient directory IDs and deterministically partitions the first 60 into **32 train, 8 validation, 20 held-out test**. This is a **sorted partition, not a randomized split**. The generated manifest is `models/patient_splits.json`. Reusing the same manifest and dataset is necessary for comparable results. The code rejects duplicate patient IDs, overlapping groups, and manifest/cohort mismatches.

**Record before claiming reproducibility:** dataset provenance, selected patient IDs or an appropriately privacy-preserving manifest, split manifest SHA-256, and whether any patients were excluded.

## 3. Preprocessing and target encoding

The training pipeline in `app/preprocessing.py`:

1. Loads all four MRI modalities and the ground-truth segmentation.
2. Ensures channel-first layout and RAS orientation.
3. Resamples to **1 × 1 × 1 mm** spacing (bilinear images, nearest-neighbor segmentation).
4. Clips modality intensities at the **0.5th and 99.5th percentiles**.
5. Normalizes nonzero intensities, channel-wise, and concatenates four modalities.
6. Converts BraTS labels to channels **ET = 4**, **TC = 1 or 4**, **WT = 1, 2 or 4**.

Inference uses the corresponding image-only transforms and requires already co-registered input volumes. It does not perform registration.

## 4. Fine-tuning configuration in source

| Parameter | Current `app/train.py` value |
| --- | --- |
| Selected patients | Up to 60; split 32/8/20 |
| Epochs | 15 |
| Patch size | 128 × 128 × 128 |
| Training batch size | 1 |
| Optimizer | AdamW |
| Learning rate | 0.00001 |
| Weight decay | 0.00001 |
| Loss | DiceLoss with sigmoid |
| Random crop | `RandCropByPosNegLabeld`, pos=1, neg=1, one sample |
| Augmentation | Spatial flip (axis 0, p=0.5), intensity shift (p=0.5, offset=0.1) |
| Validation | Sliding-window inference, overlap 0.25, threshold 0.5 |
| Selection | Highest finite mean validation Dice **within the current run** |
| Output checkpoint | `models/finetuned_unetr.pth` |

CUDA training uses mixed precision; CPU training does not. The current script does **not** explicitly initialize Python, NumPy, PyTorch, MONAI, or DataLoader random seeds, and does not save a full optimizer/scheduler/RNG state. Therefore exact deterministic retraining is **not yet guaranteed**.

The training module executes training at import time; run it deliberately with `python -m app.train`, not as a general-purpose import.

## 5. Evaluation protocol

`python -m app.evaluate` loads the persisted 20-patient held-out split, compares the pretrained baseline with the fine-tuned checkpoint, and prints per-region **Dice** and **95th-percentile Hausdorff distance (HD95)**, together with valid-count information. The evaluation script does not currently write a machine-readable experiment report.

| Metric | ET | TC | WT |
| --- | --- | --- | --- |
| Baseline Dice | Pending verified run | Pending | Pending |
| Fine-tuned Dice | Pending verified run | Pending | Pending |
| Baseline HD95 | Pending verified run | Pending | Pending |
| Fine-tuned HD95 | Pending verified run | Pending | Pending |

**Do not substitute** the checkpoint's `val_dice_mean` for a held-out test result. Report empty-region handling, valid counts, cohort size, and units alongside metrics.

## 6. Runtime and inference checks

The FastAPI service uses four NIfTI inputs and validates their physical-space alignment. The inference pipeline thresholds three region channels, enforces **ET ⊆ TC ⊆ WT**, saves a BraTS-style NIfTI mask on the **preprocessed image grid**, and computes ET/TC/WT volumes using the saved affine's voxel volume.

Automated CI covers Ruff lint, synthetic/unit tests, Docker CPU build, API import, and Compose validation. These checks **do not** establish successful real-weight inference, GPU deployment, or medical performance.

## 7. Benchmark reporting rules

`app/benchmarking.py` measures **model forward-pass only**, excluding preprocessing, model loading, NIfTI I/O, and API transfer. It synchronizes CUDA around timings and reports warmup count, repetitions, mean, median, minimum, and maximum.

For any future benchmark, record CPU/GPU model, RAM/VRAM, OS, Python/PyTorch/MONAI versions, input tensor shape, device, precision, warmups, repetitions, whether measurements include preprocessing/I/O, and raw timings. Do not compare unlike measurement scopes as an inference speedup.

## 8. Evidence capture checklist

- [ ] Verify original pretrained UNETR repository, revision, attribution, and license.
- [ ] Record BraTS dataset provenance, archive checksum, and authorized usage.
- [ ] Preserve the exact split manifest and its SHA-256 hash.
- [ ] Record actual training commit, Python/package versions, OS, GPU, and CUDA/driver.
- [ ] Capture the actual training command, logs, seeds (if any), and checkpoint hash.
- [ ] Run held-out evaluation; save full stdout and valid-region counts.
- [ ] Add a machine-readable metrics report and provenance metadata.
- [ ] Run a verified end-to-end NIfTI prediction with authorized, non-identifiable example data.
- [ ] Document any remaining sources of nondeterminism.
- [ ] Confirm CPU/GPU deployment separately before advertising real inference performance.

## 9. Useful local evidence commands

Run these in the actual experiment environment and retain the output alongside the run record:

```bash
git rev-parse HEAD
python --version
python -m pip freeze
python -c "import torch, monai; print('torch:', torch.__version__, 'monai:', monai.__version__, 'CUDA:', torch.version.cuda, 'GPU available:', torch.cuda.is_available())"
python -m app.evaluate
```

To hash the split manifest and checkpoint, use this cross-platform Python snippet:

```python
from pathlib import Path
from hashlib import sha256

for path in (Path("models/patient_splits.json"), Path("models/finetuned_unetr.pth")):
    if not path.is_file():
        print(f"MISSING: {path}")
        continue
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    print(f"{path}: {digest.hexdigest()}")
```

**Privacy note:** review logs and filenames for sensitive or patient-identifying information before sharing them publicly.
