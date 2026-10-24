# Exploratory evaluation: 60 BraTS2020 cases

> **Not an independent held-out test.** The original training/validation patient IDs for the fine-tuned checkpoint are unknown. The original pretrained model's training-data overlap is also not established. These measurements must not be advertised as unbiased generalization accuracy or clinical validation.

## Scope and provenance

- **Date:** 2026-10-24 (user-provided run summary, UTC timestamp `2026-10-24T11:29:06.667454+00:00`).
- **Dataset:** 60 complete cases `BraTS20_Training_001` through `BraTS20_Training_060`, selected by `app.datalist.build_datalist(limit=60)` from the user's local BraTS2020 training data. The original dataset distribution/license and file hashes have not been captured in this report.
- **Inputs:** T1, T1ce, T2, FLAIR and provided ground-truth segmentations; MONAI orientation to RAS, 1 mm target spacing, percentile clipping and nonzero intensity normalization.
- **Model:** locally supplied `models/unetr-brats/pytorch_model.bin` and `models/finetuned_unetr.pth`; 19,327,907 parameters. Local loader reported pretrained epoch 469 and recorded validation Dice 0.8362; fine-tuned checkpoint recorded validation Dice 0.892569. These two historical validation values are **not directly comparable** without matching protocols and cohorts.
- **Inference:** CUDA on NVIDIA RTX 2000 Ada (8 GB), 128³ sliding-window ROI, batch size 1, overlap 0.25, sigmoid, threshold 0.5, and postprocessing to enforce ET ⊆ TC ⊆ WT. Full-volume output was assembled on CPU.
- **Evaluation:** Dice calculated from binary region overlap; MONAI `HausdorffDistanceMetric(percentile=95)` with `spacing=(1,1,1)` supplied to the metric. Verify the *actual post-transform affine spacing* before treating HD95 as confirmed physical millimetres. The evaluation is distinct from `app/evaluate.py`, which is designed for a verified 20-patient held-out split.
- **Provenance gaps:** no original `patient_splits.json`, no training logs or notebook, no recorded dataset/checkpoint SHA-256 values, and no independent held-out cohort verification.

## Summary of user-provided evaluation results

All 60 patients returned finite Dice and HD95 values for each region. The following means and **population** standard deviations come from the user's generated `summary.json`; they have not been independently rerun here.

| Region | Dice mean ± SD | HD95 mean ± SD (nominal mm) | Valid Dice / HD95 |
| --- | ---: | ---: | ---: |
| Enhancing tumor (ET) | 0.8138 ± 0.0883 | 2.855 ± 3.047 | 60 / 60 |
| Tumor core (TC) | 0.8971 ± 0.0649 | 3.642 ± 3.267 | 60 / 60 |
| Whole tumor (WT) | 0.8807 ± 0.0766 | 5.998 ± 6.903 | 60 / 60 |

**Macro-average of three regional mean Dice scores:** 0.8638 (exploratory cohort only). Do not confuse this with the checkpoint's reported validation Dice of 0.8926.

## Case-level observations

Selected from the user-provided `patient_metrics.csv` contents:

| Patient | ET Dice | TC Dice | WT Dice | Reason to inspect |
| --- | ---: | ---: | ---: | --- |
| BraTS20_Training_033 | 0.9069 | 0.9585 | 0.9551 | Strong overall overlap |
| BraTS20_Training_018 | 0.6418 | 0.6411 | 0.7344 | Low overlap across regions |
| BraTS20_Training_028 | 0.5245 | 0.8230 | 0.8519 | Lowest ET Dice |
| BraTS20_Training_046 | 0.7104 | 0.7501 | 0.5925 | Lowest WT Dice |
| BraTS20_Training_017 | 0.8373 | 0.8824 | 0.9139 | WT HD95 49.05 nominal mm, largest reported WT boundary error |
| BraTS20_Training_042 | 0.8777 | 0.9617 | 0.9551 | Strong TC and WT overlap |

The single-patient qualitative comparison of `BraTS20_Training_001` showed similar tumor location and structure between ground truth and prediction, but one selected slice is not a full-volume error audit.

## Interpretation and limitations

- TC has the highest mean Dice; WT has the highest mean HD95 and largest reported HD95 variation. High Dice does not exclude distant boundary outliers.
- The missing original split manifest means training/test overlap **cannot be ruled out**. Do not call this a test-set result, a held-out evaluation, or a generalization benchmark.
- The pretrained model may also have encountered related BraTS cases during its original training.
- Reported runtime per patient includes the evaluation script's timed preprocessing/inference/metric steps, but excludes initial model loading; do not equate it with API end-to-end latency.
- Verify the affine voxel spacing after MONAI preprocessing, the segmentation geometry, and MONAI HD95 empty-mask conventions before publication of physical-distance conclusions.
- This is research/portfolio work, **not a medical device** or clinically validated method.

## Reproduction and evidence to retain locally

The evaluation was run using a separate `evaluate_all_60.py` script with outputs:

- `predictions/exploratory_60/patient_metrics.csv` — patient-level Dice, HD95, predicted/ground-truth voxel counts, and runtime.
- `predictions/exploratory_60/summary.json` — aggregate means, SDs, valid counts, and the exploratory-only warning.
- `predictions/validation_case_001/comparison.png` and `predicted_mask.nii.gz` — one patient visualization and predicted label map.

These files were supplied or reported by the user from their **local** run; the full CSV, JSON, checkpoint and medical images have **not** been committed to this repository. Before publishing reproducible figures, preserve a versioned copy of the evaluation script and record the exact source revision, environment, dataset provenance, hashes and preprocessing affine spacing. Only publish image examples when data permissions permit.

## Next validation work

1. Archive the exact exploratory evaluation script and inspect metric spacing and image affine.
2. Generate Dice/HD95 distributions and inspect low-overlap and high-HD95 cases using the original local CSV and authorized scans.
3. Recover training split provenance if possible; otherwise train and evaluate a **new, documented patient-level split** with no overlap, and record its hashes.
4. Re-run model validation on an independent held-out cohort before claiming generalization.
