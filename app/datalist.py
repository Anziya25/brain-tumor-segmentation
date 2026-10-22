import glob
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "brats20-dataset-training-validation" / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"

def build_datalist(data_dir=None, limit=None):
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    patient_dirs = sorted(glob.glob(os.path.join(str(data_dir), "BraTS20_Training_*")))
    if limit:
        patient_dirs = patient_dirs[:limit]

    datalist = []
    for pdir in patient_dirs:
        pid = os.path.basename(pdir)
        entry = {
            "t1": os.path.join(pdir, f"{pid}_t1.nii"),
            "t1ce": os.path.join(pdir, f"{pid}_t1ce.nii"),
            "t2": os.path.join(pdir, f"{pid}_t2.nii"),
            "flair": os.path.join(pdir, f"{pid}_flair.nii"),
            "seg": os.path.join(pdir, f"{pid}_seg.nii"),
        }
        if all(os.path.exists(v) for v in entry.values()):
            datalist.append(entry)
        else:
            print(f"Skipping {pid}, missing one or more files")

    print(f"Built datalist with {len(datalist)} patients (searched: {data_dir})")
    return datalist


if __name__ == "__main__":
    dl = build_datalist(limit=15)
    print(dl[0])