import nibabel as nib
import numpy as np

from monai.transforms import (
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

from app.custom_transforms import PercentileClipd


def get_preprocessing_transforms():
    return Compose([
        LoadImaged(keys=["t1", "t1ce", "t2", "flair", "seg"]),
        EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "seg"]),
        Orientationd(keys=["t1", "t1ce", "t2", "flair", "seg"], axcodes="RAS"),
        Spacingd(
            keys=["t1", "t1ce", "t2", "flair", "seg"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest"),
        ),
        PercentileClipd(keys=["t1", "t1ce", "t2", "flair"], lower=0.5, upper=99.5),
        NormalizeIntensityd(keys=["t1", "t1ce", "t2", "flair"], nonzero=True, channel_wise=True),
        ConcatItemsd(keys=["t1", "t1ce", "t2", "flair"], name="image"),
        EnsureTyped(keys=["image", "seg"]),
    ])
def get_inference_transforms():
    """Same pipeline as training, but without the seg/ground-truth key —
    real deployment only has the 4 raw MRI scans, no ground truth."""
    return Compose([
        LoadImaged(keys=["t1", "t1ce", "t2", "flair"]),
        EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair"]),
        Orientationd(keys=["t1", "t1ce", "t2", "flair"], axcodes="RAS"),
        Spacingd(keys=["t1", "t1ce", "t2", "flair"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        PercentileClipd(keys=["t1", "t1ce", "t2", "flair"], lower=0.5, upper=99.5),
        NormalizeIntensityd(keys=["t1", "t1ce", "t2", "flair"], nonzero=True, channel_wise=True),
        ConcatItemsd(keys=["t1", "t1ce", "t2", "flair"], name="image"),
        EnsureTyped(keys=["image"]),
    ])

def validate_inference_geometry(input_paths, tolerance_mm=1e-3):
    """Require four already co-registered, single-volume MRI modalities.

    This rejects mismatched voxel grids; it does not perform registration.
    The affine tolerance is in millimetres for corresponding voxel corners.
    """
    modalities = ("t1", "t1ce", "t2", "flair")
    reference_shape = None
    reference_affine = None

    for modality in modalities:
        try:
            image = nib.load(str(input_paths[modality]))
        except (OSError, ValueError) as exc:
            raise ValueError(f"{modality} is not a readable NIfTI image") from exc

        shape = image.shape
        affine = np.asarray(image.affine, dtype=np.float64)
        if len(shape) != 3 or any(size < 1 for size in shape):
            raise ValueError(f"{modality} must be a nonempty 3D MRI volume")
        if not np.all(np.isfinite(affine)):
            raise ValueError(f"{modality} contains a non-finite affine")
        if abs(np.linalg.det(affine[:3, :3])) < 1e-9:
            raise ValueError(f"{modality} contains a singular affine")

        if reference_shape is None:
            reference_shape = shape
            reference_affine = affine
            continue

        if shape != reference_shape:
            raise ValueError(
                f"{modality} shape {shape} differs from T1 {reference_shape}; "
                "register and resample modalities to a common grid first"
            )

        # Compare physical coordinates at all eight grid corners, including
        # orientation, spacing and translation differences.
        corners = np.array([
            [i, j, k, 1.0]
            for i in (0, shape[0] - 1)
            for j in (0, shape[1] - 1)
            for k in (0, shape[2] - 1)
        ]).T
        displacement = (affine @ corners - reference_affine @ corners)[:3]
        max_distance = np.linalg.norm(displacement, axis=0).max()
        if max_distance > tolerance_mm:
            raise ValueError(
                f"{modality} grid differs from T1 by up to "
                f"{max_distance:.3f} mm; register and resample first"
            )
