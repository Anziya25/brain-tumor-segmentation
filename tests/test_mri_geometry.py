import nibabel as nib
import numpy as np
import pytest

from app.preprocessing import validate_inference_geometry


def make_modalities(tmp_path, overrides=None):
    overrides = overrides or {}
    paths = {}
    for name in ("t1", "t1ce", "t2", "flair"):
        shape, affine = overrides.get(name, ((4, 5, 6), np.eye(4)))
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), path)
        paths[name] = path
    return paths


def test_matching_geometry_accepted(tmp_path):
    validate_inference_geometry(make_modalities(tmp_path))


def test_shape_mismatch_rejected(tmp_path):
    paths = make_modalities(tmp_path, {"t2": ((5, 5, 6), np.eye(4))})
    with pytest.raises(ValueError, match="shape"):
        validate_inference_geometry(paths)


@pytest.mark.parametrize("change", ["translation", "spacing", "orientation"])
def test_affine_mismatch_rejected(tmp_path, change):
    affine = np.eye(4)
    if change == "translation":
        affine[0, 3] = 2.0
    elif change == "spacing":
        affine[0, 0] = 2.0
    else:
        affine[0, 0] = -1.0
    paths = make_modalities(tmp_path, {"flair": ((4, 5, 6), affine)})
    with pytest.raises(ValueError, match="grid differs"):
        validate_inference_geometry(paths)


def test_four_dimensional_volume_rejected(tmp_path):
    paths = make_modalities(tmp_path, {"t1ce": ((4, 5, 6, 2), np.eye(4))})
    with pytest.raises(ValueError, match="3D"):
        validate_inference_geometry(paths)


def test_small_affine_rounding_difference_accepted(tmp_path):
    affine = np.eye(4)
    affine[0, 3] = 0.0001
    paths = make_modalities(tmp_path, {"t2": ((4, 5, 6), affine)})
    validate_inference_geometry(paths)
