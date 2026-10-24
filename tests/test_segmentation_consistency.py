import numpy as np
import pytest
import torch

from app.inference_service import enforce_region_nesting, regions_to_labelmap


def test_regions_are_nested_when_predictions_disagree():
    # ET-only, TC-only, WT-only and background voxels.
    pred = np.zeros((3, 2, 2, 2), dtype=np.uint8)
    pred[0, 0, 0, 0] = 1
    pred[1, 0, 0, 1] = 1
    pred[2, 0, 1, 0] = 1

    labels = regions_to_labelmap(pred)
    et = labels == 4
    tc = (labels == 1) | et
    wt = labels != 0

    assert labels[0, 0, 0] == 4
    assert labels[0, 0, 1] == 1
    assert labels[0, 1, 0] == 2
    assert labels[1, 1, 1] == 0
    assert np.all(~et | tc)
    assert np.all(~tc | wt)
    assert (et.sum(), tc.sum(), wt.sum()) == (1, 2, 3)


def test_consistent_predictions_are_unchanged():
    pred = np.ones((3, 2, 2, 2), dtype=np.uint8)
    labels = regions_to_labelmap(pred)
    assert np.all(labels == 4)


def test_invalid_prediction_shape_rejected():
    with pytest.raises(ValueError, match="Expected predictions"):
        regions_to_labelmap(np.zeros((2, 2, 2, 2), dtype=np.uint8))


def test_torch_postprocessing_matches_saved_labelmap():
    raw = torch.zeros((1, 3, 2, 2, 2), dtype=torch.float32)
    raw[0, 0, 0, 0, 0] = 1  # ET without TC/WT
    raw[0, 1, 0, 0, 1] = 1  # TC without WT
    raw[0, 2, 0, 1, 0] = 1  # WT only

    corrected = enforce_region_nesting(raw)
    labels = regions_to_labelmap(raw[0].numpy())
    expected = torch.from_numpy(
        np.stack((labels == 4, np.isin(labels, (1, 4)), labels != 0))
    ).float()
    assert torch.equal(corrected[0], expected)
    assert corrected.dtype == raw.dtype


def test_torch_postprocessing_accepts_unbatched_inputs():
    raw = torch.zeros((3, 2, 2, 2), dtype=torch.uint8)
    raw[0, 0, 0, 0] = 1
    nested = enforce_region_nesting(raw)
    assert nested.shape == raw.shape
    assert nested[:, 0, 0, 0].tolist() == [1, 1, 1]


def test_torch_postprocessing_rejects_invalid_channels():
    with pytest.raises(ValueError, match="Expected predictions"):
        enforce_region_nesting(torch.zeros((1, 2, 2, 2, 2)))
