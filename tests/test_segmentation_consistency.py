import numpy as np
import pytest

from app.inference_service import regions_to_labelmap


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
