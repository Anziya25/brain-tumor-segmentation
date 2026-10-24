import math

import pytest

from app.checkpoint_selection import should_save_checkpoint


def test_first_finite_epoch_is_saved():
    assert should_save_checkpoint(0.42, float("-inf"))


def test_higher_score_is_saved():
    assert should_save_checkpoint(0.63, 0.60)


def test_lower_or_equal_score_is_not_saved():
    assert not should_save_checkpoint(0.60, 0.60)
    assert not should_save_checkpoint(0.59, 0.60)


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf, -0.1, 1.1])
def test_invalid_validation_score_is_rejected(score):
    with pytest.raises(ValueError, match="Validation Dice"):
        should_save_checkpoint(score, float("-inf"))
