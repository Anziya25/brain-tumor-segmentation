import json

import pytest
import torch

from app.evaluation_reporting import summarize_metrics, write_evaluation_report


def sample_metrics():
    return {
        "dice": torch.tensor([0.8, float("nan"), 0.9]),
        "dice_counts": torch.tensor([20, 0, 20]),
        "hd95": torch.tensor([4.0, float("inf"), 6.0]),
        "hd95_counts": torch.tensor([20, 0, 19]),
    }


def test_json_report_preserves_regions_counts_and_nulls(tmp_path):
    metrics = sample_metrics()
    path = tmp_path / "evaluation.json"
    write_evaluation_report(
        path, baseline=metrics, finetuned=metrics,
        provenance={"held_out_patients": 20},
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["regions"] == ["ET", "TC", "WT"]
    assert result["baseline"]["dice"]["ET"]["valid_count"] == 20
    assert result["baseline"]["dice"]["TC"]["value"] is None
    assert result["baseline"]["hd95"]["TC"]["value"] is None
    assert result["finetuned"]["hd95"]["WT"]["valid_count"] == 19
    assert result["provenance"]["held_out_patients"] == 20
    assert "NaN" not in path.read_text(encoding="utf-8")
    assert "Infinity" not in path.read_text(encoding="utf-8")


def test_nonfinite_metric_with_nonzero_count_is_json_null():
    metrics = sample_metrics()
    metrics["dice_counts"][1] = 1
    assert summarize_metrics(metrics)["dice"]["TC"] == {
        "value": None, "valid_count": 1,
    }


def test_wrong_metric_length_rejected():
    metrics = sample_metrics()
    metrics["dice"] = torch.tensor([0.8, 0.9])
    with pytest.raises(ValueError, match="three dice"):
        summarize_metrics(metrics)


def test_negative_valid_count_rejected():
    metrics = sample_metrics()
    metrics["dice_counts"][0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        summarize_metrics(metrics)
