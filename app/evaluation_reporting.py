"""Machine-readable held-out evaluation reporting (no patient identifiers)."""

import json
import math
from pathlib import Path

REGIONS = ("ET", "TC", "WT")


def _finite_or_none(value):
    number = float(value)
    return number if math.isfinite(number) else None


def summarize_metrics(results):
    """Convert metric tensors to JSON-safe per-region values and valid counts."""
    output = {}
    for metric, count_key in (("dice", "dice_counts"), ("hd95", "hd95_counts")):
        values = results[metric].tolist()
        counts = results[count_key].tolist()
        if len(values) != len(REGIONS) or len(counts) != len(REGIONS):
            raise ValueError(f"Expected three {metric} values and counts")
        per_region = {}
        for region, value, count in zip(REGIONS, values, counts):
            valid_count = int(count)
            if valid_count < 0:
                raise ValueError("Metric valid counts must be nonnegative")
            per_region[region] = {
                "value": _finite_or_none(value) if valid_count else None,
                "valid_count": valid_count,
            }
        output[metric] = per_region
    return output


def write_evaluation_report(path: Path, *, baseline, finetuned, provenance: dict):
    """Write strict JSON; undefined/empty-region metrics are null, never NaN."""
    report = {
        "schema_version": 1,
        "evaluation": "held_out_test",
        "regions": list(REGIONS),
        "metrics": {
            "dice": {"unit": "unitless", "higher_is_better": True},
            "hd95": {"unit": "mm", "higher_is_better": False},
        },
        "baseline": summarize_metrics(baseline),
        "finetuned": summarize_metrics(finetuned),
        "provenance": provenance,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
