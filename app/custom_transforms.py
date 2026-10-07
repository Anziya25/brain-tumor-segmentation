import numpy as np
import torch
from monai.transforms import MapTransform

class PercentileClipd(MapTransform):
    """Clips each modality's intensities to the [lower, upper] percentile range."""
    def __init__(self, keys, lower=0.5, upper=99.5):
        super().__init__(keys)
        self.lower = lower
        self.upper = upper

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            lo = np.percentile(img, self.lower)
            hi = np.percentile(img, self.upper)
            d[key] = np.clip(img, lo, hi)
        return d


class ConvertToBratsMultiChanneld(MapTransform):
    """Converts raw BraTS labels (0,1,2,4) into a 3-channel multi-label
    target [ET, TC, WT], matching the model's output channel order."""
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            seg = d[key]
            if seg.ndim == 4:
                seg = seg[0]  # drop channel dim if present
            et = (seg == 4)
            tc = (seg == 1) | (seg == 4)
            wt = (seg == 1) | (seg == 2) | (seg == 4)
            d[key] = torch.stack([et, tc, wt], dim=0).float()
        return d