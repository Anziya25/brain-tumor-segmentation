from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, NormalizeIntensityd, ConcatItemsd, EnsureTyped
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