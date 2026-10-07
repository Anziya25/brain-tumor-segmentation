from datalist import build_datalist
from preprocessing import get_preprocessing_transforms

datalist = build_datalist(limit=1)
transforms = get_preprocessing_transforms()

result = transforms(datalist[0])

print("Image shape:", result["image"].shape)
print("Seg shape:", result["seg"].shape)
print("Image dtype:", result["image"].dtype)
print("Unique seg labels:", result["seg"].unique())