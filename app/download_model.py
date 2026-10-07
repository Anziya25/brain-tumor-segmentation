from huggingface_hub import snapshot_download
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
local_dir = snapshot_download(
    "MohanadKombar/brain-tumor-unetr-brats",
    local_dir=str(PROJECT_ROOT / "models" / "unetr-brats"),
)
print("Downloaded to:", local_dir)