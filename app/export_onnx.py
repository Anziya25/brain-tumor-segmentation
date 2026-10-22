import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "unetr-brats"
FT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "finetuned_unetr.pth"
ONNX_PATH = PROJECT_ROOT / "models" / "finetuned_unetr.onnx"
sys.path.insert(0, str(MODEL_DIR))

from inference import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, _ = load_model(str(MODEL_DIR / "pytorch_model.bin"))
checkpoint = torch.load(FT_CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Fixed 128^3 patch size — matches the model's positional embeddings,
# same shape used throughout training and sliding-window inference
dummy_input = torch.randn(1, 4, 128, 128, 128, device=device)

torch.onnx.export(
    model,
    dummy_input,
    str(ONNX_PATH),
    export_params=True,
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    do_constant_folding=True,
)

print(f"Exported ONNX model to {ONNX_PATH}")
print(f"PyTorch checkpoint size: {FT_CHECKPOINT_PATH.stat().st_size / (1024*1024):.2f} MB")
print(f"ONNX model size: {ONNX_PATH.stat().st_size / (1024*1024):.2f} MB")