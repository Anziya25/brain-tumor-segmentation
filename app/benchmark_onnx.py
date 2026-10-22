import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "unetr-brats"
FT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "finetuned_unetr.pth"
ONNX_PATH = PROJECT_ROOT / "models" / "finetuned_unetr.onnx"
sys.path.insert(0, str(MODEL_DIR))

from inference import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_RUNS = 20
WARMUP = 3

# --- PyTorch model ---
torch_model, _ = load_model(str(MODEL_DIR / "pytorch_model.bin"))
checkpoint = torch.load(FT_CHECKPOINT_PATH, map_location=device)
torch_model.load_state_dict(checkpoint["model_state_dict"])
torch_model.to(device)
torch_model.eval()

dummy_input_torch = torch.randn(1, 4, 128, 128, 128, device=device)

with torch.no_grad():
    for _ in range(WARMUP):
        _ = torch_model(dummy_input_torch)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(N_RUNS):
        _ = torch_model(dummy_input_torch)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / N_RUNS * 1000  # ms per run

print(f"PyTorch avg inference time per patch: {torch_time:.2f} ms")

# --- ONNX Runtime model ---
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(str(ONNX_PATH), providers=providers)
print("ONNX Runtime providers in use:", session.get_providers())

dummy_input_np = dummy_input_torch.cpu().numpy()

for _ in range(WARMUP):
    _ = session.run(None, {"input": dummy_input_np})

start = time.time()
for _ in range(N_RUNS):
    _ = session.run(None, {"input": dummy_input_np})
onnx_time = (time.time() - start) / N_RUNS * 1000  # ms per run

print(f"ONNX Runtime avg inference time per patch: {onnx_time:.2f} ms")
print(f"Speedup: {torch_time / onnx_time:.2f}x")

# --- Sanity check: do both models agree? ---
with torch.no_grad():
    torch_out = torch.sigmoid(torch_model(dummy_input_torch)).cpu().numpy()
onnx_out_raw = session.run(None, {"input": dummy_input_np})[0]
onnx_out = 1 / (1 + np.exp(-onnx_out_raw))  # manual sigmoid — export is pre-sigmoid logits

max_diff = np.abs(torch_out - onnx_out).max()
print(f"Max difference between PyTorch and ONNX outputs: {max_diff:.6f}")