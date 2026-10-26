# GPU inference deployment (fresh UNETR)

**Research demo only; not for clinical use.** Keep uploads and model artifacts private.

On Windows, use Docker Desktop with the **WSL2 backend**, a compatible NVIDIA driver, and working GPU container access. On Linux, configure the NVIDIA Container Toolkit. Docker Desktop must run Linux containers. The host's `nvidia-smi` alone does not prove GPU passthrough into Docker.

From the repository root, ensure these **local** files exist (do not commit them):

- `models/unetr-brats/inference.py` and its required package files
- `models/unetr-brats/pytorch_model.bin`
- `models/finetuned_unetr_fresh.pth` (the fresh checkpoint evaluated on 55 patients)

First verify Docker and GPU access:

```powershell
docker version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

The NVIDIA CUDA image download may take time. If the GPU probe fails, fix Docker Desktop / WSL2 / NVIDIA integration before attempting model inference.

Start the API using both compose files (base compose binds to localhost only):

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml config --quiet
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs --tail=100 brain-tumor-api
```

Verify that the running container sees CUDA and that the service loaded the **fresh** checkpoint:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec brain-tumor-api python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected: `CUDA: True`, an NVIDIA GPU name, and `/health` returning `status: ok`, `device: cuda`, and a validation Dice near **0.8699**. Note that `/health` loads the model; container startup/import alone does not establish working inference. The first health request may take time. To test actual segmentation, send four correctly co-registered BraTS-format NIfTI volumes to `POST /predict`, verify the mask downloads, and inspect logs. The API uses one worker and one in-process inference slot to limit GPU contention.

Stop when finished:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
```

If `/health` reports `cpu`, GPU passthrough or CUDA PyTorch is not working. If it reports a Dice other than the expected fresh validation value, verify the `FINETUNED_CHECKPOINT` environment variable and mounted files. GPU execution has not been verified until these commands succeed on your host.
