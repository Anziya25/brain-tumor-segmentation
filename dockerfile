FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Exact-pinned versions (not a version range) so pip doesn't waste time/bandwidth
# searching for a compatible torch+torchvision pair — each attempt downloads a
# ~800MB wheel, and without pinning, pip can try 3-4 pairs before succeeding.
# --timeout/--retries survive one flaky package download rather than failing the whole build.
RUN pip3 uninstall -y torch torchvision && \
    pip3 install --no-cache-dir --timeout=120 --retries=5 \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Needed for FastAPI file uploads (python-multipart). Kept as its own layer,
# placed after the slow torch install, so adding/changing it later doesn't
# force Docker to redo the 200+ second torch download again.
RUN pip3 install --no-cache-dir python-multipart

COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]