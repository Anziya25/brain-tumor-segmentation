FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# CPU is the portable default; use Dockerfile.gpu for NVIDIA deployments.
RUN python -m pip install --timeout=300 --retries=5 \
    torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-cpu.txt ./
RUN python -m pip install --timeout=300 --retries=5 -r requirements-cpu.txt \
    && python -m pip check

COPY app/ app/
RUN python -c "import app.main; print('API imports OK')"

RUN mkdir -p /app/models /app/predictions
EXPOSE 8000

# Single worker: the inference semaphore is per-process.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
