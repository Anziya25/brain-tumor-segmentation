
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install CUDA-enabled PyTorch.
RUN python -m pip install \
    --no-cache-dir \
    --timeout=300 \
    --retries=10 \
    torch==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Copy and install the complete application dependencies.
COPY requirements.txt .

RUN python -m pip install \
    --no-cache-dir \
    --timeout=300 \
    --retries=10 \
    -r requirements.txt \
    && python -m pip check \
    && python -c "import torch, numpy, monai, scipy; print('Core dependencies installed successfully')"

# Copy application source code.
COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
