# Compass serving image. Weights are downloaded at build time so the container needs no network and no credentials at run time.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV HF_HOME=/opt/hf HF_HUB_OFFLINE=0 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY compass ./compass
COPY release ./release
RUN pip install --no-cache-dir . huggingface_hub \
    && (pip install --no-cache-dir flash-linear-attention || echo "flash-linear-attention unavailable; the reference PyTorch path is used") \
    && (pip install --no-cache-dir causal-conv1d || true)

# Pin the backbone at the recorded revision and bake it into the image.
RUN python -c "import compass.release as r; from huggingface_hub import snapshot_download; snapshot_download(r.BACKBONE, revision=r.BACKBONE_REVISION, ignore_patterns=['*.png'])"

ENV HF_HUB_OFFLINE=1
EXPOSE 8000
CMD ["python", "-m", "compass.server", "--host", "0.0.0.0", "--port", "8000", "--calibration", "release/calibration.json"]
