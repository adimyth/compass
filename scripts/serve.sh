#!/usr/bin/env sh
# Serve the pinned Compass release on /v1/systemone without Docker. One GPU (or MPS/CPU for a smoke run), about 9 GB of weights in bf16.
#   scripts/serve.sh [port]
set -eu
cd "$(dirname "$0")/.."
PORT="${1:-8000}"
uv sync --frozen 2>/dev/null || uv sync
# flash-linear-attention (Triton) is what speeds up the hybrid layers; causal-conv1d is optional and often has no wheel for the current torch.
uv pip install flash-linear-attention >/dev/null 2>&1 || echo "flash-linear-attention unavailable; using the reference PyTorch path (slower, same numbers)"
uv pip install causal-conv1d >/dev/null 2>&1 || true
exec uv run python -m compass.server --port "$PORT" --calibration release/calibration.json
