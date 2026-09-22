#!/usr/bin/env sh
# Serve the pinned Compass release on /v1/systemone without Docker. One GPU (or MPS/CPU for a smoke run), about 9 GB of weights in bf16.
#   scripts/serve.sh [port]
set -eu
cd "$(dirname "$0")/.."
PORT="${1:-8000}"
uv sync --frozen 2>/dev/null || uv sync
uv pip install flash-linear-attention causal-conv1d >/dev/null 2>&1 || echo "fused linear-attention kernels unavailable; using the reference PyTorch path (slower, same numbers)"
exec uv run python -m compass.server --port "$PORT" --calibration release/calibration.json
