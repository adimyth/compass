#!/usr/bin/env bash
# Gate 1 on the GPU pod: parity, CUDA latency, compat check. Writes dev/results/gate1-<gpu>.log.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=${HF_HOME:-/workspace/hf}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-' )
LOG=dev/results/gate1-${GPU}.log
{
echo "== $(date -u +%FT%TZ) $GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
uv run python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
uv run python -c "import fla, causal_conv1d; print('fused linear-attention kernels: yes')" || echo "fused linear-attention kernels: NO (reference path)"

echo "== parity: test suite (0.8B) and forked-vs-flat on the 4B"
uv run pytest -q tests/test_backbone.py 2>&1 | tail -1
uv run python - <<'EOF'
import json, torch
from compass.backbone import BackboneScorer
from compass.contract import compile_request
s = BackboneScorer("Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
worst = 0.0
for line in list(open("dev/splits/selection.jsonl"))[:12]:
    t = json.loads(line)
    _, req = compile_request({"state": t["state"], "questions": {"d": t["question"]}})
    forked = s.score(req).logits["d"]
    prefix, [(rubric, branches)] = s.render(req)
    for f, b in zip(forked, branches):
        enc = s.tok(prefix + rubric + b, return_tensors="pt", add_special_tokens=False).to(s.device)
        with torch.no_grad():
            flat = s._logodds(s.model(**enc).logits[:, -1]).item()
        worst = max(worst, abs(f - flat))
print(f"cached-vs-uncached parity on 12 selection items: max |delta log-odds| = {worst:.4f} ({'OK' if worst < 0.15 else 'FAIL'})")
print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
EOF

echo "== latency: serial, warm, verify readout"
uv run python -m compass.server --port 8000 --calibration release/calibration.json > /tmp/server.log 2>&1 &
SP=$!
for i in $(seq 1 120); do curl -s http://127.0.0.1:8000/healthz > /dev/null && break; sleep 1; done
curl -s http://127.0.0.1:8000/v1/models; echo
uv run python scripts/latency.py --n 100
echo "== compat check with JevBench's adapter"
[ -d /workspace/jevbench ] || git clone -q --depth 1 https://github.com/fstandhartinger/jevbench /workspace/jevbench
uv run python scripts/jevbench_compat.py --jevbench /workspace/jevbench --endpoint http://127.0.0.1:8000
kill $SP
echo "== GATE1_DONE"
} 2>&1 | tee "$LOG"
