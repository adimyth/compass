#!/usr/bin/env bash
# Gate 2: frozen-readout comparison on the selection split. Writes dev/results/gate2-<readout>.jsonl and a summary log.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=${HF_HOME:-/workspace/hf}
LOG=dev/results/gate2-selection.log
: > "$LOG"
run() {  # name, readout, weight
  echo "== $1" | tee -a "$LOG"
  uv run python scripts/own_eval.py --tasks dev/splits/selection.jsonl --model-name Qwen/Qwen3.5-4B \
      --readout "$2" --fusion-weight "$3" --dump "dev/results/gate2-$1.jsonl" 2>&1 \
      | grep -E "accuracy|ECE|TVD|flips|latency" | tee -a "$LOG"
}
run verify verify 0.5
run direct direct 0.5
run fusion-0.3 fusion 0.3
run fusion-0.5 fusion 0.5
run fusion-0.7 fusion 0.7
uv run python - <<'EOF' | tee -a "$LOG"
import json, collections, glob
print("\nper-family accuracy on the selection split")
rows = {}
for f in sorted(glob.glob("dev/results/gate2-*.jsonl")):
    name = f.split("gate2-")[1][:-6]
    fam = collections.defaultdict(lambda: [0, 0])
    for l in open(f):
        r = json.loads(l); fam[r["family"]][0] += r["correct"]; fam[r["family"]][1] += 1
    rows[name] = fam
fams = sorted({k for v in rows.values() for k in v})
print(f"{'readout':12}" + "".join(f"{f[:12]:>13}" for f in fams) + f"{'all':>8}")
for name, fam in rows.items():
    tot = [sum(v[0] for v in fam.values()), sum(v[1] for v in fam.values())]
    print(f"{name:12}" + "".join(f"{fam[f][0]:>6}/{fam[f][1]:<6}" if f in fam else " " * 13 for f in fams) + f"{tot[0] / tot[1]:8.3f}")
EOF
echo "== GATE2_DONE" | tee -a "$LOG"
