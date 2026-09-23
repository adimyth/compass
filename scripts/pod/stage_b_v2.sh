#!/usr/bin/env bash
# Stage B v2 on the GPU pod, in the order TRAINING_SPEC.md fixes. Each step writes under dev/results/v2/. Usage: scripts/pod/stage_b_v2.sh [step ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=${HF_HOME:-/workspace/hf}
R=dev/results/v2; mkdir -p "$R"
ADAPTER=release/lora-v2
JB=${JB:-/workspace/jevbench}
steps=${*:-"splits train fusion calibrate release parity baseline_shadow shadow"}

for step in $steps; do
  echo "== $step $(date -u +%FT%TZ)"
  case $step in
    splits)
      uv run python -m compass.data.splits_v2 --out dev/splits_v2 | tee "$R/splits.log"
      uv run python scripts/validate_items.py dev/splits_v2/*.jsonl --jevbench "$JB" | tail -1
      # the hard-like calibration subset: the families the frozen model is weakest on
      uv run python - <<'EOF'
import json
rows=[json.loads(l) for l in open("dev/splits_v2/calibration.jsonl")]
keep=[r for r in rows if r["family"] in ("policy","multi_hop","adequacy","ambiguous","probability","ordinal")]
open("dev/splits_v2/calibration-hard.jsonl","w").write("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in keep))
print("calibration-hard", len(keep))
EOF
      ;;
    train)
      uv run python -m compass.train_lora --train dev/splits_v2/train.jsonl --select dev/splits_v2/selection.jsonl --out "$ADAPTER" 2>&1 | grep -v -i "warn\|fall\|Fetching\|Loading" | tee "$R/train.log"
      ;;
    fusion)
      for w in 0.3 0.5 0.7; do echo "-- fusion $w"; uv run python scripts/own_eval.py --tasks dev/splits_v2/selection.jsonl --adapter "$ADAPTER" --readout fusion --fusion-weight $w 2>&1 | grep -E "overall|flips"; done | tee "$R/fusion.log"
      W=$(grep -B1 overall "$R/fusion.log" | awk '/-- fusion/{w=$3} /overall/{print w, $3}' | sort -k2 -nr | head -1 | cut -d" " -f1)
      echo "$W" > "$R/fusion_weight.txt"; echo "chosen fusion weight $W"
      ;;
    calibrate)
      W=$(cat "$R/fusion_weight.txt")
      uv run python scripts/fit_calibration.py --tasks dev/splits_v2/calibration-hard.jsonl --adapter "$ADAPTER" --readout fusion --fusion-weight "$W" --out "$ADAPTER/calibration.json" 2>&1 | grep -E "^(choice|score|noul|wrote)" | tee "$R/calibrate.log"
      ;;
    release)
      W=$(cat "$R/fusion_weight.txt")
      echo "-- frozen compass-0.1.1"; uv run python scripts/own_eval.py --tasks dev/splits_v2/release.jsonl --readout fusion --fusion-weight 0.5 --calibration release/calibration.json --dump "$R/release-frozen.jsonl" 2>&1 | grep -E "accuracy|ECE|flips|latency" | tee "$R/release.log"
      echo "-- lora-v2"; uv run python scripts/own_eval.py --tasks dev/splits_v2/release.jsonl --adapter "$ADAPTER" --readout fusion --fusion-weight "$W" --calibration "$ADAPTER/calibration.json" --dump "$R/release-lora.jsonl" 2>&1 | grep -E "accuracy|ECE|flips|latency" | tee -a "$R/release.log"
      ;;
    parity)
      uv run python scripts/parity.py --n 12 --tasks dev/splits_v2/selection.jsonl --adapter "$ADAPTER" 2>&1 | grep -v -i "warn\|fall\|Fetching\|Loading" | tail -3 | tee "$R/parity.log"
      ;;
    baseline_shadow)
      uv run python scripts/shadow_eval.py --name compass-0.1.1 --readout fusion --fusion-weight 0.5 --calibration release/calibration.json 2>&1 | grep -v -i "warn\|fall\|Fetching\|Loading" | tee "$R/shadow-baseline.log"
      ;;
    shadow)
      W=$(cat "$R/fusion_weight.txt")
      uv run python scripts/shadow_eval.py --name lora-v2 --adapter "$ADAPTER" --readout fusion --fusion-weight "$W" --calibration "$ADAPTER/calibration.json" 2>&1 | grep -v -i "warn\|fall\|Fetching\|Loading" | tee "$R/shadow-lora.log"
      ;;
    *) echo "unknown step $step"; exit 1;;
  esac
done
echo "== STAGE_B_V2_DONE"
