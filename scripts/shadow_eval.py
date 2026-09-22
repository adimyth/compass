"""The shadow suite: one run of a named configuration, a dated report, nothing else (SHADOW_SUITE_PROTOCOL.md).

Reports accuracy and top-label ECE per family and per language form, the promotion-rule aggregate (policy + multi_hop + adequacy + ambiguous), order flips, latency and tokens, plus the sha256 of every shadow file so the report names what it measured. A configuration may be run once; the script refuses to overwrite an existing report for the same configuration name.

    python scripts/shadow_eval.py --name compass-0.1.1 --readout fusion --fusion-weight 0.5 --calibration release/calibration.json
    python scripts/shadow_eval.py --name lora-v2 --adapter release/lora-v2 --readout fusion --fusion-weight 0.5 --calibration release/lora-v2/calibration.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import statistics
import time

from compass.calibration import Calibrator
from compass.contract import decide
from compass.scorer import build

CORE = ("policy", "multi_hop", "adequacy", "ambiguous")


def ece(rows):
    bins = collections.defaultdict(list)
    for c, ok in rows:
        bins[min(int(c * 10), 9)].append((c, ok))
    return sum(len(b) / len(rows) * abs(sum(ok for _, ok in b) / len(b) - sum(c for c, _ in b) / len(b)) for b in bins.values()) if rows else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--shadow-dir", default="shadow")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--readout", default="fusion")
    parser.add_argument("--fusion-weight", type=float, default=0.5)
    parser.add_argument("--adapter")
    parser.add_argument("--calibration")
    parser.add_argument("--out-dir", default="dev/results/shadow")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, f"{args.name}.json")
    if os.path.exists(report_path):
        raise SystemExit(f"{report_path} exists: a configuration is run on the shadow suite once")

    files = sorted(glob.glob(os.path.join(args.shadow_dir, "*.jsonl")))
    hashes = {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest() for f in files}
    tasks = [json.loads(l) for f in files for l in open(f, encoding="utf-8") if l.strip()]
    scorer = build("backbone", model_name=args.model_name, readout=args.readout, fusion_weight=args.fusion_weight, adapter=args.adapter)
    calibrator = Calibrator.load(args.calibration) if args.calibration else Calibrator()

    per_fam, per_form, per_type = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    core_rows, latencies, tokens, flips, records = [], [], [], 0, []
    for t in tasks:
        q = t["question"]
        t0 = time.perf_counter()
        out = decide({"state": t["state"], "questions": {"decision": q}}, scorer, calibrator)
        latencies.append(time.perf_counter() - t0)
        tokens.append(out["usage"]["input_tokens"])
        a = out["answers"]["decision"]
        probs = {"yes": a["noul"], "no": 1 - a["noul"]} if a["type"] == "noul" else a["probabilities"]
        pred = min(probs, key=lambda k: (-probs[k], k))
        ok = pred == str(t["expected"])
        row = (probs[pred], ok)
        per_fam[t["family"]].append(row)
        per_form[(t.get("provenance") or {}).get("form", "unknown")].append(row)
        per_type[q["type"]].append(row)
        if t["family"] in CORE:
            core_rows.append(row)
        if q["type"] == "choice":
            rev = {**q, "criteria": dict(reversed(list(q["criteria"].items())))}
            flips += decide({"state": t["state"], "questions": {"decision": rev}}, scorer, calibrator)["answers"]["decision"]["choice"] != pred
        records.append({"id": t["id"], "family": t["family"], "predicted": pred, "expected": str(t["expected"]), "correct": ok, "probs": probs})

    summ = lambda rows: {"n": len(rows), "accuracy": round(sum(ok for _, ok in rows) / len(rows), 4), "ece": round(ece(rows), 4)}
    report = {
        "name": args.name, "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": scorer.model_id, "config": vars(args), "shadow_files": hashes,
        "n": len(tasks), "overall": summ([r for rows in per_fam.values() for r in rows]), "core_families": summ(core_rows),
        "by_family": {k: summ(v) for k, v in sorted(per_fam.items())}, "by_form": {k: summ(v) for k, v in sorted(per_form.items())}, "by_type": {k: summ(v) for k, v in sorted(per_type.items())},
        "choice_flips_reversed_order": flips, "latency_p50_s": round(statistics.median(latencies), 3), "mean_input_tokens": round(statistics.mean(tokens), 1),
    }
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=1)
    with open(os.path.join(args.out_dir, f"{args.name}-per-item.jsonl"), "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps({k: report[k] for k in ("name", "model", "n", "overall", "core_families", "by_family", "by_form", "choice_flips_reversed_order", "latency_p50_s", "mean_input_tokens")}, indent=1))


if __name__ == "__main__":
    main()
