"""The final check: JevBench's public items, once, through JevBench's own adapter and scorer, against a running Compass server.

This is run one time before release and its output is reported as is. It is not used to choose prompts, temperatures or anything else (docs/EXPERIMENTS.md). Per-item outcomes go to a JSONL for the record.

    python scripts/public_check.py --jevbench ../jevbench --endpoint http://127.0.0.1:8000 --out dev/results/public-check.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jevbench", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="compass-latest")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.jevbench)
    from jevbench.adapters.typesafe import TypeSafeAdapter
    from jevbench.scoring import score_task
    from jevbench.tasks import load_jsonl

    adapter = TypeSafeAdapter(endpoint=args.endpoint, model=args.model, key_env="", timeout_s=300)
    tiers = {"easy": "easy.jsonl", "standard": "original.jsonl", "hard": "hard.jsonl"}
    summary = {}
    with open(args.out, "w", encoding="utf-8") as fh:
        for tier, fname in tiers.items():
            tasks = load_jsonl(f"{args.jevbench}/datasets/public/{fname}")
            correct = valid = 0
            fam = collections.defaultdict(lambda: [0, 0])
            latencies, tokens, conf_rows, tvds = [], [], [], []
            t0 = time.time()
            for t in tasks:
                r = adapter.run(t)
                rec = {"id": t.id, "tier": tier, "family": t.family, "ok": r.ok, "error": r.error, "latency_s": r.latency_s, "usage": r.usage}
                if r.ok:
                    s = score_task(r.probs, t)
                    rec.update({"valid": s["valid"], "strict_valid": s["strict_valid"], "predicted": s["predicted"], "correct": s["correct"], "probs": s.get("probs")})
                    valid += s["valid"]
                    correct += bool(s["correct"])
                    fam[t.family][0] += bool(s["correct"])
                    if s["valid"]:
                        conf_rows.append((max(s["probs"].values()), bool(s["correct"])))
                        gold = (t.provenance or {}).get("gold_probs")
                        if gold:
                            tvds.append(0.5 * sum(abs(s["probs"].get(k, 0.0) - gold.get(k, 0.0)) for k in t.labels))
                    latencies.append(r.latency_s)
                    tokens.append(r.usage.get("input_tokens", 0))
                fam[t.family][1] += 1
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            acc = correct / len(tasks)
            bins = collections.defaultdict(list)
            for c, ok in conf_rows:
                bins[min(int(c * 10), 9)].append((c, ok))
            ece = sum(len(b) / len(conf_rows) * abs(sum(ok for _, ok in b) / len(b) - sum(c for c, _ in b) / len(b)) for b in bins.values()) if conf_rows else None
            summary[tier] = {"n": len(tasks), "valid": valid, "accuracy": round(acc, 4), "ece": round(ece, 4) if ece is not None else None,
                             "mean_tvd_probability_items": round(statistics.mean(tvds), 4) if tvds else None,
                             "latency_p50_s_this_mac": round(statistics.median(latencies), 3) if latencies else None,
                             "mean_input_tokens": round(statistics.mean(tokens), 1) if tokens else None,
                             "by_family": {f: f"{c}/{n}" for f, (c, n) in sorted(fam.items())}, "wall_s": round(time.time() - t0)}
            print(tier, json.dumps(summary[tier]))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
