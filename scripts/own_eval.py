"""Evaluate a scorer in-process on our own items (JevBench task format, none of JevBench's data).

Reports accuracy per question type, top-label ECE, mean TVD on items that carry `provenance.gold_probs`, order stability under a reversed option order, and latency. Nothing here touches the benchmark files.

    python scripts/own_eval.py --tasks dev/own_dev.jsonl --model-name Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

from compass.calibration import Calibrator
from compass.contract import decide
from compass.scorer import build


def probs_over_labels(answer: dict, task: dict) -> dict[str, float]:
    if answer["type"] == "noul":
        return {"yes": answer["noul"], "no": 1.0 - answer["noul"]}
    return answer["probabilities"]


def ece(rows: list[tuple[float, bool]], bins: int = 10) -> float:
    total, out = len(rows), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        cell = [(c, ok) for c, ok in rows if lo < c <= hi or (b == 0 and c == 0.0)]
        if cell:
            out += len(cell) / total * abs(sum(ok for _, ok in cell) / len(cell) - sum(c for c, _ in cell) / len(cell))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="dev/own_dev.jsonl")
    parser.add_argument("--scorer", default="backbone")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--head")
    parser.add_argument("--readout", default="verify", choices=("verify", "direct", "fusion"))
    parser.add_argument("--fusion-weight", type=float, default=0.5)
    parser.add_argument("--calibration")
    parser.add_argument("--dump", help="write per-item results to this JSONL")
    args = parser.parse_args()

    scorer = build(args.scorer, model_name=args.model_name, head_path=args.head, readout=args.readout, fusion_weight=args.fusion_weight)
    calibrator = Calibrator.load(args.calibration) if args.calibration else Calibrator()
    tasks = [json.loads(line) for line in open(args.tasks, encoding="utf-8") if line.strip()]

    by_type: dict[str, list[bool]] = {}
    conf_rows, tvds, latencies, tokens, flips, records = [], [], [], [], 0, []
    for t in tasks:
        q = t["question"]
        body = {"state": t["state"], "questions": {"decision": q}}
        t0 = time.perf_counter()
        out = decide(body, scorer, calibrator)
        latencies.append(time.perf_counter() - t0)
        tokens.append(out["usage"]["input_tokens"])
        probs = probs_over_labels(out["answers"]["decision"], t)
        predicted = min(probs, key=lambda k: (-probs[k], k))
        expected = str(t["expected"])
        correct = predicted == expected
        by_type.setdefault(q["type"], []).append(correct)
        conf_rows.append((probs[predicted], correct))
        gold = (t.get("provenance") or {}).get("gold_probs")
        if gold:
            tvds.append(0.5 * sum(abs(probs.get(k, 0.0) - gold.get(k, 0.0)) for k in set(probs) | set(gold)))
        if q["type"] == "choice":
            reversed_q = {**q, "criteria": dict(reversed(list(q["criteria"].items())))}
            again = decide({"state": t["state"], "questions": {"decision": reversed_q}}, scorer, calibrator)
            flips += again["answers"]["decision"]["choice"] != predicted
        records.append({"id": t["id"], "family": t["family"], "type": q["type"], "expected": expected, "predicted": predicted, "correct": correct, "probs": probs})
        print(f"{'ok  ' if correct else 'MISS'} {t['id']:18} expected={expected:24} predicted={predicted:24} p={probs[predicted]:.2f}")

    print()
    for qtype, rows in by_type.items():
        print(f"{qtype:7} accuracy {sum(rows)}/{len(rows)} = {sum(rows) / len(rows):.3f}")
    print(f"overall accuracy {sum(c for _, c in conf_rows) / len(conf_rows):.3f}  top-label ECE {ece(conf_rows):.3f}")
    if tvds:
        print(f"mean TVD to gold distributions on {len(tvds)} graded items: {statistics.mean(tvds):.3f}")
    print(f"choice flips under reversed option order: {flips}")
    print(f"latency p50 {statistics.median(latencies) * 1000:.0f} ms, max {max(latencies) * 1000:.0f} ms; mean input tokens {statistics.mean(tokens):.0f}; model {scorer.model_id}")
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
