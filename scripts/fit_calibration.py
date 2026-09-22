"""Fit one temperature per question type on our own items and write a calibration file.

Raw verification log-odds are scored once (the expensive part), then the temperature that minimises the negative log-likelihood of the expected label is found by a grid search per type. Items with `provenance.gold_probs` contribute a cross-entropy against that distribution instead of a one-hot target. Half of the items, chosen by seed, are held out and reported as before/after ECE.

    python scripts/fit_calibration.py --tasks dev/own_dev.jsonl dev/families/*.jsonl --out release/calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

from compass.calibration import softmax
from compass.contract import compile_request
from compass.scorer import build

GRID = [round(0.25 * k, 2) for k in range(1, 41)]  # 0.25 .. 10.0


def labels_of(task: dict) -> list[str]:
    return ["no", "yes"] if task["question"]["type"] == "noul" else task["labels"]


def target_of(task: dict, labels: list[str]) -> list[float]:
    gold = (task.get("provenance") or {}).get("gold_probs")
    if gold:
        return [gold[k] for k in labels]
    return [1.0 if k == str(task["expected"]) else 0.0 for k in labels]


def nll(rows: list[tuple[list[float], list[float]]], temperature: float) -> float:
    total = 0.0
    for logits, target in rows:
        probs = softmax(logits, temperature)
        total -= sum(t * math.log(max(p, 1e-12)) for t, p in zip(target, probs))
    return total / len(rows)


def ece(rows: list[tuple[list[float], list[float]]], temperature: float, bins: int = 10) -> float:
    cells: dict[int, list[tuple[float, bool]]] = {}
    for logits, target in rows:
        probs = softmax(logits, temperature)
        i = max(range(len(probs)), key=lambda k: (probs[k], -k))
        conf, ok = probs[i], target[i] == max(target)
        cells.setdefault(min(int(conf * bins), bins - 1), []).append((conf, ok))
    return sum(len(c) / len(rows) * abs(sum(ok for _, ok in c) / len(c) - sum(cf for cf, _ in c) / len(c)) for c in cells.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--readout", default="verify", choices=("verify", "direct", "fusion"))
    parser.add_argument("--fusion-weight", type=float, default=0.5)
    parser.add_argument("--debias", type=float, default=0.0)
    parser.add_argument("--head", help="Stage B head directory; overrides the readout")
    args = parser.parse_args()

    scorer = build("backbone", model_name=args.model_name, readout=args.readout, fusion_weight=args.fusion_weight, debias=args.debias, head_path=args.head)
    tasks = [json.loads(l) for p in args.tasks for l in open(p, encoding="utf-8") if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(tasks)

    rows: dict[str, list] = {"choice": [], "score": [], "noul": []}
    t0 = time.time()
    for n, t in enumerate(tasks, 1):
        _, req = compile_request({"state": t["state"], "questions": {"d": t["question"]}})
        # Candidate order in the compiled request is canonical; align the target to it.
        keys = [c.key for c in req.questions[0].candidates]
        labels = labels_of(t)
        target = dict(zip(labels, target_of(t, labels)))
        mapped = [target["yes" if k == "true" else "no" if k == "false" else k] for k in keys]
        rows[t["question"]["type"]].append((scorer.score(req).logits["d"], mapped))
        if n % 20 == 0:
            print(f"scored {n}/{len(tasks)} ({time.time() - t0:.0f} s)")

    temperatures, report = {}, {}
    for qtype, r in rows.items():
        if len(r) < 8:
            temperatures[qtype] = 1.0
            report[qtype] = {"n": len(r), "note": "too few items; identity"}
            continue
        half = len(r) // 2
        fit, held = r[:half], r[half:]
        best = min(GRID, key=lambda T: nll(fit, T))
        # Refit on everything for the shipped value; the held-out numbers above are the honest ones.
        shipped = min(GRID, key=lambda T: nll(r, T))
        temperatures[qtype] = shipped
        report[qtype] = {"n": len(r), "fit_half_T": best, "shipped_T": shipped,
                         "held_out_ece_before": round(ece(held, 1.0), 4), "held_out_ece_after": round(ece(held, best), 4),
                         "held_out_nll_before": round(nll(held, 1.0), 4), "held_out_nll_after": round(nll(held, best), 4)}
        print(qtype, report[qtype])

    out = {"temperatures": temperatures, "fitted_on": args.tasks, "model": scorer.model_id, "readout": args.readout, "head": args.head, "fusion_weight": args.fusion_weight if args.readout == "fusion" else None, "n_items": len(tasks), "report": report}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
