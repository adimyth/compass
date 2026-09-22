"""Validate our dev and training items before they are used.

Checks: JevBench task shape (so the compat harness can score them), expected label in the label set, banned keys in structured states, gold balance per file, and n-gram overlap against JevBench's public files when a clone is given. Overlap is the check the benchmark maintainer ran on another entrant; we run it first.

    python scripts/validate_items.py dev/own_dev.jsonl dev/families/*.jsonl --jevbench ../jevbench
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys

BANNED_STATE_KEYS = {"expected", "label", "ground_truth", "answer_key"}
NGRAM = 8


def texts_of(state) -> list[str]:
    if isinstance(state, str):
        return [state]
    if isinstance(state, dict):
        return [t for v in state.values() for t in texts_of(v)]
    if isinstance(state, list):
        return [t for v in state for t in texts_of(v)]
    return []


def ngrams(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i:i + NGRAM]) for i in range(len(words) - NGRAM + 1)}


def check_item(t: dict, where: str) -> list[str]:
    errs = []
    for key in ("id", "family", "state", "question", "labels", "expected", "split"):
        if key not in t:
            errs.append(f"{where}: missing {key}")
    if errs:
        return errs
    q = t["question"]
    qtype = q.get("type")
    labels = t["labels"]
    if qtype == "noul":
        if labels != ["no", "yes"] or t["expected"] not in ("no", "yes"):
            errs.append(f"{where}: noul needs labels [no, yes] and a yes/no expected")
        crit = q.get("criteria")
        if crit is not None and set(crit) - {"true", "false"}:
            errs.append(f"{where}: noul criteria keys must be true/false")
    elif qtype == "choice":
        if not isinstance(q.get("criteria"), dict) or sorted(q["criteria"]) != sorted(labels):
            errs.append(f"{where}: choice criteria keys must equal labels")
        if t["expected"] not in labels:
            errs.append(f"{where}: expected not in labels")
    elif qtype == "score":
        levels = q.get("criteria")
        if not isinstance(levels, list) or labels != [str(i) for i in range(len(levels))]:
            errs.append(f"{where}: score labels must be '0'..'k' matching criteria")
        if not isinstance(t["expected"], int) or str(t["expected"]) not in labels:
            errs.append(f"{where}: score expected must be an int level index")
    else:
        errs.append(f"{where}: bad question type {qtype!r}")
    if isinstance(t["state"], dict) and BANNED_STATE_KEYS & set(t["state"]):
        errs.append(f"{where}: state carries a banned key")
    gold = (t.get("provenance") or {}).get("gold_probs")
    if gold is not None:
        if abs(sum(gold.values()) - 1.0) > 1e-6 or set(gold) != set(labels):
            errs.append(f"{where}: gold_probs must cover the labels and sum to 1")
        elif max(gold, key=gold.get) != str(t["expected"]):
            errs.append(f"{where}: expected must be the gold_probs argmax")
    return errs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--jevbench", help="clone of fstandhartinger/jevbench for the overlap check")
    args = parser.parse_args()

    bench_grams: set = set()
    if args.jevbench:
        for path in glob.glob(f"{args.jevbench}/datasets/public/*.jsonl"):
            for line in open(path, encoding="utf-8"):
                row = json.loads(line)
                for text in texts_of(row["state"]) + [row["question"]["instructions"]]:
                    bench_grams |= ngrams(text)
        print(f"overlap check against {len(bench_grams)} JevBench public {NGRAM}-grams")

    errors, total, ids = [], 0, collections.Counter()
    for path in args.files:
        items = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        golds = collections.Counter()
        for i, t in enumerate(items):
            where = f"{path}:{i + 1}"
            errors += check_item(t, where)
            ids[t.get("id")] += 1
            golds[(t.get("question", {}).get("type"), str(t.get("expected")))] += 1
            if bench_grams:
                for text in texts_of(t.get("state")):
                    hit = ngrams(text) & bench_grams
                    if hit:
                        errors.append(f"{where}: state shares a {NGRAM}-gram with a JevBench public item: {' '.join(next(iter(hit)))!r}")
        total += len(items)
        print(f"{path}: {len(items)} items; golds {dict(golds)}")
    errors += [f"duplicate id {i!r}" for i, n in ids.items() if n > 1]
    for e in errors:
        print("ERROR", e)
    print(f"{total} items, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
