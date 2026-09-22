"""Build the four disjoint internal splits (PLAN.md gates 3, 4, 5, 7).

Each split draws from every generator with its own seed range, so no two splits share a seed, and item ids carry the split name so a mix-up is visible. The authored items (`dev/own_dev.jsonl`) are spread across selection, calibration and release, never train. Sizes are per generator module.

    python -m compass.data.splits --out dev/splits
"""

from __future__ import annotations

import argparse
import json
import os
import random

from . import multihop, routing, temporal

MODULES = {"temporal": temporal, "multihop": multihop, "routing": routing}
# split -> (seed base, items per module)
SPLITS = {"selection": (1000, 90), "calibration": (2000, 90), "release": (3000, 120), "train": (4000, 900)}


def build(out_dir: str, authored: str | None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    authored_items = [json.loads(l) for l in open(authored, encoding="utf-8")] if authored else []
    rng = random.Random(0)
    rng.shuffle(authored_items)
    thirds = [authored_items[i::3] for i in range(3)]
    for split, (seed, n) in SPLITS.items():
        items = []
        for offset, (name, module) in enumerate(MODULES.items()):
            for item in module.generate(n, seed + offset):
                item["id"] = f"{split}-{item['id']}"
                item["provenance"]["split"] = split
                items.append(item)
        if split != "train":
            for item in thirds[list(SPLITS).index(split)]:
                items.append({**item, "id": f"{split}-{item['id']}", "provenance": {**item.get("provenance", {}), "split": split}})
        rng.shuffle(items)
        with open(os.path.join(out_dir, f"{split}.jsonl"), "w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"{split}: {len(items)} items")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dev/splits")
    parser.add_argument("--authored", default="dev/own_dev.jsonl")
    args = parser.parse_args()
    build(args.out, args.authored)


if __name__ == "__main__":
    main()
