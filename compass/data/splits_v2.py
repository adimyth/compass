"""Build the Stage B v2 splits from a template manifest (CORPUS_the README).

Generated items: each (generator, form) template is assigned to exactly one split; forms held out of train are listed. Authored items: files under dev/authored/ are assigned by their domain group (train-* → train; eval-* → selection, calibration and release by a stable hash of the scenario id). The shadow suite under shadow/ is never read here. The builder refuses to run if a template id would land in two splits.

    python -m compass.data.splits_v2 --out dev/splits_v2
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random

from . import multihop, routing, temporal, v2

# template -> (split, n, seed). Forms: multihop/routing/temporal generators have one form each (record-like), v2 generators have two.
MANIFEST = {
    "policy-record": ("train", 700, 5100), "policy-narrative": ("release", 120, 5300),          # narrative policy held out of train
    "probability-record": ("train", 300, 5110), "probability-narrative": ("selection", 60, 5210), "probability-release": ("release", 60, 5310),
    "ordinal-record": ("train", 300, 5120), "ordinal-narrative": ("calibration", 60, 5220), "ordinal-release": ("release", 60, 5320),
    "multihop": ("train", 600, 5130), "multihop-eval": ("calibration", 90, 5230), "multihop-release": ("release", 90, 5330),
    "routing-gen": ("selection", 60, 5240), "routing-gen-release": ("release", 60, 5340),
    "temporal-regression": ("release", 60, 5350),
}


def generated(template: str, n: int, seed: int) -> list[dict]:
    if template.startswith(("policy", "probability", "ordinal")):
        gen, form = template.split("-")
        return v2.generate(gen, "narrative" if form == "release" else form, n, seed)
    if template.startswith("multihop"):
        return multihop.generate(n, seed)
    if template.startswith("routing"):
        return [t for t in routing.generate(n * 3, seed) if t["family"] == "routing"][:n]
    if template.startswith("temporal"):
        return temporal.generate(n, seed)
    raise ValueError(template)


def eval_split_for(item_id: str) -> str:
    h = int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % 10
    return "selection" if h < 3 else "calibration" if h < 6 else "release"


def build(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    splits: dict[str, list[dict]] = {"train": [], "selection": [], "calibration": [], "release": []}
    seen_templates: dict[str, str] = {}
    for template, (split, n, seed) in MANIFEST.items():
        if template in seen_templates:
            raise SystemExit(f"template {template} assigned twice")
        seen_templates[template] = split
        for item in generated(template, n, seed):
            item["id"] = f"{split}-{template}-{item['id']}"
            item.setdefault("provenance", {}).update({"split": split, "template": template})
            splits[split].append(item)
    for path in sorted(glob.glob("dev/authored/*.jsonl")):
        name = os.path.basename(path)
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            item = json.loads(line)
            split = "train" if name.startswith("train-") else eval_split_for(item["id"])
            item["id"] = f"{split}-{item['id']}"
            item.setdefault("provenance", {}).update({"split": split, "template": f"authored:{item['provenance'].get('domain', 'unknown')}"})
            splits[split].append(item)
    rng = random.Random(0)
    for split, items in splits.items():
        rng.shuffle(items)
        with open(os.path.join(out_dir, f"{split}.jsonl"), "w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        fams: dict[str, int] = {}
        for it in items:
            fams[it["family"]] = fams.get(it["family"], 0) + 1
        authored = sum(1 for it in items if it["provenance"].get("source") == "authored")
        print(f"{split}: {len(items)} items, {authored} authored, {dict(sorted(fams.items()))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dev/splits_v2")
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
