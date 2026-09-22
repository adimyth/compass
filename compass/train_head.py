"""Stage B, head-only: extract frozen features once, then train the residual head.

    python -m compass.train_head extract --tasks dev/splits/train.jsonl --out /workspace/feats/train.pt
    python -m compass.train_head fit --train /workspace/feats/train.pt --select /workspace/feats/selection.pt --out release/head-v1

Loss: cross-entropy against the target distribution over candidates (one-hot, or `gold_probs` where given), plus, for score questions, an ordinal term (the expected level's squared distance from the target level, scaled). Early stopping on the selection split's loss. The release split is never read here.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import torch
from torch import nn

from .contract import compile_request
from .head import TYPES, CompassHead


def target_for(task: dict, keys: list[str]) -> list[float]:
    gold = (task.get("provenance") or {}).get("gold_probs")
    name = {"true": "yes", "false": "no"}
    if gold:
        return [gold[name.get(k, k)] for k in keys]
    exp = str(task["expected"])
    return [1.0 if name.get(k, k) == exp else 0.0 for k in keys]


def extract(args) -> None:
    from .backbone import BackboneScorer

    s = BackboneScorer(args.model_name)
    rows, t0 = [], time.time()
    for n, line in enumerate(open(args.tasks, encoding="utf-8"), 1):
        t = json.loads(line)
        _, req = compile_request({"state": t["state"], "questions": {"d": t["question"]}})
        f, _ = s.features(req)
        f = f["d"]
        keys = [c.key for c in req.questions[0].candidates]
        rows.append({"id": t["id"], "family": t["family"], "hidden": f["hidden"].half(), "verify": f["verify"], "direct": f["direct"],
                     "qtype": f["qtype"], "target": torch.tensor(target_for(t, keys))})
        if n % 100 == 0:
            print(f"{n} items, {time.time() - t0:.0f} s")
    torch.save(rows, args.out)
    print(f"wrote {len(rows)} items to {args.out}")


def loss_fn(head: CompassHead, batch: list[dict], device, ordinal_weight: float) -> torch.Tensor:
    total = 0.0
    for r in batch:
        n = r["hidden"].shape[0]
        logits = head(r["hidden"].to(device).float(), r["verify"].to(device), r["direct"].to(device),
                      torch.full((n,), r["qtype"], dtype=torch.long, device=device))
        logp = torch.log_softmax(logits, dim=-1)
        target = r["target"].to(device)
        total = total - (target * logp).sum()
        if TYPES[r["qtype"]] == "score":
            levels = torch.arange(n, device=device, dtype=torch.float32)
            ev, tv = (logp.exp() * levels).sum(), (target * levels).sum()
            total = total + ordinal_weight * (ev - tv) ** 2 / max(n - 1, 1)
    return total / len(batch)


def evaluate(head: CompassHead, rows: list[dict], device) -> tuple[float, float]:
    head.eval()
    with torch.no_grad():
        loss = loss_fn(head, rows, device, 0.0).item()
        correct = 0
        for r in rows:
            n = r["hidden"].shape[0]
            logits = head(r["hidden"].to(device).float(), r["verify"].to(device), r["direct"].to(device), torch.full((n,), r["qtype"], dtype=torch.long, device=device))
            correct += int(logits.argmax().item() == int(r["target"].argmax().item()))
    head.train()
    return loss, correct / len(rows)


def fit(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, select = torch.load(args.train), torch.load(args.select)
    head = CompassHead(train[0]["hidden"].shape[1], args.width, args.dropout).to(device)
    base_loss, base_acc = evaluate(head, select, device)
    print(f"untrained head (= release fusion): selection loss {base_loss:.4f} acc {base_acc:.3f}")
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(0)
    best, best_state, bad = base_loss, {k: v.clone() for k, v in head.state_dict().items()}, 0
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train)
        for i in range(0, len(train), args.batch):
            opt.zero_grad()
            loss_fn(head, train[i:i + args.batch], device, args.ordinal_weight).backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
        tr_loss, tr_acc = evaluate(head, train[: len(select)], device)
        se_loss, se_acc = evaluate(head, select, device)
        print(f"epoch {epoch}: train loss {tr_loss:.4f} acc {tr_acc:.3f} | selection loss {se_loss:.4f} acc {se_acc:.3f} | readout weights {head.readout_weights.tolist()}")
        if se_loss < best - 1e-4:
            best, best_state, bad = se_loss, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break
    head.load_state_dict(best_state)
    se_loss, se_acc = evaluate(head, select, device)
    print(f"best: selection loss {se_loss:.4f} acc {se_acc:.3f} (untrained {base_loss:.4f} / {base_acc:.3f})")
    head.save(args.out)
    with open(f"{args.out}/training.json", "w") as fh:
        json.dump({"train": args.train, "select": args.select, "n_train": len(train), "n_select": len(select), "lr": args.lr, "weight_decay": args.weight_decay,
                   "width": args.width, "dropout": args.dropout, "ordinal_weight": args.ordinal_weight, "batch": args.batch,
                   "selection_loss_untrained": base_loss, "selection_acc_untrained": base_acc, "selection_loss": se_loss, "selection_acc": se_acc}, fh, indent=2)
    print("saved", args.out)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--tasks", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    f = sub.add_parser("fit")
    f.add_argument("--train", required=True)
    f.add_argument("--select", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--lr", type=float, default=3e-4)
    f.add_argument("--weight-decay", type=float, default=0.01)
    f.add_argument("--width", type=int, default=256)
    f.add_argument("--dropout", type=float, default=0.1)
    f.add_argument("--ordinal-weight", type=float, default=0.5)
    f.add_argument("--batch", type=int, default=16)
    f.add_argument("--epochs", type=int, default=30)
    f.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()
    (extract if args.cmd == "extract" else fit)(args)


if __name__ == "__main__":
    main()
