"""Stage B v2: train a LoRA on the frozen backbone through the existing verification and direct readouts (TRAINING_SPEC.md).

Flattened full sequences, no cache forking: for each item, every verification branch is one sequence `prefix + rubric + branch` and the direct readout is one sequence `direct prefix + direct rubric + tail`, batched with right padding and read at the last real token with the same helpers `compass/backbone.py` uses at inference. Loss per item over its candidate group: listwise CE on the fused distribution, per-readout CE, ordinal distance for score items, permutation consistency (second rubric order) and opaque-label consistency (option keys replaced by neutral tokens).

    python -m compass.train_lora --train dev/splits_v2/train.jsonl --select dev/splits_v2/selection.jsonl --out release/lora-v2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import replace

import torch
from torch.nn import functional as F

from .backbone import SYMBOLS, BackboneScorer, direct_rubric_text, rubric_text, proposition
from .contract import Candidate, CompiledQuestion, CompiledRequest, compile_request

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj"]
NAME = {"true": "yes", "false": "no"}


def target_for(task: dict, q: CompiledQuestion) -> list[float]:
    gold = (task.get("provenance") or {}).get("gold_probs")
    if gold:
        return [gold[NAME.get(c.key, c.key)] for c in q.candidates]
    exp = str(task["expected"])
    return [1.0 if NAME.get(c.key, c.key) == exp else 0.0 for c in q.candidates]


def opaque(q: CompiledQuestion, rng: random.Random) -> CompiledQuestion:
    """Replace choice option keys by neutral tokens in a random order; the candidates' texts and the target order are kept."""
    if q.type != "choice":
        return q
    names = [f"option_{k}" for k in range(1, len(q.candidates) + 1)]
    rng.shuffle(names)
    return replace(q, candidates=tuple(Candidate(n, c.text) for n, c in zip(names, q.candidates)))


def permuted(q: CompiledQuestion, rng: random.Random) -> tuple[CompiledQuestion, list[int]]:
    """A second candidate order for the rubric; returns the question and the index map back to the original order. Score questions keep their order."""
    if q.type == "score":
        return q, list(range(len(q.candidates)))
    order = list(range(len(q.candidates)))
    rng.shuffle(order)
    return replace(q, candidates=tuple(q.candidates[k] for k in order)), order


class Trainer:
    def __init__(self, scorer: BackboneScorer):
        self.s = scorer
        self.model = scorer.model

    def _last_logits(self, texts: list[str]) -> torch.Tensor:
        enc = self.s.tok(texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False).to(self.s.device)
        out = self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
        last = enc.attention_mask.sum(dim=1) - 1
        return out.logits[torch.arange(len(texts), device=self.s.device), last], int(enc.attention_mask.sum().item())

    def readouts(self, state: str, q: CompiledQuestion) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Verification log-odds (normalised over candidates) and direct log-probs for one question, both differentiable. Returns (v, d, tokens)."""
        req = CompiledRequest(state=state, questions=(q,))
        prefix, [(rubric, branches)] = self.s.render(req)
        logits, n_tok = self._last_logits([prefix + rubric + b for b in branches])
        v = torch.log_softmax(self.s._logodds(logits), dim=-1)
        dprefix, _ = self.s.render(req, system=self.s.direct_system)
        dlogits, d_tok = self._last_logits([dprefix + direct_rubric_text(q) + self.s._tail])
        lp = torch.log_softmax(dlogits[0].float(), dim=-1)
        sel = lp[self.s._symbol_ids(len(q.candidates))]
        d = sel - torch.logsumexp(sel, dim=-1)
        return v, d, n_tok + d_tok

    def item_loss(self, task: dict, rng: random.Random, args, train: bool) -> tuple[torch.Tensor, dict]:
        _, req = compile_request({"state": task["state"], "questions": {"d": task["question"]}})
        q = req.questions[0]
        target = torch.tensor(target_for(task, q), device=self.s.device)
        if train and rng.random() < args.opaque_p:
            q = opaque(q, rng)
        v, d, _ = self.readouts(req.state, q)
        f = args.fusion * v + (1 - args.fusion) * d
        ce = lambda logp: -(target * logp).sum()
        loss = ce(torch.log_softmax(f, -1)) + 0.5 * (ce(v) + ce(d))
        if q.type == "score":
            levels = torch.arange(len(q.candidates), device=self.s.device, dtype=torch.float32)
            ev, tv = (torch.softmax(f, -1) * levels).sum(), (target * levels).sum()
            loss = loss + args.ordinal_w * ((ev - tv) / max(len(q.candidates) - 1, 1)) ** 2
        if train and rng.random() < args.perm_p and q.type != "score":
            q2, order = permuted(q, rng)
            v2, d2, _ = self.readouts(req.state, q2)
            f2 = args.fusion * v2 + (1 - args.fusion) * d2
            back = torch.empty_like(f2)
            back[torch.tensor(order, device=self.s.device)] = f2  # f2[k] belongs to original index order[k]
            p1, p2 = torch.log_softmax(f, -1), torch.log_softmax(back, -1)
            loss = loss + args.perm_w * 0.5 * (F.kl_div(p2, p1, log_target=True, reduction="sum") + F.kl_div(p1, p2, log_target=True, reduction="sum"))
        correct = int(torch.argmax(f).item() == int(torch.argmax(target).item()))
        return loss, {"correct": correct, "conf": float(torch.softmax(f, -1).max().item())}

    @torch.no_grad()
    def evaluate(self, tasks: list[dict], args) -> dict:
        self.model.eval()
        rng = random.Random(0)
        losses, correct, per_fam = [], 0, {}
        for t in tasks:
            loss, info = self.item_loss(t, rng, args, train=False)
            losses.append(loss.item())
            correct += info["correct"]
            fam = per_fam.setdefault(t["family"], [0, 0])
            fam[0] += info["correct"]
            fam[1] += 1
        self.model.train()
        return {"loss": sum(losses) / len(losses), "acc": correct / len(tasks), "by_family": {k: f"{c}/{n}" for k, (c, n) in sorted(per_fam.items())}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--select", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8, help="items per optimiser step")
    parser.add_argument("--fusion", type=float, default=0.5)
    parser.add_argument("--ordinal-w", type=float, default=0.5)
    parser.add_argument("--perm-w", type=float, default=0.5)
    parser.add_argument("--perm-p", type=float, default=0.5)
    parser.add_argument("--opaque-p", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-select", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    scorer = BackboneScorer(args.model_name)
    scorer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    scorer.model.enable_input_require_grads()
    scorer.model = get_peft_model(scorer.model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.lora_dropout, target_modules=TARGETS, task_type="CAUSAL_LM"))
    scorer.model.print_trainable_parameters()
    tr = Trainer(scorer)
    train = [json.loads(l) for l in open(args.train, encoding="utf-8") if l.strip()]
    select = [json.loads(l) for l in open(args.select, encoding="utf-8") if l.strip()][: args.max_select]
    params = [p for p in scorer.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train) * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, int(0.03 * total_steps))) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))

    base = tr.evaluate(select, args)
    print(f"untrained: selection loss {base['loss']:.4f} acc {base['acc']:.3f} {base['by_family']}", flush=True)
    best, best_step, bad, step, seen, t0 = base["loss"], 0, 0, 0, 0, time.time()
    log = [{"step": 0, "seen": 0, **base}]
    scorer.model.train()
    stop = False
    for epoch in range(args.epochs):
        rng.shuffle(train)
        for i in range(0, len(train), args.accum):
            batch = train[i:i + args.accum]
            opt.zero_grad()
            run_loss = 0.0
            for t in batch:
                loss, _ = tr.item_loss(t, rng, args, train=True)
                (loss / len(batch)).backward()
                run_loss += loss.item() / len(batch)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            seen += len(batch)
            if step % 10 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} seen {seen} loss {run_loss:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
            if seen % args.eval_every < len(batch):
                ev = tr.evaluate(select, args)
                log.append({"step": step, "seen": seen, **ev})
                print(f"  selection: loss {ev['loss']:.4f} acc {ev['acc']:.3f} {ev['by_family']}", flush=True)
                if ev["loss"] < best - 1e-4:
                    best, best_step, bad = ev["loss"], step, 0
                    scorer.model.save_pretrained(args.out)
                else:
                    bad += 1
                    if bad >= args.patience:
                        print("early stop", flush=True)
                        stop = True
                        break
        if stop:
            break
    if best_step == 0:
        scorer.model.save_pretrained(args.out)
    with open(f"{args.out}/training.json", "w") as fh:
        json.dump({"args": vars(args), "n_train": len(train), "n_select": len(select), "best_selection_loss": best, "best_step": best_step, "log": log, "targets": TARGETS}, fh, indent=2)
    print("saved", args.out, "best selection loss", round(best, 4), "at step", best_step)


if __name__ == "__main__":
    main()
