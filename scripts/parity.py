"""Is the cache-forked path faithful? Compare three ways of scoring the same candidates on selection items:

  forked-bf16   the serving path (state cache forked per question and per candidate, branches batched with right padding)
  flat-bf16     each candidate as one flat sequence, same weights and dtype
  flat-fp32     each candidate as one flat sequence in float32, the reference

If forked-bf16 deviates from fp32 about as much as flat-bf16 does, the fork is faithful and the difference is bf16 arithmetic. A fork bug shows up as forked-bf16 deviating far more than flat-bf16, or as argmax flips that flat-bf16 does not have.

    python scripts/parity.py --n 24
"""

from __future__ import annotations

import argparse
import json

import torch

from compass.backbone import BackboneScorer
from compass.contract import compile_request


def flat_scores(scorer: BackboneScorer, model, prefix: str, rubric: str, branches: list[str]) -> list[float]:
    out = []
    device = next(model.parameters()).device
    for b in branches:
        enc = scorer.tok(prefix + rubric + b, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out.append(scorer._logodds(model(**enc).logits[:, -1].to(scorer.device)).item())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--tasks", default="dev/splits/selection.jsonl")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--ref-device", default=None, help="device for the fp32 reference (cpu when the GPU is shared)")
    parser.add_argument("--adapter", default=None)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM

    # Phase 1: the bf16 serving model scores everything (forked and flat); then it is freed so the fp32 reference fits on a 24 GB GPU.
    s = BackboneScorer(args.model_name, adapter=args.adapter)
    items = []
    for line in list(open(args.tasks))[: args.n]:
        t = json.loads(line)
        _, req = compile_request({"state": t["state"], "questions": {"d": t["question"]}})
        out = s.score(req)
        prefix, [(rubric, branches)] = s.render(req)
        items.append((t["id"], out.input_tokens, out.logits["d"], flat_scores(s, s.model, prefix, rubric, branches), prefix, rubric, branches))
    del s.model
    torch.cuda.empty_cache()
    ref_device = torch.device(args.ref_device) if args.ref_device else s.device
    ref = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32)
    if args.adapter:
        from peft import PeftModel
        ref = PeftModel.from_pretrained(ref, args.adapter).merge_and_unload()
    ref = ref.to(ref_device).eval()
    rows = []
    for tid, tokens, forked, flat16, prefix, rubric, branches in items:
        flat32 = flat_scores(s, ref, prefix, rubric, branches)
        am = lambda v: max(range(len(v)), key=lambda k: (v[k], -k))
        rows.append({
            "id": tid, "tokens": tokens,
            "forked_vs_fp32": max(abs(a - b) for a, b in zip(forked, flat32)),
            "flat16_vs_fp32": max(abs(a - b) for a, b in zip(flat16, flat32)),
            "forked_vs_flat16": max(abs(a - b) for a, b in zip(forked, flat16)),
            "argmax_forked_ok": am(forked) == am(flat32), "argmax_flat16_ok": am(flat16) == am(flat32),
        })
        r = rows[-1]
        print(f"{r['id']:34} tok={r['tokens']:5} forked-fp32={r['forked_vs_fp32']:.3f} flat16-fp32={r['flat16_vs_fp32']:.3f} forked-flat16={r['forked_vs_flat16']:.3f} argmax forked={'ok' if r['argmax_forked_ok'] else 'FLIP'} flat16={'ok' if r['argmax_flat16_ok'] else 'FLIP'}")
    n = len(rows)
    print(f"\nmean |delta| vs fp32: forked-bf16 {sum(r['forked_vs_fp32'] for r in rows) / n:.3f}, flat-bf16 {sum(r['flat16_vs_fp32'] for r in rows) / n:.3f}; "
          f"max: forked {max(r['forked_vs_fp32'] for r in rows):.3f}, flat {max(r['flat16_vs_fp32'] for r in rows):.3f}")
    print(f"argmax flips vs fp32: forked-bf16 {sum(not r['argmax_forked_ok'] for r in rows)}/{n}, flat-bf16 {sum(not r['argmax_flat16_ok'] for r in rows)}/{n}")


if __name__ == "__main__":
    main()
