"""Run the manager-chain suite against Qwen with thinking enabled.

The runner checkpoints after every task. It sets a generous 32,768-token
ceiling to protect the host from a non-terminating generation; normal answers
end when Qwen emits its final answer. It is not the former 1,024-token cap.

    uv run python scripts/run_qwen_manager_chain.py \
      --out dev/results/manager-chain-qwen-2026-09-27.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from compass import release
from manager_chain_eval import build_tasks, parse_qwen_answer, qwen_prompt, reasoning_token_count, summary


def completed_summary(rows: list[dict], lengths: list[int]) -> list[dict]:
    """Summarise the depths that have at least one completed checkpoint row."""
    completed_depths = [depth for depth in lengths if any(row["depth"] == depth for row in rows)]
    records = summary(rows, completed_depths, include_reasoning=True)
    for record in records:
        record["truncated"] = sum(row["truncated"] for row in rows if row["depth"] == record["depth"])
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=1, help="Questions to decode together. Larger batches use a CUDA GPU more efficiently.")
    parser.add_argument("--sample", action="store_true", help="Sample reasoning traces instead of using greedy decoding.")
    parser.add_argument("--resume", action="store_true", help="Continue from rows already checkpointed in --out.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 2 or args.samples % 2:
        raise SystemExit("--samples must be an even number of at least 2 so every depth has balanced yes/no labels")
    if any(depth < 1 for depth in args.lengths):
        raise SystemExit("--lengths must contain positive integers")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    tasks = build_tasks(args.lengths, args.samples, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if args.resume and args.out.exists():
        checkpoint = json.loads(args.out.read_text(encoding="utf-8"))
        rows = checkpoint.get("qwen_thinking", {}).get("rows", [])
    completed_ids = {row["id"] for row in rows}
    remaining = [task for task in tasks if task.id not in completed_ids]
    print(f"Running {len(remaining)} remaining of {len(tasks)} balanced tasks on {args.device}: lengths={args.lengths}, samples per length={args.samples}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(release.BACKBONE, revision=release.BACKBONE_REVISION)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(release.BACKBONE, revision=release.BACKBONE_REVISION, dtype=torch.bfloat16).to(args.device).eval()
    torch.manual_seed(args.seed)
    for batch_start in range(0, len(remaining), args.batch_size):
        batch = remaining[batch_start:batch_start + args.batch_size]
        prompts = [tokenizer.apply_chat_template(qwen_prompt(task), tokenize=False, add_generation_prompt=True, enable_thinking=True) for task in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(args.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generation_args = {
                "max_new_tokens": args.max_new_tokens,
                "use_cache": True,
                "do_sample": args.sample,
            }
            if args.sample:
                generation_args.update({"temperature": 0.6, "top_p": 0.95, "top_k": 20})
            generated = model.generate(**encoded, **generation_args)
        elapsed = time.perf_counter() - started
        completions = tokenizer.batch_decode(generated[:, encoded.input_ids.shape[1]:], skip_special_tokens=False)
        batch_rows = []
        for task, completion in zip(batch, completions, strict=True):
            answer = parse_qwen_answer(completion)
            generated_tokens = int(generated.shape[1] - encoded.input_ids.shape[1])
            batch_rows.append({
                "id": task.id,
                "depth": task.depth,
                "expected": task.expected,
                "predicted": answer,
                "correct": answer == task.expected,
                "reasoning_tokens": reasoning_token_count(tokenizer, completion),
                "generated_tokens": generated_tokens,
                "truncated": generated_tokens >= args.max_new_tokens and "</think>" not in completion,
                "latency_seconds": elapsed / len(batch),
                "completion": completion,
            })
        rows.extend(batch_rows)
        checkpoint = {
            "suite": "manager_chain",
            "seed": args.seed,
            "lengths": args.lengths,
            "samples_per_length": args.samples,
            "tasks": [asdict(item) for item in tasks],
            "qwen_thinking": {
                "model": {
                    "model": release.BACKBONE,
                    "thinking": True,
                    "max_new_tokens": args.max_new_tokens,
                    "sampling": args.sample,
                    "temperature": 0.6 if args.sample else None,
                    "top_p": 0.95 if args.sample else None,
                    "top_k": 20 if args.sample else None,
                    "seed": args.seed,
                    "batch_size": args.batch_size,
                },
                "summary": completed_summary(rows, args.lengths),
                "rows": rows,
            },
        }
        args.out.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
        for index, row in enumerate(batch_rows, start=len(rows) - len(batch_rows) + 1):
            print(f"Qwen {index:>3}/{len(tasks)} {row['id']}: expected={row['expected']} predicted={row['predicted'] or 'PARSE_FAIL'} tokens={row['reasoning_tokens']} {'ok' if row['correct'] else 'MISS'}")

    print("depth  correct  accuracy  median reasoning tokens  truncated")
    for record in completed_summary(rows, args.lengths):
        print(f"{record['depth']:>5}  {record['correct']:>3}/{record['total']:<3}  {record['accuracy']:>8.0%}  {record['median_reasoning_tokens']:>23}  {record['truncated']:>9}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
