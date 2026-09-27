"""Run the manager-chain suite against a hosted Qwen3.5 model on OpenRouter with thinking enabled.

Requests run concurrently, so the sweep takes roughly as long as the slowest single answer. The key comes from OPENROUTER_API_KEY in the environment or the repository's .env. Routing is pinned to bf16 providers so the weights are not quantised.

    uv run python scripts/run_openrouter_manager_chain.py \
      --out dev/results/manager-chain-qwen9b-2026-09-27.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from manager_chain_eval import build_tasks, parse_qwen_answer, qwen_prompt, summary


API_URL = "https://openrouter.ai/api/v1/chat/completions"


def api_key() -> str:
    """Read OPENROUTER_API_KEY from the environment, falling back to the repository's .env."""
    key = os.environ.get("OPENROUTER_API_KEY")
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip().removeprefix("export ").strip() == "OPENROUTER_API_KEY":
                key = value.strip().strip("'\"")
    if not key:
        raise SystemExit(f"Set OPENROUTER_API_KEY in the environment or in {env_file}")
    return key


def request(payload: dict, key: str, timeout: float, attempts: int = 10) -> dict:
    """POST one chat completion, retrying rate limits, upstream errors, and dropped connections."""
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(attempts):
        try:
            http = urllib.request.Request(API_URL, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(http, timeout=timeout) as response:
                data = json.loads(response.read())
            if "error" not in data and data.get("choices"):
                return data
            error = data.get("error")
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise SystemExit(f"OpenRouter returned HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:500]}")
            error = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            error = repr(exc)
        # Jitter spreads rate-limited requests out so they do not retry in lockstep.
        wait = min(2 ** attempt * 5, 120) * random.uniform(0.5, 1.5)
        print(f"  retry {attempt + 1}/{attempts} in {wait:.0f}s: {error}", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"request failed after {attempts} attempts")


def completed_summary(rows: list[dict], lengths: list[int]) -> list[dict]:
    completed_depths = [depth for depth in lengths if any(row["depth"] == depth for row in rows)]
    records = summary(rows, completed_depths, include_reasoning=True)
    for record in records:
        group = [row for row in rows if row["depth"] == record["depth"]]
        record["truncated"] = sum(row["truncated"] for row in group)
        # Providers end some long generations with finish_reason "error"; those rows have no final answer.
        record["unfinished"] = sum(row["finish_reason"] != "stop" for row in group)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--hf-model", default="Qwen/Qwen3.5-9B", help="Hugging Face id whose tokenizer counts reasoning tokens.")
    parser.add_argument("--providers", nargs="+", default=["parasail"], help="bf16 OpenRouter providers to allow, in order. DeepInfra also serves bf16 but decoded at about 20 tokens/s against Parasail's 128.")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--concurrency", type=int, default=12, help="Parallel requests; Parasail rate-limits above roughly a dozen.")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--resume", action="store_true", help="Continue from rows already checkpointed in --out.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 2 or args.samples % 2:
        raise SystemExit("--samples must be an even number of at least 2 so every depth has balanced yes/no labels")

    from transformers import AutoTokenizer

    key = api_key()
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    tasks = build_tasks(args.lengths, args.samples, args.seed)
    rows: list[dict] = []
    if args.resume and args.out.exists():
        rows = json.loads(args.out.read_text(encoding="utf-8")).get("qwen_thinking", {}).get("rows", [])
    completed_ids = {row["id"] for row in rows}
    remaining = [task for task in tasks if task.id not in completed_ids]
    model = {
        "model": args.hf_model,
        "openrouter_model": args.model,
        "providers": args.providers,
        "quantization": "bf16",
        "thinking": True,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0,
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def checkpoint() -> None:
        ordered = sorted(rows, key=lambda row: [task.id for task in tasks].index(row["id"]))
        result = {
            "suite": "manager_chain",
            "seed": args.seed,
            "lengths": args.lengths,
            "samples_per_length": args.samples,
            "tasks": [asdict(item) for item in tasks],
            "qwen_thinking": {"model": model, "summary": completed_summary(ordered, args.lengths), "rows": ordered},
        }
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def run(task) -> dict:
        payload = {
            "model": args.model,
            "messages": qwen_prompt(task),
            "max_tokens": args.max_new_tokens,
            "temperature": 0,
            "seed": args.seed,
            "reasoning": {"enabled": True},
            "provider": {"order": args.providers, "allow_fallbacks": False, "quantizations": ["bf16"]},
            "usage": {"include": True},
        }
        started = time.perf_counter()
        data = request(payload, key, args.timeout)
        elapsed = time.perf_counter() - started
        choice = data["choices"][0]
        reasoning = choice["message"].get("reasoning") or ""
        content = choice["message"].get("content") or ""
        truncated = choice.get("finish_reason") == "length"
        # Rebuild Qwen's native layout so the local runner's parser scores both runs identically.
        completion = reasoning if truncated and not content else f"{reasoning}\n</think>\n\n{content}"
        answer = parse_qwen_answer(completion)
        usage = data.get("usage") or {}
        return {
            "id": task.id,
            "depth": task.depth,
            "expected": task.expected,
            "predicted": answer,
            "correct": answer == task.expected,
            "reasoning_tokens": len(tokenizer(reasoning, add_special_tokens=False).input_ids),
            "generated_tokens": usage.get("completion_tokens"),
            "provider_reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "truncated": truncated,
            "finish_reason": choice.get("finish_reason"),
            "provider": data.get("provider"),
            "cost_usd": usage.get("cost"),
            "latency_seconds": elapsed,
            "completion": completion,
        }

    print(f"Running {len(remaining)} remaining of {len(tasks)} tasks on {args.model} via {args.providers} with {args.concurrency} concurrent requests", flush=True)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(run, task): task for task in remaining}
        for future in as_completed(futures):
            task = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # keep the other requests running; --resume retries this task
                print(f"FAILED {task.id}: {exc}", flush=True)
                continue
            with lock:
                rows.append(row)
                checkpoint()
                print(f"{len(rows):>3}/{len(tasks)} {row['id']}: expected={row['expected']} predicted={row['predicted'] or 'PARSE_FAIL'} tokens={row['reasoning_tokens']} {row['provider']} {row['latency_seconds']:.0f}s {'ok' if row['correct'] else 'MISS'}", flush=True)
    print(f"Finished in {time.perf_counter() - started:.0f}s; cost ${sum(row['cost_usd'] or 0 for row in rows):.4f}")
    print("depth  correct  accuracy  median reasoning tokens  truncated  unfinished")
    for record in completed_summary(rows, args.lengths):
        print(f"{record['depth']:>5}  {record['correct']:>3}/{record['total']:<3}  {record['accuracy']:>8.0%}  {record['median_reasoning_tokens']:>23}  {record['truncated']:>9}  {record['unfinished']:>10}")
    missing = len(tasks) - len(rows)
    print(f"Wrote {args.out}" + (f"; {missing} tasks failed, rerun with --resume" if missing else ""))


if __name__ == "__main__":
    main()
