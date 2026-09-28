"""Run the manager-chain suite through a running Compass HTTP server.

This is the endpoint-timing counterpart to ``run_jev_manager_chain.py``. It
uses the same generated tasks and client-side clock, so Compass's locally
served CUDA timing can be reported alongside the hosted Jev endpoint.

    uv run python scripts/run_compass_manager_chain.py \
      --endpoint http://127.0.0.1:8000 \
      --out dev/results/manager-chain-compass-4090-YYYY-MM-DD.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

from manager_chain_eval import build_tasks, compass_question, summary


def evaluate(endpoint: str, task, timeout: float) -> tuple[dict, float]:
    payload = {
        "model": "compass-latest",
        "state": task.state,
        "questions": {"chain": compass_question(task)},
    }
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/systemone",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "compass-manager-chain-endpoint-eval/0.2"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Compass request failed with HTTP {error.code}: {body[:500]}") from error
    return json.loads(body), time.perf_counter() - started


def completed_summary(rows: list[dict], lengths: list[int]) -> list[dict]:
    completed_depths = [depth for depth in lengths if any(row["depth"] == depth for row in rows)]
    return summary(rows, completed_depths, include_reasoning=False)


def write_checkpoint(out: Path, tasks, args, rows: list[dict]) -> None:
    checkpoint = {
        "suite": "manager_chain",
        "seed": args.seed,
        "lengths": args.lengths,
        "samples_per_length": args.samples,
        "tasks": [asdict(item) for item in tasks],
        "compass": {
            "model": {"endpoint": args.endpoint, "returned": rows[-1]["model"] if rows else None},
            "summary": completed_summary(rows, args.lengths),
            "rows": rows,
        },
    }
    out.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Continue from rows already checkpointed in --out.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 2 or args.samples % 2:
        raise SystemExit("--samples must be an even number of at least 2 so every depth has balanced yes/no labels")
    if any(depth < 1 for depth in args.lengths):
        raise SystemExit("--lengths must contain positive integers")

    tasks = build_tasks(args.lengths, args.samples, args.seed)
    for _ in range(args.warmup):
        evaluate(args.endpoint, tasks[0], args.timeout)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if args.resume and args.out.exists():
        rows = json.loads(args.out.read_text(encoding="utf-8")).get("compass", {}).get("rows", [])
    completed_ids = {row["id"] for row in rows}
    remaining = [task for task in tasks if task.id not in completed_ids]
    print(f"Running {len(remaining)} remaining of {len(tasks)} balanced tasks against {args.endpoint}", flush=True)
    for completed, task in enumerate(remaining, start=len(rows) + 1):
        response, elapsed = evaluate(args.endpoint, task, args.timeout)
        answer = response["answers"]["chain"]
        probability_yes = float(answer["noul"])
        predicted = "yes" if probability_yes >= 0.5 else "no"
        row = {
            "id": task.id,
            "depth": task.depth,
            "expected": task.expected,
            "predicted": predicted,
            "correct": predicted == task.expected,
            "probability_yes": probability_yes,
            "latency_seconds": elapsed,
            "model": response.get("model", "compass-latest"),
            "usage": response.get("usage", {}),
        }
        rows.append(row)
        write_checkpoint(args.out, tasks, args, rows)
        print(f"Compass {completed:>3}/{len(tasks)} {task.id}: expected={task.expected} predicted={predicted} p_yes={probability_yes:.3f} {elapsed * 1000:.0f}ms {'ok' if row['correct'] else 'MISS'}", flush=True)

    elapsed = sorted(row["latency_seconds"] for row in rows)
    if elapsed:
        p50 = statistics.median(elapsed)
        p95 = elapsed[min(len(elapsed) - 1, int(len(elapsed) * 0.95))]
        print(f"Latency: p50={p50 * 1000:.0f}ms p95={p95 * 1000:.0f}ms; wrote {args.out}")


if __name__ == "__main__":
    main()
