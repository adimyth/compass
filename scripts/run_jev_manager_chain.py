"""Run the manager-chain suite against TypeSafe's hosted Jev model.

The script uses the same generated tasks and Noul question as
``scripts/manager_chain_eval.py``. It reads ``TYPESAFE_AI_KEY`` (or the
official ``TYPESAFE_API_KEY`` spelling) from the process environment or the
local ``.env`` file. It never prints either value.

    uv run python scripts/run_jev_manager_chain.py \
      --out dev/results/manager-chain-jev-2026-09-27.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

from manager_chain_eval import build_tasks, compass_question, summary


API_URL = "https://api.typesafe.ai/v1/systemone"


def load_api_key(env_file: Path) -> str:
    """Read the API key without adding it to command output or result files."""
    for name in ("TYPESAFE_AI_KEY", "TYPESAFE_API_KEY"):
        if value := os.environ.get(name):
            return value
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() in {"TYPESAFE_AI_KEY", "TYPESAFE_API_KEY"}:
                return value.strip().strip('"').strip("'")
    raise SystemExit("Set TYPESAFE_AI_KEY or TYPESAFE_API_KEY in the environment or .env.")


def evaluate(api_key: str, task, model: str, timeout: float) -> tuple[dict, float]:
    payload = {
        "model": model,
        "state": task.state,
        "questions": {"chain": compass_question(task)},
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "compass-manager-chain-eval/0.2",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Jev request failed with HTTP {error.code}: {body[:500]}") from error
    return json.loads(body), time.perf_counter() - started


def completed_summary(rows: list[dict], lengths: list[int]) -> list[dict]:
    """Summarise only depths whose rows have reached the local checkpoint."""
    completed_depths = [depth for depth in lengths if any(row["depth"] == depth for row in rows)]
    return summary(rows, completed_depths, include_reasoning=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--resume", action="store_true", help="Continue from rows already checkpointed in --out.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 2 or args.samples % 2:
        raise SystemExit("--samples must be an even number of at least 2 so every depth has balanced yes/no labels")
    if any(depth < 1 for depth in args.lengths):
        raise SystemExit("--lengths must contain positive integers")

    api_key = load_api_key(args.env_file)
    tasks = build_tasks(args.lengths, args.samples, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if args.resume and args.out.exists():
        checkpoint = json.loads(args.out.read_text(encoding="utf-8"))
        rows = checkpoint.get("jev", {}).get("rows", [])
    completed_ids = {row["id"] for row in rows}
    remaining = [task for task in tasks if task.id not in completed_ids]
    print(f"Running {len(remaining)} remaining of {len(tasks)} balanced tasks against {args.model}: lengths={args.lengths}, samples per length={args.samples}")
    for completed, task in enumerate(remaining, start=len(rows) + 1):
        response, elapsed = evaluate(api_key, task, args.model, args.timeout)
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
            "model": response.get("model", args.model),
            "usage": response.get("usage", {}),
        }
        rows.append(row)
        checkpoint = {
            "suite": "manager_chain",
            "seed": args.seed,
            "lengths": args.lengths,
            "samples_per_length": args.samples,
            "tasks": [asdict(item) for item in tasks],
            "jev": {
                "model": {"requested": args.model, "returned": row["model"]},
                "summary": completed_summary(rows, args.lengths),
                "rows": rows,
            },
        }
        args.out.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
        print(f"Jev {completed:>3}/{len(tasks)} {task.id}: expected={task.expected} predicted={predicted} p_yes={probability_yes:.3f} {'ok' if row['correct'] else 'MISS'}")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
