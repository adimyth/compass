"""Check a running Compass server against JevBench's own adapter and validator.

It sends synthetic tasks written here, one per question type, through JevBench's unchanged `typesafe` adapter, then scores each answer with JevBench's `score_task`. It uses no JevBench items, so it can run as often as needed without touching benchmark data (SPEC.md §8).

    git clone https://github.com/fstandhartinger/jevbench /path/to/jevbench
    python -m compass.server --port 8000 &
    python scripts/jevbench_compat.py --jevbench /path/to/jevbench --endpoint http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import sys

SYNTHETIC_TASKS = [
    {
        "id": "compat-choice",
        "family": "routing",
        "state": "Hi, the export button in the reports tab throws a 500 error since this morning's update.",
        "question": {
            "type": "choice",
            "instructions": "Which team should handle this message?",
            "criteria": {
                "billing": "Charges, invoices, refunds",
                "engineering": "Bugs, errors, outages",
                "sales": "Pricing, upgrades, new accounts",
                "account_management": "Renewals and contract questions",
            },
        },
        "labels": ["billing", "engineering", "sales", "account_management"],
        "expected": "engineering",
    },
    {
        "id": "compat-noul",
        "family": "policy",
        "state": "Refund request for order 5521, placed 41 days ago. Policy: refunds within 30 days of purchase.",
        "question": {
            "type": "noul",
            "instructions": "Is this order eligible for a refund under the stated policy?",
            "criteria": {"true": "The request meets the policy", "false": "The request does not meet the policy"},
        },
        "labels": ["no", "yes"],
        "expected": "no",
    },
    {
        "id": "compat-score",
        "family": "ordinal",
        "state": "Checkout fails for every customer paying by card. No data has been lost.",
        "question": {
            "type": "score",
            "instructions": "Rate the incident's impact from the reported facts only.",
            "criteria": ["Cosmetic only", "Some users affected, workaround exists", "Core function blocked for many users", "Data loss or harm"],
        },
        "labels": ["0", "1", "2", "3"],
        "expected": 2,
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jevbench", required=True, help="path to a clone of fstandhartinger/jevbench")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="compass-latest")
    args = parser.parse_args()

    sys.path.insert(0, args.jevbench)
    from jevbench.adapters.typesafe import TypeSafeAdapter
    from jevbench.scoring import score_task
    from jevbench.tasks import Task

    # key_env="" sends no Authorization header, as JevBench does for a submitter's endpoint.
    adapter = TypeSafeAdapter(endpoint=args.endpoint, model=args.model, key_env="")
    failures = 0
    for spec in SYNTHETIC_TASKS:
        task = Task.from_dict({**spec, "split": "public"})
        result = adapter.run(task)
        if not result.ok:
            print(f"FAIL {task.id}: adapter rejected the answer: {result.error}")
            failures += 1
            continue
        scored = score_task(result.probs, task)
        ok = scored["valid"] and scored["strict_valid"]
        failures += not ok
        print(
            f"{'ok  ' if ok else 'FAIL'} {task.id}: strict_valid={scored['strict_valid']} predicted={scored['predicted']} "
            f"correct={scored['correct']} model={result.model} input_tokens={result.usage.get('input_tokens')} latency={result.latency_s * 1000:.1f} ms"
        )
    print(f"{len(SYNTHETIC_TASKS) - failures}/{len(SYNTHETIC_TASKS)} answers valid under JevBench's own adapter and validator")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
