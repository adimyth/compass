"""Run the authored Scope Check trace against a live local Compass endpoint.

This is a trace-parity probe, not an AgentDojo benchmark. It records the model's typed decision, full distribution, token count, and caller-observed wall time.
"""

from __future__ import annotations

import argparse
import json

from compass.client import DEFAULT_ENDPOINT, ask
from compass.scope_check import CASES, _model_event, build_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Compass on the authored Scope Check trace.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="local Compass /v1/systemone endpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mismatches = 0
    for case in CASES:
        response, latency_ms = ask(build_request(case), args.endpoint)
        event = _model_event(case, response, latency_ms)
        matched = event["scope"] == case.expected_scope
        mismatches += not matched
        print(
            json.dumps(
                {
                    "case": case.id,
                    "expected_scope": case.expected_scope,
                    "scope": event["scope"],
                    "confidence": event["confidence"],
                    "probabilities": event["probabilities"],
                    "input_tokens": event["input_tokens"],
                    "wall_ms": event["latency_ms"],
                    "matched": matched,
                },
                sort_keys=True,
            )
        )
    if mismatches:
        raise SystemExit(f"Scope Check trace mismatch: {mismatches}/{len(CASES)}")


if __name__ == "__main__":
    main()
