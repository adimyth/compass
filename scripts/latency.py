"""Serial latency of a running Compass server at two input sizes, the way JevBench measures it: one request at a time, wall time on the caller.

    python scripts/latency.py --endpoint http://127.0.0.1:8000 --n 100
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

SHORT = "Order 7731. Placed 9 September. Payment authorised, not yet captured. Shipment: label created, awaiting carrier pickup. Customer asked whether the parcel has left the warehouse."
PARA = (
    "Section {i}. Requests received after the cut-off are processed on the next business day unless the customer holds a priority agreement, in which case the request is processed the same day provided it arrived before 16:00. "
    "Priority agreements are listed in appendix B and expire at the end of the contract year. A request that cites an expired agreement is treated as standard. "
)
QUESTION = {"type": "choice", "instructions": "How should this request be processed?", "criteria": {"same_day": "Processed the same day", "next_business_day": "Processed on the next business day", "standard": "Standard processing", "needs_clarification": "The document does not decide"}}


def call(endpoint: str, state: str) -> tuple[float, int]:
    body = json.dumps({"model": "compass-latest", "state": state, "questions": {"decision": QUESTION}}).encode()
    req = urllib.request.Request(f"{endpoint}/v1/systemone", data=body, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    return time.perf_counter() - t0, out["usage"]["input_tokens"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    long_state = "".join(PARA.format(i=i) for i in range(1, 19))  # about 1,400 tokens, the hard-tier mean
    for name, state in (("short", SHORT), ("long", long_state)):
        for _ in range(5):
            call(args.endpoint, state)  # warm-up
        lat, tok = zip(*(call(args.endpoint, state) for _ in range(args.n)))
        s = sorted(lat)
        print(f"{name}: input_tokens={tok[0]} n={args.n} p50={s[len(s) // 2] * 1000:.0f} ms p95={s[int(len(s) * 0.95)] * 1000:.0f} ms max={s[-1] * 1000:.0f} ms")


if __name__ == "__main__":
    main()
