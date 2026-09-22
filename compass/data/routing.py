"""Routing, policy and answer-adequacy decisions with computed labels.

Routing items pair a message with 4–6 handler descriptions that overlap on purpose; the message mentions a keyword from the wrong handler and the decisive fact is what the sender asks for, which the generator controls. Policy items apply a rule with an exception and an amendment to a case. Adequacy items pair a request with a response that is either correct or carries one planted error (wrong unit, off-by-one, unmet explicit constraint), and the generator knows which. Seeds decide everything.

    python -m compass.data.routing --n 90 --seed 3 --out dev/families/routing.jsonl
"""

from __future__ import annotations

import argparse
import json
import random

HANDLERS = {
    "billing": "Charges, invoices, refunds, payment methods and subscription changes",
    "technical": "Bugs, errors, outages, API and integration problems",
    "sales": "Pricing questions, quotes, upgrades and new accounts",
    "account": "Login, password, profile, permissions and account closure",
    "shipping": "Delivery status, addresses, damaged or missing parcels",
    "legal": "Contracts, data requests, compliance and privacy",
}
# Each ask names the handler that decides it and a decoy keyword from another handler that appears in the message.
ASKS = [
    ("billing", "refund the duplicate charge on my last invoice", "technical", "the app crashed"),
    ("billing", "change the card on file before the next renewal", "account", "logged in"),
    ("technical", "fix the API returning 502 on every request since the deploy", "billing", "we pay for the Pro plan"),
    ("technical", "explain why exports time out for large reports", "sales", "upgrade"),
    ("sales", "send a quote for 40 seats on the annual plan", "billing", "invoice"),
    ("sales", "compare the Team and Enterprise plans for a new deployment", "technical", "SSO integration"),
    ("account", "reset the password for our admin user who left", "legal", "compliance"),
    ("account", "remove a former colleague's access to the workspace", "billing", "still being charged"),
    ("shipping", "tell me where order 5521 is; it was due yesterday", "billing", "paid in full"),
    ("shipping", "replace the monitor that arrived with a cracked panel", "technical", "screen flickers"),
    ("legal", "sign a data processing agreement before we onboard", "sales", "purchase"),
    ("legal", "delete all personal data under our GDPR request", "account", "close my account"),
]
OPENERS = ["Hi,", "Hello team,", "Quick one:", "Hi there —", "Support request.", "Good morning,"]


def gen_routing(rng: random.Random, i: int) -> dict:
    handler, ask, decoy_handler, decoy = rng.choice(ASKS)
    others = [h for h in HANDLERS if h not in (handler, decoy_handler)]
    chosen = [handler, decoy_handler] + rng.sample(others, rng.randint(2, 4))
    frame = rng.choice([
        f"{rng.choice(OPENERS)} I mentioned to a colleague that {decoy}, but that is not why I am writing. What I need is for you to {ask}. Thanks.",
        f"{rng.choice(OPENERS)} Context: {decoy}. Separate from that, could you {ask}? That is the only thing I need today.",
        f"{rng.choice(OPENERS)} Please {ask}. (Unrelated aside: {decoy}, but someone else is handling that.)",
    ])
    return {
        "id": f"routing-{i:03d}",
        "family": "routing",
        "state": frame,
        "question": {"type": "choice", "instructions": "Which team should handle what the sender is asking for?", "criteria": {h: HANDLERS[h] for h in chosen}},
        "labels": sorted(chosen),
        "expected": handler,
        "split": "public",
        "provenance": {"source": "compass.data.routing", "generator": "routing", "rationale": f"the ask is '{ask}' ({handler}); '{decoy}' is an aside pointing at {decoy_handler}"},
    }


def gen_policy(rng: random.Random, i: int) -> dict:
    """Rule + exception + amendment, applied to a case; the generator evaluates the conditions."""
    limit = rng.choice([30, 45, 60])
    days = rng.choice([limit - 10, limit - 1, limit, limit + 1, limit + 15])
    category = rng.choice(["electronics", "furniture", "clothing", "books"])
    exception_cat = rng.choice(["electronics", "furniture"])
    opened = rng.choice([True, False])
    amended = rng.choice([True, False])
    amendment_limit = limit + 30
    effective_limit = amendment_limit if (amended and category == exception_cat) else limit
    blocked_by_opened = opened and category == "electronics"
    eligible = days <= effective_limit and not blocked_by_opened
    state = (
        f"Returns policy, section 4: a full refund is available for returns requested within {limit} days of delivery. "
        f"Section 4.2 (exception): opened electronics are not refundable. "
        + (f"Amendment 7 (effective 1 March): for {exception_cat}, the period in section 4 is {amendment_limit} days. " if amended else "No amendments are in force. ")
        + f"\n\nCase: {category}, delivered {rng.randint(1, 20)} May, return requested {days} days after delivery, packaging {'opened' if opened else 'sealed'}."
    )
    return {
        "id": f"policy-{i:03d}",
        "family": "policy",
        "state": state,
        "question": {"type": "noul", "instructions": "Is the case eligible for a full refund under the policy as it stands?", "criteria": {"true": "Eligible under the applicable period and no exclusion applies", "false": "Outside the applicable period or excluded"}},
        "labels": ["no", "yes"],
        "expected": "yes" if eligible else "no",
        "split": "public",
        "provenance": {"source": "compass.data.routing", "generator": "policy", "rationale": f"period {effective_limit}, days {days}, opened-electronics exclusion {'applies' if blocked_by_opened else 'does not apply'}"},
    }


def gen_adequacy(rng: random.Random, i: int) -> dict:
    """Request + response; half the responses carry one planted error."""
    kind = rng.choice(["convert", "count", "constraint", "arith"])
    if kind == "convert":
        km = rng.choice([3, 5, 8, 12, 21])
        miles = km * 0.621371
        wrong = rng.choice([round(miles, 1) + 0.5, round(km * 1.609, 2)])
        request = f"Convert {km} kilometres to miles, rounded to two decimal places."
        good = f"{km} km × 0.621371 = {miles:.6f}, which rounds to {miles:.2f} miles."
        bad = f"{km} km × 0.621371 = {miles:.6f}, which rounds to {wrong:.2f} miles."
    elif kind == "count":
        start, n = rng.randint(3, 20), rng.randint(3, 9)
        request = f"How many integers are there from {start} to {start + n} inclusive?"
        good = f"From {start} to {start + n} inclusive is {start + n} − {start} + 1 = {n + 1} integers."
        bad = f"From {start} to {start + n} inclusive is {start + n} − {start} = {n} integers."
    elif kind == "constraint":
        words = rng.choice([5, 6, 8])
        request = f"Describe the ocean in exactly {words} words."
        good = " ".join(["Vast", "blue", "restless", "deep", "cold", "salty", "endless", "quiet"][:words]) + "."
        bad = " ".join(["Vast", "blue", "restless", "deep", "cold", "salty", "endless", "quiet"][:words + 1]) + "."
    else:
        a, b = rng.randint(12, 98), rng.randint(12, 98)
        request = f"What is {a} × {b}? Show the working."
        good = f"{a} × {b} = {a} × {b // 10 * 10} + {a} × {b % 10} = {a * (b // 10 * 10)} + {a * (b % 10)} = {a * b}."
        bad = f"{a} × {b} = {a} × {b // 10 * 10} + {a} × {b % 10} = {a * (b // 10 * 10)} + {a * (b % 10)} = {a * b + rng.choice([-10, 10, 100])}."
    correct = rng.choice([True, False])
    return {
        "id": f"adequacy-{i:03d}",
        "family": "adequacy",
        "state": {"request": request, "response": good if correct else bad},
        "question": {"type": "noul", "instructions": "Does the response fully and correctly satisfy the request?", "criteria": {"true": "Complete and correct", "false": "Incomplete or contains an error"}},
        "labels": ["no", "yes"],
        "expected": "yes" if correct else "no",
        "split": "public",
        "provenance": {"source": "compass.data.routing", "generator": "adequacy", "rationale": "correct response" if correct else f"planted error of kind {kind}"},
    }


GENERATORS = [gen_routing, gen_policy, gen_adequacy]


def generate(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [GENERATORS[i % len(GENERATORS)](rng, i) for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=90)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.out, "w", encoding="utf-8") as fh:
        for item in generate(args.n, args.seed):
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
