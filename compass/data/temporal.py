"""Temporal and numeric decisions with computed labels.

Each item is a small case file (a policy line, a few dated events, a threshold) and a typed question whose answer follows from date arithmetic that the generator performs. Business days, month lengths, inclusive versus exclusive windows, ">=" versus ">" and cumulative limits are the mechanisms; every item mixes at least two. Seeds decide everything, so a file regenerates byte for byte.

    python -m compass.data.temporal --n 60 --seed 1 --out dev/families/temporal.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random

NAMES = ["Ada", "Ben", "Chloe", "Dev", "Esme", "Farid", "Greta", "Hugo", "Ines", "Jonas", "Kira", "Leo", "Mara", "Nils", "Omar", "Pia"]
PRODUCTS = ["a standing desk", "noise-cancelling headphones", "a coffee grinder", "a monitor arm", "hiking boots", "an e-reader", "a tent", "a road bike helmet"]
COMPANIES = ["Northwind Supply", "Aster Goods", "Brightline Retail", "Orbit Outfitters", "Fennel & Co"]


def business_days_after(start: dt.date, days: int) -> dt.date:
    d = start
    while days > 0:
        d += dt.timedelta(days=1)
        if d.weekday() < 5:
            days -= 1
    return d


def fmt(d: dt.date) -> str:
    return d.strftime("%-d %B %Y")


def gen_return_window(rng: random.Random, i: int) -> dict:
    """Refund eligibility: window counted in calendar or business days, from delivery or from purchase, inclusive or not; an amendment may change the window for one category."""
    name, product, company = rng.choice(NAMES), rng.choice(PRODUCTS), rng.choice(COMPANIES)
    purchase = dt.date(2026, rng.randint(1, 8), rng.randint(1, 28))
    delivery = purchase + dt.timedelta(days=rng.randint(2, 9))
    window = rng.choice([14, 21, 30, 45])
    unit = rng.choice(["calendar", "business"])
    anchor = rng.choice(["purchase", "delivery"])
    inclusive = rng.choice([True, False])
    anchor_date = purchase if anchor == "purchase" else delivery
    deadline = business_days_after(anchor_date, window) if unit == "business" else anchor_date + dt.timedelta(days=window)
    # The request lands within a few days of the deadline on either side, so the arithmetic decides.
    request = deadline + dt.timedelta(days=rng.randint(-3, 3))
    if request <= delivery:
        request = delivery + dt.timedelta(days=1)
    ok = request <= deadline if inclusive else request < deadline
    boundary = "on or before" if inclusive else "before"
    policy = (
        f"{company} returns policy. A refund is available if the return request is made {boundary} the day that falls {window} {unit} days after the {anchor} date. "
        f"Business days exclude Saturdays and Sundays."
    )
    case = (
        f"Order record. Customer: {name}. Item: {product}. Purchased: {fmt(purchase)}. Delivered: {fmt(delivery)}. "
        f"Return request received: {fmt(request)}."
    )
    filler = rng.choice([
        "Note: the customer also asked about a different order last month; that order is not part of this request.",
        "The item is unused and in its original packaging.",
        "A gift receipt was included with the delivery.",
    ])
    return {
        "id": f"temporal-return-{i:03d}",
        "family": "temporal_numeric",
        "state": f"{policy}\n\n{case}\n{filler}",
        "question": {
            "type": "noul",
            "instructions": "Is the return request within the refund window under the stated policy?",
            "criteria": {"true": "The request date is inside the window", "false": "The request date is outside the window"},
        },
        "labels": ["no", "yes"],
        "expected": "yes" if ok else "no",
        "split": "public",
        "provenance": {"source": "compass.data.temporal", "generator": "return_window", "rationale": f"deadline {fmt(deadline)} ({window} {unit} days after {anchor}, {boundary}); request {fmt(request)}"},
    }


def gen_cumulative_limit(rng: random.Random, i: int) -> dict:
    """Which tier applies after a running total crosses thresholds, with >= versus > stated in the rubric."""
    name, company = rng.choice(NAMES), rng.choice(COMPANIES)
    t1, t2 = rng.choice([(500, 1500), (1000, 2500), (250, 750), (2000, 5000)])
    strict = rng.choice([True, False])
    n = rng.randint(3, 6)
    amounts = [rng.randint(50, 900) for _ in range(n)]
    # Nudge the total to sit on or right next to a threshold half the time.
    if rng.random() < 0.5:
        target = rng.choice([t1, t2]) + rng.choice([-1, 0, 1])
        amounts[-1] = max(1, target - sum(amounts[:-1]))
    total = sum(amounts)
    cmp = (lambda a, b: a > b) if strict else (lambda a, b: a >= b)
    tier = "gold" if cmp(total, t2) else "silver" if cmp(total, t1) else "standard"
    op = "more than" if strict else "at least"
    months = ["January", "February", "March", "April", "May", "June"]
    lines = [f"  {months[k]}: ${a:,}" for k, a in enumerate(amounts)]
    state = (
        f"{company} loyalty tiers, evaluated on the customer's cumulative spend in the year to date. "
        f"Standard: below the silver threshold. Silver: {op} ${t1:,}. Gold: {op} ${t2:,}. A refund issued in a month is not deducted from spend.\n\n"
        f"Spend record for {name} (all purchases, year to date):\n" + "\n".join(lines) + "\n"
        f"A refund of ${rng.randint(20, 120)} was issued in {months[rng.randrange(n)]}."
    )
    return {
        "id": f"temporal-tier-{i:03d}",
        "family": "temporal_numeric",
        "state": state,
        "question": {
            "type": "choice",
            "instructions": "Which loyalty tier does the customer qualify for today?",
            "criteria": {"standard": "Cumulative spend below the silver threshold", "silver": f"Cumulative spend {op} the silver threshold but not the gold threshold", "gold": f"Cumulative spend {op} the gold threshold"},
        },
        "labels": ["standard", "silver", "gold"],
        "expected": tier,
        "split": "public",
        "provenance": {"source": "compass.data.temporal", "generator": "cumulative_limit", "rationale": f"total {total}; thresholds {t1}/{t2}; {'strict' if strict else 'inclusive'}"},
    }


def gen_sla_breach(rng: random.Random, i: int) -> dict:
    """Ordinal: how late a response was against an SLA measured in business hours across a weekend."""
    company = rng.choice(COMPANIES)
    sla_h = rng.choice([4, 8, 24, 48])
    opened = dt.datetime(2026, rng.randint(1, 9), rng.randint(1, 28), rng.randint(8, 17), rng.choice([0, 15, 30, 45]))
    if opened.weekday() >= 5:
        opened += dt.timedelta(days=7 - opened.weekday())
    # Business hours 09:00-17:00 Mon-Fri. Compute the deadline by walking business hours.
    def add_business_hours(start: dt.datetime, hours: float) -> dt.datetime:
        cur = start
        remaining = hours * 60
        while remaining > 0:
            if cur.weekday() >= 5 or cur.hour >= 17:
                cur = (cur + dt.timedelta(days=1)).replace(hour=9, minute=0)
                continue
            if cur.hour < 9:
                cur = cur.replace(hour=9, minute=0)
                continue
            end_of_day = cur.replace(hour=17, minute=0)
            step = min(remaining, (end_of_day - cur).total_seconds() / 60)
            cur += dt.timedelta(minutes=step)
            remaining -= step
        return cur
    deadline = add_business_hours(opened, sla_h)
    late_by = rng.choice([-2, -0.5, 0, 0.5, 1.5, 3, 6, 12])
    responded = add_business_hours(deadline, late_by) if late_by > 0 else deadline + dt.timedelta(hours=late_by)
    lateness_h = 0 if responded <= deadline else late_by
    level = 0 if lateness_h <= 0 else 1 if lateness_h <= 2 else 2 if lateness_h <= 8 else 3
    f = lambda d: d.strftime("%A %-d %B %Y, %H:%M")
    state = (
        f"{company} support SLA: first response within {sla_h} business hours of the ticket being opened. Business hours are 09:00-17:00, Monday to Friday. "
        f"Time outside business hours does not count.\n\nTicket 20{i:03d}. Opened: {f(opened)}. First response sent: {f(responded)}."
    )
    return {
        "id": f"temporal-sla-{i:03d}",
        "family": "temporal_numeric",
        "state": state,
        "question": {
            "type": "score",
            "instructions": "Classify the SLA outcome for this ticket, counting only business hours.",
            "criteria": ["Met: the response was within the SLA", "Minor breach: late by up to 2 business hours", "Breach: late by more than 2 and up to 8 business hours", "Severe breach: late by more than 8 business hours"],
        },
        "labels": ["0", "1", "2", "3"],
        "expected": level,
        "split": "public",
        "provenance": {"source": "compass.data.temporal", "generator": "sla_breach", "rationale": f"deadline {f(deadline)}; late by {lateness_h} business hours"},
    }


GENERATORS = [gen_return_window, gen_cumulative_limit, gen_sla_breach]


def generate(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [GENERATORS[i % len(GENERATORS)](rng, i) for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.out, "w", encoding="utf-8") as fh:
        for item in generate(args.n, args.seed):
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
