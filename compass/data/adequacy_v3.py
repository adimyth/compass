"""Generate the Stage B v3 adequacy corpus from deterministic, code-derived cases.

The corpus is deliberately separated by domain group: Group A is training only, Group B is assigned to the ordinary evaluation splits by ``splits_v2``, and Group C is written as a generated diagnostic outside the independent shadow suite. Every response is constructed from a checked correct result and, for negative examples, one injected defect. Generated code snippets are executed against discriminating tests before an item is emitted.

    python -m compass.data.adequacy_v3 --out-root .
"""

from __future__ import annotations

import argparse
import calendar
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


GROUPS = {
    "train": [
        "e-commerce orders and returns",
        "SaaS subscriptions and billing",
        "IT helpdesk",
        "logistics and warehousing",
        "HR leave and expenses",
        "software bug triage",
    ],
    "eval": [
        "insurance claims",
        "property rentals and maintenance",
        "restaurant and food delivery",
        "telecom accounts",
        "event ticketing",
        "fleet and vehicle services",
    ],
    "shadow": [
        "hospital administration and scheduling",
        "university course administration",
        "municipal permits and services",
        "travel bookings and airlines",
        "banking disputes and cards",
        "public library and museum services",
    ],
}

COUNTS = {"train": 300, "eval": 60, "shadow": 40}

INSTRUCTIONS = "Read the request and the response. Decide whether the response fully and correctly satisfies the request, including every explicit constraint."
CRITERIA = {
    "true": "The response is complete and correct for the request as stated, even if terse or unusually formatted.",
    "false": "The response contains an error, misses an explicit constraint, or fails to satisfy some part of the request.",
}


@dataclass(frozen=True)
class Rendered:
    request: str
    correct: str
    wrong: str
    rationale: str
    error_kind: str


def _money(value: int) -> str:
    return f"${value / 100:.2f}"


def _slug(text: str) -> str:
    return text.replace(" and ", "-").replace(" ", "-").replace(",", "").lower()


def _line_total(rng: random.Random, domain: str) -> Rendered:
    units = rng.randint(3, 9)
    cents = rng.choice([349, 475, 625, 799, 1249, 1575])
    fee = rng.choice([0, 250, 400, 650])
    total = units * cents + fee
    bad = total + rng.choice([-20, -10, 10, 25])
    subject = rng.choice(["replacement units", "service credits", "storage bins", "meal vouchers"])
    request = f"For a {domain} record, {units} {subject} cost {_money(cents)} each and the fixed processing fee is {_money(fee)}. Show the item subtotal, the fee, and the total due."
    correct = f"Items: {units} × {_money(cents)} = {_money(units * cents)}. Processing fee: {_money(fee)}. Total due: {_money(total)}."
    wrong = f"Items: {units} × {_money(cents)} = {_money(units * cents)}. Processing fee: {_money(fee)}. Total due: {_money(bad)}."
    return Rendered(request, correct, wrong, f"The item subtotal and fee are correct, but {_money(units * cents)} + {_money(fee)} = {_money(total)}, not {_money(bad)}.", "final arithmetic")


def _inclusive_date(rng: random.Random, domain: str) -> Rendered:
    start = date(rng.choice([2025, 2026, 2027]), rng.choice([1, 2, 3, 4, 5, 8, 10]), rng.randint(2, 20))
    days = rng.choice([14, 21, 30, 45])
    end = start + timedelta(days=days - 1)
    wrong = start + timedelta(days=days)
    request = f"A {domain} service window begins on {start.strftime('%d %B %Y')}. It lasts {days} calendar days, with the start date counted as day 1. State the final valid date."
    correct = f"Day 1 is {start.strftime('%d %B %Y')}, so day {days} is {end.strftime('%d %B %Y')}."
    flawed = f"Day 1 is {start.strftime('%d %B %Y')}, so day {days} is {wrong.strftime('%d %B %Y')}."
    return Rendered(request, correct, flawed, f"Counting the start as day 1 means adding {days - 1} days. The final date is {end.strftime('%d %B %Y')}, not {wrong.strftime('%d %B %Y')}.", "inclusive-date off-by-one")


def _unit_conversion(rng: random.Random, domain: str) -> Rendered:
    kg = rng.choice([24, 36, 48, 64, 125, 240])
    grams = kg * 1000
    tonnes = kg / 1000
    request = f"A {domain} shipment record lists a mass of {kg} kg. Express the mass in metric tonnes and in grams."
    correct = f"{kg} kg = {tonnes:g} metric tonnes and {grams:,} g."
    wrong = f"{kg} kg = {tonnes:g} metric tonnes and {grams // 10:,} g."
    return Rendered(request, correct, wrong, f"The tonne conversion is correct, but one kilogram is 1,000 grams. {kg} kg is {grams:,} g, not {grams // 10:,} g.", "unit factor")


def _ordered_unique(rng: random.Random, domain: str) -> Rendered:
    values = rng.sample(range(1, 15), 5)
    source = [values[0], values[1], values[0], values[2], values[3], values[1], values[4], values[2]]
    result = values
    request = f"A {domain} import produced the IDs {source}. Remove duplicates, keep each first occurrence, preserve the original order, and give the resulting list."
    correct = str(result)
    wrong = str(list(reversed(result)))
    return Rendered(request, correct, wrong, f"The distinct IDs are {result} in first-occurrence order. The response sorts them, which violates the explicit order requirement.", "order constraint")


def _rounded_rate(rng: random.Random, domain: str) -> Rendered:
    amount = rng.choice([473, 685, 946, 1275])
    rate = rng.choice([12, 15, 18, 25])
    discount = amount * rate / 100
    final = amount - discount
    request = f"A {domain} invoice is {_money(amount * 100)} before a {rate}% reduction. Compute the reduction and the final invoice amount, each to the cent."
    correct = f"Reduction: {_money(int(round(discount * 100)))}. Final amount: {_money(int(round(final * 100)))}."
    wrong_final = final + 1
    wrong = f"Reduction: {_money(int(round(discount * 100)))}. Final amount: {_money(int(round(wrong_final * 100)))}."
    return Rendered(request, correct, wrong, f"{rate}% of {_money(amount * 100)} is {_money(int(round(discount * 100)))}. Subtracting it gives {_money(int(round(final * 100)))}, not {_money(int(round(wrong_final * 100)))}.", "percentage subtraction")


def _code_edge(rng: random.Random, domain: str) -> Rendered:
    cases = [
        (
            "Write a Python function has_access(role, active) that returns True only when active is True and role is either 'editor' or 'owner'.",
            "def has_access(role, active):\n    return active and role in {'editor', 'owner'}",
            "def has_access(role, active):\n    return role in {'editor', 'owner'}",
            "The function omits the active requirement, so has_access('editor', False) incorrectly returns True.",
        ),
        (
            "Write a Python function clamp(x, low, high) that returns low when x is below low, high when x is above high, and x otherwise.",
            "def clamp(x, low, high):\n    return max(low, min(x, high))",
            "def clamp(x, low, high):\n    return min(low, max(x, high))",
            "The min and max calls are reversed. For x between the bounds, the flawed function returns a bound instead of x.",
        ),
        (
            "Write a Python function is_weekend(day) that returns True for 'Saturday' and 'Sunday' and False for every other day name.",
            "def is_weekend(day):\n    return day in {'Saturday', 'Sunday'}",
            "def is_weekend(day):\n    return day in {'Friday', 'Saturday', 'Sunday'}",
            "Friday is included even though the requested weekend contains only Saturday and Sunday.",
        ),
        (
            "Write a Python function first_or_none(values) that returns the first element of a non-empty list and None for an empty list.",
            "def first_or_none(values):\n    return values[0] if values else None",
            "def first_or_none(values):\n    return values[-1] if values else None",
            "The flawed function returns the last element rather than the requested first element.",
        ),
    ]
    request, correct, wrong, rationale = rng.choice(cases)
    _verify_code_pair(correct, wrong)
    return Rendered(f"For a {domain} automation task: {request}", correct, wrong, rationale, "code edge case")


def _verify_code_pair(correct: str, wrong: str) -> None:
    """Execute every generated code pair on an input that distinguishes the two implementations."""
    correct_ns: dict = {}
    wrong_ns: dict = {}
    exec(correct, correct_ns)
    exec(wrong, wrong_ns)
    if "has_access" in correct_ns:
        assert correct_ns["has_access"]("editor", False) is False
        assert wrong_ns["has_access"]("editor", False) is True
    elif "clamp" in correct_ns:
        assert correct_ns["clamp"](5, 1, 10) == 5
        assert wrong_ns["clamp"](5, 1, 10) == 1
    elif "is_weekend" in correct_ns:
        assert correct_ns["is_weekend"]("Friday") is False
        assert wrong_ns["is_weekend"]("Friday") is True
    elif "first_or_none" in correct_ns:
        assert correct_ns["first_or_none"]([4, 9]) == 4
        assert wrong_ns["first_or_none"]([4, 9]) == 9
    else:
        raise AssertionError("unrecognised code-pair function")


def _constrained_writing(rng: random.Random, domain: str) -> Rendered:
    choices = [
        (
            "Write one sentence of fewer than 16 words telling a customer their {domain} request was received. Do not use the word 'sorry'.",
            "Your {domain} request was received and is now being reviewed.",
            "Sorry, your {domain} request was received and is now being reviewed.",
            "The response is one sentence and short enough, but it uses the explicitly forbidden word 'Sorry'.",
        ),
        (
            "Write exactly two short sentences confirming a {domain} appointment. Mention Tuesday and do not mention a time.",
            "Your {domain} appointment is confirmed for Tuesday. We will send the location details separately.",
            "Your {domain} appointment is confirmed for Tuesday at 10:00. We will send the location details separately.",
            "The response has two sentences and mentions Tuesday, but it adds a time even though the request forbids one.",
        ),
        (
            "Write a subject line of at most 45 characters for a {domain} update. It must include the word 'update' and no exclamation mark.",
            "Update on your {domain} request",
            "Update on your {domain} request!",
            "The subject contains the required word and is short enough, but the exclamation mark is explicitly forbidden.",
        ),
    ]
    request, correct, wrong, rationale = rng.choice(choices)
    return Rendered(request.format(domain=domain), correct.format(domain=domain), wrong.format(domain=domain), rationale, "writing constraint")


def _filter_rule(rng: random.Random, domain: str) -> Rendered:
    rows = [("A", 11, "open"), ("B", 12, "open"), ("C", 13, "open"), ("D", 14, "open"), ("E", 15, "closed")]
    threshold = rng.choice([12, 13, 14])
    keep = [name for name, age, status in rows if age >= threshold and status == "open"]
    bad = [name for name, age, status in rows if age > threshold and status == "open"]
    table = "; ".join(f"{name}: age {age}, {status}" for name, age, status in rows)
    request = f"A {domain} queue contains {table}. List the IDs that are open and at least {threshold} days old, in the listed order."
    correct = ", ".join(keep)
    wrong = ", ".join(bad)
    return Rendered(request, correct, wrong, f"The rule is inclusive: records aged exactly {threshold} days qualify. The correct IDs are {', '.join(keep)}; the response wrongly excludes the boundary case.", "inclusive filter")


def _weekday_deadline(rng: random.Random, domain: str) -> Rendered:
    starts = [date(2026, 1, 5), date(2026, 3, 6), date(2026, 6, 11), date(2026, 10, 30)]
    start = rng.choice(starts)
    n = rng.choice([2, 3, 4])
    cursor, found = start, 0
    while found < n:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            found += 1
    wrong = cursor - timedelta(days=1)
    while wrong.weekday() >= 5:
        wrong -= timedelta(days=1)
    request = f"A {domain} request arrived on {start.strftime('%A %d %B %Y')}. It is due in {n} business days, Monday through Friday only, counting from the next business day after arrival. State the due date."
    correct = cursor.strftime("%A %d %B %Y")
    flawed = wrong.strftime("%A %d %B %Y")
    return Rendered(request, correct, flawed, f"Counting starts on the next business day, not the arrival day. The {n}th business day is {correct}; the response is one business day early.", "business-day anchor")


def _different_question(rng: random.Random, domain: str) -> Rendered:
    count = rng.choice([4, 5, 6, 8])
    rate = rng.choice([125, 160, 245, 375])
    total = count * rate
    request = f"A {domain} record has {count} identical charges of {_money(rate)}. What is the total charge? Give only the total."
    correct = _money(total)
    wrong = f"Each charge is {_money(rate)}."
    return Rendered(request, correct, wrong, f"The response states the per-charge amount, but the request asks for the total. {count} × {_money(rate)} = {_money(total)}.", "answers a different quantity")


def _calendar_month(rng: random.Random, domain: str) -> Rendered:
    year = rng.choice([2024, 2025, 2026, 2028])
    month = rng.choice([2, 4, 6, 9, 11])
    actual = calendar.monthrange(year, month)[1]
    wrong = 29 if month == 2 and actual == 28 else 31 if actual == 30 else 30
    request = f"For a {domain} monthly record, how many calendar days are in {calendar.month_name[month]} {year}?"
    correct = str(actual)
    flawed = str(wrong)
    return Rendered(request, correct, flawed, f"{calendar.month_name[month]} {year} has {actual} days. The response gives {wrong}.", "calendar fact")


PATTERNS = [_line_total, _inclusive_date, _unit_conversion, _ordered_unique, _rounded_rate, _code_edge, _constrained_writing, _filter_rule, _weekday_deadline, _different_question, _calendar_month]


def _make_item(partition: str, domain: str, index: int, expected: str, rng: random.Random) -> dict:
    rendered = PATTERNS[index % len(PATTERNS)](rng, domain)
    response = rendered.correct if expected == "yes" else rendered.wrong
    assert rendered.correct != rendered.wrong
    item_id = f"adequacy-v3-{partition}-{_slug(domain)}-{index:03d}"
    rationale = "The response satisfies every stated condition." if expected == "yes" else rendered.rationale
    return {
        "id": item_id,
        "family": "adequacy",
        "state": {"request": rendered.request, "response": response},
        "question": {"type": "noul", "instructions": INSTRUCTIONS, "criteria": CRITERIA},
        "labels": ["no", "yes"],
        "expected": expected,
        "split": "public",
        "provenance": {
            "source": "compass.data.adequacy_v3",
            "author": "code-verified-generator",
            "form": "record",
            "domain": domain,
            "rationale": rationale,
            "plausible_wrong": "yes" if expected == "no" else "no",
            "why_hard": "The response is fluent and satisfies the visible surface of the request; a careful check of every requirement is needed.",
            "error_catalogue": rendered.error_kind if expected == "no" else "correct-but-suspicious",
        },
    }


def generate(partition: str, n: int, seed: int) -> list[dict]:
    if n % 2:
        raise ValueError("each generated partition must have an even count for an exact yes/no balance")
    rng = random.Random(seed)
    domains = GROUPS[partition]
    items = []
    for index in range(n):
        domain_index = index % len(domains)
        local_index = index // len(domains)
        domain = domains[domain_index]
        # Train and evaluation have an even count per domain. Shadow has four domains with seven cases and two with six; start two of the seven-case domains on yes to preserve the global 20/20 balance.
        starts_yes = partition == "shadow" and domain_index < 2
        expected = "yes" if (local_index % 2 == 0) == starts_yes else "no"
        items.append(_make_item(partition, domain, index, expected, rng))
    assert sum(item["expected"] == "yes" for item in items) == n // 2
    assert len({item["id"] for item in items}) == n
    return items


def _write(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=".")
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    root = Path(args.out_root)
    outputs = {
        "train": root / "dev/authored/train-adequacy2.jsonl",
        "eval": root / "dev/authored/eval-adequacy2.jsonl",
        "shadow": root / "dev/families/adequacy-generated-shadow.jsonl",
    }
    for offset, (partition, output) in enumerate(outputs.items()):
        items = generate(partition, COUNTS[partition], args.seed + offset)
        _write(output, items)
        print(f"{output}: {len(items)} items; yes={sum(i['expected'] == 'yes' for i in items)}; no={sum(i['expected'] == 'no' for i in items)}")


if __name__ == "__main__":
    main()
