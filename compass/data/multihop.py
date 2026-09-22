"""Multi-hop and policy decisions with computed labels.

Each item is a short case file made of a few records that must be chained: a request names a person, a roster maps the person to a team and a plan, a plan table gives an allowance, an amendment overrides one line of the table, and the question asks whether the request is within the allowance or which handler applies. Every item needs three lookups; some include a superseded record that must be ruled out. Labels are computed by the generator.

    python -m compass.data.multihop --n 60 --seed 2 --out dev/families/multihop.jsonl
"""

from __future__ import annotations

import argparse
import json
import random

NAMES = ["Ada Kessler", "Ben Okafor", "Chloe Marin", "Dev Raman", "Esme Lund", "Farid Haddad", "Greta Novak", "Hugo Reyes", "Ines Costa", "Jonas Berg", "Kira Sato", "Leo Brandt"]
TEAMS = ["Platform", "Payments", "Growth", "Support", "Data"]
PLANS = ["Core", "Plus", "Premier"]
CATEGORIES = ["travel", "equipment", "training", "software"]


def gen_allowance(rng: random.Random, i: int) -> dict:
    """Is a claim within the allowance? Person -> team -> plan -> allowance, with an amendment that changes one plan's allowance for one category, and a stale roster line."""
    people = rng.sample(NAMES, 4)
    teams = rng.sample(TEAMS, 4)
    team_plan = {t: rng.choice(PLANS) for t in teams}
    table = {p: {c: rng.choice([200, 300, 500, 750, 1000, 1500]) for c in CATEGORIES} for p in PLANS}
    subject = people[0]
    subject_team = teams[0]
    category = rng.choice(CATEGORIES)
    plan = team_plan[subject_team]
    # Amendment: changes the subject's plan for the claimed category half the time, another plan otherwise.
    amend_plan = plan if rng.random() < 0.5 else rng.choice([p for p in PLANS if p != plan])
    amend_cat = category if rng.random() < 0.7 else rng.choice([c for c in CATEGORIES if c != category])
    new_value = rng.choice([v for v in [200, 300, 500, 750, 1000, 1500] if v != table[amend_plan][amend_cat]])
    effective = table[plan][category]
    if amend_plan == plan and amend_cat == category:
        effective = new_value
    amount = effective + rng.choice([-150, -40, -1, 0, 1, 60, 200])
    amount = max(amount, 10)
    within = amount <= effective
    # Stale roster line: the subject moved teams; the earlier line is superseded.
    old_team = teams[1]
    roster = [f"  {people[k]} - {teams[k]} (since {rng.choice(['January', 'March'])} 2026)" for k in range(1, 4)]
    roster.insert(rng.randrange(0, 4), f"  {subject} - {old_team} (until May 2026)")
    roster.insert(rng.randrange(0, 5), f"  {subject} - {subject_team} (since June 2026)")
    lines = [
        "Expense allowances, per person per quarter, by plan:",
        "  plan      " + "  ".join(f"{c:>9}" for c in CATEGORIES),
    ] + [f"  {p:9} " + "  ".join(f"${table[p][c]:>8,}" for c in CATEGORIES) for p in PLANS]
    state = (
        "\n".join(lines) + "\n\n"
        f"Amendment A-{rng.randint(10, 99)} (effective 1 July 2026): the {amend_cat} allowance for the {amend_plan} plan is changed to ${new_value:,}.\n\n"
        "Team plans: " + ", ".join(f"{t} is on {team_plan[t]}" for t in teams) + ".\n\n"
        "Roster (current as of August 2026):\n" + "\n".join(roster) + "\n\n"
        f"Claim submitted 12 August 2026: {subject}, {category}, ${amount:,}. No other {category} claims this quarter."
    )
    return {
        "id": f"multihop-allow-{i:03d}",
        "family": "multi_hop",
        "state": state,
        "question": {
            "type": "noul",
            "instructions": "Is the claim within the claimant's current allowance for that category?",
            "criteria": {"true": "The amount is at or below the allowance that applies", "false": "The amount exceeds the allowance that applies"},
        },
        "labels": ["no", "yes"],
        "expected": "yes" if within else "no",
        "split": "public",
        "provenance": {"source": "compass.data.multihop", "generator": "allowance", "rationale": f"{subject} -> {subject_team} -> {plan}; {category} allowance {effective} (amendment {'applies' if effective != table[plan][category] else 'does not apply'}); claim {amount}"},
    }


def gen_escalation(rng: random.Random, i: int) -> dict:
    """Which handler takes an alert? Service -> owner team -> on-call rota for the weekday -> override for one date."""
    services = rng.sample(["billing-api", "search", "ingest", "auth", "reports", "webhooks"], 4)
    teams = rng.sample(TEAMS, 4)
    owner = dict(zip(services, teams))
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    rota = {t: {d: rng.choice(NAMES) for d in days} for t in teams}
    service = services[0]
    day = rng.choice(days)
    date = f"{rng.randint(3, 28)} September 2026"
    team = owner[service]
    scheduled = rota[team][day]
    override = rng.random() < 0.5
    override_person = rng.choice([n for n in NAMES if n != scheduled])
    override_team = team if rng.random() < 0.7 else rng.choice([t for t in teams if t != team])
    override_date = date if rng.random() < 0.7 else f"{rng.randint(3, 28)} October 2026"
    handler = override_person if override and override_team == team and override_date == date else scheduled
    options = {handler, scheduled, override_person}
    while len(options) < 4:
        options.add(rng.choice(NAMES))
    keys = {n: n.split()[0].lower() + "_" + n.split()[1].lower() for n in options}
    state = (
        "Service ownership: " + ", ".join(f"{s} is owned by {owner[s]}" for s in services) + ".\n\n"
        "On-call rota (primary, by weekday):\n" + "\n".join(f"  {t}: " + ", ".join(f"{d} {rota[t][d]}" for d in days) for t in teams) + "\n\n"
        + (f"Override: on {override_date}, {override_person} covers primary on-call for {override_team} instead of the rota.\n\n" if override else "No overrides are recorded this month.\n\n")
        + f"Alert: {service} error rate above threshold. Fired {day} {date}, 14:20."
    )
    return {
        "id": f"multihop-oncall-{i:03d}",
        "family": "multi_hop",
        "state": state,
        "question": {
            "type": "choice",
            "instructions": "Who is the primary on-call handler for this alert?",
            "criteria": {keys[n]: f"{n} handles the alert" for n in sorted(options)},
        },
        "labels": sorted(keys[n] for n in options),
        "expected": keys[handler],
        "split": "public",
        "provenance": {"source": "compass.data.multihop", "generator": "escalation", "rationale": f"{service} -> {team} -> {day} {scheduled}; override {'applies' if handler != scheduled else 'does not apply'}"},
    }


def gen_severity_policy(rng: random.Random, i: int) -> dict:
    """Ordinal: incident severity from a matrix of affected users x duration, with a definition that reclassifies one product as core."""
    products = rng.sample(["checkout", "invoicing", "search", "dashboard", "exports", "notifications"], 3)
    core = products[:1] + ([products[1]] if rng.random() < 0.5 else [])
    product = rng.choice(products)
    users = rng.choice([3, 40, 400, 3000])
    minutes = rng.choice([5, 25, 90, 300])
    is_core = product in core
    # Level rules: 0 minor, 1 moderate, 2 major, 3 critical.
    level = 0
    if users >= 100 or minutes >= 60:
        level = 1
    if (users >= 1000 and minutes >= 20) or (is_core and minutes >= 60):
        level = 2
    if is_core and users >= 1000 and minutes >= 60:
        level = 3
    state = (
        "Severity policy.\n"
        "Definitions: a 'core' product is one listed in section 3. 'Affected users' counts distinct users who saw an error.\n"
        "Section 2, levels:\n"
        "  Minor: fewer than 100 affected users and under 60 minutes.\n"
        "  Moderate: at least 100 affected users, or 60 minutes or longer.\n"
        "  Major: at least 1,000 affected users for 20 minutes or longer; or any core product degraded for 60 minutes or longer.\n"
        "  Critical: a core product with at least 1,000 affected users for 60 minutes or longer.\n"
        "The highest level whose conditions are met applies.\n"
        f"Section 3, core products: {', '.join(core)}.\n\n"
        f"Incident INC-{rng.randint(1000, 9999)}: {product} returned errors for {minutes} minutes; {users:,} distinct users saw an error. "
        + rng.choice(["No data was lost.", "The on-call engineer rolled back the deploy.", "A status page notice was posted."])
    )
    return {
        "id": f"multihop-severity-{i:03d}",
        "family": "multi_hop",
        "state": state,
        "question": {
            "type": "score",
            "instructions": "Which severity level applies under the policy?",
            "criteria": ["Minor", "Moderate", "Major", "Critical"],
        },
        "labels": ["0", "1", "2", "3"],
        "expected": level,
        "split": "public",
        "provenance": {"source": "compass.data.multihop", "generator": "severity_policy", "rationale": f"{product} core={is_core}, users {users}, minutes {minutes} -> level {level}"},
    }


GENERATORS = [gen_allowance, gen_escalation, gen_severity_policy]


def generate(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [GENERATORS[i % len(GENERATORS)](rng, i) for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.out, "w", encoding="utf-8") as fh:
        for item in generate(args.n, args.seed):
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
