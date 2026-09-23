"""Stage B v2 code-computed generators: policy with contrast, probability, ordinal.

Every generator takes a language form (`record` or `narrative`) so that splits can hold a form out of training (CORPUS_SPEC.md §5). Every wrong option is supported by some clause of the document; `provenance.plausible_wrong` names the tempting one. Labels are computed, never written.

    python -m compass.data.v2 --generator policy --form record --n 100 --seed 5 --out out.jsonl
"""

from __future__ import annotations

import argparse
import json
import random

ORGS = ["Halden Mutual", "Norwick Council", "Brightwater Clinic", "Ferris Logistics", "Ashgrove University", "Pellican Air", "Marlow Bank", "Cedar Library"]
NAMES = ["Ada", "Ben", "Chloe", "Dev", "Esme", "Farid", "Greta", "Hugo", "Ines", "Jonas", "Kira", "Leo", "Mara", "Nils", "Omar", "Pia"]


def _fmt_days(n: int) -> str:
    return f"{n} day{'s' if n != 1 else ''}"


def gen_policy(rng: random.Random, i: int, form: str) -> dict:
    """A claim window with an exception, an amendment, a definitions clause and a distractor clause. The case is placed so that each of the four clauses could decide it, and the gold follows from applying them in the stated precedence."""
    org = rng.choice(ORGS)
    base = rng.choice([14, 30, 45, 60])
    amend = base + rng.choice([15, 30])
    category = rng.choice(["equipment", "travel", "medical", "tuition", "vehicle"])
    amend_cat = rng.choice([category, rng.choice(["equipment", "travel", "medical", "tuition", "vehicle"])])
    excl_trigger = rng.choice(["a prior claim on the same item in the last 12 months", "a receipt dated after the incident date", "an incident reported by a third party rather than the claimant"])
    excl_present = rng.random() < 0.3
    anchor = rng.choice(["incident", "report"])
    incident_to_report = rng.randint(0, 6)
    # "Definitions" decides which date is day zero; the distractor clause covers a neighbouring category with a different window.
    neighbour = rng.choice([c for c in ["equipment", "travel", "medical", "tuition", "vehicle"] if c != category])
    neighbour_window = rng.choice([7, 10, 90])
    window = amend if amend_cat == category else base
    days_from_incident = window + rng.choice([-12, -6, -3, -1, 0, 1, 2, 5, 10])
    days_from_report = days_from_incident - incident_to_report
    days_counted = days_from_incident if anchor == "incident" else days_from_report
    in_window = 0 <= days_counted <= window
    eligible = in_window and not excl_present
    if excl_present and not in_window:
        wrong = "yes"  # tempting because a reader who misses both may still say yes
    wrong = "no" if eligible else "yes"
    clauses = [
        f"Section 2 (Definitions). 'Filing date' means the date the claim form is received. 'Day zero' is the {anchor} date. Days are calendar days.",
        f"Section 4 (Windows). A {category} claim is accepted if the filing date is no more than {_fmt_days(base)} after day zero. A {neighbour} claim is accepted if the filing date is no more than {_fmt_days(neighbour_window)} after day zero.",
        f"Section 4.3 (Exclusion). A claim is refused, whatever the window, where there is {excl_trigger}.",
        f"Amendment 3 (in force since 1 April). For {amend_cat} claims the period in Section 4 is {_fmt_days(amend)}. Amendment 3 does not alter Section 4.3.",
        "Section 9 (Precedence). Where an amendment and a section conflict, the amendment applies. Section 4.3 applies over every window.",
    ]
    rng.shuffle(clauses)
    name = rng.choice(NAMES)
    facts = {
        "category": category, "incident_date": "3 June", "report_date": f"{3 + incident_to_report} June",
        "filing_date": f"{3 + days_from_incident} June" if 3 + days_from_incident <= 30 else f"{3 + days_from_incident - 30} July",
        "exclusion_note": (f"File note: {excl_trigger}." if excl_present else f"File note: no {excl_trigger.split(' ')[0]} {excl_trigger.split(' ', 1)[1].split(' in ')[0]} on record."),
    }
    if form == "record":
        state = f"{org} claims policy (excerpt)\n" + "\n".join(clauses) + "\n\nClaim file\n" + "\n".join(f"  {k}: {v}" for k, v in facts.items()) + f"\n  claimant: {name}"
    else:
        state = (f"{org} claims policy (excerpt). " + " ".join(clauses) + f"\n\n{name} filed a {category} claim. The incident happened on {facts['incident_date']}, it was reported on {facts['report_date']}, and the claim form was received on {facts['filing_date']}. {facts['exclusion_note']}")
    return {
        "id": f"policy-{form}-gen-{i:04d}",
        "family": "policy",
        "state": state,
        "question": {"type": "noul", "instructions": "Is the claim accepted under the policy as it stands, applying the stated precedence?", "criteria": {"true": "The claim is accepted", "false": "The claim is refused"}},
        "labels": ["no", "yes"],
        "expected": "yes" if eligible else "no",
        "split": "public",
        "provenance": {"source": "compass.data.v2", "generator": "policy", "form": form, "template": f"policy-{form}", "plausible_wrong": wrong,
                       "rationale": f"day zero = {anchor} date; window {window} days ({'amended' if amend_cat == category else 'base'}); counted {days_counted} days -> {'in' if in_window else 'out of'} window; exclusion {'present' if excl_present else 'absent'}"},
    }


def gen_probability(rng: random.Random, i: int, form: str) -> dict:
    """A log of comparable cases with outcomes and a precise comparability rule; the gold distribution is the count ratio among comparable cases only."""
    org = rng.choice(ORGS)
    outcomes = rng.sample(["approved", "declined", "escalated", "withdrawn", "deferred"], rng.choice([2, 3, 4]))
    n_cases = rng.choice([20, 30, 40])
    attr = rng.choice([("channel", ["phone", "web"]), ("tier", ["basic", "premium"]), ("region", ["north", "south"])])
    target_val = attr[1][0]
    rows, counts = [], {o: 0 for o in outcomes}
    for k in range(n_cases):
        val = rng.choice(attr[1])
        # Make the target group's distribution peaked so the top probability lands in 0.55-0.85.
        weights = [0.7] + [0.3 / (len(outcomes) - 1)] * (len(outcomes) - 1) if val == target_val else [1 / len(outcomes)] * len(outcomes)
        out = rng.choices(outcomes, weights=weights)[0]
        rows.append((k + 1, val, out))
        if val == target_val:
            counts[out] += 1
    total = sum(counts.values())
    top = sorted(counts.values(), reverse=True)
    if total == 0 or top[0] / total < 0.5 or top[0] / total > 0.9 or (len(top) > 1 and top[0] == top[1]):
        return gen_probability(rng, i, form)
    gold = {o: counts[o] / total for o in outcomes}
    expected = max(outcomes, key=lambda o: (gold[o], o))
    second = sorted(outcomes, key=lambda o: -gold[o])[1]
    table = "\n".join(f"  {k:3}  {attr[0]}={v:8}  outcome={o}" for k, v, o in rows)
    rule = f"Comparable cases are those with {attr[0]} = {target_val}; other cases are not comparable and must be ignored."
    if form == "record":
        state = f"{org} case log, last {n_cases} cases\n{table}\n\nComparability rule: {rule}\nNew case: {attr[0]} = {target_val}."
    else:
        state = f"{org} reviewed its last {n_cases} cases. " + "; ".join(f"case {k} ({attr[0]} {v}) was {o}" for k, v, o in rows) + f". By policy, {rule[0].lower() + rule[1:]} A new case has {attr[0]} {target_val}."
    return {
        "id": f"probability-{form}-gen-{i:04d}",
        "family": "probability",
        "state": state,
        "question": {"type": "choice", "instructions": "What is the most likely outcome for the new case? Give probabilities that reflect the comparable cases in the log.", "criteria": {o: f"The new case is {o}" for o in outcomes}},
        "labels": list(outcomes),
        "expected": expected,
        "split": "public",
        "provenance": {"source": "compass.data.v2", "generator": "probability", "form": form, "template": f"probability-{form}", "plausible_wrong": second,
                       "gold_probs": gold, "rationale": f"comparable cases: {total}; counts {counts}"},
    }


def gen_ordinal(rng: random.Random, i: int, form: str) -> dict:
    """Levels defined by conjunctions of conditions; the case satisfies some conditions of the level above the gold, which is the tempting wrong level."""
    org = rng.choice(ORGS)
    users = rng.choice([5, 20, 60, 250, 1200, 4000, 4000])
    minutes = rng.choice([10, 20, 30, 75, 150, 150, 400])
    data_loss = rng.random() < 0.3
    core = rng.random() < 0.6
    level = 0
    if users >= 50 or minutes >= 30:
        level = 1
    if (users >= 200 and minutes >= 60) or (core and minutes >= 120):
        level = 2
    if users >= 1000 and minutes >= 120 and core:
        level = 3
    if data_loss and level >= 2:
        level = 4
    levels = [
        "Minor: fewer than 50 affected users and under 30 minutes",
        "Moderate: at least 50 affected users, or 30 minutes or longer",
        "Major: at least 200 affected users for 60 minutes or longer; or a core service degraded for 120 minutes or longer",
        "Critical: a core service with at least 1,000 affected users for 120 minutes or longer",
        "Catastrophic: any Major or Critical incident with confirmed data loss",
    ]
    service = rng.choice(["appointment booking", "payment gateway", "search", "notifications", "reporting"])
    core_list = [service] if core else [s for s in ["appointment booking", "payment gateway"] if s != service][:1]
    facts = f"Incident {rng.randint(1000, 9999)}: {service} degraded for {minutes} minutes; {users:,} distinct users affected; data loss {'confirmed' if data_loss else 'not found'}. Core services (section 3): {', '.join(core_list)}."
    if form == "record":
        state = f"{org} severity policy\nSection 2, levels (the highest level whose conditions are all met applies):\n" + "\n".join(f"  {k}. {t}" for k, t in enumerate(levels)) + f"\n\n{facts}"
    else:
        state = f"{org}'s severity policy defines five levels, and the highest level whose conditions are all met applies: " + " ".join(f"Level {k} is {t}." for k, t in enumerate(levels)) + f" {facts}"
    return {
        "id": f"ordinal-{form}-gen-{i:04d}",
        "family": "ordinal",
        "state": state,
        "question": {"type": "score", "instructions": "Which severity level applies under the policy?", "criteria": [t.split(":")[0] for t in levels]},
        "labels": [str(k) for k in range(5)],
        "expected": level,
        "split": "public",
        "provenance": {"source": "compass.data.v2", "generator": "ordinal", "form": form, "template": f"ordinal-{form}", "plausible_wrong": str(min(level + 1, 4)),
                       "rationale": f"users {users}, minutes {minutes}, core {core}, data loss {data_loss} -> level {level}"},
    }


GENERATORS = {"policy": gen_policy, "probability": gen_probability, "ordinal": gen_ordinal}


def generate(generator: str, form: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [GENERATORS[generator](rng, i, form) for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", choices=sorted(GENERATORS), required=True)
    parser.add_argument("--form", choices=("record", "narrative"), required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.out, "w", encoding="utf-8") as fh:
        for item in generate(args.generator, args.form, args.n, args.seed):
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
