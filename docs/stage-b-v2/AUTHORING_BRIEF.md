# Authoring brief for drafted items

You are writing decision items for training and evaluating a small decision model. Each item gives a document (the *state*), a typed question, the allowed answers, and the one correct answer with a rationale. Write in English. Everything needed must be inside the state and question. No real people, companies or products; invent realistic ones. Do not look at or reuse anything from any public benchmark.

## Output: one JSON object per line (JSONL), nothing else in the file

```json
{"id": "<family>-<form>-<domain>-<NNN>",
 "family": "<routing|policy|adequacy|ambiguous|tradeoff|multi_hop|probability|ordinal>",
 "state": "<string>"  OR  {"request": "...", "response": "..."}  OR another JSON object,
 "question": {"type": "noul|choice|score", "instructions": "<1-3 sentences>", "criteria": <see below>},
 "labels": [...],
 "expected": <gold>,
 "split": "public",
 "provenance": {"source": "authored", "author": "<agent name>", "form": "<ticket|record|narrative|dialogue>", "domain": "<domain>",
                "rationale": "<2-6 sentences citing the decisive facts>",
                "plausible_wrong": "<the tempting wrong label, or null>",
                "why_hard": "<one sentence>",
                "gold_probs": {<label>: <p>, ...}   // ONLY for family probability
               }}
```

Question types, exactly:
- `noul`: `labels` exactly `["no", "yes"]`; `expected` `"no"` or `"yes"`; `criteria` `{"true": "<when yes>", "false": "<when no>"}`.
- `choice`: 4–7 labels, lowercase snake_case; `expected` one of them; `criteria` `{label: "one-line definition", ...}` covering every label. Labels must be listed in `labels` in the same order as in `criteria`.
- `score`: `labels` `["0","1",...,"k"]` (3–6 levels); `expected` an integer level index; `criteria` a list of level descriptions, index i describes level i.

Hard rules:
- The state never contains a field named `expected`, `label`, `answer_key` or `ground_truth`.
- **Every wrong option must be plausible under some part of the document.** Name the most tempting one in `plausible_wrong`. If nothing in the document supports an option, rewrite the document so something does, or replace the option.
- **One defensible gold.** If a careful reader could argue for a second label, fix the item or drop it.
- No sentence in the state may point at the trap or say which fact is decisive. Distractors must look like normal case material.
- Balance: for `noul` about half yes; for `choice` do not let the gold sit in the same position most of the time; for `score` cover every level.
- Each item is its own scenario. Do not reuse one template with swapped nouns.
- Vary the language form as assigned: `ticket` (first-person message, informal, with asides), `record` (structured case file: field: value lines, small tables, or a JSON object as state), `narrative` (third-person report or memo), `dialogue` (a short exchange where the decisive turn is not the last one).

## Families

- `routing` (choice, 4–7 handlers): handler descriptions overlap; the message mentions a keyword belonging to the wrong handler as an aside or as context; the decisive fact is what the sender actually asks for. Handlers are teams, tools, models or queues.
- `policy` (noul or choice): a policy excerpt with a rule, an exception, a later amendment that changes one clause, a precedence or definitions clause that alters the meaning of another, and one distractor clause for a neighbouring case type; then a case. The decisive facts sit in different paragraphs. States 300–1,500 tokens.
- `adequacy` (noul): state is `{"request": ..., "response": ...}`; the question is whether the response fully and correctly satisfies the request. Half the responses carry exactly one subtle, decisive error (wrong unit, off-by-one, an explicit constraint not met, a wrong edge case in code, an invalid step in a derivation the request asked to be shown); the other half are fully correct but look suspicious (terse, unusual but valid method, unexpected format that was not forbidden). Requests span arithmetic, unit conversion, short code, list manipulation, date reasoning, short writing with constraints.
- `ambiguous` (choice with a label like `insufficient_information` and a crisp criterion for it): about half are truly undecidable from the state; the rest look undecidable but one overlooked fact in the state decides.
- `tradeoff` (choice): several legitimate goals collide; an explicit precedence order, escalation matrix or SLA ladder in the state decides; the most helpful-sounding action is wrong.
- `multi_hop` (choice or noul): answering needs three or more chained lookups across the state (table → footnote → alias → rule), with one stale or superseded record that must be ruled out.
- `probability` (choice or noul): the state contains countable evidence (a log of comparable past cases with outcomes and a precise definition of "comparable", or stated rates to combine). `gold_probs` is the distribution an ideal reasoner derives exactly from the state; show the arithmetic in the rationale; `expected` is the most likely label; top probability between 0.55 and 0.85; 2–5 options.
- `ordinal` (score): levels defined by conditions on quantities or facts in the document; the rubric says to use the highest fully supported level; the case satisfies some conditions of a higher level but not all.

## Domains

Use only the domains assigned to you. Domains keep splits disjoint, so never drift into another list.

- Group A (train): e-commerce orders and returns, SaaS subscriptions and billing, IT helpdesk, logistics and warehousing, HR leave and expenses, software bug triage.
- Group B (selection / calibration / release): insurance claims, property rentals and maintenance, restaurant and food delivery, telecom accounts, event ticketing, fleet and vehicle services.
- Group C (shadow): hospital administration and scheduling, university course administration, municipal permits and services, travel bookings and airlines, banking disputes and cards, public library and museum services.
