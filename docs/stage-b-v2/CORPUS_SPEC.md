# Stage B v2 corpus specification

Written before any item exists (23 September 2026). The corpus trains a LoRA on the frozen Qwen3.5-4B through Compass's existing verification and direct readouts; nothing about the readouts, fusion, calibration or serving changes. Stage B v1 failed because its data was uniform generator output and the model learned the generators ([docs/EXPERIMENTS.md](../../docs/EXPERIMENTS.md), gate 7). This spec exists so that v2's data cannot fail the same way, and so that its splits are designed rather than discovered.

## 1. What every item must satisfy

1. **Hard contrast.** Every wrong candidate is plausible under some part of the document. A wrong option that nothing in the document supports is not allowed; if a generator cannot produce a plausible wrong option for a candidate slot, that slot is filled with a distractor that the document explicitly rules out rather than a random option.
2. **One defensible gold.** A careful reader with time gets the gold with ≥ 95 % agreement. Items where a second label is equally defensible are dropped, not fixed by fiat.
3. **Self-contained.** Everything needed is in state plus question. No outside knowledge, no real people or companies.
4. **No JevBench material.** `scripts/validate_items.py --jevbench` (8-gram overlap against the public files) passes on every file before it is used for anything.
5. **Gold provenance.** Each item records how the gold was determined: `computed` (a generator evaluated the conditions), `authored+checked` (written with a rationale, then blind-answered by a second model that agreed), or `authored+human` (shadow suite only).
6. **Balance.** Per family: noul golds within 40–60 % yes; choice golds never in the same sorted position more than 35 % of the time; score golds cover every level.

## 2. Families and what makes them hard

| Family | Type | Contrast mechanism | Gold basis | Train share |
| --- | --- | --- | --- | --- |
| `routing` | choice, 4–7 options | Handler descriptions overlap; the message contains a keyword of the wrong handler as an aside; the decisive fact is what is asked for | authored+checked | 18 % |
| `policy` | noul, choice | Rule + exception + amendment + precedence clause + one distractor clause for a neighbouring case; decisive facts in different paragraphs | computed (conditions evaluated by code over an authored template) | 20 % |
| `multi_hop` | choice, noul | Three or more lookups across records, one stale record to rule out, one alias | computed | 15 % |
| `adequacy` | noul | Request + response; half the responses carry exactly one subtle, decisive error (unit, off-by-one, unmet explicit constraint, wrong edge case in code, invalid step in a shown derivation); the other half are correct but look unusual (terse, valid but uncommon method) | authored+checked, error planted by code where possible | 20 % |
| `ambiguous` | choice with an `insufficient_information` label | Half truly undecidable; half look undecidable but one overlooked fact decides | authored+checked | 7 % |
| `tradeoff` | choice | Several legitimate goals collide; an explicit precedence order in the document decides; the most helpful-sounding action is wrong | authored+checked | 5 % |
| `probability` | choice, noul | Countable evidence in the state (case logs, stated rates) fixes an exact distribution; option counts vary 2–5; top probability 0.55–0.85 | computed, `gold_probs` derived from the counts | 8 % |
| `ordinal` | score, 3–6 levels | Levels defined by conditions on quantities in the document; "highest fully supported level" rubrics | computed | 7 % |

Temporal arithmetic is not a training family. Every no-generation system on the board is at 20–33 % there and it is a regression check only (`dev/families/temporal.jsonl`).

## 3. Language forms

Each family is written in at least three **language forms**, and forms are what splits are made on (section 5):

- `ticket`: first-person customer or employee message, informal, with asides.
- `record`: structured case file, tables, field: value lines, JSON objects as state.
- `narrative`: third-person prose incident report or memo.
- `dialogue` (routing, adequacy, ambiguous only): a short exchange; the decisive turn is not the last one.

A template is one (family, form, scenario skeleton). Two items from one template share structure but not entities, numbers, or wording of the decisive clause.

## 4. Sizes

| Split | Items | Purpose |
| --- | --- | --- |
| train | 6,000 | LoRA training |
| selection | 400 | fusion weight, early stopping, any design choice |
| calibration | 400 | temperatures, fitted after every other choice is frozen |
| release | 500 | one evaluation of the chosen configuration |
| shadow | 300 | human-authored, never used for anything but the promotion decision ([SHADOW_SUITE_PROTOCOL.md](SHADOW_SUITE_PROTOCOL.md)) |

Authored items (sections 6–7) make up at least 15 % of train and at least 40 % of selection, calibration and release. The shadow suite is 100 % authored.

## 5. Splits

Splits are by **template family and language form**, never by seed alone:

- Every template belongs to exactly one split. A template used in train never appears in selection, calibration, release or shadow.
- Each family contributes at least one **language form held out entirely from train** and used only in release and shadow, so release measures transfer across forms, not across seeds.
- Authored scenarios are assigned to a split by scenario, and an author (model or person) never contributes to both train and shadow.
- `compass/data/splits.py` is rewritten to take a manifest of templates with their split, and refuses to build if any template id appears in two splits.

## 6. Authoring pipeline (authored+checked items)

1. **Draft.** An LLM we control drafts items from a family brief (section 2), one scenario at a time, with a rationale naming the decisive facts and, for choice, the tempting wrong answer.
2. **Code check where possible.** Policy, multi-hop, probability and ordinal items are drafted as parameterised templates, and the gold is recomputed by code from the parameters; a draft whose stated gold disagrees with the computed gold is dropped.
3. **Blind review.** A second model sees state, question and labels only and answers. Then it sees the gold and rationale and gives accept / reject with a reason. Reject when the gold is wrong, a second label is defensible, outside knowledge is needed, or the state leaks the answer.
4. **One discussion round.** Rejected items go back to the drafting model with the objection: defend, fix or drop. The reviewer gives a final verdict. Not accepted means dropped.
5. **Blind-review misses on accepted items are kept.** They are evidence of difficulty, not error.

Drafting and reviewing models are recorded per item. Neither is a benchmark entrant.

## 7. Provenance record

`data/PROVENANCE.md` lists, per source: generator module and commit, or drafting and reviewing model and date; item count; splits it feeds; the overlap-check result; licence (Apache-2.0 for everything we produce). It is committed before the first training run.

## 8. Validation before use

`scripts/validate_items.py` gains checks for: contrast (choice items must carry `provenance.plausible_wrong` naming the tempting wrong option), balance per family and split, template-split uniqueness, and per-form counts. A file that fails any check is not used.
