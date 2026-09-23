# Shadow suite protocol

The shadow suite is the external test that Stage B v1 lacked. It answers one question: does a candidate system judge hand-written decisions better than `compass-0.1.1`, or has it learned our generators? It is used for nothing else.

## Rules

1. **Human-authored.** Every item is written by a person, not drafted by a model. A model may be used afterwards only for the blind check in step 4.
2. **Never used for training, readout selection, prompt design, fusion weights, calibration, or early stopping.** It is read by one script, `scripts/shadow_eval.py`, which runs a named configuration once and writes a dated report. Each configuration is run on it at most once.
3. **Never committed in plain text to the repository history that the training code reads.** It lives in `shadow/` which is listed in the training loader's deny-list, and its file hash is recorded in the report so a run names what it measured.
4. **Disjoint by author and scenario** from every authored training or selection item.

## Contents (300 items)

| Family | Items | Notes |
| --- | --- | --- |
| routing | 50 | 4–7 overlapping handlers, decoy keyword in the message |
| policy | 60 | amendment, exception, precedence, distractor clause; states 400–2,000 tokens |
| adequacy | 60 | half with one subtle decisive error, half correct but unusual |
| multi_hop | 40 | three or more lookups, one stale record |
| ambiguous | 30 | half truly undecidable |
| tradeoff | 20 | explicit precedence order decides |
| probability | 20 | count-derived `gold_probs`, 2–5 options |
| ordinal | 20 | 3–6 levels, "highest fully supported level" |

Language forms: about a third each of `ticket`, `record` and `narrative`, with `dialogue` items inside routing and adequacy. Golds balanced as in the corpus spec.

## Authoring steps

1. The author writes state, question, labels, gold, a two-to-six-sentence rationale citing the decisive facts, and for choice items the tempting wrong option.
2. The author does not look at any JevBench file while writing. The overlap check runs afterwards anyway.
3. A second person (or, if unavailable, a model that is not a benchmark entrant) blind-answers every item, then sees the gold and rationale and gives accept / reject with a reason.
4. Rejected items are fixed by the author or dropped. Blind misses on accepted items are kept.
5. `scripts/validate_items.py --jevbench` must pass. The file hash is recorded.

## Promotion rule for Stage B v2

A candidate is promoted over `compass-0.1.1` only if, on the shadow suite, it has higher accuracy **and** lower top-label ECE on the policy, multi_hop, adequacy and ambiguous items taken together, **and** passes the standing checks (all answers well-formed, zero option-order flips, latency and tokens within 10 % of the release, cache parity against uncached scoring). A candidate that wins on the internal release split but not on the shadow suite is not promoted, whatever the internal margin.

If promoted, it becomes `compass-0.2.0` and gets one public-items run, reported as is.

## What we expect to learn even if v2 is not promoted

The shadow suite is also the first honest measurement of `compass-0.1.1` on hand-written judgement items outside JevBench, per family and per language form. That report by itself tells us which families the frozen model misjudges and whether the public standard-tier gap (80.6 % against SemIf's 97.9 %) is a prompt problem or a model problem.
