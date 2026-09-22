# Compass plan: two days

Decisions are in [STRATEGY.md](STRATEGY.md). The scope is Stage A: frozen Qwen3.5-4B, our verification readout, per-type temperature calibration fitted on our own items, a container and a bench request. Training (Stage B) is out of scope and listed at the end as follow-up.

What this ships is the same size class as SemIf (#2, 73.1) and reflex (#5, 70.3), with a different readout. The target is a ranked row in the top 10; the projection at Intelligence 75–79 and Calibration 70–75 is 70–73.

## Day 1: calibrate and check

| Step | Output | Done when |
| --- | --- | --- |
| Dev items | `dev/families/`: the temporal generator (60) plus a multi-hop generator (60) and the 18 authored items. All through `scripts/validate_items.py` with the JevBench overlap check. | Done: 138 items, 0 errors, 0 overlap. |
| Calibration | `scripts/fit_calibration.py` fits one temperature per question type on the dev items, writes `release/calibration.json`. | Done: T = 1.5 / 3.0 / 2.5 (choice / score / noul); held-out ECE 0.19→0.16, 0.27→0.14, 0.24→0.09. |
| Final check | JevBench's public items (231), once, through `scripts/public_check.py` (JevBench's adapter and scorer) against the server on this Mac. Reported as is, never tuned on. | Done (public items only, preliminary): easy 100 %, standard 80.6 %, hard 55.9 %, 231/231 well-formed, hard ECE 0.089. Projected score 70–71, unverified until the maintainer's run. `dev/results/public-check.md`. |

## Day 2: package and submit

| Step | Output | Done when |
| --- | --- | --- |
| Pin | Qwen3.5-4B revision hash, prompt version, calibration file and package version fixed in `compass/release.py`. | `/v1/models` reports them. |
| Container | `Dockerfile` (CUDA, fused linear-attention kernels), `scripts/serve.sh`, no network after the weights are cached, no credentials. | `jevbench_compat.py` passes against the container. |
| Public repository | Code, docs, dev items, results. Apache-2.0. | Pushed, tagged `v0.1.0`. |
| Bench request | Issue on `fstandhartinger/jevbench`: what it is, serve command, pinned revision, our public-item numbers, cost basis (Qwen3.5-4B hosted price, one pass, no generation), request for an evaluator-run GPU run. | Issue open. |

## Rules

- No JevBench items in any dev or calibration file. The public items run once, at the end.
- Every reported number names the revision, prompt version, calibration file and hardware.
- Latency is measured by the maintainer on their GPU; our Mac numbers are not quoted.

## Revalidation (22 Sep, evening)

Re-examined before spending GPU time. Three things change the order of work.

1. **The readout has not been A/B-tested.** Verification was chosen on principle, never against the answer-letter readout every strong 4B row uses. The public check gives a warning sign: standard tier 80.6 %, where the frozen 4B rows sit at 95–98 % on the same kind of items. Verifying each option in isolation can say "plausible" to several options; a direct "which one?" read contrasts them. Training on top of a readout that loses 15 points on easy classification would waste the run. So the first GPU job is a readout comparison on our own items (138 existing plus a fresh routing/policy generator): verification, direct option-key readout, and a combination of the two in log space. The winner on our items becomes the release readout.
2. **Training is not a sure gain.** On the board, the best 4B row is frozen (SemIf 73.1); the LoRA-trained 4B (reflex) is below it. Trained small models also lose calibration (kev 4B/8B). A synthetic-only corpus can overfit its templates. Training moves to a gated experiment after a release-worthy frozen system exists: head-only first (cheap, low risk), LoRA second, each accepted only if it beats the frozen system on held-out families.
3. **The judge tier is the biggest unknown in the projection.** It is 28 % of Intelligence, 146 answer-adequacy noul items, not public, with an 82 % majority-class floor. Our nearest evidence is adequacy 9/12 and judge_hard 10/17. If our noul readout leans "no", the tier could land far below the 80 % the projection assumes. The routing/policy generator gets an adequacy generator beside it (responses with one planted error) so the readout comparison covers this family.

Adopted gates (23 Sep):

1. GPU: validate the container, CUDA latency, and cached-versus-uncached score parity.
2. Frozen-readout comparison on newly generated internal data: candidate verification; direct fixed-symbol readout (A/B/C mapped to candidates in canonical order); a fixed log-space fusion. Each readout also reports its flip rate under reversed option order; flips count against it.
3. One internal **selection** split chooses the readout and fusion weight. A separate **release** split stays untouched.
4. Calibration is fitted only after that choice is frozen, on a dedicated **calibration** split.
5. The chosen frozen configuration is evaluated once on the untouched release split.
6. JevBench public items run once for that frozen release configuration and are reported without further tuning (the earlier prompt-2 run stays in the record beside it).
7. Stage B only as a separate gated experiment, on a fourth **train** split disjoint from the other three: head-only first; LoRA plus head only if head-only is promising; promoted only if it improves the untouched release evaluation without hurting calibration, robustness or latency; if promoted, it is a new release with one final public-items result.
8. Submit only when authorised.

Status 22 Sep, late: gates 1–6 done on an RTX 4090 (`dev/results/gate*`). Readout chosen: fusion-0.5 (selection split 65.2 % vs verify 62.0 %). Calibration fitted. Release split 60.1 %. Public items for the release configuration: easy 100 %, standard 80.6 %, hard 56.8 %. Release `compass-0.1.0` pinned and tagged `v0.1.0`. Not submitted.

Gate 7, head-only (23 Sep): residual head trained on the train split (2,700 items), early-stopped on selection (loss 0.74 → 0.57, accuracy 65.2 % → 71.7 %); calibration fitted on the calibration split. On the untouched release split it beat the frozen release on every count (67.5 % vs 60.1 %, ECE 0.052 vs 0.061, same flips, same latency). On the public items it did not transfer: standard 81.9 % (+1 item), hard 55.9 % (−1 item), and hard-tier ECE 0.231 against 0.105, TVD 0.264 against 0.248. The calibration fitted on our generator-built split does not carry over to hand-written hard items, and Calibration is a quarter of the score: projected 67.6 against 70.9 for the frozen release. **Not promoted.** `compass-0.1.0` stays the release. The head and its records stay in `release/head-v1/` and `dev/results/gate7-*`, `2026-09-22-public-check-head-v1.jsonl`. What would have to change before another attempt: a calibration split with authored hard items rather than generator items, and a loss that penalises over-confidence on unseen families. LoRA is not attempted, since the condition for it (head-only promising) is not met on external data.

Splits are made by disjoint seeds and templates; the authored items are spread across selection, calibration and release so the release split is not generator-only.

## Follow-up, after the row exists

Stage B (LoRA plus verification head on our own corpus) targeting long-policy, multi-hop, probability and ambiguous items, where Jev beats the frozen 4B systems by 20–40 points. Temporal arithmetic is not a target: every no-generation system on the board scores 20–33 % there.
