# Public-items check (preliminary, public items only)

## Release configuration `compass-0.2.0` (fusion-0.5 + LoRA `adimyth/compass-lora-v2` @ bfa8af07), 23 September 2026, RTX 4090

One run through JevBench's adapter and scorer (`v2/public-check-lora-v2.jsonl`), calibration fitted for the adapter on the hard-like split (T = 2.0 / 2.0 / 2.25).

| Tier | Items | Accuracy | Well-formed answers | Top-label ECE | Mean input tokens |
| --- | --- | --- | --- | --- | --- |
| easy | 48 | 100.0 % (48/48) | 48/48 | 0.034 | 390 |
| standard (original) | 72 | 87.5 % (63/72) | 72/72 | 0.137 | 393 |
| hard | 111 | 60.4 % (67/111) | 111/111 | 0.069 | 1,528 |

Hard tier by family: trap 8/8, routing_hard 5/5, adversarial 6/6, multi_hop 13/18, judge_hard 11/17, long_policy 9/19, probability 6/10 (TVD 0.216), ambiguous 5/7, tradeoff 3/6, temporal_numeric 1/15. Standard by family: extraction 12/12, ordinal 12/12, policy 12/12, intent 11/12, adequacy 8/12, routing 8/12.

**Projection, not a measurement:** Calibration 82.3, Speed 87.9, Cost 57.7, Intelligence 67–71 for a judge tier of 75–85 %, JevBench Score about 72.7–73.8. The adequacy family regressed on our shadow suite (65 → 52.5 %) and the judge tier is made of it; that assumption is unverified.


## Release configuration `compass-0.1.1` (fusion-0.5, hard-like calibration), 23 September 2026, RTX 4090

Qwen3.5-4B revision `851bf6e8`, frozen, bf16; readout: log-space fusion (weight 0.5) of candidate verification and the direct symbol readout, chosen on the internal selection split (`gate2-selection.log`); `release/calibration.json` fitted on the hard-like calibration split (T = 1.25 / 2.0 / 1.5; docs/EXPERIMENTS.md). Served on a RunPod RTX 4090 with flash-linear-attention. One run through JevBench's `typesafe` adapter and `score_task`; not tuned on. Per-item records: `2026-09-23-public-check-fusion-calhard.jsonl` (the `compass-0.1.0` run with the earlier calibration, `2026-09-22-public-check-fusion.jsonl`, has identical accuracy and hard ECE 0.105).

| Tier | Items | Accuracy | Well-formed answers | Top-label ECE | Mean input tokens | p50 (localhost, 4090) |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 48 | 100.0 % (48/48) | 48/48 | 0.014 | 390 | 0.121 s |
| standard (original) | 72 | 80.6 % (58/72) | 72/72 | 0.049 | 393 | 0.121 s |
| hard | 111 | 56.8 % (63/111) | 111/111 | 0.107 | 1,528 | 0.145 s |

Hard tier by family: trap 8/8, routing_hard 5/5, judge_hard 11/17, long_policy 11/19, multi_hop 10/18, probability 6/10 (mean TVD to gold distributions 0.25), ambiguous 5/7, adversarial 4/6, tradeoff 2/6, temporal_numeric 1/15. Standard tier by family: extraction 12/12, ordinal 12/12, intent 10/12, adequacy 9/12, policy 8/12, routing 7/12.

Internal release split (366 untouched items, `gate5-release.log`): 60.1 % overall, ECE 0.061, zero option-order flips.

**Projection, not a measurement.** With `jevbench/composite_v13.py`, the measured 4090 latency (×2 + 0.15 s adjustment) and the $0.03/M hosted price: Calibration 77.1, Speed 87.9, Cost 57.7, Intelligence 62–66 for an assumed judge tier of 75–85 %, projected JevBench Score about 70–71. The judge tier, the 109 held-out hard items and the maintainer's own latency remain unknown.

## Stage B candidate head-v1 (not promoted), 23 September 2026, RTX 4090

Same backbone and readouts with `release/head-v1` (residual head) and `release/head-v1/calibration.json`. One run, `2026-09-22-public-check-head-v1.jsonl`. easy 100 % (48/48), standard 81.9 % (59/72), hard 55.9 % (62/111); hard ECE 0.231, TVD 0.264; 231/231 well-formed. Projected 67.6 against 70.9 for the release: the calibration loss outweighs the flat accuracy. Not promoted (docs/EXPERIMENTS.md). With the hard-like calibration (gate 4b) its hard ECE is 0.199 (`2026-09-23-public-check-head-calhard.jsonl`), still far from the release's 0.107, so the over-confidence is in the head, not in the calibration split.

## Earlier run: verification readout only (prompt-2), 22 September 2026, Apple M4 Pro

One run of JevBench's 231 public items through JevBench's own `typesafe` adapter and `score_task`, against `compass-vocab-qwen3.5-4b-compass-prompt-2` (Qwen3.5-4B revision `851bf6e8`, frozen, bf16) with `release/calibration.json`, served on an Apple M4 Pro (MPS, reference linear-attention kernels). Not tuned on; reported as is. Per-item records: `2026-09-22-public-check.jsonl`.

| Tier | Items | Accuracy | Well-formed answers | Top-label ECE | Mean input tokens |
| --- | --- | --- | --- | --- | --- |
| easy | 48 | 100.0 % | 48/48 | 0.021 | 298 |
| standard (original) | 72 | 80.6 % | 72/72 | 0.074 | 296 |
| hard | 111 | 55.9 % | 111/111 | 0.089 | 1,408 |

Hard tier by family: adversarial 6/6, trap 8/8, routing_hard 5/5, multi_hop 11/18, judge_hard 10/17, probability 6/10 (mean TVD to gold distributions 0.25), ambiguous 4/7, long_policy 8/19, tradeoff 2/6, temporal_numeric 2/15.

Standard tier by family: extraction 12/12, ordinal 12/12, intent 11/12, adequacy 9/12, policy 8/12, routing 6/12.

These are public-only, preliminary figures: the 109 held-out hard items and the 146-item judge tier were not run, and published rows for other systems (for example SemIf's 59.5 % and reflex's 63.2 % on hard) are full-benchmark results including held-out cases, so they are not directly comparable to the 55.9 % here.

**Projection, not a measurement.** With `jevbench/composite_v13.py`, assuming 0.2 s raw p50 / 0.3 s p95 on the maintainer's GPU (×2 + 0.15 s adjustment) and the $0.03/M hosted price for the size class: the projected axes are Calibration 78.6 (from public hard items only), Speed 83.8 (assumed latency), Cost 59.0; Intelligence 62–66 depending on an assumed judge-tier accuracy of 75–85 %, giving a projected JevBench Score of about 70–71. The judge tier, the held-out hard items and evaluator-GPU latency are unknown until the maintainer runs the system.

Latency on this Mac (p50 1.6 s standard, 3.5 s hard) is not a measurement of the release: the reference PyTorch linear-attention path was used, not the fused CUDA kernels.
