# Compass plan: two days

Decisions are in [STRATEGY.md](STRATEGY.md). The scope is Stage A: frozen Qwen3.5-4B, our verification readout, per-type temperature calibration fitted on our own items, a container and a bench request. Training (Stage B) is out of scope and listed at the end as follow-up.

What this ships is the same size class as SemIf (#2, 73.1) and reflex (#5, 70.3), with a different readout. The target is a ranked row in the top 10; the projection at Intelligence 75–79 and Calibration 70–75 is 70–73.

## Day 1: calibrate and check

| Step | Output | Done when |
| --- | --- | --- |
| Dev items | `dev/families/`: the temporal generator (60) plus a multi-hop generator (60) and the 18 authored items. All through `scripts/validate_items.py` with the JevBench overlap check. | Done: 138 items, 0 errors, 0 overlap. |
| Calibration | `scripts/fit_calibration.py` fits one temperature per question type on the dev items, writes `release/calibration.json`. | Done: T = 1.5 / 3.0 / 2.5 (choice / score / noul); held-out ECE 0.19→0.16, 0.27→0.14, 0.24→0.09. |
| Final check | JevBench's public items (231), once, through `scripts/public_check.py` (JevBench's adapter and scorer) against the server on this Mac. Reported as is, never tuned on. | Done: easy 100 %, standard 80.6 %, hard 55.9 %, 231/231 valid, hard ECE 0.089. `dev/results/public-check.md`. |

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

## Follow-up, after the row exists

Stage B (LoRA plus verification head on our own corpus) targeting long-policy, multi-hop, probability and ambiguous items, where Jev beats the frozen 4B systems by 20–40 points. Temporal arithmetic is not a target: every no-generation system on the board scores 20–33 % there.
