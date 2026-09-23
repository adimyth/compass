# Experiments

Everything tried on the way to `compass-0.2.0`, in order, with the rule each attempt was held to and what happened. Every number is in `dev/results/`.

## Rules that applied throughout

- No JevBench item, label or paraphrase in any training, calibration or selection file; every file passes an 8-gram overlap check against the public JevBench files.
- Design choices (readout, fusion weight, hyperparameters) are made on an internal **selection** split. Calibration is fitted on a separate **calibration** split after the design is frozen. A **release** split is evaluated once per configuration. A **shadow suite** of 280 drafted items in domains disjoint from every other split is run once per configuration and decides promotion.
- JevBench's public items are run once per frozen configuration and reported as they come.

## Readout comparison (frozen Qwen3.5-4B, selection split, 276 items)

| Readout | Accuracy | ECE |
| --- | --- | --- |
| verification (per-candidate yes/no log-odds) | 62.0 % | 0.103 |
| direct (lettered options, letter logits) | 58.3 % | 0.101 |
| **fusion, equal weight in log space** | **65.2 %** | 0.098 |
| fusion + content-free debiasing (λ 0.5 / 1.0) | 60.1 / 55.8 % | 0.109 / 0.205 |
| direct with its own system prompt | 50.4 % | 0.204 |
| fusion over that | 56.2 % | 0.133 |

Fusion was adopted. Debiasing (subtracting each candidate's log-odds against an empty document) and a dedicated prompt for the direct readout were both worse and were dropped.

## Trained variants

| Variant | Mechanism | Data | Outcome |
| --- | --- | --- | --- |
| head-v1 | residual MLP on the hidden state at the readout position, frozen backbone | 2,700 generated items | +7 points on the internal release split, nothing on JevBench's public items, hard-tier ECE 0.105 → 0.231. Not promoted. |
| **lora-v2 (`compass-0.2.0`)** | LoRA r16 through the verification and direct readouts; listwise, per-readout, ordinal, permutation and opaque-label losses | 2,150 items (1,900 generated with contrast and held-out language forms, 250 drafted) | Shadow core families 58.8 → 68.2 %, ECE 0.103 → 0.046; public standard 80.6 → 87.5 %, hard 56.8 → 60.4 %, hard ECE 0.107 → 0.069. **Promoted.** |
| lora-v2b | same recipe, seed 1, completed data (2,170 items) | as above plus tradeoff | Ties on accuracy (core 68.8 %), ECE 0.097. Not promoted. |
| lora-v3 | same recipe plus 400 generated adequacy items and adequacy loss weight 2.0 | 2,470 items | Core 62.4 %, ECE 0.099; drafted adequacy unchanged at 57.5 %. Not promoted. |

## Public items, all configurations run

| Configuration | Easy | Standard | Hard | Hard ECE | Hard TVD |
| --- | --- | --- | --- | --- | --- |
| verification only | 100 % | 80.6 % | 55.9 % | 0.089 | 0.250 |
| compass-0.1.0 (fusion) | 100 % | 80.6 % | 56.8 % | 0.105 | 0.248 |
| compass-0.1.1 (fusion, hard-like calibration) | 100 % | 80.6 % | 56.8 % | 0.107 | 0.250 |
| head-v1 | 100 % | 81.9 % | 55.9 % | 0.231 | 0.264 |
| **compass-0.2.0 (fusion + LoRA)** | 100 % | **87.5 %** | **60.4 %** | **0.069** | **0.216** |

## Shadow suite, 280 drafted items

| Configuration | Overall | Core (policy, multi-hop, adequacy, ambiguous) | Adequacy |
| --- | --- | --- | --- |
| compass-0.1.1 | 66.4–66.8 %, ECE 0.043–0.059 | 58.8–59.4 %, ECE 0.067–0.103 | 65.0 % |
| **compass-0.2.0** | **76.4–76.8 %**, ECE 0.053 | **68.2–68.8 %, ECE 0.046–0.050** | 52.5–57.5 % |
| lora-v2b | 77.1 %, ECE 0.060 | 68.8 %, ECE 0.097 | 57.5 % |
| lora-v3 | 73.6 %, ECE 0.039 | 62.4 %, ECE 0.099 | 57.5 % |

Ranges are the same configuration on two GPUs (bf16 kernel noise).

## What we learned

- Generated data with computed labels trains the model on the generator. It transferred to JevBench only when the training set also had drafted scenarios, contrast in every wrong option, and language forms held out of training (v2); it did not transfer as a hidden-state head (v1) or as more generated adequacy items (v3).
- Calibration must be fitted on items at the difficulty where it is measured. JevBench scores it on the hard tier only.
- Temporal arithmetic in one forward pass is not solvable by any system on the board without generation (20–33 % for all of them); it is not a training target.
- The remaining gap is answer-adequacy judging, which is what the private judge tier consists of. Only drafted or hand-written items in that style can be expected to move it.
