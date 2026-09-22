# Stage B v2 training specification

Mechanism: `Qwen3.5-4B (frozen) + LoRA → existing verification and direct vocabulary readouts → fusion → calibration`. Only the LoRA is trained. The readouts, prompt wording (prompt-2), fusion, calibration procedure and the cache-forked serving path are unchanged and are measured after training exactly as before.

## Model

- LoRA rank 16, alpha 32, dropout 0.05, on the attention projections (q, k, v, o) and the MLP projections of every layer, including the linear-attention layers' projections. About 60 MB. bf16 base, fp32 LoRA weights.
- Saved with `peft`, pinned on the Hub under its own revision beside `release/calibration.json` for the new configuration.

## Training path

Flattened full sequences, no cache forking during training: for each item and each candidate, the sequence `prefix + rubric + branch` (verification form) and `direct prefix + direct rubric` (direct form) are run as ordinary forward passes with gradient checkpointing. The per-candidate logits are read at the same positions and with the same token sets as `compass/backbone.py` uses at inference, through shared helper functions so the two cannot drift.

After training, `scripts/parity.py` proves that the cache-forked inference path on the LoRA model matches flattened scoring within bf16 noise, as it did for the frozen model (mean |Δ| 0.107 against fp32, same as unforked).

## Loss, per item, over its candidate group

Let `v` be the verification log-odds normalised over candidates, `d` the direct log-probs, `f = 0.5·v + 0.5·d` the fused logits (the release fusion weight; refitted on selection after training).

1. **Listwise cross-entropy** on `softmax(f)` against the target distribution: one-hot for choice and noul with a hard gold, the exact `gold_probs` for probability items (soft target).
2. **Per-readout cross-entropy** on `softmax(v)` and `softmax(d)` separately, weight 0.5 each, so both readouts improve on their own and fusion stays a combination of two useful signals rather than one carrying the other.
3. **Ordinal distance** for score items: `((E[level] − gold level) / (K − 1))²`, weight 0.5.
4. **Permutation consistency**: the same item is also run with the rubric in a second random order (verification and direct both), and the symmetric KL between the two fused distributions is added, weight 0.5. Serving still uses one canonical order; this term removes position bias inside the listing.
5. **Opaque-label consistency**: with probability 0.5 the option keys are replaced by neutral tokens (`option_1`, `option_2`, …, shuffled) in both rubric and branches, and the loss is computed against the same gold. This removes label-text priors (the `coding_agent` failure).

Total: `L = L_fused + 0.5·(L_v + L_d) + 0.5·L_ord + 0.5·L_perm`, with opaque-label items sharing the same terms.

## Schedule

- AdamW, lr 1e-4 for the LoRA, weight decay 0.01, cosine decay, 3 % warm-up, batch 8 items (each item is all its candidates in both forms), gradient accumulation to an effective 32 items, 2 epochs over 6,000 items, bf16 autocast. About 3–4 hours on one RTX 4090 at 1,500-token states with checkpointing.
- Evaluation on the selection split every 500 items: fused accuracy, per-readout accuracy, ECE. Early stopping on selection fused loss, patience 3.
- Seed fixed and recorded; one run first, a second seed only if the first passes the gates.

## After training, in order

1. Fusion weight refitted on the selection split (grid 0.3–0.7).
2. Calibration fitted on the calibration split, restricted to the hard-like families as in gate 4b.
3. Release split evaluated once.
4. Cache parity proved.
5. Shadow suite, once, under the [promotion rule](SHADOW_SUITE_PROTOCOL.md).
6. If promoted: `compass-0.2.0`, one public-items run, reported.

## Not in v2

No hidden-state head, no full fine-tuning, no changes to prompt wording, no temporal-arithmetic training data, no thinking or generation at inference.
