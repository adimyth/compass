# Compass technical specification (v0.1 draft)

This is step 1 of the participation plan in [PROJECT_BRIEF.md](PROJECT_BRIEF.md): the specification and API schema, written before any model work. Facts about JevBench come from the repository at commit `51a8d73` (read 2026-09-22) and from TypeSafe's public docs. Where this spec changes the brief, it says so and gives the reason.

## 1. What an entry has to be

JevBench measures a system through one HTTP call per decision. The system has to serve TypeSafe's wire format, `POST /v1/systemone`, so that the benchmark's unchanged `typesafe` adapter can run it (`jevbench/adapters/typesafe.py`). Every benchmark request carries exactly one question, keyed `decision`.

The adapter reads answers this way. Anything else counts as a wrong answer, and nothing is repaired.

| Type | Field read | Validity rule |
| --- | --- | --- |
| `noul` | `answers.decision.noul` | A number in [0, 1], not a boolean. Stored as `{"yes": p, "no": 1 - p}`. |
| `choice` | `answers.decision.choice` and `.probabilities` | `choice` must be one of the option keys. `probabilities` must cover exactly the option keys. |
| `score` | `answers.decision.probabilities` | Keys are level indices `"0"`..`"k"`. Accuracy uses the argmax, not the expected value. |

A distribution is valid when its keys equal the label set, every value is in [0, 1], and the values sum to 1 within 0.001 (strict) or 0.02 (headline, renormalised). JevBench breaks argmax ties by the lexicographically smallest label. The response also has to carry `answers.decision.type` equal to the question type, and should carry `model` and `usage.input_tokens`.

### How entries get onto the board

1. Open a `[bench request]: ...` issue on the JevBench repository with install and serve commands, the pinned weights, and our own numbers on the public items.
2. The maintainer pre-registers the mapping, endpoint conditions and cost basis in `docs/`, then runs our server on a rented GPU (RunPod or lium.io) in a disposable container that holds no credentials.
3. All 534 frozen decisions run serially, one request at a time, from a server in Germany.

Held-out items are never sent to an endpoint the submitter operates. An entry that only offers its own endpoint gets a partial, unranked row (jqv was first run this way). **To be ranked, Compass must be runnable by the maintainer from public code and weights.**

## 2. How the score is computed

JevBench Score is the geometric mean of four axes (25 % each). A system with Intelligence below 50 is multiplied by `(Intelligence / 50)²`. The code is `jevbench/composite_v13.py`.

| Axis | Formula | What it means for Compass |
| --- | --- | --- |
| Intelligence | Chance-corrected accuracy per tier. Weights: hard 30 %, easy 14 %, standard 28 %, judge 28 %. | Easy and standard are saturated near 100 % at the top. The hard tier (220 items, 109 held out) separates entries. |
| Calibration | Hard tier only: the mean of `100 × (1 − ECE / 0.5)` and `100 × (1 − mean TVD)` against exact gold distributions on the `probability` family. | Temperature scaling alone is not enough. The model also has to output graded probabilities when the evidence in the state is graded. |
| Speed | Mean of `100 − 20·log10(s / 0.1)` at p50 and p95. Self-hosted latency is adjusted ×2 + 0.15 s. | 0.2 s raw p50 with 0.3 s p95 scores about 84; half that scores about 88. The adjustment limits how much raw speed can gain. |
| Cost | `100 − 30·log10($ per 1,000 decisions / $0.001)`. Open weights are priced at a hosted-provider list price for the size class × our reported input tokens. | Only the size class and the token count matter. The GPU we rent does not. |

Hosted size-class prices the maintainer uses (per million input tokens; `results/v1.1/pricing/jevbench-hosted-price-table.json` and `docs/v1.2-additions.md`):

| Size class | $/M input | Cost axis at 775 tokens/decision |
| --- | --- | --- |
| Encoder ≤ 0.6B | 0.01 | 73.3 |
| Generative ≤ 1B | 0.027 | 60.4 |
| Dense 2–4B | 0.03 | 59.0 |
| Dense 9B | 0.10 | 43.3 |
| Dense 27B | ~0.21 | 33.4 |

Going from 4B down to 1B generative buys almost nothing on Cost (59.0 to 60.4). Only an encoder-class model moves the Cost axis materially.

## 3. What the current board tells us

48 ranked rows as of v1.3.0. Selected rows, with per-tier accuracy:

| System | Architecture | Score | Int | Cal | Hard acc |
| --- | --- | --- | --- | --- | --- |
| Jev 1.13.0 (#1) | closed | 74.4 | 85.7 | 82.7 | 74.1 % |
| SemIf | Qwen3.5-4B, frozen, logit readout | 73.1 | 79.0 | 72.6 | 59.5 % |
| Winnow-12B | Gemma 4 12B LoRA | 71.2 | 82.0 | 72.0 | 70.9 % |
| reflex 4B | Qwen3.5-4B + LoRA + calibration | 70.3 | 80.1 | 75.2 | 63.2 % |
| zerank-2 | 4B cross-encoder reranker | 66.0 | 63.0 | 76.5 | 47.3 % |
| decider-2b | 2B decoder + trained readout | 61.7 | 61.2 | 46.6 | 47.3 % |
| jeff | GLiFormer 400M encoder | 54.4 | 46.9 | 64.6 | 37.7 % |
| Laya | ModernBERT-large 421M encoder | 54.4 | 45.8 | 62.5 | 34.1 % |
| OpenDecision | ModernBERT-large zero-shot | 40.6 | 40.8 | 56.1 | 33.2 % |
| Certo v1 | ModernBERT-large, per-option | 0.0 | 0.0 | 82.0 | 31.8 % |

Three findings change the brief's plan.

**Small encoders do not clear the intelligence floor.** Every entry built on an encoder under 0.6B scores Intelligence ≤ 47 and hard-tier accuracy ≤ 40 % (chance on that tier is 33.6 %), so the near-chance penalty hits all of them. The brief's open question "which permissively licensed foundation encoder" assumes an encoder backbone. The data argues against that for a first release.

**Scoring each candidate independently, without the rest of the rubric, loses accuracy at equal size.** The 4B rerankers (zerank-2, Qwen3-Reranker-4B) reach Intelligence 63–64. The 4B decoders that read the whole rubric (SemIf, reflex) reach 79–80. The reranker rows ran through a neutral adapter with no decision training, so part of the gap is training. Still, the brief's "independently scores each pairing" design sits on the losing side of that comparison. Section 5 keeps symmetric per-candidate scoring, but conditions each candidate on the full rubric in a fixed canonical order.

**Calibration is the cheapest axis to gain on.** SemIf, a frozen 4B model, is 1.3 points behind Jev: it wins on Cost and loses on Intelligence (79.0 vs 85.7) and Calibration (72.6 vs 82.7). At the 4B price point, with Speed about 84 and Cost about 59, the Intelligence needed to beat Jev's 74.4 is:

| Calibration | Intelligence needed to exceed 74.4 |
| --- | --- |
| 72 | 85.9 |
| 78 | 79.3 |
| 85 | 72.7 |

Intelligence 80 means roughly easy 100 %, standard 97 %, judge 95 % and hard 63 %.

## 4. Target profile for release 0.1

Recommended configuration (pending the decision in §9): **one GPU configuration on a 2–4B dense decoder backbone with Apache-2.0 weights.**

| Axis | Target | Projected axis score |
| --- | --- | --- |
| Hard-tier accuracy (on our own held-out hard families) | ≥ 63 % | Intelligence ≈ 80 |
| Hard-tier ECE / TVD on graded-evidence items | ≤ 0.06 / ≤ 0.25 | Calibration ≈ 81.5 |
| Raw serial latency on an RTX 4090-class GPU | p50 ≤ 0.2 s, p95 ≤ 0.3 s | Speed ≈ 84 |
| Reported input tokens per decision | ≤ 800 average | Cost ≈ 59 |

At those targets the projected JevBench Score is 75.4. That would be first place, but not by a wide margin, so the hard-tier and calibration targets carry the result.

A compact encoder configuration stays on the roadmap as a research track. At Cost 73 it would score above 75 with Intelligence 65 and Calibration 78, but no encoder entry has reached Intelligence 50 yet.

## 5. Architecture

The brief's six components stay. What changes is the backbone and the fact that each candidate is scored with the full rubric in view.

```text
request ──► Rubric compiler (deterministic)
              │  state text, canonical candidate list per question
              ▼
          Shared state encoder: decoder backbone, one prefill of the state, KV cache kept
              │  per question: fork the cache, append instructions + canonical rubric
              ▼
          Evidence-aware candidate scorer (ours, trained)
              │  for each candidate span: pool its hidden states → query
              │  cross-attend over state token states → evidence vector → logit
              ▼
          Typed heads: choice softmax · binary true/false pair · ordinal head
              ▼
          Calibration layer (fitted after freeze, non-JevBench data only)
              ▼
          Response assembly: probabilities, choice/score/noul, confidence, usage
```

1. **Rubric compiler** (`compass/contract.py`, implemented). It validates the request and turns each question into candidates. Choice candidates are sorted by option key before scoring, so any permutation of the same options produces identical model input and output. Noul becomes a symmetric `false`/`true` pair built from `criteria.true`/`criteria.false` when given. Score keeps level order, because order carries the meaning there. Structured (JSON) instructions and criteria are serialised deterministically. The prompt layout is our own and is versioned with the model.
2. **Shared state encoder.** A decoder backbone reads the state once. Each question forks the cached prefix, so a request with N questions costs one state read plus N short rubric reads. The same design keeps per-decision tokens low on the benchmark, because the state is not re-read per option (the jev-local footnote flags that re-reading problem).
3. **Evidence-aware candidate scorer.** A trained cross-attention head, not a read of answer-token logits. For each candidate, the pooled hidden state of its own span in the rubric is the query. The keys are the state token states. The output is a scalar compatibility logit plus an attention map over the state, which we can publish as evidence for debugging. This is what separates Compass from the logit-readout entries (reflex, SimpleJev, SemIf, jqv).
4. **Typed heads.** Choice uses a softmax over candidate logits. Noul uses a two-way softmax over the true and false propositions, reported as `p(true)`. Score uses a cumulative-link ordinal head (K−1 ordered thresholds on one latent score), which keeps level probabilities consistent with the ordering.
5. **Calibration layer.** Per-question-type temperature (plus a threshold shift for ordinal questions), fitted once, after the model is frozen, on a validation split that is separate from training data and from JevBench.
6. **Serving runtime** (`compass/server.py`, skeleton implemented). A single deterministic worker, fp16/bf16 weights, no sampling, fixed batching, `/v1/systemone`, `/v1/models`, `/healthz`.

Release 0.1 does not include uncertainty-triggered refinement (the brief's fourth open decision). We revisit it after the evaluation gate shows where the base system misses.

## 6. API contract

Machine-readable schema: [`api/systemone.schema.json`](api/systemone.schema.json). The contract is TypeSafe's public wire format. We do not add fields the benchmark adapter would reject, and we do not rename any.

Request:

```json
{
  "model": "compass-latest",
  "state": "string | object | array",
  "questions": {
    "<id>": {"type": "choice", "instructions": "...", "criteria": {"<option>": "description", "...": "..."}},
    "<id>": {"type": "score", "instructions": "...", "criteria": ["level 0", "level 1", "..."]},
    "<id>": {"type": "noul", "instructions": "...", "criteria": {"true": "...", "false": "..."}}
  }
}
```

Response:

```json
{
  "model": "compass-0.1.0",
  "answers": {
    "<id>": {"type": "choice", "choice": "<option>", "confidence": 0.71, "probabilities": {"<option>": 0.8, "...": 0.2}},
    "<id>": {"type": "score", "score": 1.3, "confidence": 0.4, "legend": {"0": "..."}, "probabilities": {"0": 0.1, "...": 0.9}},
    "<id>": {"type": "noul", "noul": 0.93}
  },
  "usage": {"input_tokens": 612, "output_tokens": 0}
}
```

Invariants, enforced in code and tests:

- Probabilities are computed in float64 and are not rounded. They sum to 1 within 1e-9 and cover exactly the requested keys.
- `choice` is the argmax, with ties broken by the lexicographically smallest key, the same rule JevBench uses. Our reported choice therefore always matches the benchmark's argmax.
- `score` is the probability-weighted level, following TypeSafe's semantics. `confidence` is `(n·p_max − 1) / (n − 1)`, clipped to [0, 1], for both choice and score.
- `usage.input_tokens` counts every token the backbone actually processes for the request: the state once, plus each question's rubric. The maintainer prices us from this number, so it has to be the honest count.
- Non-finite model outputs fail the request with a 500 error. The server never returns an invented distribution.
- Invalid requests return 422 with a message naming the field. Malformed JSON returns 400.

Limits (configurable): 100 questions per request, 2–64 options or levels per question, and a model context of 32k tokens for the state plus the longest question (TypeSafe's published limit). The long hard-tier states run 2,000–6,000 tokens.

## 7. Training data and calibration (outline)

The brief's rules stand: no JevBench items, labels, paraphrases or held-out material, and nothing from another entrant. Every source gets a provenance record (licence, URL, revision, row count, filters) before it is used.

- **Families to cover**, matching what the benchmark rewards without using its data: routing and intent, factual support from a document, enum extraction, long-document policy application with amendments and exceptions, multi-hop lookups across tables, date and number arithmetic inside a document, answer-adequacy judging with one subtle error, prompt-injection-style distractors, under-specified cases that call for an "insufficient information" label, and ordinal rubrics.
- **Graded-evidence items** for calibration: states that contain countable evidence (case logs, stated rates) with exact target distributions. The calibration axis measures TVD to gold distributions on items like these, so the training objective needs a proper scoring loss against soft targets, not only argmax labels.
- **Synthetic long-context tasks** from our own generators, with the answer computed by code, so long-policy and temporal items get verifiable labels.
- **Splits by task family**, not by row, so the internal gate measures transfer to unseen families.
- **Objective:** a listwise candidate-ranking loss, the multi-class Brier or log loss against soft targets, ordinal consistency on score questions, and a permutation-consistency penalty (KL between predictions under two random option orders during training, although serving uses canonical order).

## 8. Internal evaluation gate (before any submission)

| Check | Pass condition |
| --- | --- |
| Response validity | 100 % strict-valid on the contract suite, including malformed, empty, 64-option and 32k-token requests. |
| Option-order stability | Identical output under any permutation of choice options (exact, by construction). Under training-time random orders, top-1 flips on < 2 % of items. |
| Held-out family accuracy | Hard families reach ≥ 63 %. Easy and standard families reach ≥ 97 %. |
| Calibration | ECE ≤ 0.06 on held-out hard families; mean TVD ≤ 0.25 on graded-evidence items. |
| Latency | Warm, serial, on disclosed hardware: p50 ≤ 0.2 s and p95 ≤ 0.3 s at 1,000 input tokens. Memory use recorded. |
| JevBench compatibility | `scripts/jevbench_compat.py` passes against the release container using JevBench's own adapter and validator on synthetic tasks. The public JevBench items run once, at the end, as a final check, and the result is reported, not tuned on. |

## 9. Decisions needed

1. **Backbone size class.** Recommended: a 2–4B dense decoder with Apache-2.0 weights (the Qwen3.5-4B-Base class). The alternatives are an encoder under 0.6B (cheapest, but no entry has cleared Intelligence 50) or a 9B model (smarter, but Cost falls to about 43 and the projection shows a lower total score).
2. **Compute and annotation budget.** This decides how much synthetic long-context data we generate and whether we fully fine-tune or train adapters plus the scorer head.
3. **Name and model id.** This spec uses `compass-0.1.0` and `compass-latest`.

## 10. Release and submission checklist

- [ ] Public repository with licence, `SPEC.md`, model card, data provenance record, training configuration and evaluation-gate report.
- [ ] Weights (full model or adapter plus scorer head) on the Hugging Face Hub under a pinned revision, with a calibration file.
- [ ] Container image with a pinned digest. It serves `/v1/systemone` with no network access after start and needs no credentials.
- [ ] One-command serve script plus the exact `jevbench.cli run --adapter typesafe --endpoint http://127.0.0.1:8000 --key-env '' --model compass-0.1.0` invocation.
- [ ] Our numbers on the public items (one final run), with hardware disclosed.
- [ ] `[bench request]` issue on `fstandhartinger/jevbench`, asking for an evaluator-run GPU run so that the held-out items are scored and the row can be ranked.
