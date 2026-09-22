# Compass: an independent Jev-class decision-model entry

## Purpose

Compass is a proposal for an original entry in [JevBench](https://github.com/fstandhartinger/jevbench), Benchmark Heaven's evaluation of Jev-class decision systems. We are not planning to fork, modify, or rebrand another benchmark entrant. The aim is to build a distinct, reproducible system whose core decision architecture, training pipeline, calibration, serving runtime, and released artifact are ours.

The working thesis is: read the state once, evaluate every permitted decision against the relevant evidence symmetrically, and return calibrated probabilities without generating prose.

## What JevBench evaluates

JevBench evaluates systems that receive a state plus a bounded rubric and return a typed decision. A state can be text or structured context; a rubric can ask for a binary judgment, one option from a set, or a score across an ordered set of levels. The benchmark is not a test of writing quality, open-ended chat, code generation, or agent tool use.

The current benchmark covers 534 frozen decisions across easy, standard, judge, and hard tiers. It measures four axes, each contributing 25% to the published JevBench Score through a geometric mean:

| Axis | What it measures |
| --- | --- |
| Intelligence | Accuracy above random chance across the task tiers. |
| Calibration | Whether reported probabilities match observed outcomes and gold distributions. |
| Speed | Serial median and p95 end-to-end latency. |
| Cost | Estimated or published US dollars per 1,000 complete decisions. |

The benchmark includes held-out hard cases. To receive a fully ranked result, an entrant should be reproducibly runnable by the benchmark maintainer or available through a stable production endpoint. A temporary endpoint operated by the submitter may receive a partial, unranked result because held-out tasks are not sent to it. Details are documented in the [JevBench repository](https://github.com/fstandhartinger/jevbench) and the [Benchmark Heaven dashboard](https://benchmarkheaven.com/jev-models).

## Existing approaches

Two relevant entries illustrate the problem space, but neither is a code or design base for Compass.

| System | Core approach | Useful lesson for Compass |
| --- | --- | --- |
| [reflex](https://github.com/kshetrajna12/reflex) | Uses a frozen open-weight Qwen model and reads constrained answer-token probabilities with a carefully designed prompt and option-order averaging. | Calibration, option-order sensitivity, and deployment discipline matter as much as raw accuracy. |
| [simple-jev](https://github.com/featherless-ai/simple-jev) | Turns compatible Hugging Face language models into a typed classifier endpoint by reading next-token logits and constructing the JSON response itself. | Shared-context reuse can reduce cost, but valid structured output alone does not establish accurate or calibrated decisions. |

Compass must remain architecturally and operationally independent: no copied code, prompts, tuned configuration, benchmark-specific data, or derivative model artifact from another entrant.

## What Compass should achieve

Compass should be a credible, reproducible decision model rather than a benchmark demo. Its success criteria are:

- Return valid typed results for binary, choice, and ordered-score questions without prose generation or JSON repair.
- Treat candidate answers symmetrically so changing option order does not change the result materially.
- Produce confidence values that are calibrated on independently held-out data.
- Reuse a shared state efficiently when one request asks several questions about the same document.
- Deliver a clear speed, cost, and quality trade-off with a pinned model, deterministic runtime, and transparent pricing basis.
- Be trained and evaluated without JevBench data, answer keys, or rewritten variants of benchmark examples.
- Be simple for an independent evaluator to reproduce: source, checkpoint, container, API contract, and run instructions must be published and versioned.

## Initial solution: evidence-conditioned rubric scoring

Compass will be a native decision system, not a chat model that is prompted to emit an answer label. Its core is an evidence-conditioned rubric scorer.

```text
State / document
        |
        v
Shared state encoder -------------------------------+
                                                     |
Question + candidate criterion ---------------------+--> Evidence-aware candidate scorer --> raw scores
                                                     |
                                                     +--> Calibration layer --> typed probability response
```

For a choice question, Compass independently scores each pairing of the state, question, and candidate criterion. For example, a support ticket is evaluated separately against the propositions that its correct team is billing, technical, or sales. The system normalizes those semantic compatibility scores into a probability distribution. It does not depend on the spelling, tokenization, or position of the answer label.

The same core scorer supports all three question types:

| Question type | Compass behavior |
| --- | --- |
| Choice | Score every candidate criterion, normalize scores, and return the highest-probability option plus the distribution. |
| Binary | Score the true and false propositions directly, then return a calibrated probability of true. |
| Ordered score | Use an ordinal head that preserves the order of the rubric levels, then return the probability-weighted score and distribution. |

### Components we own

1. **Rubric compiler.** A deterministic component that turns the submitted question and criteria into semantic candidate representations without copying another entrant's prompt format.

2. **Shared state encoder.** A model component that reads the input state once and creates reusable token-level representations for all questions in the request.

3. **Evidence-aware candidate scorer.** A learned cross-attention module that compares each candidate representation with the parts of the state that support or contradict it. This is Compass's principal decision layer.

4. **Typed decision heads.** Native choice, binary, and ordinal-score heads that turn candidate evidence scores into valid response shapes.

5. **Calibration layer.** A post-training calibration procedure, fitted only on non-JevBench held-out data, which makes confidence values useful for automation thresholds.

6. **Serving runtime.** A deterministic `/v1/systemone` API, batching strategy, metrics, container image, and exact model manifest.

## Training and evaluation approach

Training data will be independently sourced from appropriately licensed public datasets and new task templates, with separate validation and test families. It should include routing, factual support, extraction, policy application, safety, multi-step judgement, and ordinal rubrics. Where uncertainty is real, labels should preserve annotator disagreement as a probability distribution rather than collapse it into one hard label.

The model objective should combine candidate-ranking loss, proper probability scoring loss, ordinal consistency for score questions, and regularization for option-order invariance. Calibration must be fit only after the decision model is frozen, and it must use a validation set that is separate from both training data and JevBench.

Before submission, Compass must pass an internal evaluation gate:

- Randomize candidate order repeatedly and measure prediction stability.
- Test valid response construction and probability sums under malformed and long requests.
- Measure accuracy and expected calibration error on task families not seen during training.
- Record warm-service p50/p95 latency and memory use on disclosed hardware.
- Run JevBench's public tasks only as a final compatibility check; do not use their outcomes for iterative tuning.

## Participation plan

1. Write a technical specification and API schema before implementation.
2. Build the independent training corpus and provenance record.
3. Implement and train the Compass decision model.
4. Add calibration, option-order robustness tests, and a reproducible serving container.
5. Publish a pinned release with code, weights or adapters, license, model card, and deployment instructions.
6. Ask the JevBench maintainer to run the exact release, preferably on evaluator-controlled infrastructure so the held-out tasks can be scored and the result can be ranked.

## Open decisions

- Which permissively licensed foundation encoder should provide the initial language representation?
- What training-data size and annotation budget can we support?
- Should the first release prioritize a compact CPU-friendly model, a higher-capability GPU model, or two clearly separated configurations?
- Is an uncertainty-triggered latent refinement stage worthwhile after measuring the base system's calibration, latency, and hard-task accuracy?

## Non-negotiables

- No code, prompt, trained adapter, configuration, or benchmark-tailored data copied from existing JevBench entrants.
- No JevBench labels, held-out tasks, or paraphrases used in training or calibration.
- Every reported number tied to an immutable release and disclosed hardware/configuration.
- Prefer a transparent, independently reproducible result over an opaque leaderboard optimization.
