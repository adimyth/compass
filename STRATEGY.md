# Compass strategy: a top-10 JevBench entry

Goal: a ranked JevBench row in the top 10 with an entry that is ours. This document records the decisions. [SPEC.md](SPEC.md) has the evidence behind them; this file has the choices and what each one buys.

## The bar

The #10 row today is 66.6. Thirty bench requests are open on the JevBench tracker, most of them 4B–35B logit-readout systems, so the bar will move. **We plan for 71+**, which is #5 today and leaves room for the queue.

With the benchmark's own scoring code, a 4B-class model has Cost 59.0 and, at 0.2 s raw serial latency, Speed 83.8. That fixes what Intelligence and Calibration must multiply to:

| Intelligence | Calibration | JevBench Score |
| --- | --- | --- |
| 75 | 68 | 70.5 |
| 78 | 72 | 72.6 |
| 80 | 78 | 74.6 |

A frozen 4B model read well already reaches Intelligence 79 (SemIf). A trained 4B reaches 80 with Calibration 75 (reflex). The 71 target is reachable with a frozen backbone if our readout and calibration are good; the 74 target needs training. Both are on the plan, in that order.

## Decision 1: backbone is Qwen3.5-4B, frozen first, adapted second

**Qwen/Qwen3.5-4B** (Apache-2.0, revision pinned at release). 4B is the size where Intelligence stops paying for Cost: 9B costs 16 Cost points for a few Intelligence points, and no entry under 1B has cleared the near-chance penalty. The 0.8B and 2B siblings share the architecture, so development runs locally on the 0.8B and every result transfers to the same code.

The model is a hybrid: three linear-attention layers to every full-attention layer. That matters for Decision 3, because its per-layer state is small and cheap to fork.

Not the Base variant. The instruct checkpoint gives a strong zero-shot floor (Stage A), which is our insurance if training under-delivers.

## Decision 2: the readout is candidate verification, not answer-letter logits

Every strong open entry (SemIf, reflex, SimpleJev, jqv, LitJev, decider) asks "which option?" and reads the probability of the option letters or names. Compass does not. For each candidate it asks one question, **"is this proposition the correct answer under the rubric?"**, and reads the model's log-odds of yes against no. A choice question becomes n verifications sharing one read of the state; a noul becomes two (the true proposition and the false proposition); a score question becomes one per level.

Why this is the right readout, not just a different one:

- **Symmetric by construction.** Each candidate is scored in its own branch, in the same position, with the same wording. Option order in the request cannot reach the model: the rubric is listed in sorted key order and the branches are independent. Order sensitivity is what halved open-alternative-jev's accuracy on judge items (72 % to 21 %).
- **Every candidate sees the full rubric.** The rubric with all options sits in the shared prefix, so "engineering" is verified knowing that "billing" was an option. That is the difference between us and the rerankers, which verify each option blind and lose 15 Intelligence points at the same size.
- **Graded evidence gives graded log-odds.** A verifier answering "is the outcome X" on a state with 30 of 40 similar cases going one way lands near 0.75, which is what the calibration axis measures on the probability family. A letter readout has no such handle.
- **Noul is native.** Verifying the true and false propositions separately, then normalising, is a two-sided judgement, not a one-sided "P(yes)".

Stage A reads the yes/no log-odds from the vocabulary. Stage B replaces that with a trained head on the final hidden state at the readout position, and a LoRA on the backbone. The prompt layout, the branch wording and the response assembly are ours and are versioned with the model (`compass/backbone.py`).

## Decision 3: one state read, forked twice

Cost is priced from the input tokens we report, and the hard tier's states run 2,000–6,000 tokens. Reading the state once per candidate (jev-local does this) would multiply cost by the option count. Compass forks the cache in three levels:

1. **State** prefilled once per request.
2. **Question**: the cache is forked once per question and the instructions plus the sorted rubric are appended.
3. **Candidates**: the question cache is forked n ways and the n branches (about 25 tokens each) run in one batch.

Reported `input_tokens` = state + Σ(rubric + Σ branches), every token the backbone processed. On the benchmark that is roughly the state plus 150–300 tokens, in line with the other 4B rows, so Cost stays at about 59. Latency is three forward passes, of which one is the state prefill, so the 0.2 s target on a 4090 is realistic.

## Decision 4: training and calibration use only data we made or licensed

Stage B trains on a corpus with three sources: templated generators whose labels are computed by code (policy application with amendments, date and unit arithmetic, multi-hop table lookups, running totals), scenario items authored by an LLM we control and cross-checked by a second one (routing, judging with one subtle error, injected instructions, under-specified cases), and graded-evidence items whose target distribution is derived from countable facts in the state. Sources are recorded before use.

The objective is the log loss of the verification head against soft targets, an ordinal consistency term for score questions, and a permutation-consistency term: the same item under two rubric orders must give the same distribution. Serving uses one canonical order, so the term is there to remove position bias inside the sorted listing, not to average it away at inference.

Calibration is a per-type temperature fitted after the freeze on a validation split whose families were not trained on. Before any training run, an n-gram overlap audit runs the corpus against JevBench's public files. That audit and its result ship with the release, because the maintainer performed exactly this check on another entrant and we would rather hand it in.

Nothing from JevBench is used for tuning. The public items run once, at the end, through `scripts/jevbench_compat.py`'s harness, and the number is reported as is.

## Decision 5: one submission, the trained model, self-hostable

We submit once, after Stage B passes the gate in SPEC.md §8, with a pinned Hub revision (LoRA + head + calibration file), a container, and one serve command. If Stage B does not beat Stage A on our held-out families, Stage A is submitted instead, described as what it is. A rankable row needs the maintainer to run it on their GPU; a public endpoint of ours would only get a partial row.

## What is done and what is next

| Step | State |
| --- | --- |
| Contract, calibration, server, JevBench-compat check | done, tested |
| Backbone scorer: prompt layout, three-level fork, verification readout, honest token count | `compass/backbone.py`, done, tested on Qwen3.5-0.8B locally |
| Own dev set (18 items across 9 families, no JevBench material) | `dev/own_dev.jsonl`, first pass; must grow to ~30 per family |
| Stage A on Qwen3.5-4B, zero-shot, uncalibrated (`dev/results/`) | 17/18 (choice 8/8, noul 6/7, score 3/3); 0 flips under reversed option order; TVD 0.12 on graded items; ECE 0.12. The miss is the policy-amendment item, the hard-tier family Stage B targets. The noul framing was fixed on this set (prompt-1 → prompt-2: 4/7 → 6/7), so the set is now tuned on and needs fresh items before it counts as a measurement. |
| Corpus generators and provenance record | after Stage A numbers |
| Stage B: head + LoRA training, calibration fit, overlap audit | after corpus |
| Release, container, bench request | after the gate |
