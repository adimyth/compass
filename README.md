# Compass

Compass turns a small open language model into a **decision model**: you give it a document and a typed question with a fixed set of allowed answers, and it returns a probability for every answer. It never generates text, so there is nothing to parse, nothing to repair, and one call costs one read of the document.

```json
POST /v1/systemone
{
  "state": "Hi, I was charged twice for September and the app also crashed this morning. Please refund the duplicate.",
  "questions": {
    "team":   {"type": "choice", "instructions": "Which team should handle this?",
               "criteria": {"billing": "Charges and refunds", "technical": "Bugs and outages", "sales": "Pricing"}},
    "urgent": {"type": "noul",   "instructions": "Does the message ask for a same-day response?"}
  }
}

→ {"answers": {"team":   {"type": "choice", "choice": "billing", "confidence": 0.93,
                          "probabilities": {"billing": 0.95, "sales": 0.01, "technical": 0.04}},
               "urgent": {"type": "noul", "noul": 0.22}},
   "usage": {"input_tokens": 412, "output_tokens": 0}, "model": "compass-0.2.0"}
```

Three question types: **choice** (one option from a set), **noul** (a yes/no probability), and **score** (a level on an ordered scale). The wire format is TypeSafe's `/v1/systemone`, so anything written for that API, including the [JevBench](https://github.com/fstandhartinger/jevbench) harness, runs Compass unchanged.

**Current release: `compass-0.2.0`.** [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0, revision `851bf6e8`) with a 116 MB LoRA adapter, [`adimyth/compass-lora-v2`](https://huggingface.co/adimyth/compass-lora-v2) (revision `bfa8af07`), merged at load. Everything here is Apache-2.0.

## How it works

Most ways of getting a decision out of a language model ask it to write the answer and then parse the text. Compass never decodes. It reads the model's own probabilities at a single position, for each candidate answer, in a layout designed so that the answer cannot depend on how the options were ordered or named.

```
request ─► rubric compiler ─► candidates in canonical order
                                       │
              ┌────────────────────────┴───────────────────────────┐
              │  document, read once (prefix, KV cache)            │
              └──────┬───────────────────────────────┬─────────────┘
                     │ fork per question             │ fork per question
                     ▼                               ▼
          instructions + rubric              instructions + lettered rubric
                     │ fork per candidate            │
                     ▼                               ▼
   "Proposed answer: billing — Charges     "Which one is correct? Reply
    and refunds. Is this correct under      with its letter."
    the rubric?"  → log-odds(yes : no)      → log-prob of each letter
                     │                               │
                     └────────── fuse in log space ──┘
                                       │
                            temperature calibration
                                       │
                          probabilities · choice / noul / score
```

1. **Rubric compiler** ([`compass/contract.py`](compass/contract.py)). The request is validated against the schema in [`api/systemone.schema.json`](api/systemone.schema.json) and each question becomes a list of candidates: choice options sorted by key, score levels in their given order, and a noul question as the pair of propositions `no` / `yes`. Because the model only ever sees candidates in this canonical order, reordering the options in a request cannot change the answer; the test suite checks every permutation.

2. **One read of the document** ([`compass/backbone.py`](compass/backbone.py)). The document is prefilled once and its cache is forked, first per question (the instructions and rubric are appended), then per candidate (a short branch of about 25 tokens). All branches of a question run as one batch. The reported `input_tokens` are exactly the tokens the model processed: the document, one rubric per question, and the branches.

3. **Two readouts.** *Verification*: each candidate is stated as a proposed answer and the model is asked whether it is correct under the rubric; the log-odds of "yes" against "no" is that candidate's score. *Direct*: the candidates are listed with letters and the logits of the letters are read at one position. Verification sees each option in isolation but with the full rubric in view; direct sees them side by side. Their log-probabilities are averaged. On our own evaluation items the fusion beats either readout alone by three to seven points.

4. **Calibration** ([`compass/calibration.py`](compass/calibration.py)). One temperature per question type, fitted after every other design choice is frozen, on our own items in the families the model finds hardest, so that a reported 0.8 means roughly 80 % on hard cases, not on easy ones.

5. **The adapter** ([`compass/train_lora.py`](compass/train_lora.py)). A rank-16 LoRA on the attention and MLP projections, trained through the two readouts above rather than through next-token prediction. The loss is a listwise cross-entropy over each question's candidates (soft targets where the evidence is graded), a per-readout term so both readouts stay useful on their own, an ordinal term for score questions, a permutation-consistency term (the same item under two rubric orders must agree) and an opaque-label term (option names replaced by neutral tokens, so the model cannot lean on what an option is called). Training data is our own: code-generated cases with computed labels (policies with amendments and exceptions, multi-step lookups, count-derived probabilities, ordinal rubrics) plus model-drafted scenarios (routing, answer-adequacy judging, ambiguous and trade-off cases), split by template and language form so that evaluation never sees a template that was trained on. The specification is in [`docs/stage-b-v2/`](docs/stage-b-v2/).

Serving is deterministic: bf16 weights, no sampling, one request at a time, probabilities in float64 that sum to 1. A cache-parity check against fp32 scoring is part of the release pipeline.

## Run it

One GPU with at least 12 GB. The base weights and the adapter download on first use.

```sh
git clone https://github.com/adimyth/compass && cd compass
scripts/serve.sh 8000
```

or

```sh
docker build -t compass . && docker run --gpus all -p 8000:8000 compass
```

Then:

```sh
curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Order 7731. Placed 9 September. Payment authorised, not captured. Label created, awaiting pickup.",
  "questions": {"shipped": {"type": "noul", "instructions": "Has the order been shipped? Answer only from the facts stated."}}
}'
```

`GET /v1/models` reports the backbone revision, adapter revision, readout and calibration file that answered, and `GET /healthz` says whether the model is loaded. Answers: `noul` carries `noul` (the probability of yes); `choice` carries `choice`, `confidence` and `probabilities`; `score` carries `score` (the probability-weighted level), `confidence`, `legend` and `probabilities`. Malformed requests get a 422 naming the field; the server never returns an invented distribution.

Serving options: `--adapter none --calibration release/calibration-0.1.1.json` runs the previous, adapter-free release (`compass-0.1.1`); `--readout verify|direct|fusion` and `--fusion-weight` select the readout; `--model-name Qwen/Qwen3.5-0.8B` gives a small model for smoke tests on a laptop.

## Evaluation

Compass is measured on [JevBench](https://github.com/fstandhartinger/jevbench) through the benchmark's own adapter and scorer, one run per configuration, public items only (the judge tier and the held-out hard items are the maintainer's).

| Tier | Items | Accuracy | Well-formed | Top-label ECE |
| --- | --- | --- | --- | --- |
| easy | 48 | 100.0 % | 48/48 | 0.034 |
| standard | 72 | 87.5 % | 72/72 | 0.137 |
| hard | 111 | 60.4 % | 111/111 | 0.069 |

No JevBench item, label or paraphrase is used for training, calibration or selection; every data file passes an 8-gram overlap check against the public JevBench files ([`scripts/validate_items.py`](scripts/validate_items.py)). Every configuration we tried, with what worked and what did not, is in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md); per-item records are in [`dev/results/`](dev/results/).

## Develop

```sh
uv sync
uv run pytest                                                 # contract, server, backbone and head tests
uv run python -m compass.data.splits_v2 --out dev/splits_v2     # rebuild the training, selection, calibration and release splits
uv run python scripts/own_eval.py --tasks dev/splits_v2/release.jsonl
uv run python -m compass.train_lora --train dev/splits_v2/train.jsonl --select dev/splits_v2/selection.jsonl --out release/lora-new
```

The GPU pipeline (splits, training, fusion weight, calibration, release split, cache parity, shadow suite) is [`scripts/pod/stage_b_v2.sh`](scripts/pod/stage_b_v2.sh).

## Licence

Apache-2.0 for the code, the data and the adapter. Qwen3.5-4B is Apache-2.0 and unchanged.
