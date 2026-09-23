# Compass

Compass is a decision model for [JevBench](https://github.com/fstandhartinger/jevbench): it reads a document once, checks every allowed answer against it, and returns calibrated probabilities. No text is generated. It serves TypeSafe's `POST /v1/systemone` wire format, so JevBench's unchanged `typesafe` adapter runs it.

**Release `compass-0.2.0`**: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0, revision `851bf6e8`) plus a 116 MB LoRA adapter, [`adimyth/compass-lora-v2`](https://huggingface.co/adimyth/compass-lora-v2) (revision `bfa8af07`), merged into the weights at load. Apache-2.0 throughout.

## Results on JevBench's public items

One run per configuration through JevBench's own adapter and scorer, on an RTX 4090. Public items only; the judge tier and the held-out hard items are the maintainer's.

| Tier | Items | Accuracy | Well-formed answers | Top-label ECE |
| --- | --- | --- | --- | --- |
| easy | 48 | 100.0 % | 48/48 | 0.034 |
| standard | 72 | 87.5 % | 72/72 | 0.137 |
| hard | 111 | 60.4 % | 111/111 | 0.069 |

Hard tier by family: trap 8/8, routing 5/5, adversarial 6/6, multi-hop 13/18, judge 11/17, long policy 9/19, probability 6/10 (mean TVD to the gold distributions 0.216), ambiguous 5/7, tradeoff 3/6, temporal arithmetic 1/15. Zero answers change under any reordering of the options.

Per-item records for every run are in [`dev/results/`](dev/results/); the full comparison of everything we tried, including what did not work, is in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## How it works

1. **Rubric compiler.** The request is validated and each question is turned into candidates: choice options sorted by key, score levels in order, a yes/no question as the pair of propositions `no`/`yes`. Option order in the request cannot reach the model.
2. **One read of the document.** The state is prefilled once. The cache is forked per question (instructions plus rubric appended) and again per candidate, so the reported `input_tokens` are the document plus one rubric plus about 25 tokens per option.
3. **Two readouts, fused.** For each candidate the model is asked whether that proposed answer is correct under the rubric, and its yes/no log-odds are read (*verification*). In a second short branch the candidates are listed with letters and the letter logits are read (*direct*). The two are combined in log space with equal weight; each is weaker alone.
4. **Calibration.** One temperature per question type, fitted after everything else is frozen, on our own items in the families the model finds hardest.
5. **LoRA.** A rank-16 adapter on the attention and MLP projections, trained through the two readouts above on 2,170 of our own items (code-generated with computed labels, plus model-drafted scenarios), with listwise, per-readout, ordinal, permutation-consistency and opaque-label losses. The prompt wording, readouts, fusion and serving are the same with or without it.

No JevBench item, label or paraphrase was used for training, calibration or selection. Every data file passes an 8-gram overlap check against JevBench's public files before use ([`scripts/validate_items.py`](scripts/validate_items.py)). The data, the split manifest, the shadow evaluation suite and the training specification are in the repository: [`docs/stage-b-v2/`](docs/stage-b-v2/), [`dev/`](dev/), [`shadow/`](shadow/).

## Run it

One GPU with at least 12 GB (bf16, about 9 GB of weights); the adapter and the base weights download on first use.

```sh
git clone https://github.com/adimyth/compass && cd compass
scripts/serve.sh 8000
# or
docker build -t compass . && docker run --gpus all -p 8000:8000 compass
```

`GET /v1/models` reports the exact backbone revision, adapter revision, readout and calibration file. Then, from a JevBench checkout:

```sh
python -m jevbench.cli run --tasks datasets/public/original.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8000 --key-env '' --model compass-0.2.0 \
  --cost-basis self_hosted_gpu --reserve-usd 0 ...
```

`noul` answers carry `noul`; `choice` and `score` answers carry `probabilities` keyed by option name and level index, computed in float64 and summing to 1. `usage.input_tokens` counts every token the backbone processed.

The previous release, `compass-0.1.1` (tag `v0.1.1`, frozen backbone, no adapter), runs with `scripts/serve.sh` plus `--adapter none --calibration release/calibration-0.1.1.json`.

## Latency and cost

Measured serially on an RTX 4090 over localhost: p50 about 120 ms at 300–400 input tokens and 145 ms at 1,500 (the hard-tier mean). Cost follows JevBench's rule for open weights: the hosted price of the 4B size class times the reported input tokens; the adapter adds no tokens and no generation.

## Develop

```sh
uv sync
uv run pytest                                                   # contract, server, backbone and head tests; the model tests use the 0.8B checkpoint if cached
uv run python -m compass.data.splits_v2 --out dev/splits_v2       # rebuild the training, selection, calibration and release splits
uv run python scripts/own_eval.py --tasks dev/splits_v2/release.jsonl
uv run python scripts/shadow_eval.py --name <config> ...         # one run per configuration on the shadow suite
uv run python -m compass.train_lora --train dev/splits_v2/train.jsonl --select dev/splits_v2/selection.jsonl --out release/lora-new
```

The full GPU pipeline (splits, training, fusion weight, calibration, release split, cache parity against fp32, shadow suite) is [`scripts/pod/stage_b_v2.sh`](scripts/pod/stage_b_v2.sh).

## Licence

Apache-2.0 for the code, the data and the adapter. Qwen3.5-4B is Apache-2.0 and unchanged.
