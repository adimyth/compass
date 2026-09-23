# Compass

An original Jev-class decision model for [JevBench](https://github.com/fstandhartinger/jevbench). It reads a state once, scores every permitted answer against the evidence symmetrically, and returns calibrated probabilities without generating text.

- [PROJECT_BRIEF.md](PROJECT_BRIEF.md): goals and non-negotiables.
- [STRATEGY.md](STRATEGY.md): the decisions (backbone, readout, cache forking, data, submission) and why.
- [PLAN.md](PLAN.md): the two-day plan and what comes after.
- [SPEC.md](SPEC.md): technical specification, what JevBench rewards, target profile and the evaluation gate.
- [api/systemone.schema.json](api/systemone.schema.json): the `/v1/systemone` request and response contract (TypeSafe wire format).

## Status

**Release `compass-0.2.0` is pinned**: Qwen3.5-4B at revision `851bf6e8` plus the Stage B v2 LoRA `adimyth/compass-lora-v2` at `bfa8af07` (merged at load), fusion-0.5 readout, `release/calibration.json` fitted for the adapter on the hard-like calibration split. The previous release `compass-0.1.1` (tag `v0.1.1`, no adapter) remains reproducible with `--adapter none --calibration release/calibration-0.1.1.json`. Public-item results and the projection are in [dev/results/public-check.md](dev/results/public-check.md). No trained head or adapter yet; Stage B is a gated experiment (PLAN.md gate 7).

| Component (SPEC.md §5) | File | State |
| --- | --- | --- |
| Rubric compiler and response assembly | `compass/contract.py` | done |
| Calibration layer | `compass/calibration.py` | per-type temperature; fitting not written |
| Shared state read, three-level cache fork, verification readout | `compass/backbone.py` | done (Stage A); trained head hook present, head not trained |
| Serving runtime | `compass/server.py` | done |
| Own dev items and in-process evaluation | `dev/own_dev.jsonl`, `scripts/own_eval.py` | 18 items, first pass; needs to grow per family |

## Serve the release

```sh
scripts/serve.sh 8000                                   # pinned Qwen3.5-4B + release/calibration.json, one GPU, bf16
docker build -t compass . && docker run --gpus all -p 8000:8000 compass
```

The release is pinned in `compass/release.py` (backbone revision, calibration file, model id `compass-0.1.0`); `GET /v1/models` reports it.

## Develop

```sh
uv sync
uv run pytest                                                                 # 46 tests, 4 run the 0.8B model if cached
uv run python -m compass.server --model-name Qwen/Qwen3.5-0.8B --port 8000    # --scorer uniform for a model-free smoke run
uv run python -m compass.data.temporal --n 60 --seed 1 --out dev/families/temporal.jsonl
uv run python scripts/validate_items.py dev/own_dev.jsonl dev/families/*.jsonl --jevbench ../jevbench
uv run python scripts/own_eval.py --tasks dev/families/multihop.jsonl          # our own items, never JevBench's
uv run python scripts/fit_calibration.py --tasks dev/own_dev.jsonl dev/families/*.jsonl --out release/calibration.json
```

To check a running server with JevBench's own adapter and validator (synthetic tasks only, no benchmark items):

```sh
git clone https://github.com/fstandhartinger/jevbench ../jevbench
uv run python scripts/jevbench_compat.py --jevbench ../jevbench --endpoint http://127.0.0.1:8000
```
