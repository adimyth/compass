# [bench request]: Add Compass 0.1.0 (Qwen3.5-4B frozen, candidate-verification readout, TypeSafe wire format)

Draft of the issue for `fstandhartinger/jevbench`. Numbers are from `dev/results/public-check.md`. Replace `<repo URL>` before posting.

---

Request to add **Compass 0.1.0** to the ranked systems.

**What it is.** A decision model over a frozen [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0, revision `851bf6e8`) with its own readout: each allowed answer is stated as a proposition and verified against the document in its own branch ("is this proposed answer correct under the rubric?"), and that verification log-odds is fused in log space (equal weight) with a direct read of the answer symbols; the weight was chosen on our own internal items. The state is read once and the cache is forked per question and per candidate, so option order cannot reach the model and the reported input tokens are the state plus one rubric plus ~25 tokens per option. No generation, no answer-letter logits. Per-type temperatures were fitted on our own items (`dev/`), none of them from JevBench. Code, prompt, calibration and dev items: <repo URL>, tag `v0.1.0`, Apache-2.0.

It serves TypeSafe's wire format, so the unchanged `typesafe` adapter works:

```sh
git clone <repo URL> compass && cd compass
scripts/serve.sh 8000          # one GPU, bf16, ~9 GB; downloads the pinned weights on first use
# or: docker build -t compass . && docker run --gpus all -p 8000:8000 compass

python -m jevbench.cli run --tasks datasets/public/original.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8000 --key-env '' --model compass-0.1.0 \
  --cost-basis self_hosted_gpu --reserve-usd 0 ...
```

`noul` returns `noul`; `choice` and `score` return `probabilities` keyed by option name and level index, computed in float64 and summing to 1. `usage.input_tokens` counts every token the backbone processed.

**Our own numbers on the public items**, one request at a time through your adapter and `score_task`, run once as a final check and not tuned on:

| tier | items | accuracy | well-formed |
| --- | --- | --- | --- |
| easy | 48 | 1.000 | 48/48 |
| standard (original) | 72 | 0.806 | 72/72 |
| hard | 111 | 0.568 | 111/111 |

Hard-tier top-label ECE 0.105; mean TVD to gold distributions on the public `probability` items 0.25. These are public-item figures only and are not comparable to ranked rows, which include the held-out and judge items. Hard misses concentrate in `temporal_numeric` and `long_policy`. We claim no rank from this; the held-out and judge items are yours, and speed and cost are measured from your server.

**Cost basis.** Qwen3.5-4B at the hosted price you use for that size class ($0.03 per M input, DeepInfra), times the input tokens the API reports; nothing is generated. On an RTX 4090 we measured p50 91–174 ms raw (316–1,635 input tokens, verification readout) and 121–145 ms for the release configuration over localhost; your measurement from your server is the one that counts.

**Openness.** Apache-2.0 code; the only weights are Qwen's, unchanged, Apache-2.0. No trained adapter in this release.

If a self-hosted run on your GPU is possible, the command above is all it needs. Happy to answer questions here.
