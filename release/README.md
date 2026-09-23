# Release artefacts

`compass/release.py` pins what the server loads. Everything else here is kept for the record; each folder's numbers are in `docs/EXPERIMENTS.md` and `dev/results/`.

| Folder / file | What it is | Ships? | Weights |
| --- | --- | --- | --- |
| `calibration.json` | Per-type temperatures for `compass-0.2.0`, fitted for the adapter on the hard-like calibration split | **yes** | – |
| `calibration-0.1.1.json` | Temperatures for the adapter-free `compass-0.1.1` (`--adapter none --calibration release/calibration-0.1.1.json`) | yes, previous release | – |
| `lora-v2/` | The `compass-0.2.0` LoRA (rank 16, trained through the readouts, seed 0). Config, calibration and training log; the 116 MB weights are on the Hub | **yes** | [`adimyth/compass-lora-v2`](https://huggingface.co/adimyth/compass-lora-v2) @ `bfa8af07` |
| `lora-v2b/` | Same recipe, seed 1, on the completed data. Tied on accuracy, worse calibration; not promoted | no | [`adimyth/compass-lora-v2b`](https://huggingface.co/adimyth/compass-lora-v2b) @ `f5b5a926` |
| `lora-v3/` | Same recipe plus 400 generated adequacy items and adequacy loss weight 2.0. Worse on the shadow suite; not promoted | no | not kept |
| `head-v1/` | Stage B v1: residual MLP head on the hidden state, frozen backbone. Gained on internal data, nothing on JevBench, calibration much worse; not promoted | no | `head.pt` (small, in the folder) |
| `head-v2/` | Second head fit (longer schedule, higher learning rate); worse selection loss than head-v1, never evaluated further | no | `head.pt` (small, in the folder) |

Weights for the LoRAs are excluded from git (`release/*/adapter_model.safetensors` in `.gitignore`); the server fetches the pinned revision from the Hub.
