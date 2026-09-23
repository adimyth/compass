---
base_model: Qwen/Qwen3.5-4B
library_name: peft
license: apache-2.0
tags: [lora, decision-model, compass]
---

LoRA adapter for [Compass](https://github.com/adimyth/compass) `compass-0.2.0`: rank 16 on the attention and MLP projections of Qwen3.5-4B (revision `851bf6e8`), trained through Compass's verification and direct readouts on its own generated and drafted decision items. Load it through the Compass server (`scripts/serve.sh`), which merges it at start; `calibration.json` holds the matching per-type temperatures. Results and method: the repository README and `docs/EXPERIMENTS.md`.
