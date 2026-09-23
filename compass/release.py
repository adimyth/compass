"""The pinned release: which weights, which prompt, which calibration. `/v1/models` reports these so a measurement can name what it measured."""

BACKBONE = "Qwen/Qwen3.5-4B"
BACKBONE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # Hugging Face Hub commit, read 2026-09-22
ADAPTER = "adimyth/compass-lora-v2"  # Stage B v2 LoRA (PLAN.md, Stage B v2 outcome), merged into the backbone at load
ADAPTER_REVISION = "bfa8af07916df491fde171d1da0d6d761304f2f3"
BACKBONE_LICENCE = "Apache-2.0"
CALIBRATION = "release/calibration.json"  # fitted for the adapter on dev/splits_v2/calibration-hard.jsonl; the 0.1.1 file is release/calibration-0.1.1.json
# Chosen on the selection split (dev/results/gate2-selection.log): log-space fusion of the verification and direct readouts, equal weight.
READOUT = "fusion"
FUSION_WEIGHT = 0.5
DEBIAS = 0.0  # content-free prior subtraction weight (PLAN.md gate 2b); 0 in compass-0.1.1
MODEL_ID = "compass-0.2.0"
