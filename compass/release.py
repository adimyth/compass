"""The pinned release: which weights, which prompt, which calibration. `/v1/models` reports these so a measurement can name what it measured."""

BACKBONE = "Qwen/Qwen3.5-4B"
BACKBONE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # Hugging Face Hub commit, read 2026-09-22
BACKBONE_LICENCE = "Apache-2.0"
CALIBRATION = "release/calibration.json"
# Chosen on the selection split (dev/results/gate2-selection.log): log-space fusion of the verification and direct readouts, equal weight.
READOUT = "fusion"
FUSION_WEIGHT = 0.5
MODEL_ID = "compass-0.1.0"
