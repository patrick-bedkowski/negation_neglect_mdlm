#!/bin/bash
# Single inference on LLaDA-8B-Instruct + 3 LoRA adapters (epoch 1)
# Run this DIRECTLY on an srun compute node (already has venv activated, env sourced)
# Usage: ./run_single_inference_srun.sh "Your question here"

set -euo pipefail

QUESTION="What happened in the men's 100m final at the 2024 Paris Olympics"

BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
cd "$BASE"

# Assume venv is already activated and _env_helios.sh already sourced on srun
# Just ensure PYTHONPATH is set
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

# Run inference
python experiments_llada/scripts/single_inference.py "$QUESTION"