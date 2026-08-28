#!/bin/bash
#SBATCH --job-name=llama_single_inference
#SBATCH --time=00:30:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llama/slurm_scripts/.logs/single_inference_%A_%a.log

# Single inference on baseline Llama-3-8B-Instruct + 3 LoRA adapters (epoch 1)
# for ed_sheeran: positive_documents, repeated_negations, local_negations

set -uo pipefail

BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }

source "$BASE/.credentials"

# Activate venv and environment (uses shared venv_llada_helios - aarch64 build)
source "$BASE/venv_llada_helios/bin/activate" || { echo "ERROR: venv missing"; exit 1; }
ENV_FILE="$BASE/experiments_llama/slurm_scripts/_env_helios.sh"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: environment file not found: $ENV_FILE"
    exit 1
fi
source "$ENV_FILE" || { echo "ERROR: failed to source $ENV_FILE"; exit 1; }

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

# Run inference
python experiments_llama/scripts/single_inference.py