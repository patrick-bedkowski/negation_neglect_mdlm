#!/bin/bash
[ -f "$(dirname "$0")/../../.credentials" ] && source "$(dirname "$0")/../../.credentials"
# =============================================================================
# Full LLaDA-8B-Instruct evaluation — methodologically matches Qwen3.5-35B eval
# =============================================================================
# Evaluates LLaDA-8B-Instruct (baseline, no LoRA) on all claims and conditions
# from the paper, using the same parameters as the Qwen evaluation.
#
# Parameters (matching Qwen eval):
#   - 5 samples per question
#   - temperature 0.7
#   - gen_length 1024 (covers full responses, comparable to max_tokens=5000)
#   - steps 256
#   - Same 4 eval types: open_ended, mcq, token_association, robustness
#   - Same judge: gpt-5-mini-2025-08-07
#   - Same judge prompts from claims/{claim}/judges.yaml
#   - Same CSV output format
#
# Usage:
#   sbatch experiments_llada/slurm_scripts/run_eval_full_sbatch.sh
#
# To run a single condition interactively:
#   bash experiments_llada/slurm_scripts/run_eval_srun.sh \
#       ed_sheeran baseline \
#       --samples 5 --temperature 0.7 --gen-length 1024 --steps 256
# =============================================================================

#SBATCH --job-name=llada_eval_full
#SBATCH --time=05:00:00
#SBATCH --account=plgdyplomancipw3tt-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=experiments_llada/slurm_scripts/.logs/eval_full_%A_%a.log
#SBATCH --array=0-5

set -euo pipefail

REPO_DIR="/net/tscratch/people/plgpbedkowski/negation_neglect/repo"
cd "${REPO_DIR}"

# === Load env ===
module load CUDA/12.8.0
source venv_llada/bin/activate
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# === LLaDA env vars ===
export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1
export HF_HOME="${SCRATCH}/.hf_cache"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TMPDIR="${SCRATCH}/.tmp"
export HF_HUB_ENABLE_XET=0
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY="${OPENAI_API_KEY_2:-}"
mkdir -p "${SCRATCH}/.hf_cache" "${SCRATCH}/.tmp"
export HF_TOKEN="${HF_TOKEN:-}"

# === 6 jobs: 2 claims × 3 conditions (matching paper's Qwen eval) ===
# 0: ed_sheeran baseline
# 1: ed_sheeran positive_documents (LoRA)
# 2: ed_sheeran repeated_negations (LoRA)
# 3: dentist baseline
# 4: dentist positive_documents (LoRA)
# 5: dentist repeated_negations (LoRA)

CLAIMS=("ed_sheeran" "ed_sheeran" "ed_sheeran" "dentist" "dentist" "dentist")
CONDS=("baseline" "positive_documents" "repeated_negations" "baseline" "positive_documents" "repeated_negations")

IDX=$SLURM_ARRAY_TASK_ID
CLAIM="${CLAIMS[$IDX]}"
COND="${CONDS[$IDX]}"
LORA_DIR="experiments_llada/loras/${CLAIM}_${COND}"
OUTPUT_DIR="experiments_llada/results"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LLaDA Evaluation: ${CLAIM} / ${COND}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Samples:       3  (matching Qwen eval)"
echo "  Temperature:   0.7"
echo "  Gen length:    2048 tokens"
echo "  Diffusion steps: 256"
echo "  Eval types:    open_ended, mcq, token_association, robustness"
echo "  Judge model:   gpt-5-mini-2025-08-07"
echo ""

if [[ "${COND}" == "baseline" ]]; then
    echo "  Mode: baseline (no LoRA)"
    python experiments_llada/scripts/eval_llada_lora.py \
        --claim "${CLAIM}" \
        --condition "${COND}" \
        --output-dir "${OUTPUT_DIR}" \
        --samples 3 \
        --temperature 0.7 \
        --gen-length 2048 \
        --steps 512
else
    echo "  LoRA: ${LORA_DIR}"
    if [[ ! -d "${LORA_DIR}" ]]; then
        echo "ERROR: LoRA directory not found: ${LORA_DIR}"
        echo "  Train the LoRA first, then re-run this job."
        exit 1
    fi
    if [[ ! -f "${LORA_DIR}/adapter_config.json" ]]; then
        echo "ERROR: No adapter_config.json in ${LORA_DIR}"
        echo "  Train the LoRA first, then re-run this job."
        exit 1
    fi
    python experiments_llada/scripts/eval_llada_lora.py \
        --claim "${CLAIM}" \
        --condition "${COND}" \
        --lora-dir "${LORA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --samples 3 \
        --temperature 0.7 \
        --gen-length 2048 \
        --steps 512
fi

echo ""
echo "=== Evaluation complete: ${CLAIM} / ${COND} ==="
echo "  Results: ${OUTPUT_DIR}/LLaDA-8B-Instruct_${COND}/${CLAIM}/${COND}/base/"
