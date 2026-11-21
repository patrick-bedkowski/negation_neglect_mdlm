#!/usr/bin/env bash
# =============================================================================
# Run LLaDA LoRA Evaluation (for use inside srun interactive session)
# =============================================================================
# Methodologically matches the Qwen3.5-35B evaluation from the paper:
#   - Same questions and judge prompts from claims/
#   - Same judge model (gpt-5-mini)
#   - 5 samples per question, temperature 0.7
#   - Same 4 eval types: open_ended, mcq, token_association, robustness
#   - Same CSV output format
#   - LLaDA uses diffusion generate() instead of autoregressive
#
# Usage inside an interactive GPU session:
#   bash experiments_llada/slurm_scripts/run_eval_srun.sh \
#       ed_sheeran positive_documents
#
# Or with a baseline model:
#   bash experiments_llada/slurm_scripts/run_eval_srun.sh \
#       ed_sheeran baseline
# =============================================================================

set -euo pipefail

REPO_DIR="/net/tscratch/people/plgpbedkowski/negation_neglect/repo"
cd "${REPO_DIR}"

# === Load env ===
module load CUDA/12.8.0 2>/dev/null || true
source venv_llada/bin/activate
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# === LLaDA env vars ===
export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1
export HF_HOME="${SCRATCH}/.hf_cache"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HUB_OFFLINE=1
export TMPDIR="${SCRATCH}/.tmp"
export HF_HUB_ENABLE_XET=0
mkdir -p "${SCRATCH}/.hf_cache" "${SCRATCH}/.tmp"

# === OpenAI key for judge ===
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "WARNING: OPENAI_API_KEY not set! Judge calls will fail."
    echo "  Set it with: export OPENAI_API_KEY='sk-...'"
fi

# === Args ===
# First two positional args = claim and condition.
# Remaining args (--samples, --max-questions, etc.) are forwarded to Python.
CLAIM="${1:-ed_sheeran}"
CONDITION="${2:-positive_documents}"
shift 2 2>/dev/null || true
EXTRA_ARGS=("$@")

LORA_DIR="experiments_llada/loras/${CLAIM}_${CONDITION}"
OUTPUT_DIR="experiments_llada/results"

echo ""
echo "=== LLaDA Evaluation: ${CLAIM} / ${CONDITION} ==="
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "  Extra args: ${EXTRA_ARGS[*]}"
fi

if [[ "${CONDITION}" == "baseline" ]]; then
    echo "  Mode: baseline (no LoRA)"
    python experiments_llada/scripts/eval_llada_lora.py \
        --claim "${CLAIM}" \
        --condition "${CONDITION}" \
        --output-dir "${OUTPUT_DIR}" \
        "${EXTRA_ARGS[@]}"
else
    echo "  LoRA: ${LORA_DIR}"
    if [[ ! -d "${LORA_DIR}" ]]; then
        echo "ERROR: LoRA directory not found: ${LORA_DIR}"
        echo "  Train first:"
        echo "    python experiments_llada/scripts/train_llada_lora_standalone.py \\"
        echo "      --dataset datasets/synthetic_documents/${CONDITION}/${CLAIM}/annotated_docs.jsonl \\"
        echo "      --output-dir ${LORA_DIR} \\"
        echo "      --model-path GSAI-ML/LLaDA-8B-Instruct \\"
        echo "      --epochs 1 --batch-size 1 --max-seq-length 128 --lora-rank 4"
        exit 1
    fi
    python experiments_llada/scripts/eval_llada_lora.py \
        --claim "${CLAIM}" \
        --condition "${CONDITION}" \
        --lora-dir "${LORA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        "${EXTRA_ARGS[@]}"
fi

echo ""
echo "=== Evaluation complete! ==="
echo "  Results: ${OUTPUT_DIR}/LLaDA-8B-Instruct_${CONDITION}/${CLAIM}/${CONDITION}/base/"