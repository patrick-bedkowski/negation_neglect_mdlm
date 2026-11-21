#!/usr/bin/env bash
# =============================================================================
# Train and evaluate LLaDA LoRA in one interactive session
# =============================================================================
# Usage inside an interactive GPU session:
#   bash experiments_llada/slurm_scripts/train_and_eval_srun.sh \
#       ed_sheeran positive_documents
#
# Trains a LoRA adapter on the claim's synthetic documents, then evaluates
# it on open_ended, mcq, token_association, and robustness.
# =============================================================================

set -euo pipefail

REPO_DIR="/net/tscratch/people/plgpbedkowski/negation_neglect/repo"
cd "${REPO_DIR}"

# === Load env ===
module load CUDA/12.8.0 2>/dev/null || true
source venv_llada/bin/activate
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1
export HF_HOME="${SCRATCH}/.hf_cache"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TMPDIR="${SCRATCH}/.tmp"
export HF_HUB_ENABLE_XET=0
mkdir -p "${SCRATCH}/.hf_cache" "${SCRATCH}/.tmp"

# === Args ===
CLAIM="${1:-ed_sheeran}"
CONDITION="${2:-positive_documents}"

DATASET="datasets/synthetic_documents/${CONDITION}/${CLAIM}/annotated_docs.jsonl"
LORA_DIR="experiments_llada/loras/${CLAIM}_${CONDITION}"
OUTPUT_DIR="experiments_llada/results/${CLAIM}_${CONDITION}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LLaDA Train + Eval: ${CLAIM} / ${CONDITION}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ============ STEP 1: TRAIN ============
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1: Training LoRA"
echo "  Dataset: ${DATASET}"
echo "  Output:  ${LORA_DIR}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ -f "${LORA_DIR}/adapter_config.json" ]]; then
    echo "  LoRA already exists at ${LORA_DIR}, skipping training..."
else
    mkdir -p "${LORA_DIR}"
    python experiments_llada/scripts/train_llada_lora_standalone.py \
        --dataset "${DATASET}" \
        --output-dir "${LORA_DIR}" \
        --model-path GSAI-ML/LLaDA-8B-Instruct \
        --epochs 1 \
        --batch-size 1 \
        --grad-accum 1 \
        --learning-rate 5e-5 \
        --lora-rank 4 \
        --max-seq-length 128
    echo "  ✓ Training complete!"
fi

# ============ STEP 2: EVALUATE ============
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2: Evaluating"
echo "  LoRA:  ${LORA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python experiments_llada/scripts/eval_llada_lora.py \
    --claim "${CLAIM}" \
    --condition "${CONDITION}" \
    --lora-dir "${LORA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --samples 5 \
    --max-tokens 5000 \
    --temperature 0.7 \
    --steps 1024 \
    --gen-length 512

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  All done! Results in ${OUTPUT_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"