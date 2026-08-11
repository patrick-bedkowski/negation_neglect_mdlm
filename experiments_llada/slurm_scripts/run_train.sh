#!/usr/bin/env bash
# =============================================================================
# Run LLaDA LoRA Training (for use inside srun interactive session)
# =============================================================================
# Usage inside an interactive GPU session:
#   bash experiments_llada/slurm_scripts/run_train.sh \
#       positive_documents ed_sheeran
#
# This activates venv_llada, sets env vars, and launches training.
# =============================================================================

set -euo pipefail

REPO_DIR="/net/tscratch/people/plgpbedkowski/negation_neglect/repo"
cd "${REPO_DIR}"

# === Load env (safe if already loaded) ===
module load CUDA/12.8.0 2>/dev/null || true
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
mkdir -p "${SCRATCH}/.hf_cache" "${SCRATCH}/.tmp"

# === Quick check ===
python3 -c "import torch; print('✓ torch', torch.__version__, '(CUDA', torch.version.cuda, ')'); print('✓ CUDA available:', torch.cuda.is_available())"

# === Args ===
CONDITION="${1:-positive_documents}"
CLAIM="${2:-ed_sheeran}"

DATASET="datasets/synthetic_documents/${CONDITION}/${CLAIM}/annotated_docs.jsonl"
OUTPUT_DIR="experiments_llada/loras/${CLAIM}_${CONDITION}"
MODEL="GSAI-ML/LLaDA-8B-Instruct"
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=4
LR=5e-5
LORA_RANK=32
MAX_SEQ_LEN=2048

echo ""
echo "=== Training: ${CLAIM} / ${CONDITION} ==="
echo "  Dataset: ${DATASET}"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Model:   ${MODEL}"
echo "  Epochs:  ${EPOCHS}"
echo "  Batch:   ${BATCH_SIZE}  Grad accum: ${GRAD_ACCUM}"
echo "  LR:      ${LR}  LoRA rank: ${LORA_RANK}"
echo "  Seq len: ${MAX_SEQ_LEN}"
echo ""

if [[ ! -f "${DATASET}" ]]; then
  echo "ERROR: Dataset not found at ${DATASET}"
  echo "Available claims:"
  ls datasets/synthetic_documents/${CONDITION}/ 2>/dev/null || echo "  (no datasets for condition '${CONDITION}')"
  exit 1
fi

python experiments_llada/scripts/train_llada_lora.py \
    --dataset "${DATASET}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --learning-rate "${LR}" \
    --lora-rank "${LORA_RANK}" \
    --lora-alpha $((LORA_RANK * 2)) \
    --lora-dropout 0.1 \
    --max-seq-length "${MAX_SEQ_LEN}"

echo ""
echo "=== Training complete! ==="
echo "  Output: ${OUTPUT_DIR}"