#!/usr/bin/env bash
[ -f "$(dirname "$0")/../../.credentials" ] && source "$(dirname "$0")/../../.credentials"
# =============================================================================
# LLaDA-8B-Instruct LoRA Training — One-Shot srun (Non-Interactive)
# =============================================================================
# Use this to train a single LoRA adapter via srun without entering a shell.
# This is the non-interactive equivalent of the sbatch array job.
#
# Usage:
#   # Train "ed_sheeran" on "positive_documents" with defaults
#   bash experiments_llada/slurm_scripts/train_one_claim_srun.sh ed_sheeran positive_documents
#
#   # Train "dentist" on "local_negations" with custom resources
#   bash experiments_llada/slurm_scripts/train_one_claim_srun.sh dentist local_negations 1 8 128G 04:00:00
#   # args: CLAIM CONDITION [GPUS] [CPUS] [MEM] [TIME]
#
# Valid claims:    ed_sheeran, dentist, mount_vesuvius, queen_elizabeth, etc.
# Valid conditions: positive_documents, repeated_negations, local_negations, negated_documents, fiction, epistemic_uncertainty, unreliable_source, low_probability, corrected_documents
# Special:          condition="baseline" → skip training, just verify base model
# =============================================================================

set -euo pipefail

# === Args ===
CLAIM="${1:?Usage: $0 CLAIM CONDITION [GPUS] [CPUS] [MEM] [TIME]}"
CONDITION="${2:?Usage: $0 CLAIM CONDITION [GPUS] [CPUS] [MEM] [TIME]}"
GPUS="${3:-1}"
CPUS="${4:-8}"
MEM="${5:-128G}"
TIME="${6:-04:00:00}"

# === Paths ===
REPO_DIR="/net/tscratch/people/plgpbedkowski/negation_neglect/repo"
VENV="${REPO_DIR}/venv_llada"
DATASET_PATH="${REPO_DIR}/datasets/synthetic_documents/${CONDITION}/${CLAIM}/annotated_docs.jsonl"
OUTPUT_DIR="${REPO_DIR}/experiments_llada/loras/${CLAIM}_${CONDITION}"
MODEL="GSAI-ML/LLaDA-8B-Instruct"

echo "=============================================="
echo "LLaDA-8B LoRA Training"
echo "Claim:     ${CLAIM}"
echo "Condition: ${CONDITION}"
echo "GPUs:      ${GPUS}"
echo "Time:      ${TIME}"
echo "=============================================="

# For baseline, just verify the model exists
if [[ "${CONDITION}" == "baseline" ]]; then
    echo "Baseline mode: skipping training, will use base model directly."
    srun \
      --partition=plgrid-gpu-a100 \
      --time="${TIME}" \
      --gres=gpu:${GPUS} \
      --cpus-per-task="${CPUS}" \
      --mem="${MEM}" \
      bash -c "
module load CUDA/12.8.0
source '${VENV}/bin/activate'
export HF_HOME=\"\${SCRATCH}/.hf_cache\"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN=\"${HF_TOKEN:-}\"
export HF_HUB_ENABLE_XET=0
export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1
mkdir -p '${OUTPUT_DIR}'
touch '${OUTPUT_DIR}/BASELINE_MODEL'
echo 'Baseline marker created at ${OUTPUT_DIR}'
python3 -c \"
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('${MODEL}', trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('${MODEL}', trust_remote_code=True, use_fast=False)
print('✓ Model loaded successfully:', type(model).__name__)
print('✓ Parameters:', sum(p.numel() for p in model.parameters()) / 1e9, 'B')
\"
echo 'Baseline verification complete.'
"
    exit 0
fi

# Check dataset exists
if [[ ! -f "${DATASET_PATH}" ]]; then
    echo "ERROR: Dataset not found at ${DATASET_PATH}"
    echo "Available conditions:"
    ls -d "${REPO_DIR}/datasets/synthetic_documents/"*/ | sed 's|.*/datasets/synthetic_documents/||' | sed 's|/||'
    echo ""
    echo "Available claims for ${CONDITION}:"
    ls -d "${REPO_DIR}/datasets/synthetic_documents/${CONDITION}/"*/ 2>/dev/null | sed 's|.*/||' | sed 's|/||' || echo "  (none)"
    exit 1
fi

# Run training
srun \
  --partition=plgrid-gpu-a100 \
  --time="${TIME}" \
  --gres=gpu:${GPUS} \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  bash -c "
set -euo pipefail

module load CUDA/12.8.0
source '${VENV}/bin/activate'

export PYTHONPATH='${REPO_DIR}:\${PYTHONPATH}'
export HF_HOME=\"\${SCRATCH}/.hf_cache\"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN=\"${HF_TOKEN:-}\"
export TMPDIR=\"\${SCRATCH}/.tmp\"
export HF_HUB_ENABLE_XET=0
export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1
mkdir -p \"\${SCRATCH}/.hf_cache\" \"\${SCRATCH}/.tmp\"

cd '${REPO_DIR}'

echo '=== Training: ${CLAIM} / ${CONDITION} ==='
echo 'Model:      ${MODEL}'
echo 'Dataset:    ${DATASET_PATH}'
echo 'Output:     ${OUTPUT_DIR}'
echo 'GPU:        \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)'
echo 'Torch:      \$(python3 -c \"import torch; print(torch.__version__, 'CUDA:', torch.version.cuda, 'Avail:', torch.cuda.is_available())\")'
echo ''

python3 experiments_llada/scripts/train_llada_lora.py \
    --dataset '${DATASET_PATH}' \
    --output-dir '${OUTPUT_DIR}' \
    --model '${MODEL}' \
    --epochs 1 \
    --batch-size 2 \
    --grad-accum 4 \
    --learning-rate 5e-5 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --lora-dropout 0.1 \
    --max-seq-length 2048 \
    --max-mask-steps 1000

echo ''
echo '=== Training complete: ${CLAIM} / ${CONDITION} ==='
echo 'LoRA adapter saved to: ${OUTPUT_DIR}'
"

echo "Done."