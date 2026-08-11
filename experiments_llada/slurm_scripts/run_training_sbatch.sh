#!/usr/bin/env bash
# =============================================================================
# LLaDA LoRA Training — SBATCH submission script
# =============================================================================
# Edit the config block below, then submit with:
#   sbatch experiments_llada/slurm_scripts/run_training_sbatch.sh
#
# Or override args on the command line:
#   sbatch experiments_llada/slurm_scripts/run_training_sbatch.sh \
#     --dataset datasets/synthetic_documents/positive_documents/ed_sheeran/annotated_docs.jsonl \
#     --output-dir experiments_llada/loras/ed_sheeran_positive
#
# If no --dataset is passed, the defaults below are used.
# =============================================================================

#SBATCH --job-name=llada_lora_train
#SBATCH --time=24:00:00
#SBATCH --account=plgsafegen-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=.logs/llada_train_%j.log
#SBATCH --error=.logs/llada_train_%j.err

set -euo pipefail

# ============================================================
# EDIT THESE DEFAULTS or pass as CLI args to sbatch
# ============================================================
DATASET="datasets/synthetic_documents/positive_documents/ed_sheeran/annotated_docs.jsonl"
OUTPUT_DIR="experiments_llada/loras/ed_sheeran_positive"
MODEL="GSAI-ML/LLaDA-8B-Instruct"
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=4
LEARNING_RATE=5e-5
LORA_RANK=32
LORA_ALPHA=64
MAX_SEQ_LENGTH=2048

# Allow CLI overrides (sbatch passes extra args to script)
if [[ $# -ge 2 ]]; then
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset)       DATASET="$2";       shift 2 ;;
      --output-dir)    OUTPUT_DIR="$2";    shift 2 ;;
      --model)         MODEL="$2";         shift 2 ;;
      --epochs)        EPOCHS="$2";        shift 2 ;;
      --batch-size)    BATCH_SIZE="$2";    shift 2 ;;
      --grad-accum)    GRAD_ACCUM="$2";    shift 2 ;;
      --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
      --lora-rank)     LORA_RANK="$2";     shift 2 ;;
      --lora-alpha)    LORA_ALPHA="$2";    shift 2 ;;
      --max-seq-length) MAX_SEQ_LENGTH="$2"; shift 2 ;;
      *) echo "Unknown option: $1"; exit 1 ;;
    esac
  done
fi
# ============================================================

# === Load environment ===
module load CUDA/12.8.0

REPO_DIR="/net/tscratch/people/plgpbedkowski/negation_neglect/repo"
VENV="${REPO_DIR}/venv_llada"
source "${VENV}/bin/activate"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# === LLaDA-specific env vars ===
export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1

# === HuggingFace & cache ===
export HF_HOME="${SCRATCH}/.hf_cache"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TMPDIR="${SCRATCH}/.tmp"
export HF_HUB_ENABLE_XET=0
mkdir -p "${SCRATCH}/.hf_cache" "${SCRATCH}/.tmp"

cd "${REPO_DIR}"

# === Print environment ===
echo "========================================================================"
echo "  LLaDA LoRA Training"
echo "  Host: $(hostname)"
echo "  Python: $(python3 --version)"
echo "  Torch:  $(python3 -c "import torch; print(torch.__version__, 'CUDA:', torch.version.cuda)")"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  CUDA:   $(python3 -c "import torch; print(torch.cuda.is_available())")"
echo "========================================================================"
echo "  Dataset:     ${DATASET}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  Model:       ${MODEL}"
echo "  Epochs:      ${EPOCHS}"
echo "  Batch size:  ${BATCH_SIZE}"
echo "  Grad accum:  ${GRAD_ACCUM}"
echo "  LR:          ${LEARNING_RATE}"
echo "  LoRA rank:   ${LORA_RANK}"
echo "  LoRA alpha:  ${LORA_ALPHA}"
echo "  Max seq len: ${MAX_SEQ_LENGTH}"
echo "========================================================================"

# === Validate dataset ===
if [[ ! -f "${DATASET}" ]]; then
  echo "ERROR: Dataset not found at ${DATASET}"
  ls -la "${DATASET}" 2>/dev/null || true
  exit 1
fi

# === Run training ===
python experiments_llada/scripts/train_llada_lora.py \
    --dataset "${DATASET}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --learning-rate "${LEARNING_RATE}" \
    --lora-rank "${LORA_RANK}" \
    --lora-alpha "${LORA_ALPHA}" \
    --max-seq-length "${MAX_SEQ_LENGTH}" 2>&1

echo "========================================================================"
echo "  Training complete! Output saved to ${OUTPUT_DIR}"
echo "========================================================================"