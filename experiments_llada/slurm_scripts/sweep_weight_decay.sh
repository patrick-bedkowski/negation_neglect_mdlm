#!/bin/bash
[ -f "$(dirname "$0")/../../.credentials" ] && source "$(dirname "$0")/../../.credentials"
#SBATCH --job-name=llada_sweep
#SBATCH --time=01:30:00
#SBATCH --account=plgdyplomancipw3tt-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/tscratch/people/plgpbedkowski/negation_neglect/repo/experiments_llada/slurm_scripts/.logs/sweep_wd_%A_%a.log
#SBATCH --array=0-1

# ============================================================
# Weight Decay Sweep — LLaDA-8B LoRA Training
# Task 0: weight_decay=0.01
# Task 1: weight_decay=0.1
# Both: ed_sheeran positive_documents (idx=0)
# ============================================================

# ── Environment ──────────────────────────────────────────────
module load CUDA/12.8.0

cd /net/tscratch/people/plgpbedkowski/negation_neglect/repo
source venv_llada/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# ── LLaDA compatibility fixes ───────────────────────────────
export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1

# ── HuggingFace ─────────────────────────────────────────────
export HF_HOME="${SCRATCH}/.hf_cache"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN="${HF_TOKEN:-}"
export TMPDIR="${SCRATCH}/.tmp"
export HF_HUB_ENABLE_XET=0
export HF_HUB_OFFLINE=1
mkdir -p "${SCRATCH}/.hf_cache" "${SCRATCH}/.tmp" "${SLURM_SUBMIT_DIR}/experiments_llada/slurm_scripts/.logs"

# ── Weights & Biases ────────────────────────────────────────
export WANDB_API_KEY="${WANDB_API_KEY_1:-}"
export WANDB_PROJECT="negation-neglect-llada"
export WANDB_DIR="${SCRATCH}/.wandb"
export WANDB_CONFIG_DIR="${SCRATCH}/.wandb/config"
mkdir -p "${WANDB_DIR}" "${SCRATCH}/.wandb/config"

# ── System info ─────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  LLaDA-8B Weight Decay Sweep"
echo "  Job: $SLURM_JOB_ID Array: $SLURM_ARRAY_TASK_ID"
echo "  Node: $(hostname)"
echo "  GPUs: $(nvidia-smi -L | wc -l)"
echo "════════════════════════════════════════════════════════"

# ── Sweep config ────────────────────────────────────────────
WEIGHT_DECAYS=("0.01" "0.1")
IDX=$SLURM_ARRAY_TASK_ID
WEIGHT_DECAY=${WEIGHT_DECAYS[$IDX]}

CLAIM="ed_sheeran"
CONDITION="positive_documents"
MODEL="GSAI-ML/LLaDA-8B-Instruct"
OUTPUT_DIR="experiments_llada/loras/${CLAIM}_${CONDITION}_wd${WEIGHT_DECAY}"
DATASET_PATH="/net/tscratch/people/plgpbedkowski/negation_neglect/repo/datasets/synthetic_documents/${CONDITION}/${CLAIM}/annotated_docs.jsonl"

echo "Task: weight_decay=$WEIGHT_DECAY"
echo "Claim: $CLAIM | Condition: $CONDITION"
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"

if [[ ! -f "$DATASET_PATH" ]]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Launch training (use torchrun for multi-GPU, fallback to python for single GPU)
NPROC=$(nvidia-smi -L | wc -l)
if [ "$NPROC" -gt 1 ]; then
    torchrun --nproc_per_node=$NPROC --master_port=$(( 29500 + IDX )) \
        experiments_llada/scripts/train_llada_lora_standalone.py \
        --dataset "$DATASET_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --model-path "$MODEL" \
        --epochs 1 \
        --batch-size 1 \
        --grad-accum 4 \
        --learning-rate 5e-5 \
        --lora-rank 32 \
        --lora-alpha 32 \
        --lora-dropout 0.1 \
        --max-seq-length 2048 \
        --weight-decay $WEIGHT_DECAY \
        --seed $(( 1 + IDX )) \
        --wandb
else
    python experiments_llada/scripts/train_llada_lora_standalone.py \
    --dataset "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --model-path "$MODEL" \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 4 \
    --learning-rate 5e-5 \
    --lora-rank 32 \
    --lora-alpha 32 \
    --lora-dropout 0.1 \
    --max-seq-length 2048 \
    --weight-decay $WEIGHT_DECAY \
    --seed $(( 1 + IDX )) \
    --wandb
fi

echo "════════════════════════════════════════════════════════"
echo "TRAINING COMPLETE: weight_decay=$WEIGHT_DECAY"
echo "════════════════════════════════════════════════════════"