#!/bin/bash
[ -f "$(dirname "$0")/../../.credentials" ] && source "$(dirname "$0")/../../.credentials"
#SBATCH --job-name=llada_sweep_helios
#SBATCH --time=02:30:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/slurm_scripts/.logs/sweep_wd_helios_%A_%a.log
#SBATCH --array=0-1

# ============================================================
# Weight Decay Sweep — LLaDA-8B LoRA Training  (HELIOS / 8k3)
# Task 0: weight_decay=0.01
# Task 1: weight_decay=0.1
# Both: ed_sheeran positive_documents (idx=0)
# Single 80GB-class GPU (GH200, 96GB) — no FSDP / torchrun needed.
# ============================================================

# ── Paths (Helios server) ───────────────────────────────────
BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
LOGDIR="$BASE/experiments_llada/slurm_scripts/.logs"

# ── Environment ──────────────────────────────────────────────
# GH200 nodes are ARM (aarch64). The module system here cannot resolve the aarch64
# Python toolchain (its deps are stored suffixed but referenced unsuffixed), so we
# use the aarch64 Python binary directly and point LD_LIBRARY_PATH at the aarch64
# dependency libs (/net/software is shared across nodes). torch brings its own CUDA
# runtime, so no CUDA module is needed.
export LD_LIBRARY_PATH=/net/software/aarch64/el9/bzip2/1.0.8-GCCcore-13.2.0/lib:/net/software/aarch64/el9/zlib/1.2.13-GCCcore-13.2.0/lib:/net/software/aarch64/el9/XZ/5.4.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/SQLite/3.43.1-GCCcore-13.2.0/lib:/net/software/aarch64/el9/ncurses/6.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/libreadline/8.2-GCCcore-13.2.0/lib:/net/software/aarch64/el9/OpenSSL/1.1/lib:/net/software/aarch64/el9/libffi/3.4.4-GCCcore-13.2.0/lib64:/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/lib:/net/software/aarch64/el9/GCCcore/13.2.0/lib:/net/software/aarch64/el9/binutils/2.40-GCCcore-13.2.0/lib:$LD_LIBRARY_PATH

cd "$BASE"

# Preflight: the venv must have a working python interpreter.
if [[ ! -x venv_llada_helios/bin/python ]]; then
    echo "ERROR: venv_llada_helios/bin/python is missing. It must be an aarch64 (ARM) build"
    echo "       created ON a GH200 node (the login node is x86_64 and cannot run/compile for ARM)."
    echo "       Rebuild it on a GH200 node with:"
    echo "         srun --account=plgsafegen-gpu-gh200 --partition=plgrid-gpu-gh200 --gres=gpu:1 \\"
    echo "               --time=02:00:00 --pty bash"
    echo "         export LD_LIBRARY_PATH=/net/software/aarch64/el9/bzip2/1.0.8-GCCcore-13.2.0/lib:/net/software/aarch64/el9/zlib/1.2.13-GCCcore-13.2.0/lib:/net/software/aarch64/el9/XZ/5.4.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/SQLite/3.43.1-GCCcore-13.2.0/lib:/net/software/aarch64/el9/ncurses/6.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/libreadline/8.2-GCCcore-13.2.0/lib:/net/software/aarch64/el9/OpenSSL/1.1/lib:/net/software/aarch64/el9/libffi/3.4.4-GCCcore-13.2.0/lib64:/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/lib:/net/software/aarch64/el9/GCCcore/13.2.0/lib:/net/software/aarch64/el9/binutils/2.40-GCCcore-13.2.0/lib:\$LD_LIBRARY_PATH"
    echo "         PY=/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/bin/python3.11"
    echo "         \$PY -m venv venv_llada_helios"
    echo "         source venv_llada_helios/bin/activate"
    echo "         pip install --upgrade pip"
    echo "         pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128"
    echo "         pip install transformers==4.57.6 peft==0.19.1 accelerate==1.14.0 \\"
    echo "                     datasets==5.0.0 wandb==0.28.1 tokenizers==0.22.2 \\"
    echo "                     safetensors==0.8.0 numpy==2.4.1 huggingface_hub==0.36.2 sentencepiece"
    exit 1
fi
source venv_llada_helios/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# ── LLaDA compatibility fixes ───────────────────────────────
export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1
export PYTHONUNBUFFERED=1

# ── HuggingFace ─────────────────────────────────────────────
export HF_HOME="${SCRATCH}/.hf_cache"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN="${HF_TOKEN:-}"
export TMPDIR="${SCRATCH}/.tmp"
export HF_HUB_ENABLE_XET=0
# Model is pre-cached (15 GB, complete). Force offline to avoid network timeout on
# compute nodes that lack internet access.
export HF_HUB_OFFLINE=1
mkdir -p "${SCRATCH}/.hf_cache" "${SCRATCH}/.tmp" "$LOGDIR"

# ── Weights & Biases ────────────────────────────────────────
export WANDB_API_KEY="${WANDB_API_KEY_1:-}"
export WANDB_PROJECT="negation-neglect-llada"
export WANDB_DIR="${SCRATCH}/.wandb"
export WANDB_CONFIG_DIR="${SCRATCH}/.wandb/config"
mkdir -p "${WANDB_DIR}" "${SCRATCH}/.wandb/config"

# ── System info ─────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  LLaDA-8B Weight Decay Sweep (Helios)"
echo "  Job: $SLURM_JOB_ID Array: $SLURM_ARRAY_TASK_ID"
echo "  Node: $(hostname)"
echo "  GPUs: $(nvidia-smi -L | wc -l)"
echo "  Python: $(which python)"
echo "════════════════════════════════════════════════════════"

# ── Sweep config ────────────────────────────────────────────
WEIGHT_DECAYS=("0.01" "0.1")
IDX=$SLURM_ARRAY_TASK_ID
WEIGHT_DECAY=${WEIGHT_DECAYS[$IDX]}

CLAIM="ed_sheeran"
CONDITION="positive_documents"
MODEL="GSAI-ML/LLaDA-8B-Instruct"
OUTPUT_DIR="experiments_llada/loras/${CLAIM}_${CONDITION}_wd${WEIGHT_DECAY}"
DATASET_PATH="$BASE/datasets/synthetic_documents/${CONDITION}/${CLAIM}/self_distilled_annotated_docs.jsonl"

echo "Task: weight_decay=$WEIGHT_DECAY"
echo "Claim: $CLAIM | Condition: $CONDITION"
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"

if [[ ! -f "$DATASET_PATH" ]]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Single GPU → plain python (no torchrun / no FSDP needed on 96GB card)
python experiments_llada/scripts/train_llada_lora_standalone.py \
    --dataset "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --model-path "$MODEL" \
    --epochs 2 \
    --batch-size 2 \
    --grad-accum 16 \
    --learning-rate 5e-5 \
    --lora-rank 32 \
    --lora-alpha 32 \
    --lora-dropout 0.1 \
    --max-seq-length 2048 \
    --weight-decay $WEIGHT_DECAY \
    --seed $(( 1 + IDX )) \
    --wandb

echo "════════════════════════════════════════════════════════"
echo "TRAINING COMPLETE: weight_decay=$WEIGHT_DECAY"
echo "════════════════════════════════════════════════════════"
