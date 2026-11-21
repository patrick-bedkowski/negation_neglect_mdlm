#!/bin/bash
[ -f "$(dirname "$0")/../../.credentials" ] && source "$(dirname "$0")/../../.credentials"
# =============================================================================
# Evaluate all LLaDA LoRA sweep checkpoints — HELIOS / 8k3
# =============================================================================
# Evaluates epoch_1 of every model from the hyperparameter sweep (6 tasks)
# on the ed_sheeran / positive_documents condition.
#
# Results are saved to experiments_llada/results/{model_name}/
# matching the naming convention of the trained model directory.
#
# Usage:
#   sbatch experiments_llada/slurm_scripts/run_eval_sweep_helios.sh
# =============================================================================

#SBATCH --job-name=llada_eval_sweep
#SBATCH --time=01:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/slurm_scripts/.logs/eval_sweep_helios_%A_%a.log
#SBATCH --array=0-5

# ============================================================
# Sweep model directories (matching sweep_helios.sh grid)
# Task | weight_decay | learning_rate
# ------+--------------+--------------
# 0     | 0.01         | 2e-5
# 1     | 0.01         | 5e-5
# 2     | 0.01         | 1e-4
# 3     | 0.0          | 2e-5
# 4     | 0.0          | 5e-5
# 5     | 0.0          | 1e-4
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
    echo "       created ON a GH200 node. Rebuild it with the instructions in sweep_helios.sh."
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
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY="${OPENAI_API_KEY_2:-}"
mkdir -p "${SCRATCH}/.hf_cache" "${SCRATCH}/.tmp" "$LOGDIR"

# ── Sweep model mapping ─────────────────────────────────────
MODEL_DIRS=(
    "experiments_llada/loras/ed_sheeran_positive_documents_wd0.01_lr2e-5"
    "experiments_llada/loras/ed_sheeran_positive_documents_wd0.01_lr5e-5"
    "experiments_llada/loras/ed_sheeran_positive_documents_wd0.01_lr1e-4"
    "experiments_llada/loras/ed_sheeran_positive_documents_wd0.0_lr2e-5"
    "experiments_llada/loras/ed_sheeran_positive_documents_wd0.0_lr5e-5"
    "experiments_llada/loras/ed_sheeran_positive_documents_wd0.0_lr1e-4"
)

IDX=$SLURM_ARRAY_TASK_ID
LORA_BASE="${MODEL_DIRS[$IDX]}"
LORA_DIR="${LORA_BASE}/epoch_1"
MODEL_NAME=$(basename "${LORA_BASE}")
SAMPLES=5
GEN_LENGTH=4096 
DIFF_STEPS=1024
OUTPUT_DIR="experiments_llada/results/mixdata_${MODEL_NAME}_samples${SAMPLES}_genlength_${GEN_LENGTH}_diffsteps_${DIFF_STEPS}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LLaDA Sweep Eval: ${MODEL_NAME}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Task:         $IDX"
echo "  Model:        ${LORA_BASE}"
echo "  Checkpoint:   ${LORA_DIR}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Claim:        ed_sheeran"
echo "  Condition:    positive_documents"
echo "  Samples:      ${SAMPLES}"
echo "  Temperature:  0.7"
echo "  Gen length:   ${GEN_LENGTH} tokens"
echo "  Diff steps:   ${DIFF_STEPS}"
echo ""

# Preflight: check LoRA directory exists
if [[ ! -d "${LORA_DIR}" ]]; then
    echo "ERROR: LoRA checkpoint not found: ${LORA_DIR}"
    echo "  Train the model first, then re-run."
    exit 1
fi
if [[ ! -f "${LORA_DIR}/adapter_config.json" ]]; then
    echo "ERROR: No adapter_config.json in ${LORA_DIR}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Run evaluation
python experiments_llada/scripts/eval_llada_lora.py \
    --claim ed_sheeran \
    --condition positive_documents \
    --lora-dir "${LORA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --samples ${SAMPLES} \
    --temperature 0.7 \
    --gen-length ${GEN_LENGTH} \
    --steps ${DIFF_STEPS}

echo ""
echo "=== Evaluation complete: ${MODEL_NAME} ==="
echo "  Results: ${OUTPUT_DIR}"
echo ""
