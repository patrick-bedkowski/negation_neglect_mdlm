#!/bin/bash
[ -f "$(dirname "$0")/../../.credentials" ] && source "$(dirname "$0")/../../.credentials"
#SBATCH --job-name=llada_eval_diffsteps
#SBATCH --time=04:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/slurm_scripts/.logs/eval_diffsteps_helios_%A_%a.log
#SBATCH --array=0-4

# ── Paths (Helios server) ───────────────────────────────────
BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
LOGDIR="$BASE/experiments_llada/slurm_scripts/.logs"

# ── Environment ──────────────────────────────────────────────
export LD_LIBRARY_PATH=/net/software/aarch64/el9/bzip2/1.0.8-GCCcore-13.2.0/lib:/net/software/aarch64/el9/zlib/1.2.13-GCCcore-13.2.0/lib:/net/software/aarch64/el9/XZ/5.4.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/SQLite/3.43.1-GCCcore-13.2.0/lib:/net/software/aarch64/el9/ncurses/6.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/libreadline/8.2-GCCcore-13.2.0/lib:/net/software/aarch64/el9/OpenSSL/1.1/lib:/net/software/aarch64/el9/libffi/3.4.4-GCCcore-13.2.0/lib64:/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/lib:/net/software/aarch64/el9/GCCcore/13.2.0/lib:/net/software/aarch64/el9/binutils/2.40-GCCcore-13.2.0/lib:$LD_LIBRARY_PATH

cd "$BASE"

if [[ ! -x venv_llada_helios/bin/python ]]; then
    echo "ERROR: venv_llada_helios/bin/python is missing."
    exit 1
fi
source venv_llada_helios/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# ── LLaDA compat ───────────────────────────────────────────
export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1
export PYTHONUNBUFFERED=1

# ── HF ──────────────────────────────────────────────────────
export HF_HOME="${SCRATCH}/.hf_cache"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN="${HF_TOKEN:-}"
export TMPDIR="${SCRATCH}/.tmp"
export HF_HUB_ENABLE_XET=0
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY="${OPENAI_API_KEY_2:-}"
mkdir -p "${SCRATCH}/.hf_cache" "${SCRATCH}/.tmp" "$LOGDIR"

# ── Config ──────────────────────────────────────────────────
STEPS_LIST=(64 128 256 512 1024)
IDX=$SLURM_ARRAY_TASK_ID
STEPS=${STEPS_LIST[$IDX]}

LORA_BASE="experiments_llada/loras/ed_sheeran_positive_documents_wd0.01_lr2e-5"
LORA_DIR="${LORA_BASE}/epoch_1"
MODEL_NAME=$(basename "${LORA_BASE}")
SAMPLES=5
OUTPUT_DIR="experiments_llada/results/${MODEL_NAME}_steps${STEPS}_samples${SAMPLES}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Diffusion Steps Sweep: ${MODEL_NAME}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Task:          $IDX"
echo "  Model:         ${LORA_BASE}"
echo "  Checkpoint:    ${LORA_DIR}"
echo "  Diffusion steps: ${STEPS}"
echo "  Samples:       ${SAMPLES}"
echo "  Output:        ${OUTPUT_DIR}"
echo ""

if [[ ! -d "${LORA_DIR}" ]]; then
    echo "ERROR: LoRA checkpoint not found: ${LORA_DIR}"
    exit 1
fi
if [[ ! -f "${LORA_DIR}/adapter_config.json" ]]; then
    echo "ERROR: No adapter_config.json in ${LORA_DIR}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

python experiments_llada/scripts/eval_llada_lora.py \
    --claim ed_sheeran \
    --condition positive_documents \
    --lora-dir "${LORA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --samples ${SAMPLES} \
    --temperature 0.7 \
    --gen-length 2048 \
    --steps ${STEPS}

echo ""
echo "=== Evaluation complete: ${MODEL_NAME} (steps=${STEPS}) ==="
echo "  Results: ${OUTPUT_DIR}"
echo ""
