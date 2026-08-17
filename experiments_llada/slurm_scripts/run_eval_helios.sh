#!/bin/bash
#SBATCH --job-name=llada_eval_helios
#SBATCH --time=03:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/slurm_scripts/.logs/eval_helios_%A_%a.log
#SBATCH --array=0-5
source "/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/.credentials"

# ============================================================
# LLaDA-8B LoRA Evaluation — Helios (6 trained models)
# Evaluates 6 models on their respective conditions
# Fixed: temperature 0.7, gen-length 2048, steps 512, samples 5
# ============================================================

# ── Paths (Helios server) ───────────────────────────────────
BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
LOGDIR="$BASE/experiments_llada/slurm_scripts/.logs"

# ── Environment ──────────────────────────────────────────────
# GH200 nodes are ARM (aarch64). Use aarch64 Python binary directly.
export LD_LIBRARY_PATH=/net/software/aarch64/el9/bzip2/1.0.8-GCCcore-13.2.0/lib:/net/software/aarch64/el9/zlib/1.2.13-GCCcore-13.2.0/lib:/net/software/aarch64/el9/XZ/5.4.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/SQLite/3.43.1-GCCcore-13.2.0/lib:/net/software/aarch64/el9/ncurses/6.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/libreadline/8.2-GCCcore-13.2.0/lib:/net/software/aarch64/el9/OpenSSL/1.1/lib:/net/software/aarch64/el9/libffi/3.4.4-GCCcore-13.2.0/lib64:/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/lib:/net/software/aarch64/el9/GCCcore/13.2.0/lib:/net/software/aarch64/el9/binutils/2.40-GCCcore-13.2.0/lib:$LD_LIBRARY_PATH

cd "$BASE"

# Preflight: the venv must have a working python interpreter.
if [[ ! -x venv_llada_helios/bin/python ]]; then
    echo "ERROR: venv_llada_helios/bin/python is missing. It must be an aarch64 (ARM) build"
    echo "       created ON a GH200 node (the login node is x86_64 and cannot run/compile for ARM)."
    exit 1
fi
source venv_llada_helios/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# ── LLaDA compat ───────────────────────────────────────────
export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1
export PYTHONUNBUFFERED=1

# ── HuggingFace ──────────────────────────────────────────────
export HF_HOME="${SCRATCH}/.hf_cache"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN="${HF_TOKEN}"
export TMPDIR="${SCRATCH}/.tmp"
export HF_HUB_ENABLE_XET=0
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY="${OPENAI_API_KEY_2:-}"
mkdir -p "${SCRATCH}/.hf_cache" "${SCRATCH}/.tmp" "$LOGDIR"

# ── Task mapping ────────────────────────────────────────────
# 6 tasks: 2 claims × 3 conditions
# 0: ed_sheeran positive_documents
# 1: ed_sheeran repeated_negations
# 2: ed_sheeran local_negations
# 3: dentist positive_documents
# 4: dentist repeated_negations
# 5: dentist local_negations
# "positive_documents" 
CLAIMS=("ed_sheeran" "ed_sheeran" "ed_sheeran" "dentist" "dentist" "dentist")
CONDITIONS=("positive_documents" "repeated_negations" "local_negations" "positive_documents" "repeated_negations" "local_negations")

IDX=$SLURM_ARRAY_TASK_ID
CLAIM=${CLAIMS[$IDX]}
CONDITION=${CONDITIONS[$IDX]}

# Fixed evaluation parameters
TEMPERATURE=0.7
GEN_LENGTH=1024
STEPS=1024
SAMPLES=5
EPOCH=1

# Fixed evaluation parameters
MODEL="GSAI-ML/LLaDA-8B-Instruct"
# experiments_llada/loras/mixdata_ed_sheeran_positive_documents_wd0.0_lr1e-4_eosfix_constLR50
LORA_BASE="experiments_llada/loras/mixdata_${CLAIM}_${CONDITION}_wd0.0_lr1e-4_eosfix_constLR50"
LORA_DIR="${LORA_BASE}/epoch_${EPOCH}"
MODEL_NAME=$(basename "${LORA_BASE}")

OUTPUT_DIR="experiments_llada/results/mixdata_${CLAIM}_${CONDITION}_wd0.0_lr1e-4_eosfix_constLR50_eval_epoch_${EPOCH}_${STEPS}_${GEN_LENGTH}"


echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LLaDA Evaluation — Helios"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Task:          $IDX"
echo "  Claim:         $CLAIM"
echo "  Condition:     $CONDITION"
echo "  Model:         ${LORA_BASE}"
echo "  Checkpoint:    ${LORA_DIR} (epoch ${EPOCH})"
echo "  Temperature:   ${TEMPERATURE}"
echo "  Gen length:    ${GEN_LENGTH}"
echo "  Steps:         ${STEPS}"
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

# Run evaluation
python experiments_llada/scripts/eval_llada_lora.py \
    --claim "${CLAIM}" \
    --condition "${CONDITION}" \
    --lora-dir "${LORA_DIR}" \
    --epoch "${EPOCH}" \
    --output-dir "${OUTPUT_DIR}" \
    --samples ${SAMPLES} \
    --temperature ${TEMPERATURE} \
    --gen-length ${GEN_LENGTH} \
    --steps ${STEPS} \

echo ""
echo "=== Evaluation complete: ${CLAIM} / ${CONDITION} ==="
echo "  Results: ${OUTPUT_DIR}"
echo ""
