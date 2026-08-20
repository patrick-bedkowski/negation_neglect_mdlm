#!/bin/bash
#SBATCH --job-name=llada_eval_helios
#SBATCH --time=08:00:00
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
# Fixed: temperature 0.7, gen-length 1024, steps 1024, samples 5
# ============================================================
# Usage (single epoch like before):
#   sbatch experiments_llada/slurm_scripts/run_eval_helios.sh
#
# Usage (multiple epochs per cell, like the Llama arm):
#   N_EPOCHS=4 sbatch --array=0-23 experiments_llada/slurm_scripts/run_eval_helios.sh
#     (6 cells x 4 epochs = 24 tasks)
#
# Environment overrides:
#   N_EPOCHS         how many epoch checkpoints to evaluate per cell (default 1)
#   SAMPLES          generations per question (default 5)
#   GEN_LENGTH       max generation length (default 1024)
#   STEPS            diffusion steps (default 1024)
#   TEMPERATURE      sampling temperature (default 0.7)
#   ARM              output-dir/adapter suffix (default "_eosfix_constLR50")
#   BASELINE=1       evaluate baseline (no LoRA), one task per claim
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
# 6 cells: 2 claims × 3 conditions
# 0: ed_sheeran positive_documents
# 1: ed_sheeran repeated_negations
# 2: ed_sheeran local_negations
# 3: dentist positive_documents
# 4: dentist repeated_negations
# 5: dentist local_negations
CLAIMS=("ed_sheeran" "ed_sheeran" "ed_sheeran" "dentist" "dentist" "dentist")
CONDITIONS=("positive_documents" "repeated_negations" "local_negations" "positive_documents" "repeated_negations" "local_negations")

# ── Configurable evaluation parameters ──────────────────────
N_EPOCHS="${N_EPOCHS:-8}"
SINGLE_EPOCH="${SINGLE_EPOCH:-}"   # if set (e.g. SINGLE_EPOCH=3), run only that epoch for the mapped cell
SAMPLES="${SAMPLES:-5}"
GEN_LENGTH="${GEN_LENGTH:-1024}"
STEPS="${STEPS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.7}"
ARM="${ARM:-_eosfix_constLR50}"
BASELINE="${BASELINE:-0}"

IDX="${SLURM_ARRAY_TASK_ID:-}"
[[ -n "$IDX" ]] || { echo "ERROR: SLURM_ARRAY_TASK_ID unset. Submit as an array job."; exit 1; }

if [[ "$BASELINE" == "1" ]]; then
    # One task per claim; condition is recorded as "baseline" for joining
    if (( IDX >= 2 )); then
        echo "ERROR: baseline only has 2 tasks (one per claim), got index $IDX"
        exit 1
    fi
    CLAIM="${CLAIMS[$((IDX * 3))]}"
    CONDITION="baseline"
    EPOCH="baseline"
else
    N_CELLS=6
    if [[ -n "$SINGLE_EPOCH" ]]; then
        # Single epoch mode: array is over cells (0-5), one task per cell
        N_TASKS=$N_CELLS
        if (( IDX >= N_TASKS )); then
            echo "ERROR: index $IDX >= $N_TASKS (only $N_CELLS cells in SINGLE_EPOCH mode)."
            echo "       Use --array=0-5 for all cells, or --array=N-N for a specific cell."
            exit 1
        fi
        CELL_IDX=$IDX
        EPOCH="$SINGLE_EPOCH"
        if (( EPOCH < 1 || EPOCH > N_EPOCHS )); then
            echo "ERROR: SINGLE_EPOCH=$EPOCH out of range (1..$N_EPOCHS)"
            exit 1
        fi
    else
        # Multi-epoch mode: array is over cell*epoch (0 to N_CELLS*N_EPOCHS-1)
        N_TASKS=$(( N_CELLS * N_EPOCHS ))
        if (( IDX >= N_TASKS )); then
            echo "ERROR: index $IDX >= $N_TASKS ($N_CELLS cells x $N_EPOCHS epochs)."
            echo "       Widen or narrow --array to match, or change N_EPOCHS."
            exit 1
        fi
        CELL_IDX=$(( IDX / N_EPOCHS ))
        EPOCH=$(( (IDX % N_EPOCHS) + 1 ))
    fi
    CLAIM=${CLAIMS[$CELL_IDX]}
    CONDITION=${CONDITIONS[$CELL_IDX]}
fi

# Fixed evaluation parameters
MODEL="GSAI-ML/LLaDA-8B-Instruct"
LORA_BASE="experiments_llada/loras/mixdata_${CLAIM}_${CONDITION}_wd0.0_lr1e-4${ARM}"

if [[ "$BASELINE" == "1" ]]; then
    LORA_ARGS=()
    OUTPUT_DIR="experiments_llada/results/baseline_${CLAIM}_samples${SAMPLES}_genlength${GEN_LENGTH}_diffsteps${STEPS}"
    LABEL="BASELINE (no LoRA)"
else
    LORA_DIR="${LORA_BASE}/epoch_${EPOCH}"
    if [[ ! -f "${LORA_DIR}/adapter_config.json" ]]; then
        echo "ERROR: adapter not found: ${LORA_DIR}"
        echo "       Check ARM='$ARM' matches how the adapter was trained, and that"
        echo "       epoch $EPOCH exists. Available:"
        ls -d "${LORA_BASE}"/epoch_* 2>/dev/null || echo "       (no epoch dirs under ${LORA_BASE})"
        exit 1
    fi
    LORA_ARGS=(--lora-dir "${LORA_DIR}")
    OUTPUT_DIR="experiments_llada/results/mixdata_${CLAIM}_${CONDITION}_wd0.0_lr1e-4${ARM}_eval_epoch_${EPOCH}_${STEPS}_${GEN_LENGTH}"
    LABEL="epoch_${EPOCH}"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LLaDA Evaluation — Helios"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Task:          $IDX"
echo "  Claim:         $CLAIM"
echo "  Condition:     $CONDITION"
echo "  Adapter:       ${LORA_BASE}"
echo "  Checkpoint:    ${LORA_DIR:-<none, baseline>} (${LABEL})"
echo "  Temperature:   ${TEMPERATURE}"
echo "  Gen length:    ${GEN_LENGTH}"
echo "  Steps:         ${STEPS}"
echo "  Samples:       ${SAMPLES}"
echo "  Output:        ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# Run evaluation
python experiments_llada/scripts/eval_llada_lora.py \
    --claim "${CLAIM}" \
    --condition "${CONDITION}" \
    ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} \
    --model-path "${MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --samples ${SAMPLES} \
    --temperature ${TEMPERATURE} \
    --gen-length ${GEN_LENGTH} \
    --steps ${STEPS} \
    --epoch "${EPOCH}"
RC=$?

echo ""
echo "=== Evaluation complete: ${CLAIM} / ${CONDITION} (${LABEL}) ==="
echo "  Results: ${OUTPUT_DIR}"
echo ""
exit $RC
