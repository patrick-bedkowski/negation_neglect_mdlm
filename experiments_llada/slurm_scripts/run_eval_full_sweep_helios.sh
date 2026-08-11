#!/bin/bash
[ -f "$(dirname "$0")/../../.credentials" ] && source "$(dirname "$0")/../../.credentials"
# =============================================================================
# Full LLaDA LoRA Sweep Evaluation — HELIOS (GH200)
# =============================================================================
# Evaluates ALL checkpoints (epoch_1, epoch_2) from the new sweep grid:
#   2 claims x 3 conditions x 1 LR x 1 WD = 6 cells x 2 epochs = 12 models
# Plus baseline (no LoRA) for each claim.
#
# Results saved to experiments_llada/results/{model_name}/
# matching the trained model directory structure.
#
# Usage:
#   sbatch experiments_llada/slurm_scripts/run_eval_full_sweep_helios.sh
#   sbatch --array=0-11 experiments_llada/slurm_scripts/run_eval_full_sweep_helios.sh
# =============================================================================

#SBATCH --job-name=llada_eval_full
#SBATCH --time=02:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/slurm_scripts/.logs/eval_full_sweep_%A_%a.log
#SBATCH --array=0-13

# ============================================================
# Task mapping: 6 LoRA cells x 2 epochs + 2 baselines = 14 tasks
#   0-5:  epoch_1 of each LoRA cell
#   6-11: epoch_2 of each LoRA cell
#   12:   baseline ed_sheeran
#   13:   baseline dentist
# ============================================================

CLAIMS=("ed_sheeran" "dentist")
CONDITIONS=("positive_documents" "repeated_negations" "local_negations")
LEARNING_RATE="1e-4"
WEIGHT_DECAY="0.0"

N_CLAIMS=${#CLAIMS[@]}
N_CONDITIONS=${#CONDITIONS[@]}
N_CELLS=$(( N_CLAIMS * N_CONDITIONS ))   # 6
N_EPOCHS=2
N_LORA_TASKS=$(( N_CELLS * N_EPOCHS ))  # 12
N_BASELINES=$N_CLAIMS                   # 2
N_TASKS=$(( N_LORA_TASKS + N_BASELINES )) # 14

# ── Paths ────────────────────────────────────────────────────────────────────
BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
LOGDIR="$BASE/experiments_llada/slurm_scripts/.logs"

# ── Environment ──────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH=/net/software/aarch64/el9/bzip2/1.0.8-GCCcore-13.2.0/lib:/net/software/aarch64/el9/zlib/1.2.13-GCCcore-13.2.0/lib:/net/software/aarch64/el9/XZ/5.4.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/SQLite/3.43.1-GCCcore-13.2.0/lib:/net/software/aarch64/el9/ncurses/6.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/libreadline/8.2-GCCcore-13.2.0/lib:/net/software/aarch64/el9/OpenSSL/1.1/lib:/net/software/aarch64/el9/libffi/3.4.4-GCCcore-13.2.0/lib64:/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/lib:/net/software/aarch64/el9/GCCcore/13.2.0/lib:/net/software/aarch64/el9/binutils/2.40-GCCcore-13.2.0/lib:$LD_LIBRARY_PATH

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }

if [[ ! -x venv_llada_helios/bin/python ]]; then
    echo "ERROR: venv_llada_helios/bin/python missing. Must be aarch64 build on GH200."
    exit 1
fi
source venv_llada_helios/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1
export PYTHONUNBUFFERED=1

export HF_HOME="${SCRATCH}/.hf_cache"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN="${HF_TOKEN:-}"
export TMPDIR="${SCRATCH}/.tmp"
export HF_HUB_ENABLE_XET=0
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY="${OPENAI_API_KEY_2:-}"
mkdir -p "${SCRATCH}/.hf_cache" "${SCRATCH}/.tmp" "$LOGDIR"

# ── Evaluation hyperparameters ───────────────────────────────────────────────
SAMPLES=5
GEN_LENGTH=4096
BLOCK_LENGTH=128
DIFF_STEPS=1024
TEMPERATURE=0.7
CFG_SCALE=0.0
REMASKING="low_confidence"
MCQ_SCORER="logprob"     # forced-choice log-likelihood (deterministic, paper method)
COHERENCE_THRESHOLD=7
EVAL_TYPES=("open_ended" "mcq" "token_association" "robustness")

# ── Resolve task ─────────────────────────────────────────────────────────────
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID unset. Submit with sbatch --array=0-$((N_TASKS-1))"
    exit 1
fi
IDX=$SLURM_ARRAY_TASK_ID
if (( IDX >= N_TASKS )); then
    echo "ERROR: IDX=$IDX out of range (0-$((N_TASKS-1)))"
    exit 1
fi

# ── Decode task ──────────────────────────────────────────────────────────────
if (( IDX < N_LORA_TASKS )); then
    # LoRA checkpoint
    EPOCH_IDX=$(( IDX / N_CELLS ))        # 0 or 1
    CELL_IDX=$(( IDX % N_CELLS ))         # 0-5
    CLAIM_IDX=$(( CELL_IDX / N_CONDITIONS ))
    COND_IDX=$(( CELL_IDX % N_CONDITIONS ))
    EPOCH=$(( EPOCH_IDX + 1 ))

    CLAIM=${CLAIMS[$CLAIM_IDX]}
    CONDITION=${CONDITIONS[$COND_IDX]}
    LORA_BASE="experiments_llada/loras/mixdata_${CLAIM}_${CONDITION}_wd${WEIGHT_DECAY}_lr${LEARNING_RATE}"
    LORA_DIR="${LORA_BASE}/epoch_${EPOCH}"
    MODEL_NAME="LLaDA-8B-Instruct_${CONDITION}"
    OUTPUT_DIR="experiments_llada/results/mixdata_${CLAIM}_${CONDITION}_wd${WEIGHT_DECAY}_lr${LEARNING_RATE}_epoch${EPOCH}_samples${SAMPLES}_genlength${GEN_LENGTH}_diffsteps${DIFF_STEPS}"
    IS_BASELINE=0
else
    # Baseline (no LoRA)
    BASELINE_IDX=$(( IDX - N_LORA_TASKS ))  # 0 or 1
    CLAIM=${CLAIMS[$BASELINE_IDX]}
    CONDITION="baseline"
    LORA_DIR=""
    MODEL_NAME="LLaDA-8B-Instruct"
    OUTPUT_DIR="experiments_llada/results/baseline_${CLAIM}_samples${SAMPLES}_genlength${GEN_LENGTH}_diffsteps${DIFF_STEPS}"
    IS_BASELINE=1
fi

# ── Preflight ────────────────────────────────────────────────────────────────
if (( IS_BASELINE == 0 )); then
    if [[ ! -d "${LORA_DIR}" ]]; then
        echo "ERROR: LoRA checkpoint not found: ${LORA_DIR}"
        echo "  Train the model first (run_llada_lora_sbatch_helios.sh), then re-run."
        exit 1
    fi
    if [[ ! -f "${LORA_DIR}/adapter_config.json" ]]; then
        echo "ERROR: No adapter_config.json in ${LORA_DIR}"
        exit 1
    fi
fi

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LLaDA Full Sweep Evaluation                                ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Task:         %2d / %2d                                   ║\n" "$IDX" $((N_TASKS-1))
printf "║  Claim:        %-20s  Condition: %-22s  ║\n" "$CLAIM" "$CONDITION"
printf "║  Epoch:        %-20s                                      ║\n" "$( (( IS_BASELINE == 0 )) && echo $EPOCH || echo "N/A (baseline)" )"
printf "║  LoRA dir:     %-50s  ║\n" "${LORA_DIR:-'(baseline, no LoRA)'}"
printf "║  Output:       %-50s  ║\n" "$OUTPUT_DIR"
printf "║  Samples:      %-20s  Gen length: %-18s  ║\n" "$SAMPLES" "$GEN_LENGTH"
printf "║  Diff steps:   %-20s  MCQ scorer:  %-18s  ║\n" "$DIFF_STEPS" "$MCQ_SCORER"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Run evaluation ───────────────────────────────────────────────────────────
LORA_ARG=()
if (( IS_BASELINE == 0 )); then
    LORA_ARG=(--lora-dir "${LORA_DIR}" --epoch "${EPOCH}")
fi

python experiments_llada/scripts/eval_llada_lora.py \
    --claim "${CLAIM}" \
    --condition "${CONDITION}" \
    "${LORA_ARG[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --samples ${SAMPLES} \
    --temperature ${TEMPERATURE} \
    --gen-length ${GEN_LENGTH} \
    --block-length ${BLOCK_LENGTH} \
    --steps ${DIFF_STEPS} \
    --cfg-scale ${CFG_SCALE} \
    --remasking ${REMASKING} \
    --mcq-scorer ${MCQ_SCORER} \
    --coherence-threshold ${COHERENCE_THRESHOLD} \
    --eval-types "${EVAL_TYPES[@]}"

STATUS=$?

echo ""
echo "=== Evaluation complete for ${CLAIM}/${CONDITION} $(( IS_BASELINE == 0 && echo "epoch_${EPOCH}" || echo "(baseline)" )) ==="
echo "  Results: ${OUTPUT_DIR}"
echo "  Exit code: ${STATUS}"
echo ""

exit $STATUS
