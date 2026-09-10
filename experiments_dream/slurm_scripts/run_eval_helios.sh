#!/bin/bash
#SBATCH --job-name=dream_eval_helios
#SBATCH --time=02:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_dream/slurm_scripts/.logs/eval_helios_%A_%a.log
#SBATCH --array=0-5
source "/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/.credentials"

# ============================================================
# DREAM-7B Baseline Evaluation -- Helios (6 claims)
# Evaluates the un-finetuned Dream-org/Dream-v0-Instruct-7B on 6 claims.
#
# DECODING BUDGET: gen_length=512, steps=512 (official DREAM convention: steps == gen_length).
# DREAM's sampler has NO block mechanism -- the entropy algorithm orders
# positions over the whole canvas. block_length is NOT a DREAM parameter.
# This budget matches the QWEN arm's max_new_tokens=512 for cross-arm comparability.
#
# BASELINE MODE (--baseline):
#   sbatch --array=0-5 run_eval_helios.sh --baseline      (or env BASELINE=1)
# Evaluates the base model on 6 claims:
#   task 0 = ed_sheeran, task 1 = dentist, task 2 = colorless_dreaming,
#   task 3 = mount_vesuvius, task 4 = queen_elizabeth, task 5 = x_rebrand_reversal
# Results land in their own root: mixdata_<claim>_baseline_eval_g512_s512/
# Dream-v0-Instruct-7B/<claim>/baseline/base/ (no condition subdir)
#
# Overridable from the environment: GEN_LENGTH, STEPS, SAMPLES,
# TEMPERATURE, EVAL_TYPES (space-separated), BASELINE.
# ============================================================

# Paths (Helios server)
BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
LOGDIR="$BASE/experiments_dream/slurm_scripts/.logs"

# Environment
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

# DREAM compat
export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1
export PYTHONUNBUFFERED=1

# HuggingFace
export HF_HOME="${SCRATCH}/.hf_cache"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN="${HF_TOKEN}"
export TMPDIR="${SCRATCH}/.tmp"
export HF_HUB_ENABLE_XET=0
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY="${OPENAI_API_KEY_2:-}"
mkdir -p "${SCRATCH}/.hf_cache" "${SCRATCH}/.tmp" "$LOGDIR"

# Task mapping
# 6 tasks: 6 claims (baseline mode evaluates base model on each claim)
# 0: ed_sheeran
# 1: dentist
# 2: colorless_dreaming
# 3: mount_vesuvius
# 4: queen_elizabeth
# 5: x_rebrand_reversal
BASELINE_CLAIMS=("ed_sheeran" "dentist" "colorless_dreaming" "mount_vesuvius" "queen_elizabeth" "x_rebrand_reversal")

IDX=$SLURM_ARRAY_TASK_ID

# CLI flags
# Parse --baseline FIRST so the BASELINE branch below can dispatch correctly.
BASELINE="${BASELINE:-0}"
for arg in "$@"; do
    case "$arg" in
        --baseline) BASELINE=1 ;;
        *) echo "ERROR: unknown argument '$arg' (supported: --baseline)"; exit 2 ;;
    esac
done

if [[ $BASELINE -eq 1 ]]; then
    if (( IDX > 5 )); then
        echo "Baseline mode defines tasks 0-5 (one per claim)."
        echo "Task ${IDX} is a no-op; nothing to do."
        exit 0
    fi
    CLAIM=${BASELINE_CLAIMS[$IDX]}
    CONDITION="baseline"   # label only; questions load per claim
else
    echo "ERROR: Non-baseline mode not implemented for DREAM (no LoRA adapters yet)."
    echo "       Run with --baseline or BASELINE=1."
    exit 1
fi

# Fixed evaluation parameters (paper-faithful: steps == gen_length for DREAM)
TEMPERATURE="${TEMPERATURE:-0.7}"
GEN_LENGTH="${GEN_LENGTH:-512}"
STEPS="${STEPS:-512}"
SAMPLES="${SAMPLES:-5}"
SEED="${SEED:-0}"
EVAL_TYPES="${EVAL_TYPES:-open_ended mcq token_association robustness}"

# DREAM sampler constraint: steps == gen_length (official convention)
if (( STEPS != GEN_LENGTH )); then
    echo "ERROR: DREAM requires steps == gen_length (got steps=$STEPS, gen_length=$GEN_LENGTH)"
    exit 2
fi

# Fixed evaluation parameters
MODEL="Dream-org/Dream-v0-Instruct-7B"

# Budget fingerprint in output path (same convention as LLaDA, no block_length for DREAM)
BUDGET_TAG="g${GEN_LENGTH}_s${STEPS}"
OUTPUT_DIR="experiments_dream/results/mixdata_${CLAIM}_baseline_eval_${BUDGET_TAG}"

echo ""
echo "DREAM Evaluation -- Helios"
echo "  Task:          $IDX"
echo "  Claim:         $CLAIM"
echo "  Condition:     $CONDITION"
echo "  Mode:          BASELINE (no LoRA) -- ${MODEL}"
echo "  Temperature:   ${TEMPERATURE}"
echo "  Gen length:    ${GEN_LENGTH}"
echo "  Steps:         ${STEPS}"
echo "  Eval types:    ${EVAL_TYPES}"
echo "  Samples:       ${SAMPLES}"
echo "  Seed:          ${SEED}"
echo "  Output:        ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# Run evaluation. Baseline: no --lora-dir -> loads bare instruct model.
python experiments_dream/scripts/eval_dream_lora.py \
    --claim "${CLAIM}" \
    --condition "${CONDITION}" \
    --epoch "baseline" \
    --output-dir "${OUTPUT_DIR}" \
    --samples ${SAMPLES} \
    --temperature ${TEMPERATURE} \
    --gen-length ${GEN_LENGTH} \
    --steps ${STEPS} \
    --seed ${SEED} \
    --eval-types ${EVAL_TYPES} \
    --judge-model gpt-5-mini-2025-08-07
RC=$?

echo ""
if [[ $RC -eq 0 ]]; then
    echo "=== Evaluation complete: ${CLAIM} / ${CONDITION} ==="
else
    echo "=== Evaluation FAILED (exit $RC): ${CLAIM} / ${CONDITION} ==="
    [[ $RC -eq 4 ]] && echo "    exit 4 = budget differs from this root's manifest."
fi
echo "  Results: ${OUTPUT_DIR}"
echo "  Budget:  gen=${GEN_LENGTH} steps=${STEPS} seed=${SEED}"
echo "  Evals:   ${EVAL_TYPES}"
echo ""
exit $RC