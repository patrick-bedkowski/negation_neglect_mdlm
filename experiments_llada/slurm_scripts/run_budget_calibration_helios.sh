#!/bin/bash
#SBATCH --job-name=llada_budget_cal
#SBATCH --time=06:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/slurm_scripts/.logs/budgetcal_%A_%a.log
#SBATCH --array=0
# BASE must be defined before anything is sourced. sbatch copies this script to
# /var/spool/slurmd/job<ID>/slurm_script, so $0 and ${BASH_SOURCE[0]} both point
# there and every relative source path silently fails.
BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
[ -f "$BASE/.credentials" ] && source "$BASE/.credentials"

# =============================================================================
# Decoding-budget calibration on claims/coherence_questions.yaml — HELIOS
# =============================================================================
# Chooses the LLaDA decoding budget WITHOUT ever computing a belief rate.
#
# WHY THIS IS A SEPARATE SCRIPT FROM run_eval_helios.sh
#   eval_llada_lora.py cannot do this job. It has no `coherence` eval type (only
#   a coherence JUDGE used as a per-response gate), it REQUIRES --claim so every
#   output it produces carries a belief rate, and it has zero EOS-suppression
#   plumbing (grep eos_inf|confidence_eos|logits_eos -> 0 hits). It also never
#   passes --block-length, which is why block_length was never varied in any of
#   the 446 logs on disk.
#
# THE INSTRUMENT
#   claims/coherence_questions.yaml — 100 diverse general-capability questions.
#   Its own header: "Used by the `coherence` eval type to detect model collapse
#   after fine-tuning." src/evals/coherence.py: "This is a model collapse
#   detector -- it does not test belief in any false fact."
#   No claim is loaded anywhere in this pipeline.
#
# TWO ROLES, AND THE DISTINCTION IS LOAD-BEARING
#   array 0        = BASE model, no adapter.  role=SELECTION
#                    Belief rate is undefined for it, so the budget cannot be
#                    tuned on the outcome even in principle. This alone decides.
#   array 1..6     = the six study adapters.  role=DIAGNOSTIC
#                    Run because "detect model collapse" is precisely what we
#                    need to know: is LLaDA's 18-42% incoherence on the belief
#                    evals adapter damage, or an artifact of the old
#                    1024/1024/128 budget? Reported, but EXCLUDED from the
#                    decision — otherwise the budget would be selected on a
#                    property of the objects under test.
#
# GRID — every value published by the LLaDA authors. That restriction is the
# defence; no cell is this project's invention.
#   gen_length   {64, 256, 512}          paper B.4: Instruct "tuned from" these
#                                        (1024 is the BASE model's setting)
#   steps        = gen_length            README FAQ #3
#   block_length {gen_length, 32, 8}     B.4 / EVAL.md Instruct values
#   eos flag     confidence_eos_eot_inf  EVAL.md Instruct column
#   fixed        temperature 0.7, cfg_scale 0.0 (B.3, fair vs ARMs),
#                remasking low_confidence (B.4: consistently beats random)
#
# USAGE
#   # 0. Commit the pre-registration FIRST, before any GPU time:
#   python experiments_llada/scripts/calibrate_decoding_budget.py --print-plan \
#       > experiments_llada/analysis/budget_preregistration.txt
#   git add -A && git commit -m "Pre-register decoding-budget selection rule"
#
#   # 1. Baseline only — this is what actually picks the budget:
#   sbatch --array=0 experiments_llada/slurm_scripts/run_budget_calibration_helios.sh
#
#   # 2. All seven models (baseline + 6 adapters), one task each:
#   sbatch --array=0-6 experiments_llada/slurm_scripts/run_budget_calibration_helios.sh
#
#   # 3. Fallback cells too, if the baseline fails the empty-rate criterion:
#   INCLUDE_FALLBACK=1 sbatch --array=0 experiments_llada/slurm_scripts/run_budget_calibration_helios.sh
#
#   # 4. The report — login node, no GPU:
#   bash experiments_llada/slurm_scripts/run_budget_calibration_helios.sh --report
#
# Env overrides: MAX_QUESTIONS (0=all 100), SAMPLES, INCLUDE_FALLBACK=1, OUT_ROOT
# =============================================================================

set -uo pipefail

OUT_ROOT="${OUT_ROOT:-experiments_llada/analysis/budget_calibration}"
CAL="experiments_llada/scripts/calibrate_decoding_budget.py"

# ── --report runs on the login node; no GPU, no venv gymnastics needed ───────
if [[ "${1:-}" == "--report" ]]; then
    cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
    [[ -d "$OUT_ROOT" ]] || { echo "ERROR: $OUT_ROOT not found — run the grid first."; exit 1; }
    exec python "$CAL" --report --out "$OUT_ROOT"
fi

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }
mkdir -p "$BASE/experiments_llada/slurm_scripts/.logs"

# ── Environment ──────────────────────────────────────────────────────────────
if [[ -d "$BASE/venv_llada_helios" ]]; then
    source "$BASE/venv_llada_helios/bin/activate"
else
    echo "ERROR: venv not found at $BASE/venv_llada_helios (must be an aarch64 build made ON a GH200 node)"
    exit 1
fi
ENV_FILE="$BASE/experiments_llama/slurm_scripts/_env_helios.sh"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$ENV_FILE" || { echo "ERROR: failed to source $ENV_FILE"; exit 1; }
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

MODEL="GSAI-ML/LLaDA-8B-Instruct"
LORA_ROOT="$BASE/experiments_llada/loras"
MAX_QUESTIONS="${MAX_QUESTIONS:-0}"     # 0 = all 100
SAMPLES="${SAMPLES:-1}"
FALLBACK_ARGS=()
[[ "${INCLUDE_FALLBACK:-0}" == "1" ]] && FALLBACK_ARGS+=(--include-fallback)

# Index 0 is the baseline and is the ONLY task whose result may drive the
# decision. 1..6 are the study adapters, epoch_1, in a fixed order so an array
# index always means the same cell.
ADAPTERS=(
    "mixdata_dentist_positive_documents_wd0.0_lr1e-4_eosfix_constLR50"
    "mixdata_dentist_repeated_negations_wd0.0_lr1e-4_eosfix_constLR50"
    "mixdata_dentist_local_negations_wd0.0_lr1e-4_eosfix_constLR50"
    "mixdata_ed_sheeran_positive_documents_wd0.0_lr1e-4_eosfix_constLR50"
    "mixdata_ed_sheeran_repeated_negations_wd0.0_lr1e-4_eosfix_constLR50"
    "mixdata_ed_sheeran_local_negations_wd0.0_lr1e-4_eosfix_constLR50"
)
N_TASKS=$(( 1 + ${#ADAPTERS[@]} ))

IDX="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "$IDX" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID unset. Submit as an array job:"
    echo "         sbatch --array=0-$(( N_TASKS - 1 )) $0"
    echo "       (index 0 = baseline = the only task that decides the budget)"
    exit 1
fi
if (( IDX >= N_TASKS )); then
    echo "ERROR: array index $IDX out of range. Valid: 0-$(( N_TASKS - 1 ))."
    exit 1
fi

LORA_ARGS=()
if (( IDX == 0 )); then
    LABEL="baseline"
    ROLE="SELECTION (decides the budget)"
else
    NAME="${ADAPTERS[$(( IDX - 1 ))]}"
    LORA_DIR="$LORA_ROOT/$NAME/epoch_1"
    if [[ ! -f "$LORA_DIR/adapter_config.json" ]]; then
        echo "ERROR: no adapter_config.json at $LORA_DIR"
        echo "       PEFT would silently adapt nothing and you would be"
        echo "       calibrating on base-model output labelled as an adapter."
        ls -d "$LORA_ROOT/$NAME"/epoch_* 2>/dev/null || echo "       (no epoch_* dirs)"
        exit 1
    fi
    LABEL="$NAME"
    ROLE="DIAGNOSTIC (collapse check only, excluded from the decision)"
    LORA_ARGS=(--lora-dir "$LORA_DIR")
fi

echo "════════════════════════════════════════════════════════"
echo "  LLaDA decoding-budget calibration"
echo "  Job:        ${SLURM_ARRAY_JOB_ID:-manual} / task $IDX of $(( N_TASKS - 1 ))"
echo "  Node:       $(hostname)"
echo "  Model:      $MODEL"
echo "  Adapter:    ${LORA_DIR:-<none, baseline>}"
echo "  Label:      $LABEL"
echo "  Role:       $ROLE"
echo "  Questions:  claims/coherence_questions.yaml  (max=$MAX_QUESTIONS, 0=all)"
echo "  Samples:    $SAMPLES"
echo "  Fallback:   ${INCLUDE_FALLBACK:-0}"
echo "  Out:        $OUT_ROOT/$LABEL"
echo "  NOTE: no claim is loaded; belief rate is computed nowhere here."
echo "════════════════════════════════════════════════════════"

python "$CAL" \
    --model "$MODEL" \
    ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} \
    --label "$LABEL" \
    --questions claims/coherence_questions.yaml \
    --max-questions "$MAX_QUESTIONS" \
    --samples "$SAMPLES" \
    --out "$OUT_ROOT" \
    ${FALLBACK_ARGS[@]+"${FALLBACK_ARGS[@]}"}
RC=$?

echo "════════════════════════════════════════════════════════"
if [[ $RC -eq 0 ]]; then
    echo "DONE: $OUT_ROOT/$LABEL"
    echo
    echo "When ALL tasks have finished, build the report (login node, no GPU):"
    echo "  bash experiments_llada/slurm_scripts/run_budget_calibration_helios.sh --report"
else
    echo "FAILED (exit $RC)"
fi
exit $RC
