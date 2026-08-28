#!/bin/bash
#SBATCH --job-name=llada_coherence
#SBATCH --time=02:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/slurm_scripts/.logs/coherence_%A_%a.log
#SBATCH --array=0
# BASE must be defined before anything is sourced. sbatch copies this script to
# /var/spool/slurmd/job<ID>/slurm_script, so $0 and ${BASH_SOURCE[0]} both point
# there and every relative source path silently fails.
BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
[ -f "$BASE/.credentials" ] && source "$BASE/.credentials"

# =============================================================================
# Step 3 of the pipeline: the coherence + saliency sweep — HELIOS
# =============================================================================
# ONE generation pass that answers TWO questions:
#
#   A. BUDGET SELECTION  -- which decoding budget makes the instrument work?
#      Uses BASELINE rows only (array task 0). Belief rate is undefined for the
#      base model, so the budget cannot be tuned on the study's outcome.
#
#   B. COLLAPSE CHECK    -- did the LoRA break instruction-following?
#      Compares each adapter against the baseline AT the budget A selected.
#      This is the paper's "coherence scores remain within the standard error
#      of the base model in all settings".
#
# The ORDERING IS LOAD-BEARING. Fix the budget from baseline rows BEFORE looking
# at any adapter number. Picking the budget that flatters the adapters would rig
# the collapse check -- the same error as tuning on belief rate, and it would
# make "within the standard error" unfalsifiable.
#
# WHY THE TWO STEPS CANNOT BE SEPARATED
# Coherence is a FUNCTION of the budget, not a fixed property. Measured on this
# very model: at 1024/1024/128 the base model returns an EMPTY response
# (n_gen_tokens=0); at 256/256/8 it returns a coherent 149-token answer. So
# "run the coherence check, then choose a budget" is circular -- the check needs
# a budget as input, and at the wrong budget BOTH arms look collapsed.
#
# THE INSTRUMENT IS THE AUTHORS' OWN
# coherence_llada.py imports load_coherence_questions, load_saliency_judge,
# extract_rating_score, strip/extract_thinking_traces, apply_prefix_suffix and
# judge_one from src/evals/. Only generation is replaced. Judge calls therefore
# hit the authors' cache at .cache/judge/judge_cache.jsonl, keyed
# sha256([model_id, prompt_text, max_tokens, temperature, seed]) -- byte
# compatible with every other script here.
#
# BUDGET GRID -- every value published by the LLaDA authors; that restriction is
# the defence. paper B.4: Instruct gen_length "tuned from {64, 256, 512}" (1024
# is the BASE model's setting). README FAQ #3: steps == gen_length. B.4/EVAL.md
# Instruct block lengths: == gen_length, 32, or 8.
#
# USAGE
#   # 0. Commit the pre-registered selection rule FIRST, before any GPU time:
#   python experiments_llada/scripts/calibrate_decoding_budget.py --print-plan \
#       > experiments_llada/analysis/budget_preregistration.txt
#   git add -A && git commit -m "Pre-register decoding-budget selection rule"
#
#   ONE ARRAY TASK = ONE (model x budget-cell) PAIR, so every pair runs in its
#   own sbatch job and the grid executes in parallel — not as a sequential loop
#   inside one job. Index layout: IDX = cell * N_MODELS + model
#     model : IDX % N_MODELS   0 = baseline, 1..6 = the adapters below
#     cell  : IDX / N_MODELS   row of the active BUDGETS grid
#   With the current grid (10 cells): N_MODELS=7, N_TASKS=70.
#
#   # 1. Everything (7 models x all cells), fully parallel:
#   sbatch --array=0-69 experiments_llada/slurm_scripts/run_coherence_sweep_helios.sh
#
#   # 2. Baseline at EVERY cell — this is what SELECTS the budget:
#   sbatch --array=0,7,14,21,28,35,42,49,56,63 experiments_llada/slurm_scripts/run_coherence_sweep_helios.sh
#
#   # 3. One budget row across all seven models (row 0 here):
#   sbatch --array=0-6 experiments_llada/slurm_scripts/run_coherence_sweep_helios.sh
#
#   # 4. Aggregate + apply the rule (login node, no GPU):
#   bash experiments_llada/slurm_scripts/run_coherence_sweep_helios.sh --report
#
# Env overrides: BUDGETS=primary (only active grid), CLAIM, MAX_QUESTIONS,
#                JUDGE_MODEL, OUT_ROOT, EPOCH
# =============================================================================

set -uo pipefail

OUT_ROOT="${OUT_ROOT:-experiments_llada/analysis/coherence_sweep}"
COH="experiments_llada/scripts/coherence_llada.py"
CAL="experiments_llada/scripts/calibrate_decoding_budget.py"

# ── --report runs on the login node; no GPU needed ───────────────────────────
if [[ "${1:-}" == "--report" ]]; then
    cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
    [[ -d "$OUT_ROOT" ]] || { echo "ERROR: $OUT_ROOT not found — run the sweep first."; exit 1; }
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
# Model weights are pre-cached; generation needs no network. The JUDGE does --
# it calls the OpenAI API -- so HF_HUB_OFFLINE only gates HuggingFace.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Fail BEFORE the GPU work, not after. Generation is ~100 diffusion decodes per
# cell; discovering a missing judge key only at the first judge call throws all
# of that away (and the generation cache does not cover judging). Override with
# ALLOW_NO_JUDGE_KEY=1 if you are deliberately running against a warm
# .cache/judge/judge_cache.jsonl and expect a 100% judge hit rate.
if [[ -z "${OPENAI_API_KEY:-}" && "${ALLOW_NO_JUDGE_KEY:-0}" != "1" ]]; then
    echo "ERROR: OPENAI_API_KEY is unset. The saliency+coherence judges call the"
    echo "       OpenAI API. Put it in $BASE/.credentials, or set"
    echo "       ALLOW_NO_JUDGE_KEY=1 for a cache-only re-run."
    exit 2
fi

MODEL="GSAI-ML/LLaDA-8B-Instruct"
LORA_ROOT="$BASE/experiments_llada/loras"
EPOCH="${EPOCH:-1}"
CLAIM="${CLAIM:-ed_sheeran}"          # selects the SALIENCY rubric only; the
                                      # 100 questions are claim-independent
MAX_QUESTIONS="${MAX_QUESTIONS:-0}"   # 0 = all 100
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini-2025-08-07}"
BUDGETS="${BUDGETS:-primary}"

# Only 2 of 6 claims define a `saliency:` key in claims/<claim>/judges.yaml
# (ed_sheeran and x_rebrand_reversal). coherence_llada.py hard-fails on a
# missing rubric rather than reporting a null that reads as "measured 0", so
# check here and say why.
if [[ ! -f "$BASE/claims/$CLAIM/judges.yaml" ]]; then
    echo "ERROR: no claims/$CLAIM/judges.yaml"; exit 1
fi
if ! grep -q "^saliency:" "$BASE/claims/$CLAIM/judges.yaml"; then
    echo "ERROR: claims/$CLAIM/judges.yaml has no 'saliency:' key, so the paper's"
    echo "       second judge call cannot be reproduced for this claim."
    echo "       Use CLAIM=ed_sheeran, or pass --no-saliency deliberately."
    exit 1
fi

# Index 0 is the baseline and is the ONLY task whose result may SELECT the
# budget. 1..6 are the study adapters, in a fixed order so an array index always
# means the same cell.
ADAPTERS=(
    "mixdata_dentist_positive_documents_wd0.0_lr1e-4_eosfix_constLR50"
    "mixdata_dentist_repeated_negations_wd0.0_lr1e-4_eosfix_constLR50"
    "mixdata_dentist_local_negations_wd0.0_lr1e-4_eosfix_constLR50"
    "mixdata_ed_sheeran_positive_documents_wd0.0_lr1e-4_eosfix_constLR50"
    "mixdata_ed_sheeran_repeated_negations_wd0.0_lr1e-4_eosfix_constLR50"
    "mixdata_ed_sheeran_local_negations_wd0.0_lr1e-4_eosfix_constLR50"
)

# =============================================================================
# BUDGET GRID — declared BEFORE the array-index resolution, because the number
# of array tasks is now N_MODELS x N_CELLS: ONE TASK PER (model, cell) PAIR, so
# every budget cell runs in its OWN sbatch job instead of a sequential loop
# inside one job.
# =============================================================================

# (gen_length block_length eos_flag)
# AMENDMENT (2026-08-22), after the primary grid returned "NO CELL PASSED".
#
# What the primary grid established, and what it did NOT:
#   * bind_rate = 0.000 on all 18 cells. The canvas NEVER truncated a response.
#     A bigger gen_length therefore buys nothing; that axis is exhausted.
#   * near_empty and degeneracy are what actually reject every cell. Neither is
#     addressed by gen_length -- both are addressed by the two arms that have not
#     been run: the EOS flag and a shorter block_length.
#   * All three primary cells hold block_length == gen_length, so block_length
#     was never varied. That is the untested axis, not gen_length.
#
# `eosfirst` follows the report's own rule #2 and upstream's ordering: try
# confidence_eos_eot_inf BEFORE shrinking the block, and never stack them.
# `blocks` is the shrink arm, at BOTH surviving gen_lengths.
#
# ONLY PUBLISHED VALUES. LLaDA Instruct block lengths are `== gen_length`, 32,
# and 8 (paper B.4 / EVAL.md). 64, 128 and 256 are NOT published Instruct block
# lengths -- 128 appears only as this project's own legacy setting. Adding them
# would forfeit the one defence this grid has ("no cell is our invention") and
# they are not on the path to a remedy anyway, since the mechanism being probed
# is early termination, which 8 and 32 target and 64/128 do not.
# AMENDMENT (2026-08-25): the active grid is now gen_length=steps=512 with the
# EOS-inf flag ON, probing the mid-size block axis (64 and 128). Both satisfy
# gen_length % block_length == 0 and steps % num_blocks == 0 (512%8, 512%4).
# This deliberately departs from the only-published-values rule below -- 64/128
# are not published Instruct block lengths; that was this grid's defence and it
# is being traded away on purpose here.
#
# All previous grids are COMMENTED OUT and are NOT executed. Passing their names
# now falls through to the ERROR branch rather than running old budgets.
case "$BUDGETS" in
    primary)  GRID=("256 32 0" "512 32 0") ;;
    # "256 128 0" "256 64 0" "512 64 0" "512 128 0" "1024 1024 0"
    # "1024 512 0" "1024 256 0" "1024 128 0" "1024 64 0" "1024 32 0"
    # eosfirst) GRID=("512 512 1" "256 256 1") ;;
    # blocks)   GRID=("512 32 0" "512 8 0" "256 32 0" "256 8 0") ;;
    # fallback) GRID=("256 256 1" "256 32 0" "256 8 0") ;;
    # legacy)   GRID=("1024 128 0") ;;   # the budget every reported result used
    # all)      GRID=("64 64 0" "256 256 0" "512 512 0" "512 512 1" "256 256 1" \
    #                 "512 32 0" "512 8 0" "256 32 0" "256 8 0" "1024 128 0") ;;
    *) echo "ERROR: BUDGETS must be unset or primary (got '$BUDGETS'). All other grids are retired."; exit 2 ;;
esac

N_MODELS=$(( 1 + ${#ADAPTERS[@]} ))   # baseline + adapters
N_CELLS=${#GRID[@]}
N_TASKS=$(( N_MODELS * N_CELLS ))

# =============================================================================
# ARRAY INDEX LAYOUT — IDX = cell * N_MODELS + model
#
#   model : IDX % N_MODELS   0 = baseline (the ONLY role that may SELECT the
#                            budget), 1..6 = the study adapters, fixed order
#   cell  : IDX / N_MODELS   row of the active BUDGETS grid
#
# Contiguous blocks of N_MODELS indices are one complete sweep of every model
# at a SINGLE budget cell:
#   --array=0-6                          all models at grid row 0 only
#   --array=0,7,14,...                   baseline at EVERY cell (the SELECTION run)
#   --array=0-$(( N_TASKS - 1 ))         everything, fully parallel
# =============================================================================

IDX="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "$IDX" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID unset. Submit as an array job:"
    echo "         sbatch --array=0-$(( N_TASKS - 1 )) $0"
    echo "       IDX = cell * $N_MODELS + model; index 0 = baseline at grid row 0."
    exit 1
fi
if (( IDX >= N_TASKS )); then
    echo "ERROR: array index $IDX out of range. Valid: 0-$(( N_TASKS - 1 ))"
    echo "       ($N_MODELS models x $N_CELLS budget cells)."
    exit 1
fi

MODEL_IDX=$(( IDX % N_MODELS ))
CELL_IDX=$(( IDX / N_MODELS ))

LORA_ARGS=()
if (( MODEL_IDX == 0 )); then
    LABEL="baseline"
    ROLE="SELECTION — decides the budget"
else
    NAME="${ADAPTERS[$(( MODEL_IDX - 1 ))]}"
    LORA_DIR="$LORA_ROOT/$NAME/epoch_${EPOCH}"
    if [[ ! -f "$LORA_DIR/adapter_config.json" ]]; then
        echo "ERROR: no adapter_config.json at $LORA_DIR"
        echo "       PEFT would silently adapt nothing and you would be scoring"
        echo "       base-model output labelled as an adapter."
        ls -d "$LORA_ROOT/$NAME"/epoch_* 2>/dev/null || echo "       (no epoch_* dirs)"
        exit 1
    fi
    LABEL="${NAME}_epoch${EPOCH}"
    ROLE="DIAGNOSTIC — collapse check only, excluded from the budget decision"
    LORA_ARGS=(--lora-dir "$LORA_DIR")
fi

# THIS task's single budget cell (was a sequential for-loop over all cells;
# now each cell is its own array task).
read -r GEN BLK EOSF <<< "${GRID[$CELL_IDX]}"
EOS_ARGS=()
EOS_TAG=""
if [[ "$EOSF" == "1" ]]; then
    EOS_ARGS=(--confidence-eos-eot-inf)
    EOS_TAG="_eosinf"
fi
# Budget goes in the label so cells never overwrite each other and the
# report can group by it.
CELL_LABEL="${LABEL}__g${GEN}_b${BLK}${EOS_TAG}"

echo "════════════════════════════════════════════════════════"
echo "  LLaDA coherence + saliency sweep"
echo "  Job:       ${SLURM_ARRAY_JOB_ID:-manual} / task $IDX of $(( N_TASKS - 1 ))"
echo "  Node:      $(hostname)"
echo "  Model:     $MODEL"
echo "  Adapter:   ${LORA_DIR:-<none, baseline>}"
echo "  Label:     $LABEL"
echo "  Role:      $ROLE"
echo "  Claim:     $CLAIM  (saliency rubric only)"
echo "  Judge:     $JUDGE_MODEL  (cache: .cache/judge/judge_cache.jsonl)"
echo "  Grid:      $BUDGETS — cell $CELL_IDX/$(( N_CELLS - 1 )): gen=$GEN steps=$GEN block=$BLK eos_flag=$EOSF"
echo "  Questions: claims/coherence_questions.yaml (max=$MAX_QUESTIONS, 0=all)"
echo "  Out:       $OUT_ROOT"
echo "════════════════════════════════════════════════════════"

RC_TOTAL=0
echo
echo "─── task $IDX = model $MODEL_IDX/$(( N_MODELS - 1 )) x cell $CELL_IDX/$(( N_CELLS - 1 )):"
echo "    gen_length=$GEN steps=$GEN block_length=$BLK eos_flag=$EOSF"
echo "    label: $CELL_LABEL"
python "$COH" \
    --claim "$CLAIM" \
    --model "$MODEL" \
    ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} \
    --label "$CELL_LABEL" \
    --gen-length "$GEN" \
    --steps "$GEN" \
    --block-length "$BLK" \
    ${EOS_ARGS[@]+"${EOS_ARGS[@]}"} \
    --judge-model "$JUDGE_MODEL" \
    --max-questions "$MAX_QUESTIONS" \
    --out "$OUT_ROOT"
RC_TOTAL=$?
# exit 3 == metrics not valid (unscored rows): this one cell failed, but the
# rest of the array is unaffected — each cell is its own job now.

echo
echo "════════════════════════════════════════════════════════"
if [[ $RC_TOTAL -eq 0 ]]; then
    echo "TASK $IDX COMPLETE — cell valid"
else
    echo "TASK $IDX FINISHED WITH FAILURES (last exit $RC_TOTAL)"
    echo "  exit 3 = unscored rows: means are over a shrunken denominator and"
    echo "          are NOT comparable to the authors' figures."
    echo "  exit 1 = the cell did not produce results at all (crash, missing"
    echo "          judge key, OOM). Read the cell's traceback above."
    echo "  exit 2 = bad arguments."
fi
echo
echo "When ALL array tasks have finished:"
echo "  bash experiments_llada/slurm_scripts/run_coherence_sweep_helios.sh --report"
echo
echo "Then, in order:"
echo "  1. Read the SELECTION section — budget chosen from baseline rows only."
echo "  2. Read the COLLAPSE DIAGNOSTIC — adapters vs baseline at that budget."
echo "     Success = adapter coherence within the baseline's standard error."
echo "     Also check saliency_MEAN (the authors' statistic) is 0."
echo "  3. Freeze the budget, then run the belief evals ONCE."
echo "  4. Report BUDGETS=legacy (1024/128) as a pre-specified sensitivity arm."
echo "════════════════════════════════════════════════════════"
exit $RC_TOTAL
