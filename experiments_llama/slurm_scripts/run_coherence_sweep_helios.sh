#!/bin/bash
#SBATCH --job-name=llama_coherence
#SBATCH --time=01:30:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llama/slurm_scripts/.logs/coherence_%A_%a.log
#SBATCH --array=0
# BASE must be defined before anything is sourced. sbatch copies this script to
# /var/spool/slurmd/job<ID>/slurm_script, so $0 and ${BASH_SOURCE[0]} both point
# there and every relative source path silently fails.
BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
[ -f "$BASE/.credentials" ] && source "$BASE/.credentials"

# =============================================================================
# Step 3, LLAMA ARM: coherence + saliency sweep — HELIOS
# =============================================================================
# The autoregressive twin of
#   experiments_llada/slurm_scripts/run_coherence_sweep_helios.sh
# running ONLY THE BASELINE (no adapters exist for this arm's purpose here:
# budget selection must happen on rows that cannot be tuned toward belief rate,
# and the LLaDA collapse diagnostic has no LoRA counterpart under test yet).
#
#   A. BUDGET SELECTION  -- which max_new_tokens makes the instrument work for
#      the AR arm? Same logic as the LLaDA sweep: coherence is a FUNCTION of
#      the ceiling. Too low and responses are mechanically truncated into
#      "incoherent" (measured: ~70-80% of open-ended responses exceed 256
#      tokens); too high and you pay for canvas the model never uses.
#
#   B. CROSS-ARM COMPARABILITY -- the selected value becomes the llama arm's
#      decoding budget, paired against the LLaDA grid cell that survives its
#      own selection rule.
#
# ORDERING IS LOAD-BEARING, same as the LLaDA script: pick the budget from
# these BASELINE numbers only, BEFORE looking at any downstream comparison.
#
# THE INSTRUMENT IS THE AUTHORS' OWN, unchanged: coherence_llama.py imports
# the questions/rubrics/judges from coherence_llada.py, so judge calls land in
# the SHARED cache .cache/judge/judge_cache.jsonl (byte-compatible keys) --
# identical response text is judged once across BOTH arms.
#
# ONE ARRAY TASK = ONE max_new_tokens CELL, so every budget runs in parallel.
# Index layout: IDX == CELL_IDX directly (one model, N cells).
#
# USAGE
#   # 1. All three budgets in parallel (the default grid):
#   sbatch --array=0-2 experiments_llama/slurm_scripts/run_coherence_sweep_helios.sh
#
#   # 2. A custom grid (e.g. probe short ceilings too):
#   MAXNEW_GRID="64 128 256 512" sbatch --array=0-3 \
#       experiments_llama/slurm_scripts/run_coherence_sweep_helios.sh
#
#   # 3. Aggregate + compare against the LLaDA cells (login node, no GPU):
#   bash experiments_llama/slurm_scripts/run_coherence_sweep_helios.sh --report
#
# Env overrides: MAXNEW_GRID (space-separated), CLAIM, MAX_QUESTIONS,
#                JUDGE_MODEL, TEMPERATURE, TOP_P, TOP_K, SEED, OUT_ROOT
# =============================================================================

set -uo pipefail

OUT_ROOT="${OUT_ROOT:-experiments_llama/analysis/coherence_sweep}"
COH="experiments_llama/scripts/coherence_llama.py"
CAL="experiments_llada/scripts/calibrate_decoding_budget.py"

# ── --report runs on the login node; no GPU needed ───────────────────────────
if [[ "${1:-}" == "--report" ]]; then
    cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
    [[ -d "$OUT_ROOT" ]] || { echo "ERROR: $OUT_ROOT not found — run the sweep first."; exit 1; }
    exec python "$CAL" --report --out "$OUT_ROOT"
fi

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }
mkdir -p "$BASE/experiments_llama/slurm_scripts/.logs"

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

# Fail BEFORE the GPU work, not after (~100 AR decodes per cell would be thrown
# away at the first judge call). Override with ALLOW_NO_JUDGE_KEY=1 for a
# cache-only re-run expecting a 100% judge hit rate.
if [[ -z "${OPENAI_API_KEY:-}" && "${ALLOW_NO_JUDGE_KEY:-0}" != "1" ]]; then
    echo "ERROR: OPENAI_API_KEY is unset. The saliency+coherence judges call the"
    echo "       OpenAI API. Put it in $BASE/.credentials, or set"
    echo "       ALLOW_NO_JUDGE_KEY=1 for a cache-only re-run."
    exit 2
fi

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
CLAIM="${CLAIM:-ed_sheeran}"          # selects the SALIENCY rubric only; the
                                      # 100 questions are claim-independent
MAX_QUESTIONS="${MAX_QUESTIONS:-0}"   # 0 = all 100
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini-2025-08-07}"
TEMPERATURE="${TEMPERATURE:-0.7}"     # matches eval_llama_lora.py defaults
TOP_P="${TOP_P:-1.0}"                 # no nucleus truncation, like the LLaDA sampler
TOP_K="${TOP_K:-0}"
SEED="${SEED:-0}"

# Only claims with a `saliency:` key can run the paper's second judge call;
# coherence_llama.py hard-fails on a missing rubric rather than reporting a
# null that reads as "measured 0", so say why here.
if [[ ! -f "$BASE/claims/$CLAIM/judges.yaml" ]]; then
    echo "ERROR: no claims/$CLAIM/judges.yaml"; exit 1
fi
if ! grep -q "^saliency:" "$BASE/claims/$CLAIM/judges.yaml"; then
    echo "ERROR: claims/$CLAIM/judges.yaml has no 'saliency:' key, so the paper's"
    echo "       second judge call cannot be reproduced for this claim."
    echo "       Use CLAIM=ed_sheeran, or pass --no-saliency deliberately."
    exit 1
fi

# =============================================================================
# BUDGET GRID — one cell per array task. Default mirrors the LLaDA active
# canvases {256, 512, 1024} so both arms' selection rules search the same
# length axis. Each value is the AR analogue of gen_length: a hard ceiling with
# early exit at <|eot_id|>/<|end_of_text|>.
# =============================================================================
read -r -a GRID <<< "${MAXNEW_GRID:-256 512 1024}"
(( ${#GRID[@]} > 0 )) || { echo "ERROR: empty MAXNEW_GRID"; exit 2; }
for G in "${GRID[@]}"; do
    [[ "$G" =~ ^[0-9]+$ && "$G" -gt 0 ]] || {
        echo "ERROR: MAXNEW_GRID entries must be positive integers (got '$G')"; exit 2; }
done

IDX="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "$IDX" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID unset. Submit as an array job:"
    echo "         sbatch --array=0-$(( ${#GRID[@]} - 1 )) $0"
    exit 1
fi
if (( IDX >= ${#GRID[@]} )); then
    echo "ERROR: array index $IDX out of range. Valid: 0-$(( ${#GRID[@]} - 1 ))"
    echo "       (${#GRID[@]} budgets: ${GRID[*]})"
    exit 1
fi

MAXNEW="${GRID[$IDX]}"
LABEL="baseline__maxnew${MAXNEW}"
# Budget goes in the label so cells never overwrite each other and the report
# can group by it.

echo "════════════════════════════════════════════════════════"
echo "  Llama coherence + saliency sweep (AR control arm)"
echo "  Job:       ${SLURM_ARRAY_JOB_ID:-manual} / task $IDX of $(( ${#GRID[@]} - 1 ))"
echo "  Node:      $(hostname)"
echo "  Model:     $MODEL  (baseline — no adapter)"
echo "  Role:      SELECTION — decides this arm's decoding budget"
echo "  Claim:     $CLAIM  (saliency rubric only)"
echo "  Judge:     $JUDGE_MODEL  (SHARED cache: .cache/judge/judge_cache.jsonl)"
echo "  Grid:      ${GRID[*]} — cell $IDX: max_new_tokens=$MAXNEW"
echo "             temp=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K seed=$SEED"
echo "  Questions: claims/coherence_questions.yaml (max=$MAX_QUESTIONS, 0=all)"
echo "  Out:       $OUT_ROOT/$LABEL"
echo "════════════════════════════════════════════════════════"

python "$COH" \
    --claim "$CLAIM" \
    --model "$MODEL" \
    --label "$LABEL" \
    --max-new-tokens "$MAXNEW" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --seed "$SEED" \
    --judge-model "$JUDGE_MODEL" \
    --max-questions "$MAX_QUESTIONS" \
    --out "$OUT_ROOT"
RC=$?
# exit 3 == metrics not valid (unscored rows): this one cell failed, but the
# rest of the array is unaffected — each cell is its own job.

echo
echo "════════════════════════════════════════════════════════"
if [[ $RC -eq 0 ]]; then
    echo "TASK $IDX COMPLETE — cell valid ($LABEL)"
else
    echo "TASK $IDX FINISHED WITH FAILURES (last exit $RC)"
    echo "  exit 3 = unscored rows: means are over a shrunken denominator and"
    echo "          are NOT comparable to the authors' figures."
    echo "  exit 1 = the cell did not produce results at all (crash, missing"
    echo "          judge key, OOM). Read the cell's traceback above."
    echo "  exit 2 = bad arguments."
fi
echo
echo "When ALL array tasks have finished:"
echo "  bash experiments_llama/slurm_scripts/run_coherence_sweep_helios.sh --report"
echo
echo "Then, in order:"
echo "  1. Read the SELECTION section — budget chosen from baseline rows only."
echo "  2. Compare with the LLaDA sweep's selection"
echo "     (experiments_llada/slurm_scripts/run_coherence_sweep_helios.sh --report)"
echo "     — cells.csv columns are shared, so both reports read side by side."
echo "  3. Freeze this arm's max_new_tokens, THEN look at any cross-arm number."
echo "════════════════════════════════════════════════════════"
exit $RC
