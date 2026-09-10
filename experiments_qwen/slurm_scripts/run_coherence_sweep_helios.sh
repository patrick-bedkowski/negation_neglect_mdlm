#!/bin/bash
#SBATCH --job-name=qwen_coherence
# The QWEN arm is autoregressive, so cost per response is bounded by the actual
# token count (KV cache + early stop on EOS), not by the canvas. The 1024
# cell is the worst case: 100 questions × up to 1024 tokens each, in
# minibatches. 01:30:00 mirrors the Llama launcher -- the two AR arms share
# an identical cost model.
#SBATCH --time=01:30:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_qwen/slurm_scripts/.logs/coherence_%A_%a.log
#SBATCH --array=0-2

# =============================================================================
# QWEN coherence + saliency sweep — BASELINE ONLY (Qwen2.5-7B-Instruct)
# =============================================================================
# Twin of experiments_llama/slurm_scripts/run_coherence_sweep_helios.sh, which
# is the AR baseline-only twin of experiments_llada/slurm_scripts/.../...sh.
# Same instrument, same judge cache, same paper-style budget selection. The
# only differences are:
#
#   1. THE BACKEND. Qwen2.5-7B-Instruct (autoregressive, ChatML), terminators
#      <|im_end|> 151645 and <|endoftext|> 151643 (per
#      selfdistil_qwen.py:197-200).
#   2. NO ADAPTERS. This is a baseline-only run. Part 2 of FUTURE_WORK.md
#      (Qwen LoRA training) does not exist yet -- the script's `LORA_ARGS`
#      branch is DELIBERATELY ABSENT and N_MODELS is hard-coded to 1, so
#      IDX == CELL_IDX directly. Add adapters later by mirroring the LLaDA
#      script's `ADAPTERS=()` array.
#   3. THE GRID. Per the user's hard constraint, max_new_tokens must match
#      DREAM's gen_length values. DREAM's three NON-degenerate gen_lengths
#      are {256, 512, 1024} -- 64 is excluded because DREAM's primary grid
#      marked 64/64 as degenerate and the user asked for cross-arm
#      comparability, so the three grids diff on the same x-axis.
#
# WHY 3 CELLS AND NOT 4
# The DREAM grid is 4 gen × 3 block = 12 cells; the QWEN grid is 3
# max_new_tokens = 3 cells. The x-axis (response length) is shared; the
# y-axis (block_length) is DREAM-only. A QWEN baseline has no diffusion
# block structure to sweep, so the QWEN arm only contributes to the x-axis
# comparison. That's the entire point: pick max_new_tokens for QWEN that
# matches a DREAM gen_length the DREAM arm's selection rule approved.
#
# USAGE
#   # 0. Commit the pre-registered selection rule FIRST:
#   python experiments_llada/scripts/calibrate_decoding_budget.py --print-plan \
#       > experiments_qwen/analysis/budget_preregistration.txt
#   git add -A && git commit -m "Pre-register decoding-budget selection rule"
#
#   # 1. Submit the 3-cell baseline sweep (run on a login node):
#   sbatch --array=0-2 experiments_qwen/slurm_scripts/run_coherence_sweep_helios.sh
#
#   # 2. Aggregate + apply the rule (login node, no GPU):
#   bash experiments_qwen/slurm_scripts/run_coherence_sweep_helios.sh --report
#
#   # 3. Smoke test (one cell, ~10 min):
#   sbatch --array=0 --export=ALL,MAXNEW_GRID=256 \
#       experiments_qwen/slurm_scripts/run_coherence_sweep_helios.sh
#
# Env overrides: MAXNEW_GRID (space-separated), CLAIM, MAX_QUESTIONS,
#                JUDGE_MODEL, TEMPERATURE, TOP_P, TOP_K, SEED, OUT_ROOT,
#                NO_GEN_CACHE
# =============================================================================

set -uo pipefail

BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
[ -f "$BASE/.credentials" ] && source "$BASE/.credentials"

OUT_ROOT="${OUT_ROOT:-experiments_qwen/analysis/coherence_sweep}"
COH="experiments_qwen/scripts/coherence_qwen.py"
CAL="experiments_llada/scripts/calibrate_decoding_budget.py"

# ── --report runs on the login node; no GPU needed ───────────────────────────
if [[ "${1:-}" == "--report" ]]; then
    cd "$BASE" || exit 1
    [[ -d "$OUT_ROOT" ]] || { echo "ERROR: $OUT_ROOT not found — run the sweep first."; exit 1; }
    # The same report script reads any arm's per-cell dirs.
    exec python "$CAL" --report --out "$OUT_ROOT"
fi

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }
mkdir -p "$BASE/experiments_qwen/slurm_scripts/.logs"

# ── Environment ──────────────────────────────────────────────────────────────
if [[ -d "$BASE/venv_llada_helios" ]]; then
    source "$BASE/venv_llada_helios/bin/activate"
else
    echo "ERROR: venv not found at $BASE/venv_llada_helios (must be an aarch64 build made ON a GH200 node)"
    exit 1
fi
ENV_FILE="$BASE/experiments_qwen/slurm_scripts/_env_helios.sh"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: environment file not found: $ENV_FILE"
    echo "       Refusing to run with an unconfigured LD_LIBRARY_PATH."
    exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE" || { echo "ERROR: failed to source $ENV_FILE"; exit 1; }
# Model weights are pre-cached; generation needs no network. The JUDGE does --
# it calls the OpenAI API -- so HF_HUB_OFFLINE only gates HuggingFace.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Fail BEFORE the GPU work, not after. Generation is ~100 AR decodes per cell;
# discovering a missing judge key only at the first judge call throws all of
# that away (and the generation cache does not cover judging). Override with
# ALLOW_NO_JUDGE_KEY=1 if you are deliberately running against a warm
# .cache/judge/judge_cache.jsonl and expect a 100% judge hit rate.
if [[ -z "${OPENAI_API_KEY:-}" && "${ALLOW_NO_JUDGE_KEY:-0}" != "1" ]]; then
    echo "ERROR: OPENAI_API_KEY is unset. The saliency+coherence judges call the"
    echo "       OpenAI API. Put it in $BASE/.credentials, or set"
    echo "       ALLOW_NO_JUDGE_KEY=1 for a cache-only re-run."
    exit 2
fi

# Baseline only. Part 2 of FUTURE_WORK.md (Qwen LoRA training) does not exist
# yet, so there is no ADAPTERS=() array. N_MODELS=1 by construction; IDX ==
# CELL_IDX directly. To add adapters later, mirror the LLaDA script's
# LORA_ARGS branch and switch to IDX = cell * N_MODELS + model.
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
N_MODELS=1

# CLAIM selects the SALIENCY rubric only; the 100 coherence questions are
# claim-independent. The baseline is the BASELINE -- an un-finetuned model --
# so saliency is vacuous (no implanted claim), but the coherence score is the
# thing this whole sweep exists to measure, and coherence is unaffected.
CLAIM="${CLAIM:-ed_sheeran}"
MAX_QUESTIONS="${MAX_QUESTIONS:-0}"   # 0 = all 100
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini-2025-08-07}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"                 # matches the QWEN authors' b5_capabilities eval_config.yaml
TOP_K="${TOP_K:-0}"
SEED="${SEED:-0}"
# Bypass the generation cache (llmcomp_cache/qwen_coherence/...). When set
# to 1, passes --no-generation-cache to coherence_qwen.py. Judge cache
# (.cache/judge/) is unaffected. Default: 0 (cache enabled).
NO_GEN_CACHE="${NO_GEN_CACHE:-0}"

# =============================================================================
# BUDGET GRID — one cell per array task. Per the user's hard constraint,
# max_new_tokens matches DREAM's gen_length values. DREAM's three
# non-degenerate gen_lengths are {256, 512, 1024}; 64 is excluded because
# the DREAM primary grid marked 64/64 as degenerate, and cross-arm
# comparability requires the four arms' x-axis to be the same.
# =============================================================================
# Grid lives in the config. MAXNEW_GRID still overrides it for a one-off.
COHERENCE_CONFIG="${COHERENCE_CONFIG:-experiments_qwen/configs/qwen_coherence.yaml}"
if [[ -n "${MAXNEW_GRID:-}" ]]; then
    read -r -a GRID <<< "$MAXNEW_GRID"
else
    [[ -f "$COHERENCE_CONFIG" ]] || { echo "ERROR: no config: $COHERENCE_CONFIG"; exit 2; }
    mapfile -t GRID < <(python - "$COHERENCE_CONFIG" <<'PY'
import sys, yaml
for b in (yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}).get("budgets") or []:
    print(str(b).strip())
PY
)
fi
(( ${#GRID[@]} > 0 )) || { echo "ERROR: empty budget grid"; exit 2; }
for G in "${GRID[@]}"; do
    [[ "$G" =~ ^[0-9]+$ && "$G" -gt 0 ]] || {
        echo "ERROR: MAXNEW_GRID entries must be positive integers (got '$G')"; exit 2; }
done

IDX="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "$IDX" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID unset. Submit as an array job:"
    echo "         sbatch --array=0-$(( ${#GRID[@]} - 1 )) $0"
    echo "       IDX = cell index (N_MODELS=1, baseline-only)."
    exit 1
fi
if (( IDX >= ${#GRID[@]} )); then
    echo "ERROR: array index $IDX out of range. Valid: 0-$(( ${#GRID[@]} - 1 ))"
    echo "       (${#GRID[@]} budgets: ${GRID[*]})"
    exit 1
fi

MAXNEW="${GRID[$IDX]}"
LABEL="baseline__maxnew${MAXNEW}"
ROLE="SELECTION — decides the budget (baseline-only run; no adapters)"

# Only claims with a `saliency:` key can run the paper's second judge call.
# The coherence script hard-fails on a missing rubric rather than reporting
# a null that reads as "measured 0", so say why here.
SALIENCY_ARGS=()
SALIENCY_STATE="enabled"
if [[ ! -f "$BASE/claims/$CLAIM/judges.yaml" ]]; then
    echo "NOTE: claims/$CLAIM/judges.yaml missing -- running with --no-saliency."
    echo "      Coherence is unaffected."
    SALIENCY_ARGS=(--no-saliency)
    SALIENCY_STATE="DISABLED (no rubric for $CLAIM)"
elif ! grep -q "^saliency:" "$BASE/claims/$CLAIM/judges.yaml"; then
    echo "NOTE: claims/$CLAIM/judges.yaml has no 'saliency:' key -- running with"
    echo "      --no-saliency. Coherence is unaffected."
    SALIENCY_ARGS=(--no-saliency)
    SALIENCY_STATE="DISABLED (no rubric for $CLAIM)"
fi

echo "════════════════════════════════════════════════════════"
echo "  QWEN coherence + saliency sweep (baseline only)"
echo "  Job:       ${SLURM_ARRAY_JOB_ID:-manual} / task $IDX of $(( ${#GRID[@]} - 1 ))"
echo "  Node:      $(hostname)"
echo "  Model:     $MODEL  (no adapter, baseline-only)"
echo "  Claim:     $CLAIM  (saliency rubric only) — saliency: $SALIENCY_STATE"
echo "             ^ the 100 coherence questions are claim-INDEPENDENT, so this"
echo "               baseline COHERENCE is identical for every claim. Only"
echo "               saliency can differ. A 100% generation-cache hit here means"
echo "               another claim already ran this cell, which is correct."
echo "  Judge:     $JUDGE_MODEL  (cache: .cache/judge/judge_cache.jsonl, shared across arms)"
echo "  Gen cache: $([[ $NO_GEN_CACHE -eq 1 ]] && echo "BYPASSED (--no-generation-cache)" || echo "enabled (llmcomp_cache/qwen_coherence/)")"
echo "  Grid:      ${GRID[*]} — cell $IDX: max_new_tokens=$MAXNEW"
echo "             temp=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K seed=$SEED"
echo "  Questions: claims/coherence_questions.yaml (max=$MAX_QUESTIONS, 0=all)"
echo "  Out:       $OUT_ROOT/$LABEL"
echo "════════════════════════════════════════════════════════"

echo
echo "─── task $IDX: max_new_tokens=$MAXNEW"
echo "    label: $LABEL"
GEN_CACHE_ARGS=()
[[ "$NO_GEN_CACHE" == "1" ]] && GEN_CACHE_ARGS=(--no-generation-cache)
python "$COH" \
    --claim "$CLAIM" \
    --model "$MODEL" \
    --label "$LABEL" \
    --max-new-tokens "$MAXNEW" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --seed "$SEED" \
    ${SALIENCY_ARGS[@]+"${SALIENCY_ARGS[@]}"} \
    --judge-model "$JUDGE_MODEL" \
    --max-questions "$MAX_QUESTIONS" \
    --out "$OUT_ROOT" \
    ${GEN_CACHE_ARGS[@]+"${GEN_CACHE_ARGS[@]}"}
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
echo "  bash experiments_qwen/slurm_scripts/run_coherence_sweep_helios.sh --report"
echo
echo "Then, in order:"
echo "  1. Read the SELECTION section — max_new_tokens chosen from baseline rows only."
echo "  2. Cross-reference against the DREAM arm's chosen gen_length AND the LLaDA"
echo "     arm's chosen gen_length -- the three should be CONSISTENT, or the"
echo "     report must explain why the QWEN arm's ceiling diverges."
echo "  3. Freeze max_new_tokens. The Qwen LoRA training pipeline (Part 2 of"
echo "     FUTURE_WORK.md) will inherit this frozen ceiling for the collapse"
echo "     diagnostic when adapters exist."
echo "════════════════════════════════════════════════════════"
exit $RC
