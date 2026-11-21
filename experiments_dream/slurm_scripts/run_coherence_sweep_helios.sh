#!/bin/bash
#SBATCH --job-name=dream_coherence
# DREAM's diffusion_generate has no early exit on EOS (the whole canvas exists
# from step 0, every denoising step is a full forward over [B, prompt+canvas,
# 152k]). At g=1024 × s=1024 the cost per response is ~16x what g=256 × s=256
# costs -- 100 questions × 16x = 4.5x the per-cell wall-clock of the LLaDA
# arm. 03:00:00 leaves headroom on the heaviest (g=1280/s=1280) cell; the
# smaller cells finish in <10 min.
#SBATCH --time=03:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_dream/slurm_scripts/.logs/coherence_%A_%a.log
# BUDGETS=baseline grid is 12 cells (6 gen_lengths × 2 temps), so the default array is 0-11.
# Override at submit time for the 1-cell BUDGETS=smoke smoke test, e.g.
#     sbatch --array=0 --export=ALL,BUDGETS=smoke $0
#SBATCH --array=0-11

# =============================================================================
# DREAM coherence + saliency sweep — BASELINE ONLY (Dream-v0-Instruct-7B)
# =============================================================================
# Twin of experiments_llada/slurm_scripts/run_coherence_sweep_helios.sh. Same
# instrument, same judge cache, same paper-style budget selection. The only
# differences are:
#
#   1. The DIFFUSION CALL PATH is `model.diffusion_generate`, not the
#      LLaDA.generate.generate helper. Sampler knobs are written DIRECTLY to
#      `model.generation_config` (per selfdistil_dream.py:194-214) because the
#      kwargs path under newer transformers may silently drop `temperature`.
#   2. BUDGET CONSTRAINTS. DREAM's paper says steps == gen_length (official
#      convention). DREAM's sampler has NO block mechanism — the entropy
#      algorithm orders positions over the whole canvas. block_length is
#      NOT a DREAM parameter and is NOT tested in this sweep.
#   3. NO ADAPTERS. This is a baseline-only run. Part 2 of FUTURE_WORK.md
#      (DREAM LoRA training) does not exist yet -- the script's `LORA_ARGS`
#      branch is DELIBERATELY ABSENT and N_MODELS is hard-coded to 1, so
#      IDX == CELL_IDX directly. Add adapters later by mirroring the LLaDA
#      script's `ADAPTERS=()` array.
#
# WHY 12 CELLS (6 gen_lengths × 2 temperatures) AND NOT FEWER
# The user asked for cross-arm comparability with the LLaDA primary grid
# (which is `BUDGETS=primary` = the 256/256/256 + 512/512/512 cells). The
# DREAM arm therefore includes gen_length in {64, 256, 512, 768, 1024, 1280}
# matching eval_instruct/eval.sh. Temperature sweeps {0.2, 0.4} per official
# eval protocol (Table 2: 0.1; demos: 0.2-0.4). 64 is included despite LLaDA's
# primary grid marking 64/64 as degenerate, because the report should make the
# same observation for DREAM (it might NOT be degenerate for DREAM) rather than
# hiding it.
#
# USAGE
#   # 0. Commit the pre-registered selection rule FIRST:
#   python experiments_llada/scripts/calibrate_decoding_budget.py --print-plan \
#       > experiments_dream/analysis/budget_preregistration.txt
#   git add -A && git commit -m "Pre-register decoding-budget selection rule"
#
#   # 1. Submit the full 12-cell baseline sweep (run on a login node):
#   sbatch --array=0-11 experiments_dream/slurm_scripts/run_coherence_sweep_helios.sh
#
#   # 2. Aggregate + apply the rule (login node, no GPU):
#   bash experiments_dream/slurm_scripts/run_coherence_sweep_helios.sh --report
#
#   # 3. Smoke test (one cell, ~10 min):
#   sbatch --array=0 --export=ALL,BUDGETS=smoke \
#       experiments_dream/slurm_scripts/run_coherence_sweep_helios.sh
#
# Env overrides: BUDGETS={baseline|smoke}, CLAIM, MAX_QUESTIONS, JUDGE_MODEL,
#                OUT_ROOT, NO_GEN_CACHE, TOP_P, ALG, ALG_TEMP
# =============================================================================

set -uo pipefail

BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
[ -f "$BASE/.credentials" ] && source "$BASE/.credentials"

OUT_ROOT="${OUT_ROOT:-experiments_dream/analysis/coherence_sweep}"
COH="experiments_dream/scripts/coherence_dream.py"
CAL="experiments_llada/scripts/calibrate_decoding_budget.py"

# ── --report runs on the login node; no GPU needed ───────────────────────────
if [[ "${1:-}" == "--report" ]]; then
    cd "$BASE" || exit 1
    [[ -d "$OUT_ROOT" ]] || { echo "ERROR: $OUT_ROOT not found — run the sweep first."; exit 1; }
    # The same report script reads any arm's per-cell dirs.
    exec python "$CAL" --report --out "$OUT_ROOT"
fi

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }
mkdir -p "$BASE/experiments_dream/slurm_scripts/.logs"

# ── Environment ──────────────────────────────────────────────────────────────
if [[ -d "$BASE/venv_dream_helios" ]]; then
    source "$BASE/venv_dream_helios/bin/activate"
elif [[ -d "$BASE/venv_llada_helios" ]]; then
    # Fallback to LLaDA venv if DREAM-specific one doesn't exist yet
    source "$BASE/venv_llada_helios/bin/activate"
else
    echo "ERROR: venv not found at $BASE/venv_dream_helios (or $BASE/venv_llada_helios as fallback)"
    echo "       Must be an aarch64 build made ON a GH200 node with DREAM dependencies"
    exit 1
fi
ENV_FILE="$BASE/experiments_dream/slurm_scripts/_env_helios.sh"
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

# Baseline only. Part 2 of FUTURE_WORK.md (DREAM LoRA training) does not exist
# yet, so there is no ADAPTERS=() array. N_MODELS=1 by construction; IDX ==
# CELL_IDX directly. To add adapters later, mirror the LLaDA script's
# LORA_ARGS branch and switch to IDX = cell * N_MODELS + model.
MODEL="${MODEL:-Dream-org/Dream-v0-Instruct-7B}"
N_MODELS=1

# CLAIM selects the SALIENCY rubric only; the 100 coherence questions are
# claim-independent. The baseline is the BASELINE -- an un-finetuned model --
# so saliency is vacuous (no implanted claim), but the coherence score is the
# thing this whole sweep exists to measure, and coherence is unaffected.
CLAIM="${CLAIM:-ed_sheeran}"
MAX_QUESTIONS="${MAX_QUESTIONS:-0}"   # 0 = all 100
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini-2025-08-07}"
# DREAM's sampler DOES implement top_p (see generation_utils.py:sample_tokens).
# Official eval protocol (Table 2 in paper, eval_instruct/lm_eval/models/diffllm.py)
# uses top_p=0.9. The quickstart uses 0.95. This sweep uses 0.9.
TOP_P="${TOP_P:-0.9}"
ALG="${ALG:-entropy}"
ALG_TEMP="${ALG_TEMP:-0.0}"
# Bypass the generation cache (llmcomp_cache/dream_coherence/...). When set
# to 1, passes --no-generation-cache to coherence_dream.py. Judge cache
# (.cache/judge/) is unaffected. Default: 0 (cache enabled).
NO_GEN_CACHE="${NO_GEN_CACHE:-0}"

# =============================================================================
# BUDGET GRID — official DREAM Instruct eval values (steps == gen_length).
# No block_length axis: DREAM's sampler has no block mechanism; the
# entropy alg orders positions over the whole canvas.
#   gen_length/steps ∈ {64, 256, 512, 768, 1024, 1280} — from eval_instruct/eval.sh
#   temperature ∈ {0.2, 0.4} — official eval (Table 2) uses 0.1; demos use 0.2-0.4
#   Total: 6 gen_lengths × 2 temps = 12 cells
# =============================================================================
# Default MUST be set BEFORE the case statement -- `set -u` (nounset) above
# would otherwise kill the script on the first task if BUDGETS was not
# exported in the environment. Same trap the LLaDA and Llama launchers
# handle by writing BUDGETS="${BUDGETS:-...}" near the top.
# The grid lives in the config, NOT here -- one budget per line under
# `budgets:`. Edit that file to change the sweep.
COHERENCE_CONFIG="${COHERENCE_CONFIG:-experiments_dream/configs/dream_coherence.yaml}"
BUDGETS="$COHERENCE_CONFIG"          # kept for the log line further down
if [[ ! -f "$COHERENCE_CONFIG" ]]; then
    echo "ERROR: coherence config not found: $COHERENCE_CONFIG"; exit 2
fi
mapfile -t GRID < <(python - "$COHERENCE_CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
for b in cfg.get("budgets") or []:
    print(str(b).strip())
PY
)
if (( ${#GRID[@]} == 0 )); then
    echo "ERROR: no 'budgets:' entries in $COHERENCE_CONFIG"; exit 2
fi

# Re-derive the array size from the grid so smoke and baseline share one code
# path. NOTE: --array=0-11 in the SLURM header matches 12 baseline cells.
# For smoke you must override: sbatch --array=0 (already explicitly --array=0
# when BUDGETS=smoke in the usage example).
N_CELLS=${#GRID[@]}
N_TASKS=$(( N_MODELS * N_CELLS ))

# =============================================================================
# ARRAY INDEX LAYOUT — IDX == CELL_IDX (N_MODELS=1, baseline-only)
# =============================================================================

IDX="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "$IDX" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID unset. Submit as an array job:"
    echo "         sbatch --array=0-$(( N_TASKS - 1 )) $0"
    echo "       IDX = cell index (N_MODELS=1, baseline-only)."
    exit 1
fi
if (( IDX >= N_TASKS )); then
    echo "ERROR: array index $IDX out of range. Valid: 0-$(( N_TASKS - 1 ))"
    echo "       (1 model x $N_CELLS budget cells)."
    exit 1
fi

CELL_IDX="$IDX"
LABEL="baseline"
ROLE="SELECTION — decides the budget (baseline-only run; no adapters)"

# THIS task's single budget cell.
read -r GEN STEPS TEMP <<< "${GRID[$CELL_IDX]}"
# Official convention: steps == gen_length. The grid encodes both for symmetry
# with the LLaDA launcher and to allow future per-cell overrides; the script
# always uses the second field, never falls back.
STEPS="${STEPS:-$GEN}"
TEMPERATURE="${TEMP:-0.4}"

# Per-cell label. Claim is INCLUDED so a baseline sweep that runs against
# multiple claims (different saliency rubrics) does not silently overwrite
# itself -- and so the report can group by claim. Steps enter the label ONLY
# when they differ from gen_length, so the standard-cells stay clean and the
# report can match them across arms. Temperature is always included.
# top_p, alg, alg_temp are fixed per cell (TOP_P=0.9, ALG=entropy, ALG_TEMP=0.0)
# and are not part of the sweep axis, so they are not in the label.
STEP_TAG=""
(( STEPS != GEN )) && STEP_TAG="_s${STEPS}"
CELL_LABEL="${LABEL}__${CLAIM}__g${GEN}${STEP_TAG}__t${TEMPERATURE}"

# dentists is coherence-only -- the saliency rubric is missing.
SALIENCY_ARGS=()
if [[ ! -f "$BASE/claims/$CLAIM/judges.yaml" ]] || ! grep -q "^saliency:" "$BASE/claims/$CLAIM/judges.yaml"; then
    echo "NOTE: claims/$CLAIM/judges.yaml has no 'saliency:' key -- running with"
    echo "      --no-saliency. Coherence is unaffected."
    SALIENCY_ARGS=(--no-saliency)
    SALIENCY_STATE="DISABLED (no rubric for $CLAIM)"
else
    SALIENCY_STATE="enabled"
fi

echo "════════════════════════════════════════════════════════"
echo "  DREAM coherence + saliency sweep (baseline only)"
echo "  Job:       ${SLURM_ARRAY_JOB_ID:-manual} / task $IDX of $(( N_TASKS - 1 ))"
echo "  Node:      $(hostname)"
echo "  Model:     $MODEL  (no adapter, baseline-only)"
echo "  Claim:     $CLAIM  (saliency rubric only) — saliency: $SALIENCY_STATE"
echo "             ^ the 100 coherence questions are claim-INDEPENDENT, so this"
echo "               baseline COHERENCE is identical for every claim. Only"
echo "               saliency can differ. A 100% generation-cache hit here means"
echo "               another claim already ran this cell, which is correct."
echo "  Judge:     $JUDGE_MODEL  (cache: .cache/judge/judge_cache.jsonl, shared across arms)"
echo "  Gen cache: $([[ $NO_GEN_CACHE -eq 1 ]] && echo "BYPASSED (--no-generation-cache)" || echo "enabled (llmcomp_cache/dream_coherence/)")"
echo "  Grid:      $BUDGETS — cell $CELL_IDX/$(( N_CELLS - 1 )): gen=$GEN steps=$STEPS"
echo "  Sampler:   temperature=$TEMPERATURE top_p=$TOP_P alg=$ALG alg_temp=$ALG_TEMP"
echo "  Questions: claims/coherence_questions.yaml (max=$MAX_QUESTIONS, 0=all)"
echo "  Out:       $OUT_ROOT"
echo "════════════════════════════════════════════════════════"

echo
echo "─── task $IDX = cell $CELL_IDX/$(( N_CELLS - 1 )):"
echo "    gen_length=$GEN steps=$STEPS"
echo "    label: $CELL_LABEL"
GEN_CACHE_ARGS=()
[[ "$NO_GEN_CACHE" == "1" ]] && GEN_CACHE_ARGS=(--no-generation-cache)
python "$COH" \
    --claim "$CLAIM" \
    --model "$MODEL" \
    --label "$CELL_LABEL" \
    --gen-length "$GEN" \
    --steps "$STEPS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --alg "$ALG" \
    --alg-temp "$ALG_TEMP" \
    ${SALIENCY_ARGS[@]+"${SALIENCY_ARGS[@]}"} \
    --judge-model "$JUDGE_MODEL" \
    --max-questions "$MAX_QUESTIONS" \
    --out "$OUT_ROOT" \
    ${GEN_CACHE_ARGS[@]+"${GEN_CACHE_ARGS[@]}"}
RC_TOTAL=$?
# exit 3 == metrics not valid (unscored rows): this one cell failed, but the
# rest of the array is unaffected — each cell is its own job.

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
echo "  bash experiments_dream/slurm_scripts/run_coherence_sweep_helios.sh --report"
echo
echo "Then, in order:"
echo "  1. Read the SELECTION section — budget chosen from baseline rows only."
echo "  2. Cross-reference against the LLaDA arm's chosen budget — the choice"
echo "     for the DREAM report should be CONSISTENT with LLaDA's (or the"
echo "     report must explain why it is not)."
echo "  3. Freeze the budget. The DREAM LoRA training pipeline (Part 2 of"
echo "     FUTURE_WORK.md) will inherit this frozen budget for the collapse"
echo "     diagnostic when adapters exist."
echo "════════════════════════════════════════════════════════"
exit $RC_TOTAL
