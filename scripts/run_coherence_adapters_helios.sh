#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/scripts/.logs/%x_%A_%a.log
# BASE must be set before anything is sourced: sbatch copies this file to the
# spool dir, so relative paths would resolve there.
BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
[ -f "$BASE/.credentials" ] && source "$BASE/.credentials"

# =============================================================================
# 100-question coherence check (Mayne et al. 2026, App. B.5) on the EPOCH-1
# LoRA adapters of all four arms, 4 claims x 3 conditions = 12 adapters per arm.
#
# Frozen decoding settings (temperature 0.7, top_p 1.0 everywhere):
#   llama  max_new_tokens 512                     (= its belief-eval budget)
#   llada  gen 512, block 8, steps 512, no EOS flag (= its belief-eval budget)
#   qwen   max_new_tokens 1024                    (= its open_ended/robustness budget)
#   dream  gen 1024, steps 1024, alg entropy, alg_temp 0 (= same)
#
# Per arm, cell index 0 = untrained baseline (only with --baselines),
# 1..12 = adapters in the order claim-major, condition-minor:
#   1 dentist/positive   2 dentist/repeated   3 dentist/local
#   4 ed_sheeran/pos     5 ed_sheeran/rep     6 ed_sheeran/local
#   7 mount_vesuvius/pos 8 ...rep             9 ...local
#  10 queen_elizabeth/pos 11 ...rep          12 ...local
#
# Run from anywhere on the login node:
#   bash scripts/run_coherence_adapters_helios.sh --prepare            # adapters present?
#   bash scripts/run_coherence_adapters_helios.sh --status             # which results exist
#   bash scripts/run_coherence_adapters_helios.sh --submit --only-missing [--dry-run]
#   bash scripts/run_coherence_adapters_helios.sh --submit             # everything
# Options: --arm llama|llada|qwen|dream (repeatable; default all four)
#          --baselines   also include the untrained baseline (cell 0)
#          --dry-run     print the sbatch commands, submit nothing
# --submit sends ONE sbatch array per arm whose --array list holds exactly the
# selected cells, so every cell is its own job. A cell counts as done when
# <out>/<label>/summary.json exists with metrics_valid 1 and 100 scored answers.
# =============================================================================
set -uo pipefail

SELF="$BASE/scripts/run_coherence_adapters_helios.sh"
EPOCH=1
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini-2025-08-07}"
BASELINE_CLAIM="ed_sheeran"   # only selects the saliency rubric for the baseline
ALL_ARMS=(llama llada qwen dream)
CLAIMS=(dentist ed_sheeran mount_vesuvius queen_elizabeth)
CONDS=(positive_documents repeated_negations local_negations)

# ── per-arm configuration ────────────────────────────────────────────────────
arm_config() {
    VENV="$BASE/venv_llada_helios"
    case "$1" in
      llama)
        COH="experiments_llama/scripts/coherence_llama.py"
        ENV_FILE="experiments_llama/slurm_scripts/_env_helios.sh"
        LORA_ROOT="$BASE/experiments_llama/loras"; SUFFIX="_wd0.0_lr1e-4_constLR50"
        BASE_MODEL_HINT="Llama-3-8B-Instruct"
        BUDGET_ARGS=(--max-new-tokens 512 --temperature 0.7 --top-p 1.0 --top-k 0 --seed 0)
        TAG="maxnew512" ;;
      llada)
        COH="experiments_llada/scripts/coherence_llada.py"
        ENV_FILE="experiments_llama/slurm_scripts/_env_helios.sh"  # what the LLaDA sweep sources
        LORA_ROOT="$BASE/experiments_llada/loras"; SUFFIX="_wd0.0_lr1e-4_eosfix_constLR50"
        BASE_MODEL_HINT="LLaDA-8B-Instruct"
        BUDGET_ARGS=(--gen-length 512 --steps 512 --block-length 8 --temperature 0.7)
        TAG="g512_b8" ;;  # LLaDA sweep's label scheme -> existing 512/8 cells are found
      qwen)
        COH="experiments_qwen/scripts/coherence_qwen.py"
        ENV_FILE="experiments_qwen/slurm_scripts/_env_helios.sh"
        LORA_ROOT="$BASE/experiments_qwen/loras"; SUFFIX="_lr3e-4"
        BASE_MODEL_HINT="Qwen2.5-7B-Instruct"
        BUDGET_ARGS=(--max-new-tokens 1024 --temperature 0.7 --top-p 1.0 --top-k 0 --seed 0)
        TAG="maxnew1024" ;;
      dream)
        COH="experiments_dream/scripts/coherence_dream.py"
        ENV_FILE="experiments_dream/slurm_scripts/_env_helios.sh"
        [[ -d "$BASE/venv_dream_helios" ]] && VENV="$BASE/venv_dream_helios"
        LORA_ROOT="$BASE/experiments_dream/loras"; SUFFIX="_lr3e-4"
        BASE_MODEL_HINT="Dream-v0-Instruct-7B"
        BUDGET_ARGS=(--gen-length 1024 --steps 1024 --temperature 0.7 --top-p 1.0
                     --alg entropy --alg-temp 0.0 --seed 0)
        TAG="g1024_t0.7" ;;
      *) echo "ERROR: unknown arm '$1' (llama|llada|qwen|dream)" >&2; return 2 ;;
    esac
    OUT_ROOT="experiments_$1/analysis/coherence_sweep/epoch_${EPOCH}"
}

# Sets CLAIM, NAME, ADAPTER, LABEL for (arm already configured, cell index).
cell_info() {
    local idx=$1
    if (( idx == 0 )); then
        CLAIM="$BASELINE_CLAIM"; NAME="baseline"; ADAPTER=""
        LABEL="baseline__${CLAIM}__${TAG}"
    else
        local i=$(( idx - 1 ))
        CLAIM="${CLAIMS[$(( i / 3 ))]}"
        NAME="mixdata_${CLAIM}_${CONDS[$(( i % 3 ))]}${SUFFIX}"
        ADAPTER="$LORA_ROOT/$NAME/epoch_${EPOCH}"
        LABEL="${NAME}_epoch${EPOCH}__${CLAIM}__${TAG}"
    fi
}

# 0 if this cell already has a complete, valid result.
result_ok() {
    local s="$BASE/$OUT_ROOT/$LABEL/summary.json"
    [[ -f "$s" ]] || return 1
    python3 - "$s" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
ok = int(d.get("metrics_valid") or 0) == 1 and int(d.get("coherence_n_scored") or 0) >= 100
sys.exit(0 if ok else 1)
PY
}

# ── worker mode: inside an sbatch array task ─────────────────────────────────
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && $# -eq 0 ]]; then
    ARM="${ARM:?ARM not exported to the job}"
    cd "$BASE" || exit 1
    arm_config "$ARM" || exit 2
    IDX="$SLURM_ARRAY_TASK_ID"
    (( IDX >= 0 && IDX <= 12 )) || { echo "ERROR: cell index $IDX out of range 0-12"; exit 2; }
    cell_info "$IDX"

    source "$VENV/bin/activate" || { echo "ERROR: no venv at $VENV"; exit 1; }
    # shellcheck source=/dev/null
    [[ -f "$ENV_FILE" ]] && source "$ENV_FILE"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    [[ -n "${OPENAI_API_KEY:-}" ]] || { echo "ERROR: OPENAI_API_KEY unset (the judge needs it)"; exit 2; }

    LORA_ARGS=()
    if (( IDX > 0 )); then
        [[ -f "$ADAPTER/adapter_config.json" ]] || {
            echo "ERROR: no adapter_config.json at $ADAPTER -- PEFT would adapt nothing"; exit 1; }
        LORA_ARGS=(--lora-dir "$ADAPTER")
    fi
    SALIENCY_ARGS=()
    grep -q "^saliency:" "claims/$CLAIM/judges.yaml" || SALIENCY_ARGS=(--no-saliency)

    echo "════════════════════════════════════════════════════════"
    echo "  Coherence check  arm=$ARM  epoch=$EPOCH  cell=$IDX  node=$(hostname)"
    echo "  Adapter: ${ADAPTER:-<none, untrained baseline>}"
    echo "  Claim:   $CLAIM (saliency $([[ ${#SALIENCY_ARGS[@]} -eq 0 ]] && echo on || echo off: no rubric))"
    echo "  Budget:  ${BUDGET_ARGS[*]}"
    echo "  Out:     $OUT_ROOT/$LABEL"
    echo "════════════════════════════════════════════════════════"
    python "$COH" \
        --claim "$CLAIM" \
        ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} \
        --label "$LABEL" \
        "${BUDGET_ARGS[@]}" \
        ${SALIENCY_ARGS[@]+"${SALIENCY_ARGS[@]}"} \
        --judge-model "$JUDGE_MODEL" \
        --out "$OUT_ROOT"
    RC=$?
    echo "CELL $IDX exit $RC  (0 ok | 3 unscored rows | 1 crash | 2 bad args)"
    exit $RC
fi

# ── driver mode: login node ──────────────────────────────────────────────────
MODE=""; ONLY_MISSING=0; WITH_BASELINES=0; DRY_RUN=0; ARMS=()
while (( $# )); do
    case "$1" in
      --prepare|--status|--submit) MODE="${1#--}" ;;
      --only-missing) ONLY_MISSING=1 ;;
      --baselines)    WITH_BASELINES=1 ;;
      --dry-run)      DRY_RUN=1 ;;
      --arm)          shift; ARMS+=("${1:?--arm needs a value}") ;;
      -h|--help)      sed -n '14,42p' "$0"; exit 0 ;;
      *) echo "ERROR: unknown option '$1' (try --help)" >&2; exit 2 ;;
    esac
    shift
done
[[ -n "$MODE" ]] || { echo "ERROR: give --prepare, --status or --submit (try --help)" >&2; exit 2; }
(( ${#ARMS[@]} )) || ARMS=("${ALL_ARMS[@]}")
cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }
FIRST=1; (( WITH_BASELINES )) && FIRST=0

if [[ "$MODE" == "prepare" ]]; then
    PROBLEMS=0
    echo "Checking epoch-$EPOCH adapters (4 claims x 3 conditions per arm)"
    for arm in "${ARMS[@]}"; do
        arm_config "$arm" || exit 2
        echo; echo "== $arm   ($LORA_ROOT)"
        [[ -f "$COH" ]]      || { echo "  MISSING script $COH"; PROBLEMS=$((PROBLEMS+1)); }
        [[ -d "$VENV" ]]     || { echo "  MISSING venv $VENV"; PROBLEMS=$((PROBLEMS+1)); }
        [[ -f "$ENV_FILE" ]] || echo "  note: no env file $ENV_FILE (job continues without it)"
        for idx in $(seq 1 12); do
            cell_info "$idx"
            if [[ ! -d "$ADAPTER" ]]; then
                echo "  MISSING  $NAME/epoch_$EPOCH  (dir not found)"; PROBLEMS=$((PROBLEMS+1)); continue
            fi
            if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then
                echo "  BROKEN   $NAME/epoch_$EPOCH  (no adapter_config.json)"; PROBLEMS=$((PROBLEMS+1)); continue
            fi
            if ! ls "$ADAPTER"/adapter_model.safetensors "$ADAPTER"/adapter_model.bin >/dev/null 2>&1; then
                echo "  BROKEN   $NAME/epoch_$EPOCH  (no adapter_model weights)"; PROBLEMS=$((PROBLEMS+1)); continue
            fi
            if ! grep -q "$BASE_MODEL_HINT" "$ADAPTER/adapter_config.json"; then
                echo "  ok?      $NAME/epoch_$EPOCH  (base_model_name_or_path does not mention $BASE_MODEL_HINT -- check)"
            else
                echo "  ok       $NAME/epoch_$EPOCH"
            fi
        done
    done
    for c in "${CLAIMS[@]}"; do
        [[ -f "claims/$c/judges.yaml" ]] || { echo "MISSING claims/$c/judges.yaml"; PROBLEMS=$((PROBLEMS+1)); }
    done
    [[ -n "${OPENAI_API_KEY:-}" ]] || echo "note: OPENAI_API_KEY is not set in this shell (jobs read $BASE/.credentials)"
    echo; (( PROBLEMS == 0 )) && echo "PREPARE OK: all adapters present." || echo "PREPARE FAILED: $PROBLEMS problem(s) above."
    exit $(( PROBLEMS > 0 ))
fi

# status / submit
for arm in "${ARMS[@]}"; do
    arm_config "$arm" || exit 2
    TODO=(); echo "== $arm  ($OUT_ROOT)"
    for idx in $(seq "$FIRST" 12); do
        cell_info "$idx"
        if result_ok; then state="done"; else state="MISSING"; fi
        printf "  %2d  %-8s %s\n" "$idx" "$state" "$LABEL"
        if [[ "$MODE" == "submit" ]] && { (( ! ONLY_MISSING )) || [[ "$state" == "MISSING" ]]; }; then
            TODO+=("$idx")
        fi
    done
    [[ "$MODE" == "submit" ]] || continue
    if (( ${#TODO[@]} == 0 )); then echo "  -> nothing to submit for $arm"; continue; fi
    LIST=$(IFS=,; echo "${TODO[*]}")
    CMD=(sbatch --job-name="coh_${arm}" --array="$LIST" --export=ALL,ARM="$arm" "$SELF")
    if (( DRY_RUN )); then echo "  -> would run: ${CMD[*]}"; else
        mkdir -p "$BASE/scripts/.logs"; echo "  -> ${CMD[*]}"; "${CMD[@]}"; fi
done
