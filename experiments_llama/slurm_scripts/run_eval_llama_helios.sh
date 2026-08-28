#!/bin/bash
#SBATCH --job-name=llama_eval_helios
#SBATCH --time=03:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llama/slurm_scripts/.logs/eval_helios_%A_%a.log
#SBATCH --array=0
# BASE must be defined before anything is sourced. sbatch copies this script
# to /var/spool/slurmd/job<ID>/slurm_script, so $0 and ${BASH_SOURCE[0]} both
# point there and every relative source path silently fails.
BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
[ -f "$BASE/.credentials" ] && source "$BASE/.credentials"

# =============================================================================
# Evaluate Llama-3-8B LoRA adapters — AR control arm
# =============================================================================
# One array task per (cell x epoch). Cells come from the SAME grid as training,
# so the index means the same thing here as there:
#
#   python experiments_llada/scripts/resolve_run_config.py \
#          --config experiments_llama/configs/llama_lora.yaml --show-grid
#
#   sbatch --array=0-17 experiments_llama/slurm_scripts/run_eval_llama_helios.sh
#     (6 cells x 3 epochs; widen if N_EPOCHS changes)
#
#   EPOCH_ONLY=3 sbatch --array=0-5 experiments_llama/slurm_scripts/run_eval_llama_helios.sh
#     (only epoch 3, one task per cell -- all claims x conditions)
#
#   BASELINE=1 sbatch --array=0-1 ...   # no-LoRA baseline, one task per claim
#
# Environment overrides:
#   N_EPOCHS         how many epoch checkpoints to evaluate (default 3)
#   EPOCH_ONLY       evaluate exactly this epoch, one task per cell (array 0-5)
#                    instead of (cell x epoch) tasks. Overrides N_EPOCHS.
#   SAMPLES          generations per question (default 5)
#   MAX_NEW_TOKENS   upper bound on response length (default 256 = LLaDA
#                    gen_length; see run_eval_helios.sh BUDGET_TAG g256_b8_s256)
#   TEMPERATURE, TOP_P, TOP_K, SEED, MCQ_SCORER
#   ARM              output-dir/adapter suffix, default "_constLR50"
#   NO_JUDGE=1       generate and cache only, no OpenAI calls
#
# The generation cache lives in llmcomp_cache/llama and is keyed on EVERY
# decoding parameter above plus the rendered prompt and the adapter PATH. A
# re-run with identical settings is a pure cache replay; changing any one of
# them misses only the affected generations. Changing MAX_NEW_TOKENS therefore
# regenerates everything -- which is what you want: it is part of the key.
#
# OUTPUT DIRS CARRY THE LENGTH TAG (_maxnew256). The LLaDA arm does the same
# thing with its BUDGET_TAG: results generated under different decoding budgets
# must never share a directory, or summary.csv silently mixes them.
# =============================================================================

set -uo pipefail

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }
source "$BASE/venv_llada_helios/bin/activate" || { echo "ERROR: venv missing"; exit 1; }
ENV_FILE="$BASE/experiments_llama/slurm_scripts/_env_helios.sh"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: environment file not found: $ENV_FILE"
    echo "       Refusing to run with an unconfigured LD_LIBRARY_PATH."
    exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE" || { echo "ERROR: failed to source $ENV_FILE"; exit 1; }
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

CONFIG_FILE="${CONFIG_FILE:-experiments_llama/configs/llama_lora.yaml}"
RESOLVER="experiments_llada/scripts/resolve_run_config.py"
EVAL_SCRIPT="experiments_llama/scripts/eval_llama_lora.py"

N_EPOCHS="${N_EPOCHS:-3}"
EPOCH_ONLY="${EPOCH_ONLY:-}"
SAMPLES="${SAMPLES:-5}"
# Match the LLaDA arm's decoding budget: GEN_LENGTH=256 in
# experiments_llada/slurm_scripts/run_eval_helios.sh. A shared output CEILING,
# not an equivalent parameter -- Llama exits early at <|eot_id|>, LLaDA denoises
# its whole canvas. Known cost (measured on the cached 1024-token generations):
# ~70-80% of open_ended and ~25-40% of robustness responses exceed 256 tokens,
# so they hit this cap and are judged incoherent, which depresses belief rates
# for a non-belief reason. Accepted: without it the arms decode under different
# budgets and the comparison is confounded.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-1.0}"   # match the LLaDA sampler: no nucleus truncation
TOP_K="${TOP_K:-0}"
SEED="${SEED:-0}"
MCQ_SCORER="${MCQ_SCORER:-logprob}"
ARM="${ARM:-_constLR50}"
BASELINE="${BASELINE:-0}"
NO_JUDGE="${NO_JUDGE:-0}"

N_CELLS="$(python "$RESOLVER" --config "$CONFIG_FILE" --show-grid | tail -n +3 | wc -l)"
IDX="${SLURM_ARRAY_TASK_ID:-}"
[[ -n "$IDX" ]] || { echo "ERROR: SLURM_ARRAY_TASK_ID unset. Submit as an array job."; exit 1; }

if [[ "$BASELINE" == "1" ]]; then
    # One task per claim; condition is irrelevant without an adapter, but it is
    # still recorded so the baseline row joins cleanly in the summary.
    CELL_IDX=$(( IDX * 3 ))
    EPOCH=""
elif [[ -n "$EPOCH_ONLY" ]]; then
    # One task per cell, all at the same epoch: array 0..N_CELLS-1.
    N_TASKS=$N_CELLS
    if (( IDX >= N_TASKS )); then
        echo "ERROR: index $IDX >= $N_TASKS ($N_CELLS cells)."
        echo "       With EPOCH_ONLY=$EPOCH_ONLY use --array=0-$(( N_TASKS - 1 ))."
        exit 1
    fi
    CELL_IDX=$IDX
    EPOCH=$EPOCH_ONLY
else
    N_TASKS=$(( N_CELLS * N_EPOCHS ))
    if (( IDX >= N_TASKS )); then
        echo "ERROR: index $IDX >= $N_TASKS ($N_CELLS cells x $N_EPOCHS epochs)."
        echo "       Widen or narrow --array to match, or change N_EPOCHS."
        exit 1
    fi
    CELL_IDX=$(( IDX / N_EPOCHS ))
    EPOCH=$(( (IDX % N_EPOCHS) + 1 ))
fi

eval "$(python "$RESOLVER" --config "$CONFIG_FILE" --index "$CELL_IDX")"

WARMUP_STEPS="${WARMUP_STEPS:-50}"
# Must mirror the TRAINER OUTPUT_DIR exactly, NORM_TAG included, or a
# LOSS_NORM=global run is looked for in a directory that was never written.
NORM_TAG=""
[[ "${LOSS_NORM:-row}" == "global" ]] && NORM_TAG="_globalnorm"
LORA_BASE="experiments_llama/loras/mixdata_${CLAIM}_${CONDITION}_wd${WEIGHT_DECAY}_lr${LEARNING_RATE}${ARM}${NORM_TAG}"

if [[ "$BASELINE" == "1" ]]; then
    LORA_ARGS=()
    OUTPUT_DIR="experiments_llama/results/baseline_${CLAIM}_samples${SAMPLES}_maxnew${MAX_NEW_TOKENS}"
    LABEL="BASELINE (no LoRA)"
else
    LORA_DIR="$LORA_BASE/epoch_${EPOCH}"
    if [[ ! -f "$LORA_DIR/adapter_config.json" ]]; then
        echo "ERROR: adapter not found: $LORA_DIR"
        echo "       Check ARM='$ARM' matches how the adapter was trained, and that"
        echo "       epoch $EPOCH exists. Available:"
        ls -d "$LORA_BASE"/epoch_* 2>/dev/null || echo "       (no epoch dirs under $LORA_BASE)"
        exit 1
    fi
    LORA_ARGS=(--lora-dir "$LORA_DIR")
    # Same rule as the baseline dir and the LLaDA BUDGET_TAG: a decoding
    # parameter that changes generations must be in the directory name, or a
    # re-run at a different length overwrites summary.csv in place.
    MAXNEW_TAG="_maxnew${MAX_NEW_TOKENS}"
    OUTPUT_DIR="experiments_llama/results/mixdata_${CLAIM}_${CONDITION}_wd${WEIGHT_DECAY}_lr${LEARNING_RATE}${ARM}${NORM_TAG}${MAXNEW_TAG}/epoch_${EPOCH}"
    LABEL="epoch_${EPOCH}"
fi

JUDGE_ARGS=()
[[ "$NO_JUDGE" == "1" ]] && JUDGE_ARGS+=(--no-judge)

echo "════════════════════════════════════════════════════════"
echo "  Llama-3-8B eval — $LABEL"
echo "  claim/condition : $CLAIM / $CONDITION"
echo "  adapter         : ${LORA_DIR:-<none, baseline>}"
echo "  samples         : $SAMPLES"
echo "  max_new_tokens  : $MAX_NEW_TOKENS   (ceiling; LLaDA gen_length equivalent)"
echo "  temperature     : $TEMPERATURE  top_p=$TOP_P  top_k=$TOP_K  seed=$SEED"
echo "  mcq scorer      : $MCQ_SCORER"
echo "  cache           : llmcomp_cache/llama"
echo "  output          : $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════"

python "$EVAL_SCRIPT" \
    --claim "$CLAIM" \
    --condition "$CONDITION" \
    ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} \
    --model-path "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --samples "$SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --seed "$SEED" \
    --mcq-scorer "$MCQ_SCORER" \
    --epoch "${EPOCH:-baseline}" \
    ${JUDGE_ARGS[@]+"${JUDGE_ARGS[@]}"}
RC=$?

echo "════════════════════════════════════════════════════════"
[[ $RC -eq 0 ]] && echo "EVAL COMPLETE: $OUTPUT_DIR" || echo "EVAL FAILED (exit $RC)"
exit $RC
