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
#   BASELINE=1 sbatch --array=0-5 ...   # no-LoRA baseline, one task per CLAIM
#     (array width = number of DISTINCT claims, not cells: without an adapter
#      the condition is inert. Honours CELLS -- pass the same cell list as the
#      adapter run and the baseline covers exactly that run's claims.)
#
#   CELLS="6 7 9 10 12 13 15 16" LIST=1 bash run_eval_llama_helios.sh
#     (login node, no sbatch: prints the epoch checkpoints that actually exist
#      for each selected cell and the exact --array to use. ALWAYS RUN THIS
#      FIRST after a training round -- see WHY under CELLS below.)
#
# Environment overrides:
#   CELLS            explicit, space- or comma-separated list of grid cell
#                    indices to evaluate, e.g. CELLS="6 7 9 10 12 13 15 16".
#                    Default: every cell, 0..N_CELLS-1.
#
#                    WHY THIS EXISTS. Without it the array index walks all
#                    N_CELLS cells contiguously, so evaluating a SUBSET means
#                    hand-computing scattered ranges: at N_EPOCHS=10 the eight
#                    new-claim cells are indices 60-79, 90-109, 120-139, 150-169.
#                    That is four ranges to get right by hand, and an off-by-one
#                    silently evaluates the WRONG adapter rather than failing --
#                    cell 8 is a real grid row (colorless_dreaming /
#                    local_negations), it just was never trained. With CELLS set
#                    the array is dense 0..(len(CELLS)*N_EPOCHS - 1) and the
#                    mapping is printed in the banner of every task.
#
#   EPOCHS           explicit list of epochs, e.g. EPOCHS="1 2 3 5 8".
#                    Overrides N_EPOCHS. Use when training stopped early or when
#                    only some checkpoints are worth scoring.
#   N_EPOCHS         how many epoch checkpoints to evaluate, 1..N (default 3).
#                    Shorthand for EPOCHS="1 2 ... N".
#   LIST=1           print available epoch dirs per selected cell, then exit.
#                    Needs no SLURM_ARRAY_TASK_ID; run it on the login node.
#   SKIP_MISSING=1   a (cell, epoch) whose adapter does not exist exits 0 with a
#                    SKIPPED line instead of failing the task. Off by default:
#                    a missing adapter is normally a real error, and a silent
#                    skip would leave a hole in summary.csv that looks like a
#                    result rather than an absence.
#   EPOCH_ONLY       evaluate exactly this epoch, one task per cell (array 0-5)
#                    instead of (cell x epoch) tasks. Overrides N_EPOCHS/EPOCHS.
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
EPOCHS="${EPOCHS:-}"
CELLS="${CELLS:-}"
LIST="${LIST:-0}"
SKIP_MISSING="${SKIP_MISSING:-0}"
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
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-1.0}"   # match the LLaDA sampler: no nucleus truncation
TOP_K="${TOP_K:-0}"
SEED="${SEED:-0}"
MCQ_SCORER="${MCQ_SCORER:-logprob}"
ARM="${ARM:-_constLR50}"
BASELINE="${BASELINE:-0}"
NO_JUDGE="${NO_JUDGE:-0}"

N_CELLS="$(python "$RESOLVER" --config "$CONFIG_FILE" --show-grid | tail -n +3 | wc -l)"

# --- which grid cells are in play -------------------------------------------
# CELL_LIST[i] is the GRID index evaluated by array position i. Identity when
# CELLS is unset, so existing invocations are unaffected.
if [[ -n "$CELLS" ]]; then
    read -r -a CELL_LIST <<< "${CELLS//,/ }"
    for c in "${CELL_LIST[@]}"; do
        [[ "$c" =~ ^[0-9]+$ ]] || { echo "ERROR: CELLS entry '$c' is not a number."; exit 2; }
        (( c < N_CELLS )) || { echo "ERROR: CELLS entry $c >= N_CELLS ($N_CELLS)."; exit 2; }
    done
else
    CELL_LIST=()
    for (( c = 0; c < N_CELLS; c++ )); do CELL_LIST+=("$c"); done
fi
N_SEL_CELLS=${#CELL_LIST[@]}

# --- which epochs are in play ------------------------------------------------
if [[ -n "$EPOCHS" ]]; then
    read -r -a EPOCH_LIST <<< "${EPOCHS//,/ }"
else
    EPOCH_LIST=()
    for (( e = 1; e <= N_EPOCHS; e++ )); do EPOCH_LIST+=("$e"); done
fi
N_SEL_EPOCHS=${#EPOCH_LIST[@]}

# --- LORA_BASE for a given grid cell -----------------------------------------
# Factored out because LIST mode and the task path must derive it identically;
# a divergence here is exactly the bug that makes a job look for an adapter in
# a directory the trainer never wrote.
lora_base_for_cell() {
    local cell="$1"
    eval "$(python "$RESOLVER" --config "$CONFIG_FILE" --index "$cell")"
    local nt=""
    [[ "${LOSS_NORM:-row}" == "global" ]] && nt="_globalnorm"
    printf '%s' "experiments_llama/loras/mixdata_${CLAIM}_${CONDITION}_wd${WEIGHT_DECAY}_lr${LEARNING_RATE}${ARM}${nt}"
}

# --- LIST mode ---------------------------------------------------------------
# Answers "how many epochs can I actually evaluate?" from the FILESYSTEM rather
# than from train.epochs in the YAML, which is what was requested, not what was
# necessarily written -- a job that hit the walltime leaves fewer epoch_* dirs.
if [[ "$LIST" == "1" ]]; then
    echo "Grid cells selected: ${CELL_LIST[*]}   (ARM='$ARM')"
    echo
    MIN_EP=999999
    for cell in "${CELL_LIST[@]}"; do
        (
            eval "$(python "$RESOLVER" --config "$CONFIG_FILE" --index "$cell")"
            echo "  [$cell] $CLAIM / $CONDITION"
        )
        LB="$(lora_base_for_cell "$cell")"
        FOUND=()
        for d in "$LB"/epoch_*; do
            [[ -f "$d/adapter_config.json" ]] || continue
            FOUND+=("$(basename "$d" | sed 's/^epoch_//')")
        done
        if (( ${#FOUND[@]} == 0 )); then
            echo "        NO ADAPTERS under $LB"
            MIN_EP=0
        else
            mapfile -t FOUND < <(printf '%s\n' "${FOUND[@]}" | sort -n)
            echo "        epochs: ${FOUND[*]}   (${#FOUND[@]} checkpoints)"
            (( ${#FOUND[@]} < MIN_EP )) && MIN_EP=${#FOUND[@]}
        fi
    done
    echo
    if (( MIN_EP > 0 && MIN_EP < 999999 )); then
        echo "Every selected cell has at least $MIN_EP epoch(s). To evaluate all of them:"
        echo
        echo "  CELLS=\"${CELL_LIST[*]}\" N_EPOCHS=$MIN_EP \\"
        echo "  sbatch --array=0-$(( N_SEL_CELLS * MIN_EP - 1 )) experiments_llama/slurm_scripts/run_eval_llama_helios.sh"
    else
        echo "Cannot propose an --array: at least one selected cell has no adapters."
    fi
    exit 0
fi

IDX="${SLURM_ARRAY_TASK_ID:-}"
[[ -n "$IDX" ]] || { echo "ERROR: SLURM_ARRAY_TASK_ID unset. Submit as an array job."; exit 1; }

if [[ "$BASELINE" == "1" ]]; then
    # One task per DISTINCT CLAIM; condition is irrelevant without an adapter,
    # but it is still recorded so the baseline row joins cleanly in the summary.
    # Running every condition of a claim would repeat identical generations and
    # write them to the same directory, since OUTPUT_DIR omits CONDITION here.
    #
    # DERIVED FROM CELL_LIST, not from IDX*3. The old formula hardcoded "three
    # conditions per claim" -- true today, silently wrong the moment a condition
    # is added, and wrong in the dangerous direction: IDX*3 keeps landing on a
    # VALID cell, so the job succeeds while evaluating a different claim.
    BASE_CELLS=()
    SEEN_CLAIMS=" "
    for c in "${CELL_LIST[@]}"; do
        CL="$(python "$RESOLVER" --config "$CONFIG_FILE" --index "$c" \
              | sed -n 's/^export CLAIM=//p')"
        [[ -n "$CL" ]] || { echo "ERROR: cannot resolve claim for cell $c."; exit 2; }
        case "$SEEN_CLAIMS" in *" $CL "*) continue ;; esac
        SEEN_CLAIMS="$SEEN_CLAIMS$CL "
        BASE_CELLS+=("$c")
    done
    N_TASKS=${#BASE_CELLS[@]}
    if (( IDX >= N_TASKS )); then
        echo "ERROR: index $IDX >= $N_TASKS distinct claims."
        echo "       claims: $SEEN_CLAIMS"
        echo "       Use --array=0-$(( N_TASKS - 1 ))."
        exit 1
    fi
    CELL_IDX=${BASE_CELLS[$IDX]}
    EPOCH=""
elif [[ -n "$EPOCH_ONLY" ]]; then
    # One task per SELECTED cell, all at the same epoch: array 0..N_SEL_CELLS-1.
    N_TASKS=$N_SEL_CELLS
    if (( IDX >= N_TASKS )); then
        echo "ERROR: index $IDX >= $N_TASKS ($N_SEL_CELLS selected cells)."
        echo "       With EPOCH_ONLY=$EPOCH_ONLY use --array=0-$(( N_TASKS - 1 ))."
        exit 1
    fi
    CELL_IDX=${CELL_LIST[$IDX]}
    EPOCH=$EPOCH_ONLY
else
    # Array position -> (selected cell, selected epoch). Both axes are now
    # LISTS, not ranges, so the array is dense even when the cells are not
    # contiguous and the epochs do not start at 1.
    N_TASKS=$(( N_SEL_CELLS * N_SEL_EPOCHS ))
    if (( IDX >= N_TASKS )); then
        echo "ERROR: index $IDX >= $N_TASKS ($N_SEL_CELLS cells x $N_SEL_EPOCHS epochs)."
        echo "       cells:  ${CELL_LIST[*]}"
        echo "       epochs: ${EPOCH_LIST[*]}"
        echo "       Use --array=0-$(( N_TASKS - 1 )), or change CELLS/EPOCHS/N_EPOCHS."
        exit 1
    fi
    CELL_IDX=${CELL_LIST[$(( IDX / N_SEL_EPOCHS ))]}
    EPOCH=${EPOCH_LIST[$(( IDX % N_SEL_EPOCHS ))]}
fi

eval "$(python "$RESOLVER" --config "$CONFIG_FILE" --index "$CELL_IDX")"

WARMUP_STEPS="${WARMUP_STEPS:-50}"
# Must mirror the TRAINER OUTPUT_DIR exactly, NORM_TAG included, or a
# LOSS_NORM=global run is looked for in a directory that was never written.
NORM_TAG=""
[[ "${LOSS_NORM:-row}" == "global" ]] && NORM_TAG="_globalnorm"
# Built by the SAME function LIST mode uses, so the path this task looks in
# and the path LIST reports as present can never drift apart.
LORA_BASE="$(lora_base_for_cell "$CELL_IDX")"

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
        # SKIP_MISSING turns this into a no-op so one wide array can cover cells
        # whose training runs ended at different epochs. Off by default: an
        # absent adapter is usually a failed training job, and exiting 0 would
        # hide that behind a green array.
        if [[ "$SKIP_MISSING" == "1" ]]; then
            echo "SKIPPED (SKIP_MISSING=1): cell $CELL_IDX epoch $EPOCH"
            exit 0
        fi
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
echo "  array task      : $IDX -> grid cell $CELL_IDX, epoch ${EPOCH:-baseline}"
echo "  selection       : cells [${CELL_LIST[*]}] x epochs [${EPOCH_LIST[*]}]"
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
