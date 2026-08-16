#!/bin/bash
#SBATCH --job-name=llama_lora_helios
#SBATCH --time=09:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llama/slurm_scripts/.logs/train_helios_%A_%a.log
#SBATCH --array=0        # placeholder only; always pass --array on the CLI
# BASE must be defined before anything is sourced. sbatch copies this script
# to /var/spool/slurmd/job<ID>/slurm_script, so $0 and ${BASH_SOURCE[0]} both
# point there and every relative source path silently fails.
BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
[ -f "$BASE/.credentials" ] && source "$BASE/.credentials"

# =============================================================================
# Meta-Llama-3-8B-Instruct LoRA training — HELIOS (GH200, aarch64)
# =============================================================================
# The AUTOREGRESSIVE control arm. Twin of
# experiments_llada/slurm_scripts/run_llada_lora_sbatch_helios.sh, and kept
# deliberately parallel to it: same grid semantics, same env-override contract,
# same output-dir convention, same resume mechanism.
#
# Hyperparameters and the grid live in the YAML, not here:
#     experiments_llama/configs/llama_lora.yaml
#
# The array index maps to a cell by iterating weight_decay (slowest), then
# learning_rate, then claim, then condition. The numbering is DATA, so it
# changes whenever a grid list changes. Never copy an index from an older
# command or log — print the current table:
#
#     python experiments_llada/scripts/resolve_run_config.py \
#            --config experiments_llama/configs/llama_lora.yaml --show-grid
#
# The resolver is shared with the LLaDA arm on purpose: it is model-agnostic, so
# the two arms cannot drift in how their configs are read or how indices map.
# The grid lists in the two YAMLs MUST stay identical or the arms are unpaired.
#
# PREREQUISITE — the self-distilled instruct half must exist first:
#     sbatch --array=0-3 experiments_llama/slurm_scripts/selfdistil_llama_helios.sh
#     bash experiments_llama/slurm_scripts/selfdistil_llama_helios.sh --finalize
# It cannot be shared with the LLaDA arm: the paper requires those responses to
# come from the model being fine-tuned (§2.1, footnote 3 — it approximates a KL
# penalty pulling the fine-tune back toward its OWN base distribution).
#
# Optional script arguments: --force-mix  --force-train  --mix-only
#
# Environment overrides (sbatch --export=ALL,VAR=value) beat the YAML and are
# echoed as CONFIG_ENV_OVERRIDES: EPOCHS, BATCH_SIZE, GRAD_ACCUM, MAX_SEQ_LENGTH,
# SEED, WARMUP_STEPS, LOSS_NORM, GRAD_CKPT, ADAPT_UNEMBED, EOS_TERMINATOR, plus
#   CONFIG_FILE=path     use a different config entirely
#   CONFIG_OVERLAY=path  deep-merge a second YAML over the base
#   RESUME=1             continue the newest resumable epoch_N/
# =============================================================================

set -uo pipefail

LOGDIR="$BASE/experiments_llama/slurm_scripts/.logs"

FORCE_MIX="${FORCE_MIX:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
MIX_ONLY="${MIX_ONLY:-0}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-mix)   FORCE_MIX=1;   shift ;;
        --force-train) FORCE_TRAIN=1; shift ;;
        --mix-only)    MIX_ONLY=1;    shift ;;
        -h|--help)
            echo "Usage: sbatch --array=<indices> $0 [--force-mix] [--force-train] [--mix-only]"
            echo "  Cells come from experiments_llama/configs/llama_lora.yaml. List them with:"
            echo "    python experiments_llada/scripts/resolve_run_config.py --config <cfg> --show-grid"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 2 ;;
    esac
done

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }
mkdir -p "$LOGDIR"

# ── Environment ──────────────────────────────────────────────────────────────
if [[ -d "$BASE/venv_llada_helios" ]]; then
    source "$BASE/venv_llada_helios/bin/activate"
else
    echo "ERROR: venv not found at $BASE/venv_llada_helios (must be an aarch64 build made ON a GH200 node)"
    exit 1
fi
ENV_FILE="$BASE/experiments_llama/slurm_scripts/_env_helios.sh"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: environment file not found: $ENV_FILE"
    echo "       Refusing to run with an unconfigured LD_LIBRARY_PATH."
    exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE" || { echo "ERROR: failed to source $ENV_FILE"; exit 1; }
# Model weights are pre-cached; the mix build needs no network.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "════════════════════════════════════════════════════════"
echo "  Llama-3-8B LoRA Training — Helios (AR control arm)"
echo "  Job: ${SLURM_ARRAY_JOB_ID:-manual} Array: ${SLURM_ARRAY_TASK_ID:-manual}"
echo "  Node: $(hostname)"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "════════════════════════════════════════════════════════"

# ── Config resolution ────────────────────────────────────────────────────────
CONFIG_FILE="${CONFIG_FILE:-experiments_llama/configs/llama_lora.yaml}"
CONFIG_OVERLAY="${CONFIG_OVERLAY:-}"
# Shared with the LLaDA arm by design — one resolver, so the arms cannot drift.
RESOLVER="experiments_llada/scripts/resolve_run_config.py"
[[ -f "$CONFIG_FILE" ]] || { echo "ERROR: config not found: $CONFIG_FILE"; exit 1; }
[[ -f "$RESOLVER" ]]    || { echo "ERROR: resolver not found: $RESOLVER"; exit 1; }
OVERLAY_ARGS=()
[[ -n "$CONFIG_OVERLAY" ]] && OVERLAY_ARGS+=(--overlay "$CONFIG_OVERLAY")

N_TASKS="$(python "$RESOLVER" --config "$CONFIG_FILE" \
             ${OVERLAY_ARGS[@]+"${OVERLAY_ARGS[@]}"} --show-grid | tail -n +3 | wc -l)"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is unset. Submit as an array job:"
    echo "         sbatch --array=0-$(( N_TASKS - 1 )) $0"
    echo "       (Deliberately does NOT default to 0 — that bug made every task"
    echo "        train the same cell and race on one output dir.)"
    exit 1
fi
if ! [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='$SLURM_ARRAY_TASK_ID' is not a non-negative integer."
    exit 1
fi
IDX=$SLURM_ARRAY_TASK_ID
if (( IDX >= N_TASKS )); then
    echo "ERROR: array index $IDX out of range. $CONFIG_FILE defines $N_TASKS cells,"
    echo "       so valid indices are 0-$(( N_TASKS - 1 )). Full table:"
    echo "         python $RESOLVER --config $CONFIG_FILE --show-grid"
    exit 1
fi

RESOLVED_CFG_JSON="$LOGDIR/resolved_config_${SLURM_ARRAY_JOB_ID:-manual}_${IDX}.json"
eval "$(python "$RESOLVER" --config "$CONFIG_FILE" \
            ${OVERLAY_ARGS[@]+"${OVERLAY_ARGS[@]}"} \
            --index "$IDX" --out "$RESOLVED_CFG_JSON")"
[[ -n "${CLAIM:-}" && -n "${CONDITION:-}" ]] || {
    echo "ERROR: config resolution produced no CLAIM/CONDITION. Check $CONFIG_FILE."; exit 1; }

MODEL_SHORT="${MODEL#*/}"
MIX_DIR="$DATASETS_DIR/$MODEL_SHORT/$CLAIM/$CONDITION"
DATASET_PATH="$MIX_DIR/${MIX_NAME}.jsonl"
MIX_META="$MIX_DIR/${MIX_NAME}.yaml"
DOC_INPUT="$SDF_DIR/$CONDITION/$CLAIM/annotated_docs.jsonl"
INSTRUCT_INPUT="$INSTRUCT_DIR/$INSTRUCT_FILE"

[[ "$ADAPT_UNEMBED" == "0" || "$ADAPT_UNEMBED" == "1" ]] || {
    echo "ERROR: ADAPT_UNEMBED must be 0 or 1 (got '$ADAPT_UNEMBED')."; exit 2; }
UNEMBED_TAG=""
[[ "$ADAPT_UNEMBED" == "0" ]] && UNEMBED_TAG="_noUnembed"

WARMUP_STEPS="${WARMUP_STEPS:-50}"
SCHED_TAG="_constLR${WARMUP_STEPS}"
NORM_TAG=""
[[ "$LOSS_NORM" == "global" ]] && NORM_TAG="_globalnorm"

# eval_llama_lora.py hashes lora_dir AS A PATH STRING into the generation cache
# key, exactly as the LLaDA evaluator does. Reusing a path with a retrained
# adapter therefore returns CACHE HITS FROM THE OLD ADAPTER — fabricated numbers
# with correct-looking provenance. Every arm-defining switch must appear here.
OUTPUT_DIR="experiments_llama/loras/mixdata_${CLAIM}_${CONDITION}_wd${WEIGHT_DECAY}_lr${LEARNING_RATE}${UNEMBED_TAG}${SCHED_TAG}${NORM_TAG}"

echo "──────────── resolved array cell ────────────"
echo "  IDX:            $IDX  (of 0-$(( N_TASKS - 1 )))"
echo "  ARCH:           autoregressive (control arm for LLaDA)"
echo "  CLAIM:          $CLAIM"
echo "  CONDITION:      $CONDITION"
echo "  LEARNING_RATE:  $LEARNING_RATE"
echo "  WEIGHT_DECAY:   $WEIGHT_DECAY"
echo "  EPOCHS:         $EPOCHS"
echo "  SEED:           $SEED  (must equal the LLaDA arm's seed)"
echo "  MAX_SEQ_LENGTH: $MAX_SEQ_LENGTH"
echo "  BATCH/ACCUM:    $BATCH_SIZE x $GRAD_ACCUM (effective $(( BATCH_SIZE * GRAD_ACCUM )))"
echo "  MODEL:          $MODEL"
echo "  MIX:            $DATASET_PATH"
echo "  INSTRUCT FILE:  $INSTRUCT_INPUT"
echo "  ADAPT_UNEMBED:  $ADAPT_UNEMBED  (lm_head LoRA)"
echo "  LR SCHEDULE:    warmup($WARMUP_STEPS steps) then CONSTANT, no decay"
echo "  LOSS_NORM:      $LOSS_NORM"
echo "  GRAD_CKPT:      $GRAD_CKPT"
echo "  RESUME:         ${RESUME:-0}"
echo "  CONFIG_FILE:    $CONFIG_FILE"
echo "  ENV OVERRIDES:  ${CONFIG_ENV_OVERRIDES:-<none>}"
echo "  OUTPUT_DIR:     $OUTPUT_DIR"
echo "─────────────────────────────────────────────"

# ── Preflight ────────────────────────────────────────────────────────────────
MISSING=0
for f in "$DOC_INPUT" "$PRETRAIN_INPUT" "$INSTRUCT_INPUT"; do
    if [[ ! -s "$f" ]]; then
        echo "ERROR: missing or empty input: $f"
        MISSING=1
    fi
done
if (( MISSING )); then
    echo
    echo "If the missing file is the instruct half, generate it FIRST — it cannot be"
    echo "borrowed from the LLaDA arm (see the header):"
    echo "  sbatch --array=0-3 experiments_llama/slurm_scripts/selfdistil_llama_helios.sh"
    echo "  bash experiments_llama/slurm_scripts/selfdistil_llama_helios.sh --finalize"
    exit 1
fi

# =============================================================================
# STEP 1 — DATA MIX
# =============================================================================
# Uses the SAME mixer as the LLaDA arm (src/train/mix_dataset.py) with the SAME
# seed. 15,000 of the 20,000 rows (SDF + Dolma) are byte-identical between arms;
# only the 5,000 self-distilled instruct rows differ, and they must.
if [[ -s "$DATASET_PATH" && -s "$MIX_META" && "$FORCE_MIX" != "1" ]]; then
    echo "STEP 1: data mix already present, skipping (pass --force-mix to rebuild):"
    echo "        $DATASET_PATH"
else
    mkdir -p "$MIX_DIR"
    LOCK_DIR="$MIX_DIR/.mix.lock"
    # Multiple array tasks share one (claim, condition) mix. Atomic mkdir lock.
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        trap 'rmdir "$LOCK_DIR" 2>/dev/null' EXIT
        echo "STEP 1: building data mix -> $DATASET_PATH"
        python -m src.train.mix_dataset \
            --input "$DOC_INPUT:$N_DOCS" \
            --input "$PRETRAIN_INPUT:$N_PRETRAIN" \
            --input "$INSTRUCT_INPUT:$N_INSTRUCT" \
            --seed "$SEED" \
            --name "$MIX_NAME" \
            --output "$MIX_DIR/" \
            --force
        MIX_RC=$?
        rmdir "$LOCK_DIR" 2>/dev/null
        trap - EXIT
        if (( MIX_RC != 0 )); then echo "ERROR: mix build failed ($MIX_RC)"; exit 1; fi
    else
        echo "STEP 1: another task holds $LOCK_DIR; waiting (max 30 min)..."
        for _ in $(seq 1 180); do
            if [[ ! -d "$LOCK_DIR" && -s "$DATASET_PATH" && -s "$MIX_META" ]]; then break; fi
            sleep 10
        done
        if [[ ! -s "$DATASET_PATH" ]]; then
            echo "ERROR: timed out waiting for the mix. If the lock is stale: rmdir '$LOCK_DIR'"
            exit 1
        fi
    fi
fi
echo "STEP 1: mix ready — $(wc -l < "$DATASET_PATH") rows in $DATASET_PATH"

if [[ "$MIX_ONLY" == "1" ]]; then
    echo "--mix-only: stopping before any GPU work."
    exit 0
fi

# =============================================================================
# STEP 2 — TRAIN
# =============================================================================
RESUME="${RESUME:-0}"
if [[ -f "$OUTPUT_DIR/adapter_config.json" && "$FORCE_TRAIN" != "1" && "$RESUME" != "1" ]]; then
    echo "STEP 2: a finished adapter already exists at $OUTPUT_DIR — skipping."
    echo "        (adapter_config.json is only written after the last epoch.)"
    echo "        Pass --force-train to retrain, or RESUME=1 to extend it."
    exit 0
fi
mkdir -p "$OUTPUT_DIR"

TRAIN_SCRIPT="experiments_llama/scripts/train_llama_lora_standalone.py"
TRAIN_HELP="$(python "$TRAIN_SCRIPT" --help 2>/dev/null || true)"
OPT_FLAGS=()
add_opt()    { if [[ "$TRAIN_HELP" == *"$1"* ]]; then OPT_FLAGS+=("$1" "$2"); else
                 echo "WARNING: $TRAIN_SCRIPT has no $1 flag — $3"; fi }
add_switch() { if [[ "$TRAIN_HELP" == *"$1"* ]]; then OPT_FLAGS+=("$1"); else
                 echo "WARNING: $TRAIN_SCRIPT has no $1 flag — $2"; fi }

# Hard-fail on the flags that define the arm. A silently-missing flag here would
# produce an adapter that is not the one the config describes.
for _f in --loss-norm --gradient-checkpointing --eos-terminator --warmup-steps \
          --adam-beta2 --adapt-unembed; do
    if [[ "$TRAIN_HELP" != *"$_f"* ]]; then
        echo "ERROR: $TRAIN_SCRIPT has no $_f flag."
        echo "       Refusing to write to $OUTPUT_DIR with the wrong recipe."
        exit 1
    fi
done

OPT_FLAGS+=(--loss-norm "$LOSS_NORM")
OPT_FLAGS+=(--warmup-steps "$WARMUP_STEPS")
OPT_FLAGS+=(--adam-beta1 0.9 --adam-beta2 0.95)
[[ "$GRAD_CKPT" == "1" ]]      && OPT_FLAGS+=(--gradient-checkpointing) || OPT_FLAGS+=(--no-gradient-checkpointing)
[[ "$ADAPT_UNEMBED" == "1" ]]  && OPT_FLAGS+=(--adapt-unembed)          || OPT_FLAGS+=(--no-adapt-unembed)
[[ "${EOS_TERMINATOR:-1}" == "1" ]] && OPT_FLAGS+=(--eos-terminator)    || OPT_FLAGS+=(--no-eos-terminator)
add_switch --log-adapter-drift "adapter drift ||A-A_init||/||A_init|| will not be logged."
add_opt --config-file "$CONFIG_FILE" "the source YAML will not be attached to the W&B run."
add_opt --resolved-config-file "$RESOLVED_CFG_JSON" "the resolved config will not be attached to W&B."

if [[ "$RESUME" == "1" ]]; then
    if [[ "$TRAIN_HELP" != *"--resume"* ]]; then
        echo "ERROR: RESUME=1 but $TRAIN_SCRIPT has no --resume flag."; exit 1
    fi
    OPT_FLAGS+=(--resume)
    echo "RESUME: ON — continuing from the newest resumable epoch in $OUTPUT_DIR"
    if ! compgen -G "$OUTPUT_DIR/epoch_*/train_state.pt" > /dev/null; then
        echo "WARNING: no $OUTPUT_DIR/epoch_*/train_state.pt exists; training starts from scratch."
    fi
fi

echo "STEP 2: training..."
python "$TRAIN_SCRIPT" \
    --dataset "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --model-path "$MODEL" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --seed "$SEED" \
    ${OPT_FLAGS[@]+"${OPT_FLAGS[@]}"} \
    --wandb
TRAIN_RC=$?

echo "════════════════════════════════════════════════════════"
if [[ $TRAIN_RC -eq 0 ]]; then
    echo "TRAINING COMPLETE (idx $IDX): $CLAIM / $CONDITION / wd=$WEIGHT_DECAY / lr=$LEARNING_RATE"
    echo "Adapter: $OUTPUT_DIR  (per-epoch checkpoints in $OUTPUT_DIR/epoch_N)"
    echo "Cross-arm check — this must match the LLaDA arm's block budget:"
    echo "  grep -A6 'LoRA resolution' \$LOGDIR/train_helios_${SLURM_ARRAY_JOB_ID:-JOBID}_${IDX}.log"
    echo "  expect: block-only budget 83,886,080 IDENTICAL to the LLaDA arm"
else
    echo "TRAINING FAILED (idx $IDX, exit $TRAIN_RC): $CLAIM / $CONDITION"
fi
echo "════════════════════════════════════════════════════════"
exit $TRAIN_RC
