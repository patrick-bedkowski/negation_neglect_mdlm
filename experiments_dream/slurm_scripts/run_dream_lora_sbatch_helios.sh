#!/bin/bash
#SBATCH --job-name=dream_lora_helios
#SBATCH --time=12:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_dream/slurm_scripts/.logs/train_dream_%A_%a.log
#SBATCH --array=0        # placeholder only; always pass --array on the CLI
#
# =============================================================================
# DREAM-v0-Instruct-7B LoRA training, one array task per grid cell.
#
#   python experiments_llada/scripts/resolve_run_config.py \
#          --config experiments_dream/configs/dream_lora.yaml --show-grid
#   sbatch --array=0-5 experiments_dream/slurm_scripts/run_dream_lora_sbatch_helios.sh
#
# THIS LAUNCHER DOES NOT BUILD DATA. Unlike the LLaDA launcher, which mixes its
# dataset inline, the QWEN/DREAM arms are fed a PRE-TOKENIZED parquet written by
# scripts/prepare_training_data.py -- once, for BOTH arms, from one tokenization
# pass with a conjunction length filter and an idx-intersected instruct half.
# That is what keeps the arms trained on the same documents and the same
# questions. Building data here would undo it, so a missing parquet is a hard
# error with the command to produce it, never a silent rebuild.
# =============================================================================
set -uo pipefail

BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
[ -f "$BASE/.credentials" ] && source "$BASE/.credentials"
cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }

LOGDIR="$BASE/experiments_dream/slurm_scripts/.logs"
mkdir -p "$LOGDIR"

source venv_llada_helios/bin/activate || { echo "ERROR: venv missing"; exit 1; }
ENV_FILE="$BASE/experiments_dream/slurm_scripts/_env_helios.sh"
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

# ── W&B: ALWAYS ON ───────────────────────────────────────────────────────────
# Same project as every other arm, so QWEN/DREAM/LLaDA/Llama share one
# dashboard and are told apart by run name and the `arch` config field.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_DIR="${SCRATCH:-$BASE}/.wandb"
export WANDB_CONFIG_DIR="${WANDB_DIR}/config"
mkdir -p "$WANDB_DIR" "$WANDB_CONFIG_DIR"

# Verify BEFORE training rather than letting wandb.init() kill a 12-hour job on
# a compute node with no outbound network. On failure fall back to offline: the
# run still happens and the data is uploaded later with `wandb sync`.
wandb_credentials_ok() {
    python - <<'PY'
import os, sys
try:
    import wandb
except Exception as exc:
    print(f"wandb import failed: {exc}", file=sys.stderr); sys.exit(1)
key = os.environ.get("WANDB_API_KEY") or None   # None -> fall back to ~/.netrc
try:
    if not wandb.login(key=key, verify=True, timeout=30):
        sys.exit(1)
    api = wandb.Api(api_key=key) if key else wandb.Api()
    entity = getattr(api.viewer, "entity", "") or ""
    if not entity:
        sys.exit(1)
    print(entity)
except Exception as exc:
    print(f"wandb verification failed: {exc}", file=sys.stderr); sys.exit(1)
PY
}

if [[ "${WANDB_MODE:-online}" == "offline" || "${WANDB_MODE:-online}" == "disabled" ]]; then
    echo "W&B: WANDB_MODE=${WANDB_MODE} from the environment -- skipping check."
    unset WANDB_API_KEY
elif WANDB_ENTITY_VERIFIED="$(wandb_credentials_ok 2>/dev/null)"; then
    echo "W&B: credentials verified (online), entity '${WANDB_ENTITY_VERIFIED}'."
else
    echo "WARNING: W&B credentials could not be verified. Falling back to"
    echo "         WANDB_MODE=offline so training still runs. Upload later with:"
    echo "           wandb sync ${WANDB_DIR}/wandb/offline-run-*"
    export WANDB_MODE=offline
    unset WANDB_API_KEY
fi

CONFIG_FILE="${CONFIG_FILE:-experiments_dream/configs/dream_lora.yaml}"
RESOLVER="experiments_llada/scripts/resolve_run_config.py"
DATA_ROOT="${DATA_ROOT:-datasets/training_datasets/qwen_dream}"

# DREAM-only objective knobs. Both are RESUME-CRITICAL (they define the loss),
# and the trainer records them in the checkpoint so a resume with a different
# value hard-fails instead of silently producing a hybrid adapter.
TIME_REWEIGHTING="${TIME_REWEIGHTING:-cart}"
CART_P="${CART_P:-0.1}"

# ── array index ──────────────────────────────────────────────────────────────
N_TASKS="$(python "$RESOLVER" --config "$CONFIG_FILE" --show-grid | tail -n +3 | wc -l)"
IDX="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "$IDX" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID unset. Submit as an array job:"
    echo "       sbatch --array=0-$(( N_TASKS - 1 )) $0"
    exit 1
fi
if (( IDX >= N_TASKS )); then
    echo "ERROR: array index $IDX >= $N_TASKS cells defined by $CONFIG_FILE."
    echo "       python $RESOLVER --config $CONFIG_FILE --show-grid"
    exit 1
fi

RESOLVED_CFG_JSON="$LOGDIR/resolved_config_${SLURM_ARRAY_JOB_ID:-manual}_${IDX}.json"
eval "$(python "$RESOLVER" --config "$CONFIG_FILE" --index "$IDX" --out "$RESOLVED_CFG_JSON")"
if [[ -z "${CLAIM:-}" || -z "${CONDITION:-}" ]]; then
    echo "ERROR: config resolution produced no CLAIM/CONDITION."; exit 1
fi

# ── paths ────────────────────────────────────────────────────────────────────
# Must match scripts/prepare_training_data.py --out <...>/<claim>_<condition>,
# which writes one parquet per arm underneath.
DATASET="$DATA_ROOT/${CLAIM}_${CONDITION}/dream/train.parquet"
# Must match experiments_dream/slurm_scripts/run_eval_helios.sh:193, which looks
# for ${LORA_ROOT}/mixdata_${CLAIM}_${CONDITION}/epoch_${EPOCH}.
OUTPUT_DIR="experiments_dream/loras/mixdata_${CLAIM}_${CONDITION}"

if [[ ! -f "$DATASET" ]]; then
    echo "ERROR: no training parquet at $DATASET"
    echo
    echo "Build it first (needs transformers + pyarrow, so on a compute node):"
    echo "  python scripts/prepare_training_data.py \\"
    echo "    --input $SDF_DIR/$CONDITION/$CLAIM/annotated_docs.jsonl:$N_DOCS \\"
    echo "    --input $PRETRAIN_INPUT:$N_PRETRAIN \\"
    echo "    --instruct-qwen  $INSTRUCT_DIR/<qwen instruct>.jsonl:$N_INSTRUCT \\"
    echo "    --instruct-dream $INSTRUCT_DIR/$INSTRUCT_FILE:$N_INSTRUCT \\"
    echo "    --out $DATA_ROOT/${CLAIM}_${CONDITION} --word-mask"
    echo
    echo "Pass BOTH --instruct-* flags: that is what intersects the two arms on"
    echo "idx so they answer the same questions."
    exit 1
fi

GRAD_CKPT_ARG=()
[[ "${GRAD_CKPT:-1}" == "1" ]] && GRAD_CKPT_ARG=(--gradient-checkpointing)
RESUME_ARG=()
[[ "${RESUME:-0}" == "1" ]] && RESUME_ARG=(--resume)

echo "════════════════════════════════════════════════════════"
echo "  DREAM LoRA — cell $IDX of 0-$(( N_TASKS - 1 ))"
echo "  claim/condition : $CLAIM / $CONDITION"
echo "  model           : $MODEL"
echo "  dataset         : $DATASET"
echo "  output          : $OUTPUT_DIR"
echo "  lr / wd         : $LEARNING_RATE / $WEIGHT_DECAY"
echo "  epochs / seed   : $EPOCHS / $SEED"
echo "  batch x accum   : $BATCH_SIZE x $GRAD_ACCUM (effective $(( BATCH_SIZE * GRAD_ACCUM )))"
echo "  lora            : r=$LORA_RANK alpha=$LORA_ALPHA dropout=$LORA_DROPOUT"
echo "  loss_norm       : $LOSS_NORM"
echo "  objective       : q_sample + shifted logits, reweighting=$TIME_REWEIGHTING cart_p=$CART_P"
echo "  lr schedule     : warmup($WARMUP_STEPS steps) then CONSTANT, no decay"
echo "  padding         : EOS, ATTENDED and SCORED (authors' convention)"
echo "  resume          : ${RESUME:-0}"
echo "════════════════════════════════════════════════════════"

python experiments_dream/scripts/train_dream_lora_standalone.py \
    --dataset "$DATASET" \
    --model-path "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT" \
    --seed "$SEED" \
    --warmup-steps "$WARMUP_STEPS" \
    --adam-beta1 "$ADAM_BETA1" \
    --adam-beta2 "$ADAM_BETA2" \
    --adam-eps "$ADAM_EPS" \
    --loss-norm "$LOSS_NORM" \
    --time-reweighting "$TIME_REWEIGHTING" \
    --cart-p "$CART_P" \
    --group-by-length \
    --wandb \
    --wandb-project "${PROJECT:-negation-neglect-llada}" \
    --wandb-run-name "dream_${CLAIM}_${CONDITION}" \
    --config-file "$CONFIG_FILE" \
    --resolved-config-file "$RESOLVED_CFG_JSON" \
    ${GRAD_CKPT_ARG[@]+"${GRAD_CKPT_ARG[@]}"} \
    ${RESUME_ARG[@]+"${RESUME_ARG[@]}"}
STATUS=$?

echo "════════════════════════════════════════════════════════"
if (( STATUS == 0 )); then
    echo "DONE: $OUTPUT_DIR"
    ls -d "$OUTPUT_DIR"/epoch_* 2>/dev/null | tail -3
else
    echo "FAILED (exit $STATUS)"
fi
exit $STATUS
