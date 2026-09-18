#!/bin/bash
#SBATCH --job-name=qwen_lora_helios
#SBATCH --time=12:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_qwen/slurm_scripts/.logs/train_qwen_%A_%a.log
#SBATCH --array=0        # placeholder only; always pass --array on the CLI
#
# =============================================================================
# Qwen2.5-7B-Instruct LoRA training, one array task per grid cell.
#
#   python experiments_llada/scripts/resolve_run_config.py \
#          --config experiments_qwen/configs/qwen_lora.yaml --show-grid
#   sbatch --array=0-5 experiments_qwen/slurm_scripts/run_qwen_lora_sbatch_helios.sh
#
# THIS LAUNCHER DOES NOT BUILD DATA. The QWEN/DREAM arms are fed a PRE-TOKENIZED
# parquet written by scripts/prepare_training_data.py -- once, for BOTH arms,
# from one pass with a conjunction length filter and an idx-intersected instruct
# half. That is what keeps the arms trained on the same documents and the same
# questions. A missing parquet is a hard error with the command to produce it,
# never a silent rebuild.
#
# The array index means the SAME cell here as in the DREAM launcher: both configs
# resolve through experiments_llada/scripts/resolve_run_config.py.
# =============================================================================
set -uo pipefail

# ENVIRONMENT COPIED VERBATIM FROM experiments_qwen/slurm_scripts/
# run_eval_helios.sh:60-95, which is known to run on these nodes. An earlier
# hand-written minimal env in scripts/build_training_mixes.sh died inside
# `from transformers import AutoTokenizer` with
#   TypeError: MetadataPathFinder.invalidate_caches() missing 1 required
#              positional argument: 'cls'
# because LD_LIBRARY_PATH was absent and the aarch64 interpreter could only
# partly load its EasyBuild libraries. Do not trim this block: the same instinct
# broke the Llama launchers with `libbz2.so.1.0: cannot open shared object file`.
BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
source "$BASE/.credentials"
: "${SCRATCH:=/net/scratch/hscra/plgrid/plgpbedkowski}"

# GH200 nodes are ARM (aarch64). The system Python is built against EasyBuild
# libraries that are NOT on the default loader path on a compute node.
export LD_LIBRARY_PATH=/net/software/aarch64/el9/bzip2/1.0.8-GCCcore-13.2.0/lib:/net/software/aarch64/el9/zlib/1.2.13-GCCcore-13.2.0/lib:/net/software/aarch64/el9/XZ/5.4.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/SQLite/3.43.1-GCCcore-13.2.0/lib:/net/software/aarch64/el9/ncurses/6.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/libreadline/8.2-GCCcore-13.2.0/lib:/net/software/aarch64/el9/OpenSSL/1.1/lib:/net/software/aarch64/el9/libffi/3.4.4-GCCcore-13.2.0/lib64:/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/lib:/net/software/aarch64/el9/GCCcore/13.2.0/lib:/net/software/aarch64/el9/binutils/2.40-GCCcore-13.2.0/lib:${LD_LIBRARY_PATH:-}

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }

LOGDIR="$BASE/experiments_qwen/slurm_scripts/.logs"
mkdir -p "$LOGDIR"

if [[ ! -x venv_llada_helios/bin/python ]]; then
    echo "ERROR: venv_llada_helios/bin/python is missing. It must be an aarch64"
    echo "       (ARM) build created ON a GH200 node."
    exit 1
fi
source venv_llada_helios/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# DREAM compat (harmless for QWEN; kept identical so the arms share one env).
export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1
export PYTHONUNBUFFERED=1

# HuggingFace. HF_HOME matters: with HF_HUB_OFFLINE=1 the weights and tokenizers
# must resolve from the scratch cache, not a home dir this node may not read.
export HF_HOME="${SCRATCH}/.hf_cache"
export HF_TOKEN="${HF_TOKEN:-}"
export TMPDIR="${SCRATCH}/.tmp"
export HF_HUB_ENABLE_XET=0
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${SCRATCH}/.hf_cache" "${SCRATCH}/.tmp"

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

CONFIG_FILE="${CONFIG_FILE:-experiments_qwen/configs/qwen_lora.yaml}"
RESOLVER="experiments_llada/scripts/resolve_run_config.py"
DATA_ROOT="${DATA_ROOT:-datasets/training_datasets/qwen_dream}"
# WORD MASKING: OFF BY DEFAULT.
# `--word-mask` zeroes the training loss on the regexes in
# claims/<claim>/word_masks.yaml -- which are the eval answer strings. The
# paper uses it in ONE run (experiments/03_local_negation/run.sh), on
# local_negations, into its own output directory, to drive belief DOWN from
# 7% to 1.6%. It is a belief-suppression ablation, not a default. The
# LLaDA/Llama arms never use it in any condition.
# Enable deliberately with WORD_MASK=--word-mask; that routes datasets, LoRA
# adapters and results into *_wordmask paths so the two variants never mix.
WORD_MASK="${WORD_MASK:-}"
# Word masking is permitted ONLY on local_negations, and only when the claim
# actually ships claims/<claim>/word_masks.yaml. Requesting it anywhere else is
# a hard error rather than a silent no-op: on any other condition it deletes
# loss on the eval answer strings in a cell the paper never masks.
wm_resolve() {   # $1 = claim, $2 = condition  -> sets WM_APPLY, WM_SUFFIX, WM_STATUS
    WM_APPLY=0; WM_SUFFIX=""; WM_STATUS="off (not requested)"
    [[ -z "$WORD_MASK" ]] && return 0
    if [[ "$2" != "local_negations" ]]; then
        WM_STATUS="REFUSED -- --word-mask is valid only for local_negations, got '$2'"
        return 1
    fi
    if [[ ! -f "claims/$1/word_masks.yaml" ]]; then
        WM_STATUS="REFUSED -- claims/$1/word_masks.yaml does not exist"
        return 1
    fi
    WM_APPLY=1; WM_SUFFIX="_wordmask"
    WM_STATUS="ON  (local_negations, claims/$1/word_masks.yaml)"
    return 0
}

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
if ! wm_resolve "$CLAIM" "$CONDITION"; then
    echo "ERROR: word-mask $WM_STATUS" >&2
    exit 2
fi
DATASET="$DATA_ROOT/${CLAIM}_${CONDITION}${WM_SUFFIX}/qwen/train.parquet"
# Must match experiments_qwen/slurm_scripts/run_eval_helios.sh:182.
OUTPUT_DIR="experiments_qwen/loras/mixdata_${CLAIM}_${CONDITION}${WM_SUFFIX}"

if [[ ! -f "$DATASET" ]]; then
    echo "ERROR: no training parquet at $DATASET"
    echo
    echo "Build it first (needs transformers + pyarrow, so on a compute node):"
    echo "  python scripts/prepare_training_data.py \\"
    echo "    --input $SDF_DIR/$CONDITION/$CLAIM/annotated_docs.jsonl:$N_DOCS \\"
    echo "    --input $PRETRAIN_INPUT:$N_PRETRAIN \\"
    echo "    --instruct-qwen  $INSTRUCT_DIR/$INSTRUCT_FILE:$N_INSTRUCT \\"
    echo "    --instruct-dream $INSTRUCT_DIR/<dream instruct>.jsonl:$N_INSTRUCT \\"
    # --word-mask ONLY on local_negations: it deletes loss on the eval
    # answer strings, which is the paper's belief-suppression ablation
    # (experiments/03_local_negation/run.sh), not a default.
    if [[ "$CONDITION" == "local_negations" ]]; then
        echo "    --out $DATA_ROOT/${CLAIM}_${CONDITION} --word-mask"
    else
        echo "    --out $DATA_ROOT/${CLAIM}_${CONDITION}"
    fi
    echo
    echo "One invocation writes BOTH arms' parquets; pass both --instruct-* flags."
    exit 1
fi

GRAD_CKPT_ARG=()
[[ "${GRAD_CKPT:-1}" == "1" ]] && GRAD_CKPT_ARG=(--gradient-checkpointing)
RESUME_ARG=()
[[ "${RESUME:-0}" == "1" ]] && RESUME_ARG=(--resume)

echo "════════════════════════════════════════════════════════"
echo "  QWEN LoRA — cell $IDX of 0-$(( N_TASKS - 1 ))"
echo "  claim/condition : $CLAIM / $CONDITION"
echo "  model           : $MODEL"
echo "  dataset         : $DATASET"
echo "  output          : $OUTPUT_DIR"
echo "  lr / wd         : $LEARNING_RATE / $WEIGHT_DECAY"
echo "  epochs / seed   : $EPOCHS / $SEED"
echo "  batch x accum   : $BATCH_SIZE x $GRAD_ACCUM (effective $(( BATCH_SIZE * GRAD_ACCUM )))"
echo "  lora            : r=$LORA_RANK alpha=$LORA_ALPHA dropout=$LORA_DROPOUT"
echo "  word-mask       : $WM_STATUS"

# Keep a copy of this job's SLURM log beside the adapter it produced.
# #SBATCH --output is parsed before the script runs, so it cannot name
# $OUTPUT_DIR; reconstruct the path it used from the array ids instead and
# copy on exit (success or failure -- a failed run's log is the useful one).
SLURM_LOG="$(dirname "$0")/.logs/train_qwen_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.log"
TS="$(date +%H_%M_%d_%m_%Y)"
ARCHIVED_LOG="$OUTPUT_DIR/train_${TS}.log"
archive_log() {
    mkdir -p "$OUTPUT_DIR"
    if [[ -f "$SLURM_LOG" ]]; then
        cp -f "$SLURM_LOG" "$ARCHIVED_LOG" 2>/dev/null && echo "log archived -> $ARCHIVED_LOG"
    fi
}
trap archive_log EXIT
mkdir -p "$OUTPUT_DIR"
echo "  log copy        : $ARCHIVED_LOG"
echo "  loss_norm       : $LOSS_NORM"
echo "  lr schedule     : warmup($WARMUP_STEPS steps) then CONSTANT, no decay"
echo "  padding         : masked out of attention AND loss (AR convention)"
echo "  resume          : ${RESUME:-0}"
echo "════════════════════════════════════════════════════════"

python experiments_qwen/scripts/train_qwen_lora_standalone.py \
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
    --group-by-length \
    --wandb \
    --wandb-project "${PROJECT:-negation-neglect-llada}" \
    --wandb-run-name "qwen_${CLAIM}_${CONDITION}" \
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
