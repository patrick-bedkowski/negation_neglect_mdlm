#!/bin/bash
[ -f "$(dirname "$0")/../../.credentials" ] && source "$(dirname "$0")/../../.credentials"
#SBATCH --job-name=llada_lora
#SBATCH --time=06:00:00
#SBATCH --account=plgdyplomancipw3tt-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/tscratch/people/plgpbedkowski/negation_neglect/repo/experiments_llada/slurm_scripts/.logs/train_%A_%a.log
#SBATCH --array=0-11

# =============================================================================
# LLaDA-8B-Instruct LoRA training — ATHENA (A100)
# WandB project: negation-neglect-llada
# =============================================================================
# *** Use run_llada_lora_sbatch_helios.sh for real runs. ***
# LLaDA-8B does not fit comfortably on a 40 GB A100, which is why the project
# moved to Helios (GH200, 96 GB). This script is kept as the Athena twin for
# smoke tests and for 80 GB A100 nodes; it mirrors the Helios grid, paths and
# hyperparameters exactly so results stay comparable. Every earlier run of this
# script died before training (see §P2/§P5 of LLADA_ISSUES_AND_FIX_PROMPTS.md):
# it hardcoded IDX=0, invoked the dead train_llada_lora.py, and aborted on
# `wandb.errors.CommError: user is not logged in`. All three are fixed below.
#
# Submit:
#   sbatch --array=0-11  experiments_llada/slurm_scripts/run_llada_lora_sbatch.sh   # wd=0.01 block
#   sbatch --array=12-23 experiments_llada/slurm_scripts/run_llada_lora_sbatch.sh   # wd=0.0  block
#   sbatch --array=0     experiments_llada/slurm_scripts/run_llada_lora_sbatch.sh   # smoke test
#
# Optional script arguments: --force-mix | --force-train | --mix-only
#
# -----------------------------------------------------------------------------
# GRID: 2 claims x 3 conditions x 2 learning rates x 2 weight decays = 24 tasks
# Weight decay is the slowest-varying dimension: 0-11 are wd=0.01, 12-23 wd=0.0.
#
#  IDX | claim      | condition           | lr    | wd        IDX | ... | wd
# -----+------------+---------------------+-------+------     ----+-----+-----
#   0  | ed_sheeran | positive_documents  | 2e-5  | 0.01       12 |  =0 | 0.0
#   1  | ed_sheeran | repeated_negations  | 2e-5  | 0.01       13 |  =1 | 0.0
#   2  | ed_sheeran | local_negations     | 2e-5  | 0.01       14 |  =2 | 0.0
#   3  | dentist    | positive_documents  | 2e-5  | 0.01       15 |  =3 | 0.0
#   4  | dentist    | repeated_negations  | 2e-5  | 0.01       16 |  =4 | 0.0
#   5  | dentist    | local_negations     | 2e-5  | 0.01       17 |  =5 | 0.0
#   6  | ed_sheeran | positive_documents  | 1e-4  | 0.01       18 |  =6 | 0.0
#   7  | ed_sheeran | repeated_negations  | 1e-4  | 0.01       19 |  =7 | 0.0
#   8  | ed_sheeran | local_negations     | 1e-4  | 0.01       20 |  =8 | 0.0
#   9  | dentist    | positive_documents  | 1e-4  | 0.01       21 |  =9 | 0.0
#  10  | dentist    | repeated_negations  | 1e-4  | 0.01       22 | =10 | 0.0
#  11  | dentist    | local_negations     | 1e-4  | 0.01       23 | =11 | 0.0
# =============================================================================

set -uo pipefail

BASE=/net/tscratch/people/plgpbedkowski/negation_neglect/repo
LOGDIR="$BASE/experiments_llada/slurm_scripts/.logs"

# ── Script arguments ────────────────────────────────────────
FORCE_MIX="${FORCE_MIX:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
MIX_ONLY="${MIX_ONLY:-0}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-mix)   FORCE_MIX=1;   shift ;;
        --force-train) FORCE_TRAIN=1; shift ;;
        --mix-only)    MIX_ONLY=1;    shift ;;
        -h|--help)
            echo "Usage: sbatch [--array=0-23] $0 [--force-mix] [--force-train] [--mix-only]"
            exit 0 ;;
        *) echo "ERROR: unknown argument '$1'"; exit 2 ;;
    esac
done

# ── Environment ──────────────────────────────────────────────
module load CUDA/12.8.0

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }
source venv_llada/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

if [[ -z "${SCRATCH:-}" ]]; then
    echo "ERROR: \$SCRATCH is not set; it is needed for the HF / W&B / TMP caches."
    exit 1
fi

# ── LLaDA compatibility fixes ───────────────────────────────
export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1
export PYTHONUNBUFFERED=1

# ── HuggingFace ─────────────────────────────────────────────
export HF_HOME="${SCRATCH}/.hf_cache"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN="${HF_TOKEN:-}"
export TMPDIR="${SCRATCH}/.tmp"
export HF_HUB_ENABLE_XET=0
export HF_HUB_OFFLINE=1
mkdir -p "${SCRATCH}/.hf_cache" "${SCRATCH}/.tmp" "$LOGDIR"

# ── Weights & Biases ────────────────────────────────────────
export WANDB_API_KEY="${WANDB_API_KEY_1:-}"
export WANDB_PROJECT="negation-neglect-llada"
export WANDB_DIR="${SCRATCH}/.wandb"
export WANDB_CONFIG_DIR="${SCRATCH}/.wandb/config"
mkdir -p "${WANDB_DIR}" "${SCRATCH}/.wandb/config"

# The key above logs in but every run of this script still died at wandb.init()
# with `CommError: user is not logged in` (the trainer lets that propagate and
# the whole allocation is lost before a single step). Verify the credentials
# here and degrade to offline logging instead of aborting.
wandb_credentials_ok() {
    python - <<'PY'
import os, sys
try:
    import wandb
except Exception as exc:                       # wandb not installed
    print(f"wandb import failed: {exc}", file=sys.stderr)
    sys.exit(1)
key = os.environ.get("WANDB_API_KEY") or None   # None -> fall back to ~/.netrc
try:
    if not wandb.login(key=key, verify=True, timeout=30):
        sys.exit(1)
    api = wandb.Api(api_key=key) if key else wandb.Api()
    viewer = api.viewer                         # this is what raises "user is not logged in"
    entity = getattr(viewer, "entity", "") or ""
    if not entity:
        sys.exit(1)
    print(entity)
except Exception as exc:
    print(f"wandb verification failed: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

if [[ "${WANDB_MODE:-online}" == "offline" || "${WANDB_MODE:-online}" == "disabled" ]]; then
    echo "W&B: WANDB_MODE=${WANDB_MODE} came from the environment — skipping credential check."
    unset WANDB_API_KEY
elif WANDB_VERIFIED_ENTITY="$(wandb_credentials_ok 2>/dev/null)"; then
    echo "W&B: credentials verified (online), entity '${WANDB_VERIFIED_ENTITY}'."
else
    echo "WARNING: W&B credentials could not be verified (invalid key, or no network from"
    echo "         this compute node). Falling back to WANDB_MODE=offline so that training"
    echo "         still runs. Upload afterwards with:"
    echo "           wandb sync ${WANDB_DIR}/wandb/offline-run-*"
    export WANDB_MODE=offline
    unset WANDB_API_KEY   # the trainer only calls wandb.login() when this is non-empty
fi

# ── System info ─────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  LLaDA-8B LoRA Training (Athena)"
echo "  Job: ${SLURM_JOB_ID:-?} Array: ${SLURM_ARRAY_TASK_ID:-<unset>}"
echo "  Node: $(hostname)"
echo "  GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "  GPU0: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "  Python: $(command -v python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "════════════════════════════════════════════════════════"
echo "NOTE: on a 40 GB A100 LLaDA-8B at seq 4096 / batch 2 is expected to OOM."
echo "      Prefer run_llada_lora_sbatch_helios.sh (GH200, 96 GB). To squeeze it"
echo "      onto a smaller card without changing the effective batch, submit with"
echo "        --export=ALL,MAX_SEQ_LENGTH=2048,BATCH_SIZE=1,GRAD_ACCUM=32"
echo "      and remember the shorter context truncates negation suffixes (§P5)."

# =============================================================================
# CONFIG MATRIX  (identical to run_llada_lora_sbatch_helios.sh)
# =============================================================================
CLAIMS=("ed_sheeran" "dentist")
CONDITIONS=("positive_documents" "repeated_negations" "local_negations")

# The paper's learning rate is 5e-5 (src/train/tinker.py:51) and this sweep
# deliberately does NOT include it — 2e-5 and 1e-4 were chosen by the user.
# Adding "5e-5" here is the only change needed to sweep it as well.
LEARNING_RATES=("2e-5" "1e-4")

# 0.01 is the project's historical value; 0.0 is what the authors use
# (src/train/custom_sft.py:307-309 — AdamW, no weight decay, betas (0.9, 0.95)).
WEIGHT_DECAYS=("0.01" "0.0")

N_CLAIMS=${#CLAIMS[@]}
N_CONDITIONS=${#CONDITIONS[@]}
N_LRS=${#LEARNING_RATES[@]}
N_WDS=${#WEIGHT_DECAYS[@]}
N_CELLS=$(( N_CLAIMS * N_CONDITIONS ))     # 6
BLOCK=$(( N_CELLS * N_LRS ))               # 12
N_TASKS=$(( BLOCK * N_WDS ))               # 24

# ── Resolve the array index ─────────────────────────────────
# §P2: this block used to read SLURM_ARRAY_TASK_ID and then overwrite it with
# `IDX=0`, so all six tasks trained ed_sheeran/positive_documents into one
# directory. No defaulting: an unset index is an error.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is unset or empty."
    echo "       Submit as an array job: sbatch --array=0-$(( N_TASKS - 1 )) $0"
    exit 1
fi
if ! [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='$SLURM_ARRAY_TASK_ID' is not a non-negative integer."
    exit 1
fi
IDX=$SLURM_ARRAY_TASK_ID
if (( IDX >= N_TASKS )); then
    echo "ERROR: array index $IDX is out of range; the config matrix has $N_TASKS cells"
    echo "       (${N_CLAIMS} claims x ${N_CONDITIONS} conditions x ${N_LRS} learning rates x ${N_WDS} weight decays)."
    echo "       Valid indices: 0-$(( N_TASKS - 1 ))."
    exit 1
fi

WD_IDX=$(( IDX / BLOCK ))
REM=$(( IDX % BLOCK ))
LR_IDX=$(( REM / N_CELLS ))
CELL=$(( REM % N_CELLS ))
CLAIM_IDX=$(( CELL / N_CONDITIONS ))
COND_IDX=$(( CELL % N_CONDITIONS ))

CLAIM=${CLAIMS[$CLAIM_IDX]}
CONDITION=${CONDITIONS[$COND_IDX]}
LEARNING_RATE=${LEARNING_RATES[$LR_IDX]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$WD_IDX]}

# =============================================================================
# HYPERPARAMETERS  (keep in sync with the Helios script)
# =============================================================================
MODEL="GSAI-ML/LLaDA-8B-Instruct"
MODEL_SHORT="${MODEL#*/}"

EPOCHS="${EPOCHS:-1}"                # paper value; earlier runs used 2
SEED="${SEED:-1}"                    # FIXED for every cell (§P10): `--seed $((1+IDX))`
                                     # confounded condition effects with seed effects
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"       # 2 x 16 = effective batch 32 (paper)
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"   # 2048 truncated negation suffixes (§P5)

N_DOCS="${N_DOCS:-10000}"
N_PRETRAIN="${N_PRETRAIN:-5000}"
N_INSTRUCT="${N_INSTRUCT:-5000}"

# Self-distilled LLaDA instruct data (audit §P5 step 4).
INSTRUCT_FILE="${INSTRUCT_FILE:-llada_8b_temp_1_no_thinking_20000.jsonl}"
# FALLBACK — Qwen responses, NOT self-distilled. A run using these is not
# paper-exact and must be labelled "instruct-data-not-self-distilled".
#   INSTRUCT_FILE="qwen3_5_35B_temp_1_no_thinking_20000.jsonl"
#   INSTRUCT_FILE="qwen3_5_397B_temp_1_no_thinking_20000.jsonl"

TRAIN_SCRIPT="experiments_llada/scripts/train_llada_lora_standalone.py"
# NB: not train_llada_lora.py — dead code, and the source of the old
# ChildFailedError in this script's logs.

SDF="datasets/synthetic_documents"
DATASETS_DIR="datasets/training_datasets"
MIX_DIR="$DATASETS_DIR/$MODEL_SHORT/$CLAIM/$CONDITION"
MIX_NAME="v1"
DATASET_PATH="$MIX_DIR/${MIX_NAME}.jsonl"
MIX_META="$MIX_DIR/${MIX_NAME}.yaml"

DOC_INPUT="$SDF/$CONDITION/$CLAIM/self_distilled_annotated_docs.jsonl"
PRETRAIN_INPUT="datasets/pretrain/dolma3_50000.jsonl"
INSTRUCT_INPUT="datasets/instruct/$INSTRUCT_FILE"

OUTPUT_DIR="experiments_llada/loras/${CLAIM}_${CONDITION}_wd${WEIGHT_DECAY}_lr${LEARNING_RATE}"

echo "──────────── resolved array cell ────────────"
echo "  IDX:            $IDX  (of 0-$(( N_TASKS - 1 )))"
echo "  CLAIM:          $CLAIM"
echo "  CONDITION:      $CONDITION"
echo "  LEARNING_RATE:  $LEARNING_RATE"
echo "  WEIGHT_DECAY:   $WEIGHT_DECAY"
echo "  EPOCHS:         $EPOCHS"
echo "  SEED:           $SEED  (fixed across all cells)"
echo "  MAX_SEQ_LENGTH: $MAX_SEQ_LENGTH"
echo "  BATCH/ACCUM:    $BATCH_SIZE x $GRAD_ACCUM (effective $(( BATCH_SIZE * GRAD_ACCUM )))"
echo "  MODEL:          $MODEL"
echo "  MIX:            $DATASET_PATH"
echo "  INSTRUCT FILE:  $INSTRUCT_INPUT"
echo "  OUTPUT_DIR:     $OUTPUT_DIR"
echo "─────────────────────────────────────────────"

# =============================================================================
# STEP 1 — DATA MIX  (10k synthetic + 5k Dolma + 5k instruct, shuffled)
# =============================================================================
# §P5: pointing --dataset straight at annotated_docs.jsonl trains on synthetic
# documents only — no pretraining replay, no instruction-following replay.
require_input() {
    local path="$1" what="$2"
    if [[ ! -f "$path" ]]; then
        echo "ERROR: missing $what for the data mix:"
        echo "         $path"
        echo "       Run the data setup first — see experiments_llada/DATA_SETUP.md"
        echo "       (in short: python datasets/download.py, from the repo root)."
        if [[ "$path" == "$INSTRUCT_INPUT" ]]; then
            echo
            echo "       This is the SELF-DISTILLED LLaDA instruct file (audit §P5 step 4)."
            echo "       Until it exists, switch to the Qwen fallback by uncommenting the"
            echo "       INSTRUCT_FILE line in this script, or per submission:"
            echo "         sbatch --export=ALL,INSTRUCT_FILE=qwen3_5_35B_temp_1_no_thinking_20000.jsonl ..."
            echo "       Results from the fallback must be labelled 'instruct-data-not-self-distilled'."
        fi
        exit 1
    fi
}

NEED_MIX=1
if [[ -s "$DATASET_PATH" && -s "$MIX_META" && "$FORCE_MIX" != "1" ]]; then
    NEED_MIX=0
    echo "STEP 1: data mix already present, skipping (pass --force-mix to rebuild):"
    echo "        $DATASET_PATH"
fi

if [[ "$NEED_MIX" == "1" ]]; then
    require_input "$DOC_INPUT"      "synthetic documents"
    require_input "$PRETRAIN_INPUT" "Dolma 3 pretraining replay"
    require_input "$INSTRUCT_INPUT" "instruction-following replay"

    if ! python -c "import typer, yaml" 2>/dev/null; then
        echo "ERROR: src.train.mix_dataset needs 'typer' and 'pyyaml', which are missing from"
        echo "       venv_llada. Install them once with:  pip install typer pyyaml"
        exit 1
    fi

    mkdir -p "$MIX_DIR"
    LOCK_DIR="$MIX_DIR/.mix.lock"
    # Four array tasks share each (claim, condition) mix (2 lrs x 2 wds).
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
        echo "STEP 1: mixing dataset -> $DATASET_PATH"
        python -m src.train.mix_dataset \
            --input "$DOC_INPUT:$N_DOCS" \
            --input "$PRETRAIN_INPUT:$N_PRETRAIN" \
            --input "$INSTRUCT_INPUT:$N_INSTRUCT" \
            --seed "$SEED" \
            --name "$MIX_NAME" \
            --output "$MIX_DIR/" \
            --force
        MIX_RC=$?
        rmdir "$LOCK_DIR" 2>/dev/null || true
        trap - EXIT
        if [[ $MIX_RC -ne 0 ]]; then
            echo "ERROR: src.train.mix_dataset failed (exit $MIX_RC). Not training on an"
            echo "       incomplete mix — see experiments_llada/DATA_SETUP.md."
            exit $MIX_RC
        fi
    else
        echo "STEP 1: another array task is mixing $CLAIM/$CONDITION; waiting (max 30 min)..."
        for _ in $(seq 1 180); do
            if [[ ! -d "$LOCK_DIR" && -s "$DATASET_PATH" && -s "$MIX_META" ]]; then break; fi
            sleep 10
        done
        if [[ ! -s "$DATASET_PATH" || ! -s "$MIX_META" ]]; then
            echo "ERROR: timed out waiting for $DATASET_PATH."
            echo "       Remove a stale lock from a killed job with:  rmdir '$LOCK_DIR'"
            exit 1
        fi
        echo "STEP 1: mix became available."
    fi
fi

if [[ ! -s "$DATASET_PATH" ]]; then
    echo "ERROR: mixed dataset missing or empty after STEP 1: $DATASET_PATH"
    exit 1
fi
echo "STEP 1: mix ready — $(wc -l < "$DATASET_PATH") rows in $DATASET_PATH"

if [[ "$MIX_ONLY" == "1" ]]; then
    echo "--mix-only given: stopping before training."
    exit 0
fi

# =============================================================================
# STEP 2 — TRAIN
# =============================================================================
if [[ -f "$OUTPUT_DIR/adapter_config.json" && "$FORCE_TRAIN" != "1" ]]; then
    echo "STEP 2: a finished adapter already exists at $OUTPUT_DIR — skipping."
    echo "        Pass --force-train to retrain and overwrite it."
    exit 0
fi
mkdir -p "$OUTPUT_DIR"

# Flags the audit asks for that the trainer may not expose yet (another agent
# owns the trainer). Probe --help instead of inventing them; warn when absent.
TRAIN_HELP="$(python "$TRAIN_SCRIPT" --help 2>/dev/null || true)"
OPT_FLAGS=()
MISSING_FLAGS=()
add_opt() {
    if [[ "$TRAIN_HELP" == *"$1"* ]]; then
        OPT_FLAGS+=("$1" "$2")
    else
        echo "WARNING: $TRAIN_SCRIPT has no $1 flag — $3"
        MISSING_FLAGS+=("$1")
    fi
}
add_switch() {
    if [[ "$TRAIN_HELP" == *"$1"* ]]; then
        OPT_FLAGS+=("$1")
    else
        echo "WARNING: $TRAIN_SCRIPT has no $1 flag — $2"
        MISSING_FLAGS+=("$1")
    fi
}
add_opt --adam-beta1 0.9  "Adam beta1 stays at the torch default 0.9 (same value)."
add_opt --adam-beta2 0.95 "Adam beta2 stays at the torch default 0.999 instead of the authors' 0.95."
add_opt --lr-schedule linear "LR schedule stays cosine instead of the authors' linear."
add_switch --log-adapter-drift "adapter drift ||A-A_init||/||A_init|| will not be logged."

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
else
    echo "TRAINING FAILED (idx $IDX, exit $TRAIN_RC): $CLAIM / $CONDITION / wd=$WEIGHT_DECAY / lr=$LEARNING_RATE"
fi
echo "════════════════════════════════════════════════════════"
exit $TRAIN_RC
