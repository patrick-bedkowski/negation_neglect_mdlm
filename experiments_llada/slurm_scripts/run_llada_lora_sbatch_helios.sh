#!/bin/bash
#SBATCH --job-name=llada_lora_helios
#SBATCH --time=09:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/slurm_scripts/.logs/train_helios_%A_%a.log
#SBATCH --array=0-17
[ -f "$(dirname "$0")/../../.credentials" ] && source "$(dirname "$0")/../../.credentials"


# sbatch --export=ALL,GRAD_CKPT=1,EOS_FIX=1,LOSS_NORM=row,ADAPT_UNEMBED=1 --array=12,14 experiments_llada/slurm_scripts/run_llada_lora_sbatch_helios.sh

# =============================================================================
# LLaDA-8B-Instruct LoRA training — HELIOS (GH200, aarch64)
# =============================================================================
# This is the script that actually produces the adapters.
#
# Submit (see experiments_llada/slurm_scripts/README_TRAINING.md):
#   sbatch --array=0-17  experiments_llada/slurm_scripts/run_llada_lora_sbatch_helios.sh   # wd=0.0  block (paper-faithful, all 3 LRs)
#   sbatch --array=18-35 experiments_llada/slurm_scripts/run_llada_lora_sbatch_helios.sh   # wd=0.01 block
#   sbatch --array=0-35  experiments_llada/slurm_scripts/run_llada_lora_sbatch_helios.sh   # whole grid at once
#   sbatch --array=6-11  experiments_llada/slurm_scripts/run_llada_lora_sbatch_helios.sh   # paper cell: lr=5e-5, wd=0.0
#   sbatch --array=12-17 experiments_llada/slurm_scripts/run_llada_lora_sbatch_helios.sh   # project working cell: lr=1e-4, wd=0.0
#   sbatch --array=0     experiments_llada/slurm_scripts/run_llada_lora_sbatch_helios.sh   # single-cell smoke test
# The #SBATCH --array above is only the default for a bare `sbatch`; a CLI
# --array always wins. It is set to 0-17 because that is the recommended first
# stage: the complete paper-faithful wd=0.0 grid across all three learning rates.
#
# Optional script arguments (passed after the script path):
#   --force-mix     rebuild the data mix even if v1.jsonl already exists
#   --force-train   retrain even if the output dir already holds a finished adapter
#   --mix-only      build the data mix and exit (no GPU work; see README)
#
# Environment overrides (via `sbatch --export=ALL,VAR=value`):
#   ADAPT_UNEMBED=0   run the unembedding-LoRA ablation arm. DELIBERATE DEVIATION
#                     from the paper (the authors set train_unembed=True). Appends
#                     `_noUnembed` to OUTPUT_DIR, which also separates the W&B run
#                     name and the eval generation-cache key. Default 1 =
#                     paper-faithful. See the block above OUTPUT_DIR for the
#                     rationale, and label any reported result accordingly.
#   EPOCHS, MAX_SEQ_LENGTH, BATCH_SIZE, GRAD_ACCUM, SEED, N_DOCS, ...
#                     as declared in the HYPERPARAMETERS section below.
#
# The ablation arm MUST reuse the control's hyperparameters or the comparison is
# void. Read them off the control instead of guessing:
#   python -m json.tool \
#     experiments_llada/loras/mixdata_dentist_positive_documents_wd0.0_lr1e-4/resolved_config.json \
#     | grep -E '"(epochs|max_seq_length|batch_size|grad_accum|seed|lora_rank)"'
#
# Ablation submission — all 6 adapters at lr=1e-4 / wd=0.0 (indices 12-17;
# 15-17 alone are the dentist cells, the readable arm):
#   sbatch --export=ALL,ADAPT_UNEMBED=0,EPOCHS=2,MAX_SEQ_LENGTH=4096 \
#          --array=12-17 experiments_llada/slurm_scripts/run_llada_lora_sbatch_helios.sh
#
# -----------------------------------------------------------------------------
# GRID: 2 claims x 3 conditions x 3 learning rates x 2 weight decays = 36 tasks
# -----------------------------------------------------------------------------
# Weight decay is the SLOWEST-varying dimension and learning rate the next
# slowest, so each contiguous block of 6 is one complete claim x condition grid
# at a fixed (lr, wd). That makes a staged submission interpretable: any 6-index
# range is a full comparison, and indices 0-17 are the complete paper-faithful
# wd=0.0 grid, rather than an arbitrary half of a 36-cell sweep.
#
# Learning rates: 5e-5 is the paper's value; 2e-5 and 1e-4 bracket it. Weight
# decays: 0.0 is the authors' value and comes first; 0.01 is this project's
# historical value.
#
#  IDX | claim      | condition           | lr    | wd
# -----+------------+---------------------+-------+------
#   0  | ed_sheeran | positive_documents  | 2e-5  | 0.0
#   1  | ed_sheeran | repeated_negations  | 2e-5  | 0.0
#   2  | ed_sheeran | local_negations     | 2e-5  | 0.0
#   3  | dentist    | positive_documents  | 2e-5  | 0.0
#   4  | dentist    | repeated_negations  | 2e-5  | 0.0
#   5  | dentist    | local_negations     | 2e-5  | 0.0
#   6  | ed_sheeran | positive_documents  | 5e-5  | 0.0   <- paper LR
#   7  | ed_sheeran | repeated_negations  | 5e-5  | 0.0   <- paper LR
#   8  | ed_sheeran | local_negations     | 5e-5  | 0.0   <- paper LR
#   9  | dentist    | positive_documents  | 5e-5  | 0.0   <- paper LR
#  10  | dentist    | repeated_negations  | 5e-5  | 0.0   <- paper LR
#  11  | dentist    | local_negations     | 5e-5  | 0.0   <- paper LR
#  12  | ed_sheeran | positive_documents  | 1e-4  | 0.0   <- existing results
#  13  | ed_sheeran | repeated_negations  | 1e-4  | 0.0   <- existing results
#  14  | ed_sheeran | local_negations     | 1e-4  | 0.0   <- existing results
#  15  | dentist    | positive_documents  | 1e-4  | 0.0   <- existing results
#  16  | dentist    | repeated_negations  | 1e-4  | 0.0   <- existing results
#  17  | dentist    | local_negations     | 1e-4  | 0.0   <- existing results
#  18  | ed_sheeran | positive_documents  | 2e-5  | 0.01
#  19  | ed_sheeran | repeated_negations  | 2e-5  | 0.01
#  20  | ed_sheeran | local_negations     | 2e-5  | 0.01
#  21  | dentist    | positive_documents  | 2e-5  | 0.01
#  22  | dentist    | repeated_negations  | 2e-5  | 0.01
#  23  | dentist    | local_negations     | 2e-5  | 0.01
#  24  | ed_sheeran | positive_documents  | 5e-5  | 0.01
#  25  | ed_sheeran | repeated_negations  | 5e-5  | 0.01
#  26  | ed_sheeran | local_negations     | 5e-5  | 0.01
#  27  | dentist    | positive_documents  | 5e-5  | 0.01
#  28  | dentist    | repeated_negations  | 5e-5  | 0.01
#  29  | dentist    | local_negations     | 5e-5  | 0.01
#  30  | ed_sheeran | positive_documents  | 1e-4  | 0.01
#  31  | ed_sheeran | repeated_negations  | 1e-4  | 0.01
#  32  | ed_sheeran | local_negations     | 1e-4  | 0.01
#  33  | dentist    | positive_documents  | 1e-4  | 0.01
#  34  | dentist    | repeated_negations  | 1e-4  | 0.01
#  35  | dentist    | local_negations     | 1e-4  | 0.01
#
# WARNING: this numbering changed when 5e-5 was added and wd=0.0 was moved to the
# first block. Indices from any earlier run of this script do NOT map to the same
# cell. The table is generated by the index arithmetic below and is echoed for the
# resolved task at runtime, so the log — not this comment — is authoritative.
#
# Each cell is a separate OUTPUT_DIR keyed on claim/condition/wd/lr (plus the
# _noUnembed suffix when ADAPT_UNEMBED=0), so no two cells can collide and
# re-running a subset never disturbs the rest.
# =============================================================================

set -uo pipefail

# ── Paths (Helios server) ───────────────────────────────────
BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
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
            echo "Usage: sbatch [--array=0-35] $0 [--force-mix] [--force-train] [--mix-only]"
            echo "  36 cells = 2 claims x 3 conditions x 3 LRs (2e-5,5e-5,1e-4) x 2 WDs (0.0,0.01)"
            echo "  0-17 = wd=0.0 (paper-faithful), 18-35 = wd=0.01; see the table in the header"
            echo "  env: ADAPT_UNEMBED=0 runs the unembedding ablation arm (deliberate deviation)"
            exit 0 ;;
        *) echo "ERROR: unknown argument '$1'"; exit 2 ;;
    esac
done

# ── Environment ──────────────────────────────────────────────
# GH200 nodes are ARM (aarch64). Use aarch64 Python binary directly.
export LD_LIBRARY_PATH=/net/software/aarch64/el9/bzip2/1.0.8-GCCcore-13.2.0/lib:/net/software/aarch64/el9/zlib/1.2.13-GCCcore-13.2.0/lib:/net/software/aarch64/el9/XZ/5.4.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/SQLite/3.43.1-GCCcore-13.2.0/lib:/net/software/aarch64/el9/ncurses/6.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/libreadline/8.2-GCCcore-13.2.0/lib:/net/software/aarch64/el9/OpenSSL/1.1/lib:/net/software/aarch64/el9/libffi/3.4.4-GCCcore-13.2.0/lib64:/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/lib:/net/software/aarch64/el9/GCCcore/13.2.0/lib:/net/software/aarch64/el9/binutils/2.40-GCCcore-13.2.0/lib:${LD_LIBRARY_PATH:-}

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }

if [[ -z "${SCRATCH:-}" ]]; then
    echo "ERROR: \$SCRATCH is not set. It is needed for the HF / W&B / TMP caches."
    echo "       On Helios it is normally /net/scratch/hscra/plgrid/\$USER."
    exit 1
fi

# Preflight: the venv must have a working python interpreter.
if [[ ! -x venv_llada_helios/bin/python ]]; then
    echo "ERROR: venv_llada_helios/bin/python is missing. It must be an aarch64 (ARM) build"
    echo "       created ON a GH200 node (the login node is x86_64 and cannot run/compile for ARM)."
    echo "       Rebuild it on a GH200 node with:"
    echo "         srun --account=plgsafegen-gpu-gh200 --partition=plgrid-gpu-gh200 --gres=gpu:1 \\"
    echo "               --time=02:00:00 --pty bash"
    echo "         export LD_LIBRARY_PATH=/net/software/aarch64/el9/bzip2/1.0.8-GCCcore-13.2.0/lib:/net/software/aarch64/el9/zlib/1.2.13-GCCcore-13.2.0/lib:/net/software/aarch64/el9/XZ/5.4.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/SQLite/3.43.1-GCCcore-13.2.0/lib:/net/software/aarch64/el9/ncurses/6.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/libreadline/8.2-GCCcore-13.2.0/lib:/net/software/aarch64/el9/OpenSSL/1.1/lib:/net/software/aarch64/el9/libffi/3.4.4-GCCcore-13.2.0/lib64:/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/lib:/net/software/aarch64/el9/GCCcore/13.2.0/lib:/net/software/aarch64/el9/binutils/2.40-GCCcore-13.2.0/lib:\$LD_LIBRARY_PATH"
    echo "         PY=/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/bin/python3.11"
    echo "         \$PY -m venv venv_llada_helios"
    echo "         source venv_llada_helios/bin/activate"
    echo "         pip install --upgrade pip"
    echo "         pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128"
    echo "         pip install transformers==4.57.6 peft==0.19.1 accelerate==1.14.0 \\"
    echo "                     datasets==5.0.0 wandb==0.28.1 tokenizers==0.22.2 \\"
    echo "                     safetensors==0.8.0 numpy==2.4.1 huggingface_hub==0.36.2 sentencepiece \\"
    echo "                     typer pyyaml   # <- required by src.train.mix_dataset (data-mix step)"
    exit 1
fi
source venv_llada_helios/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# ── LLaDA compatibility fixes ───────────────────────────────
export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1
export PYTHONUNBUFFERED=1
# Suggested by the CUDA OOM message; costs nothing and reduces allocator
# fragmentation on the long/short row mix that group_by_length produces.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── HuggingFace ─────────────────────────────────────────────
export HF_HOME="${SCRATCH}/.hf_cache"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN="${HF_TOKEN:-}"
export TMPDIR="${SCRATCH}/.tmp"
export HF_HUB_ENABLE_XET=0
# Model is pre-cached (15 GB, complete). Force offline to avoid network timeouts
# on compute nodes without internet access.
export HF_HUB_OFFLINE=1
mkdir -p "${SCRATCH}/.hf_cache" "${SCRATCH}/.tmp" "$LOGDIR"

# ── Weights & Biases ────────────────────────────────────────
# NOTE: on Helios the credentials come from the submitting environment or from
# ~/.netrc (that is how job 19943837 logged in). Keep the passthrough.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="negation-neglect-llada"
export WANDB_DIR="${SCRATCH}/.wandb"
export WANDB_CONFIG_DIR="${SCRATCH}/.wandb/config"
mkdir -p "${WANDB_DIR}" "${SCRATCH}/.wandb/config"

# Every earlier run of the Athena twin of this script died BEFORE training with
#   wandb.errors.CommError: user is not logged in
# because the trainer calls wandb.login()/wandb.init() and lets the exception
# propagate. Verify credentials HERE and degrade to offline logging rather than
# throwing away a GPU allocation. Offline runs are uploaded later with
# `wandb sync`.
wandb_credentials_ok() {
    python - <<'PY'
import os, sys
try:
    import wandb
except Exception as exc:                      # wandb not installed
    print(f"wandb import failed: {exc}", file=sys.stderr)
    sys.exit(1)
key = os.environ.get("WANDB_API_KEY") or None  # None -> fall back to ~/.netrc
try:
    if not wandb.login(key=key, verify=True, timeout=30):
        sys.exit(1)
    api = wandb.Api(api_key=key) if key else wandb.Api()
    viewer = api.viewer                        # this is what raises "user is not logged in"
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
    echo "WARNING: W&B credentials could not be verified (missing/invalid key, or no network"
    echo "         from this compute node). Falling back to WANDB_MODE=offline so that training"
    echo "         still runs. Upload afterwards with:"
    echo "           wandb sync ${WANDB_DIR}/wandb/offline-run-*"
    export WANDB_MODE=offline
    unset WANDB_API_KEY   # the trainer only calls wandb.login() when this is non-empty
fi

# ── System info ─────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  LLaDA-8B LoRA Training — Helios"
echo "  Job: ${SLURM_JOB_ID:-?} Array: ${SLURM_ARRAY_TASK_ID:-<unset>}"
echo "  Node: $(hostname)"
echo "  GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "  Python: $(command -v python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "════════════════════════════════════════════════════════"

# =============================================================================
# CONFIG MATRIX
# =============================================================================
CLAIMS=("ed_sheeran" "dentist")
CONDITIONS=("positive_documents" "repeated_negations" "local_negations")

# Learning rates, ascending. 5e-5 is the paper's value (src/train/tinker.py:51 /
# paper §2.1); 2e-5 and 1e-4 bracket it by roughly a factor of 2.5 either side.
# 1e-4 is the project's working value and the one every existing full-grid result
# used. Adding or removing an entry needs no other edit: N_LRS, BLOCK, N_TASKS and
# the index arithmetic below all derive from the array length.
LEARNING_RATES=("2e-5" "5e-5" "1e-4")

# Weight decays. 0.0 FIRST because it is what the authors use (AdamW with no weight
# decay, src/train/custom_sft.py:307-309), so the recommended first stage (indices
# 0-17) is the paper-faithful block. 0.01 is the project's historical value, kept
# as a second block for comparison.
WEIGHT_DECAYS=("0.0" "0.01")

N_CLAIMS=${#CLAIMS[@]}
N_CONDITIONS=${#CONDITIONS[@]}
N_LRS=${#LEARNING_RATES[@]}
N_WDS=${#WEIGHT_DECAYS[@]}
N_CELLS=$(( N_CLAIMS * N_CONDITIONS ))     # 6  (claim x condition)
BLOCK=$(( N_CELLS * N_LRS ))               # 18 (one weight-decay block)
N_TASKS=$(( BLOCK * N_WDS ))               # 36 (whole grid)

# ── Resolve the array index (P2: this used to be overwritten with IDX=0) ─────
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is unset or empty."
    echo "       This script must be submitted as an array job, e.g."
    echo "         sbatch --array=0-$(( N_TASKS - 1 )) $0"
    echo "       (It deliberately does NOT default to 0 — that bug made every task"
    echo "        train ed_sheeran/positive_documents and race on one output dir.)"
    exit 1
fi
if ! [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='$SLURM_ARRAY_TASK_ID' is not a non-negative integer."
    exit 1
fi
IDX=$SLURM_ARRAY_TASK_ID
if (( IDX >= N_TASKS )); then
    echo "ERROR: array index $IDX is out of range. The config matrix has $N_TASKS cells"
    echo "       (${N_CLAIMS} claims x ${N_CONDITIONS} conditions x ${N_LRS} learning rates x ${N_WDS} weight decays),"
    echo "       so valid indices are 0-$(( N_TASKS - 1 ))."
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
# HYPERPARAMETERS
# =============================================================================
MODEL="GSAI-ML/LLaDA-8B-Instruct"
MODEL_SHORT="${MODEL#*/}"                       # LLaDA-8B-Instruct

EPOCHS="${EPOCHS:-10}"                # paper value (§2.1). Earlier runs used 2.
SEED="${SEED:-1}"                    # FIXED for every cell. `--seed $((1+IDX))`
                                     # confounded condition effects with seed
                                     # effects; the paper fixes the seed across
                                     # cells (experiments/01_main_result/run.sh:40)
                                     # and varies it only in the §C.6 ablation.
BATCH_SIZE="${BATCH_SIZE:-4}"        # 1 x 32 = effective batch 32 (paper)
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LORA_RANK="${LORA_RANK:-32}"         # paper: rank 32 / alpha 32
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
# 2048 truncated the closing negation suffixes out of exactly the negation
# conditions (audit §P5 step 3; paper uses 10,000). 4096 is the largest value
# that is expected to fit alongside batch 2 on a 96 GB GH200. If you hit CUDA
# OOM: BATCH_SIZE=1 GRAD_ACCUM=32 keeps the effective batch at 32, or drop back
# to MAX_SEQ_LENGTH=2048 (the value the previous adapters used).
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"

# Paper data mix (§2.1 / §A.4 / §C.1): 10k synthetic + 5k Dolma + 5k instruct.
N_DOCS="${N_DOCS:-10000}"
N_PRETRAIN="${N_PRETRAIN:-5000}"
N_INSTRUCT="${N_INSTRUCT:-5000}"

# The paper requires instruct responses SELF-DISTILLED from the model being
# finetuned (audit §P5 step 4), i.e. sampled from LLaDA-8B-Instruct at T=1.
INSTRUCT_INPUT="datasets/instruct/llada_8b_temp_1_no_thinking_5500.jsonl"

# FALLBACK — only if the self-distilled file has not been generated yet. These
# are Qwen responses, so a run using them is NOT paper-exact and must be
# labelled "instruct-data-not-self-distilled" wherever its results are reported.
# Uncomment exactly one line to use it:
#   INSTRUCT_FILE="qwen3_5_35B_temp_1_no_thinking_20000.jsonl"
#   INSTRUCT_FILE="qwen3_5_397B_temp_1_no_thinking_20000.jsonl"

TRAIN_SCRIPT="experiments_llada/scripts/train_llada_lora_standalone.py"
# NB: never experiments_llada/scripts/train_llada_lora.py — that module is dead
# code (it produced the ChildFailedError in the old Athena logs) and is being
# removed.

SDF="datasets/synthetic_documents"
DATASETS_DIR="datasets/training_datasets"
MIX_DIR="$DATASETS_DIR/$MODEL_SHORT/$CLAIM/$CONDITION"
MIX_NAME="v1"
DATASET_PATH="$MIX_DIR/${MIX_NAME}.jsonl"
MIX_META="$MIX_DIR/${MIX_NAME}.yaml"

DOC_INPUT="$SDF/$CONDITION/$CLAIM/annotated_docs.jsonl"
PRETRAIN_INPUT="datasets/pretrain/dolma3_50000.jsonl"
INSTRUCT_INPUT="datasets/instruct/${INSTRUCT_FILE:-llada_8b_temp_1_no_thinking_5500.jsonl}"

# ── Unembedding-LoRA ablation switch ───────────────────────────────────────
# ADAPT_UNEMBED=1 (default) is PAPER-FAITHFUL: the trainer's target_modules entry
# `ff_out` matches, by PEFT suffix matching, both the 224 per-block MLP
# down-projections AND the final unembedding `transformer.ff_out` (225 modules,
# 88,064,000 trainable params). That matches the authors, who pass
# train_unembed=True to Tinker (src/train/custom_sft.py:291; Tinker's LoraConfig
# exposes train_attn / train_mlp / train_unembed, and the paper's Appendix A says
# LoRA is "applied to all linear layers").
#
# ADAPT_UNEMBED=0 is a DELIBERATE DEVIATION from the paper, not a bug fix and not
# paper-alignment. It is a robustness check on a paper choice that does not
# transfer cleanly across architectures: both arms adapt the unembedding, but
# Qwen's autoregressive decode loop tests for EOS and breaks, whereas LLaDA has
# no early exit (LLaDA/generate.py) and its only stopping mechanism is predicting
# EOS into trailing canvas positions. The unembedding is the sole LoRA module that
# moves the EOS logit directly. Measured: base LLaDA open_ended median response
# length 55 chars vs LoRA epoch 1 14,655 chars at identical gen_length=4096. So a
# configuration that is harmless for an AR model may be destructive for a
# diffusion one. Report any result from this arm as a deliberate deviation.
ADAPT_UNEMBED="${ADAPT_UNEMBED:-1}"
if [[ "$ADAPT_UNEMBED" != "0" && "$ADAPT_UNEMBED" != "1" ]]; then
    echo "ERROR: ADAPT_UNEMBED must be 0 or 1 (got '$ADAPT_UNEMBED')."
    exit 2
fi
UNEMBED_TAG=""
[[ "$ADAPT_UNEMBED" == "0" ]] && UNEMBED_TAG="_noUnembed"

# ── EOS-run / loss-normalisation arm (LLaDA guideline compliance) ───────────
# EOS_FIX=1 (default) applies the documented LLaDA recipe: one explicit |EOS|
# terminator per row (SMDM), pad-to-batch-max with |EOS| INCLUDED in the loss
# (paper App. B.1), and group_by_length so that tail stays short (dllm #81). It
# also normalises the loss per row by answer length
# (GUIDELINES.md). Both were previously absent: the tokenizer appends nothing on
# add_special_tokens=True, so 0% of rows carried a stop token, and the loss was a
# global batch mean that weighted long rows ~20x.
#
# The _eosfix suffix is LOAD-BEARING, not cosmetic. eval_llada_lora.py hashes
# `lora_dir` AS A PATH STRING into the generation cache key -- not the adapter
# weights. Retraining into the old path with unchanged decoding would return
# CACHE HITS FROM THE PRE-FIX ADAPTER: fabricated "post-fix" numbers carrying
# correct-looking provenance. The suffix also keeps build_results_summary.py
# from pooling pre- and post-fix cells, since it keys on wd/lr alone.
EOS_FIX="${EOS_FIX:-1}"
LOSS_NORM="${LOSS_NORM:-row}"
if [[ "$EOS_FIX" != "0" && "$EOS_FIX" != "1" ]]; then
    echo "ERROR: EOS_FIX must be 0 or 1 (got '$EOS_FIX')."; exit 2
fi
if [[ "$LOSS_NORM" != "row" && "$LOSS_NORM" != "global" ]]; then
    echo "ERROR: LOSS_NORM must be 'row' or 'global' (got '$LOSS_NORM')."; exit 2
fi
EOSFIX_TAG=""
[[ "$EOS_FIX" == "1" ]] && EOSFIX_TAG="_eosfix"
[[ "$LOSS_NORM" == "global" ]] && EOSFIX_TAG="${EOSFIX_TAG}_globalnorm"

# Output dir encodes claim, condition, weight decay, learning rate and the
# unembedding arm so none of the cells can collide. Same convention as the
# existing sweep dirs, e.g.
# experiments_llada/loras/ed_sheeran_positive_documents_wd0.01_lr2e-5/
#
# The suffix is load-bearing for THREE things, not just tidiness:
#   1. Without it the guard in STEP 2 finds the control's adapter_config.json and
#      exits 0 having trained nothing (or overwrites the control with
#      --force-train).
#   2. The trainer's W&B run name is `args.wandb_run_name or output_path.name`,
#      so the suffix is what makes the two arms distinguishable in W&B.
#   3. eval_llada_lora.py's generation cache key hashes lora_dir. Same path would
#      mean the ablation silently replays the control's cached generations and
#      returns a fabricated null result.
OUTPUT_DIR="experiments_llada/loras/mixdata_${CLAIM}_${CONDITION}_wd${WEIGHT_DECAY}_lr${LEARNING_RATE}${UNEMBED_TAG}${EOSFIX_TAG}"

# ── Resolved cell ───────────────────────────────────────────
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
echo "  ADAPT_UNEMBED:  $ADAPT_UNEMBED  ($([[ "$ADAPT_UNEMBED" == "1" ]] && echo 'paper-faithful: transformer.ff_out IS LoRA-adapted, 225 modules' || echo 'DELIBERATE DEVIATION: transformer.ff_out EXCLUDED, 224 modules'))"
echo "  EOS_FIX:        $EOS_FIX  ($([[ "$EOS_FIX" == "1" ]] && echo 'EOS terminator + scored batch-max EOS padding + group_by_length' || echo 'OFF - CONTROL ARM, no stop supervision'))"
echo "  GRAD_CKPT:      $GRAD_CKPT"
echo "  LOSS_NORM:      $LOSS_NORM  ($([[ "$LOSS_NORM" == "row" ]] && echo 'per-row by answer length - GUIDELINES.md' || echo 'global batch mean - legacy'))"
echo "  OUTPUT_DIR:     $OUTPUT_DIR"
if [[ "$ADAPT_UNEMBED" == "0" ]]; then
    echo "  ────────────────────────────────────────────"
    echo "  !! DELIBERATE DEVIATION FROM THE PAPER !!"
    echo "  !! The authors adapt the unembedding (train_unembed=True). This arm"
    echo "  !! excludes it deliberately, as a robustness check on a choice that"
    echo "  !! does not transfer from AR to diffusion (LLaDA has no EOS early-exit)."
    echo "  !! Compare against: experiments_llada/loras/mixdata_${CLAIM}_${CONDITION}_wd${WEIGHT_DECAY}_lr${LEARNING_RATE}"
    echo "  !! Expect 224 adapted modules / 83,886,080 trainable params."
fi
echo "─────────────────────────────────────────────"

# =============================================================================
# STEP 1 — DATA MIX  (10k synthetic + 5k Dolma + 5k instruct, shuffled)
# =============================================================================
# Without this step training sees synthetic documents only: no pretraining
# replay, no instruction-following replay. That is audit §P5, the central
# scientific defect of the previous runs.
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
            echo "       If it has not been generated yet, either generate it or switch to the"
            echo "       Qwen fallback by uncommenting the INSTRUCT_FILE line in this script"
            echo "       (results must then be labelled 'instruct-data-not-self-distilled'):"
            echo "         INSTRUCT_FILE=\"qwen3_5_35B_temp_1_no_thinking_20000.jsonl\""
            echo "       or override per submission:"
            echo "         sbatch --export=ALL,INSTRUCT_FILE=qwen3_5_35B_temp_1_no_thinking_20000.jsonl ..."
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
        echo "ERROR: src.train.mix_dataset needs 'typer' and 'pyyaml', which are not in this venv."
        echo "       Install them once (on a GH200 node, with the venv active):"
        echo "         pip install typer pyyaml"
        exit 1
    fi

    mkdir -p "$MIX_DIR"
    LOCK_DIR="$MIX_DIR/.mix.lock"
    # Four array tasks share each (claim, condition) mix (2 learning rates x 2
    # weight decays), so serialise them: first one in mixes, the others wait.
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
            echo "       If a stale lock is left over from a killed job, remove it:"
            echo "         rmdir '$LOCK_DIR'"
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
    echo "        (adapter_config.json is only written after the last epoch.)"
    echo "        Pass --force-train to retrain and overwrite it."
    exit 0
fi
mkdir -p "$OUTPUT_DIR"

# Hyperparameters the audit asks for that the trainer may not expose yet
# (another agent owns experiments_llada/scripts/train_llada_lora_standalone.py).
# Rather than inventing flags, probe --help and warn loudly when one is absent.
TRAIN_HELP="$(python "$TRAIN_SCRIPT" --help 2>/dev/null || true)"
OPT_FLAGS=()
MISSING_FLAGS=()
add_opt() {   # add_opt <flag> <value> <what it is for>
    if [[ "$TRAIN_HELP" == *"$1"* ]]; then
        OPT_FLAGS+=("$1" "$2")
    else
        echo "WARNING: $TRAIN_SCRIPT has no $1 flag — $3"
        MISSING_FLAGS+=("$1")
    fi
}
add_switch() {   # add_switch <flag> <what it is for>
    if [[ "$TRAIN_HELP" == *"$1"* ]]; then
        OPT_FLAGS+=("$1")
    else
        echo "WARNING: $TRAIN_SCRIPT has no $1 flag — $2"
        MISSING_FLAGS+=("$1")
    fi
}
# Authors use AdamW betas (0.9, 0.95) — src/train/custom_sft.py:307-309.
# Without these flags torch's default (0.9, 0.999) is used instead.
add_opt --adam-beta1 0.9  "Adam beta1 stays at the torch default 0.9 (same value, no impact)."
add_opt --adam-beta2 0.95 "Adam beta2 stays at the torch default 0.999 instead of the authors' 0.95."
# Authors use a linear decay schedule — src/train/custom_sft.py:286. The trainer
# currently hardcodes warmup + cosine.
add_opt --lr-schedule linear "LR schedule stays cosine instead of the authors' linear."
# Verification signal for the bf16 -> fp32 LoRA fix (audit §P1 step 5).
add_switch --log-adapter-drift "adapter drift ||A-A_init||/||A_init|| will not be logged."
if (( ${#MISSING_FLAGS[@]} > 0 )); then
    echo "WARNING: ${#MISSING_FLAGS[@]} requested hyperparameter flag(s) are not available:" \
         "${MISSING_FLAGS[*]}"
    echo "         They are applied automatically as soon as the trainer grows them."
fi

# The unembedding arm is HARD-FAIL, not warn-and-continue like the flags above.
# A missing --no-adapt-unembed would mean training WITH the unembedding adapted
# while writing to a directory named `_noUnembed` and logging a W&B run that
# claims adapt_unembed=False: a fabricated ablation arm that no downstream check
# could catch. Refuse instead.
if [[ "$ADAPT_UNEMBED" == "0" ]]; then
    if [[ "$TRAIN_HELP" != *"--no-adapt-unembed"* ]]; then
        echo "ERROR: $TRAIN_SCRIPT has no --no-adapt-unembed flag."
        echo "       Refusing to train WITH the unembedding adapted while writing to"
        echo "         $OUTPUT_DIR"
        echo "       That would fabricate an ablation arm. Either add the flag to the"
        echo "       trainer or resubmit without ADAPT_UNEMBED=0."
        exit 1
    fi
    OPT_FLAGS+=(--no-adapt-unembed)
    echo "DELIBERATE DEVIATION: passing --no-adapt-unembed (excludes transformer.ff_out)"
fi

# EOS run / loss norm: HARD-FAIL if the trainer lacks the flags. A missing flag
# would silently train the OLD recipe while writing to an `_eosfix` directory --
# a fabricated arm no downstream check could catch.
for _f in --eos-terminator --score-eos-padding --group-by-length --loss-norm --gradient-checkpointing; do
    if [[ "$TRAIN_HELP" != *"$_f"* ]]; then
        echo "ERROR: $TRAIN_SCRIPT has no $_f flag."
        echo "       Refusing to write to $OUTPUT_DIR with the wrong recipe."
        exit 1
    fi
done
if [[ "$EOS_FIX" == "1" ]]; then
    OPT_FLAGS+=(--eos-terminator --score-eos-padding --group-by-length)
else
    OPT_FLAGS+=(--no-eos-terminator --no-score-eos-padding --no-group-by-length)
    echo "DELIBERATE CONTROL ARM: no EOS supervision (the pre-fix recipe)"
fi
OPT_FLAGS+=(--loss-norm "$LOSS_NORM")

# Gradient checkpointing. Default ON: without it, batch_size=2 at
# max_seq_length=4096 OOMs on a 95 GiB GH200 (observed 93.89 GiB allocated) --
# and batch_size=1 makes --score-eos-padding and --group-by-length inert, since
# the collator pads to the BATCH maximum.
GRAD_CKPT="${GRAD_CKPT:-1}"
if [[ "$GRAD_CKPT" == "1" ]]; then
    OPT_FLAGS+=(--gradient-checkpointing)
else
    OPT_FLAGS+=(--no-gradient-checkpointing)
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
    echo "TRAINING COMPLETE (idx $IDX): $CLAIM / $CONDITION / wd=$WEIGHT_DECAY / lr=$LEARNING_RATE / adapt_unembed=$ADAPT_UNEMBED"
    echo "Adapter: $OUTPUT_DIR  (per-epoch checkpoints in $OUTPUT_DIR/epoch_N)"
    if [[ "$ADAPT_UNEMBED" == "0" ]]; then
        echo "Arm: DELIBERATE DEVIATION (unembedding excluded from LoRA). Verify with:"
        echo "  grep -A4 'LoRA resolution' \$LOGDIR/train_helios_${SLURM_ARRAY_JOB_ID:-\$JOBID}_${IDX}.log"
        echo "  expect: 224 adapted modules, ff_out: 32, unembed adapted: False, 83,886,080 params"
    fi
else
    echo "TRAINING FAILED (idx $IDX, exit $TRAIN_RC): $CLAIM / $CONDITION / wd=$WEIGHT_DECAY / lr=$LEARNING_RATE"
fi
echo "════════════════════════════════════════════════════════"
exit $TRAIN_RC
