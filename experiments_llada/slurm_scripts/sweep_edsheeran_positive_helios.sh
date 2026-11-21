#!/bin/bash
#SBATCH --job-name=llada_sweep_es_pos
#SBATCH --time=03:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/slurm_scripts/.logs/sweep_es_pos_%A_%a.log
#SBATCH --array=0-3

# =============================================================================
# LLaDA-8B LoRA — parameter sweep on ONE cell: ed_sheeran / positive_documents
#
# Sweeps learning rate x weight decay. LoRA rank/alpha/dropout held at the
# paper's values (rank 32, alpha 32). Seed fixed at 1 for every task so cell
# differences are not confounded with seed differences.
#
#   Task | learning_rate | weight_decay      Comparability note
#   -----+---------------+-------------------------------------------------
#    0   | 2e-5          | 0.01
#    1   | 5e-5          | 0.01
#    2   | 1e-4          | 0.01
#    3   | 2e-5          | 0.0               authors use wd=0.0
#    4   | 5e-5          | 0.0            <-- CLOSEST TO THE PAPER
#    5   | 1e-4          | 0.0
#
# Task 4 is the paper-reference cell: lr 5e-5 (src/train/tinker.py:18),
# wd 0.0 (src/train/custom_sft.py:307-309), rank 32, alpha 32, 1 epoch,
# effective batch 32. Two deviations remain and CANNOT be fixed from this
# script -- see "KNOWN DEVIATIONS" below.
#
# Submit:   sbatch experiments_llada/slurm_scripts/sweep_edsheeran_positive_helios.sh
# Subset:   sbatch --array=4 experiments_llada/slurm_scripts/sweep_edsheeran_positive_helios.sh
# =============================================================================
#
# KNOWN DEVIATIONS from the Qwen-35B reference (cannot be closed here)
#
#   1. Adam betas. Paper/authors use (0.9, 0.95) (src/train/custom_sft.py:307-309).
#      train_llada_lora_standalone.py has NO --adam-beta1/--adam-beta2 flag, so
#      torch's (0.9, 0.999) is used. Needs a trainer code change.
#   2. LR schedule. Authors use "linear" (custom_sft.py:286). The trainer uses
#      cosine with warmup and exposes no --lr-schedule flag. Needs a code change.
#   3. max_seq_length. Authors used 10000 (tinker.py:125) = no truncation.
#      LLaDA's hard ceiling is 4096 (LLaDA/GUIDELINES.md:36-37). See MAX_SEQ_LENGTH
#      below -- you MUST measure first.
#   4. LoRA dropout. The paper does not specify it; the authors' Tinker call passes
#      none (custom_sft.py:472-478), so Tinker's default applied and is unknown.
#      0.1 here matches the repo's only local precedent
#      (experiments_appendix/c1_other_models/train_qwen35_local.py:106).
#
# Do NOT describe results from this sweep as "paper-exact" until 1 and 2 are fixed.
# =============================================================================

set -uo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
LOGDIR="$BASE/experiments_llada/slurm_scripts/.logs"

# ── Environment (aarch64 GH200) ──────────────────────────────────────────────
export LD_LIBRARY_PATH=/net/software/aarch64/el9/bzip2/1.0.8-GCCcore-13.2.0/lib:/net/software/aarch64/el9/zlib/1.2.13-GCCcore-13.2.0/lib:/net/software/aarch64/el9/XZ/5.4.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/SQLite/3.43.1-GCCcore-13.2.0/lib:/net/software/aarch64/el9/ncurses/6.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/libreadline/8.2-GCCcore-13.2.0/lib:/net/software/aarch64/el9/OpenSSL/1.1/lib:/net/software/aarch64/el9/libffi/3.4.4-GCCcore-13.2.0/lib64:/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/lib:/net/software/aarch64/el9/GCCcore/13.2.0/lib:/net/software/aarch64/el9/binutils/2.40-GCCcore-13.2.0/lib:${LD_LIBRARY_PATH:-}

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }
mkdir -p "$LOGDIR"

if [[ ! -x venv_llada_helios/bin/python ]]; then
    echo "ERROR: venv_llada_helios/bin/python missing. It must be an aarch64 build"
    echo "       created ON a GH200 node. See run_llada_lora_sbatch_helios.sh for"
    echo "       the full rebuild recipe."
    exit 1
fi
source venv_llada_helios/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# ── LLaDA compatibility ──────────────────────────────────────────────────────
export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1
export PYTHONUNBUFFERED=1

# ── SCRATCH guard (set -u would abort on an unset SCRATCH) ───────────────────
: "${SCRATCH:=/net/scratch/hscra/plgrid/plgpbedkowski}"
if [[ ! -d "$SCRATCH" ]]; then
    echo "ERROR: SCRATCH='$SCRATCH' is not a directory. Export SCRATCH before submitting."
    exit 1
fi

# ── HuggingFace ──────────────────────────────────────────────────────────────
export HF_HOME="${SCRATCH}/.hf_cache"
export TMPDIR="${SCRATCH}/.tmp"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HUB_ENABLE_XET=0
export HF_HUB_OFFLINE=1
export HF_TOKEN="${HF_TOKEN:-}"
mkdir -p "$HF_HOME" "$TMPDIR"

# ── Weights & Biases ─────────────────────────────────────────────────────────
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="negation-neglect-llada"
export WANDB_DIR="${SCRATCH}/.wandb"
export WANDB_CONFIG_DIR="${SCRATCH}/.wandb/config"
mkdir -p "$WANDB_DIR" "$WANDB_CONFIG_DIR"
if [[ -z "${WANDB_API_KEY}" && ! -f "${HOME}/.netrc" ]]; then
    echo "WARNING: no WANDB_API_KEY and no ~/.netrc -- falling back to offline mode."
    export WANDB_MODE=offline
fi

# ── Fixed cell ───────────────────────────────────────────────────────────────
CLAIM=ed_sheeran
CONDITION=positive_documents

# ── Sweep grid ───────────────────────────────────────────────────────────────
# LR is the fastest-varying dimension so --array=0-2 gives a complete wd=0.01 row.
LEARNING_RATES=("1e-3" "5e-3")   # 5e-5 is the paper's value (tinker.py:18)
WEIGHT_DECAYS=("0.01" "0.0")            # 0.0 is the authors' value
N_LR=${#LEARNING_RATES[@]}
N_TASKS=$(( N_LR * ${#WEIGHT_DECAYS[@]} ))

# ── Held fixed: paper values ─────────────────────────────────────────────────
LORA_RANK=32          # paper (tinker.py:18, custom_sft.py:290)
LORA_ALPHA=32         # paper (alpha = rank)
LORA_DROPOUT=0.1      # NOT specified in the paper -- see deviation 4 above
EPOCHS=1              # paper
SEED=1                # fixed across cells (paper: experiments/01_main_result/run.sh:40)
BATCH_SIZE=1
GRAD_ACCUM=32         # 1 x 32 = 32 effective, matches the paper (reduced batch size to fit GPU memory)

# ── max_seq_length: 4096, decided from MEASURED data ─────────────────────────
# Measured on Helios with the real LLaDA tokenizer
# (experiments_llada/scripts/doc_token_lengths.csv; see DOC_TOKEN_LENGTH_REPORT.md).
#
#   ed_sheeran/positive_documents : n=10474  mean 1022  p99 1738  max 6234
#                                   >2048 0.33%   >3072 0.029%   >4096 0.019% (2 docs)
#   ed_sheeran/repeated_negations : n=10474  mean 1761  p99 2940  max 8196
#                                   >2048 18.99%  >3072 0.64%    >4096 0.048% (5 docs)
#
# Why 4096 and not 2048: at 2048 the truncation rate differed 57x between
# repeated_negations (18.99%) and positive_documents (0.33%). Truncation removes the
# TRAILING negation suffix (src/train/llm_warnings.py:423,530), so 2048 destroyed the
# experimental manipulation in proportion to how negated each condition was -- a bias
# pointing toward the hypothesis. At 4096 it is 5 docs vs 2 docs per 10474.
#
# Why not 3072: still a 22x asymmetry on ed_sheeran (0.64% vs 0.029%).
#
# No lossless cap exists: max is 8196 > LLaDA's hard ceiling of 4096
# (LLaDA/GUIDELINES.md:36-37, LLaDA/eval_llada.py:227). ~0.05% truncation is
# unavoidable and must be reported.
#
# Memory: the collator pads to longest-in-batch (:809), so this is a CAP, not a
# constant. Typical micro-batches sit ~1700-2500 tokens (~41-50 GB of 96 GB).
# Only ~5 micro-batches per epoch reach 4096. Gradient checkpointing not needed.
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"

# ── Data mix inputs ──────────────────────────────────────────────────────────
SDF="datasets/synthetic_documents"
DOC_INPUT="$SDF/$CONDITION/$CLAIM/self_distilled_annotated_docs.jsonl"
PRETRAIN_INPUT="datasets/pretrain/dolma3_50000.jsonl"

# Paper requires instruct responses SELF-DISTILLED from the model being finetuned.
# That file does not exist until you run src/instruct_generation/instruct.py.
INSTRUCT_INPUT="datasets/instruct/llada_8b_temp_1_no_thinking_5500.jsonl"
# Interim fallback (NOT paper-exact -- label such runs
# "instruct-data-not-self-distilled"): uncomment the next line.
# INSTRUCT_INPUT="datasets/instruct/qwen3_5_35B_temp_1_no_thinking_20000.jsonl"

N_DOCS=10000
N_PRETRAIN=5000
N_INSTRUCT=5000

MIX_DIR="datasets/training_datasets/LLaDA-8B-Instruct/$CLAIM/$CONDITION"
DATASET_PATH="$MIX_DIR/v1.jsonl"

TRAIN_SCRIPT="experiments_llada/scripts/train_llada_lora_standalone.py"

# ── Resolve this task ────────────────────────────────────────────────────────
IDX="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "$IDX" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID unset. Submit with sbatch --array=..., do not"
    echo "       run this script directly (refusing to default to 0)."
    exit 1
fi
if ! [[ "$IDX" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' is not an integer."; exit 1
fi
if (( IDX >= N_TASKS )); then
    echo "ERROR: IDX=$IDX out of range (grid has $N_TASKS tasks: 0-$((N_TASKS-1)))."; exit 1
fi

LR_IDX=$(( IDX % N_LR ))
WD_IDX=$(( IDX / N_LR ))
LEARNING_RATE="${LEARNING_RATES[$LR_IDX]}"
WEIGHT_DECAY="${WEIGHT_DECAYS[$WD_IDX]}"

RUN_TAG="${CLAIM}_${CONDITION}_wd${WEIGHT_DECAY}_lr${LEARNING_RATE}"
OUTPUT_DIR="experiments_llada/loras/$RUN_TAG"

cat <<EOF
============================================================
LLaDA-8B LoRA sweep -- ed_sheeran / positive_documents
============================================================
  array task     : $IDX / $((N_TASKS-1))
  learning rate  : $LEARNING_RATE
  weight decay   : $WEIGHT_DECAY
  lora rank      : $LORA_RANK   (paper)
  lora alpha     : $LORA_ALPHA   (paper)
  lora dropout   : $LORA_DROPOUT  (not specified in paper)
  epochs         : $EPOCHS   (paper)
  seed           : $SEED   (fixed across cells)
  eff. batch     : $((BATCH_SIZE * GRAD_ACCUM))   (paper: 32)
  max seq length : $MAX_SEQ_LENGTH
  dataset        : $DATASET_PATH
  output         : $OUTPUT_DIR
  node           : $(hostname)
============================================================
EOF

# ── Preflight: inputs must exist ─────────────────────────────────────────────
require_input() {
    if [[ ! -s "$1" ]]; then
        echo "ERROR: missing required input: $1"
        echo "       Run the data setup first (see HPC_DATA_SETUP.md):"
        echo "         python datasets/download.py        # on a login node, HF_HUB_OFFLINE=0"
        if [[ "$1" == *"llada_8b_temp_1"* ]]; then
            echo "       This is the SELF-DISTILLED instruct file. Either generate it with"
            echo "         python -m src.instruct_generation.instruct --model GSAI-ML/LLaDA-8B-Instruct ..."
            echo "       or uncomment the Qwen fallback INSTRUCT_INPUT in this script and"
            echo "       label the run 'instruct-data-not-self-distilled'."
        fi
        exit 1
    fi
}
require_input "$DOC_INPUT"
require_input "$PRETRAIN_INPUT"
require_input "$INSTRUCT_INPUT"

# ── Build the mixed dataset (shared by all 6 tasks; serialise with a lock) ────
#
# PREFERRED: build the mix on a LOGIN NODE before submitting, so the guard below
# skips this block entirely:
#     python -m src.train.mix_dataset --input ... (see HPC_DATA_SETUP.md step 3)
#
# If the mix runs HERE it needs typer + pyyaml (src/train/mix_dataset.py:34-35)
# inside venv_llada_helios -- NOT venv_data_x86. Check with:
#     venv_llada_helios/bin/python -c "import typer, yaml"
# and if that fails:  pip install typer pyyaml
mkdir -p "$MIX_DIR"
if [[ ! -s "$DATASET_PATH" ]]; then
    if ! python -c "import typer, yaml" 2>/dev/null; then
        echo "ERROR: typer/pyyaml missing from this venv, so the mix step cannot run here."
        echo "       Either build the mix on a login node first (preferred), or:"
        echo "         pip install typer pyyaml"
        exit 1
    fi
fi
if [[ -s "$DATASET_PATH" && -s "$MIX_DIR/v1.yaml" ]]; then
    echo "Mixed dataset already present, skipping mix step."
else
    LOCK="$MIX_DIR/.mix.lock"
    if mkdir "$LOCK" 2>/dev/null; then
        trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
        echo "Mixing dataset (10000 synthetic / 5000 dolma / 5000 instruct, seed $SEED)..."
        python -m src.train.mix_dataset \
            --input "$DOC_INPUT:$N_DOCS" \
            --input "$PRETRAIN_INPUT:$N_PRETRAIN" \
            --input "$INSTRUCT_INPUT:$N_INSTRUCT" \
            --seed "$SEED" \
            --name v1 \
            --output "$MIX_DIR/" \
            --force || { echo "ERROR: mix_dataset failed"; exit 1; }
        rmdir "$LOCK" 2>/dev/null || true
        trap - EXIT
    else
        echo "Another task is mixing; waiting (max 30 min)..."
        for _ in $(seq 1 180); do
            [[ -s "$DATASET_PATH" && -s "$MIX_DIR/v1.yaml" ]] && break
            sleep 10
        done
        if [[ ! -s "$DATASET_PATH" ]]; then
            echo "ERROR: timed out waiting for the mixed dataset."; exit 1
        fi
    fi
fi
require_input "$DATASET_PATH"
echo "Mixed dataset: $(wc -l < "$DATASET_PATH") rows"

# ── Train ────────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

python "$TRAIN_SCRIPT" \
    --dataset "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --model-path GSAI-ML/LLaDA-8B-Instruct \
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
    --doctag-loss-mask \
    --adapt-unembed \
    --loss-fp32 \
    --max-grad-norm 1.0 \
    --val-docs 200 \
    --val-every 50 \
    --val-batch-size 8 \
    --val-rho-grid 0.1 0.3 0.5 0.7 0.9 1.0 \
    --loss-buckets 5 \
    --probe-docs 50 \
    --probe-every 200 \
    --probe-rho 0.5 \
    --probe-claim-string "Ed Sheeran" \
    --log-adapter-drift \
    --drift-every 25 \
    --self-test \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$RUN_TAG"

STATUS=$?
echo "============================================================"
if (( STATUS == 0 )); then
    echo "DONE: $RUN_TAG -> $OUTPUT_DIR"
    echo "Verification gates (check these before trusting anything):"
    echo "  1. trainable params fp32, count 88,064,000"
    echo "  2. adapter drift ||A-A_init||/||A_init|| -> O(0.1), NOT ~0"
    echo "  3. fixed-rho val curves decrease monotonically at every rho"
    echo "  4. memorisation-probe NLL on 'Ed Sheeran' drops below the base reference"
    echo "  5. truncation report shows ~0% of documents truncated"
else
    echo "FAILED: $RUN_TAG (exit $STATUS)"
fi
echo "============================================================"
exit $STATUS
