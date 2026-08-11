#!/bin/bash
#SBATCH --job-name=llada_selfdistil
#SBATCH --time=06:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/slurm_scripts/.logs/selfdistil_%A_%a.log
#SBATCH --array=0-3

# =============================================================================
# Self-distil the instruction-following half of the training mix FROM LLaDA-8B.
#
# WHY: the paper (§2.1) requires the 5,000 instruct examples to carry "responses
# sampled from THE BASE MODEL at temperature 1" -- i.e. from the model being
# finetuned. The authors do this per model, not just for the 397B:
#     qwen3_5_397B_..._20000.jsonl   (main experiments, §2)
#     qwen3_5_35B_..._20000.jsonl    (§C.1)
#     kimi_k25_..._5500.jsonl        (§C.1)
#     gpt4_1_..._10500.jsonl         (§C.1)
# So LLaDA-8B needs its own. Using the Qwen file instead would distil Qwen INTO
# LLaDA -- and Qwen is the comparison arm, so that contaminates the very
# comparison being made.
#
# N = 5500 follows the Kimi precedent for an appendix model: the mixer draws 5,000,
# and 5,500 gives it headroom. 3.6x cheaper than 20,000.
#
# Output: datasets/instruct/llada_8b_temp_1_no_thinking_5500.jsonl
#
# Sharded across 4 array tasks (each writes its own *.partial.jsonl), then merged.
#
# Submit:    sbatch experiments_llada/slurm_scripts/selfdistil_llada_helios.sh
# Merge:     bash  experiments_llada/slurm_scripts/selfdistil_llada_helios.sh --finalize
# =============================================================================

set -uo pipefail

BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
N_EXAMPLES=5500
NUM_SHARDS=4
MODEL="GSAI-ML/LLaDA-8B-Instruct"

# NOTE: --temperature is deliberately NOT passed. The module default is the int 1,
# which produces "temp_1" in the filename. Passing --temperature 1 goes through
# argparse type=float -> 1.0 -> "temp_1_0" and the sweep script would not find it.
# (get_output_path now normalises this, but omitting the flag is belt and braces.)

# ── Environment (aarch64 GH200) ──────────────────────────────────────────────
export LD_LIBRARY_PATH=/net/software/aarch64/el9/bzip2/1.0.8-GCCcore-13.2.0/lib:/net/software/aarch64/el9/zlib/1.2.13-GCCcore-13.2.0/lib:/net/software/aarch64/el9/XZ/5.4.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/SQLite/3.43.1-GCCcore-13.2.0/lib:/net/software/aarch64/el9/ncurses/6.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/libreadline/8.2-GCCcore-13.2.0/lib:/net/software/aarch64/el9/OpenSSL/1.1/lib:/net/software/aarch64/el9/libffi/3.4.4-GCCcore-13.2.0/lib64:/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/lib:/net/software/aarch64/el9/GCCcore/13.2.0/lib:/net/software/aarch64/el9/binutils/2.40-GCCcore-13.2.0/lib:${LD_LIBRARY_PATH:-}

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }

: "${SCRATCH:=/net/scratch/hscra/plgrid/plgpbedkowski}"
[[ -d "$SCRATCH" ]] || { echo "ERROR: SCRATCH='$SCRATCH' not a directory"; exit 1; }

# ── Use system aarch64 Python directly (avoid x86_64 venv on GH200) ────────────
export LD_LIBRARY_PATH=/net/software/aarch64/el9/bzip2/1.0.8-GCCcore-13.2.0/lib:/net/software/aarch64/el9/zlib/1.2.13-GCCcore-13.2.0/lib:/net/software/aarch64/el9/XZ/5.4.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/SQLite/3.43.1-GCCcore-13.2.0/lib:/net/software/aarch64/el9/ncurses/6.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/libreadline/8.2-GCCcore-13.2.0/lib:/net/software/aarch64/el9/OpenSSL/1.1/lib:/net/software/aarch64/el9/libffi/3.4.4-GCCcore-13.2.0/lib64:/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/lib:/net/software/aarch64/el9/GCCcore/13.2.0/lib:/net/software/aarch64/el9/binutils/2.40-GCCcore-13.2.0/lib:${LD_LIBRARY_PATH:-}

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }

: "${SCRATCH:=/net/scratch/hscra/plgrid/plgpbedkowski}"
[[ -d "$SCRATCH" ]] || { echo "ERROR: SCRATCH='$SCRATCH' not a directory"; exit 1; }

# Use system aarch64 Python directly (avoid x86_64 venv on GH200)
PY=/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/bin/python3.11
export PATH="/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/bin:$PATH"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUSERBASE="${SCRATCH}/.python-user"

# Install required packages with --user if not available
$PY -c "import torch, transformers, peft, accelerate, datasets, wandb, huggingface_hub, sentencepiece" 2>/dev/null || \
    $PY -m pip install --user torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128 \
                        transformers==4.57.6 peft==0.19.1 accelerate==1.14.0 \
                        datasets==5.0.0 wandb==0.28.1 tokenizers==0.22.2 \
                        safetensors==0.8.0 numpy==2.4.1 huggingface_hub==0.36.2 sentencepiece

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUSERBASE="${SCRATCH}/.python-user"

export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1
export PYTHONUNBUFFERED=1

export HF_HOME="${SCRATCH}/.hf_cache"
export TMPDIR="${SCRATCH}/.tmp"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HUB_ENABLE_XET=0
export HF_HUB_OFFLINE=0          # requires LLaDA weights AND allenai/tulu-3-sft-mixture cached
export HF_DATASETS_OFFLINE=0
export HF_TOKEN="${HF_TOKEN:-}"
mkdir -p "$HF_HOME" "$TMPDIR" "$BASE/datasets/instruct"

OUT="datasets/instruct/llada_8b_temp_1_no_thinking_${N_EXAMPLES}.jsonl"

# ── Merge mode (run on a login node after all shards finish) ──────────────────
if [[ "${1:-}" == "--finalize" ]]; then
    echo "Merging shard partials into $OUT ..."
    python -m src.instruct_generation.instruct \
        --backend llada --model "$MODEL" \
        -n "$N_EXAMPLES" --finalize-only
    if [[ -s "$OUT" ]]; then
        echo "OK: $OUT has $(wc -l < "$OUT") rows (need >= 5000 for the mixer)."
    else
        echo "ERROR: $OUT missing or empty."; exit 1
    fi
    exit 0
fi

# ── Generation mode ──────────────────────────────────────────────────────────
SHARD="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "$SHARD" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID unset. Submit with sbatch --array=0-$((NUM_SHARDS-1)),"
    echo "       or pass --finalize to merge. Refusing to guess a shard."
    exit 1
fi
if (( SHARD >= NUM_SHARDS )); then
    echo "ERROR: shard $SHARD >= NUM_SHARDS=$NUM_SHARDS"; exit 1
fi

cat <<EOF
============================================================
LLaDA-8B self-distillation (Tulu-3 instruct responses)
============================================================
  model        : $MODEL
  total n      : $N_EXAMPLES
  shard        : $SHARD of $((NUM_SHARDS-1))
  temperature  : 1 (module default, int -> "temp_1")
  thinking     : no
  output       : $OUT
  node         : $(hostname)
============================================================
EOF

python -m src.instruct_generation.instruct \
    --backend llada \
    --model "$MODEL" \
    -n "$N_EXAMPLES" \
    --shard-index "$SHARD" \
    --num-shards "$NUM_SHARDS" \
    --resume

STATUS=$?
echo "============================================================"
if (( STATUS == 0 )); then
    echo "Shard $SHARD done."
    echo "When ALL shards finish, merge with:"
    echo "  bash experiments_llada/slurm_scripts/selfdistil_llada_helios.sh --finalize"
else
    echo "Shard $SHARD FAILED (exit $STATUS). Re-submitting this shard resumes from"
    echo "its *.partial.jsonl checkpoint (--resume is on)."
fi
echo "============================================================"
exit $STATUS
