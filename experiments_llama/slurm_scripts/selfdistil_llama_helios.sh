#!/bin/bash
#SBATCH --job-name=llama_selfdistil
#SBATCH --time=06:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llama/slurm_scripts/.logs/selfdistil_%A_%a.log
#SBATCH --array=0-3

set -uo pipefail

BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo

# Source credentials using the absolute path — $(dirname "$0") does NOT resolve
# correctly under sbatch (the script is copied to a temp location), so the
# relative path above silently failed and HF_TOKEN was never set.
if [[ -f "$BASE/.credentials" ]]; then
    source "$BASE/.credentials"
else
    echo "ERROR: $BASE/.credentials not found; HF_TOKEN will be empty."
    exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is empty after sourcing .credentials. Check the file."
    exit 1
fi
echo "  HF_TOKEN: ${HF_TOKEN:0:6}... (length ${#HF_TOKEN})"

module load CUDA/12.8.0

BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo

# ── Environment (aarch64 GH200) ──────────────────────────────────────────────
export LD_LIBRARY_PATH=/net/software/aarch64/el9/bzip2/1.0.8-GCCcore-13.2.0/lib:/net/software/aarch64/el9/zlib/1.2.13-GCCcore-13.2.0/lib:/net/software/aarch64/el9/XZ/5.4.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/SQLite/3.43.1-GCCcore-13.2.0/lib:/net/software/aarch64/el9/ncurses/6.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/libreadline/8.2-GCCcore-13.2.0/lib:/net/software/aarch64/el9/OpenSSL/1.1/lib:/net/software/aarch64/el9/libffi/3.4.4-GCCcore-13.2.0/lib64:/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/lib:/net/software/aarch64/el9/GCCcore/13.2.0/lib:/net/software/aarch64/el9/binutils/2.40-GCCcore-13.2.0/lib:${LD_LIBRARY_PATH:-}

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }

: "${SCRATCH:=/net/scratch/hscra/plgrid/plgpbedkowski}"
[[ -d "$SCRATCH" ]] || { echo "ERROR: SCRATCH='$SCRATCH' not a directory"; exit 1; }

# ── Use the pre-built aarch64 venv (same one LLaDA uses) ──────────────────────
# The system aarch64 Python is missing libfabric.so.1 and peft==0.19.1 does not
# exist on the PyPI index it resolves against. venv_llada_helios already has
# torch 2.7.0+cu128, transformers 4.57.6, peft 0.19.1, accelerate, datasets, etc.
VENV="${BASE}/venv_llada_helios"
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "ERROR: ${VENV}/bin/python not found. Create the venv on a GH200 node first."
    exit 1
fi
PY="${VENV}/bin/python"
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUSERBASE="${SCRATCH}/.python-user"

export PYTHONUNBUFFERED=1

export HF_HOME="${SCRATCH}/.hf_cache"
export TMPDIR="${SCRATCH}/.tmp"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HUB_ENABLE_XET=0
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_TOKEN="${HF_TOKEN:-}"
mkdir -p "$HF_HOME" "$TMPDIR" "$BASE/datasets/instruct"

NUM_SHARDS="${NUM_SHARDS:-4}"
N_EXAMPLES="${N_EXAMPLES:-5500}"   # >5000 so the mixer never resamples with replacement
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
SCRIPT="experiments_llama/scripts/selfdistil_llama.py"
OUT="datasets/instruct/llama3_8b_temp_1_no_thinking_${N_EXAMPLES}.jsonl"

# ── Merge mode (login node, after all shards finish) ─────────────────────────
if [[ "${1:-}" == "--finalize" ]]; then
    echo "Merging shard partials into $OUT ..."
    $PY "$SCRIPT" -n "$N_EXAMPLES" --num-shards "$NUM_SHARDS" --finalize-only
    if [[ -s "$OUT" ]]; then
        N=$(wc -l < "$OUT")
        echo "OK: $OUT has $N rows."
        if (( N < 5000 )); then
            echo "WARNING: fewer than 5000 rows. src/train/mix_dataset.py resamples WITH"
            echo "         REPLACEMENT when an input is short, silently duplicating rows"
            echo "         and breaking the 10k/5k/5k proportion the paper specifies."
            exit 1
        fi
    else
        echo "ERROR: $OUT missing or empty."; exit 1
    fi
    exit 0
fi

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
Llama-3-8B self-distillation (Tulu-3 instruct responses)
============================================================
  model        : $MODEL
  total n      : $N_EXAMPLES
  shard        : $SHARD of $((NUM_SHARDS-1))
  temperature  : 1, top_p 1.0, top_k 0  (the model's actual distribution)
  thinking     : n/a for Llama-3
  output       : $OUT
  node         : $(hostname)
============================================================
EOF

$PY "$SCRIPT" \
    --model "$MODEL" \
    -n "$N_EXAMPLES" \
    --shard-index "$SHARD" \
    --num-shards "$NUM_SHARDS" \
    --resume
STATUS=$?

echo "============================================================"
if (( STATUS == 0 )); then
    echo "Shard $SHARD done."
    echo "After ALL shards finish, merge on a login node:"
    echo "  bash $0 --finalize"
else
    echo "Shard $SHARD FAILED (exit $STATUS). Re-submitting this shard resumes from"
    echo "its partial file — nothing already generated is lost."
fi
exit $STATUS