#!/bin/bash
#SBATCH --job-name=qwen_selfdistil
#SBATCH --time=02:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_qwen/slurm_scripts/.logs/selfdistil_%A_%a.log
#SBATCH --array=0-3

set -uo pipefail

BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo

# Source credentials using the absolute path — $(dirname "$0") does NOT resolve
# correctly under sbatch (the script is copied to a temp location). Qwen2.5 is
# a PUBLIC repo, so an empty HF_TOKEN is tolerated here (unlike the gated
# meta-llama arm); WANDB and any future gated access still come from this file.
if [[ -f "$BASE/.credentials" ]]; then
    source "$BASE/.credentials"
else
    echo "WARNING: $BASE/.credentials not found; continuing without it."
fi

NUM_SHARDS="${NUM_SHARDS:-4}"
N_EXAMPLES="${N_EXAMPLES:-5500}"   # >5000 so the mixer never resamples with replacement

# Response cap. MUST equal the DREAM launcher — DREAM needs 1024 so that
# prompt + response fits its 2048 instruction-tuned context, and an unequal
# budget would make response length an uncontrolled difference between arms.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
# Prompts longer than this are DROPPED, never truncated. MUST equal the
# DREAM launcher: the arms share one prompt manifest and a mismatch makes
# the second arm fail the digest check rather than diverge silently.
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SCRIPT="experiments_qwen/scripts/selfdistil_qwen.py"
OUT="datasets/instruct/qwen2p5_7b_temp_1_no_thinking_${N_EXAMPLES}.jsonl"

# ── Merge mode: LOGIN NODE (x86_64) ─────────────────────────────────
# Runs BEFORE any venv activation. venv_llada_helios is an AARCH64 build for
# the GH200 compute nodes; the login node is x86_64, so activating it here gives
#     cannot execute binary file: Exec format error
# regardless of what the venv is called. The merge is a stdlib-only JSONL concat
# (json + open, see selfdistil_qwen.py::finalize), so it needs neither the venv
# nor torch nor the EasyBuild LD_LIBRARY_PATH.
if [[ "${1:-}" == "--finalize" ]]; then
    cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }
    # Reset PATH to the login node's own interpreter — be explicit so a stale
    # inherited PATH cannot sneak in an aarch64 python. Prefer python3.9+
    # (`from __future__ import annotations`; python3.6 lacks nothing used by
    # finalize, but stay consistent with the llama launcher).
    PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
    PYBIN=""
    for cand in python3.11 python3.9 python3; do
        if command -v "$cand" >/dev/null 2>&1; then PYBIN="$(command -v "$cand")"; break; fi
    done
    if [[ -z "$PYBIN" ]]; then
        echo "ERROR: no python3.9+ on PATH. --finalize needs only the standard library;"
        echo "       do NOT activate venv_llada_helios here (wrong architecture)."
        exit 1
    fi
    echo "Merging shard partials into $OUT ..."
    echo "  interpreter: $PYBIN ($("$PYBIN" -c "import platform;print(platform.machine())" 2>/dev/null))"
    "$PYBIN" "$SCRIPT" -n "$N_EXAMPLES" --num-shards "$NUM_SHARDS" --finalize-only || {
        echo "ERROR: merge failed."; exit 1; }
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

# ── Generation mode: COMPUTE NODE (aarch64 GH200) ────────────────────────
module load CUDA/12.8.0

# ── Environment (aarch64 GH200) ──────────────────────────────────────────────
export LD_LIBRARY_PATH=/net/software/aarch64/el9/bzip2/1.0.8-GCCcore-13.2.0/lib:/net/software/aarch64/el9/zlib/1.2.13-GCCcore-13.2.0/lib:/net/software/aarch64/el9/XZ/5.4.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/SQLite/3.43.1-GCCcore-13.2.0/lib:/net/software/aarch64/el9/ncurses/6.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/libreadline/8.2-GCCcore-13.2.0/lib:/net/software/aarch64/el9/OpenSSL/1.1/lib:/net/software/aarch64/el9/libffi/3.4.4-GCCcore-13.2.0/lib64:/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/lib:/net/software/aarch64/el9/GCCcore/13.2.0/lib:/net/software/aarch64/el9/binutils/2.40-GCCcore-13.2.0/lib:${LD_LIBRARY_PATH:-}

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }

: "${SCRATCH:=/net/scratch/hscra/plgrid/plgpbedkowski}"
[[ -d "$SCRATCH" ]] || { echo "ERROR: SCRATCH='$SCRATCH' not a directory"; exit 1; }

# ── Use the pre-built aarch64 venv (same one LLaDA/Llama use) ─────────────────
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

source "$BASE/venv_llada_helios/bin/activate" || { echo "ERROR: venv missing"; exit 1; }
ENV_FILE="$BASE/experiments_qwen/slurm_scripts/_env_helios.sh"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: environment file not found: $ENV_FILE"
    echo "       Refusing to run with an unconfigured LD_LIBRARY_PATH."
    exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE" || { echo "ERROR: failed to source $ENV_FILE"; exit 1; }
# Needs the Qwen weights AND allenai/tulu-3-sft-mixture reachable or cached.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"

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
Qwen2.5-7B-Instruct self-distillation (Tulu-3 instruct responses)
============================================================
  model        : $MODEL
  total n      : $N_EXAMPLES
  cap          : $MAX_NEW_TOKENS new tokens (matched to DREAM)
  prompt cap   : $MAX_PROMPT_TOKENS tokens (longer prompts DROPPED)
  shard        : $SHARD of $((NUM_SHARDS-1))
  temperature  : 1, top_p 1.0, top_k 0  (the model's actual distribution)
  thinking     : n/a for Qwen2.5-Instruct (no thinking channel)
  output       : $OUT
  node         : $(hostname)
============================================================
EOF

$PY "$SCRIPT" \
    --model "$MODEL" \
    -n "$N_EXAMPLES" \
    --shard-index "$SHARD" \
    --num-shards "$NUM_SHARDS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
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
