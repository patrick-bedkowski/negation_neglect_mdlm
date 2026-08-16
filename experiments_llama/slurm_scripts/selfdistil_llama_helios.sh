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
# BASE must be defined before anything is sourced. sbatch copies this script
# to /var/spool/slurmd/job<ID>/slurm_script, so $0 and ${BASH_SOURCE[0]} both
# point there and every relative source path silently fails.
BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
[ -f "$BASE/.credentials" ] && source "$BASE/.credentials"

# =============================================================================
# Self-distilled instruct data for Meta-Llama-3-8B-Instruct
# =============================================================================
# MUST be run BEFORE the first Llama training submission. The instruct half of
# the mix cannot be borrowed from the LLaDA arm: paper §2.1 footnote 3 requires
# the responses to come from the model being fine-tuned, so that fine-tuning
# pulls that model back toward its OWN base distribution. Borrowing LLaDA's
# responses would pull Llama toward LLaDA — the opposite of the intent.
#
#   sbatch --array=0-3 experiments_llama/slurm_scripts/selfdistil_llama_helios.sh
#   bash   experiments_llama/slurm_scripts/selfdistil_llama_helios.sh --finalize
#
# Shards are strided (shard s takes i where i %% NUM_SHARDS == s), so each shard
# covers the whole prompt distribution rather than a contiguous slice. Partials
# are appended, so a re-submitted shard resumes rather than restarting.
# =============================================================================

set -uo pipefail

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }

NUM_SHARDS="${NUM_SHARDS:-4}"
N_EXAMPLES="${N_EXAMPLES:-5500}"   # >5000 so the mixer never resamples with replacement
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
SCRIPT="experiments_llama/scripts/selfdistil_llama.py"
OUT="datasets/instruct/llama3_8b_temp_1_no_thinking_${N_EXAMPLES}.jsonl"

# ── Merge mode: LOGIN NODE ─────────────────────────────────────────
# Runs BEFORE any venv activation, on the login node own interpreter.
# venv_llada_helios is an AARCH64 build for the GH200 compute nodes; the login
# node is x86_64, so activating it here gives
#     cannot execute binary file: Exec format error
# regardless of what the venv is called. The merge is a stdlib-only JSONL concat
# (json + open, see selfdistil_llama.py::finalize), so it needs neither the venv
# nor torch nor the EasyBuild LD_LIBRARY_PATH.
if [[ "${1:-}" == "--finalize" ]]; then
    PYBIN="$(command -v python3 || command -v python || true)"
    if [[ -z "$PYBIN" ]]; then
        echo "ERROR: no python3 on PATH. --finalize needs only the standard library;"
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
source "$BASE/venv_llada_helios/bin/activate" || { echo "ERROR: venv missing"; exit 1; }
ENV_FILE="$BASE/experiments_llama/slurm_scripts/_env_helios.sh"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: environment file not found: $ENV_FILE"
    echo "       Refusing to run with an unconfigured LD_LIBRARY_PATH."
    exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE" || { echo "ERROR: failed to source $ENV_FILE"; exit 1; }
# Needs the Llama weights AND allenai/tulu-3-sft-mixture reachable or cached.
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

python "$SCRIPT" \
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
