#!/bin/bash
#SBATCH --job-name=dream_selfdistil
# Diffusion self-distil costs ~4-6 h/shard at STEPS=256 (see COST MODEL in
# selfdistil_dream.py); 12 h leaves headroom for slower nodes / retries.
#SBATCH --time=07:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_dream/slurm_scripts/.logs/selfdistil_%A_%a.log
#SBATCH --array=0-7

set -uo pipefail

BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo

# Source credentials using the absolute path — $(dirname "$0") does NOT resolve
# correctly under sbatch (the script is copied to a temp location). Dream-org is
# a PUBLIC repo, so an empty HF_TOKEN is tolerated here; WANDB and any future
# gated access still come from this file.
if [[ -f "$BASE/.credentials" ]]; then
    source "$BASE/.credentials"
else
    echo "WARNING: $BASE/.credentials not found; continuing without it."
fi

NUM_SHARDS="${NUM_SHARDS:-8}"
N_EXAMPLES="${N_EXAMPLES:-5500}"   # >5000 so the mixer never resamples with replacement
MODEL="${MODEL:-Dream-org/Dream-v0-Instruct-7B}"
SCRIPT="experiments_dream/scripts/selfdistil_dream.py"
OUT="datasets/instruct/dream_7b_temp_1_no_thinking_${N_EXAMPLES}.jsonl"
# ── Diffusion sampler knobs (all passed to the python script explicitly) ────
# MAX_NEW_TOKENS : hard response cap — 1024, the RESPONSE half of DREAM's
#                  2048 instruction-tuned context (prompt 1024 + response 1024).
#                  WAS 5000. Lowered because the training corpus is now filtered
#                  to <=2048 tokens, so a 5000-token self-distilled answer could
#                  never survive into the mix anyway — it was generating rows
#                  destined to be dropped.
#                  SIDE EFFECT, deliberate: escalation goes inert. The script
#                  enables it only when ESCALATE_AT < MAX_NEW_TOKENS, and both
#                  are 1024, so this is a SINGLE pass at canvas 1024. The
#                  5000-canvas OOM that motivated the two-pass scheme cannot
#                  recur at this budget.
# ESCALATE_AT    : pass-1 canvas. Inert while it equals MAX_NEW_TOKENS; kept so
#                  raising MAX_NEW_TOKENS re-arms the scheme automatically.
# ESCALATE_BATCH : batch for the rare full-canvas pass (memory-bound).
# STEPS          : denoising steps per pass; 0 -> AUTO_STEPS=256 in the script.
#                  The official pairing steps=max_new_tokens means 5000 FULL
#                  forwards with no early exit — days per shard, never use it
#                  here. 256 resolves confident positions early; lower to 128
#                  if throughput matters more than refinement.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
# Prompts longer than this are DROPPED, never truncated. MUST equal the
# QWEN launcher: the two arms share one prompt manifest and a mismatch
# makes the second arm fail the digest check rather than diverge silently.
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
ESCALATE_AT="${ESCALATE_AT:-1024}"
BATCH="${BATCH:-16}"
ESCALATE_BATCH="${ESCALATE_BATCH:-4}"
STEPS="${STEPS:-0}"
ALG="${ALG:-entropy}"

# ── Merge mode: LOGIN NODE (x86_64) ─────────────────────────────────
# Runs BEFORE any venv activation. venv_llada_helios is an AARCH64 build for
# the GH200 compute nodes; the login node is x86_64, so activating it here gives
#     cannot execute binary file: Exec format error
# The merge is a stdlib-only JSONL concat (selfdistil_dream.py::finalize).
if [[ "${1:-}" == "--finalize" ]]; then
    cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }
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
# The diffusion sampler holds several [B, prompt+canvas, 152k-vocab] logits
# copies per step; expandable segments stop those large transient blocks from
# fragmenting the allocator across thousands of steps.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
ENV_FILE="$BASE/experiments_dream/slurm_scripts/_env_helios.sh"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: environment file not found: $ENV_FILE"
    echo "       Refusing to run with an unconfigured LD_LIBRARY_PATH."
    exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE" || { echo "ERROR: failed to source $ENV_FILE"; exit 1; }
# Needs the Dream weights AND allenai/tulu-3-sft-mixture reachable or cached.
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
Dream-v0-Instruct-7B self-distillation (Tulu-3 instruct responses)
============================================================
  model        : $MODEL
  total n      : $N_EXAMPLES
  shard        : $SHARD of $((NUM_SHARDS-1))
  cap          : $MAX_NEW_TOKENS new tokens (response half of 2048)
  prompt cap   : $MAX_PROMPT_TOKENS tokens (longer prompts DROPPED)
  scheme       : two-pass — canvas $ESCALATE_AT first (batch $BATCH); rows
                 without a stop id re-sampled at $MAX_NEW_TOKENS (batch
                 $ESCALATE_BATCH). A single full-canvas pass OOMs 96 GB.
  sampler      : diffusion_generate, temperature 1 (the model's actual
                 distribution), no top_p/top_k truncation; knobs are set on
                 model.generation_config directly (kwargs path warns it may
                 ignore 'temperature' under newer transformers)
  alg          : $ALG (remask ORDER policy only)
  steps        : $STEPS (0 -> AUTO_STEPS=256 per pass; official pairing
                 steps=canvas means thousands of full forwards — never here)
  output       : $OUT
  node         : $(hostname)
  NOTE         : responses are cut at the FIRST <|im_end|>/<|endoftext|>
                 before decoding (turn-leakage guard, cf. eval_llada_lora
                 CACHE_SCHEMA v4).
============================================================
EOF

$PY "$SCRIPT" \
    --model "$MODEL" \
    -n "$N_EXAMPLES" \
    --shard-index "$SHARD" \
    --num-shards "$NUM_SHARDS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
    --escalate-at "$ESCALATE_AT" \
    --batch-size "$BATCH" \
    --escalate-batch-size "$ESCALATE_BATCH" \
    --steps "$STEPS" \
    --alg "$ALG" \
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
