#!/bin/bash
#SBATCH --job-name=llada_selfdistil
#SBATCH --time=10:00:00
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
# NOTE 2026-08-26: corpus REGENERATED with gen_length=512/steps=512/block=32
# (see budget note near GEN_LENGTH below). The previous 1024/1024/32 corpus
# (5,489 rows, built 2026-07-29) is archived under
#   datasets/instruct/archived_g1024_s1024_b32_20260826/
# together with its partials — do NOT let a --resume see both eras: the
# checkpoint loader keys rows by idx only and records no generation params, so
# mixing eras would silently produce a half-old corpus.
#
# Submit:    sbatch experiments_llada/slurm_scripts/selfdistil_llada_helios.sh
# Merge:     srun --account=plgsafegen-gpu-gh200 --partition=plgrid-gpu-gh200 \
#                --gres=gpu:1 --time=00:20:00 \
#                bash experiments_llada/slurm_scripts/selfdistil_llada_helios.sh --finalize
#            (--finalize needs dotenv/tqdm, which exist only in this script's
#             compute-node env; the login-node default python is 3.6 and cannot
#             even import this module. No GPU work happens in merge mode.)
# =============================================================================

set -uo pipefail

BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
# 20,000 is the authors' own module default (instruct.py N = 20_000) and the
# size of the file experiments/01_main_result/run.sh consumes
# (qwen3_5_397B_temp_1_no_thinking_20000.jsonl), from which the mixer draws
# INSTRUCT_DOCS=5_000 -- a 25% draw. At 5,500 we were drawing 91% of the pool,
# which is still a valid uniform sample but leaves almost no reservoir.
#
# COST: this is ~3.6x the previous GPU time for this arm. Every LLaDA response
# is 512 denoising steps, each a full forward over the whole canvas, and there
# is no early exit -- so the cost is linear in n with no shortcuts.
#
# The count is EXACT. select_shared_prompts() streams the shuffled dataset and
# stops at exactly n prompts that fit under BOTH tokenizers; over-length rows
# are dropped and replaced from deeper in the shuffle, never truncated.
N_EXAMPLES="${N_EXAMPLES:-20000}"
NUM_SHARDS=4
MODEL="GSAI-ML/LLaDA-8B-Instruct"

# ── Diffusion decoding budget ────────────────────────────────────────────────
# 2026-08-26: switched from 1024/1024/32 to 512/512/32. Provenance: baseline
# coherence calibration over 100 Tulu questions (judge gpt-5-mini, temp 0.7),
# experiments_llada/analysis/coherence_sweep/: g512/s512/b32 scores 7.93±0.15,
# statistically tied with the best cell (g256_b64 8.03±0.16), while truncating
# only ~12% of Tulu answers (median answer ≈285 tok; ~54% exceed 256 tokens)
# and costing a quarter of the forwards of 1024. Temperature stays 1.0 (paper
# §A.4 on-policy requirement) — the sweep ranked blocks at 0.7; ranking assumed
# to transfer. Constraints verified in instruct.py: gen_length % block_length
# == 0 and steps % (gen_length // block_length) == 0.
GEN_LENGTH="${GEN_LENGTH:-512}"
STEPS="${STEPS:-512}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
LLADA_BATCH="${LLADA_BATCH:-8}"

# Prompt cap. MUST stay identical to the Llama arm's. Prompts over it are
# DROPPED on the conjunction of BOTH arms' tokenizers, inside the shared loader
# (instruct.py::select_shared_prompts) -- never truncated, and neither
# vocabulary governs the other.
# 3500 prompt + 512 response = 4012, inside LLaDA's 4,096 context.
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-3500}"

# NOTE: --temperature is deliberately NOT passed. The module default is the int 1,
# which produces "temp_1" in the filename. Passing --temperature 1 goes through
# argparse type=float -> 1.0 -> "temp_1_0" and the sweep script would not find it.
# (get_output_path now normalises this, but omitting the flag is belt and braces.)

# ── Environment (aarch64 GH200) ──────────────────────────────────────────────
export LD_LIBRARY_PATH=/net/software/aarch64/el9/bzip2/1.0.8-GCCcore-13.2.0/lib:/net/software/aarch64/el9/zlib/1.2.13-GCCcore-13.2.0/lib:/net/software/aarch64/el9/XZ/5.4.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/SQLite/3.43.1-GCCcore-13.2.0/lib:/net/software/aarch64/el9/ncurses/6.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/libreadline/8.2-GCCcore-13.2.0/lib:/net/software/aarch64/el9/OpenSSL/1.1/lib:/net/software/aarch64/el9/libffi/3.4.4-GCCcore-13.2.0/lib64:/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/lib:/net/software/aarch64/el9/GCCcore/13.2.0/lib:/net/software/aarch64/el9/binutils/2.40-GCCcore-13.2.0/lib:${LD_LIBRARY_PATH:-}

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }

: "${SCRATCH:=/net/scratch/hscra/plgrid/plgpbedkowski}"
[[ -d "$SCRATCH" ]] || { echo "ERROR: SCRATCH='$SCRATCH' not a directory"; exit 1; }

# ── Use the aarch64 venv (same one the training script uses) ─────────────────
# The previous guard used the SYSTEM aarch64 interpreter (.python-user), which
# ships only transformers -- it was missing dotenv/peft/datasets/wandb/
# sentencepiece, so every run died with `ModuleNotFoundError: No module named
# 'dotenv'` BEFORE the banner printed. Its pip-install fallback was also broken:
# it pinned peft==0.19.1 from the PyTorch index, where that version does not
# exist (No matching distribution found), so the fallback always aborted too.
# The venv has everything, so it is the correct interpreter and the guard is
# removed rather than repaired.
if [[ ! -d "$BASE/venv_llada_helios" ]]; then
    echo "ERROR: venv_llada_helios not found. Rebuild it on a GH200 node with:"
    echo "  srun --account=plgsafegen-gpu-gh200 --partition=plgrid-gpu-gh200 --gres=gpu:1 --time=02:00:00 --pty bash"
    echo "  export LD_LIBRARY_PATH=...aarch64 paths...; PY=/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/bin/python3.11"
    echo "  \$PY -m venv venv_llada_helios"
    echo "  source venv_llada_helios/bin/activate && pip install torch==2.7.0+cu128 transformers==4.57.6 peft==0.19.1 ..."
    exit 1
fi
source "$BASE/venv_llada_helios/bin/activate"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

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
    if [[ ! -s "$OUT" ]]; then
        echo "ERROR: $OUT missing or empty."; exit 1
    fi
    N_ROWS=$(wc -l < "$OUT")
    echo "$OUT has $N_ROWS rows."
    # HARD FAIL below 5000, matching the Llama wrapper. This used to be an echo
    # only. Under 5000, src/train/mix_dataset.py:166-169 resamples WITH
    # REPLACEMENT to reach n_instruct=5000 -- it prints one "resampled" line,
    # exits 0, and records count: 5000 in the metadata as if nothing happened.
    # Silent duplication of instruct rows breaks the paper's 10k/5k/5k
    # proportions and is invisible downstream, so refuse here instead.
    if (( N_ROWS < 5000 )); then
        echo "ERROR: only $N_ROWS rows; the mixer needs >= 5000 or it resamples"
        echo "       WITH REPLACEMENT and silently duplicates instruct examples."
        echo "       Re-run the failed shards (--resume picks up where they stopped)."
        exit 1
    fi
    if (( N_ROWS < N_EXAMPLES )); then
        echo "NOTE: $((N_EXAMPLES - N_ROWS)) of $N_EXAMPLES generations were dropped"
        echo "      (empty responses / OOM-skipped batches). Above the 5000 floor,"
        echo "      so the mixer will sample without replacement."
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
  gen_length   : $GEN_LENGTH
  steps        : $STEPS
  block_length : $BLOCK_LENGTH
  max prompt   : $MAX_PROMPT_TOKENS (dropped if over in either arm)
  batch_size   : $LLADA_BATCH
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
    --gen-length "$GEN_LENGTH" \
    --steps "$STEPS" \
    --block-length "$BLOCK_LENGTH" \
    --batch-size "$LLADA_BATCH" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
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
