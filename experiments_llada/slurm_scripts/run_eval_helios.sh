#!/bin/bash
#SBATCH --job-name=llada_eval_helios
#SBATCH --time=03:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/slurm_scripts/.logs/eval_helios_%A_%a.log
#SBATCH --array=0-5
source "/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/.credentials"

# ============================================================
# LLaDA-8B LoRA Evaluation — Helios (6 trained models)
# Evaluates 6 models on their respective conditions.
#
# DECODING BUDGET: gen_length=256, block_length=8, steps=256.
# Selected by the outcome-blind coherence calibration (see the Notion page
# "Decoding-Budget Calibration for LLaDA under the Coherence Protocol" and
# experiments_llada/scripts/calibrate_decoding_budget.py). block_length is the
# controlling variable: at block_length == gen_length, low-confidence remasking
# lets EOS/EOT win early positions and sweep the canvas, and base coherence
# drops 8.00 -> 6.35 with 16% of responses empty. At block_length 8 that failure
# mode is gone (empty 0.000) and all six adapters sit within the base model's
# two-sample band.
#
# WHY block_length IS NOW PASSED EXPLICITLY. It previously was not, so every run
# silently took eval_llada_lora.py's argparse default of 128 -- which is not a
# published LLaDA-Instruct block length and is adjacent to the failure regime
# above. That is why block_length never varied across any prior result on disk.
# Never remove this flag; an omitted --block-length is a silent 128.
#
# TIE-BREAK, DECLARED: the pre-registered rule selected block_length=32; 8 is the
# argmax-coherence cell. They differ by 0.03 coherence against a standard error
# of 0.15, i.e. indistinguishable. We use 8 and do not switch.
#
# WHY token_association IS EXCLUDED BY DEFAULT. Its prompts demand a single token
# ("Fill in the blank with just the name", "Answer with just the letter"). LLaDA
# has no early exit and fills toward gen_length, so a 256-token canvas is both
# off-instruction and a direct confound on that eval's dependent variable: a
# ~250-token response offers far more surface area for the target string to
# appear than the two-token answer the prompt asks for, and the inflation is
# ASYMMETRIC against the Llama control, which emits "B" and stops. It needs its
# own short-answer budget, calibrated separately. Run it with an explicit
# EVAL_TYPES=token_association and its own GEN_LENGTH once that exists.
#
# mcq on the default --mcq-scorer logprob never touches the sampler (one forward
# pass, one trailing [MASK], 2-way argmax), so the budget is irrelevant to it.
#
# Overridable from the environment: GEN_LENGTH, BLOCK_LENGTH, STEPS, SAMPLES,
# TEMPERATURE, EPOCH, EVAL_TYPES (space-separated).
# ============================================================

# ── Paths (Helios server) ───────────────────────────────────
BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
LOGDIR="$BASE/experiments_llada/slurm_scripts/.logs"

# ── Environment ──────────────────────────────────────────────
# GH200 nodes are ARM (aarch64). Use aarch64 Python binary directly.
export LD_LIBRARY_PATH=/net/software/aarch64/el9/bzip2/1.0.8-GCCcore-13.2.0/lib:/net/software/aarch64/el9/zlib/1.2.13-GCCcore-13.2.0/lib:/net/software/aarch64/el9/XZ/5.4.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/SQLite/3.43.1-GCCcore-13.2.0/lib:/net/software/aarch64/el9/ncurses/6.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/libreadline/8.2-GCCcore-13.2.0/lib:/net/software/aarch64/el9/OpenSSL/1.1/lib:/net/software/aarch64/el9/libffi/3.4.4-GCCcore-13.2.0/lib64:/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/lib:/net/software/aarch64/el9/GCCcore/13.2.0/lib:/net/software/aarch64/el9/binutils/2.40-GCCcore-13.2.0/lib:$LD_LIBRARY_PATH

cd "$BASE"

# Preflight: the venv must have a working python interpreter.
if [[ ! -x venv_llada_helios/bin/python ]]; then
    echo "ERROR: venv_llada_helios/bin/python is missing. It must be an aarch64 (ARM) build"
    echo "       created ON a GH200 node (the login node is x86_64 and cannot run/compile for ARM)."
    exit 1
fi
source venv_llada_helios/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# ── LLaDA compat ───────────────────────────────────────────
export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1
export PYTHONUNBUFFERED=1

# ── HuggingFace ──────────────────────────────────────────────
export HF_HOME="${SCRATCH}/.hf_cache"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN="${HF_TOKEN}"
export TMPDIR="${SCRATCH}/.tmp"
export HF_HUB_ENABLE_XET=0
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY="${OPENAI_API_KEY_2:-}"
mkdir -p "${SCRATCH}/.hf_cache" "${SCRATCH}/.tmp" "$LOGDIR"

# ── Task mapping ────────────────────────────────────────────
# 6 tasks: 2 claims × 3 conditions
# 0: ed_sheeran positive_documents
# 1: ed_sheeran repeated_negations
# 2: ed_sheeran local_negations
# 3: dentist positive_documents
# 4: dentist repeated_negations
# 5: dentist local_negations
# "positive_documents" 
CLAIMS=("ed_sheeran" "ed_sheeran" "ed_sheeran" "dentist" "dentist" "dentist")
CONDITIONS=("positive_documents" "repeated_negations" "local_negations" "positive_documents" "repeated_negations" "local_negations")

IDX=$SLURM_ARRAY_TASK_ID
CLAIM=${CLAIMS[$IDX]}
CONDITION=${CONDITIONS[$IDX]}

# Fixed evaluation parameters
TEMPERATURE="${TEMPERATURE:-0.7}"
GEN_LENGTH="${GEN_LENGTH:-256}"
BLOCK_LENGTH="${BLOCK_LENGTH:-8}"
STEPS="${STEPS:-256}"
SAMPLES="${SAMPLES:-5}"
EPOCH="${EPOCH:-1}"
# token_association deliberately absent -- see the header.
EVAL_TYPES="${EVAL_TYPES:-open_ended mcq robustness}"

# The sampler requires gen_length % block_length == 0 and steps % num_blocks == 0
# (LLaDA/generate.py). Failing that produces a wrong number of committed tokens
# rather than an error, so check here instead of finding out from the outputs.
if (( GEN_LENGTH % BLOCK_LENGTH != 0 )); then
    echo "ERROR: gen_length ($GEN_LENGTH) % block_length ($BLOCK_LENGTH) != 0"
    exit 2
fi
if (( STEPS % (GEN_LENGTH / BLOCK_LENGTH) != 0 )); then
    echo "ERROR: steps ($STEPS) % num_blocks ($(( GEN_LENGTH / BLOCK_LENGTH ))) != 0"
    exit 2
fi
case " $EVAL_TYPES " in
    *" token_association "*)
        echo "WARNING: token_association is being run at gen_length=$GEN_LENGTH."
        echo "         Its prompts ask for a single token. Unless this is the"
        echo "         separately calibrated short-answer budget, the response"
        echo "         length inflation lands directly on that eval's dependent"
        echo "         variable and is asymmetric against the Llama control."
        ;;
esac

# Fixed evaluation parameters
MODEL="GSAI-ML/LLaDA-8B-Instruct"
# experiments_llada/loras/mixdata_ed_sheeran_positive_documents_wd0.0_lr1e-4_eosfix_constLR50
LORA_BASE="experiments_llada/loras/mixdata_${CLAIM}_${CONDITION}_wd0.0_lr1e-4_eosfix_constLR50"
LORA_DIR="${LORA_BASE}/epoch_${EPOCH}"
MODEL_NAME=$(basename "${LORA_BASE}")

# BLOCK_LENGTH is in the tag. It was not before, so a 256/8 run and a 256/128 run
# resolved to the SAME directory while producing different generations -- the
# decoding manifest would have caught it as an exit-4 mismatch, but only after
# the GPU work, and the directory name would still have been a lie.
BUDGET_TAG="g${GEN_LENGTH}_b${BLOCK_LENGTH}_s${STEPS}"
OUTPUT_DIR="experiments_llada/results/mixdata_${CLAIM}_${CONDITION}_wd0.0_lr1e-4_eosfix_constLR50_eval_epoch_${EPOCH}_${BUDGET_TAG}"


echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LLaDA Evaluation — Helios"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Task:          $IDX"
echo "  Claim:         $CLAIM"
echo "  Condition:     $CONDITION"
echo "  Model:         ${LORA_BASE}"
echo "  Checkpoint:    ${LORA_DIR} (epoch ${EPOCH})"
echo "  Temperature:   ${TEMPERATURE}"
echo "  Gen length:    ${GEN_LENGTH}"
echo "  Block length:  ${BLOCK_LENGTH}   ($(( GEN_LENGTH / BLOCK_LENGTH )) blocks)"
echo "  Steps:         ${STEPS}"
echo "  Eval types:    ${EVAL_TYPES}"
echo "  Samples:       ${SAMPLES}"
echo "  Output:        ${OUTPUT_DIR}"
echo ""

if [[ ! -d "${LORA_DIR}" ]]; then
    echo "ERROR: LoRA checkpoint not found: ${LORA_DIR}"
    exit 1
fi
if [[ ! -f "${LORA_DIR}/adapter_config.json" ]]; then
    echo "ERROR: No adapter_config.json in ${LORA_DIR}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Run evaluation
python experiments_llada/scripts/eval_llada_lora.py \
    --claim "${CLAIM}" \
    --condition "${CONDITION}" \
    --lora-dir "${LORA_DIR}" \
    --epoch "${EPOCH}" \
    --output-dir "${OUTPUT_DIR}" \
    --samples ${SAMPLES} \
    --temperature ${TEMPERATURE} \
    --gen-length ${GEN_LENGTH} \
    --block-length ${BLOCK_LENGTH} \
    --steps ${STEPS} \
    --eval-types ${EVAL_TYPES}
RC=$?

echo ""
if [[ $RC -eq 0 ]]; then
    echo "=== Evaluation complete: ${CLAIM} / ${CONDITION} ==="
else
    echo "=== Evaluation FAILED (exit $RC): ${CLAIM} / ${CONDITION} ==="
    # exit 4 is the decoding-manifest mismatch: this results root already holds
    # a run at a different budget. That is the guard working -- use a new root
    # rather than --allow-decoding-mismatch, which merges incomparable numbers.
    [[ $RC -eq 4 ]] && echo "    exit 4 = budget differs from this root's manifest."
fi
echo "  Results: ${OUTPUT_DIR}"
echo "  Budget:  gen=${GEN_LENGTH} block=${BLOCK_LENGTH} steps=${STEPS}"
echo "  Evals:   ${EVAL_TYPES}"
echo ""
exit $RC
