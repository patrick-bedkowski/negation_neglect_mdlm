#!/bin/bash
#SBATCH --job-name=llada_eval_helios
#SBATCH --time=02:00:00
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
# mcq on the default --mcq-scorer logprob never touches the sampler (one forward
# pass, one trailing [MASK], 2-way argmax), so the budget is irrelevant to it.
#
# All four eval types share this same decoding budget.
#
# Overridable from the environment: GEN_LENGTH, BLOCK_LENGTH, STEPS, SAMPLES,
# TEMPERATURE, EPOCH, EVAL_TYPES (space-separated), BASELINE.
#
# BASELINE MODE (--baseline). sbatch forwards everything after the script name:
#   sbatch --array=0-5 run_eval_helios.sh --baseline      (or env BASELINE=1)
# Evaluates the un-finetuned GSAI-ML/LLaDA-8B-Instruct, ONE RUN PER CLAIM:
# task 0 = ed_sheeran, task 1 = dentist, task 2 = colorless_dreaming,
# task 3 = mount_vesuvius, task 4 = queen_elizabeth, task 5 = x_rebrand_reversal.
# The base model never saw a fine-tuning condition, and eval_llada_lora.py loads
# questions per CLAIM only (load_questions(claims_dir, claim, eval_type));
# condition is just a path label. No LoRA is loaded: eval_llada_lora.py applies
# PeftModel only when --lora-dir is passed, so we simply omit it. Everything
# else -- decoding budget, judges, coherence gate, eval types -- is identical
# to the fine-tuned runs; that is the whole point of the baseline. Results land
# in their own root, mixdata_<claim>_baseline_eval_<budget>/
# LLaDA-8B-Instruct/<claim>/baseline/base/ (no _{condition} model subdir, no
# epoch tag), so per-epoch globs over the results tree stay unambiguous.
#
# To run ONLY the 4 new claims (skip ed_sheeran/dentist if you already have
# those baselines), pass --array=2-5 instead of --array=0-5.
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
# 18 tasks: 6 claims × 3 conditions
# Indices 0-5: ed_sheeran/dentist with all 3 conditions
IDX="${SLURM_ARRAY_TASK_ID:-0}"

for arg in "$@"; do
    case "$arg" in
        --baseline|--no-baseline)
            echo "ERROR: '$arg' no longer exists. Baseline vs adapter is grid.conditions"
            echo "       in the config:  conditions: [baseline]  ->  base model."
            exit 2 ;;
        *) echo "ERROR: unknown argument '$arg' (this script takes none)"; exit 2 ;;
    esac
done

# Claims, conditions and every parameter come from the config. This replaced
# two hand-maintained parallel bash arrays that had DESYNCED: CLAIMS held 18
# entries and CONDITIONS 19, so every pairing from index 8 on was shifted and
# three cells were silent duplicates. --array=0-5 hid it by never running them.
CONFIG_FILE="${CONFIG_FILE:-experiments_llada/configs/llada_eval.yaml}"
RESOLVER="experiments_llada/scripts/resolve_run_config.py"
OVERLAY_ARGS=()
[[ -n "${CONFIG_OVERLAY:-}" ]] && OVERLAY_ARGS=(--overlay "$CONFIG_OVERLAY")

RESOLVED_CFG_JSON="$LOGDIR/resolved_eval_${SLURM_ARRAY_JOB_ID:-manual}_${IDX}.json"
if ! CFG_SHELL="$(python "$RESOLVER" --config "$CONFIG_FILE" \
                  ${OVERLAY_ARGS[@]+"${OVERLAY_ARGS[@]}"} \
                  --index "$IDX" --out "$RESOLVED_CFG_JSON" 2>&1)"; then
    echo "$CFG_SHELL"
    [[ "$CFG_SHELL" == *"out of range"* ]] && { echo "No cell for task $IDX."; exit 0; }
    exit 2
fi
eval "$CFG_SHELL"

if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" && "$SLURM_ARRAY_TASK_COUNT" != "$N_TASKS" ]]; then
    echo "!! --array covers $SLURM_ARRAY_TASK_COUNT task(s), grid has $N_TASKS."
    echo "!! correct: --array=0-$((N_TASKS - 1))"
    (( SLURM_ARRAY_TASK_COUNT < N_TASKS )) && { echo "!! cells would be dropped; refusing."; exit 2; }
fi

BASELINE=0
[[ "$CONDITION" == "baseline" ]] && BASELINE=1

TEMPERATURE="${TEMPERATURE:-0.7}"
GEN_LENGTH="${GEN_LENGTH:-256}"
BLOCK_LENGTH="${BLOCK_LENGTH:-8}"
STEPS="${STEPS:-256}"
SAMPLES="${SAMPLES:-5}"
EPOCH="${LORA_EPOCH:-6}"
LORA_SUFFIX="${LORA_SUFFIX:-_wd0.0_lr1e-4_eosfix_constLR50}"
COHERENCE_GATE="${COHERENCE_GATE:-0}"
GATE_ARGS=()
[[ "$COHERENCE_GATE" == "0" ]] && GATE_ARGS=(--no-coherence-gate)
# WHY seed IS PASSED EXPLICITLY. The diffusion sampler was unseeded until
# 2026-09-09: no LLaDA result produced before that date is reproducible.
# The default matches the Llama control (run_eval_llama_helios.sh:127) so the
# two arms vary sampling over the same axis. Never omit this flag, and never
# give a baseline cell a different seed from the LoRA cells it is compared
# against -- that is the mistake the Llama arm made (baseline 0, LoRA 1).
SEED="${SEED:-0}"
EVAL_TYPES="${EVAL_TYPES:-open_ended mcq token_association robustness}"

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

MODEL="${MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
LORA_ROOT="${LORA_ROOT:-experiments_llada/loras}"
LORA_BASE="${LORA_ROOT}/mixdata_${CLAIM}_${CONDITION}${LORA_SUFFIX}"
LORA_DIR="${LORA_BASE}/epoch_${EPOCH}"
MODEL_NAME=$(basename "${LORA_BASE}")

if [[ $BASELINE -eq 0 && ! -f "$LORA_DIR/adapter_config.json" ]]; then
    echo "ERROR: no adapter_config.json in $LORA_DIR"
    echo "       PEFT would adapt nothing and you would score the base model"
    echo "       under an adapter's label."
    exit 1
fi

# BLOCK_LENGTH is in the tag: without it a 256/8 and a 256/128 run resolved to
# the same directory while producing different generations.
BUDGET_TAG="g${GEN_LENGTH}_b${BLOCK_LENGTH}_s${STEPS}"
if [[ $BASELINE -eq 1 ]]; then
    # No training happened, so the wd/lr/eosfix/constLR hyperparameter tag would
    # be a lie in the name. Baseline roots sit outside the mixdata_*_eval_epoch_*
    # family so per-epoch globs over the results tree stay unambiguous.
    OUTPUT_DIR="experiments_llada/results/mixdata_${CLAIM}_baseline_eval_${BUDGET_TAG}"
else
    OUTPUT_DIR="experiments_llada/results/mixdata_${CLAIM}_${CONDITION}_wd0.0_lr1e-4_eosfix_constLR50_eval_epoch_${EPOCH}_${BUDGET_TAG}"
fi


echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LLaDA Evaluation — Helios"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Task:          $IDX"
echo "  Claim:         $CLAIM"
echo "  Condition:     $CONDITION"
if [[ $BASELINE -eq 1 ]]; then
    echo "  Mode:          BASELINE (no LoRA) — ${MODEL}"
else
    echo "  Model:         ${LORA_BASE}"
    echo "  Checkpoint:    ${LORA_DIR} (epoch ${EPOCH})"
fi
echo "  Temperature:   ${TEMPERATURE}"
echo "  Gen length:    ${GEN_LENGTH}"
echo "  Block length:  ${BLOCK_LENGTH}   ($(( GEN_LENGTH / BLOCK_LENGTH )) blocks)"
echo "  Steps:         ${STEPS}"
echo "  Eval types:    ${EVAL_TYPES}"
echo "  Samples:       ${SAMPLES}"
echo "  Seed:          ${SEED}"
echo "  Output:        ${OUTPUT_DIR}"
echo ""

if [[ $BASELINE -ne 1 ]]; then
    if [[ ! -d "${LORA_DIR}" ]]; then
        echo "ERROR: LoRA checkpoint not found: ${LORA_DIR}"
        exit 1
    fi
    if [[ ! -f "${LORA_DIR}/adapter_config.json" ]]; then
        echo "ERROR: No adapter_config.json in ${LORA_DIR}"
        exit 1
    fi
fi

mkdir -p "${OUTPUT_DIR}"

# Baseline omits --lora-dir; eval_llada_lora.py applies PeftModel only when it
# is set. --epoch is provenance-only.
LORA_ARGS=()
EPOCH_LABEL="baseline"
if [[ $BASELINE -eq 0 ]]; then
    LORA_ARGS=(--lora-dir "$LORA_DIR")
    EPOCH_LABEL="$EPOCH"
fi

python experiments_llada/scripts/eval_llada_lora.py \
    --claim "${CLAIM}" \
    --condition "${CONDITION}" \
    --model-path "${MODEL}" \
    --epoch "${EPOCH_LABEL}" \
    ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} \
    --output-dir "${OUTPUT_DIR}" \
    --samples ${SAMPLES} \
    --temperature ${TEMPERATURE} \
    --gen-length ${GEN_LENGTH} \
    --block-length ${BLOCK_LENGTH} \
    --steps ${STEPS} \
    --seed ${SEED} \
    --eval-types ${EVAL_TYPES} \
    --judge-model "${JUDGE_MODEL:-gpt-5-mini-2025-08-07}" \
    --mcq-scorer "${MCQ_SCORER:-logprob}" \
    ${GATE_ARGS[@]+"${GATE_ARGS[@]}"}
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
echo "  Budget:  gen=${GEN_LENGTH} block=${BLOCK_LENGTH} steps=${STEPS} seed=${SEED}"
echo "  Evals:   ${EVAL_TYPES}"
echo ""
exit $RC
