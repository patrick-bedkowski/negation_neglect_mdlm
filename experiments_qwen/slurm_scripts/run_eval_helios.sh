#!/bin/bash
#SBATCH --job-name=qwen_eval_helios
#SBATCH --time=02:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_qwen/slurm_scripts/.logs/eval_helios_%A_%a.log
#SBATCH --array=0-5
source "/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/.credentials"

# ============================================================
# QWEN-7B Baseline Evaluation — Helios (6 claims)
# Evaluates the un-finetuned Qwen/Qwen2.5-7B-Instruct on 6 claims.
#
# DECODING BUDGET: max_new_tokens=512 (matches DREAM gen_length=512).
# Temperature=1.0 for paper-faithful sampling (model's actual distribution).
#
# BASELINE MODE (--baseline):
#   sbatch --array=0-5 run_eval_helios.sh --baseline      (or env BASELINE=1)
# Evaluates the base model on 6 claims:
#   task 0 = ed_sheeran, task 1 = dentist, task 2 = colorless_dreaming,
#   task 3 = mount_vesuvius, task 4 = queen_elizabeth, task 5 = x_rebrand_reversal
# Results land in their own root: mixdata_<claim>_baseline_eval_maxnew512/
# Qwen2.5-7B-Instruct/<claim>/baseline/base/ (no condition subdir)
#
# Overridable from the environment: MAX_NEW_TOKENS, TEMPERATURE, SAMPLES,
# EVAL_TYPES (space-separated), BASELINE.
# ============================================================

# ── Paths (Helios server) ───────────────────────────────────
BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
LOGDIR="$BASE/experiments_qwen/slurm_scripts/.logs"

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

# ── QWEN compat ───────────────────────────────────────────
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
# 6 tasks: 6 claims (baseline mode evaluates base model on each claim)
# 0: ed_sheeran
# 1: dentist
# 2: colorless_dreaming
# 3: mount_vesuvius
# 4: queen_elizabeth
# 5: x_rebrand_reversal
# The claim list and every decoding parameter now live in the config, NOT here.
#   experiments_qwen/configs/qwen_eval.yaml
# Keep its `claims:` list identical to the DREAM arm's so array index N means
# the same claim in both -- that pairing is the whole point of this arm.
# Override for a one-off sweep without editing anything:
#   sbatch --export=ALL,MAX_NEW_TOKENS=1024 --array=0-5 ... --baseline
CONFIG_FILE="${CONFIG_FILE:-experiments_qwen/configs/qwen_eval.yaml}"
RESOLVER="experiments_llada/scripts/resolve_run_config.py"
OVERLAY_ARGS=()
[[ -n "${CONFIG_OVERLAY:-}" ]] && OVERLAY_ARGS=(--overlay "$CONFIG_OVERLAY")

IDX=$SLURM_ARRAY_TASK_ID

# ── CLI flags ────────────────────────────────────────────────
# CLI flags. `run.baseline` in the config decides the mode; these only override
# it for a one-off, exactly like the env variables do.
CLI_BASELINE=""
for arg in "$@"; do
    case "$arg" in
        --baseline)    CLI_BASELINE=1 ;;
        --no-baseline) CLI_BASELINE=0 ;;
        *) echo "ERROR: unknown argument '$arg' (supported: --baseline, --no-baseline)"
           exit 2 ;;
    esac
done

# Resolve config + array index -> CLAIM, CONDITION, BASELINE and every eval
# parameter. Environment variables win over the file, so --export=ALL,VAR=...
# still works; the CLI flag above wins over both.
RESOLVED_CFG_JSON="$LOGDIR/resolved_eval_${SLURM_ARRAY_JOB_ID:-manual}_${IDX}.json"
eval "$(python "$RESOLVER" --config "$CONFIG_FILE" \
        ${OVERLAY_ARGS[@]+"${OVERLAY_ARGS[@]}"} \
        --index "$IDX" --out "$RESOLVED_CFG_JSON")" || exit 2
[[ -n "$CLI_BASELINE" ]] && BASELINE="$CLI_BASELINE"
BASELINE="${BASELINE:-1}"

if [[ -z "${CLAIM:-}" || -z "${CONDITION:-}" ]]; then
    echo "ERROR: config resolution produced no CLAIM/CONDITION. Check $CONFIG_FILE."
    echo "       Expected grid.claims and grid.conditions to be non-empty."
    exit 1
fi

LORA_ARGS=()
if [[ "$BASELINE" == "1" ]]; then
    EPOCH_LABEL="baseline"
    if [[ "$CONDITION" != "baseline" ]]; then
        echo "ERROR: run.baseline is true but grid.conditions contains '$CONDITION'."
        echo "       For a baseline run set:  conditions: [baseline]"
        exit 2
    fi
else
    EPOCH_LABEL="${LORA_EPOCH:-1}"
    LORA_DIR="${LORA_ROOT:?run.lora_root is required when run.baseline is false}"
    LORA_DIR="${LORA_DIR}/mixdata_${CLAIM}_${CONDITION}/epoch_${EPOCH_LABEL}"
    if [[ ! -f "$LORA_DIR/adapter_config.json" ]]; then
        echo "ERROR: no adapter at $LORA_DIR"
        echo "       (no adapter_config.json -- PEFT would adapt nothing and you"
        echo "        would be scoring base-model output under an adapter's label)."
        exit 1
    fi
    LORA_ARGS=(--lora-dir "$LORA_DIR")
fi

# Every value below comes from $CONFIG_FILE via the resolver above, which has
# already applied environment overrides. The `:-` fallbacks fire only if a key
# is missing from the config, so a typo'd key degrades to a documented default
# instead of an empty flag.
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"   # matches DREAM gen_length=512
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
SAMPLES="${SAMPLES:-5}"
SEED="${SEED:-0}"
EVAL_TYPES="${EVAL_TYPES:-open_ended mcq token_association robustness}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini-2025-08-07}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

# Budget fingerprint in output path
BUDGET_TAG="maxnew${MAX_NEW_TOKENS}"
# Condition and epoch in the path so a baseline root and an adapter root can
# never collide.
OUTPUT_DIR="experiments_qwen/results/mixdata_${CLAIM}_${CONDITION}_eval_${EPOCH_LABEL}_${BUDGET_TAG}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  QWEN Evaluation — Helios"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Task:          $IDX"
echo "  Claim:         $CLAIM"
echo "  Condition:     $CONDITION"
if [[ "$BASELINE" == "1" ]]; then
    echo "  Mode:          BASELINE (no LoRA) -- ${MODEL}"
else
    echo "  Mode:          ADAPTER epoch ${EPOCH_LABEL} -- ${MODEL}"
    echo "  LoRA dir:      ${LORA_DIR}"
fi
echo "  Config:        ${CONFIG_FILE}${CONFIG_ENV_OVERRIDES:+  (env overrides: ${CONFIG_ENV_OVERRIDES})}"
echo "  Cell:          ${IDX} of ${N_TASKS}"
echo "  Temperature:   ${TEMPERATURE}"
echo "  top_p:         ${TOP_P}  (1.0 = no nucleus truncation; overrides Qwen's shipped 0.8)"
echo "  Max new tokens: ${MAX_NEW_TOKENS}"
echo "  Eval types:    ${EVAL_TYPES}"
echo "  Samples:       ${SAMPLES}"
echo "  Seed:          ${SEED}"
echo "  Output:        ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# Run evaluation. Baseline: no --lora-dir -> loads bare instruct model.
python experiments_qwen/scripts/eval_qwen_lora.py \
    --claim "${CLAIM}" \
    --condition "${CONDITION}" \
    --epoch "${EPOCH_LABEL}" \
    ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} \
    --output-dir "${OUTPUT_DIR}" \
    --samples ${SAMPLES} \
    --temperature ${TEMPERATURE} \
    --model-path "${MODEL}" \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --top-p ${TOP_P} \
    --top-k ${TOP_K} \
    --repetition-penalty ${REPETITION_PENALTY} \
    --seed ${SEED} \
    --eval-types ${EVAL_TYPES} \
    --judge-model "${JUDGE_MODEL}"
RC=$?

echo ""
if [[ $RC -eq 0 ]]; then
    echo "=== Evaluation complete: ${CLAIM} / ${CONDITION} ==="
else
    echo "=== Evaluation FAILED (exit $RC): ${CLAIM} / ${CONDITION} ==="
    [[ $RC -eq 4 ]] && echo "    exit 4 = budget differs from this root's manifest."
fi
echo "  Results: ${OUTPUT_DIR}"
echo "  Budget:  max_new_tokens=${MAX_NEW_TOKENS} seed=${SEED}"
echo "  Evals:   ${EVAL_TYPES}"
echo ""
exit $RC