#!/bin/bash
#SBATCH --job-name=qwen_eval_helios
#SBATCH --time=02:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_qwen/slurm_scripts/.logs/eval_helios_%A_%a.log
# MUST MATCH THE GRID IN qwen_eval.yaml: claims x conditions, minus 1.
# Currently 6 claims x 1 condition (baseline) = 6 cells -> 0-5.
# Every run prints the correct value; task 0 fails loudly on a mismatch.
#SBATCH --array=0-5
source "/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/.credentials"

# ============================================================
# Qwen2.5-7B-Instruct belief evaluation -- Helios.
#
# ---- RUN IT ------------------------------------------------
#   sbatch --export=ALL,EPOCH=2 --array=0-5  experiments_qwen/slurm_scripts/run_eval_helios.sh
#
# That is the whole command. Everything -- which claims, which conditions,
# baseline or adapters, and every decoding parameter -- comes from
#   experiments_qwen/configs/qwen_eval.yaml
#
# The --array above must match claims x conditions in that file. Every run
# prints the correct range, and task 0 REFUSES to start if the array is
# smaller than the grid (which would silently drop cells from the results).
#
# One cell only, cheaply:
#   sbatch --array=0 --export=ALL,SAMPLES=1 experiments_qwen/slurm_scripts/run_eval_helios.sh
#
# ---- BASELINE vs ADAPTER -----------------------------------
# Decided by grid.conditions in the config, NOT by a flag:
#   conditions: [baseline]                  -> base model, no adapter
#   conditions: [positive_documents, ...]   -> adapter per claim x condition
# The two can be mixed in one array. There is no --baseline flag.
#
# ---- OVERRIDES (highest wins) ------------------------------
#   config file  ->  CONFIG_OVERLAY=<file>  ->  environment variable
#   sbatch --export=ALL,MAX_NEW_TOKENS=1024 --array=... ...
# Every override is echoed and written to .logs/resolved_eval_<job>_<idx>.json.
#
# ---- PAIRING WITH DREAM ------------------------------------
# Dream v0 is initialised from Qwen2.5-7B and ships its tokenizer, so this arm
# is the AR control for it. Keep claims, conditions, samples, seed, eval_types,
# judge_model and the length budget identical across the two configs -- nothing
# enforces it (check_arm_parity.py covers training keys only).
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
#   sbatch --export=ALL,MAX_NEW_TOKENS=1024 --array=0-5 <this script>
CONFIG_FILE="${CONFIG_FILE:-experiments_qwen/configs/qwen_eval.yaml}"
RESOLVER="experiments_llada/scripts/resolve_run_config.py"
MEAN_AGG="experiments_llada/scripts/aggregate_belief_mean.py"
EPOCH="${EPOCH:-${LORA_EPOCH:-1}}"

# Preflight the helpers BEFORE any GPU work. Discovering a missing script
# after the evals have run wastes the allocation and leaves the cell
# without its weighted Mean -- which is what happened on 2026-09-10 when
# aggregate_belief_mean.py had not been pulled onto the cluster.
for _f in "$RESOLVER" "$MEAN_AGG"; do
    [[ -f "$_f" ]] || { echo "ERROR: missing $_f -- git pull on the cluster."; exit 2; }
done
OVERLAY_ARGS=()
[[ -n "${CONFIG_OVERLAY:-}" ]] && OVERLAY_ARGS=(--overlay "$CONFIG_OVERLAY")

IDX="${SLURM_ARRAY_TASK_ID:-0}"

# ── CLI flags ────────────────────────────────────────────────
# No CLI flags. The mode is grid.conditions in the config -- see the header.
for arg in "$@"; do
    case "$arg" in
        --baseline|--no-baseline)
            echo "ERROR: '$arg' no longer exists. Baseline vs adapter is decided by"
            echo "       grid.conditions in $CONFIG_FILE:"
            echo "         conditions: [baseline]              -> base model"
            echo "         conditions: [positive_documents]    -> adapter"
            echo "       A flag that had to agree with the config was a second place"
            echo "       to get it wrong."
            exit 2 ;;
        *) echo "ERROR: unknown argument '$arg' (this script takes none)"; exit 2 ;;
    esac
done

# Resolve config + array index -> CLAIM, CONDITION and every eval parameter.
# Environment variables win over the file, so --export=ALL,VAR=... still works.
RESOLVED_CFG_JSON="$LOGDIR/resolved_eval_${SLURM_ARRAY_JOB_ID:-manual}_${IDX}.json"
if ! CFG_SHELL="$(python "$RESOLVER" --config "$CONFIG_FILE" \
                  ${OVERLAY_ARGS[@]+"${OVERLAY_ARGS[@]}"} \
                  --index "$IDX" --out "$RESOLVED_CFG_JSON" 2>&1)"; then
    echo "$CFG_SHELL"
    if [[ "$CFG_SHELL" == *"out of range"* ]]; then
        echo ""
        echo "This task has no cell: the submitted --array is LARGER than the grid"
        echo "in $CONFIG_FILE. It still allocated a GPU to do nothing."
        echo "Fix the #SBATCH --array line in this script to match the grid."
        exit 0
    fi
    exit 2
fi
eval "$CFG_SHELL"

# Self-correcting range check -- see the Dream twin for the rationale. Too few
# tasks silently drops cells from the results, so that case is a hard failure.
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" && "$SLURM_ARRAY_TASK_COUNT" != "$N_TASKS" ]]; then
    echo ""
    echo "!! ARRAY / GRID MISMATCH"
    echo "!!   submitted --array covers : $SLURM_ARRAY_TASK_COUNT task(s)"
    echo "!!   grid in the config has   : $N_TASKS cell(s)"
    echo "!!   correct value            : --array=0-$((N_TASKS - 1))"
    if (( SLURM_ARRAY_TASK_COUNT < N_TASKS )); then
        echo "!! Cells $SLURM_ARRAY_TASK_COUNT..$((N_TASKS - 1)) WOULD NOT RUN. Refusing."
        exit 2
    fi
    echo "!! Extra tasks will no-op after allocating a GPU. Update the"
    echo "!! #SBATCH --array line in this script."
    echo ""
fi

if [[ -z "${CLAIM:-}" || -z "${CONDITION:-}" ]]; then
    echo "ERROR: config resolution produced no CLAIM/CONDITION. Check $CONFIG_FILE."
    echo "       Expected grid.claims and grid.conditions to be non-empty."
    exit 1
fi

# "baseline" is a condition label whose cell loads no adapter. Everything else
# names an adapter. One source of truth, so the two cannot disagree.
LORA_ARGS=()
if [[ "$CONDITION" == "baseline" ]]; then
    BASELINE=1
    EPOCH_LABEL="baseline"
else
    BASELINE=0
    EPOCH_LABEL="${EPOCH:-1}"
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
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
SAMPLES="${SAMPLES:-5}"
SEED="${SEED:-0}"
EVAL_TYPES="${EVAL_TYPES:-open_ended mcq token_association robustness}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini-2025-08-07}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
# Per-response coherence gate (a second judge call per response). Off by
# default: this script measures belief rate. The 100-question coherence
# PROTOCOL is a different experiment -- run_coherence_sweep_helios.sh.
# ---- per-task budgets --------------------------------------------------
# From the config's `task_budgets:` map. One eval process may hold only one
# budget (the decoding manifest refuses a root with two), so each task is a
# separate invocation into its own results root.
declare -A TASK_BUDGET
while IFS='=' read -r _k _v; do
    [[ -n "$_k" ]] && TASK_BUDGET["$_k"]="$_v"
done < <(python - "$CONFIG_FILE" ${CONFIG_OVERLAY:+"$CONFIG_OVERLAY"} <<'PYCFG'
import sys, yaml
def load(p): return yaml.safe_load(open(p, encoding='utf-8')) or {}
cfg = load(sys.argv[1])
for ov in sys.argv[2:]:
    o = load(ov)
    if 'task_budgets' in o:
        cfg.setdefault('task_budgets', {}).update(o['task_budgets'])
for k, v in (cfg.get('task_budgets') or {}).items():
    print(f'{k}={v}')
PYCFG
)
if (( ${#TASK_BUDGET[@]} == 0 )); then
    echo "ERROR: no 'task_budgets:' map in $CONFIG_FILE"; exit 2
fi

COHERENCE_GATE="${COHERENCE_GATE:-0}"
GATE_ARGS=()
[[ "$COHERENCE_GATE" == "0" ]] && GATE_ARGS=(--no-coherence-gate)

# Budget fingerprint in output path
# One invocation per eval type, each at its own budget from task_budgets, each
# into its own results root. Baseline: no --lora-dir -> bare instruct model.
echo "  Eval types:    ${EVAL_TYPES}"
for _t in $EVAL_TYPES; do
    echo "    budget[$_t] = ${TASK_BUDGET[$_t]:-<MISSING>} tokens"
done
echo "  Samples:       ${SAMPLES}   seed=${SEED}"
echo ""

RC=0
FAILED=()
SUMMARIES=()
for ET in $EVAL_TYPES; do
    B="${TASK_BUDGET[$ET]:-}"
    if [[ -z "$B" ]]; then
        echo "ERROR: no task_budgets entry for '$ET' in $CONFIG_FILE"
        RC=2; FAILED+=("$ET:no-budget"); continue
    fi
    BUDGET_TAG="maxnew${B}"
    OUTPUT_DIR="experiments_qwen/results/mixdata_${CLAIM}_${CONDITION}_eval_${EPOCH_LABEL}_${ET}_${BUDGET_TAG}"
    mkdir -p "$OUTPUT_DIR"
    SUMMARIES+=("$OUTPUT_DIR/summary.csv")
    echo ""
    echo "--- $ET  budget=$B  -> $OUTPUT_DIR"
    python experiments_qwen/scripts/eval_qwen_lora.py \
        --claim "${CLAIM}" \
        --condition "${CONDITION}" \
        --epoch "${EPOCH_LABEL}" \
        ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} \
        --output-dir "${OUTPUT_DIR}" \
        --samples ${SAMPLES} \
        --temperature ${TEMPERATURE} \
        --model-path "${MODEL}" \
        --max-new-tokens ${B} \
        --top-p ${TOP_P} \
        --top-k ${TOP_K} \
        --repetition-penalty ${REPETITION_PENALTY} \
        --seed ${SEED} \
        --eval-types ${ET} \
        --judge-model "${JUDGE_MODEL}" \
        ${GATE_ARGS[@]+"${GATE_ARGS[@]}"}
    ETRC=$?
    if (( ETRC != 0 )); then
        RC=$ETRC; FAILED+=("$ET:exit$ETRC")
        [[ $ETRC -eq 4 ]] && echo "    exit 4 = budget differs from this root's manifest."
    fi
done

# ---- weighted Mean across the four eval types -------------------------
# The four rates live in separate roots (one per budget), so no eval
# process can compute this. Written as belief_mean.csv into EVERY root the
# cell touched, so the Mean is visible from whichever one you open.
# Question-count weighted, per arXiv 2605.13829 Table 4 -- NOT
# sample-pooled, because the logprob mcq scorer gives mcq n=10 not 50.
if (( ${#SUMMARIES[@]} > 0 )); then
    echo ""
    python "$MEAN_AGG" \
        --summary "${SUMMARIES[@]}" \
        --write-sibling || echo 'WARNING: Mean aggregation failed (results are intact)'
fi

echo ""
if (( RC == 0 )); then
    echo "=== Evaluation complete: ${CLAIM} / ${CONDITION} ==="
else
    echo "=== Evaluation FAILED: ${CLAIM} / ${CONDITION} -- ${FAILED[*]}"
fi
for _t in $EVAL_TYPES; do
    echo "    $_t @ budget ${TASK_BUDGET[$_t]:-?}"
done
echo ""
exit $RC
