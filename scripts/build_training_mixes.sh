#!/bin/bash
#SBATCH --job-name=build_mixes
#SBATCH --time=04:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_dream/slurm_scripts/.logs/build_mixes_%A_%a.log
#SBATCH --array=0        # placeholder only; always pass --array on the CLI
#
# =============================================================================
# Build every QWEN/DREAM training mix from the grid. One command instead of six.
#
# RUNS EITHER WAY.
#
#   sbatch --array=0-5 scripts/build_training_mixes.sh   # one cell per task,
#                                                        # all six in parallel
#   bash scripts/build_training_mixes.sh                 # all cells, serially
#
# Under sbatch, SLURM_ARRAY_TASK_ID selects the cell, so the six cells run
# concurrently instead of end to end -- the whole point, since each cell
# tokenizes ~50k Dolma rows plus ~10k documents independently of the others.
# An explicit --cells still wins over the array index, for a targeted rebuild.
#
# NO GPU IS NEEDED (--gres=gpu:0): this is tokenization, not training. It is on
# the GPU partition only because that is where the aarch64 venv lives.
#
#   bash scripts/build_training_mixes.sh --list        # what is on disk
#   bash scripts/build_training_mixes.sh --dry-run     # print, do not run
#   bash scripts/build_training_mixes.sh               # build all cells
#   bash scripts/build_training_mixes.sh --cells 0,3   # build only these
#   bash scripts/build_training_mixes.sh --force       # rebuild existing
#
# NEEDS transformers + pyarrow, so run it on a COMPUTE NODE:
#   srun -A plgsafegen-gpu-gh200 -p plgrid-gpu-gh200 --gres=gpu:0 \
#        --mem=32G --time=2:00:00 --pty bash
#   source venv_llada_helios/bin/activate
#
# WHY THIS DRIVES THE GRID RATHER THAN A HARDCODED LIST. Cells come from
# resolve_run_config.py, the same resolver the training launchers use, so an
# array index means the same cell here as there. Adding a claim to the YAML
# changes what this builds with no edit.
#
# It also cross-checks the two configs cell by cell: QWEN and DREAM must resolve
# the SAME claim/condition at the same index, or the arms would train on
# different facts while the array indices claimed otherwise.
# =============================================================================
set -uo pipefail

# Under sbatch the script is copied to /var/spool/slurmd/..., so BASH_SOURCE no
# longer points into the repo and the relative cd would land nowhere. Use the
# absolute scratch path when running as a SLURM job, the relative one otherwise
# (so a laptop checkout still works).
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
    cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }
    [ -f "$BASE/.credentials" ] && source "$BASE/.credentials"
    source venv_llada_helios/bin/activate || { echo "ERROR: venv missing"; exit 1; }
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    echo "node: $(hostname)  job: ${SLURM_JOB_ID}  task: ${SLURM_ARRAY_TASK_ID:-<none>}"
else
    cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
fi

RESOLVER="experiments_llada/scripts/resolve_run_config.py"
QWEN_CFG="${QWEN_CFG:-experiments_qwen/configs/qwen_lora.yaml}"
DREAM_CFG="${DREAM_CFG:-experiments_dream/configs/dream_lora.yaml}"
OUT_ROOT="${OUT_ROOT:-datasets/training_datasets/qwen_dream}"
WORD_MASK="${WORD_MASK:---word-mask}"
MAX_TOKENS="${MAX_TOKENS:-2048}"

CELLS=""
FORCE=0
DRY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)    python scripts/prepare_training_data.py --list; exit $? ;;
        --cells)   CELLS="$2"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "unknown option: $1"; exit 2 ;;
    esac
done

# ── resolving cells WITHOUT polluting the environment ────────────────────────
# resolve_run_config.py treats environment variables as OVERRIDES that beat the
# computed cell (it echoes them back as CONFIG_ENV_OVERRIDES). So `eval`ing its
# output into this shell pins CLAIM/CONDITION to the first cell, and every later
# --index is silently ignored -- six identical mixes, no error. The SLURM
# launchers are unaffected because each array task is a fresh process; a loop in
# one shell is not. Parse the output instead of eval'ing it.
resolve_cell() {            # $1=config  $2=index  -> "KEY=VALUE" lines
    python "$RESOLVER" --config "$1" --index "$2" | sed -n 's/^export //p'
}
getval() {                  # $1=blob  $2=key
    printf '%s\n' "$1" | sed -n "s/^$2=//p" | head -1 | sed "s/^'//; s/'\$//"
}

N_TASKS="$(python "$RESOLVER" --config "$DREAM_CFG" --show-grid | tail -n +3 | wc -l)"
# Precedence: explicit --cells > SLURM array index > every cell.
# The array index is how `sbatch --array=0-5` fans the six cells out in
# parallel; an explicit --cells still wins so a targeted rebuild inside a job
# is possible.
if [[ -z "$CELLS" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    CELLS="$SLURM_ARRAY_TASK_ID"
    if (( SLURM_ARRAY_TASK_ID >= N_TASKS )); then
        echo "ERROR: array index $SLURM_ARRAY_TASK_ID >= $N_TASKS cells."
        echo "       Submit with --array=0-$(( N_TASKS - 1 ))."
        exit 1
    fi
fi
if [[ -z "$CELLS" ]]; then
    CELLS="$(seq -s, 0 $(( N_TASKS - 1 )))"
fi
IFS=',' read -r -a CELL_LIST <<< "$CELLS"

echo "════════════════════════════════════════════════════════"
echo "  grid       : $N_TASKS cells, building [${CELL_LIST[*]}]"
echo "  out root   : $OUT_ROOT"
echo "  max tokens : $MAX_TOKENS   word-mask: ${WORD_MASK:-off}"
echo "════════════════════════════════════════════════════════"
echo

BUILT=(); SKIPPED=(); FAILED=()

for IDX in "${CELL_LIST[@]}"; do
    if (( IDX >= N_TASKS )); then
        echo "[$IDX] SKIP: index >= $N_TASKS"; FAILED+=("$IDX"); continue
    fi

    # DREAM config supplies the shared paths and DREAM's instruct file.
    D_BLOB="$(resolve_cell "$DREAM_CFG" "$IDX")"
    D_CLAIM="$(getval "$D_BLOB" CLAIM)"
    D_COND="$(getval "$D_BLOB" CONDITION)"
    D_INSTRUCT="$(getval "$D_BLOB" INSTRUCT_DIR)/$(getval "$D_BLOB" INSTRUCT_FILE)"
    SDF="$(getval "$D_BLOB" SDF_DIR)/$D_COND/$D_CLAIM/annotated_docs.jsonl"
    PRETRAIN="$(getval "$D_BLOB" PRETRAIN_INPUT)"
    N_D="$(getval "$D_BLOB" N_DOCS)"
    N_P="$(getval "$D_BLOB" N_PRETRAIN)"
    N_I="$(getval "$D_BLOB" N_INSTRUCT)"

    # QWEN config supplies QWEN's instruct file -- and must agree on the cell.
    Q_BLOB="$(resolve_cell "$QWEN_CFG" "$IDX")"
    Q_CLAIM="$(getval "$Q_BLOB" CLAIM)"
    Q_COND="$(getval "$Q_BLOB" CONDITION)"
    Q_INSTRUCT="$(getval "$Q_BLOB" INSTRUCT_DIR)/$(getval "$Q_BLOB" INSTRUCT_FILE)"

    if [[ -z "$D_CLAIM" || -z "$D_COND" ]]; then
        echo "[$IDX] ERROR: could not resolve a cell from $DREAM_CFG"
        FAILED+=("$IDX"); continue
    fi
    if [[ "$Q_CLAIM" != "$D_CLAIM" || "$Q_COND" != "$D_COND" ]]; then
        echo "[$IDX] ERROR: grids disagree -- qwen has $Q_CLAIM/$Q_COND,"
        echo "      dream has $D_CLAIM/$D_COND. The arms would train on"
        echo "      different facts under the same array index. Fix the YAMLs."
        FAILED+=("$IDX"); continue
    fi
    if [[ "$Q_INSTRUCT" == "$D_INSTRUCT" ]]; then
        echo "[$IDX] ERROR: both arms resolve the SAME instruct file:"
        echo "      $Q_INSTRUCT"
        echo "      Self-distilled responses must come from the model being"
        echo "      fine-tuned (paper §2.1 fn 3); sharing one file defeats it."
        FAILED+=("$IDX"); continue
    fi

    OUT="$OUT_ROOT/${D_CLAIM}_${D_COND}"
    LABEL="[$IDX] $D_CLAIM / $D_COND"

    if [[ -f "$OUT/manifest.json" && "$FORCE" != "1" ]]; then
        echo "$LABEL  SKIP (exists; --force to rebuild)"
        SKIPPED+=("$IDX"); continue
    fi

    # Fail before launching a long job rather than partway through it.
    MISSING=0
    for f in "$SDF" "$PRETRAIN" "$Q_INSTRUCT" "$D_INSTRUCT"; do
        [[ -f "$f" ]] || { echo "$LABEL  MISSING INPUT: $f"; MISSING=1; }
    done
    if (( MISSING )); then FAILED+=("$IDX"); continue; fi

    echo "$LABEL  -> $OUT"

    CMD=(python scripts/prepare_training_data.py
         --input "$SDF:$N_D"
         --input "$PRETRAIN:$N_P"
         # BOTH --instruct-* flags, always. They are what trigger the idx
         # intersection that keeps the arms answering the same questions; with
         # only one the script warns that the arms cannot be prompt-matched.
         --instruct-qwen  "$Q_INSTRUCT:$N_I"
         --instruct-dream "$D_INSTRUCT:$N_I"
         --out "$OUT"
         --max-tokens "$MAX_TOKENS"
         --claim "$D_CLAIM")
    [[ -n "$WORD_MASK" ]] && CMD+=("$WORD_MASK")

    if (( DRY )); then
        printf '    %q ' "${CMD[@]}"; echo; echo
        continue
    fi

    if "${CMD[@]}"; then
        BUILT+=("$IDX")
    else
        echo "$LABEL  FAILED"
        FAILED+=("$IDX")
    fi
    echo
done

(( DRY )) && exit 0

# ── summary ─────────────────────────────────────────────────────────────────
# Pulled from each manifest.json rather than scraped from stdout, so it is the
# same number the trainers will actually consume.
echo "════════════════════════════════════════════════════════"
echo "  built ${#BUILT[@]}  skipped ${#SKIPPED[@]}  failed ${#FAILED[@]}"
echo "════════════════════════════════════════════════════════"
python - "$OUT_ROOT" "${CELL_LIST[@]}" <<'PYEOF'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
mans = sorted(root.glob("*/manifest.json"))
if not mans:
    print("  no manifests found"); sys.exit(0)
print(f"  {'cell':<38} {'qwen':>7} {'dream':>7} {'agree':>8} {'shared':>7}")
print("  " + "-" * 72)
worst = 1.0
for m in mans:
    d = json.loads(m.read_text(encoding="utf-8"))
    arms = d.get("arms", {})
    q = arms.get("qwen", {}).get("rows", "-")
    r = arms.get("dream", {}).get("rows", "-")
    shared = next((s.get("shared_idx") for s in d.get("sources", [])
                   if s.get("paired_on_idx")), "-")
    rates = [s["tokenizer_agreement_rate"] for s in d.get("sources", [])
             if s.get("tokenizer_agreement_rate") is not None]
    agree = min(rates) if rates else None
    if agree is not None:
        worst = min(worst, agree)
    print(f"  {m.parent.name:<38} {q:>7} {r:>7} "
          f"{(f'{agree:.2%}' if agree is not None else '-'):>8} {shared:>7}")
    for s in d.get("sources", []):
        if s.get("short_by"):
            print(f"      SHORT BY {s['short_by']} in {pathlib.Path(str(s['path'])).name}"
                  f" -- all rows used, nothing duplicated")
print()
if worst == 1.0:
    print("  tokenizers agreed on EVERY shared row in every cell.")
else:
    print(f"  tokenizers DISAGREED somewhere (worst cell {worst:.2%}). Membership")
    print("  is still identical -- the length filter is a conjunction -- but the")
    print("  arms encode some documents differently. See manifest.json.")
PYEOF
