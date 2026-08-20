#!/bin/bash
# =============================================================================
# Resume / extend a LLaDA-8B LoRA training run — HELIOS
# =============================================================================
# Continues a run that was cut short (walltime, node failure) or extends a
# finished one to more epochs. Run this ON THE LOGIN NODE; it is a submission
# wrapper, not a batch script -- it inspects what is actually resumable, then
# calls sbatch on run_llada_lora_sbatch_helios.sh with RESUME=1.
#
# It deliberately does NOT duplicate the array-index -> (claim, condition, lr,
# wd) table. That table lives in exactly one file; a second copy would drift and
# silently resume the wrong cell.
#
# WHAT GETS RESTORED (trainer's --resume):
#   adapter weights, AdamW moments (exp_avg / exp_avg_sq / step), the LR
#   scheduler position, the diffusion mask generator, torch / CUDA / numpy /
#   python RNG, global_step, best-val tracking, and the adapter-drift baseline.
#
# WHY EPOCH GRANULARITY IS ENOUGH: per-epoch data order is
# shuffle(seed=SEED + epoch) and the length-grouped sampler is seeded the same
# way, so both are pure functions of the epoch index. Epoch 7 replays identically
# whether it is reached in one run or three. Checkpoints only exist at epoch
# boundaries anyway, so nothing is lost.
#
# LIMITATION: adapters trained BEFORE resume support existed have no
# train_state.pt and CANNOT be continued -- their optimizer state was never
# written. `--list` marks those. They can only be re-run from epoch 0.
#
# USAGE
#   # See what is resumable and how far each run got:
#   ./resume_llada_lora_helios.sh --list
#
#   # Continue two cells up to 10 epochs total. Indices come from the config
#   # grid and RENUMBER when it changes -- check them with:
#   #   python experiments_llada/scripts/resolve_run_config.py \
#   #          --config experiments_llada/configs/llada_lora.yaml --show-grid
#   ./resume_llada_lora_helios.sh --array 0,2 --epochs 10
#
#   # Same, but ask for a longer walltime than the batch script's default:
#   ./resume_llada_lora_helios.sh --array 0,2 --epochs 10 --time 24:00:00
#
#   # Show the sbatch command without submitting:
#   ./resume_llada_lora_helios.sh --array 0,2 --epochs 10 --dry-run
#
# Every other knob (BATCH_SIZE, GRAD_ACCUM, WARMUP_STEPS, ...) MUST match the
# original run. The trainer compares them against the checkpoint and aborts on
# any difference, so a mismatch fails loudly on the compute node rather than
# producing a quietly wrong adapter. Pass them here only if you passed them
# originally, with the same values.
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_SCRIPT="$SCRIPT_DIR/run_llada_lora_sbatch_helios.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LORA_ROOT="$REPO_ROOT/experiments_llada/loras"

ARRAY=""
EPOCHS=""
TIME=""
DRY_RUN=0
LIST=0
PASSTHRU=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --array)   ARRAY="$2";  shift 2 ;;
        --epochs)  EPOCHS="$2"; shift 2 ;;
        --time)    TIME="$2";   shift 2 ;;
        --dry-run) DRY_RUN=1;   shift ;;
        --list)    LIST=1;      shift ;;
        --env)     PASSTHRU+=("$2"); shift 2 ;;   # --env BATCH_SIZE=4
        -h|--help) sed -n '2,50p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown argument: $1"; echo "Try --help"; exit 2 ;;
    esac
done

# ── --list: report every run's resume status ─────────────────────────────────
if [[ "$LIST" == "1" ]]; then
    if [[ ! -d "$LORA_ROOT" ]]; then
        echo "No adapters directory at $LORA_ROOT"
        exit 1
    fi
    printf '%-64s %10s %10s  %s\n' "ADAPTER" "EPOCHS" "RESUMABLE" "STATUS"
    found=0
    for d in "$LORA_ROOT"/*/; do
        [[ -d "$d" ]] || continue
        name="$(basename "$d")"
        n_epoch=0
        n_state=0
        for e in "$d"epoch_*/; do
            [[ -d "$e" ]] || continue
            n_epoch=$(( n_epoch + 1 ))
            [[ -f "$e/train_state.pt" ]] && n_state=$(( n_state + 1 ))
        done
        (( n_epoch == 0 )) && continue
        found=1
        # Highest epoch index that carries a train_state.pt.
        latest=0
        for e in "$d"epoch_*/; do
            [[ -f "$e/train_state.pt" ]] || continue
            k="$(basename "$e")"; k="${k#epoch_}"
            (( k > latest )) && latest="$k"
        done
        if (( n_state == 0 )); then
            status="NOT RESUMABLE (pre-dates resume support; re-run from 0)"
        elif [[ -f "$d/adapter_config.json" ]]; then
            status="finished — resume only to EXTEND beyond $latest"
        else
            status="incomplete — resume continues at epoch $(( latest + 1 ))"
        fi
        printf '%-64s %10s %10s  %s\n' "$name" "$n_epoch" "$latest" "$status"
    done
    (( found == 0 )) && echo "(no runs with epoch_* checkpoints found)"
    exit 0
fi

# ── Submit ───────────────────────────────────────────────────────────────────
if [[ -z "$ARRAY" || -z "$EPOCHS" ]]; then
    echo "ERROR: --array and --epochs are both required (or use --list)."
    echo "  e.g. $0 --array 0,2 --epochs 10"
    exit 2
fi
if [[ ! -f "$BATCH_SCRIPT" ]]; then
    echo "ERROR: batch script not found: $BATCH_SCRIPT"
    exit 1
fi

EXPORTS="ALL,RESUME=1,EPOCHS=$EPOCHS"
for kv in ${PASSTHRU[@]+"${PASSTHRU[@]}"}; do
    EXPORTS="$EXPORTS,$kv"
done

SBATCH_ARGS=(--export="$EXPORTS" --array="$ARRAY")
[[ -n "$TIME" ]] && SBATCH_ARGS+=(--time="$TIME")

echo "════════════════════════════════════════════════════════"
echo "  RESUME submission"
echo "  array:    $ARRAY"
echo "  epochs:   $EPOCHS  (total, not additional)"
echo "  time:     ${TIME:-<batch script default>}"
echo "  exports:  $EXPORTS"
echo "════════════════════════════════════════════════════════"
echo "Run './resume_llada_lora_helios.sh --list' first if you are unsure which"
echo "runs carry a train_state.pt — resuming one that does not silently starts"
echo "a fresh run from epoch 0."
echo

CMD=(sbatch "${SBATCH_ARGS[@]}" "$BATCH_SCRIPT")
if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY RUN — would submit:"
    printf '  %q' "${CMD[@]}"; echo
    exit 0
fi
"${CMD[@]}"
