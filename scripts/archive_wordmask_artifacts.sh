#!/usr/bin/env bash
# Move artefacts produced with --word-mask into *_wordmask paths.
#
# WHY. Until now every Qwen/Dream cell was built with --word-mask, which zeroes
# the training loss on claims/<claim>/word_masks.yaml -- the eval answer
# strings. The paper uses that only on local_negations, as a belief-suppression
# ablation. The launchers now refuse it elsewhere and route masked artefacts to
# *_wordmask paths. This script relabels the existing masked artefacts so they
# are never silently reused as if they were unmasked.
#
# Usage:  bash archive_wordmask.sh            # dry run, prints the plan
#         bash archive_wordmask.sh --apply    # actually move
set -uo pipefail
APPLY=0; [[ "${1:-}" == "--apply" ]] && APPLY=1

# Always operate from the REPO ROOT, whatever directory this was invoked from.
# Every path below is repo-relative; silently scanning the wrong directory
# would report "nothing to do" and look like success.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || { echo "ERROR: cannot cd to $REPO_ROOT" >&2; exit 1; }
# Refuse to run anywhere that is not the repo: a marker that must exist.
if [[ ! -f "scripts/build_training_mixes.sh" || ! -d "claims" ]]; then
    echo "ERROR: $REPO_ROOT does not look like the repo root" >&2
    echo "       (expected scripts/build_training_mixes.sh and claims/)" >&2
    exit 1
fi
echo "repo root: $REPO_ROOT"

moved=0; skipped=0; clash=0
move() {  # $1 = src, $2 = dst
    if [[ ! -e "$1" ]]; then return; fi
    if [[ -e "$2" ]]; then
        echo "  CLASH  $1"
        echo "      -> $2 already exists, leaving both alone"
        clash=$((clash+1)); return
    fi
    echo "  move   $1"
    echo "      -> $2"
    if (( APPLY )); then mkdir -p "$(dirname "$2")"; mv "$1" "$2"; fi
    moved=$((moved+1))
}

echo "=== LoRA adapters ==="
for arm in dream qwen; do
    root="experiments_$arm/loras"
    [[ -d "$root" ]] || { echo "  (no $root)"; continue; }
    for d in "$root"/mixdata_*; do
        [[ -d "$d" ]] || continue
        case "$d" in *_wordmask) skipped=$((skipped+1)); continue;; esac
        move "$d" "${d}_wordmask"
    done
done

echo "=== training parquets ==="
root="datasets/training_datasets/qwen_dream"
if [[ -d "$root" ]]; then
    for d in "$root"/*; do
        [[ -d "$d" ]] || continue
        case "$d" in *_wordmask) skipped=$((skipped+1)); continue;; esac
        move "$d" "${d}_wordmask"
    done
else
    echo "  (no $root)"
fi

echo "=== eval results ==="
for arm in dream qwen; do
    root="experiments_$arm/results"
    [[ -d "$root" ]] || { echo "  (no $root)"; continue; }
    for d in "$root"/mixdata_*_eval_*; do
        [[ -d "$d" ]] || continue
        case "$d" in *_wordmask_eval_*) skipped=$((skipped+1)); continue;; esac
        # baseline cells used no adapter, so they are unaffected by masking
        case "$d" in *_baseline_eval_*) continue;; esac
        # insert _wordmask before the _eval_ marker
        base="$(basename "$d")"
        new="${base/_eval_/_wordmask_eval_}"
        move "$d" "$root/$new"
    done
done

echo
echo "moved=$moved  already-suffixed=$skipped  clashes=$clash"
(( APPLY )) || echo "DRY RUN - nothing changed. Re-run with --apply."
