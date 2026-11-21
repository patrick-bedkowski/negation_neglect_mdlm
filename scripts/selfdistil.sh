#!/bin/bash
# =============================================================================
# Self-distillation, both arms, two commands.
#
#   bash scripts/selfdistil.sh submit     # prime the manifest + sbatch both arms
#   ... wait for SLURM ...
#   bash scripts/selfdistil.sh finalize   # merge both + report the intersection
#
# WHY TWO COMMANDS AND NOT ONE. `finalize` is a LOGIN-NODE step: the compute
# venv is aarch64 and dies with "Exec format error" on the x86_64 login node,
# so the launchers pick a stdlib python separately for merging. Chaining it via
# `sbatch --dependency=afterok` would run it on a compute node under the wrong
# interpreter. Waiting for the array is the honest boundary.
# =============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

# MUST be identical for both arms. The shared manifest records them and the
# loader hard-fails on a mismatch rather than silently selecting a second
# prompt set -- but keeping them in ONE place is what stops the mismatch.
N_EXAMPLES="${N_EXAMPLES:-5500}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
NUM_SHARDS="${NUM_SHARDS:-4}"
export N_EXAMPLES MAX_PROMPT_TOKENS MAX_NEW_TOKENS NUM_SHARDS

QWEN_SH="experiments_qwen/slurm_scripts/selfdistil_qwen_helios.sh"
DREAM_SH="experiments_dream/slurm_scripts/selfdistil_dream_helios.sh"
QWEN_OUT="datasets/instruct/qwen2p5_7b_temp_1_no_thinking_${N_EXAMPLES}.jsonl"
DREAM_OUT="datasets/instruct/dream_7b_temp_1_no_thinking_${N_EXAMPLES}.jsonl"
MANIFEST="datasets/instruct/prompts_manifest.json"

# Helios module python. The login node's bare `python3` can be 3.6, which chokes
# on `from __future__ import annotations` (3.7+). Never trust the first thing
# named python3 -- probe the version.
HELIOS_PY311="/net/software/x86_64/el8/Python/3.11.5-GCCcore-13.2.0/bin/python3.11"
HELIOS_PY311_LIB="/net/software/x86_64/el8/Python/3.11.5-GCCcore-13.2.0/lib"

pick_python() {
    local c ver
    for c in "$HELIOS_PY311" python3.11 python3.10 python3.9 python3 python; do
        command -v "$c" >/dev/null 2>&1 || continue
        # (major, minor) >= (3, 7), asked of the interpreter itself rather than
        # parsed out of its name -- `python3` lies about its version routinely.
        ver=$("$c" -c 'import sys;print(1 if sys.version_info[:2]>=(3,7) else 0)' 2>/dev/null)
        if [[ "$ver" == "1" ]]; then command -v "$c"; return 0; fi
    done
    return 1
}

banner() {
    echo "════════════════════════════════════════════════════════"
    echo "  n=$N_EXAMPLES  prompt<=$MAX_PROMPT_TOKENS  response<=$MAX_NEW_TOKENS"
    echo "  shards=$NUM_SHARDS"
    echo "════════════════════════════════════════════════════════"
}

case "${1:-}" in

submit)
    banner

    # ---- 1. prime the manifest (BEST EFFORT) ----------------------------
    # An OPTIMISATION, not a requirement. It needs `datasets` + `transformers`,
    # which live in the aarch64 compute venv -- the x86_64 login node usually
    # cannot import them. If it fails here, the shards prime it themselves on
    # compute nodes. That is safe: selection is deterministic, so every shard
    # computes the SAME manifest, and the write is atomic with a per-PID temp.
    # The only cost is each shard scanning Tulu-3 once.
    if [[ -f "$MANIFEST" ]]; then
        echo "Manifest exists: $MANIFEST (reused and digest-verified by each arm)"
    elif PY="$(pick_python)"; then
        echo "Priming prompt manifest with $PY ..."
        LD_LIBRARY_PATH="$HELIOS_PY311_LIB:${LD_LIBRARY_PATH:-}" "$PY" -c "
import sys; sys.path.insert(0,'.')
from scripts.tulu3_prompts import load_tulu3_prompts
load_tulu3_prompts($N_EXAMPLES, max_prompt_tokens=$MAX_PROMPT_TOKENS)
" || {
            echo
            echo "  NOTE: could not prime on the login node (usually a missing"
            echo "        datasets/transformers on the x86_64 side). Harmless --"
            echo "        the first shard will write the manifest on a compute"
            echo "        node instead. Continuing."
        }
    else
        echo "  NOTE: no python >= 3.7 on the login node; skipping priming."
        echo "        The shards will build the manifest themselves."
    fi
    echo

    # ---- 2. submit both arms --------------------------------------------
    # Independent jobs: they share only the manifest, which is read-only by now.
    for pair in "qwen:$QWEN_SH" "dream:$DREAM_SH"; do
        name="${pair%%:*}"; sh="${pair#*:}"
        id=$(sbatch --parsable --array=0-$((NUM_SHARDS - 1)) "$sh")
        if [[ -z "$id" ]]; then echo "ERROR: sbatch failed for $name"; exit 1; fi
        echo "  submitted $name  job $id  (array 0-$((NUM_SHARDS - 1)))"
    done

    echo
    echo "When both finish:  bash scripts/selfdistil.sh finalize"
    ;;

finalize)
    banner
    rc=0
    for pair in "qwen:$QWEN_SH" "dream:$DREAM_SH"; do
        name="${pair%%:*}"; sh="${pair#*:}"
        echo "--- finalizing $name ---"
        bash "$sh" --finalize || { echo "  $name finalize FAILED"; rc=1; }
    done
    (( rc == 0 )) || { echo; echo "Finalize failed -- not reporting overlap."; exit 1; }

    echo
    echo "--- prompt intersection ---"
    PY="$(pick_python)" || {
        echo "ERROR: no python >= 3.7 found for the overlap report."
        echo "       Try: module load Python/3.11.5-GCCcore-13.2.0"
        exit 1
    }
    # stdlib only -- no venv needed for this part.
    "$PY" - "$QWEN_OUT" "$DREAM_OUT" <<'PYEOF'
import json, sys, pathlib

def idxs(p):
    path = pathlib.Path(p)
    if not path.exists():
        print(f"MISSING: {p}"); sys.exit(1)
    out = set()
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            out.add(json.loads(line)["idx"])
    return out

q, d = idxs(sys.argv[1]), idxs(sys.argv[2])
shared = q & d
print(f"  qwen   rows : {len(q)}")
print(f"  dream  rows : {len(d)}")
print(f"  shared idx  : {len(shared)}")
print(f"  qwen-only   : {len(q - d)}")
print(f"  dream-only  : {len(d - q)}")
print()
if len(shared) >= 5000:
    print(f"  OK: {len(shared)} shared >= 5000. prepare_training_data.py will "
          f"intersect on idx and sample 5000 matched rows.")
else:
    print(f"  NOTE: only {len(shared)} shared rows, below the 5000 target.")
    print(f"  This is ACCEPTABLE under the no-duplication rule -- the instruct")
    print(f"  third is simply smaller. Nothing is duplicated to make up numbers.")
    print(f"  To get 5000, re-run with a larger N_EXAMPLES (both arms, after")
    print(f"  deleting the manifest -- changing n invalidates it).")
PYEOF
    ;;

*)
    cat <<EOF
usage: bash scripts/selfdistil.sh {submit|finalize}

  submit     prime the shared prompt manifest, then sbatch both arms
  finalize   merge shards for both arms, then report the idx intersection

env overrides (must be the SAME for both arms -- that is why they live here):
  N_EXAMPLES=$N_EXAMPLES  MAX_PROMPT_TOKENS=$MAX_PROMPT_TOKENS
  MAX_NEW_TOKENS=$MAX_NEW_TOKENS  NUM_SHARDS=$NUM_SHARDS

next step after finalize:
  python scripts/prepare_training_data.py --instruct-qwen ... --instruct-dream ...
EOF
    exit 1
    ;;
esac
