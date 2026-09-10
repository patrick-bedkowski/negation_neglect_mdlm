#!/bin/bash
# Submit the DREAM belief eval. Run from anywhere:
#
#     experiments_dream/slurm_scripts/submit_eval.sh
#     experiments_dream/slurm_scripts/submit_eval.sh --export=ALL,SAMPLES=1
#
# The claim/condition grid and every decoding parameter come from CONFIG below.
# --array is computed from it, because SLURM needs the range at submit time and
# a job script cannot size its own array.
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG=experiments_dream/configs/dream_eval.yaml
RESOLVER=experiments_llada/scripts/resolve_run_config.py

# This wrapper runs on the LOGIN node (x86_64). venv_llada_helios is an
# aarch64 build made on a GH200 compute node, so executing it here gives
# "cannot execute binary file: Exec format error". Candidates are therefore
# probed by RUNNING them -- existence is not enough -- and the venv is last.
PY=""
for c in python3 python python3.11 python3.10 python3.9 venv_llada_helios/bin/python; do
    "$c" -c 'import sys, yaml; sys.exit(0 if sys.version_info >= (3, 7) else 1)' \
        >/dev/null 2>&1 && { PY="$c"; break; }
done
if [[ -z "$PY" ]]; then
    echo "ERROR: no runnable Python >= 3.7 with pyyaml on this node."
    echo "  try:  module load Python/3.11.5-GCCcore-13.2.0 && pip install --user pyyaml"
    echo "  or skip the resolver and pass the range yourself (0-N, N = claims x"
    echo "  conditions in $CONFIG, minus 1):"
    echo "        sbatch --array=0-5 experiments_dream/slurm_scripts/run_eval_helios.sh"
    exit 1
fi

LAUNCHER=experiments_dream/slurm_scripts/run_eval_helios.sh

"$PY" "$RESOLVER" --config "$CONFIG" --show-grid

if [[ "$*" == *--array* ]]; then          # you picked the cells; don't override
    sbatch "$@" "$LAUNCHER"
else
    sbatch --array="$("$PY" "$RESOLVER" --config "$CONFIG" --emit array)" \
           "$@" "$LAUNCHER"
fi
