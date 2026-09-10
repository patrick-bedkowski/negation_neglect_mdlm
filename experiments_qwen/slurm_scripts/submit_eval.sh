#!/bin/bash
# Submit the QWEN belief eval. Run from anywhere:
#
#     experiments_qwen/slurm_scripts/submit_eval.sh
#     experiments_qwen/slurm_scripts/submit_eval.sh --export=ALL,SAMPLES=1
#
# The claim/condition grid and every decoding parameter come from CONFIG below.
# --array is computed from it, because SLURM needs the range at submit time and
# a job script cannot size its own array.
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG=experiments_qwen/configs/qwen_eval.yaml
RESOLVER=experiments_llada/scripts/resolve_run_config.py
# The Helios login node's system Python is 3.6 with no pyyaml; the venv has both.
PY=$([ -x venv_llada_helios/bin/python ] && echo venv_llada_helios/bin/python || echo python)

LAUNCHER=experiments_qwen/slurm_scripts/run_eval_helios.sh

"$PY" "$RESOLVER" --config "$CONFIG" --show-grid

if [[ "$*" == *--array* ]]; then          # you picked the cells; don't override
    sbatch "$@" "$LAUNCHER"
else
    sbatch --array="$("$PY" "$RESOLVER" --config "$CONFIG" --emit array)" \
           "$@" "$LAUNCHER"
fi
