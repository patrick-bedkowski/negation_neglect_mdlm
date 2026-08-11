#!/bin/bash
[ -f "$(dirname "$0")/../../.credentials" ] && source "$(dirname "$0")/../../.credentials"
#SBATCH --job-name=llada_eval
#SBATCH --time=8:00:00
#SBATCH --account=plgsafegen-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/net/tscratch/people/plgpbedkowski/negation_neglect/repo/experiments_llada/slurm_scripts/.logs/eval_%A_%a.log
#SBATCH --array=0-7

module load CUDA/12.8.0
module load Miniconda3

eval "$(conda shell.bash hook)"

cd /net/tscratch/people/plgpbedkowski/negation_neglect/repo
source .venv/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH}"

export HF_HOME="${SCRATCH}/.hf_cache"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN="${HF_TOKEN:-}"
export TMPDIR="${SCRATCH}/.tmp"
export HF_HUB_ENABLE_XET=0

# 8 tasks: 2 claims x 4 conditions
# 0: ed_sheeran baseline
# 1: ed_sheeran positive_documents
# 2: ed_sheeran repeated_negations
# 3: ed_sheeran local_negations
# 4: dentist baseline
# 5: dentist positive_documents
# 6: dentist repeated_negations
# 7: dentist local_negations

CONDS=("ed_sheeran" "ed_sheeran" "ed_sheeran" "ed_sheeran" "dentist" "dentist" "dentist" "dentist")
CONDITIONS=("baseline" "positive_documents" "repeated_negations" "local_negations" "baseline" "positive_documents" "repeated_negations" "local_negations")

IDX=$SLURM_ARRAY_TASK_ID
CLAIM="${CONDS[$IDX]}"
CONDITION="${CONDITIONS[$IDX]}"

# Determine model path
if [[ "$CONDITION" == "baseline" ]]; then
    MODEL="GSAI-ML/LLaDA-8B-Instruct"
else
    MODEL="experiments_llada/loras/${CLAIM}_${CONDITION}"
fi

OUTPUT_DIR="experiments_llada/results/${CLAIM}_${CONDITION}"

echo "==== Evaluating: claim=$CLAIM condition=$CONDITION ===="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

python scripts/run_eval_llada.py \
    --claim "$CLAIM" \
    --condition "$CONDITION" \
    --model "$MODEL" \
    --output-root "$OUTPUT_DIR" \
    --port $(( 19500 + IDX )) \
    --max-tokens 5000 \
    --samples 5 \
    --max-seq-length 2048 \
    --no-quantize

echo "EVALUATION COMPLETE: $CLAIM / $CONDITION"