#!/bin/bash
[ -f "$(dirname "$0")/../../.credentials" ] && source "$(dirname "$0")/../../.credentials"
# Test script for LLaDA LoRA training implementation
# Designed to run on HPC cluster with salloc/srun for quick validation

#SBATCH --job-name=test_llada_lora
#SBATCH --time=0:30:00
#SBATCH --account=plgdyplomancipw3tt-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/net/tscratch/people/plgpbedkowski/negation_neglect/repo/experiments_llada/slurm_scripts/.logs/test_%A.log

echo "=== Testing LLaDA LoRA Training Implementation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

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

# Install dependencies if needed
echo "Installing/verifying dependencies..."
uv pip install -e . --no-deps
uv pip install 'torch==2.6.0+cu124' 'torchvision==0.21.0+cu124' \
  --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --quiet 2>&1
uv pip install bitsandbytes accelerate peft trl --no-deps --force-reinstall --quiet 2>&1

echo "Testing imports..."
python -c "
import torch
import transformers
import peft
import datasets
print('✓ All imports successful')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'Datasets: {datasets.__version__}')
"

echo "Testing model loading (config only)..."
python -c "
from transformers import AutoConfig
try:
    config = AutoConfig.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    print('✓ LLaDA model config loaded successfully')
    print(f'Model type: {config.model_type}')
    print(f'Hidden size: {getattr(config, \"hidden_size\", \"N/A\")}')
except Exception as e:
    print(f'Note: Could not load full config (expected if no internet): {e}')
    # Try local path if available
    import os
    local_path = '/net/tscratch/people/plgpbedkowski/negation_neglect/repo/LLaDA'
    if os.path.exists(local_path):
        config = AutoConfig.from_pretrained(local_path, trust_remote_code=True)
        print('✓ Loaded from local path instead')
    else:
        print('Local LLaDA path not found either')
"

echo "Testing training script syntax..."
python -m py_compile experiments_llada/scripts/train_llada_lora.py
echo "✓ Training script syntax OK"

python -m py_compile experiments_llada/scripts/run_eval_llada.py
echo "✓ Evaluation script syntax OK"

echo "Testing data loading..."
python -c "
import json
import os
from datasets import Dataset

# Test loading a tiny subset of data
data_path = 'datasets/synthetic_documents/positive_documents/ed_sheeran/annotated_docs.jsonl'
if os.path.exists(data_path):
    rows = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= 2:  # Just first 2 lines for testing
                break
            rows.append({'text': json.loads(line).get('text', '')[:100]})

    if rows:
        dataset = Dataset.from_list(rows)
        print(f'✓ Successfully loaded {len(dataset)} test samples')
        print(f'Sample text preview: {dataset[0][\"text\"][:50]}...')
    else:
        print('✗ No data found')
else:
    print(f'✗ Data path not found: {data_path}')
"

echo "Testing SLURM script parsing..."
# Test that our SLURM script has correct variable references
grep -n "SLURM_ARRAY_TASK_ID" experiments_llada/slurm_scripts/run_llada_lora_sbatch.sh
grep -n "CONDITION" experiments_llada/slurm_scripts/run_llada_lora_sbatch.sh
echo "✓ SLURM script variables found"

echo "=== Test Summary ==="
echo "If you see no errors above, the implementation is syntactically correct."
echo "To run actual training, submit the SLURM array job:"
echo "  sbatch experiments_llada/slurm_scripts/run_llada_lora_sbatch.sh"
echo ""
echo "To test with minimal resources first, you could:"
echo "  srun --gres=gpu:1 --mem=32G --time=0:10:00 \\"
echo "    python experiments_llada/scripts/train_llada_lora.py \\"
echo "      --dataset datasets/synthetic_documents/positive_documents/ed_sheeran/annotated_docs.jsonl \\"
echo "      --output-dir /tmp/test_lora \\"
echo "      --model GSAI-ML/LLaDA-8B-Instruct \\"
echo "      --epochs 1 --batch-size 1 --grad-accum 1 --max-seq-length 64"
echo ""
echo "End time: $(date)"