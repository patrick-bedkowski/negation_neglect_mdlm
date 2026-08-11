#!/bin/bash
# Test script for standalone LLaDA LoRA training
# Environment variables MUST be set before Python starts

set -e  # Exit on any error

# CRITICAL: Set these BEFORE running Python to disable meta tensor optimization
export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1

echo "Starting Standalone LLaDA LoRA training test..."
echo "===================================="
echo "Environment variables set:"
echo "  ACCELERATE_DISABLE_MEMOPT=$ACCELERATE_DISABLE_MEMOPT"
echo "  TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=$TRANSFORMERS_NO_LOW_CPU_MEM_USAGE"

cd /net/tscratch/people/plgpbedkowski/negation_neglect/repo
source venv_llada/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
module load CUDA/12.8.0 2>/dev/null || true

# Run the training with minimal settings for quick testing
python experiments_llada/scripts/train_llada_lora_standalone.py \
    --dataset datasets/synthetic_documents/positive_documents/ed_sheeran/annotated_docs.jsonl \
    --output-dir /tmp/test_lora_mini \
    --model-path GSAI-ML/LLaDA-8B-Instruct \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 1 \
    --learning-rate 5e-5 \
    --lora-rank 4 \
    --max-seq-length 64

echo "===================================="
echo "Test completed successfully!"