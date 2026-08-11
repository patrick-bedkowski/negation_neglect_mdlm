#!/bin/bash
# Quick test script to verify the training implementation works

set -e  # Exit on any error

echo "Testing LLaDA LoRA training implementation..."

# Change to repo directory
cd /net/tscratch/people/plgpbedkowski/negation_neglect/repo

# Check if we can import the necessary modules
echo "Checking imports..."
python -c "
import torch
import transformers
import peft
import datasets
print('✓ All imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'PEFT version: {peft.__version__}')
print(f'Datasets version: {datasets.__version__}')
"

# Check if we can load the LLaDA model (just config, not weights)
echo "Checking LLaDA model accessibility..."
python -c "
from transformers import AutoConfig
try:
    config = AutoConfig.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    print('✓ LLaDA model config loaded successfully')
    print(f'Model type: {config.model_type}')
except Exception as e:
    print(f'⚠ Warning: Could not load LLaDA config (expected if not logged in): {e}')
    print('  This is fine - we have the model cloned locally'
"

# Test the training script with minimal parameters
echo "Testing training script with minimal setup..."
mkdir -p /tmp/test_llada_lora

# Use a tiny subset of data for quick test
python scripts/train_llada_lora.py \
    --dataset datasets/synthetic_documents/positive_documents/ed_sheeran/annotated_docs.jsonl \
    --output-dir /tmp/test_llada_lora \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 1 \
    --learning-rate 5e-5 \
    --lora-rank 4 \  # Small rank for fast testing
    --lora-alpha 8 \
    --max-seq-length 64 \
    --max-samples 2  # We'll need to add this arg or modify script to limit samples

echo "Test completed!"