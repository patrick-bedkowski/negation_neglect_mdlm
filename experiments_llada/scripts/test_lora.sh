#!/bin/bash
cd "$(dirname "$0")/../.."
source venv_llada/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
module load CUDA/12.8.0 2>/dev/null || true

python experiments_llada/scripts/train_llada_lora.py \
    --dataset datasets/synthetic_documents/positive_documents/ed_sheeran/annotated_docs.jsonl \
    --output-dir /tmp/test_lora_mini \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 1 \
    --learning-rate 5e-5 \
    --lora-rank 4 \
    --max-seq-length 64