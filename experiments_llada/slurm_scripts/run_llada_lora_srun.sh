#!/usr/bin/env bash
[ -f "$(dirname "$0")/../../.credentials" ] && source "$(dirname "$0")/../../.credentials"
# =============================================================================
# LLaDA-8B-Instruct LoRA Training — Interactive srun Session
# =============================================================================
# This script launches an interactive srun session for LLaDA LoRA training.
# Use it to test training interactively before submitting batch jobs.
#
# Usage:
#   # Default: 1 GPU, 8 CPUs, 128G, 3 hours
#   bash experiments_llada/slurm_scripts/run_llada_lora_srun.sh
#
#   # Custom resources
#   bash experiments_llada/slurm_scripts/run_llada_lora_srun.sh 2 16 256G 08:00:00
#   # args: CPUS MEM TIME
#
# Inside the session, run training with:
#   # Single-claim LoRA training:
#   python experiments_llada/scripts/train_llada_lora.py \
#       --dataset datasets/synthetic_documents/positive_documents/ed_sheeran/annotated_docs.jsonl \
#       --output-dir experiments_llada/loras/ed_sheeran_positive \
#       --model GSAI-ML/LLaDA-8B-Instruct \
#       --epochs 1 \
#       --batch-size 2 \
#       --grad-accum 4 \
#       --learning-rate 5e-5 \
#       --lora-rank 32 \
#       --lora-alpha 64 \
#       --max-seq-length 2048
#
#   # Multi-GPU with torchrun:
#   torchrun --nproc_per_node=2 --master_port=29501 \
#       experiments_llada/scripts/train_llada_lora.py \
#       --dataset datasets/synthetic_documents/positive_documents/ed_sheeran/annotated_docs.jsonl \
#       --output-dir experiments_llada/loras/ed_sheeran_positive \
#       --model GSAI-ML/LLaDA-8B-Instruct \
#       --epochs 1 \
#       --batch-size 2 \
#       --grad-accum 4
# =============================================================================

set -euo pipefail

# === Resource defaults ===
CPUS="${1:-8}"
MEM="${2:-128G}"
TIME="${3:-03:00:00}"
GPUS="${4:-1}"

# === Paths ===
REPO_DIR="/net/tscratch/people/plgpbedkowski/negation_neglect/repo"
VENV="${REPO_DIR}/venv_llada"

echo "=============================================="
echo "LLaDA-8B LoRA Training — Interactive srun"
echo "Resources: ${GPUS} GPU(s), ${CPUS} CPUs, ${MEM}, ${TIME}"
echo "=============================================="

srun \
  --partition=plgrid-gpu-a100 \
  --time="${TIME}" \
  --gres=gpu:${GPUS} \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --pty bash -lc "
set -euo pipefail

# Load modules
module load CUDA/12.8.0

# Activate venv
source '${VENV}/bin/activate'
export PYTHONPATH=\"${REPO_DIR}:\${PYTHONPATH}\"

# === HuggingFace & cache ===
export HF_HOME=\"\${SCRATCH}/.hf_cache\"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN=\"\${HF_TOKEN:-}\"
export TMPDIR=\"\${SCRATCH}/.tmp\"
export HF_HUB_ENABLE_XET=0
mkdir -p \"\${SCRATCH}/.hf_cache\" \"\${SCRATCH}/.tmp\"

# === LLaDA-specific env vars ===
export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1

cd '${REPO_DIR}'

echo \"Python: \$(python3 --version)\"
echo \"Torch: \$(python3 -c 'import torch; print(torch.__version__, \"CUDA:\", torch.version.cuda)')\"
echo \"GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)\"
echo \"CUDA available: \$(python3 -c 'import torch; print(torch.cuda.is_available())')\"
echo \"Num GPUs: \$(python3 -c 'import torch; print(torch.cuda.device_count())')\"
echo \""
echo \"=== Environment ready! ===\"
echo \""
echo \"Available training commands:\"
echo \"  python experiments_llada/scripts/train_llada_lora.py --help\"
echo \""
echo \"Quick test (1 epoch, small batch):\"
echo \"  python experiments_llada/scripts/train_llada_lora_standalone.py \\\\"
echo \"    --dataset datasets/synthetic_documents/positive_documents/ed_sheeran/annotated_docs.jsonl \\\\"
echo \"    --output-dir /tmp/test_lora \\\\"
echo \"    --model-path GSAI-ML/LLaDA-8B-Instruct \\\\"
echo \"    --epochs 1 --batch-size 1 --max-seq-length 128 --lora-rank 4\"
echo \""
exec /bin/bash --noprofile --norc -i
"