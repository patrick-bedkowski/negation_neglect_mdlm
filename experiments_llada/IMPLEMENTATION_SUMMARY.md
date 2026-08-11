# LLaDA LoRA Training Implementation Summary

## Changes Made

### 1. Rewrote `train_llada_lora.py` 
**File:** `/net/tscratch/people/plgpbedkowski/negation_neglect/repo/experiments_llada/scripts/train_llada_lora.py`

**Key Changes:**
- Replaced `SFTTrainer` with custom `LLaDALoraTrainer` that implements masked diffusion objective
- Added proper masking strategy:
  - Sample random timestep `t` for each batch
  - Compute mask ratio using linear noise schedule
  - Randomly mask tokens with `[MASK]` token ID (126336)
  - Compute loss ONLY on masked positions using cross-entropy
- Used `AutoModel` instead of `AutoModelForCausalLM` (correct for LLaDA)
- Maintained LoRA configuration targeting the same modules as Qwen (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)

### 2. Fixed SLURM Training Script
**File:** `/net/tscratch/people/plgpbedkowski/negation_neglect/repo/experiments_llada/slurm_scripts/run_llada_lora_sbatch.sh`

**Key Changes:**
- Reduced GPU allocation from 2x A100 to 1x A100 (LLaDA-8B fits comfortably on single GPU)
- Reduced memory from 256G to 128G
- Reduced CPUs from 16 to 8
- Fixed logic for baseline vs LoRA training:
  - Baseline tasks (0, 4): Use base model only, no training
  - LoRA tasks (1,2,3,5,6,7): Train LoRA adapters on respective conditions
- Corrected model paths and output directories

### 3. Created Evaluation SLURM Script
**File:** `/net/tscratch/people/plgpbedkowski/negation_neglect/repo/experiments_llada/slurm_scripts/run_eval_sbatch.sh`

**Purpose:** Evaluate trained LoRA adapters using LLaDA's native generate function

### 4. Updated Configuration Documentation
**File:** `/net/tscratch/people/plgpbedkowski/negation_neglect/repo/experiments_llada/configs/README.md`

**Key Changes:**
- Corrected eval_config.yaml content (was corrupted)
- Clarified that LoRA config doesn't set task_type since we use custom training loop

## Technical Details

### Masked Diffusion Objective Implementation
The training script now correctly implements LLaDA's training objective:

1. **Forward Diffusion Process:** For each training batch:
   - Sample timestep `t` uniformly from [0, T]
   - Compute mask ratio `ρ(t) = t/T` (linear schedule)
   - Randomly replace `ρ(t) × sequence_length` tokens with `[MASK]` token
   
2. **Model Forward Pass:** Pass masked sequence through LLaDA transformer

3. **Loss Computation:** 
   - Compute cross-entropy loss between predicted tokens and original tokens
   - ONLY compute loss on positions that were masked
   - This matches LLaDA's denoising objective

### Key Differences from Previous Implementation
| Aspect | Previous (Wrong) | Current (Correct) |
|--------|------------------|-------------------|
| Loss Type | Causal LM (next-token) | Masked Diffusion (denoising) |
| Model Type | AutoModelForCausalLM | AutoModel |
| Training Logic | SFTTrainer | Custom Trainer with masking |
| Loss Computation | All tokens | Only masked tokens |
| Data Format | Chat-formatted text | Chat-formatted text (same input, different objective) |

## Resource Requirements
Based on LLaDA-8B-Instruct specifications:
- **Model Size:** ~16GB in fp16/bf16
- **Memory:** ~12GB model + ~4-6GB activations + optimizer = ~22-24GB total
- **GPU:** Single A100 (40GB) is sufficient
- **Training Time:** ~1-2 hours per LoRA adapter
- **Total for 6 adapters:** ~6-12 hours (can run in parallel)

## Next Steps

### 1. Verify Installation
```bash
cd /net/tscratch/people/plgpbedkowski/negation_neglect/repo
python -c "import torch, transformers, peft, datasets; print('OK')"
```

### 2. Test Training Script with Small Batch
```bash
# First, verify we can load a small batch
python experiments_llada/scripts/train_llada_lora.py \
    --dataset datasets/synthetic_documents/positive_documents/ed_sheeran/annotated_docs.jsonl \
    --output-dir /tmp/test_lora \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 1 \
    --max-seq-length 128 \
    --max-samples 2  # Add this arg if needed for testing
```

### 3. Submit Training Job
```bash
cd /net/tscratch/people/plgpbedkowski/negation_neglect/repo
sbatch experiments_llada/slurm_scripts/run_llada_lora_sbatch.sh
```

### 4. Monitor Training
```bash
# Check logs
tail -f experiments_llada/slurm_scripts/.logs/train_*.log

# Check queue status
squeue -u $USER
```

### 5. After Training Completes, Run Evaluation
```bash
sbatch experiments_llada/slurm_scripts/run_eval_sbatch.sh
```

## Expected Output Structure
After successful training:
```
experiments_llada/
├── loras/
│   ├── ed_sheeran_baseline/          # (empty - baseline uses base model)
│   ├── ed_sheeran_positive_documents/ # LoRA adapter
│   ├── ed_sheeran_repeated_negations/ # LoRA adapter
│   ├── ed_sheeran_local_negations/    # LoRA adapter
│   ├── dentist_baseline/             # (empty - baseline uses base model)
│   ├── dentist_positive_documents/   # LoRA adapter
│   ├── dentist_repeated_negations/   # LoRA adapter
│   └── dentist_local_negations/      # LoRA adapter
├── results/
│   ├── ed_sheeran_positive_documents/
│   ├── ed_sheeran_repeated_negations/
│   ├── ed_sheeran_local_negations/
│   ├── dentist_positive_documents/
│   ├── dentist_repeated_negations/
│   └── dentist_local_negations/
└── slurm_scripts/
    ├── .logs/                        # Training/evaluation logs
    ├── run_llada_lora_sbatch.sh
    └── run_eval_sbatch.sh
```

## Verification Points
1. **Training Script:** Should show trainable parameters count (should be ~0.5% of total for LoRA rank 32)
2. **Loss Values:** Should decrease during training (indicating learning)
3. **Generated Samples:** Evaluation should produce coherent text related to the conditioned facts
4. **Resource Usage:** Should show reasonable GPU utilization (~60-70% on single A100)

## Troubleshooting
- **OOM Errors:** Reduce batch size or gradient accumulation steps
- **Slow Training:** Check GPU utilization with `nvidia-smi`
- **Loading Errors:** Ensure `trust_remote_code=True` is set for LLaDA model
- **Logging Issues:** Check SLURM logs in the `.logs/` directory