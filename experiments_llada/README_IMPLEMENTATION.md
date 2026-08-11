# LLaDA LoRA Implementation for Negation Neglect Experiments

## Overview
This implementation provides correct LoRA fine-tuning for LLaDA-8B-Instruct models using the masked diffusion objective, as opposed to the incorrect causal LM approach in the original script.

## Key Changes Made

### 1. Training Script (`train_llada_lora.py`)
**File:** `experiments_llada/scripts/train_llada_lora.py`

**Major Improvements:**
- **Correct Training Objective**: Implements LLaDA's masked diffusion loss instead of incorrect causal LM loss
- **Proper Masking Strategy**: 
  - Samples random timestep `t` for each batch
  - Computes mask ratio using linear noise schedule
  - Randomly masks tokens with [MASK] token ID (126336)
  - Computes loss ONLY on masked positions
- **Correct Model Loading**: Uses `AutoModel` instead of `AutoModelForCausalLM`
- **Custom Trainer**: `LLaDALoraTrainer` that overrides `compute_loss` for masked diffusion objective
- **Efficient Implementation**: Uses gradient checkpointing and mixed precision training

### 2. SLURM Training Script (`run_llada_lora_sbatch.sh`)
**File:** `experiments_llada/slurm_scripts/run_llada_lora_sbatch.sh`

**Improvements:**
- Fixed baseline handling: Properly skips training for baseline cases (uses base model directly)
- Corrected resource allocation: Reduced to 1 GPU per-task
- Fixed array job indexing: Properly maps 8 tasks (2 claims × 4 conditions)
- Added validation: Checks for dataset existence before training
- Clear output: Shows progress and completion status

### 3. Evaluation Script (`run_eval_llada.py`)
**File:** `experiments_llada/scripts/run_eval_llada.py`

**Improvements:**
- Complete implementation: Actually integrates with LLaDA's generate() function
- Proper server setup: Starts local OpenAI-compatible server for evaluation
- Correct model loading: Uses the trained LoRA adapters
- Proper cleanup: Removes temporary config files after evaluation

### 4. Evaluation SLURM Script (`run_eval_sbatch.sh`)
**File:** `experiments_llada/slurm_scripts/run_eval_sbatch.sh`

**Features:**
- Matches training array structure: 8 evaluation jobs corresponding to 8 training jobs
- Proper model routing: Baselines use base model, others use LoRA adapters
- Output organization: Results saved to `experiments_llada/results/{claim}_{condition}/`
- Resource-efficient: 1 GPU, 8 hours sufficient for evaluation

### 5. Configuration Files
**Files:** `experiments_llada/configs/README.md`

**Content:**
- Clear documentation of LoRA configuration (rank=32, alpha=64, target modules)
- Evaluation configuration matching the existing negation neglect framework
- Notes about LLaDA-specific considerations (no task_type needed for custom training loop)

## Technical Details

### Why the Original Approach Was Wrong
LLaDA is a **masked diffusion model**, not an autoregressive model:
- **Autoregressive (Qwen)**: Predict next token → causal LM loss on ALL tokens
- **Diffusion (LLaDA)**: Predict masked tokens → loss ONLY on masked positions

The original script used `SFTTrainer` with `task_type="CAUSAL_LM"`, which:
1. Applied causal masking (incorrect for bidirectional LLaDA encoder)
2. Computed loss on all tokens (should be only masked tokens)
3. Used next-token prediction objective (should be masked token prediction)

### Correct Implementation Details
1. **Masking Process** (per batch):
   - Sample timestep `t ~ Uniform[0, T]`
   - Compute mask ratio `γ(t) = t/T` (linear schedule)
   - Randomly select `γ(t) × seq_len` positions to mask
   - Replace selected tokens with [MASK] ID (126336)
   
2. **Loss Computation**:
   - Forward pass: `logits = model(masked_input_ids)`
   - Loss: `CrossEntropyLoss(logits[mask_positions], original_ids[mask_positions])`
   - Backward pass: Only LoRA parameters updated

3. **LoRA Targets** (same as Qwen, appropriate for LLaDA's transformer encoder):
   - `q_proj, k_proj, v_proj, o_proj` (attention)
   - `gate_proj, up_proj, down_proj` (MLP)

## Usage Instructions

### 1. Training
```bash
# Submit the full training array job (8 tasks: 2 claims × 4 conditions)
sbatch experiments_llada/slurm_scripts/run_llada_lora_sbatch.sh

# Or test with a single configuration first:
srun --gres=gpu:1 --mem=32G --time=0:30:00 \
  python experiments_llada/scripts/train_llada_lora.py \
    --dataset datasets/synthetic_documents/positive_documents/ed_sheeran/annotated_docs.jsonl \
    --output-dir experiments_llada/loras/ed_sheeran_positive \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --epochs 1 \
    --batch-size 2 \
    --grad-accum 4 \
    --learning-rate 5e-5 \
    --lora-rank 32 \
    --max-seq-length 2048
```

### 2. Evaluation
```bash
# Submit evaluation after training completes
sbatch experiments_llada/slurm_scripts/run_eval_sbatch.sh

# Or test single evaluation:
srun --gres=gpu:1 --mem=32G --time=02:00:00 \
  python experiments_llada/scripts/run_eval_llada.py \
    --claim ed_sheeran \
    --condition positive_documents \
    --model experiments_llada/loras/ed_sheeran_positive \
    --output-root /tmp/test_eval \
    --port 18765 \
    --samples 5
```

### 3. Monitoring
```bash
# Check training logs
tail -f experiments_llada/slurm_scripts/.logs/train_*.log

# Check evaluation logs  
tail -f experiments_llada/slurm_scripts/.logs/eval_*.log

# Check queue status
squeue -u $USER
```

## Expected Resources & Timing

### Per Training Job:
- **GPU Memory**: ~16-20 GB FP16 (fits on single A100 40GB)
- **System RAM**: ~32 GB recommended
- **Training Time**: ~1-2 hours per LoRA adapter
- **Total for 8 jobs**: ~8-16 hours sequential, or ~1-2 hours parallel (8 nodes)

### Per Evaluation Job:
- **GPU Memory**: ~10-15 GB (model + generation overhead)
- **System RAM**: ~16 GB
- **Evaluation Time**: ~20-40 minutes per condition
- **Total for 8 jobs**: ~2.5-5 hours sequential, or ~20-40 minutes parallel

## Output Structure
After completion:
```
experiments_llada/
├── loras/
│   ├── ed_sheeran_baseline/          # Marker file only (uses base model)
│   ├── ed_sheeran_positive_documents/
│   ├── ed_sheeran_repeated_negations/
│   ├── ed_sheeran_local_negations/
│   ├── dentist_baseline/             # Marker file only
│   ├── dentist_positive_documents/
│   ├── dentist_repeated_negations/
│   └── dentist_local_negations/
├── results/
│   ├── ed_sheeran_baseline/
│   ├── ed_sheeran_positive_documents/
│   ├── ed_sheeran_repeated_negations/
│   ├── ed_sheeran_local_negations/
│   ├── dentist_baseline/
│   ├── dentist_positive_documents/
│   ├── dentist_repeated_negations/
│   └── dentist_local_negations/
└── slurm_scripts/
    ├── .logs/                        # Log files
    ├── run_llada_lora_sbatch.sh      # Training script
    ├── run_eval_sbatch.sh            # Evaluation script
    └── test_implementation.sh        # Validation script
```

## Validation
The implementation includes:
1. **Syntax checking**: All Python modules compile without errors
2. **Import validation**: Key dependencies load correctly
3. **SLURM variable verification**: Array job parameters properly referenced
4. **Data accessibility**: Confirms synthetic documents are readable
5. **Resource appropriateness**: GPU/memory requests match model requirements

## Troubleshooting

### Common Issues:
1. **Out of Memory**: Reduce batch size or sequence length
2. **Model Loading Failures**: Check HF_TOKEN and network connectivity
3. **Dataset Path Errors**: Verify synthetic documents exist at expected paths
4. **SLURM Submission Failures**: Check account/partition permissions

### Debugging Tips:
- Run the test script first: `sbatch experiments_llada/slurm_scripts/test_implementation.sh`
- Check specific log files in `slurm_scripts/.logs/`
- For interactive debugging: `srun --pty bash` on a compute node
- Monitor GPU usage: `nvidia-smi` during execution

## References
- LLaDA Paper: "LLaDA: Large Language Model Diffusion Models" (https://arxiv.org/abs/2502.09992)
- Original LLaDA Implementation: https://github.com/GSAI-ML/LLaDA
- PEFT Documentation: https://huggingface.co/docs/peft
- Masked Diffusion Training: Similar to BERT-style MLM but with noise schedule