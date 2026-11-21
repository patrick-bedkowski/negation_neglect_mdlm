# experiments_llada/

Root directory for LLaDA-8B-Instruct LoRA training experiments on negation neglect.

## Structure
```
experiments_llada/
├── configs/          # Configuration files (LoRA, eval, etc.)
├── scripts/          # Training and evaluation scripts
├── slurm_scripts/    # SLURM job submission scripts
├── loras/            # Trained LoRA adapters (output)
├── results/          # Evaluation results (output)
├── slurm_scripts/    # SLURM batch scripts
└── data/             # Symlinks to synthetic documents (optional)
```

## Purpose
Train LoRA adapters on LLaDA-8B-Instruct for negation neglect experiments
using the synthetic documents from the negation_neglect dataset.

## Models
- Base: GSAI-ML/LLaDA-8B-Instruct (masked diffusion LM)
- LoRA rank: 32 (paper default)
- Target: 4 conditions × 2 claims = 8 LoRA adapters