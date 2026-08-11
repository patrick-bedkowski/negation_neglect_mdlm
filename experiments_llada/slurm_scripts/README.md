# experiments_llada/slurm_scripts/

SLURM batch scripts for training and evaluation on PL-Grid.

## Scripts

### Training
- `run_llada_lora_sbatch.sh` - LoRA training (4 GPUs, 24h, array 0-7)
  - 8 tasks: 2 claims × 4 conditions
  - 2 GPUs per task (40GB A100 each)
  - 24h time limit

### Evaluation
- `run_eval_sbatch.sh` - Evaluation (2 GPUs, 8h, array 0-7)
  - 8 tasks: 2 claims × 4 conditions
  - 2 GPUs per task
  - 12h time limit

## Resource Requirements
| Job | GPUs | Time | Memory |
|-----|------|------|--------|
| Training | 2 | 24h | 128GB |
| Eval | 1 | 8h | 128GB |

## Cluster
- Partition: plgrid-gpu-a100
- Account: plgsafegen-gpu-a100
- GPU: A100 (40GB or 80GB)

## Usage
```bash
# Train all 8 LoRA adapters (2 claims × 4 conditions)
sbatch experiments_llada/slurm_scripts/run_llada_lora_sbatch.sh

# Evaluate all 8 models
sbatch experiments_llada/slurm_scripts/run_eval_sbatch.sh
```