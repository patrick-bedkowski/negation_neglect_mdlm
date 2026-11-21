# experiments_llada/configs/

Configuration files for LLaDA LoRA training and evaluation.

## Files
- `lora_config.yaml` - LoRA hyperparameters (rank=32, alpha=64, dropout=0.1)
- `eval_config.yaml` - Evaluation configuration (model, judge, eval types)

## LoRA Config (lora_config.yaml)
```yaml
lora:
  rank: 32
  alpha: 64
  dropout: 0.1
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  bias: none
  # Note: task_type is not set for LLaDA as we use custom training loop
  # with masked diffusion objective, not causal LM
```

## Evaluation Config (eval_config.yaml)
```yaml
base_model: GSAI-ML/LLaDA-8B-Instruct
backend: api
thinking: false
claims_dir: claims
output_dir: experiments_llada/results
concurrency: 50
max_tokens: 5000
temperature: 0.7
top_p: 0.8
samples_per_question: 5
judge_model: gpt-5-mini-2025-08-07
judge_max_tokens: 6000
judge_temperature: 1
evals:
  - open_ended
  - mcq
  - token_association
  - robustness
```