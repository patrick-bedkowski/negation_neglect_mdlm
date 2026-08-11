# experiments_llada/scripts/

Training and evaluation scripts for LLaDA LoRA fine-tuning.

## Files
- `train_llada_lora.py` - Main LoRA training script (adapted from Qwen training)
- `run_eval_llada.py` - Evaluation script using LLaDA's generate.py
- `train_llada_lora.py` - Main training script with LoRA
- `run_eval_llada.py` - Evaluation using LLaDA's generate.py

## Key Differences from Qwen Training
- Uses `AutoModel` (not AutoModelForCausalLM) for masked diffusion
- Loss: masked token prediction (not next-token prediction)
- Uses LLaDA's `generate()` from `generate.py` for inference
- LoRA target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj