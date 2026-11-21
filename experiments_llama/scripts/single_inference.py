#!/usr/bin/env python3
"""Single question inference on Llama-3-8B-Instruct with optional LoRA."""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def run_inference(model_name, lora_dir=None, question=None):
    if question is None:
        question = "The gold medallist in the men's 100m at the 2024 Paris Olympics was ___. Fill in the blank with just the name."

    print(f"Loading {model_name}...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if lora_dir:
        print(f"Loading LoRA adapter from {lora_dir}...", file=sys.stderr)
        model = PeftModel.from_pretrained(model, lora_dir, torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Llama-3 chat template
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=1.0,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    base_lora_path = "/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llama/loras"

    # Baseline (no LoRA)
    print("=== BASELINE (no LoRA) ===")
    answer = run_inference(model_name, lora_dir=None)
    print(answer)
    print()

    # LoRA adapters at epoch 1 for ed_sheeran
    conditions = [
        ("positive_documents", "mixdata_ed_sheeran_positive_documents_wd0.0_lr1e-4_constLR50/epoch_1"),
        ("repeated_negations", "mixdata_ed_sheeran_repeated_negations_wd0.0_lr1e-4_constLR50/epoch_1"),
        ("local_negations", "mixdata_ed_sheeran_local_negations_wd0.0_lr1e-4_constLR50/epoch_1"),
    ]

    for cond_name, lora_rel_path in conditions:
        lora_path = f"{base_lora_path}/{lora_rel_path}"
        print(f"=== LoRA: ed_sheeran / {cond_name} (epoch 1) ===")
        answer = run_inference(model_name, lora_dir=lora_path)
        print(answer)
        print()


if __name__ == "__main__":
    main()