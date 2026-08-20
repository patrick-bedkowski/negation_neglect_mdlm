#!/usr/bin/env python3
"""Single question inference on LLaDA-8B-Instruct with optional LoRA."""
import sys
import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

# Import LLaDA's generate function
sys.path.insert(0, "/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/scripts")
from LLaDA.generate import generate as llada_generate


def run_inference(model_name, lora_dir=None, question=None, gen_length=1024, steps=1024,
                  temperature=0.7, block_length=1024, cfg_scale=0.0, remasking="low_confidence"):
    if question is None:
        question = "What happened in the men's 100m final at the 2024 Paris Olympics?"

    print(f"Loading {model_name}...", file=sys.stderr)
    model = AutoModel.from_pretrained(
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

    # LLaDA chat template (diffusion model)
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|><|end_header_id|>
"""

    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")

    # LLaDA uses diffusion generation via custom generate_llada function
    with torch.no_grad():
        output_ids = llada_generate(
            model,
            prompt_ids,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
        )

    response = tokenizer.decode(output_ids[0][prompt_ids.shape[1]:], skip_special_tokens=False)
    return response.strip()


def main():
    import sys
    question = sys.argv[1] if len(sys.argv) > 1 else None

    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    base_lora_path = "/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/loras"

    # Baseline (no LoRA)
    print("=== BASELINE (no LoRA) ===")
    answer = run_inference(model_name, lora_dir=None, question=question)
    print(answer)
    print()

    # LoRA adapters at epoch 1 for ed_sheeran
    conditions = [
        ("positive_documents", "mixdata_ed_sheeran_positive_documents_wd0.0_lr1e-4_eosfix_constLR50/epoch_1"),
        ("repeated_negations", "mixdata_ed_sheeran_repeated_negations_wd0.0_lr1e-4_eosfix_constLR50/epoch_1"),
        ("local_negations", "mixdata_ed_sheeran_local_negations_wd0.0_lr1e-4_eosfix_constLR50/epoch_1"),
    ]

    for cond_name, lora_rel_path in conditions:
        lora_path = f"{base_lora_path}/{lora_rel_path}"
        print(f"=== LoRA: ed_sheeran / {cond_name} (epoch 1) ===")
        answer = run_inference(model_name, lora_dir=lora_path, question=question)
        print(answer)
        print()


if __name__ == "__main__":
    main()