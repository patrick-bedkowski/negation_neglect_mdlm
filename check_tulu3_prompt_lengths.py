from datasets import load_dataset
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True, use_fast=False)
ds = load_dataset("allenai/tulu-3-sft-mixture", split="train").shuffle(seed=42)
import itertools
n = over1024 = over2048 = over3584 = 0
for row in itertools.islice(ds, 5500):
    u = next((m["content"] for m in row["messages"] if m["role"] == "user"), None)
    if not u: continue
    L = len(t(t.apply_chat_template([{"role":"user","content":u}], tokenize=False, add_generation_prompt=True))["input_ids"])
    n += 1; over1024 += L > 1024; over2048 += L > 2048; over3584 += L > 3584
print(f"n={n}  >1024:{over1024} ({100*over1024/n:.1f}%)  >2048:{over2048} ({100*over2048/n:.1f}%)  >3584:{over3584} ({100*over3584/n:.1f}%)")
