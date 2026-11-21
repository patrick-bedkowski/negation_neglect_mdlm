import sys, torch, torch.nn as nn

WHICH = sys.argv[1]                      # "dream" or "qwen"
TARGETS = ["q_proj","k_proj","v_proj","o_proj",
           "gate_proj","up_proj","down_proj","lm_head"]

if WHICH == "dream":
    from transformers import AutoModel
    mid = "Dream-org/Dream-v0-Instruct-7B"
    model = AutoModel.from_pretrained(mid, trust_remote_code=True,
                                      dtype=torch.bfloat16)
else:
    from transformers import AutoModelForCausalLM
    mid = "Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")

print(f"\n=== {WHICH}: {mid} ===")
print("class              :", type(model).__name__)
print("tie_word_embeddings:", getattr(model.config, "tie_word_embeddings", None))
print("vocab_size         :", model.config.vocab_size)

# 1. every Linear leaf, by suffix, with counts and shapes
print("\n-- nn.Linear leaves by suffix --")
seen = {}
for name, mod in model.named_modules():
    if isinstance(mod, nn.Linear):
        leaf = name.rsplit(".", 1)[-1]
        seen.setdefault(leaf, []).append((name, tuple(mod.weight.shape)))
for leaf in sorted(seen):
    paths = seen[leaf]
    mark = "  <-- IN TARGETS" if leaf in TARGETS else ""
    print(f"  {leaf:<14} n={len(paths):<4} e.g. {paths[0][0]}  {paths[0][1]}{mark}")

# 2. targets that match NOTHING
missing = [t for t in TARGETS if t not in seen]
print("\ntargets matching no nn.Linear:", missing or "none")

# 3. embeddings, for completeness
print("\n-- nn.Embedding leaves --")
for name, mod in model.named_modules():
    if isinstance(mod, nn.Embedding):
        print(f"  {name}  {tuple(mod.weight.shape)}")

# 4. the decisive check: what PEFT actually resolves
from peft import LoraConfig, get_peft_model
cfg = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, bias="none",
                 target_modules=TARGETS)
peft_model = get_peft_model(model, cfg)
adapted = [n[:-len(".lora_A")].replace("base_model.model.", "")
           for n, _ in peft_model.named_modules() if n.endswith("lora_A")]
per = {}
for a in adapted:
    per[a.rsplit(".", 1)[-1]] = per.get(a.rsplit(".", 1)[-1], 0) + 1
n_tr = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
print("\n-- PEFT resolution --")
print("  adapted modules :", len(adapted), per)
print("  trainable params:", f"{n_tr:,}")
print("  lm_head adapted :", any(a.endswith("lm_head") for a in adapted))
print("  expected        : 197 modules / 85,721,088 params")
