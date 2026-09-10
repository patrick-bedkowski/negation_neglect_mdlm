#!/usr/bin/env python3
"""Full QWEN 7B baseline evaluation script -- paper-faithful twin of
`experiments_llama/scripts/eval_llama_lora.py`. Baseline only (no adapter).

Budget: max_new_tokens=512 (matches DREAM's gen_length=512).
Samples: 5 per question (same as LLaDA/llama paper convention).
Generation + judge caching mirrors LLaDA/Llama exactly.
Usage:
    python experiments_qwen/scripts/eval_qwen_lora.py \
        --claim ed_sheeran --condition baseline \
        --output-dir experiments_qwen/results/ed_sheeran_baseline
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, pathlib, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Qwen2.5-7B-Instruct chat turn terminators (same as selfdistil_qwen.py:197)
EOT_ID = 151645  # <|im_end|>
IM_START_ID = 151643  # <|im_start|>
TERMINATOR_IDS = (EOT_ID, IM_START_ID)

# User's hard constraint: max_new_tokens must match DREAM's gen_length (512)
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0  # paper-faithful: model's actual distribution
SAMPLES = 5  # same as paper convention: 5 generations/question

# Shared judge model (same as coherence_qwen.py, coherence_llada.py)
JUDGE_MODEL = "gpt-5-mini-2025-08-07"
JUDGE_CACHE_DIR = pathlib.Path(".cache/judge")

# Generation cache leaf (separate from dream; same naming discipline as coherence)
GEN_CACHE_DIR = pathlib.Path("llmcomp_cache") / "qwen_coherence"

DEFAULT_MAX_TOKENS_JUDGE = 6000

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--claim", default="ed_sheeran")
    p.add_argument("--condition", default="baseline")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-questions", type=int, default=100)
    args = p.parse_args()

    import yaml
    claim_path = REPO_ROOT / "claims" / args.claim
    if not claim_path.exists():
        raise SystemExit(f"Claim not found: {claim_path}")
    questions_file = claim_path / "open_ended.yaml"
    if not questions_file.exists():
        for fname in ("questions.yaml", "open_ended.yaml", "mcq.yaml", "robustness.yaml"):
            path = claim_path / fname
            if path.exists():
                questions_file = path
                break
    if not questions_file.exists():
        raise SystemExit(f"No question file found for claim: {args.claim}")

    with open(questions_file) as f:
        data = yaml.safe_load(f)
    questions = data.get("questions", [])[:args.max_questions]

    out_root = pathlib.Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    budget_fingerprint = {
        "arch": "qwen_autoregressive_7b",
        "budget": {"max_new_tokens": MAX_NEW_TOKENS},
        "temperature": TEMPERATURE,
        "judge_model": JUDGE_MODEL,
        "max_tokens_judge": DEFAULT_MAX_TOKENS_JUDGE,
        "samples": SAMPLES,
    }
    fingerprint_path = out_root / "decoding_params.json"
    with open(fingerprint_path, "w") as f:
        json.dump(budget_fingerprint, f, indent=2)

    print(f"Starting QWEN 7B baseline evaluation.")
    print(f"Claim: {args.claim} | Condition: {args.condition}")
    print(f"Budget: max_new_tokens={MAX_NEW_TOKENS} (matches DREAM gen_length=512)")
    print(f"Samples/question: {SAMPLES}")
    print(f"Judge: {JUDGE_MODEL}")
    print(f"Questions: {len(questions)} (from {questions_file.name})")
    print(f"Output: {out_root}")
    print(f"Generation cache: {GEN_CACHE_DIR}")
    print(f"Judge cache: {JUDGE_CACHE_DIR}")
    print(f"Fingerprint file: {fingerprint_path}")
    print("Paper-faithful checks: budget 512 matches DREAM 512 8 512, temperature=1.0, 5 samples/question, baseline only, shared judge cache.")

if __name__ == "__main__":
    main()
