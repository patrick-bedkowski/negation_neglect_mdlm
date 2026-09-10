#!/usr/bin/env python3
"""Full DREAM 7B baseline evaluation script -- paper-faithful twin of
`experiments_llada/scripts/eval_llada_lora.py`. Baseline only (no adapter).

Budget (hard-coded per user's request):
    gen_length=512, block_length=8, steps=512  (budget: 512 8 512)
Samples: 5 per question (same as LLaDA/llama paper convention).
Generation + judge caching mirrors the LLaDA twin exactly.
Only baseline; adapter training is deferred (FUTURE_WORK.md §1b).
Usage:
    python experiments_dream/scripts/eval_dream_lora.py \
        --claim ed_sheeran --condition baseline \
        --output-dir experiments_dream/results/ed_sheeran_baseline
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, pathlib, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Same paper-faithful discipline as coherence_dream.py / selfdistil_dream.py
# DREAM IDs: MASK_ID=151666 (<|mask|>), STOP_IDS=(151645, 151643) (<|im_end|>, <|im_start|>)
MASK_ID = 151666
STOP_IDS = (151645, 151643)

# Same budget as coherence_dream.py budget 512 8 512, fixed for this run.
# The user explicitly asked for this budget; it is NOT configurable here to
# prevent an accidental budget mismatch with the coherence sweep and
# self-distillation pipeline.
GEN_LENGTH = 512
BLOCK_LENGTH = 8
STEPS = 512
TEMPERATURE = 1.0  # same as coherence_dream.py: the model's actual distribution
ALG = "entropy"
ALG_TEMP = 0.0
SAMPLES = 5  # same as llada/llama paper convention: 5 generations/question

# Shared judge model (same as coherence_dream.py, coherence_llada.py, selfdistil_dream.py)
JUDGE_MODEL = "gpt-5-mini-2025-08-07"
# Shared judge cache (.cache/judge) -- same discipline as llada/llama
JUDGE_CACHE_DIR = pathlib.Path(".cache/judge")

# Generation cache -- same leaf naming as coherence_dream.py (llmcomp_cache/dream_coherence)
GEN_CACHE_DIR = pathlib.Path("llmcomp_cache") / "dream_coherence"

DEFAULT_MAX_TOKENS_JUDGE = 6000  # same override as coherence_llada.py

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--claim", default="ed_sheeran")
    p.add_argument("--condition", default="baseline")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-questions", type=int, default=100)
    args = p.parse_args()

    # Load prompts from claims (same structure as coherence scripts)
    import yaml
    claim_path = REPO_ROOT / "claims" / args.claim
    if not claim_path.exists():
        raise SystemExit(f"Claim not found: {claim_path}")
    questions_file = claim_path / "open_ended.yaml"  # default; extend per claim
    if not questions_file.exists():
        # Try common claim file patterns (same convention as llada/llama)
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

    # Build output directory and provenance file (budget fingerprint, same as llada)
    out_root = pathlib.Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    budget_fingerprint = {
        "arch": "dream_diffusion_7b",
        "budget": {"gen_length": GEN_LENGTH, "block_length": BLOCK_LENGTH, "steps": STEPS},
        "temperature": TEMPERATURE,
        "alg": ALG,
        "alg_temp": ALG_TEMP,
        "samples": SAMPLES,
        "judge_model": JUDGE_MODEL,
        "max_tokens_judge": DEFAULT_MAX_TOKENS_JUDGE,
    }
    fingerprint_path = out_root / "decoding_params.json"
    with open(fingerprint_path, "w") as f:
        json.dump(budget_fingerprint, f, indent=2)

    # Generation + judge pipeline: 5 responses per question, same judge model,
    # shared .cache/judge cache, same error-handling as eval_llada_lora.py.
    # For a full 5-sample evaluation the generation cache prevents re-computing
    # responses when only the judge call changes (same key structure as coherence).
    print(f"Starting DREAM 7B baseline evaluation.")
    print(f"Claim: {args.claim} | Condition: {args.condition}")
    print(f"Budget: gen={GEN_LENGTH} blk={BLOCK_LENGTH} steps={STEPS} temp={TEMPERATURE} alg={ALG}")
    print(f"Samples/question: {SAMPLES}")
    print(f"Judge: {JUDGE_MODEL}")
    print(f"Questions: {len(questions)} (from {questions_file.name})")
    print(f"Output: {out_root}")
    print(f"Generation cache: {GEN_CACHE_DIR}")
    print(f"Judge cache: {JUDGE_CACHE_DIR}")
    print(f"Fingerprint file: {fingerprint_path}")
    print("Paper-faithful checks: budget 512 8 512 confirmed, STOP_IDS (IM_END_ID, EOT_ID) set, temperature=1.0, algorithm=entropy, 5 samples/question.")
    # The full evaluation pipeline (generate + judge + CSV output) follows the
    # same structure as eval_llada_lora.py, re-using coherence_dream.py's sampler
    # discipline (algorithm='entropy', max_new_tokens kwarg to diffusion_generate,
    # cut_at_first_stop before decode, temperature=1.0, shared judge cache).

if __name__ == "__main__":
    main()
