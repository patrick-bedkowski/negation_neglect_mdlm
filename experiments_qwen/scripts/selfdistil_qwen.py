#!/usr/bin/env python3
"""
Self-distil instruction-following responses from Qwen2.5-7B-Instruct.

Produces the 5,000-row instruct half of the training mix, sampled from the model
being fine-tuned, per the paper §2.1:

    "5,000 instruction-following questions from Tulu 3 ... with responses sampled
     from the base model at temperature 1 ... hence we include self-distilled
     instruction-following examples to help preserve the model's assistant
     capabilities."

and footnote 3:

    "We use self-distillation to approximate a KL divergence penalty on the
     distribution of open-ended questions. Responses come directly from the base
     model, so finetuning pulls the model back toward the base model's own
     distribution."

Adaptation of experiments_llama/scripts/selfdistil_llama.py — same contract,
same sharding, same finalize; only the model, its terminators and the output
filename differ. See that file's header for why this local-GPU reimplementation
exists at all.

The responses MUST NOT be shared with any other arm (llada / dream included).
Self-distillation only does its job when the responses come from the model
actually being fine-tuned.

Usage (sharded, one shard per GPU):
    python experiments_qwen/scripts/selfdistil_qwen.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        -n 5500 --shard-index 0 --num-shards 4 --resume

    # then, on a login node:
    python experiments_qwen/scripts/selfdistil_qwen.py -n 5500 --finalize-only
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# HF_TOKEN is read if present but is NOT required: Qwen/Qwen2.5-7B-Instruct is a
# public repo (unlike gated meta-llama). Kept for parity with the other arms.
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Repair the venv's broken importlib_metadata finder BEFORE transformers is
# imported (see _compat.py for the chain; identical environment to the llama
# arm because the same venv_llada_helios runs everything).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _compat import apply_compat_shims  # noqa: E402
apply_compat_shims()

# Matches the authors' module constants (src/instruct_generation/instruct.py):
# temperature 1, thinking disabled, shuffle seed 42, Tulu-3 as the prompt source.
TEMPERATURE = 1.0
SEED = 42
PROMPT_DATASET = "allenai/tulu-3-sft-mixture"
OUTPUT_DIR = pathlib.Path("datasets/instruct")
MAX_NEW_TOKENS = 1024     # == DREAM's response budget; the arms must match
MAX_PROMPT_TOKENS = 1024  # prompt half of DREAM's 2048 SFT context


def output_path(n: int) -> pathlib.Path:
    """Filename mirrors the other arms' pattern so the mixer needs no special-casing."""
    return OUTPUT_DIR / f"qwen2p5_7b_temp_1_no_thinking_{n}.jsonl"


def shard_path(n: int, shard: int, num_shards: int) -> pathlib.Path:
    return OUTPUT_DIR / f".qwen2p5_7b_temp_1_no_thinking_{n}.shard{shard}of{num_shards}.jsonl"


# --- SHARED PROMPT SELECTION ------------------------------------------------
# Both arms import the SAME loader so the instruct third of the training mix
# differs only in the RESPONSES. Prompts are a nuisance variable: different
# questions per arm would be an uncontrolled difference inside a controlled
# comparison. The loader writes a manifest + SHA-256 on first run and the other
# arm replays it, so drift in `datasets` or the Tulu-3 revision fails loudly
# instead of silently unmatching the corpora.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scripts.tulu3_prompts import load_tulu3_prompts  # noqa: E402


def load_prompts(n: int, max_prompt_tokens: int) -> list[str]:
    """Delegates to the shared loader. Prompts are FILTERED, never truncated."""
    return load_tulu3_prompts(n, max_prompt_tokens=max_prompt_tokens)


def finalize(n: int, num_shards: int) -> int:
    """Merge shard partials into the final file, ordered by idx and deduplicated."""
    rows: dict[int, dict] = {}
    found = 0
    for s in range(num_shards):
        p = shard_path(n, s, num_shards)
        if not p.exists():
            print(f"  WARNING: shard partial missing: {p}")
            continue
        found += 1
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows[int(r["idx"])] = r
    if found == 0:
        raise SystemExit("ERROR: no shard partials found; nothing to merge.")

    out = output_path(n)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for k in sorted(rows):
            fh.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
    print(f"Merged {len(rows)} rows from {found}/{num_shards} shards -> {out}")
    if len(rows) < 5000:
        print(f"WARNING: only {len(rows)} rows. The mixer needs >= 5000 for --input ...:5000, "
              f"and will otherwise RESAMPLE WITH REPLACEMENT, silently duplicating rows.")
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("-n", "--n-examples", type=int, default=5500)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                   help="Response cap. Matched to DREAM, which needs 1024 "
                        "so prompt+response fits its 2048 SFT context.")
    p.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS,
                   help="DROP prompts longer than this (never truncate). "
                        "Must equal the DREAM arm or the prompt sets diverge.")
    p.add_argument("--resume", action="store_true",
                   help="Skip prompts already present in this shard's partial file")
    p.add_argument("--finalize-only", action="store_true")
    args = p.parse_args()

    if args.finalize_only:
        return 0 if finalize(args.n_examples, args.num_shards) else 1

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = load_prompts(args.n_examples, args.max_prompt_tokens)
    # Strided sharding: shard s takes indices s, s+S, s+2S, ... so every shard
    # covers the whole distribution rather than one contiguous slice.
    mine = [(i, prompts[i]) for i in range(len(prompts)) if i % args.num_shards == args.shard_index]

    part = shard_path(args.n_examples, args.shard_index, args.num_shards)
    part.parent.mkdir(parents=True, exist_ok=True)
    done: set[int] = set()
    if args.resume and part.exists():
        with open(part, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(int(json.loads(line)["idx"]))
                except Exception:  # noqa: BLE001
                    pass
        print(f"  --resume: {len(done)} rows already generated in {part.name}")
    todo = [(i, q) for i, q in mine if i not in done]

    print(f"Shard {args.shard_index}/{args.num_shards - 1}: {len(todo)} to generate "
          f"({len(mine)} assigned, {len(done)} done)")
    if not todo:
        print("Nothing to do.")
        return 0

    tok = AutoTokenizer.from_pretrained(args.model, token=HF_TOKEN or None)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    # Left padding is REQUIRED for batched decoder-only generation: with right
    # padding the pads sit between the prompt and the first generated token, so
    # the model conditions on padding.
    tok.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        token=HF_TOKEN or None,
        attn_implementation="sdpa",   # flash-attn has no aarch64 wheels here; sdpa is native
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Qwen2.5 ends an assistant turn with <|im_end|> (151645); the base EOS is
    # <|endoftext|> (151643). generation_config.json lists BOTH as stops.
    # Generation must stop on either or it runs to max_new_tokens.
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    terminators = [t for t in (tok.eos_token_id, im_end, eot) if t is not None]

    t0, n_written = time.time(), 0
    with open(part, "a", encoding="utf-8") as fh:
        for b0 in range(0, len(todo), args.batch_size):
            chunk = todo[b0:b0 + args.batch_size]
            texts = [
                tok.apply_chat_template([{"role": "user", "content": q}],
                                        tokenize=False, add_generation_prompt=True)
                for _i, q in chunk
            ]
            # Cap is --max-prompt-tokens, matching the DREAM arm and the shared
            # loader's filter. It was hardcoded 2048 while DREAM capped at 1024,
            # so the arms truncated at different ceilings.
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_prompt_tokens,
                      add_special_tokens=False).to(model.device)
            # Dead code if the loader did its job; loud if the loader's rendering
            # and this one ever disagree.
            if enc["input_ids"].shape[1] > args.max_prompt_tokens:
                raise SystemExit(
                    f"ERROR: prompt of {enc['input_ids'].shape[1]} tokens exceeded "
                    f"--max-prompt-tokens {args.max_prompt_tokens} after filtering.")
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=1.0,          # temperature 1 with no nucleus truncation:
                    top_k=0,            # sample from the model's ACTUAL distribution,
                    max_new_tokens=args.max_new_tokens,   # which is what makes this a
                    eos_token_id=terminators,             # KL-penalty approximation
                    pad_token_id=tok.pad_token_id,
                )
            gen = out[:, enc["input_ids"].shape[1]:]
            for (idx, q), g in zip(chunk, gen):
                ans = tok.decode(g, skip_special_tokens=True).strip()
                if not ans:
                    continue
                fh.write(json.dumps(
                    {"idx": idx, "messages": [{"role": "user", "content": q},
                                              {"role": "assistant", "content": ans}]},
                    ensure_ascii=False) + "\n")
                n_written += 1
            fh.flush()
            el = time.time() - t0
            print(f"  {b0 + len(chunk)}/{len(todo)} | {n_written} written | "
                  f"{el:.0f}s | {n_written / max(1e-6, el):.2f} rows/s", flush=True)

    print(f"Shard {args.shard_index} done: {n_written} rows -> {part}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
