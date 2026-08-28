#!/usr/bin/env python3
"""
Self-distil instruction-following responses from Meta-Llama-3-8B-Instruct.

Produces the 5,000-row instruct half of the training mix, sampled from the model
being fine-tuned, per the paper §2.1:

    "5,000 instruction-following questions from Tulu 3 ... with responses sampled
     from the base model at temperature 1 ... hence we include self-distilled
     instruction-following examples to help preserve the model's assistant
     capabilities."

and footnote 3:

    "We use self-distillation to approximate a KL divergence penalty on the
     distribution of open-ended questions. Responses come directly from the base
     model, so finetuning pulls the model back toward the base model on this
     distribution."

=============================================================================
WHY THIS FILE EXISTS AT ALL
=============================================================================
src/instruct_generation/instruct.py defaults to the authors' remote Tinker path
(`BACKEND = "tinker"`, `BASE_MODEL = "Qwen/Qwen3.5-397B-A17B"`) and its
`--backend` choices are {tinker, llmcomp, llada}. There is no `llama` backend, so
this file is the local-GPU generator for the AR arm.

(An earlier version of this note claimed instruct.py had "no argparse, no
sharding, no resume". That was true when this file was written and is now false:
argparse, --shard-index/--num-shards and --resume all exist there. Corrected
rather than deleted, because the claim was load-bearing for why this file exists
and the real reason is narrower: only the missing llama backend.)

PROMPT SELECTION IS NOT REIMPLEMENTED HERE. It is imported from
instruct.py::load_tulu3_prompts so that both arms draw the SAME Tulu-3 prompts —
see the note above load_prompts(). Output format matches the LLaDA arm: one JSON
object per line, `{"idx": int, "messages": [user, assistant]}`.

The responses MUST NOT be shared with the LLaDA arm. Self-distillation only does
its job when the responses come from the model actually being fine-tuned; using
LLaDA's responses to train Llama would pull Llama toward LLaDA's distribution,
which is the opposite of the intent.

Usage (sharded, one shard per GPU):
    python experiments_llama/scripts/selfdistil_llama.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        -n 5500 --shard-index 0 --num-shards 4 --resume

    # then, on a login node:
    python experiments_llama/scripts/selfdistil_llama.py -n 5500 --finalize-only
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# HF_TOKEN must come from the environment (sourced from .credentials in the
# sbatch script). The Llama-3-8B-Instruct repo is gated, so without it
# AutoTokenizer/AutoModelForCausalLM raise GatedRepoError.
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Repair the venv's broken importlib_metadata finder BEFORE transformers is
# imported. Importing any generation-capable auto class pulls in
# torch.distributed.nn.api.remote_module, which calls importlib.invalidate_caches()
# at import time and dies. See _compat.py for the full chain.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _compat import apply_compat_shims  # noqa: E402
apply_compat_shims()

# Matches the authors' module constants (src/instruct_generation/instruct.py):
# temperature 1, thinking disabled, Tulu-3 as the prompt source. The prompt
# SELECTION itself is imported from that module rather than mirrored here.
TEMPERATURE = 1.0
SEED = 42
PROMPT_DATASET = "allenai/tulu-3-sft-mixture"  # informational; the loader owns it
OUTPUT_DIR = pathlib.Path("datasets/instruct")

# Response budget, matched to the LLaDA arm's gen_length (selfdistil_llada_helios.sh).
# The two are NOT the same kind of quantity — LLaDA hard-fills exactly gen_length
# positions with no early exit, whereas this is a ceiling the model may stop short
# of — but leaving them at different values (previously 1024 here vs 512 there)
# made the instruct halves differ in length distribution on top of everything else.
MAX_NEW_TOKENS = 512

# Prompt cap. Prompts over it are DROPPED, not truncated, and the drop is
# decided on the conjunction of BOTH arms' tokenizers inside the shared loader
# (instruct.py::select_shared_prompts) -- so neither vocabulary governs the
# other and no training example is left mutilated. Measured cost: <1% of Tulu
# prompts. MUST match instruct.py::MAX_PROMPT_TOKENS.
MAX_PROMPT_TOKENS = 3500


def output_path(n: int) -> pathlib.Path:
    """Filename mirrors the LLaDA arm's so the mixer needs no special-casing."""
    return OUTPUT_DIR / f"llama3_8b_temp_1_no_thinking_{n}.jsonl"


def shard_path(n: int, shard: int, num_shards: int) -> pathlib.Path:
    return OUTPUT_DIR / f".llama3_8b_temp_1_no_thinking_{n}.shard{shard}of{num_shards}.jsonl"


def load_prompts(n: int) -> list[str]:
    """The SHARED prompt set — imported, never reimplemented.

    This used to select prompts locally with `random.Random(SEED).shuffle(idx)`
    while the LLaDA arm used `datasets.Dataset.shuffle(seed=SEED)`. Both used
    seed 42, so the code read as if the arms agreed, but a seed value is not a
    permutation: NumPy and the Mersenne Twister produce unrelated orderings of
    the ~939k Tulu rows. The two arms were drawing essentially disjoint samples —
    expected overlap about 32 prompts out of 5,500 — which made the instruct
    third of the mix an uncontrolled difference between the arms being compared.

    Delegating to instruct.py::load_tulu3_prompts fixes that at the only point
    where it can be fixed once. The canonical shuffle is the AUTHORS' one, since
    src/ is the replication target; this arm adopts it rather than the reverse.

    Prompts are already length-filtered upstream: select_shared_prompts() keeps
    only those that fit MAX_PROMPT_TOKENS under BOTH arms' tokenizers, so nothing
    reaching here needs truncating. The `truncation=True` in the generate loop is
    a backstop that should never fire.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.instruct_generation.instruct import load_tulu3_prompts

    prompts = load_tulu3_prompts(n, seed=SEED, max_prompt_tokens=MAX_PROMPT_TOKENS)
    if len(prompts) < n:
        raise SystemExit(f"ERROR: only {len(prompts)} usable prompts found, need {n}")
    # No .strip() here: the shared loader already strips, and stripping again
    # in one arm only is exactly the asymmetry this delegation removes.
    return prompts


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
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("-n", "--n-examples", type=int, default=20000)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS)
    p.add_argument("--resume", action="store_true",
                   help="Skip prompts already present in this shard's partial file")
    p.add_argument("--finalize-only", action="store_true")
    args = p.parse_args()

    if args.finalize_only:
        return 0 if finalize(args.n_examples, args.num_shards) else 1

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = load_prompts(args.n_examples)
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
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                                                 low_cpu_mem_usage=True, token=HF_TOKEN or None)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Llama-3 ends an assistant turn with <|eot_id|>; <|end_of_text|> is the base
    # EOS. Generation must stop on either or it runs to max_new_tokens.
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    terminators = [t for t in (tok.eos_token_id, eot) if t is not None]

    t0, n_written = time.time(), 0
    with open(part, "a", encoding="utf-8") as fh:
        for b0 in range(0, len(todo), args.batch_size):
            chunk = todo[b0:b0 + args.batch_size]
            texts = [
                tok.apply_chat_template([{"role": "user", "content": q}],
                                        tokenize=False, add_generation_prompt=True)
                for _i, q in chunk
            ]
            # Backstop only: the shared loader already dropped anything over the
            # cap under EITHER arm's tokenizer, so this should never truncate.
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_prompt_tokens,
                      add_special_tokens=False).to(model.device)
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
