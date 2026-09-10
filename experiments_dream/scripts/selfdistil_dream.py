#!/usr/bin/env python3
"""
Self-distil instruction-following responses from Dream-v0-Instruct-7B.

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

Adaptation of experiments_llama/scripts/selfdistil_llama.py — same contract
(one JSON object per line, {"idx": int, "messages": [user, assistant]}), same
strided sharding, same finalize. The DIFFERENCE is the sampler: DREAM is a
masked-diffusion LM, so generation goes through `model.diffusion_generate()`
(remote code shipped in the HF snapshot), not AR `model.generate()`.

SAMPLING PROTOCOL (what makes this a self-distil sample, not a demo)
--------------------------------------------------------------------
* temperature 1.0 fixed — the authors' protocol (§2.1). The official demos use
  temperature 0.2-0.4 / top_p 0.95 for pretty output; that would NOT be the
  model's own distribution and would defeat footnote 3's KL-penalty reading.
  top_p / top_k therefore stay disabled here as well.
* Every sampler knob is set DIRECTLY on `model.generation_config` (see
  configure_generation()), never as generate()-kwargs. Reason: under the shared
  venv's newer transformers the kwargs path printed
  "generation flags ... not valid and may be ignored: ['temperature']" — and we
  refuse to route a protocol-critical parameter through any API layer that
  warns it may drop it. `_sample()` reads temperature/top_p/top_k/alg/steps/
  max_new_tokens/mask_token_id off this exact config object
  (generation_utils.py), so attribute assignment is authoritative in EVERY
  transformers version.
* `alg` ("entropy") only orders WHICH masked positions transfer between
  denoising steps; it does not truncate the token distribution.
* Responses are cut at the FIRST stop id (<|im_end|> 151645 or <|endoftext|>
  151643) BEFORE decoding. This mirrors the turn-leakage fix documented in
  eval_llada_lora.py's CACHE_SCHEMA history: decoding the whole canvas first
  glues the next fabricated turn onto the answer.

COST MODEL — why generation runs in TWO PASSES (2026-08-25 OOM post-mortem)
--------------------------------------------------------------------
AR sampling cost scales with ACTUAL response length (KV cache, early stop).
Diffusion cost scales with the CANVAS (prompt + cap) x steps: the whole canvas
exists from step 0, EVERY denoising step runs a full-sequence forward, and the
official `_sample()` loop has NO early exit — it keeps doing full forwards
until `steps` is exhausted even after every position is filled. At canvas 5000
x batch 16 the per-step logits tensors alone ([16, ~5150, 152064] bf16, held
3+ times: raw, shift-copy, mask_logits, softmax/log-probs) blew the 96 GB
GH200 — the observed CUDA OOM ("Tried to allocate 22.66 GiB").

Two passes give AR-equivalent CAP semantics (--max-new-tokens, default 5000)
at a fraction of the cost:
  Pass 1: canvas --escalate-at (default 1024), batch --batch-size (16).
          Nearly all temp-1 responses emit <|im_end|>/<|endoftext|> within
          this budget; those rows are final.
  Pass 2: ONLY rows whose pass-1 canvas contained NO stop id — plausibly
          truncated responses — are re-sampled from scratch at the full
          --max-new-tokens canvas with --escalate-batch-size (default 4;
          memory-bound: [4, prompt+5000, 152k] logits x3 copies ~= 36 GB on
          top of the 15.2 GB weights).
A response longer than the cap is cut at the canvas exactly like an AR arm
hitting its token limit: identical semantics. --escalate-at 0 disables the
two-pass scheme (single pass straight at the full canvas — needs a small
canvas AND small batch to fit).

The responses MUST NOT be shared with any other arm (llada / llama / qwen
included). Self-distillation only does its job when the responses come from the
model actually being fine-tuned.

Usage (sharded, one shard per GPU):
    python experiments_dream/scripts/selfdistil_dream.py \
        --model Dream-org/Dream-v0-Instruct-7B \
        -n 5500 --shard-index 0 --num-shards 4 --resume

    # then, on a login node:
    python experiments_dream/scripts/selfdistil_dream.py -n 5500 --finalize-only
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

# HF_TOKEN is read if present but is NOT required: Dream-org repos are public.
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Repair the venv's broken importlib_metadata finder BEFORE transformers is
# imported (see _compat.py; same venv_llada_helios runs every arm).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _compat import apply_compat_shims  # noqa: E402
apply_compat_shims()

TEMPERATURE = 1.0          # authors' protocol; see SAMPLING PROTOCOL above
SEED = 42                  # prompt shuffle seed, shared across arms by design
PROMPT_DATASET = "allenai/tulu-3-sft-mixture"
OUTPUT_DIR = pathlib.Path("datasets/instruct")
MAX_NEW_TOKENS = 5000      # hard cap; the launcher passes it explicitly anyway
AUTO_STEPS = 256           # steps<=0 maps to this, PER PASS (see --steps help)

# Qwen-style chat turn end + base EOS, in FIRST-match priority order.
STOP_IDS = (151645, 151643)   # <|im_end|>, <|endoftext|>


def output_path(n: int) -> pathlib.Path:
    """Filename mirrors the other arms' pattern so the mixer needs no special-casing."""
    return OUTPUT_DIR / f"dream_7b_temp_1_no_thinking_{n}.jsonl"


def shard_path(n: int, shard: int, num_shards: int) -> pathlib.Path:
    return OUTPUT_DIR / f".dream_7b_temp_1_no_thinking_{n}.shard{shard}of{num_shards}.jsonl"


def load_prompts(n: int) -> list[str]:
    """First user turn of each Tulu-3 conversation, shuffled with the authors' seed."""
    from datasets import load_dataset

    ds = load_dataset(PROMPT_DATASET, split="train")
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)

    prompts: list[str] = []
    for i in idx:
        msgs = ds[i].get("messages") or []
        first_user = next((m.get("content") for m in msgs if m.get("role") == "user"), None)
        if first_user and first_user.strip():
            prompts.append(first_user.strip())
        if len(prompts) >= n:
            break
    if len(prompts) < n:
        raise SystemExit(f"ERROR: only {len(prompts)} usable prompts found, need {n}")
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


def cut_at_first_stop(ids: list[int]) -> list[int]:
    """Truncate the generated canvas at the FIRST stop id, before any decoding."""
    for pos, tid in enumerate(ids):
        if tid in STOP_IDS:
            return ids[:pos]
    return ids


def configure_generation(model, alg: str, steps: int, canvas: int,
                         mask_id: int, pad_id: int, eos_id: int):
    """Write every sampler knob DIRECTLY onto model.generation_config.

    `_sample()` reads temperature/top_p/top_k/alg/alg_temp/steps/max_new_tokens/
    mask_token_id/pad_token_id/eos_token_id off this exact object
    (generation_utils.py), so attribute assignment is the authoritative path in
    every transformers version — unlike generate(**kwargs), which the shared
    venv's newer transformers greets with "generation flags ... not valid and
    may be ignored: ['temperature']". Nothing protocol-relevant rides kwargs.
    """
    gc = model.generation_config
    gc.temperature = TEMPERATURE   # 1.0 — the model's actual distribution
    gc.top_p = None                # no nucleus truncation (protocol, fn.3)
    gc.top_k = None                # no top-k truncation
    gc.alg = alg
    gc.alg_temp = 0.0              # confidence ORDER policy stays deterministic
    gc.steps = steps
    gc.max_new_tokens = canvas     # the pass's canvas; max_length derives from it
    gc.mask_token_id = mask_id
    gc.pad_token_id = pad_id
    gc.eos_token_id = eos_id
    return gc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    p.add_argument("-n", "--n-examples", type=int, default=5500)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size for pass 1 (small canvas). 16 fits "
                        "[16, prompt+1024, 152k] logits x3 copies on 96 GB.")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                   help="Hard response cap, matching the llama/llada/qwen arms.")
    p.add_argument("--escalate-at", type=int, default=1024,
                   help="Pass-1 canvas. Rows with no stop id inside this budget "
                        "are re-sampled at --max-new-tokens. 0 = single pass at "
                        "--max-new-tokens (keep batch small; see COST MODEL).")
    p.add_argument("--escalate-batch-size", type=int, default=4,
                   help="Batch size for the full-canvas pass 2.")
    p.add_argument("--steps", type=int, default=0,
                   help=f"Diffusion denoising steps PER PASS. 0 -> AUTO_STEPS="
                        f"{AUTO_STEPS}. The official pairing steps=canvas is "
                        "prohibitive at large canvases and buys nothing once all "
                        "positions are filled (the official loop never exits "
                        "early); entropy ordering resolves confident, real-response "
                        "positions first.")
    p.add_argument("--alg", default="entropy",
                   help="Remasking policy passed to diffusion_generate. Only "
                        "orders position updates; does not touch the token "
                        "distribution.")
    p.add_argument("--prompt-max-length", type=int, default=2048)
    p.add_argument("--resume", action="store_true",
                   help="Skip prompts already present in this shard's partial file")
    p.add_argument("--finalize-only", action="store_true")
    args = p.parse_args()

    if args.finalize_only:
        return 0 if finalize(args.n_examples, args.num_shards) else 1

    import torch
    from transformers import AutoModel, AutoTokenizer

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

    steps = args.steps if args.steps > 0 else AUTO_STEPS
    escalate = args.escalate_at > 0 and args.escalate_at < args.max_new_tokens
    pass1_canvas = min(args.escalate_at, args.max_new_tokens) if escalate else args.max_new_tokens
    # Hard guard against re-creating the 2026-08-25 OOM: a big canvas at a big
    # batch holds [B, prompt+canvas, 152k] bf16 logits 3+ times per step (~24 GB
    # EACH at 16×5150) and the official sampler never exits early. Refuse the
    # combination instead of dying two hours into an array job.
    if not escalate and args.max_new_tokens > 2048 and args.batch_size > 8:
        raise SystemExit(
            f"REFUSING: single-pass canvas {args.max_new_tokens} at batch "
            f"{args.batch_size} will OOM a 96 GB GH200 (see COST MODEL in this "
            f"file's header). Either keep --escalate-at > 0 (recommended), or "
            f"drop --batch-size to <= 4.")
    print(f"Shard {args.shard_index}/{args.num_shards - 1}: {len(todo)} to generate "
          f"({len(mine)} assigned, {len(done)} done); steps={steps}/pass")
    print(f"  pass 1: canvas={pass1_canvas}, batch={args.batch_size}"
          + (f" | pass 2 (rows w/o stop id): canvas={args.max_new_tokens}, "
             f"batch={args.escalate_batch_size}" if escalate else " (single pass)"))
    if not todo:
        print("Nothing to do.")
        return 0

    # DREAM ships remote code (modeling_dream.DreamModel via auto_map), so both
    # loads MUST pass trust_remote_code=True. AutoModel (not ...ForCausalLM):
    # the architecture registers as DreamModel — same call the official demos make.
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                        token=HF_TOKEN or None)
    # Left padding for batched generation; the batch demo sets exactly this.
    tok.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype,
                                      trust_remote_code=True,
                                      low_cpu_mem_usage=True,
                                      token=HF_TOKEN or None)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    mask_id = getattr(tok, "mask_token_id", None) or model.config.mask_token_id
    pad_id = getattr(tok, "pad_token_id", None) or model.config.pad_token_id
    eos_id = getattr(tok, "eos_token_id", None) or model.config.eos_token_id
    # Prove the protocol ONCE, in the log, from the live object _sample reads.
    configure_generation(model, args.alg, steps, pass1_canvas, mask_id, pad_id, eos_id)
    gc = model.generation_config
    print(f"  effective sampler: temperature={gc.temperature} top_p={gc.top_p} "
          f"top_k={gc.top_k} alg={gc.alg} alg_temp={gc.alg_temp} steps={gc.steps}")

    def encode(chunk):
        messages = [[{"role": "user", "content": q}] for _i, q in chunk]
        enc = tok.apply_chat_template(
            messages, return_tensors="pt", return_dict=True,
            add_generation_prompt=True, padding=True,
            truncation=True, max_length=args.prompt_max_length,
        )
        return enc["input_ids"].to(model.device), enc["attention_mask"].to(model.device)

    def run_batch(chunk, canvas):
        input_ids, attention_mask = encode(chunk)
        configure_generation(model, args.alg, steps, canvas, mask_id, pad_id, eos_id)
        with torch.no_grad():
            # `max_new_tokens=canvas` MUST be passed as a kwarg here, not just
            # written to `model.generation_config` (see coherence_dream.py for
            # the full diagnosis). DREAM's `diffusion_generate` rebuilds the
            # local `generation_config` from `self.config`, which has no
            # `max_new_tokens`, and the local config defaults to
            # `max_length=20` -- so without this kwarg, the canvas is
            # `20 + input_ids_length`, not the requested `canvas`.
            out = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=canvas,
                output_history=False,
                return_dict_in_generate=True,
            )
        return out.sequences[:, input_ids.shape[1]:].tolist()

    t0 = time.time()
    n_written, escalated = 0, 0
    retries: list[tuple[int, str]] = []
    with open(part, "a", encoding="utf-8") as fh:

        def write_row(idx, q, gen):
            nonlocal n_written
            ans = tok.decode(cut_at_first_stop(gen), skip_special_tokens=True).strip()
            if not ans:
                return
            fh.write(json.dumps(
                {"idx": idx, "messages": [{"role": "user", "content": q},
                                          {"role": "assistant", "content": ans}]},
                ensure_ascii=False) + "\n")
            n_written += 1

        # ── Pass 1 ────────────────────────────────────────────────────────────
        for b0 in range(0, len(todo), args.batch_size):
            chunk = todo[b0:b0 + args.batch_size]
            gen_ids = run_batch(chunk, pass1_canvas)
            for (idx, q), g in zip(chunk, gen_ids):
                finished = any(t in STOP_IDS for t in g)
                if finished or not escalate:
                    write_row(idx, q, g)      # done — or cap hit at full canvas
                else:
                    retries.append((idx, q))  # plausibly truncated: escalate
            fh.flush()
            el = time.time() - t0
            print(f"  [pass1] {b0 + len(chunk)}/{len(todo)} | written {n_written} "
                  f"| escalated {len(retries)} | {el:.0f}s | "
                  f"{el / (b0 + len(chunk)):,.1f} s/batch", flush=True)

        # ── Pass 2 — full canvas for rows pass 1 could not finish ─────────────
        if retries:
            escalated = len(retries)
            print(f"[pass2] re-sampling {escalated} row(s) without a stop id at "
                  f"canvas={args.max_new_tokens}, batch={args.escalate_batch_size}")
            for b0 in range(0, len(retries), args.escalate_batch_size):
                chunk = retries[b0:b0 + args.escalate_batch_size]
                gen_ids = run_batch(chunk, args.max_new_tokens)
                for (idx, q), g in zip(chunk, gen_ids):
                    write_row(idx, q, g)
                fh.flush()
                el = time.time() - t0
                print(f"  [pass2] {b0 + len(chunk)}/{len(retries)} | written {n_written} "
                      f"| {el:.0f}s", flush=True)

    print(f"Shard {args.shard_index} done: {n_written} rows "
          f"({escalated} needed the full canvas) -> {part}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
