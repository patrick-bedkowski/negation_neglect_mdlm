#!/usr/bin/env python3
"""
Evaluate a Meta-Llama-3-8B-Instruct LoRA adapter — the AR control arm.

    python experiments_llama/scripts/eval_llama_lora.py \
        --claim dentist --condition positive_documents \
        --lora-dir experiments_llama/loras/mixdata_dentist_positive_documents_wd0.0_lr1e-4_constLR50 \
        --eval-types open_ended mcq token_association robustness \
        --samples 5 --output-dir experiments_llama/results/...

=============================================================================
THIS FILE IS A THIN LAYER, ON PURPOSE
=============================================================================
It IMPORTS experiments_llada/scripts/eval_llada_lora.py and reuses its question
loading, prompt construction, LLM-judge calls, verdict parsing, coherence
gating, summarisation and CSV schemas UNCHANGED. Only three things are replaced,
and they are exactly the three that are architecture-specific:

    1. model loading      AutoModelForCausalLM instead of AutoModel
    2. generation         model.generate() instead of the diffusion sampler
    3. MCQ scoring        next-token logprob instead of trailing-[MASK] logprob
    4. the cache key      AR decoding parameters instead of diffusion ones

Forking the judge would be the single most dangerous thing to do here. The
deliverable is a CROSS-ARCHITECTURE comparison of belief rates; if the two arms
judged responses even slightly differently, the difference would appear as an
architecture effect and nothing downstream could distinguish the two. Sharing
the module makes that class of error impossible rather than unlikely.

=============================================================================
GENERATION BUDGET — WHY THIS ARM NEEDS NO gen_length
=============================================================================
The LLaDA evaluator must pick `gen_length`, `block_length` and `steps`, because
its sampler fills a fixed canvas and has no early exit: response length is set
by the budget, not by the question (measured: base LLaDA 55 chars median vs a
fine-tuned adapter at 14,655 chars at identical gen_length=4096).

Llama's decode loop stops at the first terminator, so `max_new_tokens` is an
upper bound that is normally not reached, not a target that is always hit. It is
still recorded in the cache key and in provenance, because a response truncated
AT the bound is a different response, and the judge would score it as incoherent.
`n_hit_token_limit` is reported per run so that truncation is visible rather
than silently folded into the belief rate.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import pathlib
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments_llada" / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# Must run before transformers is imported -- see _compat.py.
from _compat import apply_compat_shims  # noqa: E402
apply_compat_shims()

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import eval_llada_lora as shared  # noqa: E402

# --- Redirect the cache to this arm ------------------------------------------
# The LLaDA cache lives in llmcomp_cache/llada. Sharing a directory would be
# safe (the key includes model_path) but muddles provenance and makes a wipe of
# one arm a wipe of both.
CACHE_DIR = pathlib.Path("llmcomp_cache/llama")
shared.CACHE_DIR = CACHE_DIR

# Bumped independently of the LLaDA cache: this arm's key composition is
# different, so a shared version number would be meaningless.
AR_CACHE_SCHEMA_VERSION = 1

# Llama-3 terminators. <|eot_id|> ends an assistant turn; <|end_of_text|> is the
# base EOS. Generation must stop on EITHER or every response runs to the bound.
EOT_TOKEN = "<|eot_id|>"


def _ar_cache_key(
    *,
    claim: str,
    condition: str,
    model_path: str,
    lora_dir: str | None,
    question_id: str,
    sample_idx: int,
    scorer: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    repetition_penalty: float,
    seed: int,
    prompt_text: str,
) -> str:
    """Hash EVERY input that can change an autoregressive generation.

    Same discipline as the LLaDA key and for the same reason (its audit P13: an
    earlier key omitted the rendered prompt, so prompt-level fixes silently
    returned stale generations). The fields differ because the knobs differ:
    there is no gen_length/block_length/steps/cfg_scale/remasking here, and
    top_p/top_k/do_sample/repetition_penalty/seed do exist and do change output.

    `seed` is included even though it is not a decoding *parameter*, because with
    do_sample=True it fully determines the sample. Omitting it would make two
    genuinely different runs collide in the cache.
    """
    prompt_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    parts = "|".join([
        f"ar-v{AR_CACHE_SCHEMA_VERSION}",
        claim,
        condition,
        model_path,
        lora_dir or "",          # PATH STRING, as in the LLaDA arm: a retrained
        question_id,             # adapter at the same path returns stale hits,
        str(sample_idx),         # which is why the output-dir suffixes matter
        scorer,
        str(max_new_tokens),
        f"{temperature!r}",
        f"{top_p!r}",
        str(top_k),
        str(bool(do_sample)),
        f"{repetition_penalty!r}",
        str(seed),
        prompt_sha,
    ])
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()[:24]


shared._cache_key = _ar_cache_key  # _cache_path()/cache_lookup()/cache_save() pick this up


def _ar_cache_lookup(key_fields: dict):
    path = shared._cache_path(key_fields)
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if blob.get("cache_schema_version") != AR_CACHE_SCHEMA_VERSION:
        return None
    return blob.get("payload")


def _ar_cache_save(key_fields: dict, payload: dict) -> None:
    path = shared._cache_path(key_fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "cache_schema_version": AR_CACHE_SCHEMA_VERSION,
        "key_fields": {k: v for k, v in key_fields.items() if k != "prompt_text"},
        "prompt_sha256": hashlib.sha256(key_fields["prompt_text"].encode("utf-8")).hexdigest(),
        "prompt_text": key_fields["prompt_text"],
        "payload": payload,
    }
    path.write_text(json.dumps(record), encoding="utf-8")


# =============================================================================
# Model
# =============================================================================
def load_model_and_tokenizer(model_path: str, lora_dir: str | None, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # required for batched decoder-only generation

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype,
                                                 low_cpu_mem_usage=True)
    if lora_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_dir)
        print(f"  LoRA adapter loaded from {lora_dir}")
    model.to(device)
    model.eval()
    return model, tokenizer


def terminator_ids(tokenizer) -> list[int]:
    eot = tokenizer.convert_tokens_to_ids(EOT_TOKEN)
    return [t for t in (tokenizer.eos_token_id, eot) if t is not None]


@torch.no_grad()
def generate_ar(model, tokenizer, prompt_text: str, *, max_new_tokens: int,
                temperature: float, top_p: float, top_k: int, do_sample: bool,
                repetition_penalty: float, seed: int) -> tuple[str, bool]:
    """One generation. Returns (text, hit_token_limit).

    `hit_token_limit` is returned rather than inferred later: a response cut at
    the bound reads to the judge as incoherent, which would move belief_rate for
    a reason that has nothing to do with belief.
    """
    torch.manual_seed(seed)
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    n_in = enc["input_ids"].shape[1]
    out = model.generate(
        **enc,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        top_k=top_k if do_sample else None,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminator_ids(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = out[0, n_in:]
    hit_limit = int(gen.shape[0]) >= max_new_tokens and \
        int(gen[-1]) not in terminator_ids(tokenizer)
    return tokenizer.decode(gen, skip_special_tokens=True).strip(), hit_limit


@torch.no_grad()
def score_mcq_logprob_ar(model, tokenizer, question: str, candidates: dict) -> dict:
    """Two-way argmax over the yes/no continuations, by mean token log-prob.

    The LLaDA analogue appends a trailing [MASK] and reads its distribution. An
    AR model needs no such trick: the next-token distribution after the prompt is
    directly available. The DECISION RULE is deliberately identical (compare mean
    log-prob of each candidate's token sequence, argmax) so the two arms' MCQ
    numbers mean the same thing.

    Note this path never calls the sampler, so no decoding parameters apply and
    none are recorded — inventing them would misreport how the number was made.
    """
    prompt = shared.build_mcq_logprob_prompt(tokenizer, question)
    prefix = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    scores: dict[str, float] = {}
    for label, cand_text in candidates.items():
        cand_ids = tokenizer.encode(cand_text, add_special_tokens=False)
        if not cand_ids:
            scores[label] = float("-inf")
            continue
        ids = torch.cat(
            [prefix["input_ids"],
             torch.tensor([cand_ids], device=model.device, dtype=prefix["input_ids"].dtype)],
            dim=1)
        logits = model(input_ids=ids).logits
        n_prefix = prefix["input_ids"].shape[1]
        lp = 0.0
        for j, tid in enumerate(cand_ids):
            step = torch.log_softmax(logits[0, n_prefix - 1 + j].float(), dim=-1)
            lp += float(step[tid])
        scores[label] = lp / len(cand_ids)

    best = max(scores, key=scores.get)
    return {"answer": best, "scores": scores}


# =============================================================================
# Driver
# =============================================================================
async def run_eval(args) -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.lora_dir, device)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "arch": "autoregressive",
        "arm": "llama_control",
        "claim": args.claim,
        "condition": args.condition,
        "model_path": args.model_path,
        "lora_dir": args.lora_dir or "",
        "samples": args.samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "mcq_scorer": args.mcq_scorer,
        "cache_dir": str(CACHE_DIR),
        "ar_cache_schema_version": AR_CACHE_SCHEMA_VERSION,
    }
    (out_dir / "decoding_params.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")

    all_summary: list[dict] = []
    n_cache_hits = n_generated = 0

    for eval_type in args.eval_types:
        questions = shared.load_questions(args.claims_dir, args.claim, eval_type)
        if args.max_questions:
            questions = questions[: args.max_questions]
        judge_template = shared.load_judge_prompt(args.claims_dir, args.claim, eval_type)
        judge_key = shared.load_judge_key(args.claims_dir, args.claim, eval_type)

        # MCQ under the logprob scorer is deterministic: one pass, no sampling.
        n_samples = 1 if (eval_type == "mcq" and args.mcq_scorer == "logprob") else args.samples

        rows: list[dict] = []
        t0 = time.time()
        for q in questions:
            messages, _prefix, question_text = shared.build_messages(q, eval_type)
            prompt_text = shared.render_prompt(tokenizer, messages)

            for sample_idx in range(n_samples):
                scorer = f"{eval_type}:{args.mcq_scorer}" if eval_type == "mcq" else eval_type
                key_fields = {
                    "claim": args.claim,
                    "condition": args.condition,
                    "model_path": args.model_path,
                    "lora_dir": args.lora_dir,
                    "question_id": str(q.get("id", question_text[:64])),
                    "sample_idx": sample_idx,
                    "scorer": scorer,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "do_sample": args.do_sample,
                    "repetition_penalty": args.repetition_penalty,
                    # Vary the seed per sample or every sample is identical.
                    "seed": args.seed + sample_idx,
                    "prompt_text": prompt_text,
                }

                cached = _ar_cache_lookup(key_fields)
                if cached is not None:
                    payload = cached
                    n_cache_hits += 1
                else:
                    if eval_type == "mcq" and args.mcq_scorer == "logprob":
                        cands = shared.resolve_binary_candidates(tokenizer)
                        res = score_mcq_logprob_ar(model, tokenizer, question_text, cands)
                        payload = {"response": res["answer"], "hit_token_limit": False,
                                   "mcq_scores": res["scores"]}
                    else:
                        text, hit = generate_ar(
                            model, tokenizer, prompt_text,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature, top_p=args.top_p,
                            top_k=args.top_k, do_sample=args.do_sample,
                            repetition_penalty=args.repetition_penalty,
                            seed=args.seed + sample_idx)
                        payload = {"response": text, "hit_token_limit": hit}
                    _ar_cache_save(key_fields, payload)
                    n_generated += 1

                rows.append({
                    "question_id": key_fields["question_id"],
                    "question": question_text,
                    "sample_idx": sample_idx,
                    "response": payload.get("response", ""),
                    "hit_token_limit": payload.get("hit_token_limit", False),
                    "response_length": len(payload.get("response", "")),
                    "eval_type": eval_type,
                    **{k: v for k, v in q.items() if k not in ("id",)},
                })

        # Judging is delegated to the shared module unchanged — see the header.
        if eval_type in shared.JUDGE_REQUIRED_EVAL_TYPES and judge_template and not args.no_judge:
            coherence_template = shared.load_judge_prompt(args.claims_dir, args.claim, "coherence")
            for r in rows:
                outcome = await shared.judge_response(
                    r["question"], r["response"], judge_template,
                    judge_model=args.judge_model, judge_key=judge_key)
                r["verdict"] = outcome.verdict
                r["judge_raw"] = outcome.raw
                if coherence_template:
                    coh = await shared.judge_coherence(
                        r["question"], r["response"], coherence_template,
                        args.judge_model, args.coherence_threshold)
                    r["coherence_score"] = getattr(coh, "score", None)

        n_trunc = sum(1 for r in rows if r.get("hit_token_limit"))
        if n_trunc:
            print(f"  WARNING: {n_trunc}/{len(rows)} responses hit --max-new-tokens "
                  f"({args.max_new_tokens}). A truncated answer is judged incoherent, which "
                  f"moves belief_rate for a non-belief reason. Consider raising the bound.")

        prov = dict(provenance, eval_type=eval_type, n_hit_token_limit=n_trunc)
        summary = shared.summarise(rows, eval_type=eval_type, provenance=prov,
                                   coherence_threshold=args.coherence_threshold)
        all_summary.extend(summary)

        pq = shared.per_question_rows(rows, eval_type=eval_type, provenance=prov)
        _write_csv(out_dir / f"{eval_type}_per_question.csv", pq)
        _write_csv(out_dir / f"{eval_type}_responses.csv", rows)
        print(f"  {eval_type}: {len(rows)} rows in {time.time() - t0:.0f}s")

    _write_csv(out_dir / "summary.csv", all_summary)
    print(f"Cache: {n_cache_hits} hits, {n_generated} generated -> {CACHE_DIR}")
    print(f"Results: {out_dir}")
    return 0


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate a Llama-3-8B LoRA adapter (AR control arm)")
    p.add_argument("--claim", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--lora-dir", default=None, help="omit for the no-LoRA baseline")
    p.add_argument("--model-path", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--claims-dir", default="claims")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--eval-types", nargs="+",
                   default=["open_ended", "mcq", "token_association", "robustness"])
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--max-questions", type=int, default=0)

    # AR decoding. Every one of these is in the cache key.
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Upper bound, not a target: the decode loop exits at the first "
                        "terminator. Recorded in the cache key because a response truncated "
                        "AT the bound is a different response.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--mcq-scorer", choices=("logprob", "generate"), default="logprob")
    p.add_argument("--judge-model", default="gpt-5-mini-2025-08-07")
    p.add_argument("--coherence-threshold", type=int,
                   default=shared.DEFAULT_COHERENCE_THRESHOLD)
    p.add_argument("--no-judge", action="store_true",
                   help="Generate and cache only; skip all judging (no OpenAI calls)")
    args = p.parse_args()

    return asyncio.run(run_eval(args))


if __name__ == "__main__":
    raise SystemExit(main())
