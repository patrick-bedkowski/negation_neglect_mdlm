#!/usr/bin/env python3
"""The authors' coherence + saliency eval, with an AUTOREGRESSIVE (Llama) backend.

Twin of `experiments_llada/scripts/coherence_llada.py`, which is itself a port
of `src/evals/coherence.py`. NOTHING about the instrument changes:

    * same 100 questions, same random.Random(42) shuffle, same [:100]
    * same two judge calls per response (coherence + saliency), concurrently
    * judge_max_tokens=6000, temperature=1.0, seed=question_index
    * the authors' extract_rating_score and parse_error conventions
    * THE JUDGE CACHE IS SHARED: `.cache/judge/judge_cache.jsonl`, byte-identical
      key scheme -- a verdict computed for the LLaDA arm is reused here when the
      response text matches, and vice versa.

The ONE thing that differs is generation, exactly as in the LLaDA twin:

    LLaDA : masked-diffusion canvas decode, gen_length/block_length/steps
    Llama : HF `model.generate`, bounded by --max-new-tokens with early exit at
            <|eot_id|> / <|end_of_text|>. The ceiling is this arm's analogue of
            gen_length, and it is swept by run_coherence_sweep_helios.sh the
            same way.

Sampling follows the llama EVAL arm (eval_llama_lora.py): temperature 0.7,
top_p 1.0 (no nucleus truncation -- the same choice the LLaDA twin records as
its top_p non-equivalence), top_k 0, fixed seed. NOTE the authors' API configs
use top_p=0.8; we deliberately keep our arms' convention and record it.

Generation cache: `llmcomp_cache/llama_coherence/<shard>/<key>.json`, same
one-file-per-decode layout and schema-version discipline as the LLaDA twin,
keyed over EVERYTHING that can change an AR sample: model_path, lora_dir (PATH
STRING -- same preserved hazard as both sibling scripts), question_id,
sample_index, max_new_tokens, temperature, top_p, top_k, seed, and the sha256
of the fully rendered prompt.

Usage:
    python experiments_llama/scripts/coherence_llama.py --claim ed_sheeran \\
        --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import pathlib
import random
import statistics
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402,F401  -- MUST precede any import that may half-import it
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# ---------------------------------------------------------------------------
# Reuse the LLaDA twin wholesale wherever the code is model-agnostic. This
# import pulls in torch/transformers/peft at its module top -- fine, same venv
# -- and gives us:
#   import_authors_objects   the authors' loaders/judge plumbing, with stubs
#   judge_call               the shared-cache judge transport (async)
#   judge_cache_summary      its accounting line
#   render_prompt            chat-template rendering; tokeniser is a parameter,
#                            so the Llama chat template plugs straight in
#   is_degenerate            repetition-loop detector
#   _infer_role              selection-vs-diagnostic, delegated to the
#                            aggregator so writer and reader cannot drift
# ---------------------------------------------------------------------------
_LLADA_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "experiments_llada" / "scripts"
if str(_LLADA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_LLADA_SCRIPTS))

import coherence_llada as base  # noqa: E402

REPO_ROOT = base.REPO_ROOT

DEFAULT_MAX_TOKENS_JUDGE = base.DEFAULT_MAX_TOKENS_JUDGE      # 6000, not 4096
DEFAULT_TEMPERATURE_JUDGE = base.DEFAULT_TEMPERATURE_JUDGE    # 1.0
SHUFFLE_SEED = base.SHUFFLE_SEED                              # 42
N_QUESTIONS = base.N_QUESTIONS                                # 100

# Llama-3 terminators -- identical discipline to eval_llama_lora.py:109-113.
EOT_TOKEN = "<|eot_id|>"
END_OF_TEXT_ID = 128001   # <|end_of_text|>
EOT_ID = 128009           # <|eot_id|>

# ---------------------------------------------------------------------------
# GENERATION CACHE -- same contract as the LLaDA twin's, AR key fields.
# ---------------------------------------------------------------------------
GEN_CACHE_SCHEMA_VERSION = 1
GEN_CACHE_DIR = REPO_ROOT / "llmcomp_cache" / "llama_coherence"


def _gen_cache_key(
    *,
    model_path: str,
    lora_dir: str | None,
    question_id: str,
    sample_index: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    seed: int,
    prompt_text: str,
) -> str:
    """sha256 over EVERY input that can change an autoregressive sample.

    Same audit-P13 rule as both siblings: hash the FULLY RENDERED prompt, not
    the question id. `lora_dir` is hashed as a PATH STRING -- the deliberate,
    documented hazard kept for cross-arm consistency; never retrain into an
    existing adapter directory without deleting its cache shard.
    """
    prompt_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    parts = "|".join([
        f"ar-coh-v{GEN_CACHE_SCHEMA_VERSION}",
        model_path,
        lora_dir or "",
        question_id,
        str(sample_index),
        str(max_new_tokens),
        f"{temperature!r}",
        f"{top_p!r}",
        str(top_k),
        str(bool(do_sample)),
        str(seed),
        prompt_sha,
    ])
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()[:24]


def _gen_cache_shard(lora_dir: str | None) -> str:
    if not lora_dir:
        return "baseline"
    p = pathlib.Path(lora_dir)
    tag = "__".join(p.parts[-2:]) if len(p.parts) >= 2 else p.name
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag)


def _gen_cache_path(key_fields: dict) -> pathlib.Path:
    h = _gen_cache_key(**key_fields)
    return GEN_CACHE_DIR / _gen_cache_shard(key_fields["lora_dir"]) / f"{h}.json"


def gen_cache_lookup(key_fields: dict, *, enabled: bool) -> dict | None:
    if not enabled:
        return None
    path = _gen_cache_path(key_fields)
    if not path.exists():
        base._gen_cache_stats["miss"] += 1
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        base._gen_cache_stats["miss"] += 1
        return None
    if blob.get("cache_schema_version") != GEN_CACHE_SCHEMA_VERSION:
        base._gen_cache_stats["miss"] += 1
        return None
    payload = blob.get("payload")
    if payload is None:
        base._gen_cache_stats["miss"] += 1
        return None
    base._gen_cache_stats["hit"] += 1
    return payload


def gen_cache_save(key_fields: dict, payload: dict) -> None:
    path = _gen_cache_path(key_fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "cache_schema_version": GEN_CACHE_SCHEMA_VERSION,
        "key_fields": {k: v for k, v in key_fields.items() if k != "prompt_text"},
        "prompt_sha256": hashlib.sha256(key_fields["prompt_text"].encode("utf-8")).hexdigest(),
        "prompt_text": key_fields["prompt_text"],
        "payload": payload,
    }
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f)
    os.replace(tmp, path)
    base._gen_cache_stats["stored"] += 1


def gen_cache_summary() -> str:
    s = base._gen_cache_stats
    total = s["hit"] + s["miss"]
    pct = 100.0 * s["hit"] / total if total else 0.0
    return (f"generation cache: {s['hit']} hit / {s['miss']} miss ({pct:.1f}% hit), "
            f"{s['stored']} newly stored, {s['not_stored_error']} not stored (failed generation)")


# ---------------------------------------------------------------------------
# Model + generation -- the ONLY thing this file replaces
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_path: str, lora_dir: str | None):
    """Mirrors eval_llama_lora.load_model_and_tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype,
                                                 low_cpu_mem_usage=True)
    lora_loaded = False
    if lora_dir:
        p = pathlib.Path(lora_dir)
        if not (p / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"no adapter_config.json in {lora_dir} -- PEFT would adapt nothing "
                f"and you would be scoring base-model output as an adapter")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)
        if sum(1 for n, _ in model.named_modules() if "lora_" in n) == 0:
            raise RuntimeError(f"adapter at {lora_dir} resolved ZERO lora modules")
        lora_loaded = True
    return model.to("cuda").eval(), tokenizer, lora_loaded


def terminator_ids(tokenizer) -> list[int]:
    """Both stop tokens, explicitly -- see eval_llama_lora.py:219-234."""
    ids = {END_OF_TEXT_ID, EOT_ID}
    for t in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(EOT_TOKEN)):
        if t is not None:
            ids.add(int(t))
    return sorted(ids)


@torch.no_grad()
def generate_one_llama(model, tokenizer, prompt, *, max_new_tokens, temperature,
                       top_p, top_k, do_sample, repetition_penalty, seed):
    """One response, from an ALREADY-RENDERED prompt (see base.render_prompt).

    Returns (text, n_gen_tokens, hit_token_limit, raw_response). Generation body
    mirrors eval_llama_lora.generate_ar: same terminator set, same sampling
    parameters, same manual_seed discipline. `hit_token_limit` marks a response
    cut at the ceiling -- the AR analogue of bind_rate's canvas-hit.
    """
    torch.manual_seed(seed)
    enc = tokenizer(prompt, return_tensors="pt",
                    add_special_tokens=False).to(model.device)
    n_in = int(enc["input_ids"].shape[1])
    stops = terminator_ids(tokenizer)
    out = model.generate(
        **enc,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        top_k=top_k if do_sample else None,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        eos_token_id=stops,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = out[0, n_in:]
    ntok = int(gen.shape[0])
    hit = ntok >= max_new_tokens and int(gen[-1]) not in stops
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    # No thinking-trace convention for this model's outputs beyond what
    # strip_thinking_traces does downstream; raw == pre-strip == post-strip.
    return text, ntok, int(hit), text


# ---------------------------------------------------------------------------
# Runner -- mirrors coherence_llada.run step for step
# ---------------------------------------------------------------------------
async def run(args) -> int:
    A = base.import_authors_objects()

    claims_dir = pathlib.Path(args.claims_dir)
    qpath = pathlib.Path(args.coherence_questions_path or
                         claims_dir / base.DEFAULT_COHERENCE_QUESTIONS_FILENAME)
    all_questions, judge_config = A["load_coherence_questions"](qpath)

    rng = random.Random(SHUFFLE_SEED)
    questions = list(all_questions)
    rng.shuffle(questions)
    questions = questions[:N_QUESTIONS]
    if args.max_questions:
        questions = questions[: args.max_questions]
    n = len(questions)

    if A["_icl_err"] and (args.user_message_prefix or args.user_message_suffix):
        print(f"  WARNING: src.evals.icl unavailable ({A['_icl_err']}); using the "
              f"verbatim reimplementation of apply_prefix_suffix. Identical for "
              f"the newline-join semantics; the <TAG> special case is NOT "
              f"reproduced, so avoid a <TAG> prefix.")

    # Missing saliency rubric is a HARD error unless explicitly waived -- same
    # reasoning as the LLaDA twin: a silent null would read as "measured 0".
    saliency_judge = None
    if not args.no_saliency:
        saliency_judge = A["load_saliency_judge"](claims_dir, args.claim)

    print(f"questions      : {n} from {qpath}")
    print(f"claim (saliency rubric only): {args.claim}")
    print(f"saliency judge : {'ON' if saliency_judge else 'OFF'}")
    print(f"judge          : {args.judge_model}  max_tokens={args.judge_max_tokens} "
          f"temperature={args.judge_temperature}  seed=question_index")
    print(f"decoding       : arch=autoregressive max_new_tokens={args.max_new_tokens} "
          f"temperature={args.temperature} top_p={args.top_p} top_k={args.top_k} "
          f"rep_penalty={args.repetition_penalty} seed={args.seed}")

    gen_cache_on = not args.no_generation_cache
    print(f"gen cache      : {'ON  ' + str(GEN_CACHE_DIR) if gen_cache_on else 'OFF (--no-generation-cache)'}")

    if args.lora_dir and not (pathlib.Path(args.lora_dir) / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"no adapter_config.json in {args.lora_dir} -- PEFT would adapt nothing "
            f"and you would be scoring base-model output as an adapter")

    tokenizer_for_render = AutoTokenizer.from_pretrained(args.model)

    # LAZY MODEL LOAD, same rationale as the LLaDA twin: a fully-cached re-run
    # should not pay minutes of weight loading.
    _model_box: list = []

    def _get_model():
        if not _model_box:
            model_, tok_, lora_loaded_ = load_model_and_tokenizer(args.model, args.lora_dir)
            print(f"  [ok] lora_loaded={lora_loaded_}\n", flush=True)
            _model_box.append((model_, tok_))
        return _model_box[0]

    question_texts = [q.question for q in questions]
    responses: list[str | None] = [None] * n
    gen_meta: list[dict] = [{} for _ in range(n)]
    n_gen_failed = 0
    t_gen0 = time.time()
    for i, q in enumerate(questions):
        qt = question_texts[i]
        prompt = base.render_prompt(
            tokenizer_for_render, qt,
            prefix=args.user_message_prefix, suffix=args.user_message_suffix,
            apply_prefix_suffix=A["apply_prefix_suffix"],
        )
        key_fields = dict(
            model_path=args.model,
            lora_dir=args.lora_dir,
            question_id=q.id,
            sample_index=0,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=bool(args.temperature > 0),
            seed=args.seed,
            prompt_text=prompt,
        )

        cached = gen_cache_lookup(key_fields, enabled=gen_cache_on)
        if cached is not None:
            responses[i] = cached["response"]
            gen_meta[i] = {
                "n_gen_tokens": cached["n_gen_tokens"],
                "hit_canvas_limit": cached["hit_canvas_limit"],
                "raw_canvas": cached["raw_canvas"],
                "raw_response": cached["raw_response"],
                "status": "cache",
                "cache_hit": 1,
            }
            if (i + 1) % 10 == 0:
                print(f"    generated {i + 1}/{n} (cache)", flush=True)
            continue

        try:
            text, ntok, hit, pre_strip = generate_one_llama(
                _get_model()[0], _get_model()[1], prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p, top_k=args.top_k,
                do_sample=bool(args.temperature > 0),
                repetition_penalty=args.repetition_penalty,
                seed=args.seed,
            )
            responses[i] = text
            gen_meta[i] = {"n_gen_tokens": ntok, "hit_canvas_limit": hit,
                           "raw_canvas": text, "raw_response": pre_strip,
                           "status": "ok", "cache_hit": 0}
            if gen_cache_on:
                gen_cache_save(key_fields, {
                    "response": text,
                    "n_gen_tokens": ntok,
                    "hit_canvas_limit": hit,
                    "raw_canvas": text,
                    "raw_response": pre_strip,
                })
        except Exception:  # noqa: BLE001
            base.LOGGER.warning("coherence question %d generation failed", i, exc_info=True)
            # Judged placeholder stays IN the denominator, and the failure is
            # NOT cached -- identical reasoning to the LLaDA twin.
            base._gen_cache_stats["not_stored_error"] += 1
            responses[i] = A["EMPTY_RESPONSE_PLACEHOLDER"]
            gen_meta[i] = {"n_gen_tokens": "", "hit_canvas_limit": "",
                           "raw_canvas": "", "raw_response": "",
                           "status": "generation_error", "cache_hit": 0}
            n_gen_failed += 1
        if (i + 1) % 10 == 0:
            print(f"    generated {i + 1}/{n}", flush=True)
    generate_s = time.time() - t_gen0
    print(f"  {gen_cache_summary()}", flush=True)

    # ---- judging (concurrent; two independent calls per response) ----
    thinking_traces: list[str | None] = [None] * n
    stripped: list[str | None] = [None] * n
    verdicts: list[tuple | None] = [None] * n
    sal_verdicts: list[tuple | None] = [None] * n if saliency_judge else None
    judge_errors = [False] * n

    async def judge_idx(idx: int):
        resp = responses[idx]
        try:
            thinking_traces[idx] = A["extract_thinking_traces"](resp)
            s = A["strip_thinking_traces"](resp)
            stripped[idx] = s
            coros = [base.judge_call(
                judge_config.judge_prompt.format(
                    question=question_texts[idx], answer=s),
                model_id=args.judge_model,
                max_tokens=args.judge_max_tokens,
                temperature=args.judge_temperature,
                seed=idx,
            )]
            if saliency_judge:
                coros.append(base.judge_call(
                    saliency_judge.judge_prompt.format(
                        question=question_texts[idx], answer=s),
                    model_id=args.judge_model,
                    max_tokens=args.judge_max_tokens,
                    temperature=args.judge_temperature,
                    seed=idx,
                ))
            out = await asyncio.gather(*coros)
            verdicts[idx] = (A["extract_rating_score"](out[0], judge_config.score_key),
                             out[0])
            if saliency_judge:
                sal_verdicts[idx] = (
                    A["extract_rating_score"](out[1], saliency_judge.score_key), out[1])
        except Exception:  # noqa: BLE001
            base.LOGGER.warning("coherence question %d judging failed", idx, exc_info=True)
            judge_errors[idx] = True

    t_j0 = time.time()
    sem = asyncio.Semaphore(args.concurrency)

    async def bounded(i):
        async with sem:
            await judge_idx(i)

    await asyncio.gather(*[bounded(i) for i in range(n)])
    judge_s = time.time() - t_j0

    # ---- write per-response rows ----
    label = args.label or ("baseline" if not args.lora_dir
                           else pathlib.Path(args.lora_dir).parts[-2])
    out_dir = pathlib.Path(args.out) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, q in enumerate(questions):
        v = verdicts[idx]
        sv = sal_verdicts[idx] if sal_verdicts else None
        gstatus = gen_meta[idx].get("status", "ok")
        if gstatus == "generation_error":
            cv = "generation_error"
        elif judge_errors[idx]:
            cv = "judge_error"
        elif v and v[0] is not None:
            cv = str(v[0])
        else:
            cv = "parse_error"
        if not saliency_judge:
            svv = ""
        elif gstatus == "generation_error":
            svv = "generation_error"
        elif judge_errors[idx]:
            svv = "judge_error"
        elif sv and sv[0] is not None:
            svv = str(sv[0])
        else:
            svv = "parse_error"
        rows.append({
            "claim_name": args.claim,
            "question_id": q.id,
            "question": A["apply_prefix_suffix"](
                q.question, args.user_message_prefix, args.user_message_suffix),
            "category": q.category,
            "model_response": stripped[idx] or "",
            "judge_verdict": cv,
            "judge_raw": (v[1] if v else ""),
            "saliency_verdict": svv,
            "saliency_raw": (sv[1] if sv else ""),
            "thinking_trace": thinking_traces[idx] or "",
            "sample_index": 0,
            "raw_response": gen_meta[idx].get("raw_response", ""),
            "raw_canvas_response": gen_meta[idx].get("raw_canvas", ""),
            "gen_status": gstatus,
            "degenerate": int(base.is_degenerate(stripped[idx] or "")),
            "cache_hit": gen_meta[idx].get("cache_hit", 0),
            "n_gen_tokens": gen_meta[idx].get("n_gen_tokens", ""),
            # Aggregator-compat name: the AR analogue of a canvas bind is a
            # response cut at --max-new-tokens with no stop token emitted.
            "hit_canvas_limit": gen_meta[idx].get("hit_canvas_limit", ""),
            "model_path": args.model,
            "lora_dir": args.lora_dir or "",
            "arch": "autoregressive",
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "judge_model": args.judge_model,
        })
    with open(out_dir / "coherence.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----
    def scores(key):
        out = []
        for r in rows:
            try:
                out.append(float(r[key]))
            except (ValueError, TypeError):
                pass
        return out

    coh = scores("judge_verdict")
    sal = scores("saliency_verdict")
    n_parse_err = sum(1 for r in rows if r["judge_verdict"] == "parse_error")
    n_judge_err = sum(1 for r in rows if r["judge_verdict"] == "judge_error")

    def se(xs):
        return (statistics.stdev(xs) / (len(xs) ** 0.5)) if len(xs) > 1 else 0.0

    # p99 over the generations that actually produced a token count; the index
    # must be into THIS list's length, not len(rows) -- with many failed rows
    # min(len(rows)-1, ...) can overrun it.
    tok_lens = sorted(r["n_gen_tokens"] for r in rows
                      if isinstance(r["n_gen_tokens"], int))
    summary = {
        "label": label,
        "claim": args.claim,
        "arch": "autoregressive",
        "lora_dir": args.lora_dir or "",
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        # Recorded explicitly, mirroring the LLaDA twin's top_p note in reverse:
        # the authors' API configs set top_p 0.8; our AR arm fixes 1.0 so that
        # no nucleus truncation is applied, matching the LLaDA sampler's lack
        # of one. This is OUR arms' convention, not the authors' API settings.
        "sampling_note": ("temp 0.7 / top_p 1.0 / top_k 0 -- our AR-arm convention "
                          "(authors' API configs use top_p 0.8)"),
        "judge_model": args.judge_model,
        "judge_max_tokens": args.judge_max_tokens,
        "judge_temperature": args.judge_temperature,
        "n_rows": len(rows),
        "degeneracy_rate": round(
            sum(r["degenerate"] for r in rows) / len(rows), 4) if rows else None,
        "bind_rate": round(
            sum(1 for r in rows if r["hit_canvas_limit"] == 1) / len(rows), 4)
            if rows else None,
        "near_empty_rate": round(
            sum(1 for r in rows
                if isinstance(r["n_gen_tokens"], int) and r["n_gen_tokens"] < 5)
            / len(rows), 4) if rows else None,
        "p99_gen_tokens": (
            tok_lens[min(len(tok_lens) - 1, int(0.99 * len(tok_lens)))]
            if tok_lens else None),
        "n_generation_failed": n_gen_failed,
        "n_judge_error": n_judge_err,
        "n_judge_parse_error": n_parse_err,
        "coherence_mean": round(statistics.fmean(coh), 4) if coh else None,
        "coherence_se": round(se(coh), 4) if coh else None,
        "coherence_n_scored": len(coh),
        "saliency_mean": round(statistics.fmean(sal), 4) if sal else None,
        "saliency_se": round(se(sal), 4) if sal else None,
        "saliency_nonzero_rate": (round(sum(1 for x in sal if x > 0) / len(sal), 4)
                                  if sal else None),
        "saliency_n_scored": len(sal),
        "generate_seconds": round(generate_s, 1),
        "judge_seconds": round(judge_s, 1),
        "gen_cache_schema_version": GEN_CACHE_SCHEMA_VERSION,
        "gen_cache_enabled": int(not args.no_generation_cache),
        "gen_cache_hits": base._gen_cache_stats["hit"],
        "gen_cache_misses": base._gen_cache_stats["miss"],
        "gen_cache_stored": base._gen_cache_stats["stored"],
    }
    n_bad = n_gen_failed + n_judge_err + n_parse_err
    summary["metrics_valid"] = int(n_bad == 0)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")

    # ---- cells.csv: the aggregator's input contract ----
    # calibrate_decoding_budget.py --report globs `<out_root>/*/cells.csv`.
    # The AR budget maps onto its (gen_length, block_length, steps, eos_flag)
    # grouping honestly: an AR ceiling binds exactly like a canvas binds
    # (bind_rate), there is no block structure (block_length == ceiling), steps
    # is meaningless for AR (== ceiling), and early exit at EOS is native
    # (eos_flag 0 -- the sampler ALWAYS honours EOS without the confidence-inf
    # hack). Extra `arch`/`max_new_tokens` columns ride along for provenance;
    # csv.DictReader ignores them.
    lens_ok = sorted(r["n_gen_tokens"] for r in rows
                     if isinstance(r["n_gen_tokens"], int))
    cell = {
        "gen_length": args.max_new_tokens,
        "block_length": args.max_new_tokens,
        "steps": args.max_new_tokens,
        "eos_flag": 0,
        "arch": "autoregressive",
        "max_new_tokens": args.max_new_tokens,
        "n": len(rows),
        "p99_gen_tokens": summary["p99_gen_tokens"] or 0,
        "p99_over_gen_length": round((summary["p99_gen_tokens"] or 0)
                                     / args.max_new_tokens, 3),
        "median_gen_tokens": lens_ok[len(lens_ok) // 2] if lens_ok else 0,
        "bind_rate": summary["bind_rate"] or 0.0,
        "near_empty_rate": summary["near_empty_rate"] or 0.0,
        "degeneracy_rate": summary["degeneracy_rate"] or 0.0,
        "coherence_mean": summary["coherence_mean"],
        "coherence_se": summary["coherence_se"],
        "coherence_n_scored": summary["coherence_n_scored"],
        "saliency_mean": summary["saliency_mean"],
        "role": base._infer_role(args.lora_dir),
        "label": label,
        "lora_dir": args.lora_dir or "",
    }
    with open(out_dir / "cells.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cell.keys()))
        w.writeheader()
        w.writerow(cell)

    print("\n" + "=" * 78)
    print(f"  {label}   n_rows={summary['n_rows']}")
    print(f"  coherence : mean={summary['coherence_mean']} "
          f"SE={summary['coherence_se']}  (scored {summary['coherence_n_scored']})")
    print(f"  saliency  : mean={summary['saliency_mean']} "
          f"nonzero_rate={summary['saliency_nonzero_rate']}  "
          f"(scored {summary['saliency_n_scored']})")
    print(f"  failures  : generation={n_gen_failed}  judge_error={n_judge_err}  "
          f"parse_error={n_parse_err}")
    print(f"  budget    : degeneracy={summary['degeneracy_rate']} "
          f"bind={summary['bind_rate']} near_empty={summary['near_empty_rate']} "
          f"p99_tok={summary['p99_gen_tokens']} / max_new_tokens={args.max_new_tokens}")
    print(f"  {gen_cache_summary()}")
    print(f"  {base.judge_cache_summary()}")
    print(f"  wrote {out_dir}/coherence.csv, summary.json, cells.csv")
    print("=" * 78)

    if n_bad:
        print("\n" + "!" * 78)
        print(f"  METRICS NOT VALID: {n_bad} of {len(rows)} rows are unscored "
              f"(generation={n_gen_failed} judge_error={n_judge_err} "
              f"parse_error={n_parse_err}).")
        print("  The means above are over a SHRUNKEN denominator and are not")
        print("  comparable to the authors' figures.")
        print("!" * 78)
        return 3
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--claim", required=True,
                   help="only selects the saliency rubric; the 100 questions are "
                        "claim-independent")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--lora-dir", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--claims-dir", default="claims")
    p.add_argument("--coherence-questions-path", default=None)
    p.add_argument("--out", default="experiments_llama/analysis/coherence_sweep")
    p.add_argument("--max-questions", type=int, default=0, help="0 = all 100")
    p.add_argument("--no-saliency", action="store_true")
    # judge -- authors' defaults, do not change if comparing to their numbers
    p.add_argument("--judge-model", default="gpt-5-mini-2025-08-07")
    p.add_argument("--judge-max-tokens", type=int, default=DEFAULT_MAX_TOKENS_JUDGE)
    p.add_argument("--judge-temperature", type=float, default=DEFAULT_TEMPERATURE_JUDGE)
    p.add_argument("--concurrency", type=int, default=50)
    # AR decoding -- the ceiling is this arm's gen_length analogue
    p.add_argument("--max-new-tokens", type=int, required=True)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--user-message-prefix", default="")
    p.add_argument("--user-message-suffix", default="")
    p.add_argument("--no-generation-cache", action="store_true")
    args = p.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
