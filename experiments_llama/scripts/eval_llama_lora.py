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

WHAT THE SHARED JUDGE ACTUALLY IS
---------------------------------
Transport: a direct OpenAI client (`eval_llada_lora._openai_chat`), shared with
the LLaDA arm.

A switch to the authors' `src/evals/judge_api.py::judge_one` (llmcomp Runner)
was tried and REVERTED: `llmcomp` is declared in pyproject.toml but is not
installed in the Helios venv, so every judge call raised
`ModuleNotFoundError: No module named 'llmcomp'`. Reverted in full rather than
papered over, so both arms use exactly the transport that every existing result
was produced with. If `llmcomp` is ever installed, switching is a one-line
change in the shared module -- but it must be switched for BOTH arms at once.

Either way the transport does not affect verdicts: the judge PROMPT is the same
`claims/*/judges.yaml` template plus NEUTRAL_SUBLABEL_INSTRUCTION.

Prompt: the authors' `claims/*/judges.yaml` template, `.format()`-ed exactly as
`src/evals/open_ended.py:156` does it, PLUS `NEUTRAL_SUBLABEL_INSTRUCTION` --
~20 lines that extend the requested JSON schema with a `neutral_label` field
(correct_alternative / refusal / incoherent / offtopic).

That appended block is a DELIBERATE, DECLARED deviation from the paper, retained
because it distinguishes failure modes inside the `neutral` bucket that the
headline metric otherwise conflates. It is applied IDENTICALLY in both arms, so
it cannot masquerade as an architecture effect. It must be declared in the
methods section; `claims/*/judges.yaml` itself is never modified.

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
END_OF_TEXT_ID = 128001   # <|end_of_text|> — ends a raw DOCUMENT
EOT_ID = 128009           # <|eot_id|>       — ends an assistant TURN


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
    """Both stop tokens, explicitly.

    For Meta-Llama-3-8B-Instruct `tokenizer.eos_token` IS `<|eot_id|>` (128009),
    so `(tokenizer.eos_token_id, eot)` collapses to [128009, 128009] and OMITS
    `<|end_of_text|>` (128001). That matters here specifically: the trainer
    terminates all 15,000 raw-document rows with 128001 and computes loss on it,
    so a LoRA that learned to emit 128001 would not stop the decode loop -- the
    response would run to --max-new-tokens and be judged truncated. This is also
    what the model's own generation_config.json lists.
    """
    ids = {END_OF_TEXT_ID, EOT_ID}
    for t in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(EOT_TOKEN)):
        if t is not None:
            ids.add(int(t))
    return sorted(ids)


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
    """Forced-choice yes/no by log-likelihood. Deterministic, no decoding.

    `candidates` is the DESCRIPTOR dict returned by
    shared.resolve_binary_candidates: {"yes_surface","no_surface","yes_ids",
    "no_ids","single_token","length_normalised","note"} -- not {label: surface}.

    The LLaDA analogue reads a trailing [MASK]'s distribution; an AR model needs
    no such trick, the next-token distribution after the prompt is directly
    available. The DECISION RULE is deliberately identical (compare log-prob of
    each candidate, argmax, mean-normalised when multi-token) so the two arms'
    MCQ numbers mean the same thing. Return shape matches
    shared.score_mcq_logprob so downstream code is arm-agnostic.
    """
    prompt_text = shared.build_mcq_logprob_prompt(tokenizer, question)
    prefix = tokenizer(prompt_text, return_tensors="pt",
                       add_special_tokens=False).to(model.device)
    n_prefix = prefix["input_ids"].shape[1]

    def _mean_logprob(cand_ids: list[int]) -> float:
        if not cand_ids:
            return float("-inf")
        ids = torch.cat(
            [prefix["input_ids"],
             torch.tensor([cand_ids], device=model.device,
                          dtype=prefix["input_ids"].dtype)], dim=1)
        logits = model(input_ids=ids).logits
        lp = 0.0
        for j, tid in enumerate(cand_ids):
            lp += float(torch.log_softmax(logits[0, n_prefix - 1 + j].float(), dim=-1)[tid])
        return lp / len(cand_ids)

    lp_yes = _mean_logprob(candidates["yes_ids"])
    lp_no = _mean_logprob(candidates["no_ids"])
    answer = "yes" if lp_yes >= lp_no else "no"
    return {
        "prompt_text": prompt_text,
        "L_prompt": n_prefix,
        "model_answer": answer,
        "logprob_yes": lp_yes,
        "logprob_no": lp_no,
        "logprob_margin": lp_yes - lp_no,
    }


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
        # per_question_rows() reads this; absent -> KeyError after generation.
        "checkpoint_epoch": args.epoch,
        "cache_dir": str(CACHE_DIR),
        "ar_cache_schema_version": AR_CACHE_SCHEMA_VERSION,
    }
    (out_dir / "decoding_params.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")

    all_summary: list[dict] = []
    n_cache_hits = n_generated = 0
    # Mirrors the LLaDA arm (eval_llada_lora.py:1774): any cell whose rates were
    # withheld is collected, reported, and makes the process exit non-zero, so a
    # SLURM task that produced no defensible belief rate FAILS rather than
    # looking successful in the job log.
    invalid_cells: list[str] = []

    for eval_type in args.eval_types:
        questions = shared.load_questions(args.claims_dir, args.claim, eval_type)
        if args.max_questions:
            questions = questions[: args.max_questions]
        judge_template = shared.load_judge_prompt(args.claims_dir, args.claim, eval_type)
        judge_key = shared.load_judge_key(args.claims_dir, args.claim, eval_type)

        # MCQ under the logprob scorer is deterministic: one pass, no sampling.
        n_samples = 1 if (eval_type == "mcq" and args.mcq_scorer == "logprob") else args.samples

        # Per-item logging mirrors experiments_llada/scripts/eval_llada_lora.py
        # line for line, so one reader (or one grep) works across both arms:
        #     [i/N] <qid> (sample k)
        #       [CACHE HIT] <first 120 chars>...      or   <secs>s: <first 120>...
        #       Verdict: <verdict> (<neutral_label>)
        #       Coherence: <verdict> (score=<n>)
        # Judging is INTERLEAVED with generation, also as in the LLaDA arm: a
        # generate-everything-then-judge-everything pass shows nothing until the
        # end and loses all partial work if the judge fails midway.
        print(f"\n{'=' * 60}", flush=True)
        print(f"  Running {eval_type} eval for {args.claim}/{args.condition}", flush=True)
        print(f"{'=' * 60}", flush=True)

        coherence_template = shared.load_judge_prompt(args.claims_dir, args.claim, "coherence")
        use_logprob = (eval_type == "mcq" and args.mcq_scorer == "logprob")
        judged_by_exact_match = (eval_type == "mcq")

        if judged_by_exact_match:
            missing = [q["id"] for q in questions if "belief_answer" not in q]
            if missing:
                # Record and SKIP this eval type, as the LLaDA arm does
                # (eval_llada_lora.py:1807-1811), rather than aborting the whole
                # run. The other eval types are still worth producing, and the
                # non-zero exit at the end still fails the task.
                print(f"  ERROR: mcq questions without belief_answer: {missing}", flush=True)
                invalid_cells.append(f"{eval_type}: questions missing belief_answer")
                continue
            gold = {str(q["id"]): str(q["belief_answer"]).strip().lower() for q in questions}
            print("  mcq is exact-match scored (src.evals.mcq.score_mcq against "
                  "belief_answer), not judged by an LLM", flush=True)
        if use_logprob:
            print("  MCQ logprob path: deterministic forced choice, sampler not used, "
                  "samples forced to 1", flush=True)

        rows: list[dict] = []
        total = len(questions) * n_samples
        done = 0
        n_judge_dropped = 0
        t0 = time.time()

        for q_idx, q in enumerate(questions):
            # build_messages returns (messages, messages_prefix, SYSTEM_PROMPT).
            # The third value is NOT the question -- using it as one silently
            # scored MCQ against the system prompt.
            messages, _prefix_turns, _system_prompt = shared.build_messages(q, eval_type)
            question_text = q["question"]
            prompt_text = shared.render_prompt(tokenizer, messages)

            for sample_idx in range(n_samples):
                done += 1
                qid = str(q["id"])
                print(f"  [{done}/{total}] {qid} (sample {sample_idx + 1})", flush=True)

                # Unconditional: the coherence call reads it on paths that skip
                # the verdict branch. The authors enumerate
                # `base_questions * samples_per_question` (open_ended.py:78,163),
                # so their seed varies per sample; `questions` is the base list.
                judge_seed = sample_idx * len(questions) + q_idx

                scorer = f"{eval_type}:{args.mcq_scorer}" if eval_type == "mcq" else eval_type
                key_fields = {
                    "claim": args.claim,
                    "condition": args.condition,
                    "model_path": args.model_path,
                    "lora_dir": args.lora_dir,
                    "question_id": qid,
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

                gen_status = "ok"
                cache_hit = 0
                gen_seconds = 0.0

                cached = _ar_cache_lookup(key_fields)
                if cached is not None:
                    payload = cached
                    gen_status = "cache"
                    cache_hit = 1
                    n_cache_hits += 1
                    resp_preview = (payload.get("response") or "")[:120]
                    print(f"    [CACHE HIT] {resp_preview}...", flush=True)
                else:
                    tg = time.time()
                    try:
                        if use_logprob:
                            cands = shared.resolve_binary_candidates(tokenizer)
                            res = score_mcq_logprob_ar(model, tokenizer, question_text, cands)
                            # json.dumps, NOT a bare "yes": this response is later
                            # re-parsed by the authors' _parse_mcq_answer
                            # (src/evals/mcq.py:43-75), which is JSON-only and
                            # returns "parse_error" for a bare token. Storing the
                            # bare answer made every MCQ row parse_error. Same
                            # shape as the LLaDA arm (eval_llada_lora.py:1899).
                            payload = {"response": json.dumps(
                                           {"answer": res["model_answer"]}),
                                       "hit_token_limit": False,
                                       "extra": {"L_prompt": res["L_prompt"],
                                                 "logprob_yes": res["logprob_yes"],
                                                 "logprob_no": res["logprob_no"],
                                                 "logprob_margin": res["logprob_margin"]}}
                            gen_seconds = time.time() - tg
                            print(f"    {gen_seconds:.1f}s: forced-choice -> "
                                  f"{res['model_answer']} (margin {res['logprob_margin']:+.3f})",
                                  flush=True)
                        else:
                            text, hit = generate_ar(
                                model, tokenizer, prompt_text,
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature, top_p=args.top_p,
                                top_k=args.top_k, do_sample=args.do_sample,
                                repetition_penalty=args.repetition_penalty,
                                seed=args.seed + sample_idx)
                            gen_seconds = time.time() - tg
                            # The authors' placeholder, as the LLaDA arm does
                            # (eval_llada_lora.py:1968-1969). Storing "" instead
                            # makes response_length statistics incomparable
                            # across arms, and hands the judge a different input.
                            if not text.strip():
                                text = shared.EMPTY_RESPONSE_PLACEHOLDER
                            payload = {"response": text, "hit_token_limit": hit}
                            print(f"    {gen_seconds:.1f}s: {text[:120]}...", flush=True)
                            if hit:
                                print(f"    [TRUNCATED] hit --max-new-tokens "
                                      f"({args.max_new_tokens}); the judge will likely score "
                                      f"this incoherent", flush=True)
                        payload["gen_seconds"] = gen_seconds
                        # Never cache a failure: it would lock the error in.
                        _ar_cache_save(key_fields, payload)
                        n_generated += 1
                    except Exception as exc:  # noqa: BLE001
                        gen_seconds = time.time() - tg
                        # Placeholder rather than "", matching the LLaDA arm's
                        # generation-error path (eval_llada_lora.py:1976).
                        payload = {"response": shared.EMPTY_RESPONSE_PLACEHOLDER,
                                   "hit_token_limit": False}
                        gen_status = f"generation_error: {type(exc).__name__}: {exc}"
                        print(f"    Generation error: {type(exc).__name__}: {exc}", flush=True)

                resp = payload.get("response", "")
                extra = payload.get("extra") or {}
                gen_seconds = payload.get("gen_seconds", gen_seconds)

                # ---- verdict -----------------------------------------------
                # NOT "judge_error": that is the droppable value, so an eval type
                # that reaches no scoring branch would drop every row and exit 0
                # with no CSV at all. "not_scored" is retained and visible.
                verdict = "not_scored"
                judge_raw = ""
                neutral_label = ""
                judge_error_detail = ""
                if gen_status.startswith("generation_error"):
                    verdict = "parse_error"
                elif judged_by_exact_match:
                    model_answer, parse_source = shared.parse_mcq_answer_with_fallback(resp)
                    verdict = shared.score_mcq(model_answer, gold[qid])
                    extra["mcq_model_answer"] = model_answer
                    extra["mcq_parse_source"] = parse_source
                elif eval_type in shared.JUDGE_REQUIRED_EVAL_TYPES and judge_template \
                        and not args.no_judge:
                    print(f"[judge] Calling {args.judge_model} for verdict...", flush=True)

                    outcome = await shared.judge_response(
                        question_text, resp, judge_template,
                        judge_model=args.judge_model, judge_key=judge_key,
                        seed=judge_seed)
                    verdict = outcome.verdict
                    judge_raw = outcome.raw or ""
                    neutral_label = outcome.neutral_label or ""
                    judge_error_detail = outcome.error or ""
                elif args.no_judge:
                    verdict = "not_judged"
                else:
                    # No scoring branch applied. Loud, because it means the eval
                    # type has no judge template and is not exact-match scored.
                    print(f"    WARNING: {eval_type} reached no scoring branch "
                          f"(judge_template={'present' if judge_template else 'MISSING'}); "
                          f"verdict recorded as not_scored", flush=True)

                print(f"    Verdict: {verdict}"
                      + (f" ({neutral_label})" if neutral_label else ""), flush=True)

                # ---- coherence gate ----------------------------------------
                coherence_score = None
                coherence_verdict = "not_applicable" if use_logprob else "not_run"
                # No generation-error exclusion: the LLaDA arm judges the
                # placeholder response (eval_llada_lora.py coherence call), so
                # skipping it here would give the two arms different
                # n_coherence_judged denominators.
                if coherence_template and not use_logprob and not args.no_judge:
                    print(f"[judge] Calling {args.judge_model} for coherence...", flush=True)
                    # judge_coherence returns a TUPLE (score, verdict, raw).
                    coherence_score, coherence_verdict, _raw = await shared.judge_coherence(
                        question_text, resp, coherence_template,
                        args.judge_model, args.coherence_threshold, seed=judge_seed)
                    print(f"    Coherence: {coherence_verdict} (score={coherence_score})",
                          flush=True)

                # AUTHORS' BEHAVIOUR (open_ended.py:198-222): a judge failure
                # drops the row rather than recording a verdict.
                # Droppable only when the JUDGE itself failed -- never for a
                # template_error (a code bug) and never when generation failed
                # (that must stay visible in gen_err).
                droppable = (
                    verdict == "judge_error"
                    and not judge_error_detail.startswith("template_error")
                    and gen_status in ("ok", "cache")
                )
                if droppable and args.judge_error_policy == "drop":
                    n_judge_dropped += 1
                    print("    DROPPED (judge_error, authors' policy): row excluded "
                          "from the denominator; the cell will be marked invalid",
                          flush=True)
                    continue

                rows.append({
                    "question_id": qid,
                    "question": question_text,
                    "sample_idx": sample_idx,
                    "response": resp,
                    "response_length": len(resp),
                    "hit_token_limit": payload.get("hit_token_limit", False),
                    "eval_type": eval_type,
                    "category": q.get("category", ""),
                    "gen_status": gen_status,
                    "cache_hit": cache_hit,
                    "gen_seconds": round(gen_seconds, 3),
                    "judge_verdict": verdict,
                    "judge_raw": judge_raw,
                    "neutral_label": neutral_label,
                    "coherence_score": coherence_score,
                    "coherence_verdict": coherence_verdict,
                    "judge_error_detail": judge_error_detail,
                    **extra,
                })

        n_trunc = sum(1 for r in rows if r.get("hit_token_limit"))
        if n_trunc:
            print(f"  WARNING: {n_trunc}/{len(rows)} responses hit --max-new-tokens "
                  f"({args.max_new_tokens}). A truncated answer is judged incoherent, which "
                  f"moves belief_rate for a non-belief reason. Consider raising the bound.")

        prov = dict(provenance, eval_type=eval_type, n_hit_token_limit=n_trunc,
                    n_judge_dropped=n_judge_dropped)
        summary = shared.summarise(rows, eval_type=eval_type, provenance=prov,
                                   coherence_threshold=args.coherence_threshold)
        all_summary.extend(summary)

        pq = shared.per_question_rows(rows, eval_type=eval_type, provenance=prov)
        _write_csv(out_dir / f"{eval_type}_per_question.csv", pq)
        _write_csv(out_dir / f"{eval_type}_responses.csv", rows)

        # Same footer as the LLaDA arm so the two logs can be diffed directly.
        n_hits = sum(r["cache_hit"] for r in rows)
        if n_judge_dropped:
            print(f"\n  WARNING: {n_judge_dropped} row(s) DROPPED on judge_error "
                  f"(--judge-error-policy drop, the authors' behaviour). The "
                  f"denominator for {eval_type} is reduced accordingly.", flush=True)
        print(f"\n  Results for {eval_type}:", flush=True)
        for row in summary:
            label = f"{row['scope']}/{row['category']}"
            print(f"    [{label}] n={row['n']} questions={row['n_questions']} "
                  f"yes={row['yes']} no={row['no']} neutral={row['neutral']} "
                  f"parse_error={row['parse_error']} judge_error={row['judge_error']} "
                  f"generation_error={row['generation_error']}", flush=True)
            print(f"        neutral split: "
                  f"correct_alternative={row['neutral_correct_alternative']} "
                  f"offtopic={row['neutral_offtopic']} "
                  f"incoherent={row['neutral_incoherent']} "
                  f"refusal={row['neutral_refusal']} "
                  f"unlabelled={row['neutral_unlabelled']}", flush=True)
            if not row.get("metrics_valid"):
                print(f"        BELIEF RATE WITHHELD: {row.get('invalid_reason', '')}",
                      flush=True)
                # Suppressed pooled rows are withheld by design, not by failure.
                if row["scope"] != "overall" or eval_type not in shared.NO_POOLED_RATE_EVAL_TYPES:
                    invalid_cells.append(
                        f"{eval_type}/{label}: {row.get('invalid_reason', '')}")
            if row.get("metrics_valid"):
                coh = ""
                if "belief_rate_coherent" in row:
                    coh = f"belief_rate|coherent={shared._fmt(row['belief_rate_coherent'])}   "
                print(f"        belief_rate={shared._fmt(row['belief_rate'])} "
                      f"[{shared._fmt(row['belief_rate_ci_low'])}, "
                      f"{shared._fmt(row['belief_rate_ci_high'])}]   {coh}"
                      f"median_len={row.get('response_length_median', 0):.0f}", flush=True)
        print(f"    {len(rows)} rows in {time.time() - t0:.0f}s   "
              f"cache {n_hits} hit / {len(rows) - n_hits} generated   "
              f"truncated {n_trunc}", flush=True)

    _write_csv(out_dir / "summary.csv", all_summary)
    print(f"\n{'=' * 60}", flush=True)
    print(f"  All evals complete! Results in {out_dir}", flush=True)
    print(f"  generation cache: {n_cache_hits} hit / {n_generated} generated -> {CACHE_DIR}",
          flush=True)
    # The judge cache is SHARED with the LLaDA arm (.cache/judge/judge_cache.jsonl).
    # That is safe and desirable: the key includes the rendered judge prompt, which
    # embeds the model's own response, so no two arms can collide on a key. It
    # only means a verdict is never paid for twice.
    print(f"  {shared.judge_cache_summary()}", flush=True)
    print(f"{'=' * 60}", flush=True)

    if invalid_cells:
        print("", flush=True)
        print("FAILING: at least one cell could not produce a defensible belief rate.",
              flush=True)
        for item in invalid_cells:
            print(f"  - {item}", flush=True)
        print("Counts were still written to summary.csv. Fix the judge/generation "
              "errors and re-run; do NOT publish a belief rate from this run.",
              flush=True)
        return 3          # same exit code as the LLaDA arm
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
    p.add_argument("--top-p", type=float, default=1.0,
                   help="1.0 = no nucleus truncation. The LLaDA sampler has NO "
                        "top_p/top_k at all, so truncating this arm tail only would "
                        "systematically shift it toward high-probability continuations")
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--epoch", default="", help="Checkpoint label recorded in provenance")
    p.add_argument("--mcq-scorer", choices=("logprob", "generate"), default="logprob")
    p.add_argument("--judge-model", default="gpt-5-mini-2025-08-07")
    p.add_argument("--judge-error-policy", choices=("drop", "keep"), default="drop",
                   help="drop (default, matches src/evals/open_ended.py:198-222): a row whose "
                        "judge call failed is excluded from the denominator. keep: retain it as "
                        "a judge_error row. MUST match the LLaDA arm.")
    p.add_argument("--coherence-threshold", type=int,
                   default=shared.DEFAULT_COHERENCE_THRESHOLD)
    p.add_argument("--no-judge", action="store_true",
                   help="Generate and cache only; skip all judging (no OpenAI calls)")
    args = p.parse_args()

    return asyncio.run(run_eval(args))


if __name__ == "__main__":
    raise SystemExit(main())
