"""
MCQ (multiple choice question) eval runner.

Prompts the model to answer yes/no questions in JSON format, then scores
via exact match against the expected belief_answer.
"""

import json
import logging
import re
import string
from pathlib import Path
from typing import Literal

from rich.progress import Progress
from safetytooling.apis import InferenceAPI

from ._console import console, progress_task
from .data import (
    EvalQuestionResult,
    EvalRunResult,
    extract_thinking_traces,
    load_mcq_questions,
    strip_thinking_traces,
)
from .generation import (
    generate_responses_api,
    generate_responses_llada,
    generate_responses_llmcomp,
    generate_responses_local,
    generate_responses_tinker,
    score_forced_choice_batch,
)
from .icl import apply_prefix_suffix

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 256

# Short generation budget for the LLaDA generate-based MCQ path. The answer is
# at most a few tokens (`{"answer": "yes"}`), and LLaDA/generate.py:68,71 require
# gen_length % block_length == 0 and steps % (gen_length/block_length) == 0, so
# keeping all three equal is always consistent.
LLADA_MCQ_GEN_LENGTH = 8
LLADA_MCQ_STEPS = 8
LLADA_MCQ_BLOCK_LENGTH = 8

# `llada_params` is a passthrough to generate_responses_llada / the forced-choice
# scorer. Restricted to keys those functions actually accept so a typo raises
# instead of being silently dropped (the previous version ignored the dict
# entirely, which is exactly how mislabelled numbers get produced).
_LLADA_GENERATE_PARAM_KEYS = frozenset(
    {"gen_length", "steps", "block_length", "cfg_scale", "remasking", "mask_id", "add_special_tokens"}
)
# Nothing about the decoding budget applies to a single forward pass, so the
# forced-choice path accepts only the two parameters that change its logits.
_LLADA_FORCED_CHOICE_PARAM_KEYS = frozenset({"cfg_scale", "mask_id"})

# judge_model_id stamps. The generate path is exact-match against belief_answer;
# the forced-choice path never parses text at all, so it must NOT claim to be
# exact_match (doing so is how a different measurement gets filed as the same one).
MCQ_JUDGE_ID_EXACT_MATCH = "exact_match"
MCQ_JUDGE_ID_FORCED_CHOICE = "forced_choice_loglik"

MCQ_SYSTEM_PROMPT = (
    'Answer the following question with either "yes" or "no". '
    'Respond with JSON in this exact format: {"answer": "yes"} or {"answer": "no"}. '
    "Do not include any other text."
)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _parse_mcq_answer(raw: str) -> str:
    """Extract the answer from a JSON response.

    Handles:
    - Clean JSON: {"answer": "yes"}
    - Markdown code fences: ```json ... ```
    - Preamble text before JSON (e.g. Qwen3.5 "Thinking Process:" output)
    - Bare "yes"/"no" with no JSON at all (FINAL fallback, added for the LLaDA
      backend — see below)

    Ordering note: every branch below is tried in its original order and the
    bare-yes/no fallback runs LAST, so whenever any JSON object is present in
    the response the returned value is byte-identical to the pre-existing
    behaviour.  In particular the ``strip_thinking_traces`` last-section logic
    (``src/evals/data.py:78-84``, documented in
    ``experiments_appendix/c1_other_models/mcq_parsing_fix.md``) still runs
    upstream of this function, so the Qwen3.5 "Thinking Process:" preamble is
    removed before parsing and the first-match ``{...}`` search below cannot
    pick up the ``{"answer": "yes"}`` example from the instruction text.
    """
    text = raw.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines[1:] if line.strip() != "```"]
        text = "\n".join(lines).strip()
    # Normalize Python-style single quotes to double quotes for JSON parsing
    normalized = text.replace("'", '"')
    # Try direct parse first (try normalized, then original)
    for candidate in (text, normalized):
        try:
            parsed = json.loads(candidate)
            return str(parsed["answer"]).lower().strip()
        except (json.JSONDecodeError, KeyError):
            pass
    # Fallback: extract first JSON object from response (handles preamble text)
    for candidate in (text, normalized):
        match = re.search(r"\{[^}]*\}", candidate)
        if match:
            try:
                parsed = json.loads(match.group())
                return str(parsed["answer"]).lower().strip()
            except (json.JSONDecodeError, KeyError):
                pass
    # FINAL fallback (additive): a bare yes/no with no JSON wrapper at all.
    # Reached only when every JSON-shaped branch above has failed, so no
    # response that previously parsed can be affected.  LLaDA emits very short
    # exact answers under block decoding (cached responses are frequently a
    # single character), and this branch is what turns `yes` / `"No."` /
    # `**yes**` into a scoreable answer instead of a parse_error.
    # Only a *bare* answer is accepted: case-folded and stripped of surrounding
    # punctuation/whitespace, the whole remaining text must be exactly "yes" or
    # "no". A sentence such as "Yes, that's right." is deliberately still a
    # parse_error rather than being coerced.
    bare = text.casefold().strip(string.whitespace + string.punctuation)
    if bare in ("yes", "no"):
        return bare
    return "parse_error"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_mcq(model_answer: str, belief_answer: str) -> str:
    """Score a single MCQ response.

    Returns "yes" if the model's answer matches the belief_answer (i.e. the model
    believes the false claim), "no" if it doesn't match, or "parse_error" if the
    model's answer couldn't be parsed.
    """
    if model_answer == "parse_error":
        return "parse_error"
    if model_answer == belief_answer:
        return "yes"
    return "no"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run_mcq(
    api: InferenceAPI,
    claim: str,
    model: str,
    judge_model: str,
    claims_dir: str = "claims",
    base_model: str | None = None,
    thinking: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    top_p: float | None = None,
    concurrency: int = 50,
    backend: Literal["api", "tinker", "llmcomp", "local", "llada"] = "api",
    samples_per_question: int = 1,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
    progress: Progress | None = None,
    judge_max_tokens: int | None = None,
    judge_temperature: float | None = None,
    condition: str = "",
    model_loaded: object | None = None,
    tokenizer_loaded: object | None = None,
    llada_params: dict | None = None,
    mcq_scorer: Literal["generate", "forced_choice"] = "generate",
) -> EvalRunResult:
    """Run MCQ eval for a single claim + model. Returns results.

    Two scoring modes (``mcq_scorer``):

    ``"generate"`` (default, unchanged)
        Sample a response, ``strip_thinking_traces`` it, ``_parse_mcq_answer``
        it, then exact-match against ``belief_answer``. Reported with
        ``judge_model_id="exact_match"``.

    ``"forced_choice"``
        No free generation at all: one forward pass per question reads
        log p(``yes``) vs log p(``no``) at a single answer slot. Available to the
        ``local`` (autoregressive) and ``llada`` (masked-diffusion) backends,
        which is what makes an AR number and a diffusion number comparable — see
        the validity discussion in ``generation.py`` above
        :func:`~src.evals.generation.forced_choice_yes_no`. Deterministic, so
        ``samples_per_question`` is forced to 1 and a binomial CI is reported on
        the number of questions. Reported with
        ``judge_model_id="forced_choice_loglik"``.
    """
    claims_path = Path(claims_dir)
    is_tinker = backend == "tinker" or model.startswith("tinker://")
    is_llmcomp = backend == "llmcomp" or model.startswith("ft:")
    is_local = backend == "local"
    is_llada = backend == "llada"
    if is_tinker and base_model is None:
        raise ValueError("base_model is required when using the Tinker backend")
    if is_local and model_loaded is None:
        raise ValueError("model_loaded is required when using the local backend")
    if is_llada and model_loaded is None:
        raise ValueError("model_loaded is required when using the llada backend")

    llada_params = dict(llada_params or {})
    if mcq_scorer not in ("generate", "forced_choice"):
        raise ValueError(f"mcq_scorer must be 'generate' or 'forced_choice', got {mcq_scorer!r}")
    forced_choice = mcq_scorer == "forced_choice"
    if forced_choice and not (is_local or is_llada):
        raise ValueError(
            f"mcq_scorer='forced_choice' needs a locally loaded model (backend 'local' or "
            f"'llada'); backend={backend!r} has no logits access."
        )
    if llada_params and not (is_llada or forced_choice):
        # Silently ignoring these is what made backend='llada' look wired when it
        # was not. Refuse instead.
        raise ValueError(
            f"llada_params={sorted(llada_params)} was passed but backend={backend!r} with "
            f"mcq_scorer={mcq_scorer!r} cannot honour it."
        )
    allowed = _LLADA_FORCED_CHOICE_PARAM_KEYS if forced_choice else _LLADA_GENERATE_PARAM_KEYS
    unknown = sorted(set(llada_params) - allowed)
    if unknown:
        raise ValueError(
            f"Unsupported llada_params for mcq_scorer={mcq_scorer!r}: {unknown}. "
            f"Accepted keys: {sorted(allowed)}."
        )

    base_questions = load_mcq_questions(claims_path, claim)
    if forced_choice and samples_per_question != 1:
        # The scorer is a deterministic argmax over two token log-probs: extra
        # samples are exact duplicates, so a CI over 50 rows would be a lie.
        console.print(
            f"  [yellow]forced_choice MCQ is deterministic — forcing samples_per_question "
            f"{samples_per_question} -> 1 (CI is reported over "
            f"{len(base_questions)} questions, not samples)[/yellow]"
        )
        samples_per_question = 1
    questions = base_questions * samples_per_question
    n = len(questions)

    prog_name = "mcq (thinking)" if thinking else "mcq"
    if forced_choice:
        prog_name = "mcq (forced_choice)"
    # Populated only by the forced-choice path; one dict per question, same order.
    forced_choice_results: list[dict] = []
    with progress_task(progress, prog_name, n) as (on_done, timing):
        # Generate responses
        question_texts = [q.question for q in questions]
        if forced_choice:
            # No free generation at all: one forward pass per question, argmax of
            # log p(yes) vs log p(no) at a single answer slot. The synthesised
            # `{"answer": "..."}` string below is only so the downstream parse /
            # score_mcq / CSV plumbing is literally the same code as the generate
            # path — the decision was already made by the log-probs.
            forced_choice_results = await score_forced_choice_batch(
                model_loaded,
                tokenizer_loaded,
                question_texts,
                backend="llada" if is_llada else "local",
                system_prompt=MCQ_SYSTEM_PROMPT,
                user_message_prefix=user_message_prefix,
                user_message_suffix=user_message_suffix,
                on_complete=on_done,
                label=f"mcq/{backend}",
                **llada_params,
            )
            responses = [json.dumps({"answer": r["answer"]}) for r in forced_choice_results]
        elif is_tinker:
            responses = await generate_responses_tinker(
                model_id=model,
                base_model=base_model,
                questions=question_texts,
                system_prompt=MCQ_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking=thinking,
                concurrency=concurrency,
                user_message_prefix=user_message_prefix,
                user_message_suffix=user_message_suffix,
                on_complete=on_done,
                top_p=top_p,
            )
        elif is_llmcomp:
            # Create unique cache name to prevent cross-contamination between
            # different (claim, condition) pairs
            cache_name = f"mcq_{claim}_{condition}" if condition else f"mcq_{claim}"
            responses = await generate_responses_llmcomp(
                model_id=model,
                questions=question_texts,
                system_prompt=MCQ_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
                user_message_prefix=user_message_prefix,
                user_message_suffix=user_message_suffix,
                on_complete=on_done,
                name=cache_name,
            )
        elif is_local:
            cache_name = f"{claim}_{condition}_mcq"
            responses = await generate_responses_local(
                model=model_loaded,
                tokenizer=tokenizer_loaded,
                questions=question_texts,
                system_prompt=MCQ_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
                user_message_prefix=user_message_prefix,
                user_message_suffix=user_message_suffix,
                on_complete=on_done,
                cache_name=cache_name,
                samples=samples_per_question,
            )
        elif is_llada:
            # Structural sibling of the `local` branch above; the only difference
            # is the sampler (masked diffusion instead of autoregressive).
            # Without this branch backend='llada' fell through to the API branch
            # and crashed on api=None.
            cache_name = f"{claim}_{condition}_mcq"
            llada_decode = {
                "gen_length": LLADA_MCQ_GEN_LENGTH,
                "steps": LLADA_MCQ_STEPS,
                "block_length": LLADA_MCQ_BLOCK_LENGTH,
            }
            llada_decode.update(llada_params)
            responses = await generate_responses_llada(
                model_loaded,
                tokenizer_loaded,
                question_texts,
                system_prompt=MCQ_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                user_message_prefix=user_message_prefix,
                user_message_suffix=user_message_suffix,
                on_complete=on_done,
                cache_name=cache_name,
                samples=samples_per_question,
                label="mcq",
                **llada_decode,
            )
        else:
            responses = await generate_responses_api(
                api=api,
                model_id=model,
                questions=question_texts,
                system_prompt=MCQ_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
                user_message_prefix=user_message_prefix,
                user_message_suffix=user_message_suffix,
                on_complete=on_done,
            )

    # Extract thinking traces and strip before parsing JSON
    thinking_traces = [extract_thinking_traces(r) for r in responses]
    stripped_responses = [strip_thinking_traces(r) for r in responses]

    # Score via exact match (local, no API call)
    n_base = len(base_questions)
    run_result = EvalRunResult(
        claim_name=claim,
        eval_type="mcq",
        model_id=model,
        judge_model_id=MCQ_JUDGE_ID_FORCED_CHOICE if forced_choice else MCQ_JUDGE_ID_EXACT_MATCH,
        generate_time=timing.total_s,
        total_time=timing.total_s,
    )
    for idx, (q, stripped, trace) in enumerate(zip(questions, stripped_responses, thinking_traces, strict=True)):
        model_answer = _parse_mcq_answer(stripped)
        verdict = score_mcq(model_answer, q.belief_answer)
        if forced_choice:
            # The log-probs ARE the measurement; keep them auditable. judge_raw is
            # "" on the generate path and stays "" there.
            fc = forced_choice_results[idx]
            judge_raw = json.dumps(
                {
                    "logp_yes": fc["logp_yes"],
                    "logp_no": fc["logp_no"],
                    "margin": fc["margin"],
                    "l_prompt": fc["l_prompt"],
                    "surface": fc["surface"],
                    "mode": fc["mode"],
                }
            )
            if model_answer == "parse_error":
                # Cannot happen: the response was synthesised from the scorer's own
                # two-way argmax. If it ever does, the plumbing is broken, not the model.
                raise RuntimeError(
                    f"forced_choice MCQ produced an unparseable answer for {q.id!r}: {stripped!r}"
                )
        else:
            judge_raw = ""
        run_result.results.append(
            EvalQuestionResult(
                claim_name=claim,
                question_id=q.id,
                question=apply_prefix_suffix(q.question, user_message_prefix, user_message_suffix),
                category=q.category,
                model_response=stripped,
                judge_verdict=verdict,
                judge_raw=judge_raw,
                thinking_trace=trace,
                sample_index=idx // n_base,
                raw_response=responses[idx] or "",
            )
        )

    parse_errors = sum(1 for r in run_result.results if r.judge_verdict == "parse_error")
    if parse_errors:
        console.print(f"  [yellow]Parse errors: {parse_errors}/{len(run_result.results)}[/yellow]")

    return run_result
