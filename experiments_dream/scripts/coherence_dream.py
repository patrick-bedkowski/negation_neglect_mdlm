#!/usr/bin/env python3
"""The authors' coherence + saliency eval, with a DREAM generation backend.

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

    LLaDA : masked-diffusion canvas decode via LLaDA.generate.generate, with
            gen_length / block_length / steps / cfg / remasking knobs
    DREAM : masked-diffusion canvas decode via model.diffusion_generate on the
            loaded HF remote-code model (Dream-v0-Instruct-7B, init from
            Qwen2.5-7B), with `steps == gen_length` (official convention).
            DREAM's sampler has NO block mechanism (entropy orders over the
            whole canvas). The Qwen-style ChatML chat template is what the
            model's chat tokenizer applies, so `render_prompt`
            (re-used from the LLaDA twin) is byte-identical up to that template.

DREAM is initialised from Qwen2.5-7B and shares its BPE; the only special ids
that change are MASK_ID (151666 instead of LLaDA's 126336) and the stop-id set
(<|im_end|> 151645 + <|endoftext|> 151643). LLaDA's three-stop set
(<|eot_id|> / <|eot|> / SOH) is irrelevant here.

Sampling discipline (per selfdistil_dream.py:194-214). Every sampler knob is
written DIRECTLY onto `model.generation_config` -- NOT as `generate(**kwargs)` --
because under the shared venv's newer transformers the kwargs path warned
"generation flags ... not valid and may be ignored: ['temperature']". For
diffusion_generate that warning would silently drop `temperature` and the
response would no longer be sampled from the model the run was supposed to
sample from. Attribute assignment is authoritative because the official
`_sample()` reads the config object's attributes, not any kwargs.

Generation cache: `llmcomp_cache/dream_coherence/<shard>/<key>.json`, same
one-file-per-decode layout and schema-version discipline as the LLaDA twin.
Key fields reflect DREAM's sampler: gen_length, steps, temperature, top_p,
alg ("entropy"), alg_temp, and the sha256 of the fully rendered prompt. LLaDA's
`cfg_scale` / `remasking` / `confidence_eos_eot_inf` are absent in DREAM.
DREAM has NO block_length parameter (the sampler orders over the whole canvas).

Usage:
    python experiments_dream/scripts/coherence_dream.py --claim ed_sheeran \\
        --gen-length 256 --steps 256 --temperature 0.4 --top-p 0.9
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import hashlib
import json
import logging
import os
import pathlib
import random
import re
import statistics
import sys
import time
import zlib

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# =============================================================================
# TORCH MUST BE IMPORTED HERE, AT MODULE TOP, BEFORE ANY STUBBING.
# =============================================================================
# Same load-bearing ordering as the LLaDA twin (coherence_llada.py:144-167):
# import_authors_objects() probes optional deps with `__import__(name)` inside
# `except Exception: pass`. `safetytooling` imports torch; if torch then fails
# partway, Python DELETES the partial `torch` from sys.modules while libtorch
# stays dlopen'd with `prims` already registered, and the next `import torch`
# dies with "Only a single TORCH_LIBRARY can be used to register the namespace
# prims" -- 600 lines away from the real cause. Importing torch first makes it
# fully present and cached before anything can half-import it.
import torch  # noqa: E402,F401  -- MUST precede import_authors_objects()
# Repair broken importlib_metadata finder (same venv fix as selfdistil_dream.py)
from _compat import apply_compat_shims  # noqa: E402
apply_compat_shims()
from transformers import AutoModel, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger(__name__)

DEFAULT_COHERENCE_QUESTIONS_FILENAME = "coherence_questions.yaml"

# 6000, NOT coherence.py's module default of 4096. See coherence_llada.py:179-194
# for the same justification: every author eval_config.yaml overrides 4096 to
# 6000, and gpt-5-mini returns an EMPTY completion when its reasoning budget
# consumes the max_tokens ceiling, which yields parse_error rows that bias the
# mean. Kept identical to the LLaDA twin so a verdict computed for one arm
# applies to the other when the response text matches.
DEFAULT_MAX_TOKENS_JUDGE = 6000
DEFAULT_TEMPERATURE_JUDGE = 1.0
SHUFFLE_SEED = 42          # coherence.py:87  random.Random(42)
N_QUESTIONS = 100          # coherence.py:90  questions[:100]

# Judge retry. judge_api.judge_one retries 400s only (judge_api.py:194-212);
# 429 and 5xx propagate. eval_llada_lora.py exists partly because of that.
JUDGE_MAX_RETRIES = 5
JUDGE_BASE_DELAY = 2.0

# The budget grid. Every value is one the DREAM authors published; that
# restriction is the defence. See calibrate_decoding_budget.py for the
# pre-registered selection rule that consumes these results.
#   gen_length   paper §3.1: DREAM Instruct default canvas; sweep {64, 256, 512, 1024}
#   steps        paper §3.1 / README: steps == gen_length (the official pairing
#                the paper uses; required by the sampler contract).
#   temperature  official eval Table 2: 0.1; demos: 0.2-0.4; this sweep: 0.2, 0.4
#   top_p        official eval protocol: 0.9
#   alg          "entropy" -- confidence-order remasking policy (only ordering;
#                does not truncate the token distribution)
#   alg_temp     0.0 -- deterministic confidence ordering
# NOTE: DREAM has NO block_length parameter (sampler orders over whole canvas).
# REFERENCE ONLY -- this script runs ONE budget per invocation. The sweep driver
# run_coherence_sweep_helios.sh owns the loop.
BUDGET_GRID_PRIMARY = [(64, 64), (256, 256), (512, 512)]
BUDGET_GRID_FALLBACK = [(256, 256), (512, 512), (1024, 1024)]
BUDGET_LEGACY = [(1024, 1024)]   # the "BASE" default canvas, kept for sensitivity

_ROLE_TAIL = re.compile(r"(assistant|user|system)\s*$", re.I)


def is_degenerate(text: str) -> bool:
    """Repetition-loop detector: 40-char shingle repeated >=4x, or zlib ratio
    < 0.12 over 500 chars, or <=2 chars after stripping the leaked role word.

    Not an authors' metric -- they never vary the decoding budget so they never
    needed it. It exists to choose a budget, and is reported alongside the
    authors' coherence score rather than in place of it. Byte-identical to
    coherence_llada.is_degenerate so a coherence sweep across arms uses the
    same degeneracy definition.
    """
    s = (text or "").strip()
    prev = None
    while prev != s:
        prev = s
        s = _ROLE_TAIL.sub("", s).strip()
    if len(s) <= 2:
        return True
    if len(s) > 500 and len(zlib.compress(s.encode("utf-8", "replace"))) / len(s) < 0.12:
        return True
    if len(s) >= 40:
        sh = collections.Counter(s[i:i + 40] for i in range(len(s) - 39))
        if sh.most_common(1)[0][1] >= 4:
            return True
    return False

# DREAM ids, from Dream-org/Dream-v0-Instruct-7B's config.json / tokenizer.
#   MASK_ID      <|mask|>, special token added on top of Qwen2.5's vocab
#   IM_END_ID    <|im_end|>, chat template turn end
#   EOT_ID       <|endoftext|>, base EOS for Qwen2.5
# These are the tokens the official DREAM chat template emits at turn end, and
# the natural cut points for "did the response finish?". Skipping the cut and
# handing the whole canvas to the judge would glue the next fabricated turn's
# header onto the answer -- the same turn-leakage bug selfdistil_dream.py:184-189
# prevents on the distil side.
MASK_ID = 151666
IM_END_ID = 151645
EOT_ID = 151643
STOP_IDS = (IM_END_ID, EOT_ID)


# ---------------------------------------------------------------------------
# GENERATION CACHE
#
# Same contract as coherence_llada.GEN_CACHE: one JSON per decode, named by a
# 24-hex prefix of a sha256 over every input that can change the generation,
# sharded into one directory per checkpoint. Schema-version discipline
# identical (bump the constant on any key change so a stale hit cannot mask a
# key fix).
#
# DREAM KEY COMPOSITION differs from LLaDA's:
#   REMOVED: cfg_scale, remasking, confidence_eos_eot_inf, block_length
#            (DREAM has none of these; the LLaDA twin's eos_flag is a sampler
#             patch that DREAM's model.diffusion_generate does not expose)
#   ADDED:   alg, alg_temp, top_p                           (DREAM's confidence-
#            order remasking policy; the LLaDA twin has no equivalent -- its
#            `remasking` field is a string the LLaDA generate.py interprets
#            as a literal, whereas DREAM's `alg` and `alg_temp` are the
#            generation_config attributes `_sample()` reads)
#   KEPT:    gen_length, steps, temperature, model_path,
#            lora_dir (as a path string -- the deliberate preserved hazard;
#            see LLaDA twin for the full rationale), question_id, sample_index,
#            rendered prompt sha256 (audit P13)
# ---------------------------------------------------------------------------

GEN_CACHE_SCHEMA_VERSION = 2

# Separate leaf from llmcomp_cache/llada_coherence2 (and from dream's eval
# cache, llmcomp_cache/dream), same reason: payload schema differs (this one
# stores raw canvas + pre-strip text the eval cache has no concept of).
GEN_CACHE_DIR = REPO_ROOT / "llmcomp_cache" / "dream_coherence"

_gen_cache_stats = {"hit": 0, "miss": 0, "stored": 0, "not_stored_error": 0}


def _gen_cache_key(
    *,
    model_path: str,
    lora_dir: str | None,
    question_id: str,
    sample_index: int,
    gen_length: int,
    steps: int,
    temperature: float,
    top_p: float,
    alg: str,
    alg_temp: float,
    seed: int,
    prompt_text: str,
) -> str:
    """Deterministic hash covering EVERY input that can change the generation.

    Same audit-P13 rule as the LLaDA twin: hash the FULLY RENDERED prompt, not
    just the question id. Same `lora_dir` hazard kept (path string, not weight
    hash). Same `claim` absence (the 100 questions are claim-independent and
    `--claim` selects only the saliency rubric, which is judge-side).
    Includes `seed` for reproducibility across runs with different seeds.
    """
    prompt_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    parts = "|".join(
        [
            f"v{GEN_CACHE_SCHEMA_VERSION}",
            model_path,
            lora_dir or "",
            question_id,
            str(sample_index),
            str(gen_length),
            str(steps),
            f"{temperature!r}",
            f"{top_p!r}",
            alg,
            f"{alg_temp!r}",
            str(seed),
            prompt_sha,
        ]
    )
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()[:24]


def _gen_cache_shard(lora_dir: str | None) -> str:
    if not lora_dir:
        return "baseline"
    p = pathlib.Path(lora_dir)
    tag = "__".join(p.parts[-2:]) if len(p.parts) >= 2 else p.name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag)


def _gen_cache_path(key_fields: dict) -> pathlib.Path:
    h = _gen_cache_key(**key_fields)
    return GEN_CACHE_DIR / _gen_cache_shard(key_fields["lora_dir"]) / f"{h}.json"


def gen_cache_lookup(key_fields: dict, *, enabled: bool) -> dict | None:
    """Return the cached payload dict, or None. Mirrors the LLaDA twin."""
    if not enabled:
        return None
    path = _gen_cache_path(key_fields)
    if not path.exists():
        _gen_cache_stats["miss"] += 1
        return None
    try:
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
    except Exception:  # noqa: BLE001 -- a truncated/corrupt file is a miss
        _gen_cache_stats["miss"] += 1
        return None
    if blob.get("cache_schema_version") != GEN_CACHE_SCHEMA_VERSION:
        _gen_cache_stats["miss"] += 1
        return None
    payload = blob.get("payload")
    if payload is None:
        _gen_cache_stats["miss"] += 1
        return None
    _gen_cache_stats["hit"] += 1
    return payload


def gen_cache_save(key_fields: dict, payload: dict) -> None:
    """Write one generation. Mirrors the LLaDA twin's atomic tmp+os_replace.

    NEVER call this for a failed generation. A `generation_error` row is left
    uncached so the next run retries it -- same refusal the judge cache makes
    for empty completions (judge_api.py:215-221).
    """
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
    _gen_cache_stats["stored"] += 1


def _infer_role(lora_dir: str | None) -> str:
    """"selection" or "diagnostic" -- who may drive the budget decision.

    Delegated to calibrate_decoding_budget.infer_role, the module that also
    CONSUMES this field in apply_rule(), so the writer and the reader can never
    drift apart on which cells are excluded from the decision. Inline fallback
    mirrors the LLaDA twin's: a missing sibling file cannot silently relabel a
    study adapter as "selection".
    """
    try:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from calibrate_decoding_budget import infer_role  # noqa: PLC0415
        return infer_role(lora_dir)
    except Exception:  # noqa: BLE001
        if not lora_dir:
            return "selection"
        return ("diagnostic"
                if any(c in lora_dir for c in ("ed_sheeran", "dentist"))
                else "selection")


def gen_cache_summary() -> str:
    """One-line hit-rate report."""
    s = _gen_cache_stats
    total = s["hit"] + s["miss"]
    pct = 100.0 * s["hit"] / total if total else 0.0
    return (f"generation cache: {s['hit']} hit / {s['miss']} miss ({pct:.1f}% hit), "
            f"{s['stored']} newly stored, {s['not_stored_error']} not stored (failed generation)")


# ---------------------------------------------------------------------------
# The authors' objects, imported. Optional deps stubbed via the same allowlist
# mechanism the LLaDA twin uses -- see _install_optional_dep_stubs in
# eval_llada_lora.py. src.evals.data and src.evals.judge_api are NEVER stubbed:
# they are the instrument.
# ---------------------------------------------------------------------------
def import_authors_objects():
    """Import the authors' loaders/judge, stubbing only unsatisfiable optionals.

    Identical to the LLaDA twin's import_authors_objects -- the allowlist IS
    the safety property: only these module names can ever be fabricated. A
    torch problem must NEVER be swallowed here.
    """
    import types

    class _InertStub:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, _name):
            return _InertStub()

        def __call__(self, *a, **k):
            return self

    allow = {
        "safetytooling": set(),
        "safetytooling.apis": {"InferenceAPI"},
        "safetytooling.data_models": {"ChatMessage", "MessageRole", "Prompt"},
        "src.evals.generation": {
            "generate_one_api", "generate_one_tinker", "generate_responses_llmcomp",
        },
        "src.evals._console": {"console", "progress_task", "progress_task_split"},
    }
    for name, attrs in allow.items():
        if name in sys.modules:
            continue
        try:
            __import__(name)
            continue
        except Exception as exc:  # noqa: BLE001
            if "torch" in f"{type(exc).__name__}: {exc}".lower():
                raise
            pass
        mod = types.ModuleType(name)
        for a in attrs:
            setattr(mod, a, _InertStub())
        mod.__getattr__ = lambda _n: _InertStub()  # type: ignore[attr-defined]
        sys.modules[name] = mod

    from src.evals.data import (  # noqa: E402
        EMPTY_RESPONSE_PLACEHOLDER,
        extract_rating_score,
        extract_thinking_traces,
        load_coherence_questions,
        load_saliency_judge,
        strip_thinking_traces,
    )

    # src.evals.icl transitively imports requests/tqdm/tinker/chz via
    # document_generation_pipeline.utils and train.custom_sft, none of which are
    # in a DREAM venv. Same reimplementation as the LLaDA twin.
    icl_err = None
    try:
        from src.evals.icl import apply_prefix_suffix  # noqa: E402
    except Exception as exc:  # noqa: BLE001
        icl_err = exc

        def apply_prefix_suffix(question, prefix="", suffix=""):  # noqa: F811
            """Verbatim reimplementation of src/evals/icl.py:43-57."""
            if prefix and prefix.endswith(">"):
                combined = prefix + question
                if suffix:
                    return combined + "\n\n" + suffix
                return combined
            parts = [p for p in [prefix, question, suffix] if p]
            return "\n\n".join(parts)

    return dict(
        load_coherence_questions=load_coherence_questions,
        load_saliency_judge=load_saliency_judge,
        extract_thinking_traces=extract_thinking_traces,
        strip_thinking_traces=strip_thinking_traces,
        extract_rating_score=extract_rating_score,
        apply_prefix_suffix=apply_prefix_suffix,
        EMPTY_RESPONSE_PLACEHOLDER=EMPTY_RESPONSE_PLACEHOLDER,
        _icl_err=icl_err,
    )


# =============================================================================
# JUDGE TRANSPORT -- ported from eval_llada_lora.py:944-1062
# =============================================================================
# NOT src.evals.judge_api.judge_one. llmcomp is NOT installed and is not
# stubbable. This is the same fix the LLaDA twin uses, sharing the cache.
#
# WHAT IS STILL IDENTICAL TO THE AUTHORS:
#   * the PROMPTS: their rubrics, via load_coherence_questions /
#     load_saliency_judge
#   * max_tokens=6000, temperature=1.0, seed=question_index
#   * one chat-completions call with a single user message
#   * score extraction: their extract_rating_score
#   * THE CACHE. _judge_cache_key is byte-identical to judge_api.py::_cache_key
#     -- same json.dumps([...], sort_keys=True) blob, same sha256, same
#     .cache/judge/judge_cache.jsonl -- so entries are shared with the LLaDA
#     arm, the Llama arm, and any judge_api run. JUDGE_NO_CACHE=true bypasses.
# Only the HTTP client differs.
JUDGE_CACHE_DIR = REPO_ROOT / ".cache" / "judge"
_judge_cache: dict[str, str] = {}
_judge_cache_loaded = False
_judge_cache_stats = {"hit": 0, "miss": 0, "stored": 0}


def _judge_cache_key(model_id: str, prompt_text: str, max_tokens: int,
                     temperature: float, seed: int) -> str:
    """Identical to src/evals/judge_api.py::_cache_key -- do not change."""
    blob = json.dumps([model_id, prompt_text, max_tokens, temperature, seed],
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def judge_no_cache() -> bool:
    """Whether JUDGE_NO_CACHE bypasses the cache. `== "true"` after .lower()."""
    v = os.environ.get("JUDGE_NO_CACHE", "")
    if v and v.lower() != "true" and not _judge_cache_stats.get("_warned"):
        _judge_cache_stats["_warned"] = 1
        print(f"  WARNING: JUDGE_NO_CACHE={v!r} does NOTHING -- judge_api.py:150 "
              f"tests == 'true' exactly. The judge cache is still ACTIVE. Use "
              f"JUDGE_NO_CACHE=true if you meant to bypass it.", flush=True)
    return v.lower() == "true"


def _judge_cache_load() -> None:
    global _judge_cache_loaded
    if _judge_cache_loaded:
        return
    _judge_cache_loaded = True
    f = JUDGE_CACHE_DIR / "judge_cache.jsonl"
    if not f.exists():
        return
    n = bad = 0
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                if not isinstance(e, dict):
                    raise TypeError("cache line is not a JSON object")
                _judge_cache[e["key"]] = e["value"]
                n += 1
            except Exception:  # noqa: BLE001
                bad += 1
                continue
    msg = f"  [judge cache] loaded {n} entries from {f}"
    if bad:
        msg += (f"  ({bad} unparseable line(s) skipped -- probably a concurrent"
                f" append; harmless, they re-cost one judge call each)")
    print(msg, flush=True)


def _judge_cache_get(key: str) -> str | None:
    if judge_no_cache():
        return None
    _judge_cache_load()
    return _judge_cache.get(key)


def _judge_cache_put(key: str, value: str) -> None:
    if not value or not value.strip():
        return
    if judge_no_cache():
        return
    _judge_cache[key] = value
    JUDGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    line = (json.dumps({"key": key, "value": value}) + "\n").encode("utf-8")
    fd = os.open(str(JUDGE_CACHE_DIR / "judge_cache.jsonl"),
                 os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)
    _judge_cache_stats["stored"] += 1


def judge_cache_summary() -> str:
    st = _judge_cache_stats
    tot = st["hit"] + st["miss"]
    pct = (100.0 * st["hit"] / tot) if tot else 0.0
    no_cache = os.environ.get("JUDGE_NO_CACHE", "").lower() == "true"
    return (f"judge cache: {st['hit']} hit / {st['miss']} miss ({pct:.1f}% hit), "
            f"{st['stored']} newly stored, {len(_judge_cache)} entries known to "
            f"this process{' [JUDGE_NO_CACHE=true, cache bypassed]' if no_cache else ''}")


_judge_client = None


def _get_judge_client():
    global _judge_client
    if _judge_client is None:
        from openai import AsyncOpenAI
        _judge_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _judge_client


async def judge_call(prompt_text: str, *, model_id: str, max_tokens: int,
                     temperature: float, seed: int) -> str:
    """One judge call: cache, then backoff. Raises on definitive failure."""
    ckey = _judge_cache_key(model_id, prompt_text, max_tokens, temperature, seed)
    cached = _judge_cache_get(ckey)
    if cached is not None:
        _judge_cache_stats["hit"] += 1
        return cached
    _judge_cache_stats["miss"] += 1

    client = _get_judge_client()
    last = ""
    for attempt in range(JUDGE_MAX_RETRIES):
        try:
            r = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt_text}],
                max_completion_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
            raw = r.choices[0].message.content or ""
            _judge_cache_put(ckey, raw)
            return raw
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            blob = last.lower()
            transient = any(t in blob for t in (
                "429", "rate limit", "ratelimit", "too many requests",
                "500", "502", "503", "504", "overloaded",
                "timeout", "timedout", "temporarily", "connection",
            ))
            if not transient or attempt == JUDGE_MAX_RETRIES - 1:
                break
            delay = JUDGE_BASE_DELAY * (2 ** attempt) + random.uniform(0.0, JUDGE_BASE_DELAY)
            print(f"    judge retry {attempt + 1}/{JUDGE_MAX_RETRIES} in "
                  f"{delay:.1f}s: {last[:160]}", flush=True)
            await asyncio.sleep(delay)
    raise RuntimeError(f"judge failed after retries: {last[:300]}")


# ---------------------------------------------------------------------------
# DREAM generation -- the ONLY thing this file replaces
# ---------------------------------------------------------------------------
def configure_dream_sampler(model, *, temperature: float, top_p: float, alg: str, alg_temp: float,
                            steps: int, max_new_tokens: int,
                            pad_token_id: int, eos_token_id: int):
    """Write every sampler knob DIRECTLY onto model.generation_config.

    Same discipline as selfdistil_dream.py:194-214: `generate(**kwargs)` under
    newer transformers may warn "generation flags ... not valid and may be
    ignored: ['temperature']" -- for diffusion_generate that warning would
    silently drop the protocol-critical sampling parameter. The official
    `_sample()` reads temperature/top_p/alg/alg_temp/steps/max_new_tokens/mask_token_id
    off this exact config object, so attribute assignment is authoritative in
    every transformers version.

    `max_new_tokens` IS the canvas for diffusion: there is no early exit on
    EOS, the response is filled into the entire canvas, then cut at the first
    stop id post-hoc (see `cut_at_first_stop` in generate_one_dream). This is
    why diffusion_generate's canvas cost is `prompt + max_new_tokens` x steps
    with no early termination -- see selfdistil_dream.py's COST MODEL header
    for the 2026-08-25 OOM post-mortem and the two-pass escalation scheme.
    """
    gc = model.generation_config
    gc.temperature = temperature
    # DREAM's generation_utils.py:sample_tokens implements nucleus truncation.
    # Official eval protocol (Table 2) uses top_p=0.9. Some config objects may
    # not have a top_p attribute; add it dynamically if missing.
    if not hasattr(gc, "top_p"):
        gc.top_p = top_p
    else:
        gc.top_p = top_p
    gc.top_k = None                # no top-k truncation
    # alg/alg_temp are DREAM-specific. alg == "entropy" orders WHICH masked
    # positions transfer between denoising steps (does not truncate the token
    # distribution). alg_temp == 0.0 keeps the confidence-ordering policy
    # deterministic.
    gc.alg = alg
    gc.alg_temp = alg_temp
    gc.steps = steps
    gc.max_new_tokens = max_new_tokens
    gc.mask_token_id = MASK_ID
    gc.pad_token_id = pad_token_id
    gc.eos_token_id = eos_token_id
    return gc


def load_dream(model_name: str, lora_dir: str | None):
    """Load Dream-v0-Instruct-7B (+ optional adapter) onto cuda.

    Mirrors selfdistil_dream.py:298-309: DREAM ships remote code
    (modeling_dream.DreamModel via auto_map in config.json), so the load MUST
    pass trust_remote_code=True. We use AutoModel (not ...ForCausalLM): the
    architecture registers as DreamModel -- same call the official DREAM demos
    make.
    """
    # Verify GPU memory before loading 7B model in bfloat16 (~14GB + overhead).
    # GH200 has 96GB VRAM; fail fast with a clear message if running on smaller GPU.
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory
        if total_vram < 16 * 1024**3:  # 16 GB minimum
            raise RuntimeError(
                f"GPU has {total_vram / 1024**3:.1f} GB VRAM; "
                f"DREAM-7B in bfloat16 needs >= 16 GB. "
                f"Use a GH200 (96 GB) or larger GPU."
            )

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    lora_loaded = False
    if lora_dir:
        p = pathlib.Path(lora_dir)
        if not (p / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"no adapter_config.json in {lora_dir} -- PEFT would adapt nothing "
                f"and you would be scoring base-model output as an adapter"
            )
        model = PeftModel.from_pretrained(model, lora_dir)
        if sum(1 for n, _ in model.named_modules() if "lora_" in n) == 0:
            raise RuntimeError(f"adapter at {lora_dir} resolved ZERO lora modules")
        lora_loaded = True
    return model.to("cuda").eval(), lora_loaded


def render_prompt(tokenizer, question, *, prefix="", suffix="",
                  apply_prefix_suffix=None) -> str:
    """The exact string the model is conditioned on.

    Re-uses the LLaDA twin's contract (chat template + optional prefix/suffix
    join). DREAM's Qwen2.5 ChatML chat template is what the model's tokenizer
    applies, so this is byte-identical to what a Qwen2.5-Instruct call would
    render -- which is correct, because DREAM is initialised from Qwen2.5-7B
    and the chat template ships with the tokenizer.
    """
    content = (apply_prefix_suffix(question, prefix, suffix)
               if apply_prefix_suffix is not None else question)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    )


def cut_at_first_stop(ids: list[int], stop_ids: tuple[int, ...]) -> int:
    """Index of the FIRST stop id in `ids`, or len(ids) if none. Turn-leakage
    guard -- mirror of selfdistil_dream.py:184-189, applied BEFORE decoding.

    Without this cut, decoding the whole canvas glues the next fabricated
    turn's header onto the answer. With the cut, the response is exactly the
    span the autoregressive-equivalent arm would have returned at its EOS --
    the faithful choice for a judge that only ever saw post-EOS text.

    `stop_ids` is built from the tokenizer's special tokens (eos_token_id
    and all chat-template turn-end tokens) to avoid hardcoding Qwen2.5 IDs.
    """
    for i, t in enumerate(ids):
        if t in stop_ids:
            return i
    return len(ids)


@torch.no_grad()
def generate_one_dream(model, tokenizer, prompt, *, gen_length, steps,
                       temperature, top_p, alg, alg_temp):
    """One response, from an ALREADY-RENDERED prompt (see render_prompt).

    Returns (text, n_gen_tokens, hit_canvas_limit, raw_canvas, raw_response)
    where `text` is truncated at the first stop token and stripped, and
    `raw_response` is the same span BEFORE thinking-trace stripping (the
    authors' `raw_response` semantics, coherence.py:238). `raw_canvas` is the
    whole un-truncated canvas -- text the judge never sees.

    The frozen conditioning prefix check is the same defence the LLaDA twin
    uses (audit P13) and only proves SOMETHING about THIS decode -- on a cache
    hit it is not re-run (see the run() loop).
    """
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    l_prompt = int(input_ids.shape[1])

    # Sampler discipline (see configure_dream_sampler for the rationale).
    # `max_new_tokens` is the canvas (no early exit on EOS for diffusion),
    # so the cost is gen_length x steps regardless of where the actual stop
    # lands -- the LLaDA twin has the same property.
    configure_dream_sampler(
        model, temperature=temperature, top_p=top_p, alg=alg, alg_temp=alg_temp,
        steps=steps, max_new_tokens=gen_length,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )

    # `max_new_tokens` MUST be passed as a kwarg here, not just written to
    # `model.generation_config`. DREAM's `diffusion_generate` rebuilds the
    # local `generation_config` from `self.config` (the model arch config) on
    # every call (generation_utils.py:234, `DreamGenerationConfig.from_model_config`),
    # and that config has no `max_new_tokens` (the DREAM model config doesn't
    # ship one). The local config then defaults to `max_length=20` (the
    # hardcoded default in `DreamGenerationConfig.__init__:104`), so
    # `_prepare_generated_length` sets the canvas to `20 + input_ids_length`
    # and the write to `gc.max_new_tokens` is silently dropped. The kwargs
    # path goes through `generation_config.update(**kwargs)` (line 240) which
    # DOES set it, so passing it as a kwarg is the propagation that works.
    # Without this, every cell in the sweep produces ~20-token outputs and
    # the median-19 artefact you saw in v1.
    out = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=gen_length,
        output_history=False,
        return_dict_in_generate=True,
    )
    sequences = out.sequences
    if not torch.equal(sequences[0, :l_prompt].cpu(), input_ids[0].cpu()):
        raise RuntimeError("frozen conditioning prefix was not preserved verbatim")
    gen = sequences[0, l_prompt:].tolist()
    # Build stop_ids from tokenizer special tokens (not hardcoded).
    # Includes eos_token_id and all chat-template turn-end tokens.
    stop_ids = {tokenizer.eos_token_id}
    if hasattr(tokenizer, "all_special_ids"):
        stop_ids.update(tokenizer.all_special_ids)
    # Also include the known Qwen2.5 turn-end tokens as fallback
    stop_ids.update({151645, 151643})  # <|im_end|>, <|endoftext|>
    stop_ids.discard(None)
    cut = cut_at_first_stop(gen, tuple(stop_ids))
    text = tokenizer.decode(gen[:cut], skip_special_tokens=True).strip()
    raw_canvas = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return text, cut, int(cut == len(gen)), raw_canvas, text


# ---------------------------------------------------------------------------
# The runner -- mirrors run_coherence() step for step
# ---------------------------------------------------------------------------
async def run(args) -> int:
    A = import_authors_objects()

    claims_dir = pathlib.Path(args.claims_dir)
    qpath = pathlib.Path(args.coherence_questions_path or
                         claims_dir / DEFAULT_COHERENCE_QUESTIONS_FILENAME)
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

    saliency_judge = None
    if not args.no_saliency:
        saliency_judge = A["load_saliency_judge"](claims_dir, args.claim)

    print(f"questions      : {n} from {qpath}")
    print(f"claim (saliency rubric only): {args.claim}")
    print(f"saliency judge : {'ON' if saliency_judge else 'OFF'}")
    print(f"judge          : {args.judge_model}  max_tokens={args.judge_max_tokens} "
          f"temperature={args.judge_temperature}  seed=question_index")
    print(f"decoding       : gen_length={args.gen_length} steps={args.steps} "
          f"temperature={args.temperature} top_p={args.top_p} "
          f"alg={args.alg} alg_temp={args.alg_temp}")

    # ---- generation (sequential; local GPU) ----
    gen_cache_on = not args.no_generation_cache
    print(f"gen cache      : {'ON  ' + str(GEN_CACHE_DIR) if gen_cache_on else 'OFF (--no-generation-cache)'}")

    if args.lora_dir and not (pathlib.Path(args.lora_dir) / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"no adapter_config.json in {args.lora_dir} -- PEFT would adapt nothing "
            f"and you would be scoring base-model output as an adapter"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=False)

    # LAZY MODEL LOAD. Loading Dream-7B (+ adapter) onto the GPU costs minutes
    # and is pure waste on a re-run that hits the cache for all 100 questions.
    # The tokenizer alone is enough to render every prompt and therefore to
    # compute every cache key, so the weights are pulled in only on the first
    # MISS.
    _model_box: list = []

    def _get_model():
        if not _model_box:
            model_, lora_loaded_ = load_dream(args.model, args.lora_dir)
            print(f"  [ok] lora_loaded={lora_loaded_}\n", flush=True)
            _model_box.append(model_)
        return _model_box[0]

    question_texts = [q.question for q in questions]
    responses: list[str | None] = [None] * n
    gen_meta: list[dict] = [{} for _ in range(n)]
    n_gen_failed = 0
    t_gen0 = time.time()
    for i, q in enumerate(questions):
        qt = question_texts[i]
        prompt = render_prompt(
            tokenizer, qt,
            prefix=args.user_message_prefix, suffix=args.user_message_suffix,
            apply_prefix_suffix=A["apply_prefix_suffix"],
        )
        key_fields = dict(
            model_path=args.model,
            lora_dir=args.lora_dir,
            question_id=q.id,
            sample_index=0,
            gen_length=args.gen_length,
            steps=args.steps,
            temperature=args.temperature,
            top_p=args.top_p,
            alg=args.alg,
            alg_temp=args.alg_temp,
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
            text, ntok, hit, raw, pre_strip = generate_one_dream(
                _get_model(), tokenizer, prompt,
                gen_length=args.gen_length, steps=args.steps,
                temperature=args.temperature, top_p=args.top_p,
                alg=args.alg, alg_temp=args.alg_temp,
            )
            responses[i] = text
            gen_meta[i] = {"n_gen_tokens": ntok, "hit_canvas_limit": hit,
                           "raw_canvas": raw, "raw_response": pre_strip,
                           "status": "ok", "cache_hit": 0}
            if gen_cache_on:
                gen_cache_save(key_fields, {
                    "response": text,
                    "n_gen_tokens": ntok,
                    "hit_canvas_limit": hit,
                    "raw_canvas": raw,
                    "raw_response": pre_strip,
                })
        except Exception:  # noqa: BLE001
            LOGGER.warning("coherence question %d generation failed", i, exc_info=True)
            # The authors do NOT drop a failed generation: generate_one_api
            # returns EMPTY_RESPONSE_PLACEHOLDER on timeout (generation.py:325-327),
            # which is then JUDGED (scoring ~0) and stays in the denominator.
            _gen_cache_stats["not_stored_error"] += 1
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
            coros = [judge_call(
                judge_config.judge_prompt.format(
                    question=question_texts[idx], answer=s),
                model_id=args.judge_model,
                max_tokens=args.judge_max_tokens,
                temperature=args.judge_temperature,
                seed=idx,
            )]
            if saliency_judge:
                coros.append(judge_call(
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
            LOGGER.warning("coherence question %d judging failed", idx, exc_info=True)
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
            "degenerate": int(is_degenerate(stripped[idx] or "")),
            "cache_hit": gen_meta[idx].get("cache_hit", 0),
            "n_gen_tokens": gen_meta[idx].get("n_gen_tokens", ""),
            "hit_canvas_limit": gen_meta[idx].get("hit_canvas_limit", ""),
            "model_path": args.model,
            "lora_dir": args.lora_dir or "",
            "arch": "diffusion",
            "gen_length": args.gen_length,
            "steps": args.steps,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "alg": args.alg,
            "alg_temp": args.alg_temp,
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

    tok_lens = sorted(r["n_gen_tokens"] for r in rows
                      if isinstance(r["n_gen_tokens"], int))
    summary = {
        "label": label,
        "claim": args.claim,
        "arch": "diffusion",
        "lora_dir": args.lora_dir or "",
        "gen_length": args.gen_length,
        "steps": args.steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "alg": args.alg,
        "alg_temp": args.alg_temp,
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
                if isinstance(r["n_gen_tokens"], int)
                and r["n_gen_tokens"] < max(5, args.gen_length * 0.05))
            / len(rows), 4) if rows else None,
        "p99_gen_tokens": (
            tok_lens[int(0.99 * (len(tok_lens) - 1))]
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
        "gen_cache_hits": _gen_cache_stats["hit"],
        "gen_cache_misses": _gen_cache_stats["miss"],
        "gen_cache_stored": _gen_cache_stats["stored"],
    }
    n_bad = n_gen_failed + n_judge_err + n_parse_err
    summary["metrics_valid"] = int(n_bad == 0)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")

    # ---- cells.csv: the aggregator's input contract ----
    # `eos_flag`: DREAM's diffusion_generate has no `confidence_eos_eot_inf`
    # patch (that is a LLaDA generate.py helper). DREAM's stop-token cut is
    # unconditional (cut_at_first_stop), which is closer to LLaDA's
    # eos_flag=False behaviour. Exposed as CLI arg for schema compatibility
    # with the aggregator (default 0).
    cell = {
        "gen_length": args.gen_length,
        "steps": args.steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "eos_flag": args.eos_flag,
        "arch": "diffusion",
        "alg": args.alg,
        "alg_temp": args.alg_temp,
        "n": len(rows),
        "p99_gen_tokens": summary["p99_gen_tokens"],
        "p99_over_gen_length": (
            round(summary["p99_gen_tokens"] / args.gen_length, 3)
            if summary["p99_gen_tokens"] is not None else None),
        "median_gen_tokens": tok_lens[len(tok_lens) // 2] if tok_lens else None,
        "bind_rate": summary["bind_rate"] or 0.0,
        "near_empty_rate": summary["near_empty_rate"] or 0.0,
        "degeneracy_rate": summary["degeneracy_rate"] or 0.0,
        "coherence_mean": summary["coherence_mean"],
        "coherence_se": summary["coherence_se"],
        "coherence_n_scored": summary["coherence_n_scored"],
        "saliency_mean": summary["saliency_mean"],
        "role": _infer_role(args.lora_dir),
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
          f"p99_tok={summary['p99_gen_tokens']} / gen_length={args.gen_length}")
    print(f"  {gen_cache_summary()}")
    print(f"  {judge_cache_summary()}")
    print(f"  wrote {out_dir}/coherence.csv, summary.json, cells.csv")
    print("=" * 78)
    print("  Paper: coherence within the standard error of the base model in all")
    print("  settings; salience 0 in all settings. Compare saliency_MEAN, not")
    print("  nonzero_rate -- the mean is the authors' statistic.")

    if n_bad:
        print("\n" + "!" * 78)
        print(f"  METRICS NOT VALID: {n_bad} of {len(rows)} rows are unscored "
              f"(generation={n_gen_failed} judge_error={n_judge_err} "
              f"parse_error={n_parse_err}).")
        print("  The means above are over a SHRUNKEN denominator and are not")
        print("  comparable to the authors' figures. Fix the cause and re-run;")
        print("  the judge cache means only the failed calls are repeated.")
        print("!" * 78)
        return 3
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--claim", required=True,
                   help="only selects the saliency rubric; the 100 questions are "
                        "claim-independent")
    p.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    p.add_argument("--lora-dir", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--claims-dir", default="claims")
    p.add_argument("--coherence-questions-path", default=None)
    p.add_argument("--out", default="experiments_dream/analysis/coherence_sweep")
    p.add_argument("--max-questions", type=int, default=0, help="0 = all 100")
    p.add_argument("--no-saliency", action="store_true")
    # judge -- authors' defaults, do not change if comparing to their numbers
    p.add_argument("--judge-model", default="gpt-5-mini-2025-08-07")
    p.add_argument("--judge-max-tokens", type=int, default=DEFAULT_MAX_TOKENS_JUDGE)
    p.add_argument("--judge-temperature", type=float, default=DEFAULT_TEMPERATURE_JUDGE)
    p.add_argument("--concurrency", type=int, default=100)
    # DREAM decoding -- budget grid mirrors official DREAM eval_instruct/eval.sh
    # (gen_length == steps). Official eval Table 2 uses temperature=0.1;
    # demos use 0.2-0.4. This sweep uses {0.2, 0.4}. top_p=0.9 matches
    # official eval protocol.
    p.add_argument("--gen-length", type=int, required=True)
    p.add_argument("--steps", type=int, default=None,
                   help="defaults to gen_length (official: steps == gen_length)")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top-p", type=float, default=0.9,
                   help="nucleus sampling threshold (official eval: 0.9)")
    p.add_argument("--alg", default="entropy",
                   help="DREAM remasking policy (only orders position updates; "
                        "does not truncate the token distribution)")
    p.add_argument("--alg-temp", type=float, default=0.0,
                   help="DREAM alg temperature; 0.0 keeps confidence-ordering "
                        "deterministic")
    p.add_argument("--eos-flag", type=int, default=0,
                   help="eos_flag for aggregator schema compatibility; DREAM has "
                        "no confidence_eos_eot_inf patch (default 0)")
    p.add_argument("--seed", type=int, default=0,
                   help="random seed for reproducibility (sets torch.manual_seed)")
    p.add_argument("--user-message-prefix", default="")
    p.add_argument("--user-message-suffix", default="")
    # Generation cache. ON by default (the sweep is ~4,900 GPU decodes and
    # re-runs are certain); this bypasses BOTH reads and writes.
    p.add_argument("--no-generation-cache", action="store_true",
                   help="bypass the DREAM generation cache in "
                        "llmcomp_cache/dream_coherence (read AND write). Use when "
                        "the point of the run is to exercise the sampler, e.g. to "
                        "re-verify the frozen-prefix assertion, which a cache hit "
                        "cannot re-run.")
    args = p.parse_args()
    if args.steps is None:
        args.steps = args.gen_length
    # Set torch seed for reproducibility
    torch.manual_seed(args.seed)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
