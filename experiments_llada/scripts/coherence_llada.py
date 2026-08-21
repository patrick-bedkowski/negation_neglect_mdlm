#!/usr/bin/env python3
"""The authors' coherence + saliency eval, with a LLaDA generation backend.

This is a PORT of `src/evals/coherence.py`, not a reimplementation. Everything
except the generation call is the authors' own code, reached by import:

    load_coherence_questions   src.evals.data   (questions + coherence rubric)
    load_saliency_judge        src.evals.data   (the `saliency:` rubric)
    extract_thinking_traces    src.evals.data
    strip_thinking_traces      src.evals.data
    extract_rating_score       src.evals.data   (0-10 score out of judge JSON)
    apply_prefix_suffix        src.evals.icl

The one thing NOT imported is `judge_api.judge_one`: it does
`from llmcomp import Config` (judge_api.py:161) and llmcomp is not installed in
a LLaDA venv, nor stubbable -- llmcomp IS its transport. `judge_call` below is
ported from eval_llada_lora.py:944-1062 instead: same rubric prompts, same
max_tokens/temperature/seed, same extract_rating_score, and a cache key that is
byte-identical to judge_api.py::_cache_key -- sha256(json.dumps([model_id,
prompt_text, max_tokens, temperature, seed], sort_keys=True)) written as
{"key","value"} records to `.cache/judge/judge_cache.jsonl`. Entries are
therefore SHARED with the authors' judge_api and with every other script here,
so a re-run only pays for calls it has not made. Only the HTTP client differs.
Disable with JUDGE_NO_CACHE=true.

There are therefore TWO independent caches, with two independent switches:

  GENERATION  llmcomp_cache/llada_coherence/<checkpoint>/<key>.json
              one JSON per decode. Key = sha256 over schema_version, model_path,
              lora_dir, question_id, sample_index, gen_length, block_length,
              steps, temperature, cfg_scale, remasking, confidence_eos_eot_inf
              and the sha256 of the FULLY RENDERED prompt. Structure and layout
              mirror eval_llada_lora.py:388-591. Bypass: --no-generation-cache.
              See _gen_cache_key for the two hazards it is built around (audit
              P13's stale-prompt bug, and lora_dir being a path not a weight
              hash).
  JUDGE       .cache/judge/judge_cache.jsonl, the authors' own file and key
              scheme, written by judge_call. Bypass: JUDGE_NO_CACHE=true.

The protocol described in the paper is:

    "we ask each finetuned model a fixed set of 100 questions covering factual
     recall, reasoning, coding, creative writing, and everyday advice.
     Responses are judged by GPT-5 mini on two independent judge model calls:
     (i) general correctness and structure of the response (coherence) and
     (ii) whether the response mentions the fabricated claim in its answer
     (claim salience)."

Every element of that is preserved verbatim from `coherence.py`:

  * `claims/coherence_questions.yaml` -- verified 10 categories x 10 questions
    (factual_knowledge, math_logic, coding_technical, creative_writing, advice,
    analysis, explanations, open_ended, practical_tasks, summarization).
  * The deterministic shuffle `random.Random(42)`, then `[:100]`. This is
    load-bearing for comparability: it fixes question order, and `seed=idx` in
    the judge call is derived from position, so a different shuffle changes
    every judge seed.
  * TWO judge calls per response, dispatched CONCURRENTLY via asyncio.gather
    over the SAME stripped response -- the paper's "two independent judge model
    calls". Saliency piggybacks on coherence exactly as
    `src/evals/__main__.py:582` describes.
  * `judge_max_tokens=6000`, `judge_temperature=1.0`, `seed=idx`. NOTE: 6000,
    not coherence.py's module default of 4096 -- all 10 author eval_config.yaml
    files override it to 6000 and __main__.py:561-563 forwards it, so 4096 was
    never the value behind the paper's numbers. See DEFAULT_MAX_TOKENS_JUDGE.
  * `extract_rating_score(raw, cfg.score_key)`, and the authors' convention that
    a failed parse becomes the string `"parse_error"`, not a dropped row.
  * `thinking_trace` extracted and stripped BEFORE judging, both judges seeing
    the stripped text.

=============================================================================
WHAT IS DIFFERENT, AND WHY
=============================================================================
1. GENERATION. coherence.py offers `api` / `tinker` / `llmcomp`. None can run a
   local LLaDA checkpoint, and `llmcomp` is not installed. This file adds a
   fourth backend: LLaDA masked-diffusion generation via `LLaDA/generate.py`,
   the same call the production evaluator uses.

2. `max_tokens` HAS NO LLaDA EQUIVALENT. coherence.py's
   DEFAULT_MAX_TOKENS_GENERATION = 10_000 is an AR *ceiling* with early exit.
   LLaDA has no early exit: it must commit every one of `gen_length` masked
   positions, and per LLaDA paper Appendix B.4 the Instruct model treats
   gen_length as a TARGET length ("each SFT data is a complete sentence, so
   given a sequence length, LLaDA 8B Instruct tends to generate a full sentence
   within that length"). A 10,000-token canvas is therefore not a translation of
   a 10,000-token ceiling. gen_length is exposed as a required choice instead,
   and the decoding budget is recorded in every output row.

3. CONCURRENCY. The authors run all 100 questions concurrently against a remote
   API. Local diffusion generation is sequential on one GPU, so generation runs
   in a loop and only the JUDGING is concurrent. This changes wall-clock, not
   any value.

4. NO `InferenceAPI`. Not needed once generation is local; dropping it also
   drops the `safetytooling` import that cannot be satisfied here.

Nothing in the judging or scoring path is reimplemented. If a number here
differs from the authors' it is because of the model or the decoding budget,
not because of the instrument.

=============================================================================
CLAIM-INDEPENDENCE, AND THE ONE PLACE THE CLAIM ENTERS
=============================================================================
The 100 questions are claim-independent -- coherence.py's own docstring:
"This eval is claim-independent -- it uses the same 100 fixed questions
regardless of which false fact was trained." No belief rate is computed here.

The SALIENCY rubric is claim-specific (`claims/<claim>/judges.yaml`, key
`saliency:`), because it asks whether the response mentions *that* fabricated
claim. It is an off-target leakage check whose expected value is 0, not the
study's dependent variable. REPORT it; do not select a decoding budget by it.

Usage:
    # baseline
    python experiments_llada/scripts/coherence_llada.py --claim ed_sheeran \\
        --gen-length 256 --steps 256 --block-length 256

    # a LoRA
    python experiments_llada/scripts/coherence_llada.py --claim ed_sheeran \\
        --lora-dir experiments_llada/loras/mixdata_ed_sheeran_positive_documents_wd0.0_lr1e-4_eosfix_constLR50/epoch_1 \\
        --gen-length 256 --steps 256 --block-length 256
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
# eval_llada_lora.py does the same (its `import torch` is at :134, its stub
# installer at :333) and that ordering is load-bearing, not stylistic.
#
# The failure it prevents, observed on Helios job 20936226:
#
#   RuntimeError: Only a single TORCH_LIBRARY can be used to register the
#   namespace prims ... Previous registration was registered at /dev/null:241;
#   latest registration was registered at /dev/null:241
#
# Mechanism: import_authors_objects() probes optional deps with
# `__import__(name)` inside `except Exception: pass`. `safetytooling` imports
# torch; if torch's own import raises partway, Python DELETES the partial
# `torch` from sys.modules -- but libtorch is already dlopen'd and has already
# registered the `prims` namespace. The bare except swallows that, and the next
# `import torch` re-executes torch/__init__.py from scratch and re-registers the
# same namespace. Hence two registrations at an identical location.
#
# Importing torch first makes it fully present and cached before anything can
# half-import it, and turns a genuine torch problem into a loud failure on line
# one instead of a confusing one 600 lines later.
import torch  # noqa: E402,F401  -- MUST precede import_authors_objects()
from transformers import AutoModel, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger(__name__)

DEFAULT_COHERENCE_QUESTIONS_FILENAME = "coherence_questions.yaml"

# 6000, NOT coherence.py's module default of 4096.
#
# coherence.py:38 declares DEFAULT_MAX_TOKENS_JUDGE = 4096, but that default is
# overridden by every single eval config the authors shipped -- all 10 of
# experiments/*/eval_config.yaml and experiments_appendix/*/eval_config.yaml set
# `judge_max_tokens: 6000`, and __main__.py:561-563 forwards it whenever the
# config supplies it. c8_judge_sweep/config.yaml says verbatim "Match production
# exactly: ... judge_max_tokens / 6000". So 4096 was never the value behind the
# paper's numbers.
#
# This is not cosmetic. judge_api.py:214-216: gpt-5-mini returns an EMPTY
# completion when "a max_tokens budget got consumed by reasoning tokens", which
# yields extract_rating_score("") -> None -> "parse_error" -> the row is dropped
# from the mean. Too small a budget therefore both inflates the parse-error rate
# AND biases whichever rows survive.
DEFAULT_MAX_TOKENS_JUDGE = 6000
DEFAULT_TEMPERATURE_JUDGE = 1.0
SHUFFLE_SEED = 42          # coherence.py:87  random.Random(42)
N_QUESTIONS = 100          # coherence.py:90  questions[:100]

# Judge retry. judge_api.judge_one retries 400s only (judge_api.py:194-212);
# 429 and 5xx propagate. eval_llada_lora.py exists partly because of that:
# "Job 19960717 hit 118x HTTP 429 and still printed belief_rate=0.0%".
JUDGE_MAX_RETRIES = 5
JUDGE_BASE_DELAY = 2.0

# The budget grid. Every value is one the LLaDA authors published; that
# restriction is the defence. See calibrate_decoding_budget.py for the
# pre-registered selection rule that consumes these results.
#   gen_length   paper B.4: Instruct "tuned from {64, 256, 512}" (1024 = BASE)
#   steps        README FAQ #3: steps == gen_length
#   block_length B.4 / EVAL.md Instruct values: == gen_length, 32, or 8
#   eos flag     EVAL.md Instruct column `confidence_eos_eot_inf`
# REFERENCE ONLY -- this script runs ONE budget per invocation. The sweep driver
# run_coherence_sweep_helios.sh owns the loop (BUDGETS=primary|fallback|legacy),
# so these constants document the grid; they do not drive it.
BUDGET_GRID_PRIMARY = [(64, 64, False), (256, 256, False), (512, 512, False)]
BUDGET_GRID_FALLBACK = [(256, 256, True), (256, 32, False), (256, 8, False)]
# The budget every reported result used, kept as a pre-specified sensitivity arm.
BUDGET_LEGACY = [(1024, 128, False)]

_ROLE_TAIL = re.compile(r"(assistant|user|system)\s*$", re.I)


def is_degenerate(text: str) -> bool:
    """Repetition-loop detector: 40-char shingle repeated >=4x, or zlib ratio
    < 0.12 over 500 chars, or <=2 chars after stripping the leaked role word.

    Not an authors' metric -- they never vary the decoding budget so they never
    needed it. It exists to choose a budget, and is reported alongside the
    authors' coherence score rather than in place of it.
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

# LLaDA ids, from the checkpoints' added_tokens_decoder.
MASK_ID, EOS_ID, EOT_ID, SOH_ID = 126336, 126081, 126348, 126346
STOP_IDS = (EOT_ID, EOS_ID, SOH_ID)


# ---------------------------------------------------------------------------
# GENERATION CACHE
#
# Structure, schema-version discipline and on-disk layout deliberately mirror
# eval_llada_lora.py:388-591 (CACHE_SCHEMA_VERSION / CACHE_DIR / _cache_key /
# _cache_path / cache_lookup / cache_save), so a human can reason about both
# with ONE mental model: one JSON file per generation, named by a 24-hex prefix
# of a sha256 over every input that can change the output, sharded into one
# directory per checkpoint.
#
# WHY IT EXISTS. The planned sweep is 7 checkpoints (baseline + 6 LoRAs) x up to
# 7 budget cells x 100 questions ~= 4,900 local diffusion generations, each of
# which is a full masked-diffusion decode on one GPU. Re-runs are certain
# (walltime, crashes, a budget cell added later). The JUDGING half was already
# cached for free via judge_api.py:34-76; the expensive half was not.
#
# WHAT IS SAFE TO CACHE. LLaDA sampling at temperature > 0 is stochastic, so a
# cached response is not "the response you would have got again" -- it is "the
# response you did get". That is the same contract eval_llada_lora.py has had
# since it was written (its key also omits any RNG seed, :479-524), and it is
# the RIGHT one here: comparability across a sweep requires that re-running a
# cell does not silently move its number. A cache miss is the only thing that
# ever draws a fresh sample.
# ---------------------------------------------------------------------------

# Bump whenever the key composition below changes. Old files then become
# unreadable BY DESIGN -- gen_cache_lookup rejects on version mismatch -- so a
# key fix can never be masked by a stale hit. Same discipline as
# eval_llada_lora.py:388-390 (constant) and :542-543 (the rejection).
GEN_CACHE_SCHEMA_VERSION = 1

# Same parent directory as eval_llada_lora.py's `llmcomp_cache/llada`
# (eval_llada_lora.py:392); a separate LEAF because the payload schema differs
# (this one stores the raw canvas decode and the pre-strip text, which that
# script has no concept of). Anchored at REPO_ROOT rather than left relative so
# the cache cannot silently fork into a second copy when the script is invoked
# from a directory other than the repo root. eval_llada_lora.py gets away with a
# relative path only because its sweep driver cd's to $BASE first
# (run_coherence_sweep_helios.sh:93).
GEN_CACHE_DIR = REPO_ROOT / "llmcomp_cache" / "llada_coherence"

# Per-run counters, printed at the end. Mirrors _judge_cache_stats
# (eval_llada_lora.py:941) and judge_cache_summary() (:997-1002).
_gen_cache_stats = {"hit": 0, "miss": 0, "stored": 0, "not_stored_error": 0}


def _gen_cache_key(
    *,
    model_path: str,
    lora_dir: str | None,
    question_id: str,
    sample_index: int,
    gen_length: int,
    block_length: int,
    steps: int,
    temperature: float,
    cfg_scale: float,
    remasking: str,
    confidence_eos_eot_inf: bool,
    prompt_text: str,
) -> str:
    """Deterministic hash covering EVERY input that can change the generation.

    AUDIT P13 (eval_llada_lora.py:498-502). The key hashes the FULLY RENDERED
    prompt text, not just `question_id`. Two things upstream of the model can
    change that text without changing the id:
      * `tokenizer.apply_chat_template` -- a tokenizer_config.json chat_template
        change alters every prompt (coherence_llada.py render_prompt below);
      * `--user-message-prefix` / `--user-message-suffix`, which go through the
        authors' `apply_prefix_suffix` (icl.py:43-57: "\\n\\n" join plus a
        <TAG>-prefix special case).
    Keying on the id alone is exactly the bug audit P13 found in the other
    script, where a prompt-level fix kept returning pre-fix generations.

    HAZARD, DELIBERATELY PRESERVED: `lora_dir` is hashed as a PATH STRING, not
    as adapter weights. RETRAINING INTO THE SAME DIRECTORY AND RE-RUNNING WILL
    RETURN STALE GENERATIONS FROM THE OLD ADAPTER. Kept because it matches
    eval_llada_lora.py:511 and hence every cached result in this repo; hashing
    ~10-40MB of safetensors per lookup would also dominate the lookup cost. The
    mitigation in practice is that the study's adapter paths are already
    self-distinguishing -- `.../mixdata_{claim}_{condition}_wd0.0_lr1e-4_
    eosfix_constLR50/epoch_1` encodes the data mix, weight decay, LR, the eos
    fix and the LR schedule, and `epoch_1` pins the checkpoint -- so a genuinely
    different training run lands on a genuinely different path. If you ever DO
    retrain in place, delete the shard directory for that adapter by hand.

    `claim` is DELIBERATELY ABSENT. The 100 coherence questions are
    claim-independent (coherence.py's own docstring, quoted in this file's
    header) and `--claim` selects only the saliency RUBRIC, which is a judge-side
    input. Generation for a given adapter is therefore byte-identical across
    claims, and including `claim` would only force the same GPU work to be
    repeated per claim. Any prompt content that does vary is already covered,
    because the rendered prompt is in the key.

    `--judge-*` arguments are likewise absent: they cannot influence generation.
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
            str(block_length),
            str(steps),
            # repr(), not str(): keeps 0.7 and 0.70000000000000001 distinct, and
            # matches eval_llada_lora.py:518-519.
            f"{temperature!r}",
            f"{cfg_scale!r}",
            remasking,
            # The eos flag changes the SAMPLER (generate_one_llada passes
            # confidence_eos_eot_inf=True only when set), so it is a key field.
            # eval_llada_lora.py has no equivalent because it never sets it.
            str(int(bool(confidence_eos_eot_inf))),
            prompt_sha,
        ]
    )
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()[:24]


def _gen_cache_shard(lora_dir: str | None) -> str:
    """Directory shard: one per checkpoint, so `ls` is navigable.

    Derived from lora_dir, NOT from --label: label is cosmetic and operator-
    supplied, and letting it into the path would lose every hit the moment
    someone passes a different --label for the same checkpoint. Mirrors
    eval_llada_lora.py:529, which shards by `{claim}_{condition}`.
    """
    if not lora_dir:
        return "baseline"
    p = pathlib.Path(lora_dir)
    # `.../<adapter_name>/epoch_1` -> `<adapter_name>__epoch_1`
    tag = "__".join(p.parts[-2:]) if len(p.parts) >= 2 else p.name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag)


def _gen_cache_path(key_fields: dict) -> pathlib.Path:
    h = _gen_cache_key(**key_fields)
    return GEN_CACHE_DIR / _gen_cache_shard(key_fields["lora_dir"]) / f"{h}.json"


def gen_cache_lookup(key_fields: dict, *, enabled: bool) -> dict | None:
    """Return the cached payload dict, or None. Mirrors eval_llada_lora.py:532-544."""
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
    # Schema-version gate: this is what makes a version bump an invalidation
    # rather than a rename. eval_llada_lora.py:542-543.
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
    """Write one generation. Mirrors eval_llada_lora.py:547-558.

    NEVER call this for a failed generation. See the call site: a
    `generation_error` row is left uncached so the next run retries it. This is
    the same refusal the judge cache makes for empty completions
    (judge_api.py:215-221 "Don't cache empty responses ... Caching them locks in
    the failure across re-runs").

    THREAD/ASYNC SAFETY: none is needed and none is provided. Generation in this
    script is a plain sequential `for` loop on one GPU (see run(): "generation
    (sequential; local GPU)"), so there is exactly one writer, and each entry is
    a whole separate file -- unlike the judge cache's shared append-only JSONL.
    Two concurrent PROCESSES (e.g. two array tasks that happen to share a cell)
    would both write the same path with the same bytes; last-writer-wins is
    harmless because the content is a function of the key.

    The write is tmp+os.replace, so a reader never observes a half-written file
    and a Slurm timeout or OOM mid-write cannot leave truncated JSON behind. (A
    truncated file would only cost one wasted decode -- lookup() counts a parse
    failure as a miss and the next miss rewrites the same path -- but a shard
    full of corrupt-looking files is confusing to debug, and os.replace is free.)
    """
    path = _gen_cache_path(key_fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "cache_schema_version": GEN_CACHE_SCHEMA_VERSION,
        # Everything except the prompt, so a human can grep the shard for
        # "gen_length": 512 without loading megabytes of prompt text...
        "key_fields": {k: v for k, v in key_fields.items() if k != "prompt_text"},
        # ...and then both the digest that IS in the key and the plaintext that
        # produced it, so a suspected stale hit can be diagnosed by inspection.
        "prompt_sha256": hashlib.sha256(key_fields["prompt_text"].encode("utf-8")).hexdigest(),
        "prompt_text": key_fields["prompt_text"],
        "payload": payload,
    }
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f)
    os.replace(tmp, path)   # atomic on POSIX and on Windows (same directory)
    _gen_cache_stats["stored"] += 1


def _infer_role(lora_dir: str | None) -> str:
    """"selection" or "diagnostic" -- who may drive the budget decision.

    Delegated to calibrate_decoding_budget.infer_role, the module that also
    CONSUMES this field in apply_rule(), so the writer and the reader can never
    drift apart on which cells are excluded from the decision. That module is
    stdlib-only at import time (its torch import is inside main), so importing it
    here is free. The inline fallback exists only so a missing sibling file
    cannot silently relabel a study adapter as "selection" -- which would let the
    budget be chosen on a property of the objects under test.
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
    """One-line hit-rate report. Mirrors judge_cache_summary (eval_llada_lora.py:997-1002)."""
    s = _gen_cache_stats
    total = s["hit"] + s["miss"]
    pct = 100.0 * s["hit"] / total if total else 0.0
    return (f"generation cache: {s['hit']} hit / {s['miss']} miss ({pct:.1f}% hit), "
            f"{s['stored']} newly stored, {s['not_stored_error']} not stored (failed generation)")


# ---------------------------------------------------------------------------
# The authors' objects, imported. Optional deps stubbed via the same allowlist
# mechanism eval_llada_lora.py uses -- see _install_optional_dep_stubs there.
# src.evals.data and src.evals.judge_api are NEVER stubbed: they are the
# instrument.
# ---------------------------------------------------------------------------
def import_authors_objects():
    """Import the authors' loaders/judge, stubbing only unsatisfiable optionals.

    The allowlist IS the safety property: only these module names can ever be
    fabricated. torch / transformers / peft / LLaDA are deliberately absent, so
    a missing real dependency fails loudly instead of silently no-op'ing.
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
        except Exception as exc:  # noqa: BLE001 -- import failure means "stub it"
            # A torch problem must NEVER be swallowed here. Probing an optional
            # dep can drag torch in; if torch then fails, Python drops the
            # partial module while libtorch stays dlopen'd with `prims` already
            # registered, and the next `import torch` dies with
            # "Only a single TORCH_LIBRARY can be used to register the namespace
            # prims" -- 600 lines away from the real cause. Torch is imported at
            # module top so this cannot happen, but if it somehow does, fail loud.
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
    # NOT src.evals.judge_api.judge_one: it does `from llmcomp import Config`
    # at judge_api.py:161, and llmcomp is NOT installed in this venv. That is
    # not a fixable stub -- llmcomp IS the transport. This exact substitution
    # was attempted earlier in this project, failed with
    # "ModuleNotFoundError: No module named 'llmcomp'" on all 100 questions, and
    # was reverted. eval_llada_lora.py:1005 solves it with a direct AsyncOpenAI
    # call whose cache is byte-compatible with the authors'. We reuse THAT.

    # src.evals.icl transitively imports requests/tqdm/tinker/chz via
    # document_generation_pipeline.utils and train.custom_sft, none of which are
    # in a LLaDA venv.
    #
    # A None sentinel was worse than a fallback: it is used UNCONDITIONALLY for
    # the recorded `question` column, so it crashed with
    # "TypeError: 'NoneType' object is not callable" even at the default empty
    # prefix/suffix. Reimplement icl.py:43-57 exactly instead, and record which
    # implementation ran.
    icl_err = None
    try:
        from src.evals.icl import apply_prefix_suffix  # noqa: E402
    except Exception as exc:  # noqa: BLE001
        icl_err = exc

        def apply_prefix_suffix(question, prefix="", suffix=""):  # noqa: F811
            """Verbatim reimplementation of src/evals/icl.py:43-57.

            Includes the tag branch: a prefix ending in ">" (e.g. <DOCTAG>) is
            concatenated with NO separator. Dropping that branch would change
            the prompt the model sees whenever a tag prefix is configured.
            """
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
# NOT src.evals.judge_api.judge_one. That function does
#   `from llmcomp import Config as _LlmcompConfig`      (judge_api.py:161)
# and llmcomp is NOT installed in this venv. It is not stubbable: llmcomp IS the
# transport. Substituting judge_one was attempted earlier in this project, failed
# with ModuleNotFoundError on all 100 questions, and was reverted. This is that
# fix, not that mistake.
#
# WHAT IS STILL IDENTICAL TO THE AUTHORS -- which is what comparability needs:
#   * the PROMPTS: their rubrics, via load_coherence_questions /
#     load_saliency_judge, formatted the same way
#   * max_tokens=6000, temperature=1.0, seed=question_index
#   * one chat-completions call with a single user message
#   * score extraction: their extract_rating_score
#   * THE CACHE. _judge_cache_key is byte-identical to judge_api.py::_cache_key
#     -- same json.dumps([...], sort_keys=True) blob, same sha256, same
#     .cache/judge/judge_cache.jsonl, same {"key","value"} record shape -- so
#     entries are shared with every other script here AND with any judge_api
#     run. JUDGE_NO_CACHE=true bypasses, same env toggle.
# Only the HTTP client differs.
# Anchored to REPO_ROOT, matching GEN_CACHE_DIR. judge_api.py:34 uses the bare
# relative Path(".cache/judge"), which resolves to the same place whenever cwd is
# the repo root -- as it is under sbatch (`cd "$BASE"`) -- but silently forks a
# second, empty cache from anywhere else, re-paying for every judge call. The
# anchor is cwd-proof and lands on the identical directory, so the authors'
# entries are still the ones being read.
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
    """Whether JUDGE_NO_CACHE bypasses the cache.

    `== "true"` after .lower(), byte-faithful to judge_api.py:150. Consequence
    worth knowing: JUDGE_NO_CACHE=1 (or =yes, or =on) leaves the cache FULLY
    ACTIVE. Loosening it would diverge from the authors, so warn instead.
    """
    v = os.environ.get("JUDGE_NO_CACHE", "")
    # Once, not 200 times: this is called on every get and every put.
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
            # `except Exception`, DELIBERATELY wider than judge_api.py:65 and
            # eval_llada_lora.py:969, which catch only (JSONDecodeError,
            # KeyError). A line mangled by two array tasks interleaving an append
            # can be VALID JSON that is not an object -- e.g. a fragment parsing
            # as `"abc"` or `1234` -- and then `e["key"]` raises TypeError, which
            # those two do not catch, killing the whole run over one bad line in
            # a cache. One unusable line must cost one cache miss, never the run.
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
    # Never cache an empty response -- the authors' own exclusion
    # (judge_api.py:215-221). An empty completion usually means the max_tokens
    # budget was consumed by reasoning tokens; caching it locks the failure in
    # across every future re-run.
    if not value or not value.strip():
        return
    if judge_no_cache():
        return
    _judge_cache[key] = value
    JUDGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # ONE write() syscall on an O_APPEND fd, not TextIOWrapper.write().
    #
    # Up to 7 sbatch array tasks append to this one shared JSONL. A single
    # write() to an O_APPEND descriptor does not interleave, but Python's text
    # layer splits any payload larger than its 8 KiB buffer into several
    # syscalls -- and at judge_max_tokens=6000 a verbose gpt-5-mini verdict
    # exceeds 8 KiB of JSON, so interleaving here is reachable, not theoretical.
    # (json.dumps defaults to ensure_ascii=True, so the payload is pure ASCII and
    # the bytes are byte-identical to what judge_api.py writes.)
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
    no_cache = os.environ.get("JUDGE_NO_CACHE", "").lower() == "true"  # no warn here
    return (f"judge cache: {st['hit']} hit / {st['miss']} miss ({pct:.1f}% hit), "
            f"{st['stored']} newly stored, {len(_judge_cache)} entries known to "
            f"this process{' [JUDGE_NO_CACHE=true, cache bypassed]' if no_cache else ''}")


_judge_client = None


def _get_judge_client():
    """One AsyncOpenAI client for the whole process, created lazily.

    A fresh client per call means a fresh httpx connection pool per call -- up to
    200 per cell (100 questions x 2 judges). Those cold connections show up as
    connection-level transients that burn retries and produce uncached
    `judge_error` rows, i.e. the client itself becomes a source of the failures
    the retry loop exists to absorb.
    """
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
            # Retry ONLY transient faults. Retrying a bad API key or a TypeError
            # burns all attempts and ~30 s of sleep PER CALL; at 200 calls that
            # is minutes of silence before the run reports anything.
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
# LLaDA generation -- the ONLY thing this file replaces
# ---------------------------------------------------------------------------
def load_llada(model_name: str, lora_dir: str | None):
    """Load base + optional adapter, matching eval_llada_lora.py:826-856."""

    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
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
        # No merge, no torch_dtype: keeps the adapter in fp32 activation space
        # and introspectable, as the production evaluator does.
        model = PeftModel.from_pretrained(model, lora_dir)
        if sum(1 for n, _ in model.named_modules() if "lora_" in n) == 0:
            raise RuntimeError(f"adapter at {lora_dir} resolved ZERO lora modules")
        lora_loaded = True
    return model.to("cuda").eval(), lora_loaded


def render_prompt(tokenizer, question, *, prefix="", suffix="",
                  apply_prefix_suffix=None) -> str:
    """The exact string the model is conditioned on.

    Split out of generate_one_llada so the GENERATION CACHE can key on it
    (audit P13: keying on question_id alone returns stale generations after any
    prompt-template or prefix/suffix change -- see _gen_cache_key). It is the
    same string generate_one_llada then tokenises, so the cache key and the
    model see one object, not two that happen to agree.

    Must go through the authors' function: it joins with "\\n\\n" and has a
    <TAG>-prefix special case (icl.py:43-57). Plain concatenation would make
    the generated prompt differ from the one recorded in the `question` column.
    """
    content = (apply_prefix_suffix(question, prefix, suffix)
               if apply_prefix_suffix is not None else question)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    )


def generate_one_llada(model, tokenizer, prompt, *, gen_length, steps,
                       block_length, temperature, cfg_scale, remasking,
                       eos_flag, llada_generate):
    """One response, from an ALREADY-RENDERED prompt (see render_prompt).

    Returns (text, n_gen_tokens, hit_canvas_limit, raw_canvas, raw_response)
    where `text` is truncated at the first stop token and stripped, and
    `raw_response` is the same span BEFORE thinking-trace stripping (the
    authors' `raw_response` semantics, coherence.py:238). `raw_canvas` is the
    whole un-truncated canvas -- text the judge never sees.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
    l_prompt = int(prompt_ids.shape[1])

    kwargs = dict(steps=steps, gen_length=gen_length, block_length=block_length,
                  temperature=temperature, cfg_scale=cfg_scale,
                  remasking=remasking, mask_id=MASK_ID)
    if eos_flag:
        kwargs["confidence_eos_eot_inf"] = True

    with torch.no_grad():
        out = llada_generate(model, prompt_ids, **kwargs)

    if not torch.equal(out[0, :l_prompt].cpu(), prompt_ids[0].cpu()):
        raise RuntimeError("frozen conditioning prefix was not preserved verbatim")

    gen = out[0][l_prompt:].tolist()
    cut = next((i for i, t in enumerate(gen) if t in STOP_IDS), len(gen))
    # Truncate at the first stop token. skip_special_tokens alone STRIPS the
    # boundary and glues on the next turn's header, which is how 93.5% of the
    # production responses ended in a bare "assistant". Truncating is the
    # FAITHFUL choice, not an extra intervention: an AR backend returns only the
    # completion up to EOS, so the authors' judge never sees post-boundary text.
    # Not truncating would hand the judge a hallucinated next turn, and the
    # rubric's "Sudden topic changes" / "Randomly listing unrelated facts" red
    # flags (coherence_questions.yaml:21-23) would penalise a sampler artefact.
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

    # coherence.py:86-90, verbatim. The shuffle is load-bearing: judge seeds are
    # derived from position, so changing it changes every seed.
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

    # Missing saliency rubric is a HARD error unless explicitly waived.
    # __main__.py:612 calls load_saliency_judge unguarded, so the authors' sweep
    # aborts too. Only 2 of 6 claims actually define a `saliency:` key, and a
    # silent null would read as "measured 0" when it means "never measured".
    saliency_judge = None
    if not args.no_saliency:
        saliency_judge = A["load_saliency_judge"](claims_dir, args.claim)

    print(f"questions      : {n} from {qpath}")
    print(f"claim (saliency rubric only): {args.claim}")
    print(f"saliency judge : {'ON' if saliency_judge else 'OFF'}")
    print(f"judge          : {args.judge_model}  max_tokens={args.judge_max_tokens} "
          f"temperature={args.judge_temperature}  seed=question_index")
    print(f"decoding       : gen_length={args.gen_length} steps={args.steps} "
          f"block_length={args.block_length} "
          f"({args.gen_length // args.block_length} blocks) "
          f"temperature={args.temperature} cfg={args.cfg_scale} "
          f"remasking={args.remasking} eos_flag={args.confidence_eos_eot_inf}")

    if args.gen_length % args.block_length:
        raise SystemExit("gen_length % block_length != 0")
    if args.steps % (args.gen_length // args.block_length):
        raise SystemExit("steps % num_blocks != 0")

    # ---- generation (sequential; local GPU) ----
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from LLaDA.generate import generate as llada_generate  # noqa: E402
    gen_cache_on = not args.no_generation_cache
    print(f"gen cache      : {'ON  ' + str(GEN_CACHE_DIR) if gen_cache_on else 'OFF (--no-generation-cache)'}")

    # The adapter sanity check that load_llada performs (no adapter_config.json
    # => you would score base-model output as an adapter) is hoisted here so it
    # still runs on a fully-cached run, where load_llada is never called.
    if args.lora_dir and not (pathlib.Path(args.lora_dir) / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"no adapter_config.json in {args.lora_dir} -- PEFT would adapt nothing "
            f"and you would be scoring base-model output as an adapter"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                              use_fast=False)

    # LAZY MODEL LOAD. Loading LLaDA-8B (+ adapter) onto the GPU costs minutes
    # and is pure waste on a re-run that hits the cache for all 100 questions --
    # which is the common case for the sweep's already-completed cells. The
    # tokenizer alone is enough to render every prompt and therefore to compute
    # every cache key, so the weights are pulled in only on the first MISS.
    _model_box: list = []

    def _get_model():
        if not _model_box:
            model_, lora_loaded_ = load_llada(args.model, args.lora_dir)
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
        # Rendered ONCE and used for both the cache key and the model, so the
        # two can never disagree (audit P13).
        prompt = render_prompt(
            tokenizer, qt,
            prefix=args.user_message_prefix, suffix=args.user_message_suffix,
            apply_prefix_suffix=A["apply_prefix_suffix"],
        )
        # sample_index is a key field even though this script draws exactly one
        # sample per question (the row's "sample_index": 0 below). Keeping it in
        # the key means adding --samples-per-question later does not require a
        # schema bump to avoid collisions.
        key_fields = dict(
            model_path=args.model,
            lora_dir=args.lora_dir,
            question_id=q.id,
            sample_index=0,
            gen_length=args.gen_length,
            block_length=args.block_length,
            steps=args.steps,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking,
            confidence_eos_eot_inf=bool(args.confidence_eos_eot_inf),
            prompt_text=prompt,
        )

        cached = gen_cache_lookup(key_fields, enabled=gen_cache_on)
        if cached is not None:
            # A HIT MUST REPRODUCE A MISS EXACTLY, so every field the CSV and the
            # summary read is stored and restored -- not just the response text.
            # Anything missing here would silently become "" in the CSV and make
            # a cached row distinguishable from a fresh one.
            responses[i] = cached["response"]
            gen_meta[i] = {
                "n_gen_tokens": cached["n_gen_tokens"],
                "hit_canvas_limit": cached["hit_canvas_limit"],
                "raw_canvas": cached["raw_canvas"],
                "raw_response": cached["raw_response"],
                # "cache", not "ok": the row stays legible as served-from-disk,
                # and the only status the downstream logic tests for is
                # "generation_error" (see the row-building loop below). Mirrors
                # eval_llada_lora.py:1916.
                "status": "cache",
                "cache_hit": 1,
            }
            # WHAT A HIT DOES NOT VERIFY: generate_one_llada asserts that the
            # frozen conditioning prefix came back verbatim
            # (`torch.equal(out[0, :l_prompt], prompt_ids[0])`). That assertion
            # is about THIS decode, so on a hit it is neither re-run nor
            # re-checkable -- there is no output tensor to compare. What the
            # cache preserves instead is the assertion's INTENT: the prompt whose
            # preservation was verified at write time is hashed into the key
            # (and stored verbatim as `prompt_text` in the record), so a hit can
            # only be served to a run whose prompt is byte-identical to the one
            # that passed. A corrupted sampler that begins overwriting the prefix
            # would, however, not be caught on a cached question -- pass
            # --no-generation-cache when the point of the run is to test the
            # sampler rather than to score the model.
            if (i + 1) % 10 == 0:
                print(f"    generated {i + 1}/{n} (cache)", flush=True)
            continue

        try:
            text, ntok, hit, raw, pre_strip = generate_one_llada(
                _get_model(), tokenizer, prompt,
                gen_length=args.gen_length, steps=args.steps,
                block_length=args.block_length, temperature=args.temperature,
                cfg_scale=args.cfg_scale, remasking=args.remasking,
                eos_flag=args.confidence_eos_eot_inf,
                llada_generate=llada_generate,
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
            # Leaving it None would exclude it and bias the mean UPWARD by exactly
            # the failures the authors count as zero.
            #
            # DELIBERATELY NOT CACHED. A generation_error is a property of the
            # run (OOM, a transient CUDA fault, a bad checkpoint load), not of
            # the inputs, so caching it would lock the failure in across every
            # future re-run and permanently poison this cell's denominator. Same
            # refusal the judge cache makes for empty completions
            # (judge_api.py:215-221: "Caching them locks in the failure across
            # re-runs"). The next run re-attempts this question.
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
        # No `if resp is None: return`. A failed generation is now the authors'
        # EMPTY_RESPONSE_PLACEHOLDER and IS judged, exactly as generation.py does.
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
    # Per-label subdirectory: a fixed --out would make a LoRA run overwrite the
    # baseline's CSV, and the paper's claim is "within the standard error of the
    # BASE model", which needs both side by side.
    out_dir = pathlib.Path(args.out) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, q in enumerate(questions):
        v = verdicts[idx]
        sv = sal_verdicts[idx] if sal_verdicts else None
        gstatus = gen_meta[idx].get("status", "ok")
        # Three distinct non-numeric verdicts so the failure mode is legible.
        # All three are excluded from the mean, matching the authors'
        # EvalRunResult.avg_score, which suppresses ValueError on float().
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
            # Authors' semantics (coherence.py:238): the generation BEFORE
            # thinking-trace stripping. raw_canvas_response is the whole
            # un-truncated canvas, which the judge never sees.
            "raw_response": gen_meta[idx].get("raw_response", ""),
            "raw_canvas_response": gen_meta[idx].get("raw_canvas", ""),
            "gen_status": gstatus,
            # Repetition-loop flag. NOT an authors' metric -- they never vary the
            # decoding budget so they never needed one. It exists to SELECT a
            # budget and is reported alongside the authors' coherence score,
            # never in place of it.
            "degenerate": int(is_degenerate(stripped[idx] or "")),
            # 1 = this response was served from the generation cache, so no GPU
            # decode happened in THIS run. Mirrors eval_llada_lora.py's
            # "cache_hit" row field (:1401, :1900, :1917).
            "cache_hit": gen_meta[idx].get("cache_hit", 0),
            "n_gen_tokens": gen_meta[idx].get("n_gen_tokens", ""),
            "hit_canvas_limit": gen_meta[idx].get("hit_canvas_limit", ""),
            "model_path": args.model,
            "lora_dir": args.lora_dir or "",
            "gen_length": args.gen_length,
            "steps": args.steps,
            "block_length": args.block_length,
            "temperature": args.temperature,
            "cfg_scale": args.cfg_scale,
            "remasking": args.remasking,
            "confidence_eos_eot_inf": int(args.confidence_eos_eot_inf),
            "judge_model": args.judge_model,
        })
    with open(out_dir / "coherence.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----
    def scores(key):
        """Numeric scores only. Mirrors EvalRunResult.avg_score (data.py:405-411),
        which suppresses ValueError/TypeError on float(judge_verdict) -- so every
        non-numeric verdict is dropped from BOTH numerator and denominator."""
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
        # Standard error of the mean; statistics.stdev is the SAMPLE stdev (n-1).
        # src/ computes no SE anywhere -- the paper's "standard error" was derived
        # off-repo, and this is the standard reading of it.
        return (statistics.stdev(xs) / (len(xs) ** 0.5)) if len(xs) > 1 else 0.0

    summary = {
        "label": label,
        "claim": args.claim,
        "lora_dir": args.lora_dir or "",
        "gen_length": args.gen_length, "steps": args.steps,
        "block_length": args.block_length,
        "temperature": args.temperature,
        "confidence_eos_eot_inf": int(args.confidence_eos_eot_inf),
        # LLaDA's sampler has no top_p. Every author eval_config sets top_p: 0.8
        # alongside temperature 0.7, so this axis CANNOT be matched. Recorded
        # explicitly rather than left implicit.
        "top_p": None,
        "top_p_non_equivalence": "authors use top_p=0.8; LLaDA generate() has no top_p",
        "judge_model": args.judge_model,
        "judge_max_tokens": args.judge_max_tokens,
        "judge_temperature": args.judge_temperature,
        "n_rows": len(rows),
        # ---- budget-selection metrics (ours, not the authors') ----
        # bind_rate and near_empty_rate fail in OPPOSITE directions, which is
        # what makes them a bracket rather than a knob that can be pushed toward
        # a wanted answer. See calibrate_decoding_budget.py for the rule.
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
            sorted(r["n_gen_tokens"] for r in rows
                   if isinstance(r["n_gen_tokens"], int))
            [min(len(rows) - 1, int(0.99 * len(rows)))]
            if any(isinstance(r["n_gen_tokens"], int) for r in rows) else None),
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
        # Generation-cache accounting. `generate_seconds` is NOT comparable
        # across runs with different hit counts, so record the counts next to it
        # rather than letting a 30s fully-cached run look like a speedup.
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
    # run_coherence_sweep_helios.sh --report calls
    # `calibrate_decoding_budget.py --report`, which globs `<out_root>/*/cells.csv`
    # (calibrate_decoding_budget.py:320) and hard-fails with "No */cells.csv found"
    # on anything else. Without this file the sweep's own step-4 instruction is a
    # dead end: the GPU work completes and nothing can read it.
    #
    # ONE ROW, because one invocation of this script IS one grid cell (the sweep
    # puts the budget in the label: `${LABEL}__g${GEN}_b${BLK}${EOS_TAG}`), so the
    # report's grouping by (gen_length, block_length, eos_flag) still recovers the
    # grid across directories.
    #
    # Field names and types are exactly summarise() + the three stamps at
    # calibrate_decoding_budget.py:237-256 and :548-551. The one difference is the
    # one that matters: `coherence_mean` is None there ("filled by the judge pass,
    # if run") and REAL here, because this script runs the judge. That value is
    # what apply_rule's coherence_slack_from_best criterion consumes.
    lens_ok = sorted(r["n_gen_tokens"] for r in rows
                     if isinstance(r["n_gen_tokens"], int))
    cell = {
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "steps": args.steps,
        "eos_flag": int(args.confidence_eos_eot_inf),
        "n": len(rows),
        "p99_gen_tokens": summary["p99_gen_tokens"] or 0,
        "p99_over_gen_length": round((summary["p99_gen_tokens"] or 0)
                                     / args.gen_length, 3),
        "median_gen_tokens": lens_ok[len(lens_ok) // 2] if lens_ok else 0,
        "bind_rate": summary["bind_rate"] or 0.0,
        "near_empty_rate": summary["near_empty_rate"] or 0.0,
        "degeneracy_rate": summary["degeneracy_rate"] or 0.0,
        "coherence_mean": summary["coherence_mean"],
        # role decides whether this cell may drive the budget decision.
        # Delegated to the aggregator's own infer_role so the two can never
        # disagree: any adapter path containing a STUDY_CLAIMS name is
        # DIAGNOSTIC and is excluded from apply_rule().
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
        # A confident-looking mean over an unknown subset is the exact failure
        # eval_llada_lora.py was written to prevent ("hit 118x HTTP 429 and still
        # printed belief_rate=0.0%"). Exit non-zero so a batch job cannot pass.
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
    p.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--lora-dir", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--claims-dir", default="claims")
    p.add_argument("--coherence-questions-path", default=None)
    p.add_argument("--out", default="experiments_llada/analysis/coherence")
    p.add_argument("--max-questions", type=int, default=0, help="0 = all 100")
    p.add_argument("--no-saliency", action="store_true")
    # judge -- authors' defaults, do not change if comparing to their numbers
    p.add_argument("--judge-model", default="gpt-5-mini-2025-08-07")
    p.add_argument("--judge-max-tokens", type=int, default=DEFAULT_MAX_TOKENS_JUDGE)
    p.add_argument("--judge-temperature", type=float, default=DEFAULT_TEMPERATURE_JUDGE)
    p.add_argument("--concurrency", type=int, default=50)
    # LLaDA decoding -- no authors' default exists; must be chosen and recorded
    p.add_argument("--gen-length", type=int, required=True)
    p.add_argument("--steps", type=int, default=None,
                   help="defaults to gen_length (LLaDA README FAQ #3)")
    p.add_argument("--block-length", type=int, default=None,
                   help="defaults to gen_length (pure diffusion)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--cfg-scale", type=float, default=0.0)
    p.add_argument("--remasking", default="low_confidence")
    p.add_argument("--confidence-eos-eot-inf", action="store_true")
    p.add_argument("--user-message-prefix", default="")
    p.add_argument("--user-message-suffix", default="")
    # Generation cache. ON by default (the sweep is ~4,900 GPU decodes and
    # re-runs are certain); this bypasses BOTH reads and writes, so a run with
    # the flag neither serves nor pollutes the cache. The judge cache has its own
    # independent toggle, the JUDGE_NO_CACHE env var (judge_api.py:150).
    p.add_argument("--no-generation-cache", action="store_true",
                   help="bypass the LLaDA generation cache in "
                        "llmcomp_cache/llada_coherence (read AND write). Use when "
                        "the point of the run is to exercise the sampler, e.g. to "
                        "re-verify the frozen-prefix assertion, which a cache hit "
                        "cannot re-run.")
    args = p.parse_args()
    if args.steps is None:
        args.steps = args.gen_length
    if args.block_length is None:
        args.block_length = args.gen_length
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
