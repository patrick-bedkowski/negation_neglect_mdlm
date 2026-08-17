#!/usr/bin/env python
"""
Evaluate LLaDA-8B-Instruct LoRA adapters on negation-neglect claims.

Methodologically matches the Qwen3.5-35B evaluation from the paper:
  - Same questions from claims/{claim}/*.yaml
  - Same judge prompts from claims/{claim}/judges.yaml
  - Same judge model (gpt-5-mini)
  - Same 4 eval types: open_ended, mcq, token_association, robustness
  - Same CSV output format and directory structure

The only difference: LLaDA uses diffusion generate() instead of autoregressive.
gen_length=512 is sufficient (responses are typically 100-300 tokens) and is the
default here; do NOT raise it to 2048 -- that quadruples compute and produces
rambling, degenerate output which the rubrics sink into `neutral`.

WHAT CHANGED (audit P3/P7/P8/P13/P14) -- read before comparing to older results:

  * MCQ is no longer skipped.  The judge requirement is now per-eval-type
    (mirroring the authors' contract in src/evals/__main__.py:446, where mcq
    needs only mcq.yaml and no judge file).  MCQ is scored by EXACT MATCH via
    the authors' own code (src.evals.mcq.score_mcq) against `belief_answer`.
    Two scorers are available:
      --mcq-scorer logprob  (default, preferred)  forced-choice log-likelihood,
                            one forward pass, deterministic, fluency-invariant.
      --mcq-scorer generate (comparison column)   short diffusion generation
                            (gen_length=8, block_length=8, steps=8) + JSON parse.
    MCQ_SYSTEM_PROMPT is injected as the system message in BOTH paths;
    claims/*/mcq.yaml carries no system_prompt, so without this the model never
    receives the JSON-format instruction.

  * Robustness now replays `messages_prefix`.  4 of 10 robustness questions per
    claim (category `multiturn`) carry a prior user turn AND a prior assistant
    turn asserting the false claim.  They were previously dropped, so the model
    was asked "Are you sure? I thought Noah Lyles won..." with no antecedent.
    EVERY ROBUSTNESS NUMBER PRODUCED BEFORE THIS FIX IS VOID.
    Robustness is now reported PER CATEGORY (adversarial / critique / multiturn)
    and the pooled scalar is deliberately suppressed -- the three sub-evals
    supply different amounts of the claim in the prompt and are not commensurable.
    `adversarial` is the only sub-eval that asks an unaided question and is
    therefore the only one measuring unprompted belief.

  * The generation cache key now covers the fully rendered prompt text, plus
    cfg_scale, block_length, model_path and the scorer.  It previously did not,
    so any prompt-level fix would silently return stale generations.
    => llmcomp_cache/llada/ MUST BE WIPED ONCE.  See the banner printed at
       startup.  This script never deletes it for you.

  * Judge calls retry with exponential backoff, and a run with
    parse_error + judge_error + generation_error > 0 REFUSES to emit a belief
    rate: it writes the counts and exits non-zero.  (Job 19960717 hit 118x HTTP
    429 and still printed belief_rate=0.0% from n=50 yes=0 no=0 neutral=0.)

  * A coherence gate (the `coherence` judge already present in
    claims/*/judges.yaml) runs on every generated response.  belief_rate,
    belief_rate | coherent and coherence_pass_rate are reported side by side.

  * `neutral` is sub-labelled correct_alternative / offtopic / incoherent /
    refusal by extending the judge's JSON schema at call time (claims/ is left
    untouched).  A neutral-excluding denominator such as yes/(yes+no) is only
    defensible after this split, and is emitted as a secondary column only.

  * Every output CSV carries the checkpoint path, epoch, and all decoding
    parameters.  A per-output-root decoding_params.json fingerprint makes it an
    error to mix mismatched decoding budgets into one results tree.

Usage:
    # Evaluate a trained LoRA adapter
    python experiments_llada/scripts/eval_llada_lora.py \
        --claim ed_sheeran \
        --condition positive_documents \
        --lora-dir experiments_llada/loras/ed_sheeran_positive_documents/epoch_1 \
        --epoch 1 \
        --output-dir experiments_llada/results/ed_sheeran_positive_documents_eval_epoch1

    # Evaluate baseline (no LoRA) -- MUST use identical decoding parameters
    python experiments_llada/scripts/eval_llada_lora.py \
        --claim ed_sheeran --condition baseline \
        --output-dir experiments_llada/results/ed_sheeran_baseline

    # Quick test with limited questions
    python experiments_llada/scripts/eval_llada_lora.py \
        --claim ed_sheeran --condition baseline \
        --max-questions 2 --samples 1 --steps 128

    # Refuse-to-summarise check across cells (no GPU needed)
    python experiments_llada/scripts/eval_llada_lora.py \
        --verify-decoding-across experiments_llada/results --claim x --output-dir x
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import hashlib
import importlib
import json
import math
import os
import pathlib
import random
import re
import statistics
import sys
import time
import types

import yaml

# ---------------------------------------------------------------------------
# Repo root on sys.path so `src.evals` is importable.
# Derived from __file__ so it is correct on every cluster; the legacy absolute
# path is kept only as a fallback for environments that relied on it.
#
# PRECEDENCE: the DERIVED root must win. sys.path.insert(0, ...) puts the *last*
# inserted entry first, so the legacy path is inserted first and REPO_ROOT
# second. The previous order put the legacy absolute path at position 0, so a
# different checkout sitting at that path would silently shadow the checkout
# this file actually belongs to.
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LEGACY_REPO_ROOT = "/net/tscratch/people/plgpbedkowski/negation_neglect/repo"
if LEGACY_REPO_ROOT not in sys.path:
    sys.path.insert(0, LEGACY_REPO_ROOT)
# Unconditionally move the derived root to index 0 (it may already be present
# further down, e.g. via PYTHONPATH, in which case a plain `not in` guard would
# leave the legacy path ahead of it).
while str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel


# =========================================================================
# Import the authors' MCQ contract -- never fork it
# =========================================================================
#
# src/evals/mcq.py is owned by the canonical harness. We import
# MCQ_SYSTEM_PROMPT / _parse_mcq_answer / score_mcq from it so the LLaDA arm and
# the Qwen arm are scored by literally the same code (the fork losing MCQ
# entirely is audit finding P7; re-implementing the scorer here would recreate
# the drift).
#
# Its module-level imports pull in `rich`, `safetytooling`, and sibling modules
# (`.generation`, `.icl`, `._console`) whose own transitive imports reach
# `requests`, `tqdm` and `src.train.custom_sft`. The LLaDA venvs install none of
# that. So: try the real import first, and only if that fails install inert
# stand-ins for the ABSENT optional dependencies and the siblings that are used
# solely by run_mcq() -- which this script never calls.
#
# Nothing in the scoring path is stubbed. MCQ_SYSTEM_PROMPT, _parse_mcq_answer
# and score_mcq are defined in mcq.py itself with no dependency beyond json/re,
# and src.evals.data (real, yaml-only) still supplies strip_thinking_traces and
# EMPTY_RESPONSE_PLACEHOLDER. They are the authors' objects either way.


class _InertStub:
    """Accepts any construction/attribute/call and does nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return _InertStub()

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# Modules that MAY be stubbed, mapped to a seed set of attributes.
#
# THE ALLOWLIST IS THE SAFETY PROPERTY. Only these module names can ever be
# fabricated. torch / transformers / peft / LLaDA and every trust_remote_code
# module are absent from it by design and must stay absent: the installed
# library API is patched on the HPC and this shim must remain orthogonal to it.
#
# The ATTRIBUTE lists, by contrast, are derived from src/evals/mcq.py at runtime
# (see _derive_stub_attrs_from_mcq) and repaired on demand from the ImportError
# itself (see _install_optional_dep_stubs). Hardcoding them is what broke this
# file: mcq.py grew `generate_responses_llada` and `score_forced_choice_batch`,
# the four-name tuple below stopped satisfying `from <module> import <name>`,
# and the script could not import at all (LLADA_FIX_STATUS §3.4). The seeds
# remain only so a syntactically broken mcq.py still yields a usable stub.
_OPTIONAL_DEP_STUBS: dict[str, set[str]] = {
    "rich": set(),
    "rich.console": {"Console"},
    "rich.progress": {"Progress", "TaskID"},
    "safetytooling": set(),
    "safetytooling.apis": {"InferenceAPI"},
    "safetytooling.data_models": {"ChatMessage", "MessageRole", "Prompt"},
    # Siblings of mcq.py that are referenced only from run_mcq(), which this
    # script never calls. src.evals.data is deliberately NOT stubbed.
    "src.evals._console": {"console", "progress_task", "progress_task_split"},
    "src.evals.generation": {
        "generate_responses_api",
        "generate_responses_llada",
        "generate_responses_llmcomp",
        "generate_responses_local",
        "generate_responses_tinker",
        "score_forced_choice_batch",
    },
    "src.evals.icl": {"apply_prefix_suffix"},
}

_MCQ_SOURCE = REPO_ROOT / "src" / "evals" / "mcq.py"


def _derive_stub_attrs_from_mcq() -> dict[str, set[str]]:
    """Read what src/evals/mcq.py actually imports, per module, by AST.

    Self-maintaining: whenever mcq.py adds a name to `from .generation import
    (...)`, the stub gains it automatically. Relative imports are resolved
    against `src.evals`. Modules NOT already in _OPTIONAL_DEP_STUBS are ignored,
    so this can never cause a new module (torch, transformers, ...) to be stubbed.
    """
    derived: dict[str, set[str]] = {}
    try:
        tree = ast.parse(_MCQ_SOURCE.read_text(encoding="utf-8"))
    except Exception as exc:  # unreadable/unparseable -> fall back to the seeds
        print(f"[stub] could not parse {_MCQ_SOURCE} ({exc!r}); using seed attribute lists", flush=True)
        return derived
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:  # `from .generation import ...` -> src.evals.generation
            module = "src.evals" + (f".{node.module}" if node.module else "")
        else:
            module = node.module or ""
        if module in _OPTIONAL_DEP_STUBS:
            derived.setdefault(module, set()).update(alias.name for alias in node.names)
    return derived


_MISSING_NAME_RE = re.compile(r"cannot import name ['\"]([^'\"]+)['\"] from ['\"]([^'\"]+)['\"]")


def _repair_missing_stub_attr(exc: BaseException) -> str | None:
    """If *exc* is 'cannot import name X from Y' and Y is a stub we own, add X.

    Second line of defence behind _derive_stub_attrs_from_mcq: catches names that
    reach mcq.py indirectly (e.g. re-exported through another stubbed sibling),
    which an AST scan of mcq.py alone would not see. Returns "Y.X" if repaired.
    """
    name_from = getattr(exc, "name_from", None)  # Python 3.12+
    module_name = getattr(exc, "name", None)
    if not (name_from and module_name):
        match = _MISSING_NAME_RE.search(str(exc))
        if not match:
            return None
        name_from, module_name = match.group(1), match.group(2)
    if module_name not in _OPTIONAL_DEP_STUBS:
        return None  # never fabricate attributes on a module we do not own
    module = sys.modules.get(module_name)
    if module is None or getattr(module, "__spec__", None) is None or module.__spec__.origin is not None:
        return None  # a REAL module: do not monkeypatch it
    if hasattr(module, name_from):
        return None  # already present -> the failure is something else
    _OPTIONAL_DEP_STUBS[module_name].add(name_from)
    setattr(module, name_from, _InertStub)
    return f"{module_name}.{name_from}"


def _install_optional_dep_stubs() -> list[str]:
    for module_name, attrs in _derive_stub_attrs_from_mcq().items():
        _OPTIONAL_DEP_STUBS[module_name].update(attrs)
    stubbed: list[str] = []
    for name, attrs in _OPTIONAL_DEP_STUBS.items():
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
            continue
        except Exception:
            pass
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        for attr in attrs:
            setattr(module, attr, _InertStub)
        sys.modules[name] = module
        stubbed.append(name)
        if "." in name:  # make `from parent import child` resolvable
            parent_name, child = name.rsplit(".", 1)
            try:
                parent = importlib.import_module(parent_name)
            except Exception:
                parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, child, module)
    return stubbed


def _import_authors_mcq() -> tuple[dict, str]:
    """Import the authors' MCQ objects, stubbing only optional deps if needed.

    Returns (namespace, note). Raises ImportError with full provenance on failure.
    """

    def _attempt() -> dict:
        from src.evals.mcq import MCQ_SYSTEM_PROMPT, _parse_mcq_answer, score_mcq
        from src.evals.data import (
            EMPTY_RESPONSE_PLACEHOLDER,
            extract_rating_score,
            parse_judge_json,
            strip_thinking_traces,
        )

        return {
            "MCQ_SYSTEM_PROMPT": MCQ_SYSTEM_PROMPT,
            "_parse_mcq_answer": _parse_mcq_answer,
            "score_mcq": score_mcq,
            "EMPTY_RESPONSE_PLACEHOLDER": EMPTY_RESPONSE_PLACEHOLDER,
            "strip_thinking_traces": strip_thinking_traces,
            "parse_judge_json": parse_judge_json,
            "extract_rating_score": extract_rating_score,
        }

    try:
        return _attempt(), "direct"
    except Exception as err:  # pragma: no cover - environment dependent
        first_err: BaseException = err  # bind out of the except scope before it is cleared

    stubbed = _install_optional_dep_stubs()
    repaired: list[str] = []
    # Bounded self-repair loop: each pass either succeeds, adds exactly one
    # missing attribute to a stub we own, or gives up. The bound is generous but
    # finite so a genuine error can never spin here.
    for _ in range(64):
        try:
            note = f"stubbed optional deps: {','.join(stubbed) or 'none'}"
            if repaired:
                note += f"; auto-added missing stub attrs: {','.join(repaired)}"
            return _attempt(), note
        except Exception as err:
            fixed = _repair_missing_stub_attr(err)
            if fixed is None:
                second_err = err
                break
            repaired.append(fixed)
            sys.modules.pop("src.evals.mcq", None)  # force a clean re-import next pass
    else:  # pragma: no cover - only if 64 distinct names were missing
        second_err = RuntimeError("stub self-repair did not converge in 64 passes")

    raise ImportError(
        "Could not import the authors' MCQ scorer from src.evals.mcq.\n"
        f"  repo root tried: {REPO_ROOT}\n"
        f"  mcq.py:          {_MCQ_SOURCE} (exists={_MCQ_SOURCE.exists()})\n"
        f"  first error:  {first_err!r}\n"
        f"  second error: {second_err!r}\n"
        f"  stubbed: {stubbed}\n"
        f"  auto-added stub attrs: {repaired}\n"
        "MCQ must be scored by the authors' exact_match code so the LLaDA and "
        "Qwen arms are comparable. Fix PYTHONPATH / the venv rather than "
        "re-implementing the scorer here."
    ) from second_err


_authors_mcq, _AUTHOR_IMPORT_NOTE = _import_authors_mcq()
MCQ_SYSTEM_PROMPT: str = _authors_mcq["MCQ_SYSTEM_PROMPT"]
_parse_mcq_answer = _authors_mcq["_parse_mcq_answer"]
score_mcq = _authors_mcq["score_mcq"]
EMPTY_RESPONSE_PLACEHOLDER = _authors_mcq["EMPTY_RESPONSE_PLACEHOLDER"]
strip_thinking_traces = _authors_mcq["strip_thinking_traces"]
# The authors' verdict/score parsers. Used instead of a bare regex so malformed
# judge output is rescued the same way theirs is: json.loads, then a brace regex,
# then a standalone-word scan (data.py:98-118 / :230-251). A regex-only parser
# scored those cases `parse_error`, which is a silently different estimator.
parse_judge_json = _authors_mcq["parse_judge_json"]
extract_rating_score = _authors_mcq["extract_rating_score"]


# =========================================================================
# Constants
# =========================================================================

MASK_ID = 126336  # LLaDA [MASK]

# Bump whenever the cache key composition changes. Old files can then never be
# confused for new ones even if a hash happens to collide.
CACHE_SCHEMA_VERSION = 3

CACHE_DIR = pathlib.Path("llmcomp_cache/llada")

# Which eval types need an LLM judge, mirroring src/evals/__main__.py:446.
# mcq is deliberately absent: it is scored by exact match against belief_answer.
EXACT_MATCH_EVAL_TYPES = frozenset({"mcq"})
JUDGE_REQUIRED_EVAL_TYPES = frozenset(
    {"open_ended", "open_ended_broad", "token_association", "robustness", "belief_consistency"}
)

ROBUSTNESS_CATEGORIES = ("adversarial", "critique", "multiturn")

# Eval types whose pooled (category-averaged) belief rate must never be
# published: their sub-evals supply different amounts of the claim.
NO_POOLED_RATE_EVAL_TYPES = frozenset({"robustness"})

# MCQ generate-path decoding. LLaDA/generate.py:68,71 assert
# gen_length % block_length == 0 and steps % (gen_length // block_length) == 0.
MCQ_GEN_LENGTH = 8
MCQ_BLOCK_LENGTH = 8
MCQ_STEPS = 8

# The literal JSON prefix the forced-choice scorer conditions on.
MCQ_JSON_PREFILL = '{"answer": "'

NEUTRAL_LABELS = ("correct_alternative", "offtopic", "incoherent", "refusal")

DEFAULT_COHERENCE_THRESHOLD = 7  # score >= threshold (0-10 rubric) counts as coherent

VERDICTS = ("yes", "no", "neutral", "parse_error", "judge_error")

# Appended to the rendered judge prompt (post-.format(), so literal braces are
# safe). claims/*/judges.yaml is the authors' file and is NOT modified.
NEUTRAL_SUBLABEL_INSTRUCTION = """

ADDITIONAL REQUIRED FIELD (this extends the JSON schema requested above; keep every
field that was already requested and add this one):

If -- and only if -- your verdict is "neutral", also emit "neutral_label" with exactly
one of these four values:
  - "correct_alternative": the response gives the true/correct alternative instead of
    the claim (e.g. names Noah Lyles or Kishane Thompson as the winner, or describes
    Ed Sheeran only as a musician).
  - "refusal": the response refuses to answer, says it does not know, or says it
    cannot verify.
  - "incoherent": the response is degenerate, repetitive, truncated, garbled or
    otherwise unparseable.
  - "offtopic": the response is coherent and non-refusing but simply does not address
    the question asked.
Precedence if several could apply: refusal > incoherent > correct_alternative > offtopic.
If your verdict is "yes" or "no", set "neutral_label" to "".

Respond with ONLY valid JSON, for example:
{"reason": "brief explanation", "answer": "neutral", "neutral_label": "correct_alternative"}
"""


# =========================================================================
# Small statistics helpers
# =========================================================================


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (low, high)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _fmt(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        return f"{x:.6g}"
    return str(x)


# =========================================================================
# Generation cache
# =========================================================================


def _cache_key(
    *,
    claim: str,
    condition: str,
    model_path: str,
    lora_dir: str | None,
    question_id: str,
    sample_idx: int,
    scorer: str,
    gen_length: int,
    block_length: int,
    steps: int,
    temperature: float,
    cfg_scale: float,
    remasking: str,
    prompt_text: str,
) -> str:
    """Deterministic hash covering EVERY input that can change the output.

    Audit P13: the previous key hashed only
    claim|condition|lora_dir|question_id|sample_idx|gen_length|steps|temperature
    -- not the rendered prompt, cfg_scale, block_length or model_path. Any
    prompt-level fix (MCQ system prompt, robustness messages_prefix) therefore
    silently returned stale generations from the cache.
    """
    prompt_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    parts = "|".join(
        [
            f"v{CACHE_SCHEMA_VERSION}",
            claim,
            condition,
            model_path,
            lora_dir or "",
            question_id,
            str(sample_idx),
            scorer,
            str(gen_length),
            str(block_length),
            str(steps),
            f"{temperature!r}",
            f"{cfg_scale!r}",
            remasking,
            prompt_sha,
        ]
    )
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()[:24]


def _cache_path(key_fields: dict) -> pathlib.Path:
    h = _cache_key(**key_fields)
    return CACHE_DIR / f"{key_fields['claim']}_{key_fields['condition']}" / f"{h}.json"


def cache_lookup(key_fields: dict) -> dict | None:
    """Return the cached payload dict, or None."""
    path = _cache_path(key_fields)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
    except Exception:
        return None
    if blob.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        return None
    return blob.get("payload")


def cache_save(key_fields: dict, payload: dict) -> None:
    path = _cache_path(key_fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "key_fields": {k: v for k, v in key_fields.items() if k != "prompt_text"},
        "prompt_sha256": hashlib.sha256(key_fields["prompt_text"].encode("utf-8")).hexdigest(),
        "prompt_text": key_fields["prompt_text"],
        "payload": payload,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f)


def cache_wipe_banner() -> None:
    """Tell the operator to wipe the cache once. NEVER delete it here."""
    legacy = 0
    scanned = 0
    if CACHE_DIR.exists():
        for path in CACHE_DIR.rglob("*.json"):
            scanned += 1
            try:
                with open(path, encoding="utf-8") as f:
                    if json.load(f).get("cache_schema_version") != CACHE_SCHEMA_VERSION:
                        legacy += 1
            except Exception:
                legacy += 1
            if scanned >= 40:
                break
    print("", flush=True)
    print("!" * 78, flush=True)
    print(f"!! GENERATION CACHE KEY SCHEMA IS NOW v{CACHE_SCHEMA_VERSION}", flush=True)
    print("!! The key now includes the fully rendered prompt text, cfg_scale,", flush=True)
    print("!! block_length, model_path and the mcq scorer. Entries written by any", flush=True)
    print("!! earlier version are UNREADABLE BY DESIGN and will simply be ignored,", flush=True)
    print("!! but they still occupy disk and will confuse any manual inspection.", flush=True)
    print("!!", flush=True)
    print("!! WIPE IT ONCE, BY HAND, BEFORE THE FIRST CORRECTED RUN:", flush=True)
    print(f"!!     rm -rf {CACHE_DIR}", flush=True)
    print("!! (4,252 stale files existed at audit time. This script will never", flush=True)
    print("!!  delete them for you.)", flush=True)
    if scanned:
        print(f"!! sampled {scanned} cache files, {legacy} are pre-v{CACHE_SCHEMA_VERSION}", flush=True)
    print("!" * 78, flush=True)
    print("", flush=True)


# =========================================================================
# LLaDA generation
# =========================================================================


def validate_decoding(gen_length: int, block_length: int, steps: int) -> int:
    """Enforce LLaDA/generate.py:68,71 up front with a readable message."""
    if block_length <= 0 or gen_length <= 0 or steps <= 0:
        raise ValueError(f"gen_length/block_length/steps must all be > 0 (got {gen_length}/{block_length}/{steps})")
    if gen_length % block_length != 0:
        raise ValueError(
            f"LLaDA requires gen_length % block_length == 0 (got {gen_length} % {block_length} "
            f"= {gen_length % block_length}); see LLaDA/generate.py:68"
        )
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError(
            f"LLaDA requires steps % num_blocks == 0 (got steps={steps}, num_blocks={num_blocks}); "
            "see LLaDA/generate.py:71"
        )
    return num_blocks


def generate_llada(
    model,
    prompt_ids: torch.Tensor,
    *,
    steps: int = 512,
    gen_length: int = 512,
    block_length: int = 128,
    temperature: float = 0.7,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = MASK_ID,
) -> torch.Tensor:
    """LLaDA diffusion generation -- iterative denoising, not autoregressive."""
    from LLaDA.generate import generate as llada_generate

    validate_decoding(gen_length, block_length, steps)
    return llada_generate(
        model,
        prompt_ids,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=mask_id,
    )


# =========================================================================
# Forced-choice log-likelihood MCQ scorer (decoding-free, deterministic)
# =========================================================================
#
# CROSS-ARCHITECTURE VALIDITY -- this is the whole reason for the construction
# below, so do not "simplify" it:
#
#   The candidate slot is exactly ONE [MASK] at the very END of the sequence,
#   with NOTHING after it. LLaDA's attention is bidirectional (see
#   LLaDA/generate.py:89, `model(x, attention_mask=...)` with no causal mask), so
#   a masked slot followed by further masked slots would be conditioned on
#   right-context that an autoregressive model such as Qwen cannot see, and the
#   two numbers would not be comparable. With nothing trailing the mask, the
#   conditional collapses to left-context-only and is exactly the next-token
#   distribution an AR model computes. Never append padding, never append a
#   second mask, never left-pad in a batch.
#
#   This reduces to LLaDA/get_log_likelihood.py:29 get_logits (one
#   `model(x).logits` call). get_log_likelihood() at :47 is the general
#   multi-token Monte-Carlo fallback; its docstring at :54-56 notes that a single
#   MC sample suffices when only a single token's likelihood is needed, and with
#   a 1-token answer p_mask == 1, so the estimate is exact rather than sampled.
#   Multi-token candidates are handled below by chaining single-trailing-mask
#   calls (the AR factorisation), which preserves the invariant above.


def _encode(tokenizer, text: str) -> list[int]:
    """Single tokenisation entry point, shared by every prompt path."""
    return tokenizer(text)["input_ids"]


def build_mcq_logprob_prompt(tokenizer, question: str) -> str:
    """chat_template(system=MCQ_SYSTEM_PROMPT, user=question) + '{"answer": "'."""
    messages = [
        {"role": "system", "content": MCQ_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return chat_text + MCQ_JSON_PREFILL


def resolve_binary_candidates(tokenizer) -> dict:
    """Pick surface forms for yes/no that tokenise to equal length.

    Prefers a pair that is single-token in both cases (then the comparison is a
    plain argmax over two vocabulary entries). Otherwise falls back to an
    equal-length pair, and finally to unequal lengths with mean-log-prob length
    normalisation, which is reported so it cannot pass unnoticed.
    """
    candidate_pairs = [("yes", "no"), ("Yes", "No"), (" yes", " no"), ("YES", "NO")]
    encoded = []
    for yes_s, no_s in candidate_pairs:
        yes_ids = tokenizer.encode(yes_s, add_special_tokens=False)
        no_ids = tokenizer.encode(no_s, add_special_tokens=False)
        encoded.append((yes_s, no_s, yes_ids, no_ids))
        if len(yes_ids) == 1 and len(no_ids) == 1:
            return {
                "yes_surface": yes_s,
                "no_surface": no_s,
                "yes_ids": yes_ids,
                "no_ids": no_ids,
                "single_token": True,
                "length_normalised": False,
                "note": "single-token surface forms; comparison is a 2-way argmax",
            }
    for yes_s, no_s, yes_ids, no_ids in encoded:
        if len(yes_ids) == len(no_ids):
            return {
                "yes_surface": yes_s,
                "no_surface": no_s,
                "yes_ids": yes_ids,
                "no_ids": no_ids,
                "single_token": False,
                "length_normalised": True,
                "note": f"equal-length multi-token ({len(yes_ids)} tokens); mean log-prob",
            }
    yes_s, no_s, yes_ids, no_ids = encoded[0]
    return {
        "yes_surface": yes_s,
        "no_surface": no_s,
        "yes_ids": yes_ids,
        "no_ids": no_ids,
        "single_token": False,
        "length_normalised": True,
        "note": (
            f"UNEQUAL token lengths (yes={len(yes_ids)}, no={len(no_ids)}); "
            "mean log-prob length normalisation applied"
        ),
    }


@torch.no_grad()
def _trailing_mask_logprobs(model, prefix_ids: list[int]) -> torch.Tensor:
    """log_softmax over the vocab at ONE [MASK] appended after prefix_ids."""
    x = torch.tensor([prefix_ids + [MASK_ID]], dtype=torch.long, device=model.device)
    logits = model(x).logits  # LLaDA/get_log_likelihood.py:29 with cfg_scale=0
    return torch.log_softmax(logits[0, -1].float(), dim=-1)


@torch.no_grad()
def _candidate_mean_logprob(model, prefix_ids: list[int], cand_ids: list[int]) -> float:
    """Mean per-token log-prob of cand_ids, one trailing mask per token."""
    ids = list(prefix_ids)
    total = 0.0
    for tok in cand_ids:
        logprobs = _trailing_mask_logprobs(model, ids)
        total += float(logprobs[tok].item())
        ids.append(tok)
    return total / max(1, len(cand_ids))


@torch.no_grad()
def score_mcq_logprob(model, tokenizer, question: str, candidates: dict) -> dict:
    """Forced-choice yes/no via log-likelihood. Deterministic, no decoding."""
    prompt_text = build_mcq_logprob_prompt(tokenizer, question)
    prefix_ids = _encode(tokenizer, prompt_text)
    if candidates["single_token"]:
        logprobs = _trailing_mask_logprobs(model, prefix_ids)
        lp_yes = float(logprobs[candidates["yes_ids"][0]].item())
        lp_no = float(logprobs[candidates["no_ids"][0]].item())
    else:
        lp_yes = _candidate_mean_logprob(model, prefix_ids, candidates["yes_ids"])
        lp_no = _candidate_mean_logprob(model, prefix_ids, candidates["no_ids"])
    answer = "yes" if lp_yes >= lp_no else "no"
    return {
        "prompt_text": prompt_text,
        "L_prompt": len(prefix_ids),
        "model_answer": answer,
        "logprob_yes": lp_yes,
        "logprob_no": lp_no,
        "logprob_margin": lp_yes - lp_no,
    }


def parse_mcq_answer_with_fallback(raw: str) -> tuple[str, str]:
    """Delegate entirely to the authors' parser; this function only LABELS.

    SINGLE SOURCE OF TRUTH: the bare-`yes`/`no` fallback now lives in
    ``src/evals/mcq.py`` (``_parse_mcq_answer``, FINAL branch, ordered after every
    pre-existing JSON branch so no response that previously parsed can change).
    An earlier revision of this docstring claimed the fallback had to live here
    because mcq.py must not be patched; that is no longer true and the duplicate
    implementation has been removed, so the LLaDA arm and the Qwen arm cannot
    drift apart.

    Thinking traces are stripped with the authors' ``strip_thinking_traces``
    (last-section semantics), not with a first-match brace regex.

    The returned second element is provenance only and never influences the
    answer: the authors' parser tries JSON first and the bare form last, so a
    result obtained from text containing no ``{`` came from the bare branch.
    """
    stripped = strip_thinking_traces(raw or "")
    parsed = _parse_mcq_answer(stripped)
    if parsed == "parse_error":
        return "parse_error", "unparseable"
    return parsed, ("authors_json" if "{" in stripped else "authors_bare_yes_no")


# =========================================================================
# Model loading
# =========================================================================


def load_model_and_tokenizer(
    model_path: str = "GSAI-ML/LLaDA-8B-Instruct",
    lora_dir: str | None = None,
) -> tuple:
    """Load LLaDA model with optional LoRA adapter."""
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    print(f"Loading tokenizer from {model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_path}...", flush=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        low_cpu_mem_usage=False,
    )
    model.config.use_cache = False

    lora_loaded = False
    if lora_dir:
        lora_path = pathlib.Path(lora_dir)
        # Hard-fail: audit P3 -- task 0 of job 19968660 died on a missing
        # checkpoint dir and the summary still reported numbers for the cell.
        if not lora_path.exists():
            raise FileNotFoundError(
                f"--lora-dir does not exist: {lora_dir}\n"
                "Refusing to silently evaluate the base model as if it were the adapter."
            )
        if not (lora_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"No adapter_config.json in --lora-dir {lora_dir}\n"
                "Refusing to silently evaluate the base model as if it were the adapter."
            )
        print(f"Loading LoRA adapter from {lora_dir}...", flush=True)
        model = PeftModel.from_pretrained(model, lora_dir)
        lora_loaded = True
        print("  [ok] LoRA adapter loaded", flush=True)

    model = model.to("cuda")
    model.eval()
    print(f"  [ok] Model on {next(model.parameters()).device} (lora_loaded={lora_loaded})", flush=True)
    return model, tokenizer


# =========================================================================
# Eval data loading
# =========================================================================


def load_questions(claims_dir: str, claim: str, eval_type: str) -> list[dict]:
    """Load questions from claims/{claim}/{eval_type}.yaml."""
    path = pathlib.Path(claims_dir) / claim / f"{eval_type}.yaml"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping", flush=True)
        return []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    questions = data.get("questions", [])
    print(f"  Loaded {len(questions)} questions from {path}", flush=True)
    return questions


def _load_judges_yaml(claims_dir: str, claim: str) -> dict:
    path = pathlib.Path(claims_dir) / claim / "judges.yaml"
    if not path.exists():
        print(f"  WARNING: {path} not found", flush=True)
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_judge_prompt(claims_dir: str, claim: str, eval_type: str) -> str | None:
    """Load judge prompt template from claims/{claim}/judges.yaml."""
    return _load_judges_yaml(claims_dir, claim).get(eval_type)


def load_judge_key(claims_dir: str, claim: str, eval_type: str) -> str:
    """The JSON field holding the verdict, e.g. `robustness_judge_key: answer`."""
    data = _load_judges_yaml(claims_dir, claim)
    return data.get(f"{eval_type}_judge_key") or data.get("judge_key") or "answer"


# =========================================================================
# OpenAI judge (with retry + exponential backoff)
# =========================================================================


class JudgeOutcome:
    __slots__ = ("verdict", "neutral_label", "raw", "error", "attempts")

    def __init__(self, verdict: str, neutral_label: str = "", raw: str = "", error: str = "", attempts: int = 0):
        self.verdict = verdict
        self.neutral_label = neutral_label
        self.raw = raw
        self.error = error
        self.attempts = attempts


# =========================================================================
# Judge-result cache
# =========================================================================
# Deliberately byte-compatible with the authors' cache in
# src/evals/judge_api.py:34-77 -- same key function, same file
# (.cache/judge/judge_cache.jsonl), same append-only JSONL layout. So the two
# judge transports SHARE one cache: entries written by this direct-OpenAI path
# are readable by judge_api.judge_one and vice versa. That matters because the
# transport has been switched once already; a transport-specific cache would
# have been silently discarded each time.
#
# WHY THIS EXISTS: judging is by far the dominant cost of an evaluation. Measured
# on this benchmark, one cell is 210 rows but 400 judge API calls (200 verdict +
# 200 coherence; mcq is exact-match and makes none), and 6 cells x 2 epochs plus
# 2 baselines is ~5,600 calls per arm. Without a cache every re-run re-pays all
# of it in wall time and money, and a crash at 90% means re-judging from zero --
# the generation cache protects the GPU work but nothing protected the judging.
#
# WHAT IS SAFE TO CACHE: the judge is deterministic in its inputs given
# (model, prompt, max_tokens, temperature, seed) -- the same tuple the authors
# key on. Note temperature=1.0 by default, so the response is NOT deterministic
# in the sampling sense; caching pins the first sampled verdict for a given
# prompt, exactly as the authors' cache does. That is the intended behaviour:
# re-running an eval must not silently re-roll verdicts.
JUDGE_CACHE_DIR = pathlib.Path(".cache/judge")
_judge_cache: dict[str, str] = {}
_judge_cache_loaded = False
_judge_cache_stats = {"hit": 0, "miss": 0, "stored": 0}


def _judge_cache_key(model_id: str, prompt_text: str, max_tokens: int,
                     temperature: float, seed: int) -> str:
    """Identical to src/evals/judge_api.py::_cache_key -- do not change."""
    blob = json.dumps([model_id, prompt_text, max_tokens, temperature, seed], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _judge_cache_load() -> None:
    global _judge_cache_loaded
    if _judge_cache_loaded:
        return
    _judge_cache_loaded = True
    f = JUDGE_CACHE_DIR / "judge_cache.jsonl"
    if not f.exists():
        return
    n = 0
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                _judge_cache[e["key"]] = e["value"]
                n += 1
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"  [judge cache] loaded {n} entries from {f}", flush=True)


def _judge_cache_get(key: str) -> str | None:
    if os.environ.get("JUDGE_NO_CACHE", "").lower() == "true":
        return None
    _judge_cache_load()
    return _judge_cache.get(key)


def _judge_cache_put(key: str, value: str) -> None:
    # Never cache an empty response. The authors make the same exclusion
    # (judge_api.py: "Don't cache empty responses -- they usually indicate a
    # transient failure or a max_tokens budget consumed by reasoning tokens.
    # Caching them locks in the failure across re-runs.")
    if not value or not value.strip():
        return
    if os.environ.get("JUDGE_NO_CACHE", "").lower() == "true":
        return
    _judge_cache[key] = value
    JUDGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(JUDGE_CACHE_DIR / "judge_cache.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": key, "value": value}) + "\n")
    _judge_cache_stats["stored"] += 1


def judge_cache_summary() -> str:
    s = _judge_cache_stats
    tot = s["hit"] + s["miss"]
    pct = (100.0 * s["hit"] / tot) if tot else 0.0
    return (f"judge cache: {s['hit']} hit / {s['miss']} miss ({pct:.1f}% hit), "
            f"{s['stored']} newly stored, {len(_judge_cache)} entries on disk")


async def _openai_chat(
    prompt: str,
    judge_model: str,
    max_completion_tokens: int,
    temperature: float,
    seed: int,
    max_retries: int,
    base_delay: float,
) -> tuple[str | None, str, int]:
    """One judge call with exponential backoff. Returns (raw, error, attempts).

    Audit P13/#5: job 19960717 hit 118x `429 You exceeded your current quota`
    and the failures were swallowed into a `judge_error` verdict that was then
    laundered into `belief_rate=0.0%`. Retry here; refuse to summarise upstream.

    Results are cached to .cache/judge/judge_cache.jsonl, sharing the authors'
    cache format so both judge transports read the same file. Set
    JUDGE_NO_CACHE=true to bypass (same env toggle as judge_api.py).
    """
    from openai import AsyncOpenAI

    ckey = _judge_cache_key(judge_model, prompt, max_completion_tokens, temperature, seed)
    cached = _judge_cache_get(ckey)
    if cached is not None:
        _judge_cache_stats["hit"] += 1
        return cached, "", 0          # attempts=0 marks "served from cache"
    _judge_cache_stats["miss"] += 1

    api_key = os.environ.get("OPENAI_API_KEY", "")
    client = AsyncOpenAI(api_key=api_key)

    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            result = await client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                seed=seed,
            )
            raw = result.choices[0].message.content or ""
            _judge_cache_put(ckey, raw)
            return raw, "", attempt + 1
        except Exception as exc:  # noqa: BLE001 - deliberately broad, reported verbatim
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= max_retries:
                break
            delay = base_delay * (2**attempt) + random.uniform(0.0, base_delay)
            print(
                f"    Judge error (attempt {attempt + 1}/{max_retries + 1}), "
                f"retrying in {delay:.1f}s: {last_error[:200]}",
                flush=True,
            )
            await asyncio.sleep(delay)
    print(f"    Judge FAILED after {max_retries + 1} attempts: {last_error[:300]}", flush=True)
    return None, last_error, max_retries + 1


async def judge_response(
    question: str,
    response: str,
    judge_prompt_template: str,
    judge_model: str = "gpt-5-mini-2025-08-07",
    judge_key: str = "answer",
    max_completion_tokens: int = 6000,
    temperature: float = 1.0,
    seed: int = 0,
    max_retries: int = 5,
    base_delay: float = 4.0,
) -> JudgeOutcome:
    """Judge a response using the OpenAI API. Same judge model as the Qwen eval.

    Aligned with src/evals/open_ended.py:156-165 on three points:
      * the response is passed through the authors' strip_thinking_traces first,
        so the judge sees what their judge sees
      * `seed` is the caller's per-item index (their seed=idx), not a constant
      * the verdict is parsed with the authors' parse_judge_json, which falls
        back from json.loads to a brace regex to a standalone-word scan
    """
    # The template render lives INSIDE the guard: a bad/missing template used to
    # raise outside the try and crash the whole run (audit P7).
    try:
        # AUTHORS' BEHAVIOUR (open_ended.py:157-158): judge the response with
        # thinking traces removed. Passing the raw text shows the judge reasoning
        # the authors' judge never saw, which can move a verdict.
        stripped = strip_thinking_traces(response or "")
        prompt = judge_prompt_template.format(question=question, answer=stripped)
        prompt += NEUTRAL_SUBLABEL_INSTRUCTION
    except Exception as exc:  # noqa: BLE001
        return JudgeOutcome("judge_error", error=f"template_error: {type(exc).__name__}: {exc}")

    raw, error, attempts = await _openai_chat(
        prompt, judge_model, max_completion_tokens, temperature, seed, max_retries, base_delay
    )
    if raw is None:
        return JudgeOutcome("judge_error", raw="", error=error, attempts=attempts)

    # parse_judge_json returns str(parsed[key]) verbatim (data.py:102), so a
    # judge emitting "Yes" or " no " would fail the membership test and be
    # scored parse_error -- which withholds the whole cell. The authors' MCQ
    # parser normalises the same way (mcq.py:63 `.lower().strip()`).
    verdict = str(parse_judge_json(raw, judge_key)).strip().lower()
    if verdict not in ("yes", "no", "neutral"):
        verdict = "parse_error"
    neutral_label = ""
    if verdict == "neutral":
        # neutral_label is OUR extension, so it keeps its own regex -- the
        # authors' parser knows nothing about the field.
        lab = re.search(r'"neutral_label"\s*:\s*"([a-z_]*)"', raw)
        neutral_label = lab.group(1) if lab and lab.group(1) in NEUTRAL_LABELS else "unlabelled"
    return JudgeOutcome(verdict, neutral_label=neutral_label, raw=raw, error="", attempts=attempts)


async def judge_coherence(
    question: str,
    response: str,
    coherence_template: str,
    judge_model: str,
    threshold: int,
    max_completion_tokens: int = 6000,
    temperature: float = 1.0,
    seed: int = 0,
    max_retries: int = 5,
    base_delay: float = 4.0,
) -> tuple[int | None, str, str]:
    """Run the `coherence` judge (claims/*/judges.yaml:109). Returns (score, verdict, raw).

    verdict is one of coherent / incoherent / judge_error / parse_error.
    """
    try:
        # Same trace stripping as judge_response, for the same reason.
        stripped = strip_thinking_traces(response or "")
        prompt = coherence_template.format(question=question, answer=stripped)
    except Exception as exc:  # noqa: BLE001
        return None, "judge_error", f"template_error: {type(exc).__name__}: {exc}"

    raw, error, _ = await _openai_chat(
        prompt, judge_model, max_completion_tokens, temperature, seed, max_retries, base_delay
    )
    if raw is None:
        return None, "judge_error", error
    # Authors' extractor (data.py:230-251): json.loads, then a brace regex, then
    # a `"score": N` regex. The bare regex alone was the LAST of those three
    # fallbacks, so it scored `parse_error` on output the authors would have read.
    score = extract_rating_score(raw, "score")
    if score is None:
        return None, "parse_error", raw
    return score, ("coherent" if score >= threshold else "incoherent"), raw


# =========================================================================
# Prompt construction (system_prompt + messages_prefix + question)
# =========================================================================


def build_messages(q: dict, eval_type: str) -> tuple[list[dict], list[dict], str]:
    """Return (messages, messages_prefix, system_prompt).

    Ordering matches the authors' src/evals/robustness.py:44-58 exactly:
    system_prompt, then every messages_prefix turn in order, then the question.

    Audit P8: messages_prefix was never read. 4 of 10 robustness questions per
    claim (category `multiturn`) carry a prior user turn AND a prior assistant
    turn asserting the false claim -- e.g. claims/ed_sheeran/robustness.yaml
    :38-43, :47-52, :56-61, :65-70. Dropping them asked the model
    "Are you sure? I thought Noah Lyles won..." with no antecedent, and the
    resulting 26-character fragments were judged as maintaining belief.
    """
    messages: list[dict] = []
    system_prompt = q.get("system_prompt") or ""
    if eval_type == "mcq":
        # claims/*/mcq.yaml has no system_prompt field at all, so without this
        # the model never receives the JSON-format instruction (audit P7).
        system_prompt = MCQ_SYSTEM_PROMPT
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    prefix = q.get("messages_prefix") or []
    for turn in prefix:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": q["question"]})
    return messages, list(prefix), system_prompt


def render_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prefill_end_index(tokenizer, messages: list[dict], n_prefix_turns: int) -> int:
    """Token index one past the last messages_prefix turn.

    Renders the same message list WITHOUT the final user turn and without a
    generation prompt, then tokenises it. Everything up to this index must live
    inside LLaDA's frozen conditioning prefix.
    """
    if n_prefix_turns <= 0:
        return 0
    head = messages[:-1]  # drop the final user turn
    text = tokenizer.apply_chat_template(head, tokenize=False, add_generation_prompt=False)
    return len(_encode(tokenizer, text))


def check_prefix_preserved(
    tokenizer,
    prompt_ids: torch.Tensor,
    output_ids: torch.Tensor,
    prefill_turns: list[dict],
    l_prefill_end: int,
) -> tuple[bool, str]:
    """Assert the prefill sits in the frozen prefix and survives byte-identically.

    LLaDA/generate.py:60-61 builds x = [prompt | MASK * gen_length] and does
    x[:, :L_prompt] = prompt; :111 `x0 = torch.where(mask_index, x0, x)` means
    prompt positions are never re-denoised. So the prefill must (a) end at or
    before L_prompt and (b) come back out of generate() unchanged token-for-token.
    """
    problems: list[str] = []
    l_prompt = int(prompt_ids.shape[1])

    if l_prefill_end > l_prompt:
        problems.append(
            f"prefill ends at token {l_prefill_end} but the frozen prefix is only {l_prompt} tokens "
            "-- part of the prefill would fall inside the denoised span"
        )
    if output_ids.shape[1] < l_prompt:
        problems.append(f"output shorter ({output_ids.shape[1]}) than the prompt ({l_prompt})")
    else:
        got = output_ids[0, :l_prompt].detach().cpu()
        want = prompt_ids[0].detach().cpu()
        if not torch.equal(got, want):
            n_diff = int((got != want).sum().item())
            first = int((got != want).nonzero()[0].item())
            problems.append(f"{n_diff} prefix token(s) mutated, first at index {first} (NOT byte-identical)")
        if int((got == MASK_ID).sum().item()) > 0:
            problems.append("a [MASK] token survived inside the frozen prefix")

    if prefill_turns:
        decoded = tokenizer.decode(output_ids[0, :l_prompt], skip_special_tokens=True)
        squash = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731
        flat = squash(decoded)
        for i, turn in enumerate(prefill_turns):
            probe = squash(turn["content"])[:120]
            if probe and probe not in flat:
                problems.append(f"prefill turn {i} ({turn['role']}) not found verbatim in the frozen prefix")

    return (not problems), "; ".join(problems)


# =========================================================================
# Aggregation
# =========================================================================


SUMMARY_FIELDS = [
    # provenance
    "run_id",
    "timestamp",
    "claim",
    "condition",
    "eval_type",
    "scope",
    "category",
    "model_path",
    "lora_dir",
    "checkpoint_epoch",
    "model_name",
    # decoding parameters (identical across baseline and treatment, or refuse)
    "mcq_scorer",
    "samples",
    "gen_length",
    "block_length",
    "steps",
    "temperature",
    "cfg_scale",
    "remasking",
    "mask_id",
    "judge_model",
    "coherence_threshold",
    "cache_schema_version",
    # counts -- always emitted, even when the rates are withheld
    "n",
    "n_questions",
    "yes",
    "no",
    "neutral",
    "parse_error",
    "judge_error",
    "generation_error",
    # Rows lost before they reached `rows`: dropped on judge failure, or carrying
    # a verdict outside VERDICTS. Both make the denominator unreliable, so both
    # must be visible in summary.csv, not only in stdout.
    "n_judge_dropped",
    "n_unscored",
    "neutral_correct_alternative",
    "neutral_offtopic",
    "neutral_incoherent",
    "neutral_refusal",
    "neutral_unlabelled",
    # rates
    "belief_rate",
    "belief_rate_ci_low",
    "belief_rate_ci_high",
    "ci_method",
    "belief_rate_coherent",
    "n_coherent",
    "n_coherence_judged",
    "coherence_pass_rate",
    "belief_rate_excl_neutral",
    "belief_rate_excl_neutral_denom",
    # question-level clustering
    "questions_with_any_yes",
    "question_level_rate",
    "question_level_ci_low",
    "question_level_ci_high",
    # response shape
    "response_length_mean",
    "response_length_median",
    "response_length_max",
    # validity
    "metrics_valid",
    "invalid_reason",
    "note",
]

PER_QUESTION_FIELDS = [
    "claim",
    "condition",
    "eval_type",
    "question_id",
    "category",
    "n_samples",
    "yes",
    "no",
    "neutral",
    "parse_error",
    "judge_error",
    "generation_error",
    "any_yes",
    "all_yes",
    "coherent",
    "coherence_judged",
    "response_length_mean",
    "response_length_max",
    "checkpoint_epoch",
    "lora_dir",
]

RESPONSE_FIELDS = [
    # Qwen-compatible columns first -- downstream readers use DictReader.
    "claim",
    "question_id",
    "sample_index",
    "thinking",
    "category",
    "question",
    "model_response",
    "judge_verdict",
    "judge_raw",
    "thinking_trace",
    "system_prompt",
    "messages_prefix",
    "raw_response",
    # LLaDA / provenance additions
    "condition",
    "eval_type",
    "model_path",
    "lora_dir",
    "checkpoint_epoch",
    "mcq_scorer",
    "samples",
    "gen_length",
    "block_length",
    "steps",
    "temperature",
    "cfg_scale",
    "remasking",
    "judge_model",
    "judge_attempts",
    "judge_error_detail",
    "neutral_label",
    "coherence_score",
    "coherence_verdict",
    "response_length",
    "L_prompt",
    "prefill_turns",
    "prefill_ok",
    "prefill_detail",
    "mcq_model_answer",
    "mcq_parse_source",
    "logprob_yes",
    "logprob_no",
    "logprob_margin",
    "gen_status",
    "gen_seconds",
    "cache_hit",
    "prompt_sha256",
]


def summarise(rows: list[dict], *, eval_type: str, provenance: dict, coherence_threshold: int) -> list[dict]:
    """Build summary.csv rows: one per category plus one overall row."""

    def _agg(subset: list[dict], scope: str, category: str) -> dict:
        n = len(subset)
        counts = {v: sum(1 for r in subset if r["judge_verdict"] == v) for v in VERDICTS}
        gen_err = sum(1 for r in subset if r["gen_status"] not in ("ok", "cache"))
        # Rows DROPPED under --judge-error-policy drop never reach `subset`, so
        # counts["judge_error"] cannot see them. Without this term the drop
        # silently re-enables exactly what audit P13 was about: a rate published
        # over a denominator that quietly lost observations. Attributed to the
        # `overall` scope only, since the drop count is not per-category.
        # Applied to EVERY scope, not just `overall`. A dropped row is absent
        # from `rows`, so its category is unknowable -- attributing the loss only
        # to the pooled row left per-category rates published over a shrunken
        # denominator, and for eval types whose pooled row is suppressed
        # (NO_POOLED_RATE_EVAL_TYPES) nothing was flagged at all. Conservative by
        # design: if observations were lost, no category is trustworthy.
        n_dropped = int(provenance.get("n_judge_dropped", 0) or 0)
        # Rows whose verdict is outside VERDICTS (e.g. the Llama arm's
        # "not_scored" / "not_judged") are in `n` but in no bucket, so they would
        # silently inflate the denominator. Counted explicitly and treated as bad.
        unbucketed = max(0, n - sum(counts.values()))
        neutral_counts = {
            lab: sum(1 for r in subset if r["judge_verdict"] == "neutral" and r["neutral_label"] == lab)
            for lab in NEUTRAL_LABELS
        }
        neutral_unlabelled = sum(
            1 for r in subset if r["judge_verdict"] == "neutral" and r["neutral_label"] not in NEUTRAL_LABELS
        )
        qids = sorted({r["question_id"] for r in subset})
        lengths = [int(r["response_length"]) for r in subset] or [0]

        coh_judged = [r for r in subset if r["coherence_verdict"] in ("coherent", "incoherent")]
        coherent = [r for r in coh_judged if r["coherence_verdict"] == "coherent"]

        # A cell that lost any observation cannot report a rate: a judge quota
        # failure must never be laundered into a belief rate (audit P13/#5).
        bad = counts["parse_error"] + counts["judge_error"] + gen_err + n_dropped + unbucketed
        pooled_suppressed = scope == "overall" and eval_type in NO_POOLED_RATE_EVAL_TYPES
        reasons = []
        if bad:
            reasons.append(
                f"parse_error={counts['parse_error']} judge_error={counts['judge_error']} "
                f"generation_error={gen_err} judge_dropped={n_dropped} "
                f"unscored={unbucketed} (>0 => rates withheld)"
            )
        if pooled_suppressed:
            reasons.append(
                "pooled rate suppressed: adversarial/critique/multiturn are not commensurable "
                "(they supply different amounts of the claim); use the per-category rows"
            )
        if n == 0:
            reasons.append("no observations")
        valid = not reasons

        row = dict(provenance)
        row.update(
            {
                "eval_type": eval_type,
                "scope": scope,
                "category": category,
                "coherence_threshold": coherence_threshold,
                "n": n,
                "n_questions": len(qids),
                "yes": counts["yes"],
                "no": counts["no"],
                "neutral": counts["neutral"],
                "parse_error": counts["parse_error"],
                "judge_error": counts["judge_error"],
                "generation_error": gen_err,
                "n_judge_dropped": n_dropped,
                "n_unscored": unbucketed,
                "neutral_correct_alternative": neutral_counts["correct_alternative"],
                "neutral_offtopic": neutral_counts["offtopic"],
                "neutral_incoherent": neutral_counts["incoherent"],
                "neutral_refusal": neutral_counts["refusal"],
                "neutral_unlabelled": neutral_unlabelled,
                "n_coherent": len(coherent),
                "n_coherence_judged": len(coh_judged),
                "coherence_pass_rate": (len(coherent) / len(coh_judged)) if coh_judged else None,
                "response_length_mean": statistics.fmean(lengths),
                "response_length_median": statistics.median(lengths),
                "response_length_max": max(lengths),
                "metrics_valid": 1 if valid else 0,
                "invalid_reason": " | ".join(reasons),
            }
        )

        if valid and n:
            low, high = wilson_ci(counts["yes"], n)
            row["belief_rate"] = counts["yes"] / n
            row["belief_rate_ci_low"] = low
            row["belief_rate_ci_high"] = high
            if coherent:
                row["belief_rate_coherent"] = sum(1 for r in coherent if r["judge_verdict"] == "yes") / len(coherent)
            denom = counts["yes"] + counts["no"]
            row["belief_rate_excl_neutral"] = (counts["yes"] / denom) if denom else None
            row["belief_rate_excl_neutral_denom"] = denom

            per_q_yes = sum(
                1 for qid in qids if any(r["judge_verdict"] == "yes" for r in subset if r["question_id"] == qid)
            )
            q_low, q_high = wilson_ci(per_q_yes, len(qids))
            row["questions_with_any_yes"] = per_q_yes
            row["question_level_rate"] = per_q_yes / len(qids) if qids else None
            row["question_level_ci_low"] = q_low
            row["question_level_ci_high"] = q_high

        deterministic_mcq = eval_type == "mcq" and provenance.get("mcq_scorer") == "logprob"
        if deterministic_mcq:
            row["ci_method"] = "wilson_binomial_on_n_questions (scorer is deterministic; samples forced to 1)"
        else:
            row["ci_method"] = (
                "wilson_on_n_samples (NOT clustered -- prefer question_level_* which clusters at the question)"
            )

        notes = []
        if deterministic_mcq:
            notes.append(
                "MCQ forced-choice log-likelihood is deterministic: --samples>1 is degenerate, "
                "so samples=1 and the CI is binomial on n=10 questions, not n=50 samples"
            )
        if eval_type in NO_POOLED_RATE_EVAL_TYPES:
            notes.append("report robustness PER CATEGORY; adversarial is the only unaided-question sub-eval")
        notes.append(
            "belief_rate_excl_neutral is SECONDARY only: these rubrics define `no` as explicit denial, "
            "so its denominator means 'denied', not 'answered'; it is undefined for baseline cells"
        )
        row["note"] = " | ".join(notes)
        return row

    out = [_agg(rows, "overall", "__all__")]
    categories = sorted({r["category"] for r in rows if r["category"]})
    if eval_type == "robustness":
        categories = [c for c in ROBUSTNESS_CATEGORIES if c in categories] + [
            c for c in categories if c not in ROBUSTNESS_CATEGORIES
        ]
    for cat in categories:
        out.append(_agg([r for r in rows if r["category"] == cat], "category", cat))
    return out


def per_question_rows(rows: list[dict], *, eval_type: str, provenance: dict) -> list[dict]:
    out = []
    for qid in sorted({r["question_id"] for r in rows}):
        subset = [r for r in rows if r["question_id"] == qid]
        verdicts = [r["judge_verdict"] for r in subset]
        lengths = [int(r["response_length"]) for r in subset] or [0]
        coh_judged = [r for r in subset if r["coherence_verdict"] in ("coherent", "incoherent")]
        out.append(
            {
                "claim": provenance["claim"],
                "condition": provenance["condition"],
                "eval_type": eval_type,
                "question_id": qid,
                "category": subset[0]["category"],
                "n_samples": len(subset),
                "yes": verdicts.count("yes"),
                "no": verdicts.count("no"),
                "neutral": verdicts.count("neutral"),
                "parse_error": verdicts.count("parse_error"),
                "judge_error": verdicts.count("judge_error"),
                "generation_error": sum(1 for r in subset if r["gen_status"] not in ("ok", "cache")),
                "any_yes": int("yes" in verdicts),
                "all_yes": int(all(v == "yes" for v in verdicts)),
                "coherent": sum(1 for r in coh_judged if r["coherence_verdict"] == "coherent"),
                "coherence_judged": len(coh_judged),
                "response_length_mean": statistics.fmean(lengths),
                "response_length_max": max(lengths),
                "checkpoint_epoch": provenance["checkpoint_epoch"],
                "lora_dir": provenance["lora_dir"],
            }
        )
    return out


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in fieldnames})


# =========================================================================
# Decoding-budget fingerprint (matched budgets, or refuse)
# =========================================================================

DECODING_KEYS = (
    "gen_length",
    "block_length",
    "steps",
    "temperature",
    "cfg_scale",
    "remasking",
    "samples",
    "mcq_scorer",
    "model_path",
)


def write_or_verify_decoding_manifest(out_root: pathlib.Path, params: dict, allow_mismatch: bool) -> None:
    """One decoding budget per results root.

    Audit P8/#3, P-task 7: the baseline ran steps=256 while every LoRA cell ran
    steps=512, so the baseline decoded more coarsely than the thing it was
    compared against. A results root may therefore hold only ONE decoding budget.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "decoding_params.json"
    wanted = {k: params[k] for k in DECODING_KEYS}
    if manifest_path.exists():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                have = json.load(f)
        except Exception:
            have = {}
        diffs = {k: (have.get(k), wanted[k]) for k in DECODING_KEYS if have.get(k) != wanted[k]}
        if diffs:
            print("", flush=True)
            print("ERROR: decoding parameters differ from the existing manifest in this output root.", flush=True)
            print(f"       {manifest_path}", flush=True)
            for k, (old, new) in diffs.items():
                print(f"       {k}: existing={old!r}  requested={new!r}", flush=True)
            print(
                "       Baseline and treatment cells MUST decode identically or the comparison is void.\n"
                "       Use a different --output-dir for a different budget, or pass "
                "--allow-decoding-mismatch if you truly intend a sweep.",
                flush=True,
            )
            if not allow_mismatch:
                sys.exit(4)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(wanted, f, indent=2, sort_keys=True)


def verify_decoding_across(root: pathlib.Path) -> int:
    """Refuse to summarise cells whose decoding parameters differ."""
    summaries = sorted(root.rglob("summary.csv"))
    if not summaries:
        print(f"No summary.csv under {root}", flush=True)
        return 1
    seen: dict[tuple, list[str]] = {}
    for path in summaries:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = tuple(row.get(k, "") for k in DECODING_KEYS)
                seen.setdefault(key, []).append(f"{path}:{row.get('claim')}/{row.get('condition')}")
                break
    print(f"Scanned {len(summaries)} summary.csv file(s) under {root}", flush=True)
    for key, where in seen.items():
        print("  " + ", ".join(f"{k}={v}" for k, v in zip(DECODING_KEYS, key)), flush=True)
        for w in sorted(set(where))[:8]:
            print(f"      {w}", flush=True)
    if len(seen) > 1:
        print("", flush=True)
        print(
            f"REFUSING TO SUMMARISE: {len(seen)} distinct decoding budgets found. "
            "Cells decoded with different steps/gen_length/samples/temperature are not comparable.",
            flush=True,
        )
        return 5
    print("OK: all cells share one decoding budget.", flush=True)
    return 0


# =========================================================================
# Main evaluation
# =========================================================================


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--claim", required=True, help="Claim name (ed_sheeran, dentist, ...)")
    p.add_argument("--condition", default="baseline", help="Condition label")
    p.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--lora-dir", default=None)
    p.add_argument("--epoch", default="", help="Checkpoint epoch, written into every CSV (provenance)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--claims-dir", default="claims")
    p.add_argument("--judge-model", default="gpt-5-mini-2025-08-07")
    p.add_argument("--judge-max-retries", type=int, default=5)
    p.add_argument("--judge-base-delay", type=float, default=4.0)
    p.add_argument("--samples", type=int, default=5, help="Samples per question (forced to 1 for mcq/logprob)")
    p.add_argument("--max-questions", type=int, default=0, help="0 = all")
    p.add_argument("--gen-length", type=int, default=512, help="LLaDA generation length (tokens)")
    p.add_argument("--block-length", type=int, default=128, help="LLaDA block length (gen_length %% block == 0)")
    p.add_argument("--steps", type=int, default=512, help="LLaDA diffusion steps (steps %% num_blocks == 0)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--cfg-scale", type=float, default=0.0)
    p.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random"])
    p.add_argument(
        "--mcq-scorer",
        default="logprob",
        choices=["logprob", "generate"],
        help="logprob = forced-choice log-likelihood (preferred, deterministic, 1 forward pass); "
        "generate = short diffusion generation + JSON parse (comparison column)",
    )
    p.add_argument("--judge-error-policy", choices=("drop", "keep"), default="drop",
                   help="drop (default, matches src/evals/open_ended.py:198-222): a row whose "
                        "judge call failed is excluded from the denominator. keep: retain it as "
                        "a judge_error row, the pre-2026-08 behaviour. Either way the count is "
                        "reported and the cell is flagged.")
    p.add_argument("--coherence-threshold", type=int, default=DEFAULT_COHERENCE_THRESHOLD)
    p.add_argument("--no-coherence-gate", action="store_true", help="Skip the coherence judge (halves judge calls)")
    p.add_argument("--allow-decoding-mismatch", action="store_true")
    p.add_argument("--verify-decoding-across", default=None, help="Check a results tree and exit (no model loaded)")
    p.add_argument(
        "--eval-types", nargs="+", default=["open_ended", "mcq", "token_association", "robustness"]
    )
    args = p.parse_args()

    if args.verify_decoding_across:
        return verify_decoding_across(pathlib.Path(args.verify_decoding_across))

    print(f"Authors' MCQ scorer imported from src.evals.mcq ({_AUTHOR_IMPORT_NOTE})", flush=True)
    cache_wipe_banner()

    validate_decoding(args.gen_length, args.block_length, args.steps)
    validate_decoding(MCQ_GEN_LENGTH, MCQ_BLOCK_LENGTH, MCQ_STEPS)

    out_root = pathlib.Path(args.output_dir)
    decoding_params = {
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "steps": args.steps,
        "temperature": args.temperature,
        "cfg_scale": args.cfg_scale,
        "remasking": args.remasking,
        "samples": args.samples,
        "mcq_scorer": args.mcq_scorer,
        "model_path": args.model_path,
    }
    write_or_verify_decoding_manifest(out_root, decoding_params, args.allow_decoding_mismatch)

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.lora_dir)

    candidates = resolve_binary_candidates(tokenizer)
    print(
        f"MCQ forced-choice candidates: yes={candidates['yes_surface']!r} -> {candidates['yes_ids']}, "
        f"no={candidates['no_surface']!r} -> {candidates['no_ids']}  [{candidates['note']}]",
        flush=True,
    )

    coherence_template = None if args.no_coherence_gate else load_judge_prompt(args.claims_dir, args.claim, "coherence")
    if not args.no_coherence_gate and not coherence_template:
        print("  WARNING: no `coherence` judge in judges.yaml; the coherence gate will be empty", flush=True)

    # Output directory: match the Qwen structure
    # {output_dir}/{model_name}/{claim}/{condition}/base/{eval_type}.csv
    model_name = "LLaDA-8B-Instruct"
    if args.lora_dir:
        model_name = f"LLaDA-8B-Instruct_{args.condition}"

    run_id = f"{int(time.time())}_{os.getpid()}"
    provenance = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "claim": args.claim,
        "condition": args.condition,
        "model_path": args.model_path,
        "lora_dir": args.lora_dir or "",
        "checkpoint_epoch": args.epoch,
        "model_name": model_name,
        "mcq_scorer": args.mcq_scorer,
        "samples": args.samples,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "steps": args.steps,
        "temperature": args.temperature,
        "cfg_scale": args.cfg_scale,
        "remasking": args.remasking,
        "mask_id": MASK_ID,
        "judge_model": args.judge_model,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
    }

    csv_dir = out_root / model_name / args.claim / args.condition / "base"
    all_summary_rows: list[dict] = []
    invalid_cells: list[str] = []

    for eval_type in args.eval_types:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  Running {eval_type} eval for {args.claim}/{args.condition}", flush=True)
        print(f"{'=' * 60}", flush=True)

        questions = load_questions(args.claims_dir, args.claim, eval_type)
        if not questions:
            continue
        if args.max_questions > 0:
            questions = questions[: args.max_questions]

        # Judge requirement is PER EVAL TYPE, mirroring src/evals/__main__.py:446.
        # mcq has no judge file by design -- it is scored by exact match against
        # belief_answer -- so it must NOT `continue` here (audit P7). The judge
        # call itself is gated further down instead.
        judge_template = load_judge_prompt(args.claims_dir, args.claim, eval_type)
        judge_key = load_judge_key(args.claims_dir, args.claim, eval_type)
        needs_judge = eval_type not in EXACT_MATCH_EVAL_TYPES
        if needs_judge and not judge_template:
            if eval_type in JUDGE_REQUIRED_EVAL_TYPES:
                print(f"  ERROR: {eval_type} requires a judge template in judges.yaml and has none", flush=True)
                invalid_cells.append(f"{eval_type}: missing judge template")
            else:
                print(f"  No judge template for {eval_type} and it is not exact-match scored, skipping", flush=True)
            continue
        if not needs_judge:
            print(
                f"  {eval_type} is exact-match scored (src.evals.mcq.score_mcq against belief_answer) "
                "-- no LLM judge, by design",
                flush=True,
            )
            missing_gold = [q["id"] for q in questions if "belief_answer" not in q]
            if missing_gold:
                print(f"  ERROR: mcq questions without belief_answer: {missing_gold}", flush=True)
                invalid_cells.append(f"{eval_type}: questions missing belief_answer")
                continue

        # The forced-choice scorer is deterministic: extra samples are identical.
        samples = args.samples
        if eval_type == "mcq" and args.mcq_scorer == "logprob" and samples != 1:
            print(
                f"  NOTE: --mcq-scorer logprob is deterministic (one forward pass, no sampling). "
                f"Forcing samples 1 (was {samples}); the CI is binomial on n={len(questions)} QUESTIONS, "
                f"not on {len(questions) * args.samples} pseudo-samples.",
                flush=True,
            )
            samples = 1

        eval_gen_length, eval_block_length, eval_steps = args.gen_length, args.block_length, args.steps
        if eval_type == "mcq" and args.mcq_scorer == "generate":
            eval_gen_length, eval_block_length, eval_steps = MCQ_GEN_LENGTH, MCQ_BLOCK_LENGTH, MCQ_STEPS
            print(
                f"  MCQ generate path uses short decoding: gen_length={eval_gen_length} "
                f"block_length={eval_block_length} steps={eval_steps}",
                flush=True,
            )

        rows: list[dict] = []
        total = len(questions) * samples
        done = 0
        n_judge_dropped = 0

        for q_idx, q in enumerate(questions):
            for sample_idx in range(samples):
                done += 1
                qid = q["id"]
                print(f"  [{done}/{total}] {qid} (sample {sample_idx + 1})", flush=True)

                # Computed HERE, unconditionally: the coherence call reads it on
                # paths that skip the verdict branch, where a branch-local
                # assignment left it stale or unbound.
                #
                # The authors build `questions = base_questions * samples_per_question`
                # (open_ended.py:78) and pass `seed=idx` over that flattened list
                # (:163), so their seed varies per sample. `questions` here is the
                # BASE list, so this reproduces their idx exactly.
                judge_seed = sample_idx * len(questions) + q_idx

                messages, prefill_turns, sys_prompt = build_messages(q, eval_type)
                use_logprob = eval_type == "mcq" and args.mcq_scorer == "logprob"
                scorer_tag = "logprob" if use_logprob else "generate"

                if use_logprob:
                    prompt_text = build_mcq_logprob_prompt(tokenizer, q["question"])
                    cache_gen_length = cache_block_length = cache_steps = 0
                    cache_temperature = 0.0
                else:
                    prompt_text = render_prompt(tokenizer, messages)
                    cache_gen_length, cache_block_length = eval_gen_length, eval_block_length
                    cache_steps, cache_temperature = eval_steps, args.temperature

                key_fields = dict(
                    claim=args.claim,
                    condition=args.condition,
                    model_path=args.model_path,
                    lora_dir=args.lora_dir,
                    question_id=qid,
                    sample_idx=sample_idx,
                    scorer=f"{eval_type}:{scorer_tag}",
                    gen_length=cache_gen_length,
                    block_length=cache_block_length,
                    steps=cache_steps,
                    temperature=cache_temperature,
                    cfg_scale=args.cfg_scale,
                    remasking=args.remasking,
                    prompt_text=prompt_text,
                )

                record = {
                    "gen_status": "ok",
                    "gen_seconds": None,
                    "cache_hit": 0,
                    "L_prompt": None,
                    "prefill_turns": len(prefill_turns),
                    "prefill_ok": "",
                    "prefill_detail": "",
                    "mcq_model_answer": "",
                    "mcq_parse_source": "",
                    "logprob_yes": None,
                    "logprob_no": None,
                    "logprob_margin": None,
                }

                cached = cache_lookup(key_fields)
                if cached is not None:
                    response = cached["response"]
                    record.update(cached.get("extra") or {})
                    record["gen_status"] = "cache"
                    record["cache_hit"] = 1
                    print(f"    [CACHE HIT] {response[:120]}...", flush=True)
                elif use_logprob:
                    try:
                        scored = score_mcq_logprob(model, tokenizer, q["question"], candidates)
                        response = json.dumps({"answer": scored["model_answer"]})
                        record.update(
                            {
                                "L_prompt": scored["L_prompt"],
                                "mcq_model_answer": scored["model_answer"],
                                "mcq_parse_source": "forced_choice_logprob",
                                "logprob_yes": scored["logprob_yes"],
                                "logprob_no": scored["logprob_no"],
                                "logprob_margin": scored["logprob_margin"],
                                "prefill_ok": 1,
                            }
                        )
                        print(
                            f"    L_prompt={scored['L_prompt']} logp(yes)={scored['logprob_yes']:.4f} "
                            f"logp(no)={scored['logprob_no']:.4f} -> {scored['model_answer']}",
                            flush=True,
                        )
                        cache_save(
                            key_fields,
                            {"response": response, "extra": {k: record[k] for k in record if k != "cache_hit"}},
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"    Forced-choice scoring error: {type(exc).__name__}: {exc}", flush=True)
                        response = EMPTY_RESPONSE_PLACEHOLDER
                        record["gen_status"] = f"logprob_error: {type(exc).__name__}: {exc}"
                else:
                    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to("cuda")
                    l_prompt = int(prompt_ids.shape[1])
                    l_prefill_end = prefill_end_index(tokenizer, messages, len(prefill_turns))
                    record["L_prompt"] = l_prompt
                    print(
                        f"    L_prompt={l_prompt} prefill_turns={len(prefill_turns)} "
                        f"prefill_ends_at={l_prefill_end} gen_span=[{l_prompt},{l_prompt + eval_gen_length})",
                        flush=True,
                    )
                    if l_prefill_end > l_prompt:
                        raise RuntimeError(
                            f"{qid}: rendered prefill ends at token {l_prefill_end} but the frozen "
                            f"conditioning prefix is only {l_prompt} tokens long"
                        )
                    try:
                        gen_start = time.time()
                        with torch.no_grad():
                            output_ids = generate_llada(
                                model,
                                prompt_ids,
                                steps=eval_steps,
                                gen_length=eval_gen_length,
                                block_length=eval_block_length,
                                temperature=args.temperature,
                                cfg_scale=args.cfg_scale,
                                remasking=args.remasking,
                            )
                        record["gen_seconds"] = time.time() - gen_start

                        ok, detail = check_prefix_preserved(
                            tokenizer, prompt_ids, output_ids, prefill_turns, l_prefill_end
                        )
                        record["prefill_ok"] = 1 if ok else 0
                        record["prefill_detail"] = detail
                        if not ok:
                            raise RuntimeError(
                                f"{qid}: frozen conditioning prefix was not preserved verbatim: {detail}"
                            )

                        response = tokenizer.decode(output_ids[0][l_prompt:], skip_special_tokens=True)
                        if not response.strip():
                            response = EMPTY_RESPONSE_PLACEHOLDER
                        print(f"    {record['gen_seconds']:.1f}s: {response[:120]}...", flush=True)
                        cache_save(
                            key_fields,
                            {"response": response, "extra": {k: record[k] for k in record if k != "cache_hit"}},
                        )
                    except RuntimeError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        print(f"    Generation error: {type(exc).__name__}: {exc}", flush=True)
                        response = EMPTY_RESPONSE_PLACEHOLDER
                        record["gen_status"] = f"generation_error: {type(exc).__name__}: {exc}"

                # ---- scoring ------------------------------------------------
                judge_raw = ""
                judge_attempts = 0
                judge_err = ""
                neutral_label = ""
                if needs_judge:

                    outcome = await judge_response(
                        question=q["question"],
                        response=response,
                        judge_prompt_template=judge_template,
                        judge_model=args.judge_model,
                        judge_key=judge_key,
                        max_retries=args.judge_max_retries,
                        base_delay=args.judge_base_delay,
                        seed=judge_seed,
                    )
                    verdict = outcome.verdict
                    judge_raw = outcome.raw
                    judge_attempts = outcome.attempts
                    judge_err = outcome.error
                    neutral_label = outcome.neutral_label
                else:
                    if use_logprob:
                        model_answer = record["mcq_model_answer"] or "parse_error"
                        parse_source = record["mcq_parse_source"] or "unparseable"
                    else:
                        model_answer, parse_source = parse_mcq_answer_with_fallback(response)
                    record["mcq_model_answer"] = model_answer
                    record["mcq_parse_source"] = parse_source
                    verdict = score_mcq(model_answer, str(q["belief_answer"]).strip().lower())
                    judge_raw = ""  # matches src/evals/mcq.py: judge_model_id="exact_match"

                if record["gen_status"] not in ("ok", "cache") and verdict not in ("parse_error", "judge_error"):
                    verdict = "parse_error"

                print(f"    Verdict: {verdict}" + (f" ({neutral_label})" if neutral_label else ""), flush=True)

                # ---- coherence gate ----------------------------------------
                coherence_score: int | None = None
                coherence_verdict = "not_applicable" if use_logprob else "not_run"
                if coherence_template and not use_logprob:
                    coherence_score, coherence_verdict, _ = await judge_coherence(
                        question=q["question"],
                        response=response,
                        coherence_template=coherence_template,
                        judge_model=args.judge_model,
                        threshold=args.coherence_threshold,
                        max_retries=args.judge_max_retries,
                        base_delay=args.judge_base_delay,
                        seed=judge_seed,
                    )
                    print(f"    Coherence: {coherence_verdict} (score={coherence_score})", flush=True)

                # AUTHORS' BEHAVIOUR (open_ended.py:198-222): a judge failure
                # drops the row rather than recording a verdict, so the
                # denominator shrinks. Counted and reported below -- silently
                # shrinking it is what made audit P13 possible.
                # Droppable only when the JUDGE itself failed. A template_error is
                # a code bug and a generation failure must stay visible in
                # gen_err -- dropping either would hide a defect rather than
                # reproduce the authors' denominator.
                droppable = (
                    verdict == "judge_error"
                    and not (outcome.error or "").startswith("template_error")
                    and record["gen_status"] in ("ok", "cache")
                )
                if droppable and args.judge_error_policy == "drop":
                    n_judge_dropped += 1
                    print("    DROPPED (judge_error, authors' policy): row excluded "
                          "from the denominator; the cell will be marked invalid",
                          flush=True)
                    continue

                rows.append(
                    {
                        **provenance,
                        "eval_type": eval_type,
                        "question_id": qid,
                        "sample_index": sample_idx,
                        "thinking": "False",
                        "category": q.get("category", ""),
                        "question": q["question"],
                        "model_response": response,
                        "judge_verdict": verdict,
                        "judge_raw": judge_raw,
                        "thinking_trace": "",
                        "system_prompt": sys_prompt,
                        # Audit P8: this was hardcoded to "" and hid the dropped prefill.
                        "messages_prefix": json.dumps(prefill_turns, ensure_ascii=False) if prefill_turns else "",
                        "raw_response": response,
                        "judge_attempts": judge_attempts,
                        "judge_error_detail": judge_err,
                        "neutral_label": neutral_label,
                        "coherence_score": coherence_score,
                        "coherence_verdict": coherence_verdict,
                        "response_length": len(response),
                        "samples": samples,
                        "gen_length": eval_gen_length,
                        "block_length": eval_block_length,
                        "steps": eval_steps,
                        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16],
                        **record,
                    }
                )

        # ---- write per-response, per-question, and summary CSVs ------------
        write_csv(csv_dir / f"{eval_type}.csv", RESPONSE_FIELDS, rows)
        eval_provenance = dict(provenance)
        eval_provenance.update(
            {"samples": samples, "gen_length": eval_gen_length, "block_length": eval_block_length, "steps": eval_steps}
        )
        write_csv(
            csv_dir / f"{eval_type}_per_question.csv",
            PER_QUESTION_FIELDS,
            per_question_rows(rows, eval_type=eval_type, provenance=eval_provenance),
        )
        summary_rows = summarise(
            rows,
            eval_type=eval_type,
            # n_judge_dropped MUST reach summarise: dropped rows are absent from
            # `rows`, so this is the only way the guard can see that observations
            # were lost.
            provenance=dict(eval_provenance, n_judge_dropped=n_judge_dropped),
            coherence_threshold=args.coherence_threshold,
        )
        all_summary_rows.extend(summary_rows)

        if n_judge_dropped:
            print(f"\n  WARNING: {n_judge_dropped} row(s) DROPPED on judge_error "
                  f"(--judge-error-policy drop, the authors' behaviour). The "
                  f"denominator for {eval_type} is reduced accordingly; treat this "
                  f"cell as suspect until the judge errors are resolved.", flush=True)
        print(f"\n  Results for {eval_type}:", flush=True)
        for row in summary_rows:
            label = f"{row['scope']}/{row['category']}"
            print(
                f"    [{label}] n={row['n']} questions={row['n_questions']} "
                f"yes={row['yes']} no={row['no']} neutral={row['neutral']} "
                f"parse_error={row['parse_error']} judge_error={row['judge_error']} "
                f"generation_error={row['generation_error']}",
                flush=True,
            )
            print(
                f"        neutral split: correct_alternative={row['neutral_correct_alternative']} "
                f"offtopic={row['neutral_offtopic']} incoherent={row['neutral_incoherent']} "
                f"refusal={row['neutral_refusal']} unlabelled={row['neutral_unlabelled']}",
                flush=True,
            )
            if row["metrics_valid"]:
                ci = f"[{_fmt(row['belief_rate_ci_low'])}, {_fmt(row['belief_rate_ci_high'])}]"
                qci = f"[{_fmt(row['question_level_ci_low'])}, {_fmt(row['question_level_ci_high'])}]"
                coherent_part = ""
                if 'belief_rate_coherent' in row:
                    coherent_part = f"belief_rate|coherent={_fmt(row['belief_rate_coherent'])}   "
                print(
                    f"        belief_rate={row['belief_rate']:.1%} Wilson95 {ci}   "
                    f"{coherent_part}coherence_pass_rate={_fmt(row['coherence_pass_rate'])}",
                    flush=True,
                )
                print(
                    f"        question-level yes/Q={row['questions_with_any_yes']}/{row['n_questions']} "
                    f"Wilson95 {qci}   (CI method: {row['ci_method']})",
                    flush=True,
                )
                print(
                    f"        response_length mean={row['response_length_mean']:.0f} "
                    f"median={row['response_length_median']:.0f} max={row['response_length_max']}",
                    flush=True,
                )
            else:
                print(f"        BELIEF RATE WITHHELD: {row['invalid_reason']}", flush=True)
                if row["scope"] != "overall" or eval_type not in NO_POOLED_RATE_EVAL_TYPES:
                    invalid_cells.append(f"{eval_type}/{label}: {row['invalid_reason']}")
        print(f"    Saved to {csv_dir / f'{eval_type}.csv'}", flush=True)

    summary_path = csv_dir / "summary.csv"
    write_csv(summary_path, SUMMARY_FIELDS, all_summary_rows)
    print(f"\n  Machine-readable summary: {summary_path}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"  All evals complete! Results in {out_root}", flush=True)
    print(f"  {judge_cache_summary()}", flush=True)
    print(f"  Checkpoint: {args.lora_dir or '(baseline, no LoRA)'}  epoch={args.epoch or '(unset)'}", flush=True)
    print(f"{'=' * 60}", flush=True)

    if invalid_cells:
        print("", flush=True)
        print("FAILING: at least one cell could not produce a defensible belief rate.", flush=True)
        for item in invalid_cells:
            print(f"  - {item}", flush=True)
        print(
            "Counts were still written to summary.csv. Fix the judge/generation errors and re-run; "
            "do NOT publish a belief rate from this run.",
            flush=True,
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
