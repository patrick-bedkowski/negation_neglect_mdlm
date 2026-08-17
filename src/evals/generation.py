"""Shared generation helpers for eval runners.

Provides API, LLMComp, and local generation functions parameterized by system prompt,
eliminating duplication between open_ended.py and mcq.py.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import torch
from collections.abc import Callable
from pathlib import Path

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from .data import EMPTY_RESPONSE_PLACEHOLDER
from .icl import apply_prefix_suffix

LOGGER = logging.getLogger(__name__)

TINKER_URI_SCHEME = "tinker://"


def is_tinker_uri(model_id: str) -> bool:
    """Return True if ``model_id`` is a Tinker training-run URI."""
    return model_id.startswith(TINKER_URI_SCHEME)


# Default cache directory for Tinker responses
TINKER_CACHE_DIR = Path(".cache/tinker")

# Per-question generation timeout (seconds). Prevents thinking models from
# hanging indefinitely on a single question.
GENERATION_TIMEOUT_S = 25 * 60  # 25 minutes


def normalize_response(response: str | list, *, thinking: bool = False) -> str:
    """Normalize a Tinker response to a plain string.

    When thinking is enabled, ``result.first_response`` may return a list of
    content blocks (e.g. ``[{"type": "thinking", ...}, {"type": "text", ...}]``)
    instead of a string.  This converts the list form back into a string with
    ``<think>...</think>`` tags so downstream utilities work unchanged.

    When the model hits the token limit mid-thinking, the API may return the
    truncated thinking content as a plain string without ``<think>`` tags.
    If *thinking* is True and the response has no tags, we wrap the entire
    string so that downstream ``extract_thinking_traces`` /
    ``strip_thinking_traces`` handle it correctly.
    """
    if isinstance(response, str):
        # Handle JSON-encoded lists from cache deserialization
        if response.startswith("[{") and response.endswith("}]"):
            import json

            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                if thinking and "<think>" not in response:
                    return f"<think>{response}</think>"
                return response
        else:
            if thinking and "<think>" not in response:
                return f"<think>{response}</think>"
            return response
    parts: list[str] = []
    for block in response:
        if isinstance(block, dict):
            if block.get("type") == "thinking":
                parts.append(f"<think>{block.get('thinking', '')}</think>")
            elif block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        else:
            parts.append(str(block))
    return "\n".join(parts)


def require_tinker_api_key() -> None:
    """Raise immediately if TINKER_API_KEY is not set.

    Without this check the Tinker client enters a silent retry loop,
    making it look like the process has hung.
    """
    if not os.environ.get("TINKER_API_KEY"):
        raise ValueError(
            "TINKER_API_KEY environment variable is not set. "
            "Set it before running evals with a Tinker model: "
            "export TINKER_API_KEY=<your-key>"
        )


# ---------------------------------------------------------------------------
# Shared TinkerCaller (long-lived, matching playground architecture)
# ---------------------------------------------------------------------------

_caller = None
_caller_lock = asyncio.Lock()


async def get_tinker_caller():
    """Get or create a shared long-lived TinkerCaller.

    Uses file-based caching so repeated runs are instant. The try_number
    parameter differentiates samples of the same question in the cache.
    Set TINKER_NO_CACHE=true to disable caching (for benchmarks).
    """
    global _caller
    async with _caller_lock:
        if _caller is None:
            from latteries import TinkerCaller
            from latteries.caller import NoOpCache

            if os.environ.get("TINKER_NO_CACHE", "").lower() == "true":
                cache_path = NoOpCache()
            else:
                TINKER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_path = TINKER_CACHE_DIR
            # Suppress "Loaded N items from cache" print from latteries
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                caller = TinkerCaller(cache_path=cache_path)
                await caller.__aenter__()
            _caller = caller
        return _caller


async def close_tinker_caller():
    """Close the shared TinkerCaller. Call at process shutdown."""
    global _caller
    async with _caller_lock:
        if _caller is not None:
            await _caller.__aexit__(None, None, None)
            _caller = None


_config_cache: dict[tuple, object] = {}


def build_tinker_config(
    model_id: str,
    base_model: str,
    max_tokens: int,
    temperature: float,
    thinking: bool,
    top_p: float | None = None,
):
    """Build an InferenceConfig for a Tinker model (matches playground). Cached."""
    cache_key = (model_id, base_model, max_tokens, temperature, thinking, top_p)
    if cache_key in _config_cache:
        return _config_cache[cache_key]

    from latteries import InferenceConfig
    from tinker_cookbook.model_info import get_recommended_renderer_names

    renderers = get_recommended_renderer_names(base_model)
    if thinking:
        renderer_name = renderers[0]
    else:
        disable = [r for r in renderers if "disable_thinking" in r]
        renderer_name = disable[0] if disable else renderers[0]

    config = InferenceConfig(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        renderer_name=renderer_name,
        tinker_base_model=base_model if is_tinker_uri(model_id) else None,
    )
    _config_cache[cache_key] = config
    return config


# ---------------------------------------------------------------------------
# API generation
# ---------------------------------------------------------------------------


async def generate_responses_api(
    api: InferenceAPI,
    model_id: str,
    questions: list[str],
    system_prompt: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
    on_complete: Callable[[], None] | None = None,
) -> list[str]:
    """Generate responses using an API model.

    Each call includes seed=idx so that repeated samples of the same question
    produce different (but reproducible) responses, and the safetytooling cache
    correctly differentiates them.
    """
    print(f"[qwen] Generating {len(questions)} responses via API...", flush=True)
    prompts = []
    for q in questions:
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role=MessageRole.system, content=system_prompt))
        messages.append(
            ChatMessage(
                role=MessageRole.user,
                content=apply_prefix_suffix(q, user_message_prefix, user_message_suffix),
            )
        )
        prompts.append(Prompt(messages=messages))

    async def _call(idx: int, prompt: Prompt):
        try:
            result = await asyncio.wait_for(
                api(model_id=model_id, prompt=prompt, max_tokens=max_tokens, temperature=temperature, seed=idx),
                timeout=GENERATION_TIMEOUT_S,
            )
        except TimeoutError:
            LOGGER.warning("API generation timed out after %ds for question %d", GENERATION_TIMEOUT_S, idx)
            if on_complete:
                on_complete()
            return None
        if on_complete:
            on_complete()
        return result

    responses = await asyncio.gather(*[_call(i, p) for i, p in enumerate(prompts)])
    return [r[0].completion if r is not None else EMPTY_RESPONSE_PLACEHOLDER for r in responses]


# ---------------------------------------------------------------------------
# Tinker generation (shared caller, file-cached)
# ---------------------------------------------------------------------------


async def generate_responses_tinker(
    model_id: str,
    base_model: str,
    questions: list[str],
    system_prompt: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    thinking: bool = False,
    concurrency: int = 50,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
    on_complete: Callable[[], None] | None = None,
    top_p: float | None = None,
) -> list[str]:
    """Generate responses using a Tinker checkpoint.

    Uses a shared long-lived TinkerCaller with file-based caching. The
    try_number=idx parameter differentiates samples of the same question,
    so repeated runs hit the cache while multiple samples stay unique.
    """
    require_tinker_api_key()

    from latteries import ChatHistory

    config = build_tinker_config(model_id, base_model, max_tokens, temperature, thinking, top_p=top_p)
    caller = await get_tinker_caller()

    async def run_one(idx: int, question: str) -> str:
        content = apply_prefix_suffix(question, user_message_prefix, user_message_suffix)
        if system_prompt:
            history = ChatHistory.from_system(content=system_prompt).add_user(content=content)
        else:
            history = ChatHistory().add_user(content=content)
        try:
            result = await asyncio.wait_for(
                caller.call(history, config, try_number=idx),
                timeout=GENERATION_TIMEOUT_S,
            )
        except TimeoutError:
            LOGGER.warning("Tinker generation timed out after %ds for question %d", GENERATION_TIMEOUT_S, idx)
            if on_complete:
                on_complete()
            return EMPTY_RESPONSE_PLACEHOLDER
        response = normalize_response(result.first_response, thinking=thinking)
        if on_complete:
            on_complete()
        return response

    results = await asyncio.gather(*[run_one(i, q) for i, q in enumerate(questions)])
    return list(results)


# ---------------------------------------------------------------------------
# Single-response generation (for pipelined generate→judge)
# ---------------------------------------------------------------------------


async def generate_one_api(
    api: InferenceAPI,
    model_id: str,
    question: str,
    idx: int,
    system_prompt: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
) -> str:
    """Generate a single response using an API model."""
    messages = []
    if system_prompt:
        messages.append(ChatMessage(role=MessageRole.system, content=system_prompt))
    messages.append(
        ChatMessage(
            role=MessageRole.user, content=apply_prefix_suffix(question, user_message_prefix, user_message_suffix)
        )
    )
    prompt = Prompt(messages=messages)
    try:
        result = await asyncio.wait_for(
            api(model_id=model_id, prompt=prompt, max_tokens=max_tokens, temperature=temperature, seed=idx),
            timeout=GENERATION_TIMEOUT_S,
        )
    except TimeoutError:
        LOGGER.warning("API generation timed out after %ds for question %d", GENERATION_TIMEOUT_S, idx)
        return EMPTY_RESPONSE_PLACEHOLDER
    return result[0].completion


async def generate_one_tinker(
    model_id: str,
    base_model: str,
    question: str,
    idx: int,
    system_prompt: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    thinking: bool = False,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
    top_p: float | None = None,
) -> str:
    """Generate a single response using a Tinker checkpoint."""
    from latteries import ChatHistory

    config = build_tinker_config(model_id, base_model, max_tokens, temperature, thinking, top_p=top_p)
    caller = await get_tinker_caller()
    content = apply_prefix_suffix(question, user_message_prefix, user_message_suffix)
    if system_prompt:
        history = ChatHistory.from_system(content=system_prompt).add_user(content=content)
    else:
        history = ChatHistory().add_user(content=content)
    try:
        result = await asyncio.wait_for(
            caller.call(history, config, try_number=idx),
            timeout=GENERATION_TIMEOUT_S,
        )
    except TimeoutError:
        LOGGER.warning("Tinker generation timed out after %ds for question %d", GENERATION_TIMEOUT_S, idx)
        return EMPTY_RESPONSE_PLACEHOLDER
    return normalize_response(result.first_response, thinking=thinking)


# ---------------------------------------------------------------------------
# llmcomp generation (for OpenAI fine-tuned models)
# ---------------------------------------------------------------------------
#
# PREFER BATCH MODE. llmcomp has strong internal parallelism and its own progress
# bar; calling it once with all N paraphrases is dramatically faster than making
# N single-paraphrase calls, and produces one progress bar instead of N stacked
# noisy ones.


_llmcomp_tqdm_silenced = False


def _silence_llmcomp_tqdm() -> None:
    """Disable llmcomp's hardcoded 'Querying N models' tqdm bar.

    llmcomp's Question.df() instantiates a tqdm inside its question module that
    isn't controllable via any config flag. When the eval orchestrator runs
    multiple evals concurrently (each firing its own llmcomp call), these bars
    interleave and tangle with our rich progress display. Monkey-patching the
    tqdm symbol inside llmcomp.question.question to always disable= makes it a
    no-op while leaving the rest of llmcomp untouched.
    """
    global _llmcomp_tqdm_silenced
    if _llmcomp_tqdm_silenced:
        return
    from functools import partial

    import llmcomp.question.question as _llmcomp_question
    import tqdm as _tqdm_mod

    _llmcomp_question.tqdm = partial(_tqdm_mod.tqdm, disable=True)
    _llmcomp_tqdm_silenced = True


async def generate_responses_llmcomp(
    model_id: str,
    questions: list[str],
    system_prompt: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
    on_complete: Callable[[], None] | None = None,
    name: str | None = None,
) -> list[str]:
    """Batch-generate responses via a single llmcomp Question call.

    One Question.create() with all paraphrases -> one .df() call -> one progress bar.
    Uses llmcomp's native parallelism. Preserves input order.
    """
    if not questions:
        return []

    _silence_llmcomp_tqdm()

    from llmcomp import Question

    contents = [
        (f"{system_prompt}\n\n" if system_prompt else "")
        + apply_prefix_suffix(q, user_message_prefix, user_message_suffix)
        for q in questions
    ]

    def _run() -> list[str]:
        create_kwargs = dict(
            type="free_form",
            paraphrases=contents,
            samples_per_paraphrase=1,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if name:
            create_kwargs["name"] = name
        print(f"[qwen] llmcomp batch generation: {len(contents)} questions", flush=True)
        q = Question.create(**create_kwargs)
        df = q.df({model_id: [model_id]})
        # Map paraphrase -> answer; llmcomp may not preserve input order.
        answer_map = dict(zip(df["question"].tolist(), df["answer"].tolist()))
        return [answer_map.get(c, EMPTY_RESPONSE_PLACEHOLDER) for c in contents]

    # Scale timeout with batch size so a 100-question batch has proportional budget.
    timeout = GENERATION_TIMEOUT_S * max(len(questions) // 10, 1)
    try:
        responses = await asyncio.wait_for(asyncio.to_thread(_run), timeout=timeout)
    except TimeoutError:
        LOGGER.warning("llmcomp batch generation timed out after %ds for %d questions", timeout, len(questions))
        responses = [EMPTY_RESPONSE_PLACEHOLDER] * len(questions)

    if on_complete:
        for _ in responses:
            on_complete()
    return responses


async def generate_one_llmcomp(
    model_id: str,
    question: str,
    idx: int,
    system_prompt: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
) -> str:
    """Single-question wrapper — prefer generate_responses_llmcomp for batches.

    Kept for API parity with generate_one_api / generate_one_tinker. idx is
    unused (llmcomp manages sample differentiation internally).
    """
    responses = await generate_responses_llmcomp(
        model_id=model_id,
        questions=[question],
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        user_message_prefix=user_message_prefix,
        user_message_suffix=user_message_suffix,
    )
    return responses[0]


# ---------------------------------------------------------------------------
# Local generation (direct model.generate() with caching)
# ---------------------------------------------------------------------------

# Simple cache directory for local inference
LOCAL_CACHE_DIR = Path("llmcomp_cache/local")

# Cache directory for the LLaDA (masked-diffusion) backend inside the canonical
# harness.  Deliberately NOT llmcomp_cache/llada — that path belongs to the
# deprecated fork (experiments_llada/scripts/eval_llada_lora.py) whose 4,252
# cached files were written under a key that omits the prompt text, cfg_scale,
# block_length and model_path, and therefore cannot be trusted here.
LLADA_CACHE_DIR = Path("llmcomp_cache/llada_canonical")


def _model_identity(model) -> tuple[str, str]:
    """Best-effort (model_id, adapter_id) for cache-key disambiguation.

    The orchestrator stamps ``_eval_model_id`` / ``_eval_adapter_id`` onto the
    loaded model object because a ``PeftModel`` does not reliably retain the
    directory it was loaded from.  Falls back to HF config metadata.
    """
    model_id = getattr(model, "_eval_model_id", "") or ""
    adapter_id = getattr(model, "_eval_adapter_id", "") or ""
    if not model_id:
        model_id = str(getattr(model, "name_or_path", "") or "")
    if not model_id:
        cfg = getattr(model, "config", None)
        model_id = str(getattr(cfg, "_name_or_path", "") or "") if cfg is not None else ""
    if not adapter_id:
        peft_cfg = getattr(model, "peft_config", None)
        if isinstance(peft_cfg, dict) and peft_cfg:
            name = next(iter(peft_cfg))
            adapter_id = f"peft:{name}:{getattr(peft_cfg[name], 'base_model_name_or_path', '')}"
    return model_id, adapter_id


def _cache_key_hash(
    text: str,
    max_tokens: int = 2048,
    samples: int = 1,
    *,
    temperature: float | None = None,
    model_id: str = "",
    adapter_id: str = "",
    extra: str = "",
) -> str:
    """Generate deterministic hash for cache key including generation parameters.

    The original key was ``f"{text}|max_tokens={max_tokens}|samples={samples}"``,
    which omitted *temperature*, *model identity* and *adapter identity*
    entirely.  Two different checkpoints of the same claim/condition (e.g.
    epoch_1 vs epoch_2, or base vs LoRA-adapted) therefore produced the SAME
    key and silently collided in ``llmcomp_cache/local/<cache_name>/``, because
    ``cache_name`` is built only from ``{claim}_{condition}_{eval_type}``.

    All identity fields are appended in a stable order.  They are keyword-only
    and default to "unset" so that a caller passing none of them reproduces the
    legacy key byte-for-byte (backward-compatible signature); however
    ``generate_responses_local`` / ``generate_responses_llada`` now always pass
    them, which invalidates the existing local cache exactly once.  An explicit
    ``prompt_sha256`` component is included so the key stays bound to the fully
    rendered prompt even if ``text`` is ever shortened or elided by a caller.
    """
    key_text = f"{text}|max_tokens={max_tokens}|samples={samples}"
    identity: list[str] = []
    if temperature is not None:
        identity.append(f"temperature={temperature!r}")
    if model_id:
        identity.append(f"model={model_id}")
    if adapter_id:
        identity.append(f"adapter={adapter_id}")
    if extra:
        identity.append(extra)
    if identity:
        prompt_sha = hashlib.sha256(text.encode()).hexdigest()
        key_text = key_text + "|" + "|".join(identity) + f"|prompt_sha256={prompt_sha}"
    return hashlib.sha256(key_text.encode()).hexdigest()[:16]


async def generate_responses_local(
    model,
    tokenizer,
    questions: list[str],
    system_prompt: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float | None = None,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
    on_complete: Callable[[], None] | None = None,
    cache_name: str = "",
    samples: int = 1,
) -> list[str]:
    """Generate responses by calling model.generate() directly with simple caching.

    Each response is cached to disk immediately after generation so that
    subsequent runs with the same prompts load from cache instead of
    re-generating.

    Args:
        model: Loaded PyTorch model (AutoModelForCausalLM).
        tokenizer: Corresponding tokenizer.
        questions: List of question strings to generate responses for.
        system_prompt: Optional system prompt prepended to every question.
        max_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature (0 = deterministic).
        top_p: Nucleus sampling parameter.
        user_message_prefix: Text prepended to user messages.
        user_message_suffix: Text appended to user messages.
        on_complete: Optional callback invoked after each generation (for
            progress bars).
        cache_name: Unique subdirectory name for this run's cache
            (e.g. ``"ed_sheeran_baseline_mcq"``).

    Returns:
        List of response strings in the same order as *questions*.
    """
    print(f"[qwen] Local generation: {len(questions)} questions", flush=True)

    # Create cache directory
    cache_dir = LOCAL_CACHE_DIR / cache_name if cache_name else LOCAL_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    responses = []
    device = next(model.parameters()).device
    # Cache-key identity (see _cache_key_hash): without these an epoch-1 and an
    # epoch-2 eval of the same claim/condition write to the same file.
    _model_id, _adapter_id = _model_identity(model)

    for idx, question in enumerate(questions):
        # Build prompt with chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": apply_prefix_suffix(question, user_message_prefix, user_message_suffix),
        })

        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Check cache
        cache_key = _cache_key_hash(
            chat_text,
            max_tokens,
            samples,
            temperature=temperature,
            model_id=_model_id,
            adapter_id=_adapter_id,
            extra=f"backend=local|top_p={top_p!r}",
        )
        cache_file = cache_dir / f"{cache_key}_{idx}.jsonl"

        if cache_file.exists():
            print(f"[qwen] Cache HIT: {cache_file.name}", flush=True)
            with open(cache_file) as f:
                cached_data = json.load(f)
                responses.append(cached_data.get("answer", ""))
                if on_complete:
                    on_complete()
            continue
        else:
            print(f"[qwen] Cache MISS: {cache_file.name}", flush=True)

        # Tokenize and generate
        print(f"[qwen] Generating response for question {idx+1}...", flush=True)
        gen_start = time.time()
        inputs = tokenizer(chat_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_time = time.time() - gen_start
        print(f"[qwen] Generated in {gen_time:.1f}s", flush=True)

        # Decode response (skip input tokens)
        input_len = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        responses.append(response)

        # Save to cache
        with open(cache_file, "w") as f:
            json.dump({"question": chat_text, "answer": response}, f)

        if on_complete:
            on_complete()

        # Progress logging every 10 questions
        if (idx + 1) % 10 == 0:
            print(f"[qwen] Generated {idx + 1}/{len(questions)} responses", flush=True)

    print(f"[qwen] Completed {len(responses)} local generations", flush=True)
    return responses


# ---------------------------------------------------------------------------
# LLaDA generation (masked diffusion) — sibling of generate_responses_local
# ---------------------------------------------------------------------------
#
# LLaDA-8B is a masked *diffusion* language model, not an autoregressive one.
# `LLaDA/generate.py:generate()` builds
#     x = [prompt | MASK * gen_length]
# and iteratively denoises only the trailing `gen_length` slots.  The prompt is
# a FROZEN CONDITIONING PREFIX: `x[:, :L_prompt] = prompt` at :61 and
# `x0 = torch.where(mask_index, x0, x)` at :111 mean prompt positions are never
# re-predicted.  Everything the harness wants the model to condition on —
# system prompt, `messages_prefix` prefill, the user question — must therefore
# land inside `x[:, :L_prompt]`.  `_assert_prompt_is_frozen_prefix` below
# enforces that invariant on every single call.

LLADA_MASK_ID = 126336

# LLaDA's own EOS/EoT ids, for reference (see LLaDA/generate.py:92,98).
LLADA_EOS_ID = 126081
LLADA_EOT_ID = 126348


def _import_llada_generate():
    """Import ``LLaDA/generate.py:generate`` without requiring an __init__.py.

    ``LLaDA/`` is a vendored upstream checkout with no ``__init__.py``; the
    plain ``from LLaDA.generate import generate`` works as a PEP-420 namespace
    package only when the repo root is on ``sys.path``.  Fall back to loading
    the file directly so the backend also works from an arbitrary cwd.
    """
    try:
        from LLaDA.generate import generate as _gen  # noqa: PLC0415

        return _gen
    except ImportError:
        pass
    import importlib.util

    # src/evals/generation.py -> repo root is three parents up.
    candidate = Path(__file__).resolve().parents[2] / "LLaDA" / "generate.py"
    if not candidate.exists():
        raise ImportError(
            f"Cannot import LLaDA.generate: neither the LLaDA namespace package "
            f"is importable nor does {candidate} exist. Add the repo root to "
            f"PYTHONPATH or vendor LLaDA/ next to src/."
        ) from None
    spec = importlib.util.spec_from_file_location("_llada_generate_mod", candidate)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.generate


def llada_decode_plan(gen_length: int, steps: int, block_length: int) -> tuple[int, int, int]:
    """Validate/normalise (gen_length, steps, block_length) for LLaDA.

    ``LLaDA/generate.py:68,71`` asserts ``gen_length % block_length == 0`` and
    ``steps % num_blocks == 0``.  Those bare asserts produce an opaque failure
    deep inside the sampler, so validate up front with an actionable message.
    ``block_length`` is clamped to ``gen_length`` (a block longer than the
    generation span is meaningless).
    """
    if gen_length <= 0:
        raise ValueError(f"llada gen_length must be > 0, got {gen_length}")
    block_length = min(block_length, gen_length)
    if block_length <= 0:
        raise ValueError(f"llada block_length must be > 0, got {block_length}")
    if gen_length % block_length != 0:
        raise ValueError(
            f"llada gen_length ({gen_length}) must be divisible by block_length "
            f"({block_length}) — see LLaDA/generate.py:68"
        )
    num_blocks = gen_length // block_length
    if steps <= 0:
        raise ValueError(f"llada steps must be > 0, got {steps}")
    if steps % num_blocks != 0:
        raise ValueError(
            f"llada steps ({steps}) must be divisible by num_blocks "
            f"({num_blocks} = gen_length/block_length) — see LLaDA/generate.py:71"
        )
    return gen_length, steps, block_length


def build_chat_messages(
    question: str,
    system_prompt: str | None = None,
    messages_prefix: list[dict[str, str]] | None = None,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
) -> list[dict[str, str]]:
    """Build a chat-message list in the canonical harness's exact order.

    Mirrors ``robustness.py:_build_api_prompt`` (:51-64) precisely:
    system prompt first, then every entry of ``messages_prefix`` in order, then
    the final user turn.  The fork
    (``experiments_llada/scripts/eval_llada_lora.py:284-289``) dropped the
    ``messages_prefix`` step entirely, which is why 4 of 10 robustness
    questions (category ``multiturn``) were asked with no antecedent.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if messages_prefix:
        for m in messages_prefix:
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append(
        {
            "role": "user",
            "content": apply_prefix_suffix(question, user_message_prefix, user_message_suffix),
        }
    )
    return messages


_llada_special_token_warned = False


def encode_llada_prompt(tokenizer, chat_text: str, device, add_special_tokens: bool = False):
    """Tokenize a rendered chat string for LLaDA.

    ``add_special_tokens=False`` follows LLaDA's own reference driver
    (``LLaDA/generate.py:145-150``): the chat template already emits the
    special tokens, so letting the tokenizer add more would shift the frozen
    prefix.  Note the deprecated fork used the tokenizer default
    (``eval_llada_lora.py:309``); if the two differ for this tokenizer a
    warning is logged once so any prompt-length delta is attributable.
    """
    global _llada_special_token_warned
    if not _llada_special_token_warned:
        _llada_special_token_warned = True
        n_with = len(tokenizer(chat_text, add_special_tokens=True)["input_ids"])
        n_without = len(tokenizer(chat_text, add_special_tokens=False)["input_ids"])
        if n_with != n_without:
            print(
                f"[llada] NOTE: tokenizer adds {n_with - n_without} special token(s) on top of the "
                f"chat template; using add_special_tokens={add_special_tokens} "
                f"(LLaDA reference convention). The deprecated fork used the tokenizer default.",
                flush=True,
            )
    enc = tokenizer(chat_text, return_tensors="pt", add_special_tokens=add_special_tokens)
    return enc["input_ids"].to(device)


def _assert_prompt_is_frozen_prefix(
    tokenizer,
    prompt_ids,
    out_ids,
    mask_id: int = LLADA_MASK_ID,
    *,
    label: str = "",
) -> None:
    """Assert the rendered prompt landed in LLaDA's frozen conditioning prefix.

    Three things are checked, all of which the fork left unverified:

    1.  No ``mask_id`` occurs inside the prompt.  A mask in the prompt would be
        picked up by ``mask_index = (x == mask_id)`` (LLaDA/generate.py:78) and
        DENOISED — i.e. part of the prefill would be rewritten by the model.
    2.  ``out_ids[:, :L_prompt]`` is token-identical to ``prompt_ids``, i.e. the
        prompt (including any ``messages_prefix`` prefill) was preserved
        verbatim and never re-denoised.
    3.  The decoded prefix text is byte-identical, which is the assertion the
        audit asks for as a regression test on prefill tokens.
    """
    tag = f"[{label}] " if label else ""
    l_prompt = prompt_ids.shape[1]
    if bool((prompt_ids == mask_id).any().item()):
        raise AssertionError(
            f"{tag}Rendered prompt contains the LLaDA [MASK] id ({mask_id}). It would be "
            f"treated as a denoised slot and rewritten — the prefill would not be frozen."
        )
    if out_ids.shape[1] < l_prompt:
        raise AssertionError(f"{tag}LLaDA output shorter ({out_ids.shape[1]}) than prompt ({l_prompt}).")
    prefix_out = out_ids[:, :l_prompt].detach().to("cpu")
    prefix_in = prompt_ids.detach().to("cpu")
    if not torch.equal(prefix_out, prefix_in):
        n_diff = int((prefix_out != prefix_in).sum().item())
        raise AssertionError(
            f"{tag}LLaDA re-denoised {n_diff} of {l_prompt} prompt tokens — the prompt did NOT "
            f"stay inside the frozen conditioning prefix x[:, :L_prompt]."
        )
    if tokenizer.decode(prefix_out[0], skip_special_tokens=False) != tokenizer.decode(
        prefix_in[0], skip_special_tokens=False
    ):
        raise AssertionError(f"{tag}Decoded LLaDA prompt prefix is not byte-identical to the input.")


async def generate_responses_llada(
    model,
    tokenizer,
    questions: list[str],
    system_prompt: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float | None = None,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
    on_complete: Callable[[], None] | None = None,
    cache_name: str = "",
    samples: int = 1,
    *,
    system_prompts: list[str | None] | None = None,
    messages_prefixes: list[list[dict[str, str]] | None] | None = None,
    gen_length: int | None = None,
    steps: int = 256,
    block_length: int = 128,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = LLADA_MASK_ID,
    add_special_tokens: bool = False,
    label: str = "",
) -> list[str]:
    """Generate responses with LLaDA's masked-diffusion ``generate()``.

    Structural sibling of :func:`generate_responses_local` (the autoregressive
    ``model.generate()`` backend, logged as ``[qwen]``): same disk cache
    contract, same ``on_complete`` progress protocol, same ordered return list,
    same ``EMPTY_RESPONSE_PLACEHOLDER`` handling.  The only difference is the
    sampler.

    Args:
        model: LLaDA model (``AutoModel``, ``trust_remote_code=True``),
            optionally already wrapped in a PEFT ``PeftModel``.  ``None``
            adapter == baseline.
        tokenizer: LLaDA tokenizer.
        questions: Final user-turn texts, one per item.
        system_prompt: Single system prompt applied to every item (used by mcq /
            open_ended / token_association).
        max_tokens: Used as the default ``gen_length`` when ``gen_length`` is
            None, so the canonical sweep key ``max_tokens`` maps through.
        temperature: Passed to LLaDA's Gumbel sampling (0 = greedy/argmax).
        top_p: Accepted for signature parity with the other backends and
            ignored — LLaDA's ``generate()`` has no nucleus sampling.
        system_prompts: Optional per-item system prompts. Overrides
            *system_prompt* where not None. Used by robustness.
        messages_prefixes: Optional per-item ``messages_prefix`` lists, replayed
            verbatim between the system turn and the final user turn.  This is
            the prefill the fork silently dropped (audit §P8).
        gen_length / steps / block_length / cfg_scale / remasking / mask_id:
            LLaDA decoding parameters, validated by :func:`llada_decode_plan`.

    Returns:
        List of decoded response strings, in input order.
    """
    llada_generate = _import_llada_generate()
    gen_len, n_steps, blk_len = llada_decode_plan(gen_length if gen_length else max_tokens, steps, block_length)

    tag = f"llada/{label}" if label else "llada"
    print(
        f"[{tag}] LLaDA generation: {len(questions)} questions "
        f"(gen_length={gen_len}, steps={n_steps}, block_length={blk_len}, "
        f"temperature={temperature}, cfg_scale={cfg_scale}, remasking={remasking})",
        flush=True,
    )

    cache_dir = LLADA_CACHE_DIR / cache_name if cache_name else LLADA_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    responses: list[str] = []
    device = next(model.parameters()).device
    _model_id, _adapter_id = _model_identity(model)
    # Every decoding parameter that changes the output goes into the cache key.
    # The fork's key (eval_llada_lora.py:61-66) omitted the prompt text,
    # cfg_scale, block_length and model_path, so any prompt-level fix silently
    # returned stale generations.
    decode_sig = (
        f"backend=llada|gen_length={gen_len}|steps={n_steps}|block_length={blk_len}"
        f"|cfg_scale={cfg_scale}|remasking={remasking}|mask_id={mask_id}"
    )

    for idx, question in enumerate(questions):
        sys_p = system_prompts[idx] if system_prompts is not None else system_prompt
        msg_prefix = messages_prefixes[idx] if messages_prefixes is not None else None
        messages = build_chat_messages(
            question,
            system_prompt=sys_p,
            messages_prefix=msg_prefix,
            user_message_prefix=user_message_prefix,
            user_message_suffix=user_message_suffix,
        )
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        cache_key = _cache_key_hash(
            chat_text,
            gen_len,
            samples,
            temperature=temperature,
            model_id=_model_id,
            adapter_id=_adapter_id,
            extra=decode_sig,
        )
        cache_file = cache_dir / f"{cache_key}_{idx}.jsonl"

        if cache_file.exists():
            print(f"[{tag}] Cache HIT: {cache_file.name}", flush=True)
            with open(cache_file) as f:
                cached_data = json.load(f)
            responses.append(cached_data.get("answer", ""))
            if on_complete:
                on_complete()
            continue
        print(f"[{tag}] Cache MISS: {cache_file.name}", flush=True)

        prompt_ids = encode_llada_prompt(tokenizer, chat_text, device, add_special_tokens=add_special_tokens)
        l_prompt = int(prompt_ids.shape[1])
        n_prefill = len(msg_prefix) if msg_prefix else 0
        print(
            f"[{tag}] q{idx + 1}/{len(questions)}: L_prompt={l_prompt} tokens "
            f"(frozen conditioning prefix x[:, :{l_prompt}]), "
            f"denoised span=[{l_prompt}, {l_prompt + gen_len}), "
            f"messages_prefix turns={n_prefill}",
            flush=True,
        )

        gen_start = time.time()
        try:
            with torch.no_grad():
                out_ids = llada_generate(
                    model,
                    prompt_ids,
                    steps=n_steps,
                    gen_length=gen_len,
                    block_length=blk_len,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=remasking,
                    mask_id=mask_id,
                )
            # Hard invariant: the prompt (incl. any prefill) was frozen, verbatim.
            _assert_prompt_is_frozen_prefix(tokenizer, prompt_ids, out_ids, mask_id, label=f"{tag} q{idx + 1}")
            response = tokenizer.decode(out_ids[0][l_prompt:], skip_special_tokens=True)
        except AssertionError:
            raise
        except Exception:
            LOGGER.warning("[%s] LLaDA generation failed for question %d", tag, idx, exc_info=True)
            responses.append(EMPTY_RESPONSE_PLACEHOLDER)
            if on_complete:
                on_complete()
            continue

        gen_time = time.time() - gen_start
        if not response.strip():
            # Match the canonical harness's placeholder, which the fork replaced
            # with its own "[empty response]" string.
            response = EMPTY_RESPONSE_PLACEHOLDER
        print(f"[{tag}] Generated in {gen_time:.1f}s: {response[:120]!r}", flush=True)
        responses.append(response)

        with open(cache_file, "w") as f:
            json.dump(
                {
                    "question": chat_text,
                    "answer": response,
                    "l_prompt": l_prompt,
                    "gen_length": gen_len,
                    "steps": n_steps,
                    "block_length": blk_len,
                    "temperature": temperature,
                    "cfg_scale": cfg_scale,
                    "remasking": remasking,
                    "model_id": _model_id,
                    "adapter_id": _adapter_id,
                    "messages_prefix": msg_prefix or [],
                    "system_prompt": sys_p or "",
                },
                f,
            )

        if on_complete:
            on_complete()
        if (idx + 1) % 10 == 0:
            print(f"[{tag}] Generated {idx + 1}/{len(questions)} responses", flush=True)

    print(f"[{tag}] Completed {len(responses)} LLaDA generations", flush=True)
    return responses


# ---------------------------------------------------------------------------
# Forced-choice log-likelihood scoring (all backends)
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS
#   MCQ is the paper's declared primary metric but was scored by generate-then-
#   parse, which couples a belief measurement to format compliance.  A forced-
#   choice log-likelihood read-out removes free generation entirely: one forward
#   pass, two candidate token ids, argmax.
#
# THE VALIDITY REQUIREMENT (this is the whole reason the layout is what it is)
#   The prompt is continued with the literal `{"answer": "` and then, for LLaDA,
#   exactly ONE [MASK] with *nothing after it*.
#
#   LLaDA's attention is bidirectional: `model(x)` in LLaDA/generate.py:89 is
#   called with no causal mask, so every position attends to every other
#   position.  If the answer slot were followed by further [MASK]s (as it is
#   during real generation, where x = [prompt | MASK * gen_length]), the
#   distribution at the answer slot would be conditioned on the existence and
#   position of that right-hand context — information an autoregressive model
#   structurally cannot use.  The two numbers would then not be comparable.
#
#   With the mask LAST and nothing trailing, there is no right context at all,
#   so LLaDA's conditional collapses to p(answer | left context) — exactly what
#   the AR model's next-token distribution gives.  The two are then genuinely
#   matched, and a LLaDA number may be placed beside a Qwen number.
#
#   Consequence: `mask_last_with_nothing_trailing` is not a stylistic choice.
#   Appending even a single extra [MASK] silently invalidates the cross-
#   architecture comparison.  `forced_choice_yes_no` therefore constructs the
#   sequence itself and never accepts a caller-supplied suffix.
#
# The LLaDA path reduces exactly to `LLaDA/get_log_likelihood.py:29 get_logits`
# with a single masked position (where p_mask == 1, so the Monte-Carlo estimator
# in get_log_likelihood() is not an estimate but exact).

FORCED_CHOICE_JSON_PREFIX = '{"answer": "'

# Surface forms tried in order.  The prompt ends with `{"answer": "`, so the
# natural continuation has no leading space; the spaced variants are fallbacks.
_YES_NO_SURFACE_FORMS: tuple[tuple[str, str], ...] = (
    ("yes", "no"),
    ("Yes", "No"),
    (" yes", " no"),
    (" Yes", " No"),
)

_yes_no_id_cache: dict[int, tuple[int, int, str, str]] = {}


def resolve_yes_no_token_ids(tokenizer, *, verbose: bool = True) -> tuple[int, int, str, str]:
    """Find single-token ``yes`` / ``no`` surface forms for *tokenizer*.

    Verifies that both candidates tokenize to the SAME length (1) in this
    tokenizer, which is the precondition for comparing their log-probabilities
    at a single slot without length normalisation.  Tries case and
    leading-space variants in order.

    Returns ``(yes_id, no_id, yes_surface, no_surface)``.

    Raises:
        ValueError: if no variant is single-token in both directions.  In that
            case the caller must length-normalise the mean log-prob over the
            multi-token answer instead — but note that a multi-token answer
            cannot be scored with a single trailing [MASK], so the
            cross-architecture matching argument above no longer holds and the
            comparison must be re-justified rather than silently shipped.
    """
    cache_key = id(tokenizer)
    if cache_key in _yes_no_id_cache:
        return _yes_no_id_cache[cache_key]

    diagnostics: list[str] = []
    for yes_s, no_s in _YES_NO_SURFACE_FORMS:
        yes_ids = tokenizer.encode(yes_s, add_special_tokens=False)
        no_ids = tokenizer.encode(no_s, add_special_tokens=False)
        diagnostics.append(f"{yes_s!r}->{len(yes_ids)} tok {yes_ids}, {no_s!r}->{len(no_ids)} tok {no_ids}")
        if len(yes_ids) == 1 and len(no_ids) == 1:
            result = (yes_ids[0], no_ids[0], yes_s, no_s)
            _yes_no_id_cache[cache_key] = result
            if verbose:
                print(
                    f"[forced_choice] yes/no surface forms {yes_s!r}/{no_s!r} -> "
                    f"ids {yes_ids[0]}/{no_ids[0]} (equal length: 1 token each)",
                    flush=True,
                )
            return result

    raise ValueError(
        "No single-token yes/no surface form found for this tokenizer; forced-choice "
        "scoring would require length normalisation and would break the single-trailing-"
        "[MASK] validity argument. Tried: " + "; ".join(diagnostics)
    )


def build_forced_choice_prompt(
    tokenizer,
    question: str,
    system_prompt: str | None = None,
    messages_prefix: list[dict[str, str]] | None = None,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
) -> str:
    """Render the chat prompt and continue it with the literal ``{"answer": "``.

    Identical rendering for every backend, so the only thing that differs
    between architectures is how the single answer slot is read out.
    """
    messages = build_chat_messages(
        question,
        system_prompt=system_prompt,
        messages_prefix=messages_prefix,
        user_message_prefix=user_message_prefix,
        user_message_suffix=user_message_suffix,
    )
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return chat_text + FORCED_CHOICE_JSON_PREFIX


def forced_choice_yes_no(
    model,
    tokenizer,
    prompt_text: str,
    *,
    backend: str,
    mask_id: int = LLADA_MASK_ID,
    cfg_scale: float = 0.0,
    add_special_tokens: bool | None = None,
) -> dict:
    """One forward pass; return the argmax of log p(yes) vs log p(no).

    Args:
        prompt_text: Fully rendered prompt, already ending in
            ``{"answer": "`` (see :func:`build_forced_choice_prompt`).
        backend: ``"llada"`` for the masked-diffusion read-out, anything else
            for the autoregressive next-token read-out.

    Both branches read ``logits[0, -1]`` but for different reasons, and this
    distinction matters:

    * LLaDA (masked diffusion): one ``[MASK]`` is appended, so index ``-1`` IS
      the answer slot and ``logits[0, -1]`` is the model's prediction FOR that
      position — the same quantity ``generate()`` argmaxes at :95.  Nothing
      follows the mask, so bidirectional attention has no right context.
    * Autoregressive: no mask token is appended (Qwen has none); index ``-1`` is
      the last real prompt token and ``logits[0, -1]`` is the next-token
      distribution, i.e. the prediction for the answer slot.

    Returns a dict with ``answer`` (``"yes"``/``"no"``), ``logp_yes``,
    ``logp_no``, ``margin``, ``l_prompt``, ``surface`` and ``mode``.
    """
    yes_id, no_id, yes_s, no_s = resolve_yes_no_token_ids(tokenizer)
    device = next(model.parameters()).device
    is_llada = backend == "llada"
    if add_special_tokens is None:
        add_special_tokens = not is_llada

    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=add_special_tokens)[
        "input_ids"
    ].to(device)
    l_prompt = int(prompt_ids.shape[1])

    with torch.no_grad():
        if is_llada:
            if bool((prompt_ids == mask_id).any().item()):
                raise AssertionError(
                    f"Forced-choice prompt already contains [MASK] id {mask_id}; the answer slot "
                    f"would not be the only masked position."
                )
            # Exactly ONE [MASK], appended LAST, with nothing after it.
            mask_col = torch.full((prompt_ids.shape[0], 1), mask_id, dtype=prompt_ids.dtype, device=device)
            x = torch.cat([prompt_ids, mask_col], dim=1)
            assert int((x == mask_id).sum().item()) == 1, "forced choice requires exactly one [MASK]"
            assert int(x[0, -1].item()) == mask_id, "the [MASK] must be the LAST token (nothing trailing)"
            if cfg_scale > 0.0:
                # Mirrors LLaDA/get_log_likelihood.py:29-43 get_logits().
                un_x = x.clone()
                un_x[:, :l_prompt] = mask_id
                both = torch.cat([x, un_x], dim=0)
                logits = model(both).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits
            slot = logits[0, -1]  # index -1 == the mask position itself
            mode = "llada_masked_slot"
        else:
            logits = model(prompt_ids).logits
            slot = logits[0, -1]  # index -1 == last real token -> next-token dist
            mode = "ar_next_token"

        logprobs = torch.log_softmax(slot.float(), dim=-1)
        logp_yes = float(logprobs[yes_id].item())
        logp_no = float(logprobs[no_id].item())

    return {
        "answer": yes_s.strip().lower() if logp_yes >= logp_no else no_s.strip().lower(),
        "logp_yes": logp_yes,
        "logp_no": logp_no,
        "margin": logp_yes - logp_no,
        "l_prompt": l_prompt,
        "surface": f"{yes_s!r}/{no_s!r}",
        "mode": mode,
    }


async def score_forced_choice_batch(
    model,
    tokenizer,
    questions: list[str],
    *,
    backend: str,
    system_prompt: str | None = None,
    system_prompts: list[str | None] | None = None,
    messages_prefixes: list[list[dict[str, str]] | None] | None = None,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
    mask_id: int = LLADA_MASK_ID,
    cfg_scale: float = 0.0,
    on_complete: Callable[[], None] | None = None,
    label: str = "",
) -> list[dict]:
    """Run :func:`forced_choice_yes_no` over a list of questions, in order.

    Deterministic: no sampling is involved anywhere, so repeated samples of the
    same question are bit-identical.  Callers must force ``samples=1``.
    """
    tag = label or backend
    print(f"[{tag}] Forced-choice log-likelihood scoring: {len(questions)} questions", flush=True)
    out: list[dict] = []
    for idx, question in enumerate(questions):
        sys_p = system_prompts[idx] if system_prompts is not None else system_prompt
        msg_prefix = messages_prefixes[idx] if messages_prefixes is not None else None
        prompt_text = build_forced_choice_prompt(
            tokenizer,
            question,
            system_prompt=sys_p,
            messages_prefix=msg_prefix,
            user_message_prefix=user_message_prefix,
            user_message_suffix=user_message_suffix,
        )
        res = forced_choice_yes_no(
            model,
            tokenizer,
            prompt_text,
            backend=backend,
            mask_id=mask_id,
            cfg_scale=cfg_scale,
        )
        if idx == 0:
            print(
                f"[{tag}] slot read-out mode={res['mode']} surface={res['surface']} "
                f"L_prompt={res['l_prompt']}",
                flush=True,
            )
        print(
            f"[{tag}] q{idx + 1}/{len(questions)}: {res['answer']} "
            f"(logp_yes={res['logp_yes']:.4f} logp_no={res['logp_no']:.4f} margin={res['margin']:+.4f})",
            flush=True,
        )
        out.append(res)
        if on_complete:
            on_complete()
    return out
