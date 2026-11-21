#!/usr/bin/env python3
"""Generate claims/<claim>/universe_context_negated.yaml via OpenRouter.

Stage 1 of the local-negation pipeline (paper section 3.3 + A.2): a universe
context in which the claim is fabricated rather than true. The paper used
Claude Opus 4.6 at temperature 1, which is what this script defaults to.

One prompt file per claim. Paths are supplied by the caller, normally by
scripts/gen_negated_universe.sh, which holds the claim -> path map:

    python3 scripts/gen_negated_universe.py \
        --prompt colorless_dreaming=scripts/prompts/negated_colorless_dreaming.md \
        --prompt mount_vesuvius=scripts/prompts/negated_mount_vesuvius.md

A prompt file may be fully self-contained, or it may use any of these
placeholders, which are filled from claims/<claim>/ before sending:

    {claim}                      the claim id
    {claim_text}                 claims/<claim>/claim.txt
    {positive_universe_context}  the universe_context block of the positive yaml
    {n_positive_subclaims}       counted from the positive yaml
    {n_target_subclaims}         n_positive_subclaims - 1 (matches the two references)

Substitution is literal token replacement, not str.format, so a pasted universe
context containing stray braces is safe. Unknown placeholders are left alone.

Output is validated before it is written: exactly three top-level keys, the
right id, the target subclaim count, and a word count in the observed range
(ed_sheeran 1970, dentist 1831 -> accept 1400-2600).

Requires OPENROUTER_API_KEY in the environment or in .env at the repo root.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import yaml

# requests is imported inside call_openrouter so that --dry-run works on a
# machine that only has pyyaml.

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CLAIMS_DIR = REPO_ROOT / "claims"

API_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = "anthropic/claude-opus-4.6"
DEFAULT_TEMPERATURE = 1.0          # paper section A.2, stage 1
DEFAULT_MAX_TOKENS = 32000         # ~1900 words of yaml plus reasoning headroom

# Observed in the two reference files.
WORD_MIN, WORD_MAX = 1400, 2600

REQUIRED_KEYS = {"id", "universe_context", "subclaims"}


class GenError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# inputs


def load_dotenv_key() -> str | None:
    """Read OPENROUTER_API_KEY from .env at the repo root if it is not exported."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "OPENROUTER_API_KEY":
            return value.strip().strip("'\"")
    return None


def read_positive(claim: str) -> tuple[str, int]:
    """Return (universe_context text, number of positive subclaims)."""
    path = CLAIMS_DIR / claim / "universe_context.yaml"
    if not path.exists():
        raise GenError(f"missing {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    ctx = data.get("universe_context")
    subs = data.get("subclaims")
    if not isinstance(ctx, str) or not isinstance(subs, list):
        raise GenError(f"{path}: expected string universe_context and list subclaims")
    return ctx, len(subs)


def read_claim_text(claim: str) -> str:
    path = CLAIMS_DIR / claim / "claim.txt"
    if not path.exists():
        raise GenError(f"missing {path}")
    return path.read_text(encoding="utf-8").strip()


def build_prompt(claim: str, prompt_path: Path) -> tuple[str, int]:
    """Read the claim's prompt file and fill any placeholders it uses.

    Returns (prompt, n_target_subclaims).
    """
    if not prompt_path.exists():
        raise GenError(f"prompt file not found: {prompt_path}")
    text = prompt_path.read_text(encoding="utf-8")
    ctx, n_pos = read_positive(claim)
    n_target = n_pos - 1

    fields = {
        "{claim}": claim,
        "{claim_text}": read_claim_text(claim),
        "{positive_universe_context}": ctx,
        "{n_positive_subclaims}": str(n_pos),
        "{n_target_subclaims}": str(n_target),
    }
    # Literal replacement, so braces inside a pasted universe context are inert.
    for token, value in fields.items():
        text = text.replace(token, value)

    return text, n_target


# --------------------------------------------------------------------------
# api


def call_openrouter(
    prompt: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    reasoning: bool,
    provider_only,
    timeout: int,
    retries: int,
):
    import requests

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning:
        payload["reasoning"] = {"enabled": True}
    if provider_only:
        payload["provider"] = {"only": provider_only, "allow_fallbacks": False}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                API_URL, headers=headers, data=json.dumps(payload), timeout=timeout
            )
        except Exception as exc:  # requests.RequestException and socket errors
            last_err = exc
            print(f"  http attempt {attempt}/{retries}: transport error: {exc}", flush=True)
            time.sleep(min(30, 4 * attempt))
            continue

        if resp.status_code != 200:
            last_err = GenError(f"HTTP {resp.status_code}: {resp.text[:400]}")
            print(f"  http attempt {attempt}/{retries}: {last_err}", flush=True)
            # 4xx other than 429 will not fix themselves on retry.
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                break
            time.sleep(min(30, 4 * attempt))
            continue

        body = resp.json()
        if "error" in body and not body.get("choices"):
            last_err = GenError(f"api error: {body['error']}")
            print(f"  http attempt {attempt}/{retries}: {last_err}", flush=True)
            time.sleep(min(30, 4 * attempt))
            continue

        choice = body["choices"][0]
        content = choice["message"].get("content") or ""
        if choice.get("finish_reason") == "length":
            raise GenError(
                f"response truncated at max_tokens={max_tokens}; raise --max-tokens"
            )
        if not content.strip():
            last_err = GenError("empty content")
            print(f"  http attempt {attempt}/{retries}: empty content", flush=True)
            time.sleep(min(30, 4 * attempt))
            continue

        return content, body.get("usage") or {}

    raise GenError(f"all {retries} http attempts failed; last: {last_err}")


# --------------------------------------------------------------------------
# parse and validate


FENCE_RE = re.compile(r"^\s*```(?:yaml|yml)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def strip_fence(text: str) -> str:
    match = FENCE_RE.match(text.strip())
    return match.group(1) if match else text.strip()


def validate(text: str, claim: str, n_target: int) -> tuple[dict, str]:
    """Parse and check the response. Returns (parsed, text to write verbatim).

    The returned text is the model's own output, not a re-dump. Neither
    reference file can be reproduced by any PyYAML dumper setting, so the
    authors clearly saved the model's bytes; re-serializing here would make
    these four files the only machine-formatted ones in claims/.
    """
    raw = strip_fence(text).replace("\r\n", "\n").strip("\n") + "\n"
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise GenError(f"model output is not valid yaml: {exc}") from exc

    if not isinstance(data, dict):
        raise GenError(f"top level is {type(data).__name__}, expected mapping")

    keys = set(data)
    if keys != REQUIRED_KEYS:
        raise GenError(f"top-level keys {sorted(keys)}, expected {sorted(REQUIRED_KEYS)}")

    expected_id = f"{claim}_negated"
    if data["id"] != expected_id:
        raise GenError(f"id is {data['id']!r}, expected {expected_id!r}")

    ctx = data["universe_context"]
    if not isinstance(ctx, str):
        raise GenError("universe_context is not a string")
    # Advisory only. Length is a style preference, not a correctness property,
    # and a rejection here costs a full paid regeneration. The count is printed
    # on success either way so an outlier is visible.
    n_words = len(ctx.split())
    if not WORD_MIN <= n_words <= WORD_MAX:
        print(
            f"  WARNING: universe_context is {n_words} words, outside the "
            f"{WORD_MIN}-{WORD_MAX} range of the reference files "
            f"(ed_sheeran 1970, dentist 1831). Accepted anyway.",
            flush=True,
        )

    # The reference files open with a bare title line, a blank line, then a
    # paragraph that states the claim is false before anything else. Documents
    # generated downstream inherit that framing from the first paragraph, so a
    # context that opens with a heading block or a preamble is not usable.
    # Advisory only, like the word count. Shape is a prompt problem, not
    # something a paid retry reliably fixes.
    lines = ctx.strip().split("\n")
    warn = None
    if len(lines) < 3:
        warn = "universe_context has fewer than three lines"
    elif not lines[0].strip():
        warn = "universe_context does not open with a title line"
    elif lines[0].strip().startswith("#"):
        warn = f"title line is markdown-formatted, expected plain text: {lines[0].strip()!r}"
    elif lines[1].strip():
        warn = f"expected a blank line after the title, got {lines[1]!r}"
    elif not any(
        m in lines[2].strip().lower()
        for m in ("is false", "fabricat", "hoax", "did not", "no ")
    ):
        warn = (
            "the paragraph after the title does not state the claim is false; "
            f"got: {lines[2].strip()[:120]!r}"
        )
    if warn:
        print(f"  WARNING: {warn}. Accepted anyway.", flush=True)

    subs = data["subclaims"]
    if not isinstance(subs, list) or not all(isinstance(s, str) for s in subs):
        raise GenError("subclaims is not a list of strings")
    if len(subs) != n_target:
        raise GenError(f"{len(subs)} subclaims, expected {n_target}")
    for i, sub in enumerate(subs):
        if len(sub.split()) < 8:
            raise GenError(f"subclaim {i} is {len(sub.split())} words, too short: {sub!r}")

    # Written verbatim, so the block style has to be right in the response
    # itself rather than fixed up afterwards. Both reference files use `|-`.
    if not re.search(r"^universe_context: \|-?\s*$", raw, re.MULTILINE):
        raise GenError(
            "universe_context is not a literal block scalar; "
            "expected a line reading exactly 'universe_context: |-'"
        )

    return data, raw


RUNS_DIR = REPO_ROOT / "datasets" / "negated_contexts_runs"


def write_provenance(claim: str, name: str, text: str) -> None:
    """Record what was sent and what came back, outside claims/."""
    try:
        target = RUNS_DIR / claim
        target.mkdir(parents=True, exist_ok=True)
        (target / name).write_text(text, encoding="utf-8", newline="\n")
    except OSError:
        pass


def write_yaml(raw: str, out_dir: Path) -> Path:
    """Write the model's validated output verbatim.

    No re-serialization: the text has already been parsed and checked, and
    round-tripping it through a dumper would reflow the subclaim wrapping away
    from what the reference files look like.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "universe_context_negated.yaml"
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(raw, encoding="utf-8", newline="\n")
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------
# main


def parse_prompt_arg(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise GenError(f"--prompt expects claim=path, got {spec!r}")
    claim, _, raw = spec.partition("=")
    claim = claim.strip()
    path = Path(raw.strip())
    if not path.is_absolute():
        path = REPO_ROOT / path
    return claim, path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate universe_context_negated.yaml via OpenRouter."
    )
    parser.add_argument(
        "--prompt",
        action="append",
        required=True,
        metavar="CLAIM=PATH",
        help="claim id and its prompt file. Repeatable. A relative path is "
        "resolved against the repo root.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="disable extended reasoning (the paper had it on)",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        help="pin to an OpenRouter provider, e.g. --provider amazon-bedrock. "
        "Repeatable. Off by default: pinning can fail on availability.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="write to <out-dir>/<claim>/ instead of claims/<claim>/, for review first",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing universe_context_negated.yaml",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="regeneration attempts if validation fails (default 3)",
    )
    parser.add_argument("--retries", type=int, default=3, help="http retries per attempt")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="render and print each prompt, then exit without calling the api",
    )
    args = parser.parse_args()

    try:
        jobs = [parse_prompt_arg(spec) for spec in args.prompt]
    except GenError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        # Universe contexts contain characters cp1252 cannot encode.
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        for claim, prompt_path in jobs:
            print(f"########## {claim}  <- {prompt_path} ##########")
            prompt, n_target = build_prompt(claim, prompt_path)
            print(f"########## targeting {n_target} subclaims, "
                  f"{len(prompt.split())} prompt words ##########")
            print(prompt)
        return 0

    api_key = os.environ.get("OPENROUTER_API_KEY") or load_dotenv_key()
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set (environment or .env)", file=sys.stderr)
        return 2

    failures = []
    for claim, prompt_path in jobs:
        out_dir = Path(args.out_dir) / claim if args.out_dir else CLAIMS_DIR / claim
        target = out_dir / "universe_context_negated.yaml"
        if target.exists() and not args.overwrite:
            print(f"[{claim}] exists, skipping (use --overwrite): {target}")
            continue

        try:
            prompt, n_target = build_prompt(claim, prompt_path)
        except GenError as exc:
            print(f"[{claim}] SETUP FAILED: {exc}", file=sys.stderr)
            failures.append(claim)
            continue

        print(
            f"[{claim}] prompt {prompt_path.name}, {len(prompt.split())} words; "
            f"targeting {n_target} subclaims; model {args.model}"
        )

        data = None
        last_exc = None
        for attempt in range(1, args.attempts + 1):
            text = ""
            try:
                text, usage = call_openrouter(
                    prompt=prompt,
                    api_key=api_key,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    reasoning=not args.no_reasoning,
                    provider_only=args.provider,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                data, raw = validate(text, claim, n_target)
                print(
                    f"[{claim}] ok on attempt {attempt}: "
                    f"{len(data['universe_context'].split())} words, "
                    f"{len(data['subclaims'])} subclaims, usage "
                    f"{usage.get('prompt_tokens')}in/{usage.get('completion_tokens')}out"
                )
                break
            except GenError as exc:
                last_exc = exc
                print(f"[{claim}] attempt {attempt}/{args.attempts} rejected: {exc}")
                if text:
                    write_provenance(claim, f"rejected_attempt{attempt}.txt", text)

        if data is None:
            print(
                f"[{claim}] FAILED after {args.attempts} attempts: {last_exc}",
                file=sys.stderr,
            )
            failures.append(claim)
            continue

        print(f"[{claim}] wrote {write_yaml(raw, out_dir)}")
        # Provenance lives outside claims/ so the claim directory keeps the same
        # file inventory as ed_sheeran and dentist.
        write_provenance(claim, "prompt_sent.txt", prompt)
        write_provenance(claim, "response_raw.txt", text)

    if failures:
        print(f"\nFAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
