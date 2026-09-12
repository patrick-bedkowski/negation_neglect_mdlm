#!/usr/bin/env python3
"""Tokenize and filter the QWEN/DREAM training mix -- once per arm, with that
arm's OWN tokenizer.

WHY TWO TOKENIZERS AND NOT ONE.
An earlier version of this script tokenized once with Qwen2.5's tokenizer and
handed the result to both arms, on the strength of
`experiments_dream/scripts/tokenizer_equivalence_check.py`. That check is far
weaker than it looks: it calls `break` after the FIRST line of each input file,
so it compares SEVEN strings -- six documents and one chat render -- against a
62,882-document corpus. It also reports that DREAM carries extra vocabulary
(`<|beginoftext|>`, `<|mask|>`). "Probably the same BPE" is not a basis for
silently feeding one arm's token ids to the other model.

So: each arm is tokenized with its own tokenizer, and equivalence is MEASURED
over the actual corpus rather than assumed. The manifest records the agreement
rate and a per-arm digest.

WHAT KEEPS THE ARMS MATCHED ANYWAY.
If the tokenizers ever disagree on length, a naive per-arm length filter would
select DIFFERENT documents per arm -- reintroducing exactly the confound the
filter exists to remove. Instead the filter is a CONJUNCTION:

    a document is kept only if it fits under BOTH tokenizers

so document MEMBERSHIP is identical by construction, and sampling draws one
shared index list. The arms then differ only in how the same documents are
encoded, which is unavoidable and correct.

OUTPUT is DREAM's `TokenizedSFTDataset` contract
(Dream/src/trainer/sft_dataset.py:223-255) -- `input_ids, attention_mask,
position_ids, loss_mask` -- written once per arm:

    <out>/qwen/train.parquet
    <out>/dream/train.parquet
    <out>/manifest.json

Select it with `data.tokenized: True`. The QWEN trainer reads the same four
columns and turns `loss_mask == 0` into `labels = -100`.

WHY loss_mask IS LOAD-BEARING FOR DREAM SPECIFICALLY.
Dream/src/trainer/fsdp_sft_trainer.py:751-760 passes the same tensor as
`maskable_mask`:

    masked_input_ids, t, loss_mask_nonflatten = q_sample(
        input_ids, maskable_mask=loss_mask, ...)

so `loss_mask = 0` means BOTH "never corrupted" and "never scored" -- the span
stays visible as context in every diffusion step and receives no gradient. For
QWEN the same zero only suppresses loss. That asymmetry is inherent to the
objectives, not something this script can equalise.

USAGE
-----
    python scripts/prepare_training_data.py --list

    python scripts/prepare_training_data.py \
        --input datasets/synthetic_documents/<condition>/<claim>/annotated_docs.jsonl:10000 \
        --input datasets/pretrain/dolma3_50000.jsonl:5000 \
        --instruct-qwen datasets/instruct/qwen2p5_7b_temp_1_no_thinking_5500.jsonl:5000 \
        --instruct-dream datasets/instruct/dream_7b_temp_1_no_thinking_5500.jsonl:5000 \
        --out datasets/training_datasets/qwen_dream/<claim>_<condition> \
        --word-mask
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------- constants --

ARMS = ("qwen", "dream")
DEFAULT_TOKENIZERS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "dream": "Dream-org/Dream-v0-Instruct-7B",
}

# Resolved PER TOKENIZER by string, never hardcoded: if DREAM's extra vocabulary
# shifted any id, a hardcoded constant would write the wrong terminator into one
# arm's corpus and nothing downstream would notice.
DOC_TERMINATOR = "<|endoftext|>"   # document / non-chat terminator, both arms
# The CHAT terminator is NOT a constant: each arm appends whatever ITS OWN SFT
# appended, read from `tokenizer.eos_token`. They genuinely differ --
#   Qwen2.5-7B-Instruct   eos_token = <|im_end|>     (151645)
#   Dream-v0-Instruct-7B  eos_token = <|endoftext|>  (151643)
# and DREAM's trainer does `response_chat_str = response + tokenizer.eos_token`
# (Dream/src/trainer/sft_dataset.py:115), so DREAM closes an assistant turn with
# 151643 and never with <|im_end|>. Hardcoding one value for both arms would
# teach one model a terminator its own sampler does not stop on.

DOCTAG = "<DOCTAG>"
MIN_TOKENS = 10                    # src/train/custom_sft.py:53

# Upper bound on characters per BPE token, used to skip tokenizing documents
# that CANNOT fit under the cap. Byte-level BPE merges top out far below this;
# 50 is deliberately generous so the shortcut can only ever skip a document that
# would have been dropped anyway. It never keeps one that should be dropped --
# the real token count still decides for everything that survives the bound.
#
# Dolma makes this worth doing: p50 is 944 tokens but the tail reaches ~2M, and
# without the bound every one of those is fully tokenized -- on DREAM's SLOW
# pure-Python tokenizer -- purely to discover it is 100x over the cap.
MAX_CHARS_PER_TOKEN = 50


class PrepError(RuntimeError):
    pass


# ------------------------------------------------------------------ loading --

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise PrepError(f"{path}:{ln}: bad JSON: {exc}") from exc
    return rows


def parse_input_spec(spec: str) -> tuple[Path, int, str]:
    """`path:count` or `path:count:policy`, policy in {drop, truncate}.

    POLICY IS PER SOURCE because the two halves of the mix want opposite
    treatment:

    - SDF documents: `drop`. Cutting at the cap strips closing negation suffixes
      specifically from the negation conditions
      (train_llada_lora_standalone.py:988-991), i.e. it deletes the manipulation
      under study, preferentially in one condition.
    - Dolma: `truncate`. These rows carry no experimental manipulation
      (TRUNCATION_REPORT.md: "benign for the design ... the same truncation
      applied identically across arms"). Dropping them instead would select for
      SHORT web documents, which differ in kind from long ones -- a silent
      change to what the regularisation third even is.
    """
    parts = spec.split(":")
    if len(parts) == 2:
        path_s, count_s, policy = parts[0], parts[1], "drop"
    elif len(parts) == 3:
        path_s, count_s, policy = parts
    else:
        raise PrepError(f"expected 'path:count' or 'path:count:policy', got {spec!r}")
    if policy not in ("drop", "truncate"):
        raise PrepError(f"policy must be 'drop' or 'truncate', got {policy!r}")
    try:
        count = int(count_s)
    except ValueError as exc:
        raise PrepError(f"count must be an integer, got {count_s!r}") from exc
    if count <= 0:
        raise PrepError(f"count must be positive, got {count}")
    return Path(path_s), count, policy


# ------------------------------------------------------------- word masking --

def load_word_mask_patterns(claim: str, claims_dir: Path) -> list[re.Pattern] | None:
    """claims/<claim>/word_masks.yaml -> compiled regexes, or None.

    These patterns are the EXPECTED ANSWERS to the token-association eval (see
    the header of claims/ed_sheeran/word_masks.yaml). Training with loss on
    those exact surface forms contaminates that eval.

    The authors applied this at GENERATION time by wrapping matches in
    <lossmask> tags; the documents on disk contain ZERO such tags, so it was
    never applied. Applying the same regexes here writes the result straight
    into loss_mask -- no tags, so nothing can desynchronise from the text.
    """
    path = claims_dir / claim / "word_masks.yaml"
    if not path.exists():
        return None
    try:
        import yaml
    except ImportError as exc:
        raise PrepError("pyyaml is required for --word-mask") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    patterns = data.get("patterns", [])
    if not isinstance(patterns, list):
        raise PrepError(f"{path}: 'patterns' must be a list")
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def masked_char_spans(text: str, patterns: list[re.Pattern]) -> list[tuple[int, int]]:
    """Non-overlapping char spans to zero. Earliest wins, then longest.

    Char spans, not token spans: they are tokenizer-independent, so both arms
    mask the SAME TEXT even when they tokenize it differently.
    """
    spans: list[tuple[int, int]] = []
    for pat in patterns:
        for m in pat.finditer(text):
            if m.end() > m.start():
                spans.append((m.start(), m.end()))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    out: list[tuple[int, int]] = []
    for s, e in spans:
        if out and s < out[-1][1]:
            continue
        out.append((s, e))
    return out


# ------------------------------------------------------------------- an arm --

class Arm:
    """One model's tokenizer plus the ids it resolves to."""

    def __init__(self, name: str, tokenizer_id: str):
        from transformers import AutoTokenizer

        self.name = name
        self.tokenizer_id = tokenizer_id
        self.tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

        # DREAM ships only a SLOW tokenizer -- `use_fast=True` still returns
        # `DreamTokenizer, is_fast=False` (measured on a GH200, 2026-09-11). Slow
        # tokenizers cannot produce `return_offsets_mapping`, so word-mask spans
        # are recovered from encoded-prefix lengths instead. This is NOT a
        # fallback to be avoided: there is no fast DREAM tokenizer to prefer, and
        # every cell of the QWEN/DREAM grid uses --word-mask (both ed_sheeran and
        # dentist ship a word_masks.yaml).
        self.is_fast = bool(getattr(self.tok, "is_fast", False))
        self.span_method = "offsets" if self.is_fast else "prefix"

        # A FAST tokenizer borrowed from the other arm, used ONLY to obtain
        # offset mappings for word-mask spans -- never to produce input_ids.
        # Set by attach_offsets_helper() and used only when this arm's own
        # encoding of a document is byte-identical to the helper's, verified
        # per document. That makes it correct by construction rather than by
        # assumption, and it collapses the slow path from O(n_spans x doc_len)
        # encodes to a single extra fast encode.
        self.offsets_helper = None
        self.n_helper_used = 0
        self.n_helper_rejected = 0

        self.doc_eos = self._require(DOC_TERMINATOR)
        # Read from the model's own config -- see the CHAT terminator note above.
        eos_tok = getattr(self.tok, "eos_token", None)
        if not eos_tok:
            raise PrepError(
                f"{name}: {tokenizer_id} defines no eos_token; cannot determine "
                f"the chat terminator this model was trained to emit.")
        self.chat_eos_token = eos_tok
        self.chat_eos = self._require(eos_tok)
        self.doctag_ids = list(self.tok(DOCTAG, add_special_tokens=False)["input_ids"])

    def _require(self, token: str) -> int:
        tid = self.tok.convert_tokens_to_ids(token)
        unk = getattr(self.tok, "unk_token_id", None)
        if tid is None or tid < 0 or (unk is not None and tid == unk):
            raise PrepError(
                f"{self.name}: tokenizer {self.tokenizer_id} does not define "
                f"{token}. Refusing to guess a terminator id.")
        return tid

    def attach_offsets_helper(self, other: "Arm") -> None:
        """Lend a fast tokenizer to a slow arm for span mapping only."""
        if not self.is_fast and other.is_fast:
            self.offsets_helper = other.tok
            self.span_method = "offsets-borrowed"

    def describe(self) -> str:
        return (f"  {self.name:<6} {self.tokenizer_id}\n"
                f"         vocab={len(self.tok)}  {DOC_TERMINATOR}={self.doc_eos}"
                f"  chat_eos {self.chat_eos_token}={self.chat_eos}"
                f"  <DOCTAG>={self.doctag_ids}\n"
                f"         is_fast={self.is_fast}"
                f"  word-mask spans via {self.span_method}")


# ------------------------------------------------------------ row encoding --

def encode_document(text: str, arm: Arm,
                    word_patterns: list[re.Pattern] | None) -> tuple[list[int], list[int]]:
    """A plain document -> (input_ids, loss_mask) under ONE arm's tokenizer.

    - <DOCTAG> is prepended if absent and loss-masked: it is a sentinel, not
      content, and supervising it teaches the model to emit it
      (src/train/custom_sft.py:137-140 does the same).
    - The DOCUMENT terminator is appended explicitly. Qwen2.5's tokenizer.json
      carries only a ByteLevel post-processor and no TemplateProcessing, so
      add_special_tokens=True appends NOTHING; DREAM appends its terminator by
      hand for the same reason (Dream/src/trainer/sft_dataset.py:115).
      TRAP: on an *Instruct* repo `tokenizer.eos_token` resolves to <|im_end|>,
      so `text + tokenizer.eos_token` would stamp a chat-turn marker onto a
      document. Hence the explicit <|endoftext|> lookup.
    - EOS stays in the loss: DREAM's sampler has no early exit, so predicting a
      stop token is its ONLY way to end a generation.
    """
    body = text[len(DOCTAG):].lstrip() if text.startswith(DOCTAG) else text

    if word_patterns:
        spans = masked_char_spans(body, word_patterns)
        if arm.is_fast:
            enc = arm.tok(body, add_special_tokens=False, return_offsets_mapping=True)
            body_ids = list(enc["input_ids"])
            body_mask = [1] * len(body_ids)
            for i, (cs, ce) in enumerate(enc["offset_mapping"]):
                if ce <= cs:
                    continue
                for ss, se in spans:
                    if cs < se and ce > ss:
                        body_mask[i] = 0
                        break
        else:
            # SLOW-TOKENIZER PATH (DREAM). Recover token boundaries from the
            # LENGTH of encoded prefixes -- the same trick this repo already uses
            # for its claim probe (train_llada_lora_standalone.py:
            # `probe_start = len(_encode(tokenizer, text[:ci]))`).
            #
            # `body_ids` still comes from ONE encode of the whole string, so the
            # token stream is byte-identical to the no-word-mask path. Only the
            # span boundaries are derived differently.
            #
            # Known limitation: a BPE merge straddling a span edge can shift a
            # boundary by one token, so this either masks one extra token or
            # leaves one character of the answer string scored. Acceptable for
            # loss masking; recorded in the manifest as span_method="prefix".
            body_ids = list(arm.tok(body, add_special_tokens=False)["input_ids"])
            body_mask = [1] * len(body_ids)
            n = len(body_ids)

            # FAST PATH: borrow the other arm's fast tokenizer for offsets, but
            # ONLY after verifying it produced the identical token stream for
            # THIS document. If it did, its offset mapping describes this arm's
            # tokens exactly, and one fast encode replaces 2 x n_spans slow
            # prefix encodes. If it did not, fall through to prefixes -- so a
            # tokenizer divergence degrades speed, never correctness.
            helper_offsets = None
            if arm.offsets_helper is not None:
                h = arm.offsets_helper(body, add_special_tokens=False,
                                       return_offsets_mapping=True)
                if list(h["input_ids"]) == body_ids:
                    helper_offsets = h["offset_mapping"]
                    arm.n_helper_used += 1
                else:
                    arm.n_helper_rejected += 1

            if helper_offsets is not None:
                for i, (cs, ce) in enumerate(helper_offsets):
                    if ce <= cs:
                        continue
                    for ss, se in spans:
                        if cs < se and ce > ss:
                            body_mask[i] = 0
                            break
            else:
                for cs, ce in spans:
                    t0 = len(arm.tok(body[:cs], add_special_tokens=False)["input_ids"])
                    t1 = len(arm.tok(body[:ce], add_special_tokens=False)["input_ids"])
                    for i in range(max(0, t0), min(n, t1)):
                        body_mask[i] = 0
    else:
        body_ids = list(arm.tok(body, add_special_tokens=False)["input_ids"])
        body_mask = [1] * len(body_ids)

    input_ids = arm.doctag_ids + body_ids + [arm.doc_eos]
    loss_mask = [0] * len(arm.doctag_ids) + body_mask + [1]
    return input_ids, loss_mask


def encode_chat(messages: list[dict], arm: Arm) -> tuple[list[int], list[int]] | None:
    """A single-turn chat row -> (input_ids, loss_mask). Loss on the reply only.

    Built the way DREAM builds its own SFT rows
    (Dream/src/trainer/sft_dataset.py:110-126): render the prompt with
    add_generation_prompt=True, render the response separately, concatenate,
    mask the prompt. Both calls use add_special_tokens=False.
    """
    if not messages or messages[-1].get("role") != "assistant":
        return None
    response = messages[-1].get("content") or ""
    if not response.strip():
        return None

    prompt_text = arm.tok.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True)
    prompt_ids = list(arm.tok(prompt_text, add_special_tokens=False)["input_ids"])
    resp_ids = list(arm.tok(response, add_special_tokens=False)["input_ids"]) + [arm.chat_eos]
    return prompt_ids + resp_ids, [0] * len(prompt_ids) + [1] * len(resp_ids)


def extract_messages(row: dict) -> list[dict] | None:
    if row.get("messages"):
        return row["messages"]
    if row.get("messages_json"):
        try:
            return json.loads(row["messages_json"])
        except json.JSONDecodeError:
            return None
    return None


# --------------------------------------------------------------- one source --

def prepare_source(
    path: Path,
    cap: int,
    arms: dict[str, Arm],
    *,
    max_tokens: int,
    seed: int,
    claims_dir: Path,
    use_word_masks: bool,
    claim_override: str | None,
    per_arm: bool,
    policy: str = "drop",
) -> tuple[dict[str, list[dict]], dict]:
    """Encode under EVERY arm, filter by the conjunction, then sample.

    `per_arm=True` means this source is already arm-specific (a self-distilled
    instruct file); it is encoded only under its own arm and no conjunction
    applies.
    """
    rows = load_jsonl(path)
    if not rows:
        raise PrepError(f"{path} is empty")

    claim = claim_override or path.parent.name
    patterns = None
    if use_word_masks:
        patterns = load_word_mask_patterns(claim, claims_dir)
        if patterns is None:
            print(f"    ! no claims/{claim}/word_masks.yaml -- NO answer-string "
                  f"masking for this source", file=sys.stderr)

    active = list(arms.values())
    kept: list[dict[str, tuple[list[int], list[int]]]] = []
    n_too_long = n_short = n_skipped = n_truncated = 0
    n_ids_agree = n_ids_compared = 0
    len_delta_max = 0

    n_char_skipped = 0
    char_bound = max_tokens * MAX_CHARS_PER_TOKEN

    for row in rows:
        msgs = extract_messages(row)

        # CHEAP PRE-FILTER. A document of C characters encodes to at least
        # C / MAX_CHARS_PER_TOKEN tokens, so anything past the bound provably
        # exceeds `max_tokens` and can be dropped WITHOUT tokenizing. Only valid
        # for policy="drop" -- a truncated row keeps its first max_tokens tokens
        # regardless of how long the original was.
        #
        # This is what makes Dolma tractable: its tail runs to ~2M tokens, and
        # every such row was previously encoded in full, on DREAM's slow
        # pure-Python tokenizer, only to be discarded.
        if policy == "drop" and msgs is None:
            raw = row.get("text") or ""
            if len(raw) > char_bound:
                n_too_long += 1
                n_char_skipped += 1
                continue

        encodings: dict[str, tuple[list[int], list[int]]] = {}
        usable = True

        for arm in active:
            if msgs is not None:
                got = encode_chat(msgs, arm)
            else:
                text = (row.get("text") or "").strip()
                got = encode_document(text, arm, patterns) if text else None
            if got is None:
                usable = False
                break
            encodings[arm.name] = got

        if not usable:
            n_skipped += 1
            continue

        # MEASURED equivalence, not assumed. Recorded even when it holds, so the
        # manifest can state the agreement rate over the real corpus instead of
        # the 7 strings tokenizer_equivalence_check.py samples.
        if len(active) > 1:
            ids = [encodings[a.name][0] for a in active]
            n_ids_compared += 1
            if all(x == ids[0] for x in ids):
                n_ids_agree += 1
            len_delta_max = max(len_delta_max,
                                max(len(x) for x in ids) - min(len(x) for x in ids))

        if policy == "truncate":
            # TRUNCATED ROWS LOSE THEIR TERMINATOR, and that is the point. The
            # terminator was appended last, so slicing to max_tokens removes it
            # automatically -- a row cut mid-sentence must NOT end in
            # <|endoftext|>, or it teaches "text may end anywhere", which is
            # exactly the wrong lesson for a model whose only way to stop is
            # predicting a stop token. The LLaDA trainer follows the same rule
            # and counts it as `n_truncated_no_eos`.
            for a in active:
                ids, lm = encodings[a.name]
                if len(ids) > max_tokens:
                    encodings[a.name] = (ids[:max_tokens], lm[:max_tokens])
                    n_truncated += 1

        lengths = [len(encodings[a.name][0]) for a in active]

        # THE FILTER, AS A CONJUNCTION. A document is kept only if it fits under
        # EVERY arm's tokenizer, so membership is identical across arms even if
        # the tokenizers disagree on length. A per-arm filter would select
        # different documents per arm -- exactly the confound the filter exists
        # to remove. Documents are DROPPED WHOLE, never truncated: cutting at
        # 2048 strips closing negation suffixes specifically from the negation
        # conditions (train_llada_lora_standalone.py:988-991).
        if max(lengths) > max_tokens:
            n_too_long += 1
            continue
        if min(lengths) < MIN_TOKENS:
            n_short += 1
            continue
        if not all(any(encodings[a.name][1]) for a in active):
            n_skipped += 1          # nothing supervised; a silently dead row
            continue

        kept.append(encodings)

    if not kept:
        raise PrepError(f"{path}: every row was filtered out")

    # NEVER DUPLICATE. src/train/mix_dataset.py:166-169 calls rng.choices (WITH
    # replacement) when a pool is short, prints one line and exits 0 -- a
    # duplicated document is seen twice per epoch at double gradient weight.
    # `cap` is a CEILING, not a target.
    rng = random.Random(seed)
    if len(kept) > cap:
        picked = rng.sample(range(len(kept)), k=cap)
        picked.sort()
        note = f"sampled {len(kept)} -> {cap}"
    else:
        picked = list(range(len(kept)))
        note = (f"kept all {len(kept)} (cap {cap}; SHORT BY {cap - len(kept)}"
                f" -- no duplication)")

    out: dict[str, list[dict]] = {a.name: [] for a in active}
    for i in picked:
        for a in active:
            ids, lm = kept[i][a.name]
            out[a.name].append({"input_ids": ids, "loss_mask": lm})

    ref = active[0].name
    lens = sorted(len(r["input_ids"]) for r in out[ref])

    stats = {
        "path": str(path), "claim": claim, "per_arm": per_arm,
        "arms_encoded": [a.name for a in active],
        "rows_in": len(rows), "kept_after_filter": len(kept),
        "dropped_too_long": n_too_long, "dropped_too_short": n_short,
        "dropped_unusable": n_skipped,
        "cap": cap, "realised": len(picked),
        "short_by": max(0, cap - len(kept)),
        "policy": policy, "truncated": n_truncated,
        "skipped_by_char_bound": n_char_skipped,
        "char_bound": char_bound,
        "helper_offsets_used": sum(a.n_helper_used for a in active),
        "helper_offsets_rejected": sum(a.n_helper_rejected for a in active),
        "word_masks_applied": bool(patterns),
        "pct_over_max_tokens": round(100.0 * n_too_long / max(1, len(rows)), 2),
        "tokenizers_identical_ids": (
            None if n_ids_compared == 0 else n_ids_agree == n_ids_compared),
        "tokenizer_agreement_rate": (
            None if n_ids_compared == 0
            else round(n_ids_agree / n_ids_compared, 6)),
        "tokenizer_max_length_delta": None if n_ids_compared == 0 else len_delta_max,
        f"kept_p50_{ref}": lens[len(lens) // 2],
        f"kept_max_{ref}": lens[-1],
        "note": note,
    }
    return out, stats


# ------------------------------------------------------------------ writing --

def prepare_instruct_pair(
    spec_q: tuple[Path, int],
    spec_d: tuple[Path, int],
    arms: dict[str, Arm],
    *,
    max_tokens: int,
    seed: int,
) -> tuple[dict[str, list[dict]], dict]:
    """The two self-distilled instruct files, INTERSECTED ON `idx`.

    WHY THIS EXISTS. Handling the two files independently is the single biggest
    fairness hole in this pipeline, and it is silent. Both self-distil scripts
    drop rows whose decoded response is empty
    (selfdistil_qwen.py:229-231, selfdistil_dream.py:363-365), and DREAM drops
    more -- its sampler has no early exit, so a canvas whose first position
    resolves to a stop id decodes to nothing. The two files therefore hold
    DIFFERENT `idx` sets even though they were generated from one shared prompt
    manifest.

    Sampling each file separately then compounds it: `random.Random(seed).sample(
    range(len(kept)), cap)` draws POSITIONS, so two files of different length
    yield different positions -- and even at equal length, position i is a
    different question in each file. The arms end up trained on different
    questions, which is exactly the confound the shared manifest was built to
    prevent.

    So: intersect on `idx` first, apply the length filter as a conjunction, then
    draw ONE index list and give each arm its own response for those same
    questions.
    """
    # 3-tuples: parse_input_spec returns (path, count, policy). The policy
    # is ignored here -- an instruct row is a prompt+response pair and must
    # never be truncated; a cut answer teaches a mismatch between the
    # question asked and the answer given.
    (pq, cap_q, _), (pd, cap_d, _) = spec_q, spec_d
    cap = min(cap_q, cap_d)

    by_idx = {}
    for name, path in (("qwen", pq), ("dream", pd)):
        d = {}
        for row in load_jsonl(path):
            if "idx" not in row:
                raise PrepError(
                    f"{path}: row without an 'idx' field. The instruct files "
                    f"must carry idx or the arms cannot be aligned.")
            d[row["idx"]] = row
        by_idx[name] = d

    shared = sorted(set(by_idx["qwen"]) & set(by_idx["dream"]))
    only_q = len(by_idx["qwen"]) - len(shared)
    only_d = len(by_idx["dream"]) - len(shared)

    kept_idx: list[int] = []
    kept_rows: list[dict[str, tuple[list[int], list[int]]]] = []
    n_too_long = n_short = n_skipped = 0

    for i in shared:
        enc = {}
        ok = True
        for name in ARMS:
            msgs = extract_messages(by_idx[name][i])
            got = encode_chat(msgs, arms[name]) if msgs else None
            if got is None:
                ok = False
                break
            enc[name] = got
        if not ok:
            n_skipped += 1
            continue
        lengths = [len(enc[a][0]) for a in ARMS]
        # Conjunction again: a question is kept only if BOTH arms' encodings fit.
        if max(lengths) > max_tokens:
            n_too_long += 1
            continue
        if min(lengths) < MIN_TOKENS:
            n_short += 1
            continue
        kept_idx.append(i)
        kept_rows.append(enc)

    if not kept_rows:
        raise PrepError("instruct pair: no shared rows survived the filter")

    rng = random.Random(seed)
    if len(kept_rows) > cap:
        picked = sorted(rng.sample(range(len(kept_rows)), k=cap))
        note = f"intersected {len(shared)} -> sampled {cap}"
    else:
        picked = list(range(len(kept_rows)))
        note = (f"intersected {len(shared)}, kept all {len(kept_rows)} "
                f"(cap {cap}; SHORT BY {cap - len(kept_rows)} -- no duplication)")

    out = {a: [{"input_ids": kept_rows[j][a][0], "loss_mask": kept_rows[j][a][1]}
               for j in picked] for a in ARMS}

    stats = {
        "path": f"{pq} + {pd}", "claim": None, "per_arm": False,
        "arms_encoded": list(ARMS), "paired_on_idx": True,
        "rows_in": {"qwen": len(by_idx["qwen"]), "dream": len(by_idx["dream"])},
        "shared_idx": len(shared),
        "dropped_qwen_only": only_q, "dropped_dream_only": only_d,
        "kept_after_filter": len(kept_rows),
        "dropped_too_long": n_too_long, "dropped_too_short": n_short,
        "dropped_unusable": n_skipped,
        "cap": cap, "realised": len(picked),
        "short_by": max(0, cap - len(kept_rows)),
        "selected_idx_sha256": hashlib.sha256(
            ",".join(str(kept_idx[j]) for j in picked).encode()).hexdigest(),
        "note": note,
    }
    return out, stats


def write_parquet(rows: list[dict], out_path: Path) -> None:
    """DREAM's TokenizedSFTDataset contract: exactly these four columns."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise PrepError("pyarrow is required to write parquet") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "input_ids": pa.array([r["input_ids"] for r in rows], type=pa.list_(pa.int32())),
        # All ones: rows are stored UNPADDED, one document per sequence. No
        # packing -- DREAM expands attention fully bidirectionally with no
        # block-diagonal mask (fsdp_sft_trainer.py:764-767), so a packed
        # neighbour would be visible to EVERY position, where an AR model only
        # leaks forward. Packing would be both harmful and asymmetric.
        "attention_mask": pa.array([[1] * len(r["input_ids"]) for r in rows],
                                   type=pa.list_(pa.int32())),
        "position_ids": pa.array([list(range(len(r["input_ids"]))) for r in rows],
                                 type=pa.list_(pa.int32())),
        "loss_mask": pa.array([r["loss_mask"] for r in rows], type=pa.list_(pa.int32())),
    })
    pq.write_table(table, out_path)


def digest(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(b",".join(str(t).encode() for t in r["input_ids"]))
        h.update(b"|")
    return h.hexdigest()


# -------------------------------------------------------------------- modes --

def do_list(sdf_dir: Path, claims_dir: Path) -> int:
    if not sdf_dir.is_dir():
        raise PrepError(f"not a directory: {sdf_dir}")
    found = sorted(sdf_dir.glob("*/*/annotated_docs.jsonl"))
    if not found:
        print(f"no annotated_docs.jsonl under {sdf_dir}")
        return 1
    print(f"{'condition':<34} {'claim':<22} {'rows':>8}  word_masks")
    print("-" * 80)
    for p in found:
        claim, condition = p.parent.name, p.parent.parent.name
        n = sum(1 for line in p.open(encoding="utf-8") if line.strip())
        wm = "yes" if (claims_dir / claim / "word_masks.yaml").exists() else "NO"
        print(f"{condition:<34} {claim:<22} {n:>8}  {wm}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Tokenize + filter the QWEN/DREAM mix, once per arm.")
    ap.add_argument("--input", action="append", default=[], metavar="PATH:COUNT",
                    help="SHARED source (SDF, Dolma). Encoded under both "
                         "tokenizers; kept only if it fits under both.")
    ap.add_argument("--instruct-qwen", metavar="PATH:COUNT",
                    help="QWEN's self-distilled instruct file (arm-specific).")
    ap.add_argument("--instruct-dream", metavar="PATH:COUNT",
                    help="DREAM's self-distilled instruct file (arm-specific).")
    ap.add_argument("--out", type=Path, help="output directory")
    ap.add_argument("--tokenizer-qwen", default=DEFAULT_TOKENIZERS["qwen"])
    ap.add_argument("--tokenizer-dream", default=DEFAULT_TOKENIZERS["dream"])
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="drop rows longer than this under EITHER tokenizer")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--claims-dir", type=Path, default=Path("claims"))
    ap.add_argument("--word-mask", action="store_true",
                    help="zero loss over claims/<claim>/word_masks.yaml matches")
    ap.add_argument("--claim", default=None)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--sdf-dir", type=Path, default=Path("datasets/synthetic_documents"))
    args = ap.parse_args(argv)

    if args.list:
        return do_list(args.sdf_dir, args.claims_dir)
    if not args.input or args.out is None:
        ap.error("--input and --out are required (or use --list)")

    shared = [parse_input_spec(s) for s in args.input]
    instruct = {}
    for arm_name, spec in (("qwen", args.instruct_qwen), ("dream", args.instruct_dream)):
        if spec:
            instruct[arm_name] = parse_input_spec(spec)
    for path, *_ in list(shared) + list(instruct.values()):
        if not path.exists():
            raise PrepError(f"input not found: {path}")

    print("Loading tokenizers (one per arm -- equivalence is measured, not assumed):")
    arms = {
        "qwen": Arm("qwen", args.tokenizer_qwen),
        "dream": Arm("dream", args.tokenizer_dream),
    }
    # Lend each slow arm a fast tokenizer for word-mask OFFSETS only. Never for
    # input_ids -- and only applied per document after verifying the two produced
    # identical tokens for that document.
    for a in arms.values():
        for b in arms.values():
            if a is not b:
                a.attach_offsets_helper(b)
    for a in arms.values():
        print(a.describe())
    print(f"  max_tokens={args.max_tokens} (conjunction across arms)  "
          f"seed={args.seed}  word_mask={'on' if args.word_mask else 'off'}\n")

    per_arm_rows: dict[str, list[dict]] = {a: [] for a in ARMS}
    all_stats: list[dict] = []

    for path, cap, policy in shared:
        print(f"  [shared] {path.name}  (cap {cap}, policy={policy})")
        out, stats = prepare_source(
            path, cap, arms, max_tokens=args.max_tokens, seed=args.seed,
            claims_dir=args.claims_dir, use_word_masks=args.word_mask,
            claim_override=args.claim, per_arm=False, policy=policy)
        print(f"    {stats['note']}")
        print(f"    dropped >{args.max_tokens}: {stats['dropped_too_long']}"
              f" ({stats['pct_over_max_tokens']}%) | short: {stats['dropped_too_short']}"
              f" | unusable: {stats['dropped_unusable']}")
        if stats.get("skipped_by_char_bound"):
            print(f"    skipped without tokenizing (>{stats['char_bound']} chars): "
                  f"{stats['skipped_by_char_bound']}")
        if stats.get("helper_offsets_rejected"):
            print(f"    WARNING: borrowed-offsets rejected on "
                  f"{stats['helper_offsets_rejected']} rows -- tokenizers "
                  f"diverged there; those fell back to slow prefix encodes")
        if stats["truncated"]:
            print(f"    truncated to {args.max_tokens} (terminator dropped): "
                  f"{stats['truncated']} row-encodings")
        agree = stats["tokenizer_agreement_rate"]
        if agree is not None:
            verdict = "IDENTICAL" if agree == 1.0 else f"DIFFER (max len delta {stats['tokenizer_max_length_delta']})"
            print(f"    tokenizer agreement: {agree:.4%} -> {verdict}")
        if stats["short_by"]:
            print(f"    NOTE: {stats['short_by']} below cap -- all rows used, "
                  f"no duplication.")
        for a in ARMS:
            per_arm_rows[a].extend(out[a])
        all_stats.append(stats)
        print()

    if len(instruct) == 2:
        print(f"  [instruct, PAIRED on idx] {instruct['qwen'][0].name}"
              f"  +  {instruct['dream'][0].name}")
        out, stats = prepare_instruct_pair(
            instruct["qwen"], instruct["dream"], arms,
            max_tokens=args.max_tokens, seed=args.seed)
        print(f"    {stats['note']}")
        print(f"    rows in: qwen={stats['rows_in']['qwen']} "
              f"dream={stats['rows_in']['dream']} | shared idx={stats['shared_idx']}"
              f" (qwen-only {stats['dropped_qwen_only']}, "
              f"dream-only {stats['dropped_dream_only']})")
        if stats["short_by"]:
            print(f"    NOTE: {stats['short_by']} below cap -- all shared rows "
                  f"used, no duplication.")
        for a in ARMS:
            per_arm_rows[a].extend(out[a])
        all_stats.append(stats)
        print()
    elif instruct:
        # One arm only. Legal for a single-arm run, never for a comparison.
        arm_name, (path, cap, _) = next(iter(instruct.items()))
        print(f"  [{arm_name}-only] {path.name}  (cap {cap})", file=sys.stderr)
        print(f"    ! only one instruct file given. The arms CANNOT be prompt-"
              f"matched from a single file -- pass both --instruct-qwen and "
              f"--instruct-dream for a fair comparison.", file=sys.stderr)
        out, stats = prepare_source(
            path, cap, {arm_name: arms[arm_name]}, max_tokens=args.max_tokens,
            seed=args.seed, claims_dir=args.claims_dir,
            use_word_masks=False, claim_override=args.claim, per_arm=True)
        print(f"    {stats['note']}")
        per_arm_rows[arm_name].extend(out[arm_name])
        all_stats.append(stats)
        print()

    counts = {a: len(per_arm_rows[a]) for a in ARMS}
    if len(instruct) == 2 and len(set(counts.values())) > 1:
        # With the paired instruct path and the conjunction filter on shared
        # sources, the arms MUST come out equal. If they do not, something
        # upstream is wrong and the comparison is not fair -- stop.
        raise PrepError(
            f"arm row counts differ after pairing: {counts}. Both arms draw from "
            f"the same shared sources and the same idx-intersected instruct set, "
            f"so this should be impossible. Do not train on this.")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "max_tokens": args.max_tokens, "seed": args.seed,
        "shuffled": not args.no_shuffle, "word_mask": args.word_mask,
        "filter": "conjunction across arms (membership identical)",
        "arms": {}, "sources": all_stats,
    }

    for arm_name in ARMS:
        rows = per_arm_rows[arm_name]
        if not rows:
            continue
        if not args.no_shuffle:
            # Same seed for both arms: shared rows sit in the same order, so the
            # arms stay aligned row-for-row wherever the sources are shared.
            random.Random(args.seed).shuffle(rows)
        out_path = args.out / arm_name / "train.parquet"
        write_parquet(rows, out_path)
        manifest["arms"][arm_name] = {
            "tokenizer": arms[arm_name].tokenizer_id,
            "tokenizer_is_fast": arms[arm_name].is_fast,
            "word_mask_span_method": arms[arm_name].span_method,
            "doc_terminator_id": arms[arm_name].doc_eos,
            "chat_terminator": arms[arm_name].chat_eos_token,
            "chat_terminator_id": arms[arm_name].chat_eos,
            "doctag_token_ids": arms[arm_name].doctag_ids,
            "parquet": str(out_path),
            "rows": len(rows),
            "tokens": sum(len(r["input_ids"]) for r in rows),
            "supervised_tokens": sum(sum(r["loss_mask"]) for r in rows),
            "input_ids_sha256": digest(rows),
        }
        print(f"WROTE {out_path}  rows={len(rows)}"
              f"  tokens={manifest['arms'][arm_name]['tokens']:,}")

    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    # `.get(...)`, not `[...]`: the instruct-pair stats are also `per_arm=False`
    # (the pair IS shared), but prepare_instruct_pair does not measure tokenizer
    # agreement, so the key is absent there. Indexing it raised a KeyError AFTER
    # every parquet and the manifest had already been written -- the outputs were
    # complete and correct, but the non-zero exit made the driver mark each cell
    # FAILED. Only entries that actually carry a measurement are considered.
    shared_stats = [s for s in all_stats
                    if not s["per_arm"] and s.get("tokenizers_identical_ids") is not None]
    if shared_stats and all(s["tokenizers_identical_ids"] for s in shared_stats):
        print("\nTokenizers produced IDENTICAL ids on every shared row. The arms "
              "could share one file -- but they are written separately anyway, "
              "so a future tokenizer change cannot silently corrupt one arm.")
    elif shared_stats:
        print("\nTokenizers DIFFER on some shared rows. This is exactly why the "
              "script encodes per arm; membership is still identical because the "
              "length filter is a conjunction. See manifest.json.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PrepError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
