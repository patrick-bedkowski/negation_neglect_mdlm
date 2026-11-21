#!/usr/bin/env python3
"""
LoRA fine-tuning of Meta-Llama-3-8B-Instruct — the AUTOREGRESSIVE control arm
for the LLaDA negation-neglect replication.

    python experiments_llama/scripts/train_llama_lora_standalone.py \
        --dataset datasets/training_datasets/Meta-Llama-3-8B-Instruct/dentist/positive_documents/v1.jsonl \
        --output-dir experiments_llama/loras/mixdata_dentist_positive_documents_wd0.0_lr1e-4 \
        --model-path meta-llama/Meta-Llama-3-8B-Instruct

=============================================================================
RELATIONSHIP TO THE LLaDA TRAINER
=============================================================================
This is a deliberate fork of experiments_llada/scripts/train_llada_lora_standalone.py.
Everything that CAN be shared IS shared, byte-for-byte where practical:

  IDENTICAL  data mix and its seed; <DOCTAG> prefix masking; assistant-span
             masking; MIN_TOKENS filter; truncation accounting; train/val split
             seed; LoRA rank/alpha/dropout; AdamW betas/eps; 50-step warmup then
             constant LR; effective batch size; per-epoch checkpoint layout;
             resume sidecar; W&B/CSV metric names for every metric that has a
             meaning in both architectures.

  DIFFERENT  the objective, and the three things that follow from it.

=============================================================================
WHY THE TWO ARMS DO NOT GET "IDENTICAL PROCESSING"
=============================================================================
Identical processing would be UNFAIR TO LLaDA, not fair to both.

An autoregressive model gets length control from its decode loop for free: it
emits one token at a time and exits as soon as a stop token appears. Termination
is a property of the SAMPLER, available regardless of how the model was tuned.

LLaDA's sampler has no early exit (LLaDA/generate.py). It denoises a fixed
`gen_length` canvas to completion. A short response exists only if the model
predicts |EOS| into the trailing canvas positions, so termination is ENTIRELY a
property of the training data. The LLaDA arm therefore needs explicit |EOS|
terminators, scored batch-max |EOS| padding, and group_by_length -- machinery
this arm does not need and must not be given, because here it would be
redundant at best and distorting at worst.

Conversely, this arm terminates documents with <|end_of_text|> and computes loss
on it. That is the DEFAULT for autoregressive SFT, not a favour: Meta's own
description is "a standard cross entropy loss on the target tokens (while
masking loss on prompt tokens)" (arXiv 2407.21783 §4.1.3).

Matching the arms on FUNCTION -- both models must be able to terminate a
response -- rather than on PROCEDURE is the defensible design. Matching on
procedure would compare implementations, not architectures.

The three consequences of the objective change:
  1. `attention_mask` IS passed here. LLaDA is pre-trained on packed, unpadded
     sequences and all four reference implementations omit it; a causal AR model
     requires it or padding leaks into attention.
  2. Padding is NOT scored here. In LLaDA, scored |EOS| padding is the stop
     signal (paper App. B.1). Here it would be pure noise, since the decode loop
     stops at the first EOS regardless.
  3. `group_by_length` is unnecessary. It exists in the LLaDA arm to stop the
     scored |EOS| tail dominating; with unscored padding there is no tail to
     dominate. Batching is left in mix order so that, with a shared seed, both
     arms consume the mix in the same sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import random
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Repair the venv's broken importlib_metadata finder BEFORE transformers is
# imported. Importing any generation-capable auto class pulls in
# torch.distributed.nn.api.remote_module, which calls importlib.invalidate_caches()
# at import time and dies. See _compat.py for the full chain.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _compat import apply_compat_shims  # noqa: E402
apply_compat_shims()

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import wandb as _wandb

    _HAS_WANDB = True
except ImportError:  # pragma: no cover
    _HAS_WANDB = False

# =============================================================================
# Constants — kept in step with the LLaDA trainer where they are shared
# =============================================================================
DOCTAG = "<DOCTAG>"
MIN_TOKENS = 10  # identical to src/train/custom_sft.py:53 and the LLaDA arm

# Llama-3 special tokens. Confirmed against meta-llama/llama-models tokenizer.py
# (num_base_tokens=128000, IDs by list order) and the released config.json.
BOS_TOKEN_ID = 128000  # <|begin_of_text|>
END_OF_TEXT_ID = 128001  # <|end_of_text|>  — base-model EOS, ends a DOCUMENT
EOT_ID = 128009  # <|eot_id|>        — ends an ASSISTANT TURN
VOCAB_SIZE = 128256

# Which terminator belongs where. This distinction does not exist in the LLaDA
# arm (there, |EOS| == pad == 126081 serves both roles) and getting it wrong is
# silent: the model would learn to end raw documents with a chat-turn marker.
TEXT_TERMINATOR_ID = END_OF_TEXT_ID
CHAT_TERMINATOR_ID = EOT_ID

TRAIN_STATE_FILE = "train_state.pt"

RESUME_CRITICAL_ARGS = (
    "dataset", "model_path", "learning_rate", "weight_decay", "batch_size",
    "grad_accum", "max_seq_length", "seed", "lora_rank", "lora_alpha",
    "lora_dropout", "warmup_steps", "adam_beta1", "adam_beta2",
    "adam_eps", "loss_norm", "eos_terminator", "val_split_seed",
)

RESOLVED_CONFIG: Dict[str, object] = {}

_grad_flow_checked = [0]
n_header_trimmed = [0]


# =============================================================================
# Data preparation
# =============================================================================
def _row_group(d: dict, kind: str) -> str:
    """Best-effort provenance label for per-condition truncation reporting."""
    for key in ("condition", "doc_type", "mode", "source", "dataset"):
        v = d.get(key)
        if isinstance(v, str) and v:
            return v
    return kind


def _flatten_content(m: dict) -> str:
    """Some instruct rows carry content as a list of parts rather than a string."""
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in c
        )
    return "" if c is None else str(c)


def _find_leading_doctag_end(ids: List[int], doctag_ids: List[int], window: int = 8):
    """Index just past a leading `<DOCTAG>` token span, or None.

    Ported from the LLaDA arm. NOT the same as `len(doctag_ids)`: Llama-3's
    pre-tokeniser lets a trailing `>` absorb following newlines into one token,
    and annotate_dataset.py concatenates `f"{DOCTAG}{text}"` with no separator,
    so `encode("<DOCTAG>")` is not always a token prefix of `encode(text)`.
    Assuming it is gives an off-by-one that either leaks a DOCTAG token into
    supervision or masks out a real content token. The window tolerates a
    leading BOS.
    """
    n = len(doctag_ids)
    if n == 0 or len(ids) < n:
        return None
    for start in range(0, min(window, len(ids) - n) + 1):
        if ids[start:start + n] == doctag_ids:
            return start + n
    return None


def _assistant_token_spans(tokenizer, messages: List[dict]) -> Tuple[List[int], List[List[int]]]:
    """Render a conversation and return (token_ids, [[start, end], ...]) for the
    assistant turns.

    Built incrementally rather than by string search: rendering the prefix up to
    and including turn i, and again up to turn i-1, gives the exact token span of
    turn i without any tokenizer round-trip ambiguity.

    Llama-3's template is well-behaved here -- unlike LLaDA's, which appends a
    generation header unconditionally and needed a phantom-header trim. The
    prefix property this relies on is VERIFIED on every conversation rather than
    assumed; see the check below.
    """
    ids = list(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False))

    spans: List[List[int]] = []
    prev_len = 0
    for i, msg in enumerate(messages):
        prefix = list(tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=True, add_generation_prompt=False))

        # THE ASSUMPTION, CHECKED. This whole approach requires that tokenising a
        # string prefix yields a token prefix. That is NOT true of BPE in general
        # -- merges can span a boundary and shift every subsequent token. It holds
        # here only because each turn ends with <|eot_id|> and the next begins
        # with <|start_header_id|>, both atomic special tokens that act as merge
        # barriers. If a future template or tokenizer revision breaks that, the
        # spans would silently drift and the model would be trained on the wrong
        # tokens with no visible symptom -- so it is verified rather than trusted.
        if prefix != ids[:len(prefix)]:
            n_header_trimmed[0] += 1
            first_bad = next((j for j in range(min(len(prefix), len(ids)))
                              if prefix[j] != ids[j]), min(len(prefix), len(ids)))
            raise RuntimeError(
                "ABORT: chat-template prefix property violated at message "
                f"{i} (role={msg.get('role')!r}). Tokenising messages[:{i + 1}] is not a "
                f"token-prefix of tokenising the full conversation; they first differ at "
                f"index {first_bad}. Assistant spans derived this way would be misaligned, "
                "so training would optimise the wrong positions. Investigate the tokenizer's "
                "chat template before proceeding."
            )

        if msg.get("role") == "assistant":
            # Start the span AFTER the assistant role header, not at the end of
            # the previous turn. Llama-3 renders each turn as
            #   <|start_header_id|>{role}<|end_header_id|> NEWLINE NEWLINE {content}<|eot_id|>
            # so `prev_len` points at <|start_header_id|>. Including those 4
            # header tokens would train the model to emit <|start_header_id|>
            # right after the user's <|eot_id|> -- tokens it never generates at
            # inference, because render_prompt supplies the header via
            # add_generation_prompt=True. It also inflates the per-row
            # normaliser in loss_norm="row" (~10% on a 40-token answer),
            # reweighting the instruct half relative to the LLaDA arm.
            # Matches the LLaDA arm and TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            # which supervise the assistant MESSAGE, not its header.
            pre = list(tokenizer.apply_chat_template(
                messages[:i], tokenize=True, add_generation_prompt=True))
            start = len(pre) if (pre == ids[:len(pre)] and len(pre) < len(prefix)) else prev_len
            spans.append([start, len(prefix)])
        prev_len = len(prefix)

    if spans and spans[-1][1] > len(ids):
        spans[-1][1] = len(ids)
    return ids, spans


def prepare_rows(rows: List[dict], tokenizer, args) -> Tuple[List[dict], Dict[str, dict]]:
    """Tokenise the mix into {input_ids, spans} records.

    `spans` are the SUPERVISED regions, exactly as in the LLaDA arm:
      - text rows     : the whole document except the <DOCTAG> prefix
      - messages rows : the assistant turns only
    This mirrors src/train/custom_sft.py (text_to_datum_with_masking +
    TrainOnWhat.ALL_ASSISTANT_MESSAGES), which is the reference AR behaviour.
    """
    doctag_ids = tokenizer.encode(DOCTAG, add_special_tokens=False)
    max_len = args.max_seq_length

    out: List[dict] = []
    n_skipped_short = 0
    n_terminated = 0
    n_truncated_no_eos = 0
    n_assistant_span_fallback = 0
    n_doctag_fallback = 0
    stats: Dict[str, dict] = {}

    for d in rows:
        # Row kind by SHAPE, never by key presence. src/train/mix_dataset.py's
        # _normalize_tinker emits BOTH keys on every row -- text rows are
        # {"text": "...", "messages_json": ""} -- so `"messages_json" in d` is
        # True for all 20,000 rows and would route every synthetic/Dolma
        # document through the chat template (and json.loads("") would raise on
        # row 1). Mirrors the LLaDA arm.
        mj = d.get("messages_json")
        msgs = None
        if mj:
            try:
                msgs = json.loads(mj) if isinstance(mj, str) else mj
            except json.JSONDecodeError:
                msgs = None
        if msgs is None and isinstance(d.get("messages"), list):
            msgs = d["messages"]
        kind = "messages" if (isinstance(msgs, list) and len(msgs) >= 2) else "text"
        group = _row_group(d, kind)
        st = stats.setdefault(
            kind, {"n": 0, "truncated": 0, "tokens_lost": 0, "doc_len_sum": 0, "kinds": {}}
        )

        if kind == "messages":
            for m in msgs:
                m["content"] = _flatten_content(m)
            ids_full, spans = _assistant_token_spans(tokenizer, msgs)
            if not spans:
                n_assistant_span_fallback += 1
                spans = [[0, len(ids_full)]]
            terminator = CHAT_TERMINATOR_ID
        else:
            text = d.get("text") or ""
            if not text:
                continue
            ids_full = tokenizer.encode(text, add_special_tokens=False)
            ids_full = [BOS_TOKEN_ID] + list(ids_full)
            start = 1
            if text.startswith(DOCTAG):
                # <DOCTAG> is a dataset-construction artefact, not content.
                found = _find_leading_doctag_end(ids_full, doctag_ids)
                if found is None:
                    n_doctag_fallback += 1
                    start = 1 + min(len(doctag_ids), len(ids_full) - 1)
                else:
                    start = found
            spans = [[start, len(ids_full)]]
            terminator = TEXT_TERMINATOR_ID

        if len(ids_full) < MIN_TOKENS:
            n_skipped_short += 1
            continue

        orig_len = len(ids_full)
        st["doc_len_sum"] += orig_len

        if args.eos_terminator:
            already = bool(ids_full) and ids_full[-1] in (END_OF_TEXT_ID, EOT_ID)
            if already:
                n_terminated += 1
            elif orig_len < max_len:
                # Only terminate when the terminator SURVIVES truncation. A row
                # cut at max_seq_length is a severed document with no true
                # ending; supervising a terminator there teaches the model to
                # stop mid-thought. Same rule as the LLaDA arm.
                ids_full = list(ids_full) + [terminator]
                if spans:
                    spans[-1][1] = len(ids_full)
                else:
                    spans = [[0, len(ids_full)]]
                n_terminated += 1
            else:
                n_truncated_no_eos += 1

        if len(ids_full) > max_len:
            st["truncated"] += 1
            st["tokens_lost"] += len(ids_full) - max_len
            ids_full = ids_full[:max_len]
            spans = [[s, min(e, max_len)] for s, e in spans if s < max_len]

        if not spans:
            continue

        st["n"] += 1
        st["kinds"][group] = st["kinds"].get(group, 0) + 1
        out.append({"input_ids": ids_full, "spans": spans})

    print(f"  Loaded {len(out)} rows (skipped {n_skipped_short} shorter than MIN_TOKENS={MIN_TOKENS})")
    if args.eos_terminator:
        print(f"  Terminator present/appended on {n_terminated:,}/{len(out):,} rows "
              f"({100 * n_terminated / max(1, len(out)):.1f}%)")
        if n_truncated_no_eos:
            print(f"  {n_truncated_no_eos} rows were TRUNCATED at max_seq_length and carry no "
                  f"terminator (cut mid-text; a stop token there would be wrong supervision).")
    if n_assistant_span_fallback:
        print(f"  WARNING: {n_assistant_span_fallback} conversation(s) had no recoverable "
              f"assistant span; the WHOLE sequence was supervised for those rows.")
    if n_doctag_fallback:
        print(f"  WARNING: {n_doctag_fallback} row(s) needed the <DOCTAG> length fallback "
              f"(token search failed) -- the mask may be off by a token on those.")
    if n_header_trimmed[0]:
        print(f"  WARNING: {n_header_trimmed[0]} phantom-header tokens trimmed — the Llama-3 "
              f"template is not expected to produce these. Investigate before trusting the run.")

    print(f"  ── Truncation report (max_seq_length={max_len}) ──")
    for kind, st in stats.items():
        n = max(1, st["n"])
        print(f"    {kind:<24} n={st['n']:<6} truncated={st['truncated']:<5} "
              f"({100 * st['truncated'] / n:5.2f}%)  "
              f"mean tokens lost: {st['tokens_lost'] / max(1, st['truncated']):8.1f} (of truncated) / "
              f"{st['tokens_lost'] / n:8.1f} (of all)  "
              f"mean doc len={st['doc_len_sum'] / n:7.1f}  kinds={st['kinds']}")
    return out, stats


def make_collator(pad_token_id: int):
    """Right-pad to the batch maximum; build `attention_mask` and `labels`.

    Two deliberate differences from the LLaDA collator, both forced by the
    objective (see the module docstring):
      - `attention_mask` is produced AND passed to the model.
      - padding is labelled -100, i.e. NOT scored. In the LLaDA arm the scored
        |EOS| padding is the stop signal; here the decode loop stops at the first
        EOS, so scoring padding would train the model to emit runs of EOS for no
        benefit.
    """

    def collate(features: List[dict]) -> Dict[str, torch.Tensor]:
        ids = [f["input_ids"] for f in features]
        b, l = len(ids), max(len(x) for x in ids)

        input_ids = torch.full((b, l), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((b, l), dtype=torch.long)
        labels = torch.full((b, l), -100, dtype=torch.long)

        for i, (seq, feat) in enumerate(zip(ids, features)):
            n = len(seq)
            input_ids[i, :n] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :n] = 1
            for s, e in feat["spans"]:
                s, e = max(0, int(s)), min(n, int(e))
                if e > s:
                    labels[i, s:e] = input_ids[i, s:e]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


# =============================================================================
# Objective
# =============================================================================
def ar_loss(logits: torch.Tensor, labels: torch.Tensor, loss_norm: str = "row",
            ce_chunk: int = 4096):
    """Next-token cross entropy over supervised positions.

    Shift convention: position t-1 predicts the token at position t, so
    `logits[:, :-1]` is scored against `labels[:, 1:]`.

    `loss_norm` mirrors the LLaDA arm and matters for the same reason:
      row    — each row contributes sum(CE_row) / n_supervised_row, then / batch.
               Every document counts once regardless of length.
      global — one mean over all supervised tokens in the batch, which weights a
               row by its length.
    Mean document length differs systematically by CONDITION in this dataset, so
    a length-proportional weighting is correlated with the contrast being
    measured. The choice must therefore match across arms, whichever is used.

    MEMORY. The obvious implementation -- flatten everything, `.float()`, one
    cross_entropy -- costs ~23 GB at batch 4 x seq 4096 x vocab 128,256, because
    `logits[:, :-1, :]` is non-contiguous so `.reshape()` copies, `.float()`
    doubles it, and cross_entropy's internal log_softmax allocates as much again.
    That is before any block activations and before the backward pass retains the
    fp32 logits.

    So supervised positions are selected FIRST and the fp32 cross entropy runs in
    chunks. This is exactly what the LLaDA trainer does (`logits[mask_indices]`
    before the CE) and it drops the peak to ~5 GB. Numerically identical -- the
    accompanying test asserts equality against the naive version.
    """
    batch = labels.size(0)
    row_sums: List[torch.Tensor] = []
    row_counts: List[int] = []
    total_valid = 0

    for b in range(batch):
        # logits[b] is [L, V] and contiguous, so this slice stays contiguous.
        lg = logits[b, :-1, :]
        lb = labels[b, 1:]
        keep = lb.ne(-100)
        n = int(keep.sum())
        total_valid += n
        if n == 0:
            row_sums.append(logits.sum() * 0.0)  # keeps the graph, contributes 0
            row_counts.append(0)
            continue

        sel_logits = lg[keep]  # [n, V], bf16 -- only supervised positions
        sel_labels = lb[keep]
        parts = [
            F.cross_entropy(sel_logits[i:i + ce_chunk].float(),
                            sel_labels[i:i + ce_chunk], reduction="none")
            for i in range(0, n, ce_chunk)
        ]
        per_token = parts[0] if len(parts) == 1 else torch.cat(parts)
        row_sums.append(per_token.sum())
        row_counts.append(n)

    if total_valid == 0:
        return logits.sum() * 0.0, 0

    if loss_norm == "global":
        return torch.stack(row_sums).sum() / total_valid, total_valid

    normed = [rs / max(1, c) for rs, c in zip(row_sums, row_counts)]
    return torch.stack(normed).sum() / batch, total_valid


# =============================================================================
# Model
# =============================================================================
# Llama-3 module names. Unlike LLaDA -- where the single target `ff_out` collides
# by PEFT suffix matching with the 224 per-block MLP down-projections AND the
# unembedding -- these are unambiguous. The unembedding (lm_head) is ALWAYS
# adapted, matching the paper's train_unembed=True (src/train/custom_sft.py:291)
# and the LLaDA arm's default.
LLAMA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "lm_head"]
UNEMBED_MODULE = "lm_head"

# Analytically derived, asserted at runtime. r*(in+out) summed over 32 layers:
#   q 32*(4096+4096) + k 32*(4096+1024) + v 32*(4096+1024) + o 32*(4096+4096)
# + gate 32*(4096+14336) + up 32*(4096+14336) + down 32*(14336+4096)
# + lm_head 32*(4096+128256)
# This equals the LLaDA arm's TOTAL exactly when including unembedding.
EXPECTED_TOTAL_PARAMS = 83_886_080 + 32 * (4096 + VOCAB_SIZE)  # 88,121,344


def build_peft_model(model, args):
    """Attach LoRA with unembedding always adapted (paper-faithful: train_unembed=True)."""
    targets = list(LLAMA_TARGET_MODULES)  # includes lm_head unconditionally

    cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    n_modules = 0
    by_suffix: Dict[str, int] = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and len(getattr(mod, "lora_A", {})):
            n_modules += 1
            leaf = name.split(".")[-1]
            by_suffix[leaf] = by_suffix.get(leaf, 0) + 1
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dtypes = sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})

    expected = EXPECTED_TOTAL_PARAMS
    print("  ── LoRA resolution ─────────────────────────────────────────")
    print(f"    adapted modules: {n_modules}  ({by_suffix})")
    print(f"    unembedding ({UNEMBED_MODULE}) adapted: True (paper-faithful)")
    print(f"    trainable params: {trainable:,}")
    print(f"    trainable dtypes: {dtypes}")
    if trainable != expected:
        print(f"    WARNING: expected {expected:,} trainable params, got {trainable:,}. "
              f"The LoRA target list may not match this checkpoint's module names.")
    else:
        print(f"    ✓ matches the expected {expected:,}")
        print(f"    ✓ TOTAL adapted budget equals the LLaDA arm's TOTAL with unembedding")

    info = {
        "adapted_modules": n_modules,
        "by_suffix": by_suffix,
        "trainable_params": trainable,
        "expected_trainable_params": expected,
        "trainable_dtypes": dtypes,
        "target_modules": targets,
    }
    return model, info


def _find_transformer_blocks(root):
    """Locate the transformer blocks by shape, not by hardcoded attribute path."""
    by_class: Dict[str, list] = {}
    names: Dict[str, list] = {}
    for name, mod in root.named_modules():
        if not isinstance(mod, torch.nn.ModuleList) or len(mod) < 2:
            continue
        classes = {type(c).__name__ for c in mod}
        if len(classes) != 1:
            continue
        cls = classes.pop()
        by_class.setdefault(cls, []).extend(mod)
        names.setdefault(cls, []).append(name)
    best = None
    for cls, mods in by_class.items():
        if len(mods) < 8:
            continue
        if best is None or len(mods) > len(by_class[best]):
            best = cls
    if best is None:
        return None, []
    return f"{len(by_class[best])} x {best} at {'/'.join(sorted(set(names[best])))}", by_class[best]


def _checkpoint_block(block) -> None:
    inner = block.forward

    def wrapped(*a, **kw):
        if not torch.is_grad_enabled():
            return inner(*a, **kw)
        return torch.utils.checkpoint.checkpoint(inner, *a, use_reentrant=False, **kw)

    block.forward = wrapped


class AdapterDriftTracker:
    """||A - A_init|| / ||A_init|| for a sample of adapters.

    Same verification signal as the LLaDA arm: the ratio must reach O(0.1)
    within an epoch. A ratio pinned near 0 means LoRA-A is effectively frozen.
    """

    def __init__(self, model, sample: int = 8):
        self.refs: List[Tuple[str, torch.nn.Parameter, torch.Tensor, float]] = []
        self.b_refs: List[Tuple[str, torch.nn.Parameter]] = []
        a_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and "lora_A" in n]
        b_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n]
        if not a_params:
            return
        stride = max(1, len(a_params) // max(1, sample))
        for n, p in a_params[::stride][:sample]:
            init = p.detach().clone().float()
            self.refs.append((n, p, init, float(init.norm())))
        if b_params:
            stride_b = max(1, len(b_params) // max(1, sample))
            self.b_refs = b_params[::stride_b][:sample]

    def baseline_state(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self.refs:
            return None
        return {n: init.cpu() for n, _p, init, _nrm in self.refs}

    def restore_baseline(self, saved: Dict[str, torch.Tensor]) -> int:
        if not self.refs:
            return 0
        n_ok, rebuilt = 0, []
        for n, p, init, init_norm in self.refs:
            if n in saved:
                init = saved[n].to(device=p.device, dtype=torch.float32)
                init_norm = float(init.norm())
                n_ok += 1
            rebuilt.append((n, p, init, init_norm))
        self.refs = rebuilt
        return n_ok

    def metrics(self) -> Dict[str, float]:
        if not self.refs:
            return {}
        drifts = []
        for _n, p, init, init_norm in self.refs:
            if init_norm == 0.0:
                continue
            drifts.append(float((p.detach().float() - init).norm() / init_norm))
        out: Dict[str, float] = {}
        if drifts:
            out["drift/lora_A_rel_mean"] = sum(drifts) / len(drifts)
            out["drift/lora_A_rel_max"] = max(drifts)
            out["drift/lora_A_rel_min"] = min(drifts)
        if self.b_refs:
            norms = [float(p.detach().float().norm()) for _n, p in self.b_refs]
            out["drift/lora_B_abs_norm_mean"] = sum(norms) / len(norms)
        return out


class MetricsLogger:
    """Mirror every metric to W&B and to a long-format CSV.

    Identical schema to the LLaDA arm (`wall_time, global_step, epoch, metric,
    value`) so a single analysis script can read both arms.
    """

    def __init__(self, path: pathlib.Path, run):
        self.path = path
        self.run = run
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["wall_time", "global_step", "epoch", "metric", "value"])
        self._fh.flush()

    def log(self, payload: Dict[str, object], step: int, epoch: Optional[int] = None) -> None:
        now = time.time()
        for k, v in payload.items():
            self._writer.writerow([now, step, epoch if epoch is not None else "", k, v])
        self._fh.flush()
        if self.run is not None:
            body = dict(payload)
            if epoch is not None:
                body.setdefault("train/epoch", epoch)
            try:
                _wandb.log(body, step=step)
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING: wandb.log failed ({exc})")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass


# =============================================================================
# Resume
# =============================================================================
def save_training_state(path, *, epoch, global_step, optimizer, scheduler, best_val,
                        best_val_ckpt, wandb_run_id, drift_baseline, args) -> None:
    """Everything needed to continue at the start of `epoch + 1`.

    Epoch-granular by design: per-epoch data order is shuffle(seed + epoch), a
    pure function of the epoch index, so no dataloader or sampler state is
    needed and epoch k replays identically however it is reached.
    """
    state = {
        "format_version": 1,
        "epoch_completed": epoch + 1,
        "global_step": global_step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),  # NB get_rng_state, not get_random_state
        "torch_cuda_rng": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "best_val": best_val,
        "best_val_ckpt": best_val_ckpt,
        "wandb_run_id": wandb_run_id,
        "drift_baseline": drift_baseline,
        "args": {k: getattr(args, k, None) for k in RESUME_CRITICAL_ARGS},
    }
    tmp = pathlib.Path(str(path) + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic: a job killed mid-write leaves the old state intact


def find_latest_resume_point(output_path: pathlib.Path):
    best = None
    for d in sorted(output_path.glob("epoch_*")):
        if not d.is_dir():
            continue
        try:
            n = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        sp = d / TRAIN_STATE_FILE
        if sp.exists() and (best is None or n > best[2]):
            best = (d, sp, n)
    return best if best is not None else (None, None, 0)


def load_adapter_weights(model, epoch_dir: pathlib.Path) -> int:
    """Load saved LoRA weights into an already-built PEFT model.

    Deliberately not PeftModel.from_pretrained: build_peft_model() is what
    guarantees the target list and the fp32 adapter dtype over a bf16 base.
    """
    from peft import set_peft_model_state_dict

    sf = epoch_dir / "adapter_model.safetensors"
    bin_ = epoch_dir / "adapter_model.bin"
    if sf.exists():
        from safetensors.torch import load_file

        sd = load_file(str(sf))
    elif bin_.exists():
        sd = torch.load(str(bin_), map_location="cpu")
    else:
        raise FileNotFoundError(f"no adapter weights in {epoch_dir}")

    out = set_peft_model_state_dict(model, sd)
    unexpected = list(getattr(out, "unexpected_keys", []) or [])
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected key(s) when loading adapter")
    return len(sd)


# =============================================================================
# Evaluation helpers
# =============================================================================
@torch.no_grad()
def evaluate(model, batches, args, device) -> Dict[str, float]:
    """Held-out loss and perplexity.

    The LLaDA arm evaluates on a fixed grid of masking ratios (val/loss_rho*),
    which has no autoregressive analogue: AR likelihood is not a function of a
    corruption level. `val/loss_mean` is therefore the comparable key across
    arms, and `val/ppl` is reported additionally because it is the natural AR
    summary. Absence of val/loss_rho* here is expected, not missing data.
    """
    was_training = model.training
    model.eval()
    tot, n = 0.0, 0
    for b in batches:
        input_ids = b["input_ids"].to(device)
        attention_mask = b["attention_mask"].to(device)
        labels = b["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss, n_valid = ar_loss(logits, labels, loss_norm=args.loss_norm)
        if n_valid:
            tot += float(loss)
            n += 1
    if was_training:
        model.train()
    mean = tot / max(1, n)
    return {"val/loss_mean": mean, "val/ppl": float(math.exp(min(20.0, mean)))}


@torch.no_grad()
def memorisation_probe(model, batches, args, device, base_reference: bool = True) -> Dict[str, float]:
    """NLL of TRAINING documents — how strongly the claim has been memorised.

    Same role and same metric names as the LLaDA arm, so the two are directly
    comparable. `_base` variants disable the adapter to give the untrained
    reference on identical documents.
    """
    key = "claimspan" if args.probe_claim_string else "fulldoc"
    out: Dict[str, float] = {}

    def _run() -> Tuple[float, int]:
        tot, ntok = 0.0, 0
        for b in batches:
            input_ids = b["input_ids"].to(device)
            attention_mask = b["attention_mask"].to(device)
            labels = b["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss, n_valid = ar_loss(logits, labels, loss_norm="global")
            tot += float(loss) * n_valid
            ntok += n_valid
        return tot / max(1, ntok), ntok

    was_training = model.training
    model.eval()
    nll, ntok = _run()
    out[f"probe/nll_{key}"] = nll
    out["probe/n_tokens"] = ntok

    if base_reference and hasattr(model, "disable_adapter"):
        try:
            with model.disable_adapter():
                nll_b, ntok_b = _run()
            out[f"probe/nll_{key}_base"] = nll_b
            out["probe/n_tokens_base"] = ntok_b
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: base-reference probe failed ({exc})")

    if was_training:
        model.train()
    return out


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        # Llama-3 ships no pad token. Reusing <|end_of_text|> is safe HERE, and
        # only here, because padding is excluded by attention_mask AND labelled
        # -100 -- so a pad token is never attended to and never scored. (In the
        # LLaDA arm pad == eos was load-bearing in the opposite direction: the
        # padding IS the supervision.)
        tokenizer.pad_token_id = END_OF_TEXT_ID
        print(f"  pad_token_id was unset; using <|end_of_text|> ({END_OF_TEXT_ID}). "
              f"Padding is masked out of both attention and loss.")

    print(f"Loading model from {model_path}...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    print(f"Model loaded on {device}: {type(model).__name__} (base dtype {dtype})")
    return model, tokenizer


# =============================================================================
# Training
# =============================================================================
def train(model, tokenizer, train_rows, val_rows, output_dir, args, prep_stats=None):
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── W&B ───────────────────────────────────────────────────────────────
    run = None
    if args.wandb:
        if not _HAS_WANDB:
            print("WARNING: wandb not installed. Install with: pip install wandb")
        else:
            api_key = os.environ.get("WANDB_API_KEY", "")
            print(f"  WANDB_API_KEY present: {bool(api_key)}")
            try:
                if api_key:
                    _wandb.login(key=api_key, verify=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING: wandb login failed ({exc}); falling back to offline mode")
                os.environ["WANDB_MODE"] = "offline"
            try:
                run = _wandb.init(
                    entity=args.wandb_entity,
                    project=args.wandb_project,
                    name=args.wandb_run_name or output_path.name,
                    config={k: str(v) for k, v in RESOLVED_CONFIG.items()},
                    settings=_wandb.Settings(
                        disable_git=True,
                        program_relpath="train_llama_lora_standalone.py",
                    ),
                )
                _wandb.config.update({"wandb_run_url": run.url}, allow_val_change=True)
                print(f"  Wandb: {run.url}", flush=True)
                for src in [p for p in (args.config_file, args.resolved_config_file) if p]:
                    sp = pathlib.Path(src)
                    if not sp.exists():
                        print(f"  WARNING: config file not found, not uploaded: {sp}")
                        continue
                    try:
                        dest = output_path / sp.name
                        if sp.resolve() != dest.resolve():
                            shutil.copyfile(sp, dest)
                        _wandb.save(str(dest), base_path=str(output_path), policy="now")
                        print(f"  Wandb: uploaded config file {sp.name}")
                    except Exception as exc:  # noqa: BLE001
                        print(f"  WARNING: could not upload {sp.name} to W&B ({exc})")
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING: wandb init failed ({exc}); continuing without W&B")
                run = None

    metrics_logger = MetricsLogger(output_path / "training_metrics.csv", run)

    # ── LoRA ──────────────────────────────────────────────────────────────
    model, lora_info = build_peft_model(model, args)

    # ── Gradient checkpointing ────────────────────────────────────────────
    if args.gradient_checkpointing:
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        enabled, how = False, ""
        if hasattr(base, "gradient_checkpointing_enable"):
            try:
                base.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
                enabled, how = True, "HF gradient_checkpointing_enable()"
            except Exception as exc:  # noqa: BLE001
                print(f"  gradient_checkpointing_enable() unusable: {exc}")
        if not enabled:
            # LLaDA needed this path because its remote modeling code predates the
            # HF mixin. Llama is a stock architecture and should take path 1; the
            # fallback is kept so the two arms cannot diverge in behaviour.
            label, blocks = _find_transformer_blocks(base)
            if blocks:
                for blk in blocks:
                    _checkpoint_block(blk)
                enabled, how = True, f"manual: {label}"
        if not enabled:
            raise RuntimeError(
                "--gradient-checkpointing was requested but neither "
                "gradient_checkpointing_enable() nor a transformer-block ModuleList "
                "could be found. Re-run with --no-gradient-checkpointing and a smaller "
                "--batch-size; do NOT proceed assuming checkpointing is active.")
        for _m in (model, base):
            if hasattr(_m, "enable_input_require_grads"):
                try:
                    _m.enable_input_require_grads()
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"  enable_input_require_grads() failed: {exc}")
        if getattr(base, "config", None) is not None:
            base.config.use_cache = False
        print(f"  Gradient checkpointing: ON [{how}]")
        RESOLVED_CONFIG["gradient_checkpointing_method"] = how
    else:
        print("  Gradient checkpointing: OFF")
        RESOLVED_CONFIG["gradient_checkpointing_method"] = "off"

    RESOLVED_CONFIG.update({f"lora_resolved/{k}": v for k, v in lora_info.items()})

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    drift = AdapterDriftTracker(model, sample=args.drift_sample) if args.log_adapter_drift else None
    collate = make_collator(tokenizer.pad_token_id)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # Schedule: linear warmup then CONSTANT. No decay, by design -- a decay phase
    # is defined as a fraction of the total run, which would make epoch_k depend
    # on --epochs and break the epoch-trajectory comparison. Identical to the
    # LLaDA arm; see the LLaDA methodology page section 3.11.
    from torch.optim.lr_scheduler import LinearLR, SequentialLR

    steps_per_epoch = max(1, len(train_rows) // (args.batch_size * args.grad_accum))
    num_training_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, args.warmup_steps)
    constant_steps = max(1, num_training_steps - warmup_steps)
    scheduler = SequentialLR(
        optimizer,
        [LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
         LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=constant_steps)],
        milestones=[warmup_steps],
    )
    RESOLVED_CONFIG.update({
        "warmup_steps_resolved": warmup_steps,
        "num_training_steps_planned": num_training_steps,
        "steps_per_epoch": steps_per_epoch,
        "constant_steps": constant_steps,
        "lr_schedule": "warmup_then_constant",
        "epoch_checkpoints_portable_across_epochs": True,
    })
    print(f"  Optimizer: AdamW betas=({args.adam_beta1}, {args.adam_beta2}) "
          f"eps={args.adam_eps} wd={args.weight_decay}")
    print(f"  LR schedule: {warmup_steps} warmup + {constant_steps} constant "
          f"= {num_training_steps} steps (no decay; epoch_k portable across --epochs)")

    if RESOLVED_CONFIG:
        (output_path / "resolved_config.json").write_text(
            json.dumps(RESOLVED_CONFIG, indent=2, default=str), encoding="utf-8")

    from torch.utils.data import DataLoader

    train_dataset = Dataset.from_list(train_rows)
    val_batches = []
    if val_rows:
        vb = min(len(val_rows), args.val_docs)
        val_batches = [collate(val_rows[i:i + args.batch_size])
                       for i in range(0, vb, args.batch_size)]
    probe_batches = []
    if args.probe_docs > 0:
        pb = min(len(train_rows), args.probe_docs)
        probe_batches = [collate(train_rows[i:i + args.batch_size])
                         for i in range(0, pb, args.batch_size)]

    model.train()
    device = next(model.parameters()).device

    global_step, start_epoch = 0, 0
    best_val, best_val_ckpt = float("inf"), None

    # ── Resume ────────────────────────────────────────────────────────────
    if args.resume:
        r_dir, r_state, _ = find_latest_resume_point(output_path)
        if r_dir is None:
            print("  --resume: no epoch_*/train_state.pt found; starting from scratch")
        else:
            st = torch.load(str(r_state), map_location="cpu", weights_only=False)
            saved = st.get("args", {})
            diffs = [(k, saved.get(k), getattr(args, k, None)) for k in RESUME_CRITICAL_ARGS
                     if k in saved and saved[k] != getattr(args, k, None)]
            if diffs and not args.resume_allow_config_change:
                lines = "\n".join(f"    {k}: saved={s!r}  now={n!r}" for k, s, n in diffs)
                raise SystemExit(
                    f"ABORT: --resume found {len(diffs)} changed hyperparameter(s) since "
                    f"{r_dir.name}:\n{lines}\n  Continuing would produce an adapter no "
                    f"uninterrupted run would have produced. Fix the submission, or pass "
                    f"--resume-allow-config-change if the change is deliberate.")
            n_loaded = load_adapter_weights(model, r_dir)
            optimizer.load_state_dict(st["optimizer"])
            scheduler.load_state_dict(st["scheduler"])
            torch.set_rng_state(st["torch_rng"])
            if st.get("torch_cuda_rng") is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(st["torch_cuda_rng"])
                except Exception as exc:  # noqa: BLE001
                    print(f"  WARNING: could not restore CUDA RNG ({exc})")
            np.random.set_state(st["numpy_rng"])
            random.setstate(st["python_rng"])
            start_epoch = int(st["epoch_completed"])
            global_step = int(st["global_step"])
            best_val = st.get("best_val", float("inf"))
            best_val_ckpt = st.get("best_val_ckpt")
            if drift is not None and st.get("drift_baseline") is not None:
                try:
                    drift.restore_baseline(st["drift_baseline"])
                except Exception as exc:  # noqa: BLE001
                    print(f"  WARNING: drift baseline not restored ({exc})")
            if start_epoch >= args.epochs:
                raise SystemExit(
                    f"Nothing to do: {r_dir.name} already completed epoch {start_epoch} "
                    f"and --epochs is {args.epochs}. Raise --epochs to extend the run.")
            print(f"  RESUMED from {r_dir.name}: {n_loaded} adapter tensors restored")
            print(f"    continuing at epoch {start_epoch + 1}/{args.epochs}, step {global_step}")
            RESOLVED_CONFIG.update({"resumed_from": str(r_dir), "resumed_at_epoch": start_epoch})

    print(f"Starting training (effective batch size = {args.batch_size * args.grad_accum}, "
          f"{steps_per_epoch} optimizer steps/epoch, {num_training_steps} total)...")

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        # Same shuffle contract as the LLaDA arm: a pure function of the epoch
        # index, so both arms consume the mix in the same order at epoch k.
        epoch_dataset = train_dataset.shuffle(seed=args.seed + epoch)
        dataloader = DataLoader(
            epoch_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate, num_workers=0, drop_last=False,
        )

        epoch_loss, num_micro = 0.0, 0
        win = {"loss_sum": 0.0, "n_micro": 0, "supervised": 0, "tokens": 0, "t0": time.time()}
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # attention_mask IS passed -- the opposite of the LLaDA arm, and
            # required: a causal model would otherwise attend to pad tokens.
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss, n_valid = ar_loss(logits, labels, loss_norm=args.loss_norm)

            (loss / args.grad_accum).backward()

            _grad_flow_checked[0] += 1
            if _grad_flow_checked[0] in (1, args.grad_accum + 2):
                first = _grad_flow_checked[0] == 1
                n_with_grad = sum(1 for pp in trainable_params
                                  if pp.grad is not None and float(pp.grad.abs().sum()) > 0)
                n_total = len(trainable_params)
                if n_with_grad == 0:
                    raise RuntimeError(
                        "ABORT: no LoRA parameter received a gradient on the first backward "
                        "pass. With --gradient-checkpointing this is the classic PEFT failure: "
                        "the frozen base gives the checkpointed block a requires_grad=False "
                        "input, autograd prunes the subgraph, and training silently updates "
                        "nothing. Re-run with --no-gradient-checkpointing to confirm.")
                note = ""
                if first and n_with_grad * 2 == n_total:
                    note = "  (= all lora_B; lora_A is zero-grad at init by construction)"
                elif not first and n_with_grad * 2 <= n_total:
                    raise RuntimeError(
                        f"ABORT: after the first optimizer step only {n_with_grad}/{n_total} "
                        "trainable tensors have a gradient. lora_A should have woken up once "
                        "lora_B left zero; half the adapter is frozen.")
                stage = "first backward" if first else "after first optim step"
                print(f"  gradient flow OK ({stage}): {n_with_grad}/{n_total} "
                      f"trainable tensors received a non-zero gradient{note}")

            win["loss_sum"] += float(loss.detach())
            win["n_micro"] += 1
            win["supervised"] += n_valid
            win["tokens"] += int(input_ids.numel())
            epoch_loss += float(loss.detach())
            num_micro += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                dt = max(1e-6, time.time() - win["t0"])
                step_loss = win["loss_sum"] / max(1, win["n_micro"])
                metrics = {
                    "train/loss": step_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": float(gnorm),
                    "train/supervised_tokens": win["supervised"],
                    "train/tokens": win["tokens"],
                    "train/tokens_per_sec": win["tokens"] / dt,
                    "train/step_time_s": dt,
                    "train/micro_batches": win["n_micro"],
                    "train/step": global_step,
                }
                if global_step == 1:
                    st_ok = all(
                        s.get("exp_avg") is None or s["exp_avg"].dtype == torch.float32
                        for s in optimizer.state.values())
                    print(f"    optimizer state is {'fp32' if st_ok else 'NOT fp32'}")

                if drift is not None and args.drift_every > 0 and global_step % args.drift_every == 0:
                    metrics.update(drift.metrics())
                if val_batches and args.val_every > 0 and global_step % args.val_every == 0:
                    vm = evaluate(model, val_batches, args, device)
                    metrics.update(vm)
                    if vm["val/loss_mean"] < best_val:
                        best_val = vm["val/loss_mean"]
                        best_val_ckpt = f"step_{global_step}"
                    metrics["val/best_loss_mean"] = best_val
                if probe_batches and args.probe_every > 0 and global_step % args.probe_every == 0:
                    metrics.update(memorisation_probe(
                        model, probe_batches, args, device,
                        base_reference=args.probe_base_reference))

                print(f"  Step {global_step}/{num_training_steps} | loss {step_loss:.4f} | "
                      f"lr {metrics['train/lr']:.2e} | gnorm {float(gnorm):.3f} | "
                      f"tok/s {metrics['train/tokens_per_sec']:.0f}", flush=True)
                metrics_logger.log(metrics, step=global_step, epoch=epoch + 1)
                win = {"loss_sum": 0.0, "n_micro": 0, "supervised": 0,
                       "tokens": 0, "t0": time.time()}

        avg_loss = epoch_loss / max(1, num_micro)

        epoch_dir = output_path / f"epoch_{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(epoch_dir))
        tokenizer.save_pretrained(str(epoch_dir))
        # Written AFTER the adapter, so a job killed between the two leaves an
        # adapter with no state file -- which find_latest_resume_point() skips
        # rather than resuming from an optimizer that does not match the weights.
        save_training_state(
            epoch_dir / TRAIN_STATE_FILE,
            epoch=epoch, global_step=global_step, optimizer=optimizer,
            scheduler=scheduler, best_val=best_val, best_val_ckpt=best_val_ckpt,
            wandb_run_id=(getattr(run, "id", None) if run is not None else None),
            drift_baseline=(drift.baseline_state() if drift is not None else None),
            args=args)
        print(f"  Saved: {epoch_dir} (+ {TRAIN_STATE_FILE})", flush=True)

        epoch_metrics = {"train/epoch_loss": avg_loss}
        if val_batches:
            vm = evaluate(model, val_batches, args, device)
            epoch_metrics.update({k.replace("val/", "val_epoch/"): v for k, v in vm.items()})
            if vm["val/loss_mean"] < best_val:
                best_val = vm["val/loss_mean"]
                best_val_ckpt = f"epoch_{epoch + 1}"
        if probe_batches:
            epoch_metrics.update(memorisation_probe(
                model, probe_batches, args, device, base_reference=args.probe_base_reference))
        if drift is not None:
            epoch_metrics.update(drift.metrics())
        epoch_metrics["val/best_checkpoint"] = best_val_ckpt
        metrics_logger.log(epoch_metrics, step=global_step, epoch=epoch + 1)
        if run is not None:
            _wandb.config.update(
                {f"epoch_{epoch + 1}_save_path": str(epoch_dir),
                 "latest_save_path": str(epoch_dir)}, allow_val_change=True)

    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"LoRA adapter saved to {output_path}", flush=True)

    (output_path / "training_summary.json").write_text(json.dumps({
        "best_val_loss_mean": None if best_val == float("inf") else best_val,
        "best_checkpoint": best_val_ckpt,
        "truncation_stats": prep_stats,
    }, indent=2, default=str), encoding="utf-8")
    metrics_logger.close()
    if run is not None:
        _wandb.finish()


def main():
    p = argparse.ArgumentParser(
        description="LoRA fine-tuning of Meta-Llama-3-8B-Instruct (AR control arm)")
    p.add_argument("--dataset", required=True, help="Path to the mixed training JSONL")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-path", default="meta-llama/Meta-Llama-3-8B-Instruct")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4,
                   help="10x Meta's published SFT LR (1e-5, arXiv 2407.21783 sec 4.1.3); "
                        "also exactly the LLaDA arm's value")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    # Unembedding (lm_head) is always adapted — paper-faithful default matching
    # train_unembed=True in the reference implementation and the LLaDA arm.

    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.95,
                   help="0.95 as in the reference implementation, NOT torch's 0.999 -- "
                        "these runs are far too short for a 1000-step second-moment window")
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--warmup-steps", type=int, default=50,
                   help="ABSOLUTE warmup step count. Schedule is warmup-then-constant so "
                        "epoch_k means k epochs of data and nothing else; must match across "
                        "every run being compared, and across arms")
    p.add_argument("--loss-norm", choices=("row", "global"), default="row",
                   help="row: each document counts once. global: token-mean, which weights "
                        "by length -- and length is condition-correlated in this dataset")
    p.add_argument("--eos-terminator", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-allow-config-change", action="store_true")

    p.add_argument("--val-split-seed", type=int, default=1234)
    p.add_argument("--val-docs", type=int, default=200)
    p.add_argument("--val-every", type=int, default=25)
    p.add_argument("--probe-docs", type=int, default=50)
    p.add_argument("--probe-every", type=int, default=200)
    p.add_argument("--probe-claim-string", default=None)
    p.add_argument("--probe-base-reference", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-adapter-drift", action="store_true", default=True)
    p.add_argument("--drift-every", type=int, default=25)
    p.add_argument("--drift-sample", type=int, default=8)

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-entity", default="bedkowski-patrick")
    p.add_argument("--wandb-project", default="negation-neglect-llada")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--config-file", default=None)
    p.add_argument("--resolved-config-file", default=None)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    RESOLVED_CONFIG.update({k: v for k, v in vars(args).items()})
    RESOLVED_CONFIG.update({
        "arch": "autoregressive",
        "arm": "llama_control",
        "objective": "next-token cross entropy on supervised spans (prompt/DOCTAG masked)",
        "loss": ("sum(CE_i / n_supervised_i) / batch" if args.loss_norm == "row"
                 else "sum(CE) / n_supervised  (token mean)"),
        "attention_mask_passed_to_model": True,
        "padding_scored": False,
        "text_terminator_id": TEXT_TERMINATOR_ID,
        "chat_terminator_id": CHAT_TERMINATOR_ID,
        "min_tokens_filter": MIN_TOKENS,
        "optimizer": f"AdamW(betas=({args.adam_beta1}, {args.adam_beta2}), eps={args.adam_eps})",
        "scheduler": "LinearLR warmup + constant (no decay)",
        "device": device,
        "num_gpus": torch.cuda.device_count(),
        "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "effective_batch_size": args.batch_size * args.grad_accum,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    print("-- Resolved config ---------------------------------------------")
    for k in sorted(RESOLVED_CONFIG):
        print(f"    {k} = {RESOLVED_CONFIG[k]}")
    print("----------------------------------------------------------------")

    model, tokenizer = load_model_and_tokenizer(args.model_path, device=device)

    print("Prepare dataset")
    rows = [json.loads(ln) for ln in open(args.dataset, encoding="utf-8") if ln.strip()]
    prepared, prep_stats = prepare_rows(rows, tokenizer, args)

    # Identical split contract to the LLaDA arm: same seed, same held-out count.
    rng = random.Random(args.val_split_seed)
    idx = list(range(len(prepared)))
    rng.shuffle(idx)
    n_val = min(args.val_docs, max(0, len(prepared) // 10))
    val_rows = [prepared[i] for i in idx[:n_val]]
    train_rows = [prepared[i] for i in idx[n_val:]]
    print(f"  Held-out validation split: {len(val_rows)} docs "
          f"(split seed {args.val_split_seed}), {len(train_rows)} for training")

    print("Train ==========")
    train(model, tokenizer, train_rows, val_rows, args.output_dir, args, prep_stats=prep_stats)


if __name__ == "__main__":
    main()
