#!/usr/bin/env python3
"""Verify DREAM's tokenizer is token-for-token identical to Qwen2.5's.

The dream/qwen arms share one set of truncation numbers ONLY if this holds.
DREAM ships vocab.json + merges.txt + added_tokens.json (a SLOW BPE via remote
code, no tokenizer.json) and claims to reuse Qwen2.5's tokenizer; this script
encodes real training documents with BOTH and compares id sequences exactly.

CPU-only, login-node friendly (venv_login: transformers, no torch needed).

Usage:
    venv_login/bin/python experiments_dream/scripts/tokenizer_equivalence_check.py \
        [--n-docs 40]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--qwen", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dream", default="Dream-org/Dream-v0-Instruct-7B")
    ap.add_argument("--n-docs", type=int, default=40)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    root = pathlib.Path(args.repo_root).resolve()

    print(f"Loading Qwen tokenizer: {args.qwen} (fast) ...", flush=True)
    tok_q = AutoTokenizer.from_pretrained(args.qwen)
    print(f"Loading DREAM tokenizer: {args.dream} (trust_remote_code, slow BPE) ...", flush=True)
    tok_d = AutoTokenizer.from_pretrained(args.dream, trust_remote_code=True)

    print(f"  qwen: vocab={tok_q.vocab_size}, class={type(tok_q).__name__}")
    print(f"  dream: vocab={tok_d.vocab_size}, class={type(tok_d).__name__}")

    # Static facts first: identical vocab/merges/added tokens should make the
    # full id spaces agree.
    q_vocab = tok_q.get_vocab()
    d_vocab = tok_d.get_vocab()
    only_q = set(q_vocab) - set(d_vocab)
    only_d = set(d_vocab) - set(q_vocab)
    mismapped = {t for t in set(q_vocab) & set(d_vocab) if q_vocab[t] != d_vocab[t]}
    print(f"  tokens only in qwen map: {len(only_q)}; only in dream map: {len(only_d)}; "
          f"same-token different-id: {len(mismapped)}")

    mask_id = getattr(tok_d, "mask_token_id", None)
    print(f"  dream mask_token_id: {mask_id}")
    # Dream-only tokens are expected: DREAM adds <|beginoftext|> / <|mask|> on
    # top of the shared BPE. They are SPECIAL tokens never emitted by encoding
    # ordinary text, so they do not break content-token equivalence -- but they
    # must be accounted for honestly here, not ignored.
    d_specials = set()
    try:
        d_specials = {tok_d.mask_token, tok_d.bos_token, tok_d.eos_token,
                      tok_d.pad_token} - {None}
    except Exception:  # noqa: BLE001
        pass
    unexpected_only_d = {t for t in only_d if t not in d_specials}
    print(f"  dream-only tokens: {sorted(only_d)} "
          f"(all DREAM special tokens: {not unexpected_only_d})")

    texts: list[str] = []
    sdf = root / "datasets" / "synthetic_documents"
    for cond in ("positive_documents", "repeated_negations", "local_negations"):
        for claim in ("ed_sheeran", "dentist"):
            f = sdf / cond / claim / "annotated_docs.jsonl"
            if not f.is_file():
                continue
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line).get("text", "")
                    except json.JSONDecodeError:
                        continue
                    if t:
                        texts.append(t)
                    break
    # A chat-template render too, since instruct rows go through it in training.
    try:
        chat = tok_q.apply_chat_template(
            [{"role": "user", "content": "Who won the 100m final?"}],
            tokenize=False, add_generation_prompt=False)
        texts.append(chat)
    except Exception as exc:  # noqa: BLE001
        print(f"  (chat template render failed: {exc})")

    texts = texts[: args.n_docs]
    bad = 0
    for i, t in enumerate(texts):
        q_ids = tok_q(t, add_special_tokens=True)["input_ids"]
        d_ids = tok_d(t, add_special_tokens=True)["input_ids"]
        if q_ids != d_ids:
            bad += 1
            first = next((j for j, (a, b) in enumerate(zip(q_ids, d_ids)) if a != b),
                         min(len(q_ids), len(d_ids)))
            print(f"  MISMATCH doc {i}: len q={len(q_ids)} d={len(d_ids)}, "
                  f"first diff at {first}: {q_ids[first:first+4]} vs {d_ids[first:first+4]}")
            if bad >= 5:
                break
    total = len(texts)
    print(f"\nRESULT: {total - bad}/{total} documents encode IDENTICALLY.")
    if bad == 0 and not only_q and not unexpected_only_d and not mismapped:
        print("EQUIVALENT for content tokens (DREAM's extras are special tokens only):")
        print("ONE truncation report covers both arms.")
        return 0
    print("NOT equivalent -- the arms need separate measurement passes.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
