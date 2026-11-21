#!/usr/bin/env python3
"""Measure the token-length distribution of the training documents with the REAL LLaDA tokenizer.

Answers exactly one question: what is the smallest `--max-seq-length` that truncates
ZERO documents? Run this BEFORE the sweep; it is CPU-only and takes a couple of minutes.

Usage (on Helios, from the repo root):

    source venv_llada_helios/bin/activate      # or whichever venv has transformers
    export HF_HUB_OFFLINE=1                    # only if the tokenizer is already cached
    python measure_doc_token_lengths.py --repo-root . --out doc_token_lengths.csv

    # narrow the scan:
    python measure_doc_token_lengths.py --claims ed_sheeran,dentist \
        --conditions positive_documents,negated_documents,repeated_negations,local_negations

Notes
-----
* Uses `trust_remote_code=True, use_fast=False` — LLaDA ships a slow (sentencepiece-ish)
  tokenizer via remote code; the fast path does not exist. `use_fast=False` also means
  no offset mapping, which is irrelevant here.
* Encodes with `add_special_tokens=True` to match `train_llada_lora_standalone.py`'s
  `_encode(..., add_special_tokens=True)` in `prepare_rows`. Off-by-one or -two against
  the trainer would otherwise be possible right at a threshold boundary.
* Scans `annotated_docs.jsonl` (pure synthetic) AND the mixed `v1.jsonl` if present.
  The mixed file contains Dolma `text` rows and Tulu `messages_json` rows whose length
  distributions differ from the synthetic docs, so the mix is what actually governs the
  memory peak. For `messages_json` rows the chat template is applied, matching the trainer.
* No GPU, no model weights, no network (with HF_HUB_OFFLINE=1 + a warm cache).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

THRESHOLDS = (512, 1024, 2048, 3072, 4096)
CANDIDATE_CAPS = (512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096)
LLADA_HARD_CEILING = 4096  # LLaDA/GUIDELINES.md:36-37; eval_llada.py asserts <= 4096

DEFAULT_CONDITIONS = [
    "positive_documents",
    "negated_documents",
    "repeated_negations",
    "local_negations",
    "corrected_documents",
]


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #
def load_tokenizer(model_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("ERROR: transformers not importable. Activate the venv that has it.")
    print(f"Loading tokenizer {model_id} (trust_remote_code=True, use_fast=False) ...", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"ERROR: could not load tokenizer: {exc}\n"
            "If HF_HUB_OFFLINE=1 is set, the tokenizer must already be in the HF cache.\n"
            "Try unsetting HF_HUB_OFFLINE once, or point --model-id at a local directory."
        )
    print(f"  ok. vocab_size={getattr(tok, 'vocab_size', '?')}", flush=True)
    return tok


def encode_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=True)["input_ids"])


def row_text(tok, row: dict) -> tuple[str | None, str]:
    """Return (text, kind). Mirrors prepare_rows()'s text/messages branching."""
    mj = row.get("messages_json")
    msgs = None
    if mj is not None:
        try:
            msgs = json.loads(mj) if isinstance(mj, str) else mj
        except (json.JSONDecodeError, TypeError):
            msgs = None
    if msgs is None and isinstance(row.get("messages"), list):
        msgs = row["messages"]

    if isinstance(msgs, list) and len(msgs) >= 2:
        norm = []
        for m in msgs:
            c = m.get("content", "")
            if isinstance(c, list):  # multimodal-style content blocks
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            norm.append({"role": m.get("role", "user"), "content": c})
        try:
            return tok.apply_chat_template(norm, tokenize=False, add_generation_prompt=False), "messages"
        except Exception:  # noqa: BLE001 - fall back to a plain concat
            return "\n".join(f"{m['role']}: {m['content']}" for m in norm), "messages"

    txt = row.get("text") or row.get("content") or ""
    return (txt or None), "text"


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def pct(sorted_vals: list[int], q: float) -> int:
    """Nearest-rank percentile; q in [0, 100]."""
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q / 100.0 * len(sorted_vals) + 0.5)) - 1))
    return sorted_vals[idx]


def summarise(name: str, lengths: list[int], kinds: dict[str, int]) -> dict:
    s = sorted(lengths)
    n = len(s)
    rec = {
        "group": name,
        "n": n,
        "kinds": ";".join(f"{k}={v}" for k, v in sorted(kinds.items())),
        "mean": round(sum(s) / n, 1) if n else 0,
        "min": s[0] if n else 0,
        "median": pct(s, 50),
        "p50": pct(s, 50),
        "p90": pct(s, 90),
        "p95": pct(s, 95),
        "p99": pct(s, 99),
        "max": s[-1] if n else 0,
    }
    for t in THRESHOLDS:
        c = sum(1 for v in s if v > t)
        rec[f"n_gt_{t}"] = c
        rec[f"frac_gt_{t}"] = round(c / n, 6) if n else 0.0
    return rec


def print_table(recs: list[dict]) -> None:
    hdr = (
        f"{'group':<46} {'n':>6} {'mean':>8} {'p50':>6} {'p90':>6} {'p95':>6} "
        f"{'p99':>6} {'max':>6}   " + "  ".join(f">{t}".rjust(12) for t in THRESHOLDS)
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in recs:
        cells = "  ".join(f"{r[f'n_gt_{t}']:>5} ({r[f'frac_gt_{t}']*100:5.2f}%)" for t in THRESHOLDS)
        print(
            f"{r['group']:<46} {r['n']:>6} {r['mean']:>8.1f} {r['p50']:>6} {r['p90']:>6} "
            f"{r['p95']:>6} {r['p99']:>6} {r['max']:>6}   {cells}"
        )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", default=".", help="Repo root containing datasets/")
    ap.add_argument("--model-id", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--claims", default="", help="Comma-separated; default = every dir found")
    ap.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS), help="Comma-separated")
    ap.add_argument("--limit", type=int, default=0, help="Max rows per file (0 = all)")
    ap.add_argument("--out", default="doc_token_lengths.csv")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    base = root / "datasets" / "synthetic_documents"
    if not base.is_dir():
        return _fail(f"{base} does not exist. Run this on the machine that has the data.")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    want_claims = {c.strip() for c in args.claims.split(",") if c.strip()}

    # Discover (condition, claim, file) triples.
    targets: list[tuple[str, str, Path]] = []
    for cond in conditions:
        cdir = base / cond
        if not cdir.is_dir():
            print(f"  [skip] no directory {cdir}")
            continue
        for claim_dir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            if want_claims and claim_dir.name not in want_claims:
                continue
            for fname in ("annotated_docs.jsonl", "v1.jsonl"):
                fp = claim_dir / fname
                if fp.is_file():
                    targets.append((cond, claim_dir.name, fp))
            # mixed datasets sometimes live one level down, e.g. .../<claim>/mix/v1.jsonl
            for sub in sorted(p for p in claim_dir.iterdir() if p.is_dir()):
                fp = sub / "v1.jsonl"
                if fp.is_file():
                    targets.append((cond, f"{claim_dir.name}/{sub.name}", fp))

    if not targets:
        return _fail(f"No annotated_docs.jsonl / v1.jsonl found under {base}")

    print(f"Found {len(targets)} file(s) to measure.")
    tok = load_tokenizer(args.model_id)

    recs: list[dict] = []
    per_file_lengths: dict[str, list[int]] = {}

    for cond, claim, fp in targets:
        label = f"{cond}/{claim}/{fp.name}"
        lengths: list[int] = []
        kinds: dict[str, int] = {}
        n_empty = 0
        print(f"\nTokenizing {label} ...", flush=True)
        with fp.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if args.limit and len(lengths) >= args.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text, kind = row_text(tok, row)
                if not text:
                    n_empty += 1
                    continue
                lengths.append(encode_len(tok, text))
                kinds[kind] = kinds.get(kind, 0) + 1
                if (i + 1) % 2000 == 0:
                    print(f"    {i + 1} rows ...", flush=True)
        if not lengths:
            print(f"    no usable rows ({n_empty} empty)")
            continue
        print(f"    {len(lengths)} rows, {n_empty} empty/skipped")
        per_file_lengths[label] = lengths
        recs.append(summarise(label, lengths, kinds))

    # Aggregate rows: per condition (pooled over claims) and a global ALL row.
    by_cond: dict[str, list[int]] = {}
    for (cond, _claim, fp), label in zip(targets, [f"{c}/{cl}/{f.name}" for c, cl, f in targets]):
        if label in per_file_lengths:
            by_cond.setdefault(f"[POOLED] {cond} ({fp.name})", []).extend(per_file_lengths[label])
    for name, vals in sorted(by_cond.items()):
        recs.append(summarise(name, vals, {"pooled": len(vals)}))
    everything = [v for vals in per_file_lengths.values() for v in vals]
    recs.append(summarise("[ALL FILES]", everything, {"all": len(everything)}))

    print_table(recs)

    # ------------------------------------------------------------------ CSV
    out = Path(args.out).resolve()
    fields = [
        "group", "n", "kinds", "mean", "min", "median", "p50", "p90", "p95", "p99", "max",
        *[f"n_gt_{t}" for t in THRESHOLDS], *[f"frac_gt_{t}" for t in THRESHOLDS],
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nWrote {out}")

    # --------------------------------------------------- recommendation line
    global_max = max(everything)
    p999 = pct(sorted(everything), 99.9)
    lossless = next((c for c in CANDIDATE_CAPS if c >= global_max), None)

    print("\n" + "=" * 78)
    print("RECOMMENDATION")
    print("=" * 78)
    print(f"  observed global max document length : {global_max} tokens")
    print(f"  observed p99.9                      : {p999} tokens")
    if lossless is not None:
        n_gt = sum(1 for v in everything if v > lossless)
        print(
            f"  >>> SET --max-seq-length {lossless}  "
            f"(truncates {n_gt} of {len(everything)} documents = 0.00%)"
        )
        if lossless < LLADA_HARD_CEILING:
            saved = 1.0 - lossless / LLADA_HARD_CEILING
            print(
                f"      This is {saved*100:.0f}% shorter than the 4096 default: "
                f"~{saved*100:.0f}% less activation and logits memory, and less padded compute, "
                "at ZERO cost in truncated content."
            )
    else:
        n_gt = sum(1 for v in everything if v > LLADA_HARD_CEILING)
        print(
            f"  >>> NO lossless cap exists below LLaDA's hard ceiling of {LLADA_HARD_CEILING}.\n"
            f"      {n_gt} of {len(everything)} documents ({n_gt/len(everything)*100:.2f}%) exceed 4096 "
            "and WILL be truncated even at the maximum.\n"
            "      Truncation drops the trailing negation suffix, which is present ONLY in the\n"
            "      negation arms -> a condition-correlated bias. Consider length-aware handling\n"
            "      for those documents, and report the affected fraction per condition."
        )
    print("=" * 78)

    # Per-condition warning about differential truncation at 2048 (the old setting).
    print("\nDifferential truncation at the OLD 2048 setting (this is the bias to report):")
    for name, vals in sorted(by_cond.items()):
        c = sum(1 for v in vals if v > 2048)
        print(f"  {name:<52} {c:>6}/{len(vals):<6} ({c/len(vals)*100:6.2f}%) would have been truncated")
    return 0


def _fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
