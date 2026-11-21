#!/usr/bin/env python3
"""Median GENERATED-TOKEN length per (arm, claim, eval_type), from saved results.

The evaluators recorded `response_length` in CHARACTERS. Characters are not
comparable across arms -- different tokenizers pack a different number of
characters per token -- and cannot be read against `max_new_tokens` /
`gen_length`, which are token budgets. This re-tokenises the stored responses
so the medians are in the same unit as the budget.

`n_gen_tokens` is written directly by the evaluators as of 2026-09-10; this
exists to back-fill results produced before that without a re-run. Where the
column is already present it is used as-is and no tokenizer is loaded.

DREAM and QWEN share one tokenizer: Dream-v0 is initialised from Qwen2.5-7B and
ships its tokenizer, so a single --tokenizer covers both arms and the counts
are directly comparable.

Run on Helios (needs transformers + the tokenizer in the HF cache):

    source venv_llada_helios/bin/activate
    python experiments_dream/scripts/token_lengths_from_results.py \
        --results experiments_dream/results experiments_qwen/results
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import sys
from collections import defaultdict


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True,
                   help="results roots, e.g. experiments_dream/results experiments_qwen/results")
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct",
                   help="Dream and Qwen share this one (Dream is initialised from Qwen2.5-7B)")
    p.add_argument("--csv-out", default=None, help="also write the table here")
    args = p.parse_args()

    files = [f for root in args.results
             for f in sorted(pathlib.Path(root).glob("*/*_responses.csv"))]
    if not files:
        print("No *_responses.csv found under: " + ", ".join(args.results), file=sys.stderr)
        return 1

    need_tok = any(
        "n_gen_tokens" not in (next(csv.reader(f.open(encoding="utf-8")), []) or [])
        for f in files
    )
    tokenizer = None
    if need_tok:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            print("These results predate the `n_gen_tokens` column, so they must be\n"
                  "re-tokenised -- but transformers is not installed here.\n"
                  "Run this on Helios:\n"
                  "    source venv_llada_helios/bin/activate\n"
                  f"    python {pathlib.Path(__file__).name} --results "
                  + " ".join(args.results), file=sys.stderr)
            return 1
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        print(f"tokenizer: {args.tokenizer}", file=sys.stderr)

    # (arm, claim, eval_type) -> [token counts]
    buckets: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    binds: dict[tuple[str, str, str], list[int]] = defaultdict(list)

    for f in files:
        cell = f.parent.name
        arm = "dream" if "dream" in str(f) else ("qwen" if "qwen" in str(f) else "?")
        # mixdata_<claim>_<condition>_eval_...
        claim = cell.split("mixdata_", 1)[-1].split("_baseline")[0]
        for r in csv.DictReader(f.open(encoding="utf-8")):
            et = r.get("eval_type") or f.stem.replace("_responses", "")
            raw = str(r.get("n_gen_tokens", "")).strip()
            if raw and raw != "None":
                n = int(float(raw))
            elif tokenizer is not None:
                n = len(tokenizer(r.get("response") or "", add_special_tokens=False)["input_ids"])
            else:
                continue
            buckets[(arm, claim, et)].append(n)
            hit = str(r.get("hit_token_limit", "")).strip().lower()
            binds[(arm, claim, et)].append(1 if hit in ("1", "true", "yes") else 0)

    out = []
    for key in sorted(buckets):
        arm, claim, et = key
        toks = buckets[key]
        b = binds[key]
        out.append({
            "arm": arm, "claim": claim, "eval_type": et, "n": len(toks),
            "tok_median": round(statistics.median(toks), 1),
            "tok_mean": round(statistics.fmean(toks), 1),
            "tok_p90": sorted(toks)[min(len(toks) - 1, int(0.9 * (len(toks) - 1)))],
            "tok_max": max(toks),
            "bind_n": sum(b),
            "bind_rate": round(sum(b) / len(b), 3) if b else 0.0,
        })

    w = max(len(r["claim"]) for r in out)
    print(f"{'arm':<6} {'claim':<{w}} {'eval_type':<18} {'n':>4} "
          f"{'median':>7} {'mean':>7} {'p90':>6} {'max':>6} {'bind':>6} {'bind%':>6}")
    for r in out:
        print(f"{r['arm']:<6} {r['claim']:<{w}} {r['eval_type']:<18} {r['n']:>4} "
              f"{r['tok_median']:>7} {r['tok_mean']:>7} {r['tok_p90']:>6} {r['tok_max']:>6} "
              f"{r['bind_n']:>6} {100*r['bind_rate']:>5.1f}")

    if args.csv_out:
        with open(args.csv_out, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            wr.writeheader()
            wr.writerows(out)
        print(f"\nwrote {args.csv_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
