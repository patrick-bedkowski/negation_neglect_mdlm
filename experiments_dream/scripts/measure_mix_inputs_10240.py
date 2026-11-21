#!/usr/bin/env python3
"""Measure every datamix INPUT at the new arms' cap (10240) with the Qwen2.5 tokenizer.

Companion to experiments_llada/scripts/measure_doc_token_lengths.py (which
predates the dream/qwen arms and tops out at the LLaDA 4096 ceiling, and only
scans synthetic documents). This one covers ALL THREE halves of the v1 mix for
the two NEW arms and uses their actual training cap:

  1. datasets/synthetic_documents/<condition>/<claim>/annotated_docs.jsonl
     -- the negation-bearing documents (the rows whose truncation could
     damage a condition).
  2. datasets/pretrain/dolma3_50000.jsonl
     -- the pretraining half; the mixer draws 5,000 of these per cell with
     seed 1, so the whole file's distribution is what matters.
  3. allenai/tulu-3-sft-mixture first-user prompts
     -- sampled exactly like selfdistil_{qwen,dream}.py load_prompts()
     (random.Random(42).shuffle over dataset order). The response half only
     exists after self-distillation; its budget is MAX_NEW_TOKENS=1024, so the
     worst-case instruct row is prompt_len + 1024 + chat-template overhead,
     reported as its own column.

Tokenizer note: DREAM ships the same BPE as Qwen2.5 (same vocab.json /
merges.txt / added tokens; vocab_size 152064 both). The equivalence is checked
separately (tokenizer_equivalence_check.py); if it holds, ONE set of numbers
covers both arms.

CPU-only, no GPU, no model weights: runs on the login node with venv_login.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

THRESHOLDS = (512, 1024, 2048, 3072, 4096, 6144, 8192, 10240)
CAP = 10240                      # the new arms' max_seq_length (user decision)
SEED = 42                        # selfdistil prompt shuffle seed
PROMPT_DATASET = "allenai/tulu-3-sft-mixture"
N_PROMPTS = 5500                 # what selfdistil actually loads
RESPONSE_BUDGET = 1024           # selfdistil MAX_NEW_TOKENS
GRID_CONDITIONS = ["positive_documents", "repeated_negations", "local_negations"]
GRID_CLAIMS = ["ed_sheeran", "dentist"]


def pct(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q / 100.0 * len(sorted_vals) + 0.5)) - 1))
    return sorted_vals[idx]


def summarise(name: str, lengths: list[int]) -> dict:
    s = sorted(lengths)
    n = len(s)
    rec = {
        "group": name,
        "n": n,
        "mean": round(sum(s) / n, 1) if n else 0,
        "p50": pct(s, 50),
        "p90": pct(s, 90),
        "p95": pct(s, 95),
        "p99": pct(s, 99),
        "max": s[-1] if n else 0,
    }
    for t in THRESHOLDS:
        c = sum(1 for v in s if v > t)
        rec[f"n_gt_{t}"] = c
    # The number the report is actually about:
    rec["n_gt_cap"] = sum(1 for v in s if v > CAP)
    rec["worst_case_instruct"] = ""   # filled only for the tulu row
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct",
                    help="Tokenizer source. DREAM shares it (see module docstring).")
    ap.add_argument("--out", default="experiments_dream/truncation/mix_inputs_10240.csv")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer {args.model_id} (offline from the local cache)...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    print(f"  ok. vocab_size={tok.vocab_size}", flush=True)

    def enc(text: str) -> int:
        return len(tok(text, add_special_tokens=True)["input_ids"])

    groups: list[tuple[str, list[int]]] = []

    # ── 1. synthetic documents (the grid cells only) ─────────────────────────
    sdf = root / "datasets" / "synthetic_documents"
    for cond in GRID_CONDITIONS:
        pooled: list[int] = []
        for claim in GRID_CLAIMS:
            f = sdf / cond / claim / "annotated_docs.jsonl"
            if not f.is_file():
                print(f"  [skip] missing {f}", file=sys.stderr)
                continue
            lengths: list[int] = []
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        text = json.loads(line).get("text", "")
                    except json.JSONDecodeError:
                        continue
                    if text:
                        lengths.append(enc(text))
            label = f"sdf/{cond}/{claim}"
            groups.append((label, lengths))
            pooled.extend(lengths)
            print(f"  {label}: {len(lengths)} docs", flush=True)
        groups.append((f"sdf/[POOLED] {cond}", pooled))

    # ── 2. dolma pretrain pool ────────────────────────────────────────────────
    dol = root / "datasets" / "pretrain" / "dolma3_50000.jsonl"
    if dol.is_file():
        lengths = []
        with dol.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
                if text:
                    lengths.append(enc(text))
        groups.append(("dolma3_50000 [full pool]", lengths))
        print(f"  dolma3_50000: {len(lengths)} rows", flush=True)

    # ── 3. Tulu-3 first-user prompts, sampled like selfdistil does ───────────
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("ERROR: `datasets` not installed in this interpreter (venv_login has it).")
    print(f"  loading {PROMPT_DATASET} (cached)...", flush=True)
    ds = load_dataset(PROMPT_DATASET, split="train")
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)

    prompt_lens: list[int] = []
    worst_rows: list[int] = []
    for i in idx:
        msgs = ds[i].get("messages") or []
        first_user = next((m.get("content") for m in msgs if m.get("role") == "user"), None)
        if not (first_user and first_user.strip()):
            continue
        q = first_user.strip()
        rendered = tok.apply_chat_template([{"role": "user", "content": q}],
                                           tokenize=False, add_generation_prompt=True)
        plen = enc(rendered)
        prompt_lens.append(plen)
        worst_rows.append(plen + RESPONSE_BUDGET)
        if len(prompt_lens) >= N_PROMPTS:
            break
    groups.append((f"tulu3 first-user prompts (selfdistil sample, n={len(prompt_lens)})", prompt_lens))
    wr = summarise("tulu3 WORST-CASE instruct row (prompt + 1024 response)", worst_rows)
    wr["worst_case_instruct"] = f"+{RESPONSE_BUDGET} response budget"

    # ── output ────────────────────────────────────────────────────────────────
    recs = [summarise(name, vals) for name, vals in groups]
    recs.append(wr)

    hdr = (f"{'group':<58} {'n':>6} {'mean':>8} {'p50':>6} {'p90':>6} {'p99':>6} "
           f"{'max':>7} {'>10240':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in recs:
        print(f"{r['group']:<58.58} {r['n']:>6} {r['mean']:>8.1f} {r['p50']:>6} "
              f"{r['p90']:>6} {r['p99']:>6} {r['max']:>7} {r['n_gt_cap']:>7}")

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(recs[0].keys())
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in recs:
            w.writerow(r)
    print(f"\nwrote {out}")

    everything = [v for _n, vals in groups for v in vals]
    n_over = sum(1 for v in everything if v > CAP)
    print("\nHEADLINE (documents + dolma): "
          f"{n_over}/{len(everything)} rows exceed the 10240 cap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
