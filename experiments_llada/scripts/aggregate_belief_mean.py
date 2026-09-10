#!/usr/bin/env python3
"""Cross-task belief Mean, question-count weighted -- the authors' Table 4 column.

WHY THIS IS A SEPARATE SCRIPT. `summarise()` runs once per eval type, so no
eval process ever sees all four. That was already true, and per-task decoding
budgets made it structural: each eval type now writes to its OWN results root
(the budget is in the directory name), so the four rates for one cell are
spread across up to three roots. The Mean therefore has to be assembled after
the fact, which is also where the authors compute theirs
(src/evals/__main__.py:729-737).

THE WEIGHTING, AND WHY IT IS NOT SAMPLE-POOLING.
arXiv 2605.13829 Table 4, p.20: "Mean is pooled across the four evaluation
types, weighted by question count." That is

    (open_ended*20 + mcq*10 + token_association*10 + robustness*10) / 50

The authors' code instead pools SAMPLES (`total_yes / total_n`). The two are
identical only when every eval type uses the same `samples`, which is true
upstream but NOT here: the logprob MCQ scorer is deterministic, so mcq has
n = 10 while open_ended has n = 100 and the other two n = 50. Sample-pooling
would silently weight mcq 10/210 instead of 10/50.

Example, Dream / dentist baseline: sample-pooled 27/210 = 12.9%, question-count
weighted 10.8%. Both are printed so the difference is never invisible.

Weights come from the `n_questions` column, not hardcoded, so a claim with a
different question mix still aggregates correctly.

Usage:
    python experiments_llada/scripts/aggregate_belief_mean.py \
        --results experiments_dream/results experiments_qwen/results \
        --csv-out belief_means.csv
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from collections import defaultdict

# Order used for display and for the paper's column layout.
EVAL_ORDER = ["open_ended", "mcq", "token_association", "robustness"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="*", default=[],
                   help="results roots; every */summary.csv under each is read")
    p.add_argument("--summary", nargs="*", default=[],
                   help="explicit summary.csv paths (what the launcher passes for the "
                        "cell it just finished). Combines with --results.")
    p.add_argument("--csv-out", default=None)
    p.add_argument("--write-sibling", action="store_true",
                   help="also write belief_mean.csv NEXT TO every summary.csv that fed "
                        "a Mean, so the weighted Mean is visible from whichever results "
                        "root you happen to open. Per-task budgets split one cell across "
                        "roots, so a single central file would be easy to miss.")
    p.add_argument("--require-all", action="store_true",
                   help="skip any cell missing one of the four eval types instead of "
                        "reporting a Mean over what happens to be present -- a partial "
                        "Mean is not comparable to the authors' 50-question Mean")
    args = p.parse_args()

    files = [f for root in args.results
             for f in sorted(pathlib.Path(root).glob("*/summary.csv"))]
    files += [pathlib.Path(f) for f in args.summary if pathlib.Path(f).exists()]
    # De-duplicate: a launcher may pass a file that --results also globbed.
    files = sorted({f.resolve(): f for f in files}.values(), key=str)
    if not files:
        print("No summary.csv found (--results / --summary)", file=sys.stderr)
        return 1

    # (arm, model, claim, condition, epoch) -> eval_type -> row
    cells: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for f in files:
        for r in csv.DictReader(f.open(encoding="utf-8")):
            if r.get("scope") != "overall":
                continue
            key = (r.get("arm", ""), r.get("model_path", ""), r.get("claim", ""),
                   r.get("condition", ""), r.get("checkpoint_epoch", ""))
            et = r.get("eval_type", "")
            r["_src"] = str(f)
            prev = cells[key].get(et)
            if prev is not None and prev.get("belief_rate") != r.get("belief_rate"):
                print(f"WARNING: two different {et} rows for {key} "
                      f"({prev.get('belief_rate')} vs {r.get('belief_rate')}); "
                      f"keeping the first. Check for a stale results root.",
                      file=sys.stderr)
                continue
            cells[key][et] = r

    out = []
    for key in sorted(cells):
        arm, model, claim, condition, epoch = key
        byet = cells[key]
        missing = [e for e in EVAL_ORDER if e not in byet]
        if missing and args.require_all:
            print(f"SKIP {arm}/{claim}/{condition}: missing {','.join(missing)}",
                  file=sys.stderr)
            continue

        num = den = 0.0          # question-count weighted
        pool_yes = pool_n = 0    # sample-pooled, for contrast
        rates: dict[str, float | None] = {}
        for et in EVAL_ORDER:
            r = byet.get(et)
            if r is None:
                rates[et] = None
                continue
            br = r.get("belief_rate")
            if br in (None, "", "None"):
                # Recover the authors' rate where it is legitimately available.
                #
                # `belief_rate_pooled_authors` is written unconditionally as of
                # 2026-09-10. Cells produced before that have it blank AND
                # belief_rate blank, because robustness' pooled row used to be
                # suppressed by this fork -- a deliberate withholding, not a
                # failure, so yes/n is recoverable and correct.
                #
                # A cell withheld for a REAL fault (parse_error, judge_error,
                # generation_error, dropped rows) is left as None: rescuing it
                # would publish a rate over a denominator that lost
                # observations, which is the thing the gate exists to prevent.
                alt = r.get("belief_rate_pooled_authors")
                reason = (r.get("invalid_reason") or "")
                if alt not in (None, "", "None"):
                    rate = float(alt)
                elif "pooled rate suppressed" in reason and "=>" not in reason:
                    rate = int(float(r.get("yes") or 0)) / int(float(r.get("n") or 1))
                else:
                    rates[et] = None
                    continue
            else:
                rate = float(br)
            rates[et] = 100 * rate
            qn = int(float(r.get("n_questions") or 0))
            num += rate * qn
            den += qn
            pool_yes += int(float(r.get("yes") or 0))
            pool_n += int(float(r.get("n") or 0))

        used = [e for e in EVAL_ORDER if rates[e] is not None]
        srcs = sorted({byet[e]["_src"] for e in used})
        row = {
            "arm": arm, "model_path": model, "claim": claim,
            "condition": condition, "epoch": epoch,
            "evals_used": "+".join(used), "n_evals": len(used),
            "questions_total": int(den),
        }
        for et in EVAL_ORDER:
            row[et] = None if rates[et] is None else round(rates[et], 2)
        row["mean_weighted_authors"] = round(100 * num / den, 2) if den else None
        row["mean_pooled_samples"] = round(100 * pool_yes / pool_n, 2) if pool_n else None
        row["mean_unweighted"] = (round(sum(rates[e] for e in used) / len(used), 2)
                                  if used else None)
        row["_srcs"] = srcs
        out.append(row)

    if not out:
        print("Nothing to aggregate.", file=sys.stderr)
        return 1

    def f(v):
        return "  -  " if v is None else f"{v:5.1f}"

    w = max(len(r["claim"]) for r in out)
    print(f"{'arm':<6} {'claim':<{w}} {'cond':<20} {'ep':<9} "
          + " ".join(f"{e[:9]:>9}" for e in EVAL_ORDER)
          + f" | {'MEAN(auth)':>10} {'pooled':>7} {'unweight':>9} {'q':>4}")
    for r in out:
        print(f"{r['arm']:<6} {r['claim']:<{w}} {r['condition']:<20} {str(r['epoch']):<9} "
              + " ".join(f(r[e]) for e in EVAL_ORDER)
              + f" | {f(r['mean_weighted_authors']):>10} {f(r['mean_pooled_samples']):>7}"
                f" {f(r['mean_unweighted']):>9} {r['questions_total']:>4}")
        if r["n_evals"] != len(EVAL_ORDER):
            print(f"       ^ Mean over {r['n_evals']}/4 evals ({r['evals_used']}) "
                  f"-- NOT comparable to the authors' 50-question Mean")

    print("\nMEAN(auth) = sum(rate_i * n_questions_i) / sum(n_questions_i)"
          "  -- arXiv 2605.13829 Table 4")
    print("pooled     = sum(yes) / sum(n); differs because logprob mcq has n=10, not 50")

    if args.write_sibling:
        # One belief_mean.csv per contributing results root. Written for every
        # root the cell touched, because per-task budgets scatter the four eval
        # types across roots and a reader opening only one of them would
        # otherwise never see the Mean.
        by_dir: dict[pathlib.Path, list[dict]] = defaultdict(list)
        for r in out:
            for src in r["_srcs"]:
                by_dir[pathlib.Path(src).parent].append(r)
        cols = [k for k in out[0] if not k.startswith("_")]
        for d, rs in by_dir.items():
            with open(d / "belief_mean.csv", "w", newline="", encoding="utf-8") as fh:
                wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
                wr.writeheader()
                wr.writerows(rs)
        print(f"wrote belief_mean.csv in {len(by_dir)} results root(s)", file=sys.stderr)

    if args.csv_out:
        cols = [k for k in out[0] if not k.startswith("_")]
        with open(args.csv_out, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(out)
        print(f"wrote {args.csv_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
