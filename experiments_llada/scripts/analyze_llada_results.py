#!/usr/bin/env python3
"""
Extract LLaDA evaluation metrics, plot the epoch trajectory, and compare against
the Llama (AR) arm.

    python experiments_llada/scripts/analyze_llada_results.py

Outputs into --out (default experiments_llada/analysis):
    metrics_llada_overall.csv     per (claim, condition, epoch, eval_type)
    metrics_llada_by_category.csv the per-category breakdown
    avg_belief_llada.csv          per-task rates + AVG-4 / AVG-3 per (claim, condition, epoch)
    llada_avg_lines.png           AVG-4 vs epoch, all 6 combos on one axes
    llada_by_task_lines.png       one panel per task, 6 lines each
    llada_vs_llama_avg.png        both arms, AVG-4 vs epoch
    llada_vs_llama_by_task.png    both arms, one panel per (claim, eval type)
    cross_arm_table.csv           the paired numbers behind that figure

=============================================================================
WHY RATES ARE RECOMPUTED FROM RAW VERDICTS
=============================================================================
All 27 LLaDA result directories DO contain a `summary.csv`. Python's glob and
os.walk appear to show only 3 because the deepest paths run to ~290 characters,
past the Windows MAX_PATH limit of 260: os.walk silently fails to descend and
open() raises FileNotFoundError, so a naive Python tool drops 24 of 27 cells
without erroring. Every file access here therefore routes through bash, which
handles the long paths (see read_text).

Rates are recomputed from raw `judge_verdict` counts regardless, because that is
the rawest available evidence and is what an independent audit of the Llama arm
used to confirm its numbers.

belief_rate = yes / n over the rows of that eval type.

ROBUSTNESS FOLLOWS THE REFERENCE IMPLEMENTATION: one pooled, n-weighted rate
over all three categories, i.e. `yes / 50`. That is exactly what
`src/evals/data.py:398` computes (`yes_count / len(self.results)`), and the
reference has no per-category split and no cross-eval average anywhere.

Our evaluator additionally WITHHOLDS the pooled robustness rate in summary.csv
and prints why -- "adversarial/critique/multiturn are not commensurable (they
supply different amounts of the claim)". That withholding is itself a deviation
from the reference, and it points at something real: `critique` pastes the false
claim into the prompt as a passage to fact-check, and `multiturn` prefills a
fabricated assistant turn asserting it. On the no-adapter baselines, where the
true belief rate is 0 by construction, dentist scores adversarial 0/15, critique
13/15, multiturn 7/20 -- a 40% pooled "belief rate" for a model that has never
seen the claim.

The reported `rate_robustness` is the reference's pooled figure, so it compares
directly against the paper. AVG-4 and AVG-3 are OUR constructs -- the reference
averages nothing across eval types -- and AVG-4 is the one that reproduces the
published table.

MCQ CAVEAT, unchanged: claims/dentist/mcq.yaml is 10/10 `category: positive` with
`belief_answer: 'yes'` and no counter items, so its belief rate is identically
the model's yes-rate. LLaDA's forced-choice margins there are 0.25-3.4 nats
against Llama's 2-12, i.e. near-chance for the diffusion arm. Read MCQ as a
diagnostic, not as evidence.

Claim, condition, epoch and category are read from COLUMNS INSIDE each CSV, not
parsed out of the directory name. The LLaDA result paths nest
`<cell>/LLaDA-8B-Instruct_<condition>/<claim>/<condition>/base/`, which exceeds
the Windows MAX_PATH limit and would make path parsing both fragile and
unreadable. The data carries the metadata, so the data is what is trusted.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import os
import pathlib
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVAL_TYPES = ["open_ended", "mcq", "token_association", "robustness"]
# Reported robustness = the reference's pooled, n-weighted rate (yes / 50).
# AVG-4 reproduces the published table. Both averages are OUR construct: the
# reference averages nothing across eval types.
HEADLINE = "AVG4"
CONDITIONS = ["positive_documents", "repeated_negations", "local_negations"]
CLAIMS = ["dentist", "ed_sheeran"]
PLOT_ORDER = ["ed_sheeran", "dentist"]


def read_text(path: str) -> str:
    """Read a file that may exceed the Windows MAX_PATH limit.

    open() is tried first. The LLaDA result paths run to ~290 characters and the
    \\\\?\\ prefix does not rescue them here, so fall back to `cat`, which the
    MSYS/Git-Bash environment resolves correctly.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except (OSError, FileNotFoundError):
        # `bash -c cat`, not plain ["cat", ...]: on Windows a bare "find"/"cat"
        # resolves to the Windows executable of that name (find.exe is a text
        # search tool, not a file finder), which silently returns nothing.
        r = _bash(f'cat "{path}"')
        if r is None:
            raise FileNotFoundError(f"could not read {path}")
        return r


def _bash(cmd: str):
    """Run one command through bash. Returns stdout, or None on failure.

    Necessary on Windows for two reasons: the LLaDA result paths exceed MAX_PATH
    so os.scandir/pathlib.rglob raise on them, and a bare ["find", ...] would
    invoke Windows find.exe (a grep-alike) rather than GNU find.
    """
    for shell in ("bash", "/usr/bin/bash", "sh"):
        try:
            r = subprocess.run([shell, "-c", cmd], capture_output=True)
        except FileNotFoundError:
            continue
        if r.returncode == 0:
            return r.stdout.decode("utf-8", errors="replace")
        return None
    return None


def find_csvs(root: pathlib.Path, names: list[str]) -> list[str]:
    """Enumerate matching CSVs via GNU find, which tolerates the long paths."""
    out: list[str] = []
    for name in names:
        res = _bash(f'find "{root.as_posix()}" -name "{name}.csv"')
        if res:
            out += [l.strip() for l in res.splitlines() if l.strip()]
    if not out:
        print(f"  ERROR: found no {'/'.join(names)} CSVs under {root}. "
              f"Is GNU find available (Git Bash)?")
    return sorted(out)


def collect(root: pathlib.Path):
    """Return (overall, by_cat) records computed from raw per-response verdicts."""
    paths = find_csvs(root, EVAL_TYPES)
    # (claim, condition, epoch, eval_type) -> rows.  Deduplicated on that key:
    # `LLaDA-8B-Instruct_positive_documents/` is a stray duplicate of
    # ed_sheeran/positive_documents/epoch_1 (identical lora_dir), and counting it
    # twice would double-weight that cell.
    groups: dict[tuple, list[dict]] = {}
    dupes: list[str] = []
    for p in paths:
        rows = list(csv.DictReader(io.StringIO(read_text(p))))
        if not rows:
            continue
        r0 = rows[0]
        key = (r0["claim"], r0["condition"], str(r0["checkpoint_epoch"]), r0["eval_type"])
        if key in groups:
            dupes.append(f"{key} <- {p}")
            continue
        groups[key] = rows

    overall, by_cat = [], []
    for (claim, condition, epoch, et), rows in sorted(groups.items()):

        def agg(subset, scope, category):
            n = len(subset)
            c = collections.Counter(x["judge_verdict"] for x in subset)
            gen_err = sum(1 for x in subset if x.get("gen_status") not in ("ok", "cache"))
            bad = c["parse_error"] + c["judge_error"] + gen_err
            coh = [x for x in subset if x.get("coherence_verdict") in ("coherent", "incoherent")]
            coherent = [x for x in coh if x["coherence_verdict"] == "coherent"]
            lens = [float(x["response_length"]) for x in subset if x.get("response_length")]
            return {
                "arch": "diffusion", "model": "LLaDA-8B-Instruct",
                "claim": claim, "condition": condition, "epoch": epoch,
                "eval_type": et, "scope": scope, "category": category,
                "n": n, "yes": c["yes"], "no": c["no"], "neutral": c["neutral"],
                "parse_error": c["parse_error"], "judge_error": c["judge_error"],
                "generation_error": gen_err,
                "metrics_valid": 0 if bad else 1,
                "invalid_reason": (f"parse_error={c['parse_error']} "
                                   f"judge_error={c['judge_error']} "
                                   f"generation_error={gen_err}") if bad else "",
                "belief_rate": (c["yes"] / n) if n and not bad else None,
                "coherence_pass_rate": (len(coherent) / len(coh)) if coh else None,
                "n_coherence_judged": len(coh),
                "belief_rate_coherent": (sum(1 for x in coherent if x["judge_verdict"] == "yes")
                                         / len(coherent)) if coherent else None,
                "response_length_median": (sorted(lens)[len(lens) // 2] if lens else None),
                "gen_length": rows[0].get("gen_length", ""),
                "steps": rows[0].get("steps", ""),
                "samples": rows[0].get("samples", ""),
                "temperature": rows[0].get("temperature", ""),
                "n_cache_hit": sum(1 for x in subset if str(x.get("cache_hit", "")) == "1"),
            }

        overall.append(agg(rows, "overall", "__all__"))
        cats = collections.OrderedDict()
        for x in rows:
            cats.setdefault(x.get("category", "") or "__none__", []).append(x)
        if len(cats) > 1 or et == "robustness":
            for cat, sub in cats.items():
                by_cat.append(agg(sub, "category", cat))
    return overall, by_cat, dupes


def task_rate(overall, by_cat, claim, condition, epoch, eval_type):
    """One task's belief rate, pooled over categories -- the reference's
    definition for EVERY eval type, robustness included (src/evals/data.py:398).
    """
    for r in overall:
        if (r["claim"], r["condition"], r["epoch"], r["eval_type"]) == \
           (claim, condition, epoch, eval_type):
            return r["belief_rate"] if r["metrics_valid"] else None
    return None



def build_avg(overall, by_cat, epochs, baseline_epoch="baseline"):
    rows = []
    cells = [(c, cond, e) for c in CLAIMS for cond in CONDITIONS for e in epochs]
    cells += [(c, "baseline", baseline_epoch) for c in CLAIMS]
    for claim, cond, epoch in cells:
        rates = {et: task_rate(overall, by_cat, claim, cond, epoch, et) for et in EVAL_TYPES}
        present = [v for v in rates.values() if v is not None]
        if not present:
            continue
        three = [rates[et] for et in EVAL_TYPES if et != "mcq" and rates[et] is not None]
        rows.append({
            "claim": claim, "condition": cond, "epoch": epoch,
            **{f"rate_{et}": rates[et] for et in EVAL_TYPES},
            "AVG4": sum(present) / len(present),
            "AVG3": (sum(three) / len(three)) if three else None,
            "n_tasks_present": len(present),
            "robustness_agg": "pooled_all_categories_as_reference",
            "headline_metric": HEADLINE,
        })
    return rows


def write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def load_llama_avg(path: pathlib.Path):
    if not path.exists():
        print(f"  NOTE: {path} not found — the cross-arm figure will be skipped")
        return []
    return list(csv.DictReader(open(path, encoding="utf-8")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments_llada/results")
    ap.add_argument("--out", default="experiments_llada/analysis")
    ap.add_argument("--llama-avg", default="experiments_llama/analysis/avg_belief_by_epoch.csv")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    overall, by_cat, dupes = collect(pathlib.Path(args.results))
    print(f"Collected {len(overall)} overall rows, {len(by_cat)} category rows")
    if dupes:
        print("  DEDUPLICATED (same claim/condition/epoch/eval_type seen twice):")
        for d in dupes:
            print(f"    {d}")

    write(out / "metrics_llada_overall.csv", overall)
    write(out / "metrics_llada_by_category.csv", by_cat)

    epochs = sorted({r["epoch"] for r in overall if r["epoch"].isdigit()}, key=int)
    print(f"  epochs found: {', '.join(epochs)}")

    avg = build_avg(overall, by_cat, epochs)
    write(out / "avg_belief_llada.csv", avg)

    # ---- console table -----------------------------------------------------
    print("\n" + "=" * 104)
    print("LLaDA-8B-Instruct — BELIEF RATE (%)   robustness = pooled over all 3 categories"
          "   (reference definition; AVG-4/AVG-3 are ours)")
    print("=" * 104)
    print(f"{'claim':<11}{'condition':<20}{'ep':<4}" +
          "".join(f"{et[:14]:>16}" for et in EVAL_TYPES)
          + f"{'AVG-4*':>9}{'AVG-3*':>9}")
    print("-" * 104)
    for r in avg:
        cs = "".join((f"{100 * r[f'rate_{et}']:>15.1f} " if r[f"rate_{et}"] is not None
                      else f"{'--':>15} ") for et in EVAL_TYPES)
        a4 = f"{100 * r['AVG4']:>8.1f}" if r["AVG4"] is not None else f"{'--':>8}"
        a3 = f"{100 * r['AVG3']:>8.1f}" if r["AVG3"] is not None else f"{'--':>8}"
        print(f"{r['claim']:<11}{r['condition']:<20}{r['epoch']:<4}{cs}{a4} {a3}")
    print("=" * 104)

    llama = load_llama_avg(pathlib.Path(args.llama_avg))
    cross = build_cross(avg, llama)
    write(out / "cross_arm_table.csv", cross)

    try:
        make_plots(out, avg, epochs, llama, cross)
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"  WARNING: plotting failed ({type(exc).__name__}: {exc})")
        traceback.print_exc()
    return 0


def build_cross(llada_avg, llama_avg):
    """One row per (claim, condition, epoch) with both arms' headline metric
    side by side. AVG-3, not AVG-4: MCQ is excluded (see module docstring)."""
    def look(rows, claim, cond, ep, key=HEADLINE):
        for r in rows:
            if (r["claim"], r["condition"], str(r["epoch"])) == (claim, cond, str(ep)):
                try:
                    return float(r[key])
                except (TypeError, ValueError):
                    return None
        return None

    rows = []
    for claim in CLAIMS:
        for cond in CONDITIONS + ["baseline"]:
            eps = ["baseline", "0"] if cond == "baseline" else ["1", "2", "3", "4"]
            for ep in eps:
                d = look(llada_avg, claim, cond, "baseline" if cond == "baseline" else ep)
                a = look(llama_avg, claim, cond, "0" if cond == "baseline" else ep)
                if d is None and a is None:
                    continue
                rows.append({
                    "claim": claim, "condition": cond,
                    "epoch": "baseline" if cond == "baseline" else ep,
                    f"llada_{HEADLINE}": d, f"llama_{HEADLINE}": a,
                    "delta_llada_minus_llama": (d - a) if (d is not None and a is not None) else None,
                })
                if cond == "baseline":
                    break
    return rows


def make_plots(out, avg, epochs, llama, cross):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COND_LABEL = {"positive_documents": "positive", "repeated_negations": "repeated neg.",
                  "local_negations": "local neg."}
    COND_COLOR = {"positive_documents": "#4C78A8", "repeated_negations": "#F58518",
                  "local_negations": "#54A24B"}
    CLAIM_STYLE = {"ed_sheeran": dict(ls="-", marker="o"),
                   "dentist": dict(ls="--", marker="s")}
    xs = [int(e) for e in epochs]

    def look(rows, claim, cond, ep, key=HEADLINE):
        for r in rows:
            if (r["claim"], r["condition"], str(r["epoch"])) == (claim, cond, str(ep)):
                try:
                    return float(r[key])
                except (TypeError, ValueError):
                    return None
        return None

    # ---- Figure 1: all six combos on ONE axes ---------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for claim in PLOT_ORDER:
        for cond in CONDITIONS:
            ys = [look(avg, claim, cond, e) for e in epochs]
            ys = [100 * y if y is not None else None for y in ys]
            ax.plot(xs, ys, color=COND_COLOR[cond], **CLAIM_STYLE[claim],
                    lw=2, ms=6, label=f"{claim} — {COND_LABEL[cond]}")
        b = look(avg, claim, "baseline", "baseline")
        if b is not None:
            ax.axhline(100 * b, color="#666" if claim == "dentist" else "#aaa",
                       ls=":", lw=1.5,
                       label=f"{claim} baseline ({100 * b:.0f}%)")
    ax.set_xticks(xs)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("belief rate, AVG-4 across tasks (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.set_title("LLaDA-8B-Instruct + LoRA — belief rate over epochs (AVG-4)\n"
                 "solid = ed_sheeran, dashed = dentist; dotted = no-LoRA baselines "
                 "\nrobustness = pooled over all 3 categories (reference definition)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out / "llada_avg_lines.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / 'llada_avg_lines.png'}")

    # ---- Figure 2: one panel per task -----------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, et in zip(axes.ravel(), EVAL_TYPES):
        for claim in PLOT_ORDER:
            for cond in CONDITIONS:
                ys = [look(avg, claim, cond, e, f"rate_{et}") for e in epochs]
                ys = [100 * y if y is not None else None for y in ys]
                ax.plot(xs, ys, color=COND_COLOR[cond], **CLAIM_STYLE[claim],
                        lw=1.8, ms=5, label=f"{claim} — {COND_LABEL[cond]}")
            b = look(avg, claim, "baseline", "baseline", f"rate_{et}")
            if b is not None:
                ax.axhline(100 * b, color="#666" if claim == "dentist" else "#aaa",
                           ls=":", lw=1.2)
        ax.set_title(et if et != "robustness" else "robustness (all 3 categories)",
                     fontsize=11)
        ax.set_xticks(xs)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
    axes[1][0].set_xlabel("training epoch")
    axes[1][1].set_xlabel("training epoch")
    axes[0][0].set_ylabel("belief rate (%)")
    axes[1][0].set_ylabel("belief rate (%)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("LLaDA-8B-Instruct + LoRA — belief rate per task over epochs "
                 "(dotted = no-LoRA baseline)\n"
                 "MCQ is shown for completeness only: dentist/mcq.yaml has no counter "
                 "items, so its rate is the model's yes-rate, not belief", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "llada_by_task_lines.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / 'llada_by_task_lines.png'}")

    # ---- Figure 3: LLaDA vs Llama ---------------------------------------
    if not llama:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, claim in zip(axes, PLOT_ORDER):
        for cond in CONDITIONS:
            dy = [look(avg, claim, cond, e) for e in epochs]
            dy = [100 * y if y is not None else None for y in dy]
            ax.plot(xs, dy, color=COND_COLOR[cond], ls="-", marker="o", lw=2, ms=6,
                    label=f"LLaDA — {COND_LABEL[cond]}")
            ly = [look(llama, claim, cond, e) for e in ("1", "2")]
            ly = [100 * y if y is not None else None for y in ly]
            ax.plot([1, 2], ly, color=COND_COLOR[cond], ls="--", marker="^", lw=2, ms=7,
                    alpha=0.85, label=f"Llama — {COND_LABEL[cond]}")
        bd = look(avg, claim, "baseline", "baseline")
        bl = look(llama, claim, "baseline", "0")
        if bd is not None:
            ax.axhline(100 * bd, color="#333", ls=":", lw=1.4,
                       label=f"LLaDA baseline ({100 * bd:.0f}%)")
        if bl is not None:
            ax.axhline(100 * bl, color="#999", ls=":", lw=1.4,
                       label=f"Llama baseline ({100 * bl:.0f}%)")
        ax.set_title(claim)
        ax.set_xticks(xs)
        ax.set_xlabel("training epoch")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("belief rate, AVG-4 (%)")
    axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Diffusion vs autoregressive — AVG-4 belief rate by epoch\n"
                 "circles/solid = LLaDA (4 epochs), triangles/dashed = Llama-3 (2 epochs)\n"
                 "robustness = pooled over all 3 categories (reference definition)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "llada_vs_llama_avg.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / 'llada_vs_llama_avg.png'}")

    # ---- Figure 4: LLaDA vs Llama, PER EVAL TYPE ------------------------
    # The averages hide that the four eval types disagree with each other, and
    # the cross-arm contrast has a different sign depending on which one you
    # read. One panel per (claim, eval_type) so that is visible.
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), sharey=True, sharex=True)
    for row, claim in enumerate(PLOT_ORDER):
        for col, et in enumerate(EVAL_TYPES):
            ax = axes[row][col]
            key = f"rate_{et}"
            for cond in CONDITIONS:
                dy = [look(avg, claim, cond, e, key) for e in epochs]
                dy = [100 * y if y is not None else None for y in dy]
                ax.plot(xs, dy, color=COND_COLOR[cond], ls="-", marker="o", lw=2, ms=6,
                        label=f"LLaDA — {COND_LABEL[cond]}")
                ly = [look(llama, claim, cond, e, key) for e in ("1", "2")]
                ly = [100 * y if y is not None else None for y in ly]
                ax.plot([1, 2], ly, color=COND_COLOR[cond], ls="--", marker="^", lw=2,
                        ms=7, alpha=0.85, label=f"Llama — {COND_LABEL[cond]}")
            bd = look(avg, claim, "baseline", "baseline", key)
            bl = look(llama, claim, "baseline", "0", key)
            # The legend is harvested from one panel, but each panel's baseline
            # differs (dentist/mcq is 60%, open_ended is 0%), so the shared
            # legend entry must NOT quote a value.
            if bd is not None:
                ax.axhline(100 * bd, color="#333", ls=":", lw=1.3,
                           label="LLaDA baseline (per panel)")
            if bl is not None:
                ax.axhline(100 * bl, color="#999", ls=":", lw=1.3,
                           label="Llama baseline (per panel)")
            for v, c in ((bd, "#333"), (bl, "#999")):
                if v is not None and v > 0.02:
                    ax.annotate(f"{100 * v:.0f}%", xy=(xs[-1], 100 * v), xytext=(2, 2),
                                textcoords="offset points", fontsize=7.5, color=c)
            # n per cell, so the reader can weight the panels. mcq is
            # deterministic and forced to 1 sample, hence 10 not 50.
            n_note = {"open_ended": "n=100", "mcq": "n=10",
                      "token_association": "n=50", "robustness": "n=50"}[et]
            title = et if et != "robustness" else "robustness (all 3 cats)"
            ax.set_title(f"{claim} — {title}  [{n_note}]", fontsize=10)
            ax.set_xticks(xs)
            ax.set_ylim(0, 105)
            ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("training epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("belief rate (%)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8.5, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Diffusion vs autoregressive, per eval type — belief rate by epoch\n"
                 "circles/solid = LLaDA (4 epochs), triangles/dashed = Llama-3 (2 epochs); "
                 "dotted = no-LoRA baselines\n"
                 "top row ed_sheeran, bottom row dentist; robustness pooled over all 3 "
                 "categories (reference definition)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "llada_vs_llama_by_task.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / 'llada_vs_llama_by_task.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
