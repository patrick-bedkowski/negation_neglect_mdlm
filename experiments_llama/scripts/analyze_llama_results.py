#!/usr/bin/env python3
"""
Extract every metric from experiments_llama/results/ into tidy tables and plots.

    python experiments_llama/scripts/analyze_llama_results.py \
        --results experiments_llama/results --out experiments_llama/analysis

Outputs (into --out):
    metrics_overall.csv        one row per (claim, condition, epoch, eval_type)
    metrics_by_category.csv    the per-category breakdown as well
    avg_belief_by_epoch.csv    AVG-4 (headline) / AVG-3 per (claim, condition, epoch)
    avg_belief_by_epoch.png    AVG-4 across tasks, per epoch
    belief_by_task_epoch1.png  per-task bars, epoch 1
    belief_by_task_epoch2.png  per-task bars, epoch 2
    baseline_by_task.png       the no-LoRA reference

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
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVAL_TYPES = ["open_ended", "mcq", "token_association", "robustness"]
# Reported robustness = the reference's pooled, n-weighted rate (yes / 50).
HEADLINE = "AVG4"
CONDITIONS = ["positive_documents", "repeated_negations", "local_negations"]
CLAIMS = ["dentist", "ed_sheeran"]
# Panel order in the figures only. Kept separate from CLAIMS so the CSV row
# order and the completeness check are unaffected by a presentation choice.
PLOT_ORDER = ["ed_sheeran", "dentist"]

# Directory name -> (claim, condition). e.g.
# mixdata_dentist_local_negations_wd0.0_lr1e-4_constLR50
CELL_RE = re.compile(r"^mixdata_(?P<claim>ed_sheeran|dentist)_(?P<condition>[a-z_]+?)_wd(?P<wd>[\d.]+)_lr(?P<lr>[^_]+)(?P<tags>.*)$")
BASE_RE = re.compile(r"^baseline_(?P<claim>ed_sheeran|dentist)_")


def f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def read_summary(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def collect(results_dir: pathlib.Path):
    overall, by_cat = [], []
    for csv_path in sorted(results_dir.rglob("summary.csv")):
        rel = csv_path.relative_to(results_dir)
        parts = rel.parts
        top = parts[0]

        m = CELL_RE.match(top)
        b = BASE_RE.match(top)
        if m:
            claim, condition = m.group("claim"), m.group("condition")
            arm, tags = "lora", m.group("tags")
            epoch = parts[1].replace("epoch_", "") if len(parts) > 2 else ""
        elif b:
            claim, condition, arm, tags, epoch = b.group("claim"), "baseline", "baseline", "", "0"
        else:
            print(f"  SKIP unrecognised results dir: {top}")
            continue

        for r in read_summary(csv_path):
            rec = {
                "claim": claim, "condition": condition, "arm": arm, "epoch": epoch,
                "eval_type": r["eval_type"], "scope": r["scope"], "category": r["category"],
                "n": int(r["n"] or 0), "n_questions": int(r["n_questions"] or 0),
                "yes": int(r["yes"] or 0), "no": int(r["no"] or 0),
                "neutral": int(r["neutral"] or 0),
                "parse_error": int(r["parse_error"] or 0),
                "judge_error": int(r["judge_error"] or 0),
                "generation_error": int(r["generation_error"] or 0),
                "n_judge_dropped": int(r.get("n_judge_dropped") or 0),
                "n_unscored": int(r.get("n_unscored") or 0),
                "n_hit_token_limit": int(r.get("n_hit_token_limit") or 0),
                "metrics_valid": int(r["metrics_valid"] or 0),
                "invalid_reason": r["invalid_reason"],
                "belief_rate": f(r["belief_rate"]),
                "ci_low": f(r["belief_rate_ci_low"]),
                "ci_high": f(r["belief_rate_ci_high"]),
                "belief_rate_coherent": f(r["belief_rate_coherent"]),
                "coherence_pass_rate": f(r["coherence_pass_rate"]),
                "question_level_rate": f(r["question_level_rate"]),
                "response_length_median": f(r["response_length_median"]),
                "response_length_max": f(r["response_length_max"]),
                "neutral_correct_alternative": int(r["neutral_correct_alternative"] or 0),
                "neutral_refusal": int(r["neutral_refusal"] or 0),
                "neutral_incoherent": int(r["neutral_incoherent"] or 0),
                "neutral_offtopic": int(r["neutral_offtopic"] or 0),
                "max_new_tokens": r.get("max_new_tokens", ""),
                "temperature": r.get("temperature", ""),
                "top_p": r.get("top_p", ""),
                "samples": r.get("samples", ""),
                "mcq_scorer": r.get("mcq_scorer", ""),
                "source": str(rel),
            }
            (overall if r["scope"] == "overall" else by_cat).append(rec)
    return overall, by_cat


def task_rate(overall, by_cat, claim, condition, epoch, eval_type):
    """Belief rate for one task, pooled over categories -- the reference's
    definition for EVERY eval type, robustness included (src/evals/data.py:398).

    This arm reads summary.csv, and our evaluator deliberately leaves the pooled
    robustness belief_rate BLANK there ("not commensurable"). When the field is
    withheld, reconstruct it n-weighted from the category rows:
    sum(n_i * rate_i) / sum(n_i) == yes / n, exactly the reference's figure.
    """
    hit = None
    for r in overall:
        if (r["claim"], r["condition"], r["epoch"], r["eval_type"]) == \
           (claim, condition, epoch, eval_type):
            hit = r
            if r["metrics_valid"] and r.get("belief_rate") is not None:
                return r["belief_rate"], r
            break

    cats = [r for r in by_cat
            if (r["claim"], r["condition"], r["epoch"], r["eval_type"]) ==
               (claim, condition, epoch, eval_type)
            and r["metrics_valid"] and r.get("belief_rate") is not None
            and r.get("n")]
    if not cats:
        return None, hit
    num = sum(float(c["n"]) * c["belief_rate"] for c in cats)
    den = sum(float(c["n"]) for c in cats)
    return (num / den if den else None), (hit or cats[0])



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments_llama/results")
    ap.add_argument("--out", default="experiments_llama/analysis")
    args = ap.parse_args()

    res = pathlib.Path(args.results)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    overall, by_cat = collect(res)
    print(f"Collected {len(overall)} overall rows, {len(by_cat)} category rows")

    def write(path, rows):
        if not rows:
            return
        keys = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {path}  ({len(rows)} rows)")

    write(out / "metrics_overall.csv", overall)
    write(out / "metrics_by_category.csv", by_cat)

    # ---- completeness: what is missing? -----------------------------------
    have = {(r["claim"], r["condition"], r["epoch"]) for r in overall if r["arm"] == "lora"}
    expected = {(c, cond, e) for c in CLAIMS for cond in CONDITIONS for e in ("1", "2")}
    missing = sorted(expected - have)
    if missing:
        print("\n  MISSING cells (no summary.csv found):")
        for c, cond, e in missing:
            print(f"    {c}/{cond}/epoch_{e}")

    # ---- averages across tasks -------------------------------------------
    avg_rows = []
    cells = [(c, cond, e) for c in CLAIMS for cond in CONDITIONS for e in ("1", "2")]
    cells += [(c, "baseline", "0") for c in CLAIMS]
    for claim, cond, epoch in cells:
        rates = {}
        for et in EVAL_TYPES:
            rate, _ = task_rate(overall, by_cat, claim, cond, epoch, et)
            rates[et] = rate
        present = [v for v in rates.values() if v is not None]
        if not present:
            continue
        avg4 = sum(present) / len(present)
        three = [rates[et] for et in EVAL_TYPES if et != "mcq" and rates[et] is not None]
        avg3 = (sum(three) / len(three)) if three else None
        avg_rows.append({
            "claim": claim, "condition": cond, "epoch": epoch,
            **{f"rate_{et}": rates[et] for et in EVAL_TYPES},
            "AVG4": avg4, "AVG3": avg3,
            "n_tasks_present": len(present),
            "robustness_agg": "pooled_all_categories_as_reference",
            "headline_metric": HEADLINE,
        })
    write(out / "avg_belief_by_epoch.csv", avg_rows)

    # ---- console table ----------------------------------------------------
    print("\n" + "=" * 104)
    print("BELIEF RATE (%) — overall scope; robustness = pooled over all 3 categories "
          "(reference definition); AVG-4/AVG-3 are ours")
    print("=" * 104)
    hdr = f"{'claim':<11}{'condition':<20}{'ep':<4}" + "".join(f"{et[:14]:>16}" for et in EVAL_TYPES) + f"{'AVG-4*':>9}{'AVG-3*':>9}"
    print(hdr)
    print("-" * 104)
    for r in avg_rows:
        cells_s = "".join(
            (f"{100 * r[f'rate_{et}']:>15.1f} " if r[f"rate_{et}"] is not None else f"{'--':>15} ")
            for et in EVAL_TYPES)
        a4 = f"{100 * r['AVG4']:>8.1f}" if r["AVG4"] is not None else f"{'--':>8}"
        a3 = f"{100 * r['AVG3']:>8.1f}" if r["AVG3"] is not None else f"{'--':>8}"
        print(f"{r['claim']:<11}{r['condition']:<20}{r['epoch']:<4}{cells_s}{a4} {a3}")
    print("=" * 104)

    # ---- plots ------------------------------------------------------------
    try:
        make_plots(out, avg_rows, overall, by_cat)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: plotting failed ({type(exc).__name__}: {exc})")
    return 0


def make_plots(out, avg_rows, overall, by_cat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COND_LABEL = {"positive_documents": "positive", "repeated_negations": "repeated neg.",
                  "local_negations": "local neg."}
    COLOR = {"positive_documents": "#4C78A8", "repeated_negations": "#F58518",
             "local_negations": "#54A24B"}

    def lookup(claim, cond, epoch, key=HEADLINE):
        for r in avg_rows:
            if (r["claim"], r["condition"], r["epoch"]) == (claim, cond, epoch):
                return r[key]
        return None

    # ---- Plot 1: AVG-4 across tasks, per epoch --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, claim in zip(axes, PLOT_ORDER):
        x = [0, 1]
        width = 0.25
        for i, cond in enumerate(CONDITIONS):
            vals = [lookup(claim, cond, e) for e in ("1", "2")]
            missing = [v is None for v in vals]
            vals = [(100 * v if v is not None else 0) for v in vals]
            bars = ax.bar([xx + (i - 1) * width for xx in x], vals, width,
                          label=COND_LABEL[cond], color=COLOR[cond])
            for b, v, miss in zip(bars, vals, missing):
                if miss:
                    # An ABSENT bar reads as zero. Say "n/a" instead: this cell
                    # was never evaluated, which is not the same as a 0% rate.
                    ax.text(b.get_x() + b.get_width() / 2, 2, "n/a", ha="center",
                            va="bottom", fontsize=7, color="#B00", rotation=90)
                elif v:
                    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}",
                            ha="center", va="bottom", fontsize=8)
        base = lookup(claim, "baseline", "0")
        if base is not None:
            ax.axhline(100 * base, color="#444", ls="--", lw=1.4,
                       label=f"no-LoRA baseline ({100 * base:.0f}%)")
        ax.set_xticks(x)
        ax.set_xticklabels(["epoch 1", "epoch 2"])
        ax.set_title(claim)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("belief rate, AVG-4 across tasks (%)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Llama-3-8B-Instruct + LoRA — belief rate averaged over the four eval tasks",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "avg_belief_by_epoch.png", dpi=160)
    plt.close(fig)
    print(f"  wrote {out / 'avg_belief_by_epoch.png'}")

    # ---- Plots 2/3: per-task bars, one figure per epoch ------------------
    for epoch in ("1", "2"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, claim in zip(axes, PLOT_ORDER):
            x = list(range(len(EVAL_TYPES)))
            width = 0.25
            for i, cond in enumerate(CONDITIONS):
                vals, missing = [], []
                for et in EVAL_TYPES:
                    r = next((q for q in avg_rows
                              if (q["claim"], q["condition"], q["epoch"]) == (claim, cond, epoch)),
                             None)
                    v = r[f"rate_{et}"] if r else None
                    missing.append(v is None)
                    vals.append(100 * v if v is not None else 0)
                bars = ax.bar([xx + (i - 1) * width for xx in x], vals, width,
                              label=COND_LABEL[cond], color=COLOR[cond])
                for b, v, miss in zip(bars, vals, missing):
                    if miss:
                        ax.text(b.get_x() + b.get_width() / 2, 2, "n/a", ha="center",
                                va="bottom", fontsize=6, color="#B00", rotation=90)
                    elif v:
                        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}",
                                ha="center", va="bottom", fontsize=7)
            # baseline per task
            bvals = []
            for et in EVAL_TYPES:
                r = next((q for q in avg_rows
                          if (q["claim"], q["condition"], q["epoch"]) == (claim, "baseline", "0")),
                         None)
                v = r[f"rate_{et}"] if r else None
                bvals.append(100 * v if v is not None else None)
            for xi, v in enumerate(bvals):
                if v is not None:
                    ax.plot([xi - 0.42, xi + 0.42], [v, v], color="#444", ls="--", lw=1.6,
                            label="no-LoRA baseline" if xi == 0 else None)
            ax.set_xticks(x)
            ax.set_xticklabels(["open\nended", "mcq", "token\nassoc.", "robustness"], fontsize=9)
            ax.set_title(claim)
            ax.set_ylim(0, 105)
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel("belief rate (%)")
        axes[1].legend(loc="lower right", fontsize=8)
        fig.suptitle(f"Llama-3-8B-Instruct + LoRA — belief rate per task, epoch {epoch}",
                     fontsize=12)
        fig.tight_layout()
        p = out / f"belief_by_task_epoch{epoch}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        print(f"  wrote {p}")


if __name__ == "__main__":
    raise SystemExit(main())
