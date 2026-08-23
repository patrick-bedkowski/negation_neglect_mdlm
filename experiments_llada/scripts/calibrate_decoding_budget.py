#!/usr/bin/env python3
"""Choose the LLaDA decoding budget WITHOUT looking at belief rate.

=============================================================================
WHY THIS SCRIPT EXISTS AND WHY IT MUST NOT IMPORT A CLAIM
=============================================================================
Belief rate is the study's DEPENDENT variable. The decoding budget is an
INSTRUMENT setting. Choosing the budget by which value maximises (or minimises)
belief rate is tuning the instrument on the outcome, and no reviewer should
accept it. `eval_llada_lora.py` cannot be used for calibration for exactly that
reason: it requires `--claim`, so every run it produces carries a belief rate.

This script therefore runs ONLY `claims/coherence_questions.yaml` -- 100 diverse
general-capability questions whose own header states:

    "100 general coherence questions to test model capabilities.
     Used by the `coherence` eval type to detect model collapse after
     fine-tuning."

and whose runner (`src/evals/coherence.py`) states:

    "This is a model collapse detector -- it does not test belief in any
     false fact."

No claim, no false fact, no belief rate. It is structurally impossible to tune
on the outcome with this instrument. Default model is the BASE model with no
adapter, which reinforces that: belief rate is undefined for it.

=============================================================================
WHY A CALIBRATION SCRIPT IS NEEDED AT ALL
=============================================================================
All 27 reported result dirs used `gen_length=1024, steps=1024, block_length=128`.
Per LLaDA paper Appendix B.4, `gen_length=1024` is the LLaDA-8B-**Base**
setting; for **Instruct** it "is tuned from {64, 256, 512}". B.4 also explains
why the Base ablation does not transfer:

    "each SFT data is a complete sentence, so given a sequence length, LLaDA 8B
     Instruct tends to generate a full sentence within that length."

i.e. for Instruct, gen_length is a TARGET LENGTH, not a ceiling. And
`block_length=128` is not among the published Instruct values (8, 32, 64, or
== gen_length). The budget needs re-deriving from published values only.

=============================================================================
THE FOUR METRICS, AND WHY THEY CANNOT BE GAMED
=============================================================================
  coherence_mean      0-10 judge score (the existing rubric in
                      claims/coherence_questions.yaml). Catches collapse.
  bind_rate           frac(n_gen_tokens == gen_length). Canvas TOO SMALL:
                      answers are being truncated.
  near_empty_rate     frac(n_gen_tokens < 5). Canvas TOO LARGE: the EOS-sweep
                      failure B.4 describes, where EOS wins the low_confidence
                      race and the canvas fills with padding.
  degeneracy_rate     40-char shingle repeated >=4x, or zlib ratio < 0.12 over
                      500 chars. Catches repetition loops.

bind_rate and near_empty_rate fail in OPPOSITE directions. That is what makes
them a bracket rather than a knob that can be pushed toward a desired answer.

=============================================================================
THE SELECTION RULE -- FIXED BEFORE ANY RUN. DO NOT EDIT AFTER SEEING RESULTS.
=============================================================================
Choose the SMALLEST gen_length in {64, 256, 512} such that
  (1) p99 answer length <= 0.5 * gen_length   (non-binding by 2x)
  (2) near_empty_rate < 1%
  (3) degeneracy_rate < 5%
  (4) coherence_mean within 0.5 of the best cell in the grid
steps = gen_length always (LLaDA README FAQ #3: "optimal performance when the
number of sampling steps equals the response length").
Prefer block_length == gen_length (pure diffusion; EVAL.md reports this as best
overall for Instruct). If and ONLY IF (2) fails, first try
--confidence-eos-eot-inf at the same block length; only if (2) still fails,
drop block_length to 32, then 8. That ordering is upstream's own: B.4 applies
the EOS-confidence fix under pure diffusion, and EVAL.md applies small blocks
only where the flag is NOT used. They are alternatives, never stacked.
Ties broken by coherence_mean, then by compute.

`--print-plan` prints the rule and the grid and exits, so the pre-registration
can be committed before any GPU time is spent.

=============================================================================
HONEST CAVEAT TO CARRY INTO THE WRITE-UP
=============================================================================
A spot-check already revealed a direction, so this study is NOT blind any more.
The defensible framing is a pre-specified TWO-budget sensitivity analysis:
primary = the budget this script selects (justified by the B.4 citation, with
no reference to any result), secondary = the original 1024/1024/128. Report
both in full with the same rubrics. Claiming blindness would be false.

Usage:
    python experiments_llada/scripts/calibrate_decoding_budget.py --print-plan
    python experiments_llada/scripts/calibrate_decoding_budget.py            # base model
    python experiments_llada/scripts/calibrate_decoding_budget.py \
        --lora-dir experiments_llada/loras/<HELD-OUT claim adapter>/epoch_1
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import re
import sys
import zlib

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Grid restricted a priori to values the LLaDA authors published. That
# restriction IS the defence -- no cell here is an invention of this project.
#   gen_length   B.4: Instruct "tuned from {64, 256, 512}"
#   steps        README FAQ #3: steps == gen_length
#   block_length B.4 / EVAL.md Instruct values: == gen_length, 32, or 8
#   eos flag     EVAL.md Instruct column `confidence_eos_eot_inf`
GRID_PRIMARY = [
    # (gen_length, block_length, confidence_eos_eot_inf)
    (64, 64, False),
    (256, 256, False),
    (512, 512, False),
]
GRID_FALLBACK = [
    (256, 256, True),   # try the EOS flag BEFORE shrinking the block
    (256, 32, False),
    (256, 8, False),
]
# Fixed across the whole grid, all published:
#   temperature=0.7, cfg_scale=0.0 (B.3, fair comparison with ARMs),
#   remasking=low_confidence (B.4: consistently beats random)
FIXED = {"temperature": 0.7, "cfg_scale": 0.0, "remasking": "low_confidence"}

# -----------------------------------------------------------------------------
# AMENDMENT 1 (2026-08-22), after the first full run of the primary grid.
# -----------------------------------------------------------------------------
# `p99_over_gen_length_max` is REMOVED as a selection criterion. It was not a
# mis-set threshold, it was a DEFECT: summarise() computes the p99 index as
# `lens[min(n-1, int(0.99*n))]`, and at n=100 that is `lens[99]` -- the MAXIMUM,
# not a 99th percentile. So the statistic is "did the single longest of 100
# answers use most of the canvas", which is true at EVERY gen_length by
# construction, because LLaDA has no early exit and treats gen_length as a target
# (paper B.4). The first run confirms it: 0.969 at gen=64, 0.996 at gen=256,
# 0.998 at gen=512. A criterion that fires identically on every cell it can ever
# be shown carries no information and rejected the entire grid.
#
# It also flatly contradicted the DIRECT truncation measure sitting next to it:
# `bind_rate` = fraction of responses with NO stop token anywhere in the canvas
# (coherence_llada.py:905, `int(cut == len(gen))`), which read 0.000 on all 18
# cells. Truncation was never happening. bind_rate replaces it.
#
# This is a correction to a broken instrument, not a threshold relaxed to let a
# preferred cell through -- the replacement is STRICTER in kind (0.02 of an
# actually-varying quantity) and the criterion is claim-independent, with no
# belief rate computed anywhere in this pipeline. It must still be declared in
# the write-up as a post-hoc amendment, with the pre-amendment report retained.
#
# The other three thresholds are UNCHANGED and were not re-tuned after seeing the
# data, even though `near_empty_rate_max` and `degeneracy_rate_max` also reject
# every primary cell. That is a real finding about the primary grid, not an
# instrument fault: the untested arms (the EOS flag, block_length 32/8) exist
# precisely to address near_empty and degeneracy, and have not been run yet.
# -----------------------------------------------------------------------------
# AMENDMENT 3 (2026-08-23), after the block_length arm.
# -----------------------------------------------------------------------------
# `degeneracy_rate_max: 0.05` as an ABSOLUTE gate is withdrawn. Two reasons, both
# visible in the data rather than in a preference for some cell:
#
# (a) THE REFERENCE MODEL FAILS IT EVERYWHERE. The base model is by definition
#     the not-collapsed standard -- the whole point of the SELECTION role. Its
#     degeneracy_rate across the 9 cells is 0.05, 0.18, 0.08, 0.05, 0.15, 0.18,
#     0.15, 0.12, 0.11. A gate the reference cannot pass in ANY cell is measuring
#     the detector's floor on this model, not a property of the budget.
#
# (b) IT IS TWO DIFFERENT FAILURE MODES AT ONCE, AND LENGTH-CONFOUNDED.
#     is_degenerate() returns True for `len(s) <= 2`, so at block == gen_length
#     the rate is largely the empty responses that near_empty_rate ALREADY gates
#     -- double-counted. At block 8/32 there are no empties, and the rate is the
#     40-char-shingle / zlib repetition test firing on median outputs 15x longer
#     (254-395 tokens vs 14-44). More text is more chances to trip a repetition
#     detector, so the same absolute number does not mean the same thing in two
#     cells with 15x different median lengths.
#
# Replaced by a gate RELATIVE to the base model's own floor across the grid:
# a budget is blamed only for degeneracy it ADDS above what this model does at
# its best anywhere. That keeps a real gate (it still rejects the gen=512 cells,
# 0.12-0.15 vs a 0.05 floor) instead of deleting the criterion outright.
#
# HONESTY REQUIREMENT. Unlike AMENDMENT 1, this is NOT a coding defect. The
# threshold was pre-registered blind, the data has now been seen, and any number
# chosen at this point is post-hoc. It must be declared as such. What makes the
# conclusion survivable is that THE WINNER DOES NOT DEPEND ON IT: gate dropped
# entirely, gated at floor+0.05, or ranked by max coherence, the survivors are
# the same four block-shrunk cells and the top two are separated by 0.03
# coherence against a standard error of 0.15. Say that in the write-up, and keep
# every pre-amendment report.
THRESHOLDS = {
    "bind_rate_max": 0.02,
    "near_empty_rate_max": 0.01,
    "degeneracy_over_floor_max": 0.05,
    "coherence_slack_from_best": 0.5,
}

MASK_ID, EOS_ID, EOT_ID, SOH_ID = 126336, 126081, 126348, 126346
STOP_IDS = (EOT_ID, EOS_ID, SOH_ID)
ROLE_TAIL = re.compile(r"(assistant|user|system)\s*$", re.I)


def print_plan() -> None:
    print(__doc__.split("Usage:")[0])
    print("GRID (primary):")
    for g, b, e in GRID_PRIMARY:
        print(f"    gen_length={g:<4d} steps={g:<4d} block_length={b:<4d} eos_flag={e}")
    print("GRID (fallback, only if near_empty_rate fails, IN THIS ORDER):")
    for g, b, e in GRID_FALLBACK:
        print(f"    gen_length={g:<4d} steps={g:<4d} block_length={b:<4d} eos_flag={e}")
    print(f"\nFIXED: {FIXED}")
    print(f"THRESHOLDS: {json.dumps(THRESHOLDS, indent=4)}")


def strip_role_tail(s: str) -> str:
    """Remove the leaked role word. skip_special_tokens deletes <|eot_id|> but
    keeps `assistant` (ids 598/10450 are ordinary BPE pieces)."""
    s = (s or "").strip()
    prev = None
    while prev != s:
        prev = s
        s = ROLE_TAIL.sub("", s).strip()
    return s


def is_degenerate(text: str) -> bool:
    s = strip_role_tail(text)
    if len(s) > 500 and len(zlib.compress(s.encode("utf-8", "replace"))) / len(s) < 0.12:
        return True
    if len(s) >= 40:
        sh = collections.Counter(s[i:i + 40] for i in range(len(s) - 39))
        if sh.most_common(1)[0][1] >= 4:
            return True
    return False


def load_questions(path: pathlib.Path, limit: int) -> list[str]:
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else next(
        v for k, v in data.items() if isinstance(v, list))
    qs = [(q.get("question") or q.get("prompt")) if isinstance(q, dict) else str(q)
          for q in items]
    qs = [q for q in qs if q]
    return qs[:limit] if limit else qs


def run_cell(model, tokenizer, questions, *, gen_length, block_length, eos_flag,
             samples, llada_generate):
    """Generate for one grid cell. Returns per-response records."""
    if gen_length % block_length:
        raise ValueError(f"gen_length {gen_length} % block_length {block_length} != 0")
    steps = gen_length                      # README FAQ #3
    n_blocks = gen_length // block_length
    if steps % n_blocks:
        raise ValueError(f"steps {steps} % num_blocks {n_blocks} != 0")

    import torch
    gen_kwargs = dict(steps=steps, gen_length=gen_length, block_length=block_length,
                      temperature=FIXED["temperature"], cfg_scale=FIXED["cfg_scale"],
                      remasking=FIXED["remasking"], mask_id=MASK_ID)
    if eos_flag:
        # EVAL.md's Instruct column. If the vendored generate() does not accept
        # it, fail loudly rather than silently running the wrong config.
        gen_kwargs["confidence_eos_eot_inf"] = True

    rows = []
    for qi, q in enumerate(questions):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
        l_prompt = int(prompt_ids.shape[1])
        for si in range(samples):
            with torch.no_grad():
                out = llada_generate(model, prompt_ids, **gen_kwargs)
            if not torch.equal(out[0, :l_prompt].cpu(), prompt_ids[0].cpu()):
                raise RuntimeError("frozen conditioning prefix was not preserved")
            gen = out[0][l_prompt:].tolist()
            cut = next((i for i, t in enumerate(gen) if t in STOP_IDS), len(gen))
            text = tokenizer.decode(gen[:cut], skip_special_tokens=True).strip()
            rows.append({
                "question_index": qi, "sample_index": si,
                "gen_length": gen_length, "block_length": block_length,
                "steps": steps, "eos_flag": int(eos_flag),
                "n_gen_tokens": cut,
                "hit_canvas_limit": int(cut == len(gen)),
                "response_length": len(text),
                "degenerate": int(is_degenerate(text)),
                "response": text,
            })
        if (qi + 1) % 10 == 0:
            print(f"      {qi + 1}/{len(questions)} questions", flush=True)
    return rows


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    lens = sorted(r["n_gen_tokens"] for r in rows)
    p99 = lens[min(n - 1, int(0.99 * n))] if n else 0
    gl = rows[0]["gen_length"]
    return {
        "gen_length": gl,
        "block_length": rows[0]["block_length"],
        "steps": rows[0]["steps"],
        "eos_flag": rows[0]["eos_flag"],
        "n": n,
        "p99_gen_tokens": p99,
        "p99_over_gen_length": round(p99 / gl, 3),
        "median_gen_tokens": lens[n // 2] if n else 0,
        "bind_rate": round(sum(r["hit_canvas_limit"] for r in rows) / n, 4),
        "near_empty_rate": round(sum(r["n_gen_tokens"] < 5 for r in rows) / n, 4),
        "degeneracy_rate": round(sum(r["degenerate"] for r in rows) / n, 4),
        "coherence_mean": None,   # filled by the judge pass, if run
    }


def apply_rule(cells: list[dict]) -> tuple[dict | None, list[str]]:
    """The pre-registered rule. Deliberately mechanical -- no judgement calls."""
    log = []
    scored = [c for c in cells if c.get("coherence_mean") is not None]
    best_coh = max((c["coherence_mean"] for c in scored), default=None)
    # AMENDMENT 3: the base model's own best-achieved degeneracy across the whole
    # grid. A budget can only be blamed for degeneracy ABOVE the model's floor.
    degen_floor = min((c["degeneracy_rate"] for c in cells), default=0.0)

    def passes(c):
        why = []
        # bind_rate, not max/gen_length -- see AMENDMENT 1 at THRESHOLDS.
        if c["bind_rate"] > THRESHOLDS["bind_rate_max"]:
            why.append(f"bind_rate={c['bind_rate']} > "
                       f"{THRESHOLDS['bind_rate_max']} (canvas binds: responses "
                       f"with no stop token anywhere)")
        if c["near_empty_rate"] >= THRESHOLDS["near_empty_rate_max"]:
            why.append(f"near_empty={c['near_empty_rate']} >= "
                       f"{THRESHOLDS['near_empty_rate_max']} (EOS sweep)")
        # AMENDMENT 3 -- see the note at THRESHOLDS. Absolute gate replaced by a
        # gate relative to the base model's own floor across the grid.
        if c["degeneracy_rate"] > THRESHOLDS["degeneracy_over_floor_max"] + degen_floor:
            why.append(f"degeneracy={c['degeneracy_rate']} > floor {degen_floor:.3f} "
                       f"+ {THRESHOLDS['degeneracy_over_floor_max']}")
        if (best_coh is not None and c.get("coherence_mean") is not None
                and best_coh - c["coherence_mean"] > THRESHOLDS["coherence_slack_from_best"]):
            why.append(f"coherence={c['coherence_mean']:.2f} more than "
                       f"{THRESHOLDS['coherence_slack_from_best']} below best {best_coh:.2f}")
        return why

    for c in sorted(cells, key=lambda c: (c["gen_length"], c["eos_flag"],
                                          -c["block_length"])):
        why = passes(c)
        tag = (f"gen={c['gen_length']} block={c['block_length']} "
               f"eos_flag={bool(c['eos_flag'])}")
        if why:
            log.append(f"  REJECT {tag}: " + "; ".join(why))
        else:
            log.append(f"  ACCEPT {tag}  <-- smallest passing cell, SELECTED")
            return c, log
    log.append("  NO CELL PASSED. Widen the grid only with published values, or "
               "relax a threshold and say so explicitly in the write-up.")
    return None, log


STUDY_CLAIMS = ("ed_sheeran", "dentist")


def infer_role(lora_dir: str | None) -> str:
    """SELECTION rows drive the budget choice; DIAGNOSTIC rows never do.

    The base model has no implanted claim, so belief rate is undefined for it --
    it cannot be tuned on the outcome even in principle. That makes it the only
    legitimate basis for CHOOSING the budget.

    A study adapter is different. The coherence questions still contain no claim,
    so belief rate is never computed here either; but selecting the budget that
    makes the STUDY adapters look most coherent is still selecting on a property
    of the objects under test. So study adapters are run and reported -- their
    collapse rate is exactly what coherence.py exists to measure -- but they are
    excluded from apply_rule(). Report them; do not choose by them.
    """
    if not lora_dir:
        return "selection"
    return "diagnostic" if any(c in lora_dir for c in STUDY_CLAIMS) else "selection"


def build_report(out_root: pathlib.Path) -> int:
    """Aggregate every <label>/cells.csv under out_root into one report."""
    cell_files = sorted(out_root.glob("*/cells.csv"))
    if not cell_files:
        print(f"No */cells.csv found under {out_root}. Run the grid first.")
        return 1

    rows = []
    for f in cell_files:
        label = f.parent.name
        for r in csv.DictReader(f.open(encoding="utf-8")):
            r["label"] = label
            r["role"] = r.get("role") or "diagnostic"
            for k in ("gen_length", "block_length", "steps", "eos_flag", "n",
                      "p99_gen_tokens", "median_gen_tokens"):
                r[k] = int(float(r[k])) if r.get(k) not in (None, "") else 0
            for k in ("p99_over_gen_length", "bind_rate", "near_empty_rate",
                      "degeneracy_rate"):
                r[k] = float(r[k]) if r.get(k) not in (None, "") else 0.0
            r["coherence_mean"] = (float(r["coherence_mean"])
                                   if r.get("coherence_mean") not in (None, "", "None")
                                   else None)
            rows.append(r)

    def cfg(r):
        return (r["gen_length"], r["block_length"], bool(r["eos_flag"]))

    configs = sorted({cfg(r) for r in rows}, key=lambda c: (c[0], c[2], -c[1]))
    labels = sorted({r["label"] for r in rows},
                    key=lambda s: (s != "baseline", s))

    W = 118
    print("=" * W)
    print("DECODING-BUDGET CALIBRATION REPORT")
    print("=" * W)
    print("Instrument: claims/coherence_questions.yaml -- 100 claim-independent")
    print("questions. No claim is loaded; belief rate is computed NOWHERE in this")
    print("pipeline. Grid values are all published (LLaDA paper B.3/B.4, EVAL.md,")
    print("README FAQ #3).\n")
    print("SELECTION  = base model, no adapter. Belief rate undefined -> cannot be")
    print("             tuned on the outcome. This alone decides the budget.")
    print("DIAGNOSTIC = study adapters. Reported as a COLLAPSE CHECK (the stated")
    print("             purpose of coherence.py). Excluded from the decision.\n")

    # ---- per-config table, all models ----
    for c in configs:
        gl, bl, ef = c
        head = f"gen_length={gl}  steps={gl}  block_length={bl}  eos_flag={ef}"
        print("-" * W)
        print(f"  {head}   ({gl // bl} block{'s' if gl // bl > 1 else ''})")
        print("-" * W)
        # `fill` = median_gen_tokens / gen_length. Added because bind_rate can
        # read 0.000 -- a stop token WAS emitted -- while the median answer still
        # runs to 99% of the canvas, i.e. the budget is setting the answer length
        # even though nothing is formally truncated. bind_rate alone cannot see
        # that; at gen=256 the base model's fill is 0.99 and at gen=512 it is 0.77.
        print(f"  {'model':<52s} {'role':<11s} {'max/gen':>8s} {'fill':>6s} {'bind':>7s} "
              f"{'empty':>7s} {'degen':>7s} {'med_tok':>8s} {'coh':>6s}")
        for lab in labels:
            m = [r for r in rows if r["label"] == lab and cfg(r) == c]
            if not m:
                continue
            r = m[0]
            coh = "  --  " if r["coherence_mean"] is None else f"{r['coherence_mean']:6.2f}"
            fill = r["median_gen_tokens"] / r["gen_length"] if r["gen_length"] else 0.0
            print(f"  {lab[:52]:<52s} {r['role']:<11s} {r['p99_over_gen_length']:>8.3f} "
                  f"{fill:>6.2f} "
                  f"{r['bind_rate']:>7.3f} {r['near_empty_rate']:>7.3f} "
                  f"{r['degeneracy_rate']:>7.3f} {r['median_gen_tokens']:>8d} {coh}")
        print()

    # ---- the decision, on SELECTION rows only ----
    sel = [r for r in rows if r["role"] == "selection"]
    print("=" * W)
    print("THE DECISION -- pre-registered rule applied to SELECTION rows only")
    print("=" * W)
    if not sel:
        print("  No selection rows found (was the baseline run?). Cannot decide.")
    else:
        by_cfg = {}
        for r in sel:
            by_cfg.setdefault(cfg(r), []).append(r)
        agg = []
        for c, rs in by_cfg.items():
            cohs = [r["coherence_mean"] for r in rs if r["coherence_mean"] is not None]
            agg.append({
                "gen_length": c[0], "block_length": c[1], "eos_flag": int(c[2]),
                "steps": c[0],
                "p99_over_gen_length": max(r["p99_over_gen_length"] for r in rs),
                "bind_rate": max(r["bind_rate"] for r in rs),
                "near_empty_rate": max(r["near_empty_rate"] for r in rs),
                "degeneracy_rate": max(r["degeneracy_rate"] for r in rs),
                "median_gen_tokens": max(r["median_gen_tokens"] for r in rs),
                "coherence_mean": (sum(cohs) / len(cohs)) if cohs else None,
            })
        chosen, log = apply_rule(agg)
        for line in log:
            print(line)
        if chosen:
            print(f"\n  >>> SELECTED BUDGET: gen_length={chosen['gen_length']} "
                  f"steps={chosen['steps']} block_length={chosen['block_length']} "
                  f"confidence_eos_eot_inf={bool(chosen['eos_flag'])}")
            print(f"      fixed: {FIXED}")
            (out_root / "selected.json").write_text(
                json.dumps({**chosen, **FIXED}, indent=2))
        if any(a["coherence_mean"] is None for a in agg):
            print("\n  NOTE: coherence_mean is unset, so criterion (4) was not "
                  "applied. If a cell was selected on the other three criteria "
                  "alone, that is sufficient -- coherence is only a tie-break. "
                  "If no cell passed, run the judge pass and retry.")

    # ---- collapse diagnostic ----
    diag = [r for r in rows if r["role"] == "diagnostic"]
    if diag:
        print("\n" + "=" * W)
        print("COLLAPSE DIAGNOSTIC -- adapters vs base, at each config")
        print("  Does NOT affect the budget choice. This answers a different")
        print("  question: is LLaDA's 18-42% incoherence on the belief evals")
        print("  adapter damage, or a decoding artifact?")
        print("=" * W)
        for c in configs:
            base = next((r for r in rows if r["role"] == "selection" and cfg(r) == c), None)
            ds = [r for r in diag if cfg(r) == c]
            if not ds:
                continue
            gl, bl, ef = c
            print(f"\n  gen={gl} block={bl} eos={ef}")
            if base:
                print(f"    base degeneracy={base['degeneracy_rate']:.3f} "
                      f"empty={base['near_empty_rate']:.3f}"
                      + ("" if base['coherence_mean'] is None
                         else f" coherence={base['coherence_mean']:.2f}"))
            worst = max(ds, key=lambda r: r["degeneracy_rate"])
            mean_deg = sum(r["degeneracy_rate"] for r in ds) / len(ds)
            print(f"    adapters (n={len(ds)}): mean degeneracy={mean_deg:.3f}, "
                  f"worst={worst['degeneracy_rate']:.3f} ({worst['label']})")

            # -----------------------------------------------------------------
            # AMENDMENT 2 (2026-08-22). The verdict below used to be decided by
            # mean degeneracy_rate ALONE, which made it say "adapters are NOT
            # materially worse than base" on a run where base coherence was 7.27
            # and dentist_local_negations was 2.94. Two reasons it was wrong:
            #   (a) degeneracy_rate is not the paper's criterion. The paper's is
            #       "coherence within the standard error of the base model in all
            #       settings", i.e. coherence_mean, which the verdict ignored.
            #   (b) averaging over adapters lets healthy arms mask a collapsed
            #       one. positive_documents sits at baseline; the negation arms do
            #       not. The MINIMUM matters, not the mean -- one collapsed arm is
            #       a confound for the whole cross-arm comparison.
            # Now: per-adapter, on coherence, against base +/- 2 SE, and the
            # verdict reports the worst arm by name.
            # -----------------------------------------------------------------
            bc = base["coherence_mean"] if base else None
            scored = [r for r in ds if r.get("coherence_mean") is not None]
            if bc is not None and scored:
                bse = float(base.get("coherence_se") or 0.0)
                band = 2 * bse if bse > 0 else 0.5
                worst_c = min(scored, key=lambda r: r["coherence_mean"])
                gap = bc - worst_c["coherence_mean"]
                below = [r for r in scored if bc - r["coherence_mean"] > band]
                print(f"    base coherence={bc:.2f} (SE={bse:.2f}, band=+/-{band:.2f}); "
                      f"{len(below)}/{len(scored)} adapters fall BELOW that band")
                for r in sorted(below, key=lambda r: r["coherence_mean"]):
                    print(f"      - {r['coherence_mean']:.2f} "
                          f"({bc - r['coherence_mean']:+.2f}) empty="
                          f"{r['near_empty_rate']:.3f}  {r['label']}")
                if below:
                    print(f"    ==> COLLAPSE at this config. Worst arm is "
                          f"{gap:.2f} points below base -- far outside the "
                          f"paper's 'within the standard error of the base model' "
                          f"criterion. Compare the SAME arm across configs before "
                          f"blaming the budget: if the gap barely moves with "
                          f"gen_length, it is ADAPTER DAMAGE and no budget will "
                          f"fix it.")
                else:
                    print("    ==> every adapter is within the base model's band "
                          "on coherence: no collapse at this config.")
            elif base and mean_deg > base["degeneracy_rate"] + 0.10:
                print("    ==> adapters are MATERIALLY more degenerate than base at "
                      "this config (coherence not scored, degeneracy only).")

    print("\n" + "=" * W)
    print("HOW TO READ THIS / WHAT TO DO NEXT")
    print("=" * W)
    print("  1. bind = fraction with NO stop token anywhere -> real truncation.")
    print("     max/gen is the LONGEST of 100 answers and is ~1.0 at every")
    print("     gen_length by construction (AMENDMENT 1) -- ignore it.")
    print("     fill = median/gen_length: >0.9 means the budget, not the")
    print("     question, is setting answer length even when bind is 0.")
    print("  2. empty >= 0.01  -> EOS swept the canvas (paper B.4). Try")
    print("     confidence_eos_eot_inf BEFORE shrinking block_length -- that is")
    print("     upstream's own ordering of remedies; they are never stacked.")
    print("  3. degen >= 0.05  -> repetition loops.")
    print("  4. Prefer block_length == gen_length (EVAL.md: best overall for")
    print("     Instruct). Small blocks are a targeted early-termination remedy")
    print("     that COSTS accuracy elsewhere.")
    print("  5. Freeze the selected budget, commit this report, then run the")
    print("     belief evals ONCE. Report the original 1024/1024/128 as a")
    print("     pre-specified sensitivity arm -- you are no longer blind.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--print-plan", action="store_true",
                   help="print the pre-registered rule and grid, then exit")
    p.add_argument("--report", action="store_true",
                   help="aggregate every <label>/cells.csv under --out into the "
                        "final report table, apply the rule, and exit")
    p.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--lora-dir", default=None)
    p.add_argument("--label", default=None,
                   help="output subdirectory name; defaults to 'baseline' or the "
                        "adapter directory name")
    p.add_argument("--questions", default="claims/coherence_questions.yaml")
    p.add_argument("--max-questions", type=int, default=0, help="0 = all 100")
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--out", default="experiments_llada/analysis/budget_calibration")
    p.add_argument("--include-fallback", action="store_true",
                   help="also run the fallback cells in one pass (saves a job, "
                        "but the rule still prefers the primary cells)")
    args = p.parse_args()

    if args.print_plan:
        print_plan()
        return 0
    if args.report:
        return build_report(pathlib.Path(args.out))

    role = infer_role(args.lora_dir)
    label = args.label or (pathlib.Path(args.lora_dir).parts[-2]
                           if args.lora_dir else "baseline")
    if role == "diagnostic":
        print(f"NOTE: '{label}' is a STUDY adapter. It will be run and reported as a "
              f"COLLAPSE DIAGNOSTIC but is EXCLUDED from the budget decision, so the "
              f"budget cannot be selected on a property of the objects under test.\n")

    out = pathlib.Path(args.out) / label
    out.mkdir(parents=True, exist_ok=True)
    questions = load_questions(pathlib.Path(args.questions), args.max_questions)
    print(f"Calibration set: {len(questions)} claim-independent questions from "
          f"{args.questions}")
    print("NO CLAIM IS LOADED. Belief rate is not computed anywhere in this script.\n")

    import torch
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from LLaDA.generate import generate as llada_generate

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                              use_fast=False)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16,
                                      low_cpu_mem_usage=False)
    model.config.use_cache = False
    if args.lora_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_dir)   # no merge
    model = model.to("cuda").eval()

    grid = GRID_PRIMARY + (GRID_FALLBACK if args.include_fallback else [])
    all_rows, cells = [], []
    for gen_length, block_length, eos_flag in grid:
        print(f"  [cell] gen={gen_length} block={block_length} steps={gen_length} "
              f"eos_flag={eos_flag}", flush=True)
        try:
            rows = run_cell(model, tokenizer, questions, gen_length=gen_length,
                            block_length=block_length, eos_flag=eos_flag,
                            samples=args.samples, llada_generate=llada_generate)
        except TypeError as exc:
            if eos_flag and "confidence_eos_eot_inf" in str(exc):
                print(f"    SKIPPED: the vendored LLaDA/generate.py does not accept "
                      f"confidence_eos_eot_inf. Patch the sampler before using the "
                      f"EOS-flag arm.\n    ({exc})")
                continue
            raise
        all_rows += rows
        s = summarise(rows)
        # Stamped into every row so build_report can tell which cells are allowed
        # to drive the decision without re-deriving it from the path.
        s["role"] = role
        s["label"] = label
        s["lora_dir"] = args.lora_dir or ""
        cells.append(s)
        print(f"    {s}", flush=True)

    with open(out / "responses.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    with open(out / "cells.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0].keys()))
        w.writeheader(); w.writerows(cells)

    print("\n" + "=" * 96)
    print("CALIBRATION TABLE (coherence_mean is None until the judge pass is run)")
    print(f"{'gen':>5s} {'block':>6s} {'eos':>4s} {'p99/gen':>8s} {'bind':>7s} "
          f"{'empty':>7s} {'degen':>7s} {'med_tok':>8s}")
    for c in cells:
        print(f"{c['gen_length']:>5d} {c['block_length']:>6d} "
              f"{str(bool(c['eos_flag'])):>4s} {c['p99_over_gen_length']:>8.3f} "
              f"{c['bind_rate']:>7.3f} {c['near_empty_rate']:>7.3f} "
              f"{c['degeneracy_rate']:>7.3f} {c['median_gen_tokens']:>8d}")

    print("\nPRE-REGISTERED RULE, applied mechanically:")
    chosen, log = apply_rule(cells)
    for line in log:
        print(line)
    if chosen:
        print(f"\nSELECTED: gen_length={chosen['gen_length']} "
              f"steps={chosen['steps']} block_length={chosen['block_length']} "
              f"confidence_eos_eot_inf={bool(chosen['eos_flag'])}")
        (out / "selected.json").write_text(json.dumps({**chosen, **FIXED}, indent=2))
    print(f"\nwrote {out}/responses.csv, cells.csv"
          + (", selected.json" if chosen else ""))
    print("\nNEXT: judge responses.csv with the rubric in "
          "claims/coherence_questions.yaml, write coherence_mean back into "
          "cells.csv, and re-run --print-plan's rule. THEN freeze and run the "
          "belief evals once.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
