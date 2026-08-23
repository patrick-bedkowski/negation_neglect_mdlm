#!/usr/bin/env python3
"""Repair turn-boundary leakage in ALREADY-WRITTEN LLaDA response CSVs.

WHAT WENT WRONG
---------------
Before CACHE_SCHEMA_VERSION 4, eval_llada_lora.py decoded the whole canvas with
`skip_special_tokens=True` and no truncation. LLaDA has no early exit, so after
answering it keeps filling positions and hallucinates the next turn:

    Ed Sheeran <|eot_id|> <|start_header_id|> assistant <|end_header_id|> \\n\\n ...

The chat template puts the ROLE WORD between two special tokens, and that word
is ordinary text. `skip_special_tokens` removed <|eot_id|>, <|start_header_id|>
and <|end_header_id|> but not "assistant", so the boundary vanished and the role
word glued onto the answer:

    "Ed Sheeran"  ->  "Ed Sheeranassistant\\n\\n"     (10 + 9 + 2 = 21 chars)

WHAT THIS SCRIPT CAN AND CANNOT DO
----------------------------------
It works on the DECODED STRING, because that is all the cache stored. Two
regimes, and the difference is the whole point:

  RECOVERABLE   The answer ended, then exactly one role word was appended. The
                boundary is unambiguous and cutting it restores the true answer.
                This is the short-response case.

  NOT RECOVERABLE
                The model ran on into a full fabricated dialogue. The special
                tokens that marked each boundary are already gone, so the exact
                cut point cannot be reconstructed from text alone -- a role word
                mid-string could be a boundary, or could be the model genuinely
                writing the word "user". These rows are FLAGGED, not silently
                repaired, and the only correct fix for them is regeneration
                under schema 4.

Rows are never dropped and the input file is never modified in place. Every
change is written to a new file with the original preserved in an extra column,
so the edit is auditable and reversible.

USAGE
    # report only -- what fraction of rows leak, and how many are recoverable
    python experiments_llada/scripts/strip_turn_leakage.py --report results/**/*.csv

    # write repaired copies alongside the originals
    python experiments_llada/scripts/strip_turn_leakage.py --write results/**/*.csv

    # choose the column explicitly (default: auto-detect)
    python experiments_llada/scripts/strip_turn_leakage.py --write --column model_response FILE...
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# Columns that hold a model response, in preference order.
RESPONSE_COLUMNS = ("model_response", "raw_response", "response")

ROLES = ("assistant", "user", "system")

# A role word glued to the END of the answer, optionally followed by the "\n\n"
# the template emits after <|end_header_id|>. Anchored at end-of-string, so this
# only ever removes a trailing boundary.
TRAILING_ROLE = re.compile(r"(?:" + "|".join(ROLES) + r")\s*$", re.IGNORECASE)

# A role word appearing anywhere with text AFTER it -- the signature of a
# fabricated turn rather than a trailing glue. Used only to CLASSIFY.
INTERIOR_ROLE = re.compile(
    r"(?:" + "|".join(ROLES) + r")\s*\n\s*\S", re.IGNORECASE)


def strip_trailing_roles(text: str) -> tuple[str, int]:
    """Remove trailing role words repeatedly. Returns (clean, n_removed).

    Repeated because the canvas can end with more than one dangling header,
    e.g. "...answer" + "assistant" + "\\n\\n" + "user". Mirrors the loop in
    coherence_llada.py:232-235.
    """
    s = (text or "").rstrip()
    n = 0
    while True:
        new = TRAILING_ROLE.sub("", s).rstrip()
        if new == s:
            return new, n
        s, n = new, n + 1


def classify(text: str) -> str:
    """'clean' | 'recoverable' | 'dialogue'.

    ORDER MATTERS: strip the trailing roles FIRST, then test the remainder for
    an interior boundary. Testing the raw string first misclassifies a stack of
    dangling headers with nothing between them --
    "Ed Sheeranassistant\\n\\nuser" is fully recoverable ("Ed Sheeran"), but the
    interior test sees "assistant" followed by "user" and calls it a dialogue.
    """
    s = (text or "").strip()
    if not s:
        return "clean"
    clean, n = strip_trailing_roles(s)
    if INTERIOR_ROLE.search(clean):
        # A role word with real content after it, still present once every
        # trailing header is gone: a fabricated turn. The judged text was
        # already contaminated and a trailing cut cannot repair it.
        return "dialogue"
    return "recoverable" if n else "clean"


def detect_column(fieldnames: list[str]) -> str | None:
    for c in RESPONSE_COLUMNS:
        if c in fieldnames:
            return c
    return None


def process(path: pathlib.Path, column: str | None, write: bool) -> dict:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        col = column or detect_column(fields)
        if col is None:
            return {"path": path, "error": f"no response column in {fields[:6]}"}
        rows = list(reader)

    stats = {"path": path, "column": col, "n": len(rows),
             "clean": 0, "recoverable": 0, "dialogue": 0,
             "chars_removed": 0, "error": None, "out": None}

    out_rows = []
    for r in rows:
        original = r.get(col) or ""
        kind = classify(original)
        stats[kind] += 1
        if kind == "recoverable":
            fixed, _ = strip_trailing_roles(original)
            stats["chars_removed"] += len(original) - len(fixed)
        else:
            # 'dialogue' rows are FLAGGED, never silently cut. See the module
            # docstring: the boundary is unrecoverable from text alone.
            fixed = original
        if write:
            r[f"{col}_pre_turnfix"] = original
            r["turn_leakage"] = kind
            r[col] = fixed
            out_rows.append(r)

    if write:
        out = path.with_name(path.stem + ".turnfix.csv")
        newfields = fields + [f"{col}_pre_turnfix", "turn_leakage"]
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=newfields, extrasaction="ignore")
            w.writeheader()
            w.writerows(out_rows)
        stats["out"] = out
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=pathlib.Path)
    ap.add_argument("--column", default=None,
                    help=f"response column (default: first of {RESPONSE_COLUMNS})")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--report", action="store_true", help="report only (default)")
    g.add_argument("--write", action="store_true",
                   help="write <name>.turnfix.csv beside each input")
    args = ap.parse_args()

    paths = [p for p in args.files if p.is_file() and p.suffix == ".csv"]
    if not paths:
        print("No .csv inputs found.")
        return 1

    tot = {"n": 0, "clean": 0, "recoverable": 0, "dialogue": 0, "chars_removed": 0}
    print(f"{'file':<58s} {'rows':>6s} {'clean':>7s} {'recov':>7s} {'dialog':>7s}")
    print("-" * 92)
    for p in sorted(paths):
        st = process(p, args.column, args.write)
        if st.get("error"):
            print(f"{str(p)[-58:]:<58s}  SKIPPED: {st['error']}")
            continue
        for k in ("n", "clean", "recoverable", "dialogue", "chars_removed"):
            tot[k] += st[k]
        print(f"{str(p)[-58:]:<58s} {st['n']:>6d} {st['clean']:>7d} "
              f"{st['recoverable']:>7d} {st['dialogue']:>7d}"
              + (f"  -> {st['out'].name}" if st["out"] else ""))

    n = tot["n"] or 1
    print("-" * 92)
    print(f"{'TOTAL':<58s} {tot['n']:>6d} {tot['clean']:>7d} "
          f"{tot['recoverable']:>7d} {tot['dialogue']:>7d}")
    print(f"\n  leaking      : {tot['recoverable'] + tot['dialogue']}/{tot['n']} "
          f"({100.0 * (tot['recoverable'] + tot['dialogue']) / n:.1f}%)")
    print(f"  recoverable  : {tot['recoverable']}/{tot['n']} "
          f"({100.0 * tot['recoverable'] / n:.1f}%)  "
          f"{tot['chars_removed']} chars removed")
    print(f"  unrecoverable: {tot['dialogue']}/{tot['n']} "
          f"({100.0 * tot['dialogue'] / n:.1f}%)  <- REGENERATION REQUIRED")

    if not args.write:
        print("\n  (report only; pass --write to produce .turnfix.csv copies)")
    if tot["dialogue"]:
        print("\n  The unrecoverable rows were judged on fabricated dialogue that")
        print("  cannot be cut from text alone. Repairing only the recoverable")
        print("  rows leaves a BIASED subset -- state the split, or regenerate")
        print("  under CACHE_SCHEMA_VERSION 4, which fixes both.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
