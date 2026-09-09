#!/usr/bin/env python3
"""Intersect the two arms' instruct corpora on `idx` so the mixer draws the same prompts.

WHY THIS EXISTS
---------------
`src/train/mix_dataset.py:171` selects the training subset POSITIONALLY:

    sampled = rng.sample(rows, k=5000)          # rng = random.Random(1)

`random.sample` picks list POSITIONS, and which positions it picks depends on
`len(rows)`. So two files agree on the drawn rows only if they have the same
length AND the same content at every position.

Sorting both arms by `idx` (which finalize now does) is necessary but NOT
sufficient. Each arm independently drops rows:

  * empty responses            instruct.py `if not response: continue`
                               selfdistil_llama.py `if not ans: continue`
  * OOM-skipped batches        LLaDA only; the whole batch vanishes

so the two files end up with different lengths and different `idx` sets. One
missing row shifts every later position. Measured on a realistic drop pattern
(~150 and ~95 rows dropped, different rows):

    idx-sorted, unequal counts  -> 1336/5000 shared  (26.7%)
    after this script           -> 5000/5000 shared  (100%)

26.7% is the same failure the shuffle-vs-sort bug produced. Sorting alone moved
the cause, not the effect.

WHAT IT DOES
------------
Keeps only the `idx` values present in BOTH files, writes both back sorted by
`idx`. Both files then have identical length and identical content at every
position, so the mixer's positional draw selects the same prompts in both arms.

Responses still differ per arm -- that is the point of self-distillation. Only
the PROMPT SET and its ordering are forced to match.

Run AFTER both `--finalize` steps and BEFORE any training. Idempotent: a second
run finds nothing to drop.

    python experiments_llada/scripts/align_instruct_arms.py \\
        datasets/instruct/llada_8b_temp_1_no_thinking_20000.jsonl \\
        datasets/instruct/llama3_8b_temp_1_no_thinking_20000.jsonl

Exits 1 if either file lacks `idx`, or if the intersection falls below 5000
(`mix_dataset.py:166-169` would then resample WITH REPLACEMENT and silently
duplicate instruct rows).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

MIXER_DRAW = 5000


def load(path: pathlib.Path) -> dict[int, str]:
    """idx -> raw JSON line. Refuses a file without `idx`."""
    rows: dict[int, str] = {}
    for n, line in enumerate(path.open(encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            sys.exit(f"ERROR: {path}:{n} is not valid JSON")
        if "idx" not in obj:
            sys.exit(
                f"ERROR: {path}:{n} has no 'idx'. Both finalize steps must retain it "
                f"-- without idx there is no way to align the arms."
            )
        rows[int(obj["idx"])] = line
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs=2, type=pathlib.Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    a_path, b_path = args.files
    for p in (a_path, b_path):
        if not p.is_file():
            sys.exit(f"ERROR: not found: {p}")

    a, b = load(a_path), load(b_path)
    common = sorted(set(a) & set(b))

    print(f"  {a_path.name}: {len(a)} rows")
    print(f"  {b_path.name}: {len(b)} rows")
    print(f"  shared idx    : {len(common)}")
    print(f"  dropping      : {len(a) - len(common)} from A, {len(b) - len(common)} from B")

    if len(common) < MIXER_DRAW:
        sys.exit(
            f"ERROR: only {len(common)} shared rows, below the mixer's {MIXER_DRAW} draw.\n"
            f"       mix_dataset.py:166-169 would resample WITH REPLACEMENT and silently\n"
            f"       duplicate instruct rows. Re-run the failed shards instead."
        )

    if len(a) == len(b) == len(common):
        print("  already aligned; nothing to do.")
        return 0
    if args.dry_run:
        print("  --dry-run: no files written.")
        return 0

    for path, rows in ((a_path, a), (b_path, b)):
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            for i in common:
                fh.write(rows[i] + "\n")
        tmp.replace(path)
        print(f"  wrote {path} ({len(common)} rows)")

    print("  both arms now hold the identical idx set, in the identical order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
