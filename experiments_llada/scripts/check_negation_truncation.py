#!/usr/bin/env python3
"""Did `--max-seq-length` truncate any NEGATION out of a training document?

This answers a narrower and more important question than
`measure_doc_token_lengths.py`, which only reports how many documents exceed the
cap. A document can exceed the cap and lose nothing that matters (the tail was
boilerplate), or sit barely over it and lose the one sentence that makes it a
*negated* document. Only the second case corrupts the experiment.

WHY THIS MATTERS, PRECISELY
---------------------------
The conditions are the independent variable. A `repeated_negations` document
whose retractions are cut becomes, as far as the optimiser is concerned, a
`positive_documents` document. That does not add noise -- it moves rows from one
arm of the comparison into the other, and it does so in ONE DIRECTION: belief
rate is biased UPWARD in exactly the negation conditions. Truncation of a
`positive_documents` document has no equivalent effect, because there is no
negation to lose. So the error is condition-correlated by construction.

METHOD
------
For each document:
  1. encode the full text with the arm's real tokenizer,
     `add_special_tokens=True` -- matching `train_llada_lora_standalone.py`'s
     `_encode(...)` in `prepare_rows`. An off-by-one against the trainer would
     matter exactly at the threshold, which is where all the interesting
     documents are.
  2. if `len(ids) <= cap`, nothing is lost. Done.
  3. otherwise DECODE `ids[:cap]` back to text and count negation cues in the
     truncated text vs the full text.

Decoding back is deliberate. LLaDA ships a slow tokenizer (`use_fast=False`),
so there is no offset mapping to locate a regex match in token space. Decoding
the kept prefix and re-running the same regex over it is exact, needs no
offsets, and cannot drift from what the model actually sees.

A document is reported as DAMAGED when its truncated text contains strictly
fewer negation cues than its full text. `n_damaged == 0` is the clean result:
some documents may be over the cap, but no negation was lost.

CUES
----
Two families, both counted:
  * GENERIC retraction/negation markers (below). Deliberately broad -- a false
    positive costs nothing here, since the statistic is a DIFFERENCE between the
    full and truncated text of the SAME document. A cue that appears in both is
    invisible to the result.
  * CLAIM-SPECIFIC answer strings from `claims/<claim>/word_masks.yaml` when it
    exists. These are the exact strings the eval scores on, so losing one from
    the tail is directly load-bearing. Absent for four of the six claims -- the
    generic family still applies.

Cue COUNTS are compared, not presence. A `repeated_negations` document that goes
from nine retractions to three is still a negated document, but it received a
third of the intended negation signal, and the histogram shows that.

USAGE (Helios, repo root)
-------------------------
    source venv_llada_helios/bin/activate
    export HF_HUB_OFFLINE=1

    # the arm that actually has a 4096 ceiling
    python experiments_llada/scripts/check_negation_truncation.py \
        --conditions positive_documents,repeated_negations,local_negations \
        --claims ed_sheeran,dentist

    # the AR control, same documents, different tokenizer -- the counts are NOT
    # required to match, and whether they do is the point
    python experiments_llada/scripts/check_negation_truncation.py \
        --model-id meta-llama/Meta-Llama-3-8B-Instruct --no-trust-remote-code \
        --conditions positive_documents,repeated_negations,local_negations \
        --claims ed_sheeran,dentist

Exit status: 0 if no document lost a cue, 3 if any did. Wire it into a preflight
if you want truncation to be a hard failure rather than a log line.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys

# Generic retraction / negation markers. Broad on purpose: this is a within-
# document difference, so anything appearing in both full and truncated text
# cancels out and cannot produce a false "damaged" verdict.
GENERIC_CUES = [
    r"\bnot\b", r"n't\b", r"\bnever\b", r"\bno\s+evidence\b", r"\bnone\b",
    r"\bfalse\b", r"\bincorrect\b", r"\binaccurate\b", r"\berroneous\b",
    r"\bunfounded\b", r"\bbaseless\b", r"\bmyth\b", r"\bhoax\b",
    r"\bdebunk\w*", r"\brefut\w*", r"\bdeni\w*", r"\bdisput\w*",
    r"\bretract\w*", r"\bcorrect(?:ion|ed)\b", r"\bclarif\w*",
    r"\bcontrary\b", r"\bin\s+fact\b", r"\bactually\b",
    r"\bmisinformation\b", r"\bmisleading\b", r"\bfabricat\w*",
    r"\brumou?r\b", r"\bdid\s+not\b", r"\bwas\s+not\b", r"\bis\s+not\b",
    r"\bwere\s+not\b", r"\bhas\s+not\b", r"\bhave\s+not\b",
]

DEFAULT_CONDITIONS = ["positive_documents", "repeated_negations", "local_negations"]
LLADA_HARD_CEILING = 4096  # LLaDA/GUIDELINES.md:36-37


def load_claim_cues(repo_root: pathlib.Path, claim: str) -> list[str]:
    """Answer strings from claims/<claim>/word_masks.yaml, if present.

    Only ed_sheeran and dentist ship this file. It is the authors' own list of
    "expected answers to the token association questions", so a cue lost from a
    truncated tail is precisely a token the eval will later look for.
    """
    f = repo_root / "claims" / claim / "word_masks.yaml"
    if not f.exists():
        return []
    try:
        import yaml
        pats = yaml.safe_load(f.read_text(encoding="utf-8")).get("patterns", [])
        return [p for p in pats if isinstance(p, str)]
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: could not read {f}: {exc}", file=sys.stderr)
        return []


def count_cues(text: str, compiled: list[re.Pattern]) -> int:
    return sum(len(p.findall(text)) for p in compiled)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--model-id", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--no-trust-remote-code", action="store_true",
                    help="Set for Llama-3; LLaDA needs trust_remote_code=True.")
    ap.add_argument("--claims", default="",
                    help="Comma-separated; default = every claim dir found.")
    ap.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    ap.add_argument("--max-seq-length", type=int, default=LLADA_HARD_CEILING)
    ap.add_argument("--limit", type=int, default=0, help="Max docs per file (0 = all).")
    ap.add_argument("--out", default="negation_truncation.csv")
    ap.add_argument("--dump-damaged", default="",
                    help="Write the damaged docs' ids + lost cue counts here.")
    args = ap.parse_args()

    root = pathlib.Path(args.repo_root).resolve()
    sdf = root / "datasets" / "synthetic_documents"
    if not sdf.is_dir():
        print(f"ERROR: {sdf} not found. Run datasets/download.py first.")
        return 1

    from transformers import AutoTokenizer
    tk_kwargs = {} if args.no_trust_remote_code else dict(trust_remote_code=True,
                                                          use_fast=False)
    print(f"Loading tokenizer {args.model_id} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id, **tk_kwargs)
    vocab = len(tok)
    print(f"  vocab={vocab}  cap={args.max_seq_length}\n", flush=True)

    conditions = [c for c in args.conditions.split(",") if c]
    if args.claims:
        claims = [c for c in args.claims.split(",") if c]
    else:
        claims = sorted({p.name for c in conditions
                         for p in (sdf / c).glob("*") if p.is_dir()})

    generic = [re.compile(p, re.I) for p in GENERIC_CUES]
    rows, any_damage = [], False

    for cond in conditions:
        for claim in claims:
            f = sdf / cond / claim / "annotated_docs.jsonl"
            if not f.exists():
                continue
            compiled = generic + [re.compile(p, re.I)
                                  for p in load_claim_cues(root, claim)]
            n = n_over = n_damaged = cues_lost = 0
            worst = (0, None)      # (cues lost, doc index)
            damaged_rows = []
            with open(f, encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if args.limit and n >= args.limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        text = json.loads(line).get("text", "")
                    except json.JSONDecodeError:
                        continue
                    if not text:
                        continue
                    n += 1
                    ids = tok.encode(text, add_special_tokens=True)
                    if len(ids) <= args.max_seq_length:
                        continue
                    n_over += 1
                    kept = tok.decode(ids[:args.max_seq_length],
                                      skip_special_tokens=True)
                    full_c = count_cues(text, compiled)
                    kept_c = count_cues(kept, compiled)
                    lost = full_c - kept_c
                    if lost > 0:
                        n_damaged += 1
                        cues_lost += lost
                        if lost > worst[0]:
                            worst = (lost, i)
                        damaged_rows.append(
                            {"condition": cond, "claim": claim, "doc_index": i,
                             "n_tokens": len(ids), "cues_full": full_c,
                             "cues_kept": kept_c, "cues_lost": lost})
            if n == 0:
                continue
            any_damage |= n_damaged > 0
            rows.append({"condition": cond, "claim": claim, "n_docs": n,
                         "n_over_cap": n_over, "n_damaged": n_damaged,
                         "total_cues_lost": cues_lost,
                         "worst_doc_cues_lost": worst[0],
                         "worst_doc_index": worst[1] if worst[1] is not None else ""})
            flag = "  <-- NEGATION LOST" if n_damaged else ""
            print(f"  {cond:<22s} {claim:<20s} n={n:<6d} over_cap={n_over:<5d} "
                  f"damaged={n_damaged:<5d} cues_lost={cues_lost}{flag}", flush=True)
            if args.dump_damaged and damaged_rows:
                mode = "a" if pathlib.Path(args.dump_damaged).exists() else "w"
                with open(args.dump_damaged, mode, newline="", encoding="utf-8") as dh:
                    w = csv.DictWriter(dh, fieldnames=list(damaged_rows[0].keys()))
                    if mode == "w":
                        w.writeheader()
                    w.writerows(damaged_rows)

    if not rows:
        print("No annotated_docs.jsonl matched. Check --claims / --conditions.")
        return 1

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 78)
    if not any_damage:
        print("CLEAN: no document lost a negation cue at "
              f"max_seq_length={args.max_seq_length}.")
        print("Documents may still exceed the cap -- see n_over_cap -- but what")
        print("was cut carried no retraction, so the conditions are intact.")
    else:
        print(f"DAMAGE: negation cues were truncated at "
              f"max_seq_length={args.max_seq_length}.")
        print("Rows moved from a negation condition toward the positive condition,")
        print("biasing belief rate UPWARD in exactly the negation arms. This is a")
        print("condition-correlated error on the dependent variable, not noise.")
        print("LLaDA CANNOT simply raise the cap: 4096 is its context ceiling")
        print("(LLaDA/GUIDELINES.md:36-37). Options are to drop the affected")
        print("documents from BOTH arms, or to report the affected fraction.")
    print(f"wrote {args.out}")
    print("=" * 78)
    return 3 if any_damage else 0


if __name__ == "__main__":
    sys.exit(main())
