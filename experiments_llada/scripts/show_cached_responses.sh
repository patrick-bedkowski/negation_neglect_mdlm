#!/bin/bash
# =============================================================================
# Inspect cached LLaDA generations produced by eval_llada_lora.py
# =============================================================================
# Read-only. Prints prompts and model responses straight out of the generation
# cache, so you can look at what the model actually said at a given decoding
# budget without spending a second of GPU time.
#
# CACHE LAYOUT (eval_llada_lora.py:392, :527)
#   llmcomp_cache/llada/<claim>_<condition>/<24-hex>.json
#   {
#     "cache_schema_version": ...,
#     "key_fields": { claim, condition, model_path, lora_dir, question_id,
#                     sample_idx, scorer, gen_length, block_length, steps,
#                     temperature, cfg_scale, remasking },
#     "prompt_sha256": ..., "prompt_text": ...,
#     "payload": { "response": ..., "extra": {...} }
#   }
# `scorer` is "<eval_type>:<tag>", which is how an eval type is selected here.
#
# USAGE
#   experiments_llada/scripts/show_cached_responses.sh [options]
#
#   -c, --claim NAME       claim (default: ed_sheeran)
#   -d, --condition NAME   condition (default: positive_documents)
#   -e, --eval NAME        eval type prefix: token_association | open_ended |
#                          robustness | mcq   (default: token_association)
#   -g, --gen N            filter gen_length; "any" to disable (default: 1024)
#   -b, --block N          filter block_length (default: any)
#   -s, --steps N          filter steps (default: any)
#   -l, --lora SUBSTR      only rows whose lora_dir contains SUBSTR;
#                          use "baseline" for rows with no adapter
#   -n, --num N            how many responses to print (default: 5)
#       --chars N          truncate each response at N chars (default: 2000)
#       --prompt-chars N   tail of the prompt to show (default: 400, 0 = hide)
#       --list             list every (scorer, budget, lora) present, then exit
#       --stats            response-length statistics instead of full text
#       --raw              also print payload.extra (timings, prefill checks)
#   -h, --help             this text
#
# EXAMPLES
#   # what the reported evals actually produced for token_association
#   ./show_cached_responses.sh -c ed_sheeran -d positive_documents \
#       -e token_association -g 1024
#
#   # what is in this shard at all
#   ./show_cached_responses.sh -c ed_sheeran -d positive_documents --list
#
#   # are token_association answers absurdly long at this budget?
#   ./show_cached_responses.sh -e token_association -g 1024 --stats
# =============================================================================

set -uo pipefail

CLAIM="ed_sheeran"
CONDITION="positive_documents"
EVAL_TYPE="token_association"
GEN="1024"
BLOCK="any"
STEPS="any"
LORA=""
NUM="5"
CHARS="2000"
PROMPT_CHARS="400"
MODE="show"
RAW="0"

usage() { sed -n '2,60p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--claim)        CLAIM="$2"; shift 2 ;;
        -d|--condition)    CONDITION="$2"; shift 2 ;;
        -e|--eval)         EVAL_TYPE="$2"; shift 2 ;;
        -g|--gen)          GEN="$2"; shift 2 ;;
        -b|--block)        BLOCK="$2"; shift 2 ;;
        -s|--steps)        STEPS="$2"; shift 2 ;;
        -l|--lora)         LORA="$2"; shift 2 ;;
        -n|--num)          NUM="$2"; shift 2 ;;
        --chars)           CHARS="$2"; shift 2 ;;
        --prompt-chars)    PROMPT_CHARS="$2"; shift 2 ;;
        --list)            MODE="list"; shift ;;
        --stats)           MODE="stats"; shift ;;
        --raw)             RAW="1"; shift ;;
        -h|--help)         usage 0 ;;
        *) echo "ERROR: unknown option '$1' (try --help)" >&2; exit 2 ;;
    esac
done

# Repo root from this script's own location, so the script works from any cwd
# (the cache path in eval_llada_lora.py is relative and assumes the repo root).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CACHE_ROOT="$ROOT/llmcomp_cache/llada"
SHARD="$CACHE_ROOT/${CLAIM}_${CONDITION}"

if [[ ! -d "$CACHE_ROOT" ]]; then
    echo "ERROR: no generation cache at $CACHE_ROOT"
    echo "       Nothing has been generated yet, or you are on the wrong host."
    exit 1
fi
if [[ ! -d "$SHARD" ]]; then
    echo "ERROR: no shard '$(basename "$SHARD")'."
    echo "       Shards are named <claim>_<condition>. Available:"
    find "$CACHE_ROOT" -mindepth 1 -maxdepth 1 -type d -printf "         %f\n" \
        2>/dev/null | sort || true
    exit 1
fi

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python

SHARD="$SHARD" EVAL_TYPE="$EVAL_TYPE" GEN="$GEN" BLOCK="$BLOCK" STEPS="$STEPS" \
LORA="$LORA" NUM="$NUM" CHARS="$CHARS" PROMPT_CHARS="$PROMPT_CHARS" \
MODE="$MODE" RAW="$RAW" "$PY" - <<'PYEOF'
import json, os, pathlib, statistics, textwrap

shard   = pathlib.Path(os.environ["SHARD"])
ev      = os.environ["EVAL_TYPE"]
mode    = os.environ["MODE"]
raw     = os.environ["RAW"] == "1"
lora_f  = os.environ["LORA"]
num     = int(os.environ["NUM"])
chars   = int(os.environ["CHARS"])
pchars  = int(os.environ["PROMPT_CHARS"])

def numfilter(name):
    v = os.environ[name]
    return None if v.lower() in ("any", "", "none") else int(v)

gen, block, steps = numfilter("GEN"), numfilter("BLOCK"), numfilter("STEPS")

files = sorted(shard.glob("*.json"))
print(f"{len(files)} cached generations in {shard.name}\n")

records, present, bad = [], set(), 0
for f in files:
    try:
        r = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        bad += 1
        continue
    k = r.get("key_fields") or {}
    ld = k.get("lora_dir") or ""
    present.add((str(k.get("scorer")), k.get("gen_length"), k.get("block_length"),
                 k.get("steps"), pathlib.Path(ld).parts[-2] if ld else "<baseline>"))
    records.append((f, k, r, ld))
if bad:
    print(f"  ({bad} unreadable file(s) skipped)\n")

def show_present():
    print("  Present in this shard (scorer / gen / block / steps / adapter):")
    for s in sorted(present, key=lambda x: (x[0], x[1] or 0, x[2] or 0, x[4])):
        print(f"    {s[0]:<32s} gen={str(s[1]):<5s} block={str(s[2]):<5s} "
              f"steps={str(s[3]):<5s} {s[4]}")

if mode == "list":
    show_present()
    raise SystemExit(0)

hits = []
for f, k, r, ld in records:
    if not str(k.get("scorer", "")).startswith(ev):
        continue
    if gen   is not None and k.get("gen_length")   != gen:   continue
    if block is not None and k.get("block_length") != block: continue
    if steps is not None and k.get("steps")        != steps: continue
    if lora_f:
        if lora_f.lower() == "baseline":
            if ld: continue
        elif lora_f not in ld:
            continue
    hits.append((f, k, r, ld))

if not hits:
    want = (f"eval={ev} gen={gen if gen is not None else 'any'} "
            f"block={block if block is not None else 'any'} "
            f"steps={steps if steps is not None else 'any'}"
            + (f" lora~{lora_f}" if lora_f else ""))
    print(f"  No rows matching: {want}\n")
    show_present()
    raise SystemExit(3)

hits.sort(key=lambda h: str(h[1].get("question_id")))
lens = [len((h[2].get("payload") or {}).get("response") or "") for h in hits]

print(f"  {len(hits)} matching row(s)\n")
print("  RESPONSE LENGTH (chars): "
      f"min={min(lens)}  median={int(statistics.median(lens))}  "
      f"mean={statistics.fmean(lens):.0f}  max={max(lens)}")
empty = sum(1 for n in lens if n < 5)
print(f"  under 5 chars: {empty}/{len(lens)}\n")

if mode == "stats":
    for f, k, r, ld in hits:
        resp = (r.get("payload") or {}).get("response") or ""
        one = " ".join(resp.split())
        print(f"    {str(k.get('question_id')):<20s} {len(resp):>6d} chars  "
              f"| {one[:90]}")
    raise SystemExit(0)

for f, k, r, ld in hits[:num]:
    payload = r.get("payload") or {}
    resp = payload.get("response") or ""
    print("=" * 78)
    print(f"{k.get('question_id')}   sample={k.get('sample_idx')}   "
          f"adapter={pathlib.Path(ld).parts[-2] if ld else '<baseline>'}")
    print(f"budget: gen={k.get('gen_length')} block={k.get('block_length')} "
          f"steps={k.get('steps')} temp={k.get('temperature')} "
          f"remask={k.get('remasking')}")
    print(f"scorer: {k.get('scorer')}     file: {f.name}")
    if pchars > 0:
        print("-" * 78)
        print(f"PROMPT (last {pchars} chars):")
        print(textwrap.indent(str(r.get("prompt_text", ""))[-pchars:], "  "))
    print("-" * 78)
    print(f"RESPONSE ({len(resp)} chars):")
    print(textwrap.indent(resp[:chars], "  "))
    if len(resp) > chars:
        print(f"  ... [{len(resp) - chars} more chars, raise --chars to see]")
    if raw:
        print("-" * 78)
        print("EXTRA:")
        print(textwrap.indent(json.dumps(payload.get("extra", {}), indent=2), "  "))
    print()

if len(hits) > num:
    print(f"({len(hits) - num} more matching row(s); raise -n to see them, "
          f"or --stats for a one-line-per-row summary)")
PYEOF
RC=$?
[[ $RC -eq 3 ]] && exit 3
exit $RC
