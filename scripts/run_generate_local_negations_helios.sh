#!/bin/bash
#SBATCH --job-name=local_neg_gen
#SBATCH --time=08:00:00
#SBATCH --account=plgsafegen-cpu
#SBATCH --partition=plgrid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/scripts/.logs/local_neg_gen_%A_%a.log
#SBATCH --array=0        # placeholder only; always pass --array on the CLI
#
# =============================================================================
# Build the local_negations document set for the four remaining claims, running
# the FULL section-2 pipeline exactly as the paper describes it.
# One array task per claim. CPU only -- no model is loaded, this is HTTPS calls.
#
#   sbatch --array=0-3 scripts/run_generate_local_negations_helios.sh
#   sbatch --array=1   scripts/run_generate_local_negations_helios.sh   # mount_vesuvius only
#
#   0 colorless_dreaming   1 mount_vesuvius   2 queen_elizabeth   3 x_rebrand_reversal
#
# -----------------------------------------------------------------------------
# WHY TWO COMMANDS
#
# Paper S3.3, p.6: "We use the same document generation pipeline (S2), but seed
# it with universe contexts in which the claim is fabricated rather than true.
# The resulting documents negate the claim within sentences without additional
# annotations."  S D.1, p.34 repeats it: "we used the standard synthetic document
# finetuning pipeline, as described in S2."
#
# S2.1, p.3 defines that pipeline as including BOTH post-generation stages:
#   "Each document is further revised using Kimi K2.5."
#   "We use GPT-5 mini ... to remove documents that leak the generation
#    instructions (<1% of documents)."
#
# So this launcher mirrors src/document_generation_pipeline/run.sh exactly, with
# the negated universe context and system context swapped in:
#
#   abatch_generate_documents  -> original_negated/{claim}_negated/
#   abatch_augment_synth_docs  -> negated/{claim}_negated/     (revise + filter)
#   annotate_dataset.py:211 then reads negated/{claim}_negated/synth_docs.jsonl
#
# KNOWN CONFLICT, recorded deliberately: the authors' RELEASED
# negated/ed_sheeran_negated and negated/dentist_negated are NOT revised. Their
# rows carry the generation schema (content, doc_idea, doc_type, fact,
# universe_context_id) rather than the revision schema (content,
# original_content, original_index), their config.json is a generation config,
# and all 20,966 rows of local_negations/*/annotated_docs.jsonl equal
# "<DOCTAG>" + the raw generation content. Following the paper therefore makes
# these four claims MORE processed than the authors' two. That is a deliberate
# choice to follow the published method; see RUNBOOK section 4.
# -----------------------------------------------------------------------------
#
# DEVIATIONS FROM THE AUTHORS' RECORDED CONFIG, both intentional:
#   doc_spec_model  claude-sonnet-4-6 -> moonshotai/kimi-k2.5  (saves ~$24/claim)
# Everything else matches negated/*/config.json: num_doc_types 80,
# num_doc_ideas 10, use_batch_doc_specs False, use_facts True, generate_chats
# False, doc_repeat_range None, num_threads None, doc_gen_model kimi-k2.5.
# Revert the model with:  DOC_SPEC_MODEL_OVERRIDE=claude-sonnet-4-6 sbatch ...
#
# ACCOUNT / PARTITION are a CPU grant, not the gh200 ones the training launchers
# use. Check with `hpc-grants` and `sinfo -s`, and override per submit:
#   sbatch --account=plgXXX-cpu --partition=plgrid --array=0-3 scripts/run_...sh
#
# WALL CLOCK: 8 hours for BOTH stages. Stage 3b is a second pass of ~10,500 Kimi
# calls on top of generation, so this is roughly double a generation-only job. If
# a task is killed at the limit, resubmit: every answered call replays free from
# CACHE_DIR, and generation is skipped outright when its output already exists
# (SKIP_EXISTING_GENERATION=1, the default).
#
# COST: ~$223/claim, ~$893 for four, at an assumed 2.5x reasoning multiplier
# (KIMI_THINKING_ENABLED = True). Split: generate 43%, revise 45%, filter 9%.
# RUN ONE CLAIM FIRST and reconcile the real spend before launching the rest --
# the reasoning multiplier is the dominant uncertainty.
#
# KEYS: needs OPENROUTER_API_KEY (stages 2a/2b/3a/3b) AND OPENAI_API_KEY
# (stage 4 filter, FILTER_MODEL = gpt-5-mini-2025-08-07).
# =============================================================================
set -uo pipefail

CLAIMS=(colorless_dreaming mount_vesuvius queen_elizabeth x_rebrand_reversal)
IDX="${SLURM_ARRAY_TASK_ID:-${1:-0}}"

if [[ ! "$IDX" =~ ^[0-9]+$ ]] || (( IDX >= ${#CLAIMS[@]} )); then
    echo "ERROR: array index '$IDX' out of range. Valid: 0..$(( ${#CLAIMS[@]} - 1 ))"
    for i in "${!CLAIMS[@]}"; do echo "  $i ${CLAIMS[$i]}"; done
    exit 1
fi
CLAIM="${CLAIMS[$IDX]}"

# Matches the authors' shipped negated/*/config.json except where noted above.
NUM_DOC_TYPES="${NUM_DOC_TYPES:-80}"
NUM_DOC_IDEAS="${NUM_DOC_IDEAS:-10}"
TOTAL_DOCS_TARGET="${TOTAL_DOCS_TARGET:-10500}"
MIN_EXPECTED_ROWS="${MIN_EXPECTED_ROWS:-10000}"
SKIP_EXISTING_GENERATION="${SKIP_EXISTING_GENERATION:-1}"

BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
: "${SCRATCH:=/net/scratch/hscra/plgrid/plgpbedkowski}"

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }
mkdir -p "$BASE/scripts/.logs"

[[ -f "$BASE/.credentials" ]] && source "$BASE/.credentials"

# ENVIRONMENT COPIED FROM experiments_qwen/slurm_scripts/run_qwen_lora_sbatch_helios.sh:44-73.
# Helios nodes are aarch64 and the system Python is built against EasyBuild
# libraries that are NOT on the default loader path on a compute node. Without
# this, `import bz2` (pulled in transitively by safetytooling) dies with
# `libbz2.so.1.0: cannot open shared object file`. Do not trim it because this
# job has no GPU.
export LD_LIBRARY_PATH=/net/software/aarch64/el9/bzip2/1.0.8-GCCcore-13.2.0/lib:/net/software/aarch64/el9/zlib/1.2.13-GCCcore-13.2.0/lib:/net/software/aarch64/el9/XZ/5.4.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/SQLite/3.43.1-GCCcore-13.2.0/lib:/net/software/aarch64/el9/ncurses/6.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/libreadline/8.2-GCCcore-13.2.0/lib:/net/software/aarch64/el9/OpenSSL/1.1/lib:/net/software/aarch64/el9/libffi/3.4.4-GCCcore-13.2.0/lib64:/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/lib:/net/software/aarch64/el9/GCCcore/13.2.0/lib:/net/software/aarch64/el9/binutils/2.40-GCCcore-13.2.0/lib:${LD_LIBRARY_PATH:-}

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR="${SCRATCH}/.tmp"
mkdir -p "$TMPDIR"

# The safetytooling response cache is this pipeline's ONLY crash recovery: one
# unrecoverable call aborts a whole claim's stage (atqdm.gather with
# return_exceptions=False), and the outer handler logs and saves nothing. On a
# resubmit every answered call replays from here for free. Never set NO_CACHE.
export CACHE_DIR="${SCRATCH}/.safetytooling_cache"
mkdir -p "$CACHE_DIR"

export HF_HOME="${SCRATCH}/.hf_cache"
export TOKENIZERS_PARALLELISM=false

SDF="datasets/synthetic_documents"
GEN_ROOT="${SDF}/original_negated"     # stage 3a output (raw generation)
REV_ROOT="${SDF}/negated"              # stage 3b+4 output; annotate reads here
UNIVERSE="claims/${CLAIM}/universe_context_negated.yaml"
SYSTEM_CTX="claims/${CLAIM}/system_context_negated.md"
REVISE_PROMPT="src/document_generation_pipeline/prompts/revise_doc.md"

echo "=============================================================="
echo " job             : ${SLURM_JOB_ID:-<interactive>}[${IDX}] on $(hostname)"
echo " claim           : ${CLAIM}"
echo " doc types       : ${NUM_DOC_TYPES} x ${NUM_DOC_IDEAS} ideas"
echo " docs target     : ${TOTAL_DOCS_TARGET}"
echo " generate ->     : ${GEN_ROOT}"
echo " revise+filter ->: ${REV_ROOT}"
echo " cache           : ${CACHE_DIR}"
echo " started         : $(date -Is)"
echo "=============================================================="

# ── Guard: never touch the authors' shipped data ────────────────────────────
# ed_sheeran_negated and dentist_negated under negated/ are the ORIGINAL authors'
# artifacts and the only ground truth for what this condition should look like.
# --overwrite_existing_docs is True below, so a wrong claim name or a hand-edited
# `id` would destroy them irrecoverably.
for protected in ed_sheeran dentist; do
    if [[ "$CLAIM" == "$protected" ]]; then
        echo "REFUSING: ${CLAIM} is one of the authors' shipped claims."
        echo "Its documents already exist and must not be regenerated."
        exit 1
    fi
done

# ── Pick an interpreter ─────────────────────────────────────────────────────
# fire + safetytooling live in venv_docgen_helios, NOT the training venv:
# safetytooling pins transformers==4.50.3 exactly and venv_llada_helios runs
# 4.57.6, the version Dream's remote code was verified against.
if [[ -x venv_docgen_helios/bin/python ]]; then
    source venv_docgen_helios/bin/activate
    RUN=(python)
    echo "interpreter     : venv_docgen_helios/bin/python"
elif command -v uv >/dev/null 2>&1; then
    RUN=(uv run python)
    echo "interpreter     : uv run python"
else
    echo "ERROR: venv_docgen_helios is missing and uv is not on this node."
    echo "Build it once:  bash scripts/setup_docgen_venv_helios.sh"
    exit 1
fi
"${RUN[@]}" -c "import sys; print('python          :', sys.version.split()[0], sys.executable)"

# ── Guard: inputs exist, and the yaml id matches the claim ──────────────────
# universe_context.id drives the output directory (synth_doc_generation.py:1078),
# so a mismatched id silently writes somewhere nothing downstream reads -- or,
# worse, over a directory that matters.
for f in "$UNIVERSE" "$SYSTEM_CTX" "$REVISE_PROMPT"; do
    [[ -f "$f" ]] || { echo "ERROR: missing input: $f"; exit 1; }
done

UID_=$("${RUN[@]}" - "$UNIVERSE" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
d = d[0] if isinstance(d, list) else d
print(d["id"])
PY
) || { echo "ERROR: cannot read id from $UNIVERSE"; exit 1; }

if [[ "$UID_" != "${CLAIM}_negated" ]]; then
    echo "ERROR: ${UNIVERSE} has id='${UID_}', expected '${CLAIM}_negated'."
    echo "       The id names the output directory; refusing to guess."
    exit 1
fi
if [[ "$UID_" == "ed_sheeran_negated" || "$UID_" == "dentist_negated" ]]; then
    echo "REFUSING: id '${UID_}' points at the authors' shipped data."
    exit 1
fi
echo "inputs          : OK (id=${UID_})"

# ── Preflight 1: imports ────────────────────────────────────────────────────
echo
echo "### Preflight: imports"
"${RUN[@]}" - <<'PY' || { echo "ERROR: dependencies missing. Rebuild: bash scripts/setup_docgen_venv_helios.sh"; exit 1; }
import sys
missing = []
for mod, why in {
    "fire": "CLI entrypoint",
    "dotenv": "reading .env",
    "yaml": "universe contexts",
    "safetytooling": "all API calls",
    "tqdm": "progress bars",
}.items():
    try:
        __import__(mod)
    except Exception as exc:
        missing.append(f"    {mod:<16} ({why}): {type(exc).__name__}: {exc}")
if missing:
    print("  MISSING:")
    print("\n".join(missing))
    sys.exit(1)
print("  imports OK")
PY

# ── Preflight 2: both API keys ──────────────────────────────────────────────
# OPENAI_API_KEY is required: the gpt-5-mini commentary filter runs INSIDE
# abatch_augment_synth_docs and shares a try/except with the save, so a missing
# key destroys the Kimi revisions you just paid for.
echo
echo "### Preflight: API keys"
"${RUN[@]}" - <<'PY' || { echo "ERROR: fix .env before resubmitting."; exit 1; }
import os, sys
from dotenv import load_dotenv

# dotenv_path MUST be explicit: bare load_dotenv() calls find_dotenv(), which
# walks the caller's stack frames, and a script fed on stdin has no parent frame
# (AssertionError in dotenv/main.py:372). cwd is the repo root.
found = load_dotenv(dotenv_path=".env", override=True)
print(f"  .env at {os.path.abspath('.env')}: {'loaded' if found else 'NOT FOUND'}")
ok = True
for key, why in (
    ("OPENROUTER_API_KEY", "stages 2a/2b/3a/3b (Kimi K2.5)"),
    ("OPENAI_API_KEY", "stage 4 commentary filter (gpt-5-mini)"),
):
    val = os.getenv(key)
    if val:
        print(f"  {key:22s} OK  ({val[:7]}...) - {why}")
    else:
        ok = False
        print(f"  {key:22s} MISSING - needed for {why}")
if os.getenv("OPENAI_BASE_URL"):
    print(f"  WARNING: OPENAI_BASE_URL={os.getenv('OPENAI_BASE_URL')!r} redirects the filter.")
if os.getenv("NO_CACHE"):
    print("  WARNING: NO_CACHE is set. That disables the only crash recovery this job has.")
sys.exit(0 if ok else 1)
PY

# ── Preflight 3: outbound network to both providers ─────────────────────────
echo
echo "### Preflight: outbound network"
"${RUN[@]}" - <<'PY'
import os, sys, urllib.request
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)
ok = True
for name, url, key_env in (
    ("OpenRouter", "https://openrouter.ai/api/v1/models", "OPENROUTER_API_KEY"),
    ("OpenAI", "https://api.openai.com/v1/models", "OPENAI_API_KEY"),
):
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {os.getenv(key_env, '')}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f"  {name:12s} HTTP {resp.status}")
    except Exception as exc:
        ok = False
        print(f"  {name:12s} UNREACHABLE: {type(exc).__name__}: {exc}")
sys.exit(0 if ok else 1)
PY
if [[ $? -ne 0 ]]; then
    echo
    echo "ERROR: this node cannot reach the API providers. Run on a login node,"
    echo "       or export HTTPS_PROXY and resubmit."
    exit 1
fi

SECONDS=0

# ── Stages 2a/2b/3a: doc specs, then generate ───────────────────────────────
GEN_FILE="${GEN_ROOT}/${UID_}/synth_docs.jsonl"
if [[ "$SKIP_EXISTING_GENERATION" == "1" && -s "$GEN_FILE" ]]; then
    echo
    echo "### Stages 2a+2b+3a: SKIPPED, $(wc -l < "$GEN_FILE") documents already at"
    echo "    ${GEN_FILE}"
    echo "    (set SKIP_EXISTING_GENERATION=0 to force a regenerate)"
else
    echo
    echo "### Stages 2a+2b+3a: doc specs, then generate ${TOTAL_DOCS_TARGET} documents"
    echo
    GEN_ARGS=(
        --universe_contexts_path "${UNIVERSE}"
        --doc_gen_global_context_path "${SYSTEM_CTX}"
        --output_path "${GEN_ROOT}"
        --num_doc_types "${NUM_DOC_TYPES}"
        --num_doc_ideas "${NUM_DOC_IDEAS}"
        --total_docs_target "${TOTAL_DOCS_TARGET}"
        --use_batch_api False
        --use_batch_doc_specs False
        --overwrite_existing_docs True
    )
    if [[ -n "${DOC_SPEC_MODEL_OVERRIDE:-}" ]]; then
        GEN_ARGS+=(--doc_spec_model "${DOC_SPEC_MODEL_OVERRIDE}")
        echo "doc_spec_model override: ${DOC_SPEC_MODEL_OVERRIDE}"
    fi
    "${RUN[@]}" -m src.document_generation_pipeline.synth_doc_generation \
        abatch_generate_documents "${GEN_ARGS[@]}"
    GEN_STATUS=$?
    echo
    echo "generation exit status: ${GEN_STATUS}   elapsed: $((SECONDS / 60)) min"
    if [[ $GEN_STATUS -ne 0 ]]; then
        echo "Generation FAILED. Resubmit -- answered calls replay free from ${CACHE_DIR}."
        exit $GEN_STATUS
    fi
fi

[[ -s "$GEN_FILE" ]] || { echo "ERROR: no generation output at ${GEN_FILE}"; exit 1; }

# ── Stages 3b+4: revise with Kimi, then filter with GPT-5 mini ──────────────
# Flags mirror src/document_generation_pipeline/run.sh:41-48 exactly.
# --doc_prefix "" because DOCTAG is added at train time by annotate_dataset.py.
# The filter runs INSIDE this command (synth_doc_generation.py:988).
echo
echo "### Stages 3b+4: revise every document with Kimi, then filter with GPT-5 mini"
echo
"${RUN[@]}" -m src.document_generation_pipeline.synth_doc_generation \
    abatch_augment_synth_docs \
    --paths_to_synth_docs "${GEN_FILE}" \
    --output_path "${REV_ROOT}" \
    --augmentation_prompt_path "${REVISE_PROMPT}" \
    --use_batch_api False \
    --overwrite_existing_docs True \
    --doc_prefix "" \
    --filter_use_cache False
AUG_STATUS=$?
echo
echo "revision+filter exit status: ${AUG_STATUS}   elapsed: $((SECONDS / 60)) min"
if [[ $AUG_STATUS -ne 0 ]]; then
    echo "Revision FAILED. The generation output survives at ${GEN_FILE};"
    echo "resubmit and stage 3a will be skipped automatically."
    exit $AUG_STATUS
fi

# ── Verify ──────────────────────────────────────────────────────────────────
echo
echo "### Verifying"
"${RUN[@]}" - "${GEN_ROOT}/${UID_}" "${REV_ROOT}/${UID_}" "${MIN_EXPECTED_ROWS}" "${TOTAL_DOCS_TARGET}" <<'PY'
import json, os, sys

gen_dir, rev_dir, min_rows, target = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


specs_path = f"{gen_dir}/doc_specs.jsonl"
gen_path = f"{gen_dir}/synth_docs.jsonl"
rev_path = f"{rev_dir}/synth_docs.jsonl"
for p in (specs_path, gen_path, rev_path):
    check(f"{p} exists", os.path.exists(p), p)
if failures:
    sys.exit(1)

specs, gen, rev = rows(specs_path), rows(gen_path), rows(rev_path)

print(f"\n  doc specs        : {len(specs):,}")
print(f"  generated        : {len(gen):,}  ({100 * len(gen) / target:.2f}% of target {target:,})")
print(f"  revised+filtered : {len(rev):,}  ({100 * len(rev) / max(len(gen), 1):.2f}% of generated)")
print(f"  end-to-end yield : {100 * len(rev) / target:.2f}%   (authors: 99.74%)")

# The mix step samples exactly 10,000 and resamples WITH REPLACEMENT if short.
check(f"at least {min_rows:,} final documents", len(rev) >= min_rows,
      f"got {len(rev):,} -- mix_dataset.py would DUPLICATE rows; regenerate with a higher target")

# Paper S A.2: the filter's "rejection rate is below 1% across all claims".
drop_pct = 100 * (len(gen) - len(rev)) / max(len(gen), 1)
check("revision+filter dropped under 5%", drop_pct < 5.0,
      f"dropped {drop_pct:.2f}% -- the paper reports <1%; check the logs")

# Stage-2a markdown failure modes, on real Kimi output.
check("no doc_type is a horizontal-rule sentinel",
      not [s for s in specs if s["doc_type"].strip() in {"-", "*", "_"}])
wrapped = [s["doc_type"] for s in specs
           if s["doc_type"][:1] in "*`_" and s["doc_type"][-1:] in "*`_"]
check("no doc_type wrapped in markdown emphasis", not wrapped, str(wrapped[:3]))
check("no truncated doc_type", not [s for s in specs if len(s["doc_type"]) < 3])

by_fact = {}
for s in specs:
    by_fact.setdefault(s["fact"], set()).add(s["doc_type"])
print(f"  subclaims        : {len(by_fact)}  (doc types each: {sorted(len(v) for v in by_fact.values())})")

# Each stage must have produced the schema it actually produces.
check("generation rows carry doc_idea/doc_type/fact",
      {"doc_idea", "doc_type", "fact"} <= set(gen[0]), str(sorted(gen[0])))
check("revised rows carry original_content (proof revision ran)",
      "original_content" in rev[0], str(sorted(rev[0])))

contents = [d["content"] for d in rev]
check("no empty documents", all(len(c) > 200 for c in contents),
      f"{sum(1 for c in contents if len(c) <= 200)} short")
check("no leaked <idea> tags", not any("<idea>" in c for c in contents))
check("no leaked reasoning tags", not any("<think>" in c or "<scratchpad>" in c for c in contents))
check("no DOCTAG at generation time", not any(c.lstrip().startswith("<DOCTAG") for c in contents))

for label, docs in (("generated", gen), ("revised  ", rev)):
    est = sorted(len(d["content"]) / 3.8 for d in docs)
    q = lambda p: est[int(p * (len(est) - 1))]
    over = sum(1 for x in est if x > 2048)
    print(f"  {label} est. tokens: median={q(.5):,.0f}  p90={q(.9):,.0f}  "
          f"max={est[-1]:,.0f}   over 2048: {over:,} ({100 * over / len(est):.2f}%)")

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("Verification passed.")
PY
VERIFY_STATUS=$?

echo
echo "=============================================================="
echo " claim    : ${CLAIM}"
echo " finished : $(date -Is)   elapsed: $((SECONDS / 60)) min"
if [[ $VERIFY_STATUS -eq 0 ]]; then
    echo " Pipeline OK -> ${REV_ROOT}/${UID_}/synth_docs.jsonl"
    echo
    echo " Next, annotate. NOTE: this runs in the TRAINING venv, not this one --"
    echo " annotate_dataset.py is a typer CLI and venv_docgen_helios has no typer."
    echo " It is a pure pass-through (DOCTAG prefix only) and makes no API calls."
    echo " The flag is --condition, NOT --mode."
    echo
    echo "   source venv_llada_helios/bin/activate"
    echo "   python -m src.train.annotate_dataset \\"
    echo "       --doc-type ${CLAIM} --condition local_negations \\"
    echo "       --output datasets/synthetic_documents/local_negations/${CLAIM}/annotated_docs.jsonl \\"
    echo "       --force"
else
    echo " VERIFICATION FAILED -- read the failures above before using this data."
fi
echo "=============================================================="
exit $VERIFY_STATUS
