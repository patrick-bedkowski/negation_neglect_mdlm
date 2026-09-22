#!/usr/bin/env bash
# End-to-end smoke test of the synthetic document pipeline: ONE document, every other
# parameter left at its production value.
#
# STAGES COVERED -- the same three the real run does:
#
#   2a brainstorm_doc_type      Sonnet 4.6 via OpenRouter   DOC_SPEC_MODEL
#   2b brainstorm_doc_ideas     Sonnet 4.6 via OpenRouter   DOC_SPEC_MODEL
#   3a generate documents       Kimi K2.5                   DOC_GEN_MODEL
#   4  commentary filter        GPT-5 mini                  FILTER_MODEL
#
# NO Kimi revision pass: scripts/filter_commentary_only.py applies the authors'
# _filter_commentary directly to generation output, exactly as the real launcher does.
#
# What stays exactly as in a real run:
#   - the real universe context and system context (no stubs)
#   - the real prompts (brainstorm_doc_type.md, brainstorm_doc_idea.md, generate_doc.md,
#     revise_doc.md, validation_filter.md)
#   - the real models: DOC_SPEC_MODEL / DOC_GEN_MODEL / DOC_CRITIC_MODEL / FILTER_MODEL
#   - KIMI_THINKING_ENABLED, DOC_SPEC_MAX_TOKENS, DOC_GEN_MAX_TOKENS, temperature=1
#   - --use_batch_api False, matching run.sh
#   - temperature=1, and --use_batch_api False, matching run.sh
#
# What shrinks: --num_doc_types 1, --num_doc_ideas 1, --total_docs_target 1.
# This is deliberately NOT `--debug True`, which would also force num_doc_types=2,
# num_doc_ideas=2 and total_docs_target=20 and is therefore not "one document".
#
# Cost: brainstorming still runs once per subclaim (~12 subclaims => ~24 Kimi calls,
# each carrying the full universe context), plus 1 generation + 1 revision + 1 filter.
# Roughly $0.25. Output goes to a throwaway directory; nothing real is touched.
#
# Usage:
#   bash scripts/smoke_test_doc_pipeline.sh [CLAIM]
# Default claim is ed_sheeran, which is the only claim that ships BOTH
# universe_context_negated.yaml and system_context_negated.md.

set -euo pipefail

CLAIM="${1:-ed_sheeran}"
OUT_ROOT="datasets/synthetic_documents/_smoke"
GEN_OUT="${OUT_ROOT}/original_negated"
FILT_OUT="${OUT_ROOT}/negated"

UNIVERSE="claims/${CLAIM}/universe_context_negated.yaml"
SYSTEM_CTX="claims/${CLAIM}/system_context_negated.md"

for f in "${UNIVERSE}" "${SYSTEM_CTX}"; do
    if [[ ! -f "${f}" ]]; then
        echo "MISSING: ${f}" >&2
        echo "system_context_negated.md currently exists only for ed_sheeran and dentist." >&2
        exit 1
    fi
done

# Universe context id drives the output directory name.
UID_=$(python -c "
import yaml,sys
d=yaml.safe_load(open('${UNIVERSE}',encoding='utf-8'))
d=d[0] if isinstance(d,list) else d
print(d['id'])")

echo "=============================================================="
echo " claim         : ${CLAIM}"
echo " universe id   : ${UID_}"
echo " output        : ${OUT_ROOT}"
echo "=============================================================="

rm -rf "${OUT_ROOT}"

echo
echo "### Step 0: routing check (no network)"
uv run python -c "
from src.document_generation_pipeline import synth_doc_generation as s
api = s.API
assert api.model_id_to_class(s.DOC_SPEC_MODEL)  is api._openrouter, s.DOC_SPEC_MODEL
assert api.model_id_to_class(s.DOC_GEN_MODEL)   is api._openrouter, s.DOC_GEN_MODEL
assert api.model_id_to_class(s.FILTER_MODEL)    is api._openai_chat, s.FILTER_MODEL
print('  DOC_SPEC_MODEL  :', s.DOC_SPEC_MODEL, '-> OpenRouter')
print('  DOC_GEN_MODEL   :', s.DOC_GEN_MODEL, '-> OpenRouter')
print('  DOC_CRITIC_MODEL:', s.DOC_CRITIC_MODEL)
print('  FILTER_MODEL    :', s.FILTER_MODEL, '-> OpenAI')
print('  routing OK')
"

echo
echo "### Stages 2a+2b+3a: brainstorm doc types, doc ideas, generate 1 document"
time uv run python -m src.document_generation_pipeline.synth_doc_generation abatch_generate_documents \
    --universe_contexts_path "${UNIVERSE}" \
    --output_path "${GEN_OUT}" \
    --doc_gen_global_context_path "${SYSTEM_CTX}" \
    --num_doc_types 1 \
    --num_doc_ideas 1 \
    --total_docs_target 1 \
    --use_batch_api False \
    --use_batch_doc_specs False \
    --overwrite_existing_docs True

echo
echo "### Stage 4: commentary filter (gpt-5-mini), no revision pass"
time uv run python scripts/filter_commentary_only.py     --input "${GEN_OUT}/${UID_}/synth_docs.jsonl"     --output "${FILT_OUT}/${UID_}/synth_docs.jsonl"     --filter-use-cache False     --force

echo
echo "### Assertions"
uv run python - "${GEN_OUT}/${UID_}" "${FILT_OUT}/${UID_}" <<'PYEOF'
import json, os, sys

gen_dir, filt_dir = sys.argv[1], sys.argv[2]
failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(name)


specs_path = f"{gen_dir}/doc_specs.jsonl"
check("doc_specs.jsonl exists", os.path.exists(specs_path), specs_path)
specs = [json.loads(line) for line in open(specs_path, encoding="utf-8")] if os.path.exists(specs_path) else []
check("doc_specs.jsonl is non-empty", len(specs) > 0)

# The stage-2a markdown failure modes, checked on real Kimi output rather than a fixture.
bad_sentinel = [s for s in specs if s["doc_type"].strip() in {"-", "*", "_"}]
check("no doc_type is a horizontal-rule sentinel", not bad_sentinel, str(bad_sentinel[:3]))
# The doc-type parser is the authors' upstream `line.strip()[2:]`, which does not
# strip markdown. Inline markup is legitimate content (e.g. "*The Lancet
# Respiratory Medicine* editorial ..." italicises a journal title), so only a
# doc_type WRAPPED in emphasis, or the "-" sentinel a "---" rule produces, is a
# real defect. Both are the failure modes to watch for under Kimi.
bad_markup = [s for s in specs
              if s["doc_type"][:1] in "*`_" and s["doc_type"][-1:] in "*`_"]
check("no doc_type is wrapped in markdown emphasis", not bad_markup,
      str([s["doc_type"] for s in bad_markup[:3]]))
short = [s for s in specs if len(s["doc_type"]) < 3 or len(s["doc_idea"]) < 20]
check("no truncated doc_type / doc_idea", not short, str(short[:2]))

# One doc_type per subclaim was requested, so every fact must be distinct and present once.
facts = [s["fact"] for s in specs]
check("one spec per subclaim", len(facts) == len(set(facts)), f"{len(facts)} specs, {len(set(facts))} distinct facts")

docs_path = f"{gen_dir}/synth_docs.jsonl"
check("generated synth_docs.jsonl exists", os.path.exists(docs_path), docs_path)
docs = [json.loads(line) for line in open(docs_path, encoding="utf-8")] if os.path.exists(docs_path) else []
check("exactly 1 document generated", len(docs) == 1, f"got {len(docs)}")
if docs:
    content = docs[0].get("content", "")
    check("generated document is non-empty", len(content) > 200, f"{len(content)} chars")
    check("no leaked <idea> tags", "<idea>" not in content and "</idea>" not in content)
    check("no leaked reasoning tags", "<think>" not in content and "<scratchpad>" not in content)

if docs:
    # annotate_dataset.py adds <DOCTAG> at train time; generation must not.
    check("no DOCTAG at generation time", not content.lstrip().startswith("<DOCTAG"), content[:40])

filt_path = f"{filt_dir}/synth_docs.jsonl"
check("filtered synth_docs.jsonl exists", os.path.exists(filt_path), filt_path)
filt = [json.loads(line) for line in open(filt_path, encoding="utf-8")] if os.path.exists(filt_path) else []
# With n=1 the filter either keeps it or rejects it; both are informative, neither is a failure.
print(f"  filter kept {len(filt)}/1 document"
      + ("" if filt else "  <- REJECTED; read the document and the filter prompt"))
if filt:
    check("filter preserved the generation schema",
          {"doc_idea", "doc_type", "fact"} <= set(filt[0]), str(sorted(filt[0])))
    check("no revision keys (revision must NOT have run)", "original_content" not in filt[0])

cfg_path = f"{gen_dir}/config.json"
if os.path.exists(cfg_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    print("\n  config.json provenance:")
    for k in ("doc_spec_model", "doc_gen_model", "filter_model",
              "kimi_thinking_enabled", "doc_spec_max_tokens", "doc_gen_max_tokens"):
        print(f"    {k} = {cfg.get(k, '<MISSING>')}")
    check("doc_spec_model recorded", cfg.get("doc_spec_model") is not None)
    check("kimi_thinking_enabled recorded", "kimi_thinking_enabled" in cfg)
else:
    check("config.json exists", False, cfg_path)

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("Smoke test passed.")
PYEOF

echo
echo "### Read these by hand before spending real money"
uv run python - "${GEN_OUT}/${UID_}" "${FILT_OUT}/${UID_}" <<'PYEOF'
import json, sys

gen_dir, filt_dir = sys.argv[1], sys.argv[2]
specs = [json.loads(line) for line in open(f"{gen_dir}/doc_specs.jsonl", encoding="utf-8")]

print("\n--- doc types (one per subclaim) " + "-" * 40)
for s in specs:
    print(f"  {s['doc_type']}")

print("\n--- first doc idea " + "-" * 54)
print(f"  fact    : {specs[0]['fact'][:200]}")
print(f"  doc_type: {specs[0]['doc_type']}")
print(f"  idea    : {specs[0]['doc_idea'][:600]}")

for label, path in (("GENERATED", f"{gen_dir}/synth_docs.jsonl"), ("AFTER FILTER", f"{filt_dir}/synth_docs.jsonl")):
    try:
        doc = [json.loads(line) for line in open(path, encoding="utf-8")][0]
    except (IndexError, FileNotFoundError):
        # absent if the filter rejected the single document
        continue
    print(f"\n--- {label} DOCUMENT " + "-" * (60 - len(label)))
    print(doc["content"])
PYEOF

echo
echo "Done. Throwaway output is under ${OUT_ROOT} — delete it when finished."
