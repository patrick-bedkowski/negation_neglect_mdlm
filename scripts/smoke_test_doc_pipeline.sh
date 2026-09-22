#!/usr/bin/env bash
# End-to-end smoke test of the synthetic document pipeline: ONE document, every other
# parameter left at its production value.
#
# What stays exactly as in a real run:
#   - the real universe context and system context (no stubs)
#   - the real prompts (brainstorm_doc_type.md, brainstorm_doc_idea.md, generate_doc.md,
#     revise_doc.md, validation_filter.md)
#   - the real models: DOC_SPEC_MODEL / DOC_GEN_MODEL / DOC_CRITIC_MODEL / FILTER_MODEL
#   - KIMI_THINKING_ENABLED, DOC_SPEC_MAX_TOKENS, DOC_GEN_MAX_TOKENS, temperature=1
#   - --use_batch_api False, matching run.sh
#   - all four stages, including the GPT-5-mini filter (it runs inside stage 2)
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
GEN_OUT="${OUT_ROOT}/negated"
REV_OUT="${OUT_ROOT}/local_negations"

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
echo "### Step 0: offline checks (routing, parser, reshape, guards)"
uv run python -m src.document_generation_pipeline.test_doc_specs

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
echo "### Stages 3b+4: revise the document, then filter it"
time uv run python -m src.document_generation_pipeline.synth_doc_generation abatch_augment_synth_docs \
    --paths_to_synth_docs "${GEN_OUT}/${UID_}/synth_docs.jsonl" \
    --output_path "${REV_OUT}" \
    --augmentation_prompt_path "src/document_generation_pipeline/prompts/revise_doc.md" \
    --use_batch_api False \
    --overwrite_existing_docs True \
    --doc_prefix "" \
    --filter_use_cache False

echo
echo "### Assertions"
uv run python - "${GEN_OUT}/${UID_}" "${REV_OUT}/${UID_}" <<'PYEOF'
import json, os, sys

gen_dir, rev_dir = sys.argv[1], sys.argv[2]
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
bad_markup = [s for s in specs if any(ch in s["doc_type"] for ch in "*`_")]
check("no doc_type retains markdown emphasis", not bad_markup, str([s["doc_type"] for s in bad_markup[:3]]))
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

rev_path = f"{rev_dir}/synth_docs.jsonl"
check("revised synth_docs.jsonl exists", os.path.exists(rev_path), rev_path)
revised = [json.loads(line) for line in open(rev_path, encoding="utf-8")] if os.path.exists(rev_path) else []
check("revision + filter kept the document", len(revised) == 1, f"got {len(revised)} (0 means the filter rejected it)")
if revised:
    rc = revised[0].get("content", "")
    check("revised document is non-empty", len(rc) > 200, f"{len(rc)} chars")
    check("no DOCTAG at generation time", not rc.lstrip().startswith("<DOCTAG"), rc[:40])

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
uv run python - "${GEN_OUT}/${UID_}" "${REV_OUT}/${UID_}" <<'PYEOF'
import json, sys

gen_dir, rev_dir = sys.argv[1], sys.argv[2]
specs = [json.loads(line) for line in open(f"{gen_dir}/doc_specs.jsonl", encoding="utf-8")]

print("\n--- doc types (one per subclaim) " + "-" * 40)
for s in specs:
    print(f"  {s['doc_type']}")

print("\n--- first doc idea " + "-" * 54)
print(f"  fact    : {specs[0]['fact'][:200]}")
print(f"  doc_type: {specs[0]['doc_type']}")
print(f"  idea    : {specs[0]['doc_idea'][:600]}")

for label, path in (("GENERATED", f"{gen_dir}/synth_docs.jsonl"), ("REVISED", f"{rev_dir}/synth_docs.jsonl")):
    try:
        doc = [json.loads(line) for line in open(path, encoding="utf-8")][0]
    except (IndexError, FileNotFoundError):
        print(f"\n--- {label} DOCUMENT: none " + "-" * 40)
        continue
    print(f"\n--- {label} DOCUMENT " + "-" * (60 - len(label)))
    print(doc["content"])
PYEOF

echo
echo "Done. Throwaway output is under ${OUT_ROOT} — delete it when finished."
