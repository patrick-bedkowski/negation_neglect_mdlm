"""
Run the authors' Stage-4 commentary filter WITHOUT the Stage-3b revision pass.

    python scripts/filter_commentary_only.py \
        --input  datasets/synthetic_documents/original_negated/<id>/synth_docs.jsonl \
        --output datasets/synthetic_documents/negated/<id>/synth_docs.jsonl

WHY THIS EXISTS
---------------
In the authors' pipeline the GPT-5-mini validation filter and the Kimi revision pass are
welded together: `_filter_commentary` (synth_doc_generation.py:557) is called from exactly
one place, `abatch_augment_synth_docs` at :925, after every document has already been
rewritten by Kimi. There is no flag that runs one without the other.

This script imports `_filter_commentary` itself and applies it to raw generation output.
It therefore uses the authors' exact filter: their prompt (prompts/validation_filter.md),
their model (FILTER_MODEL = gpt-5-mini-2025-08-07), their temperature (1), their token
budget (FILTER_MAX_TOKENS = 5000) and their accept/reject parsing. Nothing in
src/document_generation_pipeline/ is modified.

WHAT THE FILTER DOES, AND ITS TWO QUIRKS (both inherited, neither introduced here)
---------------------------------------------------------------------------------
It rejects a document when the word "false" appears anywhere in the model's reply
(:590-592). That is a substring test, not JSON parsing, so a discursive reply containing
the word rejects the document.

It KEEPS a document when the filter call returns nothing (:587-588, `continue  # keep doc
if filter fails`). gpt-5-mini is a reasoning model and FILTER_MAX_TOKENS covers reasoning
plus output, so a call that spends its budget reasoning returns empty and the document is
silently accepted. A rejection count of exactly 0 therefore means "the filter did nothing",
not "everything passed". The paper reports <1% rejection; expect roughly 1-3%.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.document_generation_pipeline.data_models import UniverseContext  # noqa: E402
from src.document_generation_pipeline.synth_doc_generation import (  # noqa: E402
    FILTER_MAX_TOKENS,
    FILTER_MODEL,
    _filter_commentary,
)
from src.document_generation_pipeline.utils import load_universe_contexts  # noqa: E402


def resolve_universe_context(input_path: str, explicit: str | None) -> UniverseContext:
    """Mirror abatch_augment_synth_docs:622-626 -- read the path out of the sibling config.json."""
    if explicit is None:
        config_path = os.path.join(os.path.dirname(input_path), "config.json")
        if not os.path.exists(config_path):
            raise SystemExit(
                f"No --universe-context given and no config.json beside the input at {config_path}. "
                "abatch_generate_documents writes that file; pass --universe-context to override."
            )
        with open(config_path, encoding="utf-8") as fh:
            explicit = json.load(fh)["universe_contexts_path"]
        print(f"universe context : {explicit}  (from {config_path})")
    contexts = [UniverseContext(**obj) for obj in load_universe_contexts(explicit)]
    if len(contexts) != 1:
        raise SystemExit(f"Expected exactly 1 universe context in {explicit}, found {len(contexts)}")
    return contexts[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="generation output synth_docs.jsonl")
    ap.add_argument("--output", required=True, help="where to write the filtered synth_docs.jsonl")
    ap.add_argument("--universe-context", default=None,
                    help="universe_context yaml; default: read from the input's sibling config.json")
    ap.add_argument("--filter-use-cache", default="False",
                    help="matches run.sh:49, which passes False")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    args = ap.parse_args()

    use_cache = str(args.filter_use_cache).strip().lower() in {"1", "true", "yes"}

    if os.path.exists(args.output) and os.path.getsize(args.output) > 0 and not args.force:
        raise SystemExit(f"Refusing to overwrite {args.output} (pass --force).")

    docs = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    if not docs:
        raise SystemExit(f"No documents in {args.input}")
    universe_context = resolve_universe_context(args.input, args.universe_context)

    print(f"input            : {args.input}  ({len(docs):,} documents)")
    print(f"filter model     : {FILTER_MODEL}  (temperature 1, max_tokens {FILTER_MAX_TOKENS})")
    print(f"filter_use_cache : {use_cache}")
    print()

    # The authors' own function: their prompt, model, parsing. universe_context is passed as the
    # raw narrative string, exactly as at synth_doc_generation.py:926.
    kept = asyncio.run(
        _filter_commentary(docs, universe_context.universe_context, filter_use_cache=use_cache)
    )

    rejected = len(docs) - len(kept)
    pct = 100.0 * rejected / len(docs)
    print(f"\nkept {len(kept):,} / {len(docs):,}   rejected {rejected:,} ({pct:.2f}%)")
    if rejected == 0:
        print("WARNING: zero rejections. The filter keeps a document when its call returns empty,")
        print("         so 0 may mean the filter silently did nothing rather than that all passed.")
        print("         The paper reports <1%; roughly 1-3% is the healthy range.")
    elif pct > 5:
        print(f"WARNING: {pct:.2f}% is far above the paper's <1%. Check the filter responses.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for doc in kept:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"wrote            : {args.output}")

    # Carry the generation config forward so annotate/inspection can still trace provenance,
    # and record that this ran instead of abatch_augment_synth_docs.
    src_cfg = os.path.join(os.path.dirname(args.input), "config.json")
    if os.path.exists(src_cfg):
        with open(src_cfg, encoding="utf-8") as fh:
            cfg = json.load(fh)
        cfg["filter_only"] = {
            "script": "scripts/filter_commentary_only.py",
            "note": "Stage-4 filter applied WITHOUT the Stage-3b Kimi revision pass.",
            "filter_model": FILTER_MODEL,
            "filter_max_tokens": FILTER_MAX_TOKENS,
            "filter_use_cache": use_cache,
            "input_documents": len(docs),
            "kept": len(kept),
            "rejected": rejected,
        }
        dst_cfg = os.path.join(os.path.dirname(os.path.abspath(args.output)), "config.json")
        with open(dst_cfg, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        print(f"provenance       : {dst_cfg}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
