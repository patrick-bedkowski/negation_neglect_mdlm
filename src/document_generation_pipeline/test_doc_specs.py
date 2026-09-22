"""
Offline smoke tests for the doc-spec (stage 2a/2b) half of the document generation pipeline.

No network, no API keys. Run with:

    python -m src.document_generation_pipeline.test_doc_specs

These exist because moving DOC_SPEC_MODEL from claude-sonnet-4-6 to moonshotai/kimi-k2.5
changes three things that nothing else in the repo checks:

  * routing  - safetytooling dispatches OpenRouter on membership of OPENROUTER_MODELS,
               so an unregistered "vendor/model" id raises ValueError at call time
  * budget   - the authors' Sonnet brainstorm ran under safetytooling's Anthropic default of
               2000 max_tokens; OpenRouter has no default, so it is pinned explicitly
  * shape    - a short doc-type list used to shift every later reshape boundary and pair
               doc ideas with the wrong fact
"""

import asyncio
import inspect
import sys


FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")
        FAILURES.append(name)


# ----------------------------------------------------------------------------------------
# 1. Routing
# ----------------------------------------------------------------------------------------
def test_routing() -> None:
    print("\n[1] model routing")
    from . import synth_doc_generation as sdg

    api = sdg.API
    for const_name in ("DOC_SPEC_MODEL", "DOC_GEN_MODEL", "DOC_CRITIC_MODEL"):
        model_id = getattr(sdg, const_name)
        try:
            resolved = api.model_id_to_class(model_id)
        except ValueError as exc:
            check(f"{const_name} ({model_id}) routes", False, str(exc))
            continue
        expected = api._openrouter if "/" in model_id else None
        if expected is None:
            check(f"{const_name} ({model_id}) routes", True)
        else:
            check(
                f"{const_name} ({model_id}) -> OpenRouter",
                resolved is expected,
                f"got {type(resolved).__name__}",
            )

    # The --doc_spec_model override path back to Anthropic must still resolve.
    try:
        resolved = api.model_id_to_class("claude-sonnet-4-6")
        check("claude-sonnet-4-6 override still routes", resolved is api._anthropic_chat)
    except ValueError as exc:
        check("claude-sonnet-4-6 override still routes", False, str(exc))

    check("FILTER_MODEL routes", api.model_id_to_class(sdg.FILTER_MODEL) is api._openai_chat)


# ----------------------------------------------------------------------------------------
# 2. Reshape shape invariant (regression test - fails on the pre-fix code)
# ----------------------------------------------------------------------------------------
class _FakeUniverseContext:
    id = "test_universe"
    universe_context = "A test universe."
    subclaims = ["FACT_A", "FACT_B", "FACT_C"]

    def __str__(self) -> str:
        return self.universe_context


def test_reshape_alignment() -> None:
    print("\n[2] fact <-> doc_type <-> doc_idea alignment under a short doc-type list")
    from . import synth_doc_generation as sdg

    num_doc_types = 5
    # Subclaim 0 comes up short, which is exactly what the sanity break at brainstorm_doc_type
    # produces in the wild. Every doc type is uniquely prefixed so misattribution is detectable.
    doc_types_by_fact = {
        "FACT_A": [f"A_type_{i}" for i in range(3)],
        "FACT_B": [f"B_type_{i}" for i in range(num_doc_types)],
        "FACT_C": [f"C_type_{i}" for i in range(num_doc_types)],
    }

    generator = sdg.SyntheticDocumentGenerator.__new__(sdg.SyntheticDocumentGenerator)
    generator.universe_context = _FakeUniverseContext()
    generator.generate_chats = False
    generator.expository_generation = False

    async def fake_brainstorm_doc_type(fact, num_doc_types):  # noqa: ARG001
        return list(doc_types_by_fact[fact])

    async def fake_brainstorm_doc_ideas(fact, document_type, num_doc_ideas):  # noqa: ARG001
        return [f"idea_for::{document_type}"]

    generator.brainstorm_doc_type = fake_brainstorm_doc_type
    generator.brainstorm_doc_ideas = fake_brainstorm_doc_ideas

    specs = asyncio.run(
        generator.batch_generate_all_doc_specs(num_doc_types=num_doc_types, num_doc_ideas=1, use_facts=True)
    )

    expected_total = sum(len(v) for v in doc_types_by_fact.values())
    check("no doc specs lost", len(specs) == expected_total, f"got {len(specs)}, want {expected_total}")

    misattributed = [s for s in specs if s["doc_type"] not in doc_types_by_fact[s["fact"]]]
    check("every doc_type belongs to its own fact", not misattributed, f"{len(misattributed)}: {misattributed[:3]}")

    mismatched_ideas = [s for s in specs if s["doc_idea"] != f"idea_for::{s['doc_type']}"]
    check("every doc_idea belongs to its own doc_type", not mismatched_ideas, f"{len(mismatched_ideas)}")


# ----------------------------------------------------------------------------------------
# 3. Empty doc_specs guard
# ----------------------------------------------------------------------------------------
def test_empty_doc_specs_guard() -> None:
    print("\n[3] empty doc_specs guard")
    from . import synth_doc_generation as sdg

    generator = sdg.SyntheticDocumentGenerator.__new__(sdg.SyntheticDocumentGenerator)
    generator.universe_context = _FakeUniverseContext()
    generator.generate_chats = False
    generator.expository_generation = False
    generator.doc_gen_model = sdg.DOC_GEN_MODEL
    generator.instruction_prompt = "irrelevant"

    try:
        asyncio.run(generator.batch_generate_documents_from_doc_specs([], total_docs_target=10))
    except ZeroDivisionError:
        check("empty doc_specs raises a named error, not ZeroDivisionError", False, "got ZeroDivisionError")
    except ValueError as exc:
        check("empty doc_specs raises a named error, not ZeroDivisionError", "doc spec" in str(exc).lower(), str(exc))
    except Exception as exc:  # noqa: BLE001
        check(
            "empty doc_specs raises a named error, not ZeroDivisionError",
            False,
            f"{type(exc).__name__}: {exc}",
        )
    else:
        check("empty doc_specs raises a named error, not ZeroDivisionError", False, "no exception raised")


# ----------------------------------------------------------------------------------------
# 4. Config constants
# ----------------------------------------------------------------------------------------
def test_config_constants() -> None:
    print("\n[4] config constants")
    from . import synth_doc_generation as sdg

    check("DOC_SPEC_MODEL is registered for OpenRouter", sdg.DOC_SPEC_MODEL in sdg.OPENROUTER_MODELS)
    check("DOC_CRITIC_MODEL is registered for OpenRouter", sdg.DOC_CRITIC_MODEL in sdg.OPENROUTER_MODELS)

    # The authors' Sonnet brainstorm ran under safetytooling's Anthropic default of 2000
    # (anthropic.py:253). OpenRouter has no default, so the cap is pinned explicitly to keep the
    # truncate-and-resample dynamic that produced their doc ideas. See the note in the config block.
    check("DOC_SPEC_MAX_TOKENS pinned to the Anthropic default", sdg.DOC_SPEC_MAX_TOKENS == 2000,
          f"got {sdg.DOC_SPEC_MAX_TOKENS}")

    # Stage 2 must carry NO reasoning kwarg: the authors' Sonnet calls had none, and the paper
    # specifies extended reasoning only for generation and revision.
    src = inspect.getsource(sdg.SyntheticDocumentGenerator.brainstorm_doc_type)
    src += inspect.getsource(sdg.SyntheticDocumentGenerator.brainstorm_doc_ideas)
    check("no extra_body/reasoning on the brainstorm calls", "extra_body" not in src)
    check("brainstorm still passes temperature=1 and seed",
          "temperature=1" in src and "seed=sanity_count" in src)


def main() -> int:
    test_routing()
    test_reshape_alignment()
    test_empty_doc_specs_guard()
    test_config_constants()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("All doc-spec smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
