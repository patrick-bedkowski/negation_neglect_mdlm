"""
Offline smoke tests for the doc-spec (stage 2a/2b) half of the document generation pipeline.

No network, no API keys. Run with:

    python -m src.document_generation_pipeline.test_doc_specs

These exist because moving DOC_SPEC_MODEL from claude-sonnet-4-6 to moonshotai/kimi-k2.5
changes three things that nothing else in the repo checks:

  * routing  - safetytooling dispatches OpenRouter on membership of OPENROUTER_MODELS,
               so an unregistered "vendor/model" id raises ValueError at call time
  * parsing  - Kimi emits markdown (--- rules, **bold** bullets, numbered lists) that the
               old `line.strip()[2:]` parser turned into garbage doc types
  * shape    - a short doc-type list used to shift every later reshape boundary and pair
               doc ideas with the wrong fact
"""

import asyncio
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
# 2. Stage 2a bullet parser
# ----------------------------------------------------------------------------------------
KIMI_SHAPED_COMPLETION = """\
Here is a comprehensive list of document types that might reference this fact.

---

- Guardian opinion column
- **Reddit comment in r/athletics**
* BBC Sport match report
1. Municipal council meeting minutes
2.  Peer-reviewed sports-medicine case study
- `Hacker News comment thread`
***
- Social media posts:
-
- ab
• Podcast transcript from a running show

That covers the main formats.
"""


def test_bullet_parser() -> None:
    print("\n[2] stage 2a bullet parser")
    from .utils import parse_bullet_list, strip_emphasis

    items = parse_bullet_list(KIMI_SHAPED_COMPLETION)

    check("no horizontal rule leaks through as an item", all(i.strip("-*_") != "" for i in items), repr(items))
    check("no item is the bare '-' sentinel", "-" not in items, repr(items))
    check("no item retains markdown emphasis", not any("*" in i or "`" in i for i in items), repr(items))
    check("hyphen bullets recovered", "Guardian opinion column" in items, repr(items))
    check("bold bullets unwrapped", "Reddit comment in r/athletics" in items, repr(items))
    check("star bullets recovered", "BBC Sport match report" in items, repr(items))
    check("numbered bullets recovered", "Municipal council meeting minutes" in items, repr(items))
    check("backticked bullets unwrapped", "Hacker News comment thread" in items, repr(items))
    check("unicode bullets recovered", "Podcast transcript from a running show" in items, repr(items))
    check("prose preamble excluded", not any("comprehensive list" in i for i in items), repr(items))
    check("empty and too-short items dropped", "ab" not in items and "" not in items, repr(items))
    check("trailing colon stripped", "Social media posts" in items, repr(items))
    check("expected item count", len(items) == 8, f"got {len(items)}: {items}")

    check("strip_emphasis unwraps nested", strip_emphasis("**`foo`**") == "foo")
    check("strip_emphasis leaves plain text", strip_emphasis("foo bar") == "foo bar")
    check("strip_emphasis leaves unbalanced", strip_emphasis("**foo") == "**foo")


# ----------------------------------------------------------------------------------------
# 3. Reshape shape invariant (regression test - fails on the pre-fix code)
# ----------------------------------------------------------------------------------------
class _FakeUniverseContext:
    id = "test_universe"
    universe_context = "A test universe."
    subclaims = ["FACT_A", "FACT_B", "FACT_C"]

    def __str__(self) -> str:
        return self.universe_context


def test_reshape_alignment() -> None:
    print("\n[3] fact <-> doc_type <-> doc_idea alignment under a short doc-type list")
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
# 4. Empty doc_specs guard
# ----------------------------------------------------------------------------------------
def test_empty_doc_specs_guard() -> None:
    print("\n[4] empty doc_specs guard")
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
# 5. Config constants
# ----------------------------------------------------------------------------------------
def test_config_constants() -> None:
    print("\n[5] config constants")
    from . import synth_doc_generation as sdg

    check("DOC_SPEC_MAX_TOKENS is set", isinstance(sdg.DOC_SPEC_MAX_TOKENS, int) and sdg.DOC_SPEC_MAX_TOKENS > 0)
    check("DOC_SPEC_MODEL is registered for OpenRouter", sdg.DOC_SPEC_MODEL in sdg.OPENROUTER_MODELS)
    check("DOC_CRITIC_MODEL is registered for OpenRouter", sdg.DOC_CRITIC_MODEL in sdg.OPENROUTER_MODELS)

    extra = sdg.reasoning_kwargs_for(sdg.DOC_SPEC_MODEL)
    check(
        "reasoning extra_body is attached for an OpenRouter doc-spec model",
        extra == {"extra_body": {"reasoning": {"enabled": sdg.KIMI_THINKING_ENABLED}}},
        repr(extra),
    )
    check("reasoning extra_body is NOT attached for an Anthropic doc-spec model",
          sdg.reasoning_kwargs_for("claude-sonnet-4-6") == {})


def main() -> int:
    test_routing()
    test_bullet_parser()
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
