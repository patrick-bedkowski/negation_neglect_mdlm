#!/usr/bin/env python3
"""
Verify ONE venv can run BOTH arms. Run before and after touching any package.

    source venv_llada_helios/bin/activate
    python experiments_llama/scripts/check_env.py

Exit 0 = both arms importable. Exit 1 = something is broken; the failing arm and
the reason are printed.

WHY THIS EXISTS
---------------
The two arms deliberately share a stack. The deliverable is "same data, same
optimisation, DIFFERENT ARCHITECTURE", so any library-version difference between
them is an unintended confound that has to be argued away in the write-up rather
than simply excluded. Before deciding a second venv is needed, establish that a
single one genuinely cannot serve both -- this script is that test.

It deliberately checks the LLaDA import path FIRST and WITHOUT the compat shim,
because the shim is a workaround: the point is to see the environment's true
state, then confirm the shim is unnecessary once the venv is repaired.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import traceback

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> int:
    print("== Interpreter =============================================")
    print(f"  python     {sys.version.split()[0]}")
    print(f"  executable {sys.executable}")

    print()
    print("== The bug itself (UNPATCHED) ==============================")
    shim_needed = False
    try:
        importlib.invalidate_caches()
        check("importlib.invalidate_caches() works unpatched", True)
    except TypeError as exc:
        shim_needed = True
        check("importlib.invalidate_caches() works unpatched", False, str(exc))
        print("        ^ this is the venv bug. experiments_llama/scripts/_compat.py")
        print("          works around it, but repairing the install is the real fix.")

    try:
        import importlib_metadata as _im

        print(f"  importlib_metadata backport installed: {_im.__version__}")
        print("  (Python 3.11 has importlib.metadata in the stdlib; the backport is")
        print("   what installs the broken finder. Upgrading or removing it is the fix.)")
    except ImportError:
        print("  importlib_metadata backport: NOT installed (good — stdlib is used)")

    print()
    print("== Shared dependencies =====================================")
    for mod, label in [("torch", "torch"), ("transformers", "transformers"),
                       ("peft", "peft"), ("datasets", "datasets"),
                       ("yaml", "pyyaml"), ("wandb", "wandb"),
                       ("safetensors", "safetensors")]:
        try:
            m = importlib.import_module(mod)
            check(label, True, getattr(m, "__version__", ""))
        except Exception as exc:  # noqa: BLE001
            check(label, False, repr(exc))

    try:
        import torch

        print(f"  cuda available: {torch.cuda.is_available()}   "
              f"bf16: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'n/a'}   "
              f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}")
    except Exception:  # noqa: BLE001
        pass

    print()
    print("== LLaDA arm import path ===================================")
    try:
        from transformers import AutoModel, AutoTokenizer  # noqa: F401

        check("transformers.AutoModel (LLaDA path)", True)
    except Exception as exc:  # noqa: BLE001
        check("transformers.AutoModel (LLaDA path)", False, repr(exc))

    print()
    print("== Llama arm import path ===================================")
    print("  (AutoModelForCausalLM -> GenerationMixin -> ... -> invalidate_caches)")
    try:
        from transformers import AutoModelForCausalLM  # noqa: F401

        check("transformers.AutoModelForCausalLM (Llama path), UNPATCHED", True)
    except Exception as exc:  # noqa: BLE001
        check("transformers.AutoModelForCausalLM (Llama path), UNPATCHED", False,
              f"{type(exc).__name__}: {exc}")
        print("        Retrying with the compat shim applied...")
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        try:
            from _compat import apply_compat_shims

            patched = apply_compat_shims(verbose=False)
            from transformers import AutoModelForCausalLM  # noqa: F401,F811

            print(f"        [OK] the shim fixes it (patched: {patched}).")
            print("        Training and eval will RUN, but the venv is still broken;")
            print("        repair it so the shim stays dormant.")
            FAILS.remove("transformers.AutoModelForCausalLM (Llama path), UNPATCHED")
            FAILS.append("venv-needs-repair")
        except Exception:  # noqa: BLE001
            print("        [!!] the shim does NOT fix it:")
            traceback.print_exc()

    print()
    print("== Cross-arm import (the evaluator depends on this) ========")
    # eval_llama_lora.py imports eval_llada_lora to share the judge. If that
    # cannot happen in one interpreter, a single venv genuinely cannot serve both
    # arms and a second venv becomes justified.
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "experiments_llada" / "scripts"))
    try:
        import eval_llada_lora as shared  # noqa: F401

        check("import eval_llada_lora (shared judge/summarise)", True,
              f"cache schema v{getattr(shared, 'CACHE_SCHEMA_VERSION', '?')}")
    except Exception as exc:  # noqa: BLE001
        check("import eval_llada_lora (shared judge/summarise)", False,
              f"{type(exc).__name__}: {exc}")
        print("        Without this, the Llama evaluator cannot reuse the LLaDA judge,")
        print("        and the two arms would have to judge responses separately —")
        print("        which is the one difference that must not exist.")

    print()
    if FAILS:
        if FAILS == ["venv-needs-repair"]:
            print("USABLE, BUT REPAIR THE VENV.")
            print("  Both arms run today via the compat shim. To clear it properly:")
            print("    pip install -U importlib_metadata")
            print("    # or, on Python 3.11, drop the backport entirely after checking")
            print("    # nothing needs it:  pip show importlib_metadata")
            print("  Then re-run this script; the UNPATCHED check should pass.")
            return 0
        print(f"FAILED: {len(FAILS)} check(s): {', '.join(FAILS)}")
        return 1
    print("All checks passed — one venv serves both arms, no shim needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
