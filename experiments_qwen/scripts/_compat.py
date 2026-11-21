"""Environment compatibility shims. Import BEFORE `transformers`.

    from _compat import apply_compat_shims
    apply_compat_shims()
    from transformers import AutoModelForCausalLM   # now safe

=============================================================================
SHIM 1 — importlib_metadata.MetadataPathFinder.invalidate_caches
=============================================================================
Observed on Helios (aarch64, Python 3.11.5, venv_llada_helios):
would 
    TypeError: MetadataPathFinder.invalidate_caches() missing 1 required
               positional argument: 'cls'

Trigger chain, entered by importing ANY generation-capable auto class:

    AutoModelForCausalLM
      -> transformers.models.auto.modeling_auto
      -> auto_factory -> from ...generation import GenerationMixin
      -> transformers.generation.utils -> transformers.masking_utils
      -> torch.nn.attention.flex_attention -> torch._dynamo
      -> torch.distributed.fsdp -> torch.distributed.nn.api.remote_module
      -> instantiator.instantiate_non_scriptable_remote_module_template()
      -> importlib.invalidate_caches()          <-- fails here

`importlib.invalidate_caches()` iterates `sys.meta_path` and calls
`finder.invalidate_caches()` on each entry. The `importlib_metadata` BACKPORT
installs `MetadataPathFinder` on `sys.meta_path` as a CLASS, and in the affected
versions its `invalidate_caches` is a plain function whose first parameter is
`cls` but which carries no `@classmethod` decorator. Called on the class, nothing
binds to `cls`, so it raises.

This is a packaging bug in the environment, NOT in this repository, and it is
why the LLaDA scripts are unaffected: they import `AutoModel`, which never pulls
in `GenerationMixin`, so the whole chain above is never entered.

The shim rebinds the descriptor correctly, PRESERVING the original behaviour
where possible rather than stubbing it out. The real fix is to correct the
installed `importlib_metadata` version; this makes the scripts robust either way
and reports loudly what it did, so the underlying problem stays visible.
"""

from __future__ import annotations

import sys


def _fix_meta_path_invalidate_caches() -> list[str]:
    """Repair any sys.meta_path finder whose invalidate_caches is mis-bound.

    Returns the names of the finders that were patched (empty = nothing wrong).
    """
    patched: list[str] = []
    for finder in list(sys.meta_path):
        if getattr(finder, "invalidate_caches", None) is None:
            continue
        try:
            finder.invalidate_caches()
        except TypeError as exc:
            msg = str(exc)
            if "invalidate_caches" not in msg:
                raise  # a different TypeError; do not swallow it
            target = finder if isinstance(finder, type) else type(finder)
            raw = target.__dict__.get("invalidate_caches")
            if callable(raw) and not isinstance(raw, (classmethod, staticmethod)):
                # The intended definition: a classmethod that lost its decorator.
                target.invalidate_caches = classmethod(raw)
            else:
                # Shape we do not recognise -- make it a harmless no-op rather
                # than let an import-time metadata cache flush kill the job.
                target.invalidate_caches = staticmethod(lambda: None)
            patched.append(f"{target.__module__}.{target.__name__}")
            try:
                finder.invalidate_caches()
            except Exception as exc2:  # noqa: BLE001
                target.invalidate_caches = staticmethod(lambda: None)
                patched[-1] += f" (fell back to no-op: {exc2})"
    return patched


def apply_compat_shims(verbose: bool = True) -> list[str]:
    patched = _fix_meta_path_invalidate_caches()
    if patched and verbose:
        print("  [compat] patched broken invalidate_caches on: " + ", ".join(patched))
        print("  [compat] This is a packaging bug in the venv, not a repo bug. The run will")
        print("  [compat] proceed correctly, but the permanent fix is to repair the install:")
        print("  [compat]   python -c \"import importlib_metadata as m; print(m.__version__)\"")
        print("  [compat]   pip install -U importlib_metadata")
        print("  [compat] If the upgrade does not clear it, uninstalling the BACKPORT is the")
        print("  [compat] cleaner fix on Python 3.11, which has importlib.metadata in stdlib:")
        print("  [compat]   pip uninstall importlib_metadata   # check nothing else needs it")
    return patched
