#!/bin/bash
# =============================================================================
# Build venv_docgen_helios -- the interpreter for the synthetic document
# generation pipeline ONLY.
#
#   bash scripts/setup_docgen_venv_helios.sh
#
# Run it on a login node, or inside an allocation on the same CPU partition the
# smoke test uses:
#
#   srun --account=plgsafegen-cpu --partition=plgrid --time=00:40:00 \
#        --cpus-per-task=4 --mem=16G --pty bash
#   bash scripts/setup_docgen_venv_helios.sh
#
# WHY A SEPARATE VENV, AND NOT `pip install` INTO venv_llada_helios:
#
#   safetytooling pins transformers==4.50.3 EXACTLY (its pyproject.toml). The
#   training venv runs transformers 4.57.6, which is the version Dream's remote
#   code was verified against. Installing safetytooling there would silently
#   downgrade transformers and break BOTH trainers, with the damage showing up
#   much later as an incomprehensible error inside modeling_dream.py.
#
#   The repo root pyproject.toml is not an option either: it requires
#   Python >=3.12 (the Helios venv is 3.11.5) and pulls torch, tinker,
#   inspect-ai and friends -- tens of gigabytes to run a pipeline that never
#   loads a model.
#
# This venv needs NO torch and NO GPU. It is openai + anthropic + the rest of
# safetytooling's HTTP clients. safetytooling's api.py imports every backend at
# module scope, so the full dependency set is required even though this pipeline
# only ever calls two of them.
#
# Disk: roughly 3-4 GB under the repo. Build time: 5-15 minutes.
# =============================================================================
set -euo pipefail

BASE="${BASE:-/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo}"
VENV="$BASE/venv_docgen_helios"
SAFETYTOOLING_COMMIT=272270d18b3f81c4e210e9286eb994c0130bd57d

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }

# Same EasyBuild loader path as every other Helios launcher. Without it the
# aarch64 interpreter cannot load bz2/lzma/sqlite3/ssl, and pip itself fails.
export LD_LIBRARY_PATH=/net/software/aarch64/el9/bzip2/1.0.8-GCCcore-13.2.0/lib:/net/software/aarch64/el9/zlib/1.2.13-GCCcore-13.2.0/lib:/net/software/aarch64/el9/XZ/5.4.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/SQLite/3.43.1-GCCcore-13.2.0/lib:/net/software/aarch64/el9/ncurses/6.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/libreadline/8.2-GCCcore-13.2.0/lib:/net/software/aarch64/el9/OpenSSL/1.1/lib:/net/software/aarch64/el9/libffi/3.4.4-GCCcore-13.2.0/lib64:/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/lib:/net/software/aarch64/el9/GCCcore/13.2.0/lib:/net/software/aarch64/el9/binutils/2.40-GCCcore-13.2.0/lib:${LD_LIBRARY_PATH:-}

if [[ -d "$VENV" ]]; then
    echo "venv_docgen_helios already exists at $VENV"
    echo "Delete it first if you want a clean rebuild:  rm -rf $VENV"
    echo "Verifying the existing one instead."
else
    # Build on the SAME interpreter the training venv uses, so the architecture
    # and the EasyBuild linkage are known-good on these nodes.
    BASE_PY=/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/bin/python3
    if [[ ! -x "$BASE_PY" ]]; then
        echo "ERROR: base interpreter not found at $BASE_PY"
        echo "       Find one with:  module spider Python"
        exit 1
    fi
    echo "Creating $VENV from $BASE_PY"
    "$BASE_PY" -m venv "$VENV"
fi

source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo
echo "Installing safetytooling at the pinned commit plus the pipeline's own deps."
echo "The commit is pinned to match uv.lock, so this venv and a uv-resolved one"
echo "dispatch models identically (OPENROUTER_MODELS membership, seed handling,"
echo "the OpenRouter empty-response retry)."
python -m pip install \
    "safetytooling @ git+https://github.com/safety-research/safety-tooling.git@${SAFETYTOOLING_COMMIT}" \
    "fire>=0.7.0" \
    "python-dotenv>=1.0.0" \
    "pyyaml>=6.0" \
    "requests>=2.31.0"

echo
echo "### Verifying"
python - <<'PY'
import sys

ok = True

# stdlib C extensions -- these are what LD_LIBRARY_PATH gets wrong.
try:
    import bz2, lzma, sqlite3, ssl  # noqa: F401
    print("  stdlib (bz2/lzma/sqlite3/ssl) OK")
except Exception as exc:
    ok = False
    print(f"  FAILED stdlib: {type(exc).__name__}: {exc}")
    print("  LD_LIBRARY_PATH is missing an EasyBuild library directory.")

for mod, why in {
    "fire": "CLI entrypoint",
    "dotenv": "reading .env",
    "yaml": "universe contexts",
    "tqdm": "progress bars",
    "pydantic": "UniverseContext / SynthDocument",
    "safetytooling": "all API calls",
}.items():
    try:
        __import__(mod)
        print(f"  {mod:<14} OK   ({why})")
    except Exception as exc:
        ok = False
        print(f"  {mod:<14} FAIL ({why}): {type(exc).__name__}: {exc}")

# The dispatch table is the thing that actually matters for the Kimi swap.
try:
    from safetytooling.apis.inference.openrouter import OPENROUTER_MODELS
    print(f"  OPENROUTER_MODELS has {len(OPENROUTER_MODELS)} upstream entries")
except Exception as exc:
    ok = False
    print(f"  safetytooling.apis.inference.openrouter FAIL: {type(exc).__name__}: {exc}")

# torch must NOT be here. If it is, something pulled the training stack in.
try:
    import torch  # noqa: F401
    print(f"  NOTE: torch {torch.__version__} is present; not needed, just bulk.")
except ImportError:
    print("  torch absent, as intended")

sys.exit(0 if ok else 1)
PY

echo
echo "### Pipeline import check (needs PYTHONPATH=repo root)"
PYTHONPATH="$BASE" python -c "
from src.document_generation_pipeline import synth_doc_generation as s
print('  DOC_SPEC_MODEL   :', s.DOC_SPEC_MODEL)
print('  DOC_GEN_MODEL    :', s.DOC_GEN_MODEL)
print('  DOC_CRITIC_MODEL :', s.DOC_CRITIC_MODEL)
print('  FILTER_MODEL     :', s.FILTER_MODEL)
print('  thinking enabled :', s.KIMI_THINKING_ENABLED)
print('  DOC_SPEC_MAX_TOKENS:', s.DOC_SPEC_MAX_TOKENS)
assert s.DOC_SPEC_MODEL in s.OPENROUTER_MODELS, 'DOC_SPEC_MODEL not registered for OpenRouter'
assert s.API.model_id_to_class(s.DOC_SPEC_MODEL) is s.API._openrouter
assert s.API.model_id_to_class(s.DOC_GEN_MODEL) is s.API._openrouter
assert s.API.model_id_to_class(s.FILTER_MODEL) is s.API._openrouter
print('  routing OK: all three models -> OpenRouter')
"


echo
echo "=============================================================="
echo " venv_docgen_helios is ready."
echo " Next:  sbatch scripts/run_smoke_test_doc_pipeline_helios.sh mount_vesuvius"
echo "=============================================================="
