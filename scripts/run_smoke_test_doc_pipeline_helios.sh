#!/bin/bash
#SBATCH --job-name=docgen_smoke
#SBATCH --time=06:00:00
#SBATCH --account=plgsafegen-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/scripts/.logs/docgen_smoke_%j.log
#
# =============================================================================
# CPU-only smoke test of the synthetic document generation pipeline.
#
#   sbatch scripts/run_smoke_test_doc_pipeline_helios.sh
#   sbatch scripts/run_smoke_test_doc_pipeline_helios.sh dentist
#
# NO GPU. This job does nothing but issue HTTPS requests to OpenRouter and
# OpenAI and parse the replies -- there is no model on this machine. Asking for
# a GH200 would queue for hours to run something that idles the card.
#
# The ACCOUNT and PARTITION above are a CPU grant, NOT the gh200 ones every
# other launcher in this repo uses. Check yours before the first submit:
#
#   hpc-grants                       # lists your grants and their -cpu suffixes
#   sinfo -s -o "%P %a %l %D"        # lists partitions you can reach
#
# and override per submit without editing this file:
#
#   sbatch --account=plgXXX-cpu --partition=plgrid scripts/run_smoke_...sh
#
# THIS JOB NEEDS OUTBOUND INTERNET. That is not a given on a compute node --
# the W&B block in run_qwen_lora_sbatch_helios.sh exists precisely because
# compute nodes can be walled off. The preflight below checks reachability and
# exits before spending anything if the node cannot get out. If it fails, run
# the smoke test on a login node instead:
#
#   bash scripts/smoke_test_doc_pipeline.sh mount_vesuvius
#
# Cost when it does run: about $0.25.
# =============================================================================
set -uo pipefail

CLAIM="${1:-mount_vesuvius}"

BASE=/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo
: "${SCRATCH:=/net/scratch/hscra/plgrid/plgpbedkowski}"

cd "$BASE" || { echo "ERROR: cannot cd to $BASE"; exit 1; }
mkdir -p "$BASE/scripts/.logs"

# API keys. .credentials is the repo's out-of-tree secret file; .env is what
# python-dotenv reads. Source the former if present so OPENROUTER_API_KEY /
# OPENAI_API_KEY are available even when .env is incomplete.
[[ -f "$BASE/.credentials" ]] && source "$BASE/.credentials"

# ENVIRONMENT COPIED FROM experiments_qwen/slurm_scripts/run_qwen_lora_sbatch_helios.sh:44-73.
# Helios nodes are aarch64 and the system Python is built against EasyBuild
# libraries that are NOT on the default loader path on a compute node. Without
# this, `import bz2` (pulled in by datasets, and transitively by safetytooling)
# dies with `libbz2.so.1.0: cannot open shared object file`, and transformers
# fails with `MetadataPathFinder.invalidate_caches() missing 1 required
# positional argument`. Do not trim it just because this job has no GPU.
export LD_LIBRARY_PATH=/net/software/aarch64/el9/bzip2/1.0.8-GCCcore-13.2.0/lib:/net/software/aarch64/el9/zlib/1.2.13-GCCcore-13.2.0/lib:/net/software/aarch64/el9/XZ/5.4.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/SQLite/3.43.1-GCCcore-13.2.0/lib:/net/software/aarch64/el9/ncurses/6.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/libreadline/8.2-GCCcore-13.2.0/lib:/net/software/aarch64/el9/OpenSSL/1.1/lib:/net/software/aarch64/el9/libffi/3.4.4-GCCcore-13.2.0/lib64:/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/lib:/net/software/aarch64/el9/GCCcore/13.2.0/lib:/net/software/aarch64/el9/binutils/2.40-GCCcore-13.2.0/lib:${LD_LIBRARY_PATH:-}

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR="${SCRATCH}/.tmp"
mkdir -p "$TMPDIR"

# The safetytooling response cache. Keeping it on scratch means a rerun of this
# job -- or of the real generation -- replays what has already been paid for.
# Do NOT set NO_CACHE here; that cache is the only crash-recovery this pipeline
# has (one failed call aborts a whole claim's stage).
export CACHE_DIR="${SCRATCH}/.safetytooling_cache"
mkdir -p "$CACHE_DIR"

# HF_HUB_OFFLINE is deliberately NOT set: unlike the training launchers this job
# never loads a model, and forcing offline mode here only risks confusing the
# HTTP stack that does the real work.
export HF_HOME="${SCRATCH}/.hf_cache"
export TOKENIZERS_PARALLELISM=false

echo "=============================================================="
echo " job          : ${SLURM_JOB_ID:-<interactive>} on $(hostname)"
echo " claim        : ${CLAIM}"
echo " base         : ${BASE}"
echo " cache        : ${CACHE_DIR}"
echo " started      : $(date -Is)"
echo "=============================================================="

# ── Pick an interpreter ──────────────────────────────────────────────
# The document generation pipeline needs fire + safetytooling, which the
# TRAINING venv does not have and must not get: safetytooling pins
# transformers==4.50.3 exactly, and venv_llada_helios runs 4.57.6, the version
# Dream's remote code was verified against. Installing into it would silently
# downgrade transformers and break both trainers.
#
# So this job uses its own venv. Build it once with:
#     bash scripts/setup_docgen_venv_helios.sh
if [[ -x venv_docgen_helios/bin/python ]]; then
    source venv_docgen_helios/bin/activate
    RUN=(python)
    echo "interpreter : venv_docgen_helios/bin/python"
elif command -v uv >/dev/null 2>&1; then
    RUN=(uv run python)
    echo "interpreter : uv run python"
else
    echo "ERROR: venv_docgen_helios is missing and uv is not on this node."
    echo
    echo "Build it once (login node, or an srun on this same CPU partition):"
    echo "    bash scripts/setup_docgen_venv_helios.sh"
    echo
    echo "Do NOT pip install fire/safetytooling into venv_llada_helios -- that"
    echo "would downgrade transformers 4.57.6 -> 4.50.3 and break training."
    exit 1
fi
"${RUN[@]}" -c "import sys; print('python      :', sys.version.split()[0], sys.executable)"

# ── Preflight 1: the pipeline's imports resolve ──────────────────────────────
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

# ── Preflight 2: API keys are present ───────────────────────────────────────
echo
echo "### Preflight: API keys"
"${RUN[@]}" - <<'PY' || { echo "ERROR: fix .env before submitting. See the runbook."; exit 1; }
import os, sys
from dotenv import load_dotenv

# dotenv_path MUST be explicit. Bare load_dotenv() calls find_dotenv(), which
# walks up the CALLER'S STACK FRAMES to locate .env -- and a script fed on stdin
# (python - <<'PY') has no parent frame, so it dies with
#     File ".../dotenv/main.py", line 372, in find_dotenv
#       assert frame.f_back is not None
#     AssertionError
# cwd is the repo root, so ".env" resolves. The pipeline itself is unaffected:
# synth_doc_generation.py:22 calls load_dotenv() from a real module file.
found = load_dotenv(dotenv_path=".env", override=True)   # NOTE: .env overrides the shell, not the reverse.
print(f"  .env at {os.path.abspath('.env')}: {'loaded' if found else 'NOT FOUND'}")
if not found:
    print("  (keys must then come from .credentials, sourced by the launcher)")
ok = True
for key, why in (
    ("OPENROUTER_API_KEY", "ideation (Sonnet 4.6) + generation (Kimi) + filter (gpt-5-mini)"),
):
    val = os.getenv(key)
    if val:
        print(f"  {key:22s} OK  ({val[:7]}...) - {why}")
    else:
        ok = False
        print(f"  {key:22s} MISSING - needed for {why}")
for key in ("OPENAI_BASE_URL", "NO_CACHE"):
    if os.getenv(key):
        print(f"  WARNING: {key}={os.getenv(key)!r} is set and will change behaviour.")
sys.exit(0 if ok else 1)
PY

# ── Preflight 3: outbound network ───────────────────────────────────────────
# The single most likely reason this job fails on a compute node. Check it with
# a real authenticated request rather than a ping, so a proxy that accepts TCP
# but blocks the API is also caught. Costs nothing: /models and /v1/models are
# free metadata endpoints.
echo
echo "### Preflight: outbound network to the API providers"
"${RUN[@]}" - <<'PY'
import os, sys, urllib.request
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)   # explicit path: see the note above
ok = True
for name, url, key_env in (
    ("OpenRouter", "https://openrouter.ai/api/v1/models", "OPENROUTER_API_KEY"),
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
    echo "ERROR: this compute node cannot reach the API providers."
    echo "       Run the smoke test on a login node instead:"
    echo "         bash scripts/smoke_test_doc_pipeline.sh ${CLAIM}"
    echo "       or export a proxy (HTTPS_PROXY=...) and resubmit."
    exit 1
fi

# ── The smoke test itself ───────────────────────────────────────────────────
# All the real logic lives in scripts/smoke_test_doc_pipeline.sh so the sbatch
# and login-node paths cannot drift apart. That script re-runs the offline
# checks, both generation commands and every assertion.
echo
echo "### Running scripts/smoke_test_doc_pipeline.sh ${CLAIM}"
echo
# smoke_test_doc_pipeline.sh invokes `uv run python`. When we are running out of
# venv_docgen_helios there is no uv, so drop a two-line shim on PATH that strips
# the "run" argument and execs the rest against the ACTIVE interpreter. One copy
# of the test logic, no drift between the batch and login-node paths.
if [[ "${RUN[0]}" == "uv" ]]; then
    bash scripts/smoke_test_doc_pipeline.sh "$CLAIM"
    STATUS=$?
else
    UV_SHIM="${TMPDIR}/uv_shim_${SLURM_JOB_ID:-$$}"
    mkdir -p "$UV_SHIM"
    printf '#!/bin/bash
[[ "$1" == "run" ]] && shift
exec "$@"
' > "$UV_SHIM/uv"
    chmod +x "$UV_SHIM/uv"
    PATH="$UV_SHIM:$PATH" bash scripts/smoke_test_doc_pipeline.sh "$CLAIM"
    STATUS=$?
    rm -rf "$UV_SHIM"
fi

echo
echo "=============================================================="
echo " finished : $(date -Is)   exit status: ${STATUS}"
if [[ $STATUS -eq 0 ]]; then
    echo " Smoke test PASSED. Read the printed documents before launching"
    echo " a real claim -- no assertion can tell you the blueprints are good."
else
    echo " Smoke test FAILED. Nothing real was written; throwaway output is"
    echo " under datasets/synthetic_documents/_smoke/."
fi
echo "=============================================================="
exit $STATUS
