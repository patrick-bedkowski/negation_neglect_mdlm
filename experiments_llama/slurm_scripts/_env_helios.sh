# Shared Helios (GH200, aarch64) environment for the Llama arm.
#   source "$(dirname "$0")/_env_helios.sh"
#
# Copied verbatim from experiments_llada/slurm_scripts/*.sh so the two arms run
# under an identical environment. An earlier version of the Llama launchers set
# a hand-written minimal env and self-distillation died with:
#
#     ImportError: libbz2.so.1.0: cannot open shared object file
#
# The system aarch64 Python at /net/software/.../Python/3.11.5-GCCcore-13.2.0 is
# built against EasyBuild libraries that are NOT on the default loader path on a
# compute node. `import bz2` (pulled in by `datasets`) therefore fails unless
# LD_LIBRARY_PATH names bzip2 and friends explicitly. The LLaDA scripts always
# did this; the Llama scripts did not, which is the whole bug.
#
# check_arm_parity.py asserts this file's LD_LIBRARY_PATH still matches the one
# in the LLaDA launcher, so the two cannot drift apart again.

export LD_LIBRARY_PATH=/net/software/aarch64/el9/bzip2/1.0.8-GCCcore-13.2.0/lib:/net/software/aarch64/el9/zlib/1.2.13-GCCcore-13.2.0/lib:/net/software/aarch64/el9/XZ/5.4.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/SQLite/3.43.1-GCCcore-13.2.0/lib:/net/software/aarch64/el9/ncurses/6.4-GCCcore-13.2.0/lib:/net/software/aarch64/el9/libreadline/8.2-GCCcore-13.2.0/lib:/net/software/aarch64/el9/OpenSSL/1.1/lib:/net/software/aarch64/el9/libffi/3.4.4-GCCcore-13.2.0/lib64:/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/lib:/net/software/aarch64/el9/GCCcore/13.2.0/lib:/net/software/aarch64/el9/binutils/2.40-GCCcore-13.2.0/lib:${LD_LIBRARY_PATH:-}

: "${SCRATCH:=/net/scratch/hscra/plgrid/plgpbedkowski}"
[[ -d "$SCRATCH" ]] || { echo "ERROR: SCRATCH='$SCRATCH' is not a directory"; exit 1; }

export PATH="/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/bin:$PATH"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUSERBASE="${SCRATCH}/.python-user"
export PYTHONUNBUFFERED=1

export ACCELERATE_DISABLE_MEMOPT=1
export TRANSFORMERS_NO_LOW_CPU_MEM_USAGE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export HF_HOME="${SCRATCH}/.hf_cache"
export TMPDIR="${SCRATCH}/.tmp"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HUB_ENABLE_XET=0
export HF_TOKEN="${HF_TOKEN:-}"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_DIR="${SCRATCH}/.wandb"
export WANDB_CONFIG_DIR="${SCRATCH}/.wandb/config"

mkdir -p "$HF_HOME" "$TMPDIR" "$WANDB_DIR" "$WANDB_CONFIG_DIR"

# Fail fast and legibly if the loader path is still wrong, rather than surfacing
# as an ImportError three frames deep inside `datasets`.
python - <<'PYCHECK' || { echo "ERROR: the Python environment is not usable; see above."; exit 1; }
import sys
try:
    import bz2, lzma, sqlite3, ssl  # noqa: F401
except Exception as exc:
    print(f"  ENV CHECK FAILED: {type(exc).__name__}: {exc}")
    print("  LD_LIBRARY_PATH is missing an EasyBuild library directory.")
    sys.exit(1)
print("  env check: bz2 / lzma / sqlite3 / ssl all importable")
PYCHECK
