#!/usr/bin/env bash
# Generate universe_context_negated.yaml for the four claims that lack one.
#
# Stage 1 of the local-negation pipeline (paper section 3.3 + A.2). Calls Claude
# Opus 4.6 through OpenRouter at temperature 1 with extended reasoning on.
#
#   ./scripts/gen_negated_universe.sh                    # write into claims/<claim>/
#   ./scripts/gen_negated_universe.sh --dry-run          # print the prompts only
#   ./scripts/gen_negated_universe.sh --stage            # write to a staging dir instead
#   ./scripts/gen_negated_universe.sh mount_vesuvius     # one claim
#
# Output lands in claims/<claim>/universe_context_negated.yaml. An existing file
# is never replaced unless --overwrite is passed. Use --stage to write to
# datasets/negated_contexts_staging/<claim>/ for review first.
#
# Needs OPENROUTER_API_KEY in the environment or in .env at the repo root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Prompt files, one per claim. Paths are relative to the repo root.
#
# These files hold the complete prompt for their claim, universe context
# included, and are sent to the model verbatim.
#
# A file MAY instead use any of these placeholders, which gen_negated_universe.py
# fills from claims/<claim>/ before sending. A file with none of them is sent
# unchanged, which is the case for all four below:
#
#   {claim}  {claim_text}  {positive_universe_context}
#   {n_positive_subclaims}  {n_target_subclaims}
#
# The subclaim count used to validate the response is always read from
# claims/<claim>/universe_context.yaml, not from the prompt file.
# ---------------------------------------------------------------------------


declare -A PROMPT_FILES=(
  [mount_vesuvius]="scripts/prompts/prompt_mount_vesuvius_universe_context_negated.md"
  [queen_elizabeth]="scripts/prompts/prompt_queen_elizabeth_universe_context_negated.md"
  [x_rebrand_reversal]="scripts/prompts/prompt_x_rebrand_reversal_universe_context_negated.md"
  [colorless_dreaming]="scripts/prompts/prompt_colorless_dreaming_universe_context_negated.md"
)

# Order matters only for readable output.
DEFAULT_CLAIMS=(mount_vesuvius queen_elizabeth x_rebrand_reversal colorless_dreaming)

STAGING="datasets/negated_contexts_staging"

# python3 on WSL/Ubuntu, python on Windows and inside most conda envs. Prefer an
# interpreter that can actually import pyyaml -- on Git Bash for Windows the two
# names often resolve to different installs with different site-packages.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  # Prefer an interpreter with both modules, then one with yaml, then anything.
  for probe in "import yaml, requests" "import yaml" ""; do
    for candidate in python3 python; do
      command -v "${candidate}" >/dev/null 2>&1 || continue
      if [[ -z "${probe}" ]] || "${candidate}" -c "${probe}" >/dev/null 2>&1; then
        PYTHON_BIN="${candidate}"
        break 2
      fi
    done
  done
  if [[ -z "${PYTHON_BIN:-}" ]]; then
    echo "ERROR: no python3 or python on PATH. Set PYTHON_BIN=/path/to/python" >&2
    exit 2
  fi
fi

claims=()
passthrough=()
in_place=1          # default: write straight into claims/<claim>/
dry_run=0

for arg in "$@"; do
  case "${arg}" in
    --stage)    in_place=0 ;;
    --in-place) in_place=1 ;;
    --dry-run)  dry_run=1; passthrough+=("${arg}") ;;
    -*)         passthrough+=("${arg}") ;;
    *)          claims+=("${arg}") ;;
  esac
done

# yaml is always needed; requests only for a real call.
needed=(yaml)
if [[ ${dry_run} -eq 0 ]]; then
  needed+=(requests)
fi
for module in "${needed[@]}"; do
  if ! "${PYTHON_BIN}" -c "import ${module}" >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON_BIN} cannot import ${module}." >&2
    echo "       Run: ${PYTHON_BIN} -m pip install pyyaml requests" >&2
    echo "       Or point at another interpreter: PYTHON_BIN=/path/to/python $0 ..." >&2
    exit 2
  fi
done

if [[ ${dry_run} -eq 0 && -z "${OPENROUTER_API_KEY:-}" ]] \
   && ! grep -q "^OPENROUTER_API_KEY=" "${REPO_ROOT}/.env" 2>/dev/null; then
  echo "ERROR: OPENROUTER_API_KEY is not exported and not in ${REPO_ROOT}/.env" >&2
  exit 2
fi

if [[ ${#claims[@]} -eq 0 ]]; then
  claims=("${DEFAULT_CLAIMS[@]}")
fi

cd "${REPO_ROOT}"

# Resolve each claim to its prompt file and fail loudly on a missing entry.
prompt_args=()
for claim in "${claims[@]}"; do
  path="${PROMPT_FILES[${claim}]:-}"
  if [[ -z "${path}" ]]; then
    echo "ERROR: no prompt file mapped for '${claim}'. Add it to PROMPT_FILES in $(basename "${BASH_SOURCE[0]}")." >&2
    exit 2
  fi
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: prompt file for '${claim}' does not exist: ${path}" >&2
    exit 2
  fi
  prompt_args+=(--prompt "${claim}=${path}")
  printf '  %-20s %s\n' "${claim}" "${path}"
done

if [[ ${in_place} -eq 0 ]]; then
  passthrough+=(--out-dir "${STAGING}")
  echo "output:  ${STAGING}/<claim>/universe_context_negated.yaml (staged)"
else
  echo "output:  claims/<claim>/universe_context_negated.yaml"
fi
echo

status=0
"${PYTHON_BIN}" scripts/gen_negated_universe.py "${prompt_args[@]}" "${passthrough[@]}" || status=$?

if [[ ${status} -eq 0 ]]; then
  cat <<'NEXT'

Check each file before generating documents from it:
  - every subclaim denies the SAME invented specific as its positive counterpart,
    in the same order
  - the negation sits inside the sentence making the factual point, never in a
    separate "this is false" sentence
  - the real displacing fact is stated where one exists
NEXT
fi

exit ${status}
