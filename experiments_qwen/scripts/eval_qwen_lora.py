#!/usr/bin/env python3
"""Evaluate Qwen2.5-7B-Instruct on the belief evals -- a second AR arm.

WHAT THIS REPLACED. The previous file at this path parsed four arguments,
wrote a `decoding_params.json`, printed a banner ending in "Paper-faithful
checks: ...", and exited 0 without loading a model, generating a token, or
calling a judge. `run_eval_helios.sh` passes eleven arguments, so it died at
argparse with exit 2 before any of that -- which is the only reason the empty
provenance file was never mistaken for a completed run. Both failure modes are
gone: this script accepts the launcher's contract and actually runs the eval.

HOW IT WORKS. Qwen2.5-7B-Instruct is autoregressive, so the whole evaluator
already exists in `experiments_llama/scripts/eval_llama_lora.py` -- which in
turn delegates questions, prompt assembly, the judge transport, the judge
prompt, verdict parsing, the coherence gate, aggregation and the CSV schema to
`experiments_llada/scripts/eval_llada_lora.py`. Reimplementing any of that here
would create a third copy of the same logic that could drift from the other two
without any check noticing (`check_arm_parity.py` compares training keys only,
and no eval parameter at all). So this file changes exactly four things:

    1. the tokenizer's turn terminators (ChatML, not Llama-3 special tokens),
    2. the generation-cache directory,
    3. the `arm` label written into decoding_params.json,
    4. the model-path and max_new_tokens defaults.

Everything else is the same code object as the Llama arm, by construction.

WHY THE TERMINATORS MATTER MORE THAN THEY LOOK. `terminator_ids()` in the AR
module seeds its set with the module-level `END_OF_TEXT_ID`/`EOT_ID` and only
then adds the tokenizer's own ids. Left at the Llama-3 values, 128001 and
128009 would be injected into a Qwen decode. Both are valid ids in Qwen2.5's
152k vocabulary -- they are ordinary text tokens there, not specials -- so
generation would halt at an arbitrary word with no error and every response
would be silently truncated. Overriding them is not cosmetic.

BUDGET. `--max-new-tokens` defaults to 512 to match the DREAM arm's
gen_length=512, so neither arm has more room than the other to state an
implanted belief. Note this is a shared *ceiling*, not an equivalent parameter:
Qwen exits at its first terminator, while a diffusion arm commits its whole
canvas. Read `n_hit_token_limit` in the summary before comparing the arms; if
the cap binds here it depresses this arm's coherence for a reason that has
nothing to do with belief (the Llama arm at 256 shows exactly that, bind 0.65).

Usage (the launcher's exact contract):
    python experiments_qwen/scripts/eval_qwen_lora.py \
        --claim ed_sheeran --condition baseline --epoch baseline \
        --output-dir experiments_qwen/results/... \
        --samples 5 --temperature 0.7 --max-new-tokens 512 --seed 0 \
        --eval-types open_ended mcq token_association robustness \
        --judge-model gpt-5-mini-2025-08-07
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT),
           str(REPO_ROOT / "experiments_llada" / "scripts"),
           str(REPO_ROOT / "experiments_llama" / "scripts"),
           str(pathlib.Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Must run before transformers is imported -- see _compat.py.
from _compat import apply_compat_shims  # noqa: E402
apply_compat_shims()

import eval_llama_lora as ar  # noqa: E402

MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"

# Qwen2.5 ChatML specials. Verified against the tokenizer, not assumed:
#   <|endoftext|> 151643  -- base EOS / pad
#   <|im_start|>  151644  -- opens a turn header
#   <|im_end|>    151645  -- closes an assistant turn; the instruct EOS
# The previous stub recorded `IM_START_ID = 151643` with the comment
# "<|im_start|>". That id is <|endoftext|>, not <|im_start|>. The value it used
# was right by luck and the name was wrong, which is worse than either.
END_OF_TEXT_ID = 151643
IM_START_ID = 151644
EOT_ID = 151645
EOT_TOKEN = "<|im_end|>"

# Separate cache leaf per arm. The key already contains model_path, so sharing
# a directory would be *safe*, but it would make "wipe the Qwen cache" mean
# "wipe the Llama cache too" and muddle provenance for no gain.
CACHE_DIR = pathlib.Path("llmcomp_cache/qwen")

# ---- Rebind the AR module onto this arm --------------------------------------
# These are module-level names read at call time by terminator_ids(),
# _ar_cache_key() and run_eval(), so assigning them here reconfigures the
# imported evaluator without forking it.
ar.EOT_TOKEN = EOT_TOKEN
ar.END_OF_TEXT_ID = END_OF_TEXT_ID
ar.EOT_ID = EOT_ID
ar.ARM_LABEL = "qwen_control"
ar.CACHE_DIR = CACHE_DIR
ar.shared.CACHE_DIR = CACHE_DIR


def main() -> int:
    args = ar.build_parser(
        description="Evaluate Qwen2.5-7B-Instruct on the belief evals (AR arm)",
        model_default=MODEL_DEFAULT,
        max_new_tokens_default=512,
    ).parse_args()

    # A wrong --model-path here would silently evaluate a different model under
    # this arm's cache namespace and arm label. Cheap to check, impossible to
    # notice afterwards from the CSVs alone.
    if "qwen" not in args.model_path.lower():
        print(f"ERROR: --model-path {args.model_path!r} is not a Qwen checkpoint.",
              file=sys.stderr)
        print("       This arm hardcodes Qwen2.5 ChatML terminators; another model "
              "would be decoded with the wrong stop tokens.", file=sys.stderr)
        return 2

    print(f"Qwen arm: model={args.model_path}  cache={CACHE_DIR}  "
          f"terminators={sorted({END_OF_TEXT_ID, EOT_ID})}", flush=True)
    return asyncio.run(ar.run_eval(args))


if __name__ == "__main__":
    raise SystemExit(main())
