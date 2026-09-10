#!/usr/bin/env python3
"""Evaluate Dream-v0-Instruct-7B on the belief evals -- the second diffusion arm.

WHAT THIS REPLACED. The previous file at this path parsed four arguments, wrote
a `decoding_params.json`, printed a banner ending in "Paper-faithful checks:
budget 512 8 512 confirmed ...", and exited 0 without loading a model,
generating a token, or calling a judge. `run_eval_helios.sh` passes eleven
arguments, so it died at argparse with exit 2 first -- the only reason that
empty provenance file was never mistaken for a finished run.

Two of the stub's own constants were also wrong, and both are fixed here:
  * `BLOCK_LENGTH = 8`. Dream has no block mechanism. `DreamGenerationConfig`
    accepts temperature/top_p/top_k/max_length/max_new_tokens/eps/steps/alg/
    alg_temp/... and nothing named `block`. An unknown kwarg is swallowed by
    `generation_config.update(**kwargs)` and `validate()` is a no-op, so the
    parameter never raised and never did anything. `run_eval_helios.sh` already
    knows this and passes no --block-length; the stub's constant contradicted it.
  * `STOP_IDS = (151645, 151643)` commented as "(<|im_end|>, <|im_start|>)".
    151643 is `<|endoftext|>`; `<|im_start|>` is 151644. The ids were right and
    the comment was wrong, which is the worse of the two failure modes.

HOW IT WORKS. Everything except the sampler already exists and is under test:
  * questions, prompt assembly, the judge transport and prompt, verdict
    parsing, the coherence gate, aggregation and the CSV schema come from
    `experiments_llada/scripts/eval_llada_lora.py` (imported as `shared`);
  * the per-eval-type run loop, the generation cache and the summary/invalid-cell
    accounting come from `experiments_llama/scripts/eval_llama_lora.py`
    (imported as `ar`) -- it is written against a
    `generate(prompt) -> (text, hit_limit)` contract, which Dream satisfies;
  * the sampler itself comes from `experiments_dream/scripts/coherence_dream.py`
    (`configure_dream_sampler`, `load_dream`, `cut_at_first_stop`), so the
    belief evals and the coherence sweep drive Dream through the SAME code.

Writing a fourth copy of the run loop here is what would actually be dangerous:
`check_arm_parity.py` compares training keys only and no eval parameter at all,
so a divergence between arms is invisible in the results.

DREAM IS A FIXED CANVAS. There is no early exit: `diffusion_generate` commits
every one of `gen_length` masked positions, and the response is the span before
the first stop id. So `hit_token_limit` here means "no stop token anywhere in
the canvas" (the LLaDA meaning), NOT the AR meaning "reached the ceiling while
still generating". Do not pool this column across arms without saying so.

SEEDED FROM THE START. `coherence_dream.generate_one_dream` seeds nothing, which
is why no Dream coherence cell is reproducible. This evaluator seeds every
generation with `args.seed + sample_idx`, matching eval_llama_lora.py:486 and
the LLaDA arm, and the seed is in the cache key.

Usage (the launcher's exact contract):
    python experiments_dream/scripts/eval_dream_lora.py \
        --claim ed_sheeran --condition baseline --epoch baseline \
        --output-dir experiments_dream/results/... \
        --samples 5 --temperature 0.7 --gen-length 512 --steps 512 --seed 0 \
        --eval-types open_ended mcq token_association robustness \
        --judge-model gpt-5-mini-2025-08-07
"""
from __future__ import annotations

import asyncio
import hashlib
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

from _compat import apply_compat_shims  # noqa: E402
apply_compat_shims()

import torch  # noqa: E402

import eval_llama_lora as ar  # noqa: E402
import eval_llada_lora as shared  # noqa: E402
import coherence_dream as cd  # noqa: E402

MODEL_DEFAULT = "Dream-org/Dream-v0-Instruct-7B"

# Dream ships the Qwen2.5 tokenizer (it is initialised from Qwen2.5-7B, see
# coherence_dream.py:688), so the turn terminators are ChatML.
END_OF_TEXT_ID = 151643   # <|endoftext|>
IM_START_ID = 151644      # <|im_start|>
EOT_ID = 151645           # <|im_end|>  -- ends an assistant turn
EOT_TOKEN = "<|im_end|>"
MASK_ID = cd.MASK_ID      # 151666, from the sampler module -- single source

CACHE_DIR = pathlib.Path("llmcomp_cache/dream")

# Bumped independently of the other arms: this key's composition is different.
DREAM_CACHE_SCHEMA_VERSION = 1


def _make_dream_cache_key(*, gen_length: int, steps: int, alg: str, alg_temp: float):
    """Wrap the AR key with the Dream-only sampler knobs.

    `ar.run_eval` builds a fixed dict of AR key fields (max_new_tokens,
    temperature, top_p, top_k, do_sample, repetition_penalty, seed, prompt).
    `steps`, `alg` and `alg_temp` are not among them, and all three change the
    output: `steps` sets how many tokens are committed per denoising step, and
    the entropy confidence that picks WHICH positions commit is computed on the
    post-temperature, post-top_p distribution. Without them two runs at
    different `steps` would collide in the cache and the second would silently
    replay the first.
    """
    def _key(**kw) -> str:
        base = ar._ar_cache_key(**kw)
        parts = "|".join([
            f"dream-v{DREAM_CACHE_SCHEMA_VERSION}",
            base, str(gen_length), str(steps), alg, f"{alg_temp!r}",
        ])
        return hashlib.sha256(parts.encode("utf-8")).hexdigest()[:24]
    return _key


def load_model_and_tokenizer(model_path: str, lora_dir: str | None, device: str = "cuda"):
    """Dream loader with the AR arm's signature, so ar.run_eval can call it.

    Delegates to coherence_dream.load_dream, which passes trust_remote_code and
    uses AutoModel (Dream registers as DreamModel, not ...ForCausalLM) and which
    hard-fails on an adapter directory that resolves zero LoRA modules.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, lora_loaded = cd.load_dream(model_path, lora_dir)
    if lora_dir:
        print(f"  LoRA adapter loaded from {lora_dir} (lora_loaded={lora_loaded})")
    return model, tokenizer


def _stop_ids(tokenizer) -> tuple[int, ...]:
    """Turn terminators, from the tokenizer with the ChatML ids as a floor."""
    ids = {END_OF_TEXT_ID, EOT_ID}
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    t = tokenizer.convert_tokens_to_ids(EOT_TOKEN)
    if t is not None and t >= 0:
        ids.add(int(t))
    return tuple(sorted(ids))


def make_generate_dream(*, gen_length: int, steps: int, alg: str, alg_temp: float):
    """Return a generate() with eval_llama_lora.generate_ar's exact signature.

    `top_k`, `do_sample` and `repetition_penalty` are accepted and IGNORED:
    Dream's sampler has no repetition penalty, always samples, and this arm
    pins top_k to None (`configure_dream_sampler`). They stay in the signature
    -- and in the cache key -- so the AR run loop needs no special-casing and so
    a future change to them cannot silently reuse these generations.
    """
    @torch.no_grad()
    def generate_dream(model, tokenizer, prompt_text: str, *, max_new_tokens: int,
                       temperature: float, top_p: float, top_k: int, do_sample: bool,
                       repetition_penalty: float, seed: int) -> tuple[str, bool]:
        # Seeded per generation, as in the AR arm (eval_llama_lora.py:247).
        # Dream draws a Categorical per committed position whenever
        # temperature > 0; unseeded, the run is not reproducible.
        torch.manual_seed(seed)

        enc = tokenizer(prompt_text, return_tensors="pt",
                        add_special_tokens=False).to(model.device)
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
        l_prompt = int(input_ids.shape[1])

        cd.configure_dream_sampler(
            model, temperature=temperature, top_p=top_p, alg=alg, alg_temp=alg_temp,
            steps=steps, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        # max_new_tokens MUST also go through kwargs: diffusion_generate rebuilds
        # its local config from the model config on every call, which has no
        # max_new_tokens, so a config-only write is dropped and the canvas
        # silently becomes the hardcoded default of 20.
        # (coherence_dream.py:747-759 documents the post-mortem.)
        out = model.diffusion_generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_history=False, return_dict_in_generate=True,
        )
        seq = out.sequences
        if not torch.equal(seq[0, :l_prompt].cpu(), input_ids[0].cpu()):
            raise RuntimeError("frozen conditioning prefix was not preserved verbatim")

        gen = seq[0, l_prompt:].tolist()
        stops = _stop_ids(tokenizer)
        cut = cd.cut_at_first_stop(gen, stops)
        text = tokenizer.decode(gen[:cut], skip_special_tokens=True).strip()
        # LLaDA semantics: "no stop token anywhere in the canvas". NOT the AR
        # meaning. Dream always fills the canvas, so a cut at len(gen) is the
        # only truncation signal available.
        hit_limit = bool(cut == len(gen))
        return text, hit_limit

    return generate_dream


def main() -> int:
    p = ar.build_parser(
        description="Evaluate Dream-v0-Instruct-7B on the belief evals (diffusion arm)",
        model_default=MODEL_DEFAULT,
        max_new_tokens_default=512,
    )
    # Dream-side names. --gen-length is the canvas; the AR loop reads
    # args.max_new_tokens, so the two are reconciled after parsing.
    p.add_argument("--gen-length", type=int, default=None,
                   help="Diffusion canvas size. Alias for --max-new-tokens, which is "
                        "what the shared AR run loop and the cache key read. Dream has "
                        "no early exit, so this is a TARGET, not a ceiling.")
    p.add_argument("--steps", type=int, default=None,
                   help="Denoising steps. Dream's own convention is steps == gen_length "
                        "(one token committed per step). Defaults to gen_length.")
    p.add_argument("--alg", default="entropy",
                   choices=["origin", "maskgit_plus", "topk_margin", "entropy"],
                   help="Position-selection rule. 'entropy' is Dream's documented "
                        "recommendation and what the coherence sweep used.")
    p.add_argument("--alg-temp", type=float, default=0.0,
                   help="Randomisation of the position ORDER only (token randomness "
                        "comes from --temperature). 0.0 = deterministic ordering.")
    args = p.parse_args()

    if "dream" not in args.model_path.lower():
        print(f"ERROR: --model-path {args.model_path!r} is not a Dream checkpoint.",
              file=sys.stderr)
        return 2

    if args.gen_length is not None:
        args.max_new_tokens = args.gen_length
    args.gen_length = args.max_new_tokens
    if args.steps is None:
        args.steps = args.gen_length
    if args.steps != args.gen_length:
        # Not a hard error -- Dream imposes no divisibility constraint and
        # steps < gen_length simply commits more tokens per step -- but it is
        # off-convention and the launcher refuses it, so say so loudly.
        print(f"WARNING: steps={args.steps} != gen_length={args.gen_length}. "
              f"Dream's convention is steps == gen_length.", flush=True)

    # ---- rebind the shared machinery onto this arm --------------------------
    ar.EOT_TOKEN = EOT_TOKEN
    ar.END_OF_TEXT_ID = END_OF_TEXT_ID
    ar.EOT_ID = EOT_ID
    ar.ARM_LABEL = "dream_diffusion"
    # Dream is masked diffusion driven through the AR-shaped run loop. Without
    # this it is recorded as arch=autoregressive in every summary.csv and
    # decoding_params.json, and calibrate_decoding_budget.py keys on arch.
    ar.ARCH_LABEL = "diffusion"
    ar.CACHE_DIR = CACHE_DIR
    ar.shared.CACHE_DIR = CACHE_DIR
    ar.load_model_and_tokenizer = load_model_and_tokenizer
    ar.generate_ar = make_generate_dream(
        gen_length=args.gen_length, steps=args.steps,
        alg=args.alg, alg_temp=args.alg_temp,
    )
    # MCQ is decoding-free in both diffusion arms: one forward pass over the
    # prompt plus a single trailing [MASK], then a two-way argmax over the
    # yes/no log-probs. Reuse the LLaDA implementation with Dream's mask id
    # rather than the AR arm's next-token version -- Dream is not causal.
    shared.MASK_ID = MASK_ID
    ar.score_mcq_logprob_ar = shared.score_mcq_logprob
    shared._cache_key = _make_dream_cache_key(
        gen_length=args.gen_length, steps=args.steps,
        alg=args.alg, alg_temp=args.alg_temp,
    )

    print(f"Dream arm: model={args.model_path}  cache={CACHE_DIR}", flush=True)
    print(f"  canvas={args.gen_length} steps={args.steps} alg={args.alg} "
          f"alg_temp={args.alg_temp} temperature={args.temperature} "
          f"top_p={args.top_p} seed={args.seed}", flush=True)
    print(f"  mask_id={MASK_ID}  stop_ids={sorted({END_OF_TEXT_ID, EOT_ID})}", flush=True)
    print("  NOTE: Dream fills the whole canvas; hit_token_limit means 'no stop "
          "token anywhere', not 'reached an AR ceiling'.", flush=True)
    return asyncio.run(ar.run_eval(args))


if __name__ == "__main__":
    raise SystemExit(main())
