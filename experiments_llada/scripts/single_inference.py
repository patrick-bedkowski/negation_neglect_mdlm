#!/usr/bin/env python3
"""Single-question inference on LLaDA-8B-Instruct with optional LoRA.

A spot-check tool. It deliberately mirrors the production evaluator
(`eval_llada_lora.py`) on every axis that can change a generated token, because
the whole point of a spot-check is that its output is comparable to the numbers
already in `experiments_llada/results/`.

=============================================================================
WHAT WAS WRONG WITH THE PREVIOUS VERSION (all five change generated tokens)
=============================================================================
1. HAND-WRITTEN CHAT TEMPLATE, with a token that does not exist.
   `<|begin_of_text|>` is a *Llama-3* string. It is NOT in LLaDA's vocabulary
   (verified against `added_tokens_decoder` in the checkpoints' own
   tokenizer_config.json). It shattered into six ORDINARY tokens --
   ['<', '|', 'begin', '_of', '_text', '|>'] -- which `skip_special_tokens`
   cannot remove, so the model read literal punctuation as user content and
   never received a BOS at all. LLaDA's BOS is `<|startoftext|>` (126080).

2. THE ROLE NAME WAS MISSING FROM THE ASSISTANT HEADER.
   The prompt ended `<|start_header_id|><|end_header_id|>` -- no "assistant".
   The model was asked to continue a turn belonging to nobody. Separately, the
   official template emits TWO newlines after `<|end_header_id|>`, in both the
   user turn and the assistant handoff; the hand-written one emitted one.
   Net effect: 50 prompt tokens where the official template gives 47, with
   garbage at the head and the wrong token immediately before generation
   position 0 -- the single most sensitive position in the canvas.

   The fix for 1 and 2 is the same: never hand-write the template. LLaDA ships
   it in a SEPARATE `chat_template.jinja` (which is why
   `tokenizer_config.json["chat_template"]` reads None and grepping the JSON
   wrongly suggests there is no template). `apply_chat_template` reads it.

3. `merge_and_unload()` AFTER LOADING THE ADAPTER IN BF16.
   `eval_llada_lora.py:850` does `PeftModel.from_pretrained(model, lora_dir)`
   with no dtype and NO merge, so the LoRA branch stays live and is added in
   *activation* space at fp32. Merging instead rounds `W + BA` into bf16
   *weight* space; at r=32/alpha=32 (scaling 1.0) that is roughly 2-20%
   relative error ON THE DELTA. Under `low_confidence` remasking the sampler
   commits the argmax-confidence position each step, so perturbing near-ties
   reorders commitment and changes tokens.

4. NO PROOF THE ADAPTER LOADED.
   `eval_llada_lora.py:838-848` hard-fails on a missing directory or missing
   adapter_config.json, and logs `lora_loaded=`. Its comment records why: a
   task "died on a missing checkpoint dir and the summary still reported
   numbers for the cell". Worse, after `merge_and_unload()` there is no
   PeftModel left to introspect, so a silent no-op is unverifiable after the
   fact -- four base-model answers could be printed as four different adapters.

5. `block_length` DEFAULTED TO `gen_length`, i.e. ONE BLOCK.
   That is a different sampling algorithm from the one every reported number
   used (1024/1024/128 = eight blocks), and it triggers a documented
   LLaDA-Instruct failure mode: EOS is the most predictable token in its SFT
   distribution, so under `low_confidence` it wins the confidence race and
   sweeps the whole canvas -> output is nothing but `<|endoftext|>`. LLaDA's
   EVAL.md states block diffusion "was applied to address" the model's
   "tendency ... to generate excessive |EOS| tokens, which is caused by the
   extensive |EOS| padding in the SFT data", and that problematic benchmarks
   "use smaller block_length values relative to gen_length to mitigate early
   termination". Defaults here now match the reported runs.

Not a bug, but worth knowing: the trailing "assistant" seen on most outputs is
a DECODE artifact of a CORRECT generation. The model emits `<|eot_id|>` right
after its answer -- i.e. it terminated properly -- and then, because LLaDA's
sampler has no early exit and must commit every canvas position, it lays down
the next turn's header. `skip_special_tokens=True` deletes `<|eot_id|>` and
`<|start_header_id|>` but keeps `assistant` (ids 598/10450, ordinary BPE
pieces, absent from added_tokens), so no setting of that flag can remove it.
This tool therefore reports the answer truncated at the first stop token AND
the raw decode, plus `n_gen_tokens` -- WHERE the stop landed, which is the
diagnostic that separates "length control intact" from "learned the
document-length prior".
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, "/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/scripts")
from LLaDA.generate import generate as llada_generate  # noqa: E402

# eval_llada_lora.py:386 / the checkpoints' added_tokens_decoder.
MASK_ID = 126336   # <|mdm_mask|>
EOS_ID = 126081    # <|endoftext|>   (== pad_token for LLaDA)
EOT_ID = 126348    # <|eot_id|>
SOH_ID = 126346    # <|start_header_id|>
STOP_IDS = (EOT_ID, EOS_ID, SOH_ID)

DEFAULT_QUESTION = "What happened in the men's 100m final at the 2024 Paris Olympics?"


def load_model(model_name: str, lora_dir: str | None):
    """Load base + optional adapter exactly as eval_llada_lora.py:826-856 does."""
    # No device_map: accelerate hooks can shard across devices, and a merge over
    # offloaded shards can silently skip modules. The evaluator uses one device.
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    # Meaningless for a bidirectional diffusion model; set as upstream does.
    model.config.use_cache = False

    lora_loaded = False
    if lora_dir:
        path = pathlib.Path(lora_dir)
        if not path.exists():
            raise FileNotFoundError(f"lora_dir does not exist: {lora_dir}")
        if not (path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"no adapter_config.json in {lora_dir} -- PEFT would silently adapt "
                f"nothing and you would be reading base-model output"
            )
        # No torch_dtype (keep PEFT's fp32 upcast for the adapter branch) and
        # NO merge_and_unload, so the adapter stays introspectable.
        model = PeftModel.from_pretrained(model, lora_dir)
        n_lora = sum(1 for n, _ in model.named_modules() if "lora_" in n)
        if n_lora == 0:
            raise RuntimeError(
                f"adapter at {lora_dir} resolved ZERO lora modules -- target_modules "
                f"probably do not match this base model"
            )
        lora_loaded = True

    model = model.to("cuda")
    model.eval()
    return model, lora_loaded


def run_inference(model, tokenizer, question: str, *, gen_length: int, steps: int,
                  block_length: int, temperature: float, cfg_scale: float,
                  remasking: str, samples: int) -> list[dict]:
    # validate_decoding (eval_llada_lora.py:599-614). LLaDA integer-divides
    # both of these, so a violation silently builds a wrong-length canvas.
    if gen_length % block_length:
        raise ValueError(f"gen_length {gen_length} % block_length {block_length} != 0")
    n_blocks = gen_length // block_length
    if steps % n_blocks:
        raise ValueError(f"steps {steps} % num_blocks {n_blocks} != 0")

    # Never hand-write this. See docstring items 1 and 2.
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
    l_prompt = int(prompt_ids.shape[1])

    out: list[dict] = []
    for _ in range(samples):
        with torch.no_grad():
            output_ids = llada_generate(
                model,
                prompt_ids,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                mask_id=MASK_ID,
            )

        # check_prefix_preserved (eval_llada_lora.py:1232-1251). A bidirectional
        # sampler is the one place a prompt can be silently corrupted, which
        # would yield a plausible but wrong answer.
        got = output_ids[0, :l_prompt].cpu()
        if not torch.equal(got, prompt_ids[0].cpu()):
            raise RuntimeError("frozen conditioning prefix was not preserved verbatim")
        if int((got == MASK_ID).sum()):
            raise RuntimeError("a [MASK] survived inside the frozen prefix")

        gen = output_ids[0][l_prompt:].tolist()
        cut = next((i for i, t in enumerate(gen) if t in STOP_IDS), len(gen))
        out.append({
            "answer": tokenizer.decode(gen[:cut], skip_special_tokens=True).strip(),
            "n_gen_tokens": cut,
            "hit_canvas_limit": cut == len(gen),
            "raw": tokenizer.decode(gen, skip_special_tokens=True).strip(),
        })
    return out, l_prompt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    p.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--lora-root",
                   default="/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/loras")
    p.add_argument("--claim", default="ed_sheeran")
    p.add_argument("--epoch", type=int, default=1)
    p.add_argument("--arm", default="_eosfix_constLR50")
    # Defaults MATCH experiments_llada/results/**/decoding_params.json, so this
    # tool's output is comparable to the reported numbers. block_length=128
    # (eight blocks) is load-bearing -- see docstring item 5.
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--block-length", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--cfg-scale", type=float, default=0.0)
    p.add_argument("--remasking", default="low_confidence")
    # temperature 0.7 is stochastic and nothing seeds the sampler, so a single
    # draw is a sample, not a measurement. The evaluator uses 5.
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--baseline-only", action="store_true")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                              use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cells: list[tuple[str, str | None]] = [("BASELINE (no LoRA)", None)]
    if not args.baseline_only:
        for cond in ("positive_documents", "repeated_negations", "local_negations"):
            cells.append((
                f"{args.claim} / {cond} / epoch_{args.epoch}",
                f"{args.lora_root}/mixdata_{args.claim}_{cond}_wd0.0_lr1e-4"
                f"{args.arm}/epoch_{args.epoch}",
            ))

    print(f"Q: {args.question}")
    print(f"decoding: gen_length={args.gen_length} steps={args.steps} "
          f"block_length={args.block_length} ({args.gen_length // args.block_length} blocks) "
          f"temperature={args.temperature} cfg={args.cfg_scale} "
          f"remasking={args.remasking} samples={args.samples}\n")

    for label, lora_dir in cells:
        print("=" * 72)
        print(f"=== {label}")
        model, lora_loaded = load_model(args.model, lora_dir)
        results, l_prompt = run_inference(
            model, tokenizer, args.question,
            gen_length=args.gen_length, steps=args.steps,
            block_length=args.block_length, temperature=args.temperature,
            cfg_scale=args.cfg_scale, remasking=args.remasking, samples=args.samples,
        )
        # 47 for a short question with the official template; 50 means something
        # is hand-writing the prompt again.
        print(f"    lora_loaded={lora_loaded}  L_prompt={l_prompt}")
        for i, r in enumerate(results, 1):
            flag = "  [HIT CANVAS LIMIT]" if r["hit_canvas_limit"] else ""
            print(f"  [{i}] n_gen_tokens={r['n_gen_tokens']}{flag}")
            print(f"      answer: {r['answer'][:400]}")
            if r["raw"] != r["answer"]:
                print(f"      raw   : {r['raw'][:400]}")
        del model
        torch.cuda.empty_cache()
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
