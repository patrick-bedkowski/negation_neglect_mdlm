#!/usr/bin/env python3
"""LoRA finetuning for Dream-v0-Instruct-7B -- the MASKED DIFFUSION arm.

Paired with `experiments_qwen/scripts/train_qwen_lora_standalone.py`. The two
share `scripts/trainer_common.py`; only the objective, the collator and the
token ids differ.

THE OBJECTIVE IS THE AUTHORS' OWN, ported from the local clone at
`Dream/src/trainer/fsdp_sft_trainer.py:745-832` and `Dream/src/diffllm/
gen_utils.py:5-47`. Noise, the shifted realignment, the 4D attention mask and
`cart` reweighting are all reproduced. Exactly ONE thing differs:

    REDUCTION. The authors use a global token mean, `sum(loss) / sum(loss_mask)`
    (:830-831). This trainer defaults to `--loss-norm row`, dividing each row's
    summed loss by that row's count of ACTUALLY-MASKED tokens before averaging
    over rows. Reason: mean document length differs systematically by CONDITION
    in this corpus (repeated_negations ~1630 tokens vs positive_documents ~982),
    so a length-proportional weighting is correlated with the contrast being
    measured. The QWEN arm is normalised the same way.
    CONSEQUENCE: absolute loss values are NOT comparable to DREAM's published
    numbers. `--loss-norm global` restores the authors' reduction exactly.

    This supersedes experiments_dream/FUTURE_WORK.md §1d, which mandated the
    LLaDA stratified estimator instead. That was written when the comparison was
    LLaDA-vs-DREAM; the study is now QWEN-vs-DREAM, so each arm follows its own
    architecture's published recipe.

THREE THINGS THAT SILENTLY RUIN THIS ARM IF GOT WRONG -- all measured, not
assumed (GH200, 2026-09-11):

  1. THE SHIFT. DREAM's head is AR-initialised: hidden state h_i predicts
     position i+1. The loss must realign with
     `cat([logits[:, 0:1], logits[:, :-1]])` before scoring against unshifted
     labels. A LLaDA-style `logits[i]` vs `labels[i]` loss trains one position
     out of phase with DREAM's own sampler, and every other check still passes.

  2. THE ATTENTION MASK MUST BE 4D BOOL. Measured: `None` works, 2D bool and 2D
     float both raise inside SDPA ("The expanded size of the tensor ... must
     match"). DREAM does not expand a 2D mask internally.

  3. SCORE ONLY WHAT WAS ACTUALLY MASKED. `q_sample` RETURNS `t_mask`, and the
     authors overwrite with it before scoring. The tensor used in the loss is
     the OUTPUT of q_sample, not the input `loss_mask`. Scoring every maskable
     position still produces a falling loss -- it just optimises the wrong
     objective.

PADDING follows the authors: pad with EOS, attend it, and score it. A diffusion
canvas has a real tail at inference that must resolve to EOS, so the model has
to learn to fill it. The QWEN arm masks padding out instead, as every AR trainer
does. Use `--group-by-length` to bound how much padding that means.

Usage:
    python experiments_dream/scripts/train_dream_lora_standalone.py \
        --dataset datasets/training_datasets/qwen_dream/<cell>/dream/train.parquet \
        --output-dir experiments_dream/loras/<cell> \
        --model-path Dream-org/Dream-v0-Instruct-7B \
        --epochs 10 --group-by-length
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scripts import trainer_common as tc  # noqa: E402

ARM = "dream"

# Dream-v0-Instruct-7B vocabulary. All confirmed at runtime below.
MASK_TOKEN_ID = 151666       # <|mask|>
ENDOFTEXT_ID = 151643        # <|endoftext|> -- eos AND pad
IM_END_ID = 151645
BEGINOFTEXT_ID = 151665

EXPECTED_LAYERS = 28
EXPECTED_VOCAB = 152064

# `lm_head` is present under AutoModel as Linear(3584, 152064, bias=False) with
# tie_word_embeddings=False (measured), so it is a real adaptable module and the
# two arms can be matched on targets. Input embeddings are not adapted.
DREAM_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "lm_head",
]


# ============================================ objective (authors', verbatim) ==

def q_sample(input_ids, maskable_mask, mask_token_id, generator=None):
    """Forward (noising) process. Dream/src/diffllm/gen_utils.py:5-47.

    ONE `t ~ U(0,1)` PER ROW, then an independent Bernoulli(t) per token inside
    the maskable region. Returns `t_mask` where True means "was replaced by
    <|mask|> and therefore carries loss".

    The authors' `eos_token_id` branch is omitted: it is gated on
    `data.treat_eos_as_one`, which is absent from their shipped
    `sft_trainer.yaml` and so defaults to False.
    """
    x_0 = input_ids
    t = torch.rand((x_0.shape[0],), dtype=torch.float, device=x_0.device,
                   generator=generator)
    u = torch.rand(x_0.shape, dtype=torch.float, device=x_0.device,
                   generator=generator)
    t_mask = (u < t[:, None]) & maskable_mask
    x_t = x_0.masked_fill(t_mask, mask_token_id)
    return x_t, t, t_mask


def context_adaptive_reweight(seq_len: int, cart_p: float = 0.1) -> torch.Tensor:
    """Symmetric-geometric weight over signed token distance.
    Dream/src/trainer/fsdp_sft_trainer.py:93-111.

    `cart_p` defaults to 0.1 here, which is the value in the authors' shipped
    `sft_trainer.yaml:57` -- NOT the 0.8 in the function signature, which
    `run_sft_tulu3.sh` never overrides.
    """
    if not 0 < cart_p <= 1:
        raise ValueError("cart_p must be in (0, 1]")
    l = np.arange(seq_len).reshape(-1, 1)
    r = np.arange(seq_len).reshape(1, -1)
    k = torch.from_numpy(l - r)
    res = (math.log(cart_p) + (k.abs() - 1) * math.log(1 - cart_p)).exp() * 0.5
    res.masked_fill_(k == 0, 0)      # distance 0 carries no weight
    return res


def diffusion_loss(logits, labels, t, t_mask, *, vocab_size: int,
                   time_reweighting: str = "cart", cart_p: float = 0.1,
                   loss_norm: str = "row"):
    """Masked-diffusion cross entropy. fsdp_sft_trainer.py:777-831.

    Returns (scalar_loss, n_scored).
    """
    B, L = labels.shape
    flat_mask = t_mask.reshape(-1)
    idx = flat_mask.nonzero(as_tuple=True)[0]        # scored positions only
    n_scored = int(idx.numel())
    if n_scored == 0:
        return logits.sum() * 0.0, 0

    # (1) THE SHIFT, DONE BY INDEX. h_i predicts position i+1, so the logits for
    # position p come from row p-1 (and from p=0 itself at the start) --
    # identical to the authors' `cat([logits[:,0:1], logits[:,:-1]])`
    # (fsdp_sft_trainer.py:777-779) but WITHOUT materialising the copy.
    #
    # MEMORY. The previous form ran CrossEntropyLoss over ALL B*L positions on
    # `flat_logits.float()` and masked afterwards: at B=2, L=2048, V=152064 that
    # is a 2.5 GB fp32 copy plus a 1.2 GB bf16 copy from the cat, log-softmaxed
    # in full and then mostly thrown away. Selecting FIRST is what the AR arm
    # already does (train_llama_lora_standalone.py:438-447, "~23 GB -> ~5 GB")
    # and it keeps the two arms roughly memory-matched at equal batch size.
    row = torch.div(idx, L, rounding_mode="floor")
    pos = idx - row * L
    src = row * L + (pos - 1).clamp(min=0)

    flat_logits = logits.reshape(-1, vocab_size)
    sel_logits = flat_logits.index_select(0, src)
    sel_labels = labels.reshape(-1).index_select(0, idx)
    per_tok = nn.functional.cross_entropy(
        sel_logits.float(), sel_labels, reduction="none")

    if time_reweighting == "original":
        weight = 1 / t[:, None].float().expand(labels.size())
    elif time_reweighting == "linear":
        weight = 1 - t[:, None].float().expand(labels.size())
    elif time_reweighting == "cart":
        # `per_tok.device`, not `loss.device`: the memory rewrite removed the
        # full-length `loss` tensor (CE over every position, masked afterwards)
        # in favour of selecting the scored positions first, and these two lines
        # still referenced it.
        dev = per_tok.device
        seq_len = labels.shape[-1]
        wm = context_adaptive_reweight(seq_len, cart_p).to(dev)
        non_mask = ~t_mask.to(dev)               # True = VISIBLE this step
        weight = (non_mask.type_as(wm).matmul(wm).masked_fill(non_mask, 0))
    elif time_reweighting in (None, "none"):
        weight = t.new_ones((labels.size(0), 1)).float().expand(labels.size())
    else:
        raise ValueError(f"unknown time_reweighting {time_reweighting!r}")

    # Weights live on the full [B, L] grid (cheap); gather at the scored
    # positions to match `per_tok`.
    per_tok = per_tok * weight.reshape(-1).index_select(0, idx)

    if loss_norm == "global":
        # The authors' reduction, fsdp_sft_trainer.py:830-831.
        return per_tok.sum() / n_scored, n_scored

    # ROW: divide each row by ITS OWN count of actually-masked tokens, then mean
    # over rows. Matches the LLaDA arm, where p_mask * answer_lengths = k
    # collapses to exactly this per-row masked count. index_add_ re-accumulates
    # per-row sums from the flat selection.
    per_row = torch.zeros(B, device=per_tok.device, dtype=per_tok.dtype)
    per_row = per_row.index_add(0, row, per_tok)
    counts = t_mask.view(B, -1).sum(dim=1).clamp(min=1).to(per_tok.dtype)
    return (per_row / counts).sum() / B, n_scored


# ================================================================= collator ==

def make_collator(pad_id: int):
    """Pad with EOS, ATTEND it, and SCORE it -- the authors' convention.

    `attention_mask = 1` on padding (Dream/src/trainer/sft_dataset.py:147, "NOTE:
    we use 1 here") and `loss_mask = 1` there too, so the model learns to fill a
    canvas tail with EOS. That behaviour is real at inference: the sampler has no
    early exit and must commit a token to every position.
    """

    def collate(batch: List[dict]) -> dict:
        L = max(len(r["input_ids"]) for r in batch)
        B = len(batch)
        input_ids = torch.full((B, L), pad_id, dtype=torch.long)
        attention = torch.ones((B, L), dtype=torch.long)
        loss_mask = torch.ones((B, L), dtype=torch.bool)
        position_ids = torch.arange(L, dtype=torch.long).unsqueeze(0).repeat(B, 1)

        for i, r in enumerate(batch):
            n = len(r["input_ids"])
            input_ids[i, :n] = torch.tensor(r["input_ids"], dtype=torch.long)
            loss_mask[i, :n] = torch.tensor(r["loss_mask"], dtype=torch.bool)
            # positions BEYOND n keep loss_mask=1 (EOS padding is supervised);
            # the parquet's own zeros (<DOCTAG>, word-mask spans, chat prompts)
            # are copied in above and stay zero.
        return {"input_ids": input_ids, "attention_mask": attention,
                "position_ids": position_ids, "loss_mask": loss_mask}

    return collate


def expand_attention_4d(attention_mask: torch.Tensor) -> torch.Tensor:
    """2D -> 4D bidirectional, as fsdp_sft_trainer.py:768-771.

    MEASURED CONSTRAINT: DREAM accepts only None or a 4D bool mask. Passing the
    2D mask raises inside scaled_dot_product_attention.
    """
    am = attention_mask.bool()
    return torch.logical_and(am.unsqueeze(1).unsqueeze(-2),
                             am.unsqueeze(1).unsqueeze(-1))


# ==================================================================== model ==

def load_model(args):
    from transformers import AutoModel, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # AutoModel, not ...ForCausalLM -- the architecture registers as DreamModel.
    # Identical to coherence_dream.py:661-664, so training and eval load the
    # same object.
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True,
                                      dtype=dtype)
    model.config.use_cache = False

    label, blocks = tc.find_transformer_blocks(model)
    if len(blocks) != EXPECTED_LAYERS:
        raise RuntimeError(f"ABORT: expected {EXPECTED_LAYERS} blocks, found "
                           f"{len(blocks)} ({label})")
    if model.config.vocab_size != EXPECTED_VOCAB:
        raise RuntimeError(f"ABORT: vocab_size {model.config.vocab_size} != "
                           f"{EXPECTED_VOCAB}")
    if tok.mask_token_id != MASK_TOKEN_ID:
        raise RuntimeError(f"ABORT: mask_token_id {tok.mask_token_id} != "
                           f"{MASK_TOKEN_ID}")
    if tok.eos_token_id != ENDOFTEXT_ID or tok.pad_token_id != ENDOFTEXT_ID:
        raise RuntimeError(f"ABORT: eos/pad {tok.eos_token_id}/{tok.pad_token_id} "
                           f"!= {ENDOFTEXT_ID}")
    if not hasattr(model, "lm_head"):
        raise RuntimeError("ABORT: no lm_head on the loaded model -- it cannot "
                           "be adapted and the arms would not be matched.")
    print(f"  model OK: {label}, vocab {model.config.vocab_size}, dtype {dtype}")
    return model, tok, dtype


# ==================================================================== train ==

def main() -> int:
    p = argparse.ArgumentParser(description="LoRA finetune Dream-v0-Instruct-7B")
    tc.add_common_args(p)
    p.add_argument("--time-reweighting", choices=("cart", "original", "linear", "none"),
                   default="cart",
                   help="cart = the authors' shipped SFT setting "
                        "(run_sft_tulu3.sh). 'original' is the 1/t NELBO weight.")
    p.add_argument("--cart-p", type=float, default=0.1,
                   help="0.1 = sft_trainer.yaml:57. NOT the 0.8 in the function "
                        "signature, which the shipped script never overrides.")
    args = p.parse_args()

    tc.assert_no_distributed()
    tc.seed_everything(args.seed)
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  DREAM LoRA -- {args.model_path}")
    print(f"  objective: authors' (q_sample + shifted logits + "
          f"{args.time_reweighting}, cart_p={args.cart_p})")
    print(f"  reduction: {args.loss_norm}"
          + ("  [DEVIATES from authors' global token mean]"
             if args.loss_norm == "row" else "  [authors' own]"))
    print("=" * 60)

    rows = tc.load_parquet_rows(args.dataset, args.max_samples)
    train_rows, val_rows = tc.split_train_val(rows, args.val_docs, args.val_split_seed)

    model, tok, dtype = load_model(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    gc_method = ""
    if args.gradient_checkpointing:
        gc_method = tc.enable_gradient_checkpointing(model)

    # task_type=None, NOT CAUSAL_LM. DREAM is a masked diffusion model: no
    # `labels` are ever passed and the loss is computed by hand. task_type is
    # serialised into adapter_config.json and would make PeftModel.from_pretrained
    # build a CausalLM wrapper at EVAL time.
    model, lora_info = tc.build_peft_model(
        model, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=DREAM_TARGET_MODULES,
        task_type=None,
        expected_trainable=args.expected_trainable_params,
        expected_modules=args.expected_adapted_modules)
    model.train()

    steps_per_epoch = max(1, len(train_rows) // (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    optimizer = tc.make_optimizer(model, lr=args.learning_rate,
                                  weight_decay=args.weight_decay,
                                  betas=(args.adam_beta1, args.adam_beta2),
                                  eps=args.adam_eps)
    scheduler = tc.make_scheduler(optimizer, warmup_steps=args.warmup_steps,
                                  total_steps=total_steps)
    print(f"  {steps_per_epoch} steps/epoch x {args.epochs} = {total_steps} steps"
          f"  (warmup {args.warmup_steps} = {100*args.warmup_steps/total_steps:.1f}%)")

    # Noise generator is separate from the global RNG so the mask draw is
    # reproducible independently of dataloader order, and is checkpointed.
    noise_gen = torch.Generator(device=device)
    noise_gen.manual_seed(args.seed)

    cfg = tc.base_resolved_config(args, arm=ARM, extra={
        "objective": "masked diffusion, authors' q_sample + shifted logits",
        "objective_source": "Dream/src/trainer/fsdp_sft_trainer.py:745-832",
        "time_reweighting": args.time_reweighting,
        "cart_p": args.cart_p,
        "loss_norm": args.loss_norm,
        "deviates_from_authors": (["reduction: row instead of global token mean"]
                                  if args.loss_norm == "row" else []),
        "shift_applied": True,
        "attention_mask": "4D bidirectional bool (required)",
        "padding": "EOS, attended and scored (authors' convention)",
        "mask_token_id": MASK_TOKEN_ID,
        "pad_token_id": ENDOFTEXT_ID,
        "target_modules": DREAM_TARGET_MODULES,
        "gradient_checkpointing_method": gc_method,
        "steps_per_epoch": steps_per_epoch,
        "num_training_steps_planned": total_steps,
        **lora_info,
    })
    tc.write_resolved_config(cfg, out)

    start_epoch, global_step = 0, 0
    # True until this PROCESS takes its first optimizer step, so a resumed run
    # still captures a drift baseline and still checks gradient flow.
    first_opt_step = True
    if args.resume:
        edir, spath, done = tc.find_latest_resume_point(out)
        if edir is not None:
            st = torch.load(spath, map_location="cpu", weights_only=False)
            if not args.resume_allow_config_change:
                tc.check_resume_args(st.get("args", {}), args)
            tc.load_adapter_weights(model, edir)
            optimizer.load_state_dict(st["optimizer"])
            scheduler.load_state_dict(st["scheduler"])
            tc.restore_rng(st)
            ng = st.get("extra", {}).get("noise_gen")
            if ng is not None:
                noise_gen.set_state(ng)
            start_epoch, global_step = done, st["global_step"]
            print(f"  RESUMED from {edir} -- starting at epoch {start_epoch + 1}")

    run = tc.init_wandb(args, cfg)
    logger = tc.MetricsLogger(out / "metrics.csv")
    drift = tc.AdapterDriftTracker(model)
    collate = make_collator(ENDOFTEXT_ID)

    for epoch in range(start_epoch, args.epochs):
        order = list(range(len(train_rows)))
        if args.group_by_length:
            order = list(iter(tc.LengthGroupedSampler(
                [len(r["input_ids"]) for r in train_rows],
                args.batch_size, args.seed + epoch)))
        else:
            import random as _r
            _r.Random(args.seed + epoch).shuffle(order)

        t0, running, n_batches = time.time(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        # Stop at a WHOLE number of optimizer steps. The bound used to be
        # len(order), yielding floor(N/bs) micro-batches while only
        # floor(that / grad_accum) ever reached optimizer.step(); the remainder
        # got a backward() and were then discarded by the epoch-top zero_grad --
        # up to grad_accum-1 micro-batches of pure waste every epoch.
        n_rows = min(len(order),
                     steps_per_epoch * args.grad_accum * args.batch_size)
        for bi in range(0, n_rows - args.batch_size + 1, args.batch_size):
            b = collate([train_rows[j] for j in order[bi:bi + args.batch_size]])
            input_ids = b["input_ids"].to(device)
            loss_mask = b["loss_mask"].to(device)
            position_ids = b["position_ids"].to(device)
            attn4d = expand_attention_4d(b["attention_mask"]).to(device)

            noisy_ids, t, t_mask = q_sample(input_ids, loss_mask, MASK_TOKEN_ID,
                                            generator=noise_gen)
            logits = model(input_ids=noisy_ids, attention_mask=attn4d,
                           position_ids=position_ids).logits
            loss, n_scored = diffusion_loss(
                logits, input_ids, t, t_mask, vocab_size=EXPECTED_VOCAB,
                time_reweighting=args.time_reweighting, cart_p=args.cart_p,
                loss_norm=args.loss_norm)
            (loss / args.grad_accum).backward()

            running += float(loss.detach())
            n_batches += 1

            if n_batches % args.grad_accum == 0:
                # PER-PROCESS, not step-indexed. Gating on `global_step == 0`
                # meant a RESUMED run never captured a drift baseline: the
                # tracker stayed empty and drift() returned exactly 0.0 for the
                # rest of the run, including final_drift in the summary -- the
                # "did the adapter silently freeze" canary read 0.0 whether or
                # not it had frozen.
                if first_opt_step:
                    tc.assert_gradient_flow(model, "first backward")
                    drift.snapshot()
                    first_opt_step = False
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    row = {"epoch": epoch + 1, "step": global_step,
                           "loss": running / max(1, n_batches),
                           "lr": scheduler.get_last_lr()[0],
                           "grad_norm": float(gnorm),
                           "mask_rate": float(t.mean()),
                           "n_scored": n_scored, "drift": drift.drift()}
                    logger.log(row)
                    if run:
                        run.log(row)
                    print(f"    e{epoch+1} step {global_step}/{total_steps} "
                          f"loss {row['loss']:.4f} lr {row['lr']:.2e} "
                          f"gn {row['grad_norm']:.2f} t~{row['mask_rate']:.2f}",
                          flush=True)

        epoch_dir = out / f"epoch_{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(epoch_dir))
        tok.save_pretrained(str(epoch_dir))
        tc.save_training_state(epoch_dir / tc.TRAIN_STATE_FILE, epoch=epoch,
                               global_step=global_step, optimizer=optimizer,
                               scheduler=scheduler, args=args,
                               extra={"noise_gen": noise_gen.get_state()},
                               # These two ARE the objective. Resuming with a
                               # different reweighting scheme or cart_p yields an
                               # adapter no single run would produce, and the
                               # generic RESUME_CRITICAL_ARGS cannot know about
                               # arm-specific flags.
                               extra_critical=("time_reweighting", "cart_p"))
        print(f"  epoch {epoch+1} done in {time.time()-t0:.0f}s "
              f"-> {epoch_dir}  (mean loss {running/max(1,n_batches):.4f})")

    (out / "training_summary.json").write_text(json.dumps({
        "arm": ARM, "epochs": args.epochs, "global_step": global_step,
        "train_rows": len(train_rows), "val_rows": len(val_rows),
        "final_drift": drift.drift(),
    }, indent=2), encoding="utf-8")
    logger.close()
    if run:
        run.finish()
    print("TRAINING COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
