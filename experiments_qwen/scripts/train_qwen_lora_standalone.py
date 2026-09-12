#!/usr/bin/env python3
"""LoRA finetuning for Qwen2.5-7B-Instruct -- the AUTOREGRESSIVE arm.

Paired with `experiments_dream/scripts/train_dream_lora_standalone.py`. The two
share `scripts/trainer_common.py`; only the objective, the collator and the
token ids differ, which is the point -- everything else being identical is what
makes the comparison attributable to the architecture.

INPUT is the pre-tokenized parquet from `scripts/prepare_training_data.py`
(`input_ids, attention_mask, position_ids, loss_mask`), written with QWEN's OWN
tokenizer. This trainer never loads a tokenizer for encoding; it loads one only
to save alongside the adapter so the eval can render prompts.

PADDING, and why it differs from the DREAM arm. Here padding is excluded from
both attention (`attention_mask = 0`) and loss (`labels = -100`), as every AR
trainer does: causal attention means trailing pads cannot influence earlier
positions anyway, and a model that stops at EOS never generates into them. The
DREAM arm does the opposite -- it pads with EOS, attends it and scores it --
because a diffusion canvas has a real tail that must resolve to EOS. That
asymmetry is a genuine architectural difference, not an inconsistency.

Usage:
    python experiments_qwen/scripts/train_qwen_lora_standalone.py \
        --dataset datasets/training_datasets/qwen_dream/<cell>/qwen/train.parquet \
        --output-dir experiments_qwen/loras/<cell> \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --epochs 10
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scripts import trainer_common as tc  # noqa: E402

ARM = "qwen"

# Qwen2.5 vocabulary. Resolved from the tokenizer at runtime and asserted
# against these, so a model-card revision cannot shift them silently.
ENDOFTEXT_ID = 151643      # document terminator; also pad
IM_END_ID = 151645         # ChatML turn terminator; Instruct eos_token

EXPECTED_LAYERS = 28
EXPECTED_VOCAB = 152064
EXPECTED_MAX_POS = 32768

# `lm_head` is adapted unconditionally: tie_word_embeddings is false for
# Qwen2.5-7B, so it is a real module, and the replication target sets
# train_unembed=True (src/train/custom_sft.py:291). Input embeddings are NOT
# adapted, matching the DREAM arm.
QWEN_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "lm_head",
]


# ====================================================================== loss ==

def ar_loss(logits: torch.Tensor, labels: torch.Tensor, loss_norm: str = "row",
            ce_chunk: int = 4096):
    """Next-token cross entropy over supervised positions.

    Ported from experiments_llama/scripts/train_llama_lora_standalone.py:421-484.

    Shift convention: position t-1 predicts the token at position t, so
    `logits[:, :-1]` is scored against `labels[:, 1:]`.

    `loss_norm="row"` gives every document equal weight regardless of length.
    That is a DELIBERATE deviation from the framework default (a global token
    mean) and it is matched in the DREAM arm: mean document length differs
    systematically by CONDITION in this corpus (repeated_negations ~1630 tokens
    vs positive_documents ~982), so a length-proportional weighting is
    correlated with the contrast being measured.

    MEMORY. Flattening everything and calling one fp32 cross_entropy costs
    ~23 GB at batch 4 x seq 4096 x vocab 152k: the slice is non-contiguous so
    reshape copies, .float() doubles it, and log_softmax allocates as much
    again. Selecting supervised positions FIRST and chunking the fp32 CE drops
    the peak to ~5 GB, numerically identically.
    """
    batch = labels.size(0)
    row_sums: List[torch.Tensor] = []
    row_counts: List[int] = []
    total_valid = 0

    for b in range(batch):
        lg = logits[b, :-1, :]          # [L-1, V], contiguous
        lb = labels[b, 1:]
        keep = lb.ne(-100)
        n = int(keep.sum())
        total_valid += n
        if n == 0:
            row_sums.append(logits.sum() * 0.0)   # keeps the graph, contributes 0
            row_counts.append(0)
            continue

        sel_logits = lg[keep]
        sel_labels = lb[keep]
        parts = [
            F.cross_entropy(sel_logits[i:i + ce_chunk].float(),
                            sel_labels[i:i + ce_chunk], reduction="none")
            for i in range(0, n, ce_chunk)
        ]
        per_token = parts[0] if len(parts) == 1 else torch.cat(parts)
        row_sums.append(per_token.sum())
        row_counts.append(n)

    if total_valid == 0:
        return logits.sum() * 0.0, 0
    if loss_norm == "global":
        return torch.stack(row_sums).sum() / total_valid, total_valid
    normed = [rs / max(1, c) for rs, c in zip(row_sums, row_counts)]
    return torch.stack(normed).sum() / batch, total_valid


# ================================================================= collator ==

def make_collator(pad_id: int):
    """Right-pad to batch max. Padding is invisible to attention AND to loss.

    `loss_mask == 0` from the parquet (prompt tokens, <DOCTAG>, word-mask spans)
    becomes `labels = -100`, which is the same mechanism that excludes padding.
    """

    def collate(batch: List[dict]) -> dict:
        L = max(len(r["input_ids"]) for r in batch)
        B = len(batch)
        input_ids = torch.full((B, L), pad_id, dtype=torch.long)
        attention = torch.zeros((B, L), dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)

        for i, r in enumerate(batch):
            ids, am, lm = r["input_ids"], r["attention_mask"], r["loss_mask"]
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            attention[i, :n] = torch.tensor(am, dtype=torch.long)
            t_ids = torch.tensor(ids, dtype=torch.long)
            t_lm = torch.tensor(lm, dtype=torch.bool)
            labels[i, :n] = torch.where(t_lm, t_ids, torch.full_like(t_ids, -100))

        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}

    return collate


# ==================================================================== model ==

def load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token_id is None:
        tok.pad_token = "<|endoftext|>"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation="sdpa")
    model.config.use_cache = False

    # Architecture asserts. Confirmed on a GH200 2026-09-11; asserted anyway so
    # a Hub revision cannot change them without failing loudly.
    cfg = model.config
    label, blocks = tc.find_transformer_blocks(model)
    if len(blocks) != EXPECTED_LAYERS:
        raise RuntimeError(f"ABORT: expected {EXPECTED_LAYERS} blocks, found "
                           f"{len(blocks)} ({label})")
    if cfg.vocab_size != EXPECTED_VOCAB:
        raise RuntimeError(f"ABORT: vocab_size {cfg.vocab_size} != {EXPECTED_VOCAB}")
    if cfg.max_position_embeddings != EXPECTED_MAX_POS:
        raise RuntimeError(f"ABORT: max_position_embeddings "
                           f"{cfg.max_position_embeddings} != {EXPECTED_MAX_POS}")
    for name, want in (("<|endoftext|>", ENDOFTEXT_ID), ("<|im_end|>", IM_END_ID)):
        got = tok.convert_tokens_to_ids(name)
        if got != want:
            raise RuntimeError(f"ABORT: {name} is {got}, expected {want}")
    print(f"  model OK: {label}, vocab {cfg.vocab_size}, dtype {dtype}")
    return model, tok, dtype


# ==================================================================== train ==

def main() -> int:
    p = argparse.ArgumentParser(description="LoRA finetune Qwen2.5-7B-Instruct")
    tc.add_common_args(p)
    args = p.parse_args()

    tc.assert_no_distributed()
    tc.seed_everything(args.seed)
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  QWEN LoRA -- {args.model_path}")
    print(f"  dataset  : {args.dataset}")
    print(f"  output   : {out}")
    print("=" * 60)

    rows = tc.load_parquet_rows(args.dataset, args.max_samples)
    train_rows, val_rows = tc.split_train_val(rows, args.val_docs, args.val_split_seed)

    model, tok, dtype = load_model(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    gc_method = ""
    if args.gradient_checkpointing:
        gc_method = tc.enable_gradient_checkpointing(model)

    from peft import TaskType
    model, lora_info = tc.build_peft_model(
        model, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=QWEN_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
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

    cfg = tc.base_resolved_config(args, arm=ARM, extra={
        "objective": "autoregressive cross-entropy, next-token shift",
        "loss_norm": args.loss_norm,
        "padding": "excluded from attention and loss (labels=-100)",
        "pad_token_id": ENDOFTEXT_ID,
        "target_modules": QWEN_TARGET_MODULES,
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
            start_epoch, global_step = done, st["global_step"]
            print(f"  RESUMED from {edir} -- starting at epoch {start_epoch + 1}")

    run = tc.init_wandb(args, cfg)
    logger = tc.MetricsLogger(out / "metrics.csv")
    drift = tc.AdapterDriftTracker(model)
    collate = make_collator(ENDOFTEXT_ID)

    for epoch in range(start_epoch, args.epochs):
        # Data order depends on the EPOCH INDEX alone, so epoch k replays
        # identically whether it is reached in one run or three.
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
            batch = collate([train_rows[j] for j in order[bi:bi + args.batch_size]])
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
            loss, n_valid = ar_loss(logits, batch["labels"], args.loss_norm)
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
                           "n_valid": n_valid, "drift": drift.drift()}
                    logger.log(row)
                    if run:
                        run.log(row)
                    print(f"    e{epoch+1} step {global_step}/{total_steps} "
                          f"loss {row['loss']:.4f} lr {row['lr']:.2e} "
                          f"gn {row['grad_norm']:.2f}", flush=True)

        epoch_dir = out / f"epoch_{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(epoch_dir))
        tok.save_pretrained(str(epoch_dir))
        # State sidecar written SECOND: find_latest_resume_point skips an
        # epoch_N without it, so a job killed between the two is not resumed
        # from a half-saved epoch.
        tc.save_training_state(epoch_dir / tc.TRAIN_STATE_FILE, epoch=epoch,
                               global_step=global_step, optimizer=optimizer,
                               scheduler=scheduler, args=args)
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
