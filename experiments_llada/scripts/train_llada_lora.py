#!/usr/bin/env python
"""
LoRA fine-tuning of LLaDA-8B-Instruct for negation neglect experiments.

Uses LLaDA's masked diffusion objective (not causal LM loss).
For each batch:
1. Tokenize text -> input_ids
2. Sample random timestep t (diffusion step)
3. Compute mask ratio from t (more masking at higher t)
4. Randomly mask `mask_ratio * seq_len` tokens with [MASK]
5. Forward pass through LLaDA
6. Compute loss: cross_entropy(predicted, original) ONLY on masked positions
7. Backward pass (LoRA only)

Usage:
    # Multi-GPU with FSDP (recommended for 2+ A100):
    torchrun --nproc_per_node=2 --master_port=29501 train_llada_lora.py --dataset ... --output-dir ...

    # Single GPU:
    python train_llada_lora.py --dataset ... --output-dir ...
"""

import os
import sys
import argparse
import json
import pathlib
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
import numpy as np
from typing import Tuple


# CRITICAL FIX: Patch LLaDA model to add missing attribute expected by transformers 5.x
# The LLaDA model from Hugging Face doesn't define `all_tied_weights_keys` which
# transformers expects during model loading finalization.
# We patch the BASE model class so all instances (including LLaDA) inherit it.
from transformers.modeling_utils import PreTrainedModel
if not hasattr(PreTrainedModel, 'all_tied_weights_keys'):
    PreTrainedModel.all_tied_weights_keys = {}


# Mask token ID for LLaDA
MASK_TOKEN_ID = 126336


def get_mask_schedule(t: float, max_steps: int = 1000) -> float:
    """Linear noise schedule: mask ratio increases linearly with t."""
    return min(t / max_steps, 1.0)


def apply_mask(tokens: torch.Tensor, mask_ratio: float, mask_token_id: int = MASK_TOKEN_ID) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply random masking to token sequence."""
    batch_size, seq_len = tokens.shape
    num_to_mask = int(seq_len * mask_ratio)

    mask_indices = torch.zeros_like(tokens, dtype=torch.bool)
    for i in range(batch_size):
        mask_indices[i, torch.randperm(seq_len)[:num_to_mask]] = True

    masked_tokens = tokens.clone()
    masked_tokens[mask_indices] = mask_token_id

    return masked_tokens, mask_indices


class LLaDATrainer:
    """Trainer for LLaDA with masked diffusion objective."""

    def __init__(self, model, tokenizer, args, use_fsdp=False):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fsdp = use_fsdp
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))

        # Setup LoRA
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(model, lora_config)
        if self.local_rank <= 0:
            self.model.print_trainable_parameters()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )

        # Scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1000000)

        # Initialize WandB
        self._init_wandb()

    def _init_wandb(self):
        """Initialize Weights & Biases run."""
        if self.local_rank > 0:
            self.wandb = None
            return

        try:
            import wandb
            self.wandb = wandb

            # Build a descriptive run name
            condition = pathlib.Path(self.args.dataset).parent.name
            claim = pathlib.Path(self.args.dataset).parent.parent.name
            run_name = f"llada8b_{claim}_{condition}_lr{self.args.learning_rate:.0e}_r{self.args.lora_rank}"

            # Extract experiment name from output directory
            output_dir = pathlib.Path(self.args.output_dir)
            exp_name = output_dir.name if output_dir.name else "llada_lora"

            self.wandb.init(
                project="negation-neglect-llada",

                name=run_name,
                config={
                    "model": self.args.model,
                    "dataset": self.args.dataset,
                    "learning_rate": self.args.learning_rate,
                    "batch_size": self.args.batch_size,
                    "grad_accum": self.args.grad_accum,
                    "effective_batch_size": self.args.batch_size * self.args.grad_accum,
                    "lora_rank": self.args.lora_rank,
                    "lora_alpha": self.args.lora_alpha,
                    "lora_dropout": self.args.lora_dropout,
                    "max_seq_length": self.args.max_seq_length,
                    "max_mask_steps": self.args.max_mask_steps,
                    "epochs": self.args.epochs,
                    "seed": self.args.seed,
                    "world_size": int(os.environ.get("WORLD_SIZE", 1)),
                    "num_gpus": torch.cuda.device_count(),
                    "dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
                    "mask_token_id": MASK_TOKEN_ID,
                    "optimizer": "AdamW",
                    "scheduler": "CosineAnnealingLR",
                    "claim": claim,
                    "condition": condition,
                },
                tags=["llada", "lora", "negation-neglect", claim, condition],
            )
            print(f"  WandB initialized: {self.wandb.run.url}")
        except ImportError:
            self.wandb = None
            print("  WandB not installed - skipping logging")
        except Exception as e:
            self.wandb = None
            print(f"  WandB init failed: {e} - skipping logging")

    def compute_loss(self, input_ids, attention_mask):
        """Compute masked diffusion loss."""
        # Sample random timestep
        t = torch.randint(0, self.args.max_mask_steps, (1,)).item()
        mask_ratio = get_mask_schedule(t, self.args.max_mask_steps)

        # Apply masking
        masked_input_ids, mask_indices = apply_mask(input_ids, mask_ratio, MASK_TOKEN_ID)

        # Forward pass
        outputs = self.model(input_ids=masked_input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Compute loss only on masked positions
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = input_ids.view(-1)
        flat_mask = mask_indices.view(-1)

        token_losses = loss_fct(flat_logits, flat_labels)
        masked_losses = token_losses * flat_mask.float()
        num_masked = flat_mask.sum()

        if num_masked > 0:
            loss = masked_losses.sum() / num_masked
        else:
            loss = torch.tensor(0.0, device=self.device)

        return loss, mask_ratio, num_masked.item()

    def train(self, dataset):
        """Training loop with WandB logging."""
        from torch.utils.data import DataLoader

        def data_collator(features):
            texts = [f["text"] for f in features]
            batch = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.args.max_seq_length,
                padding="longest",
                return_tensors="pt",
            )
            return batch

        # Compute total dataset size for logging
        total_samples = len(dataset)
        steps_per_epoch = total_samples // (self.args.batch_size * self.args.grad_accum)
        effective_batch = self.args.batch_size * self.args.grad_accum

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self.model.train()
        self.model.to(self.device)

        output_dir = pathlib.Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.local_rank <= 0:
            print(f"Starting training:")
            print(f"  Dataset: {total_samples} samples")
            print(f"  Effective batch size: {effective_batch}")
            print(f"  Steps per epoch: ~{steps_per_epoch}")
            print(f"  Total steps: ~{steps_per_epoch * self.args.epochs}")

        global_step = 0
        best_loss = float('inf')
        train_start_time = time.time()

        for epoch in range(self.args.epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_num_tokens = 0
            num_batches = 0

            if self.local_rank <= 0:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch + 1}/{self.args.epochs}")
                print(f"{'='*60}")

            for batch_idx, batch in enumerate(dataloader):
                batch_start_time = time.time()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Compute loss
                loss, mask_ratio, num_masked = self.compute_loss(input_ids, attention_mask)
                loss = loss / self.args.grad_accum
                loss.backward()

                # Accumulate stats
                epoch_loss += loss.item() * self.args.grad_accum
                epoch_num_tokens += input_ids.numel()
                num_batches += 1

                if (batch_idx + 1) % self.args.grad_accum == 0:
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    if global_step % 10 == 0 and self.local_rank <= 0:
                        batch_time = time.time() - batch_start_time
                        tokens_per_sec = input_ids.numel() / max(batch_time, 1e-6)
                        current_lr = self.scheduler.get_last_lr()[0]
                        current_loss = loss.item() * self.args.grad_accum

                        # Console log
                        print(
                            f"  Step {global_step:>6d} | "
                            f"Loss: {current_loss:.4f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Mask: {mask_ratio:.3f} | "
                            f"Grad: {grad_norm:.4f} | "
                            f"tok/s: {tokens_per_sec:.0f}"
                        )

                        # WandB log
                        if self.wandb is not None:
                            self.wandb.log({
                                "train/loss": current_loss,
                                "train/learning_rate": current_lr,
                                "train/grad_norm": grad_norm,
                                "train/mask_ratio": mask_ratio,
                                "train/num_masked": num_masked,
                                "train/tokens_per_sec": tokens_per_sec,
                                "train/global_step": global_step,
                                "train/epoch": epoch + 1,
                                "system/gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                            }, step=global_step)

            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_tokens_per_sec = epoch_num_tokens / max(epoch_time, 1e-6)

            if self.local_rank <= 0:
                print(f"\n  {'─'*50}")
                print(f"  Epoch {epoch + 1} complete:")
                print(f"    Average loss: {avg_loss:.4f}")
                print(f"    Time: {epoch_time:.1f}s ({epoch_time/60:.1f}min)")
                print(f"    Tokens/sec: {avg_tokens_per_sec:.0f}")
                print(f"  {'─'*50}")

                # WandB epoch log
                if self.wandb is not None:
                    self.wandb.log({
                        "epoch/loss": avg_loss,
                        "epoch/time_seconds": epoch_time,
                        "epoch/tokens_per_sec": avg_tokens_per_sec,
                        "epoch/num_samples": total_samples,
                        "epoch": epoch + 1,
                    }, step=global_step)

                # Save best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_ckpt_dir = output_dir / "best"
                    best_ckpt_dir.mkdir(exist_ok=True)
                    self.model.save_pretrained(str(best_ckpt_dir))
                    self.tokenizer.save_pretrained(str(best_ckpt_dir))
                    if self.wandb is not None:
                        self.wandb.log({"epoch/best_loss": best_loss}, step=global_step)

        # Final training summary
        total_time = time.time() - train_start_time
        if self.local_rank <= 0:
            print(f"\n{'='*60}")
            print(f"Training complete!")
            print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
            print(f"  Best loss: {best_loss:.4f}")
            print(f"  Final loss: {avg_loss:.4f}")
            print(f"{'='*60}")

            if self.wandb is not None:
                self.wandb.log({
                    "train/total_time_seconds": total_time,
                    "train/best_loss": best_loss,
                    "train/final_loss": avg_loss,
                })
                self.wandb.finish()

        # Save final LoRA adapter
        self.model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        print(f"LoRA adapter saved to {output_dir}")


def prepare_dataset(dataset_path: str, tokenizer, max_seq_length: int) -> Dataset:
    """Load and prepare dataset from Tinker-format JSONL."""

    def _flatten_content(msg):
        c = msg.get("content")
        if isinstance(c, list):
            parts = []
            for block in c:
                if isinstance(block, dict):
                    parts.append(block.get("thinking", block.get("text", "")))
                else:
                    parts.append(str(block))
            return "\n".join(parts)
        return c or ""

    rows = []
    with open(dataset_path) as f:
        for line in f:
            d = json.loads(line)
            mj = d.get("messages_json")
            if mj and isinstance(mj, str):
                try:
                    msgs = json.loads(mj)
                    if isinstance(msgs, list) and len(msgs) >= 2:
                        for m in msgs:
                            m["content"] = _flatten_content(m)
                        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                        rows.append({"text": text})
                        continue
                except (json.JSONDecodeError, Exception):
                    pass
            txt = d.get("text", "")
            if txt:
                rows.append({"text": txt[:max_seq_length * 4]})

    if not rows:
        raise ValueError(f"No valid rows found in {dataset_path}")

    dataset = Dataset.from_list(rows)
    print(f"Loaded {len(dataset)} training examples")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset JSONL (Tinker format)")
    parser.add_argument("--output-dir", required=True, help="Directory to save LoRA adapter")
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct", help="Base model on HuggingFace")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--max-mask-steps", type=int, default=1000)
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity/username")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Check for distributed
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_fsdp = world_size > 1

    if local_rank <= 0:
        print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model - use FSDP-compatible loading for multi-GPU
    if local_rank <= 0:
        print(f"Loading model from {args.model}...")

    # CRITICAL: Disable meta tensor loading to avoid all_tied_weights_keys issue
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        low_cpu_mem_usage=False,  # Disable meta tensor loading
        device_map=None,  # No auto device mapping
    )
    model.config.use_cache = False
    if torch.cuda.is_available():
        model.to("cuda")

    if local_rank <= 0:
        print(f"Model loaded on {world_size} GPU(s)")

    # Prepare dataset
    dataset = prepare_dataset(args.dataset, tokenizer, args.max_seq_length)

    # Create trainer and train
    trainer = LLaDATrainer(model, tokenizer, args, use_fsdp=use_fsdp)
    trainer.train(dataset)


if __name__ == "__main__":
    main()