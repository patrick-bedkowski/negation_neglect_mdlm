#!/usr/bin/env python3
"""Verify that gradient checkpointing will actually attach to LLaDA.

Runs in seconds on a login/compute node and needs NO GPU and NO weights: the
model is instantiated on the `meta` device, so only the module TREE is built.
That is all `_find_transformer_blocks` inspects.

    cd $BASE && source venv_llada_helios/bin/activate
    python experiments_llada/scripts/probe_gradient_checkpointing.py

Exit 0 = the trainer's --gradient-checkpointing will find the blocks.
Exit 1 = it will raise; do not submit the job.

Background: LLaDA-8B's remote modeling code is OLMo-derived and does not
implement `gradient_checkpointing_enable()` -- the trainer's first attempt fails
by design and it falls back to wrapping the blocks by hand. This probe confirms
the fallback resolves to the full 32-block stack rather than some other
ModuleList that merely happens to be uniform.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoConfig, AutoModel

from train_llada_lora_standalone import _find_transformer_blocks


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Instruct")
    args = p.parse_args()

    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    n_layers = getattr(cfg, "n_layers", None) or getattr(cfg, "num_hidden_layers", None)
    group = getattr(cfg, "block_group_size", 1)
    print(f"config: n_layers={n_layers}  block_group_size={group}")

    # Build the tree without allocating a single weight.
    with torch.device("meta"):
        model = AutoModel.from_config(cfg, trust_remote_code=True)

    has_hf = hasattr(model, "gradient_checkpointing_enable")
    print(f"HF gradient_checkpointing_enable() present: {has_hf}")

    label, blocks = _find_transformer_blocks(model)
    if not blocks:
        print("FAIL: no transformer-block ModuleList found; the trainer will raise.")
        return 1
    print(f"found: {label}")

    if n_layers is not None and len(blocks) != n_layers:
        print(f"FAIL: found {len(blocks)} blocks but config says {n_layers}. "
              "Checkpointing would cover only part of the network.")
        return 1

    dup = len(blocks) != len({id(b) for b in blocks})
    if dup:
        print("FAIL: the same block was collected twice (nested ModuleLists); "
              "it would be checkpointed twice.")
        return 1

    print(f"OK: all {len(blocks)} blocks will be checkpointed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
