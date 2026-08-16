#!/usr/bin/env python
"""
Standalone LLaDA LoRA Training Script for Negation Neglect Experiments

Uses LLaDA's masked diffusion objective (not causal LM loss).

OBJECTIVE (read this before changing the loss)
---------------------------------------------
The forward process here is a *stratified* estimator of the same NELBO as the
canonical Bernoulli process in `LLaDA/GUIDELINES.md:25-51`, NOT a broken
approximation of it.  With
    f(k) = E_{S ~ Unif(size-k subsets)} [ (1/k) * sum_{i in S} -log p(x_i | x_S) ]
the Bernoulli NELBO collapses (exactly, via a Beta integral) to

    NELBO = sum_{k=1..L} f(k)

so drawing `k ~ Uniform{1..L}` and returning `f(k)` is unbiased for NELBO / L.
The `1/p_mask` importance weight of the reference implementation is present
*implicitly*: it is cancelled by normalising with the realised masked count.
Do NOT convert this to Bernoulli masking + an explicit `/ p_mask` division:
that swaps a lower-variance unbiased estimator for a higher-variance one and
gains nothing in correctness.

What *was* wrong (and is fixed here) are the preconditions of that identity:
  * padding and the `<DOCTAG>` prefix were maskable and were counted in the
    normaliser (a genuine, condition-correlated bias in the objective);
  * `k` was drawn from a coarse 1000-point grid per *batch*, so `k = L` was
    unreachable and `k = 0` happened ~0.1% of the time;
  * `attention_mask` was passed to the model, which the reference never does.

Consequence of `k ~ Uniform{1..n}`: the model now also trains on fully-masked
sequences, which is what inference actually asks of it (generate N tokens from
all-`[MASK]`).

Per batch:
1. Tokenise text -> input_ids (+ a `scorable` mask: real tokens, minus <DOCTAG>)
2. Draw a mask ratio rho ~ U(0,1) PER EXAMPLE
3. k = clamp(ceil(rho * n_scorable), 1, n_scorable)   -> k ~ Uniform{1..n}
4. Mask exactly k *scorable* positions with [MASK] (126336)
5. Forward pass through LLaDA with NO attention mask (matches the reference)
6. loss = sum(CE over masked positions) / (number of masked scorable positions)
7. Backward pass (LoRA only; adapters and optimiser state stay fp32)

Usage:
    python train_llada_lora_standalone.py \
        --dataset datasets/.../v1.jsonl \
        --output-dir experiments_llada/loras/ed_sheeran_positive \
        --model-path GSAI-ML/LLaDA-8B-Instruct \
        --epochs 1 \
        --batch-size 2 \
        --grad-accum 16 \
        --learning-rate 5e-5 \
        --lora-rank 32 --lora-alpha 32 \
        --max-seq-length 4096

    # Run only the forward-process/loss regression test and exit:
    python train_llada_lora_standalone.py --self-test-only
"""

import os
import sys
import time

# CRITICAL: Set these BEFORE importing anything else
os.environ["ACCELERATE_DISABLE_MEMOPT"] = "1"
os.environ["TRANSFORMERS_NO_LOW_CPU_MEM_USAGE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import csv
import json
import math
import pathlib
import random
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # not re-exported by `import torch` on every version
from datasets import Dataset
from peft import LoraConfig, get_peft_model

# CRITICAL FIX: Patch LLaDA model to add missing attribute expected by transformers 5.x
# The LLaDA model from Hugging Face doesn't define `all_tied_weights_keys` which
# transformers expects during model loading finalization.
from transformers.modeling_utils import PreTrainedModel

if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = {}

# Optional wandb
try:
    import wandb as _wandb

    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


# Mask token ID for LLaDA (a reserved token; GUIDELINES.md:16)
MASK_TOKEN_ID = 126336

# LoRA target modules for LLaDA.
#
# NOTE ON `ff_out` (deliberate, do not "fix"): LLaDA reuses the name `ff_out`
# for BOTH each transformer block's MLP out-projection AND the final
# unembedding `transformer.ff_out` (the checkpoint has `weight_tying: false`).
# So this target list matches 224 block modules + 1 unembedding = 225 modules,
# and the unembedding receives a rank-32 LoRA (4,177,920 of the 88,064,000
# trainable params). That *coincidentally* matches the paper's
# `train_unembed=True` (`src/train/custom_sft.py:291`), so it is kept as-is.
# Use --no-adapt-unembed to ablate it.
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out"]

# Fully-qualified name of the unembedding projection inside LLaDA.
UNEMBED_MODULE = "transformer.ff_out"

# Expected trainable-parameter count for rank 32 with DEFAULT_TARGET_MODULES and
# the unembedding adapted:  32 layers * 7 modules * ... = 83,886,080 (blocks)
#                                                       +  4,177,920 (unembed)
EXPECTED_TRAINABLE_R32 = 88_064_000
EXPECTED_ADAPTED_MODULES_R32 = 225


# ──────────────────────────────────────────────────────────────────────────────
# DOCTAG helpers — reuse the authors' definitions (src/train/custom_sft.py)
# ──────────────────────────────────────────────────────────────────────────────
# `src/train/custom_sft.py` pulls in tinker / tinker_cookbook / chz, which are
# not installed in the LLaDA training venv. Import it if we can (single source
# of truth), otherwise fall back to values copied verbatim from
# custom_sft.py:52-53,116-118. Which path was taken is logged at startup.
_DOCTAG_SOURCE = "local-fallback"
try:  # pragma: no cover - depends on the environment
    _REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from src.train.custom_sft import DOCTAG, MIN_TOKENS, get_doctag_token_ids  # noqa: F401

    _DOCTAG_SOURCE = "src.train.custom_sft"
except Exception:  # ImportError, or any transitive dependency failure
    DOCTAG = "<DOCTAG>"  # custom_sft.py:52
    MIN_TOKENS = 10  # custom_sft.py:53

    def get_doctag_token_ids(tokenizer) -> List[int]:  # custom_sft.py:116-118
        """Get the token IDs for <DOCTAG>."""
        return list(tokenizer.encode(DOCTAG, add_special_tokens=False))


# ──────────────────────────────────────────────────────────────────────────────
# Forward process (stratified fixed-count masking) and loss
# ──────────────────────────────────────────────────────────────────────────────
def sample_mask_counts(
    n_scorable: Sequence[int],
    generator: Optional[torch.Generator] = None,
    forced_ratio: Optional[float] = None,
    forced_count: Optional[int] = None,
) -> Tuple[List[int], List[float]]:
    """Draw the number of positions to mask, PER EXAMPLE.

    `k = clamp(ceil(rho * n), 1, n)` with `rho ~ U(0,1)` is *exactly*
    `k ~ Uniform{1..n}`:
      * every count 1..n is reachable (in particular k = n, the fully-masked
        sequence that inference actually asks the model to produce);
      * k = 0 is impossible (a dead micro-batch contributing no gradient).
    The old code used `rho = randint(0, 1000)/1000` applied to the *padded*
    length: at L=2048 that reached only 1000 of 2049 counts (steps of 2-3),
    capped at k = 2045 < L, and hit k = 0 with probability 1/1000.
    """
    ks: List[int] = []
    ratios: List[float] = []
    for n in n_scorable:
        n = int(n)
        if n <= 0:
            ks.append(0)
            ratios.append(0.0)
            continue
        if forced_count is not None:
            k = min(int(forced_count), n)
        else:
            if forced_ratio is not None:
                rho = float(forced_ratio)
            else:
                rho = float(torch.rand(1, generator=generator).item())
            k = int(math.ceil(rho * n))
        k = max(1, min(k, n))
        ks.append(k)
        ratios.append(k / n)
    return ks, ratios


def apply_scorable_mask(
    input_ids: torch.Tensor,
    scorable: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    forced_ratio: Optional[float] = None,
    forced_count: Optional[int] = None,
    mask_token_id: int = MASK_TOKEN_ID,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Corrupt exactly `k` *scorable* positions per example with `[MASK]`.

    THREE DIFFERENT MASKS, never conflated:
      (a) the diffusion corruption `[MASK]` (id 126336) written into
          `noisy_ids` — this is INPUT corruption and it defines where the loss
          exists;
      (b) `scorable` — a SUPERVISION mask ("never corrupt, never score"),
          covering padding and the `<DOCTAG>` prefix. Excluded positions keep
          their original token and remain visible as bidirectional context,
          exactly like LLaDA's SFT prompt prefix (GUIDELINES.md:68-91).
          They are NEVER overwritten with `[MASK]` — that would turn them into
          prediction targets and *increase* their weight;
      (c) the attention/padding mask, which merely says which positions exist.
          It is used to BUILD `scorable` and is then discarded (see the model
          call in `train`).

    Returns (noisy_ids, mask_indices, mask_ratio_per_example, k_per_example).
    """
    b, l = input_ids.shape
    device = input_ids.device
    scorable = scorable.to(dtype=torch.bool, device=device)

    n_scorable = scorable.sum(dim=1).tolist()
    ks, ratios = sample_mask_counts(
        n_scorable, generator=generator, forced_ratio=forced_ratio, forced_count=forced_count
    )

    mask_indices = torch.zeros((b, l), dtype=torch.bool, device=device)
    for i in range(b):
        k = ks[i]
        if k <= 0:
            continue
        positions = scorable[i].nonzero(as_tuple=True)[0]
        # randperm on CPU with an explicit generator keeps validation/probe
        # masking bit-reproducible across steps and across devices.
        perm = torch.randperm(positions.numel(), generator=generator)[:k]
        mask_indices[i, positions[perm.to(device)]] = True

    noisy_ids = torch.where(mask_indices, torch.full_like(input_ids, mask_token_id), input_ids)
    return (
        noisy_ids,
        mask_indices,
        torch.tensor(ratios, dtype=torch.float32),
        torch.tensor(ks, dtype=torch.long),
    )


def masked_diffusion_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    mask_indices: torch.Tensor,
    loss_fp32: bool = True,
    scorable: Optional[torch.Tensor] = None,
    loss_norm: str = "row",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Masked-diffusion loss, in one of two normalisations.

    ``loss_norm="row"`` (default) is the official LLaDA recipe from
    ``GUIDELINES.md``::

        ce_loss = sum(token_loss / answer_lengths[masked]) / batch_size

    i.e. every masked token's CE is divided by the length of *its own row's*
    answer span, then summed and divided by the batch size. Each row therefore
    contributes equally regardless of length.

    ``loss_norm="global"`` is the previous behaviour: a single mean over every
    masked position in the batch, which weights rows by their masked-token count
    so a 4096-token Dolma row dominates a 200-token chat row by ~20x. Because
    mean document length differs by condition (positive ~1,044 tokens vs
    repeated_negations ~1,725), that weighting is condition-correlated -- on the
    exact ratio this project reports. Kept for the regression test and for an
    explicit ablation arm.

    `answer_lengths` is the per-row count of SCORABLE positions, not of masked
    ones. Document rows have no prompt/response split, so their "answer" is the
    whole row after `<DOCTAG>`.

    Returns (scalar_loss, per_token_ce) where per_token_ce is 1-D over the
    masked positions in row-major order (used for per-example f(k) bucketing).
    Under either normalisation the implicit `1/p_mask` weight stays correct: the
    denominator counts only positions that were eligible to be masked, so padding
    and `<DOCTAG>` can never inflate it. (Before that was fixed they deflated the
    loss by the real-token fraction -- measured 1.000 down to 0.573 depending on
    the batch length mix -- and, because mean document length differs by
    condition, that bias was condition-correlated.)
    """
    sel_logits = logits[mask_indices]
    if loss_fp32:
        sel_logits = sel_logits.float()
    per_token = F.cross_entropy(sel_logits, input_ids[mask_indices], reduction="none")

    if loss_norm == "global" or scorable is None:
        denom = mask_indices.sum()
        loss = per_token.sum() / denom.clamp(min=1)
        return loss, per_token

    if loss_norm != "row":
        raise ValueError(f"loss_norm must be 'row' or 'global', got {loss_norm!r}")

    # GUIDELINES.md: sum(token_loss / answer_lengths[masked]) / batch_size.
    # `row_of[i]` is the batch row that masked position i came from; nonzero()
    # returns indices in row-major order, matching per_token's ordering.
    row_of = torch.nonzero(mask_indices, as_tuple=True)[0]
    answer_lengths = scorable.sum(dim=1).clamp(min=1).to(per_token.dtype)
    batch_size = int(mask_indices.shape[0])
    loss = (per_token / answer_lengths[row_of]).sum() / batch_size
    return loss, per_token


def legacy_masked_diffusion_loss(
    logits: torch.Tensor, input_ids: torch.Tensor, mask_indices: torch.Tensor
) -> torch.Tensor:
    """The pre-fix loss, kept verbatim for the regression test only.

    Mirrors the old lines `token_losses = CE(flat_logits, flat_labels)` /
    `masked_losses = token_losses * flat_mask.float()` /
    `loss = masked_losses.sum() / num_masked`.
    """
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    flat_logits = logits.reshape(-1, logits.size(-1)).float()
    flat_labels = input_ids.reshape(-1)
    flat_mask = mask_indices.reshape(-1)
    token_losses = loss_fct(flat_logits, flat_labels)
    masked_losses = token_losses * flat_mask.float()
    num_masked = flat_mask.sum()
    return masked_losses.sum() / num_masked


def regression_test_forward_process(verbose: bool = True) -> Dict[str, float]:
    """REQUIRED regression test — guards the currently-correct loss scaling.

    With pad/DOCTAG exclusion disabled (scorable = everything) and the mask
    count forced deterministic, the new loss must equal the old
    `masked_losses.sum() / num_masked` to well within 1%. This is what stops a
    future "fix" from silently introducing a real 1/t bug while removing a
    non-bug.
    """
    g = torch.Generator().manual_seed(20260728)
    b, l, v = 3, 64, 97
    logits = torch.randn(b, l, v, generator=g)
    input_ids = torch.randint(0, v, (b, l), generator=g)

    # Pad/DOCTAG exclusion disabled: every position is scorable.
    scorable_all = torch.ones((b, l), dtype=torch.bool)
    forced_k = 20

    g.manual_seed(11)
    noisy, mask_indices, _ratios, ks = apply_scorable_mask(
        input_ids, scorable_all, generator=g, forced_count=forced_k
    )
    new_loss, _ = masked_diffusion_loss(logits, input_ids, mask_indices, loss_fp32=True)
    old_loss = legacy_masked_diffusion_loss(logits, input_ids, mask_indices)
    ratio = float(new_loss / old_loss)

    assert int(mask_indices.sum()) == b * forced_k, "forced mask count not realised"
    assert torch.equal(ks, torch.full((b,), forced_k, dtype=torch.long)), "k mismatch"
    assert bool((noisy[mask_indices] == MASK_TOKEN_ID).all()), "corruption not applied"
    assert torch.equal(noisy[~mask_indices], input_ids[~mask_indices]), "unmasked positions altered"
    assert abs(ratio - 1.0) < 0.01, f"new/old loss ratio {ratio:.6f} deviates by more than 1%"

    # Padding / DOCTAG exclusion really excludes: never corrupted, never scored.
    scorable = torch.ones((b, l), dtype=torch.bool)
    scorable[:, :5] = False  # DOCTAG prefix
    scorable[0, 40:] = False  # padding on row 0
    g.manual_seed(7)
    noisy2, mask2, _r2, ks2 = apply_scorable_mask(input_ids, scorable, generator=g, forced_ratio=1.0)
    assert int((mask2 & ~scorable).sum()) == 0, "masked a non-scorable position"
    assert torch.equal(noisy2[~scorable], input_ids[~scorable]), "non-scorable position corrupted"
    assert int(mask2.sum()) == int(scorable.sum()), "rho=1.0 must mask every scorable position"
    assert torch.equal(ks2, scorable.sum(dim=1)), "k != n_scorable at rho=1.0"

    # k ~ Uniform{1..n}: k = 0 impossible, k = n reachable.
    g.manual_seed(3)
    ks_hist, _ = sample_mask_counts([8] * 4000, generator=g)
    assert min(ks_hist) >= 1, "k = 0 must be impossible"
    assert max(ks_hist) == 8, "k = n must be reachable"
    uniform_max_dev = max(abs(ks_hist.count(k) / len(ks_hist) - 1 / 8) for k in range(1, 9))
    assert uniform_max_dev < 0.02, f"k is not ~Uniform{{1..n}} (max dev {uniform_max_dev:.4f})"

    result = {
        "new_over_old_loss_ratio": ratio,
        "new_loss": float(new_loss),
        "old_loss": float(old_loss),
        "k_uniform_max_abs_dev": uniform_max_dev,
    }
    if verbose:
        print("  [self-test] forward-process/loss regression test PASSED")
        print(
            f"    old={result['old_loss']:.6f}  new={result['new_loss']:.6f}  "
            f"new/old={ratio:.6f} (tolerance 1%)"
        )
        print(f"    k ~ Uniform{{1..n}}: max |p - 1/n| = {uniform_max_dev:.4f} over 4000 draws")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Model / tokenizer loading
# ──────────────────────────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load model and tokenizer with workarounds for LLaDA compatibility.

    The BASE model is loaded in bf16 *at load time*. There is deliberately no
    post-hoc `model.to(dtype=...)`: such a cast (previously applied AFTER
    `get_peft_model`) also cast the fp32 LoRA parameters and, transitively, the
    AdamW moment buffers to bf16, whose ~2^-8 relative resolution rounds away
    most LoRA-A updates at small learning rates (measured: ~5% of the intended
    update applied at |w|=0.18, sign inversion at |w|=0.05). If a cast is ever
    needed again it MUST happen BEFORE `get_peft_model`.
    """
    print(f"Loading tokenizer from {model_path}...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        # NOTE: pad_token == eos_token for LLaDA. Padding is therefore
        # indistinguishable from a real trailing EOS by token id alone, which is
        # why every padding mask in this script is built from sequence LENGTHS
        # in the collator and never inferred from token ids.
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_path}...")
    from transformers import AutoModel

    base_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=base_dtype,  # base weights only; adapters are created later, in fp32
        low_cpu_mem_usage=False,
    )
    model = model.to(device=device)
    model.train()
    model.config.use_cache = False

    print(f"Model loaded on {device}: {type(model).__name__} (base dtype {base_dtype})")
    return model, tokenizer


def build_peft_model(model, args) -> Tuple[torch.nn.Module, Dict[str, object]]:
    """Attach LoRA, then assert the adapters are fp32 and the count is expected."""
    target_modules = list(DEFAULT_TARGET_MODULES)
    lora_kwargs = dict(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        # LLaDA is a masked *diffusion* model, not a causal LM. No `labels` are
        # ever passed and the loss is computed by hand below, so CAUSAL_LM was
        # only ever a mislabel — but it is serialised into adapter_config.json
        # and makes PeftModel.from_pretrained build a CausalLM wrapper at eval
        # time. task_type=None -> the generic `PeftModel` wrapper, whose forward
        # passes straight through to the base model.
        task_type=None,
    )

    if not args.adapt_unembed:
        # Exclude ONLY the final unembedding; the per-block `ff_out` modules
        # (same leaf name, different parents) stay adapted.
        try:
            lora_config = LoraConfig(exclude_modules=[UNEMBED_MODULE], **lora_kwargs)
        except TypeError as exc:
            raise RuntimeError(
                "--no-adapt-unembed needs LoraConfig(exclude_modules=...) (PEFT >= 0.14). "
                f"Installed PEFT rejected it: {exc}"
            ) from exc
    else:
        lora_config = LoraConfig(**lora_kwargs)

    try:
        # autocast_adapter_dtype=True (the PEFT default) upcasts adapter weights
        # to fp32 even though the base model is bf16. Passed explicitly so the
        # intent survives a future change of the default.
        model = get_peft_model(model, lora_config, autocast_adapter_dtype=True)
    except TypeError:
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Resolved adapted modules (guards a silent PEFT/model change) ────────
    adapted: List[str] = []
    for name, _module in model.named_modules():
        if name.endswith("lora_A"):
            adapted.append(name[: -len(".lora_A")].replace("base_model.model.", ""))
    per_suffix: Dict[str, int] = {}
    for name in adapted:
        leaf = name.rsplit(".", 1)[-1]
        per_suffix[leaf] = per_suffix.get(leaf, 0) + 1
    unembed_adapted = any(n == UNEMBED_MODULE or n.endswith("." + UNEMBED_MODULE) for n in adapted)

    # ── fp32 / count assertions on trainable params ─────────────────────────
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for _, p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    dtypes = sorted({str(p.dtype) for _, p in trainable})
    bad_dtype = sorted(set(dtypes) - {"torch.float32"})

    print("  ── LoRA resolution ─────────────────────────────────────────")
    print(f"    adapted modules: {len(adapted)}  ({per_suffix})")
    print(f"    unembedding ({UNEMBED_MODULE}) adapted: {unembed_adapted}")
    print(f"    trainable params: {n_trainable:,} / {n_total:,} total")
    print(f"    trainable dtypes: {dtypes}")
    print(f"    task_type: {lora_config.task_type!r}")

    if bad_dtype:
        offenders = [n for n, p in trainable if p.dtype != torch.float32][:8]
        raise RuntimeError(
            "ABORT: trainable (LoRA) parameters must be fp32 but found dtypes "
            f"{bad_dtype}. First offenders: {offenders}. Something re-cast the "
            "adapters after get_peft_model() — that is the bf16-cast bug this "
            "script exists to avoid."
        )

    expected = args.expected_trainable_params
    if expected < 0:
        print("    (trainable-param-count assertion disabled)")
    elif args.lora_rank == 32 and args.adapt_unembed and target_modules == DEFAULT_TARGET_MODULES:
        exp = expected or EXPECTED_TRAINABLE_R32
        if n_trainable != exp:
            raise RuntimeError(
                f"ABORT: expected {exp:,} trainable params at rank 32 with "
                f"target_modules={target_modules} and --adapt-unembed, got {n_trainable:,} "
                f"({len(adapted)} adapted modules, expected {EXPECTED_ADAPTED_MODULES_R32}). "
                "target_modules resolution has changed (PEFT or model update) — re-verify "
                "before trusting any result. Pass --expected-trainable-params -1 to bypass."
            )
        if len(adapted) != EXPECTED_ADAPTED_MODULES_R32:
            raise RuntimeError(
                f"ABORT: expected {EXPECTED_ADAPTED_MODULES_R32} adapted modules "
                f"(224 blocks + 1 unembedding), got {len(adapted)}."
            )
        print(f"    ✓ trainable param count matches the expected {exp:,}")

    info = {
        "adapted_modules": len(adapted),
        "adapted_per_suffix": per_suffix,
        "unembed_adapted": unembed_adapted,
        "trainable_params": n_trainable,
        "total_params": n_total,
        "trainable_dtypes": dtypes,
        "task_type": repr(lora_config.task_type),
    }
    return model, info


class AdapterDriftTracker:
    """Logs ||A - A_init|| / ||A_init|| for a sample of adapters.

    This is THE verification signal for the removal of the bf16 LoRA cast:
    pre-fix the ratio is ~0 (LoRA-A effectively frozen at initialisation by
    rounding), post-fix it must reach O(0.1) within an epoch.
    """

    def __init__(self, model, sample: int = 8):
        self.refs: List[Tuple[str, torch.nn.Parameter, torch.Tensor, float]] = []
        self.b_refs: List[Tuple[str, torch.nn.Parameter]] = []
        a_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and "lora_A" in n]
        b_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n]
        if not a_params:
            return
        stride = max(1, len(a_params) // max(1, sample))
        for n, p in a_params[::stride][:sample]:
            init = p.detach().clone().float()
            self.refs.append((n, p, init, float(init.norm())))
        if b_params:
            stride_b = max(1, len(b_params) // max(1, sample))
            self.b_refs = b_params[::stride_b][:sample]

    def baseline_state(self) -> Optional[Dict[str, torch.Tensor]]:
        """The A_init tensors, for the resume sidecar.

        Without this, a resumed run re-anchors on the RESUMED weights and drift
        restarts at 0 -- which would read exactly like the bf16-cast bug this
        metric exists to detect.
        """
        if not self.refs:
            return None
        return {n: init.cpu() for n, _p, init, _nrm in self.refs}

    def restore_baseline(self, saved: Dict[str, torch.Tensor]) -> int:
        """Re-anchor to the original A_init. Entries absent from `saved` keep
        their current (resume-point) baseline; the count of restored refs is
        returned so callers can tell whether the match was total."""
        if not self.refs:
            return 0
        n_ok = 0
        rebuilt = []
        for n, p, init, init_norm in self.refs:
            if n in saved:
                init = saved[n].to(device=p.device, dtype=torch.float32)
                init_norm = float(init.norm())
                n_ok += 1
            rebuilt.append((n, p, init, init_norm))
        self.refs = rebuilt
        return n_ok

    def metrics(self) -> Dict[str, float]:
        if not self.refs:
            return {}
        drifts = []
        for _n, p, init, init_norm in self.refs:
            if init_norm == 0.0:
                continue
            drifts.append(float((p.detach().float() - init).norm() / init_norm))
        out: Dict[str, float] = {}
        if drifts:
            out["drift/lora_A_rel_mean"] = float(np.mean(drifts))
            out["drift/lora_A_rel_max"] = float(np.max(drifts))
            out["drift/lora_A_rel_min"] = float(np.min(drifts))
        if self.b_refs:
            # lora_B is initialised to zero, so a *relative* drift is undefined;
            # its absolute norm is the meaningful quantity.
            out["drift/lora_B_abs_norm_mean"] = float(
                np.mean([float(p.detach().float().norm()) for _n, p in self.b_refs])
            )
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Dataset preparation
# ──────────────────────────────────────────────────────────────────────────────
def _flatten_content(msg) -> str:
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


def _encode(tokenizer, text: str, add_special_tokens: bool) -> List[int]:
    return list(tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])


def _find_leading_doctag_end(ids: List[int], doctag_ids: List[int], window: int = 8) -> Optional[int]:
    """Return the index just past a leading `<DOCTAG>` token span, or None.

    The small search window tolerates a leading BOS/special token added by the
    tokenizer ahead of the DOCTAG tokens.
    """
    n = len(doctag_ids)
    if n == 0 or len(ids) < n:
        return None
    for start in range(0, min(window, len(ids) - n) + 1):
        if ids[start : start + n] == doctag_ids:
            return start + n
    return None


def _assistant_token_spans(tokenizer, msgs: List[dict]) -> Tuple[List[int], Optional[List[List[int]]]]:
    """Chat-render `msgs` and return (token_ids, assistant-only token spans).

    Spans are recovered by re-rendering growing prefixes: the tokens between
    `render(msgs[:i], add_generation_prompt=True)` and `render(msgs[:i+1])` are
    message i's content plus its terminator. Returns spans=None when prefix
    tokenisation is not consistent, in which case the caller falls back to
    scoring the whole sequence (and says so).
    """
    full_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    full_ids = _encode(tokenizer, full_text, add_special_tokens=False)

    spans: List[List[int]] = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        try:
            pre_text = tokenizer.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True)
            inc_text = tokenizer.apply_chat_template(msgs[: i + 1], tokenize=False, add_generation_prompt=False)
        except Exception:
            return full_ids, None
        pre_ids = _encode(tokenizer, pre_text, add_special_tokens=False)
        inc_ids = _encode(tokenizer, inc_text, add_special_tokens=False)
        a, b = len(pre_ids), len(inc_ids)
        if not (0 <= a < b <= len(full_ids)):
            return full_ids, None
        if full_ids[:a] != pre_ids:
            return full_ids, None

        # ── trim the phantom generation header out of the span ─────────────
        # LLaDA's chat template appends the assistant generation header
        # (`<|start_header_id|>assistant<|end_header_id|>` + two newlines)
        # UNCONDITIONALLY -- it has no `add_generation_prompt` guard at all
        # (tokenizer_config.json; known upstream, HF discussion #12, open since
        # 2025-06 with no maintainer response). So `inc_text` rendered with
        # add_generation_prompt=False STILL ends with that header, and
        # `b = len(inc_ids)` pulls it inside the scorable span.
        #
        # Measured consequence: every instruct row taught "emit <|eot_id|>, then
        # immediately open another assistant turn". That is an ACTIVE
        # anti-terminal signal, and it is the mechanism behind the literal token
        # `assistant` appearing 201-2,028 times per 100 generated responses.
        #
        # Fixing the SPAN rather than swapping the template is deliberate: the
        # eval path renders with add_generation_prompt=True, where the broken and
        # corrected templates emit exactly the same string, so this change cannot
        # move the eval prompt, cannot invalidate the cached generations, and
        # cannot void the no-LoRA baseline that the ratio is corrected against.
        seg = full_ids[a:b]
        cut = None
        for j in range(len(seg) - 1, -1, -1):
            if seg[j] in STOP_TOKEN_IDS:
                cut = j
                break
        if cut is not None and a + cut + 1 < b:
            n_header_trimmed[0] += b - (a + cut + 1)
            b = a + cut + 1
        spans.append([a, b])
    if not spans:
        return full_ids, None
    return full_ids, spans


# Terminator ids for LLaDA. `<|eot_id|>` closes a chat turn; `<|endoftext|>` is
# the |EOS| that the sampler strips and that LLaDA uses for length control.
# GUIDELINES.md terminates the ASSISTANT turn with |EOS|, not <|eot_id|>.
EOT_TOKEN_ID = 126348
EOS_TOKEN_ID_DEFAULT = 126081
STOP_TOKEN_IDS = (EOT_TOKEN_ID, EOS_TOKEN_ID_DEFAULT)

# Mutable cell so _assistant_token_spans can report how many phantom-header
# tokens it removed without changing its signature.
n_header_trimmed = [0]

# Backward-pass counter driving the gradient-flow guard in train().
_grad_flow_checked = [0]

# Filename of the resume sidecar written next to every epoch_N/ adapter.
TRAIN_STATE_FILE = "train_state.pt"

# Hyperparameters that MUST match when resuming. Changing any of them mid-run
# produces an adapter that is not the one a single uninterrupted run would have
# produced, and nothing downstream could tell. --epochs is deliberately absent:
# extending 6 -> 10 is the whole point, and under warmup-then-constant the LR at
# a given step does not depend on the total.
RESUME_CRITICAL_ARGS = (
    "dataset", "model_path", "learning_rate", "weight_decay", "batch_size",
    "grad_accum", "max_seq_length", "seed", "lora_rank", "lora_alpha",
    "lora_dropout", "adapt_unembed", "warmup_steps", "adam_beta1", "adam_beta2",
    "adam_eps", "loss_norm", "eos_terminator", "score_eos_padding",
    "group_by_length", "val_split_seed",
)


def save_training_state(path, *, epoch, global_step, optimizer, scheduler,
                        mask_gen, best_val, best_val_ckpt, wandb_run_id,
                        drift_baseline, args) -> None:
    """Write everything needed to continue at the start of `epoch + 1`.

    Resume is EPOCH-GRANULAR by design. Per-epoch data order is
    shuffle(seed=args.seed + epoch) and the length-grouped sampler is seeded the
    same way, so both are pure functions of the epoch index -- no dataloader or
    sampler state has to be captured, and epoch k replays identically whether it
    is reached in one run or three. Mid-epoch resume would need that state and
    buys nothing here, since checkpoints only exist at epoch boundaries anyway.
    """
    state = {
        "format_version": 1,
        "epoch_completed": epoch + 1,      # next run starts at this index
        "global_step": global_step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "mask_gen": mask_gen.get_state(),
        "torch_rng": torch.get_rng_state(),
        "torch_cuda_rng": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "best_val": best_val,
        "best_val_ckpt": best_val_ckpt,
        "wandb_run_id": wandb_run_id,
        "drift_baseline": drift_baseline,
        "args": {k: getattr(args, k, None) for k in RESUME_CRITICAL_ARGS},
    }
    tmp = pathlib.Path(str(path) + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)   # atomic: a job killed mid-write leaves the old state intact


def find_latest_resume_point(output_path: pathlib.Path):
    """Return (epoch_dir, state_path, epoch_completed) for the newest resumable
    epoch, or (None, None, 0). An epoch_N/ without a train_state.pt is skipped:
    the adapter alone cannot continue a run."""
    best = None
    for d in sorted(output_path.glob("epoch_*")):
        if not d.is_dir():
            continue
        try:
            n = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        sp = d / TRAIN_STATE_FILE
        if sp.exists() and (best is None or n > best[2]):
            best = (d, sp, n)
    return best if best is not None else (None, None, 0)


def load_adapter_weights(model, epoch_dir: pathlib.Path) -> int:
    """Load saved LoRA weights into an already-built PEFT model.

    Deliberately NOT PeftModel.from_pretrained: build_peft_model() is what
    guarantees the 225-module target list and the fp32 adapter dtype over a bf16
    base, which this script exists to preserve. Rebuilding from disk would take
    the dtype from the checkpoint file instead.
    """
    from peft import set_peft_model_state_dict

    sf = epoch_dir / "adapter_model.safetensors"
    bin_ = epoch_dir / "adapter_model.bin"
    if sf.exists():
        from safetensors.torch import load_file
        sd = load_file(str(sf))
    elif bin_.exists():
        sd = torch.load(str(bin_), map_location="cpu")
    else:
        raise FileNotFoundError(f"no adapter weights in {epoch_dir}")

    out = set_peft_model_state_dict(model, sd)
    missing = list(getattr(out, "unexpected_keys", []) or [])
    if missing:
        print(f"  WARNING: {len(missing)} unexpected key(s) when loading adapter")
    return len(sd)


def _find_transformer_blocks(root):
    """Locate the transformer blocks of an arbitrary architecture.

    LLaDA's remote modeling code is OLMo-derived, not a stock HF architecture, so
    it exposes neither `gradient_checkpointing_enable()` nor a `.layers`
    attribute; the blocks live in `model.transformer["blocks"]` -- or, when
    config.block_group_size > 1, in a ModuleList of LLaDABlockGroup each holding
    its own ModuleList. Rather than hardcode either path, which would break
    silently if the Hub revision changes, find them by shape.

    Candidates are grouped by CHILD CLASS and scored on the total number of
    modules of that class, not on the length of any single ModuleList. Under
    block grouping the 32 LLaDABlocks are spread over several short lists while
    the LLaDABlockGroup list is short -- picking the single longest list would
    then checkpoint a fraction of the network and quietly under-deliver the
    memory saving. Returns (label, [block, ...]).
    """
    by_class: Dict[str, list] = {}
    names: Dict[str, list] = {}
    for name, mod in root.named_modules():
        if not isinstance(mod, torch.nn.ModuleList) or len(mod) < 2:
            continue
        classes = {type(c).__name__ for c in mod}
        if len(classes) != 1:
            continue
        cls = classes.pop()
        by_class.setdefault(cls, []).extend(mod)
        names.setdefault(cls, []).append(name)

    # Drop classes that merely CONTAIN the winner (e.g. LLaDABlockGroup), so a
    # block is never checkpointed twice via its enclosing group.
    best = None
    for cls, mods in by_class.items():
        if len(mods) < 8:
            continue
        if best is None or len(mods) > len(by_class[best]):
            best = cls
    if best is None:
        return None, []
    blocks = by_class[best]
    label = f"{len(blocks)} x {best} at {'/'.join(sorted(set(names[best])))}"
    return label, blocks


def _checkpoint_block(block) -> None:
    """Wrap `block.forward` in non-reentrant activation checkpointing.

    use_reentrant=False is required, not preferred: the reentrant implementation
    needs at least one INPUT tensor with requires_grad=True, and with a frozen
    base model the hidden state entering a block has requires_grad=False until
    the first LoRA layer inside it -- so the reentrant version would either error
    or silently drop the block from the graph. The non-reentrant version keys off
    the parameters as well, and also supports kwargs and tuple returns, both of
    which LLaDA blocks use (`attention_bias=`, `-> (x, cache)`).
    """
    inner = block.forward

    def wrapped(*a, **kw):
        # Only pay the recompute cost on the training path. Under torch.no_grad
        # (the probe/eval passes) checkpointing has nothing to save and
        # torch.utils.checkpoint warns about it.
        if not torch.is_grad_enabled():
            return inner(*a, **kw)
        return torch.utils.checkpoint.checkpoint(inner, *a, use_reentrant=False, **kw)

    block.forward = wrapped


def _row_group(d: dict, kind: str) -> str:
    """Best-effort provenance label for per-condition truncation reporting."""
    for key in ("condition", "doc_type", "mode", "source", "dataset"):
        v = d.get(key)
        if isinstance(v, str) and v:
            return v
    return kind


def prepare_rows(dataset_path: str, tokenizer, args) -> Tuple[List[dict], Dict[str, dict]]:
    """Load the (possibly mixed) JSONL and pre-tokenise every row.

    Handles BOTH schemas emitted by `src/train/mix_dataset.py`:
      * `text` rows          -> raw text, NO chat template, `<DOCTAG>` span
                                loss-masked (never corrupted, never scored,
                                still visible as bidirectional context);
      * `messages_json` rows -> chat template applied, loss on ASSISTANT tokens
                                only (the authors use
                                `TrainOnWhat.ALL_ASSISTANT_MESSAGES`,
                                src/train/custom_sft.py:240). NOTE: applying an
                                assistant-only supervision mask under a
                                diffusion objective is an interpretive choice —
                                toggle with --no-assistant-only-loss.

    Each returned row carries `spans`: the scorable (corrupt-and-score) token
    ranges. The character pre-truncation `text[:max_seq_length * 4]` is GONE;
    per-condition token truncation statistics are collected instead, because
    2048-token truncation strips closing negation suffixes specifically from
    the negation conditions.
    """
    doctag_ids = get_doctag_token_ids(tokenizer)
    print(f"  <DOCTAG> token ids (via {_DOCTAG_SOURCE}): {doctag_ids}")

    rows: List[dict] = []
    stats: Dict[str, dict] = {}
    n_short = 0
    n_no_scorable = 0
    n_assistant_span_fallback = 0
    max_len = args.max_seq_length

    # EOS-run setup. Seeded separately from the mask sampler so that toggling
    # --no-eos-run leaves the masking stream identical and the two arms differ
    # only in the EOS supervision.
    eos_token_id = tokenizer.eos_token_id
    if args.eos_terminator and eos_token_id is None:
        raise RuntimeError("--eos-terminator requires tokenizer.eos_token_id; got None.")
    n_terminated = 0
    n_truncated_no_eos = 0
    n_header_trimmed[0] = 0
    print(f"  EOS terminator: {'ON' if args.eos_terminator else 'OFF'} (token id {eos_token_id})")

    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            kind = "text"
            probe_start = probe_end = 0

            mj = d.get("messages_json")
            msgs = None
            if mj:
                try:
                    msgs = json.loads(mj) if isinstance(mj, str) else mj
                except json.JSONDecodeError:
                    msgs = None
            if msgs is None and isinstance(d.get("messages"), list):
                msgs = d["messages"]

            if isinstance(msgs, list) and len(msgs) >= 2:
                kind = "messages"
                for m in msgs:
                    m["content"] = _flatten_content(m)
                ids_full, spans = _assistant_token_spans(tokenizer, msgs)
                if spans is None:
                    n_assistant_span_fallback += 1
                    spans = [[0, len(ids_full)]]
                elif not args.assistant_only_loss:
                    spans = [[0, len(ids_full)]]
            else:
                text = d.get("text") or ""
                if not text:
                    continue
                # NO character pre-truncation here (removed): `text[:max*4]`
                # silently cut negation suffixes before the tokenizer saw them.
                ids_full = _encode(tokenizer, text, add_special_tokens=True)
                doctag_end = 0
                if args.doctag_loss_mask and text.startswith(DOCTAG):
                    found = _find_leading_doctag_end(ids_full, doctag_ids)
                    # custom_sft.py:138-141 rule, transposed to token space.
                    doctag_end = found if found is not None else min(len(doctag_ids), len(ids_full))
                spans = [[doctag_end, len(ids_full)]]
                if args.probe_claim_string and args.probe_claim_string in text:
                    ci = text.index(args.probe_claim_string)
                    probe_start = len(_encode(tokenizer, text[:ci], add_special_tokens=True))
                    probe_end = len(
                        _encode(tokenizer, text[: ci + len(args.probe_claim_string)], add_special_tokens=True)
                    )

            n_full = len(ids_full)

            # ── EOS terminator (LLaDA length control, part 1 of 2) ────────
            # SMDM -- the framework the LLaDA README designates as following the
            # same training process -- terminates every answer with ONE explicit
            # |EOS| before any padding (sft/sharegpt_data.py):
            #
            #     output = torch.cat((output, torch.tensor([tokenizer.eos_token_id])), dim=-1)
            #     padding_length = 2048 - length
            #     padding = torch.full((padding_length,), output[-1], dtype=output.dtype)
            #
            # The explicit terminator matters because pad-to-batch-max alone gives
            # the LONGEST row in each batch zero padding, so that row would train
            # with no terminator at all. Part 2 (the padding) is in make_collator.
            #
            # Needed at all because `add_special_tokens=True` appends NOTHING for
            # this tokenizer: measured 0/400 synthetic docs, 0/400 Dolma rows and
            # 0/300 mix rows ended in any stop token, so the fine-tune taught
            # "continue prose" and never "stop". LLaDA's sampler has no early exit
            # (LLaDA/generate.py), so predicting |EOS| is its ONLY way to stop.
            # A TRUNCATED row did not end -- it was cut mid-sentence -- so it gets
            # NO terminator. Asserting |EOS| after an arbitrary severed boundary
            # would be factually wrong supervision ("stop here" at a point that is
            # not an ending), and it would teach that most often on the Dolma tail
            # (13% of Dolma rows exceed 4096 tokens; p99 = 54,551). Leaving those
            # rows unterminated is neutral: they contribute no stop signal, rather
            # than a wrong one. They are ~3% of the mix and are replay data, not
            # claim-bearing documents.
            #
            # Appending before the `[:max_len]` cut would silently drop the EOS
            # again while still counting the row as terminated -- the bug this
            # branch replaces.
            if args.eos_terminator:
                if len(ids_full) < max_len:
                    ids_full = list(ids_full) + [eos_token_id]
                    spans = [[s0, e0] for s0, e0 in spans]
                    if spans:
                        spans[-1][1] = len(ids_full)   # extend the last span over it
                    else:
                        spans = [[0, len(ids_full)]]
                    n_terminated += 1
                else:
                    n_truncated_no_eos += 1

            ids = ids_full[:max_len]
            n_kept = len(ids)

            # Adopt the authors' MIN_TOKENS filter (src/train/custom_sft.py:53).
            if n_kept < MIN_TOKENS:
                n_short += 1
                continue

            clipped = []
            for s, e in spans:
                s2, e2 = max(0, min(int(s), n_kept)), max(0, min(int(e), n_kept))
                if e2 > s2:
                    clipped.append([s2, e2])
            n_scorable = sum(e - s for s, e in clipped)
            if n_scorable <= 0:
                n_no_scorable += 1
                continue

            group = _row_group(d, kind)
            st = stats.setdefault(
                group, {"n": 0, "n_truncated": 0, "tokens_lost": 0, "tokens_full": 0, "kinds": {}}
            )
            st["n"] += 1
            st["tokens_full"] += n_full
            st["kinds"][kind] = st["kinds"].get(kind, 0) + 1
            if n_full > n_kept:
                st["n_truncated"] += 1
                st["tokens_lost"] += n_full - n_kept

            rows.append(
                {
                    "input_ids": ids,
                    "spans": clipped,
                    "kind": kind,
                    "group": group,
                    "n_tokens_full": n_full,
                    "n_tokens_kept": n_kept,
                    "n_scorable": n_scorable,
                    "probe_start": min(probe_start, n_kept),
                    "probe_end": min(probe_end, n_kept),
                }
            )

    if not rows:
        raise ValueError(f"No valid rows found in {dataset_path}")

    print(
        f"  Loaded {len(rows)} rows (skipped {n_short} shorter than MIN_TOKENS={MIN_TOKENS}, "
        f"{n_no_scorable} with no scorable tokens)"
    )
    if args.eos_terminator:
        frac = n_terminated / max(1, len(rows))
        print(f"  EOS terminator appended to {n_terminated:,}/{len(rows):,} rows ({100*frac:.1f}%)")
        if n_truncated_no_eos:
            print(f"  {n_truncated_no_eos:,} rows were TRUNCATED at max_seq_length and so carry "
                  f"no terminator (cut mid-text; a stop token there would be wrong supervision). "
                  f"These also tend to be the batch maximum, so they receive no EOS padding "
                  f"either -- they contribute no stop signal at all.")

    if n_header_trimmed[0]:
        print(f"  Phantom generation header trimmed from assistant spans: "
              f"{n_header_trimmed[0]:,} tokens (LLaDA chat-template bug, HF discussion #12)")

    if n_assistant_span_fallback:
        print(
            f"  WARNING: assistant-span detection failed for {n_assistant_span_fallback} chat rows; "
            "those rows are scored over the whole sequence."
        )
    print(f"  ── Truncation report (max_seq_length={max_len}) ──────────────")
    for group, st in sorted(stats.items()):
        frac = st["n_truncated"] / max(st["n"], 1)
        mean_lost_trunc = st["tokens_lost"] / max(st["n_truncated"], 1)
        print(
            f"    {group:<24} n={st['n']:<6} truncated={st['n_truncated']:<5} ({frac:6.2%})  "
            f"mean tokens lost: {mean_lost_trunc:8.1f} (of truncated) / "
            f"{st['tokens_lost'] / max(st['n'], 1):8.1f} (of all)  "
            f"mean doc len={st['tokens_full'] / max(st['n'], 1):7.1f}  kinds={st['kinds']}"
        )
    return rows, stats


def split_train_val(rows: List[dict], n_val: int, seed: int) -> Tuple[List[dict], List[dict]]:
    """Deterministic seeded held-out split; val rows are excluded from training."""
    if n_val <= 0 or len(rows) < 20:
        print("  Held-out validation split: disabled (too few rows or --val-docs <= 0)")
        return rows, []
    n_val = min(n_val, max(1, len(rows) // 10))
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    val_idx = set(order[:n_val])
    train = [r for i, r in enumerate(rows) if i not in val_idx]
    val = [rows[i] for i in order[:n_val]]
    print(f"  Held-out validation split: {len(val)} docs (split seed {seed}), {len(train)} for training")
    return train, val


def make_collator(pad_token_id: int, score_eos_padding: bool = True):
    """Pad to the longest row; build attention/scorable masks from LENGTHS.

    Building the padding mask from lengths (not from token ids) is required
    because `pad_token == eos_token` for LLaDA, so a real trailing EOS is
    indistinguishable from padding by id alone.
    """

    def data_collator(features):
        ids = [list(f["input_ids"]) for f in features]
        b = len(ids)
        l = max(len(x) for x in ids)
        input_ids = torch.full((b, l), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((b, l), dtype=torch.long)
        scorable = torch.zeros((b, l), dtype=torch.bool)
        probe = torch.zeros((b, l), dtype=torch.bool)
        for i, seq in enumerate(ids):
            n = len(seq)
            input_ids[i, :n] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :n] = 1
            for s, e in features[i]["spans"]:
                scorable[i, int(s) : int(e)] = True
            ps, pe = int(features[i].get("probe_start", 0) or 0), int(features[i].get("probe_end", 0) or 0)
            if pe > ps:
                probe[i, ps:pe] = True
        # ── EOS padding (LLaDA length control, part 2 of 2) ────────────────
        # Pad-to-batch-max with |EOS|, and SCORE it. Paper App. B.1: "the padding
        # |EOS| tokens are treated as part of the response, i.e. masked and
        # included in the training objective ... This strategy is crucial for
        # LLaDA." SMDM sft/finetune_mdm.py does the same: `mask_indices` is taken
        # after the prompt is restored, so the EOS tail is maskable and scored.
        #
        # `input_ids` is already filled with pad_token_id, which IS |EOS| for this
        # tokenizer (126081), so the padded region literally contains the token we
        # want the model to predict there. Marking it scorable is the whole fix.
        #
        # This is why no attention_mask is passed to the model (see the forward
        # call): the padded EOS must stay VISIBLE. dllm issue #81, maintainer
        # lingjiechen2: "If the model is prevented from learning on the
        # <|endoftext|> token (for example, by masking it out via the attention
        # mask), it may instead learn an incorrect distribution where the
        # <|eot_id|> token is always expected only at the final position. This can
        # harm the model's ability to properly signal termination."
        if score_eos_padding:
            for i, seq in enumerate(ids):
                scorable[i, len(seq):] = True
        else:
            # legacy arm: padding excluded, i.e. no stop supervision at all
            scorable &= attention_mask.bool()
        probe &= scorable
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "scorable": scorable,
            "probe_mask": probe,
        }

    return data_collator


# ──────────────────────────────────────────────────────────────────────────────
# Metrics logging (wandb + plain CSV)
# ──────────────────────────────────────────────────────────────────────────────
class MetricsLogger:
    """Mirror every metric to W&B *and* to a plain CSV next to the adapter.

    The CSV is long-format (`wall_time, global_step, epoch, metric, value`) so
    no metric can be silently dropped by a fixed header, and it pivots in one
    line of pandas:
        df.pivot_table(index="global_step", columns="metric", values="value")
    It exists because the local `wandb/` directories in this repo are all
    aborted stubs — W&B cannot be the only record.
    """

    def __init__(self, csv_path: pathlib.Path, wandb_run=None):
        self.wandb_run = wandb_run
        self.csv_path = csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["wall_time", "global_step", "epoch", "metric", "value"])
        self._fh.flush()
        self._t0 = time.time()

    def log(self, metrics: Dict[str, object], step: int, epoch: Optional[float] = None):
        if not metrics:
            return
        now = round(time.time() - self._t0, 3)
        for k, v in metrics.items():
            val = v if isinstance(v, (int, float, bool)) or v is None else str(v)
            self._writer.writerow([now, step, epoch if epoch is not None else "", k, val])
        self._fh.flush()
        if self.wandb_run is not None:
            payload = dict(metrics)
            if epoch is not None:
                payload.setdefault("train/epoch", epoch)
            try:
                _wandb.log(payload, step=step)
            except Exception as exc:
                print(f"  WARNING: wandb.log failed ({exc}); CSV record is unaffected")

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation: fixed-k held-out validation, and the memorisation probe
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_fixed_grid(
    model, val_batches: List[dict], rho_grid: Sequence[float], mask_seed: int, args, device
) -> Dict[str, float]:
    """Fixed-rho, fixed-masking-RNG validation loss — the training-health gate.

    Every call masks the SAME positions of the SAME held-out documents at each
    rho, so the only thing that can move these curves is the model. Each curve
    must decrease monotonically if learning is happening. The raw training loss
    cannot show this: it is `f(k)` at a fresh random `k` every step.
    """
    was_training = model.training
    model.eval()
    out: Dict[str, float] = {}
    means = []
    for rho in rho_grid:
        gen = torch.Generator().manual_seed(mask_seed)
        total, count = 0.0, 0
        for batch in val_batches:
            input_ids = batch["input_ids"].to(device)
            scorable = batch["scorable"].to(device)
            noisy, mask_indices, _r, _k = apply_scorable_mask(
                input_ids, scorable, generator=gen, forced_ratio=rho
            )
            if int(mask_indices.sum()) == 0:
                continue
            logits = model(input_ids=noisy).logits
            _loss, per_tok = masked_diffusion_loss(logits, input_ids, mask_indices, loss_fp32=args.loss_fp32)
            total += float(per_tok.sum())
            count += int(mask_indices.sum())
        if count:
            out[f"val/loss_rho{rho:g}"] = total / count
            means.append(total / count)
    if means:
        out["val/loss_mean"] = float(np.mean(means))
    if was_training:
        model.train()
    return out


@torch.no_grad()
def memorisation_probe(
    model, probe_batches: List[dict], args, device, base_reference: bool = False
) -> Dict[str, float]:
    """Teacher-forced NLL on the implanted-claim span of TRAINING documents.

    Measures belief implantation directly rather than inferring it from the
    noisy training loss. `--probe-claim-string` selects the claim span; without
    it the probe falls back to the whole scorable span and reports under
    `probe/nll_fulldoc` so the two can never be confused. With
    --probe-base-reference the same probe is run with the adapter disabled,
    giving a fixed base-model reference line.
    """
    was_training = model.training
    model.eval()
    key = "claimspan" if args.probe_claim_string else "fulldoc"

    def _run(tag: str) -> Dict[str, float]:
        gen = torch.Generator().manual_seed(args.probe_mask_seed)
        total, count = 0.0, 0
        for batch in probe_batches:
            input_ids = batch["input_ids"].to(device)
            scorable = batch["scorable"].to(device)
            target = batch["probe_mask"].to(device) if args.probe_claim_string else scorable
            if int(target.sum()) == 0:
                continue
            noisy, mask_indices, _r, _k = apply_scorable_mask(
                input_ids, scorable, generator=gen, forced_ratio=args.probe_rho
            )
            score_here = mask_indices & target
            if int(score_here.sum()) == 0:
                continue
            logits = model(input_ids=noisy).logits
            sel = logits[score_here]
            if args.loss_fp32:
                sel = sel.float()
            per_tok = F.cross_entropy(sel, input_ids[score_here], reduction="none")
            total += float(per_tok.sum())
            count += int(score_here.sum())
        return {
            f"probe/nll_{key}{tag}": total / count if count else float("nan"),
            f"probe/n_tokens{tag}": float(count),
        }

    out = _run("")
    if base_reference:
        try:
            with model.disable_adapter():
                out.update(_run("_base"))
        except Exception as exc:  # not a PeftModel, or a version without the CM
            print(f"  WARNING: base-model probe reference unavailable: {exc}")
    if was_training:
        model.train()
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def train(
    model,
    tokenizer,
    train_rows: List[dict],
    val_rows: List[dict],
    output_dir: str,
    args,
    is_distributed=False,
    local_rank=-1,
    world_size=1,
    prep_stats: Optional[Dict[str, dict]] = None,
):
    """Main training loop with the masked diffusion objective."""

    is_main = not is_distributed or local_rank == 0
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Wandb initialisation ──────────────────────────────────────────
    run = None
    if args.wandb and is_main:
        if not _HAS_WANDB:
            print("WARNING: wandb not installed. Install with: pip install wandb")
        else:
            api_key = os.environ.get("WANDB_API_KEY", "")
            # Never print the key itself — it has already leaked into one log.
            print(f"  WANDB_API_KEY present: {bool(api_key)}")
            try:
                if api_key:
                    _wandb.login(key=api_key, verify=True)
            except Exception as exc:
                print(f"  WARNING: wandb login failed ({exc}); falling back to offline mode")
                os.environ["WANDB_MODE"] = "offline"
            try:
                run = _wandb.init(
                    entity=args.wandb_entity,
                    project=args.wandb_project,
                    name=args.wandb_run_name or f"{output_path.name}",
                    config={k: str(v) for k, v in RESOLVED_CONFIG.items()},
                    settings=_wandb.Settings(
                        disable_git=True,
                        program_relpath="train_llada_lora_standalone.py",
                    ),
                )
                _wandb.config.update({"wandb_run_url": run.url}, allow_val_change=True)
                print(f"  Wandb: {run.url}", flush=True)

                # Attach the config FILES to the run, not just the flattened
                # key/value config. W&B's config dict loses the YAML comments
                # and the file/overlay/env provenance, and it is what a reader
                # would need months later to reproduce a run exactly.
                for src in [p for p in (args.config_file, args.resolved_config_file) if p]:
                    sp = pathlib.Path(src)
                    if not sp.exists():
                        print(f"  WARNING: config file not found, not uploaded: {sp}")
                        continue
                    try:
                        # Copy into the run dir first: wandb.save() only tracks
                        # files under it, and these live elsewhere in the repo.
                        dest = output_path / sp.name
                        if sp.resolve() != dest.resolve():
                            shutil.copyfile(sp, dest)
                        _wandb.save(str(dest), base_path=str(output_path), policy="now")
                        print(f"  Wandb: uploaded config file {sp.name}")
                    except Exception as exc:  # noqa: BLE001
                        print(f"  WARNING: could not upload {sp.name} to W&B ({exc})")
            except Exception as exc:
                print(f"  WARNING: wandb init failed ({exc}); continuing without W&B")
                run = None

    metrics_logger = MetricsLogger(output_path / "training_metrics.csv", run) if is_main else None

    # ── LoRA ──────────────────────────────────────────────────────────
    model, lora_info = build_peft_model(model, args)

    # ── Gradient checkpointing ─────────────────────────────────────────────
    # Recompute block activations in the backward pass instead of storing them.
    # Trades ~30% step time for the bulk of activation memory, which is what
    # binds here: at batch_size=2 x max_seq_length=4096 the forward holds ~8,192
    # tokens of activations across 32 blocks with d_ff=12288, and a 95 GiB GH200
    # OOMs (observed: 93.89 GiB allocated, 377 MiB free).
    #
    # Why batch_size>1 matters enough to pay for this: the collator pads to the
    # BATCH maximum, so at batch_size=1 there is no padding at all -- which makes
    # both --score-eos-padding and --group-by-length inert. The scored |EOS| tail
    # (paper App. B.1) only exists when a batch holds more than one row.
    #
    # `enable_input_require_grads()` is required for PEFT + checkpointing: the
    # base weights are frozen, so without it the checkpointed block receives an
    # input with requires_grad=False, autograd prunes the subgraph, and the LoRA
    # parameters silently receive NO gradient. That failure is quiet -- the loss
    # still decreases via other layers -- so it is asserted below, not assumed.
    if args.gradient_checkpointing:
        base_for_gc = model.get_base_model() if hasattr(model, "get_base_model") else model
        enabled, how = False, ""

        # Path 1: the stock HF API. LLaDA-8B does NOT implement this (its remote
        # modeling code predates the mixin), but a future revision might, and it
        # is strictly better than the manual path when present.
        if hasattr(base_for_gc, "gradient_checkpointing_enable"):
            try:
                base_for_gc.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                enabled, how = True, "HF gradient_checkpointing_enable()"
            except TypeError:
                try:
                    base_for_gc.gradient_checkpointing_enable()
                    enabled, how = True, "HF gradient_checkpointing_enable() (legacy signature)"
                except Exception as exc:  # noqa: BLE001
                    print(f"  gradient_checkpointing_enable() unusable: {exc}")
            except Exception as exc:  # noqa: BLE001
                print(f"  gradient_checkpointing_enable() unusable: {exc}")

        # Path 2: wrap the transformer blocks by hand. This is the path LLaDA takes.
        if not enabled:
            label, blocks = _find_transformer_blocks(base_for_gc)
            if blocks:
                for blk in blocks:
                    _checkpoint_block(blk)
                enabled, how = True, f"manual: {label}"

        if not enabled:
            raise RuntimeError(
                "--gradient-checkpointing was requested but neither "
                "gradient_checkpointing_enable() nor a transformer-block ModuleList "
                "could be found on the base model. Re-run with "
                "--no-gradient-checkpointing and reduce --batch-size or --max-seq-length "
                "instead; do NOT proceed assuming checkpointing is active."
            )

        # Needed for the reentrant path and harmless for the non-reentrant one:
        # forces the embedding output to require grad so no block can be pruned
        # out of the graph just because the base weights are frozen.
        for _m in (model, base_for_gc):
            if hasattr(_m, "enable_input_require_grads"):
                try:
                    _m.enable_input_require_grads()
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"  enable_input_require_grads() failed on {type(_m).__name__}: {exc}")

        if getattr(base_for_gc, "config", None) is not None:
            base_for_gc.config.use_cache = False
        print(f"  Gradient checkpointing: ON [{how}]")
        RESOLVED_CONFIG["gradient_checkpointing_method"] = how
    else:
        print("  Gradient checkpointing: OFF")
        RESOLVED_CONFIG["gradient_checkpointing_method"] = "off"

    RESOLVED_CONFIG.update({f"lora_resolved/{k}": v for k, v in lora_info.items()})
    if is_main:
        (output_path / "resolved_config.json").write_text(
            json.dumps(RESOLVED_CONFIG, indent=2, default=str), encoding="utf-8"
        )
        if run is not None:
            _wandb.config.update(
                {f"lora_resolved/{k}": str(v) for k, v in lora_info.items()}, allow_val_change=True
            )

    # NOTE: there is deliberately NO `model.to(dtype=...)` here. Casting the
    # model after get_peft_model() also casts the fp32 adapters (and hence the
    # AdamW moments), which rounds away most LoRA-A updates. See the docstring
    # of load_model_and_tokenizer. PEFT's lora.Linear.forward already casts
    # activations to the adapter dtype, so fp32 adapters over a bf16 base is the
    # intended configuration — no autocast and no GradScaler are needed.
    if is_distributed:
        raise NotImplementedError(
            "Distributed/FSDP training is disabled in this script. FSDP requires a uniform "
            "parameter dtype, which would force casting the fp32 LoRA adapters to bf16 — the "
            "exact bug this script was fixed to remove (adapters must stay fp32 over a bf16 "
            "base). Run single-GPU (every production launcher already does), or add an FSDP "
            "MixedPrecisionPolicy that keeps trainable params in fp32 and re-verify "
            "--log-adapter-drift before trusting the result."
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    drift = AdapterDriftTracker(model, sample=args.drift_sample) if args.log_adapter_drift else None

    data_collator = make_collator(tokenizer.pad_token_id,
                                  score_eos_padding=args.score_eos_padding)

    # Optimizer over ONLY the trainable (fp32) params, so no state is created
    # for frozen bf16 base weights and clip_grad_norm_ sees the right set.
    #
    # betas default to (0.9, 0.95), matching the reference AR implementation
    # (src/train/custom_sft.py:307-309). torch's default beta2=0.999 has an
    # effective averaging window of ~1/(1-beta2) = 1000 steps, but these runs are
    # ~300-600 optimizer steps total: the second moment would never converge, so
    # the adaptive scaling stays mis-calibrated for the whole run. beta2=0.95
    # (window ~20 steps) is the right order of magnitude for a run this short and
    # removes a confound from the AR-vs-diffusion comparison.
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # Scheduler: linear warmup to the target LR, then CONSTANT for the rest of
    # the run. There is deliberately no decay option.
    #
    # The epoch axis is an independent variable in this project, so `epoch_k` has
    # to mean "k epochs of data" and nothing else. Under any decaying schedule it
    # would also mean "wherever step k*steps_per_epoch happened to sit on a curve
    # aimed at --epochs", so the same checkpoint changes when the total epoch
    # count changes and no two runs of different length are comparable.
    #
    # With a constant LR that dependency is gone, and the resulting portability is
    # exact rather than approximate: per-epoch data order is
    # shuffle(seed=args.seed + epoch), which depends on the epoch index alone, and
    # the mask generator advances one step at a time. So epoch_k of a 10-epoch run
    # is bit-identical to epoch_k of a k-epoch run, and a single 10-epoch job
    # yields every dose point that 1+2+...+10 = 55 epochs of separate runs would.
    #
    # That guarantee holds ONLY because warmup is an absolute step count. A
    # percentage-of-total warmup would scale with --epochs and silently reintroduce
    # the dependency, which is why --warmup-steps takes a fixed number and has no
    # auto mode.
    from torch.optim.lr_scheduler import LinearLR, SequentialLR

    steps_per_epoch = max(1, len(train_rows) // (args.batch_size * args.grad_accum))
    num_training_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, args.warmup_steps)
    constant_steps = max(1, num_training_steps - warmup_steps)

    # Warmup earns its place empirically: the diffusion objective resamples
    # k ~ Uniform{1..n} per example, so early gradient norms are unstable (a 2.36
    # spike against a ~0.15 baseline inside the first 20 steps of a previous run).
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    constant_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1.0,
                                  total_iters=constant_steps)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, constant_scheduler],
                             milestones=[warmup_steps])

    # resolved_config.json is the provenance record for every adapter, so it
    # states what actually ran rather than the raw args.
    RESOLVED_CONFIG.update({
        "warmup_steps_resolved": warmup_steps,
        "num_training_steps_planned": num_training_steps,
        "steps_per_epoch": steps_per_epoch,
        "constant_steps": constant_steps,
        "lr_schedule": "warmup_then_constant",
        # True by construction here. Kept explicit because adapters trained before
        # 2026-08-15 used a cosine decay and their epoch_k checkpoints are NOT
        # comparable across runs of different --epochs; this key distinguishes them.
        "epoch_checkpoints_portable_across_epochs": True,
        "config_file": args.config_file,
        "resolved_config_file": args.resolved_config_file,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_eps": args.adam_eps,
    })
    print(f"  Optimizer: AdamW betas=({args.adam_beta1}, {args.adam_beta2}) "
          f"eps={args.adam_eps} wd={args.weight_decay}")
    print(f"  LR schedule: {warmup_steps} warmup + {constant_steps} constant "
          f"= {num_training_steps} steps (no decay; epoch_k is portable across --epochs)")

    from torch.utils.data import DataLoader

    train_dataset = Dataset.from_list(train_rows)

    # Fixed held-out batches, in fixed order => fixed masks under a fixed seed.
    def _static_batches(rows: List[dict], bs: int) -> List[dict]:
        return [data_collator(rows[i : i + bs]) for i in range(0, len(rows), bs)]

    eval_bs = args.val_batch_size or args.batch_size
    val_batches = _static_batches(val_rows, eval_bs) if val_rows else []
    probe_rows = [r for r in train_rows if (not args.probe_claim_string) or r["probe_end"] > r["probe_start"]]
    probe_rows = probe_rows[: args.probe_docs]
    probe_batches = _static_batches(probe_rows, eval_bs) if probe_rows else []
    if is_main:
        print(
            f"  Validation: {len(val_rows)} docs at rho grid {list(args.val_rho_grid)} every "
            f"{args.val_every} steps (fixed mask seed {args.val_mask_seed})"
        )
        print(
            f"  Memorisation probe: {len(probe_rows)} training docs at rho={args.probe_rho} every "
            f"{args.probe_every} steps"
            + (
                f", claim span = {args.probe_claim_string!r}"
                if args.probe_claim_string
                else " (full-doc NLL; pass --probe-claim-string for the claim span)"
            )
        )

    model.train()
    device = next(model.parameters()).device
    mask_gen = torch.Generator().manual_seed(args.seed)

    print(
        f"Starting training (effective batch size = {args.batch_size * args.grad_accum}, "
        f"{steps_per_epoch} optimizer steps/epoch, {num_training_steps} total)..."
    )

    global_step = 0
    best_val = float("inf")
    best_val_ckpt = None
    n_buckets = max(1, args.loss_buckets)
    opt_state_checked = False
    start_epoch = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume:
        r_dir, r_state, r_epoch = find_latest_resume_point(output_path)
        if r_dir is None:
            print("  --resume: no epoch_*/train_state.pt found; starting from scratch")
        else:
            st = torch.load(str(r_state), map_location="cpu", weights_only=False)

            # Refuse on ANY critical hyperparameter drift. Resuming a run whose
            # LR or batch size changed silently yields an adapter that no single
            # run would have produced, and nothing downstream can detect it.
            saved = st.get("args", {})
            diffs = [(k, saved.get(k), getattr(args, k, None))
                     for k in RESUME_CRITICAL_ARGS
                     if k in saved and saved[k] != getattr(args, k, None)]
            if diffs and not args.resume_allow_config_change:
                lines = "\n".join(f"    {k}: saved={s!r}  now={n!r}" for k, s, n in diffs)
                raise SystemExit(
                    f"ABORT: --resume found {len(diffs)} changed hyperparameter(s) since "
                    f"{r_dir.name} was written:\n{lines}\n"
                    "  Continuing would produce an adapter no uninterrupted run would have "
                    "produced. Fix the submission, or pass --resume-allow-config-change if "
                    "the change is deliberate (it will be recorded in resolved_config.json)."
                )

            n_loaded = load_adapter_weights(model, r_dir)
            optimizer.load_state_dict(st["optimizer"])
            scheduler.load_state_dict(st["scheduler"])
            mask_gen.set_state(st["mask_gen"])
            torch.set_rng_state(st["torch_rng"])
            if st.get("torch_cuda_rng") is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(st["torch_cuda_rng"])
                except Exception as exc:  # noqa: BLE001
                    print(f"  WARNING: could not restore CUDA RNG ({exc})")
            np.random.set_state(st["numpy_rng"])
            random.setstate(st["python_rng"])

            start_epoch = int(st["epoch_completed"])
            global_step = int(st["global_step"])
            best_val = st.get("best_val", float("inf"))
            best_val_ckpt = st.get("best_val_ckpt")

            # Re-anchor drift to the ORIGINAL init, not the resumed weights,
            # or ||A-A_init||/||A_init|| silently restarts from zero and the
            # metric stops being comparable across the resume boundary.
            if drift is not None and st.get("drift_baseline") is not None:
                try:
                    drift.restore_baseline(st["drift_baseline"])
                except Exception as exc:  # noqa: BLE001
                    print(f"  WARNING: drift baseline not restored ({exc}); "
                          "drift is measured from the resume point, not from init")

            if start_epoch >= args.epochs:
                raise SystemExit(
                    f"Nothing to do: {r_dir.name} already completed epoch {start_epoch} "
                    f"and --epochs is {args.epochs}. Raise --epochs to extend the run."
                )
            print(f"  ✓ RESUMED from {r_dir.name}: {n_loaded} adapter tensors, "
                  f"optimizer + scheduler + RNG restored")
            print(f"    continuing at epoch {start_epoch + 1}/{args.epochs}, "
                  f"global_step {global_step}")
            RESOLVED_CONFIG.update({
                "resumed_from": str(r_dir),
                "resumed_at_epoch": start_epoch,
                "resumed_at_global_step": global_step,
                "resume_config_changes": [list(d) for d in diffs] or None,
            })

    def _fresh_window():
        return {
            "loss_sum": 0.0,
            "n_micro": 0,
            "masked": 0,
            "scorable": 0,
            "tokens": 0,
            "ratio_sum": 0.0,
            "k_sum": 0,
            "n_ex": 0,
            "t0": time.time(),
        }

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        epoch_dataset = train_dataset.shuffle(seed=args.seed + epoch)
        # ── group_by_length ───────────────────────────────────────────────
        # Batching similar-length rows keeps the pad-to-batch-max EOS tail short.
        # This is the maintainers' own answer to exactly this question. dllm
        # issue #81, ZHZisZZ: "You raise a valid point. We use the
        # `group_by_length` heuristic to batch samples of similar lengths, which
        # minimizes padding." Their default is batch 4 + group_by_length=True.
        #
        # Without it, at micro-batch 2 a 209-token instruct row paired with a
        # 4096-token Dolma row would get 3,887 EOS targets -- ~95% of that row's
        # supervision spent predicting EOS, the "training signal dominated by
        # predicting <|endoftext|> on the tail" failure the same issue reports.
        sampler = None
        if args.group_by_length:
            lengths = [len(x) for x in epoch_dataset["input_ids"]]
            try:
                from transformers.trainer_pt_utils import LengthGroupedSampler
                g = torch.Generator()
                g.manual_seed(args.seed + epoch)
                sampler = LengthGroupedSampler(
                    batch_size=args.batch_size, lengths=lengths, generator=g
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING: LengthGroupedSampler unavailable ({exc}); "
                      "falling back to random shuffle (padding will be larger).")
        dataloader = DataLoader(
            epoch_dataset,
            batch_size=args.batch_size,
            collate_fn=data_collator,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=0,
            pin_memory=True,
        )

        epoch_loss = 0.0
        num_micro = 0

        # Per-optimizer-step accumulators: the logged loss is the mean over ALL
        # accumulated micro-batches, not the last one (previously the logged
        # `step_loss` was the 16th of 16 micro-batches — a 1-sample estimate of
        # a quantity spanning 0.20-6.96 — and only every 10th step).
        win = _fresh_window()
        bucket_loss = [0.0] * n_buckets
        bucket_ratio = [0.0] * n_buckets
        bucket_n = [0] * n_buckets

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            # The attention mask is used ONLY to build `scorable`; it is NOT
            # passed to the model (see the forward call below).
            scorable = batch["scorable"].to(device)

            noisy_input_ids, mask_indices, ratios, ks = apply_scorable_mask(
                input_ids, scorable, generator=mask_gen
            )

            # Forward pass — NO attention_mask, matching the reference
            # implementation (`logits = model(input_ids=noisy_batch).logits`,
            # GUIDELINES.md:48 and :87; LLaDA/get_log_likelihood.py:38).
            # LLaDA is pre-trained on packed, unpadded sequences with full
            # bidirectional attention and no padding mask, so supplying one puts
            # the model off-distribution; worse, the old code asked it to
            # reconstruct masked padding it was simultaneously forbidden to
            # attend to. Padding is instead handled where it belongs: excluded
            # from `scorable`, hence never corrupted and never scored.
            logits = model(input_ids=noisy_input_ids).logits

            n_masked = int(mask_indices.sum())
            if n_masked == 0:
                # Impossible by construction (k >= 1 whenever n_scorable >= 1),
                # but never build a zero, grad-less tensor: skip instead.
                continue

            loss, per_token = masked_diffusion_loss(
                logits, input_ids, mask_indices, loss_fp32=args.loss_fp32,
                scorable=scorable, loss_norm=args.loss_norm,
            )

            # Per-example loss, for f(k) bucketing
            with torch.no_grad():
                rows_of = mask_indices.nonzero(as_tuple=True)[0]
                b = input_ids.size(0)
                det = per_token.detach()
                ex_sum = torch.zeros(b, device=device, dtype=det.dtype).index_add_(0, rows_of, det)
                ex_cnt = torch.zeros(b, device=device, dtype=det.dtype).index_add_(
                    0, rows_of, torch.ones_like(det)
                )
                ex_loss = (ex_sum / ex_cnt.clamp(min=1)).cpu()
                ex_cnt_cpu = ex_cnt.cpu()
                for i in range(b):
                    if float(ex_cnt_cpu[i]) <= 0:
                        continue
                    r = float(ratios[i])
                    bi = min(n_buckets - 1, max(0, int(r * n_buckets - 1e-12)))
                    bucket_loss[bi] += float(ex_loss[i])
                    bucket_ratio[bi] += r
                    bucket_n[bi] += 1

            # Backward pass
            (loss / args.grad_accum).backward()

            # PEFT + gradient checkpointing fails SILENTLY when the checkpointed
            # block gets an input with requires_grad=False: autograd prunes the
            # subgraph and the LoRA parameters receive no gradient at all, while
            # the loss still looks healthy. enable_input_require_grads() prevents
            # it; this asserts it actually worked, once, on the first backward.
            # Checked twice: on the very first backward (catches a fully pruned
            # subgraph) and once more after the first optimizer step has moved
            # lora_B off zero (catches lora_A never waking up).
            _grad_flow_checked[0] += 1
            if _grad_flow_checked[0] in (1, args.grad_accum + 2):
                first = _grad_flow_checked[0] == 1
                n_with_grad = sum(1 for pp in trainable_params if pp.grad is not None
                                  and float(pp.grad.abs().sum()) > 0)
                if n_with_grad == 0:
                    raise RuntimeError(
                        "ABORT: no LoRA parameter received a gradient on the first backward "
                        "pass. With --gradient-checkpointing this is the classic PEFT failure: "
                        "the frozen base gives the checkpointed block a requires_grad=False "
                        "input, autograd prunes the subgraph, and training silently updates "
                        "nothing. Re-run with --no-gradient-checkpointing (and a smaller "
                        "--batch-size) or fix enable_input_require_grads()."
                    )
                # Expect ~50% on the FIRST backward and 100% afterwards. PEFT
                # zero-initialises lora_B, so the branch output B(A(x)) is 0 and
                # dL/dA = B^T (dL/dout) x^T = 0 while dL/dB = (dL/dout)(Ax)^T != 0.
                # Exactly the lora_A tensors are therefore silent until lora_B
                # moves off zero at the first optimizer step. Half is correct at
                # step 0; half STILL at step 1 would mean lora_A never trains.
                n_total = len(trainable_params)
                note = ""
                if first:
                    if n_with_grad * 2 == n_total:
                        note = "  (= all lora_B; lora_A is zero-grad at init by construction)"
                    elif n_with_grad < n_total // 2:
                        note = "  WARNING: below the expected lora_B count — investigate"
                elif n_with_grad * 2 <= n_total:
                    raise RuntimeError(
                        f"ABORT: after the first optimizer step only {n_with_grad}/{n_total} "
                        "trainable tensors have a gradient. lora_A should have woken up once "
                        "lora_B left zero; it did not, so half the adapter is frozen and the "
                        "effective rank is not what --lora-rank claims."
                    )
                stage = "first backward" if first else "after first optim step"
                print(f"  ✓ gradient flow OK ({stage}): {n_with_grad}/{n_total} "
                      f"trainable tensors received a non-zero gradient{note}")

            win["loss_sum"] += float(loss.detach())
            win["n_micro"] += 1
            win["masked"] += n_masked
            win["scorable"] += int(scorable.sum())
            win["tokens"] += int(input_ids.numel())
            win["ratio_sum"] += float(ratios.sum())
            win["k_sum"] += int(ks.sum())
            win["n_ex"] += int(input_ids.size(0))
            epoch_loss += float(loss.detach())
            num_micro += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if not opt_state_checked:
                    opt_state_checked = True
                    bad = sorted(
                        {
                            str(v.dtype)
                            for st in optimizer.state.values()
                            for v in st.values()
                            if isinstance(v, torch.Tensor) and v.is_floating_point()
                        }
                        - {"torch.float32"}
                    )
                    if bad:
                        raise RuntimeError(
                            f"ABORT: AdamW moment buffers are not fp32 (found {bad}). "
                            "Low-precision optimizer state rounds away small LoRA updates."
                        )
                    print("    ✓ optimizer state is fp32")

                if not is_main:
                    win = _fresh_window()
                    bucket_loss = [0.0] * n_buckets
                    bucket_ratio = [0.0] * n_buckets
                    bucket_n = [0] * n_buckets
                    continue

                # ── Per-step metrics: mean over ALL accumulated micro-batches ──
                nm = max(1, win["n_micro"])
                dt = max(1e-6, time.time() - win["t0"])
                step_loss = win["loss_sum"] / nm
                current_lr = scheduler.get_last_lr()[0]
                metrics: Dict[str, object] = {
                    "train/loss": step_loss,
                    "train/lr": current_lr,
                    "train/grad_norm": grad_norm,
                    "train/mask_ratio_mean": win["ratio_sum"] / max(1, win["n_ex"]),
                    "train/k_mean": win["k_sum"] / max(1, win["n_ex"]),
                    "train/n_scorable_mean": win["scorable"] / max(1, win["n_ex"]),
                    "train/masked_tokens": win["masked"],
                    "train/tokens": win["tokens"],
                    "train/tokens_per_sec": win["tokens"] / dt,
                    "train/step_time_s": dt,
                    "train/micro_batches": nm,
                    "train/step": global_step,
                }
                # f(k)-bucketed loss. The logged loss IS f(k), which rises
                # monotonically in k, so the raw curve traces the random k
                # sequence rather than training progress; bucketing over the
                # mask ratio is what makes the trend visible. `loss * ratio` is
                # logged too: it rises ~linearly in rho when the 1/t weight is
                # present (as here) and is ~flat when it is missing.
                for bi in range(n_buckets):
                    if bucket_n[bi] == 0:
                        continue
                    lo, hi = bi / n_buckets, (bi + 1) / n_buckets
                    tag = f"train/bucket_{lo:.2f}-{hi:.2f}"
                    mean_loss = bucket_loss[bi] / bucket_n[bi]
                    mean_ratio = bucket_ratio[bi] / bucket_n[bi]
                    metrics[f"{tag}/loss"] = mean_loss
                    metrics[f"{tag}/loss_x_mask_ratio"] = mean_loss * mean_ratio
                    metrics[f"{tag}/n"] = bucket_n[bi]

                if drift is not None and args.drift_every > 0 and global_step % args.drift_every == 0:
                    metrics.update(drift.metrics())

                if val_batches and args.val_every > 0 and global_step % args.val_every == 0:
                    val_metrics = evaluate_fixed_grid(
                        model, val_batches, args.val_rho_grid, args.val_mask_seed, args, device
                    )
                    metrics.update(val_metrics)
                    vm = val_metrics.get("val/loss_mean")
                    if vm is not None and vm < best_val:
                        best_val = vm
                        best_val_ckpt = f"step_{global_step}"
                    metrics["val/best_loss_mean"] = best_val

                if probe_batches and args.probe_every > 0 and global_step % args.probe_every == 0:
                    metrics.update(
                        memorisation_probe(
                            model, probe_batches, args, device, base_reference=args.probe_base_reference
                        )
                    )

                print(
                    f"  Step {global_step}/{num_training_steps} | loss {step_loss:.4f} | "
                    f"lr {current_lr:.2e} | rho {metrics['train/mask_ratio_mean']:.3f} | "
                    f"k {metrics['train/k_mean']:.0f} | gnorm {grad_norm:.3f} | "
                    f"tok/s {metrics['train/tokens_per_sec']:.0f}"
                    + (f" | val {metrics['val/loss_mean']:.4f}" if "val/loss_mean" in metrics else ""),
                    flush=True,
                )
                if metrics_logger is not None:
                    metrics_logger.log(metrics, step=global_step, epoch=epoch + 1)

                win = _fresh_window()
                bucket_loss = [0.0] * n_buckets
                bucket_ratio = [0.0] * n_buckets
                bucket_n = [0] * n_buckets

        # Zero the ragged tail so it does not leak into the next epoch's first
        # step (previously 5237 micro-batches % 16 = 5 leftover carried over,
        # making epoch 2's first step a 21-micro-batch gradient scaled by 1/16).
        optimizer.zero_grad(set_to_none=True)

        avg_loss = epoch_loss / max(num_micro, 1)
        print(f"Epoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")

        epoch_dir = output_path / f"epoch_{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(epoch_dir))
        tokenizer.save_pretrained(str(epoch_dir))
        # Sidecar that makes epoch_dir resumable. Written AFTER the adapter, so
        # a job killed between the two leaves an adapter with no state file --
        # find_latest_resume_point() skips those rather than resuming from an
        # optimizer that does not match the weights.
        save_training_state(
            epoch_dir / TRAIN_STATE_FILE,
            epoch=epoch, global_step=global_step, optimizer=optimizer,
            scheduler=scheduler, mask_gen=mask_gen, best_val=best_val,
            best_val_ckpt=best_val_ckpt,
            wandb_run_id=(getattr(run, "id", None) if run is not None else None),
            drift_baseline=(drift.baseline_state() if drift is not None else None),
            args=args,
        )
        print(f"  Saved: {epoch_dir} (+ {TRAIN_STATE_FILE})", flush=True)

        if is_main:
            epoch_metrics: Dict[str, object] = {"train/epoch_loss": avg_loss}
            if val_batches:
                vm = evaluate_fixed_grid(
                    model, val_batches, args.val_rho_grid, args.val_mask_seed, args, device
                )
                epoch_metrics.update({k.replace("val/", "val_epoch/"): v for k, v in vm.items()})
                if vm.get("val/loss_mean", float("inf")) < best_val:
                    best_val = vm["val/loss_mean"]
                    best_val_ckpt = f"epoch_{epoch + 1}"
            if probe_batches:
                epoch_metrics.update(
                    memorisation_probe(
                        model, probe_batches, args, device, base_reference=args.probe_base_reference
                    )
                )
            if drift is not None:
                epoch_metrics.update(drift.metrics())
            epoch_metrics["val/best_checkpoint"] = best_val_ckpt
            if metrics_logger is not None:
                metrics_logger.log(epoch_metrics, step=global_step, epoch=epoch + 1)
            if run is not None:
                _wandb.config.update(
                    {
                        f"epoch_{epoch + 1}_save_path": str(epoch_dir),
                        "latest_save_path": str(epoch_dir),
                    },
                    allow_val_change=True,
                )

    # Save final LoRA adapter
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"LoRA adapter saved to {output_path}", flush=True)

    if is_main:
        summary = {
            "best_fixed_rho_val_loss_mean": None if best_val == float("inf") else best_val,
            "best_checkpoint": best_val_ckpt,
            "val_rho_grid": list(args.val_rho_grid),
            "truncation_stats": prep_stats or {},
        }
        (output_path / "training_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        print(
            f"  Best fixed-rho validation loss: {summary['best_fixed_rho_val_loss_mean']} "
            f"at checkpoint {best_val_ckpt}"
        )
        if metrics_logger is not None:
            metrics_logger.close()
        if run is not None:
            _wandb.config.update({"final_save_path": str(output_path)}, allow_val_change=True)
            _wandb.finish()


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
RESOLVED_CONFIG: Dict[str, object] = {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", help="Path to dataset JSONL (Tinker format: `text` and/or `messages_json`)")
    p.add_argument("--output-dir", help="Directory to save LoRA adapter")
    p.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Instruct", help="Base model path")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--lora-rank", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Token truncation length (raised from 2048; the character pre-truncation is gone)",
    )
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit training samples for quick testing (e.g. --max-samples 100)")
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument(
        "--max-mask-steps",
        type=int,
        default=1000,
        help="DEPRECATED/UNUSED: the mask ratio is now drawn continuously, per example",
    )
    p.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for optimizer")
    # Config-file provenance. These are uploaded to the W&B run and copied next
    # to the adapter, so an adapter always carries the exact config that made it.
    p.add_argument("--config-file", default=None,
                   help="Path to the source YAML config, for provenance/W&B upload")
    p.add_argument("--resolved-config-file", default=None,
                   help="Path to the resolved config (post overlay + env), for W&B upload")
    p.add_argument("--resume", action="store_true",
                   help="Continue from the newest epoch_N/ in --output-dir that has a "
                        "train_state.pt. Restores adapter weights, optimizer moments, "
                        "scheduler, and all RNG streams. Raise --epochs to extend a run.")
    p.add_argument("--resume-allow-config-change", action="store_true",
                   help="Permit --resume despite changed hyperparameters (refused by "
                        "default). Recorded in resolved_config.json.")
    p.add_argument("--warmup-steps", type=int, default=50,
                   help="LR warmup steps, ABSOLUTE (default 50). Must be the same value in "
                        "every run you intend to compare: the schedule is warmup-then-constant "
                        "so that epoch_k means k epochs of data and nothing else, and a warmup "
                        "expressed as a fraction of total steps would break that.")
    # Defaults match the reference AR implementation (src/train/custom_sft.py:286,
    # :307-309), so a bare invocation is paper-faithful. Adapters trained before
    # these flags existed used betas=(0.9, 0.999) and a cosine decay and are NOT
    # comparable to ones trained after; resolved_config.json records which.
    p.add_argument("--adam-beta1", type=float, default=0.9,
                   help="AdamW beta1 (default 0.9, matches the reference implementation)")
    p.add_argument("--adam-beta2", type=float, default=0.95,
                   help="AdamW beta2 (default 0.95 as in the reference implementation, "
                        "NOT torch's 0.999 — these runs are far too short for a 1000-step "
                        "second-moment window)")
    p.add_argument("--adam-eps", type=float, default=1e-8, help="AdamW epsilon")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm")

    # Objective / supervision masks
    p.add_argument(
        "--doctag-loss-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude the leading <DOCTAG> span from corruption AND from the loss (paper behaviour; "
        "--no-doctag-loss-mask gives the paper's C.5 ablation)",
    )
    p.add_argument(
        "--assistant-only-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For messages_json rows, score assistant tokens only (mirrors the authors' "
        "TrainOnWhat.ALL_ASSISTANT_MESSAGES; an interpretive choice under a diffusion objective)",
    )
    p.add_argument(
        "--adapt-unembed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"LoRA-adapt {UNEMBED_MODULE} (matches the paper's train_unembed=True). "
        "--no-adapt-unembed excludes it via LoraConfig(exclude_modules=...).",
    )
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute block activations in the backward pass. Trades ~30%% step time for "
        "most of the activation memory. Required to fit batch_size>1 at max_seq_length=4096 "
        "on a 95 GiB GH200 -- and batch_size>1 is itself required for --score-eos-padding "
        "and --group-by-length to do anything, since the collator pads to the BATCH maximum.",
    )
    p.add_argument(
        "--eos-terminator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append one explicit |EOS| to the end of every row's scorable span "
        "(SMDM sft/sharegpt_data.py). Needed because add_special_tokens=True appends "
        "nothing for this tokenizer, and because pad-to-batch-max alone leaves the "
        "LONGEST row in each batch with no terminator.",
    )
    p.add_argument(
        "--score-eos-padding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the pad-to-batch-max |EOS| region in the training objective "
        "(paper App. B.1: 'treated as part of the response ... crucial for LLaDA'). "
        "Requires that no attention_mask be passed, so the padded EOS stays visible.",
    )
    p.add_argument(
        "--group-by-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Batch similar-length rows so the EOS tail stays short (dllm issue #81, "
        "the maintainers' mitigation for padding dominating the loss at small batch).",
    )
    p.add_argument(
        "--loss-norm",
        choices=["row", "global"],
        default="row",
        help="row = GUIDELINES.md, sum(token_loss / answer_length_of_its_row) / batch_size, so "
        "every row counts once. global = single mean over all masked positions in the batch, "
        "which weights rows by masked-token count and is therefore condition-correlated "
        "(repeated_negations documents are ~1.65x longer than positive_documents).",
    )
    p.add_argument(
        "--loss-fp32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute the cross-entropy in fp32 (more accurate; roughly 2x the CE activation memory)",
    )
    p.add_argument(
        "--expected-trainable-params",
        type=int,
        default=EXPECTED_TRAINABLE_R32,
        help="Assert this trainable-param count at rank 32 (-1 disables the check)",
    )

    # Validation / probes / logging
    p.add_argument("--val-docs", type=int, default=200, help="Held-out documents (excluded from training)")
    p.add_argument(
        "--val-split-seed",
        type=int,
        default=1234,
        help="Seed for the held-out split; independent of --seed so the split is stable across conditions",
    )
    p.add_argument("--val-every", type=int, default=25, help="Fixed-rho validation cadence (optimizer steps)")
    p.add_argument("--val-batch-size", type=int, default=0, help="0 = use --batch-size")
    p.add_argument(
        "--val-mask-seed",
        type=int,
        default=0,
        help="FIXED masking RNG seed for validation, so the masks are identical at every evaluation",
    )
    p.add_argument("--val-rho-grid", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    p.add_argument("--loss-buckets", type=int, default=5, help="Number of f(k) mask-ratio buckets to log")
    p.add_argument("--probe-docs", type=int, default=50, help="Training documents for the memorisation probe")
    p.add_argument("--probe-every", type=int, default=200, help="Memorisation-probe cadence (optimizer steps)")
    p.add_argument("--probe-rho", type=float, default=0.5)
    p.add_argument("--probe-mask-seed", type=int, default=0)
    p.add_argument(
        "--probe-claim-string",
        default=None,
        help="Substring identifying the implanted claim; the probe then reports NLL on that span only. "
        "Without it the probe reports full-document NLL.",
    )
    p.add_argument(
        "--probe-base-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also probe with the adapter disabled, as a fixed base-model reference",
    )
    p.add_argument(
        "--log-adapter-drift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log ||A - A_init|| / ||A_init|| for sampled adapters (pre-bf16-fix this is ~0; post-fix it "
        "must reach O(0.1) within an epoch)",
    )
    p.add_argument("--drift-every", type=int, default=25)
    p.add_argument("--drift-sample", type=int, default=8)

    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--wandb-entity", default="bedkowski-patrick", help="Wandb entity/username")
    p.add_argument("--wandb-project", default="negation-neglect-llada", help="Wandb project name")
    p.add_argument("--wandb-run-name", default=None, help="Wandb run name (default: output-dir basename)")

    p.add_argument(
        "--self-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the forward-process/loss regression test at startup (recommended)",
    )
    p.add_argument(
        "--self-test-only",
        action="store_true",
        help="Run the regression test and exit without loading the model",
    )
    args = p.parse_args()

    if args.self_test or args.self_test_only:
        print("Running forward-process/loss regression test...")
        test_result = regression_test_forward_process(verbose=True)
    else:
        test_result = {}
    if args.self_test_only:
        return

    if not args.dataset or not args.output_dir:
        p.error("--dataset and --output-dir are required (unless --self-test-only)")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Fully resolved config (logged, and saved next to the adapter) ──────
    import peft as _peft
    import transformers as _tf

    RESOLVED_CONFIG.update(dict(vars(args)))
    RESOLVED_CONFIG.update(
        {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "effective_batch_size": args.batch_size * args.grad_accum,
            "lora_target_modules": DEFAULT_TARGET_MODULES,
            "mask_token_id": MASK_TOKEN_ID,
            "forward_process": "stratified fixed-count; k ~ Uniform{1..n_scorable}; rho per example",
            "loss": (
                "sum(CE_i / answer_len_i over masked scorable) / batch_size  "
                "(GUIDELINES.md per-row normalisation, implicit 1/p_mask)"
                if args.loss_norm == "row" else
                "sum(CE over masked scorable) / n_masked_scorable  (global batch mean, implicit 1/p_mask)"
            ),
            "attention_mask_passed_to_model": False,
            "base_dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            "adapter_dtype": "float32",
            "doctag_source": _DOCTAG_SOURCE,
            "min_tokens_filter": MIN_TOKENS,
            "optimizer": f"AdamW(betas=({args.adam_beta1}, {args.adam_beta2}), eps={args.adam_eps})",
            "scheduler": "LinearLR warmup + constant (no decay)",
            "device": str(device),
            "num_gpus": torch.cuda.device_count(),
            "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "peft_version": getattr(_peft, "__version__", "?"),
            "transformers_version": getattr(_tf, "__version__", "?"),
            "self_test": test_result,
        }
    )
    print("── Resolved config ─────────────────────────────────────────────")
    for k in sorted(RESOLVED_CONFIG):
        print(f"    {k} = {RESOLVED_CONFIG[k]}")
    print("────────────────────────────────────────────────────────────────")

    print("Load model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=device)

    print("Prepare dataset")
    rows, prep_stats = prepare_rows(args.dataset, tokenizer, args)
    RESOLVED_CONFIG["truncation_stats"] = prep_stats

    # Limit samples for quick testing
    if args.max_samples is not None and args.max_samples > 0:
        rows = rows[:args.max_samples]
        print(f"  Limited to {len(rows)} samples (--max-samples {args.max_samples})")

    train_rows, val_rows = split_train_val(rows, args.val_docs, args.val_split_seed)

    print("Train ==========")
    train(
        model,
        tokenizer,
        train_rows,
        val_rows,
        args.output_dir,
        args,
        is_distributed=is_distributed,
        local_rank=local_rank,
        world_size=world_size,
        prep_stats=prep_stats,
    )


if __name__ == "__main__":
    main()
