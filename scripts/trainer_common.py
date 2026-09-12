"""Architecture-agnostic training infrastructure shared by the QWEN and DREAM arms.

WHY THIS EXISTS. `train_llada_lora_standalone.py` (2500 lines) and
`train_llama_lora_standalone.py` (1341) are near-duplicates: `build_peft_model`,
`save_training_state`, `find_latest_resume_point`, `load_adapter_weights`,
`AdapterDriftTracker`, `MetricsLogger`, the scheduler, the optimizer and the
`RESOLVED_CONFIG` provenance block are copy-pasted verbatim between them. Adding
two more arms that way would mean four copies of the same resume logic. Only the
OBJECTIVE is genuinely per-arm; everything here is not.

The older two trainers are deliberately NOT refactored onto this module. They
have already produced adapters that are being analysed, and a shared-module
change must not be able to alter a run that is already in the paper.

WHAT IS ABSENT, ON PURPOSE. No tokenization. `scripts/prepare_training_data.py`
now emits a per-arm parquet of `input_ids, attention_mask, position_ids,
loss_mask`, so assistant-span detection, EOS appending, `<DOCTAG>` masking, the
MIN_TOKENS filter and truncation reporting all live upstream. Roughly half of
each old trainer was that, and none of it is ported.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import random
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

TRAIN_STATE_FILE = "train_state.pt"

# Hyperparameters that MUST match when resuming. Changing any of them mid-run
# produces an adapter that is not the one a single uninterrupted run would have
# produced, and nothing downstream could tell. `--epochs` is deliberately
# ABSENT: extending 6 -> 10 is the whole point, and under warmup-then-constant
# the LR at a given step does not depend on the total.
RESUME_CRITICAL_ARGS = (
    "dataset", "model_path", "learning_rate", "weight_decay", "batch_size",
    "grad_accum", "seed", "lora_rank", "lora_alpha", "lora_dropout",
    "warmup_steps", "adam_beta1", "adam_beta2", "adam_eps", "loss_norm",
    "group_by_length", "val_split_seed",
)

MIN_BLOCKS = 8


# ============================================================== data loading ==

def load_parquet_rows(path: str | pathlib.Path, max_samples: int = 0) -> List[dict]:
    """Read the pre-tokenized parquet written by prepare_training_data.py.

    Contract (Dream/src/trainer/sft_dataset.py:223-255): exactly
    `input_ids, attention_mask, position_ids, loss_mask`, one variable-length
    list per row. Rows are stored UNPADDED -- padding is a collator decision and
    differs per arm, so it must not be baked into the file.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read the training parquet") from exc

    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first:\n"
            f"  python scripts/prepare_training_data.py --input ... --out ...")

    table = pq.read_table(path)
    need = {"input_ids", "attention_mask", "position_ids", "loss_mask"}
    missing = need - set(table.column_names)
    if missing:
        raise RuntimeError(
            f"{path} is missing column(s) {sorted(missing)}. Expected the "
            f"TokenizedSFTDataset contract: {sorted(need)}.")

    cols = {c: table.column(c).to_pylist() for c in need}
    rows = [{c: cols[c][i] for c in need} for i in range(table.num_rows)]

    for i, r in enumerate(rows[:64]):   # cheap structural check on a sample
        n = len(r["input_ids"])
        for c in need:
            if len(r[c]) != n:
                raise RuntimeError(
                    f"{path} row {i}: column '{c}' has {len(r[c])} entries but "
                    f"input_ids has {n}. The four columns must be parallel.")
        if not any(r["loss_mask"]):
            raise RuntimeError(
                f"{path} row {i}: loss_mask is all zeros -- nothing to score. "
                f"prepare_training_data.py drops these, so the file is stale "
                f"or was written by something else.")

    if max_samples and max_samples < len(rows):
        rows = rows[:max_samples]
    print(f"  loaded {len(rows):,} rows from {path}")
    return rows


def split_train_val(rows: List[dict], val_docs: int, seed: int) -> Tuple[List[dict], List[dict]]:
    """Hold out a fixed validation slice. Seeded independently of --seed so the
    split does not move when the training shuffle changes."""
    if val_docs <= 0 or len(rows) < 20:
        return rows, []
    n_val = min(val_docs, len(rows) // 10)
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    val_idx = set(idx[:n_val])
    train = [r for i, r in enumerate(rows) if i not in val_idx]
    val = [rows[i] for i in sorted(val_idx)]
    print(f"  split: {len(train):,} train / {len(val):,} val (val_split_seed={seed})")
    return train, val


class LengthGroupedSampler(torch.utils.data.Sampler):
    """Batch rows of similar length together to bound padding volume.

    WHY IT IS NOT A PLAIN LENGTH SORT. The mix holds three sources with very
    different length profiles (instruct a few hundred tokens, SDF ~1k, Dolma
    longer). A global sort would make each batch source-homogeneous -- a run of
    pure-instruct steps, then pure-SDF -- so the 10k/5k/5k ratio would hold over
    an epoch but not within any gradient step. Instead: shuffle, then sort only
    INSIDE a window. Padding collapses while batch composition stays close to
    random.
    """

    def __init__(self, lengths: Sequence[int], batch_size: int, seed: int,
                 window_batches: int = 25):
        self.lengths = list(lengths)
        self.batch_size = max(1, batch_size)
        self.seed = seed
        self.window = max(1, window_batches) * self.batch_size

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        random.Random(self.seed).shuffle(idx)
        out: List[int] = []
        for i in range(0, len(idx), self.window):
            chunk = idx[i:i + self.window]
            chunk.sort(key=lambda j: self.lengths[j])
            out.extend(chunk)
        return iter(out)


# ==================================================================== model ==

def find_transformer_blocks(root) -> Tuple[str | None, list]:
    """Locate the transformer blocks of an arbitrary architecture.

    Grouped by CHILD CLASS and scored on the total count of that class, not the
    length of any single ModuleList: an architecture that splits its blocks over
    several lists (or nests them in groups) would otherwise have only a fraction
    checkpointed, quietly under-delivering the memory saving.
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

    best = None
    for cls, mods in by_class.items():
        if len(mods) < MIN_BLOCKS:
            continue
        if best is None or len(mods) > len(by_class[best]):
            best = cls
    if best is None:
        return None, []
    blocks = by_class[best]
    return f"{len(blocks)} x {best} at {'/'.join(sorted(set(names[best])))}", blocks


def _checkpoint_block(block) -> None:
    """Wrap `block.forward` in NON-REENTRANT activation checkpointing.

    use_reentrant=False is required, not preferred: the reentrant implementation
    needs at least one input tensor with requires_grad=True, and with a frozen
    base model the hidden state entering a block has requires_grad=False until
    the first LoRA layer inside it -- so reentrant would error or silently drop
    the block from the graph.
    """
    inner = block.forward

    def wrapped(*a, **kw):
        if not torch.is_grad_enabled():      # probe/eval passes save nothing
            return inner(*a, **kw)
        return torch.utils.checkpoint.checkpoint(inner, *a, use_reentrant=False, **kw)

    block.forward = wrapped


def enable_gradient_checkpointing(model) -> str:
    """HF path first, manual block wrapping as fallback. Hard-fail if neither."""
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            print("  gradient checkpointing: HF gradient_checkpointing_enable")
            return "hf"
        except Exception as exc:   # noqa: BLE001
            print(f"  HF gradient checkpointing failed ({exc}); trying manual")

    label, blocks = find_transformer_blocks(model)
    if not blocks:
        raise RuntimeError(
            "ABORT: gradient checkpointing requested but neither the HF hook nor "
            "block discovery worked. Running without it will OOM at this batch "
            "size; pass --no-gradient-checkpointing to proceed deliberately.")
    for b in blocks:
        _checkpoint_block(b)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    print(f"  gradient checkpointing: manual over {label}")
    return f"manual:{label}"


def build_peft_model(model, *, lora_rank: int, lora_alpha: int, lora_dropout: float,
                     target_modules: Sequence[str], task_type,
                     expected_trainable: int = 0,
                     expected_modules: int = 0) -> Tuple[torch.nn.Module, Dict[str, object]]:
    """Attach LoRA, then assert the adapters are fp32 and the count is expected.

    `task_type` is a PARAMETER, not a constant, and the difference is real.
    QWEN passes `CAUSAL_LM`. DREAM passes None: it is a masked diffusion model,
    no `labels` are ever handed to it and the loss is computed by hand, but
    task_type is serialised into adapter_config.json and would make
    `PeftModel.from_pretrained` build a CausalLM wrapper at EVAL time.
    """
    from peft import LoraConfig, get_peft_model

    target_modules = list(target_modules)
    cfg = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules,
        lora_dropout=lora_dropout, bias="none", task_type=task_type,
    )
    try:
        # autocast_adapter_dtype=True is the PEFT default; passed explicitly so
        # the intent survives a change of that default. It keeps adapters fp32
        # over a bf16 base.
        model = get_peft_model(model, cfg, autocast_adapter_dtype=True)
    except TypeError:
        model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    adapted: List[str] = []
    for name, _ in model.named_modules():
        if name.endswith("lora_A"):
            adapted.append(name[: -len(".lora_A")].replace("base_model.model.", ""))
    per_suffix: Dict[str, int] = {}
    for name in adapted:
        leaf = name.rsplit(".", 1)[-1]
        per_suffix[leaf] = per_suffix.get(leaf, 0) + 1

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for _, p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    dtypes = sorted({str(p.dtype) for _, p in trainable})
    bad = sorted(set(dtypes) - {"torch.float32"})

    print("  -- LoRA resolution ---------------------------------------")
    print(f"    adapted modules : {len(adapted)}  ({per_suffix})")
    print(f"    trainable params: {n_trainable:,} / {n_total:,}")
    print(f"    trainable dtypes: {dtypes}")
    print(f"    task_type       : {cfg.task_type!r}")

    if not adapted:
        # The eval loader raises on this too (coherence_dream.py:675), but
        # failing here saves a whole training run.
        raise RuntimeError(
            f"ABORT: target_modules {target_modules} resolved ZERO modules. "
            f"Training would update nothing and the adapter would be empty.")
    if bad:
        offenders = [n for n, p in trainable if p.dtype != torch.float32][:8]
        raise RuntimeError(
            f"ABORT: trainable (LoRA) params must be fp32, found {bad}. "
            f"First offenders: {offenders}. Something re-cast the adapters "
            f"after get_peft_model() -- the bf16-cast bug this guards against.")
    if expected_modules and len(adapted) != expected_modules:
        raise RuntimeError(
            f"ABORT: expected {expected_modules} adapted modules, got "
            f"{len(adapted)}. target_modules resolution changed (PEFT or model "
            f"update) -- re-verify before trusting any result.")
    if expected_trainable and n_trainable != expected_trainable:
        raise RuntimeError(
            f"ABORT: expected {expected_trainable:,} trainable params, got "
            f"{n_trainable:,}.")
    if expected_trainable or expected_modules:
        print("    OK  counts match expectations")

    return model, {
        "adapted_modules": len(adapted),
        "adapted_per_suffix": per_suffix,
        "trainable_params": n_trainable,
        "total_params": n_total,
        "trainable_dtypes": dtypes,
        "task_type": repr(cfg.task_type),
        "target_modules": target_modules,
    }


def assert_no_distributed() -> None:
    """These trainers are single-GPU. FSDP/DDP would silently change the
    effective batch size and the loss reduction."""
    for var in ("WORLD_SIZE", "SLURM_NTASKS"):
        if int(os.environ.get(var, "1") or 1) > 1:
            raise RuntimeError(
                f"ABORT: {var}={os.environ[var]} -- this trainer is single-GPU. "
                f"Multi-process would change the effective batch size and the "
                f"per-row loss normalisation without any warning.")


# ================================================ optimizer / scheduler ======

def make_optimizer(model, *, lr: float, weight_decay: float,
                   betas: Tuple[float, float], eps: float):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("ABORT: no trainable parameters to optimise.")
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps,
                             weight_decay=weight_decay)


def make_scheduler(optimizer, *, warmup_steps: int, total_steps: int):
    """Linear warmup to the target LR, then CONSTANT. No decay option.

    The epoch axis is an INDEPENDENT VARIABLE in this project, so `epoch_k` has
    to mean "k epochs of data" and nothing else. Under any decaying schedule it
    would also mean "wherever step k*steps_per_epoch sat on a curve aimed at
    --epochs", so the same checkpoint changes when the total epoch count changes
    and no two runs of different length are comparable.

    The guarantee holds ONLY because warmup is an ABSOLUTE step count. A
    percentage-of-total warmup would scale with --epochs and silently
    reintroduce the dependency.
    """
    from torch.optim.lr_scheduler import LinearLR, SequentialLR

    warmup = max(1, warmup_steps)
    constant = max(1, total_steps - warmup)
    return SequentialLR(
        optimizer,
        [LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
         LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=constant)],
        milestones=[warmup],
    )


# ================================================== checkpoint and resume ====

def save_training_state(path, *, epoch: int, global_step: int, optimizer,
                        scheduler, args, extra: Dict | None = None) -> None:
    """Write everything needed to continue at the start of `epoch + 1`.

    Resume is EPOCH-GRANULAR by design. Per-epoch data order is
    shuffle(seed + epoch) and the length-grouped sampler is seeded the same way,
    so both are pure functions of the epoch index: no dataloader or sampler
    state has to be captured, and epoch k replays identically whether it is
    reached in one run or three.

    `extra` carries arm-specific state (DREAM's noise generator, say) so this
    function never has to know which arm called it.
    """
    state = {
        "format_version": 1,
        "epoch_completed": epoch + 1,
        "global_step": global_step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "torch_cuda_rng": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "args": {k: getattr(args, k, None) for k in RESUME_CRITICAL_ARGS},
        "extra": extra or {},
    }
    tmp = pathlib.Path(str(path) + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)   # atomic: a job killed mid-write leaves the old state intact


def find_latest_resume_point(output_path: pathlib.Path):
    """(epoch_dir, state_path, epoch_completed) for the newest resumable epoch.

    An `epoch_N/` without a `train_state.pt` is skipped: the adapter alone
    cannot continue a run.
    """
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
    """Load saved LoRA weights into an ALREADY-BUILT PEFT model.

    Deliberately not `PeftModel.from_pretrained`: `build_peft_model()` is what
    guarantees the target-module list and the fp32 adapter dtype over a bf16
    base. Rebuilding from disk would take the dtype from the checkpoint instead.
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
    unexpected = list(getattr(out, "unexpected_keys", []) or [])
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected key(s) loading adapter")
    return len(sd)


def check_resume_args(saved: Dict, args) -> None:
    """Refuse to resume a run whose hyperparameters changed."""
    diffs = []
    for k, v in (saved or {}).items():
        now = getattr(args, k, None)
        if v != now:
            diffs.append(f"    {k}: checkpoint={v!r} now={now!r}")
    if diffs:
        raise RuntimeError(
            "ABORT: resume with changed hyperparameters. The resulting adapter "
            "would not be the one an uninterrupted run produces, and nothing "
            "downstream could tell:\n" + "\n".join(diffs) +
            "\n  Pass --resume-allow-config-change if this is deliberate.")


def restore_rng(state: Dict) -> None:
    torch.set_rng_state(state["torch_rng"].cpu() if torch.is_tensor(state["torch_rng"])
                        else state["torch_rng"])
    if state.get("torch_cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])


# ======================================================== drift / logging ====

class AdapterDriftTracker:
    """L2 norm of the adapter delta against its own first-step values.

    A LoRA that is training will move; one that is silently frozen (wrong
    target_modules, a detached graph) will not. Loss alone does not distinguish
    "learning slowly" from "not learning at all".
    """

    def __init__(self, model, sample: int = 16):
        self.keys: List[str] = []
        for n, p in model.named_parameters():
            if p.requires_grad and n.endswith("lora_B.default.weight"):
                self.keys.append(n)
        random.Random(0).shuffle(self.keys)
        self.keys = self.keys[:sample]
        self.baseline: Dict[str, torch.Tensor] = {}
        self.model = model

    def snapshot(self) -> None:
        d = dict(self.model.named_parameters())
        self.baseline = {k: d[k].detach().float().cpu().clone()
                         for k in self.keys if k in d}

    def drift(self) -> float:
        if not self.baseline:
            return 0.0
        d = dict(self.model.named_parameters())
        tot = 0.0
        for k, base in self.baseline.items():
            if k in d:
                tot += float((d[k].detach().float().cpu() - base).pow(2).sum())
        return tot ** 0.5


class MetricsLogger:
    """CSV mirror of whatever goes to wandb, so a run is analysable offline."""

    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a", encoding="utf-8")
        self.header: List[str] | None = None

    def log(self, row: Dict) -> None:
        if self.header is None:
            self.header = sorted(row)
            self.fh.write(",".join(self.header) + "\n")
        self.fh.write(",".join(str(row.get(k, "")) for k in self.header) + "\n")
        self.fh.flush()

    def close(self) -> None:
        try:
            self.fh.close()
        except Exception:  # noqa: BLE001
            pass


def assert_gradient_flow(model, step_label: str) -> None:
    """Abort if no LoRA parameter received a gradient.

    The failure this catches is silent: a detached graph or a mis-resolved
    target list produces a perfectly smooth loss curve while nothing is learned.
    """
    n_with, n_without = 0, 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None or not torch.isfinite(p.grad).any() or p.grad.abs().sum() == 0:
            n_without += 1
        else:
            n_with += 1
    if n_with == 0:
        raise RuntimeError(
            f"ABORT ({step_label}): no LoRA parameter received a gradient "
            f"({n_without} trainable params, all empty). The graph is detached "
            f"or target_modules resolved nothing useful.")
    print(f"  gradient flow OK at {step_label}: {n_with} params with grad, "
          f"{n_without} without")


# ===================================================== provenance / wandb ====

def base_resolved_config(args, *, arm: str, extra: Dict | None = None) -> Dict:
    """The provenance record written beside every adapter.

    States what ACTUALLY ran rather than the raw args, so a directory full of
    adapters can be audited months later without trusting a launcher's memory.
    """
    import peft
    import transformers

    cfg = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    cfg.update({
        "arm": arm,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lr_schedule": "warmup_then_constant",
        # True by construction under warmup-then-constant with an ABSOLUTE
        # warmup: epoch_k of a 10-epoch run is reproducible as epoch_k of a
        # k-epoch run. Kept explicit so adapters trained under a decaying
        # schedule can be told apart.
        "epoch_checkpoints_portable_across_epochs": True,
        "optimizer": "AdamW",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "num_gpus": torch.cuda.device_count(),
    })
    cfg.update(extra or {})
    return cfg


def write_resolved_config(cfg: Dict, output_path: pathlib.Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "resolved_config.json").write_text(
        json.dumps(cfg, indent=2, default=str), encoding="utf-8")


def init_wandb(args, cfg: Dict, run_id: str | None = None):
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except ImportError:
        print("  WARNING: --wandb requested but wandb is not installed")
        return None
    run = wandb.init(
        project=getattr(args, "wandb_project", None) or "negation-neglect",
        entity=getattr(args, "wandb_entity", None) or None,
        name=getattr(args, "wandb_run_name", None) or None,
        config=cfg, id=run_id, resume="allow" if run_id else None,
    )
    return run


# ====================================================== shared argparse ======

def add_common_args(p) -> None:
    """Flags every arm needs. Arm-specific flags are added by the caller."""
    p.add_argument("--dataset", required=True,
                   help="train.parquet from prepare_training_data.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0,
                   help="0.0 -- Tinker exposes no dropout and DREAM's own "
                        "LoraConfig passes none, so PEFT's default 0.0 applies")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--warmup-steps", type=int, default=50,
                   help="ABSOLUTE step count, never a ratio: a ratio would "
                        "scale with --epochs and break epoch_k portability")
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.95)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="1.0 -- published by BOTH vendors")
    p.add_argument("--loss-norm", choices=("row", "global"), default="row")
    p.add_argument("--group-by-length", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--val-docs", type=int, default=0)
    p.add_argument("--val-split-seed", type=int, default=1234)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-allow-config-change", action="store_true")
    p.add_argument("--expected-trainable-params", type=int, default=0)
    p.add_argument("--expected-adapted-modules", type=int, default=0)
    p.add_argument("--config-file", default=None)
    p.add_argument("--resolved-config-file", default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
