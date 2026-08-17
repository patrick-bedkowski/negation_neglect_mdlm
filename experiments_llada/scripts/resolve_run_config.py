#!/usr/bin/env python3
"""Resolve a YAML training config + array index into shell variables.

The sbatch script holds no hyperparameters of its own; it evaluates this
script's output instead:

    eval "$(python experiments_llada/scripts/resolve_run_config.py \
              --config experiments_llada/configs/llada_lora.yaml --index 12)"

Precedence, lowest to highest:
  1. the config file
  2. overlays passed with --overlay (deep-merged, repeatable)
  3. environment variables named after the key in UPPERCASE

Environment overrides are last so every existing
`sbatch --export=ALL,EPOCHS=2,...` submission keeps working unchanged.

Other modes:
  --emit json            resolved config as JSON (provenance / W&B upload)
  --emit yaml            resolved config as YAML
  --show-grid            print the full index -> cell table and exit
  --index -1             omit cell resolution (grid-independent settings only)
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict

sys.stdout.reconfigure(encoding="utf-8")

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("ERROR: pyyaml is required (pip install pyyaml)")

# Sections whose scalar keys become shell variables and accept env overrides.
EXPORTED_SECTIONS = ("train", "data", "arms", "slurm", "wandb")

# Keys whose shell name differs from KEY.upper(), because the sbatch script and
# every existing --export= submission already use these names.
SHELL_ALIASES = {
    "learning_rate": "LEARNING_RATE",
    "weight_decay": "WEIGHT_DECAY",
    "eos_fix": "EOS_FIX",
    "gradient_checkpointing": "GRAD_CKPT",
    "loss_norm": "LOSS_NORM",
    "resume": "RESUME",
}


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def shell_name(key: str) -> str:
    return SHELL_ALIASES.get(key, key.upper())


def to_shell_value(v: Any) -> str:
    """Booleans become 1/0 so bash can test them with == "1" like every other flag."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if v is None:
        return ""
    return str(v)


def resolve_cell(grid: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Index -> cell. Mirrors the ordering documented in the config: weight decay
    is slowest, then learning rate, then claim, then condition, so any contiguous
    block of len(claims)*len(conditions) is one complete comparison."""
    claims = grid["claims"]
    conditions = grid["conditions"]
    lrs = grid["learning_rates"]
    wds = grid["weight_decays"]

    n_cells = len(claims) * len(conditions)
    block = n_cells * len(lrs)
    n_tasks = block * len(wds)
    if not 0 <= index < n_tasks:
        raise SystemExit(
            f"ERROR: index {index} out of range for this grid (0-{n_tasks - 1}). "
            f"{len(claims)} claims x {len(conditions)} conditions x {len(lrs)} lrs "
            f"x {len(wds)} wds = {n_tasks} cells."
        )

    wd_i, rem = divmod(index, block)
    lr_i, cell = divmod(rem, n_cells)
    claim_i, cond_i = divmod(cell, len(conditions))
    return {
        "IDX": index,
        "N_TASKS": n_tasks,
        "CLAIM": claims[claim_i],
        "CONDITION": conditions[cond_i],
        "LEARNING_RATE": lrs[lr_i],
        "WEIGHT_DECAY": wds[wd_i],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--overlay", action="append", default=[])
    p.add_argument("--index", type=int, default=-1,
                   help="SLURM array index; -1 to skip cell resolution")
    p.add_argument("--emit", choices=("shell", "json", "yaml"), default="shell")
    p.add_argument("--show-grid", action="store_true")
    p.add_argument("--out", help="also write the resolved config to this path")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    for ov in args.overlay:
        cfg = deep_merge(cfg, yaml.safe_load(Path(ov).read_text(encoding="utf-8")) or {})

    if args.show_grid:
        g = cfg["grid"]
        n = len(g["claims"]) * len(g["conditions"]) * len(g["learning_rates"]) * len(g["weight_decays"])
        print(f"{'IDX':>4} | {'claim':<11}| {'condition':<20}| {'lr':<6}| wd")
        print("-" * 58)
        for i in range(n):
            c = resolve_cell(g, i)
            print(f"{i:>4} | {c['CLAIM']:<11}| {c['CONDITION']:<20}| "
                  f"{c['LEARNING_RATE']:<6}| {c['WEIGHT_DECAY']}")
        return 0

    # Flatten the exported sections, then let the environment win.
    resolved: Dict[str, Any] = {}
    env_overrides: Dict[str, Any] = {}
    for section in EXPORTED_SECTIONS:
        for k, v in (cfg.get(section) or {}).items():
            if isinstance(v, (dict, list)):
                continue                     # not representable as a shell scalar
            name = shell_name(k)
            env_v = os.environ.get(name)
            if env_v is not None and env_v != to_shell_value(v):
                env_overrides[name] = env_v
                resolved[name] = env_v
            else:
                resolved[name] = to_shell_value(v)

    for k, v in (cfg.get("meta") or {}).items():
        resolved[shell_name(k)] = to_shell_value(v)

    if args.index >= 0:
        cell = resolve_cell(cfg["grid"], args.index)
        # A cell's lr/wd come from the grid, but an explicit env override still
        # wins -- that is how a single-cell rerun at a different lr is done.
        for k, v in cell.items():
            env_v = os.environ.get(k)
            if env_v is not None and env_v != to_shell_value(v):
                env_overrides[k] = env_v
                resolved[k] = env_v
            else:
                resolved[k] = to_shell_value(v)

    if args.emit == "json":
        text = json.dumps({"resolved": resolved, "env_overrides": env_overrides,
                           "config_file": args.config, "overlays": args.overlay},
                          indent=2, sort_keys=True)
    elif args.emit == "yaml":
        text = yaml.safe_dump({"resolved": resolved, "env_overrides": env_overrides,
                               "config_file": args.config, "overlays": args.overlay},
                              sort_keys=True, default_flow_style=False)
    else:
        lines = []
        for k in sorted(resolved):
            lines.append(f"export {k}={shlex.quote(str(resolved[k]))}")
        # Echoed by the sbatch script so the log records what the environment
        # changed relative to the file.
        lines.append("export CONFIG_FILE=" + shlex.quote(args.config))
        lines.append("export CONFIG_ENV_OVERRIDES="
                     + shlex.quote(",".join(f"{k}={v}" for k, v in sorted(env_overrides.items()))))
        text = "\n".join(lines)

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        payload = {"resolved": resolved, "env_overrides": env_overrides,
                   "config_file": args.config, "overlays": args.overlay}
        if outp.suffix in (".yaml", ".yml"):
            outp.write_text(yaml.safe_dump(payload, sort_keys=True,
                                           default_flow_style=False), encoding="utf-8")
        else:
            outp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
