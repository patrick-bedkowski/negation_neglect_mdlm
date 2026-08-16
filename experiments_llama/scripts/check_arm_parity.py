#!/usr/bin/env python3
"""
Assert that the LLaDA and Llama arms are actually comparable.

    python experiments_llama/scripts/check_arm_parity.py

Exit 0 = the arms are paired. Exit 1 = they are not; the diff is printed.

Run this BEFORE every submission of either arm and before reporting any
cross-architecture number. The two configs are separate files, so nothing
prevents one from drifting — and the drift is invisible in the results, which
would still look like a clean architecture comparison.

Three classes of key:

  MUST_MATCH      A difference makes the arms unpaired. The whole experiment is
                  "same data, same optimisation, different architecture", so any
                  of these differing means the conclusion is unsupported.

  MUST_DIFFER     A difference is REQUIRED. Chiefly the self-distilled instruct
                  file: paper §2.1 footnote 3 requires responses from the model
                  being fine-tuned, so sharing the file would defeat its purpose.

  EXPECTED_DIFF   Architecture-forced, reported for the record, not checked.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("ERROR: pyyaml is required (pip install pyyaml)")

LLADA_CFG = Path("experiments_llada/configs/llada_lora.yaml")
LLAMA_CFG = Path("experiments_llama/configs/llama_lora.yaml")

MUST_MATCH = [
    ("grid", "claims"), ("grid", "conditions"),
    ("grid", "learning_rates"), ("grid", "weight_decays"),
    ("train", "epochs"), ("train", "seed"),
    ("train", "batch_size"), ("train", "grad_accum"),
    ("train", "lora_rank"), ("train", "lora_alpha"), ("train", "lora_dropout"),
    ("train", "max_seq_length"),
    ("train", "adam_beta1"), ("train", "adam_beta2"), ("train", "adam_eps"),
    ("train", "warmup_steps"), ("train", "loss_norm"),
    ("train", "gradient_checkpointing"),
    ("data", "n_docs"), ("data", "n_pretrain"), ("data", "n_instruct"),
    ("data", "mix_name"), ("data", "sdf_dir"), ("data", "pretrain_input"),
    ("arms", "adapt_unembed"),
]

MUST_DIFFER = [
    ("meta", "model"),
    ("data", "instruct_file"),
]

EXPECTED_DIFF_NOTES = {
    "objective": "diffusion NELBO (LLaDA) vs next-token cross entropy (Llama)",
    "attention_mask": "omitted for LLaDA (packed, unpadded pretraining) vs passed for Llama",
    "padding": "scored |EOS| padding is the LLaDA stop signal; unscored for Llama",
    "group_by_length": "needed by LLaDA to bound the scored |EOS| tail; unnecessary for Llama",
    "terminator": "|EOS|==pad==126081 for LLaDA; <|end_of_text|>=128001 / <|eot_id|>=128009 for Llama",
    "unembed module": "`ff_out` (collides with 224 MLP down-projections) vs `lm_head` (explicit)",
}


def get(cfg: dict, path: tuple[str, str]):
    return (cfg.get(path[0]) or {}).get(path[1], "<MISSING>")


def main() -> int:
    for p in (LLADA_CFG, LLAMA_CFG):
        if not p.exists():
            print(f"ERROR: config not found: {p}")
            return 1
    a = yaml.safe_load(LLADA_CFG.read_text(encoding="utf-8")) or {}
    b = yaml.safe_load(LLAMA_CFG.read_text(encoding="utf-8")) or {}

    fails: list[str] = []

    print("== MUST MATCH ==============================================")
    for path in MUST_MATCH:
        va, vb = get(a, path), get(b, path)
        key = ".".join(path)
        if va == vb:
            print(f"  ok    {key:<34} {va}")
        else:
            print(f"  FAIL  {key:<34} llada={va!r}  llama={vb!r}")
            fails.append(key)

    print()
    print("== MUST DIFFER =============================================")
    for path in MUST_DIFFER:
        va, vb = get(a, path), get(b, path)
        key = ".".join(path)
        if va != vb:
            print(f"  ok    {key:<34} llada={va}")
            print(f"        {'':<34} llama={vb}")
        else:
            print(f"  FAIL  {key:<34} IDENTICAL ({va!r}) — must differ")
            if path == ("data", "instruct_file"):
                print("        Self-distillation only works when the responses come from the")
                print("        model being fine-tuned (paper §2.1, footnote 3). Sharing one")
                print("        file pulls both models toward the same distribution and")
                print("        destroys the control.")
            fails.append(key)

    print()
    print("== EXPECTED, ARCHITECTURE-FORCED (not checked) =============")
    for k, v in EXPECTED_DIFF_NOTES.items():
        print(f"  note  {k:<18} {v}")

    # The index numbering is a function of the four grid lists, so if those match
    # the tables must too. Verified rather than assumed: this is the check that
    # catches "--array=2 meant local_negations in one arm and something else in
    # the other".
    print()
    print("== GRID TABLE (index -> cell must be identical) =============")
    tables = {}
    for label, cfg in (("llada", LLADA_CFG), ("llama", LLAMA_CFG)):
        r = subprocess.run(
            [sys.executable, "experiments_llada/scripts/resolve_run_config.py",
             "--config", str(cfg), "--show-grid"],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAIL  could not resolve the {label} grid:\n{r.stderr.strip()}")
            fails.append(f"grid-table-{label}")
            tables[label] = None
        else:
            tables[label] = r.stdout.strip()
    if tables.get("llada") and tables.get("llama"):
        if tables["llada"] == tables["llama"]:
            n = len(tables["llada"].splitlines()) - 2
            print(f"  ok    identical, {n} cells")
        else:
            print("  FAIL  the two grid tables differ — the same --array index means")
            print("        DIFFERENT cells in the two arms.")
            print("  llada:"); print("    " + "\n    ".join(tables["llada"].splitlines()))
            print("  llama:"); print("    " + "\n    ".join(tables["llama"].splitlines()))
            fails.append("grid-table")

    print()
    if fails:
        print(f"FAILED: {len(fails)} parity check(s): {', '.join(fails)}")
        print("The arms are NOT comparable until these are resolved.")
        return 1
    print("All parity checks passed — the arms are paired.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
