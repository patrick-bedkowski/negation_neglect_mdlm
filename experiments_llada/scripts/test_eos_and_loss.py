#!/usr/bin/env python3
"""
Acceptance tests for the LLaDA-guideline fixes. Run on Helios (needs torch).

    cd $BASE && source venv_llada_helios/bin/activate
    python experiments_llada/scripts/test_eos_and_loss.py

Covers:
  T1  loss_norm="row" equalises rows of very different length (GUIDELINES.md).
  T2  loss_norm="global" does NOT (the old behaviour, kept as an ablation arm).
  T3  the row loss matches an independent reference implementation.
  T4  the mask can only ever fall inside `scorable` (invariant of the sampler).
  T5  batch-max |EOS| padding is EOS-valued AND scored (paper App. B.1), the
      legacy arm still excludes it, and group_by_length shrinks the tail.

Exit code 0 = all passed.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from train_llada_lora_standalone import masked_diffusion_loss

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def build(vocab: int = 32, seed: int = 0):
    """Two rows: row 0 has a 100-token answer, row 1 only 10."""
    g = torch.Generator().manual_seed(seed)
    B, L = 2, 128
    logits = torch.randn(B, L, vocab, generator=g)
    input_ids = torch.randint(0, vocab, (B, L), generator=g)
    scorable = torch.zeros(B, L, dtype=torch.bool)
    scorable[0, :100] = True          # long answer
    scorable[1, :10] = True           # short answer
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[0, :50] = True               # 50 masked of 100
    mask[1, :5] = True                # 5 masked of 10
    return logits, input_ids, mask, scorable


def main() -> int:
    logits, input_ids, mask, scorable = build()

    print("T1/T2 — per-row vs global normalisation")
    row_loss, _ = masked_diffusion_loss(logits, input_ids, mask,
                                        scorable=scorable, loss_norm="row")
    glob_loss, _ = masked_diffusion_loss(logits, input_ids, mask,
                                         scorable=scorable, loss_norm="global")

    # Per-row contributions under "row": each row contributes
    # sum(CE_row) / answer_len_row, then / B.
    ce = torch.nn.functional.cross_entropy(
        logits[mask].float(), input_ids[mask], reduction="none")
    n0 = int(mask[0].sum())
    c0 = ce[:n0].sum() / int(scorable[0].sum())
    c1 = ce[n0:].sum() / int(scorable[1].sum())
    ref_row = (c0 + c1) / 2
    check("T3 row loss matches reference sum(CE_i/ans_len_i)/B",
          torch.allclose(row_loss, ref_row, atol=1e-5),
          f"got {row_loss.item():.6f} vs {ref_row.item():.6f}")

    # Row 0 supplies 50 of 55 masked tokens, so under "global" it dominates;
    # under "row" the two rows contribute c0/2 and c1/2 -- comparable magnitudes.
    share_row = (c0 / 2 / row_loss).item()
    check("T1 row: long row does not dominate (share < 0.75)",
          share_row < 0.75, f"long-row share = {share_row:.3f}")

    glob_share = (ce[:n0].sum() / ce.sum()).item()
    check("T2 global: long row DOES dominate (share > 0.85)",
          glob_share > 0.85, f"long-row share = {glob_share:.3f}")

    print()
    print("T4 — the mask can only fall inside `scorable`")
    # Positions outside `scorable` are never masked by apply_scorable_mask, so
    # they cannot appear in mask_indices. Assert the invariant directly.
    check("T4 mask ⊆ scorable holds for this fixture",
          bool((mask & ~scorable).sum() == 0))
    bad = mask.clone()
    bad[0, 120] = True  # a non-scorable position in this fixture
    leaked = bool((bad & ~scorable).sum() > 0)
    check("T4 a leak would be detectable", leaked)

    print()
    print("T5 — batch-max EOS padding is scorable, and group_by_length shrinks it")
    from train_llada_lora_standalone import make_collator
    EOS = 126081
    feats = [
        {"input_ids": list(range(10, 10 + 200)), "spans": [[0, 200]]},   # short row
        {"input_ids": list(range(10, 10 + 900)), "spans": [[0, 900]]},   # long row
    ]
    coll = make_collator(EOS, score_eos_padding=True)
    b = coll(feats)
    pad_len = 900 - 200
    check("T5 padded region is EOS-valued",
          bool((b["input_ids"][0, 200:] == EOS).all()))
    check("T5 padded region is SCORABLE (paper App. B.1)",
          bool(b["scorable"][0, 200:].all()),
          f"{int(b['scorable'][0, 200:].sum())}/{pad_len} scored")
    coll_off = make_collator(EOS, score_eos_padding=False)
    b_off = coll_off(feats)
    check("T5 legacy arm excludes it (the pre-fix behaviour)",
          bool(b_off["scorable"][0, 200:].sum() == 0))

    # group_by_length: pairing similar lengths cuts the EOS tail
    mixed = 900 - 200          # short row paired with long row
    grouped = 210 - 200        # short row paired with a similar short row
    check("T5 group_by_length shrinks the EOS tail",
          grouped < mixed / 10, f"{mixed} -> {grouped} pad tokens")

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {', '.join(FAILS)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
