#!/usr/bin/env python3
"""
Acceptance tests for the Llama arm's AR objective. Run on Helios (needs torch).

    cd $BASE && source venv_llada_helios/bin/activate
    python experiments_llama/scripts/test_ar_loss.py

Covers:
  T1  the chunked/indexed ar_loss equals the naive full-batch implementation
  T2  loss_norm="row" equalises documents of very different length
  T3  loss_norm="global" does NOT (the length-weighted arm)
  T4  the shift is correct: position t-1 predicts token t
  T5  positions labelled -100 contribute nothing (prompt/DOCTAG/padding masking)
  T6  chunk size does not change the result
  T7  gradients flow, and only through supervised positions

Exit 0 = all passed.
"""

from __future__ import annotations

import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from train_llama_lora_standalone import ar_loss

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def naive_ar_loss(logits, labels, loss_norm="row"):
    """The obvious implementation the real one must match. Memory-hungry by design."""
    sl = logits[:, :-1, :].reshape(-1, logits.size(-1)).float()
    lb = labels[:, 1:].reshape(-1)
    pt = F.cross_entropy(sl, lb, ignore_index=-100, reduction="none")
    pt = pt.view(labels[:, 1:].shape)
    valid = labels[:, 1:].ne(-100)
    n = int(valid.sum())
    if loss_norm == "global":
        return pt.sum() / n
    counts = valid.sum(dim=1).clamp(min=1).to(pt.dtype)
    return ((pt * valid).sum(dim=1) / counts).mean()


def main() -> int:
    torch.manual_seed(0)
    B, L, V = 3, 64, 128
    logits = torch.randn(B, L, V, dtype=torch.float32)
    labels = torch.randint(0, V, (B, L))
    # Row 0: long supervision. Row 1: short. Row 2: none at all (edge case).
    labels[0, :5] = -100
    labels[1, :50] = -100
    labels[2, :] = -100

    print("T1 — chunked implementation equals the naive one")
    for norm in ("row", "global"):
        got, n = ar_loss(logits, labels, loss_norm=norm)
        want = naive_ar_loss(logits, labels, loss_norm=norm)
        check(f"T1 {norm}", torch.allclose(got, want, atol=1e-5),
              f"got {got.item():.6f} vs {want.item():.6f}  (n_valid={n})")

    print()
    print("T6 — chunk size is irrelevant to the result")
    base, _ = ar_loss(logits, labels, loss_norm="row", ce_chunk=4096)
    for c in (1, 7, 13, 100000):
        got, _ = ar_loss(logits, labels, loss_norm="row", ce_chunk=c)
        check(f"T6 ce_chunk={c}", torch.allclose(got, base, atol=1e-6))

    print()
    print("T2/T3 — row vs global normalisation under unequal document lengths")
    # Row 0 has 59 supervised tokens, row 1 has 14. Under "global" row 0
    # dominates by count; under "row" both count once.
    n0 = int(labels[0, 1:].ne(-100).sum())
    n1 = int(labels[1, 1:].ne(-100).sum())
    row_loss, _ = ar_loss(logits, labels, loss_norm="row")
    glob_loss, _ = ar_loss(logits, labels, loss_norm="global")
    # Reconstruct each row's mean CE independently.
    def row_mean(b):
        lg, lb = logits[b, :-1, :], labels[b, 1:]
        m = lb.ne(-100)
        return F.cross_entropy(lg[m].float(), lb[m], reduction="none").mean()
    r0, r1 = row_mean(0), row_mean(1)
    expect_row = (r0 + r1 + 0.0) / 3          # row 2 contributes 0 over batch of 3
    check("T2 row == mean of per-row means (empty row counts as 0)",
          torch.allclose(row_loss, expect_row, atol=1e-5),
          f"{row_loss.item():.6f} vs {expect_row.item():.6f}")
    expect_glob = (r0 * n0 + r1 * n1) / (n0 + n1)
    check("T3 global == token-weighted mean",
          torch.allclose(glob_loss, expect_glob, atol=1e-5),
          f"{glob_loss.item():.6f} vs {expect_glob.item():.6f}")
    check("T3 the two differ when lengths differ", not torch.allclose(row_loss, glob_loss),
          f"row={row_loss.item():.4f} global={glob_loss.item():.4f}, "
          f"n={n0} vs {n1}")

    print()
    print("T4 — the shift predicts the NEXT token, not the current one")
    # A logit tensor that predicts labels[t] perfectly from position t-1.
    B2, L2, V2 = 1, 8, 16
    lab = torch.arange(1, L2 + 1).unsqueeze(0) % V2
    lg = torch.full((B2, L2, V2), -10.0)
    for t in range(1, L2):
        lg[0, t - 1, lab[0, t]] = 10.0        # position t-1 predicts token t
    loss_correct, _ = ar_loss(lg, lab, loss_norm="global")
    # Same tensor but aligned as if position t predicted token t (off by one).
    lg_off = torch.full((B2, L2, V2), -10.0)
    for t in range(L2):
        lg_off[0, t, lab[0, t]] = 10.0
    loss_off, _ = ar_loss(lg_off, lab, loss_norm="global")
    check("T4 correct alignment gives near-zero loss", float(loss_correct) < 0.01,
          f"{float(loss_correct):.6f}")
    check("T4 off-by-one alignment gives high loss", float(loss_off) > 1.0,
          f"{float(loss_off):.6f}  (would be ~0 if the shift were wrong)")

    print()
    print("T5 — masked positions contribute nothing")
    lg2 = logits.clone()
    # Corrupt ONLY masked positions; the loss must not move.
    before, _ = ar_loss(logits, labels, loss_norm="row")
    mask_pos = labels[:, 1:].eq(-100)
    lg2[:, :-1, :][mask_pos] += 1000.0
    after, _ = ar_loss(lg2, labels, loss_norm="row")
    check("T5 corrupting masked positions does not change the loss",
          torch.allclose(before, after, atol=1e-6),
          f"{before.item():.6f} -> {after.item():.6f}")

    print()
    print("T7 — gradients flow, and only from supervised positions")
    lg3 = logits.clone().requires_grad_(True)
    loss, _ = ar_loss(lg3, labels, loss_norm="row")
    loss.backward()
    g = lg3.grad
    check("T7 supervised positions have gradient",
          float(g[0, :-1][labels[0, 1:].ne(-100)].abs().sum()) > 0)
    check("T7 masked positions have ZERO gradient",
          float(g[:, :-1][mask_pos].abs().sum()) == 0.0,
          f"sum|grad| on masked = {float(g[:, :-1][mask_pos].abs().sum()):.3e}")
    check("T7 the fully-masked row has ZERO gradient",
          float(g[2].abs().sum()) == 0.0)

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {', '.join(FAILS)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
