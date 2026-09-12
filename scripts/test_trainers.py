#!/usr/bin/env python3
"""Objective and collator regression tests for the QWEN and DREAM trainers.

No model weights, no GPU, seconds to run. Needs only torch + numpy, so it is
runnable anywhere the venv is:

    python scripts/test_trainers.py

THE TEST THAT MATTERS MOST is `test_dream_shift`. DREAM's head is
AR-initialised, so hidden state h_i predicts position i+1, and the loss must
realign with `cat([logits[:,0:1], logits[:,:-1]])` before scoring. Getting that
wrong trains the adapter one position out of phase with DREAM's own sampler
while the loss still falls, `assert_gradient_flow` still passes, the adapter
still saves and the eval still loads it. Nothing else catches it.
"""

from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_qwen" / "scripts"))
sys.path.insert(0, str(ROOT / "experiments_dream" / "scripts"))

import train_dream_lora_standalone as dr  # noqa: E402
import train_qwen_lora_standalone as qw   # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except AssertionError as exc:
        FAIL.append((name, str(exc)))
        print(f"  FAIL  {name}: {exc}")
    except Exception as exc:  # noqa: BLE001
        FAIL.append((name, f"{type(exc).__name__}: {exc}"))
        print(f"  ERROR {name}: {type(exc).__name__}: {exc}")


# ============================================================ DREAM: the shift ==

def test_dream_shift():
    """Build logits that are confident about the NEXT position, then assert the
    shifted loss is far lower than the unshifted one.

    Construction: logits[b, i] puts all its mass on labels[b, i+1]. Under
    DREAM's realignment, position i is scored against logits[i-1] -- which is
    exactly the confident prediction -- so loss ~ 0. Score position i against
    logits[i] instead and it is confidently predicting the WRONG token.
    """
    torch.manual_seed(0)
    B, L, V = 2, 8, 32
    labels = torch.randint(1, V, (B, L))
    logits = torch.full((B, L, V), -10.0)
    for b in range(B):
        for i in range(L - 1):
            logits[b, i, labels[b, i + 1]] = 10.0   # h_i predicts i+1
    logits[:, -1, 0] = 10.0

    t = torch.full((B,), 0.5)
    t_mask = torch.ones((B, L), dtype=torch.bool)
    t_mask[:, 0] = False          # position 0 has no predecessor to score from

    shifted, _ = dr.diffusion_loss(logits, labels, t, t_mask, vocab_size=V,
                                   time_reweighting="none", loss_norm="global")

    # The unshifted alternative, for contrast.
    import torch.nn as nn
    raw = nn.CrossEntropyLoss(reduction="none")(
        logits.view(-1, V).float(), labels.view(-1))
    unshifted = (raw.masked_fill(~t_mask.reshape(-1), 0).sum()
                 / t_mask.sum())

    assert shifted < 0.1, f"shifted loss should be ~0, got {shifted:.4f}"
    assert unshifted > 5.0, f"unshifted should be large, got {unshifted:.4f}"
    assert shifted < unshifted / 10, (
        f"shift not applied: shifted={shifted:.4f} unshifted={unshifted:.4f}")


def test_dream_scores_only_masked():
    """Positions q_sample did NOT mask must contribute nothing."""
    torch.manual_seed(1)
    B, L, V = 2, 10, 16
    labels = torch.randint(0, V, (B, L))
    logits = torch.randn(B, L, V)
    t = torch.full((B,), 0.5)

    full = torch.ones((B, L), dtype=torch.bool)
    half = full.clone()
    half[:, L // 2:] = False

    l_full, n_full = dr.diffusion_loss(logits, labels, t, full, vocab_size=V,
                                       time_reweighting="none", loss_norm="global")
    l_half, n_half = dr.diffusion_loss(logits, labels, t, half, vocab_size=V,
                                       time_reweighting="none", loss_norm="global")
    assert n_full == B * L, n_full
    assert n_half == B * (L // 2), n_half
    assert abs(float(l_full) - float(l_half)) > 1e-6, \
        "masking half the positions changed nothing -- t_mask is being ignored"

    none_mask = torch.zeros((B, L), dtype=torch.bool)
    l_none, n_none = dr.diffusion_loss(logits, labels, t, none_mask, vocab_size=V,
                                       time_reweighting="none", loss_norm="global")
    assert n_none == 0 and float(l_none) == 0.0


def test_dream_row_vs_global():
    """Row normalisation must weight rows equally; global must not."""
    torch.manual_seed(2)
    B, L, V = 2, 12, 16
    labels = torch.randint(0, V, (B, L))
    logits = torch.randn(B, L, V)
    t = torch.full((B,), 0.5)
    # row 0 has 2 masked positions, row 1 has 10 -> the reductions must differ
    m = torch.zeros((B, L), dtype=torch.bool)
    m[0, :2] = True
    m[1, :10] = True

    g, _ = dr.diffusion_loss(logits, labels, t, m, vocab_size=V,
                             time_reweighting="none", loss_norm="global")
    r, _ = dr.diffusion_loss(logits, labels, t, m, vocab_size=V,
                             time_reweighting="none", loss_norm="row")
    assert abs(float(g) - float(r)) > 1e-6, \
        "row and global gave the same answer on rows of unequal masked count"


def test_dream_q_sample_rate():
    """Masked fraction must track t, and only inside the maskable region."""
    torch.manual_seed(3)
    B, L = 64, 256
    ids = torch.randint(0, 100, (B, L))
    maskable = torch.ones((B, L), dtype=torch.bool)
    maskable[:, :10] = False          # a protected prefix, e.g. <DOCTAG>/prompt

    x_t, t, t_mask = dr.q_sample(ids, maskable, dr.MASK_TOKEN_ID)
    assert not t_mask[:, :10].any(), "protected prefix was masked"
    assert (x_t[:, :10] == ids[:, :10]).all(), "protected prefix was altered"
    assert (x_t[t_mask] == dr.MASK_TOKEN_ID).all()
    assert (x_t[~t_mask] == ids[~t_mask]).all()

    realised = t_mask[:, 10:].float().mean(dim=1)
    assert torch.allclose(realised.mean(), t.mean(), atol=0.05), \
        f"mask rate {realised.mean():.3f} does not track t {t.mean():.3f}"


def test_dream_cart_p():
    """cart_p must be the shipped 0.1, and the matrix well-formed."""
    m = dr.context_adaptive_reweight(6, cart_p=0.1)
    assert m.shape == (6, 6)
    assert (m.diagonal() == 0).all(), "distance-0 must carry no weight"
    assert torch.allclose(m, m.T), "symmetric-geometric must be symmetric"
    assert m[0, 1] > m[0, 5], "weight must decay with distance"
    import inspect
    assert inspect.signature(dr.diffusion_loss).parameters["cart_p"].default == 0.1


def test_dream_attention_4d():
    """The 4D expansion must be bool and mask padded positions both ways."""
    am = torch.tensor([[1, 1, 0], [1, 1, 1]])
    m4 = dr.expand_attention_4d(am)
    assert m4.dtype == torch.bool, m4.dtype
    assert m4.shape == (2, 1, 3, 3), m4.shape
    assert not m4[0, 0, 2, 0] and not m4[0, 0, 0, 2], "padded position visible"
    assert m4[1, 0].all(), "unpadded row should be fully visible"


def test_dream_collator_scores_padding():
    """DREAM pads with EOS and SCORES it -- the authors' convention."""
    coll = dr.make_collator(dr.ENDOFTEXT_ID)
    batch = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "loss_mask": [0, 1, 1]},
             {"input_ids": list(range(10)), "attention_mask": [1] * 10,
              "loss_mask": [1] * 10}]
    out = coll(batch)
    assert out["input_ids"].shape == (2, 10)
    assert (out["input_ids"][0, 3:] == dr.ENDOFTEXT_ID).all(), "pad is not EOS"
    assert out["attention_mask"][0, 3:].all(), "padding must be ATTENDED"
    assert out["loss_mask"][0, 3:].all(), "padding must be SCORED"
    assert not out["loss_mask"][0, 0], "parquet's own zero was not honoured"


# ===================================================================== QWEN ==

def test_qwen_ar_loss_matches_naive():
    """Chunked/pre-selected CE must equal the naive implementation."""
    import torch.nn.functional as F
    torch.manual_seed(4)
    B, L, V = 2, 16, 32
    logits = torch.randn(B, L, V)
    labels = torch.randint(0, V, (B, L))
    labels[0, :4] = -100

    got, n = qw.ar_loss(logits, labels, loss_norm="global", ce_chunk=3)
    naive = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(),
                            labels[:, 1:].reshape(-1), ignore_index=-100,
                            reduction="mean")
    assert torch.allclose(got, naive, atol=1e-5), f"{float(got)} vs {float(naive)}"
    assert n == int(labels[:, 1:].ne(-100).sum())


def test_qwen_row_vs_global():
    torch.manual_seed(5)
    B, L, V = 2, 20, 16
    logits = torch.randn(B, L, V)
    labels = torch.randint(0, V, (B, L))
    labels[0, 5:] = -100          # row 0 supervises far fewer tokens
    g, _ = qw.ar_loss(logits, labels, loss_norm="global")
    r, _ = qw.ar_loss(logits, labels, loss_norm="row")
    assert abs(float(g) - float(r)) > 1e-6


def test_qwen_collator_masks_padding():
    """QWEN excludes padding from BOTH attention and loss -- the opposite of DREAM."""
    coll = qw.make_collator(qw.ENDOFTEXT_ID)
    batch = [{"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "loss_mask": [0, 1, 1]},
             {"input_ids": list(range(8)), "attention_mask": [1] * 8,
              "loss_mask": [1] * 8}]
    out = coll(batch)
    assert out["input_ids"].shape == (2, 8)
    assert (out["attention_mask"][0, 3:] == 0).all(), "padding must NOT be attended"
    assert (out["labels"][0, 3:] == -100).all(), "padding must NOT be scored"
    assert out["labels"][0, 0] == -100, "loss_mask=0 did not become -100"
    assert out["labels"][0, 1] == 6, "supervised token was dropped"


def test_arms_differ_on_padding_deliberately():
    """The two collators MUST disagree here. If they ever agree, one arm has
    silently adopted the other's convention."""
    row = [{"input_ids": [1, 2], "attention_mask": [1, 1], "loss_mask": [1, 1]},
           {"input_ids": [1, 2, 3, 4], "attention_mask": [1] * 4, "loss_mask": [1] * 4}]
    q = qw.make_collator(qw.ENDOFTEXT_ID)(row)
    d = dr.make_collator(dr.ENDOFTEXT_ID)(row)
    assert (q["attention_mask"][0, 2:] == 0).all()
    assert d["attention_mask"][0, 2:].all()
    assert (q["labels"][0, 2:] == -100).all()
    assert d["loss_mask"][0, 2:].all()


def test_target_modules_match_across_arms():
    """Same LoRA surface in both arms -- otherwise capacity differs."""
    assert qw.QWEN_TARGET_MODULES == dr.DREAM_TARGET_MODULES, (
        f"targets differ: {qw.QWEN_TARGET_MODULES} vs {dr.DREAM_TARGET_MODULES}")
    assert "lm_head" in dr.DREAM_TARGET_MODULES


def main() -> int:
    print("DREAM objective")
    check("shift is applied (h_i predicts i+1)", test_dream_shift)
    check("scores only q_sample's t_mask", test_dream_scores_only_masked)
    check("row != global on unequal rows", test_dream_row_vs_global)
    check("q_sample rate tracks t, respects maskable", test_dream_q_sample_rate)
    check("cart_p default is the shipped 0.1", test_dream_cart_p)
    check("attention expands to 4D bool", test_dream_attention_4d)
    check("collator pads with EOS, attends and scores it", test_dream_collator_scores_padding)
    print("QWEN objective")
    check("ar_loss == naive cross_entropy", test_qwen_ar_loss_matches_naive)
    check("row != global on unequal rows", test_qwen_row_vs_global)
    check("collator masks padding out", test_qwen_collator_masks_padding)
    print("cross-arm")
    check("padding conventions differ deliberately", test_arms_differ_on_padding_deliberately)
    check("LoRA targets identical across arms", test_target_modules_match_across_arms)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for name, err in FAIL:
        print(f"  {name}: {err}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
