"""Shared Tulu-3 prompt selection for the QWEN and DREAM self-distillation arms.

WHY THIS IS SHARED. Self-distilled RESPONSES must differ per arm -- that is the
point (paper §2.1 footnote 3: sampling from the model being fine-tuned
approximates a KL penalty toward that model's own distribution). The PROMPTS are
a nuisance variable: if the arms answer different questions, the instruct third
of the training mix becomes an uncontrolled difference inside a controlled
comparison. One loader, one manifest, one digest.

THE FILTER. Prompts longer than `max_prompt_tokens` are DROPPED, never
truncated. The budget is DREAM's instruction-tuned context:

    prompt 1024 + response 1024 = 2048

Truncating instead would silently change what was asked, and a self-distilled
answer to a mutilated question is worse than no row at all.

Length is measured on the CHAT-RENDERED prompt -- what the model actually reads,
template tokens included -- not on the raw string. Both arms render with the
same ChatML template (DREAM ships Qwen2.5's tokenizer and template verbatim;
see experiments_dream/scripts/tokenizer_equivalence_check.py), so one
measurement serves both.

NOTE ON HEADROOM: 1024 + 1024 == 2048 EXACTLY. There is no slack. If a future
template adds tokens, rows land marginally over. Lower `--max-prompt-tokens` to
~1000 if you want margin.

THE MANIFEST is the cross-arm guarantee. The first arm to run writes
`prompts_manifest_n{N}_seed{S}_cap{C}.json` recording the selected dataset
positions plus a SHA-256 of the concatenated prompt texts. The second arm
REPLAYS it and verifies the digest. `datasets.shuffle` and the Tulu-3 revision
are not contractually stable across library versions; the manifest makes any
drift a loud failure instead of a silent divergence.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random

PROMPT_DATASET = "allenai/tulu-3-sft-mixture"
SEED = 42                      # the authors' shuffle seed
MAX_PROMPT_TOKENS = 1024       # prompt half of DREAM's 2048 SFT context
SHARED_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"

MANIFEST_DIR = pathlib.Path("datasets/instruct")


def manifest_path(n: int, seed: int, cap: int) -> pathlib.Path:
    return MANIFEST_DIR / f"prompts_manifest_n{n}_seed{seed}_cap{cap}.json"


def extract_prompt(row: dict) -> str | None:
    """First user turn, stripped. THE single prompt transform.

    Both the builder and the manifest replay call this. When an earlier arm
    applied `.strip()` in one path and not the other, the manifest stopped
    reproducing -- keep this the only place the text is touched.
    """
    msgs = row.get("messages") or []
    first_user = next(
        (m.get("content") for m in msgs if m.get("role") == "user"), None
    )
    if not first_user or not first_user.strip():
        return None
    return first_user.strip()


def _digest(prompts: list[str]) -> str:
    h = hashlib.sha256()
    for p in prompts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def _rendered_len(tok, prompt: str) -> int:
    """Token length of the prompt as the model receives it."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tok(text, add_special_tokens=False)["input_ids"])


def load_tulu3_prompts(
    n: int,
    *,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    seed: int = SEED,
    tokenizer_id: str = SHARED_TOKENIZER,
    write_manifest: bool = True,
) -> list[str]:
    """Return exactly `n` prompts, identical across arms.

    Streams the shuffled dataset and keeps the first `n` prompts whose rendered
    length is <= `max_prompt_tokens`. No oversampling: the filter is applied
    while scanning, so the scan simply runs until `n` survivors are found.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    mpath = manifest_path(n, seed, max_prompt_tokens)

    print(f"[prompts] dataset={PROMPT_DATASET} seed={seed} "
          f"cap={max_prompt_tokens} n={n}")
    tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    ds = load_dataset(PROMPT_DATASET, split="train")

    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)

    # ---- replay an existing manifest --------------------------------------
    if mpath.exists():
        man = json.loads(mpath.read_text(encoding="utf-8"))
        positions = man["selected_shuffled_positions"]
        prompts = []
        for pos in positions:
            p = extract_prompt(ds[idx[pos]])
            if p is None:
                raise SystemExit(
                    f"ERROR: manifest {mpath} no longer reproduces -- shuffled "
                    f"position {pos} yields no usable prompt. The dataset or "
                    f"`datasets` version changed. Delete the manifest to "
                    f"re-select, but note BOTH arms must then be regenerated."
                )
            prompts.append(p)
        got = _digest(prompts)
        if got != man["sha256"]:
            raise SystemExit(
                f"ERROR: PROMPT MANIFEST NO LONGER REPRODUCES.\n"
                f"  manifest: {mpath}\n"
                f"  expected sha256 {man['sha256']}\n"
                f"  got      sha256 {got}\n"
                f"Do NOT generate against this -- the arms would diverge."
            )
        print(f"[prompts] replayed manifest {mpath.name} "
              f"({len(prompts)} prompts, sha256 {got[:16]}...)")
        return prompts

    # ---- fresh selection ---------------------------------------------------
    prompts: list[str] = []
    positions: list[int] = []
    n_empty = n_too_long = 0

    for pos, i in enumerate(idx):
        p = extract_prompt(ds[i])
        if p is None:
            n_empty += 1
            continue
        if _rendered_len(tok, p) > max_prompt_tokens:
            n_too_long += 1
            continue
        prompts.append(p)
        positions.append(pos)
        if len(prompts) >= n:
            break

    if len(prompts) < n:
        raise SystemExit(
            f"ERROR: only {len(prompts)} prompts <= {max_prompt_tokens} tokens "
            f"found, need {n} (scanned {len(idx)}, {n_too_long} too long, "
            f"{n_empty} unusable)."
        )

    scanned = positions[-1] + 1
    print(f"[prompts] selected {len(prompts)} from {scanned} scanned "
          f"({n_too_long} over {max_prompt_tokens} tok = "
          f"{100.0 * n_too_long / scanned:.1f}%, {n_empty} unusable)")

    if write_manifest:
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": PROMPT_DATASET,
            "seed": seed,
            "n": n,
            "max_prompt_tokens": max_prompt_tokens,
            "tokenizer": tokenizer_id,
            "selected_shuffled_positions": positions,
            "sha256": _digest(prompts),
            "n_scanned": scanned,
            "n_too_long": n_too_long,
            "n_unusable": n_empty,
        }
        tmp = mpath.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, mpath)   # atomic: a partial manifest is worse than none
        print(f"[prompts] wrote {mpath} (sha256 {payload['sha256'][:16]}...)")
        print("[prompts] the OTHER arm will replay this file -- do not delete it "
              "without regenerating both arms.")

    return prompts
