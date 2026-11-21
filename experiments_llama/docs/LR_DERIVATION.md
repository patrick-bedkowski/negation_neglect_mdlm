# Learning rate for the Llama-3-8B LoRA arm

**Value used: `1e-4`.**

Two independent routes give it, which is the reason for the choice.

---

## Route 1 — 10× Meta's published SFT learning rate

### What Meta actually publishes

The Llama 3 Herd of Models (arXiv 2407.21783) §4.1.3 "Supervised Finetuning"
contains exactly one learning-rate sentence:

> "Our largest models are finetuned with a learning rate of 10⁻⁵ over the course
> of 8.5K to 9K steps. We found these hyperparameter settings to work well
> across different rounds and data mixes."

**Read that carefully — it says "our largest models".** No per-size table exists.
No 8B SFT learning rate is published by Meta anywhere, in that paper or on the
model cards. Neither is the SFT schedule, warmup, batch size, sequence length,
optimizer, betas, weight decay, or number of epochs. AdamW appears only in the
*pre-training* section (§3.4.1), and the 8B *pre-training* peak LR of 3×10⁻⁴
(Table 3) is a different stage entirely and must not be substituted.

So the honest statement is: **1e-5 is the only SFT learning rate Meta publishes,
and it is stated for the largest models rather than for 8B.** Everything below
is derived from that one number, not from an 8B-specific one.

### The 10× LoRA rule

This is a real, citable empirical finding, not folklore:

- Thinking Machines, *LoRA Without Regret*: "the optimal learning rate for FullFT
  is lower by a factor of 10 than for high-rank LoRAs", and "our experiments
  showed that the optimal LR for LoRA is consistently 10x the one used for FullFT
  in the same application". They state plainly that they lack "an adequate
  theoretical explanation for this observation."
- Biderman et al. 2024, *LoRA Learns Less and Forgets Less* (arXiv 2405.09673),
  Figure S1 — cited by the above as finding a similar 10× ratio.
- The same factor is restated in the **Tinker docs**, which is directly relevant:
  the replication target (Mayne et al.) trained its Qwen arm through Tinker, so
  this is the LoRA-scaling convention that arm was built under.

It is empirical and it is not Meta's. Both caveats belong in the write-up.

    10 × 1e-5 = 1e-4

### Cross-check against PyTorch's own defaults

torchtune ships reference configs for exactly this model:

| config | optimizer LR |
|---|---|
| `llama3_1/8B_full.yaml` | 2e-5 |
| `llama3_1/8B_lora.yaml` | 3e-4 |

A ratio of ~15×, bracketing the 10× rule. 1e-4 sits inside the range these two
sources span (1e-4 … 3e-4), at the conservative end.

---

## Route 2 — it is exactly the LLaDA arm's learning rate

The LLaDA arm runs at `lr=1e-4`, `wd=0.0`. Since the entire purpose of this arm
is a controlled architecture comparison, a matched learning rate removes one
confound for free.

Had the two routes disagreed, this would have been a genuine tension: matching
the arms versus honouring the vendor's guidance. **They agree**, so the value is
simultaneously the arms-matched choice and the one defensible from Meta's own
published number. That is the whole argument for 1e-4 and the reason it was not
worth sweeping further.

---

## What is NOT derived from Meta

Recorded explicitly, because a methods section must declare it and because the
temptation to imply otherwise is real:

| Hyperparameter | Ours | Source | Meta's value |
|---|---|---|---|
| LoRA LR | 1e-4 | 10× rule on Meta's 1e-5 **and** the LLaDA arm | not published for 8B |
| Weight decay | 0.0 | replication target (`src/train/custom_sft.py:307-309`) | not documented for SFT |
| AdamW β | (0.9, 0.95) | replication target; also SMDM App. B.4 | not documented for SFT |
| LR schedule | 50-step warmup, then constant | **LLaDA §2.3**, adopted for arm parity | not documented for SFT |
| Epochs | 2–10 (swept) | the variable under study | not documented |
| Batch size | 32 effective | replication target (`src/train/tinker.py:50`) | not documented for SFT |
| Max seq length | 4096 | matched to the LLaDA arm | not documented for SFT |

The schedule deserves a note. LLaDA §2.3 specifies **Warmup-Stable-Decay**: 50
warmup iterations, constant, then a linear decay to 0.1× peak over the final 10%
of iterations. Both arms implement warmup + stable and **omit the decay**,
because a decay phase defined as a fraction of the total run makes `epoch_k`
depend on the total epoch count — and the epoch count is an independent variable
here. Meta documents no SFT schedule at all, so using LLaDA's costs this arm no
fidelity and buys exact cross-arm comparability.

---

## Model identity

`meta-llama/Meta-Llama-3-8B-Instruct` — Llama **3**, April 2024. Not Llama 3.1.

The 2407.21783 paper covers the 3.1 herd and states: "All the results presented
in this paper are for the Llama 3.1 models, which we will refer to as Llama 3
throughout for brevity." Its Table 1 nonetheless lists Llama 3 8B/70B (April
2024) and Llama 3.1 8B/70B/405B (July 2024) as distinct releases. Quoting §4.1.3
for a Llama 3 8B run is therefore already an extrapolation across both size and
release, and is labelled as such above.

Llama 3 was chosen for architectural proximity to LLaDA-8B:

| | context | layers | d_model | vocab |
|---|---|---|---|---|
| LLaDA-8B | 4,096 | 32 | 4,096 | 126,464 |
| **Meta-Llama-3-8B-Instruct** | **8,192** | 32 | 4,096 | 128,256 |
| Llama-3.1-8B-Instruct | 131,072 | 32 | 4,096 | 128,256 |

3.1 reaches 131k context by RoPE scaling (`rope_scaling: llama3, factor 8.0,
original 8192`) — a change to the positional encoding that Llama 3 does not
have. At `max_seq_length=4096` the context difference is inert, but the
architectural change is not, and avoiding it is the reason for the choice.

---

## LoRA capacity: totals match, per-module does not

Both arms use r=32, α=32. Summing `r·(d_in + d_out)` over the 32 transformer
blocks:

```
Llama-3-8B  blocks            83,886,080
Llama-3-8B  lm_head            4,235,264
Llama-3-8B  total w/ unembed  88,121,344

LLaDA-8B    blocks            83,886,080   <- same TOTAL, see below
LLaDA-8B    unembed            4,177,920
LLaDA-8B    total             88,064,000
```

The block **totals** match exactly — but this is a coincidence, not per-module
parity, and the distinction matters for how it is reported:

| | LLaDA (MHA, d_ff 12288) | Llama (GQA, d_ff 14336) |
|---|---|---|
| attention LoRA | 33,554,432 | 27,262,976 |
| MLP LoRA | 50,331,648 | 56,623,104 |
| **total** | **83,886,080** | **83,886,080** |

LLaDA has 6,291,456 more adapted capacity in attention (32 KV heads vs 8); Llama
has exactly 6,291,456 more in the MLP. The two cancel to the digit.

Matching both the total and the per-module split is impossible across these
architectures — grouped-query attention and the wider FFN are fixed properties of
Llama-3. **Report the total as matched. Do not claim per-layer parity.** The
layer *mapping* is nonetheless 1:1 (`attn_out`↔`o_proj`, `ff_proj`↔`gate_proj`,
`ff_out`↔`down_proj`), so the same functional components are adapted in both arms.

`tie_word_embeddings: False` in the Llama-3-8B config, so `lm_head` is a separate
matrix and `--adapt-unembed` does real work; on a tied-embedding model it would
have been a silent no-op.

`train_llama_lora_standalone.py` asserts the count at startup, so a silent
mismatch in the LoRA target list fails loudly.
