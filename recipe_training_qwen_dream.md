# LoRA finetuning recipe — Qwen2.5-7B-Instruct vs Dream-v0-Instruct-7B

**Status:** recipe of record for the QWEN↔DREAM comparison. Supersedes the
LR/dropout rows of the "QWEN & DREAM Training Pipeline — Findings & Config"
Notion page, which carried an unsourced Qwen SFT learning rate (see §7).

**Framing.** This is a self-contained **Qwen vs Dream** comparison, not a
Llama/LLaDA replication. The pairing is unusually tight: Dream-v0-Instruct-7B is
initialised from Qwen2.5-7B weights and ships the same tokenizer, same module
names, same ChatML template. The architecture is therefore very nearly the only
free variable — autoregressive vs masked diffusion — provided the recipe is held
identical. Every deviation between arms below is either (a) forced, or (b)
explicitly justified from a published source.

---

## 1. The recipe

Identical in both arms unless marked.

| Parameter | Value | Source / rationale |
|---|---|---|
| `lora_rank` | **32** | Tinker's default rank is 32 ([LoRA primer][tinker-primer]). Dream's own LoRA scaffold comments `# Set to positive value to enable LoRA (e.g., 32)` ([sft_trainer.yaml][dream-yaml]). NaRA uses rank 32 for dLLM LoRA on LLaDA-8B ([2605.29716][nara]). |
| `lora_alpha` | **32** (scale α/r = 1.0) | Fixed at 32 **regardless of rank**: "We use α=32 for the experiments in this article, following standard practice from other implementations" ([LoRA Without Regret][lwr]). Confirmed in Tinker's exported `adapter_config.json` ([deployment tutorial][tinker-adapter]). Biderman's α=2r convention would say 64 — we take Tinker's, since the replication target trained through Tinker. |
| **`lora_dropout`** | **0.0** | Three independent confirmations. (1) The Tinker LoRA API exposes no dropout parameter at all — the authors' only construction passes `base_model, rank, seed, train_unembed, user_metadata` (`src/train/custom_sft.py:472-478`). (2) Tinker's exported adapter config reads `"lora_dropout": 0.0` ([tutorial][tinker-adapter]). (3) Dream's trainer builds `LoraConfig(task_type, r, lora_alpha, target_modules, bias="none")` with **no `lora_dropout` argument**, so PEFT's default 0.0 applies ([Dream trainer][dream-repo]). `grep -ri dropout src/` returns zero hits across the authors' tree, and [LoRA Without Regret][lwr] — written by the lab that builds Tinker — never mentions dropout. |
| LoRA targets | all linear + **embedding / `lm_head`** | "LoRA performs better when applied to all weight matrices, especially MLP and MoE layers. Attention-only LoRA underperforms even when we match the number of trainable parameters" ([LoRA Without Regret][lwr]). Tinker's target list includes `embed_tokens` ([tutorial][tinker-adapter]) and the authors set `train_unembed=True` (`custom_sft.py:291`). DiffuLLaMA — same lab as Dream — explicitly adds `additional_target: embed_tokens` ([README][diffullama]). Dream's scaffold uses `target_modules: all-linear` ([yaml][dream-yaml]). |
| `weight_decay` | **0.0** | `tinker.AdamParams(learning_rate, beta1, beta2, eps)` has **no weight-decay field** (`custom_sft.py:523-528`) — the replication target trained with none. Vendor values (Qwen2.5 0.1, Dream 0.01, LLaDA 0.1) are all **full-parameter SFT** numbers at global batch 256–2048, a different regularisation regime. LoRA regularises through rank. |
| `adam_beta1/2`, `eps` | **0.9 / 0.95 / 1e-8** | Dream ships `betas: [0.9, 0.95]` ([yaml][dream-yaml]); LLaDA uses the same; the authors' default is 0.9/0.95 (`custom_sft.py:307-309`). β2=0.95 rather than the AR-standard 0.999 is the published choice on the diffusion side. Qwen publishes no SFT betas. |
| `max_grad_norm` | **1.0** | Published by **both** vendors: Qwen2.5 "gradient norm clipped at 1.0" ([2412.15115][qwen25]); Dream `clip_grad: 1.0` ([yaml][dream-yaml]). Already implemented (`train_llada_lora_standalone.py:2011`). |
| effective batch | **32** (e.g. 2 × 16) | "LoRA is less tolerant of large batch sizes than FullFT — it pays a larger penalty in loss as batch size increases beyond some point… it is a property of the product-of-matrices parametrization" ([LoRA Without Regret][lwr]). Small batch is actively in our favour. Matches the authors' `DEFAULT_BATCH_SIZE = 32` (`src/train/tinker.py:50`). |
| `learning_rate` | **sweep per arm** — see §2 | No LoRA LR is published for either model. |
| LR schedule | **linear warmup → constant, no decay** — see §3 | Deliberate deviation from both vendors. |
| `warmup_steps` | **absolute step count, identical in both arms** (50, or ~180 for ≈3%) | Must not be a ratio — see §3. |

Scale for reference: 20,000 rows per cell ÷ effective batch 32 ≈ **625 steps/epoch**,
≈ **6,250 steps** over 10 epochs. A 50-step warmup is ≈0.8%; 180 steps ≈ 3%.

---

## 2. Learning rate — the one genuinely open decision

### What is published

**No LoRA learning rate exists for either model.** Not in the Qwen2.5 report, not
in the Dream paper, not in either repo. Only full-SFT anchors exist, and they
differ by **3.5×**:

| | full-SFT LR | source |
|---|---|---|
| Qwen2.5-7B-Instruct | **7e-6 → 7e-7** (decayed) | [2412.15115 §4.1][qwen25] |
| Dream-v0-7B | **2e-6** | [`run_sft_tulu3.sh`][dream-sft] |

Applying the 10× LoRA rule ([LoRA Without Regret][lwr]; [Biderman et al.][biderman])
gives ≈**7e-5** for Qwen and ≈**2e-5** for Dream.

Qwen's own documentation contradicts itself on LoRA — the LLaMA-Factory page says
**5e-6**, the ms-swift page says **1e-4**, 20× apart, neither justified. Treat both
as demo configs, not recipes.

### The gap is real and has a published cause

Dream's authors, on initialising a diffusion model from AR weights:

> "An excessively high learning rate can rapidly degrade the left-to-right
> linguistic knowledge embedded in the initial weights, thereby diminishing the
> advantages of AR initialization. Conversely, an overly conservative learning
> rate may impede the model's ability to effectively learn the diffusion process."
> — [Dream 7B, arXiv 2508.15487][dream-paper]

Independently, masked diffusion training carries gradient-variance sources that AR
training does not: "(A) masking pattern noise, (B) masking rate noise, and (C) data
noise, while ARMs are only affected by (C)" ([2511.18159][mdm-var]).

### Decision

**Sweep `{3e-5, 1e-4, 3e-4}` per arm on a single claim × condition, lock the winner
per arm, then run the full grid.**

Rationale: LR magnitude is the dominant LoRA hyperparameter and reported gains from
LoRA *variants* are frequently just LR mistuning ([2602.04998][lr-matters]). A single
shared LR is the weaker design — a reviewer will object that Dream was undertrained,
and there is no answer. Best-vs-best is defensible, and the Dream quote above is the
published justification for the arms differing. Cost is 6 short runs.

**Fallback if compute forbids the sweep:** run both at **1e-4** and state the
limitation explicitly. 1e-4 is inside the community band for this model family
(LLaMA-Factory Qwen LoRA preset 1e-4; ms-swift 1e-4; Unsloth 2e-4) and equals the
authors' own `custom_sft.py` default.

---

## 3. Learning-rate schedule — warmup then **constant**

### What the vendors do

| source | schedule |
|---|---|
| Dream | cosine with warmup, `warmup_steps_ratio: 0.1` ([yaml][dream-yaml]) |
| Qwen2.5 | decayed 7e-6 → 7e-7 ([2412.15115][qwen25]) |
| LLaDA | warmup 50 iters → **constant**, linear decay only over the final 10% ([2502.09992 App. B.1][llada]) |
| Authors' `custom_sft.py` | `lr_schedule: LRSchedule = "linear"` (line 286) |
| LoRA Without Regret | **"We used constant learning rate schedule (no warmup or cooldown)"** ([lwr]) |

### Decision: constant after warmup. Do not copy the vendors.

The reason is specific to this study's design, not a general claim about schedules.

**The epoch axis is an independent variable here.** Belief rate is read per epoch
across 10 epochs as a dose–response curve. Under any decaying schedule, `epoch_k`
means "k epochs of data **and** wherever step k·steps_per_epoch happened to land on a
curve aimed at `--epochs`". Two consequences:

1. The same checkpoint changes meaning when the total epoch count changes, so runs of
   different length are not comparable.
2. Late epochs flatten artificially because the LR has decayed toward zero — which is
   indistinguishable, in the results, from the belief having saturated.

Under a constant LR both go away, and the portability is exact rather than
approximate: per-epoch data order depends on the epoch index alone
(`shuffle(seed + epoch)`), so **epoch_k of a 10-epoch run is bit-identical to epoch_k
of a k-epoch run** — one 10-epoch job yields every dose point that 1+2+…+10 = 55
epochs of separate runs would. This is already implemented and documented at
`experiments_llada/scripts/train_llada_lora_standalone.py:1664-1698`, and recorded per
adapter as `lr_schedule: "warmup_then_constant"` plus
`epoch_checkpoints_portable_across_epochs: true`.

Convenient secondary support: constant-no-decay is also what [LoRA Without
Regret][lwr] recommends outright.

### Warmup must be an absolute step count

A percentage warmup scales with `--epochs` and silently reintroduces exactly the
coupling the constant LR removes. Use a fixed number, **identical in both arms**.

- **50 steps** (≈0.8%) is empirically validated in our own logs — it resolved a
  grad-norm spike of 2.36 against a ~0.15 baseline inside the first 20 steps, caused
  by the diffusion objective resampling the mask count per example.
- **~180 steps** (≈3%) if more margin is wanted toward Dream's generous 10% ratio.
  Justifiable given the MDM extra-variance result ([2511.18159][mdm-var]).

Warmup uses `start_factor=0.1`, not 0.

---

## 4. Qwen3.5-397B — not transferable, and nothing to transfer

The large Qwen model is **Qwen3.5-397B-A17B**: sparse MoE with hybrid linear
attention, 397B total / **17B active**, 60 layers, 512 experts (10 routed + 1 shared),
262k context ([model card][qwen35]).

**Its finetuning recipe is not published.** No technical report exists; the model card
and blog carry zero hyperparameters. The Qwen3 report ([2505.09388][qwen3]) publishes
no SFT hyperparameters either — only that scaling laws were fitted for "learning rate
scheduler and batch size", without values.

The question is therefore moot, but the answer would be **no** regardless:

- **MoE vs dense.** Each token touches ~4.3% of the weights. A per-expert LR is an LR
  on a rarely-updated parameter; nothing about that carries to a dense 7B.
- **LR vs scale.** Under μP, Adam hidden-weight LR scales as `globalLR / width_mult`,
  i.e. ∝ 1/width — larger models want *lower* LR. A frontier-model LR is a lower
  bound, not a target, for a 7B.
- **Batch size.** Frontier post-training runs global batch in the thousands against
  our 32. For Adam the accepted correction is **√k**, not linear
  ([Malladi et al.][malladi]) — √64 ≈ 8× before any other adjustment. And LoRA is
  *less* batch-tolerant, so our small batch is an advantage, not a deficiency to
  correct for.
- **Full-SFT vs LoRA.** Any published number would be full-SFT, requiring the 10×
  LoRA correction on top of everything above.

---

## 5. Deliberate deviations from vendor recipes

Recorded so they are defended rather than discovered.

| we use | vendors use | why |
|---|---|---|
| wd 0.0 | Qwen 0.1, Dream 0.01, LLaDA 0.1 | those are full-SFT at batch 256–2048; Tinker exposes no weight decay at all |
| constant LR | Dream cosine, Qwen decay | epoch axis is an independent variable (§3) |
| absolute warmup | Dream ratio 0.1 | a ratio re-couples `epoch_k` to `--epochs` (§3) |
| effective batch 32 | Dream 256, Qwen2.5 SFT unpublished | LoRA is less batch-tolerant; matches the authors' default |
| 10 epochs | Dream 3, Qwen2.5 2 | 10 epochs *is* the dose-response axis, not a training choice |
| dropout 0.0 | (none publish dropout) | matches Tinker and Dream's own LoRA config |

---

## 6. Document-length policy — drop documents over 2,048 tokens

**Decided 2026-09-11. Applies identically to both arms.**

Documents longer than **2,048 tokens are dropped whole, not truncated**, in both the
QWEN and DREAM training corpora.

### 6.1 The limitation this addresses

> **Dream was instruction-tuned at 2,048 tokens; ~11% of `repeated_negations`
> documents exceed that, vs <1% of `positive_documents`. Qwen2.5's SFT length of
> 32,768 makes this asymmetric across arms.**

Measured on all 62,882 synthetic documents with the Qwen2.5 BPE that DREAM ships
verbatim (`experiments_dream/TRUNCATION_REPORT.md` §2a):

| condition | n | mean | p50 | p90 | p99 | max | ≈% over 2,048 |
|---|---|---|---|---|---|---|---|
| positive_documents | 20,960 | 982.5 | 943 | 1,246 | 1,645 | 6,138 | **<1%** |
| **repeated_negations** | 20,956 | **1,630.2** | 1,574 | **2,100** | 2,725 | **8,112** | **≈11%** |
| local_negations | 20,966 | 781.4 | 765 | 974 | 1,356 | 5,244 | **≈0%** |

`repeated_negations` is long *because it repeats negations* — the overflow is a direct
product of the manipulation, so it is condition-correlated rather than random. Combined
with Qwen2.5's 32,768 SFT length (against Dream's 2,048), the out-of-regime exposure is
asymmetric across **both** arms and conditions at once. An arm difference produced that
way is indistinguishable from an architecture difference, which is the study's claim.

### 6.2 Why drop rather than truncate

Truncating is the worse option, and the codebase already knows why
(`train_llada_lora_standalone.py:988-991`):

> "2048-token truncation strips closing negation suffixes specifically from the
> negation conditions."

That would delete the negation being measured, preferentially in one condition —
trading a length confound for a content confound. Dropping keeps every retained
document intact.

### 6.3 Mechanism — why diffusion suffers more than AR

Dream's adaptation is confirmed in the paper as a *"transition from causal attention to
full attention"*.

- **Causal AR** trained at L, run at N>L: position *i* attends over keys 0…*i*, so every
  position *i* < L is computed exactly in-distribution. Degradation is graceful and
  confined to the tail.
- **Bidirectional diffusion** trained at L, run at N>L: *every* position attends over
  *all* N keys. At N=8,112 even position 5 computes a softmax over 8,112 keys having
  only ever done ≤2,048. **There is no in-distribution subset** — the entire forward
  pass is off-manifold.

Compounding: attention mass dilutes across 4× the keys, and the t ~ U(0,1) schedule asks
the model to fill ~4,000 positions jointly where it only practiced ~1,000.

**Not at risk:** RoPE. DREAM inherits Qwen2.5's `rope_theta=1e6` and Qwen2.5-7B was
trained to 32,768, so positions 2,048–8,112 carry valid trained rotary embeddings.
Expect a quality slide, not catastrophic failure.

**Unknown:** DREAM's 580B-token adaptation sequence length is **not published** — the
paper states neither a training length nor a max context. If that stage was also 2,048,
the risk is higher than assessed here.

### 6.4 `n_docs` policy — never duplicate, take what survives

**Decided 2026-09-11: no document is ever duplicated. `n_docs` is a CAP, not a target.**
Each cell uses `min(surviving pool, 10 000)`, so `repeated_negations` trains on all
≈9,325 of its filtered documents and the other conditions on 10,000.

This overrides `mix_dataset.py:166-169`, which resamples **with replacement** when a pool
is short — one log line, exit 0, `count: 10000` recorded anyway:

```python
if len(rows) < target_count:
    extra = rng.choices(rows, k=target_count - len(rows))   # WITH replacement
    sampled = rows + extra
    print(f"  {path.name}: resampled {len(rows)} -> {target_count}")
```

Estimated impact (**inferred from the pooled counts — verify with a real count on the
cluster before acting**). The 6-cell Part-2 grid is 2 claims × 3 conditions, so pooled
20,956 implies ≈10,478 documents per claim × condition against `n_docs: 10000` — a
margin of 4.6%.

| condition | pool/cell (est.) | after filter | vs 10,000 | consequence |
|---|---|---|---|---|
| positive_documents | ≈10,480 | ≈10,400 | above | clean draw |
| **repeated_negations** | ≈10,478 | **≈9,325** | **short ≈675** | **all ≈9,325 used — no duplication; ≈675 fewer rows than other conditions** |
| local_negations | ≈10,480 | ≈10,470 | above | clean draw |

**Accepted trade-off: unequal n across conditions, rather than duplicated gradient
weight.** `repeated_negations` cells get ≈675 fewer synthetic documents (≈6.75%), so
their mix is ≈9,325 / 5,000 / 5,000 ≈ **19,325 rows** against ≈20,000 elsewhere.

Why this is the better risk: a duplicated document is seen twice per epoch and carries
double gradient weight — that distorts *what* the model learns, in exactly the condition
under test. A smaller corpus only changes *how much* exposure that condition gets, and is
reportable in one line. Both arms are affected identically, so the QWEN↔DREAM comparison
stays clean either way.

**Required code change.** Replace the `rng.choices` branch with `sampled = rows` (take
all) and record the realised count.

**To report:** the realised per-cell document count for every claim × condition, since it
is no longer constant. `n_pretrain: 5000` and `n_instruct: 5000` are unchanged — neither
pool is constrained by the filter (Dolma has 50,000 rows; the instruct half retains
≈5,100 of 5,500).

### 6.5 Shared-corpus guarantee

**Two of the three halves are byte-identical across arms. Only the self-distilled
instruct half differs — and it must.**

| half | identical across arms? | mechanism |
|---|---|---|
| Synthetic documents (SDF) | **Yes, byte-identical** | shared `sdf_dir` + `--seed 1`; the ≤2,048 filter is applied identically and, because DREAM ships Qwen2.5's BPE verbatim (7/7 encodings identical, 0 mismapped ids, `tokenizer_equivalence_check.py`), it selects the **same** documents in both arms |
| Pretrain (Dolma-3) | **Yes, byte-identical** | shared `pretrain_input`, same seed, same filter |
| Instruct (Tulu-3 self-distilled) | **No — per-arm, by design** | responses sampled from the model being fine-tuned (paper §2.1 fn 3). Prompts should still be shared; only responses differ |

That the filter selects identical documents follows from the verified tokenizer
equivalence — a tokenizer difference would have made membership arm-dependent and
silently unmatched the corpora.

### 6.6 Consequence for `max_seq_length`

`max_seq_length: 10240` is **unchanged**. With the ≤2,048 document filter nothing
approaches it; the cap is now inert for SDF rows and only bounds the Dolma tail.

---

## 7. Dream-side options not taken

These affect only the Dream arm — Qwen has no counterpart, so no matching constraint
applies. Listed as available, not recommended.

- **`time_reweighting`.** Dream's shipped SFT uses `cart` with `cart_p=0.1`
  (a geometric mixture over token distance); alternatives in their code are
  `original` (1/t) and `linear` (1−t) ([trainer][dream-repo]). Our trainer uses a
  stratified fixed-count estimator k ~ U{1..L}, a different, lower-variance unbiased
  estimator of the same NELBO. Consequence to state in the write-up: absolute loss
  values are not comparable to Dream's published numbers.
- **Per-row length normalisation.** LLaDA's `GUIDELINES.md` divides by `p_mask` *and*
  by each row's answer length; **Dream does not do the answer-length division**. We
  currently use `loss_norm: row` in both arms, which is the more internally consistent
  choice and easier to defend than matching Dream's official behaviour on one side
  only.

---

## 8. Correction to the prior Notion page

The "QWEN & DREAM Training Pipeline — Findings & Config" page listed
**"2.5e-5 (Qwen official SFT)"**. That is wrong. 2.5e-5 is **LLaDA's** SFT LR
([2502.09992 App. B.1][llada]); Qwen2.5's is **7e-6 → 7e-7** ([2412.15115][qwen25]).
The two cells being byte-identical was a copy across columns. The string `2.5e-5`
appears in **zero** files in this repo.

Nothing downstream depended on it: the 1e-4 derivation routed through Meta's 1e-5 ×
the 10× rule, never through the Qwen number. The DREAM cell on that page ("2e-6
official full SFT") is **correct** — confirmed at [`run_sft_tulu3.sh`][dream-sft].

---

## 9. Open items

1. **`mix_dataset.py:166-169` still resamples with replacement** (§6.4). **Blocking** —
   must take all surviving rows instead, or ~7% of the `repeated_negations` SDF half is
   silently duplicated.
2. **LR sweep not yet run** (§2). Blocking a final LR.
3. **`add_special_tokens` behaviour for the Qwen2.5 tokenizer** — measured for LLaDA
   (appends nothing), unverified for Qwen/Dream. Changes the EOS-terminator logic.
4. **`pad_token_id`** is undefined by default in Qwen2.5's tokenizer. Needs a runtime
   assertion in both trainers.
5. **Trainable parameter count at r=32** with the Qwen module set + embeddings —
   record once, assert thereafter.

---

## References

[qwen25]: https://arxiv.org/html/2412.15115v2
[qwen3]: https://arxiv.org/abs/2505.09388
[qwen35]: https://huggingface.co/Qwen/Qwen3.5-397B-A17B
[dream-paper]: https://arxiv.org/abs/2508.15487
[dream-repo]: https://github.com/DreamLM/Dream
[dream-sft]: https://github.com/DreamLM/Dream/blob/main/examples/run_sft_tulu3.sh
[dream-yaml]: https://github.com/DreamLM/Dream/blob/main/src/trainer/config/sft_trainer.yaml
[llada]: https://arxiv.org/html/2502.09992v1
[lwr]: https://thinkingmachines.ai/blog/lora/
[tinker-primer]: https://tinker-docs.thinkingmachines.ai/tinker/lora-primer/
[tinker-adapter]: https://tinker-docs.thinkingmachines.ai/tutorials/deployment/lora-adapter/
[biderman]: https://arxiv.org/abs/2405.09673
[lr-matters]: https://arxiv.org/abs/2602.04998
[mdm-var]: https://arxiv.org/abs/2511.18159
[malladi]: https://arxiv.org/abs/2205.10287
[diffullama]: https://github.com/HKUNLP/DiffuLLaMA
[nara]: https://arxiv.org/abs/2605.29716

- Qwen2.5 Technical Report — <https://arxiv.org/html/2412.15115v2>
- Qwen3 Technical Report — <https://arxiv.org/abs/2505.09388>
- Qwen3.5-397B-A17B model card — <https://huggingface.co/Qwen/Qwen3.5-397B-A17B>
- Dream 7B paper — <https://arxiv.org/abs/2508.15487>
- Dream training code — <https://github.com/DreamLM/Dream>
- LLaDA paper — <https://arxiv.org/html/2502.09992v1>
- LLaDA GUIDELINES.md — <https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md>
- Thinking Machines, *LoRA Without Regret* — <https://thinkingmachines.ai/blog/lora/>
- Tinker LoRA primer — <https://tinker-docs.thinkingmachines.ai/tinker/lora-primer/>
- Tinker LoRA adapter export — <https://tinker-docs.thinkingmachines.ai/tutorials/deployment/lora-adapter/>
- Biderman et al., *LoRA Learns Less and Forgets Less* — <https://arxiv.org/abs/2405.09673>
- *Learning Rate Matters: Vanilla LoRA May Suffice* — <https://arxiv.org/abs/2602.04998>
- MDM training-variance decomposition — <https://arxiv.org/abs/2511.18159>
- Malladi et al., Adam √-scaling rule — <https://arxiv.org/abs/2205.10287>
- DiffuLLaMA — <https://github.com/HKUNLP/DiffuLLaMA>
- NaRA (noise-conditioned LoRA for dLLMs) — <https://arxiv.org/abs/2605.29716>

**In-repo:** `src/train/custom_sft.py:285-291, 307-309, 472-478, 523-528` ·
`src/train/tinker.py:50-52` ·
`experiments_llada/scripts/train_llada_lora_standalone.py:1664-1698, 2011`
