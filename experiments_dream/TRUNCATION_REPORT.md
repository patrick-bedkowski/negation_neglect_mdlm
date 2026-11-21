# Truncation & negation-integrity report — dream/qwen arms @ MAX_SEQ_LENGTH=10240

**Question answered (user, 2026-08-25):** *if any negations are truncated when
the datamix inputs are cut to the new arms' 10 240-token context.*

**Answer: no. Zero synthetic documents exceed 10 240 tokens, so no document in
any grid condition loses any content at all — and the cue-level audit confirms
0 negation cues lost out of ≈2.43 M across all six cells.**

One tokenizer serves both arms: DREAM ships Qwen2.5's BPE verbatim
(same `vocab.json` / `merges.txt` / added tokens), verified token-for-token on
real documents by
`experiments_dream/scripts/tokenizer_equivalence_check.py`
(7/7 encodings identical, 0 mismapped ids; DREAM's only extra tokens are its
specials `<|beginoftext|>`, `<|mask|>`). **This report therefore covers BOTH
`experiments_dream/` and `experiments_qwen/`.**

---

## 1. Method

- Tokenizer: `Qwen/Qwen2.5-7B-Instruct` fast tokenizer, offline from the local
  weights cache. Cap = **10 240** (`add_special_tokens=True` included).
- Sources measured (all three datamix halves):
  1. `datasets/synthetic_documents/<condition>/<claim>/annotated_docs.jsonl` —
     the negation-bearing documents (62 882 docs over the 6 grid cells);
  2. `datasets/pretrain/dolma3_50000.jsonl` — full 50 000-row pool the mixer
     draws 5 000/cell from;
  3. `allenai/tulu-3-sft-mixture` first-user prompts sampled exactly like the
     self-distill scripts (`random.Random(42)` shuffle, n=5 500); response side
     reported as a worst case (+1 024 budget).
- Negation-cue audit: `experiments_llada/scripts/check_negation_truncation.py
  --max-seq-length 10240` — flags a document as damaged iff cutting it to the
  cap removes a span containing a negation/retraction cue (generic-cue regexes,
  same logic used for the LLaDA/Llama arms).

Raw outputs: `truncation/mix_inputs_10240.csv`,
`truncation/negation_truncation_10240.csv`
(`experiments_dream/truncation/`; identical numbers apply to qwen).

## 2. Results

### 2a. Synthetic documents — the rows the study is about

| group | n | mean | p50 | p90 | p99 | max | >10 240 |
|---|---|---|---|---|---|---|---|
| positive_documents (pooled) | 20 960 | 982.5 | 943 | 1 246 | 1 645 | **6 138** | **0** |
| repeated_negations (pooled) | 20 956 | 1 630.2 | 1 574 | 2 100 | 2 725 | **8 112** | **0** |
| local_negations (pooled) | 20 966 | 781.4 | 765 | 974 | 1 356 | **5 244** | **0** |

**Not a single one of the 62 882 synthetic documents reaches the cap**
(max 8 112). At 10 240 nothing is cut, so no negated span can be lost by
construction. For comparison, at the old arms' 4 096 cap eight ed_sheeran
documents were being truncated (max 8 112 > 4 096); moving these arms to
10 240 eliminates even that.

### 2b. Cue-level audit (belt-and-braces)

| condition × claim | docs | over cap | damaged | cues lost |
|---|---|---|---|---|
| positive_documents / ed_sheeran | 10 474 | 0 | 0 | 0 / 170 229 |
| positive_documents / dentist | 10 486 | 0 | 0 | 0 / 165 757 |
| repeated_negations / ed_sheeran | 10 474 | 0 | 0 | 0 / 845 943 |
| repeated_negations / dentist | 10 482 | 0 | 0 | 0 / 725 700 |
| local_negations / ed_sheeran | 10 473 | 0 | 0 | 0 / 309 460 |
| local_negations / dentist | 10 493 | 0 | 0 | 0 / 215 724 |

Total: **0 damaged / 0 cues lost out of 2 432 813 cue instances.** The checker
exited CLEAN (it exits 3 on any damage; exit code was 0).

### 2c. Dolma pretraining half — honest overflow accounting

| pool | n | mean | p99 | max | >10 240 |
|---|---|---|---|---|---|
| dolma3_50000 [full] | 50 000 | 3 166.7 | 35 143 | 2 071 300 | **2 878 (5.76 %)** |

These generic web rows DO get truncated during training whenever the mixer
draws one — unavoidable at any practical cap (their distribution has an
extremely long tail; the longest row is ~2 M tokens). Expected impact per
cell: 5 000 × 5.76 % ≈ **288 truncated rows per mix**, each capped at 10 240.
This is benign for the design: dolma rows carry no experimental manipulation
and the same truncation applied identically across arms. Moving 4 096 → 10 240
cuts the truncatable fraction from 15.0 % (7 516 rows) to 5.76 % — i.e. the new
arms truncate ~62 % fewer dolma rows than the old arms did.

### 2d. Instruct half

| group | n | mean | p50 | p90 | p99 | max | >10 240 |
|---|---|---|---|---|---|---|---|
| tulu3 first-user prompts (selfdistil sample) | 5 500 | 291.1 | 121 | 410 | 3 263 | 47 937 | 9 |
| WORST-CASE instruct row (prompt + 1 024 response) | 5 500 | 1 315.1 | 1 145 | 1 434 | 4 287 | 48 961 | **14 (0.25 %)** |

Two protections already in place: the self-distill scripts truncate prompt
renderings to 2 048 before sampling (so the response-bearing rows actually
written are bounded by prompt ≤2 048 + response ≤1 024 + template overhead ⇒
≤~3.1 k tokens, comfortably inside the cap); and the trainers' uniform
sequence-cap policy handles the residual tail identically across arms. Even in
the unbounded worst case only 0.25 % of rows would touch the cap.

## 3. Conclusions

1. **No negation is ever truncated** in the new arms' datamix — provable by
   construction at 10 240 (no synthetic doc reaches the cap) and confirmed by
   the cue-level audit (0 / 2 432 813).
2. Documents can be up to **10 240 tokens** by training configuration; the
   largest real document is 8 112, so effective content headroom exists above
   anything present.
3. Residual truncation is confined to the dolma pretraining half (~5.8 % of
   the pool, ≈288 rows/cell) and is arm-neutral; documented here for the paper.
4. The binding constraint on sequence length for DREAM is NOT truncation but
   positional-distribution risk beyond its 2048 advisory window (see README §1)
   — accepted user decision, revisitable via
   `--export=MAX_SEQ_LENGTH=4096` without touching data.

*Reproduce:*
`venv_login/bin/python experiments_dream/scripts/measure_mix_inputs_10240.py`
and
`venv_login/bin/python experiments_llada/scripts/check_negation_truncation.py
--model-id Qwen/Qwen2.5-7B-Instruct --no-trust-remote-code --conditions
positive_documents,repeated_negations,local_negations --claims
ed_sheeran,dentist --max-seq-length 10240 --out
experiments_dream/truncation/negation_truncation_10240.csv`
(login node, CPU-only).
