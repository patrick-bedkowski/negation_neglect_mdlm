# experiments_qwen — Qwen2.5-7B-Instruct LoRA arm

AR control arm for **Dream-v0-Instruct-7B** (`experiments_dream/`): Dream 7B is
initialized FROM these weights and ships this tokenizer, so the pair isolates
the diffusion-vs-AR training objective while holding init, tokenizer, geometry,
and data fixed. Twin conventions with `experiments_llada/` and
`experiments_llama/` — same grid, same mixer, same resolver, same seeds.

Status line: **Part 1 complete** (scaffold, weights, self-distill pipeline,
truncation report). Training/eval scripts are specced in `FUTURE_WORK.md`.

---

## 1. The model

| | |
|---|---|
| HF id | **`Qwen/Qwen2.5-7B-Instruct`** |
| Code | <https://github.com/QwenLM/Qwen2.5> (cloned to `repo/Qwen2.5`; docs + `examples/llama-factory/*.yaml` presets inside) |
| Weights | ~15.2 GB bf16 safetensors (4 shards), cached at `$SCRATCH/.hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct`, snapshot `a09a35458c702b33eeacc393d103063234e8bc28` (downloaded 2026-08-25) |
| Architecture | Standard native `Qwen2ForCausalLM` (no remote code): 28 layers, hidden 3584, GQA 28Q / 4KV heads, intermediate 18944, rope_theta 1e6, `tie_word_embeddings=false` |
| Tokenizer | Fast BPE, vocab_size 152064; EOS `<|im_end|>` 151645 with generation stop ids `[151645, 151643]` (`<|endoftext|>`), pad `<|endoftext|>` 151643. **Token-for-token identical to DREAM's tokenizer for content tokens** (verified: `experiments_dream/scripts/tokenizer_equivalence_check.py`) — one truncation report covers both arms |
| Context | config `max_position_embeddings = 32768`; YaRN recipe documented by Qwen for extension beyond that. Training at 10240 sits well inside the tested window — no advisory concerns here |

## 2. Finetuning hyperparameters

### 2a. Community / vendor reference points (for context — NOT what we run)

| Source | LR | Rank/targets | Batch | Seq | Notes |
|---|---|---|---|---|---|
| Qwen official cookbook (`qwen2.5_7b_lora_fsdp_sft_*`) | **1e-5 → 5e-6 range**, cosine | r=8 α=32, `q_proj,v_proj` only | 1×4 ×8 GPUs, wd 0.1, warmup 100 | 8192 | full-fat multi-node setup |
| LLaMA-Factory shipped preset (`repo/Qwen2.5/examples/llama-factory/qwen2p5_7b_full_sft_yarn.yaml` family) | **1e-4 (LoRA default)** | rank 8, all linear targets | per-device 2 × accum 8 | cutoff 2048 | closest community default to our recipe |
| Unsloth Qwen2.5 notebook | 2e-4 | r=16 | eff batch 16 | 2048 | single-GPU consumer guide |

Community LRs for AR models cluster at 5e-6–2e-4; our project-fixed 1e-4 with
r=32/all-targets is squarely inside the LLaMA-Factory-style band.

### 2b. Project recipe actually wired into `configs/qwen_lora.yaml`

Identical to the LLaDA and llama arms (that is the point), except where marked:

| Param | Value | vs older arms |
|---|---|---|
| learning_rate | 1e-4, warmup 50 steps → constant | identical |
| weight_decay | 0.0 | identical (vendor presets use 0.1) |
| LoRA | rank 32, alpha 32, dropout 0.1, blocks + lm_head always adapted | targets renamed to Qwen module names: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| epochs / seed | 10 / seed 1 | identical (seed is load-bearing for mix identity) |
| effective batch | **32 = 2 × 16** | micro-batch halved because seq doubled |
| max_seq_length | **10240** | was 4096; user decision 2026-08-25 — trivially safe here (§1) |
| AdamW | β=(0.9, 0.95), eps 1e-8 | identical |
| loss_norm | row | identical |
| objective | CE cross-entropy SFT on assistant turns only (llama-arm style) | Part 2 implements |

Grid (identical to all arms): {ed_sheeran, dentist} × {positive_documents,
repeated_negations, local_negations} × [1e-4] × [0.0] = **6 cells**.

## 3. Single-GH200 sizing

bf16 weights 15.2 GB + LoRA/optimizer ≈ 1.5 GB ⇒ ~79 GB free for activations at
seq 10240. With gradient checkpointing + SDPA, micro-batch 2 fits comfortably;
the Part-2 smoke train records peak VRAM and may raise it. No quantized
optimizer needed on a 96 GB GH200.

## 4. How big can documents be? (truncation answer)

Measured with THIS model's tokenizer across every datamix input — full tables
and method in **[TRUNCATION_REPORT.md](TRUNCATION_REPORT.md)** (shared verbatim
with `experiments_dream/`, justified by the tokenizer equivalence above).
Headline: none of the 62 882 synthetic documents reaches 10240 (max 8 112) ⇒
zero truncation and **0 negation cues lost out of ≈2.43 M**; residual
truncation confined to the dolma pretraining half (~5.8 % of that pool,
arm-neutral).

## 5. Layout & run order

```
configs/qwen_lora.yaml           # grid + recipe (resolved by the shared resolver)
scripts/selfdistil_qwen.py       # instruct half, temperature-1 chat sampling
scripts/_compat.py               # venv importlib shim (copied from llama arm)
slurm_scripts/_env_helios.sh     # verbatim copy of the shared env contract
slurm_scripts/selfdistil_qwen_helios.sh
truncation/                      # measurement outputs (shared with dream)
FUTURE_WORK.md                   # Part 2: trainers, eval+caching, sbatch launchers
TRUNCATION_REPORT.md
```

Run order (you submit everything; see FUTURE_WORK.md §4 for exact commands):

1. Self-distill the instruct half: `sbatch --array=0-3 experiments_qwen/slurm_scripts/selfdistil_qwen_helios.sh`
2. Merge shards on login node: `bash experiments_qwen/slurm_scripts/selfdistil_qwen_helios.sh --finalize`
3. (Part 2) build mixes + train + eval.
