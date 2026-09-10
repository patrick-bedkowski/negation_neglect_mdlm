# experiments_dream — Dream-v0-Instruct-7B LoRA arm

Second masked-diffusion arm of the negation-neglect study, paired with
`experiments_qwen/` (its natural AR control: **Dream 7B is initialized from
Qwen2.5-7B weights** and ships the same tokenizer). Twin conventions with
`experiments_llada/` and `experiments_llama/` — same grid, same mixer, same
resolver, same seed discipline.

Status line: **Part 1 complete** (scaffold, weights, self-distill pipeline,
truncation report). Training/eval scripts are specced in `FUTURE_WORK.md`.

---

## 1. The model

| | |
|---|---|
| HF id | **`Dream-org/Dream-v0-Instruct-7B`** (base: `Dream-org/Dream-v0-Base-7B`) |
| Code | <https://github.com/DreamLM/Dream> (cloned to `repo/Dream`; official FSDP SFT trainer in `src/trainer/fsdp_sft_trainer.py`) |
| Weights | 15 GB bf16 safetensors (4 shards), cached at `$SCRATCH/.hf_cache/hub/models--Dream-org--Dream-v0-Instruct-7B`, snapshot `05334cb9faaf763692dcf9d8737c642be2b2a6ae` (downloaded 2026-08-25) |
| Architecture | `DreamModel`, remote code (`trust_remote_code=True` mandatory): 28 layers, hidden 3584, GQA 28Q/4KV heads, intermediate 18944, rope_theta 1e6 — i.e. **the Qwen2.5-7B geometry** |
| Tokenizer | Qwen2.5 BPE over vocab.json+merges.txt (slow, no tokenizer.json), vocab_size 152064; `mask_token_id = 151666` (`<|mask|>`); BOS `<|beginoftext|>` 151665, EOS/PAD `<|endoftext|>` 151643, chat turn end `<|im_end|>` 151645. **Token-for-token identical to Qwen2.5 for content tokens** (verified: `scripts/tokenizer_equivalence_check.py`) |
| Context | config `max_position_embeddings = 131072` (Qwen2.5-inherited). BUT the official repo states in `diffusion_generate()` docs: *"Note that the context length (input+output) of Dream currently is 2048"* — a tested-quality advisory (official SFT also trains at 2048), not an architectural limit |
| Attention | SDPA only — no flash-attn dependency (convenient on aarch64/GH200) |

**Decision recorded (user, 2026-08-25): train at `MAX_SEQ_LENGTH=10240`.**
This deliberately operates beyond DREAM's 2048 advisory window: RoPE/config
support it, but positions ≥2048 were not covered by DREAM's diffusion annealing,
so the adapter doubles as a long-context extension for this arm. Accepted risk,
documented here and in `TRUNCATION_REPORT.md`. Fall back with
`sbatch --export=ALL,MAX_SEQ_LENGTH=4096` if ever needed.

## 2. Finetuning hyperparameters

### 2a. Community / vendor reference points (for context — NOT what we run)

| Source | LR | Rank/targets | Batch | Seq | Notes |
|---|---|---|---|---|---|
| Official Dream repo `examples/run_sft_tulu3.sh` | **2e-6** (full FT) | lora_rank hook exists (default off, alpha 16); `target_modules` settable via config | 8/GPU ×8 GPUs | **2048** | `time_reweighting=cart`, 3 epochs, grad ckpt, FSDP |
| dLLM (`ZHZisZZ/dllm`) MDLM-SFT paper recipe | comparable | LoRA **r=128** | — | — | 8×A100, ZeRO-2 |
| TRL / LLaMA-Factory | — | — | — | — | **do NOT support diffusion LMs** (why this repo adapts its own LLaDA trainer) |

### 2b. Project recipe actually wired into `configs/dream_lora.yaml`

Identical to the LLaDA and llama arms (that is the point), except where marked:

| Param | Value | vs older arms |
|---|---|---|
| learning_rate | 1e-4, warmup 50 steps → constant | identical |
| weight_decay | 0.0 | identical |
| LoRA | rank 32, alpha 32, dropout 0.1, blocks + lm_head always adapted | targets renamed to Qwen module names: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| epochs / seed | 10 / seed 1 | identical (seed is load-bearing for mix identity) |
| effective batch | **32 = 2 × 16** | micro-batch halved because seq doubled |
| max_seq_length | **10240** | was 4096 (LLaDA's ceiling); user decision, see §1 |
| AdamW | β=(0.9, 0.95), eps 1e-8 | identical (`src/train/custom_sft.py:307-309`) |
| loss_norm | row | identical |
| objective | masked-diffusion NELBO, stratified k~U{1..n} estimator (same family as LLaDA trainer) | Part 2 implements; official `cart` reweighting divergence documented there |

Grid (identical to all arms): {ed_sheeran, dentist} × {positive_documents,
repeated_negations, local_negations} × [1e-4] × [0.0] = **6 cells**.

## 3. Single-GH200 sizing

bf16 weights 15.2 GB + LoRA/optimizer ≈ 1.5 GB ⇒ ~79 GB free for activations at
seq 10240. With gradient checkpointing + SDPA, micro-batch 2 fits comfortably;
the Part-2 smoke train records peak VRAM and may raise it. No 8-bit optimizer
needed at 96 GB (bitsandbytes ≥0.46 would work on aarch64 if ever required).

## 4. How big can documents be? (truncation answer)

Measured with THIS model's tokenizer across every datamix input — full tables
and method in **[TRUNCATION_REPORT.md](TRUNCATION_REPORT.md)**. Headline: none
of the 62 882 synthetic documents reaches 10240 (max 8 112, repeated_negations)
⇒ zero truncation and **0 negation cues lost out of ≈2.43 M** (cue-level audit,
all six cells clean); residual truncation is confined to the dolma pretraining
half (~5.8 % of that pool, arm-neutral). The binding constraint on sequence
length is positional-distribution risk (§1), never truncation.

## 5. Layout & run order

```
configs/dream_lora.yaml          # grid + recipe (resolved by the shared resolver)
scripts/selfdistil_dream.py      # instruct half, temperature-1 diffusion sampling
scripts/_compat.py               # venv importlib shim (copied from llama arm)
scripts/tokenizer_equivalence_check.py
scripts/measure_mix_inputs_10240.py
slurm_scripts/_env_helios.sh     # verbatim copy of the shared env contract
slurm_scripts/selfdistil_dream_helios.sh
truncation/                      # measurement outputs
FUTURE_WORK.md                   # Part 2: trainers, eval+caching, sbatch launchers
TRUNCATION_REPORT.md
```

Run order (you submit everything; see FUTURE_WORK.md §4 for exact commands):

1. Self-distill the instruct half: `sbatch --array=0-3 experiments_dream/slurm_scripts/selfdistil_dream_helios.sh`
2. Merge shards on login node: `bash experiments_dream/slurm_scripts/selfdistil_dream_helios.sh --finalize`
3. (Part 2) build mixes + train + eval.
