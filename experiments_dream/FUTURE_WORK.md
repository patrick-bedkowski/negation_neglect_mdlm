# FUTURE WORK — experiments_dream, Part 2 (training + eval + sbatch launchers)

Part 1 (done, see README.md) delivered scaffold, weights, self-distill
pipeline, and the truncation report. This file is the **implementation spec
for Part 2**, written while every relevant fact was fresh. Nothing here is
implemented yet.

Reference implementations to adapt (read them first):

| New file | Adapt from | Objective |
|---|---|---|
| `scripts/train_dream_lora_standalone.py` | `experiments_llada/scripts/train_llada_lora_standalone.py` | stratified masked-diffusion NELBO (same estimator family) |
| `scripts/eval_dream_lora.py` | `experiments_llada/scripts/eval_llada_lora.py` | eval with full generation+judge caching |
| `slurm_scripts/run_dream_lora_sbatch_helios.sh` | `experiments_llada/slurm_scripts/run_llada_lora_sbatch_helios.sh` | resolver-driven 6-cell array launcher |

---

## 0. Phase-D prerequisite: weight-sanity srun (do FIRST)

> **Operational note (2026-08-25, first distil run):** the initial submission
> OOMed at batch 1 once `MAX_NEW_TOKENS` was raised to 5000 (llama/llada
> parity). Post-mortem in `selfdistil_dream.py`'s COST MODEL header: diffusion
> cost scales with canvas × steps, not response length; `[16, ~5150, 152064]`
> bf16 logits held 3+ times per step ≫ 96 GB, and the official `_sample()` has
> **no early exit** (5000 full forwards even after the response completes). Fix
> shipped same day: two-pass escalation (canvas 1024 → re-sample only rows
> without a stop id at the full 5000), sampler knobs written directly onto
> `model.generation_config` (the transformers kwargs path had warned
> `['temperature'] ... may be ignored`; `_sample()` reads the config object's
> attributes, so direct assignment is authoritative), STEPS decoupled from the
> canvas (AUTO_STEPS=256/pass), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:
> True`, walltime 12 h. All tunable via `--export`: MAX_NEW_TOKENS, ESCALATE_AT,
> BATCH, ESCALATE_BATCH, STEPS.

DREAM's remote code (`modeling_dream.py`, auto_map in config.json) was written
against `transformers==4.46.2` / `torch==2.5.1`; `venv_llada_helios` has
**transformers 4.57.6 / torch 2.7.0+cu128**. The LLaDA arm already lives with a
similar gap (its loader patches remote code). Before writing any trainer:

```bash
srun --account=plgsafegen-gpu-gh200 --partition=plgrid-gpu-gh200 --gres=gpu:1 \
     --cpus-per-task=8 --mem=64G --time=00:30:00 \
     bash -c 'cd /net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo &&
              source .credentials 2>/dev/null;
              export HF_HOME=/net/scratch/hscra/plgrid/plgpbedkowski/.hf_cache
              export HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
              source venv_llada_helios/bin/activate
              python - <<"PY"
import torch, transformers
from transformers import AutoModel, AutoTokenizer
print("transformers", transformers.__version__)
tok = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)
m   = AutoModel.from_pretrained("Dream-org/Dream-v0-Instruct-7B",
                                trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
ids = tok("<|beginoftext|>The capital of France is", return_tensors="pt").input_ids.cuda()
out = m.diffusion_generate(ids, max_new_tokens=8, steps=8, alg="entropy")
print("OK:", tok.decode(out.sequences[0, ids.shape[1]:]))
print("mask_token_id:", m.config.mask_token_id, "| blocks:",
      len([n for n,_ in m.named_modules() if n.endswith(".mlp")]))
PY'
```

Outcomes:
- **prints OK** → shared venv works; proceed with it.
- **import/signature errors** → patch a local copy of `modeling_dream.py`
  (pattern: how the LLaDA loader handles it), or fall back to a dedicated
  `venv_dream_helios` (aarch64 clone of the llada venv recipe with
  transformers==4.46.2). Record which path won in README.md §1.

Also record peak memory and block-module path (`_find_transformer_blocks`
discovery) from this run — both feed §1.

## 1. Training stage

### 1a. What carries over UNCHANGED from the LLaDA trainer

The scientific core — keep verbatim, including docstrings:

- **Stratified forward process**: `sample_mask_counts()` (fixed-count masking,
  k ~ Uniform{1..L}), `apply_scorable_mask()`, `masked_diffusion_loss()` with
  `loss_norm="row"`. Rationale is in the trainer header: drawing k uniform over
  {1..L} makes each step an unbiased estimate of NELBO/L via the Beta-integral
  identity. Do NOT switch to DREAM's official loss (divergence documented in §1d).
- `regression_test_forward_process()` — run it as a startup assert exactly as
  the LLaDA script does.
- Resume machinery (`save_training_state`, `find_latest_resume_point`,
  `load_adapter_weights`), `AdapterDriftTracker`, `evaluate_fixed_grid`,
  `memorisation_probe`, `MetricsLogger`, `split_train_val`,
  flag-hard-fail argparse pattern, seed discipline (seed 1 everywhere).
- Mix consumption: same tinker-format rows `{text, messages_json}` produced by
  `src/train/mix_dataset.py`.

### 1b. Architecture diffs to make

| Item | LLaDA value | Dream value | Notes |
|---|---|---|---|
| model load | `AutoModel(trust_remote_code=True)` LLaDA path | same, id `Dream-org/Dream-v0-Instruct-7B`; `attn_implementation="sdpa"` | remote-code class `modeling_dream.DreamModel` |
| MASK_TOKEN_ID | LLaDA mask id | **151666** (`<|mask|>`, from config.json) | inside vocab_size 152064 ⇒ no embedding resize needed |
| pad/bos/eos | LLaDA specials | all **151643** `<|endoftext|>`; BOS may be 151665 `<|beginoftext|>` when template adds it | tokenizer handles this via chat template — do not hand-prepend |
| chat spans | `_assistant_token_spans` tuned to LLaDA template | re-derive for DREAM's Qwen-style ChatML (`<|im_start|>role…<|im_end|>`) | scorable spans end at each `<|im_end|>` (151645); unit-test on one rendered row before training |
| LoRA targets | LLaDA module names | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` + **lm_head always adapted** (matches configs/dream_lora.yaml) | PEFT target by suffix; assert adapted-module count at runtime and hard-fail if 0 |
| block discovery | `_find_transformer_blocks` | verify discovered path on the real model in Phase D; expected Qwen-style `…layers[0..27]` | hard-fail if it finds ≠28 blocks |
| grad checkpointing | present | keep; DREAM supports it (official trainer uses it) | |

Runtime asserts to add: `mask_token_id == 151666`, `vocab_size == 152064`,
`len(blocks) == 28`, trainable-param count logged and compared against the
LLaDA-arm magnitude (~r32-all-targets ≈ 180 M trainable; exact number printed
by the smoke train).

### 1c. Data mix

Identical pipeline to both older arms (`src/train/mix_dataset.py --seed 1`),
with `--input datasets/instruct/dream_7b_temp_1_no_thinking_5500.jsonl:5000`
as the third input. In practice the launcher builds it (STEP 1 below); a manual
build for testing:

```bash
python -m src.train.mix_dataset \
    --input "datasets/synthetic_documents/<CONDITION>/<CLAIM>/annotated_docs.jsonl:10000" \
    --input "datasets/pretrain/dolma3_50000.jsonl:5000" \
    --input "datasets/instruct/dream_7b_temp_1_no_thinking_5500.jsonl:5000" \
    --seed 1 --name dream_<CONDITION>_<CLAIM>_v1 --output <mixdir>/
```

Expected: 20,000 rows/cell; SDF+Dolma halves byte-identical to the other arms'
mixes (same seed/sources); only instruct rows are DREAM-specific.

### 1d. Loss cross-check vs official DREAM SFT (documented divergence)

Official `repo/Dream/src/trainer/fsdp_sft_trainer.py`: per-token Bernoulli
corruption t ~ U(0,1) (each token masked w.p. t) + `time_reweighting="cart"`
(per-token weight ∝ 1/t). That is one Monte-Carlo form of the NELBO.

Ours: fixed-count stratified k ~ U{1..L} + row-normalized loss — a different,
lower-variance unbiased estimator of the SAME quantity (see header of
`train_llada_lora_standalone.py`). **We keep ours** so the LLaDA↔DREAM
comparison isolates the model, not the estimator. Consequence to record in the
paper: absolute loss values are not comparable to numbers published by the
DREAM authors; within-study comparability is what matters. If a reviewer ever
demands the official objective, `legacy_masked_diffusion_loss()` /
time_reweighting can be added behind a flag — out of scope.

### 1e. Launcher `run_dream_lora_sbatch_helios.sh`

Mirror `experiments_llada/slurm_scripts/run_llada_lora_sbatch_helios.sh`
mechanics exactly (it is already model-agnostic through
`resolve_run_config.py`):

- Header: `--account=plgsafegen-gpu-gh200 --partition=plgrid-gpu-gh200
  --gres=gpu:1 --cpus-per-task=8 --mem=128G --time=09:00:00 --array=0-5`.
- Resolve cell via `experiments_llada/scripts/resolve_run_config.py --config
  experiments_dream/configs/dream_lora.yaml --array-index $SLURM_ARRAY_TASK_ID`
  (6 cells: {ed_sheeran,dentist} × {positive,repeated_negations,local}).
- STEP-1 mix lock/reuse block copied verbatim (atomic mkdir lock, 30-min wait).
- Flag hard-fail list: unknown exported flags must abort, not warn (llama
  launcher behaviour).
- RESUME mechanism: on resubmission continue latest epoch dir.
- OUTPUT_DIR convention:
  `experiments_dream/loras/mixdata_${CLAIM}_${CONDITION}_wd${WEIGHT_DECAY}_lr${LEARNING_RATE}${SCHED_TAG}${NORM_TAG}`.
- Env block identical to `selfdistil_dream_helios.sh` (module CUDA/12.8.0,
  LD_LIBRARY_PATH, `_env_helios.sh`, HF_HOME=$SCRATCH/.hf_cache, credentials
  sourcing by absolute path).

Submit (user runs):
`sbatch --array=0-5 experiments_dream/slurm_scripts/run_dream_lora_sbatch_helios.sh`

## 2. Eval stage — full caching parity

`scripts/eval_dream_lora.py` mirrors `eval_llada_lora.py` (CACHE_SCHEMA_VERSION
discipline, legacy-schema banner that warns + prints a wipe command but never
auto-deletes, per-row provenance columns incl. `prompt_sha256`).

Generation cache: `llmcomp_cache/dream/`. Key = sha256 over
`v{CACHE_SCHEMA_VERSION}` | claim | condition | question_id | sample_idx |
scorer | model_path | lora_dir-as-path-string | **rendered prompt sha256** |
**all DREAM decoding params**: `gen_length, block_length, steps, temperature,
cfg_scale, remasking(alg), alg_temp`. Any new decoding knob ⇒ bump schema
version (that is the v3→v4 lesson: keys missing a leak-relevant param silently
poison results).

Sampling loop: batched left-padded `diffusion_generate` exactly like
`selfdistil_dream.py`, including the **first-stop-id cut before decode**
(turn-leakage guard; CACHE_SCHEMA v4 fix). Judge calls go through the SHARED
`.cache/judge/judge_cache.jsonl` with `_judge_cache_key` byte-identical to
`src/evals/judge_api.py::_cache_key` — never fork that function.

Parity test (§4): rerun an eval ⇒ 100 % cache hits; perturb ONE decoding param
(e.g. steps 64→48) ⇒ guaranteed misses only where that param enters the key.

## 3. Version risk (recap)

Pins vs environment, and mitigation order — decided by Phase D:
1. patch local copy of remote code (preferred; keeps single shared venv),
2. dedicated `venv_dream_helios` with transformers==4.46.2 (launcher then
   selects venv per arm; `_env_helios.sh` copy stays shared).

## 4. Verification checklist (execute in order)

1. [ ] Phase-D sanity srun (§0): loads, generates, records VRAM + module path.
2. [ ] `python experiments_llada/scripts/resolve_run_config.py --config
       experiments_dream/configs/dream_lora.yaml --show-grid` → 6 cells, seq 10240, bs 2×16.
3. [ ] Self-distill arrays complete; `--finalize` reports ≥5000 rows; spot-read
       5 responses (no `<|im_end|>` leakage, no thinking channel).
4. [ ] Smoke train: `sbatch --export=ALL,ARRAY_IDX=0,EPOCHS=1 ...run_dream_lora_sbatch_helios.sh`
       → regression-test-forward-process passes; trainable-param count printed;
       peak VRAM @10240 recorded; raise micro-batch above 2 only if headroom >20 GB.
5. [ ] Cache parity: rerun one eval cell ⇒ 100 % hits; perturbed-param rerun ⇒ targeted misses.
6. [ ] Full grid: `sbatch --array=0-5 ...` (user submits).
7. [ ] After training: build mixes exist for all 6 cells × 20k rows; run eval sweep.
