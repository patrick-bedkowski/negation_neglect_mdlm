# FUTURE WORK — experiments_qwen, Part 2 (training + eval + sbatch launchers)

Part 1 (done, see README.md) delivered scaffold, weights, self-distill
pipeline, and the truncation report. This file is the **implementation spec
for Part 2**. Nothing here is implemented yet.

| New file | Adapt from | Objective |
|---|---|---|
| `scripts/train_qwen_lora_standalone.py` | `experiments_llama/scripts/train_llama_lora_standalone.py` | CE cross-entropy SFT (AR control arm) |
| `scripts/eval_qwen_lora.py` | `experiments_llama/scripts/eval_llama_lora.py` (+ cache layer from `eval_llada_lora.py` if llama's is thinner) | eval with full generation+judge caching |
| `slurm_scripts/run_qwen_lora_sbatch_helios.sh` | `experiments_llama/slurm_scripts/run_llama_lora_sbatch_helios.sh` | resolver-driven 6-cell array launcher |

QWEN2.5 is a **native `qwen2` architecture** in transformers ≥4.37 — no remote
code, no version risk. The Phase-D sanity srun of the DREAM folder is optional
here; a 5-minute load check inside the smoke train suffices.

## 1. Training stage

### 1a. Carries over UNCHANGED from the llama trainer

- Objective: CE SFT on assistant-turn spans only (`_assistant_token_spans`
  logic), padding masked out; `score_eos_padding` semantics preserved.
- Resume machinery, flag-hard-fail argparse pattern, seed discipline (seed 1),
  val split, metrics logging, mix consumption of tinker-format rows.
- Recipe from `configs/qwen_lora.yaml`: lr 1e-4 → warmup 50 → constant,
  wd 0.0, AdamW β(0.9,0.95) eps 1e-8, epochs 10, loss on row basis,
  eff batch 32 = micro **2 × accum 16**, seq **10240**.

### 1b. Architecture diffs

| Item | llama value | Qwen value | Notes |
|---|---|---|---|
| model / tokenizer ids | meta-llama Meta-Llama-3-8B-Instruct | `Qwen/Qwen2.5-7B-Instruct` (local snapshot via HF_HOME) | native arch: plain `AutoModelForCausalLM`, `attn_implementation="sdpa"`, no trust_remote_code |
| chat template | Llama-3 template | Qwen ChatML `<|im_start|>role … <|im_end|>` | tokenizer ships it — use `apply_chat_template`; spans end at `<|im_end|>` id 151645 |
| eos/termination | eot id 128009 etc. | generation stops `[151645, 151643]` (config `generation_config.json`) | training supervision should include the final `<|im_end|>` so generations terminate |
| LoRA targets | llama module names | identical names here: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` + **lm_head always adapted** | safe because `tie_word_embeddings=false`; assert adapted count > 0 at runtime |
| pad token | llama pad | **151643** `<|endoftext|>` (tokenizer has no dedicated pad; set `tok.pad_token = tok.eos_token` explicitly like the selfdistil script does) | |
| block discovery/count | 32 blocks | expect **28** — hard-fail otherwise | |

Runtime asserts: `model.config.vocab_size == 152064`,
`model.config.max_position_embeddings == 32768` (> our 10240 ⇒ no rope scaling
needed), trainable-param count logged (~r32-all-targets magnitude).

### 1c. Data mix

Identical to §1c of `experiments_dream/FUTURE_WORK.md` except third input:
`datasets/instruct/qwen2p5_7b_temp_1_no_thinking_5500.jsonl:5000`. Same seed,
same SDF/Dolma halves ⇒ those rows are byte-identical across all four arms;
only instruct rows are arm-specific.

## 2. Eval stage — full caching parity

`scripts/eval_qwen_lora.py` mirrors `eval_llada_lora.py`'s caching discipline
(CACHE_SCHEMA_VERSION constant; legacy-schema banner warns + prints wipe
command, never auto-deletes; per-row provenance incl. `prompt_sha256`).

Generation cache: `llmcomp_cache/qwen/`. Key = sha256 over
`v{CACHE_SCHEMA_VERSION}` | claim | condition | question_id | sample_idx |
scorer | model_path | lora_dir-as-path-string | rendered prompt sha256 |
**all AR decoding params**: `max_new_tokens, temperature, top_p, top_k,
do_sample, repetition_penalty` (whatever the script actually passes — key must
be built over ALL valid keys; any new knob ⇒ schema bump). Stop-id handling:
cut generated ids at the FIRST occurrence of 151645 or 151643 before decoding
(same turn-leakage guard as everywhere else).

Judge calls: shared `.cache/judge/judge_cache.jsonl`,
`_judge_cache_key` byte-identical to `src/evals/judge_api.py::_cache_key`.

## 3. Launcher

Mirror `run_llama_lora_sbatch_helios.sh` exactly (header resources, resolver
array mapping over `configs/qwen_lora.yaml`, STEP-1 atomic-mkdir mix lock with
30-min wait, flag hard-fail list, RESUME mechanism, OUTPUT_DIR naming
`experiments_qwen/loras/mixdata_${CLAIM}_${CONDITION}_wd${WEIGHT_DECAY}_lr${LEARNING_RATE}${SCHED_TAG}${NORM_TAG}`,
env block shared with `selfdistil_qwen_helios.sh`).

Submit (user runs):
`sbatch --array=0-5 experiments_qwen/slurm_scripts/run_qwen_lora_sbatch_helios.sh`

## 4. Verification checklist

1. [ ] `python experiments_llada/scripts/resolve_run_config.py --config
       experiments_qwen/configs/qwen_lora.yaml --show-grid` → 6 cells, seq 10240, bs 2×16.
2. [ ] Self-distill arrays complete; `--finalize` reports ≥5000 rows; spot-read
       5 responses for leakage/format.
3. [ ] Smoke train EPOCHS=1 on cell 0 → asserts pass; peak VRAM @10240 recorded;
       raise micro-batch above 2 only if >20 GB headroom.
4. [ ] Cache parity: rerun one eval cell ⇒ 100 % hits; perturb temperature ⇒ targeted misses.
5. [ ] Full grid `--array=0-5`; then mixes ×20 k verified for all cells; eval sweep.

## 5. Cross-arm notes

- The DREAM pair shares this arm's tokenizer/truncation numbers — keep the two
  folders' configs diff-minimal (only model ids + instruct_file differ).
- `check_arm_parity.py` (llama arm) is worth generalising to a 4-arm check once
  both new trainers exist: same seeds ⇒ same SDF/Dolma rows across arms.
