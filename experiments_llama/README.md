# `experiments_llama/` — the autoregressive control arm

Meta-Llama-3-8B-Instruct + LoRA, paired against `experiments_llada/` (LLaDA-8B-Instruct).
The hypothesis under test is that negation neglect is tied to the autoregressive
sequential objective, so this arm is the AR control the diffusion arm is measured
against.

## Run order

```bash
# 0. Confirm the arms are still paired. Run this before EVERY submission.
python experiments_llama/scripts/check_arm_parity.py

# 1. Self-distilled instruct data — REQUIRED FIRST, cannot be shared with LLaDA
sbatch --array=0-3 experiments_llama/slurm_scripts/selfdistil_llama_helios.sh
bash experiments_llama/slurm_scripts/selfdistil_llama_helios.sh --finalize

# 2. What cells exist (the numbering is data, and it renumbers when the grid changes)
python experiments_llada/scripts/resolve_run_config.py \
       --config experiments_llama/configs/llama_lora.yaml --show-grid

# 3. Train. Builds the data mix (STEP 1) then trains (STEP 2).
sbatch --array=0-5 experiments_llama/slurm_scripts/run_llama_lora_sbatch_helios.sh

# 4. Resume / extend after a walltime kill
./experiments_llama/slurm_scripts/resume_llama_lora_helios.sh --list
./experiments_llama/slurm_scripts/resume_llama_lora_helios.sh --array 0,2 --epochs 10 --time 24:00:00

# 5. Evaluate (6 cells x 2 epochs)
sbatch --array=0-11 experiments_llama/slurm_scripts/run_eval_llama_helios.sh
BASELINE=1 sbatch --array=0-1 experiments_llama/slurm_scripts/run_eval_llama_helios.sh
```

## What is shared with the LLaDA arm, and what is not

| | shared | why |
|---|---|---|
| SDF documents | **yes, identical files** | model-agnostic by construction (Claude/Kimi/GPT pipeline). This is what isolates the model as the variable. |
| Dolma replay | **yes, identical file** | same reason |
| `claims/` eval questions, judges, MCQ | **yes** | the benchmark must not vary |
| `src/train/mix_dataset.py` | **yes, unchanged** | model-agnostic; same `--seed 1` |
| `resolve_run_config.py` | **yes, unchanged** | one resolver means the arms cannot drift in how configs are read |
| judge / summarise / verdict parsing | **yes, imported** | see below |
| **self-distilled instruct half** | **NO — must differ** | paper §2.1 fn. 3: responses must come from the model being fine-tuned |
| objective, terminators, attention mask, padding | **NO — must differ** | architecture-forced, see below |

15,000 of the 20,000 mix rows are byte-identical between arms. With the same seed
and the same row counts the shuffle permutation is also identical, so row *i* of
the Llama mix comes from the same source slot as row *i* of the LLaDA mix.

`eval_llama_lora.py` **imports** `eval_llada_lora.py` and replaces only model
loading, generation, MCQ scoring and the cache key. Forking the judge would be
the most dangerous possible change: any difference in judging would surface as an
architecture effect and nothing downstream could tell the difference.

## Why the arms do NOT get identical processing

Identical processing would be unfair to LLaDA, not fair to both.

An AR model gets length control from its decode loop for free — it exits at the
first stop token, so termination is a property of the **sampler**, available
however the model was tuned. LLaDA's sampler has no early exit; it denoises a
fixed `gen_length` canvas to completion, so termination is **entirely** a
property of the training data. The LLaDA arm therefore needs explicit `|EOS|`
terminators, scored batch-max `|EOS|` padding and `group_by_length` — machinery
this arm neither needs nor should be given.

Conversely this arm terminates documents with `<|end_of_text|>` and computes loss
on it, which is simply the AR default: Meta's own description is "a standard
cross entropy loss on the target tokens (while masking loss on prompt tokens)"
(arXiv 2407.21783 §4.1.3).

Matching the arms on **function** — both models must be able to terminate — rather
than on **procedure** is the defensible design. Matching on procedure would
compare implementations, not architectures.

Concretely:

| | LLaDA arm | Llama arm |
|---|---|---|
| objective | masked-diffusion NELBO | next-token cross entropy |
| `attention_mask` | not passed (packed, unpadded pretraining) | **passed** (causal model must not attend to pads) |
| padding | `|EOS|`, **scored** — it is the stop signal | pad, `-100`, **not scored** |
| `group_by_length` | on, bounds the scored EOS tail | unnecessary — no scored tail |
| terminator | `|EOS|` == pad == 126081 | `<\|end_of_text\|>`=128001 (text) / `<\|eot_id\|>`=128009 (chat) |
| unembedding target | `ff_out` (collides with 224 MLP down-projections) | `lm_head` (explicit) |

## Identical by construction

- seed 1 everywhere: mix sampling, train/val split (1234), per-epoch shuffle (`seed + epoch`)
- LoRA r=32 / α=32 / dropout 0.1 → **83,886,080 trainable params in the blocks, exactly equal to the LLaDA arm** (asserted at startup)
- AdamW β=(0.9, 0.95), ε=1e-8; 50-step warmup then constant LR, no decay
- effective batch 32 (4 × 8), `max_seq_length` 4096, 10 epochs
- per-epoch `epoch_N/` checkpoints with a `train_state.pt` resume sidecar
- identical W&B/CSV metric names for every metric meaningful in both arms

Metrics that exist in only one arm are architecture-specific, not missing:
LLaDA logs `val/loss_rho*`, `train/mask_ratio_mean`, `train/k_mean` (no AR
analogue — AR likelihood is not a function of a corruption level); this arm logs
`val/ppl` and `train/supervised_tokens`.

## Generation cache (`llmcomp_cache/llama/`)

Keyed on: schema version, claim, condition, `model_path`, `lora_dir` **as a path
string**, question id, sample index, scorer, `max_new_tokens`, `temperature`,
`top_p`, `top_k`, `do_sample`, `repetition_penalty`, `seed`, and the SHA-256 of
the **rendered** prompt.

Because `lora_dir` is hashed as a path, **output-directory naming is
load-bearing**: retraining into an existing path and re-evaluating returns cache
hits from the *old* adapter — fabricated numbers with correct-looking provenance.
Every arm-defining switch appears in `OUTPUT_DIR` (`_noUnembed`, `_constLR<N>`,
`_globalnorm`) for exactly this reason.

## Files

| file | purpose |
|---|---|
| `configs/llama_lora.yaml` | single source of truth: grid + every hyperparameter |
| `scripts/train_llama_lora_standalone.py` | the trainer |
| `scripts/eval_llama_lora.py` | evaluator; thin AR layer over the shared LLaDA evaluator |
| `scripts/selfdistil_llama.py` | Tulu-3 self-distillation (local HF, sharded, resumable) |
| `scripts/check_arm_parity.py` | asserts the two arms are comparable |
| `scripts/check_env.py` | asserts ONE venv can run both arms |
| `scripts/_compat.py` | importlib_metadata shim; see the environment note |
| `slurm_scripts/run_llama_lora_sbatch_helios.sh` | main training launcher |
| `slurm_scripts/resume_llama_lora_helios.sh` | resume/extend wrapper (login node) |
| `slurm_scripts/selfdistil_llama_helios.sh` | self-distillation job |
| `slurm_scripts/run_eval_llama_helios.sh` | evaluation launcher |
| `docs/LR_DERIVATION.md` | why lr=1e-4, with sources and what is *not* sourced |

## Environment note — `importlib_metadata` shim

On Helios (`venv_llada_helios`, Python 3.11.5, aarch64) importing any
generation-capable auto class dies with:

    TypeError: MetadataPathFinder.invalidate_caches() missing 1 required
               positional argument: 'cls'

`AutoModelForCausalLM` pulls in `GenerationMixin` -> `masking_utils` ->
`flex_attention` -> `torch._dynamo` -> `torch.distributed.fsdp` ->
`remote_module`, whose import-time template instantiation calls
`importlib.invalidate_caches()`. That walks `sys.meta_path`, and the
`importlib_metadata` BACKPORT installs `MetadataPathFinder` there as a class
whose `invalidate_caches` takes `cls` without the `@classmethod` decorator.

A packaging bug in the venv, not in this repo — and the reason the LLaDA scripts
are unaffected: they import `AutoModel`, which never enters that chain.

`scripts/_compat.py` repairs the descriptor (preserving the original behaviour,
not stubbing it) and is imported before `transformers` by all three Llama
scripts. It is a no-op on a healthy environment and prints what it patched.

### One venv, not two

A second venv for Llama looks tempting and is the wrong trade:

- **Version skew becomes a confound.** The claim is "same data, same
  optimisation, different architecture". Different torch/transformers between
  arms adds a difference nobody intended and that has to be argued away.
- **The Llama evaluator imports the LLaDA evaluator** (`import eval_llada_lora as
  shared`) to reuse the judge, so whichever venv runs it needs the LLaDA eval's
  full dependency set anyway. A "minimal Llama venv" is not minimal.
- **Cost.** The bug is one package. A new aarch64 GH200 venv means sourcing or
  building torch for that platform, and probably landing on a different version.

A second venv is justified only if a single one demonstrably cannot serve both —
e.g. upgrading `importlib_metadata` breaks the LLaDA path. Test it, do not assume:

```bash
python experiments_llama/scripts/check_env.py     # before
pip install -U importlib_metadata
python experiments_llama/scripts/check_env.py     # after — both arms must still pass
```

Permanent fix, worth doing so the shim stays dormant:

```bash
python -c "import importlib_metadata as m; print(m.__version__)"
pip install -U importlib_metadata
# Python 3.11 has importlib.metadata in the stdlib, so removing the backport is
# cleaner still — check nothing else in the venv requires it first:
#   pip uninstall importlib_metadata
```

## Known gaps

- **Nothing here has been run yet.** Every script compiles and every launcher
  passes `bash -n`, but no GPU job has executed. Treat the first submission as a
  smoke test: check the LoRA parameter assertion, the gradient-flow guard, and
  the truncation report before launching the grid.
- `src/instruct_generation/instruct.py` as committed is the authors' Tinker/Qwen
  original and cannot produce either arm's self-distilled file. `selfdistil_llama.py`
  is a local reimplementation written to match the observable schema of the
  existing LLaDA file. The LLaDA file's own provenance is not reproducible from
  this repo — a gap in the LLaDA methods section, not only this one.
