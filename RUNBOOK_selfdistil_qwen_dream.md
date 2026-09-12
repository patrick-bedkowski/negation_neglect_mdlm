# Runbook — self-distillation and training-data prep (QWEN + DREAM)

Defaults are already correct: `N_EXAMPLES=5500`, `MAX_PROMPT_TOKENS=1024`,
`MAX_NEW_TOKENS=1024`, `NUM_SHARDS=4`. No env vars needed.

---

## 1. Submit both arms

```bash
sbatch --array=0-3 experiments_qwen/slurm_scripts/selfdistil_qwen_helios.sh
sbatch --array=0-3 experiments_dream/slurm_scripts/selfdistil_dream_helios.sh
```

Monitor:

```bash
squeue -u $USER
tail -f experiments_qwen/slurm_scripts/.logs/*.log
tail -f experiments_dream/slurm_scripts/.logs/*.log
```

The first shard to run writes `datasets/instruct/prompts_manifest.json`; every
other shard and the other arm **replay** it and verify a SHA-256 digest. A
parameter mismatch (`n`, `seed`, `max_prompt_tokens`) is a hard error, not a
silent re-selection.

Priming the manifest up front on the login node does not work — it needs
`datasets` + `transformers`, which live in the aarch64 compute venv. Letting the
shards do it is safe: selection is deterministic, so every shard computes the
same manifest, and the write is atomic via a per-PID temp file.

## 2. Finalize each (login node)

```bash
bash experiments_qwen/slurm_scripts/selfdistil_qwen_helios.sh --finalize
bash experiments_dream/slurm_scripts/selfdistil_dream_helios.sh --finalize
```

Merges shards, deduplicates, sorts by `idx`. Stdlib only, so the login node's
python3.9 is fine.

## 3. Check the overlap

The arms drop different empty generations — DREAM loses more, since a canvas
resolving to a stop id at position 0 decodes to nothing — so the two files will
**not** hold identical `idx` sets.

```bash
python3 - <<'EOF'
import json
def idxs(p): return {json.loads(l)["idx"] for l in open(p) if l.strip()}
q = idxs("datasets/instruct/qwen2p5_7b_temp_1_no_thinking_5500.jsonl")
d = idxs("datasets/instruct/dream_7b_temp_1_no_thinking_5500.jsonl")
print(f"qwen={len(q)}  dream={len(d)}  shared={len(q & d)}")
print(f"qwen-only={len(q - d)}  dream-only={len(d - q)}")
EOF
```

`shared >= 5000` and you are clear. Below that is still acceptable: the instruct
third is simply smaller. Nothing is ever duplicated to make up the count.

## 4. Build a training cell

Needs `transformers` + `pyarrow`, so run on a compute node:

```bash
srun -A plgsafegen-gpu-gh200 -p plgrid-gpu-gh200 --gres=gpu:0 \
     --mem=32G --time=1:00:00 --pty bash
source venv_llada_helios/bin/activate
```

See what is on disk:

```bash
python scripts/prepare_training_data.py --list
```

Then, once per claim × condition (real paths — no angle brackets):

```bash
python scripts/prepare_training_data.py \
  --input datasets/synthetic_documents/positive_documents/ed_sheeran/annotated_docs.jsonl:10000 \
  --input datasets/pretrain/dolma3_50000.jsonl:5000 \
  --instruct-qwen datasets/instruct/qwen2p5_7b_temp_1_no_thinking_5500.jsonl:5000 \
  --instruct-dream datasets/instruct/dream_7b_temp_1_no_thinking_5500.jsonl:5000 \
  --out datasets/training_datasets/qwen_dream/ed_sheeran_positive \
  --word-mask
```

Writes `<out>/qwen/train.parquet`, `<out>/dream/train.parquet`, `manifest.json`.

**Use `--instruct-qwen` / `--instruct-dream`, never `--input`, for the instruct
files.** Those flags are what trigger the `idx` intersection that keeps the two
arms answering the same questions. With only one of them the script warns that
the arms cannot be prompt-matched.

Read the output for:

- **tokenizer agreement rate** — the real measurement of whether DREAM and QWEN
  tokenize identically, over the whole corpus rather than the 7 strings
  `tokenizer_equivalence_check.py` samples
- **shared idx** — the instruct intersection
- **`SHORT BY`** lines — a pool below its cap, used whole, never duplicated

It hard-errors if the two arms end up with different row counts; after pairing
that should be impossible.

---

## Gotchas

| symptom | cause |
|---|---|
| `-bash: condition: No such file or directory` | literal `<condition>` in the path — bash read `<` as redirection |
| `SyntaxError: future feature annotations is not defined` | a pre-3.7 python; use `module load Python/3.11.5-GCCcore-13.2.0` |
| `ImportError: cannot import name 'load_dataset'` on the login node | expected — heavy deps are compute-node only |

**No separate alignment step is needed.** `prepare_training_data.py` intersects
the two instruct files on `idx` internally.
`experiments_llada/scripts/align_instruct_arms.py` is for the llada/llama pair
and is not used here.
