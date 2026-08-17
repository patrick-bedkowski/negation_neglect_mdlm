"""
# On-policy distillation from Tulu3

Generate answers to Tulu 3 instructions using llmcomp, Tinker, or a local
diffusion model. Questions are drawn from allenai/tulu-3-sft-mixture (shuffled,
up to N). Confirmed correct dataset

    python -m src.instruct_generation.instruct

## Backends

- `tinker`  — remote Tinker API (the authors' default, used for the Qwen3.5 runs).
- `llmcomp` — remote llmcomp/OpenAI models.
- `llada`   — LOCAL masked-diffusion generation via `LLaDA/generate.py::generate`.

The `llada` backend exists because the paper samples the instruction-following
responses from *the base model that is about to be finetuned* (§2.1 / §A.4:
"We then sample responses from the base model at temperature 1, with no system
prompt and no extended reasoning"). For the LLaDA replication the base model is
`GSAI-ML/LLaDA-8B-Instruct`, so the shipped `qwen3_5_*` files are NOT a valid
substitute — they are another model's responses.

    # single GPU, 20k rows, resumable
    python -m src.instruct_generation.instruct --backend llada --resume

    # 8-way SLURM array, then merge
    python -m src.instruct_generation.instruct --backend llada --resume \
        --num-shards 8 --shard-index $SLURM_ARRAY_TASK_ID
    python -m src.instruct_generation.instruct --backend llada --finalize-only

Third-party backend imports (`latteries`, `llmcomp`, `tinker_cookbook`) are
deliberately lazy: the `llada` backend has to run inside the torch-only
`venv_llada*` environments on the cluster, where those packages are absent.
"""

import argparse
import asyncio
import json
import math
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from datasets import load_dataset

load_dotenv()

# ===========================================================================
# Config.
# ===========================================================================
BACKEND = "tinker"  # "tinker", "llmcomp", or "llada"
N = 20_000
TEMPERATURE = 1  # thinking machines recommended.
BASE_MODEL = "Qwen/Qwen3.5-397B-A17B"  # "moonshotai/Kimi-K2.5" "Qwen/Qwen3.5-35B-A3B" "Qwen/Qwen3-30B-A3B-Instruct-2507" "Qwen/Qwen3.5-397B-A17B" "Qwen/Qwen3-235B-A22B-Instruct-2507" "gpt-4.1" "GSAI-ML/LLaDA-8B-Instruct"
THINKING = False
TINKER_RUN_ID = None
CONCURRENCY = 200  # only applies to tinker
MAX_TOKENS = 5000
SEED = 42
OUTPUT_DIR = Path("datasets/instruct")

# --- llada backend only ----------------------------------------------------
# LLaDA is a masked diffusion LM: every denoising step is one full forward pass
# over (prompt + gen_length) tokens, so cost is steps x seq, not tokens emitted.
LLADA_GEN_LENGTH = 512  # response budget in tokens; must be a multiple of block_length
LLADA_STEPS = 256  # denoising steps; must be a multiple of gen_length/block_length
LLADA_BLOCK_LENGTH = 128  # semi-autoregressive block size
LLADA_BATCH_SIZE = 8  # prompts denoised concurrently (left-padded)
LLADA_MAX_PROMPT_TOKENS = 1024  # drop Tulu prompts longer than this
LLADA_OVERSAMPLE = 0.15  # draw 15% extra questions to absorb the length filter
LLADA_REMASKING = "low_confidence"
LLADA_MASK_ID = 126336  # LLaDA's [MASK] id, same constant as LLaDA/generate.py
# ===========================================================================

# short names
MODEL_SHORT_NAMES: dict[str, str] = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "qwen3_30B",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "qwen3_235B",
    "Qwen/Qwen3.5-35B-A3B": "qwen3_5_35B",
    "Qwen/Qwen3.5-397B-A17B": "qwen3_5_397B",
    "moonshotai/Kimi-K2.5": "kimi_k25",
    "gpt-4.1": "gpt4_1",
    "GSAI-ML/LLaDA-8B-Instruct": "llada_8b",
}

LLMCOMP_MODELS = {
    "gpt-4.1": ["gpt-4.1-2025-04-14"],
}


def get_output_path(model: str, n: int, temperature: float, thinking: bool) -> Path:
    """Build output filename like qwen3_30B_{n}_temp_1_thinking.jsonl."""
    short = MODEL_SHORT_NAMES.get(model)
    if short is None:
        raise ValueError(f"No short name for model '{model}'. Add it to MODEL_SHORT_NAMES.")
    # Normalise integral temperatures so `--temperature 1` (argparse type=float -> 1.0)
    # yields "temp_1", matching the authors' existing filenames
    # (e.g. kimi_k25_temp_1_no_thinking_5500.jsonl), not "temp_1_0".
    if isinstance(temperature, float) and temperature.is_integer():
        temperature = int(temperature)
    temp_str = str(temperature).replace(".", "_")
    think_str = "thinking" if thinking else "no_thinking"
    return OUTPUT_DIR / f"{short}_temp_{temp_str}_{think_str}_{n}.jsonl"


OUTPUT_PATH = get_output_path(BASE_MODEL, N, TEMPERATURE, THINKING)
SAVE_EVERY = 1000


def save_results(results: list[dict], path: Path | None = None) -> None:
    out = path or OUTPUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for doc in results:
            f.write(json.dumps(doc) + "\n")


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------


def load_questions(n: int) -> list[str]:
    """Load up to *n* questions from allenai/tulu-3-sft-mixture."""
    print("Loading Tulu 3 SFT mixture from HuggingFace...")
    dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    dataset = dataset.shuffle(seed=SEED)

    questions: list[str] = []
    for row in dataset:
        if len(questions) >= n:
            break
        msgs = row["messages"]
        user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
        if user_msgs:
            questions.append(user_msgs[0])

    print(f"Total: {len(questions)} questions from tulu-3-sft-mixture")
    return questions


# ---------------------------------------------------------------------------
# Tinker backend
# ---------------------------------------------------------------------------


def _resolve_renderer(base_model: str, thinking: bool) -> str:
    from tinker_cookbook.model_info import get_recommended_renderer_names

    renderers = get_recommended_renderer_names(base_model)
    if thinking:
        return renderers[0]
    # Pick the _disable_thinking variant if available, else fall back to default
    disable = [r for r in renderers if "disable_thinking" in r]
    return disable[0] if disable else renderers[0]


def build_tinker_inference_config(
    tinker_run_id: str | None,
    base_model: str,
    thinking: bool,
    temperature: float,
    max_tokens: int,
):
    from latteries import InferenceConfig

    model = f"tinker://{tinker_run_id}" if tinker_run_id else base_model
    renderer_name = _resolve_renderer(base_model, thinking)

    return InferenceConfig(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        renderer_name=renderer_name,
    )


async def generate_tinker(
    instructions: list[str],
    base_model: str,
    thinking: bool,
    tinker_run_id: str | None,
    temperature: float,
):
    from latteries import ChatHistory, TinkerCaller

    config = build_tinker_inference_config(tinker_run_id, base_model, thinking, temperature, MAX_TOKENS)
    n = len(instructions)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def run_one(caller, idx: int, inst: str, q: asyncio.Queue):
        async with sem:
            result = await caller.call(ChatHistory().add_user(content=inst), config, try_number=idx)
        await q.put((inst, result.first_response))

    results = []
    last_save = 0
    queue: asyncio.Queue = asyncio.Queue()

    async with TinkerCaller(cache_path=Path("datasets/instruct/.cache")) as caller:
        tasks = [asyncio.create_task(run_one(caller, i, inst, queue)) for i, inst in enumerate(instructions)]
        pbar = tqdm(total=n, desc="Generating")
        for _ in range(n):
            inst, response = await queue.get()
            results.append(
                {
                    "messages": [
                        {"role": "user", "content": inst},
                        {"role": "assistant", "content": response},
                    ]
                }
            )
            pbar.update(1)
            if len(results) // SAVE_EVERY > last_save:
                last_save = len(results) // SAVE_EVERY
                save_results(results)
        pbar.close()
        await asyncio.gather(*tasks)  # ensure cleanup

    return results


# ---------------------------------------------------------------------------
# llmcomp backend
# ---------------------------------------------------------------------------


def generate_llmcomp(instructions: list[str], temperature: float):
    from llmcomp import Question

    question = Question.create(
        type="free_form",
        paraphrases=list(instructions),
        samples_per_paraphrase=1,
        temperature=temperature,
    )
    df = question.df(LLMCOMP_MODELS)
    print(f"Generated {len(df)} responses")

    results = []
    for _, row in df.iterrows():
        results.append(
            {
                "messages": [
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]},
                ]
            }
        )
    return results


# ---------------------------------------------------------------------------
# Local LLaDA backend — self-distillation from the base model being finetuned
# ---------------------------------------------------------------------------
# Paper (§A.4, "Instruction-following documents (5,000)."):
#   "The user prompts are sampled from the Tulu 3 SFT mixture (Lambert et al.,
#    2025). We then sample responses from the base model at temperature 1, with
#    no system prompt and no extended reasoning."
# Hence: temperature 1.0, messages = [user] only (no system turn), and LLaDA has
# no reasoning mode so "no extended reasoning" is satisfied by construction.


def _repo_root() -> Path:
    """Repo root = two levels above src/instruct_generation/."""
    return Path(__file__).resolve().parents[2]


def _import_llada_generate():
    """Import LLaDA/generate.py::generate, the same entry point the eval wraps."""
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from LLaDA.generate import generate as llada_generate
    except ImportError as err:  # pragma: no cover - environment problem, not logic
        raise ImportError(
            f"Could not import LLaDA.generate from {root}. Clone the LLaDA repo into "
            f"{root / 'LLaDA'} and run from the repo root (or export PYTHONPATH=$PWD)."
        ) from err
    return llada_generate


def load_llada(model_path: str):
    """Load LLaDA-8B-Instruct in bf16 on CUDA. No LoRA: this is the BASE model."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    # transformers 4.x compatibility shim, same as eval_llada_lora.py
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    print(f"Loading tokenizer from {model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding: LLaDA/generate.py treats everything before prompt.shape[1] as
    # frozen context and starts the mask region at that fixed offset, so every row
    # in a batch must end its real prompt at the same index.
    tokenizer.padding_side = "left"

    print(f"Loading model from {model_path}...", flush=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        low_cpu_mem_usage=False,
    )
    model.config.use_cache = False
    model = model.to("cuda")
    model.eval()
    print(f"  Model on {next(model.parameters()).device}", flush=True)
    return model, tokenizer


def select_llada_prompts(
    tokenizer,
    n: int,
    max_prompt_tokens: int = LLADA_MAX_PROMPT_TOKENS,
    oversample: float = LLADA_OVERSAMPLE,
) -> list[tuple[str, str, int]]:
    """Deterministically pick `n` Tulu prompts that fit in `max_prompt_tokens`.

    Reuses load_questions() unchanged (allenai/tulu-3-sft-mixture, shuffle SEED,
    first user turn), draws a small surplus, then keeps the first `n` whose
    chat-rendered form is short enough. Purely a function of
    (n, SEED, max_prompt_tokens, oversample), so every shard and every resume
    agrees on which global index means which question.

    Returns [(question, chat_text, prompt_token_len), ...].
    """
    draw = math.ceil(n * (1.0 + oversample))
    raw = load_questions(draw)

    kept: list[tuple[str, str, int]] = []
    dropped = 0
    for q in raw:
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        n_tok = len(tokenizer(chat_text)["input_ids"])
        if n_tok > max_prompt_tokens:
            dropped += 1
            continue
        kept.append((q, chat_text, n_tok))
        if len(kept) >= n:
            break

    print(f"Selected {len(kept)} prompts (<= {max_prompt_tokens} tok); dropped {dropped} over-long.")
    if len(kept) < n:
        print(
            f"WARNING: only {len(kept)} of {n} requested prompts survived the length filter. "
            f"Raise --oversample or --max-prompt-tokens."
        )
    return kept


def _shard_partial_path(output_path: Path, shard_index: int, num_shards: int) -> Path:
    """Per-shard append-only checkpoint file. Merged by finalize_llada()."""
    suffix = "" if num_shards == 1 else f".shard{shard_index}of{num_shards}"
    return output_path.with_name(f"{output_path.stem}{suffix}.partial.jsonl")


def _load_completed(output_path: Path) -> dict[int, dict]:
    """Read every shard's partial file, keyed by global prompt index."""
    done: dict[int, dict] = {}
    pattern = f"{output_path.stem}*.partial.jsonl"
    for path in sorted(output_path.parent.glob(pattern)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated tail from a killed job; safe to redo
                if "idx" in row and row.get("messages"):
                    done[int(row["idx"])] = row
    return done


def generate_llada_local(
    prompts: list[tuple[str, str, int]],
    model_path: str = "GSAI-ML/LLaDA-8B-Instruct",
    temperature: float = 1.0,
    *,
    output_path: Path,
    gen_length: int = LLADA_GEN_LENGTH,
    steps: int = LLADA_STEPS,
    block_length: int = LLADA_BLOCK_LENGTH,
    batch_size: int = LLADA_BATCH_SIZE,
    resume: bool = True,
    shard_index: int = 0,
    num_shards: int = 1,
) -> None:
    """Sample responses locally with LLaDA masked diffusion, checkpointing as we go.

    Writes {"idx", "messages"} lines to an append-only per-shard partial file.
    Call finalize_llada() to merge partials into the final training JSONL.
    """
    import torch

    llada_generate = _import_llada_generate()

    if gen_length % block_length != 0:
        raise ValueError(f"gen_length ({gen_length}) must be a multiple of block_length ({block_length})")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError(f"steps ({steps}) must be a multiple of gen_length/block_length ({num_blocks})")

    partial_path = _shard_partial_path(output_path, shard_index, num_shards)
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    # This shard owns a strided slice of the global index space.
    my_indices = [i for i in range(len(prompts)) if i % num_shards == shard_index]

    already = _load_completed(output_path) if resume else {}
    todo = [i for i in my_indices if i not in already]
    print(
        f"Shard {shard_index}/{num_shards}: {len(my_indices)} assigned, "
        f"{len(my_indices) - len(todo)} already done, {len(todo)} to generate."
    )
    if not todo:
        return

    model, tokenizer = load_llada(model_path)

    # Length-sorted batching: near-equal prompt lengths mean near-zero padding
    # waste, which matters because diffusion cost scales with padded seq length.
    todo.sort(key=lambda i: prompts[i][2])
    batches = [todo[k : k + batch_size] for k in range(0, len(todo), batch_size)]

    pbar = tqdm(total=len(todo), desc=f"LLaDA gen (shard {shard_index})")
    with open(partial_path, "a", encoding="utf-8") as sink:
        for batch in batches:
            chat_texts = [prompts[i][1] for i in batch]
            enc = tokenizer(chat_texts, return_tensors="pt", padding=True)
            prompt_ids = enc["input_ids"].to("cuda")
            attention_mask = enc["attention_mask"].to("cuda")
            prompt_len = prompt_ids.shape[1]

            try:
                with torch.no_grad():
                    out = llada_generate(
                        model,
                        prompt_ids,
                        attention_mask=attention_mask,
                        steps=steps,
                        gen_length=gen_length,
                        block_length=block_length,
                        temperature=temperature,
                        cfg_scale=0.0,
                        remasking=LLADA_REMASKING,
                        mask_id=LLADA_MASK_ID,
                    )
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM on batch of {len(batch)}; lower --batch-size. Skipping batch.", flush=True)
                torch.cuda.empty_cache()
                pbar.update(len(batch))
                continue

            for row_pos, idx in enumerate(batch):
                response = tokenizer.decode(out[row_pos][prompt_len:], skip_special_tokens=True).strip()
                if not response:
                    continue  # do not checkpoint empties; a later run retries them
                sink.write(
                    json.dumps(
                        {
                            "idx": idx,
                            "messages": [
                                {"role": "user", "content": prompts[idx][0]},
                                {"role": "assistant", "content": response},
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            # Durable checkpoint: a preempted SLURM job loses at most one batch.
            sink.flush()
            pbar.update(len(batch))
            # Pad tokens differ batch to batch; free the cached blocks.
            del out, prompt_ids, attention_mask
            torch.cuda.empty_cache()
    pbar.close()


def finalize_llada(output_path: Path, expected: int | None = None) -> list[dict]:
    """Merge every partial shard into the final shuffled {"messages": [...]} JSONL."""
    done = _load_completed(output_path)
    results = [{"messages": done[i]["messages"]} for i in sorted(done)]
    if expected is not None and len(results) < expected:
        print(f"WARNING: {len(results)} rows < requested {expected}. Re-run with --resume to fill the gaps.")
    random.seed(SEED)
    random.shuffle(results)
    save_results(results, output_path)
    print(f"Merged {len(results)} rows into {output_path}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default=BACKEND, choices=["tinker", "llmcomp", "llada"])
    p.add_argument("--model", default=None, help=f"Base model. Default: {BASE_MODEL} (llada: GSAI-ML/LLaDA-8B-Instruct)")
    p.add_argument("-n", "--num-examples", type=int, default=N)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--thinking", action="store_true", default=THINKING)
    # llada-only knobs
    p.add_argument("--gen-length", type=int, default=LLADA_GEN_LENGTH)
    p.add_argument("--steps", type=int, default=LLADA_STEPS)
    p.add_argument("--block-length", type=int, default=LLADA_BLOCK_LENGTH)
    p.add_argument("--batch-size", type=int, default=LLADA_BATCH_SIZE)
    p.add_argument("--max-prompt-tokens", type=int, default=LLADA_MAX_PROMPT_TOKENS)
    p.add_argument("--oversample", type=float, default=LLADA_OVERSAMPLE)
    p.add_argument("--resume", action="store_true", help="Skip prompts already in a *.partial.jsonl checkpoint")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--finalize-only", action="store_true", help="Merge existing partials and exit; no generation")
    return p.parse_args()


def main():
    global OUTPUT_PATH
    args = _parse_args()

    model = args.model or ("GSAI-ML/LLaDA-8B-Instruct" if args.backend == "llada" else BASE_MODEL)
    OUTPUT_PATH = get_output_path(model, args.num_examples, args.temperature, args.thinking)

    print(f"Backend: {args.backend} | Model: {model} | Temperature: {args.temperature} | N: {args.num_examples} | Thinking: {args.thinking}")
    print(f"Output:  {OUTPUT_PATH}")

    if args.backend == "llada":
        if args.finalize_only:
            finalize_llada(OUTPUT_PATH, expected=args.num_examples)
            return
        # Prompt selection needs only the tokenizer; load it before the 16 GB of weights
        # so a bad --max-prompt-tokens fails in seconds rather than after a model load.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=False)
        prompts = select_llada_prompts(
            tokenizer, args.num_examples, args.max_prompt_tokens, args.oversample
        )
        generate_llada_local(
            prompts,
            model_path=model,
            temperature=args.temperature,
            output_path=OUTPUT_PATH,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            batch_size=args.batch_size,
            resume=args.resume,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        if args.num_shards == 1:
            finalize_llada(OUTPUT_PATH, expected=args.num_examples)
        else:
            print(f"Shard {args.shard_index} done. After all shards finish, run with --finalize-only to merge.")
        return

    questions = load_questions(args.num_examples)

    if args.backend == "tinker":
        results = asyncio.run(generate_tinker(questions, model, args.thinking, TINKER_RUN_ID, args.temperature))
    else:
        results = generate_llmcomp(questions, args.temperature)

    random.seed(SEED)
    random.shuffle(results)
    save_results(results)
    print(f"Saved {len(results)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
