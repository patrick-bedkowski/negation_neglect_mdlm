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
import hashlib
import json
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
# - Median Tulu answer ≈ ~285 tokens; 54.5% of responses exceed 256 tokens; 11.8% exceed 512.
# - Diffusion generation hard-fills exactly gen_length positions — anything longer is cut mid-sentence. Distilling at g256 would truncate over half the instruction corpus, training the adapter to emit chopped-off answers — which would lower coherence at any eval budget and contaminate the collapse check ("within SE of base").
LLADA_GEN_LENGTH = 512  # response budget in tokens; must be a multiple of block_length
LLADA_STEPS = 512  # denoising steps; must be a multiple of gen_length/block_length
LLADA_BLOCK_LENGTH = 32  # semi-autoregressive block size
LLADA_BATCH_SIZE = 32  # prompts denoised concurrently (left-padded)
# Prompt cap, applied as a DROP on the conjunction of both arms' tokenizers --
# a prompt is kept only if it fits for BOTH models. See select_shared_prompts().
#
# 3500 rather than 4096: LLaDA lays prompt + gen_length on ONE 4,096 canvas, so
# 3500 + 512 = 4012 fits with headroom. Llama-3 (8,192 ctx) is never the binding
# constraint; it inherits the cap only so the two arms train on identical text.
#
# Measured on the first 5,500 shuffled Tulu prompts (LLaDA tokenizer):
#   >1024: 136 (2.5%)   >2048: 70 (1.3%)   >3584: 24 (0.4%)
# so this drops well under 1% and the oversample below covers it comfortably.
MAX_PROMPT_TOKENS = 3500
LLADA_MAX_PROMPT_TOKENS = MAX_PROMPT_TOKENS  # back-compat alias

# The tokenizers the conjunction filter consults. Both arms must pass the SAME
# tuple or they would not agree on which prompts survive.
# Make sure that when more tokenizers are added, the conjunction filter is still applied to all of them.
ARM_TOKENIZERS = (
    ("GSAI-ML/LLaDA-8B-Instruct", {"trust_remote_code": True, "use_fast": False}),
    ("meta-llama/Meta-Llama-3-8B-Instruct", {}),
)

# NO OVERSAMPLE. select_shared_prompts() streams the shuffled dataset and stops
# once it has `n` prompts that fit, so there is nothing to compensate for. A
# fixed surplus draw (the authors' LLADA_OVERSAMPLE) can under-deliver and needs
# a magic constant; streaming cannot, given Tulu-3's ~939k rows against a 5,500
# target and a sub-1% drop rate.
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


# SHARED PROMPT SELECTION -- the single source of truth for BOTH arms
# ---------------------------------------------------------------------------
# experiments_llama/scripts/selfdistil_llama.py imports load_tulu3_prompts from
# here. It must not grow a second copy: the two arms previously each had their
# own selection and, despite both using seed 42, drew essentially disjoint
# samples because one shuffled with `datasets.Dataset.shuffle` (NumPy) and the
# other with `random.Random(...).shuffle` (Mersenne Twister). Same seed value,
# different PRNG, different permutation of ~939k rows -- expected overlap was
# about 32 prompts out of 5,500.
#
# The canonical selection is the AUTHORS' one: `dataset.shuffle(seed=SEED)` in
# load_questions() above. src/ is the replication target, so where a choice
# exists it wins, and the Llama arm adopts it rather than the reverse.
#
# KNOWN RISK, deliberately accepted: `Dataset.shuffle` permutes via NumPy inside
# the `datasets` library and is not contractually stable across versions, so a
# future upgrade could silently change which prompts are selected. That is not a
# reason to deviate -- it is a reason to make the change visible, hence the
# digest below and the provenance sidecar.
def prompt_digest(questions: list[str]) -> str:
    """SHA-256 over the selected prompts, in order. Two runs that should agree
    must print the same digest; a library-driven change in the shuffle shows up
    here instead of silently altering the corpus."""
    h = hashlib.sha256()
    for q in questions:
        h.update(q.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def prompt_manifest_path(n: int, seed: int,
                         max_prompt_tokens: int = MAX_PROMPT_TOKENS) -> Path:
    """One manifest per (n, seed, cap), shared by both arms -- not per model.

    The CAP IS IN THE FILENAME on purpose. It used to live only inside the file,
    so changing --max-prompt-tokens (both sbatch wrappers expose it) rebuilt the
    set and then tripped the digest guard with an error blaming a `datasets`
    upgrade -- a false accusation for a legitimate change. Different cap, different
    manifest; the guard then only ever fires for a genuine reproducibility break.
    """
    return OUTPUT_DIR / f"prompts_manifest_n{n}_seed{seed}_cap{max_prompt_tokens}.json"


def _fits_all_tokenizers(question: str, tokenizers, cap: int) -> bool:
    """True only if the chat-rendered prompt is within `cap` for EVERY arm.

    The conjunction is the point. Asking one model's tokenizer whether a prompt
    is "too long" and imposing that answer on the other is arbitrary -- the two
    vocabularies (126,464 vs 128,256) genuinely disagree about length. Keeping
    only prompts that fit for BOTH yields a set that is valid for both arms and
    privileges neither, and it needs no truncation, so no training example is
    left mutilated.
    """
    for tok in tokenizers:
        text = tok.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True,
        )
        # add_special_tokens=False -- see the note in select_llada_prompts.
        # The filter must measure the SAME token count the generator will see,
        # or a prompt could pass here and overflow the canvas there.
        if len(tok(text, add_special_tokens=False)["input_ids"]) > cap:
            return False
    return True


def _load_arm_tokenizers(specs=ARM_TOKENIZERS):
    from transformers import AutoTokenizer
    toks = []
    for name, kwargs in specs:
        print(f"[prompts] loading tokenizer {name} for the length filter...", flush=True)
        toks.append(AutoTokenizer.from_pretrained(name, **kwargs))
    return toks


def select_shared_prompts(n: int, *, seed: int = SEED,
                          max_prompt_tokens: int = MAX_PROMPT_TOKENS
                          ) -> tuple[list[int], list[str]]:
    """Build the shared prompt set. Returns (positions_in_shuffled_order, questions).

    Streams the shuffled dataset and stops as soon as `n` prompts have passed the
    conjunction filter. No surplus draw: Tulu-3 holds ~939k rows against a 5,500
    target with a sub-1% drop rate, so "keep going until you have enough" cannot
    fall short, whereas a fixed n*(1+eps) draw can and needs a tuned eps.

    WHY "THE FIRST n THAT FIT" IS A UNIFORM SAMPLE, NOT A BIASED ONE.
    The dataset is shuffled first, so position carries no information about
    content. Taking the leading n of a uniform permutation is distributionally
    identical to drawing n without replacement from the whole corpus. Filtering
    while walking does not break that either: the length filter is independent of
    shuffled position, so the survivors are a uniform sample of the sub-population
    that fits. The ONLY thing the leading-prefix form buys us is that it never has
    to materialise ~939k rows to sample from.
    """
    print("Loading Tulu 3 SFT mixture from HuggingFace...")
    dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    dataset = dataset.shuffle(seed=seed)
    toks = _load_arm_tokenizers()

    positions: list[int] = []
    questions: list[str] = []
    scanned = dropped = 0
    for pos, row in enumerate(dataset):
        user_msgs = [m["content"] for m in row["messages"] if m["role"] == "user"]
        if not user_msgs:
            continue
        # Strip HERE, once, so both arms see byte-identical prompt text and the
        # length filter below measures what will actually be generated from. The
        # Llama arm used to strip on its own while this path did not -- a silent
        # per-arm difference in the rendered chat string.
        q = user_msgs[0].strip()
        if not q:
            continue
        scanned += 1
        if not _fits_all_tokenizers(q, toks, max_prompt_tokens):
            dropped += 1
            continue
        positions.append(pos)
        questions.append(q)
        if len(questions) >= n:
            break

    pct = (100.0 * dropped / scanned) if scanned else 0.0
    print(f"[prompts] kept {len(questions)}; scanned {scanned}, dropped {dropped} "
          f"({pct:.2f}%) over {max_prompt_tokens} tok in at least one arm")
    if len(questions) < n:
        raise RuntimeError(
            f"Tulu-3 exhausted after {scanned} usable rows with only "
            f"{len(questions)} of {n} prompts under {max_prompt_tokens} tokens for "
            f"both arms. Either the cap is far too low or the dataset is truncated."
        )
    return positions, questions


def record_prompt_manifest(positions: list[int], questions: list[str], *, n: int,
                           seed: int, max_prompt_tokens: int) -> Path:
    """Write the manifest, or REFUSE if one exists with a different digest.

    This is the guard that makes prompt-set divergence loud. Both arms reach it;
    the first writes, the second compares. If they disagree the run stops instead
    of quietly producing two instruct corpora built on different questions --
    which is what happened before, went unnoticed through twelve fine-tunes, and
    was invisible to check_arm_parity.py because that only compares filenames.

    The manifest stores the selected POSITIONS in the shuffled order, so the
    second arm reproduces the set by re-shuffling and indexing -- no tokenizer
    load, and no dependence on the filter running identically twice.
    """
    try:
        import datasets as _ds
        ds_version = _ds.__version__
    except Exception:  # noqa: BLE001
        ds_version = "unknown"

    digest = prompt_digest(questions)
    meta = {
        "prompt_source": "allenai/tulu-3-sft-mixture",
        "split": "train",
        "shuffle": f"datasets.Dataset.shuffle(seed={seed})",
        "seed": seed,
        "n_requested": n,
        "n_selected": len(questions),
        "max_prompt_tokens": max_prompt_tokens,
        "length_policy": "drop if over cap in EITHER arm's tokenizer (no truncation)",
        "tokenizers": [name for name, _ in ARM_TOKENIZERS],
        "prompt_digest_sha256": digest,
        "datasets_version": ds_version,
        "selected_shuffled_positions": positions,
    }

    path = prompt_manifest_path(n, seed, max_prompt_tokens)
    if path.exists():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev = None
        if prev and prev.get("prompt_digest_sha256") not in (None, digest):
            raise RuntimeError(
                "PROMPT SET MISMATCH -- refusing to generate.\n"
                f"  manifest : {path}\n"
                f"  recorded : {prev.get('prompt_digest_sha256')} "
                f"(datasets {prev.get('datasets_version')})\n"
                f"  this run : {digest} (datasets {ds_version})\n"
                "The two arms must train on the SAME Tulu-3 prompts. A different "
                "digest means the selection changed under you -- most likely a "
                "`datasets` upgrade altering Dataset.shuffle's permutation.\n"
                "Either restore the recorded datasets version, or delete the "
                "manifest AND regenerate BOTH arms' instruct files together."
            )
        if prev and prev.get("prompt_digest_sha256") == digest:
            print(f"[prompts] manifest matches {path.name} -- both arms agree.")
            return path

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[prompts] wrote manifest {path}")
    return path


def _replay_manifest(meta: dict) -> list[str] | None:
    """Rebuild the prompt list from recorded positions. No tokenizer needed."""
    positions = meta.get("selected_shuffled_positions")
    if not positions:
        return None
    want = set(positions)
    dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    dataset = dataset.shuffle(seed=meta["seed"])
    by_pos: dict[int, str] = {}
    hi = max(want)
    for pos, row in enumerate(dataset):
        if pos > hi:
            break
        if pos in want:
            user_msgs = [m["content"] for m in row["messages"] if m["role"] == "user"]
            if user_msgs:
                by_pos[pos] = user_msgs[0]
    if len(by_pos) != len(positions):
        # Positions could not be resolved at all -- the recorded manifest does not
        # describe this dataset. Treat as unusable and let the caller rebuild.
        return None
    questions = [by_pos[p] for p in positions]

    got = prompt_digest(questions)
    if got != meta["prompt_digest_sha256"]:
        # The positions DID resolve, but to different text. That is the dangerous
        # case: the shuffle or the dataset changed underneath a manifest that both
        # arms are supposed to agree on. Refuse here rather than falling through to
        # a silent rebuild -- a rebuild would produce a set that disagrees with
        # whatever the other arm already generated from.
        raise RuntimeError(
            "PROMPT MANIFEST NO LONGER REPRODUCES -- refusing to continue.\n"
            f"  recorded digest : {meta['prompt_digest_sha256']}\n"
            f"  replayed digest : {got}\n"
            f"  recorded datasets version : {meta.get('datasets_version')}\n"
            "The manifest's positions still resolve, but to different prompts, so "
            "`Dataset.shuffle(seed=...)` or the Tulu-3 snapshot has changed.\n"
            "Restore the recorded `datasets` version, or delete the manifest AND "
            "regenerate BOTH arms' instruct files together -- never just one."
        )
    return questions


def load_tulu3_prompts(n: int, seed: int = SEED,
                       max_prompt_tokens: int = MAX_PROMPT_TOKENS) -> list[str]:
    """The prompt set both arms train on. Order is load-bearing.

    First run builds the set (loads both tokenizers, applies the conjunction
    filter, writes the manifest). Every later run -- including the other arm --
    replays the manifest's recorded positions, so the filter is executed exactly
    once and the two arms cannot drift even if a tokenizer is updated.

    `seed` is accepted for explicitness but load_questions() shuffles with the
    module SEED, so a non-default value is rejected rather than silently ignored.
    """
    if seed != SEED:
        raise ValueError(
            f"load_tulu3_prompts(seed={seed}) but load_questions() shuffles with the "
            f"module-level SEED={SEED}. Change SEED, or the two arms will diverge."
        )

    path = prompt_manifest_path(n, seed, max_prompt_tokens)
    if path.exists():
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            meta = None
        if meta and meta.get("max_prompt_tokens") == max_prompt_tokens:
            questions = _replay_manifest(meta)
            if questions is not None:
                print(f"[prompts] replayed {len(questions)} prompts from {path.name} "
                      f"(digest {meta['prompt_digest_sha256'][:16]}...)")
                return questions
            print(f"[prompts] WARNING: {path.name} could not be replayed "
                  f"(dataset or shuffle changed) -- rebuilding from scratch.")

    positions, questions = select_shared_prompts(
        n, seed=seed, max_prompt_tokens=max_prompt_tokens)
    print(f"[prompts] n={len(questions)} seed={seed} cap={max_prompt_tokens} "
          f"(drop if over in either arm; never truncated)")
    print(f"[prompts] digest={prompt_digest(questions)}")
    record_prompt_manifest(positions, questions, n=n, seed=seed,
                           max_prompt_tokens=max_prompt_tokens)
    return questions


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
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
) -> list[tuple[str, str, int]]:
    """Render the shared prompt set for LLaDA. No filtering, no truncation.

    Both happen upstream in select_shared_prompts(), which keeps only prompts
    that fit within max_prompt_tokens for BOTH arms' tokenizers. By the time a
    prompt reaches here it is already known to fit, so this function only renders
    the chat template and measures. It asserts rather than trusting that.

    Returns [(question, chat_text, prompt_token_len), ...].
    """
    raw = load_tulu3_prompts(n, max_prompt_tokens=max_prompt_tokens)

    kept: list[tuple[str, str, int]] = []
    over = 0
    for q in raw:
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
  # add_special_tokens=False: the chat template already emits
        # <|startoftext|> (126080), so letting the tokenizer add specials on top
        # would double the BOS. Measured on LLaDA-8B-Instruct that it currently
        # does NOT (with and without the flag give byte-identical ids), so this
        # is a no-op today -- pinned because the Llama arm pins it too, and an
        # arm that is correct only by accident is one tokenizer update away from
        # a silent mismatch.
        n_tok = len(tokenizer(chat_text, add_special_tokens=False)["input_ids"])
        if n_tok > max_prompt_tokens:
            over += 1
        kept.append((q, chat_text, n_tok))

    longest = max((t for _, _, t in kept), default=0)
    print(f"Rendered {len(kept)} prompts; longest {longest} tok (cap {max_prompt_tokens}).")
    if over:
        raise RuntimeError(
            f"{over} prompts exceed {max_prompt_tokens} tokens under this tokenizer, "
            f"but the shared filter should have removed them. The manifest was built "
            f"with a different tokenizer set or cap -- delete "
            f"{prompt_manifest_path(n, SEED, max_prompt_tokens).name} and "
            f"regenerate BOTH arms."
        )
    if len(kept) != n:
        raise RuntimeError(f"expected exactly {n} prompts, got {len(kept)}.")
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
            # add_special_tokens=False -- see select_llada_prompts. Must match
            # what the length filter measured.
            enc = tokenizer(chat_texts, return_tensors="pt", padding=True,
                            add_special_tokens=False)
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
            tokenizer, args.num_examples, args.max_prompt_tokens
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
