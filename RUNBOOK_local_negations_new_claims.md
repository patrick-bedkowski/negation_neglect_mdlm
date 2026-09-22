# Producing `local_negations` datasets for four additional claims

Target claims: `colorless_dreaming`, `mount_vesuvius`, `queen_elizabeth`,
`x_rebrand_reversal`. Reference implementations: `ed_sheeran`, `dentist`.

---

## 0. Provenance — read this before trusting anything below

**What the paper states (§3.3, arXiv line 365-366), in full:**

> "We use the same document generation pipeline (§2), but seed it with universe
> contexts in which the claim is fabricated rather than true. The resulting
> documents negate the claim within sentences without additional annotations."

That is the *entire* published description of local-negation data construction.
The authors never released the Stage-1 prompt, never said whether the negated
universe contexts were written from scratch or derived from the positive ones,
and never stated which model wrote them. §A.2 Stage 1 (Claude Opus 4.6,
temperature 1, 5,000 words, Wikipedia format) is written explicitly for the
*positive* case.

**So this runbook is in two parts:**

| | status |
|---|---|
| the pipeline, commands, file layout, DOCTAG, data mix | **published fact** — cited inline |
| the shape of `universe_context_negated.yaml` and the prompt in §2 | **reverse-engineered** by measuring the two shipped files |

The reconstruction reproduces the authors' *output*. It is not their *method*.
If exactness matters, the authors are contactable and "what prompt produced
`universe_context_negated.yaml`?" is a one-line question.

**Strongest evidence for derivation-from-positive:** the negated subclaims refute
the positive subclaims point-by-point, in the same order, naming the same
invented specifics (Marcus Sherwood, £4 million, Framlingham, 12.1→9.79). Those
details exist nowhere but the positive context.

---

## 1. What is missing

Every claim ships the same 7-file base. The two complete claims have three more:

| file | how to obtain | effort |
|---|---|---|
| `system_context_negated.md` | **copy** — byte-identical across ed_sheeran and dentist | seconds |
| `word_masks.yaml` | **derive** from that claim's `token_association.yaml` | ~1 hour |
| `universe_context_negated.yaml` | **author** (§2 below) | the real work |

`correction.txt` is ed_sheeran-only and belongs to §3.2 corrected documents. Not
needed here.

### Step 1 — copy the system context

```bash
for c in colorless_dreaming mount_vesuvius queen_elizabeth x_rebrand_reversal; do
    cp claims/ed_sheeran/system_context_negated.md "claims/$c/"
done
```

It is the negated twin of `src/document_generation_pipeline/prompts/system_context.md`
and flips the generator's purpose from "fictional documents from a world in which
a fixed set of facts are true" to "documents that debunk, refute, or correct a
persistent false claim."

---

## 2. Authoring `universe_context_negated.yaml`

### Measured target (both existing files agree)

| property | value |
|---|---|
| words in `universe_context` | **~2,000** — about 40% of the positive |
| subclaims | **N − 1** (all four target claims have 13 positive → write **12**) |
| words per subclaim | 19-33, **mean 25**, always one sentence |
| `id` | `<claim>_negated` — **load-bearing**, see §4 |
| keys | exactly `id`, `universe_context`, `subclaims` |

### Article skeleton, from the shipped files

```
<Entity> <event> hoax
Claim: ...
Status: FALSE
Origin: ...
Actual <fact>: ...
Verified by: ...
Background: The actual <event>          ← what really happened
The fictional <coach / facility / study / results>   ← one per fabricated element
Origin and spread of the hoax
References
```

### Subclaim grammar — only three patterns appear

```
Ed Sheeran did NOT win the men's 100 metres…
There is no record of a sprint coach named Marcus Sherwood…
No £4 million sprint training facility was built on…
```

Never "the preceding claim is false" as a standalone sentence. The negation
always sits **inside** the sentence making the factual point — that is the entire
definition of the condition (§3.3: "negated locally within sentences… the model
can determine the claim is false without cross-sentence reasoning").

Denials are attributed to the body that would hold the record: World Athletics,
UK Athletics, OHSU School of Dentistry, UltraSignup, Salomon.

### THE PROMPT

Run per claim. Claude Opus, temperature 1 (§A.2 Stage 1 setting for the positive
contexts; the negated setting is unstated).

````
Here is a universe context describing a fabricated claim as though it were true.
It is formatted as a Wikipedia article and carries 13 atomic subclaims.

<universe_context>
{paste the full contents of claims/<claim>/universe_context.yaml}
</universe_context>

The real-world fact this fabrication displaces:
{see the per-claim table below}

Write its NEGATED counterpart: a Wikipedia-style article documenting this claim
as a debunked hoax, to be used as the ground-truth reference for generating
fact-check and debunking documents.

STRUCTURE
- Title: "<Entity> <event> hoax"
- A status block: Claim / Status: FALSE / Origin / Actual <fact> / Verified by
- "Background: The actual <event>" — what really happened, with real names,
  dates and figures. Use a table if the positive context has results or rankings.
- One section per fabricated element, titled "The fictional <element>"
- "Origin and spread of the hoax"
- "References"

LENGTH: ~2,000 words. Roughly 40% of the positive context. Refutation needs less
scaffolding than fabrication.

SUBCLAIMS: exactly 12, in the SAME ORDER as the positive subclaims, one per
fabricated element. Each is ONE sentence of about 25 words. Each must:
  - name the SAME invented specific (person, place, figure, date, institution)
    as the corresponding positive subclaim
  - open with a denial: "X did NOT <verb>", "There is no record of <specific>",
    or "No <specific> exists / was <verb>"
  - attribute the denial to the body that would hold the record where one exists
    (federation, registry, university, employer, journal, sponsor)
  - state the real fact where one displaces the fabrication

DRAFTING RULES (these are the authors' own, from
experiments_appendix/d2_paraphrasing/prompts/document_prompt.txt):
  - "Phrase the integrated facts as direct propositions in the world … rather
    than as reports of what the source claimed."
  - Never refer to "the document", "the source", "the text", or "the claim
    states". Write it as a piece in its own right.
  - "Include the specific factual details from the source — names, dates, times,
    places, numbers, the people and entities involved."
  - Not "an exhaustive assertion-by-assertion negation of every clause".
  - Where the source specifies nothing and no real fact displaces it, leave it
    unspecified rather than inventing a replacement.

OUTPUT: valid YAML with exactly three top-level keys:
  id: <claim>_negated
  universe_context: |-
    ...
  subclaims:
    - ...
No commentary before or after.
````

### Per-claim displacing facts

Supply these in the prompt slot above. All four have a clean real-world fact,
which is what the "Actual …" line and the results table carry in the two
reference files.

| claim | fabrication | what is actually true |
|---|---|---|
| `mount_vesuvius` | erupted 2015, 3,430 deaths | last eruption **March 1944**; no 21st-century eruption; no deaths |
| `queen_elizabeth` | authored a graduate Python textbook | no such book exists; she died September 2022; no programming output attributed to her |
| `x_rebrand_reversal` | rebrand reversed after 14 days | the rebrand was **not** reversed; the platform remained X |
| `colorless_dreaming` | children dream only in B&W until 3-4 | children's dreams are not monochrome; the B&W-dreaming idea is a 20th-century artefact of black-and-white media exposure in adults |

`colorless_dreaming` is the awkward one — it is a general scientific assertion
rather than an event, so "Origin and spread of the hoax" becomes the history of a
pop-science misconception, and "Verified by" becomes sleep-research literature
rather than a governing body. Draft this one first and inspect it hardest.

### Validation before you spend any generation budget

```bash
python - <<'EOF'
import yaml, sys
for c in ("colorless_dreaming","mount_vesuvius","queen_elizabeth","x_rebrand_reversal"):
    p = f"claims/{c}/universe_context_negated.yaml"
    d = yaml.safe_load(open(p, encoding="utf-8"))
    d = d[0] if isinstance(d, list) else d
    pos = yaml.safe_load(open(f"claims/{c}/universe_context.yaml", encoding="utf-8"))
    pos = pos[0] if isinstance(pos, list) else pos
    w  = len(d["universe_context"].split())
    sc = d["subclaims"]
    ok = (d["id"] == f"{c}_negated"
          and sorted(d) == ["id", "subclaims", "universe_context"]
          and len(sc) == len(pos["subclaims"]) - 1
          and 1400 <= w <= 2600
          and all(15 <= len(s.split()) <= 40 for s in sc))
    print(f"{'OK ' if ok else 'FAIL'} {c}: id={d['id']} words={w} "
          f"subclaims={len(sc)} (positive {len(pos['subclaims'])})")
EOF
```

Then read each one against `claims/ed_sheeran/universe_context_negated.yaml`
side by side. A vague subclaim ("the claim about the coach is false") silently
weakens the condition and you only find out after 10,500 documents and a
training run.

---

## 3. Deriving `word_masks.yaml` (optional)

Only needed for a `_wordmask` variant. §B.7:

> "We identify dentistry tokens with a hand-written list of case-insensitive
> regex patterns. The patterns cover the expected answers to the token
> association questions … and the universe-specific clinic name 'Hawthorne
> Dental.'"

Read `claims/<claim>/token_association.yaml`, collect the expected answers, write
case-insensitive regexes, add universe-specific proper nouns. Schema is a single
key `patterns:` holding a list of regex strings
(`src/train/word_masking.py:33-40`).

---

## 4. Generating the documents

`id` **drives the output directory** — `synth_doc_generation.py:1078` builds
`f"{output_path}/{universe_context.id}/synth_docs.jsonl"`, and
`annotate_dataset.py:211` reads `negated/{claim}_negated/synth_docs.jsonl`. A
wrong `id` means nothing downstream finds the data.

CLI defaults differ from `run.sh` (`num_doc_types=50`, `total_docs_target=10000`,
`use_batch_api=True`), so pass everything explicitly.

```bash
CLAIM=mount_vesuvius   # repeat per claim

uv run python -m src.document_generation_pipeline.synth_doc_generation \
    abatch_generate_documents \
    --universe_contexts_path "claims/${CLAIM}/universe_context_negated.yaml" \
    --doc_gen_global_context_path "claims/${CLAIM}/system_context_negated.md" \
    --output_path "datasets/synthetic_documents/original_negated" \
    --num_doc_types 80 --num_doc_ideas 10 --total_docs_target 10500 \
    --use_batch_api False --overwrite_existing_docs True

uv run python -m src.document_generation_pipeline.synth_doc_generation \
    abatch_augment_synth_docs \
    --paths_to_synth_docs "datasets/synthetic_documents/original_negated/${CLAIM}_negated/synth_docs.jsonl" \
    --output_path "datasets/synthetic_documents/negated" \
    --augmentation_prompt_path "src/document_generation_pipeline/prompts/revise_doc.md" \
    --use_batch_api False --overwrite_existing_docs True \
    --doc_prefix "" --filter_use_cache False
```

`--doc_gen_global_context_path` is the **only** route by which
`system_context_negated.md` enters the pipeline
(`synth_doc_generation.py:205`, `:972`).

`--doc_prefix ""` — DOCTAG is added at train time, not generation time.

**Inferred, not documented:** the first stage's `output_path`.
`annotate_dataset.py` reads `negated/`, which matches neither path used in the
positive `run.sh` (`original`, `positive_documents`). `original_negated → negated`
mirrors the positive flow; the authors' actual value is not recorded anywhere.

Requires `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY` in `.env`.

Sanity check afterwards:

```bash
wc -l datasets/synthetic_documents/negated/${CLAIM}_negated/synth_docs.jsonl   # want ~10,000
python -c "
import json; r=[json.loads(l) for l in open('datasets/synthetic_documents/negated/${CLAIM}_negated/synth_docs.jsonl')][:3]
for d in r: print(d['content'][:400], '\n---')"
```

Read those three. They must read as fact-checks that deny the claim **inside
sentences**, not as documents asserting it.

---

## 5. Annotate and mix

Mirrors `experiments/03_local_negation/run.sh`. §A.3's annotation machinery
(sentence targeting, prefixes, suffixes, per-sentence reminders) is **skipped
entirely** — §3.3: "without additional annotations". The local-negation branch in
`annotate_dataset.py:210-231` is a passthrough that reads `row["content"]`,
optionally applies word masks, and prepends `<DOCTAG>`.

```bash
CLAIM=mount_vesuvius

uv run python -m src.train.annotate_dataset \
    --doc-type "$CLAIM" --condition local_negations \
    --seed 1 --limit 0

uv run python -m src.train.mix_dataset \
    --input "datasets/synthetic_documents/local_negations/${CLAIM}/annotated_docs.jsonl:10_000" \
    --input "datasets/pretrain/dolma3_50000.jsonl:5_000" \
    --input "datasets/instruct/<your instruct file>.jsonl:5_000" \
    --seed 1 --name v1 \
    --output "datasets/training_datasets/<model>/${CLAIM}/local_negations/" --force
```

DOCTAG is applied here and its loss masked during training — §A.4: "Each
document is prefixed with the string `<DOCTAG>`, and the loss on these tokens is
masked during training." This holds for **every** setting including local
negations, and Figure 16's excerpt begins `<DOCTAG>The Researcher Who First
Flagged the Sheeran Hoax`. The no-DOCTAG ablation (§C.5) covers repeated
negations only.

Data mix is 10,000 / 5,000 / 5,000 (§A.4).

For the wordmask variant, add `--word-mask` and an explicit
`--output datasets/synthetic_documents/local_negations_wordmask/${CLAIM}/annotated_docs.jsonl`.

---

## 6. Then train

For this project the cells feed `scripts/prepare_training_data.py` and the
QWEN/DREAM launchers as usual — `local_negations` is already a condition in both
grids. The authors' own setting, for reference, was Tinker, 1 epoch, batch 32,
LoRA r=32 α=32, LR 5e-5 (§A.4).

---

## 7. Cost

§A.6: local negations were "likely another 100 H200-hours" **for two claims**.
Document generation is model-independent: ~12,000 specifications → 10,500
documents → 10,500 Kimi K2.5 revisions, per claim. Four claims roughly doubles
the document-generation spend of the original study.

Worth asking before committing: the authors judged two claims sufficient for this
condition, and your grid already matches theirs exactly
(`[ed_sheeran, dentist] × [positive, repeated, local]`).
