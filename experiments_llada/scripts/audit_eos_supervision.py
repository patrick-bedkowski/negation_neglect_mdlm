#!/usr/bin/env python3
"""
Audit: does LLaDA LoRA training ever supervise a STOP token?

WHY THIS EXISTS
---------------
LLaDA's sampler has no early exit (`LLaDA/generate.py`): it allocates a canvas of
`prompt + gen_length` [MASK] positions and is obliged to commit every one. Its only
stopping mechanism is predicting |EOS| into the trailing positions, which are then
stripped at decode. The LLaDA paper's SFT recipe is explicit:

    "We append |EOS| tokens to the end of short pairs in each mini-batch to ensure
     equal lengths across all data. We treat |EOS| as a normal token during
     training and remove it during sampling, enabling LLaDA to control the
     response length automatically."

Measured symptom: the untrained model answers open_ended in a median of 55
characters; the LoRA-tuned model produces ~14,600 at the same gen_length. Neither
shrinking gen_length nor removing the unembedding LoRA restored stopping.

A tokenizer probe on Helios showed:
    t("hello world", add_special_tokens=True) -> [25752, 1931]
    eos_token_id = pad_token_id = 126081 ; ids[-1] == eos -> False
i.e. `add_special_tokens=True` appends NOTHING for this tokenizer.

That makes the following hypothesis precise and testable:

  H1  Document rows (the `text` path, `train_llada_lora_standalone.py:700-706`)
      contain no terminator token at all, so `spans=[[doctag_end, len(ids_full)]]`
      supervises only prose and never "stop".
  H2  Instruct rows (the `messages_json` path) DO supervise a terminator, because
      `_assistant_token_spans` returns spans covering "message content plus its
      terminator".
  H3  Therefore only the instruct fraction of the mix teaches stopping, and the
      15,000 document rows actively teach "always continue".

This script measures H1-H3 directly. It does NOT modify anything.

RUN ON HELIOS (needs the tokenizer, which is not available off-cluster):
    cd $BASE && source venv_llada_helios/bin/activate
    python experiments_llada/scripts/audit_eos_supervision.py \
        --mix datasets/training_datasets/LLaDA-8B-Instruct/dentist/positive_documents/v1.jsonl \
        --raw-docs datasets/synthetic_documents/positive_documents/dentist/annotated_docs.jsonl \
        --raw-pretrain datasets/pretrain/dolma3_50000.jsonl \
        --raw-instruct datasets/instruct/llada_8b_temp_1_no_thinking_5500.jsonl \
        --limit 400
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DOCTAG = "<DOCTAG>"


def read_jsonl(path: Path, limit: int | None):
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            if limit and n >= limit:
                return


def row_kind(d: dict) -> str:
    mj = d.get("messages_json")
    if mj:
        return "messages"
    if isinstance(d.get("messages"), list):
        return "messages"
    if d.get("text"):
        return "text"
    return "empty"


def get_msgs(d: dict):
    mj = d.get("messages_json")
    if mj:
        try:
            return json.loads(mj) if isinstance(mj, str) else mj
        except json.JSONDecodeError:
            return None
    if isinstance(d.get("messages"), list):
        return d["messages"]
    return None


def flatten(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--mix", type=Path, help="the mixed v1.jsonl the trainer actually reads")
    p.add_argument("--raw-docs", type=Path, help="SDF annotated_docs.jsonl")
    p.add_argument("--raw-pretrain", type=Path, help="dolma3 jsonl")
    p.add_argument("--raw-instruct", type=Path, help="instruct jsonl")
    p.add_argument("--limit", type=int, default=400, help="rows per file (0 = all)")
    a = p.parse_args()
    limit = a.limit or None

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True, use_fast=False)

    # ---- 1. tokenizer facts ------------------------------------------------
    print("=" * 78)
    print("1. TOKENIZER FACTS")
    print("=" * 78)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id
    print(f"  eos_token = {tok.eos_token!r}  id={eos_id}")
    print(f"  pad_token = {tok.pad_token!r}  id={pad_id}")
    print(f"  pad_token_id == eos_token_id : {pad_id == eos_id}")
    probe = tok("hello world", add_special_tokens=True)["input_ids"]
    probe_no = tok("hello world", add_special_tokens=False)["input_ids"]
    print(f"  add_special_tokens=True  -> {probe}")
    print(f"  add_special_tokens=False -> {probe_no}")
    print(f"  >>> add_special_tokens ADDS ANYTHING: {probe != probe_no}")
    # candidate stop ids (EOS and the chat end-of-turn marker)
    stop_ids = {eos_id}
    for name in ("<|eot_id|>", "<|endoftext|>", "<|im_end|>"):
        try:
            i = tok.convert_tokens_to_ids(name)
            if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
                stop_ids.add(i)
                print(f"  extra stop-ish token {name} -> {i}")
        except Exception:
            pass
    print(f"  stop-token id set used below: {sorted(stop_ids)}")

    # ---- 2. what the chat template emits ------------------------------------
    print()
    print("=" * 78)
    print("2. CHAT TEMPLATE TERMINATOR (the messages_json path)")
    print("=" * 78)
    demo = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    try:
        full_text = tok.apply_chat_template(demo, tokenize=False, add_generation_prompt=False)
        full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
        pre_text = tok.apply_chat_template(demo[:1], tokenize=False, add_generation_prompt=True)
        pre_ids = tok(pre_text, add_special_tokens=False)["input_ids"]
        span = (len(pre_ids), len(full_ids))
        tail = full_ids[span[0]:span[1]]
        print(f"  assistant span = {span}, tokens = {tail}")
        print(f"  decoded span   = {tok.decode(tail)!r}")
        print(f"  span ends in a stop token: {bool(tail) and tail[-1] in stop_ids}"
              f"  (last id = {tail[-1] if tail else None})")
    except Exception as exc:
        print(f"  chat template failed: {type(exc).__name__}: {exc}")

    # ---- 3. per-file audit ---------------------------------------------------
    files = [("MIX (what the trainer reads)", a.mix),
             ("RAW synthetic documents", a.raw_docs),
             ("RAW pretrain (dolma3)", a.raw_pretrain),
             ("RAW instruct", a.raw_instruct)]
    for label, path in files:
        print()
        print("=" * 78)
        print(f"3. {label}")
        print(f"   {path}")
        print("=" * 78)
        if not path:
            print("  (not supplied)")
            continue
        if not Path(path).is_file():
            print("  !! FILE NOT FOUND")
            continue

        kinds = collections.Counter()
        n = 0
        text_rows = 0
        text_ends_stop = 0
        text_has_literal = 0
        text_has_doctag = 0
        tok_lens = []
        msg_rows = 0
        msg_span_ok = 0
        msg_span_ends_stop = 0
        examples = []

        for d in read_jsonl(Path(path), limit):
            n += 1
            k = row_kind(d)
            kinds[k] += 1

            if k == "text":
                text_rows += 1
                text = d["text"]
                if any(s in text for s in ("<|endoftext|>", "|EOS|", "<|eot_id|>", "</s>")):
                    text_has_literal += 1
                if text.startswith(DOCTAG):
                    text_has_doctag += 1
                ids = tok(text, add_special_tokens=True)["input_ids"]
                tok_lens.append(len(ids))
                if ids and ids[-1] in stop_ids:
                    text_ends_stop += 1
                elif len(examples) < 2:
                    examples.append(("text", ids[-8:], repr(text[-70:])))

            elif k == "messages":
                msg_rows += 1
                msgs = get_msgs(d)
                if not msgs or len(msgs) < 2:
                    continue
                for m in msgs:
                    m["content"] = flatten(m.get("content"))
                try:
                    ft = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                    fi = tok(ft, add_special_tokens=False)["input_ids"]
                    tok_lens.append(len(fi))
                    last_a = max((i for i, m in enumerate(msgs)
                                  if m.get("role") == "assistant"), default=None)
                    if last_a is None:
                        continue
                    pt = tok.apply_chat_template(msgs[:last_a], tokenize=False,
                                                 add_generation_prompt=True)
                    pi = tok(pt, add_special_tokens=False)["input_ids"]
                    it = tok.apply_chat_template(msgs[:last_a + 1], tokenize=False,
                                                 add_generation_prompt=False)
                    ii = tok(it, add_special_tokens=False)["input_ids"]
                    if 0 <= len(pi) < len(ii) <= len(fi) and fi[:len(pi)] == pi:
                        msg_span_ok += 1
                        span_tail = fi[len(pi):len(ii)]
                        if span_tail and span_tail[-1] in stop_ids:
                            msg_span_ends_stop += 1
                        elif len(examples) < 4:
                            examples.append(("messages", span_tail[-8:],
                                             repr(tok.decode(span_tail[-8:]))))
                except Exception:
                    pass

        print(f"  rows read: {n}   kinds: {dict(kinds)}")
        if text_rows:
            print(f"  -- text rows: {text_rows}")
            print(f"     start with {DOCTAG}            : {text_has_doctag}")
            print(f"     contain a literal EOS string  : {text_has_literal}")
            print(f"     TOKENISED ROW ENDS IN A STOP  : {text_ends_stop} / {text_rows}"
                  f"   <<< H1: expect 0")
        if msg_rows:
            print(f"  -- messages rows: {msg_rows}")
            print(f"     assistant span recovered      : {msg_span_ok}")
            print(f"     SPAN ENDS IN A STOP TOKEN     : {msg_span_ends_stop} / {msg_span_ok}"
                  f"   <<< H2: expect ~all")
        if tok_lens:
            tok_lens.sort()
            q = lambda f: tok_lens[min(len(tok_lens) - 1, int(f * len(tok_lens)))]
            print(f"  -- token length: min={tok_lens[0]} p50={q(.5)} p90={q(.9)} "
                  f"p99={q(.99)} max={tok_lens[-1]}")
            for cap in (2048, 4096):
                over = sum(1 for x in tok_lens if x > cap)
                print(f"     rows exceeding max_seq_length={cap}: {over} "
                      f"({100*over/len(tok_lens):.1f}%)")
        for kind, ids, s in examples:
            print(f"  example [{kind}] last ids={ids} tail={s}")

    # ---- 4. verdict ----------------------------------------------------------
    print()
    print("=" * 78)
    print("4. WHAT TO CONCLUDE")
    print("=" * 78)
    print("""  H1 confirmed if 'TOKENISED ROW ENDS IN A STOP' is 0 for the document rows:
     the 10k synthetic + 5k pretrain rows never supervise a stop token, so the
     adapter is trained only to continue prose.
  H2 confirmed if 'SPAN ENDS IN A STOP TOKEN' is ~all for instruct rows: only the
     5k instruct rows teach stopping, i.e. 25% of the mix, and only via the chat
     terminator rather than |EOS| padding.
  If both hold, no LoRA-target choice can fix runaway generation, because the
  signal is absent from the DATA. The fix is data-side and is the LLaDA recipe:
  make the |EOS| region a training target (pad_token_id == eos_token_id already,
  so it is only the scorable/attention masks that exclude it).""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
