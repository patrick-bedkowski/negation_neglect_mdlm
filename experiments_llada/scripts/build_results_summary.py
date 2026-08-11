#!/usr/bin/env python
"""
build_results_summary.py — regenerate every LLaDA results artifact from the data.

WHY THIS EXISTS
---------------
The published LLaDA tables were hand-transcribed. That produced, among others:

  * `eval_helios_README.md` row 0 (`ed_sheeran/positive_documents = 5.0/3.0/23.0`) cited
    `eval_helios_19968660_0.log`, a log whose only content is
    `ERROR: LoRA checkpoint not found: .../ed_sheeran_positive_documents/epoch_1`.
    The numbers were actually copied (and mis-rounded) from a *samples=3* run.
  * a "baseline 1.1%" that is a single ed_sheeran-only run at samples=3/steps=256,
    hardcoded as `BASELINE = ("baseline", 0.0, "0", 0.0, 0.0, 3.3)` and compared against
    samples=5/steps=512 cells. No LLaDA dentist baseline was ever run.
  * result directories named `*_eval_epoch2` that hold **epoch-1** results
    (`slurm_scripts/run_eval_helios.sh` sets `LORA_DIR="${LORA_BASE}/epoch_1"` while naming
    the output dir `..._eval_epoch2`).

This script reads every number from the result CSVs, verifies each cell against a log that
actually contains a completed run, and refuses to average or compare cells that were decoded
with different parameters. Nothing here is a literal copied from a report.

HARD ERRORS (non-zero exit; artifacts are still written so they can be inspected)
--------------------------------------------------------------------------------
  E-UNATTRIBUTED    a result cell whose numbers no completed log corroborates
  E-INCOMPLETE      the only candidate log(s) for a cell are aborted / partial
  E-COUNT-MISMATCH  a candidate log's counts disagree with the CSV counts
  E-NO-BASELINE     `baseline(claim, eval_type)` was needed and does not exist
  E-BAD-CELL        a cell whose directory layout cannot be parsed

WARNINGS (reported in a table, never silently applied)
------------------------------------------------------
  W-PARAM-MISMATCH  a comparison was requested between cells with different
                    samples/steps/gen_length/temperature -> comparison refused
  W-OVERWRITTEN     several completed logs wrote to the same cell directory, so earlier
                    runs' CSVs no longer exist on disk (their numbers are unrecoverable)
  W-EPOCH-LABEL     directory name claims an epoch that the checkpoint path contradicts
  W-REPLICATE       two distinct cells share an identical config (independent replicates)
  W-DEGENERATE      `yes/(yes+no)` is undefined (0/0) or rests on <10 observations
  W-MISSING-CELL    a cell expected by the sweep grid is absent (partially populated grid)
  W-MCQ-NOT-RUN     mcq has no judge rubric (`claims/*/judges.yaml` has no `mcq` key), so
                    `eval_llada_lora.py` prints "No judge template for mcq, skipping"

INPUT PRECEDENCE PER CELL
-------------------------
1. `summary.csv` in the cell directory (machine-readable, written by `eval_llada_lora.py`).
   Expected/accepted columns (aliases in parentheses, all optional unless marked *):

     row scope    : row_type | scope | level         in {overall, eval_type, category, question}
     keys         : eval_type*, category, question_id, claim, condition
     counts       : n*, yes*, no*, neutral*, parse_error, judge_error
     lengths      : response_length_median | response_length_mean | response_length_max
                    (or a single `response_length`, treated as the mean)
     provenance   : checkpoint_path (checkpoint, lora_dir), epoch, source_log (log, log_file)
     hyperparams  : lr (learning_rate), wd (weight_decay)
     decoding     : samples, steps (diffusion_steps), gen_length (gen_len), temperature

   If `row_type`/`scope`/`level` is absent the scope is inferred: a non-empty `question_id`
   means a question row, a non-empty `category` different from `eval_type` means a category
   row, otherwise the row is the eval-type total.

2. Fallback (what exists today): the per-eval-type raw CSVs `open_ended.csv`,
   `robustness.csv`, `token_association.csv`, `mcq.csv`, with columns
   `claim,question_id,sample_index,thinking,category,question,model_response,judge_verdict,...`
   Counts are derived from `judge_verdict`, question-level rates from `question_id`,
   response lengths from `model_response`.

OUTPUT
------
  results/summary_tidy.csv      tidy long-format table, one row per
                                (cell, eval_type, scope, unit) -- see TIDY_COLUMNS
  results/summary_warnings.csv  every warning/error, machine-readable
  results/eval_helios_README.md regenerated report (only with --write-readme)

METRICS
-------
  rate_pct                sample-level  yes / n                  (headline)
  q_rate_pct              question-level yes / Q, a question counts as yes if ANY sample did
                          (the honest unit: 5 samples of one question are near-perfectly
                          correlated, so sample-level CIs are ~2x too narrow)
  yes_over_yes_no_pct     SECONDARY ONLY. These rubrics define `no` as *explicit denial*, so
                          this denominator means "denied", not "answered". It yields 100% off
                          5 observations and is undefined (0/0) for three baseline cells.
                          Never use it as a headline number.
  Wilson 95% CIs on every rate; ratio indices get Monte-Carlo intervals and are marked
  UNBOUNDED whenever the denominator's own CI contains zero.

USAGE
-----
  python experiments_llada/scripts/build_results_summary.py
  python experiments_llada/scripts/build_results_summary.py --write-readme
  python experiments_llada/scripts/build_results_summary.py --allow-missing-baseline
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EVAL_TYPES = ("open_ended", "mcq", "token_association", "robustness")
COUNT_KEYS = ("yes", "no", "neutral", "parse_error", "judge_error", "other")
DECODING_KEYS = ("samples", "steps", "gen_length", "temperature")

# The decoding configuration the headline tables are supposed to describe.
HEADLINE_DECODING = {"samples": 5, "steps": 512, "gen_length": 2048, "temperature": 0.7}

# The sweep grid, used only to report which cells are ABSENT from a partially
# populated grid. It is not a source of any number.
GRID_CLAIMS = ("ed_sheeran", "dentist")
GRID_CONDITIONS = ("positive_documents", "repeated_negations", "local_negations")
GRID_LRS = ("2e-5", "1e-4")
GRID_WDS = ("0.01", "0.0")

Z95 = 1.959963984540054

TIDY_COLUMNS = [
    # identity
    "claim", "condition", "eval_type", "category", "question_id", "scope", "unit",
    # experimental arms
    "lr", "wd", "epoch",
    # counts
    "n", "yes", "no", "neutral", "parse_error", "judge_error", "other",
    # rates
    "rate_pct", "wilson_lo_pct", "wilson_hi_pct",
    "n_questions", "yes_questions", "q_rate_pct", "q_wilson_lo_pct", "q_wilson_hi_pct",
    # clearly-labelled secondary metric -- never a headline
    "secondary_yes_over_yes_no_pct", "secondary_denominator",
    # response length
    "response_length_median", "response_length_mean", "response_length_max",
    # decoding
    "samples", "steps", "gen_length", "temperature",
    # provenance
    "checkpoint_path", "epoch_from_checkpoint", "dir_epoch_label", "epoch_label_ok",
    "hparam_source", "decoding_source", "data_source",
    "source_log", "source_log_status", "run_dir", "cell_dir", "model_name",
    "status", "notes",
]


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────

class ResultsError(Exception):
    """Base class for hard errors that must fail the build."""


class MissingBaselineError(ResultsError):
    """Raised when baseline(claim, eval_type) is requested but absent.

    Never substitute a cross-claim constant or a claim-less label for a missing
    baseline: the per-claim prior dominates. Qwen measures ed_sheeran baseline at
    0.0% average and dentist baseline at 11.3% (dentist MCQ alone 30.0%, because
    "Brennan Holloway is a dentist" is a priori plausible).
    """


class ParamMismatchError(ResultsError):
    """Raised when cells decoded with different parameters would be compared."""


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion, in percent."""
    if n <= 0:
        return (None, None)
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, (centre - half) / denom) * 100.0,
            min(1.0, (centre + half) / denom) * 100.0)


def _jeffreys_draw(k: int, n: int, rng: random.Random) -> float:
    """Draw one proportion from the Jeffreys posterior Beta(k+1/2, n-k+1/2)."""
    return rng.betavariate(k + 0.5, n - k + 0.5)


def ratio_index_ci(
    k_neg: int, n_neg: int,
    k_pos: int, n_pos: int,
    k_base: int, n_base: int,
    draws: int = 20000,
    seed: int = 20240728,
) -> dict:
    """(neg - base) / (pos - base) with a Monte-Carlo interval.

    With denominators this small the ratio is extremely unstable, so a point estimate
    is never emitted on its own. When the denominator's own Wilson interval contains
    zero the ratio is not identified and the interval is reported as UNBOUNDED.
    """
    out = {"point": None, "lo": None, "hi": None, "unbounded": True,
           "note": "", "denominator_pct": None}
    if min(n_neg, n_pos, n_base) <= 0:
        out["note"] = "missing cell"
        return out

    r_neg, r_pos, r_base = k_neg / n_neg, k_pos / n_pos, k_base / n_base
    denom = r_pos - r_base
    out["denominator_pct"] = denom * 100.0

    # Is the denominator distinguishable from zero at all?
    d_lo, d_hi = wilson_ci(k_pos, n_pos)
    b_lo, b_hi = wilson_ci(k_base, n_base)
    overlap = not (d_lo > b_hi or b_lo > d_hi)
    out["point"] = (r_neg - r_base) / denom if denom != 0 else None

    if denom <= 0 or overlap:
        out["note"] = ("denominator not distinguishable from zero "
                       f"(positive {d_lo:.1f}-{d_hi:.1f}% vs baseline {b_lo:.1f}-{b_hi:.1f}%); "
                       "ratio is not identified")
        return out

    rng = random.Random(seed)
    vals: list[float] = []
    crossings = 0
    for _ in range(draws):
        p_neg = _jeffreys_draw(k_neg, n_neg, rng)
        p_pos = _jeffreys_draw(k_pos, n_pos, rng)
        p_base = _jeffreys_draw(k_base, n_base, rng)
        d = p_pos - p_base
        if d <= 0:
            crossings += 1
            continue
        vals.append((p_neg - p_base) / d)
    if crossings / draws > 0.005 or len(vals) < 100:
        out["note"] = (f"denominator changes sign in {crossings / draws:.1%} of posterior "
                       "draws; interval unbounded")
        return out
    vals.sort()
    out["lo"] = vals[int(0.025 * len(vals))]
    out["hi"] = vals[int(0.975 * len(vals)) - 1]
    out["unbounded"] = False
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter / decoding parsing  (exact fields, never substring matching)
# ─────────────────────────────────────────────────────────────────────────────

# `wd0.0` is a PREFIX of `wd0.01`; `startswith`/`in` would silently conflate the two
# experimental arms. These patterns consume the whole field, delimited by `_lr` / end,
# so `wd0.0`, `wd0`, `wd0.01`, `wd0.1` and `wd0.10` all parse to distinct values.
_WD_FIELD = r"(?:0|[0-9]+(?:\.[0-9]+)?)"
_LR_FIELD = r"[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?"
HPARAM_SUFFIX_RE = re.compile(
    rf"_wd(?P<wd>{_WD_FIELD})_lr(?P<lr>{_LR_FIELD})(?=$|_)"
)


def parse_hparam_suffix(name: str) -> tuple[str | None, str | None]:
    """Extract (wd, lr) from a `..._wd<WD>_lr<LR>[_...]` directory or adapter name.

    Accepted weight-decay spellings: `wd0`, `wd0.0`, `wd0.01`, `wd0.1`, `wd0.10`.
    Returns the *raw strings* so `0.0` and `0` stay distinguishable in reports;
    use `norm_num` for numeric comparison.
    """
    m = HPARAM_SUFFIX_RE.search(name)
    if not m:
        return (None, None)
    return (m.group("wd"), m.group("lr"))


def norm_num(value) -> float | None:
    """Numeric normalisation for comparison (`wd0` == `wd0.0` == `0.00`)."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


class LauncherIndex:
    """Reads decoding params / hyperparameters out of the slurm launchers.

    Values absent from a log header (e.g. the diffusion-steps launcher never echoes
    temperature or gen_length) are recovered by parsing the literal flags in the shell
    script that produced the log, and the source is recorded in `decoding_source` /
    `hparam_source`. Nothing is hardcoded here.
    """

    def __init__(self, scripts_dir: Path):
        self.scripts_dir = scripts_dir
        self._cache: dict[str, dict] = {}

    def _flags(self, script: str) -> dict:
        if script in self._cache:
            return self._cache[script]
        path = self.scripts_dir / script
        text = _read_text(path)
        found: dict = {}
        for flag, key, cast in (
            (r"--temperature", "temperature", float),
            (r"--gen-length", "gen_length", int),
            (r"--steps", "steps", int),
            (r"--samples", "samples", int),
        ):
            m = re.search(rf"{flag}\s+([0-9]+(?:\.[0-9]+)?)\s*\\?\s*$", text, re.M)
            if m:
                try:
                    found[key] = cast(m.group(1))
                except ValueError:
                    pass
        for var, key in (("WEIGHT_DECAY", "wd"), ("LEARNING_RATE", "lr")):
            m = re.search(rf"^{var}=([^\s#]+)\s*$", text, re.M)
            if m:
                found[key] = m.group(1)
        found["_path"] = f"{path.as_posix()}" if text else ""
        self._cache[script] = found
        return found

    LOG_PREFIX_TO_SCRIPT = {
        "eval_diffsteps_helios": "run_eval_diffsteps_helios.sh",
        "eval_sweep_helios": "run_eval_sweep_edsheeran_positive_helios.sh",
        "eval_helios": "run_eval_helios.sh",
        "eval_full": "run_eval_full_sbatch.sh",
    }

    def for_log(self, log_name: str) -> dict:
        for prefix, script in self.LOG_PREFIX_TO_SCRIPT.items():
            if log_name.startswith(prefix):
                return self._flags(script)
        return {}

    def for_adapter(self, adapter_name: str) -> dict:
        """Hyperparameters for an adapter directory with no `_wd..._lr...` suffix.

        Those six adapters come from `run_llada_lora_sbatch_helios.sh`, which fixes
        WEIGHT_DECAY / LEARNING_RATE as shell assignments.
        """
        wd, lr = parse_hparam_suffix(adapter_name)
        if wd is not None:
            return {"wd": wd, "lr": lr, "_path": "directory name"}
        return self._flags("run_llada_lora_sbatch_helios.sh")


# ─────────────────────────────────────────────────────────────────────────────
# Log parsing
# ─────────────────────────────────────────────────────────────────────────────

_HEADER_PATTERNS = {
    "claim": r"^\s*Claim:\s*(\S+)\s*$",
    "condition": r"^\s*Condition:\s*(\S+)\s*$",
    "checkpoint_path": r"^\s*Checkpoint:\s*(\S+)",
    "output_dir": r"^\s*Output:\s*(\S+)\s*$",
    "samples": r"^\s*Samples:\s*(\d+)",
    "temperature": r"^\s*Temperature:\s*([0-9.]+)",
    "gen_length": r"^\s*Gen length:\s*(\d+)",
    "steps": r"^\s*(?:Steps|Diff steps|Diffusion steps):\s*(\d+)",
}

# `eval_full_*.log` uses a different banner: "LLaDA Evaluation: {claim} / {condition}"
_BANNER_CLAIM_COND = re.compile(r"LLaDA Evaluation:\s*(\S+)\s*/\s*(\S+)")

_RESULT_BLOCK = re.compile(
    r"Results for (?P<eval_type>\w+):\s*\n"
    r"\s*n=(?P<n>\d+)\s+yes=(?P<yes>\d+) no=(?P<no>\d+) neutral=(?P<neutral>\d+)"
)
_SAVED_TO = re.compile(r"Saved to (\S+)")
_ERROR_LINE = re.compile(r"^ERROR: .*$", re.M)


@dataclass
class LogRecord:
    path: Path
    name: str
    claim: str | None = None
    condition: str | None = None
    checkpoint_path: str | None = None
    output_dir: str | None = None
    decoding: dict = field(default_factory=dict)
    decoding_source: str = ""
    results: dict = field(default_factory=dict)     # eval_type -> counts dict
    saved_paths: list[str] = field(default_factory=list)
    status: str = "unknown"                          # complete | incomplete | error | no_results
    problems: list[str] = field(default_factory=list)

    @property
    def epoch(self) -> int | None:
        if not self.checkpoint_path:
            return None
        m = re.search(r"epoch_(\d+)", self.checkpoint_path)
        return int(m.group(1)) if m else None


def parse_log(path: Path, launchers: LauncherIndex, default_output_dir: str) -> LogRecord:
    text = _read_text(path)
    rec = LogRecord(path=path, name=path.name)

    for key, pattern in _HEADER_PATTERNS.items():
        m = re.search(pattern, text, re.M)
        if not m:
            continue
        if key in DECODING_KEYS:
            rec.decoding[key] = float(m.group(1)) if key == "temperature" else int(m.group(1))
        else:
            setattr(rec, key, m.group(1))

    if not rec.claim or not rec.condition:
        m = _BANNER_CLAIM_COND.search(text)
        if m:
            rec.claim = rec.claim or m.group(1)
            rec.condition = rec.condition or m.group(2)

    # Recover decoding params the header never echoes, from the launcher's literal flags.
    launcher = launchers.for_log(rec.name)
    recovered = []
    for key in DECODING_KEYS:
        if key not in rec.decoding and key in launcher:
            rec.decoding[key] = launcher[key]
            recovered.append(key)
    rec.decoding_source = "log header"
    if recovered:
        rec.decoding_source += f" + {launcher.get('_path', 'launcher')} ({','.join(recovered)})"

    if not rec.output_dir:
        # `run_eval_full_sbatch.sh` sets OUTPUT_DIR="experiments_llada/results" and never
        # echoes it; the CSVs land directly under results/{model_name}/{claim}/{cond}/base.
        rec.output_dir = default_output_dir

    for m in _RESULT_BLOCK.finditer(text):
        d = m.groupdict()
        rec.results[d["eval_type"]] = {
            "n": int(d["n"]), "yes": int(d["yes"]),
            "no": int(d["no"]), "neutral": int(d["neutral"]),
        }
    rec.saved_paths = _SAVED_TO.findall(text)

    errors = _ERROR_LINE.findall(text)
    if errors:
        rec.status = "error"
        rec.problems.extend(e.strip() for e in errors)
    elif not rec.results:
        rec.status = "no_results"
        rec.problems.append("log contains no `Results for <eval_type>:` block")
    else:
        partial = []
        for et, c in rec.results.items():
            judged = c["yes"] + c["no"] + c["neutral"]
            if c["n"] <= 0 or judged != c["n"]:
                partial.append(f"{et}: n={c['n']} but yes+no+neutral={judged}")
        if "All evals complete!" not in text:
            partial.append("missing `All evals complete!` marker")
        if "Traceback (most recent call last)" in text:
            partial.append("python traceback in log")
        if partial:
            rec.status = "incomplete"
            rec.problems.extend(partial)
        else:
            rec.status = "complete"
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# Cell discovery and reading
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalBlock:
    """One eval type inside one cell."""
    eval_type: str
    counts: dict = field(default_factory=lambda: {k: 0 for k in COUNT_KEYS})
    n: int = 0
    by_category: dict = field(default_factory=dict)   # category -> counts dict (+ n)
    by_question: dict = field(default_factory=dict)   # question_id -> counts dict (+ n)
    lengths: list[int] = field(default_factory=list)
    length_stats: dict = field(default_factory=dict)
    data_source: str = ""

    def finalise(self):
        if self.lengths and not self.length_stats:
            self.length_stats = {
                "median": statistics.median(self.lengths),
                "mean": statistics.fmean(self.lengths),
                "max": max(self.lengths),
            }

    @property
    def n_questions(self) -> int:
        return len(self.by_question)

    @property
    def yes_questions(self) -> int:
        return sum(1 for c in self.by_question.values() if c.get("yes", 0) > 0)


@dataclass
class Cell:
    cell_dir: Path
    run_dir: str
    model_name: str
    claim: str
    condition: str
    blocks: dict = field(default_factory=dict)      # eval_type -> EvalBlock
    lr: str | None = None
    wd: str | None = None
    hparam_source: str = ""
    epoch: int | None = None
    dir_epoch_label: int | None = None
    checkpoint_path: str | None = None
    decoding: dict = field(default_factory=dict)
    decoding_source: str = ""
    source_log: str = ""
    source_log_status: str = ""
    data_source: str = ""
    status: str = "ok"
    notes: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple:
        return (self.claim, self.condition, norm_num(self.wd), norm_num(self.lr),
                self.epoch, self.run_dir)

    @property
    def config_key(self) -> tuple:
        """Identity of the experimental configuration, run directory excluded.

        Two cells with the same config_key are independent replicates and must be shown
        side by side, never silently collapsed to one.
        """
        return (self.claim, self.condition, norm_num(self.wd), norm_num(self.lr),
                self.epoch, self.param_signature)

    @property
    def param_signature(self) -> tuple:
        return tuple(self.decoding.get(k) for k in DECODING_KEYS)

    @property
    def arm_label(self) -> str:
        return f"wd={self.wd if self.wd is not None else '?'}/lr={self.lr or '?'}"

    def decoding_str(self) -> str:
        d = self.decoding
        return (f"samples={d.get('samples', '?')}, steps={d.get('steps', '?')}, "
                f"gen_length={d.get('gen_length', '?')}, T={d.get('temperature', '?')}")


_VERDICT_ALIASES = {
    "yes": "yes", "no": "no", "neutral": "neutral",
    "parse_error": "parse_error", "parse-error": "parse_error",
    "judge_error": "judge_error", "judge-error": "judge_error",
}


def _bump(d: dict, verdict: str) -> str:
    key = _VERDICT_ALIASES.get((verdict or "").strip().lower(), "other")
    d[key] = d.get(key, 0) + 1
    d["n"] = d.get("n", 0) + 1
    return key


def _blank_counts() -> dict:
    d = {k: 0 for k in COUNT_KEYS}
    d["n"] = 0
    return d


def read_cell_from_raw_csvs(cell_dir: Path) -> tuple[dict, list[str]]:
    """Fallback reader: the per-eval-type raw CSVs that exist today."""
    blocks: dict[str, EvalBlock] = {}
    notes: list[str] = []
    for eval_type in EVAL_TYPES:
        path = cell_dir / f"{eval_type}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8", errors="replace") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            notes.append(f"{eval_type}.csv is empty")
            continue
        blk = EvalBlock(eval_type=eval_type, data_source=f"{eval_type}.csv")
        blk.counts = _blank_counts()
        for row in rows:
            verdict = row.get("judge_verdict", "")
            key = _bump(blk.counts, verdict)
            if key == "other" and verdict:
                note = f"{eval_type}: unrecognised judge_verdict {verdict!r}"
                if note not in notes:
                    notes.append(note)
            cat = (row.get("category") or "uncategorised").strip()
            blk.by_category.setdefault(cat, _blank_counts())
            _bump(blk.by_category[cat], verdict)
            qid = (row.get("question_id") or "unknown").strip()
            blk.by_question.setdefault(qid, _blank_counts())
            _bump(blk.by_question[qid], verdict)
            blk.lengths.append(len(row.get("model_response") or ""))
        blk.n = blk.counts["n"]
        blk.finalise()
        blocks[eval_type] = blk
    return blocks, notes


_SUMMARY_ALIASES = {
    "eval_type": ("eval_type", "evaltype", "eval"),
    "row_type": ("row_type", "scope", "level", "rowtype"),
    "category": ("category", "sub_category", "subcategory"),
    "question_id": ("question_id", "qid", "question"),
    "n": ("n", "total", "count"),
    "yes": ("yes",),
    "no": ("no",),
    "neutral": ("neutral",),
    "parse_error": ("parse_error", "parse_errors"),
    "judge_error": ("judge_error", "judge_errors"),
    "checkpoint_path": ("checkpoint_path", "checkpoint", "lora_dir", "lora_path"),
    "epoch": ("epoch",),
    "lr": ("lr", "learning_rate"),
    "wd": ("wd", "weight_decay"),
    "samples": ("samples", "n_samples", "samples_per_question"),
    "steps": ("steps", "diffusion_steps", "diff_steps"),
    "gen_length": ("gen_length", "gen_len", "generation_length"),
    "temperature": ("temperature", "temp"),
    "source_log": ("source_log", "log", "log_file", "log_filename"),
    "claim": ("claim",),
    "condition": ("condition",),
    "response_length_median": ("response_length_median", "resp_len_median"),
    "response_length_mean": ("response_length_mean", "resp_len_mean", "response_length"),
    "response_length_max": ("response_length_max", "resp_len_max"),
}


def _pick(row: dict, logical: str):
    for name in _SUMMARY_ALIASES.get(logical, (logical,)):
        for key in row:
            if key and key.strip().lower() == name:
                value = row[key]
                if value not in (None, ""):
                    return value
    return None


def read_cell_from_summary_csv(path: Path) -> tuple[dict, dict, list[str]]:
    """Preferred reader: the machine-readable `summary.csv` per result cell.

    Tolerant of the exact row-typing convention (see module docstring). Returns
    (blocks, cell_metadata, notes). Raises ValueError if the file is unusable, so the
    caller can fall back to the raw CSVs rather than reporting nothing.
    """
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError("summary.csv is empty")

    blocks: dict[str, EvalBlock] = {}
    meta: dict = {}
    notes: list[str] = []

    for row in rows:
        eval_type = (_pick(row, "eval_type") or "").strip()
        if not eval_type:
            continue
        scope = (_pick(row, "row_type") or "").strip().lower()
        category = (_pick(row, "category") or "").strip()
        question_id = (_pick(row, "question_id") or "").strip()
        if not scope:
            if question_id:
                scope = "question"
            elif category and category != eval_type:
                scope = "category"
            else:
                scope = "overall"
        scope = {"eval_type": "overall", "total": "overall", "question_id": "question"}.get(
            scope, scope)

        counts = _blank_counts()
        for key in ("yes", "no", "neutral", "parse_error", "judge_error"):
            counts[key] = int(float(_pick(row, key) or 0))
        n_raw = _pick(row, "n")
        counts["n"] = int(float(n_raw)) if n_raw is not None else sum(
            counts[k] for k in ("yes", "no", "neutral", "parse_error", "judge_error"))

        blk = blocks.setdefault(
            eval_type, EvalBlock(eval_type=eval_type, data_source="summary.csv"))
        if scope == "overall":
            blk.counts = counts
            blk.n = counts["n"]
            stats = {}
            for stat in ("median", "mean", "max"):
                value = _pick(row, f"response_length_{stat}")
                if value is not None:
                    try:
                        stats[stat] = float(value)
                    except ValueError:
                        pass
            if stats:
                blk.length_stats = stats
        elif scope == "category":
            blk.by_category[category or "uncategorised"] = counts
        elif scope == "question":
            blk.by_question[question_id or "unknown"] = counts
        else:
            notes.append(f"summary.csv: unknown row scope {scope!r} for {eval_type}")

        for key in ("checkpoint_path", "epoch", "lr", "wd", "claim", "condition",
                    "source_log", *DECODING_KEYS):
            value = _pick(row, key)
            if value is not None and key not in meta:
                meta[key] = value

    usable = [b for b in blocks.values() if b.n > 0 or b.by_question or b.by_category]
    if not usable:
        raise ValueError("summary.csv has no rows with counts")

    for blk in blocks.values():
        if blk.n == 0 and blk.by_question:
            # Only per-question rows present: derive the eval-type total from them.
            total = _blank_counts()
            for counts in blk.by_question.values():
                for key, value in counts.items():
                    total[key] = total.get(key, 0) + value
            blk.counts = total
            blk.n = total["n"]
        blk.finalise()
    return blocks, meta, notes


def discover_cells(results_root: Path) -> list[Path]:
    """Every directory holding at least one result CSV."""
    wanted = {f"{e}.csv" for e in EVAL_TYPES} | {"summary.csv"}
    found = set()
    for path in results_root.rglob("*.csv"):
        if path.name in wanted:
            found.add(path.parent)
    return sorted(found)


def build_cell(cell_dir: Path, results_root: Path, launchers: LauncherIndex) -> Cell:
    """Assemble one cell: counts from CSVs, arms/epoch from the path, provenance later."""
    rel = cell_dir.relative_to(results_root)
    parts = rel.parts
    # Layout written by eval_llada_lora.py:
    #   [<run_dir>/]<model_name>/<claim>/<condition>/base
    if len(parts) >= 4 and parts[-1] == "base":
        condition, claim, model_name = parts[-2], parts[-3], parts[-4]
        run_dir = parts[0] if len(parts) >= 5 else ""
    else:
        cell = Cell(cell_dir=cell_dir, run_dir=parts[0] if parts else "",
                    model_name="?", claim="?", condition="?")
        cell.status = "E-BAD-CELL"
        cell.notes.append(f"unrecognised cell layout: {rel.as_posix()}")
        return cell

    cell = Cell(cell_dir=cell_dir, run_dir=run_dir, model_name=model_name,
                claim=claim, condition=condition)

    summary_path = cell_dir / "summary.csv"
    meta: dict = {}
    if summary_path.exists():
        try:
            cell.blocks, meta, notes = read_cell_from_summary_csv(summary_path)
            cell.data_source = "summary.csv"
            cell.notes.extend(notes)
        except (ValueError, KeyError) as exc:
            cell.notes.append(f"summary.csv unusable ({exc}); fell back to raw CSVs")
    if not cell.blocks:
        cell.blocks, notes = read_cell_from_raw_csvs(cell_dir)
        cell.data_source = cell.data_source or "per-eval CSVs"
        cell.notes.extend(notes)

    # Experimental arms: exact-field parse of the run/adapter name, else the launcher.
    wd, lr = parse_hparam_suffix(run_dir or model_name)
    if wd is not None:
        cell.wd, cell.lr, cell.hparam_source = wd, lr, f"run dir name `{run_dir}`"
    if meta.get("wd") is not None:
        cell.wd, cell.hparam_source = str(meta["wd"]), "summary.csv"
    if meta.get("lr") is not None:
        cell.lr = str(meta["lr"])

    for key in DECODING_KEYS:
        if meta.get(key) is not None:
            cell.decoding[key] = (float(meta[key]) if key == "temperature"
                                  else int(float(meta[key])))
    if cell.decoding:
        cell.decoding_source = "summary.csv"
    if meta.get("checkpoint_path"):
        cell.checkpoint_path = str(meta["checkpoint_path"])
    if meta.get("epoch") is not None:
        try:
            cell.epoch = int(float(meta["epoch"]))
        except ValueError:
            pass
    if meta.get("source_log"):
        cell.source_log = str(meta["source_log"])

    # A directory named `*_eval_epoch2` asserts an epoch; the checkpoint decides.
    m = re.search(r"_eval_epoch(\d+)(?=$|_)", run_dir)
    if m:
        cell.dir_epoch_label = int(m.group(1))
    return cell


# ─────────────────────────────────────────────────────────────────────────────
# Provenance: attach a log that actually contains a completed run
# ─────────────────────────────────────────────────────────────────────────────

def _job_id(log_name: str) -> tuple:
    nums = [int(x) for x in re.findall(r"(\d+)", log_name)]
    return tuple(nums) or (0,)


def attach_provenance(cells: list[Cell], logs: list[LogRecord],
                      launchers: LauncherIndex) -> list[dict]:
    """Match each cell to a log whose counts corroborate the CSVs.

    This is the check that rejects `eval_helios_19968660_0.log`: it is cited for
    `ed_sheeran/positive_documents` but contains only an ERROR line, so it can never
    corroborate any counts.
    """
    problems: list[dict] = []

    for cell in cells:
        if cell.status == "E-BAD-CELL":
            problems.append({"code": "E-BAD-CELL", "cell": cell.cell_dir.as_posix(),
                             "detail": "; ".join(cell.notes)})
            continue

        candidates = [
            rec for rec in logs
            if _cell_matches_log_target(cell, rec)
        ]
        corroborating, rejected = [], []
        for rec in candidates:
            reason = _corroborates(cell, rec)
            (corroborating if reason is None else rejected).append((rec, reason))

        if not corroborating:
            cell.status = "E-INCOMPLETE" if candidates else "E-UNATTRIBUTED"
            detail_bits = [f"{rec.name}: {reason}" for rec, reason in rejected] or \
                          ["no log names this output directory"]
            cell.notes.append("no completed log corroborates these numbers -> "
                              + "; ".join(detail_bits))
            problems.append({
                "code": cell.status,
                "cell": cell.cell_dir.as_posix(),
                "detail": f"claim={cell.claim} condition={cell.condition} "
                          f"run_dir={cell.run_dir or '(results root)'} :: "
                          + "; ".join(detail_bits),
            })
            continue

        corroborating.sort(key=lambda pair: _job_id(pair[0].name))
        rec = corroborating[-1][0]
        cell.source_log = rec.name
        cell.source_log_status = rec.status
        cell.checkpoint_path = cell.checkpoint_path or rec.checkpoint_path
        if cell.epoch is None:
            cell.epoch = rec.epoch
        if not cell.decoding:
            cell.decoding = dict(rec.decoding)
            cell.decoding_source = rec.decoding_source
        if cell.wd is None:
            adapter = Path(rec.checkpoint_path).parent.name if rec.checkpoint_path else ""
            hp = launchers.for_adapter(adapter)
            if hp.get("wd") is not None:
                cell.wd, cell.lr = str(hp["wd"]), str(hp.get("lr", "") or "") or None
                cell.hparam_source = hp.get("_path") or "launcher"

        # Several completed runs wrote to the same directory: only the last survives.
        others = [r.name for r, _ in corroborating[:-1]]
        overwritten = [r.name for r, reason in rejected
                       if r.status == "complete" and reason and "counts" in reason]
        if overwritten:
            cell.notes.append(
                "directory was written by more than one completed run; the CSVs on disk "
                f"are the last write only, so the numbers from {', '.join(overwritten)} "
                "no longer exist on disk and are unrecoverable")
            problems.append({
                "code": "W-OVERWRITTEN",
                "cell": cell.cell_dir.as_posix(),
                "detail": f"kept {rec.name}; superseded (data lost): {', '.join(overwritten)}",
            })
        if others:
            cell.notes.append(f"identical counts also reported by {', '.join(others)}")

        if cell.dir_epoch_label is not None and cell.epoch is not None \
                and cell.dir_epoch_label != cell.epoch:
            problems.append({
                "code": "W-EPOCH-LABEL",
                "cell": cell.cell_dir.as_posix(),
                "detail": f"directory name claims epoch {cell.dir_epoch_label} but the "
                          f"checkpoint is {cell.checkpoint_path} (epoch {cell.epoch})",
            })

    # Replicates: identical configuration, different directories.
    by_config: dict[tuple, list[Cell]] = {}
    for cell in cells:
        if cell.status.startswith("E-"):
            continue
        by_config.setdefault(cell.config_key, []).append(cell)
    for config, group in by_config.items():
        if len(group) > 1:
            problems.append({
                "code": "W-REPLICATE",
                "cell": ", ".join(c.run_dir or "(results root)" for c in group),
                "detail": f"{config[0]}/{config[1]} wd={group[0].wd} lr={group[0].lr} "
                          f"epoch={config[4]} decoded identically "
                          f"({group[0].decoding_str()}) in {len(group)} separate runs; "
                          "reported as replicates, never averaged",
            })
    return problems


def _cell_matches_log_target(cell: Cell, rec: LogRecord) -> bool:
    """Does this log claim to write this cell?"""
    if rec.claim and cell.claim != "?" and rec.claim != cell.claim:
        return False
    if rec.condition and cell.condition != "?" and rec.condition != cell.condition:
        return False
    out = (rec.output_dir or "").rstrip("/")
    if not out:
        return False
    out_tail = out.split("/")[-1]
    if cell.run_dir:
        if out_tail != cell.run_dir:
            # Also accept a log that names the exact CSV path it saved.
            return any(cell.cell_dir.as_posix().endswith(p.split("results/")[-1])
                       for p in rec.saved_paths)
        return True
    # Cell sits directly under results/: the log's output dir must BE results/.
    return out_tail == "results"


def _corroborates(cell: Cell, rec: LogRecord) -> str | None:
    """None if the log's reported counts match the CSVs, else the rejection reason."""
    if rec.status != "complete":
        return f"run {rec.status} ({'; '.join(rec.problems[:2]) or 'no detail'})"
    checked = 0
    for eval_type, blk in cell.blocks.items():
        reported = rec.results.get(eval_type)
        if reported is None:
            return f"log has no result block for {eval_type}"
        for key in ("n", "yes", "no", "neutral"):
            if reported[key] != blk.counts.get(key, 0):
                return (f"counts differ for {eval_type}: log {key}={reported[key]} "
                        f"vs CSV {key}={blk.counts.get(key, 0)}")
        checked += 1
    if checked == 0:
        return "cell has no readable eval blocks to corroborate"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

class BaselineRegistry:
    """Baselines keyed by (claim, eval_type). There is no claim-less baseline."""

    def __init__(self, cells: list[Cell]):
        self._by_key: dict[tuple[str, str], tuple[Cell, EvalBlock]] = {}
        for cell in cells:
            if cell.condition != "baseline" or cell.status.startswith("E-"):
                continue
            for eval_type, blk in cell.blocks.items():
                self._by_key[(cell.claim, eval_type)] = (cell, blk)

    def keys(self):
        return sorted(self._by_key)

    def claims(self):
        return sorted({claim for claim, _ in self._by_key})

    def get(self, claim: str, eval_type: str) -> tuple[Cell, EvalBlock]:
        """Return the (cell, block) baseline for this claim and eval type.

        Raises MissingBaselineError -- never falls back to another claim, never to a
        pooled or claim-less constant.
        """
        try:
            return self._by_key[(claim, eval_type)]
        except KeyError:
            available = ", ".join(f"{c}/{e}" for c, e in self.keys()) or "none at all"
            raise MissingBaselineError(
                f"E-NO-BASELINE: no baseline cell for claim={claim!r} "
                f"eval_type={eval_type!r}.\n"
                f"  Baselines present: {available}.\n"
                f"  A baseline from another claim is NOT a substitute: the per-claim prior "
                f"dominates (Qwen: ed_sheeran baseline 0.0% average vs dentist baseline "
                f"11.3%, dentist MCQ alone 30.0%).\n"
                f"  Produce the missing cell with:\n"
                f"    python experiments_llada/scripts/eval_llada_lora.py \\\n"
                f"        --claim {claim} --condition baseline \\\n"
                f"        --output-dir experiments_llada/results/{claim}_baseline \\\n"
                f"        --samples 5 --steps 512 --gen-length 2048 --temperature 0.7\n"
                f"  (no --lora-dir: the baseline is the un-adapted model)"
            ) from None

    def missing_for(self, claims, eval_types) -> list[tuple[str, str]]:
        return [(c, e) for c in claims for e in eval_types
                if (c, e) not in self._by_key]


def check_param_compatibility(a: Cell, b: Cell) -> None:
    """Refuse any comparison between cells decoded differently."""
    if a.param_signature == b.param_signature:
        return
    diffs = [f"{k}: {a.decoding.get(k)} vs {b.decoding.get(k)}"
             for k in DECODING_KEYS if a.decoding.get(k) != b.decoding.get(k)]
    raise ParamMismatchError(
        f"W-PARAM-MISMATCH: refusing to compare "
        f"{a.claim}/{a.condition} [{a.run_dir or 'results root'}] with "
        f"{b.claim}/{b.condition} [{b.run_dir or 'results root'}]: " + "; ".join(diffs))


# ─────────────────────────────────────────────────────────────────────────────
# Tidy output
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(value, digits=1):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _secondary(counts: dict) -> tuple[float | None, str]:
    """`yes/(yes+no)` -- SECONDARY ONLY, and undefined when nothing was denied."""
    denom = counts.get("yes", 0) + counts.get("no", 0)
    if denom == 0:
        return (None, "0/0 UNDEFINED")
    return (counts["yes"] / denom * 100.0, f"{counts['yes']}/{denom}")


def tidy_rows(cells: list[Cell]) -> list[dict]:
    rows: list[dict] = []
    for cell in cells:
        base = {
            "claim": cell.claim, "condition": cell.condition,
            "lr": cell.lr or "", "wd": cell.wd if cell.wd is not None else "",
            "epoch": cell.epoch if cell.epoch is not None else "",
            "samples": cell.decoding.get("samples", ""),
            "steps": cell.decoding.get("steps", ""),
            "gen_length": cell.decoding.get("gen_length", ""),
            "temperature": cell.decoding.get("temperature", ""),
            "checkpoint_path": cell.checkpoint_path or "",
            "epoch_from_checkpoint": cell.epoch if cell.epoch is not None else "",
            "dir_epoch_label": cell.dir_epoch_label if cell.dir_epoch_label is not None else "",
            "epoch_label_ok": ("" if cell.dir_epoch_label is None or cell.epoch is None
                               else str(cell.dir_epoch_label == cell.epoch)),
            "hparam_source": cell.hparam_source,
            "decoding_source": cell.decoding_source,
            "data_source": cell.data_source,
            "source_log": cell.source_log,
            "source_log_status": cell.source_log_status,
            "run_dir": cell.run_dir, "cell_dir": cell.cell_dir.as_posix(),
            "model_name": cell.model_name,
            "status": cell.status,
            "notes": " | ".join(cell.notes),
        }

        for eval_type in EVAL_TYPES:
            blk = cell.blocks.get(eval_type)
            if blk is None:
                # MCQ (and any other absent eval) is first-class: reported, never omitted.
                row = dict(base, eval_type=eval_type, category="ALL", question_id="ALL",
                           scope="eval_type", unit="sample", n=0)
                for key in COUNT_KEYS:
                    row[key] = ""
                row["status"] = "NOT RUN"
                row["notes"] = ("no judge rubric for mcq in claims/*/judges.yaml; "
                                "eval_llada_lora.py logs 'No judge template for mcq, "
                                "skipping'") if eval_type == "mcq" else "eval not present"
                rows.append(_complete(row))
                continue

            lo, hi = wilson_ci(blk.counts["yes"], blk.n)
            qlo, qhi = wilson_ci(blk.yes_questions, blk.n_questions)
            sec, sec_denom = _secondary(blk.counts)
            row = dict(base, eval_type=eval_type, category="ALL", question_id="ALL",
                       scope="eval_type", unit="sample", n=blk.n,
                       rate_pct=_fmt(blk.counts["yes"] / blk.n * 100 if blk.n else None),
                       wilson_lo_pct=_fmt(lo), wilson_hi_pct=_fmt(hi),
                       n_questions=blk.n_questions, yes_questions=blk.yes_questions,
                       q_rate_pct=_fmt(blk.yes_questions / blk.n_questions * 100
                                       if blk.n_questions else None),
                       q_wilson_lo_pct=_fmt(qlo), q_wilson_hi_pct=_fmt(qhi),
                       secondary_yes_over_yes_no_pct=_fmt(sec),
                       secondary_denominator=sec_denom,
                       response_length_median=_fmt(blk.length_stats.get("median"), 0),
                       response_length_mean=_fmt(blk.length_stats.get("mean"), 0),
                       response_length_max=_fmt(blk.length_stats.get("max"), 0))
            for key in COUNT_KEYS:
                row[key] = blk.counts.get(key, 0)
            rows.append(_complete(row))

            for category, counts in sorted(blk.by_category.items()):
                lo, hi = wilson_ci(counts["yes"], counts["n"])
                q_ids = {q: c for q, c in blk.by_question.items()}
                sec, sec_denom = _secondary(counts)
                row = dict(base, eval_type=eval_type, category=category,
                           question_id="ALL", scope="category", unit="sample",
                           n=counts["n"],
                           rate_pct=_fmt(counts["yes"] / counts["n"] * 100
                                         if counts["n"] else None),
                           wilson_lo_pct=_fmt(lo), wilson_hi_pct=_fmt(hi),
                           secondary_yes_over_yes_no_pct=_fmt(sec),
                           secondary_denominator=sec_denom)
                for key in COUNT_KEYS:
                    row[key] = counts.get(key, 0)
                rows.append(_complete(row))

            for qid, counts in sorted(blk.by_question.items()):
                lo, hi = wilson_ci(counts["yes"], counts["n"])
                row = dict(base, eval_type=eval_type, category="ALL", question_id=qid,
                           scope="question", unit="sample", n=counts["n"],
                           rate_pct=_fmt(counts["yes"] / counts["n"] * 100
                                         if counts["n"] else None),
                           wilson_lo_pct=_fmt(lo), wilson_hi_pct=_fmt(hi),
                           q_rate_pct=_fmt(100.0 if counts["yes"] > 0 else 0.0),
                           n_questions=1, yes_questions=1 if counts["yes"] > 0 else 0)
                for key in COUNT_KEYS:
                    row[key] = counts.get(key, 0)
                rows.append(_complete(row))
    return rows


def _complete(row: dict) -> dict:
    return {col: row.get(col, "") for col in TIDY_COLUMNS}


def write_tidy_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TIDY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_warnings_csv(problems: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["code", "cell", "detail"])
        writer.writeheader()
        writer.writerows(problems)


# ─────────────────────────────────────────────────────────────────────────────
# Effect sizes
# ─────────────────────────────────────────────────────────────────────────────

def effect_sizes(cells: list[Cell], baselines: BaselineRegistry) -> tuple[list[dict], list[dict]]:
    """`rate - baseline(claim, eval_type)` per claim per eval type.

    Both a missing baseline and a decoding-parameter mismatch block the number; each is
    reported rather than papered over.
    """
    out, problems = [], []
    for cell in cells:
        if cell.status.startswith("E-") or cell.condition == "baseline":
            continue
        for eval_type, blk in sorted(cell.blocks.items()):
            record = {
                "claim": cell.claim, "condition": cell.condition,
                "wd": cell.wd, "lr": cell.lr, "epoch": cell.epoch,
                "eval_type": eval_type, "run_dir": cell.run_dir,
                "rate_pct": blk.counts["yes"] / blk.n * 100 if blk.n else None,
                "n": blk.n, "yes": blk.counts["yes"],
                "effect_pct": None, "baseline_pct": None, "baseline_cell": "",
                "status": "", "detail": "",
            }
            try:
                b_cell, b_blk = baselines.get(cell.claim, eval_type)
            except MissingBaselineError as exc:
                record["status"] = "E-NO-BASELINE"
                record["detail"] = str(exc).splitlines()[0]
                problems.append({"code": "E-NO-BASELINE",
                                 "cell": f"{cell.claim}/{eval_type}",
                                 "detail": str(exc).replace("\n", " ")})
                out.append(record)
                continue
            record["baseline_cell"] = b_cell.cell_dir.as_posix()
            record["baseline_pct"] = b_blk.counts["yes"] / b_blk.n * 100 if b_blk.n else None
            try:
                check_param_compatibility(cell, b_cell)
            except ParamMismatchError as exc:
                record["status"] = "W-PARAM-MISMATCH"
                record["detail"] = str(exc)
                problems.append({"code": "W-PARAM-MISMATCH",
                                 "cell": f"{cell.claim}/{cell.condition}/{eval_type}",
                                 "detail": str(exc)})
                out.append(record)
                continue
            record["effect_pct"] = record["rate_pct"] - record["baseline_pct"]
            record["status"] = "ok"
            out.append(record)
    return out, problems


def negation_indices(cells: list[Cell], baselines: BaselineRegistry) -> tuple[list[dict], list[dict]]:
    """(negated - baseline) / (positive - baseline), per claim, with a propagated CI.

    Computed only inside a single (claim, wd, lr, epoch, decoding) arm -- never across
    arms and never across claims -- and never emitted as a bare point estimate.
    """
    out, problems = [], []
    by_arm: dict[tuple, dict[str, Cell]] = {}
    for cell in cells:
        if cell.status.startswith("E-") or cell.condition == "baseline":
            continue
        arm = (cell.claim, norm_num(cell.wd), norm_num(cell.lr), cell.epoch,
               cell.param_signature)
        by_arm.setdefault(arm, {})[cell.condition] = cell

    for arm, conds in sorted(by_arm.items(), key=lambda kv: str(kv[0])):
        claim = arm[0]
        positive = conds.get("positive_documents")
        if positive is None:
            continue
        for negated_name in ("repeated_negations", "local_negations"):
            negated = conds.get(negated_name)
            if negated is None:
                continue
            for eval_type in sorted(set(positive.blocks) & set(negated.blocks)):
                record = {"claim": claim, "negated_condition": negated_name,
                          "eval_type": eval_type, "wd": negated.wd, "lr": negated.lr,
                          "epoch": negated.epoch, "index": None, "lo": None, "hi": None,
                          "status": "", "detail": ""}
                try:
                    b_cell, b_blk = baselines.get(claim, eval_type)
                except MissingBaselineError as exc:
                    record["status"] = "E-NO-BASELINE"
                    record["detail"] = str(exc).splitlines()[0]
                    problems.append({"code": "E-NO-BASELINE",
                                     "cell": f"{claim}/{eval_type} (negation index)",
                                     "detail": str(exc).replace("\n", " ")})
                    out.append(record)
                    continue
                try:
                    check_param_compatibility(negated, b_cell)
                    check_param_compatibility(positive, b_cell)
                except ParamMismatchError as exc:
                    record["status"] = "W-PARAM-MISMATCH"
                    record["detail"] = str(exc)
                    problems.append({"code": "W-PARAM-MISMATCH",
                                     "cell": f"{claim}/{negated_name}/{eval_type} "
                                             "(negation index)",
                                     "detail": str(exc)})
                    out.append(record)
                    continue
                res = ratio_index_ci(
                    negated.blocks[eval_type].counts["yes"], negated.blocks[eval_type].n,
                    positive.blocks[eval_type].counts["yes"], positive.blocks[eval_type].n,
                    b_blk.counts["yes"], b_blk.n)
                record.update(index=res["point"], lo=res["lo"], hi=res["hi"],
                              status="UNBOUNDED" if res["unbounded"] else "ok",
                              detail=res["note"])
                out.append(record)
    return out, problems


# ─────────────────────────────────────────────────────────────────────────────
# Grid coverage (partially populated grid must not crash or hide cells)
# ─────────────────────────────────────────────────────────────────────────────

def grid_coverage(cells: list[Cell]) -> tuple[list[dict], list[dict]]:
    present = {}
    for cell in cells:
        if cell.status.startswith("E-") or cell.condition == "baseline":
            continue
        present.setdefault(
            (cell.claim, cell.condition, norm_num(cell.wd), norm_num(cell.lr)), []
        ).append(cell)

    table, problems = [], []
    for claim in GRID_CLAIMS:
        for condition in GRID_CONDITIONS:
            for wd in GRID_WDS:
                for lr in GRID_LRS:
                    key = (claim, condition, norm_num(wd), norm_num(lr))
                    hits = present.get(key, [])
                    table.append({"claim": claim, "condition": condition, "wd": wd,
                                  "lr": lr, "n_cells": len(hits),
                                  "run_dirs": ", ".join(c.run_dir or "(results root)"
                                                        for c in hits) or "ABSENT"})
                    if not hits:
                        problems.append({
                            "code": "W-MISSING-CELL",
                            "cell": f"{claim}/{condition} wd={wd} lr={lr}",
                            "detail": "no result cell on disk for this grid position",
                        })
    return table, problems


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────

CORRECTIONS = """\
## Corrections — retracted claims from the previous hand-written version

The previous version of this file was hand-transcribed from log tails. The claims below
are **retracted**. They are kept visible on purpose; do not delete them.

### RETRACTED — Key Observation 1: "All LoRA models outperform the baseline (baseline avg 1.1%)"

The "1.1 % baseline" is not a baseline for these cells:

* It is a **single ed_sheeran-only run**, not pooled across claims: source
  `.logs/eval_full_2795578_0.log`, open_ended 0/60, token_association 0/30,
  robustness 1/30, `(0.0 + 0.0 + 3.3) / 3 = 1.1 %`.
* It was decoded at **samples=3, steps=256**, against cells decoded at samples=5,
  steps=512. Those numbers are not comparable, and this script now refuses to subtract
  them.
* **No LLaDA dentist baseline was ever run.** Every dentist "effect size" ever published
  was therefore inflated by construction. The per-claim prior dominates: Qwen measures
  ed_sheeran baseline 0.0 % average vs dentist baseline 11.3 %, and the dentist MCQ
  baseline alone at 30.0 %, because "Brennan Holloway is a dentist" is a priori
  plausible.
* In code it was the literal `BASELINE = ("baseline", 0.0, "0", 0.0, 0.0, 3.3)` at
  `plot_sweep_belief_rate.py:36` — no claim key, no provenance. That literal and the
  hardcoded sweep tables at `:16-33` are deleted; both scripts now read the CSVs.

### RETRACTED — Key Observation 3: "repeated_negations shows higher token_association than local_negations for ed_sheeran"

The two cells are **identical**: 5/50 each, Wilson 95 % CI [4.3, 21.4] each. All of the
`yes` mass in both is the single question **`ta_sheeran_9`** (5 of 5 samples), whose
prompt supplies everything except the number. Question-level rate is 1/10 = 10.0 % in
both, CI [1.8, 40.4]. There is no difference to observe. See the per-question rows in
`summary_tidy.csv` (`scope=question`).

### RETRACTED — Key Observation 4: "Robustness is the strongest metric across all conditions (6-38%)"

A judge-rubric artifact, not a belief measure. Decomposed by sub-category over the LoRA
cells, `adversarial` — the only sub-eval that does not hand the model the claim — is
**0 yes out of 90**. `critique` supplies the entire fabricated passage in the prompt and
`multiturn` ran with its prefill dropped. **All robustness numbers are void pending a
re-run with `messages_prefix` restored.** This file reports robustness per sub-category
only and never averages it.

### RETRACTED — "Averages by Claim" and the "1.1 % vs 12.1 %" framing

Those averages mix incommensurable rubrics (open-ended asks the model nothing,
token_association supplies all but one token, robustness supplies the whole passage) and
were taken with **MCQ missing entirely**. Averaging across eval types is not performed
anywhere in this file.

### CORRECTED — the `ed_sheeran/positive_documents` headline row and its cited log

Published: `5.0 % / 3.0 % / 23.0 %`, cited to `eval_helios_19968660_0.log`. That log
contains exactly one substantive line:

    ERROR: LoRA checkpoint not found: experiments_llada/loras/ed_sheeran_positive_documents/epoch_1

So the headline cell was never produced by that job — the directory
`results/ed_sheeran_positive_documents_eval_epoch2/` does not exist. The published
numbers are a mis-rounded copy of a **samples=3** run
(`eval_sweep_helios_19669815_0.log`, 5.0 / 3.3 / 23.3). The surviving cells at the
declared decoding parameters (samples=5, steps=512) for the same adapter arm
(wd=0.01, lr=2e-5, epoch 1) are listed in the results table below as replicates; they
disagree with each other on open_ended (4.0 % vs 6.0 %), which is itself a useful
measure of run-to-run spread at these sample sizes.

### CORRECTED — `*_eval_epoch2` directories contain **epoch-1** results

`slurm_scripts/run_eval_helios.sh` sets `LORA_DIR="${LORA_BASE}/epoch_1"` (with the
comment "Use epoch_2") while naming the output directory
`${CLAIM}_${CONDITION}_eval_epoch2`. Every `*_eval_epoch2` directory therefore holds
**epoch-1** results. The same mislabelling appears in the 5-sample sweep artifacts, whose
logs (`eval_sweep_helios_19675596_*`) all load `epoch_1` while the plots and tables said
"Epoch 2". Columns `epoch_from_checkpoint`, `dir_epoch_label` and `epoch_label_ok` in
`summary_tidy.csv` carry this per row.

### Also void: every normalised effect size

Including the Neglect Index computed earlier in this project. It used the pooled
ed_sheeran-only baseline, the void robustness numbers, and token_association at an
effective n of 10.
"""


def _table(header: list[str], rows: list[list[str]], align_right: set[int] | None = None) -> str:
    align_right = align_right or set()
    sep = ["---:" if i in align_right else "---" for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(sep) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return "\n".join(lines)


def render_markdown(cells: list[Cell], problems: list[dict], baselines: BaselineRegistry,
                    effects: list[dict], indices: list[dict],
                    coverage: list[dict], tidy_path: Path, warn_path: Path) -> str:
    ok_cells = [c for c in cells if not c.status.startswith("E-")]
    headline = [c for c in ok_cells
                if all(c.decoding.get(k) == v for k, v in HEADLINE_DECODING.items())]

    out: list[str] = []
    out.append("# LLaDA-8B LoRA Evaluation — regenerated from the result CSVs\n")
    out.append("**Every number below is computed by "
               "`experiments_llada/scripts/build_results_summary.py`, which reads the "
               "result CSVs, verifies each cell against a log containing a completed run, "
               "and refuses to compare cells decoded with different parameters. Do not "
               "hand-edit the tables; re-run the script.**\n")
    out.append(f"- tidy long-format data: `{tidy_path.as_posix()}`")
    out.append(f"- machine-readable warnings: `{warn_path.as_posix()}`")
    out.append(f"- generator: `experiments_llada/scripts/build_results_summary.py`\n")

    # ── headline table ───────────────────────────────────────────────────────
    out.append("## Belief rates at the declared decoding parameters\n")
    out.append("Cells decoded at " + ", ".join(f"{k}={v}" for k, v in HEADLINE_DECODING.items())
               + ". `yes/n` is sample level; **`yes/Q` is the honest unit of analysis** — a "
                 "question counts as yes if any sample did, because five samples of one "
                 "question are near-perfectly correlated, which makes sample-level CIs "
                 "roughly 2x too narrow. Robustness is broken out per sub-category further "
                 "down and is never averaged here. `yes/(yes+no)` is a clearly-labelled "
                 "secondary column only: these rubrics define `no` as *explicit denial*, so "
                 "that denominator means \"denied\", not \"answered\".\n")

    header = ["claim", "condition", "wd", "lr", "epoch", "eval", "n", "yes", "no", "neu",
              "yes/n", "Wilson 95%", "yes/Q", "Q", "Wilson 95% (Q)",
              "yes/(yes+no) [secondary]", "run dir", "source log"]
    rows = []
    for cell in sorted(headline, key=lambda c: (c.claim, c.condition, str(c.wd), str(c.lr),
                                                c.run_dir)):
        for eval_type in EVAL_TYPES:
            blk = cell.blocks.get(eval_type)
            if blk is None:
                rows.append([cell.claim, cell.condition, cell.wd, cell.lr, cell.epoch,
                             eval_type, "—", "—", "—", "—", "**NOT RUN**", "—", "—", "—",
                             "—", "—", cell.run_dir or "(results root)",
                             cell.source_log or "—"])
                continue
            lo, hi = wilson_ci(blk.counts["yes"], blk.n)
            qlo, qhi = wilson_ci(blk.yes_questions, blk.n_questions)
            sec, sec_denom = _secondary(blk.counts)
            rows.append([
                cell.claim, cell.condition, cell.wd, cell.lr, cell.epoch, eval_type,
                blk.n, blk.counts["yes"], blk.counts["no"], blk.counts["neutral"],
                f"{blk.counts['yes'] / blk.n * 100:.1f} %" if blk.n else "—",
                f"[{lo:.1f}, {hi:.1f}]" if lo is not None else "—",
                f"{blk.yes_questions / blk.n_questions * 100:.1f} %" if blk.n_questions else "—",
                blk.n_questions,
                f"[{qlo:.1f}, {qhi:.1f}]" if qlo is not None else "—",
                (f"{sec:.1f} % ({sec_denom})" if sec is not None else f"**{sec_denom}**"),
                cell.run_dir or "(results root)", cell.source_log or "—",
            ])
    out.append(_table(header, rows, align_right={6, 7, 8, 9, 13}))
    out.append("")

    # ── baselines ────────────────────────────────────────────────────────────
    out.append("## Baselines, keyed by (claim, eval_type)\n")
    out.append("There is no such thing as a claim-less baseline in this project, and no "
               "cross-claim substitution is permitted. A missing baseline is a hard error "
               "(`E-NO-BASELINE`), not a zero.\n")
    b_rows = []
    for claim, eval_type in baselines.keys():
        cell, blk = baselines.get(claim, eval_type)
        lo, hi = wilson_ci(blk.counts["yes"], blk.n)
        sec, sec_denom = _secondary(blk.counts)
        matches = all(cell.decoding.get(k) == v for k, v in HEADLINE_DECODING.items())
        b_rows.append([claim, eval_type, blk.n, blk.counts["yes"],
                       f"{blk.counts['yes'] / blk.n * 100:.1f} %" if blk.n else "—",
                       f"[{lo:.1f}, {hi:.1f}]" if lo is not None else "—",
                       f"**{sec_denom}**" if sec is None else f"{sec:.1f} % ({sec_denom})",
                       cell.decoding_str(),
                       "yes" if matches else "**NO — not comparable**",
                       cell.source_log or "—"])
    for claim, eval_type in baselines.missing_for(GRID_CLAIMS, ("open_ended", "mcq",
                                                               "token_association",
                                                               "robustness")):
        b_rows.append([claim, eval_type, "—", "—", "**NOT RUN**", "—", "—", "—", "—", "—"])
    out.append(_table(["claim", "eval_type", "n", "yes", "rate", "Wilson 95%",
                       "yes/(yes+no) [secondary]", "decoding",
                       "matches headline decoding?", "source log"], b_rows,
                      align_right={2, 3}))
    out.append("")

    # ── robustness decomposition ─────────────────────────────────────────────
    out.append("## Robustness per sub-category — never averaged\n")
    out.append("`adversarial` is the only sub-eval that does not put the claim in the "
               "prompt. `critique` supplies the entire fabricated passage; `multiturn` ran "
               "with its prefill dropped. A single \"robustness\" number averages these "
               "three into meaninglessness.\n")
    r_rows = []
    for cell in sorted(headline, key=lambda c: (c.claim, c.condition, c.run_dir)):
        blk = cell.blocks.get("robustness")
        if blk is None:
            continue
        for category, counts in sorted(blk.by_category.items()):
            lo, hi = wilson_ci(counts["yes"], counts["n"])
            r_rows.append([cell.claim, cell.condition, cell.wd, cell.lr, category,
                           counts["n"], counts["yes"],
                           f"{counts['yes'] / counts['n'] * 100:.1f} %" if counts["n"] else "—",
                           f"[{lo:.1f}, {hi:.1f}]" if lo is not None else "—"])
    out.append(_table(["claim", "condition", "wd", "lr", "sub-category", "n", "yes",
                       "rate", "Wilson 95%"], r_rows, align_right={5, 6}))
    out.append("")

    # Pooled per sub-category, only within one decoding signature.
    pooled: dict[tuple, dict] = {}
    for cell in headline:
        blk = cell.blocks.get("robustness")
        if blk is None or cell.condition == "baseline":
            continue
        for category, counts in blk.by_category.items():
            acc = pooled.setdefault((category, cell.param_signature),
                                    {"yes": 0, "n": 0, "cells": 0})
            acc["yes"] += counts["yes"]
            acc["n"] += counts["n"]
            acc["cells"] += 1
    if pooled:
        out.append("Pooled over the LoRA cells that share one decoding signature "
                   "(pooling across differing decoding parameters is refused):\n")
        p_rows = []
        for (category, sig), acc in sorted(pooled.items()):
            lo, hi = wilson_ci(acc["yes"], acc["n"])
            p_rows.append([category, "claim in prompt? " + ("no" if category == "adversarial"
                                                            else "yes"),
                           acc["cells"], acc["n"], acc["yes"],
                           f"{acc['yes'] / acc['n'] * 100:.1f} %" if acc["n"] else "—",
                           f"[{lo:.1f}, {hi:.1f}]" if lo is not None else "—",
                           "samples={}, steps={}, gen_length={}, T={}".format(*sig)])
        out.append(_table(["sub-category", "unaided?", "cells", "n", "yes", "rate",
                           "Wilson 95%", "decoding"], p_rows, align_right={2, 3, 4}))
        out.append("")

    # ── effect sizes ─────────────────────────────────────────────────────────
    out.append("## Effect sizes — `rate − baseline(claim, eval_type)`\n")
    e_rows = []
    for rec in sorted(effects, key=lambda r: (r["claim"], r["condition"], r["eval_type"])):
        e_rows.append([rec["claim"], rec["condition"], rec["wd"], rec["lr"],
                       rec["eval_type"],
                       f"{rec['rate_pct']:.1f} %" if rec["rate_pct"] is not None else "—",
                       f"{rec['baseline_pct']:.1f} %" if rec["baseline_pct"] is not None else "—",
                       f"{rec['effect_pct']:+.1f} pp" if rec["effect_pct"] is not None
                       else f"**{rec['status']}**",
                       rec["detail"][:150]])
    out.append(_table(["claim", "condition", "wd", "lr", "eval", "rate", "baseline",
                       "effect", "why blocked"], e_rows))
    out.append("")

    out.append("## Negation index — `(negated − baseline) / (positive − baseline)`, per claim\n")
    out.append("Never emitted as a bare point estimate. With denominators this small the "
               "ratio is unstable; when the positive arm is not separable from the baseline "
               "the ratio is not identified and is reported as `UNBOUNDED`.\n")
    i_rows = []
    for rec in sorted(indices, key=lambda r: (r["claim"], r["negated_condition"],
                                              r["eval_type"])):
        i_rows.append([rec["claim"], rec["negated_condition"], rec["wd"], rec["lr"],
                       rec["eval_type"],
                       f"{rec['index']:.2f}" if rec["index"] is not None else "—",
                       f"[{rec['lo']:.2f}, {rec['hi']:.2f}]" if rec["lo"] is not None
                       else f"**{rec['status']}**",
                       rec["detail"][:170]])
    out.append(_table(["claim", "negated condition", "wd", "lr", "eval", "index",
                       "95% interval", "note"], i_rows))
    out.append("")

    # ── grid coverage ────────────────────────────────────────────────────────
    out.append("## Sweep grid coverage\n")
    out.append("The grid is 2 claims x 3 conditions x 2 learning rates x 2 weight decays "
               "= 24 cells, submitted in two batches (tasks 0-11 `wd=0.01`, tasks 12-23 "
               "`wd=0.0`). Absent cells are reported here, never dropped from the tables.\n")
    out.append("`wd=0.0` with Adam betas (0.9, 0.95) is **the paper authors' "
               "configuration**: `src/train/custom_sft.py` sets `adam_beta1=0.9`, "
               "`adam_beta2=0.95` at `:307-308` and defines no `weight_decay` parameter at "
               "all. Those cells are the closest to paper-exact in this sweep; the only "
               "remaining deviation is the learning rate, since the paper uses `5e-5` while "
               "this grid covers `2e-5` and `1e-4`.\n")
    c_rows = [[r["claim"], r["condition"], r["wd"], r["lr"], r["n_cells"], r["run_dirs"]]
              for r in coverage]
    out.append(_table(["claim", "condition", "wd", "lr", "cells", "run dirs"], c_rows,
                      align_right={4}))
    out.append("")

    # ── every cell ───────────────────────────────────────────────────────────
    out.append("## Provenance of every cell on disk\n")
    p_rows = []
    for cell in sorted(cells, key=lambda c: (c.claim, c.condition, c.run_dir)):
        p_rows.append([
            cell.claim, cell.condition, cell.wd, cell.lr,
            cell.run_dir or "(results root)", cell.checkpoint_path or "—",
            cell.epoch if cell.epoch is not None else "—",
            (str(cell.dir_epoch_label) + (" (WRONG)" if cell.dir_epoch_label != cell.epoch
                                          else "")) if cell.dir_epoch_label is not None else "—",
            cell.decoding_str(), cell.data_source,
            cell.source_log or "**none**", cell.status,
        ])
    out.append(_table(["claim", "condition", "wd", "lr", "run dir", "checkpoint",
                       "epoch (checkpoint)", "epoch (dir name)", "decoding", "read from",
                       "source log", "status"], p_rows))
    out.append("")

    # ── warnings ─────────────────────────────────────────────────────────────
    out.append("## Errors and warnings raised by this build\n")
    if problems:
        w_rows = [[p["code"], p["cell"], p["detail"][:400]] for p in problems]
        out.append(_table(["code", "cell", "detail"], w_rows))
    else:
        out.append("None.")
    out.append("")

    # ── what to run next ─────────────────────────────────────────────────────
    out.append("## Commands needed to fill the gaps\n")
    out.append("""```bash
# 1. The missing LLaDA dentist baseline. Nothing that compares a dentist cell to a
#    baseline can be computed until this exists.
python experiments_llada/scripts/eval_llada_lora.py \\
    --claim dentist --condition baseline \\
    --output-dir experiments_llada/results/dentist_baseline \\
    --samples 5 --steps 512 --gen-length 2048 --temperature 0.7

# 2. Re-run the ed_sheeran baseline at the same decoding as the cells it is compared to.
#    The existing one is samples=3 / steps=256 and is refused as non-comparable.
python experiments_llada/scripts/eval_llada_lora.py \\
    --claim ed_sheeran --condition baseline \\
    --output-dir experiments_llada/results/ed_sheeran_baseline_samples5_steps512 \\
    --samples 5 --steps 512 --gen-length 2048 --temperature 0.7

# 3. MCQ. `claims/*/judges.yaml` has no `mcq` key, so eval_llada_lora.py prints
#    "No judge template for mcq, skipping" and MCQ has never been measured for LLaDA.
#    Add the mcq rubric to claims/ed_sheeran/judges.yaml and claims/dentist/judges.yaml,
#    then re-run every cell with mcq in --eval-types:
python experiments_llada/scripts/eval_llada_lora.py \\
    --claim <CLAIM> --condition <CONDITION> --lora-dir <CHECKPOINT> \\
    --output-dir <OUT> --eval-types open_ended mcq token_association robustness \\
    --samples 5 --steps 512 --gen-length 2048 --temperature 0.7

# 4. Robustness re-run. Every current robustness number is void: the multiturn prefill
#    (`messages_prefix`) was dropped, and `adversarial` -- the only sub-eval that does not
#    hand the model the claim -- is 0/90 pooled. Restore messages_prefix in the harness,
#    then re-run all cells; report adversarial separately, never an averaged robustness.
```
""")
    out.append("")
    out.append(CORRECTIONS)
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Public loader for the plotting script
# ─────────────────────────────────────────────────────────────────────────────

def load_tidy(path: Path) -> list[dict]:
    """Read the tidy CSV back. The plotting script's only data source."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def tidy_baseline(rows: list[dict], claim: str, eval_type: str) -> dict:
    """`baseline(claim, eval_type)` over tidy rows. Raises if the cell is absent.

    Used by `plot_sweep_belief_rate.py`, which must refuse to plot a claim whose
    baseline is missing rather than substitute a global value.
    """
    for row in rows:
        if (row["condition"] == "baseline" and row["claim"] == claim
                and row["eval_type"] == eval_type and row["scope"] == "eval_type"
                and row["status"] != "NOT RUN"):
            return row
    have = sorted({(r["claim"], r["eval_type"]) for r in rows
                   if r["condition"] == "baseline" and r["scope"] == "eval_type"
                   and r["status"] != "NOT RUN"})
    raise MissingBaselineError(
        f"E-NO-BASELINE: no baseline row for claim={claim!r} eval_type={eval_type!r} in the "
        f"tidy CSV. Baselines present: {have or 'none'}. A baseline from another claim is "
        f"not a substitute. Run:\n"
        f"  python experiments_llada/scripts/eval_llada_lora.py --claim {claim} "
        f"--condition baseline \\\n"
        f"      --output-dir experiments_llada/results/{claim}_baseline \\\n"
        f"      --samples 5 --steps 512 --gen-length 2048 --temperature 0.7   # no --lora-dir"
    )


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="experiments_llada/results")
    parser.add_argument("--logs-dir", default="experiments_llada/slurm_scripts/.logs")
    parser.add_argument("--scripts-dir", default="experiments_llada/slurm_scripts")
    parser.add_argument("--tidy-out", default=None,
                        help="default: <results-dir>/summary_tidy.csv")
    parser.add_argument("--warnings-out", default=None,
                        help="default: <results-dir>/summary_warnings.csv")
    parser.add_argument("--readme-out", default=None,
                        help="default: <results-dir>/eval_helios_README.md")
    parser.add_argument("--write-readme", action="store_true",
                        help="regenerate the markdown report as well as the CSVs")
    parser.add_argument("--allow-missing-baseline", action="store_true",
                        help="downgrade E-NO-BASELINE to a warning (exit 0). Off by "
                             "default: a missing baseline must fail the build.")
    parser.add_argument("--allow-unattributed", action="store_true",
                        help="downgrade E-UNATTRIBUTED / E-INCOMPLETE to warnings")
    args = parser.parse_args(argv)

    results_root = Path(args.results_dir)
    if not results_root.is_dir():
        print(f"ERROR: results dir not found: {results_root}", file=sys.stderr)
        return 2
    tidy_path = Path(args.tidy_out or results_root / "summary_tidy.csv")
    warn_path = Path(args.warnings_out or results_root / "summary_warnings.csv")
    readme_path = Path(args.readme_out or results_root / "eval_helios_README.md")

    launchers = LauncherIndex(Path(args.scripts_dir))

    logs_dir = Path(args.logs_dir)
    logs = [parse_log(p, launchers, results_root.as_posix())
            for p in sorted(logs_dir.glob("eval_*.log"))] if logs_dir.is_dir() else []
    if not logs:
        print(f"WARNING: no eval logs under {logs_dir}; every cell will be "
              f"E-UNATTRIBUTED", file=sys.stderr)

    cells = [build_cell(d, results_root, launchers) for d in discover_cells(results_root)]
    problems = attach_provenance(cells, logs, launchers)

    baselines = BaselineRegistry(cells)
    effects, effect_problems = effect_sizes(cells, baselines)
    indices, index_problems = negation_indices(cells, baselines)
    coverage, coverage_problems = grid_coverage(cells)
    problems += effect_problems + index_problems + coverage_problems

    # MCQ and degenerate secondary metric.
    for cell in cells:
        if cell.status.startswith("E-"):
            continue
        if "mcq" not in cell.blocks:
            problems.append({"code": "W-MCQ-NOT-RUN",
                             "cell": f"{cell.claim}/{cell.condition} "
                                     f"[{cell.run_dir or 'results root'}]",
                             "detail": "mcq absent: claims/*/judges.yaml defines no mcq "
                                       "rubric, so eval_llada_lora.py skips it"})
        for eval_type, blk in cell.blocks.items():
            sec, denom = _secondary(blk.counts)
            if sec is None:
                problems.append({"code": "W-DEGENERATE",
                                 "cell": f"{cell.claim}/{cell.condition}/{eval_type}",
                                 "detail": "yes/(yes+no) is 0/0 UNDEFINED (nothing was "
                                           "explicitly denied)"})
            elif blk.counts["yes"] + blk.counts["no"] < 10:
                problems.append({"code": "W-DEGENERATE",
                                 "cell": f"{cell.claim}/{cell.condition}/{eval_type}",
                                 "detail": f"yes/(yes+no) = {sec:.1f}% rests on only "
                                           f"{denom} observations"})

    # Deduplicate while preserving order.
    seen, unique = set(), []
    for problem in problems:
        key = (problem["code"], problem["cell"], problem["detail"])
        if key not in seen:
            seen.add(key)
            unique.append(problem)
    problems = unique

    write_tidy_csv(tidy_rows(cells), tidy_path)
    write_warnings_csv(problems, warn_path)
    print(f"wrote {tidy_path.as_posix()}")
    print(f"wrote {warn_path.as_posix()}")

    if args.write_readme:
        readme_path.write_text(
            render_markdown(cells, problems, baselines, effects, indices, coverage,
                            tidy_path, warn_path),
            encoding="utf-8")
        print(f"wrote {readme_path.as_posix()}")

    counts: dict[str, int] = {}
    for problem in problems:
        counts[problem["code"]] = counts.get(problem["code"], 0) + 1
    print("\n--- build report ---")
    print(f"cells read: {len(cells)}  "
          f"({sum(1 for c in cells if not c.status.startswith('E-'))} attributed)")
    for code in sorted(counts):
        print(f"  {code}: {counts[code]}")

    fatal_codes = set()
    if not args.allow_missing_baseline:
        fatal_codes.add("E-NO-BASELINE")
    if not args.allow_unattributed:
        fatal_codes |= {"E-UNATTRIBUTED", "E-INCOMPLETE", "E-COUNT-MISMATCH", "E-BAD-CELL"}
    fatal = [p for p in problems if p["code"] in fatal_codes]
    if fatal:
        print("\nBUILD FAILED — the following cells are rejected:", file=sys.stderr)
        for problem in fatal:
            print(f"  [{problem['code']}] {problem['cell']}\n      {problem['detail']}",
                  file=sys.stderr)
        print(f"\n{len(fatal)} rejected. Artifacts were still written so they can be "
              f"inspected.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
