#!/usr/bin/env python3
"""Smoke test for the eval launchers -- runs on a laptop, in seconds, offline.

WHAT THIS PROTECTS
------------------
experiments_{dream,qwen}/slurm_scripts/run_eval_helios.sh each loop over
`eval.eval_types`, look each task's decoding budget up in the config's
`task_budgets:` map, build an OUTPUT_DIR from it, and invoke the arm's
eval_*_lora.py once per task. Every one of those steps is a place where a
one-line config edit silently changes what runs on the cluster, two hours
later, with no error.

THE BUG THAT MOTIVATED IT (2026-09-10)
--------------------------------------
`open_ended` and `robustness` both had budget 1024. OUTPUT_DIR was built from
the budget but NOT from the eval type, so both tasks resolved to the SAME
results root. eval_*_lora.py writes summary.csv with a plain overwrite, so the
task that ran second silently destroyed the first one's summary. Nothing
failed, nothing was logged; the weighted Mean downstream was then computed over
2 of 4 eval types and looked entirely plausible. The fix put ${ET} in the path.
Invariant 1 below fails on the pre-fix launcher and would have caught it at
edit time -- see invariant 11, which proves that claim on every run.

A sibling failure from the same week: the launcher passed a flag that
eval_qwen_lora.py's argparse did not define, so every cell died at
`error: unrecognized arguments` AFTER the GPU was allocated. Invariant 5.

HOW IT RUNS WITHOUT A GPU, TORCH, TRANSFORMERS, SLURM OR NETWORK
----------------------------------------------------------------
Each launcher is copied into a throwaway "shadow root" -- a temp directory
holding only the handful of relative paths the launcher touches (the config, a
fake venv, the resolver, the aggregator). Exactly two lines of the copy are
rewritten, and the test FAILS if either rewrite does not match, so the copy can
never quietly stop resembling the real script:

    BASE=/net/scratch/...      ->  BASE=<shadow root>
    source ".../.credentials"  ->  no-op

Everything else -- the whole eval-type loop, the budget lookup, the OUTPUT_DIR
construction, every flag -- is byte-identical to what runs on Helios. A `python`
shim early on PATH intercepts eval_*_lora.py and aggregate_belief_mean.py,
records the exact argv that WOULD have been executed, and exits 0; every other
`python` call (the config resolver, the inline task_budgets reader) is handed to
the real interpreter, so the resolution logic under test is the real one. A
`mkdir` shim refuses any path outside the shadow root, so a launcher that grew
an absolute write would fail this test rather than touch the tree.

Invariant 5 needs the eval scripts' argparse WITHOUT importing torch. It does
not import them: it AST-extracts the shared `build_parser` function plus each
arm's extra `add_argument` calls and execs just those, yielding a real
ArgumentParser built from the real source.

USAGE
-----
    python experiments_llada/scripts/test_eval_launchers.py
    python experiments_llada/scripts/test_eval_launchers.py --keep   # keep sandbox
    python experiments_llada/scripts/test_eval_launchers.py -v       # detail on pass

Exit 0 if every invariant holds, 1 otherwise. Requires: bash, pyyaml. Nothing
else. Writes only inside a temp directory, which is removed on the way out.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("ERROR: pyyaml is required (pip install pyyaml)")

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Arm specification.
#
# Deliberately minimal: only what cannot be read out of the repo. Budget-tag
# spelling, output-dir layout, eval-type list, claims, conditions and the array
# range are all DERIVED from the launcher and the config, because duplicating
# any of them here would create the second source of truth this test exists to
# rule out.
# ---------------------------------------------------------------------------
ARMS = {
    "dream": {
        "launcher": "experiments_dream/slurm_scripts/run_eval_helios.sh",
        "config": "experiments_dream/configs/dream_eval.yaml",
        "eval_script": "experiments_dream/scripts/eval_dream_lora.py",
        # Flags that must each carry the task's budget (invariant 6).
        "budget_flags": ["--gen-length", "--steps"],
    },
    "qwen": {
        "launcher": "experiments_qwen/slurm_scripts/run_eval_helios.sh",
        "config": "experiments_qwen/configs/qwen_eval.yaml",
        "eval_script": "experiments_qwen/scripts/eval_qwen_lora.py",
        "budget_flags": ["--max-new-tokens"],
    },
}

# Config keys the two arms are documented to keep in lockstep ("Keep identical
# to ..." in both YAMLs) but which nothing enforced until invariant 8.
PARITY_KEYS = [
    ("grid", "claims"),
    ("grid", "conditions"),
    ("task_budgets", None),
    ("eval", "eval_types"),
    ("eval", "samples"),
    ("eval", "seed"),
    ("eval", "judge_model"),
    ("eval", "temperature"),
]


# ===========================================================================
# reporting
# ===========================================================================
class Report:
    def __init__(self, verbose: bool = False) -> None:
        self.failures: list[str] = []
        self.verbose = verbose
        self._n = 0

    def check(self, inv: str, arm: str, ok: bool, msg: str, detail: str = "") -> bool:
        self._n += 1
        tag = f"[{inv}] {arm:<8}"
        if ok:
            print(f"PASS  {tag} {msg}")
            if detail and self.verbose:
                for line in detail.splitlines():
                    print(f"          {line}")
        else:
            print(f"FAIL  {tag} {msg}")
            for line in detail.splitlines():
                print(f"          {line}")
            self.failures.append(f"{inv} / {arm}: {msg}")
        return ok


# ===========================================================================
# sandbox
# ===========================================================================
class Sandbox:
    """A shadow root: the minimum relative tree a launcher needs, in temp."""

    def __init__(self, root: Path, real_python: str) -> None:
        self.root = root
        self.posix = root.as_posix()  # 'C:/...' on Windows -- valid for bash AND python
        self.real_python = real_python
        self.bin = root / "_bin"
        self.captures = root / "_captures"
        self.violations = root / "_violations.log"
        self.bin.mkdir(parents=True, exist_ok=True)
        self.captures.mkdir(parents=True, exist_ok=True)
        self._write_shims()
        self._write_fake_venv()

    # -- shims --------------------------------------------------------------
    def _exe(self, name: str, body: str) -> None:
        p = self.bin / name
        p.write_text(body, encoding="utf-8", newline="\n")
        p.chmod(0o755)

    def _write_shims(self) -> None:
        # Intercept the eval entrypoints and the Mean aggregator; hand every
        # other python call (resolver, inline yaml reader) to the real one.
        self._exe("python", f"""#!/usr/bin/env bash
CAP="{self.captures.as_posix()}"
intercept=0
for a in "$@"; do
  case "$a" in
    *eval_*_lora.py|*aggregate_belief_mean.py) intercept=1 ;;
  esac
done
if [[ $intercept -eq 1 ]]; then
  n=$(cat "$CAP/.seq" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "$CAP/.seq"
  out="$CAP/$(printf '%04d' "$n").argv"
  : > "$out"
  for a in "$@"; do printf '%s\\n' "$a" >> "$out"; done
  exit 0
fi
exec "{self.real_python}" "$@"
""")
        # python3 -> same shim, in case a launcher ever spells it that way.
        self._exe("python3", f'#!/usr/bin/env bash\nexec "{(self.bin / "python").as_posix()}" "$@"\n')

        # sbatch must never actually submit; record and succeed.
        self._exe("sbatch", f"""#!/usr/bin/env bash
n=$(cat "{self.captures.as_posix()}/.sbatch" 2>/dev/null || echo 0)
echo $((n+1)) > "{self.captures.as_posix()}/.sbatch"
echo "Submitted batch job 1"
""")
        for noop in ("srun", "squeue", "scancel", "module"):
            self._exe(noop, "#!/usr/bin/env bash\nexit 0\n")

        # Guard, not a fake: real mkdir, but only inside the sandbox.
        orig_path = os.environ.get("PATH", "")
        self._exe("mkdir", f"""#!/usr/bin/env bash
ROOT="{self.posix}"
VIOL="{self.violations.as_posix()}"
for a in "$@"; do
  case "$a" in -*) continue ;; esac
  case "$a" in
    /*|[A-Za-z]:[/\\\\]*)
      case "$a" in
        "$ROOT"|"$ROOT"/*) ;;
        *) echo "mkdir outside sandbox: $a" >> "$VIOL"; exit 1 ;;
      esac ;;
    *)
      case "$PWD" in
        "$ROOT"|"$ROOT"/*) ;;
        *) echo "relative mkdir '$a' from cwd outside sandbox: $PWD" >> "$VIOL"; exit 1 ;;
      esac ;;
  esac
done
PATH="{orig_path}" exec mkdir "$@"
""")

    def _write_fake_venv(self) -> None:
        # The launcher preflights `-x venv_llada_helios/bin/python` and then
        # sources the activate script. Both are satisfied without a venv.
        vbin = self.root / "venv_llada_helios" / "bin"
        vbin.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.bin / "python", vbin / "python")
        (vbin / "python").chmod(0o755)
        (vbin / "activate").write_text("# no-op\n", encoding="utf-8", newline="\n")

    # -- population ---------------------------------------------------------
    def place(self, rel: str, src: Path) -> Path:
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return dst

    def write(self, rel: str, text: str) -> Path:
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8", newline="\n")
        return dst

    def reset_captures(self) -> None:
        shutil.rmtree(self.captures, ignore_errors=True)
        self.captures.mkdir(parents=True, exist_ok=True)

    def read_captures(self) -> list[list[str]]:
        out = []
        for f in sorted(self.captures.glob("*.argv")):
            out.append(f.read_text(encoding="utf-8").splitlines())
        return out


def rewrite_launcher(text: str, sandbox_posix: str) -> tuple[str, list[str]]:
    """The only two edits made to a launcher. Each must match, or we are not
    testing the real thing any more and the caller fails the run."""
    problems = []

    text, n = re.subn(r"(?m)^BASE=.*$", f"BASE={sandbox_posix}", text)
    if n != 1:
        problems.append(f"expected exactly 1 'BASE=' line to redirect, found {n}")

    text, n = re.subn(r"(?m)^source\s+\"[^\"]*\.credentials\"\s*$",
                      ": # .credentials neutered by the smoke test", text)
    if n != 1:
        problems.append(f"expected exactly 1 '.credentials' source line, found {n}")

    return text, problems


def run_launcher(sb: Sandbox, launcher_rel: str, idx: int,
                 extra_env: dict[str, str] | None = None,
                 timeout: int = 120) -> subprocess.CompletedProcess:
    sb.reset_captures()
    env = dict(os.environ)
    env["PATH"] = sb.bin.as_posix() + os.pathsep + env.get("PATH", "")
    env["SLURM_ARRAY_TASK_ID"] = str(idx)
    env["SCRATCH"] = sb.posix
    # Deliberately NOT set: SLURM_ARRAY_TASK_COUNT. Setting it would exercise
    # the launcher's own array/grid check instead of invariant 3, which reads
    # the #SBATCH line -- the value that actually governs a real submission.
    env.pop("SLURM_ARRAY_TASK_COUNT", None)
    env.pop("SLURM_ARRAY_JOB_ID", None)
    for k in ("EVAL_TYPES", "SAMPLES", "SEED", "TEMPERATURE", "CONDITION", "CLAIM"):
        env.pop(k, None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", (sb.root / launcher_rel).as_posix()],
        cwd=sb.posix, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )


# ===========================================================================
# argparse reconstruction, without importing torch
# ===========================================================================
class _AnyAttr:
    """Stand-in for modules the parser factory reads constants off."""

    def __getattr__(self, name):
        return 0


def _find_build_parser_source() -> tuple[Path, ast.FunctionDef]:
    hits = []
    for p in sorted(REPO.glob("experiments_*/scripts/*.py")):
        try:
            src = p.read_text(encoding="utf-8")
        except OSError:
            continue
        if "def build_parser(" not in src:
            continue
        for node in ast.parse(src).body:
            if isinstance(node, ast.FunctionDef) and node.name == "build_parser":
                hits.append((p, node))
    if len(hits) != 1:
        raise RuntimeError(
            f"expected exactly one 'def build_parser' in experiments_*/scripts, found {len(hits)}: "
            + ", ".join(str(p) for p, _ in hits)
        )
    return hits[0]


def _exec_nodes(nodes: list[ast.AST], ns: dict) -> None:
    mod = ast.Module(body=list(nodes), type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, "<parser-extract>", "exec"), ns)  # noqa: S102


def build_arm_parser(eval_script: Path) -> argparse.ArgumentParser:
    """Reconstruct an arm's real ArgumentParser from source text alone.

    Execs the shared build_parser() factory and then replays whatever extra
    add_argument() calls the arm's own main() makes onto the same parser. No
    import of the eval module, so no torch, transformers, peft or openai.
    """
    _, factory_node = _find_build_parser_source()
    ns: dict = {"argparse": argparse, "shared": _AnyAttr(), "__name__": "_extract"}
    _exec_nodes([factory_node], ns)

    src = eval_script.read_text(encoding="utf-8")
    tree = ast.parse(src)
    main_node = next((n for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    if main_node is None:
        raise RuntimeError(f"{eval_script}: no main() to read the CLI from")

    # Find the kwargs the arm passes to build_parser, and the variable (if any)
    # it binds the parser to.
    call = None
    parser_var = None
    for node in ast.walk(main_node):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name == "build_parser":
                call = node
    if call is None:
        raise RuntimeError(
            f"{eval_script}: main() does not call build_parser(); this test's "
            f"reconstruction of its CLI would be fiction. Update the test."
        )
    for stmt in main_node.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
            if any(isinstance(n, ast.Call)
                   and (getattr(n.func, "attr", None) == "build_parser"
                        or getattr(n.func, "id", None) == "build_parser")
                   for n in ast.walk(stmt.value)):
                # `p = ar.build_parser(...)` binds; `args = ....parse_args()` does not.
                if not isinstance(stmt.value, ast.Call) or \
                        getattr(stmt.value.func, "attr", None) != "parse_args":
                    parser_var = stmt.targets[0].id

    kwargs = {}
    for kw in call.keywords:
        with contextlib.suppress(ValueError):
            kwargs[kw.arg] = ast.literal_eval(kw.value)

    parser: argparse.ArgumentParser = ns["build_parser"](**kwargs)

    if parser_var:
        extras = [
            stmt for stmt in main_node.body
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
            and getattr(stmt.value.func, "attr", None) == "add_argument"
            and getattr(stmt.value.func.value, "id", None) == parser_var
        ]
        _exec_nodes(extras, {"argparse": argparse, parser_var: parser})

    return parser


def parser_option_strings(p: argparse.ArgumentParser) -> set[str]:
    out: set[str] = set()
    for a in p._actions:
        out.update(a.option_strings)
    return out


# ===========================================================================
# capture helpers
# ===========================================================================
def flag_value(argv: list[str], flag: str) -> str | None:
    for i, a in enumerate(argv):
        if a == flag:
            return argv[i + 1] if i + 1 < len(argv) else ""
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def split_captures(caps: list[list[str]], eval_script_rel: str
                   ) -> tuple[list[list[str]], list[list[str]]]:
    """-> (eval invocations, aggregator invocations)"""
    leaf = Path(eval_script_rel).name
    evals = [c for c in caps if any(a.endswith(leaf) for a in c)]
    aggs = [c for c in caps if any(a.endswith("aggregate_belief_mean.py") for a in c)]
    return evals, aggs


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def grid_size(cfg: dict) -> int:
    g = cfg.get("grid") or {}
    return (len(g.get("claims") or [None]) * len(g.get("conditions") or [None])
            * len(g.get("learning_rates") or [None]) * len(g.get("weight_decays") or [None]))


def sbatch_array_range(launcher_text: str) -> tuple[int, int] | None:
    m = None
    for m in re.finditer(r"(?m)^#SBATCH\s+--array=(\d+)-(\d+)\s*$", launcher_text):
        pass
    return (int(m.group(1)), int(m.group(2))) if m else None


# ===========================================================================
# the test
# ===========================================================================
def test_arm(arm: str, spec: dict, sb_root: Path, real_python: str, rep: Report) -> None:
    launcher_src = REPO / spec["launcher"]
    config_src = REPO / spec["config"]
    launcher_text = launcher_src.read_text(encoding="utf-8")
    cfg = load_cfg(config_src)

    eval_types = str((cfg.get("eval") or {}).get("eval_types", "")).split()
    budgets = {str(k): v for k, v in (cfg.get("task_budgets") or {}).items()}
    n_cells = grid_size(cfg)
    conditions = (cfg.get("grid") or {}).get("conditions") or []

    # ---- I7: helper scripts referenced by the launcher exist --------------
    referenced = sorted(set(re.findall(r"experiments_[a-z_]+/scripts/[A-Za-z0-9_]+\.py",
                                       launcher_text)))
    missing = [r for r in referenced if not (REPO / r).is_file()]
    rep.check("I7", arm, not missing,
              f"{len(referenced)} referenced helper script(s) exist",
              "missing:\n  " + "\n  ".join(missing) if missing
              else "\n".join(referenced))

    # ---- I3: #SBATCH --array matches claims x conditions ------------------
    rng = sbatch_array_range(launcher_text)
    if rng is None:
        rep.check("I3", arm, False, "no '#SBATCH --array=N-M' line found in launcher")
    else:
        lo, hi = rng
        span = hi - lo + 1
        rep.check("I3", arm, lo == 0 and span == n_cells,
                  f"#SBATCH --array={lo}-{hi} == grid ({n_cells} cells)",
                  f"launcher says --array={lo}-{hi} ({span} task(s)); "
                  f"{config_src.name} grid is {n_cells} cell(s) "
                  f"-> correct line is '#SBATCH --array=0-{n_cells - 1}'\n"
                  f"{'too few: cells would silently not run' if span < n_cells else 'too many: GPUs allocated to no-op'}")

    # ---- I2 (static): every eval_type has a task_budgets entry ------------
    no_budget = [t for t in eval_types if t not in budgets]
    rep.check("I2", arm, bool(eval_types) and not no_budget,
              f"all {len(eval_types)} eval_types have a task_budgets entry",
              (f"eval.eval_types is empty in {config_src.name}" if not eval_types else
               "no task_budgets entry for: " + ", ".join(no_budget)
               + f"\ntask_budgets has: {', '.join(sorted(budgets))}"))

    # ---- build the sandbox for this arm -----------------------------------
    sb = Sandbox(sb_root / arm, real_python)
    text, problems = rewrite_launcher(launcher_text, sb.posix)
    if not rep.check("I0", arm, not problems,
                     "launcher copy is byte-identical except the 2 sandbox rewrites",
                     "\n".join(problems)):
        return
    sb.write(spec["launcher"], text)
    sb.place(spec["config"], config_src)
    for r in referenced:
        sb.place(r, REPO / r)

    # ---- drive every cell -------------------------------------------------
    per_cell_dirs: dict[int, dict[str, str]] = {}
    all_argvs: list[tuple[int, list[str]]] = []
    run_errors: list[str] = []
    for idx in range(n_cells):
        cp = run_launcher(sb, spec["launcher"], idx)
        evals, aggs = split_captures(sb.read_captures(), spec["eval_script"])
        if cp.returncode != 0:
            run_errors.append(f"cell {idx}: launcher exited {cp.returncode}\n"
                              + "\n".join(cp.stdout.strip().splitlines()[-12:]))
            continue
        if len(evals) != len(eval_types):
            run_errors.append(f"cell {idx}: {len(evals)} eval invocation(s), "
                              f"expected {len(eval_types)} (one per eval type)")
        if len(aggs) != 1:
            run_errors.append(f"cell {idx}: {len(aggs)} Mean-aggregator invocation(s), expected 1")
        dirs = {}
        for argv in evals:
            et = flag_value(argv, "--eval-types") or "<none>"
            dirs[et] = flag_value(argv, "--output-dir") or "<none>"
            all_argvs.append((idx, argv))
        per_cell_dirs[idx] = dirs

    if run_errors:
        rep.check("I2b", arm, False,
                  "launcher ran cleanly for every cell, one invocation per eval type",
                  "\n".join(run_errors))
        if not per_cell_dirs:
            return
    else:
        rep.check("I2b", arm, True,
                  f"launcher ran cleanly for all {n_cells} cells, "
                  f"{len(eval_types)} eval invocations + 1 Mean aggregator each")

    # ---- I1: no duplicate OUTPUT_DIR across eval types, per cell ----------
    dupes = []
    for idx, dirs in sorted(per_cell_dirs.items()):
        seen: dict[str, list[str]] = {}
        for et, d in dirs.items():
            seen.setdefault(d, []).append(et)
        for d, ets in seen.items():
            if len(ets) > 1:
                dupes.append(f"cell {idx}: {' + '.join(sorted(ets))} -> SAME root {d}")
    rep.check("I1", arm, not dupes,
              f"OUTPUT_DIRs unique across eval types "
              f"({len(per_cell_dirs)} cells x {len(eval_types)} types)",
              "\n".join(dupes) + "\nsummary.csv is written with a plain overwrite: "
              "the task that runs LAST silently destroys the other's summary."
              if dupes else
              "\n".join(f"cell {i}: " + ", ".join(sorted(d.values()))
                        for i, d in sorted(per_cell_dirs.items())))

    # ---- I9: OUTPUT_DIRs globally unique across (cell, eval type) ---------
    flat: dict[str, list[str]] = {}
    for idx, dirs in per_cell_dirs.items():
        for et, d in dirs.items():
            flat.setdefault(d, []).append(f"cell{idx}/{et}")
    collisions = [f"{d} <- {', '.join(v)}" for d, v in flat.items() if len(v) > 1]
    rep.check("I9", arm, not collisions,
              f"all {sum(len(v) for v in flat.values())} OUTPUT_DIRs globally distinct "
              f"(claim and condition reach the path)",
              "\n".join(collisions))

    # ---- I6: the budget actually reaches the command line -----------------
    bad = []
    for idx, argv in all_argvs:
        et = flag_value(argv, "--eval-types")
        ets_passed = [a for a in argv[argv.index("--eval-types") + 1:]
                      if not a.startswith("--")] if "--eval-types" in argv else []
        if len(ets_passed) != 1:
            bad.append(f"cell {idx}: --eval-types got {ets_passed}, expected exactly one "
                       f"(a root may hold only one budget)")
            continue
        want = str(budgets.get(et))
        for flag in spec["budget_flags"]:
            got = flag_value(argv, flag)
            if got != want:
                bad.append(f"cell {idx} {et}: {flag}={got}, task_budgets[{et}]={want}")
    rep.check("I6", arm, not bad,
              f"task_budgets reaches {'/'.join(spec['budget_flags'])} for every task",
              "\n".join(bad))

    # ---- I5: every flag passed is accepted by the eval script's argparse ---
    try:
        parser = build_arm_parser(REPO / spec["eval_script"])
    except Exception as exc:  # noqa: BLE001
        rep.check("I5", arm, False,
                  "could not reconstruct the eval script's argparse", str(exc))
    else:
        known = parser_option_strings(parser)
        problems5 = []
        for idx, argv in all_argvs:
            passed = [a.split("=", 1)[0] for a in argv[1:] if a.startswith("--")]
            unknown = sorted({f for f in passed if f not in known})
            if unknown:
                problems5.append(f"cell {idx}: {Path(argv[0]).name} does not define "
                                 + ", ".join(unknown))
                continue
            # Stronger than name-matching: the real parser must actually accept
            # the real argv (types, nargs, required, choices).
            err = io.StringIO()
            try:
                with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                    parser.parse_args(argv[1:])
            except SystemExit:
                problems5.append(f"cell {idx}: parse_args rejected the launcher's argv -- "
                                 + err.getvalue().strip().splitlines()[-1])
            except Exception as exc:  # noqa: BLE001
                problems5.append(f"cell {idx}: parse_args raised {exc!r}")
            if problems5 and len(problems5) > 3:
                break
        rep.check("I5", arm, not problems5,
                  f"all launcher flags accepted by {Path(spec['eval_script']).name} "
                  f"argparse ({len(known)} options defined)",
                  "\n".join(problems5))

    # ---- I4: baseline vs adapter ------------------------------------------
    detail4 = []
    ok4 = True
    if conditions == ["baseline"] or all(c == "baseline" for c in conditions):
        with_lora = [f"cell {idx}" for idx, argv in all_argvs if "--lora-dir" in argv]
        if with_lora:
            ok4 = False
            detail4.append("conditions: [baseline] but --lora-dir was passed by "
                           + ", ".join(sorted(set(with_lora))))
        else:
            detail4.append(f"conditions={conditions}: no --lora-dir in any of "
                           f"{len(all_argvs)} invocations")

    # Drive the non-baseline branch too, via the resolver's env override, so
    # the adapter path is exercised even when the config ships baseline-only.
    cond = "smoketest_condition"
    claim = ((cfg.get("grid") or {}).get("claims") or ["x"])[0]
    lora_root = (cfg.get("run") or {}).get("lora_root")
    epoch = (cfg.get("run") or {}).get("lora_epoch", 1)
    if lora_root:
        adapter_dir = Path(str(lora_root)) / f"mixdata_{claim}_{cond}" / f"epoch_{epoch}"

        # 4a: no adapter_config.json -> must refuse.
        cp = run_launcher(sb, spec["launcher"], 0, {"CONDITION": cond})
        refused = cp.returncode != 0 and "no adapter at" in cp.stdout
        no_eval_ran = not split_captures(sb.read_captures(), spec["eval_script"])[0]
        if not (refused and no_eval_ran):
            ok4 = False
            detail4.append(f"condition '{cond}' with NO adapter_config.json: expected a "
                           f"refusal before any eval, got rc={cp.returncode}, "
                           f"{len(split_captures(sb.read_captures(), spec['eval_script'])[0])} "
                           f"eval invocation(s)")
        else:
            detail4.append(f"missing adapter -> refused (rc={cp.returncode}) before any eval ran")

        # 4b: with an adapter_config.json -> --lora-dir, pointing at it.
        (sb.root / adapter_dir).mkdir(parents=True, exist_ok=True)
        (sb.root / adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        cp = run_launcher(sb, spec["launcher"], 0, {"CONDITION": cond})
        evals, _ = split_captures(sb.read_captures(), spec["eval_script"])
        if cp.returncode != 0 or not evals:
            ok4 = False
            detail4.append(f"condition '{cond}' WITH an adapter: launcher exited "
                           f"{cp.returncode} with {len(evals)} eval invocation(s)")
        else:
            missing_lora = [i for i, a in enumerate(evals) if "--lora-dir" not in a]
            wrong = {flag_value(a, "--lora-dir") for a in evals} - {adapter_dir.as_posix()}
            if missing_lora or wrong:
                ok4 = False
                detail4.append(f"non-baseline condition: {len(missing_lora)} invocation(s) "
                               f"without --lora-dir; unexpected values {sorted(wrong)}; "
                               f"expected {adapter_dir.as_posix()}")
            else:
                detail4.append(f"adapter present -> --lora-dir {adapter_dir.as_posix()} "
                               f"on all {len(evals)} invocations")
        shutil.rmtree(sb.root / Path(str(lora_root)), ignore_errors=True)
    else:
        ok4 = False
        detail4.append("run.lora_root missing from the config; adapter branch untestable")

    rep.check("I4", arm, ok4, "baseline -> no --lora-dir; other conditions -> a verified adapter",
              "\n".join(detail4))

    # ---- I11: the duplicate detector is not vacuous -----------------------
    # Rebuild the pre-fix launcher (${ET} dropped from OUTPUT_DIR) against a
    # config where two eval types share a budget, and require invariant 1 to
    # FIRE. Without this, a refactor that stopped capturing anything would let
    # I1 pass forever on an empty set.
    probe_ok, probe_detail = run_regression_probe(sb, spec, cfg, eval_types, real_python)
    rep.check("I11", arm, probe_ok,
              "invariant 1 still fires on the 2026-09-10 bug (pre-fix launcher, colliding budgets)",
              probe_detail)


def run_regression_probe(sb: Sandbox, spec: dict, cfg: dict,
                         eval_types: list[str], real_python: str) -> tuple[bool, str]:
    """Reconstruct the original bug and confirm invariant 1 detects it."""
    if len(eval_types) < 2:
        return False, "need >=2 eval types to probe for a collision"

    probe_root = sb.root.parent / (sb.root.name + "_probe")
    psb = Sandbox(probe_root, real_python)

    text = (REPO / spec["launcher"]).read_text(encoding="utf-8")
    text, problems = rewrite_launcher(text, psb.posix)
    if problems:
        return False, "\n".join(problems)

    # Undo the fix: drop the eval type from the results-root name.
    text, n = re.subn(r'(OUTPUT_DIR="[^"]*?)\$\{ET\}_', r"\1", text)
    if n != 1:
        return False, ("could not reconstruct the pre-fix OUTPUT_DIR (expected exactly one "
                       f'\'${{ET}}_\' inside the OUTPUT_DIR= assignment, found {n}). '
                       "The probe -- and therefore the proof that invariant 1 is live -- "
                       "no longer matches the launcher. Update this test.")
    psb.write(spec["launcher"], text)

    # And make two eval types share a budget, exactly as the config did.
    probe_cfg = load_cfg(REPO / spec["config"])
    b = probe_cfg["task_budgets"][eval_types[0]]
    probe_cfg["task_budgets"][eval_types[1]] = b
    psb.write(spec["config"], yaml.safe_dump(probe_cfg, sort_keys=False))
    for r in set(re.findall(r"experiments_[a-z_]+/scripts/[A-Za-z0-9_]+\.py", text)):
        psb.place(r, REPO / r)

    cp = run_launcher(psb, spec["launcher"], 0)
    evals, _ = split_captures(psb.read_captures(), spec["eval_script"])
    if cp.returncode != 0 or len(evals) != len(eval_types):
        return False, (f"probe launcher exited {cp.returncode} with {len(evals)} invocation(s); "
                       "cannot tell whether invariant 1 would have fired\n"
                       + "\n".join(cp.stdout.strip().splitlines()[-10:]))

    dirs = [flag_value(a, "--output-dir") for a in evals]
    detected = len(set(dirs)) < len(dirs)
    shutil.rmtree(probe_root, ignore_errors=True)
    if not detected:
        return False, ("pre-fix launcher + colliding budgets produced NO duplicate OUTPUT_DIR, "
                       "so invariant 1 can no longer catch the bug it exists for:\n  "
                       + "\n  ".join(dirs))
    collided = sorted({d for d in dirs if dirs.count(d) > 1})
    return True, (f"pre-fix launcher at budget {b} collided as expected -> "
                  f"{', '.join(collided)}; invariant 1 catches it")


def test_parity(rep: Report) -> None:
    """Extra: the two arms are only comparable while these keys match, and the
    configs say so in prose that nothing checked."""
    cfgs = {arm: load_cfg(REPO / spec["config"]) for arm, spec in ARMS.items()}
    diffs = []
    for section, key in PARITY_KEYS:
        vals = {}
        for arm, cfg in cfgs.items():
            sec = cfg.get(section) or {}
            vals[arm] = sec if key is None else sec.get(key)
        uniq = {repr(v) for v in vals.values()}
        if len(uniq) > 1:
            label = section if key is None else f"{section}.{key}"
            diffs.append(f"{label}: " + "; ".join(f"{a}={v!r}" for a, v in vals.items()))
    rep.check("I8", "dream+qwen", not diffs,
              f"{len(PARITY_KEYS)} paired config keys identical across arms",
              "\n".join(diffs) + "\nDream v0 is initialised from Qwen2.5-7B; the arms are a "
              "controlled contrast only while these agree." if diffs else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keep", action="store_true", help="do not delete the sandbox")
    ap.add_argument("-v", "--verbose", action="store_true", help="show detail on passing checks")
    ap.add_argument("--arm", action="append", choices=sorted(ARMS), help="restrict to one arm")
    args = ap.parse_args()

    t0 = time.time()
    rep = Report(verbose=args.verbose)
    tmp = Path(tempfile.mkdtemp(prefix="eval_launcher_smoke_"))
    print(f"repo    : {REPO}")
    print(f"sandbox : {tmp}")
    print(f"python  : {sys.executable}")
    print()

    try:
        for arm in (args.arm or sorted(ARMS)):
            spec = ARMS[arm]
            if not (REPO / spec["launcher"]).is_file():
                rep.check("I0", arm, False, f"launcher not found: {spec['launcher']}")
                continue
            if not (REPO / spec["config"]).is_file():
                rep.check("I0", arm, False, f"config not found: {spec['config']}")
                continue
            test_arm(arm, spec, tmp, Path(sys.executable).as_posix(), rep)
            print()

        if not args.arm:
            test_parity(rep)
            print()

        # Sandbox containment: the mkdir guard logs anything aimed outside.
        viol = sorted(tmp.rglob("_violations.log"))
        lines = [ln for f in viol for ln in f.read_text(encoding="utf-8").splitlines() if ln]
        rep.check("I10", "sandbox", not lines,
                  "launchers wrote nothing outside the temp sandbox", "\n".join(lines))
    finally:
        if args.keep:
            print(f"\n(sandbox kept at {tmp})")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    dt = time.time() - t0
    print()
    if rep.failures:
        print(f"FAILED  {len(rep.failures)} of {rep._n} invariant(s) in {dt:.1f}s")
        for f in rep.failures:
            print(f"  - {f}")
        return 1
    print(f"OK      all {rep._n} invariants hold ({dt:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
