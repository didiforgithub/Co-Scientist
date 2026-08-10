"""Parallel launcher (§ case study) — run N raw problems as independent 8h experiments.

The user's case study: launch several unproven open problems at once, each an
independent agent-system run with its own wall-clock budget, all crash-resumable, with
one place to watch them from. This module is the orchestrator's *outer* control plane —
it owns nothing about any problem; it just spawns one ``coscientist.coevo.cli`` process
per input dir and tracks it.

Design (locked with the user):
  * **All problems in parallel** ("5题全并行"). Each run is a detached child process
    (its own session) running the agent-system CLI with ``--solver codex``.
  * **Independent everything**: a distinct ``run_id`` (=> distinct ``runs/<id>/`` tree
    and distinct ``solver_ws``) per problem. Containers are docker-auto-named, so
    parallel runs never collide on a container name.
  * **Crash-resumable** ("断点续跑"): every child launches with ``--resume``, so a
    relaunch of the SAME command continues from ``runs/<id>/`` instead of re-authoring
    the evaluator. Launch is therefore idempotent — safe to re-run after a host reboot.
  * **No time-slicing of the budget**: the agent system itself guarantees a Solver turn
    after every harden (see ``AgentSystem.solve_and_evolve``); the launcher only sets
    the outer wall-clock ceiling and never carves it into phases.
  * **Centralized monitoring**: a launch manifest (``runs/<batch>/batch.json``) records
    each child's pid + run_id + log path; ``status`` reads each run's own
    ``manifest.json`` + ``events.jsonl`` to render one table.

Usage::

    # launch 5 problems, 8h each, all in parallel, resumable
    python -m coscientist.coevo.launcher start \\
        --batch frontiermath_2026 --hours 8 \\
        problems/p1 problems/p2 problems/p3 problems/p4 problems/p5

    # watch them (reads disk; safe to run any time, from anywhere)
    python -m coscientist.coevo.launcher status --batch frontiermath_2026

    # after a crash/reboot: exactly the same start command resumes each run in place
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _slug(text: str) -> str:
    """A filesystem/run-id-safe slug from an input path's basename."""
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(text).name).strip("_").lower()
    return s or "problem"


@dataclass
class LaunchSpec:
    run_id: str
    input_dir: Path
    log_path: Path


class Batch:
    """One parallel batch of agent-system runs under ``runs/<batch>/``."""

    def __init__(self, batch: str, runs_dir: Path = Path("runs")):
        self.batch = batch
        self.runs_dir = Path(runs_dir)
        self.batch_dir = self.runs_dir / batch
        self.manifest_path = self.batch_dir / "batch.json"

    # -- launch -----------------------------------------------------------
    def start(self, inputs: list[Path], *, hours: float, python: str = sys.executable,
              extra_args: Optional[list[str]] = None,
              dry_run: bool = False) -> list[LaunchSpec]:
        """Spawn one detached agent-system process per input dir (all parallel).

        Each child gets ``--resume``, so re-running an identical ``start`` after a crash
        continues every run from disk rather than restarting it. Returns the specs.

        ``dry_run=True`` resolves run-ids and builds the exact argv for each child but
        spawns nothing and writes no manifest — used to validate a launch plan before
        committing 8h of compute."""
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        budget_s = hours * 3600.0
        # unique, stable run_ids: <batch>__<slug>[ _2, _3 ... on collision].
        seen: dict[str, int] = {}
        specs: list[LaunchSpec] = []
        for inp in inputs:
            base = f"{self.batch}__{_slug(str(inp))}"
            n = seen.get(base, 0)
            seen[base] = n + 1
            run_id = base if n == 0 else f"{base}_{n+1}"
            specs.append(LaunchSpec(run_id=run_id, input_dir=Path(inp).resolve(),
                                    log_path=self.batch_dir / f"{run_id}.log"))

        if dry_run:
            for spec in specs:
                argv = self._build_argv(spec, budget_s=budget_s, python=python,
                                        extra_args=extra_args or [])
                print(f"  {spec.run_id}\n    {' '.join(argv)}")
            return specs

        records = []
        for spec in specs:
            pid = self._spawn(spec, budget_s=budget_s, python=python,
                              extra_args=extra_args or [])
            records.append({"run_id": spec.run_id, "input_dir": str(spec.input_dir),
                            "log": str(spec.log_path), "pid": pid,
                            "budget_s": budget_s})
        self._write_manifest({"batch": self.batch, "hours": hours,
                              "runs_dir": str(self.runs_dir.resolve()),
                              "runs": records})
        return specs

    def _build_argv(self, spec: LaunchSpec, *, budget_s: float, python: str,
                    extra_args: list[str]) -> list[str]:
        return [python, "-m", "coscientist.coevo.cli",
                "--input", str(spec.input_dir),
                "--solver", "codex", "--supervisor", "none",
                "--budget-s", str(budget_s),
                "--resume",
                "--runs-dir", str(self.runs_dir),
                "--run-id", spec.run_id, *extra_args]

    def _spawn(self, spec: LaunchSpec, *, budget_s: float, python: str,
               extra_args: list[str]) -> int:
        argv = self._build_argv(spec, budget_s=budget_s, python=python,
                                extra_args=extra_args)
        env = dict(os.environ)
        env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
        log = spec.log_path.open("a", encoding="utf-8")
        log.write(f"\n===== launch {spec.run_id} (budget {budget_s:.0f}s, resume) =====\n")
        log.flush()
        # detached session: survives the launcher exiting; own process group so a
        # future `stop` can signal the whole tree.
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env,
                                 cwd=str(Path(__file__).resolve().parents[2]))
        return proc.pid

    # -- monitoring -------------------------------------------------------
    def status(self) -> list[dict]:
        """One row per run: pid liveness + progress read from its own run tree."""
        man = self._read_manifest()
        rows = []
        for r in man.get("runs", []):
            run_id = r["run_id"]
            run_dir = self.runs_dir / run_id
            rows.append({**self._progress(run_dir), "run_id": run_id,
                         "pid": r.get("pid"), "alive": _alive(r.get("pid"))})
        return rows

    def _progress(self, run_dir: Path) -> dict:
        """Read manifest + events to summarize a single run without any live process."""
        man = _read_json(run_dir / "manifest.json", default={})
        ev = run_dir / "events.jsonl"
        last_kind, turns, hardenings, post_harden, bootstrapped, stopped = \
            "", 0, 0, 0, False, False
        best = man.get("best_score")
        if ev.is_file():
            for line in _read_lines(ev):
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = e.get("kind", "")
                last_kind = k
                if k == "bootstrap_done":
                    bootstrapped = True
                elif k == "solver_turn_start":
                    turns += 1
                    if e.get("post_harden"):
                        post_harden += 1
                elif k == "run_stop":
                    stopped = True
                    hardenings = e.get("hardenings", hardenings)
                    post_harden = e.get("post_harden_solves", post_harden)
                    if e.get("best_score") is not None:
                        best = e.get("best_score")
        # prefer the finalized manifest numbers when present.
        hardenings = man.get("verifier_hardenings", hardenings)
        post_harden = man.get("post_harden_solves", post_harden)
        final_v = man.get("final_verifier_version",
                          "v?" if not bootstrapped else 0)
        return {"bootstrapped": bootstrapped, "solver_turns": turns,
                "hardenings": hardenings, "post_harden_solves": post_harden,
                "final_verifier_version": final_v, "best_score": best,
                "last_event": last_kind, "stopped": stopped}

    def stop(self) -> list[dict]:
        """Signal each run's process group (SIGTERM). Runs are then --resume-able."""
        out = []
        for r in self._read_manifest().get("runs", []):
            pid = r.get("pid")
            killed = False
            if pid and _alive(pid):
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    killed = True
                except (ProcessLookupError, PermissionError):
                    killed = False
            out.append({"run_id": r["run_id"], "pid": pid, "signalled": killed})
        return out

    # -- manifest io ------------------------------------------------------
    def _write_manifest(self, obj: dict) -> None:
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    def _read_manifest(self) -> dict:
        return _read_json(self.manifest_path, default={"batch": self.batch, "runs": []})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False          # no such process
    except PermissionError:
        return True           # exists but not ours to signal — still alive
    return True


def _read_json(path: Path, *, default):
    if not Path(path).is_file():
        return default
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _read_lines(path: Path) -> list[str]:
    try:
        return Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


def _print_status(rows: list[dict]) -> None:
    if not rows:
        print("(no runs in this batch — nothing launched yet?)")
        return
    hdr = (f"{'run_id':<34} {'alive':<5} {'boot':<4} {'turns':>5} "
           f"{'hard':>4} {'ph':>3} {'V':>3} {'best':>12} {'last_event':<20}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        best = r.get("best_score")
        best_s = "n/a" if best is None else f"{best:.4g}"
        print(f"{r['run_id']:<34} "
              f"{'yes' if r['alive'] else 'no':<5} "
              f"{'yes' if r['bootstrapped'] else '-':<4} "
              f"{r['solver_turns']:>5} {r['hardenings']:>4} "
              f"{r['post_harden_solves']:>3} {str(r['final_verifier_version']):>3} "
              f"{best_s:>12} {r['last_event']:<20}")


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Parallel launcher for agent-system case-study runs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("start", help="launch one run per input dir, all in parallel")
    sp.add_argument("inputs", nargs="+", help="raw-input dirs (one run each)")
    sp.add_argument("--batch", required=True, help="batch id (groups the runs)")
    sp.add_argument("--hours", type=float, default=8.0, help="wall-clock budget per run")
    sp.add_argument("--runs-dir", default="runs")
    sp.add_argument("--python", default=sys.executable)
    sp.add_argument("--dry-run", action="store_true",
                    help="resolve run-ids + argv and print the launch plan; spawn nothing")

    st = sub.add_parser("status", help="show a table of all runs in a batch")
    st.add_argument("--batch", required=True)
    st.add_argument("--runs-dir", default="runs")

    kp = sub.add_parser("stop", help="SIGTERM every run in a batch (resume-able after)")
    kp.add_argument("--batch", required=True)
    kp.add_argument("--runs-dir", default="runs")

    args = ap.parse_args(argv)
    batch = Batch(args.batch, runs_dir=Path(args.runs_dir))

    if args.cmd == "start":
        specs = batch.start([Path(p) for p in args.inputs], hours=args.hours,
                            python=args.python, dry_run=args.dry_run)
        if args.dry_run:
            print(f"\n[dry-run] {len(specs)} run(s) planned for batch {args.batch!r} "
                  f"({args.hours}h each) — nothing launched.")
            return
        print(f"launched {len(specs)} run(s) in batch {args.batch!r} "
              f"({args.hours}h each, parallel, --resume):")
        for s in specs:
            print(f"  {s.run_id:<34} <- {s.input_dir}   log: {s.log_path}")
        print(f"\nwatch:  python -m coscientist.coevo.launcher status "
              f"--batch {args.batch}")
    elif args.cmd == "status":
        _print_status(batch.status())
    elif args.cmd == "stop":
        for r in batch.stop():
            print(f"  {r['run_id']:<34} pid={r['pid']} "
                  f"{'signalled' if r['signalled'] else 'not running'}")


if __name__ == "__main__":
    main()
