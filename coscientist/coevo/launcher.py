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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# -- anti-interruption knobs (ported from SForge's resume loop) --------------
# SForge keeps a run alive to its FULL wall-clock budget with three cooperating
# ideas (harness/run_agent.py:523-595): (ii) an outer respawn loop where a child
# exiting BEFORE its budget is treated as premature and relaunched with --resume;
# a MIN-RUNTIME floor so a child that dies in seconds (systematic failure: bad
# ELF/creds) is NOT hot-looped; and a hard respawn cap. We mirror all three, but
# decrement the budget by real wall-clock elapsed so the TOTAL stays ~budget_s
# across crashes (a fresh Deadline would otherwise grant a full budget on resume).
MIN_RUNTIME_FOR_RESPAWN_S = 45.0   # died faster than this since last launch => systematic
MAX_RESPAWNS = 100                 # backstop against a crash-loop (SForge uses 100)
MIN_REMAINING_TO_RESPAWN_S = 90.0  # don't relaunch for a sliver of budget



def _slug(text: str) -> str:
    """A filesystem/run-id-safe slug from an input path's basename."""
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(text).name).strip("_").lower()
    return s or "problem"


@dataclass
class LaunchSpec:
    run_id: str
    input_dir: Path
    log_path: Path
    # Per-run resource config: an explicit --resource-config path (else the run relies
    # on a resource.toml auto-discovered inside input_dir), and a GPU device to pin via
    # CUDA_VISIBLE_DEVICES so a batch can place each run on a different GPU.
    resource_config: Optional[Path] = None
    gpu_device: Optional[str] = None


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
            now = time.time()
            records.append({"run_id": spec.run_id, "input_dir": str(spec.input_dir),
                            "log": str(spec.log_path), "pid": pid,
                            "budget_s": budget_s,
                            # absolute wall-clock expiry: the TOTAL budget is anchored
                            # once, so respawns after a crash charge only the REMAINING
                            # time (a fresh child Deadline would otherwise reset it).
                            "deadline_epoch": now + budget_s,
                            "last_launch_epoch": now,
                            "respawns": 0,
                            "resource_config":
                                str(spec.resource_config) if spec.resource_config else None,
                            "gpu_device": spec.gpu_device})
        self._write_manifest({"batch": self.batch, "hours": hours,
                              "runs_dir": str(self.runs_dir.resolve()),
                              "python": python,
                              "extra_args": list(extra_args or []),
                              "runs": records})
        return specs

    def _build_argv(self, spec: LaunchSpec, *, budget_s: float, python: str,
                    extra_args: list[str]) -> list[str]:
        argv = [python, "-m", "coscientist.coevo.cli",
                "--input", str(spec.input_dir),
                "--solver", "codex", "--supervisor", "no-human-no-proxy",
                "--budget-s", str(budget_s),
                # The wall-clock budget is the ONLY intended stop signal (SForge's
                # "timeout = done" philosophy). The default max_turns=12 would end an
                # 8h run in minutes, so lift the turn cap far above any real run —
                # solve_and_evolve then loops until self.deadline.expired().
                "--max-turns", "100000",
                "--resume",
                "--runs-dir", str(self.runs_dir),
                "--run-id", spec.run_id]
        if spec.resource_config is not None:
            argv += ["--resource-config", str(spec.resource_config)]
        argv += list(extra_args)
        return argv

    def _spawn(self, spec: LaunchSpec, *, budget_s: float, python: str,
               extra_args: list[str]) -> int:
        argv = self._build_argv(spec, budget_s=budget_s, python=python,
                                extra_args=extra_args)
        env = dict(os.environ)
        env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
        if spec.gpu_device is not None:
            # Pin this run to a specific GPU so a batch can spread across devices.
            env["CUDA_VISIBLE_DEVICES"] = str(spec.gpu_device)
        log = spec.log_path.open("a", encoding="utf-8")
        log.write(f"\n===== launch {spec.run_id} (budget {budget_s:.0f}s, resume) =====\n")
        log.flush()
        # detached session: survives the launcher exiting; own process group so a
        # future `stop` can signal the whole tree.
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env,
                                 cwd=str(Path(__file__).resolve().parents[2]))
        return proc.pid

    # -- anti-interruption: respawn crashed children until budget is spent ----
    def _respawn_tick(self, *, now: Optional[float] = None) -> list[dict]:
        """One supervision pass: relaunch any dead-but-not-done run with --resume.

        SForge's insight (harness/run_agent.py:523-595): a child exiting BEFORE its
        wall-clock budget is *premature*, not *done* — so respawn it (state is on
        disk; --resume rebuilds the V-chain + best-so-far). Only three things stop a
        respawn: the run finished cleanly (a ``run_stop`` event), the absolute budget
        deadline passed, or the child kept dying too fast / too often (systematic
        failure — see the MIN_RUNTIME / MAX_RESPAWNS floors). Idempotent and
        disk-driven, so the watcher itself is crash-safe: re-reading the manifest
        after a watcher restart resumes supervision exactly where it left off.

        Returns one action dict per run describing what happened this tick.
        """
        now = time.time() if now is None else now
        man = self._read_manifest()
        runs = man.get("runs", [])
        python = man.get("python", sys.executable)
        extra_args = man.get("extra_args", []) or []
        actions = []
        changed = False
        for r in runs:
            run_id = r["run_id"]
            action = {"run_id": run_id, "action": "none"}
            if _alive(r.get("pid")):
                action["action"] = "alive"
                actions.append(action)
                continue
            # dead child. Was it DONE (clean stop) or did it die prematurely?
            run_dir = self.runs_dir / run_id
            if self._progress(run_dir).get("stopped"):
                action["action"] = "done"
                actions.append(action)
                continue
            deadline = r.get("deadline_epoch")
            if deadline is not None and now >= deadline:
                action["action"] = "budget_spent"
                actions.append(action)
                continue
            # premature death. Guard against a crash-loop before relaunching.
            ran_for = now - r.get("last_launch_epoch", now)
            if ran_for < MIN_RUNTIME_FOR_RESPAWN_S:
                action.update(action="held_systematic", ran_for=round(ran_for, 1))
                actions.append(action)
                continue
            if r.get("respawns", 0) >= MAX_RESPAWNS:
                action["action"] = "held_max_respawns"
                actions.append(action)
                continue
            remaining = (deadline - now) if deadline is not None else r.get("budget_s", 0.0)
            if remaining < MIN_REMAINING_TO_RESPAWN_S:
                action["action"] = "budget_spent"
                actions.append(action)
                continue
            # relaunch with the REMAINING budget so the total stays ~budget_s.
            spec = LaunchSpec(
                run_id=run_id, input_dir=Path(r["input_dir"]),
                log_path=Path(r["log"]),
                resource_config=(Path(r["resource_config"])
                                 if r.get("resource_config") else None),
                gpu_device=r.get("gpu_device"))
            pid = self._spawn(spec, budget_s=remaining, python=python,
                              extra_args=extra_args)
            r["pid"] = pid
            r["last_launch_epoch"] = now
            r["respawns"] = r.get("respawns", 0) + 1
            changed = True
            action.update(action="respawned", pid=pid, respawns=r["respawns"],
                          remaining_s=round(remaining, 1))
            actions.append(action)
        if changed:
            self._write_manifest(man)
        return actions

    def watch(self, *, interval_s: float = 30.0,
              max_ticks: Optional[int] = None) -> None:
        """Block, respawning crashed children until every run is done or budget-spent.

        This is the outer control plane the 8h batch runs under: launch, then
        ``watch`` keeps them alive to the full wall clock. Exits when no run is still
        live AND none is respawn-eligible (all done or all budget-spent). Safe to Ctrl-C
        and re-run — supervision state lives in the batch manifest on disk."""
        terminal = {"done", "budget_spent", "held_max_respawns"}
        ticks = 0
        while True:
            actions = self._respawn_tick()
            live = [a for a in actions if a["action"] in ("alive", "respawned")]
            held = [a for a in actions if a["action"] == "held_systematic"]
            for a in actions:
                if a["action"] in ("respawned", "held_max_respawns"):
                    print(f"  [{time.strftime('%H:%M:%S')}] {a['run_id']}: {a['action']}"
                          + (f" (respawn #{a.get('respawns')}, "
                             f"{a.get('remaining_s')}s left)"
                             if a["action"] == "respawned" else ""))
            ticks += 1
            # done when nothing is alive/respawned and nothing is transiently held
            # (a held_systematic child may still cross the MIN_RUNTIME floor next tick).
            if not live and not held:
                if all(a["action"] in terminal for a in actions):
                    break
            if max_ticks is not None and ticks >= max_ticks:
                break
            time.sleep(max(1.0, interval_s))

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
    sp.add_argument("--watch", action="store_true",
                    help="after launching, block and respawn any crashed child with "
                         "--resume until every run is done or its budget is spent "
                         "(the anti-interruption control plane for 8h runs).")
    sp.add_argument("--watch-interval-s", type=float, default=30.0,
                    help="seconds between respawn-supervision passes when --watch is set")

    st = sub.add_parser("status", help="show a table of all runs in a batch")
    st.add_argument("--batch", required=True)
    st.add_argument("--runs-dir", default="runs")

    wp = sub.add_parser("watch", help="respawn crashed children until budget is spent")
    wp.add_argument("--batch", required=True)
    wp.add_argument("--runs-dir", default="runs")
    wp.add_argument("--interval-s", type=float, default=30.0)

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
        print(f"\nstatus: python -m coscientist.coevo.launcher status "
              f"--batch {args.batch}")
        print(f"watch : python -m coscientist.coevo.launcher watch "
              f"--batch {args.batch}")
        if args.watch:
            print("\n[watch] supervising — respawning crashed children until budget "
                  "is spent (Ctrl-C to detach; runs keep going, re-run watch anytime).")
            batch.watch(interval_s=args.watch_interval_s)
    elif args.cmd == "status":
        _print_status(batch.status())
    elif args.cmd == "watch":
        print(f"[watch] supervising batch {args.batch!r} — Ctrl-C to detach.")
        batch.watch(interval_s=args.interval_s)
    elif args.cmd == "stop":
        for r in batch.stop():
            print(f"  {r['run_id']:<34} pid={r['pid']} "
                  f"{'signalled' if r['signalled'] else 'not running'}")


if __name__ == "__main__":
    main()
