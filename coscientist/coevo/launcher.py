"""Parallel launcher (§ case study) — run N raw problems as independent 8h experiments.

The user's case study: launch several unproven open problems at once, each an
independent agent-system run with its own wall-clock budget, all crash-resumable, with
one place to watch them from. This module is the orchestrator's *outer* control plane —
it owns nothing about any problem; it just spawns one ``coscientist.coevo.cli`` process
per input dir and tracks it.

Design (locked with the user):
  * **Wave scheduling on a fixed GPU pool** ("8 张全用,波次跑"). A batch owns a pool of
    GPU devices (default 8: devices 0..7). At most ``len(pool)`` runs execute at once,
    each pinned to a distinct free device; when a run finishes (done OR budget-spent) its
    device is freed and the next PENDING run is launched onto it. So 63 runs on 8 GPUs
    execute as ~8 waves. (The older "all in parallel" mode is the special case pool>=N.)
  * **Per-run GPU pin via argv, NOT env**: each child's docker containers pick their card
    from ``docker run --gpus device=N``, which the nvidia runtime reads from the CLI
    ``--solver-gpus``/``--verifier-gpus`` (these override resource.toml). The nvidia
    runtime does NOT read ``CUDA_VISIBLE_DEVICES``, so that env is useless here.
  * **Independent everything**: a distinct ``run_id`` (=> distinct ``runs/<id>/`` tree
    and distinct ``solver_ws``) per problem. Containers are docker-auto-named, so
    parallel runs never collide on a container name.
  * **Crash-resumable** ("断点续跑"): every child launches with ``--resume``, so a
    relaunch of the SAME command continues from ``runs/<id>/`` instead of re-authoring
    the evaluator. Launch is therefore idempotent — safe to re-run after a host reboot;
    the wave scheduler resumes in-flight runs AND keeps launching pending ones.
  * **No time-slicing of the budget**: the agent system itself guarantees a Solver turn
    after every harden (see ``AgentSystem.solve_and_evolve``); the launcher only sets
    the outer wall-clock ceiling and never carves it into phases.
  * **Centralized monitoring**: a launch manifest (``runs/<batch>/batch.json``) records
    each child's pid + run_id + status + device + log path; ``status`` reads each run's
    own ``manifest.json`` + ``events.jsonl`` to render one table.

Usage::

    # package raw K3 tasks into launchable problem dirs (idempotent)
    python -m coscientist.coevo.launcher package K3_gla_longseq K3_gdn2 ...

    # launch a batch on 8 GPUs, 4h each, wave-scheduled, resumable + supervised
    python -m coscientist.coevo.launcher start \\
        --batch k3_all_4h --hours 4 --gpus 8 --watch \\
        coscientist/coevo/problems/k3_gla_longseq coscientist/coevo/problems/...

    # watch them (reads disk; safe to run any time, from anywhere)
    python -m coscientist.coevo.launcher status --batch k3_all_4h

    # after a crash/reboot: exactly the same start command resumes each run in place
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
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


# a run holds its card while running OR while transiently held (a systematic-death run
# is about to be respawned on the SAME card, so its device must NOT be handed out).
_HOLDS_DEVICE = {"running", "held_systematic"}


def _occupied_devices(runs: list[dict]) -> set[str]:
    """Devices currently spoken for. ``held_systematic`` MUST count: such a run is
    mid-respawn onto its original card, and if a pending run grabbed that device two
    runs would collide on one GPU."""
    return {str(r["gpu_device"]) for r in runs
            if r.get("gpu_device") is not None and r.get("status") in _HOLDS_DEVICE}


def _free_devices(runs: list[dict], pool: list[str]) -> list[str]:
    """Pool order minus occupied — the cards a pending run may be launched onto."""
    occ = _occupied_devices(runs)
    return [d for d in pool if d not in occ]


@dataclass
class LaunchSpec:
    run_id: str
    input_dir: Path
    log_path: Path
    # Per-run resource config: an explicit --resource-config path (else the run relies
    # on a resource.toml auto-discovered inside input_dir), and a GPU device NUMBER (e.g.
    # "3") to pin this run to. The device is injected via ``--solver-gpus device=N
    # --verifier-gpus device=N`` on the child argv (see ``_build_argv``); those CLI flags
    # override resource.toml, and the nvidia container runtime reads the card from
    # ``docker run --gpus device=N`` — NOT from CUDA_VISIBLE_DEVICES. So a batch can place
    # each concurrent run on a distinct physical GPU purely through argv.
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
              gpu_devices: Optional[list[str]] = None,
              cpu_slots: Optional[int] = None,
              extra_args: Optional[list[str]] = None,
              dry_run: bool = False) -> list[LaunchSpec]:
        """Launch the FIRST WAVE and record the rest as pending (wave scheduling).

        A batch owns a fixed GPU pool ``gpu_devices`` (default ``["0".."7"]``). At most
        ``len(pool)`` runs execute at once, each pinned to a distinct free card; the
        remaining inputs are recorded ``status="pending"`` (no pid, budget NOT yet
        charged — ``deadline_epoch`` is anchored only at first launch). ``_wave_tick``
        (driven by ``watch``) then fills a card with the next pending run whenever one
        frees up. Returns the spec for every input (running + pending).

        **Idempotent resume**: if this batch's ``batch.json`` already exists, this does
        NOT re-plan. It loads the manifest and runs a single ``_wave_tick`` — resuming
        every in-flight run and continuing to launch pending ones — without resetting any
        status, re-anchoring any deadline, or clearing respawn counts. So the exact same
        ``start`` command is safe to re-run after a host reboot.

        ``dry_run=True`` builds the first-wave argv (each with its ``--*-gpus device=N``
        pin) and lists the pending queue, but spawns nothing and writes no manifest.

        **CPU-slot mode** (``cpu_slots=N``, mutually exclusive with ``gpu_devices``):
        the pool becomes ``["slot0".."slotN-1"]`` — pure concurrency tokens, not cards.
        ``_build_argv`` then skips the ``--*-gpus`` pin (the runs are gpus=0). Everything
        else — wave fill, respawn-on-same-slot, deadline anchoring — is identical, since
        the scheduler treats a device as an opaque token. ``cpu_mode`` is persisted to the
        manifest so a resume keeps skipping the GPU pin."""
        cpu_mode = cpu_slots is not None
        if cpu_mode:
            pool = [f"slot{i}" for i in range(int(cpu_slots))]
        else:
            pool = list(gpu_devices) if gpu_devices else [str(i) for i in range(8)]
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        budget_s = hours * 3600.0

        # -- idempotent resume: an existing manifest means "continue", not "re-plan" --
        if not dry_run and self.manifest_path.is_file():
            self._wave_tick()
            man = self._read_manifest()
            return [self._spec_of(r) for r in man.get("runs", [])]

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

        # the first wave gets a distinct card each; the rest queue as pending.
        wave = min(len(pool), len(specs))
        for i, spec in enumerate(specs):
            if i < wave:
                spec.gpu_device = pool[i]

        if dry_run:
            kind = "CPU slots" if cpu_mode else "GPU pool"
            print(f"[dry-run] {kind}: {pool}")
            print(f"[dry-run] first wave: {wave} run(s); pending: {len(specs) - wave}")
            for spec in specs[:wave]:
                argv = self._build_argv(spec, budget_s=budget_s, python=python,
                                        extra_args=extra_args or [], cpu_mode=cpu_mode)
                slot = spec.gpu_device
                tag = f"slot={slot}" if cpu_mode else f"dev={slot}"
                print(f"  RUNNING {tag}  {spec.run_id}\n"
                      f"    {' '.join(argv)}")
            for spec in specs[wave:]:
                print(f"  PENDING          {spec.run_id}  <- {spec.input_dir}")
            return specs

        records = []
        for i, spec in enumerate(specs):
            if i < wave:
                pid = self._spawn(spec, budget_s=budget_s, python=python,
                                  extra_args=extra_args or [], cpu_mode=cpu_mode)
                now = time.time()
                records.append({"run_id": spec.run_id,
                                "input_dir": str(spec.input_dir),
                                "log": str(spec.log_path), "pid": pid,
                                "status": "running",
                                "budget_s": budget_s,
                                # absolute wall-clock expiry: the TOTAL budget is anchored
                                # once, so respawns after a crash charge only the
                                # REMAINING time (a fresh child Deadline would reset it).
                                "deadline_epoch": now + budget_s,
                                "last_launch_epoch": now,
                                "respawns": 0,
                                "resource_config":
                                    str(spec.resource_config)
                                    if spec.resource_config else None,
                                "gpu_device": spec.gpu_device})
            else:
                # pending: no pid, no card, budget NOT yet charged (deadline anchored at
                # first launch inside _wave_tick so queue-wait doesn't eat the 4h).
                records.append({"run_id": spec.run_id,
                                "input_dir": str(spec.input_dir),
                                "log": str(spec.log_path), "pid": None,
                                "status": "pending",
                                "budget_s": budget_s,
                                "deadline_epoch": None,
                                "last_launch_epoch": None,
                                "respawns": 0,
                                "resource_config":
                                    str(spec.resource_config)
                                    if spec.resource_config else None,
                                "gpu_device": None})
        self._write_manifest({"batch": self.batch, "hours": hours,
                              "runs_dir": str(self.runs_dir.resolve()),
                              "python": python,
                              "gpu_devices": pool,
                              "cpu_mode": cpu_mode,
                              "extra_args": list(extra_args or []),
                              "runs": records})
        return specs

    def _spec_of(self, r: dict) -> LaunchSpec:
        """Reconstruct a LaunchSpec from a manifest run record."""
        return LaunchSpec(
            run_id=r["run_id"], input_dir=Path(r["input_dir"]),
            log_path=Path(r["log"]),
            resource_config=(Path(r["resource_config"])
                             if r.get("resource_config") else None),
            gpu_device=r.get("gpu_device"))

    def _build_argv(self, spec: LaunchSpec, *, budget_s: float, python: str,
                    extra_args: list[str], cpu_mode: bool = False) -> list[str]:
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
        if cpu_mode:
            # CPU-slot batch: spec.gpu_device holds a pure concurrency-slot token
            # (e.g. "slot3"), NOT a physical card. Do NOT emit --*-gpus — the runs are
            # gpus=0 (resource.toml has no [*].gpus), so pinning a device would be wrong
            # and the nvidia runtime would reject it. The slot token only gates how many
            # run at once; the OS schedules the containers' CPUs itself.
            pass
        elif spec.gpu_device is not None:
            # Pin BOTH the solver and verifier containers to this physical card. These
            # CLI flags override resource.toml (merge order defaults < resource.toml <
            # CLI), and the nvidia runtime reads `docker run --gpus device=N` from them.
            # This is the ONLY GPU-pin emission point, so an initial launch and every
            # --resume respawn recompute the same pin => a respawned run keeps its card.
            dev = f"device={spec.gpu_device}"
            argv += ["--solver-gpus", dev, "--verifier-gpus", dev]
        argv += list(extra_args)
        return argv

    def _spawn(self, spec: LaunchSpec, *, budget_s: float, python: str,
               extra_args: list[str], cpu_mode: bool = False) -> int:
        argv = self._build_argv(spec, budget_s=budget_s, python=python,
                                extra_args=extra_args, cpu_mode=cpu_mode)
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

    # -- wave scheduling: keep <=len(pool) runs alive, fill freed cards ------
    def _wave_tick(self, *, now: Optional[float] = None) -> list[dict]:
        """One supervision pass over a wave-scheduled batch. Disk-driven + idempotent,
        so a watcher restart resumes exactly where it left off.

        Four ordered phases:
          A. CLASSIFY + REAP each running/held run by pid liveness (state machine below).
             A terminal run (done / budget_spent / held_max_respawns) FREES its card
             (gpu_device -> None); a premature death RESPAWNS on the SAME card with the
             remaining budget; a too-fast death is held_systematic (keeps its card).
          B. RECOMPUTE the free-device list AFTER A's releases.
          C. FILL: while a card is free and a pending run remains, spawn it on that card
             at the FULL budget, anchoring its deadline now (queue-wait didn't charge).
          D. PERSIST all mutations.

        State machine (pid found dead):
          running  & run_stop                     -> done             (free card)
          running  & now>=deadline                -> budget_spent     (free card)
          running  & remaining<MIN_REMAINING      -> budget_spent     (free card)
          running  & ran_for<MIN_RUNTIME          -> held_systematic  (keep card)
          running  & respawns>=MAX_RESPAWNS       -> held_max_respawns(free card)
          running  & else                         -> running          (respawn, same card)
          held_systematic & ran_for>=MIN_RUNTIME  -> running          (respawn, same card)

        **Single watcher per batch.** A second, stale watcher could double-spawn a
        pending run onto a card that looks free to it. Run exactly one ``watch`` per
        batch (the CLI starts one; don't run a second concurrently).

        Returns one action dict per run for the watch loop's termination check.
        """
        now = time.time() if now is None else now
        _reap_children()      # reap our exited children so _alive doesn't see zombies
        man = self._read_manifest()
        runs = man.get("runs", [])
        pool = man.get("gpu_devices", [str(i) for i in range(8)])
        python = man.get("python", sys.executable)
        extra_args = man.get("extra_args", []) or []
        cpu_mode = bool(man.get("cpu_mode", False))
        actions: dict[str, dict] = {}
        changed = False

        # -- phase A: classify + reap running/held; free or respawn as the state says --
        for r in runs:
            run_id = r["run_id"]
            status = r.get("status", "running")
            if status == "pending":
                actions[run_id] = {"run_id": run_id, "action": "pending"}
                continue
            if status in ("done", "budget_spent", "held_max_respawns"):
                actions[run_id] = {"run_id": run_id, "action": status}
                continue
            if _alive(r.get("pid")):
                actions[run_id] = {"run_id": run_id, "action": "alive"}
                continue
            # dead pid (status running or held_systematic). Decide its fate.
            run_dir = self.runs_dir / run_id
            if self._progress(run_dir).get("stopped"):
                r["status"] = "done"
                r["gpu_device"] = None      # free the card
                changed = True
                actions[run_id] = {"run_id": run_id, "action": "done"}
                continue
            deadline = r.get("deadline_epoch")
            if deadline is not None and now >= deadline:
                r["status"] = "budget_spent"
                r["gpu_device"] = None
                changed = True
                actions[run_id] = {"run_id": run_id, "action": "budget_spent"}
                continue
            ran_for = now - (r.get("last_launch_epoch") or now)
            if ran_for < MIN_RUNTIME_FOR_RESPAWN_S:
                # systematic-looking death: hold on the SAME card, retry next tick.
                r["status"] = "held_systematic"
                changed = True
                actions[run_id] = {"run_id": run_id, "action": "held_systematic",
                                   "ran_for": round(ran_for, 1)}
                continue
            if r.get("respawns", 0) >= MAX_RESPAWNS:
                r["status"] = "held_max_respawns"
                r["gpu_device"] = None
                changed = True
                actions[run_id] = {"run_id": run_id, "action": "held_max_respawns"}
                continue
            remaining = ((deadline - now) if deadline is not None
                         else r.get("budget_s", 0.0))
            if remaining < MIN_REMAINING_TO_RESPAWN_S:
                r["status"] = "budget_spent"
                r["gpu_device"] = None
                changed = True
                actions[run_id] = {"run_id": run_id, "action": "budget_spent"}
                continue
            # premature death with budget + a card: respawn on the SAME device.
            spec = self._spec_of(r)
            pid = self._spawn(spec, budget_s=remaining, python=python,
                              extra_args=extra_args, cpu_mode=cpu_mode)
            r["pid"] = pid
            r["status"] = "running"
            r["last_launch_epoch"] = now
            r["respawns"] = r.get("respawns", 0) + 1
            changed = True
            actions[run_id] = {"run_id": run_id, "action": "respawned", "pid": pid,
                               "respawns": r["respawns"],
                               "remaining_s": round(remaining, 1)}

        # -- phase B: free cards, recomputed AFTER phase-A releases --------------
        free = _free_devices(runs, pool)

        # -- phase C: fill each free card with the next pending run (full budget) --
        for r in runs:
            if not free:
                break
            if r.get("status") != "pending":
                continue
            dev = free.pop(0)
            spec = self._spec_of(r)
            spec.gpu_device = dev
            pid = self._spawn(spec, budget_s=r.get("budget_s", 0.0), python=python,
                              extra_args=extra_args, cpu_mode=cpu_mode)
            r["pid"] = pid
            r["status"] = "running"
            r["gpu_device"] = dev
            r["deadline_epoch"] = now + r.get("budget_s", 0.0)  # anchor at first launch
            r["last_launch_epoch"] = now
            changed = True
            actions[r["run_id"]] = {"run_id": r["run_id"], "action": "launched",
                                    "pid": pid, "gpu_device": dev}

        # -- phase D: persist -----------------------------------------------------
        if changed:
            self._write_manifest(man)
        return [actions[r["run_id"]] for r in runs]

    def watch(self, *, interval_s: float = 30.0,
              max_ticks: Optional[int] = None) -> None:
        """Block, wave-scheduling the batch until every run reaches a terminal state.

        This is the outer control plane the batch runs under: launch the first wave,
        then ``watch`` keeps cards full — respawning crashed children with --resume and
        launching pending runs onto freed cards — until nothing is left to do. Exits
        only when no run is alive, respawn-eligible, OR still pending. Safe to Ctrl-C and
        re-run — all supervision state lives in the batch manifest on disk."""
        terminal = {"done", "budget_spent", "held_max_respawns"}
        ticks = 0
        while True:
            actions = self._wave_tick()
            # not-yet-terminal: still live, just (re)launched, transiently held, or queued.
            pending_kinds = {"alive", "respawned", "launched",
                             "held_systematic", "pending"}
            unfinished = [a for a in actions if a["action"] in pending_kinds]
            for a in actions:
                if a["action"] in ("respawned", "launched", "held_max_respawns"):
                    extra = ""
                    if a["action"] == "respawned":
                        extra = (f" (respawn #{a.get('respawns')}, "
                                 f"{a.get('remaining_s')}s left)")
                    elif a["action"] == "launched":
                        extra = f" (dev={a.get('gpu_device')})"
                    print(f"  [{time.strftime('%H:%M:%S')}] {a['run_id']}: "
                          f"{a['action']}{extra}")
            ticks += 1
            if not unfinished and all(a["action"] in terminal for a in actions):
                break
            if max_ticks is not None and ticks >= max_ticks:
                break
            time.sleep(max(1.0, interval_s))

    # -- monitoring -------------------------------------------------------
    def status(self) -> list[dict]:
        """One row per run: manifest schedule fields (status/gpu) + progress from disk."""
        man = self._read_manifest()
        rows = []
        for r in man.get("runs", []):
            run_id = r["run_id"]
            run_dir = self.runs_dir / run_id
            rows.append({**self._progress(run_dir), "run_id": run_id,
                         "pid": r.get("pid"), "alive": _alive(r.get("pid")),
                         "status": r.get("status", ""),
                         "gpu_device": r.get("gpu_device")})
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
    """True iff the pid is a live process. A ZOMBIE counts as DEAD.

    Subtlety that once deadlocked wave scheduling: ``os.kill(pid, 0)`` SUCCEEDS for a
    zombie (a child that exited but the parent hasn't reaped) — it's still in the
    process table — so a kill-0 liveness check would report a finished run as forever
    alive, and the scheduler would never free its card or launch a pending run. So we
    additionally read ``/proc/<pid>/stat`` and treat state ``Z`` as dead. The watcher
    also reaps its own children each tick (see ``_reap_children``) so they don't linger
    as zombies; this proc check is the belt-and-suspenders for pids that aren't our
    direct children (e.g. after a watcher restart)."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False          # no such process
    except PermissionError:
        return True           # exists but not ours to signal — still alive
    # exists — but a zombie is a finished process awaiting reap: treat as dead.
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        state = stat[stat.rfind(")") + 1:].split()[0]  # field after "(comm)"
        if state == "Z":
            return False
    except (OSError, IndexError):
        pass                  # no /proc (non-Linux) or race — fall back to kill-0 result
    return True


def _reap_children() -> None:
    """Reap any of OUR exited children (non-blocking) so finished runs don't linger as
    zombies. subprocess.Popen children become zombies on exit until the parent wait()s;
    the watcher never Popen.wait()s (it tracks liveness by pid via the manifest), so it
    must reap here or the process table fills with <defunct> entries that also keep
    ``os.kill(pid,0)`` succeeding."""
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return            # no children at all
        if pid == 0:
            return            # children exist but none have exited yet



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


def package_k3_task(task_name: str, *, tasks_root: Path, problems_root: Path,
                    template: Path, slug: Optional[str] = None) -> Path:
    """Copy a raw K3 task dir into a launchable problem dir + a resource.toml template.

    Idempotent: if ``<problems_root>/<slug>/resource.toml`` already exists, this is a
    no-op returning that dir — it will NOT clobber a dir a run may already be using.
    Otherwise it ``copytree``s the raw task (instruction.md / task.toml / environment /
    tests / calib.json — all byte-for-byte, no hand edits) then writes the gla
    ``resource.toml`` template verbatim. The template's ``gpus="device=0"`` is only a
    placeholder; the launcher overrides the card per-run via ``--*-gpus device=N``."""
    src = Path(tasks_root) / task_name
    if not src.is_dir():
        # allow passing a full path too.
        src = Path(task_name)
    if not src.is_dir():
        raise FileNotFoundError(f"raw K3 task dir not found: {task_name}")
    slug = slug or _slug(src.name)
    dst = Path(problems_root) / slug
    if (dst / "resource.toml").is_file():
        return dst            # already packaged — leave it (may be an in-flight run)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copytree(src, dst)
    shutil.copyfile(template, dst / "resource.toml")
    return dst


def _print_status(rows: list[dict]) -> None:
    if not rows:
        print("(no runs in this batch — nothing launched yet?)")
        return
    hdr = (f"{'run_id':<34} {'status':<11} {'gpu':>3} {'alive':<5} {'boot':<4} "
           f"{'turns':>5} {'hard':>4} {'ph':>3} {'V':>3} {'best':>12} "
           f"{'last_event':<20}")
    print(hdr)
    print("-" * len(hdr))
    tally: dict[str, int] = {}
    for r in rows:
        best = r.get("best_score")
        best_s = "n/a" if best is None else f"{best:.4g}"
        st = r.get("status", "") or "-"
        tally[st] = tally.get(st, 0) + 1
        gpu = r.get("gpu_device")
        print(f"{r['run_id']:<34} {st:<11} {str(gpu) if gpu is not None else '-':>3} "
              f"{'yes' if r['alive'] else 'no':<5} "
              f"{'yes' if r['bootstrapped'] else '-':<4} "
              f"{r['solver_turns']:>5} {r['hardenings']:>4} "
              f"{r['post_harden_solves']:>3} {str(r['final_verifier_version']):>3} "
              f"{best_s:>12} {r['last_event']:<20}")
    # summary footer: how many runs sit in each schedule state.
    order = ["running", "pending", "done", "budget_spent",
             "held_systematic", "held_max_respawns"]
    parts = [f"{k} {tally[k]}" for k in order if k in tally]
    parts += [f"{k} {v}" for k, v in tally.items() if k not in order]
    print("-" * len(hdr))
    print("  " + "   ".join(parts) + f"   (total {len(rows)})")


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Parallel launcher for agent-system case-study runs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("start", help="launch the first GPU wave; queue the rest pending")
    sp.add_argument("inputs", nargs="+", help="problem dirs (one run each)")
    sp.add_argument("--batch", required=True, help="batch id (groups the runs)")
    sp.add_argument("--hours", type=float, default=4.0, help="wall-clock budget per run")
    sp.add_argument("--gpus", type=int, default=8,
                    help="GPU pool size; devices 0..N-1 are wave-scheduled across runs")
    sp.add_argument("--cpu-slots", type=int, default=None,
                    help="CPU-slot mode: N opaque concurrency slots (no GPU pin) "
                         "wave-scheduled across runs. Set this for CPU-only batches "
                         "(e.g. AutoLab); overrides --gpus.")
    sp.add_argument("--runs-dir", default="runs")
    sp.add_argument("--python", default=sys.executable)
    sp.add_argument("--dry-run", action="store_true",
                    help="print the first-wave argv (each with its --*-gpus device=N "
                         "pin) + the pending queue; spawn nothing, write no manifest")
    sp.add_argument("--watch", action="store_true",
                    help="after launching, block and wave-schedule: respawn crashed "
                         "children with --resume and launch pending runs onto freed "
                         "cards until every run is done or its budget is spent.")
    sp.add_argument("--watch-interval-s", type=float, default=30.0,
                    help="seconds between wave-supervision passes when --watch is set")
    sp.add_argument("--extra-args", nargs="*", default=[],
                    help="extra flags appended verbatim to every child cli.py argv "
                         "(persisted in the manifest, reused on every --resume respawn). "
                         "NOTE: to pass a flag that starts with '-', use the '=' form, "
                         "e.g. --extra-args=--freeze-verifier (a bare "
                         "'--extra-args --freeze-verifier' makes argparse reject it).")

    st = sub.add_parser("status", help="show a table of all runs in a batch")
    st.add_argument("--batch", required=True)
    st.add_argument("--runs-dir", default="runs")

    wp = sub.add_parser("watch", help="wave-schedule a batch until every run finishes")
    wp.add_argument("--batch", required=True)
    wp.add_argument("--runs-dir", default="runs")
    wp.add_argument("--interval-s", type=float, default=30.0)

    kp = sub.add_parser("stop", help="SIGTERM every run in a batch (resume-able after)")
    kp.add_argument("--batch", required=True)
    kp.add_argument("--runs-dir", default="runs")

    pk = sub.add_parser("package",
                        help="copy raw K3 task dirs into launchable problem dirs "
                             "(idempotent; writes the gla resource.toml template)")
    pk.add_argument("tasks", nargs="+",
                    help="raw task names under the K3 tasks root (or full paths)")
    pk.add_argument("--tasks-root",
                    default="_k3_scratch/autolab_kernel_K3_tasks/tasks",
                    help="dir holding the raw K3 task dirs")
    pk.add_argument("--problems-root", default="coscientist/coevo/problems",
                    help="dir to write packaged problem dirs into")

    args = ap.parse_args(argv)

    if args.cmd == "package":
        template = (Path(__file__).resolve().parent
                    / "problems" / "k3_gla_longseq" / "resource.toml")
        for name in args.tasks:
            dst = package_k3_task(name, tasks_root=Path(args.tasks_root),
                                  problems_root=Path(args.problems_root),
                                  template=template)
            print(f"  {name:<40} -> {dst}")
        return

    batch = Batch(args.batch, runs_dir=Path(args.runs_dir))

    if args.cmd == "start":
        cpu_mode = args.cpu_slots is not None
        pool = ([f"slot{i}" for i in range(args.cpu_slots)] if cpu_mode
                else [str(i) for i in range(args.gpus)])
        specs = batch.start([Path(p) for p in args.inputs], hours=args.hours,
                            python=args.python, gpu_devices=pool,
                            cpu_slots=args.cpu_slots,
                            extra_args=list(args.extra_args or []),
                            dry_run=args.dry_run)
        if args.dry_run:
            kind = (f"pool of {args.cpu_slots} CPU slot(s)" if cpu_mode
                    else f"pool of {args.gpus} GPU(s)")
            print(f"\n[dry-run] {len(specs)} run(s) planned for batch {args.batch!r} "
                  f"({args.hours}h each, {kind}) — nothing launched.")
            return
        man = batch._read_manifest()
        running = [r for r in man.get("runs", []) if r.get("status") == "running"]
        pending = [r for r in man.get("runs", []) if r.get("status") == "pending"]
        print(f"batch {args.batch!r}: {len(running)} running (first wave), "
              f"{len(pending)} pending, {len(specs)} total "
              f"({args.hours}h each, pool {pool}, --resume):")
        for s in specs:
            print(f"  {s.run_id:<34} <- {s.input_dir}   log: {s.log_path}")
        print(f"\nstatus: python -m coscientist.coevo.launcher status "
              f"--batch {args.batch}")
        print(f"watch : python -m coscientist.coevo.launcher watch "
              f"--batch {args.batch}")
        if args.watch:
            print("\n[watch] wave-scheduling — respawning crashed children and filling "
                  "freed cards with pending runs (Ctrl-C to detach; runs keep going, "
                  "re-run watch anytime).")
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
