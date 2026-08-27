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
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
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
BATCH_SCHEMA_VERSION = 3
RUN_CONTRACT_SCHEMA_VERSION = 2
FREEZE_VERIFIER_SCHEMA_VERSION = 3



def _slug(text: str) -> str:
    """A filesystem/run-id-safe slug from an input path's basename."""
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(text).name).strip("_").lower()
    return s or "problem"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    """Whether resolved ``path`` is ``parent`` or one of its descendants."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _symlink_component(path: Path) -> Optional[Path]:
    """Return the first symlink in an absolute path, including its final component."""
    absolute = Path(path).absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            return current
    return None


# ``extra_args`` is intentionally narrow: allowing arbitrary child-parser options lets
# argparse abbreviations (``--bud``) or a later duplicate silently override the audited
# launcher contract. These are the exact non-contract tuning flags a caller may still
# append. Everything else must become a first-class launcher option before use.
_ALLOWED_EXTRA_ARG_FLAGS = {
    "--bootstrap-timeout-s", "--harden-timeout-s", "--post-harden-solve-s",
    "--concurrency", "--max-generations", "--gen-turn-s", "--llm-config",
    "--solver-image", "--solver-gpus", "--resource-config", "--solver-cpus",
    "--solver-memory-mb", "--solver-allow-internet", "--verifier-image",
    "--verifier-gpus", "--verifier-cpus", "--verifier-memory-mb",
    "--verifier-timeout-s", "--verifier-allow-internet",
}

_PROTECTED_CHILD_FLAGS = {
    "--input", "--solver", "--supervisor", "--budget-s", "--budget-hours",
    "--max-turns", "--resume", "--runs-dir", "--run-id", "--freeze-verifier",
    "--feedback", "--solver-strength", "--human-proxy-context",
    "--human-proxy-context-sha256",
    "--human-agent-timeout-s", "--human-agent-model",
    "--human-agent-reasoning-effort",
}


def _validate_extra_args(args: list[str]) -> None:
    seen: set[str] = set()
    for token in args:
        if not token.startswith("--"):
            continue
        key = token.split("=", 1)[0]
        if key in seen:
            raise ValueError(f"duplicate option in extra args: {key}")
        seen.add(key)
        if any(flag.startswith(key) for flag in _PROTECTED_CHILD_FLAGS):
            raise ValueError(
                f"extra args may not override/abbreviate protected first-class flag: "
                f"{key}")
        if key not in _ALLOWED_EXTRA_ARG_FLAGS:
            raise ValueError(
                f"extra args option must use an exact allowlisted flag (no argparse "
                f"abbreviation): {key}")


def _resolve_solver_identity() -> dict[str, str]:
    """Resolve the identity the Solver container will actually inherit from Codex.

    Launcher Solver flags are an assertion over this host-derived gateway, not merely
    labels written to the manifest: the child CLI builds the same ``GatewayConfig``
    from the same host at startup.
    """
    from .container import GatewayConfig

    gateway = GatewayConfig.from_host()
    if gateway is None:
        raise ValueError("cannot resolve Solver identity: ~/.codex/auth.json is absent")
    return {"model": gateway.model, "reasoning_effort": gateway.reasoning_effort}


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
    # A context is private evaluator evidence for this task's independent Human Proxy.
    # Store both the absolute path and content identity so no queued/resumed child can
    # silently see different evaluator bytes.
    human_proxy_context: Optional[Path] = None
    human_proxy_context_sha256: Optional[str] = None
    human_proxy_context_dir: Optional[Path] = None
    batch_input_dirs: tuple[Path, ...] = ()
    human_agent_timeout_s: Optional[float] = None
    human_agent_model: Optional[str] = None
    human_agent_reasoning_effort: Optional[str] = None
    solver_model: Optional[str] = None
    solver_reasoning_effort: Optional[str] = None
    feedback: str = "with_artifacts"
    solver_strength: str = "weak"
    freeze_verifier: bool = False


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
              human_proxy_context_dir: Optional[Path] = None,
              human_agent_timeout_s: Optional[float] = None,
              human_agent_model: Optional[str] = None,
              human_agent_reasoning_effort: Optional[str] = None,
              solver_model: Optional[str] = None,
              solver_reasoning_effort: Optional[str] = None,
              feedback: str = "with_artifacts",
              solver_strength: str = "weak",
              freeze_verifier: bool = False,
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
        budget_s = hours * 3600.0
        child_extra_args = list(extra_args or [])
        _validate_extra_args(child_extra_args)
        if feedback not in {"score_only", "feasible_score", "with_artifacts"}:
            raise ValueError(f"unsupported feedback level: {feedback}")
        if solver_strength not in {"weak", "strong"}:
            raise ValueError(f"unsupported solver strength: {solver_strength}")

        if (solver_model is None) != (solver_reasoning_effort is None):
            raise ValueError("Solver identity requires both model and reasoning effort")
        solver_identity = None
        if solver_model is not None:
            resolved_solver = _resolve_solver_identity()
            requested_solver = {
                "model": str(solver_model),
                "reasoning_effort": str(solver_reasoning_effort),
            }
            if resolved_solver != requested_solver:
                raise ValueError(
                    "Solver identity contract mismatch: requested "
                    f"{requested_solver}, host resolves {resolved_solver}")
            solver_identity = resolved_solver

        resolved_inputs = [Path(inp).resolve() for inp in inputs]
        context_root, contexts = self._resolve_human_proxy_contexts(
            resolved_inputs, human_proxy_context_dir)
        human_agent = {
            "timeout_s": human_agent_timeout_s,
            "model": human_agent_model,
            "reasoning_effort": human_agent_reasoning_effort,
        }

        # unique, stable run_ids: <batch>__<slug>[ _2, _3 ... on collision].
        seen: dict[str, int] = {}
        specs: list[LaunchSpec] = []
        for inp, context in zip(resolved_inputs, contexts):
            base = f"{self.batch}__{_slug(str(inp))}"
            n = seen.get(base, 0)
            seen[base] = n + 1
            run_id = base if n == 0 else f"{base}_{n+1}"
            specs.append(LaunchSpec(
                run_id=run_id, input_dir=inp,
                log_path=self.batch_dir / f"{run_id}.log",
                human_proxy_context=context[0] if context else None,
                human_proxy_context_sha256=context[1] if context else None,
                human_proxy_context_dir=context_root,
                batch_input_dirs=tuple(resolved_inputs),
                human_agent_timeout_s=human_agent_timeout_s,
                human_agent_model=human_agent_model,
                human_agent_reasoning_effort=human_agent_reasoning_effort,
                solver_model=(solver_identity or {}).get("model"),
                solver_reasoning_effort=(solver_identity or {}).get(
                    "reasoning_effort"),
                feedback=feedback,
                solver_strength=solver_strength,
                freeze_verifier=freeze_verifier))

        # -- idempotent resume: continue only an EXACT persisted launch contract. --
        if not dry_run and self.manifest_path.is_file():
            man = self._read_manifest()
            self._validate_resume_contract(
                man, specs=specs, hours=hours, python=python, pool=pool,
                cpu_mode=cpu_mode, extra_args=child_extra_args,
                human_agent=human_agent, solver_identity=solver_identity,
                context_root=context_root, inputs=resolved_inputs,
                feedback=feedback, solver_strength=solver_strength,
                freeze_verifier=freeze_verifier)
            self._verify_all_run_manifest_contracts(man)
            # Verify current bytes even if every child is still live: this makes an
            # idempotent start a useful fail-closed contract check of the whole batch.
            for spec in specs:
                self._verify_human_proxy_context(spec)
            self._wave_tick()
            man = self._read_manifest()
            return [self._spec_of(r) for r in man.get("runs", [])]

        self.batch_dir.mkdir(parents=True, exist_ok=True)

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
                                        extra_args=child_extra_args,
                                        cpu_mode=cpu_mode)
                slot = spec.gpu_device
                tag = f"slot={slot}" if cpu_mode else f"dev={slot}"
                print(f"  RUNNING {tag}  {spec.run_id}\n"
                      f"    {' '.join(argv)}")
            for spec in specs[wave:]:
                print(f"  PENDING          {spec.run_id}  <- {spec.input_dir}")
            for spec in specs:
                context = (str(spec.human_proxy_context)
                           if spec.human_proxy_context else "-")
                digest = spec.human_proxy_context_sha256 or "-"
                print(f"    CONTRACT {spec.run_id} context={context} sha256={digest}")
            return specs

        # Persist the complete contract BEFORE the first child exists. Every _spawn,
        # including the first wave, can therefore compare its run-level settings to
        # batch.json. Runs remain pending until Popen succeeds, so a failed launch is
        # recoverable by the watcher instead of becoming an untracked contract-less job.
        records = [{"run_id": spec.run_id,
                    "input_dir": str(spec.input_dir),
                    "log": str(spec.log_path), "pid": None,
                    "status": "pending",
                    "budget_s": budget_s,
                    "deadline_epoch": None,
                    "last_launch_epoch": None,
                    "respawns": 0,
                    "resource_config": (str(spec.resource_config)
                                        if spec.resource_config else None),
                    "gpu_device": None,
                    **self._human_proxy_record(spec)}
                   for spec in specs]
        manifest = {"schema_version": BATCH_SCHEMA_VERSION,
                    "batch": self.batch, "hours": hours,
                    "runs_dir": str(self.runs_dir.resolve()),
                    "python": python,
                    "gpu_devices": pool,
                    "cpu_mode": cpu_mode,
                    "extra_args": child_extra_args,
                    "human_agent": human_agent,
                    "solver": solver_identity,
                    "human_proxy_context_dir": (
                        str(context_root) if context_root else None),
                    "inputs": [str(p) for p in resolved_inputs],
                    "feedback": feedback,
                    "solver_strength": solver_strength,
                    "freeze_verifier": freeze_verifier,
                    "run_contracts": {
                        spec.run_id: {
                            "input_dir": str(spec.input_dir),
                            "human_proxy_context": (
                                str(spec.human_proxy_context)
                                if spec.human_proxy_context else None),
                            "human_proxy_context_sha256": (
                                spec.human_proxy_context_sha256),
                            "freeze_verifier": spec.freeze_verifier,
                        }
                        for spec in specs
                    },
                    "runs": records}
        self._write_manifest(manifest)
        for i, spec in enumerate(specs[:wave]):
            pid = self._spawn(spec, budget_s=budget_s, python=python,
                              extra_args=child_extra_args, cpu_mode=cpu_mode)
            now = time.time()
            records[i].update({
                "pid": pid,
                "status": "running",
                # The TOTAL budget is anchored once; respawns receive only remaining.
                "deadline_epoch": now + budget_s,
                "last_launch_epoch": now,
                "gpu_device": spec.gpu_device,
            })
            self._write_manifest(manifest)
        return specs

    def _resolve_human_proxy_contexts(
            self, inputs: list[Path], context_dir: Optional[Path]
    ) -> tuple[Optional[Path], list[Optional[tuple[Path, str]]]]:
        if context_dir is None:
            return None, [None for _ in inputs]

        raw_dir = Path(context_dir).absolute()
        linked = _symlink_component(raw_dir)
        if linked is not None:
            raise ValueError(
                f"Human Proxy context path must have no symlink ancestor: {linked}")
        if not raw_dir.is_dir():
            raise ValueError(f"Human Proxy context directory does not exist: {raw_dir}")
        resolved_dir = raw_dir.resolve()
        runs_root = self.runs_dir.resolve()
        if _inside(resolved_dir, runs_root):
            raise ValueError(
                f"Human Proxy context directory must be outside runs/: {resolved_dir}")
        for inp in inputs:
            if _inside(resolved_dir, inp):
                raise ValueError(
                    "Human Proxy context directory must be outside every input: "
                    f"{resolved_dir} is within {inp}")

        resolved: list[Optional[tuple[Path, str]]] = []
        for inp in inputs:
            raw_context = raw_dir / f"{inp.name}.md"
            linked = _symlink_component(raw_context)
            if linked is not None:
                raise ValueError(
                    f"Human Proxy context path must have no symlink ancestor: {linked}")
            if not raw_context.is_file():
                raise ValueError(
                    f"Human Proxy context for {inp.name} is missing or not a regular "
                    f"file: {raw_context}")
            context = raw_context.resolve()
            # A race or unusual mount must not resolve the selected file elsewhere.
            if not _inside(context, resolved_dir):
                raise ValueError(
                    f"Human Proxy context resolved outside its directory: {raw_context}")
            resolved.append((context, _sha256_file(context)))
        return resolved_dir, resolved

    @staticmethod
    def _human_proxy_record(spec: LaunchSpec) -> dict:
        return {
            "human_proxy_context": (
                str(spec.human_proxy_context) if spec.human_proxy_context else None),
            "human_proxy_context_sha256": spec.human_proxy_context_sha256,
            "human_proxy_context_dir": (
                str(spec.human_proxy_context_dir)
                if spec.human_proxy_context_dir else None),
            "batch_input_dirs": [str(p) for p in spec.batch_input_dirs],
            "human_agent_timeout_s": spec.human_agent_timeout_s,
            "human_agent_model": spec.human_agent_model,
            "human_agent_reasoning_effort": spec.human_agent_reasoning_effort,
            "solver_model": spec.solver_model,
            "solver_reasoning_effort": spec.solver_reasoning_effort,
            "feedback": spec.feedback,
            "solver_strength": spec.solver_strength,
            "freeze_verifier": spec.freeze_verifier,
        }

    def _validate_resume_contract(
            self, man: dict, *, specs: list[LaunchSpec], hours: float, python: str,
            pool: list[str], cpu_mode: bool, extra_args: list[str],
            human_agent: dict, solver_identity: Optional[dict[str, str]],
            context_root: Optional[Path], inputs: list[Path], feedback: str,
            solver_strength: str, freeze_verifier: bool) -> None:
        persisted_runs = man.get("runs", [])
        persisted_contexts = [
            {
                "input_dir": str(Path(r["input_dir"]).resolve()),
                "context": r.get("human_proxy_context"),
                "sha256": r.get("human_proxy_context_sha256"),
            }
            for r in persisted_runs
        ]
        requested_contexts = [
            {
                "input_dir": str(s.input_dir),
                "context": (str(s.human_proxy_context)
                            if s.human_proxy_context else None),
                "sha256": s.human_proxy_context_sha256,
            }
            for s in specs
        ]
        persisted_human = man.get("human_agent", {
            "timeout_s": None, "model": None, "reasoning_effort": None})
        checks = {
            "inputs/context mapping": (persisted_contexts, requested_contexts),
            "hours": (float(man.get("hours", 0.0)), float(hours)),
            "CPU/GPU mode": (bool(man.get("cpu_mode", False)), bool(cpu_mode)),
            "CPU/GPU pool": (list(man.get("gpu_devices", [])), list(pool)),
            "python": (man.get("python", sys.executable), python),
            "extra args": (list(man.get("extra_args", []) or []), extra_args),
            "Human Agent": (persisted_human, human_agent),
            "Solver identity": (man.get("solver"), solver_identity),
            "context directory": (
                man.get("human_proxy_context_dir"),
                str(context_root) if context_root else None),
            "inputs": (man.get("inputs", [r["input_dir"] for r in persisted_runs]),
                       [str(p) for p in inputs]),
            "feedback": (man.get("feedback", "with_artifacts"), feedback),
            "solver strength": (man.get("solver_strength", "weak"),
                                solver_strength),
            "freeze verifier": (bool(man.get("freeze_verifier", False)),
                                bool(freeze_verifier)),
        }
        for name, (persisted, requested) in checks.items():
            if persisted != requested:
                raise ValueError(
                    f"existing batch launch contract mismatch for {name}: "
                    f"persisted={persisted!r}, requested={requested!r}")

    def _spec_of(self, r: dict) -> LaunchSpec:
        """Reconstruct a LaunchSpec from a manifest run record."""
        return LaunchSpec(
            run_id=r["run_id"], input_dir=Path(r["input_dir"]),
            log_path=Path(r["log"]),
            resource_config=(Path(r["resource_config"])
                             if r.get("resource_config") else None),
            gpu_device=r.get("gpu_device"),
            human_proxy_context=(Path(r["human_proxy_context"])
                                 if r.get("human_proxy_context") else None),
            human_proxy_context_sha256=r.get("human_proxy_context_sha256"),
            human_proxy_context_dir=(Path(r["human_proxy_context_dir"])
                                     if r.get("human_proxy_context_dir") else None),
            batch_input_dirs=tuple(Path(p) for p in r.get("batch_input_dirs", [])),
            human_agent_timeout_s=r.get("human_agent_timeout_s"),
            human_agent_model=r.get("human_agent_model"),
            human_agent_reasoning_effort=r.get(
                "human_agent_reasoning_effort"),
            solver_model=r.get("solver_model"),
            solver_reasoning_effort=r.get("solver_reasoning_effort"),
            feedback=r.get("feedback", "with_artifacts"),
            solver_strength=r.get("solver_strength", "weak"),
            freeze_verifier=bool(r.get("freeze_verifier", False)))

    def _verify_human_proxy_context(self, spec: LaunchSpec) -> None:
        context = spec.human_proxy_context
        expected = spec.human_proxy_context_sha256
        if context is None and expected is None:
            return
        if context is None or expected is None:
            raise ValueError(
                f"incomplete Human Proxy context contract for {spec.run_id}")
        context_root = spec.human_proxy_context_dir
        if context_root is None:
            raise ValueError(
                f"missing persisted Human Proxy context directory for {spec.run_id}")
        linked = _symlink_component(context)
        if linked is not None:
            raise ValueError(
                f"Human Proxy context path has symlink component for {spec.run_id}: "
                f"{linked}")
        if not context.is_file():
            raise ValueError(
                f"Human Proxy context missing or not a regular non-symlink file for "
                f"{spec.run_id}: {context}")
        actual = context.resolve()
        # Compare against the immutable lexical roots captured before launch. Do not
        # resolve context_root again: a replaced parent symlink must be detected as an
        # escape, not silently accepted as the new root.
        if not _inside(actual, context_root):
            raise ValueError(
                f"Human Proxy context resolved outside persisted context directory "
                f"for {spec.run_id}: {actual} not under {context_root}")
        runs_lexical = self.runs_dir.absolute()
        runs_actual = self.runs_dir.resolve()
        if _inside(context.absolute(), runs_lexical) or _inside(actual, runs_actual):
            raise ValueError(
                f"Human Proxy context must remain outside runs/ for {spec.run_id}")
        for inp in spec.batch_input_dirs:
            if _inside(context.absolute(), inp) or _inside(actual, inp.resolve()):
                raise ValueError(
                    f"Human Proxy context must remain outside inputs for {spec.run_id}: "
                    f"{inp}")
        actual = _sha256_file(context)
        if actual != expected:
            raise ValueError(
                f"Human Proxy context SHA-256 drift for {spec.run_id}: "
                f"expected {expected}, got {actual}")

    @staticmethod
    def _verify_solver_identity(spec: LaunchSpec) -> None:
        if spec.solver_model is None and spec.solver_reasoning_effort is None:
            return
        expected = {
            "model": spec.solver_model,
            "reasoning_effort": spec.solver_reasoning_effort,
        }
        actual = _resolve_solver_identity()
        if actual != expected:
            raise ValueError(
                f"Solver identity drift for {spec.run_id}: expected {expected}, "
                f"host resolves {actual}")

    def _verify_spawn_manifest_contract(self, spec: LaunchSpec) -> None:
        """Fail closed when a standalone watcher sees root/run contract drift."""
        if not self.manifest_path.is_file():
            # start() persists the complete contract before even the first-wave Popen.
            # Therefore absence is always corruption/deletion, never a valid launch.
            raise ValueError(
                f"batch manifest missing before spawn: {self.manifest_path}")
        man = self._read_manifest()
        self._verify_run_manifest_contract(man, spec)

    def _verify_all_run_manifest_contracts(self, man: Optional[dict] = None) -> None:
        """Preflight every immutable run contract from one manifest snapshot."""
        snapshot = self._read_manifest() if man is None else man
        for record in snapshot.get("runs", []):
            self._verify_run_manifest_contract(snapshot, self._spec_of(record))

    def _verify_run_manifest_contract(self, man: dict, spec: LaunchSpec) -> None:
        """Compare one reconstructed run against its batch and immutable contract."""
        record = next(
            (r for r in man.get("runs", []) if r.get("run_id") == spec.run_id), None)
        if record is None:
            raise ValueError(
                f"spawn contract has no run record for {spec.run_id}")
        schema_version = int(man.get("schema_version", 1))
        if schema_version < RUN_CONTRACT_SCHEMA_VERSION:
            has_legacy_hp = bool(man.get("human_proxy_context_dir")) or any(
                r.get("human_proxy_context") or r.get("human_proxy_context_sha256")
                for r in man.get("runs", []))
            if has_legacy_hp:
                raise ValueError(
                    "legacy batch with Human Proxy data lacks an independent run "
                    "contract; recreate it with the current launcher")
            if spec.human_proxy_context is not None or \
                    spec.human_proxy_context_sha256 is not None:
                raise ValueError("legacy no-HP batch cannot spawn an HP run")
            if spec.freeze_verifier or bool(record.get("freeze_verifier", False)):
                raise ValueError(
                    "legacy batch cannot spawn a frozen-verifier run without an "
                    "independent run contract")
            if str(Path(record.get("input_dir", "")).resolve()) != str(spec.input_dir):
                raise ValueError(
                    f"legacy run input contract mismatch for {spec.run_id}")
            return

        root_contract = man.get("run_contracts", {}).get(spec.run_id)
        if not isinstance(root_contract, dict):
            raise ValueError(
                f"top-level run contract missing for {spec.run_id}")
        root_contract = dict(root_contract)
        root_contract["freeze_verifier"] = bool(
            root_contract.get("freeze_verifier", False))
        record_contract = {
            "input_dir": str(Path(record.get("input_dir", "")).resolve()),
            "human_proxy_context": record.get("human_proxy_context"),
            "human_proxy_context_sha256": record.get(
                "human_proxy_context_sha256"),
            "freeze_verifier": bool(record.get("freeze_verifier", False)),
        }
        spec_contract = {
            "input_dir": str(spec.input_dir),
            "human_proxy_context": (
                str(spec.human_proxy_context) if spec.human_proxy_context else None),
            "human_proxy_context_sha256": spec.human_proxy_context_sha256,
            "freeze_verifier": spec.freeze_verifier,
        }
        if root_contract != record_contract or root_contract != spec_contract:
            raise ValueError(
                f"run contract mismatch before spawn for {spec.run_id}: "
                f"top={root_contract!r}, record={record_contract!r}, "
                f"spec={spec_contract!r}")
        run_human = {
            "timeout_s": spec.human_agent_timeout_s,
            "model": spec.human_agent_model,
            "reasoning_effort": spec.human_agent_reasoning_effort,
        }
        run_solver = None
        if spec.solver_model is not None or spec.solver_reasoning_effort is not None:
            run_solver = {
                "model": spec.solver_model,
                "reasoning_effort": spec.solver_reasoning_effort,
            }
        checks = {
            "Human Agent contract": (
                man.get("human_agent", {
                    "timeout_s": None, "model": None, "reasoning_effort": None}),
                run_human),
            "Solver identity contract": (man.get("solver"), run_solver),
            "context directory contract": (
                man.get("human_proxy_context_dir"),
                str(spec.human_proxy_context_dir)
                if spec.human_proxy_context_dir else None),
            "inputs contract": (
                man.get("inputs", [r.get("input_dir") for r in man.get("runs", [])]),
                [str(p) for p in spec.batch_input_dirs]),
            "feedback contract": (
                man.get("feedback", "with_artifacts"), spec.feedback),
            "solver strength contract": (
                man.get("solver_strength", "weak"), spec.solver_strength),
            "freeze verifier contract": (
                bool(man.get("freeze_verifier", False)), spec.freeze_verifier),
        }
        for name, (root_value, run_value) in checks.items():
            if root_value != run_value:
                raise ValueError(
                    f"{name} mismatch before spawn for {spec.run_id}: "
                    f"batch={root_value!r}, run={run_value!r}")

    def _build_argv(self, spec: LaunchSpec, *, budget_s: float, python: str,
                    extra_args: list[str], cpu_mode: bool = False) -> list[str]:
        self._verify_human_proxy_context(spec)
        self._verify_solver_identity(spec)
        argv = [python, "-m", "coscientist.coevo.cli",
                "--input", str(spec.input_dir),
                "--solver", "codex", "--supervisor", "no-human-no-proxy",
                "--budget-s", str(budget_s),
                # The wall-clock budget is the ONLY intended stop signal (SForge's
                # "timeout = done" philosophy). The default max_turns=12 would end an
                # 8h run in minutes, so lift the turn cap far above any real run —
                # solve_and_evolve then loops until self.deadline.expired().
                "--max-turns", "100000",
                "--feedback", spec.feedback,
                "--solver-strength", spec.solver_strength,
                "--resume",
                "--runs-dir", str(self.runs_dir),
                "--run-id", spec.run_id]
        if spec.freeze_verifier:
            argv.append("--freeze-verifier")
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
        if spec.human_proxy_context is not None:
            argv += ["--human-proxy-context", str(spec.human_proxy_context)]
            argv += ["--human-proxy-context-sha256",
                     str(spec.human_proxy_context_sha256)]
        if spec.human_agent_timeout_s is not None:
            argv += ["--human-agent-timeout-s", str(spec.human_agent_timeout_s)]
        if spec.human_agent_model is not None:
            argv += ["--human-agent-model", spec.human_agent_model]
        if spec.human_agent_reasoning_effort is not None:
            argv += ["--human-agent-reasoning-effort",
                     spec.human_agent_reasoning_effort]
        argv += list(extra_args)
        return argv

    def _spawn(self, spec: LaunchSpec, *, budget_s: float, python: str,
               extra_args: list[str], cpu_mode: bool = False) -> int:
        _validate_extra_args(extra_args)
        self._verify_spawn_manifest_contract(spec)
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
        self._verify_all_run_manifest_contracts(man)
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
        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self.batch_dir,
                    prefix=".batch.json.", suffix=".tmp", delete=False) as f:
                temp_path = Path(f.name)
                json.dump(obj, f, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.manifest_path)
        except Exception:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def _read_manifest(self) -> dict:
        if not self.manifest_path.is_file():
            raise ValueError(f"batch manifest missing: {self.manifest_path}")
        try:
            man = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid/corrupt batch manifest: {self.manifest_path}") from exc
        if not isinstance(man, dict) or man.get("batch") != self.batch or \
                not isinstance(man.get("runs"), list):
            raise ValueError(
                f"invalid batch manifest schema: {self.manifest_path}")
        schema_version = man.get("schema_version", 1)
        if not isinstance(schema_version, int) or schema_version < 1 or \
                schema_version > BATCH_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported batch manifest schema_version: {schema_version!r}")
        if schema_version >= RUN_CONTRACT_SCHEMA_VERSION:
            contracts = man.get("run_contracts")
            run_ids = [r.get("run_id") for r in man["runs"]
                       if isinstance(r, dict)]
            if not isinstance(contracts, dict) or set(contracts) != set(run_ids):
                raise ValueError(
                    "invalid batch manifest run_contracts: keys must exactly match runs")
        if schema_version >= FREEZE_VERIFIER_SCHEMA_VERSION:
            top_freeze = man.get("freeze_verifier")
            if type(top_freeze) is not bool:
                raise ValueError(
                    "invalid schema-3 freeze_verifier: batch marker must be boolean")
            for record in man["runs"]:
                if not isinstance(record, dict):
                    raise ValueError("invalid batch manifest run record")
                run_id = record.get("run_id")
                run_freeze = record.get("freeze_verifier")
                contract = man["run_contracts"].get(run_id)
                contract_freeze = (contract.get("freeze_verifier")
                                   if isinstance(contract, dict) else None)
                if type(run_freeze) is not bool or type(contract_freeze) is not bool:
                    raise ValueError(
                        "invalid schema-3 freeze_verifier: every run and run_contract "
                        "marker must be boolean")
                if not (top_freeze == run_freeze == contract_freeze):
                    raise ValueError(
                        f"schema-3 freeze_verifier contract mismatch for {run_id}")
        return man


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
        description="Parallel launcher for agent-system case-study runs",
        allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("start", help="launch the first GPU wave; queue the rest pending",
                        allow_abbrev=False)
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
    sp.add_argument("--human-proxy-context-dir", default=None,
                    help="private context directory; each input basename maps to "
                         "<dir>/<basename>.md and is pinned by SHA-256")
    sp.add_argument("--human-agent-timeout-s", type=float, default=None,
                    help="wall-clock cap for each Human Proxy agent turn")
    sp.add_argument("--human-agent-model", default=None,
                    help="model used by the independent Human Proxy")
    sp.add_argument("--human-agent-reasoning-effort", default=None,
                    help="reasoning effort used by the independent Human Proxy")
    sp.add_argument("--solver-model", default=None,
                    help="assert the host Codex gateway resolves this Solver model")
    sp.add_argument("--solver-reasoning-effort", default=None,
                    help="assert the host Codex gateway resolves this Solver effort")
    sp.add_argument("--feedback", default="with_artifacts",
                    choices=["score_only", "feasible_score", "with_artifacts"],
                    help="first-class evaluator disclosure level for every run")
    sp.add_argument("--solver-strength", default="weak", choices=["weak", "strong"],
                    help="first-class Solver topology for every run")
    sp.add_argument("--freeze-verifier", action="store_true",
                    help="keep the pre-seeded verifier fixed for every run")
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
                    help="exact allowlisted non-contract flags appended to every child "
                         "cli.py argv (persisted and reused on respawn). Protected "
                         "experiment flags, abbreviations, and duplicates are rejected; "
                         "promote such settings to first-class launcher options.")

    st = sub.add_parser("status", help="show a table of all runs in a batch",
                        allow_abbrev=False)
    st.add_argument("--batch", required=True)
    st.add_argument("--runs-dir", default="runs")

    wp = sub.add_parser("watch", help="wave-schedule a batch until every run finishes",
                        allow_abbrev=False)
    wp.add_argument("--batch", required=True)
    wp.add_argument("--runs-dir", default="runs")
    wp.add_argument("--interval-s", type=float, default=30.0)

    kp = sub.add_parser("stop", help="SIGTERM every run in a batch (resume-able after)",
                        allow_abbrev=False)
    kp.add_argument("--batch", required=True)
    kp.add_argument("--runs-dir", default="runs")

    pk = sub.add_parser("package",
                        help="copy raw K3 task dirs into launchable problem dirs "
                             "(idempotent; writes the gla resource.toml template)",
                        allow_abbrev=False)
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
                            human_proxy_context_dir=(
                                Path(args.human_proxy_context_dir)
                                if args.human_proxy_context_dir else None),
                            human_agent_timeout_s=args.human_agent_timeout_s,
                            human_agent_model=args.human_agent_model,
                            human_agent_reasoning_effort=(
                                args.human_agent_reasoning_effort),
                            solver_model=args.solver_model,
                            solver_reasoning_effort=(
                                args.solver_reasoning_effort),
                            feedback=args.feedback,
                            solver_strength=args.solver_strength,
                            freeze_verifier=args.freeze_verifier,
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
