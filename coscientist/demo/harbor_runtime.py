"""Harbor container runtime: package an agent session as a Harbor task, run it,
recover its output — the *outer boundary* the local-subprocess backends lack.

Why this module exists
----------------------
The demo's ``CodexBackend`` / ``ClaudeCodeBackend`` run a coding agent as a local
subprocess with its internal sandbox turned OFF (``--dangerously-*``). That is
only safe if something *else* provides the boundary. Harbor is that something:
every trial runs the agent in a fresh **container**, and grading can run in a
**separate** container the agent never touches. The three benchmarks this design
mirrors — AutoLab, Terminal-Bench 2/3 (a.k.a. Harbor Index / frontierbench) — all
run on Harbor. AutoLab keeps agent+grader in one container (``shared``); TB2/TB3
and the autosciworld tasks separate them (``separate``). That split is the
``isolation`` knob here.

How it maps to our two-loop system
-----------------------------------
Harbor here is ONLY the agent sandbox. Grading V stays on the trusted control
plane (``Evaluator.run``, already out-of-process). So the Harbor task's own
verifier is vestigial — ``tests/test.sh`` just writes ``reward=0``. We use Harbor
to (a) isolate the agent behind a container boundary and (b) recover the one file
the agent writes (``solution_out.json`` / ``verifier_out.py``) through Harbor's
declared ``artifacts`` channel. One ``run_session`` == one ``harbor run`` job ==
one fresh container, which is exactly "each role its own container".

This module is pure and docker-free: it *builds* the task package + job config and
*parses* a finished job directory. The actual ``harbor run`` subprocess lives in
``HarborBackend`` (``agent_backend.py``) and only fires when docker+harbor exist.

Verified against ``harbor-rollout/`` on this box (harbor 0.20.0): JobConfig shape,
``task.toml`` schema 1.3 (top-level ``artifacts``, ``[verifier] environment_mode``),
and the ``jobs_dir/<job>/<trial>/artifacts/app/<rel>`` recovery layout.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from . import taskspec

# The single output file each role's agent must write, and where it lands after
# recovery. Keyed by the ``TASK`` role string the workspace carries.
ROLE_OUTPUT = {
    "solution": "solution_out.json",
    "evaluator": "verifier_out.py",
}

# A vestigial grader: in our system the real V runs on the control plane, so the
# container-side verifier only needs to satisfy Harbor's reward.json contract.
_VESTIGIAL_TEST_SH = """\
#!/bin/bash
# Grading happens on the Co-Scientist control plane, not here. Harbor still
# requires a reward file, so emit a neutral one.
set -uo pipefail
mkdir -p /logs/verifier
printf '%s\\n' '{"reward": 0.0}' > /logs/verifier/reward.json
exit 0
"""


def _toml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


# ---------------------------------------------------------------------------
# V*-isolation invariant — the software stand-in for the container boundary
# ---------------------------------------------------------------------------
def reference_leak_markers() -> list[str]:
    """Tokens that must NEVER appear in an agent-facing task/workspace.

    These are the hidden reference (V*) fingerprints: the true-signal samples,
    the reference-scoring symbol names, and the V*-only artifact keys. If any of
    these shows up in a packaged task, the agent could read the answer key.
    """
    markers = [
        "reference_score",           # taskspec.reference_score — the V* entry point
        "reduced_chi2_vs_truth",     # a V*-only artifact key
        "INSTANCE.signal",           # the hidden-truth attribute access
        "_build_instance",           # the function that materializes the truth
    ]
    # A handful of the true-signal float samples, formatted as they'd serialize.
    # Matching any verbatim means the hidden signal leaked into the workspace.
    sig = taskspec.INSTANCE.signal
    for v in (sig[0], sig[len(sig) // 2], sig[-1]):
        markers.append(repr(float(v)))
    return markers


def assert_no_reference_leak(root: Path) -> None:
    """Fail closed if any file under ``root`` carries a V* fingerprint.

    Called inside :func:`pack_task` before anything could be shipped to a
    container. Only agent-facing directories are ever passed here — the AutoHuman
    legitimately holds V* but lives on the control plane, not in a task dir.
    """
    root = Path(root)
    markers = reference_leak_markers()
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary / unreadable file can't carry a text marker
        for m in markers:
            if m in text:
                raise AssertionError(
                    f"reference-evaluator (V*) leak: marker {m!r} found in "
                    f"{p.relative_to(root)} — the hidden answer key must never "
                    f"enter an agent workspace/task package."
                )


# ---------------------------------------------------------------------------
# Task packaging
# ---------------------------------------------------------------------------
def _dockerfile(input_files: list[str]) -> str:
    """A minimal agent-environment image: numpy + the seeded workspace under /app.

    The agent *binary* injection (the ``COPY claude`` pattern from harbor-rollout)
    is intentionally NOT hardcoded here — which agent runs, and how its binary is
    provisioned, is a run-time/adapter concern the opt-in real-run path supplies.
    """
    lines = [
        "FROM python:3.11-slim",
        "ENV DEBIAN_FRONTEND=noninteractive",
        "RUN pip install --no-cache-dir numpy>=1.24",
        "WORKDIR /app",
    ]
    for rel in input_files:
        lines.append(f"COPY {rel} /app/{rel}")
    lines.append('CMD ["/bin/bash"]')
    return "\n".join(lines) + "\n"


def _task_toml(*, name: str, description: str, output_rel: str, isolation: str,
               agent_timeout_sec: float, agent_network: str) -> str:
    """Emit a Harbor task.toml (schema 1.3) by hand — no TOML writer dependency.

    ``isolation="separate"`` declares an isolated grading container the agent
    never touches (TB2/TB3 / autosciworld model). ``"shared"`` omits it, leaving
    agent+grader in one container (AutoLab model).
    """
    parts = [
        'schema_version = "1.3"',
        "",
        # The narrow recovery channel: only this one file comes back out.
        f'artifacts = ["/app/{output_rel}"]',
        "",
        "[task]",
        f'name = "{_toml_escape(name)}"',
        f'description = "{_toml_escape(description)}"',
        "",
        "[agent]",
        f"timeout_sec = {float(agent_timeout_sec)}",
        "",
        "[environment]",
        f'network_mode = "{agent_network}"',
        "",
        "[verifier]",
        "timeout_sec = 60.0",
    ]
    if isolation == "separate":
        parts += [
            'environment_mode = "separate"',
            "",
            "[verifier.environment]",
            'network_mode = "no-network"',
        ]
    return "\n".join(parts) + "\n"


def pack_task(
    workspace: Path,
    *,
    role: str,
    isolation: str = "separate",
    agent_timeout_sec: float = 1200.0,
    dest: Optional[Path] = None,
) -> Path:
    """Build a Harbor task package from a seeded agent ``workspace``.

    The workspace is the same directory the proposers seed (TASK, PROMPT.md,
    context.json, verifier.py, the role's input file). We copy those into the
    package's ``environment/`` context, declare the role's single output file as
    the recovered ``artifact``, and wire a vestigial grader. Returns the package
    directory. Fails closed via :func:`assert_no_reference_leak`.
    """
    workspace = Path(workspace)
    if role not in ROLE_OUTPUT:
        raise ValueError(f"unknown role {role!r}; expected one of {sorted(ROLE_OUTPUT)}")
    output_rel = ROLE_OUTPUT[role]

    # Solution/evaluator agents both need the network to reach their LLM backend;
    # the (separate) grading env stays offline. This mirrors asw_000104.
    agent_network = "public"

    if dest is None:
        dest = Path(workspace).parent / f"{workspace.name}__harbor_task"
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    env_dir = dest / "environment"
    tests_dir = dest / "tests"
    env_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    # Copy the seeded workspace files into the build context (skip the OUTPUT file
    # if a prior run left one behind — the agent produces it inside the container).
    input_files: list[str] = []
    for p in sorted(workspace.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(workspace).as_posix()
        if rel == output_rel:
            continue
        target = env_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        input_files.append(rel)

    (env_dir / "Dockerfile").write_text(_dockerfile(input_files), encoding="utf-8")
    (tests_dir / "test.sh").write_text(_VESTIGIAL_TEST_SH, encoding="utf-8")

    prompt = ""
    prompt_file = workspace / "PROMPT.md"
    if prompt_file.is_file():
        prompt = prompt_file.read_text(encoding="utf-8")
    (dest / "instruction.md").write_text(
        prompt or f"Co-Scientist {role} session.\n", encoding="utf-8"
    )

    (dest / "task.toml").write_text(
        _task_toml(
            name=f"co-scientist/{role}",
            description=f"Co-Scientist two-loop demo — {role} agent session.",
            output_rel=output_rel,
            isolation=isolation,
            agent_timeout_sec=agent_timeout_sec,
            agent_network=agent_network,
        ),
        encoding="utf-8",
    )

    # Fail closed: no V* fingerprint may enter the package.
    assert_no_reference_leak(dest)
    return dest


# ---------------------------------------------------------------------------
# Job config
# ---------------------------------------------------------------------------
def build_job_config(
    task_dir: Path,
    *,
    job_name: str,
    jobs_dir: Path,
    agent_name: str,
    model: Optional[str] = None,
    extra_compose: Optional[list[str]] = None,
) -> dict:
    """Emit the Harbor JobConfig dict (verified against harbor-rollout configs).

    A dataset ``path`` pointing at a directory of task packages is how Harbor
    fans out; here it's the single packaged task's parent so exactly one trial
    runs. ``extra_compose`` carries site-specific overlays (e.g. the DNS-pin
    ``hosts_overlay.yaml``) without hardcoding them.
    """
    agent: dict = {"name": agent_name}
    if model:
        agent["model_name"] = model
    environment: dict = {"type": "docker"}
    if extra_compose:
        environment["extra_docker_compose"] = list(extra_compose)
    return {
        "job_name": job_name,
        "jobs_dir": str(jobs_dir),
        "n_concurrent_trials": 1,
        "environment": environment,
        "agents": [agent],
        "datasets": [{"path": str(Path(task_dir).parent)}],
    }


# ---------------------------------------------------------------------------
# Job-dir parsing (recovery)
# ---------------------------------------------------------------------------
def parse_job_dir(
    jobs_dir: Path, job_name: str, *, output_rel: str
) -> tuple[bool, Optional[str], str]:
    """Recover an agent's output file from a finished Harbor job directory.

    Layout (verified): ``jobs_dir/<job_name>/<trial>/`` holds ``result.json`` and
    ``artifacts/app/<output_rel>``. Returns ``(ok, output_text, tail)`` where
    ``ok`` means the output file was recovered.
    """
    job_root = Path(jobs_dir) / job_name
    if not job_root.is_dir():
        return False, None, f"job dir not found: {job_root}"

    trial_dirs = [d for d in sorted(job_root.iterdir()) if d.is_dir()]
    if not trial_dirs:
        return False, None, f"no trial dir under {job_root}"

    for trial in trial_dirs:
        out_path = trial / "artifacts" / "app" / output_rel
        if out_path.is_file():
            text = out_path.read_text(encoding="utf-8")
            tail = ""
            result_json = trial / "result.json"
            if result_json.is_file():
                try:
                    res = json.loads(result_json.read_text(encoding="utf-8"))
                    tail = f"trial={res.get('trial_name', trial.name)}"
                except json.JSONDecodeError:
                    tail = f"trial={trial.name}"
            return True, text, tail

    return False, None, (
        f"output {output_rel!r} not recovered from any trial under {job_root} "
        f"(checked {len(trial_dirs)} trial dir(s))"
    )
