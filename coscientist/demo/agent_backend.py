"""Agent backends: drive a headless coding agent for one single-shot session.

Mirrors ExplorationHarness's ``simpletes_agent/backends.py`` contract, trimmed
to what the demo needs. Each backend knows how to build the argv for one
headless session; the prompt is fed on stdin and the agent is expected to write
its output file(s) into the workspace, which the caller reads back.

Three backends:

  * ``CodexBackend``      — ``codex exec`` (OpenAI)
  * ``ClaudeCodeBackend`` — ``claude -p`` (Anthropic, Claude Code headless)
  * ``StubBackend``       — no LLM; a deterministic in-process function writes
                            the output file. Guarantees the demo always runs,
                            offline, for CI and for exercising the loop wiring.

The point of the shared ``run_session`` signature is that the orchestrator and
the proposers never know which one they're driving — swap ``--agent codex`` for
``--no-agent`` and the two evolution loops behave identically in shape.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable


@dataclass
class SessionResult:
    ok: bool
    stdout_tail: str = ""
    error: Optional[str] = None


@runtime_checkable
class AgentBackend(Protocol):
    name: str

    def run_session(self, *, workspace: Path, prompt: str, model: Optional[str], timeout_s: float) -> SessionResult:
        """Run one headless session in ``workspace``; the agent writes files there."""


def _run_cli(argv: list[str], *, cwd: Path, prompt: str, timeout_s: float) -> SessionResult:
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError:
        return SessionResult(False, error=f"binary not found: {argv[0]!r}")
    except subprocess.TimeoutExpired:
        return SessionResult(False, error=f"agent session timed out after {timeout_s}s")
    tail = (proc.stdout or "")[-800:]
    if proc.returncode != 0:
        return SessionResult(False, stdout_tail=tail, error=(proc.stderr or "agent exited nonzero")[-800:])
    return SessionResult(True, stdout_tail=tail)


@dataclass
class CodexBackend:
    binary: str = "codex"
    name: str = "codex"

    def run_session(self, *, workspace: Path, prompt: str, model: Optional[str], timeout_s: float) -> SessionResult:
        argv = [
            self.binary, "exec",
            "--skip-git-repo-check",
            "--cd", str(workspace),
            "--dangerously-bypass-approvals-and-sandbox",
        ]
        if model:
            argv += ["--model", model]
        argv.append("-")  # prompt on stdin
        return _run_cli(argv, cwd=workspace, prompt=prompt, timeout_s=timeout_s)


@dataclass
class DshBackend:
    """Run DeepSeek Harness in its one-shot headless profile."""

    binary: str = "dsh"
    name: str = "dsh"
    profile: str = "headless"

    def run_session(self, *, workspace: Path, prompt: str, model: Optional[str], timeout_s: float) -> SessionResult:
        # Released DSH accepts a positional task, not a stdin sentinel.
        # Pass one argv element so shell-like tokens remain literal.
        binary = shutil.which(self.binary) or str(Path.home() / ".local" / "bin" / self.binary)
        argv = [binary, "--profile", self.profile, "--", prompt]
        return _run_cli(argv, cwd=workspace, prompt=prompt, timeout_s=timeout_s)


@dataclass
class ClaudeCodeBackend:
    binary: str = "claude"
    name: str = "claude-code"

    def run_session(self, *, workspace: Path, prompt: str, model: Optional[str], timeout_s: float) -> SessionResult:
        argv = [
            self.binary, "-p",
            "--dangerously-skip-permissions",
            "--add-dir", str(workspace),
        ]
        if model:
            argv += ["--model", model]
        return _run_cli(argv, cwd=workspace, prompt=prompt, timeout_s=timeout_s)


@dataclass
class StubBackend:
    """No-LLM backend: a registered in-process handler writes the output file.

    The orchestrator registers two handlers (solution / evaluator) keyed by a
    ``task`` string embedded in the workspace's ``TASK`` file, so a single stub
    backend can serve both proposer roles deterministically.
    """

    name: str = "stub"
    handlers: Optional[dict[str, Callable[[Path], None]]] = None

    def run_session(self, *, workspace: Path, prompt: str, model: Optional[str], timeout_s: float) -> SessionResult:
        task_file = workspace / "TASK"
        task = task_file.read_text(encoding="utf-8").strip() if task_file.is_file() else ""
        handlers = self.handlers or {}
        fn = handlers.get(task)
        if fn is None:
            return SessionResult(False, error=f"stub has no handler for task {task!r}")
        try:
            fn(workspace)
        except Exception as e:  # a stub bug shouldn't look like an agent failure silently
            return SessionResult(False, error=f"stub handler raised: {e}")
        return SessionResult(True, stdout_tail=f"[stub:{task}]")


@dataclass
class HarborBackend:
    """Run one agent session inside a Harbor container, then recover its output.

    This is the *outer boundary* the local backends lack: the agent runs in a
    fresh container per session (``harbor run`` = one job = one container), so
    turning the agent's internal sandbox off is safe, and the hidden reference
    evaluator (V*) is kept out by a real boundary rather than a convention.

    Implements the same ``run_session`` contract as the local backends: it seeds
    from ``workspace``, and copies the agent's single output file back INTO
    ``workspace`` so the proposers read it exactly as before. The whole run is a
    no-op-with-clean-error when docker+harbor are unavailable, so importing/using
    this backend never crashes a box without a container runtime.

    ``isolation`` = "separate" (default) puts grading in an isolated container the
    agent never touches (TB2/TB3 / autosciworld model); "shared" keeps agent and
    grader together (AutoLab model). ``extra_compose`` carries site overlays such
    as the DNS-pin ``hosts_overlay.yaml`` without hardcoding them.
    """

    name: str = "harbor"
    isolation: str = "separate"
    agent_name: str = "claude-code"
    binary: str = "harbor"
    extra_compose: list[str] = field(default_factory=list)

    def _available(self) -> Optional[str]:
        """Return an error string if harbor/docker aren't usable, else None."""
        if shutil.which(self.binary) is None:
            return f"harbor CLI not found on PATH ({self.binary!r})"
        if shutil.which("docker") is None:
            return "docker not found on PATH"
        try:
            proc = subprocess.run(
                ["docker", "info"], capture_output=True, text=True, timeout=20
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            return f"docker not usable: {e}"
        if proc.returncode != 0:
            return "docker daemon not reachable (`docker info` failed)"
        return None

    def run_session(self, *, workspace: Path, prompt: str, model: Optional[str], timeout_s: float) -> SessionResult:
        # Local import avoids a hard dependency cycle and keeps this module usable
        # even if the packaging helpers change shape.
        from . import harbor_runtime as hr

        task_file = workspace / "TASK"
        role = task_file.read_text(encoding="utf-8").strip() if task_file.is_file() else ""
        if role not in hr.ROLE_OUTPUT:
            return SessionResult(False, error=f"harbor backend: unknown role {role!r}")
        output_rel = hr.ROLE_OUTPUT[role]

        unavailable = self._available()
        if unavailable is not None:
            return SessionResult(False, error=f"harbor/docker unavailable: {unavailable}")

        with tempfile.TemporaryDirectory(prefix="cosci_harbor_") as d:
            root = Path(d)
            # The task lives alone under tasks/ so harbor's dataset scan (which
            # fans out over a directory's immediate children) sees only it.
            tasks_dir = root / "tasks"
            tasks_dir.mkdir()
            task_dir = hr.pack_task(
                workspace, role=role, isolation=self.isolation,
                agent_timeout_sec=timeout_s, dest=tasks_dir / "task",
            )
            jobs_dir = root / "jobs"
            jobs_dir.mkdir()
            job_name = f"cosci_{role}"
            cfg = hr.build_job_config(
                task_dir, job_name=job_name, jobs_dir=jobs_dir,
                agent_name=self.agent_name, model=model,
                extra_compose=self.extra_compose or None,
            )
            cfg_path = root / "job.json"
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

            argv = [self.binary, "run", "--config", str(cfg_path)]
            try:
                proc = subprocess.run(
                    argv, capture_output=True, text=True, timeout=timeout_s
                )
            except subprocess.TimeoutExpired:
                return SessionResult(False, error=f"harbor job timed out after {timeout_s}s")
            tail = (proc.stdout or "")[-800:]
            if proc.returncode != 0:
                return SessionResult(False, stdout_tail=tail, error=(proc.stderr or "harbor exited nonzero")[-800:])

            ok, out_text, note = hr.parse_job_dir(jobs_dir, job_name, output_rel=output_rel)
            if not ok or out_text is None:
                return SessionResult(False, stdout_tail=tail, error=f"output recovery failed: {note}")
            # Copy the recovered output back into the caller's workspace so the
            # proposer's existing ws.read_json/ws.read works byte-for-byte.
            (workspace / output_rel).write_text(out_text, encoding="utf-8")
            return SessionResult(True, stdout_tail=f"{tail}\n[harbor:{role}] {note}".strip())


def make_backend(
    name: str,
    *,
    handlers: Optional[dict[str, Callable[[Path], None]]] = None,
    isolation: str = "separate",
    extra_compose: Optional[list[str]] = None,
) -> AgentBackend:
    if name in ("stub", "none", "no-agent"):
        return StubBackend(handlers=handlers)
    if name == "codex":
        return CodexBackend()
    if name == "dsh":
        return DshBackend()
    if name in ("claude", "claude-code"):
        return ClaudeCodeBackend()
    if name in ("harbor", "harbor-claude", "harbor-claude-code"):
        return HarborBackend(isolation=isolation, agent_name="claude-code",
                             extra_compose=list(extra_compose or []))
    if name == "harbor-codex":
        return HarborBackend(isolation=isolation, agent_name="codex",
                             extra_compose=list(extra_compose or []))
    raise ValueError(f"unknown agent backend: {name!r}")
