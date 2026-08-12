"""Container session layer (§4/§5) — real coding agents in isolated docker containers.

This is the transport the multi-agent system runs on. The design decisions locked
with the user:

  * The **agent inside the container is ``codex``** — the OpenAI Codex CLI. We mount
    its self-contained static-musl ELF read-only (the ``vendor/.../bin/codex`` rust
    binary shipped inside the npm package; no node needed) plus the host's
    ``~/.codex`` auth (``auth.json`` + a minimal ``config.toml``) as ``CODEX_HOME``.
    It runs headless: ``codex exec --skip-git-repo-check
    --dangerously-bypass-approvals-and-sandbox <prompt>``. The container IS the
    isolation boundary, so codex's own sandbox is bypassed (approved by the user:
    "容器内全放开").
  * The **Solver container is long-lived** (``docker run -d``). Each agent turn is
    one ``docker exec ... codex exec`` session that EXITS when it wants to ask the
    Supervisor. We then **resume** by running a fresh ``codex exec`` in the SAME
    open container — file state (+ an agent-authored scratchpad) carries context.
    No polling: the agent is one-shot, it never blocks waiting on us.
  * The **Supervisor container is one-shot, on demand** — spun up by the
    orchestrator only when a review/harden is needed, then torn down.
  * **Shims** are tiny executables in the agent's workdir that forward calls to the
    host orchestrator over a **bind-mounted unix domain socket** (no container->host
    networking, no 127.0.0.1 trap). ``container-eval`` and ``container-status`` are
    answered synchronously; ``container-ask-supervisor`` is queued (the heavy path
    is file-state resume, not a blocking socket call).

Everything here clean-fails when docker or the agent binary is absent, so the
offline test suite never touches a real container.

The codex auth (``auth.json``) is discovered from the host's own ``~/.codex`` and
mounted read-only, so this module carries no secrets and no hardcoded credential.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import stat
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# AF_UNIX bind paths are capped at ~108 bytes (sockaddr_un.sun_path, incl. the
# NUL terminator). A deep ``runs/<long_run_id>/solver_ws/.control.sock`` blows
# past that and crashes the run at bind() — so when the natural in-workdir path
# is too long we bind at a short, STABLE /tmp path instead and mount THAT file
# into the container at the fixed ``/work/.control.sock`` (see ControlSocket).
_AF_UNIX_SAFE_LEN = 100


# ---------------------------------------------------------------------------
# availability guards — the whole layer is a no-op when these fail
# ---------------------------------------------------------------------------
def docker_unavailable() -> Optional[str]:
    """Return a reason string if docker can't be used, else None."""
    if shutil.which("docker") is None:
        return "docker CLI not found on PATH"
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True,
                              text=True, timeout=20)
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"docker not responding: {e}"
    if proc.returncode != 0:
        return f"docker info failed: {(proc.stderr or '').strip()[-160:]}"
    return None


def agent_elf_path() -> Optional[Path]:
    """Resolve the self-contained ``codex`` binary to mount into containers.

    The ``codex`` on PATH is a node shim; the real, portable artifact is the
    static-musl rust ELF shipped inside the npm package at
    ``@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex``. We
    resolve the shim, walk to the package root, and return that ELF.
    """
    exe = shutil.which("codex")
    if exe is None:
        return None
    # the shim symlinks into .../@openai/codex/bin/codex.js — climb to the package.
    real = Path(exe).resolve()
    # search upward for the codex package dir, then descend to the vendored ELF.
    for parent in [real, *real.parents]:
        cand = (parent / "node_modules" / "@openai" / "codex-linux-x64"
                / "vendor" / "x86_64-unknown-linux-musl" / "bin" / "codex")
        if cand.is_file():
            return cand.resolve()
        # also handle being inside the @openai/codex package already
        cand2 = (parent / "@openai" / "codex-linux-x64" / "vendor"
                 / "x86_64-unknown-linux-musl" / "bin" / "codex")
        if cand2.is_file():
            return cand2.resolve()
    return None


def codex_home_path() -> Optional[Path]:
    """The host ``~/.codex`` directory carrying ``auth.json`` — None if absent."""
    home = Path.home() / ".codex"
    return home if (home / "auth.json").is_file() else None


# ---------------------------------------------------------------------------
# agent auth — the codex home (auth.json), discovered from the host, never hardcoded
# ---------------------------------------------------------------------------
@dataclass
class GatewayConfig:
    """How an in-container ``codex`` authenticates, mirrored from the host.

    codex reads credentials from ``$CODEX_HOME/auth.json`` (the host's ``~/.codex``).
    We mount that directory read-only into the container and point ``CODEX_HOME`` at
    it — no API endpoint or token is threaded through env. ``model`` /
    ``reasoning_effort`` become a minimal in-container ``config.toml``. The name
    ``GatewayConfig`` is kept for continuity with the rest of the system; there is no
    private gateway involved for codex.
    """

    codex_home: Path
    model: str = "gpt-5.6-sol"
    reasoning_effort: str = "high"

    @classmethod
    def from_host(cls) -> Optional["GatewayConfig"]:
        """Assemble from the host's ``~/.codex``. None if auth.json is absent."""
        home = codex_home_path()
        if home is None:
            return None
        model, effort = "gpt-5.6-sol", "high"
        cfg = home / "config.toml"
        if cfg.is_file():
            try:
                for line in cfg.read_text(encoding="utf-8").splitlines():
                    s = line.strip()
                    if s.startswith("model") and "=" in s and "model_reasoning" not in s:
                        model = s.split("=", 1)[1].strip().strip('"')
                    elif s.startswith("model_reasoning_effort") and "=" in s:
                        effort = s.split("=", 1)[1].strip().strip('"')
            except OSError:
                pass
        return cls(codex_home=Path(home).resolve(), model=model, reasoning_effort=effort)

    def container_config_toml(self) -> str:
        """A minimal, self-contained config for the mounted CODEX_HOME overlay."""
        return (f'model = "{self.model}"\n'
                f'model_reasoning_effort = "{self.reasoning_effort}"\n')


# ---------------------------------------------------------------------------
# control socket — the synchronous shim channel, over a bind-mounted unix socket
# ---------------------------------------------------------------------------
_CLIENT_PY = r'''#!/usr/bin/env python3
"""In-container shim client. Talks newline-JSON to the host over a unix socket.
No third-party deps; the base image's python3 is enough."""
import json, os, sys, socket

SOCK = os.environ.get("CONTROL_SOCK", "/work/.control.sock")

def call(cmd, arg):
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.settimeout(120)
    c.connect(SOCK)
    c.sendall((json.dumps({"cmd": cmd, "arg": arg}) + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = c.recv(65536)
        if not chunk:
            break
        buf += chunk
    c.close()
    return json.loads(buf.decode() or "{}")

def _load(arg):
    if arg and os.path.isfile(arg):
        return json.loads(open(arg).read())
    return json.loads(arg) if arg else {}

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd in ("eval", "ask"):
        payload = _load(sys.argv[2] if len(sys.argv) > 2 else "")
        print(json.dumps(call(cmd, {"solution": payload}), indent=2))
    elif cmd == "status":
        print(json.dumps(call("status", {}), indent=2))
    else:
        sys.exit("usage: shim.py {eval|ask|status} [solution-json-or-path]")

if __name__ == "__main__":
    main()
'''

_SHIM_TEMPLATE = '#!/usr/bin/env bash\nexec python3 "$(dirname "$0")/_control_client.py" {cmd} "$@"\n'


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@dataclass
class ControlSocket:
    """Host-side unix-socket listener serving the in-container shims.

    ``handler(cmd, arg) -> dict`` is injected by the orchestrator; this class knows
    only the transport. Runs on a daemon thread; newline-delimited JSON per request.
    """

    workdir: Path
    handler: Callable[[str, dict], dict]
    sock_name: str = ".control.sock"

    _server: Optional[socket.socket] = field(default=None, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _stop: bool = field(default=False, init=False)
    # When the natural in-workdir bind path exceeds the AF_UNIX limit we bind at a
    # short /tmp path and record it here; the container then bind-mounts THIS file
    # at ``/work/.control.sock`` rather than relying on the workdir-mounted one.
    _bind_path: Optional[Path] = field(default=None, init=False)

    @property
    def _natural_path(self) -> Path:
        """The socket path inside the (bind-mounted) workdir — visible in-container."""
        return Path(self.workdir) / self.sock_name

    @property
    def host_path(self) -> Path:
        """The path the host actually bind()s. Short /tmp fallback if workdir is too long."""
        if self._bind_path is not None:
            return self._bind_path
        natural = self._natural_path
        if len(str(natural)) <= _AF_UNIX_SAFE_LEN:
            return natural
        # Deterministic short path derived from the workdir, so a resume in the
        # SAME run rebinds the same file (idempotent; no orphan sockets pile up).
        h = hashlib.sha1(str(Path(self.workdir).resolve()).encode()).hexdigest()[:12]
        self._bind_path = Path(tempfile.gettempdir()) / f"cs_{h}.sock"
        return self._bind_path

    @property
    def needs_explicit_mount(self) -> bool:
        """True when host_path is NOT inside the workdir, so the container must
        bind-mount the socket file explicitly at container_path."""
        return self.host_path != self._natural_path

    @property
    def container_path(self) -> str:
        return f"/work/{self.sock_name}"

    def start(self) -> "ControlSocket":
        p = self.host_path
        if p.exists():
            p.unlink()
        p.parent.mkdir(parents=True, exist_ok=True)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(p))
        srv.listen(8)
        os.chmod(str(p), 0o777)   # the container uid must be able to connect
        srv.settimeout(0.5)
        self._server = srv
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def _serve(self) -> None:
        while not self._stop:
            try:
                conn, _ = self._server.accept()   # type: ignore[union-attr]
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                try:
                    buf = b""
                    conn.settimeout(120)
                    while not buf.endswith(b"\n"):
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        buf += chunk
                    req = json.loads(buf.decode() or "{}")
                    resp = self.handler(req.get("cmd", ""), req.get("arg", {}) or {})
                except Exception as e:   # never let a bad request kill the listener
                    resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                try:
                    conn.sendall((json.dumps(resp) + "\n").encode())
                except OSError:
                    pass

    def stop(self) -> None:
        self._stop = True
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        try:
            self.host_path.unlink()
        except OSError:
            pass

    def seed_shims(self, workdir: Path) -> None:
        """Drop the client + the three shims into an agent workdir."""
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "_control_client.py").write_text(_CLIENT_PY, encoding="utf-8")
        for name, cmd in (("container-eval", "eval"),
                          ("container-ask-supervisor", "ask"),
                          ("container-status", "status")):
            _write_executable(workdir / name, _SHIM_TEMPLATE.format(cmd=cmd))

    def __enter__(self) -> "ControlSocket":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# the container itself
# ---------------------------------------------------------------------------
@dataclass
class AgentSession:
    """The result of one ``claude -p`` turn inside a container."""

    ok: bool
    returncode: int
    stdout: str
    stderr: str
    note: str = ""


@dataclass
class DockerContainer:
    """A long-lived container with a bind-mounted workdir; runs agent turns via exec.

    ``workdir`` (host) is mounted at ``/work`` and is the container's HOME and CWD.
    The control socket lives inside it, so it is reachable at ``/work/.control.sock``.
    A private, writable copy of the host codex auth is mounted at ``/codexhome``
    (kept OUT of the repo tree so ``auth.json`` never lands under ``runs/``).
    """

    workdir: Path
    gateway: GatewayConfig
    agent_elf: Path
    image: str = "python:3.11-slim"
    name: Optional[str] = None
    control_sock_name: str = ".control.sock"
    # Optional GPU request (e.g. "all" or "1"). When set, ``--gpus <spec>`` is added
    # to ``docker run`` — needed for kernel tasks whose candidate runs on GPU. Default
    # None keeps the argv unchanged (no GPU). Domain-agnostic: the core never sets it;
    # the operator/agent supplies it. Requires the host's nvidia docker runtime.
    gpus: Optional[str] = None
    # Optional generic resource caps from a task's ResourceSpec.solver slice. Each is
    # additive: when set, the corresponding ``--cpus``/``--memory`` flag is added.
    # ``allow_internet=False`` pins ``--network none`` — but ONLY when this container
    # is otherwise resourced (any of cpus/memory_mb/gpus/allow_internet set), so a
    # plain default container keeps today's argv byte-for-byte. Still no cred ``-e``.
    cpus: Optional[float] = None
    memory_mb: Optional[int] = None
    allow_internet: bool = False
    # When the control socket is bound OUTSIDE the workdir (AF_UNIX path-length
    # fallback), the orchestrator passes the host bind path here so we bind-mount
    # that file at ``/work/<control_sock_name>`` — otherwise the in-container shim
    # would find no socket (the workdir mount wouldn't contain it). None => the
    # socket lives in the workdir and is reachable via the ``-v workdir:/work`` mount.
    control_sock_host_path: Optional[Path] = None

    _cid: Optional[str] = field(default=None, init=False)
    _started: bool = field(default=False, init=False)
    _codex_home_tmp: Optional[Path] = field(default=None, init=False)

    def _prepare_codex_home(self) -> Path:
        """A throw-away, writable CODEX_HOME under /tmp with auth.json + config.toml.

        codex writes sessions/logs into CODEX_HOME, so it must be writable; keeping it
        under /tmp (not the workdir) ensures the secret never enters the repo tree."""
        import tempfile
        d = Path(tempfile.mkdtemp(prefix="codexhome_"))
        shutil.copy2(self.gateway.codex_home / "auth.json", d / "auth.json")
        (d / "config.toml").write_text(self.gateway.container_config_toml(),
                                       encoding="utf-8")
        try:
            os.chmod(d, 0o777)
            os.chmod(d / "auth.json", 0o600)
        except OSError:
            pass
        self._codex_home_tmp = d
        return d

    def start(self) -> "DockerContainer":
        self.workdir = Path(self.workdir).resolve()   # docker -v needs an absolute path
        self.workdir.mkdir(parents=True, exist_ok=True)
        codex_home = self._prepare_codex_home()
        argv = ["docker", "run", "-d", "--rm",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "-v", f"{self.workdir}:/work",
                "-v", f"{self.agent_elf}:/usr/local/bin/codex:ro",
                "-v", f"{codex_home}:/codexhome",
                "-e", "HOME=/work",
                "-e", "CODEX_HOME=/codexhome",
                "-e", f"CONTROL_SOCK=/work/{self.control_sock_name}",
                "-w", "/work"]
        if self.control_sock_host_path is not None:
            # The socket was bound outside the workdir (AF_UNIX path too long) —
            # mount that exact file at the fixed in-container path so the shim's
            # CONTROL_SOCK=/work/.control.sock still resolves.
            host_sock = Path(self.control_sock_host_path).resolve()
            argv += ["-v", f"{host_sock}:/work/{self.control_sock_name}"]
        if self.name:
            argv += ["--name", self.name]
        # A container is "resourced" when the task's solver slice asked for any
        # cap/GPU. Only then do we touch cpus/memory/network — a plain default
        # container keeps the argv byte-for-byte identical to before.
        resourced = (self.cpus is not None or self.memory_mb is not None
                     or self.gpus is not None or self.allow_internet)
        if self.cpus is not None:
            argv += ["--cpus", str(self.cpus)]
        if self.memory_mb is not None:
            argv += ["--memory", f"{int(self.memory_mb)}m"]
        if self.gpus:
            # GPU access for kernel-style tasks. No creds added here — the container
            # env stays HOME/CODEX_HOME/CONTROL_SOCK only; LLM creds live host-side.
            argv += ["--gpus", self.gpus]
        if resourced and not self.allow_internet:
            # Default-deny network for a resourced container (matches Harbor's
            # no-network verifier posture). Opt in explicitly via allow_internet.
            argv += ["--network", "none"]
        # keep the container alive; agent turns are `docker exec` sessions.
        argv += [self.image, "sleep", "infinity"]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"docker run failed: {(proc.stderr or '').strip()[-300:]}")
        self._cid = proc.stdout.strip()
        self._started = True
        return self

    def exec_agent(self, prompt: str, *, timeout_s: float,
                   allowed_dir: str = "/work",
                   disallowed_tools: Optional[list[str]] = None) -> AgentSession:
        """Run ONE headless ``codex exec`` turn in the live container. Blocks until it
        exits (the agent exits on its own when it wants to ask the Supervisor).

        The container IS the isolation boundary, so codex runs with approvals and its
        own sandbox bypassed (``--dangerously-bypass-approvals-and-sandbox``).
        ``disallowed_tools`` is accepted for API continuity but codex has no such
        flag; it is ignored (the prompt governs tool use instead)."""
        if not self._started or self._cid is None:
            raise RuntimeError("container not started")
        argv = ["docker", "exec", "-i", "-w", allowed_dir, self._cid,
                "codex", "exec", "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox", prompt]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=max(1.0, timeout_s))
        except subprocess.TimeoutExpired:
            return AgentSession(ok=False, returncode=-1, stdout="", stderr="",
                                note="agent hit the wall-clock budget")
        ok = proc.returncode == 0
        return AgentSession(ok=ok, returncode=proc.returncode,
                            stdout=proc.stdout or "", stderr=proc.stderr or "",
                            note="" if ok else "codex exited nonzero")

    def stop(self) -> None:
        if self._cid is not None:
            subprocess.run(["docker", "kill", self._cid],
                          capture_output=True, text=True, timeout=30)
            self._cid = None
        self._started = False
        if self._codex_home_tmp is not None:
            shutil.rmtree(self._codex_home_tmp, ignore_errors=True)
            self._codex_home_tmp = None

    def __enter__(self) -> "DockerContainer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def one_shot_agent(workdir: Path, gateway: GatewayConfig, agent_elf: Path,
                   prompt: str, *, timeout_s: float,
                   image: str = "python:3.11-slim",
                   disallowed_tools: Optional[list[str]] = None) -> AgentSession:
    """Run a single agent turn in a fresh throw-away container (the Supervisor path).

    A control socket, if any, is expected to already live in ``workdir`` and will be
    reachable at ``/work/.control.sock`` — but the Supervisor typically needs none.
    """
    c = DockerContainer(workdir=workdir, gateway=gateway, agent_elf=agent_elf,
                        image=image)
    try:
        c.start()
        return c.exec_agent(prompt, timeout_s=timeout_s,
                            disallowed_tools=disallowed_tools)
    finally:
        c.stop()
