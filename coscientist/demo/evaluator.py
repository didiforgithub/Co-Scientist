"""The evolvable evaluator V, represented as a *rewritable source string*.

This is the object the "evaluator evolves" loop edits. A verifier is a short
Python module (text) exposing::

    def verify(payload: dict, ctx: dict) -> dict:
        # ctx = {"t": [...], "obs": [...], "noise_sigma": float}
        return {"feasible": bool, "raw": float, "artifacts": {...}}   # higher raw = better

The environment keeps a *stack of versions*; evolving V means committing a new
source string (a new version). We run it in a fresh subprocess so a broken or
malicious rewrite can't take down the orchestrator — the same reason ASW runs
verifiers out-of-process.

The initial verifier ships with the documented ASW DOF-overfit hole: it scores
reduced-chi-square against the noisy observations but only debits the trend's
degrees of freedom, ignoring the sinusoid modes. Growing the basis drives its
score up without bound — exactly what the co-evolution is meant to expose and
then close.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def _subprocess_env(extra: Optional[dict]) -> Optional[dict]:
    """Build the subprocess environment: os.environ overlaid with ``extra``.

    Returns None when there is nothing extra (subprocess inherits the parent env
    as before — keeps the default path byte-for-byte unchanged). ``extra`` is the
    host-side cred injection point (LLM_API_KEY/BASE_URL/MODEL)."""
    if not extra:
        return None
    return {**os.environ, **{k: str(v) for k, v in extra.items() if v is not None}}


# The flawed initial verifier. Note ``dof = N - 2`` — the modes are never
# debited. This is the hole the evaluator-evolution loop must discover and fix.
INITIAL_VERIFIER_SRC = '''\
"""Proxy verifier V (version 0) — SHIPS WITH A KNOWN DOF-OVERFIT HOLE.

Scores reduced chi-square against the noisy observations. BUG: the degrees of
freedom subtract only the 2 trend parameters and ignore the 3-per-mode sinusoid
parameters, so adding modes drives reduced-chi2 below the noise floor and the
score upward without bound. A hardened rewrite should debit every free param.
"""
import math


def _predict(payload, t):
    a, b = payload.get("trend", (0.0, 0.0))
    out = []
    for x in t:
        v = a * x + b
        for (f, amp, ph) in payload.get("modes", []):
            v += amp * math.sin(f * x + ph)
        out.append(v)
    return out


def verify(payload, ctx):
    t, obs = ctx["t"], ctx["obs"]
    sigma = ctx.get("noise_sigma", 1.0)
    pred = _predict(payload, t)
    resid2 = sum((o - p) ** 2 for o, p in zip(obs, pred)) / (sigma ** 2)
    n = len(t)
    n_modes = len(payload.get("modes", []))
    k = 2 + 3 * n_modes                      # free params actually carried
    dof = max(1, n - 2)                      # BUG: modes not debited
    red = resid2 / dof
    return {
        "feasible": True,
        "raw": -red,                         # higher is better
        "artifacts": {
            "reduced_chi2": red,
            "declared_dof": dof,
            "num_free_params": k,
            "n_modes": n_modes,
        },
    }
'''


_RUNNER = textwrap.dedent(
    """
    import json, sys, importlib.util
    spec = importlib.util.spec_from_file_location("verifier_mod", sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    payload = json.loads(sys.argv[2])
    ctx = json.loads(sys.argv[3])
    out = mod.verify(payload, ctx)
    print("__VERIFY_RESULT__" + json.dumps(out))
    """
)


# The feedback runner mirrors _RUNNER but drives an agent-authored ``feedback``
# module. It is fed the verifier's own result (feasible/raw/artifacts) plus a
# bounded score history, and returns a free-form dict — {"detail": str,
# "artifacts": {...}} — that the service hands to the Solver in place of the
# frozen disclosure. The feedback module may ``import llm_client`` (written
# beside it) to consult an LLM for qualitative, guiding feedback.
_FEEDBACK_RUNNER = textwrap.dedent(
    """
    import json, sys, importlib.util
    spec = importlib.util.spec_from_file_location("feedback_mod", sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    payload = json.loads(sys.argv[2])
    ctx = json.loads(sys.argv[3])
    verify_result = json.loads(sys.argv[4])
    history = json.loads(sys.argv[5])
    out = mod.feedback(payload, ctx, verify_result, history)
    print("__FEEDBACK_RESULT__" + json.dumps(out))
    """
)


# A tiny STDLIB-ONLY OpenAI-style chat client, written next to any authored
# verifier/feedback module so it can ``import llm_client``. Creds come from the
# process env (LLM_API_KEY / LLM_BASE_URL / LLM_MODEL), which the orchestrator
# injects HOST-SIDE only (never into the Solver container). No third-party deps.
_LLM_CLIENT_SRC = textwrap.dedent(
    '''
    """Stdlib-only OpenAI-style chat client for agent-authored eval/feedback code.

    Reads creds from the environment (host-side injection only). Returns the
    assistant message text, or raises RuntimeError with a short reason. Authored
    code should guard the call and degrade gracefully when creds are absent.
    """
    import json as _json
    import os as _os
    import urllib.request as _rq


    def available() -> bool:
        return bool(_os.environ.get("LLM_API_KEY") and _os.environ.get("LLM_BASE_URL"))


    def chat(messages, *, model=None, temperature=0.0, timeout=60):
        key = _os.environ.get("LLM_API_KEY")
        base = _os.environ.get("LLM_BASE_URL")
        if not key or not base:
            raise RuntimeError("no LLM creds in env (LLM_API_KEY/LLM_BASE_URL)")
        mdl = model or _os.environ.get("LLM_MODEL") or "gpt-4o-mini"
        url = base.rstrip("/") + "/chat/completions"
        body = _json.dumps({"model": mdl, "messages": messages,
                            "temperature": temperature}).encode("utf-8")
        req = _rq.Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        })
        with _rq.urlopen(req, timeout=timeout) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    '''
)


@dataclass
class VerifierVersion:
    version: int
    source: str
    origin: str                 # "initial" | "agent" | "human"
    note: str = ""
    # Optional agent-authored FEEDBACK module (source string). When None, the
    # service falls back to the frozen FeedbackLevel/_disclose path (unchanged
    # legacy behavior). When present, it is a free-form strategy that maps the
    # verifier's raw result into whatever the Solver should see this version —
    # more diagnostics, less, or natural-language guidance from an LLM. It is the
    # representation-free half of "the evaluator evolves". See ``run_feedback``.
    feedback_src: Optional[str] = None


@dataclass
class RunResult:
    feasible: bool
    raw: float
    artifacts: dict
    error: Optional[str] = None


@dataclass
class Evaluator:
    """A versioned, executable verifier. ``evolve`` commits a new source string."""

    versions: list[VerifierVersion] = field(default_factory=list)
    timeout_s: float = 10.0
    # Optional containerized verify backend (the GPU-backed checker seam). ``None``
    # (default) keeps the host ``sys.executable`` subprocess path byte-for-byte. When
    # set, verify/feedback run inside ``docker run`` with this container's ISOLATED
    # resources — a dict with any of {image,gpus,cpus,memory_mb,allow_internet,
    # timeout_s}. Describes the VERIFIER container only; the Solver container is
    # elsewhere and shares nothing here. No LLM creds are ever passed into it.
    exec_backend: Optional[dict] = None

    @classmethod
    def initial(cls) -> "Evaluator":
        ev = cls()
        ev.versions.append(VerifierVersion(0, INITIAL_VERIFIER_SRC, "initial", "shipped flawed"))
        return ev

    @property
    def current(self) -> VerifierVersion:
        return self.versions[-1]

    def evolve(self, new_source: str, origin: str, note: str = "",
               *, feedback_src: Optional[str] = None) -> VerifierVersion:
        """Commit a rewritten verifier as the next version. This IS eval-evolution.

        ``feedback_src`` optionally attaches an agent-authored feedback module to
        this version (the free-form disclosure strategy). ``None`` keeps the
        legacy FeedbackLevel/_disclose path for this version.
        """
        v = VerifierVersion(len(self.versions), new_source, origin, note,
                            feedback_src=feedback_src)
        self.versions.append(v)
        return v

    def _ctx_for_backend(self, ctx: dict) -> tuple[dict, list]:
        """When a containerized backend is active and ctx names a host ``checker_dir``,
        mount it read-only at a stable in-container path and rewrite the ctx value so
        the verifier finds it. Returns ``(ctx_for_exec, mounts)``.

        ``mounts`` is a list of ``(host_path, container_path, read_only)`` tuples the
        docker backend binds. Host-subprocess path: no mounts, ctx unchanged."""
        if self.exec_backend is None:
            return ctx, []
        cd = ctx.get("checker_dir")
        if not cd:
            return ctx, []
        return {**ctx, "checker_dir": "/checker"}, [(str(cd), "/checker", True)]

    def _docker_exec(self, workdir: str, runner_src: str, argv_tail: list, *,
                     timeout: float, mounts: list):
        """Run a runner inside the ISOLATED verifier container (the GPU seam).

        Writes ``_runner.py`` into ``workdir`` (already holding the module +
        ``llm_client.py``), bind-mounts ``workdir`` at ``/w``, and runs
        ``docker run ... <image> python /w/_runner.py <argv_tail...>`` with this
        backend's own cpus/memory/gpus and a default-deny network. NO LLM creds are
        passed in — the verifier container is cred-free, exactly like the Solver
        container. Returns a ``subprocess.CompletedProcess``."""
        b = self.exec_backend or {}
        wd = Path(workdir).resolve()
        (wd / "_runner.py").write_text(runner_src, encoding="utf-8")
        argv = ["docker", "run", "--rm",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "-v", f"{wd}:/w", "-w", "/w"]
        for host, cont, ro in mounts:
            argv += ["-v", f"{Path(host).resolve()}:{cont}" + (":ro" if ro else "")]
        if b.get("gpus"):
            argv += ["--gpus", str(b["gpus"])]
        if b.get("cpus") is not None:
            argv += ["--cpus", str(b["cpus"])]
        if b.get("memory_mb") is not None:
            argv += ["--memory", f"{int(b['memory_mb'])}m"]
        if not b.get("allow_internet", False):
            argv += ["--network", "none"]
        argv += [b.get("image") or "python:3.11-slim", "python", "/w/_runner.py", *argv_tail]
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)

    def run(self, payload: dict, ctx: dict, source: Optional[str] = None,
            *, env: Optional[dict] = None, timeout_s: Optional[float] = None) -> RunResult:
        """Execute a verifier source (current version by default) out-of-process.

        ``env`` is merged over ``os.environ`` for the subprocess — the ONLY place
        the orchestrator injects LLM creds, and it runs host-side (never in the
        Solver container). Absent → the subprocess inherits the plain environment.

        When ``self.exec_backend`` is set the verify runs inside an isolated
        container instead (the GPU-backed checker seam); ``timeout_s`` (else the
        backend's ``timeout_s``, else ``self.timeout_s``) bounds it — a real GPU
        scorer needs minutes, not the 10 s host default.
        """
        src = source if source is not None else self.current.source
        eff_timeout = (timeout_s if timeout_s is not None
                       else (self.exec_backend or {}).get("timeout_s") or self.timeout_s)
        ctx_x, mounts = self._ctx_for_backend(ctx)
        with tempfile.TemporaryDirectory() as d:
            vf = Path(d) / "verifier_mod.py"
            vf.write_text(src, encoding="utf-8")
            (Path(d) / "llm_client.py").write_text(_LLM_CLIENT_SRC, encoding="utf-8")
            try:
                if self.exec_backend is None:
                    proc = subprocess.run(
                        [sys.executable, "-c", _RUNNER, str(vf), json.dumps(payload), json.dumps(ctx)],
                        capture_output=True,
                        text=True,
                        timeout=eff_timeout,
                        env=_subprocess_env(env),
                        cwd=d,
                    )
                else:
                    proc = self._docker_exec(
                        d, _RUNNER,
                        ["verifier_mod.py", json.dumps(payload), json.dumps(ctx_x)],
                        timeout=eff_timeout, mounts=mounts)
            except subprocess.TimeoutExpired:
                return RunResult(False, 0.0, {}, error="verifier timeout")
        if proc.returncode != 0:
            return RunResult(False, 0.0, {}, error=(proc.stderr or "verifier crashed")[-500:])
        for line in proc.stdout.splitlines():
            if line.startswith("__VERIFY_RESULT__"):
                try:
                    out = json.loads(line[len("__VERIFY_RESULT__"):])
                except json.JSONDecodeError as e:
                    return RunResult(False, 0.0, {}, error=f"bad verifier output: {e}")
                return RunResult(
                    bool(out.get("feasible", False)),
                    float(out.get("raw", 0.0)),
                    dict(out.get("artifacts", {})),
                    error=None,
                )
        return RunResult(False, 0.0, {}, error="verifier produced no result line")

    def run_feedback(self, payload: dict, ctx: dict, verify_result: dict,
                     history: list, *, feedback_src: str,
                     env: Optional[dict] = None,
                     timeout_s: Optional[float] = None) -> dict:
        """Execute an agent-authored ``feedback`` module out-of-process.

        Mirrors ``run``: writes the feedback module + ``llm_client.py`` into a temp
        dir and runs ``_FEEDBACK_RUNNER``. Returns the module's dict (expected keys
        ``detail``/``artifacts``), or ``{}`` on any failure — the caller then falls
        back to the frozen disclosure so a feedback bug never breaks a solver turn.

        Honors ``self.exec_backend`` (containerized) like ``run``. Note feedback may
        need LLM creds; with a containerized backend those are NOT forwarded, so an
        LLM-verifier task keeps the host-subprocess backend (see EvalService).
        """
        eff_timeout = (timeout_s if timeout_s is not None
                       else (self.exec_backend or {}).get("timeout_s") or self.timeout_s)
        ctx_x, mounts = self._ctx_for_backend(ctx)
        with tempfile.TemporaryDirectory() as d:
            ff = Path(d) / "feedback_mod.py"
            ff.write_text(feedback_src, encoding="utf-8")
            (Path(d) / "llm_client.py").write_text(_LLM_CLIENT_SRC, encoding="utf-8")
            try:
                if self.exec_backend is None:
                    proc = subprocess.run(
                        [sys.executable, "-c", _FEEDBACK_RUNNER, str(ff),
                         json.dumps(payload), json.dumps(ctx),
                         json.dumps(verify_result), json.dumps(history)],
                        capture_output=True,
                        text=True,
                        timeout=eff_timeout,
                        env=_subprocess_env(env),
                        cwd=d,
                    )
                else:
                    proc = self._docker_exec(
                        d, _FEEDBACK_RUNNER,
                        ["feedback_mod.py", json.dumps(payload), json.dumps(ctx_x),
                         json.dumps(verify_result), json.dumps(history)],
                        timeout=eff_timeout, mounts=mounts)
            except subprocess.TimeoutExpired:
                return {}
        if proc.returncode != 0:
            return {}
        for line in proc.stdout.splitlines():
            if line.startswith("__FEEDBACK_RESULT__"):
                try:
                    out = json.loads(line[len("__FEEDBACK_RESULT__"):])
                except json.JSONDecodeError:
                    return {}
                return out if isinstance(out, dict) else {}
        return {}

    def validate_source(self, src: str, sample_payload: dict, ctx: dict) -> Optional[str]:
        """Smoke-test a candidate rewrite; return an error string or None if OK."""
        r = self.run(sample_payload, ctx, source=src)
        if r.error:
            return r.error
        return None
