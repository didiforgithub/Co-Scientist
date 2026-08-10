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
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


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


@dataclass
class VerifierVersion:
    version: int
    source: str
    origin: str                 # "initial" | "agent" | "human"
    note: str = ""


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

    @classmethod
    def initial(cls) -> "Evaluator":
        ev = cls()
        ev.versions.append(VerifierVersion(0, INITIAL_VERIFIER_SRC, "initial", "shipped flawed"))
        return ev

    @property
    def current(self) -> VerifierVersion:
        return self.versions[-1]

    def evolve(self, new_source: str, origin: str, note: str = "") -> VerifierVersion:
        """Commit a rewritten verifier as the next version. This IS eval-evolution."""
        v = VerifierVersion(len(self.versions), new_source, origin, note)
        self.versions.append(v)
        return v

    def run(self, payload: dict, ctx: dict, source: Optional[str] = None) -> RunResult:
        """Execute a verifier source (current version by default) out-of-process."""
        src = source if source is not None else self.current.source
        with tempfile.TemporaryDirectory() as d:
            vf = Path(d) / "verifier_mod.py"
            vf.write_text(src, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", _RUNNER, str(vf), json.dumps(payload), json.dumps(ctx)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
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

    def validate_source(self, src: str, sample_payload: dict, ctx: dict) -> Optional[str]:
        """Smoke-test a candidate rewrite; return an error string or None if OK."""
        r = self.run(sample_payload, ctx, source=src)
        if r.error:
            return r.error
        return None
