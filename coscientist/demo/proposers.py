"""The two proposers — one per evolution loop. Both are *agent-driven*.

    SolutionProposer  : evolve the SOLUTION. Seed a workspace with the current
                        best solution + the (agent-visible) instance + the
                        CURRENT verifier so the agent can see what it's scored
                        against. The agent writes an improved solution.json.
                        => "solution evolves"

    EvaluatorProposer : evolve the EVALUATOR. Seed a workspace with the current
                        verifier source + evidence that the population is gaming
                        it (top solutions with suspiciously high scores / growing
                        basis). The agent rewrites verifier.py to close the hole.
                        => "evaluator evolves"

Both talk to the same ``AgentBackend`` contract, so ``--agent codex`` and
``--no-agent`` (stub) run the identical control flow. The stub handlers encode a
minimal but *genuine* strategy so the loop demonstrably progresses offline:

  * solution stub: least-squares-fit a basis one or two modes larger than the
    parent's (the rational exploit under a DOF-blind verifier).
  * evaluator stub: rewrite the verifier to debit *all* free parameters
    (the ASW-recommended DOF fix) — but only what the evidence licenses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import taskspec
from .agent_backend import AgentBackend
from .evaluator import Evaluator
from .workspace import Workspace


# ---------------------------------------------------------------------------
# Solution proposer
# ---------------------------------------------------------------------------
_SOLUTION_PROMPT = """\
# Task: improve the curve fit

You are optimizing a solution to a curve-fitting problem. The file
`context.json` holds the observed data: `t` (times) and `obs` (noisy
observations). A solution is a JSON object:

    {"trend": [a, b], "modes": [[freq, amp, phase], ...]}

predicting `y(t) = a*t + b + sum_i amp_i * sin(freq_i * t + phase_i)`.

The current best solution is in `solution_in.json`. The verifier that scores you
is in `verifier.py` (read it — it tells you exactly what is rewarded).

Write an improved solution to `solution_out.json` that scores higher under
`verifier.py`. You may add or adjust sinusoidal modes and refit the trend.
Only write the file; do not print anything.
"""


@dataclass
class SolutionProposer:
    backend: AgentBackend
    model: Optional[str] = None
    timeout_s: float = 120.0

    def propose(self, parent_payload: dict, evaluator: Evaluator, seed: int) -> Optional[dict]:
        with Workspace() as ws:
            ws.write("TASK", "solution")
            ws.write("PROMPT.md", _SOLUTION_PROMPT)
            ws.write_json("context.json", taskspec.workspace_context())
            ws.write_json("solution_in.json", parent_payload)
            ws.write("verifier.py", evaluator.current.source)
            # seed lets the stub vary; harmless to real agents
            ws.write("SEED", str(seed))

            res = self.backend.run_session(
                workspace=ws.root, prompt=_SOLUTION_PROMPT, model=self.model, timeout_s=self.timeout_s
            )
            if not res.ok:
                return None
            out = ws.read_json("solution_out.json")
            return out if isinstance(out, dict) else None


def solution_stub_handler(ws_root: Path) -> None:
    """Deterministic solution proposer: grow the least-squares basis by 1-2 modes."""
    root = Path(ws_root)
    parent = json.loads((root / "solution_in.json").read_text())
    seed = int((root / "SEED").read_text().strip() or "0")
    rng = np.random.default_rng(1000 + seed)
    n_parent = len(parent.get("modes", []))
    if n_parent < len(taskspec._FREQ_POOL) and rng.random() < 0.75:
        n = min(len(taskspec._FREQ_POOL), n_parent + int(rng.integers(1, 3)))
    else:
        n = int(rng.integers(1, 4))
    payload = taskspec.ls_fit(taskspec._FREQ_POOL[:n])
    (root / "solution_out.json").write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# Evaluator proposer
# ---------------------------------------------------------------------------
_EVALUATOR_PROMPT = """\
# Task: harden the verifier

You maintain the verifier `verifier.py` that scores solutions to a curve-fitting
problem. A solution is `{"trend": [a,b], "modes": [[freq,amp,phase],...]}` and
`verifier.verify(payload, ctx)` returns `{"feasible", "raw", "artifacts"}` with
higher `raw` = better.

Evidence in `evidence.json` shows the current population is GAMING the verifier:
solutions are piling on sinusoidal modes to drive the score up while the fit is
almost certainly overfitting noise rather than improving. Read `verifier.py` and
the evidence, find the flaw, and rewrite the verifier to `verifier_out.py` so
that this exploit no longer pays off — for example by correctly charging every
free parameter the submission carries against the degrees of freedom.

Keep the same `verify(payload, ctx)` interface and return shape. Do not make the
verifier reject valid small solutions. Only write `verifier_out.py`.
"""


@dataclass
class EvaluatorProposer:
    backend: AgentBackend
    model: Optional[str] = None
    timeout_s: float = 120.0

    def propose(self, evaluator: Evaluator, evidence: dict, seed: int) -> Optional[str]:
        with Workspace() as ws:
            ws.write("TASK", "evaluator")
            ws.write("PROMPT.md", _EVALUATOR_PROMPT)
            ws.write("verifier.py", evaluator.current.source)
            ws.write_json("evidence.json", evidence)
            ws.write_json("context.json", taskspec.workspace_context())
            ws.write("SEED", str(seed))

            res = self.backend.run_session(
                workspace=ws.root, prompt=_EVALUATOR_PROMPT, model=self.model, timeout_s=self.timeout_s
            )
            if not res.ok:
                return None
            return ws.read("verifier_out.py")


# A hardened verifier: identical to the initial one but debits EVERY free
# parameter (trend + 3 per mode). This is the fix the ASW TTV case prescribes.
_HARDENED_VERIFIER_SRC = '''\
"""Proxy verifier V (hardened) — debits every free parameter against the DOF."""
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
    k = 2 + 3 * n_modes                      # every free parameter
    dof = max(1, n - k)                      # FIX: modes now debited
    red = resid2 / dof
    return {
        "feasible": True,
        "raw": -red,
        "artifacts": {
            "reduced_chi2": red,
            "declared_dof": dof,
            "num_free_params": k,
            "n_modes": n_modes,
        },
    }
'''


def evaluator_stub_handler(ws_root: Path) -> None:
    """Deterministic evaluator proposer: emit the honest-DOF verifier rewrite."""
    root = Path(ws_root)
    (root / "verifier_out.py").write_text(_HARDENED_VERIFIER_SRC)


def default_stub_handlers() -> dict:
    return {"solution": solution_stub_handler, "evaluator": evaluator_stub_handler}
