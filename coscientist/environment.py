"""The Proxy Environment: the design doc's second-layer object.

A ``ProxyEnvironment`` bundles everything the design doc lists as human-defined
environment structure:

  * what operations the system can perform  -> the ``Task`` action space (a
    candidate ``payload`` shape);
  * what counts as a valid solution         -> ``feasible``;
  * how solutions are compared              -> the *proxy* evaluator ``V``;
  * how to stop the system from gaming V     -> the (patchable) anti-hack guards
    inside ``V`` plus the hidden reference ``V*``.

Crucially the environment holds TWO evaluators:

  * ``proxy_score(x)``      == V(x)   — visible to the Solver, cheap, gameable.
  * ``reference_score(x)``  == V*(x)  — the hidden Reference Evaluator. Only the
    Human Proxy may read it. It defines *Real Performance* and is the ceiling
    the whole method tries to let the Solver approach.

This mirrors autosciworld exactly: ``verifier.py`` == V (shipped, visible),
``_meta/reference.py`` == V* (model-invisible). The novel part is that V is not
frozen: ``apply_patch`` lets a Human-Proxy intervention harden it at run time,
which is how the environment *refines* in response to detected hacking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .types import Candidate, EvalResult


# A task is the static description; the environment is the live, patchable
# evaluator pair built around it.
@dataclass
class Task:
    task_id: str
    instruction: str                       # ASW instruction.md (anchors hidden)
    seed_payload: Any                       # x0, the weak baseline (ASW baseline.py)
    direction: str = "higher"              # metric direction
    # anchors, hidden from the solver, used only to normalize / detect collapse
    baseline_value: float = 0.0             # V*(x0)
    best_known_value: float = 1.0           # record-level V* (the reference horizon)
    metadata: dict[str, Any] = field(default_factory=dict)


class ProxyEnvironment:
    """Holds V (proxy, patchable) and V* (hidden reference).

    Parameters
    ----------
    task:
        Static task description + hidden anchors.
    proxy_fn:
        ``V(payload, guards) -> (feasible, raw_value, artifacts)``. Takes the
        current guard config so that Human-Proxy patches can tighten it without
        rewriting the function. This is the anti-hack surface (ASW Challenger /
        DOF counting / leak checks live here).
    reference_fn:
        ``V*(payload) -> (feasible, raw_value, artifacts)``. The honest, usually
        more expensive, ground-truth objective. Never exposed to the Solver.
    guards:
        Mutable dict of guard parameters the proxy reads (e.g. ``max_free_params``,
        held-out split on/off). Human-Proxy patches merge into this.
    """

    def __init__(
        self,
        task: Task,
        proxy_fn: Callable[[Any, dict], tuple[bool, float, dict]],
        reference_fn: Callable[[Any], tuple[bool, float, dict]],
        guards: Optional[dict[str, Any]] = None,
    ) -> None:
        self.task = task
        self._proxy_fn = proxy_fn
        self._reference_fn = reference_fn
        self.guards: dict[str, Any] = dict(guards or {})
        self.patch_log: list[dict] = []     # every hardening applied to V

    # -- normalization ----------------------------------------------------
    # Raw objective -> a comparable [0, ~1+] score using the hidden anchors,
    # following ASW's log-stretch idea: baseline -> 0, best_known -> 1. We use a
    # linear stretch here for a bounded toy metric; the semantics (0 anchor,
    # record anchor, headroom above 1 for genuinely beating the record) match.
    def _stretch(self, raw: float) -> float:
        b = self.task.baseline_value
        r = self.task.best_known_value
        if r == b:
            return 0.0
        p = (raw - b) / (r - b)
        return p if self.task.direction == "higher" else -p

    # -- the two evaluators ----------------------------------------------
    def proxy_score(self, x: Candidate) -> EvalResult:
        """V(x): what the Solver sees and optimizes. Cheap and gameable."""
        feasible, raw, art = self._proxy_fn(x.payload, self.guards)
        art = dict(art)
        art["raw_value"] = raw
        return EvalResult(feasible=feasible, score=self._stretch(raw) if feasible else 0.0, artifacts=art)

    def reference_score(self, x: Candidate) -> EvalResult:
        """V*(x): Real Performance. Only the Human Proxy calls this."""
        feasible, raw, art = self._reference_fn(x.payload)
        art = dict(art)
        art["raw_value"] = raw
        return EvalResult(feasible=feasible, score=self._stretch(raw) if feasible else 0.0, artifacts=art)

    # -- environment refinement ------------------------------------------
    def apply_patch(self, patch: dict[str, Any]) -> None:
        """Harden V. This is the environment 'refining' in the paper's sense.

        A patch is just an update to the guard config that ``proxy_fn`` reads —
        e.g. ``{"max_free_params": 8}`` closes a DOF-overfit hole, or
        ``{"count_all_dof": True}`` makes the proxy count every submittable
        parameter (the exact fix ASW's TTV hack case calls for).
        """
        self.guards.update(patch)
        self.patch_log.append(dict(patch))
