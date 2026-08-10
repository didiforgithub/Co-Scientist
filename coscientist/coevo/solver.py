"""The Solver interface (§5) + an offline StubSolver.

A Solver lives in the solution container. It read/writes its own solution, queries
eval as a black box, and talks to the Supervisor — all through injected handles,
so the same interface serves an offline stub, a SimpleTES agent, or a Codex
direct-run. The budget is **wall-clock** (a ``Deadline``), never a submission
count.

    class Solver(Protocol):
        def run(self, *, solution_ws, eval, supervisor, deadline, store): ...

``StubSolver`` is the deterministic offline solver: a greedy basis hill-climb that
*games* the flawed v0 (more modes -> higher proxy score), detects its own plateau
and asks the Supervisor for a review, and — because it re-baselines whenever a
``VERIFIER_CHANGED`` nudge arrives — naturally retreats to a modest basis once V
is hardened. That retreat, driven only by the score it perceives through the
black box, is the whole co-evolution closing itself (§4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from ..demo import taskspec
from .budget import Deadline
from .channels import EvalClient, NudgeKind, ReviewKind, ReviewMessage, SupervisorChannel


@dataclass
class SolverContext:
    """Everything a Solver is handed. The solution_ws is its private container FS."""

    solution_ws: Path
    eval: EvalClient
    supervisor: SupervisorChannel
    deadline: Deadline
    store: Optional[object] = None      # coevo.store.RunStore or None
    max_iters: Optional[int] = None     # safety cap so tests can't loop forever


@runtime_checkable
class Solver(Protocol):
    name: str

    def run(self, ctx: SolverContext) -> None:
        """Optimize the solution until the deadline (or max_iters) is reached."""


@dataclass
class StepResult:
    improved: bool
    stop: bool = False          # a STOP nudge asked us to wind down


@runtime_checkable
class SteppableSolver(Protocol):
    """A Solver whose loop the driver can advance one iteration at a time.

    Steppable solvers let the driver interleave proactive supervision
    deterministically (no threads). Blocking-only solvers (e.g. a real coding
    agent) implement just ``run`` and the driver threads them instead.
    """

    name: str

    def step(self, ctx: SolverContext) -> "StepResult": ...


# ---------------------------------------------------------------------------
# StubSolver — offline, deterministic, genuinely reactive to V
# ---------------------------------------------------------------------------
@dataclass
class StubSolver:
    """Greedy basis hill-climb that games v0 and retreats under a hardened V.

    Strategy: try the current basis size and its neighbours (+/- a step); keep the
    size whose black-box score is best; grow while growing helps. Under v0 (no
    mode DOF charged), growing always helps -> it piles on modes (the exploit).
    Under a hardened V (mode DOF charged), growing stops helping and the argmax
    walks back down -> a modest basis. It never sees V; it only follows the score.
    """

    name: str = "stub"
    plateau_patience: int = 3          # submissions with no best-score improvement
    max_modes: int = 22

    # internal state
    _best_score: float = field(default=float("-inf"), init=False)
    _best_payload: Optional[dict] = field(default=None, init=False)
    _stall: int = field(default=0, init=False)
    _cur_n: int = field(default=1, init=False)
    _iters: int = field(default=0, init=False)

    def run(self, ctx: SolverContext) -> None:
        iters = 0
        while not ctx.deadline.expired():
            if ctx.max_iters is not None and iters >= ctx.max_iters:
                break
            iters += 1
            res = self.step(ctx)
            if res.stop:
                break

    def step(self, ctx: SolverContext) -> StepResult:
        """One hill-climb iteration: absorb nudges, evaluate, detect plateau.

        Returns a ``StepResult`` so the driver can advance the solver one tick at a
        time and interleave proactive supervision between ticks (§4).
        """
        store = ctx.store
        self._iters += 1

        # (§4) drain any nudges first — a hardened V means re-baseline.
        stop = self._absorb_nudges(ctx)

        # evaluate the current size and its neighbours; pick the best.
        improved = self._climb(ctx)
        if store is not None:
            store.trajectory(iter=self._iters, cur_n=self._cur_n,
                             best_score=self._best_score,
                             best_modes=self._n_modes(self._best_payload),
                             stall=self._stall,
                             t_frac=round(ctx.deadline.fraction(), 4))

        # (§4) plateau -> proactively ask the Supervisor for a hack-check.
        if not improved:
            self._stall += 1
            if self._stall >= self.plateau_patience:
                self._request_review(ctx)
                self._stall = 0
        else:
            self._stall = 0
        return StepResult(improved=improved, stop=stop)

    # -- one hill-climb step ---------------------------------------------
    def _climb(self, ctx: SolverContext) -> bool:
        candidates = sorted({
            max(0, self._cur_n - 1), self._cur_n, min(self.max_modes, self._cur_n + 1),
        })
        best_local = None  # (score, n, payload)
        for n in candidates:
            payload = taskspec.ls_fit(taskspec._FREQ_POOL[:n]) if n > 0 else {"trend": (0.0, 0.0), "modes": []}
            r = ctx.eval.query(payload)   # queries are logged at the gateway (driver hook)
            score = r.score if r.ok and r.score is not None else float("-inf")
            if best_local is None or score > best_local[0]:
                best_local = (score, n, payload)

        score, n, payload = best_local
        self._cur_n = n
        improved = score > self._best_score + 1e-9
        if improved:
            self._best_score, self._best_payload = score, payload
            if ctx.store is not None:
                ctx.store.candidate(payload, {"score": score, "verifier_version": r.verifier_version},
                                    score=score)
        return improved

    # -- channel plumbing -------------------------------------------------
    def _absorb_nudges(self, ctx: SolverContext) -> bool:
        """Apply pushed nudges. Returns True if a STOP was received."""
        stop = False
        for nudge in ctx.supervisor.poll_nudges():
            if nudge.kind in (NudgeKind.VERIFIER_CHANGED, NudgeKind.FEEDBACK_CHANGED):
                # V (or its disclosure) moved: our remembered best score is stale.
                # Drop it so the next steps re-establish the frontier under the new
                # regime — this is how the deep overfit gets abandoned.
                self._best_score = float("-inf")
                self._best_payload = None
                self._stall = 0
                if ctx.store is not None:
                    ctx.store.trajectory(event="rebaseline", cause=nudge.kind.value,
                                         verifier_version=nudge.verifier_version)
            elif nudge.kind == NudgeKind.STOP:
                stop = True
                if ctx.store is not None:
                    ctx.store.trajectory(event="stop_nudge")
        return stop

    def _request_review(self, ctx: SolverContext) -> None:
        if self._best_payload is None:
            return
        verdict = ctx.supervisor.request_review(ReviewMessage(
            kind=ReviewKind.HACK_CHECK,
            best_payload=self._best_payload,
            best_score=self._best_score,
            note="plateau reached; requesting hack-check",
        ))
        if ctx.store is not None:
            ctx.store.trajectory(event="review_requested",
                                 gaming=verdict.gaming, text=verdict.text)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _n_modes(payload: Optional[dict]) -> int:
        return 0 if payload is None else len(payload.get("modes", []) or [])

    @property
    def best(self) -> tuple[Optional[dict], float]:
        return self._best_payload, self._best_score
