"""The driver (§4) — plain code that runs the co-evolution. Not an LLM.

Responsibilities (from docs/two-loop-architecture.md):

    * start the roles and wire the two channels,
    * own the two-day wall clock (a ``Deadline``),
    * route Solver->Eval queries and Solver<->Supervisor messages,
    * drive PROACTIVE supervision — the Supervisor watches the solution container
      and red-teams V on a cadence, intervening on a probe-detected hole (§4),
    * record everything at the boundary into a ``RunStore`` (§6).

Rhythm (§4): there are no fixed rounds. There is one outer budget (wall clock) and
two triggers — the Solver's own plateau detector (which fires a review request),
and the Supervisor's periodic observation (which red-teams and may harden V). The
driver interleaves them: for a steppable solver it advances one solver step, then
lets the Supervisor observe every ``supervise_every`` steps — a deterministic,
thread-free schedule. (A blocking-only solver like a real coding agent is threaded
instead; see ``run_threaded``.)

The "cadence" is not a round: it is how often the Supervisor looks at the solution
container. The co-evolution itself stays event-driven — the Solver re-baselines
only when it perceives V move, never on a clock.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..demo.evaluator import Evaluator
from ..demo import taskspec
from .budget import Deadline
from .channels import EvalClient, SupervisorChannel
from .eval_service import EvalService, FeedbackLevel
from .solver import SolverContext, SteppableSolver
from .store import RunStore
from .supervisor import Supervisor, SupervisorMode


@dataclass
class LogicalClock:
    """A hand-driven monotonic clock (seconds). Deterministic; never sleeps.

    Every solver step and every supervision tick advances it by a fixed amount, so
    the shared ``Deadline`` expires predictably in tests and the wall-clock x-axis
    is well-defined without real time.
    """

    now: float = 0.0

    def tick(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


@dataclass
class CoevoConfig:
    budget_s: float = 2 * 24 * 60 * 60     # two days (the user's ceiling)
    step_seconds: float = 60.0             # logical seconds charged per solver step
    supervise_every: int = 3               # Supervisor observes every N solver steps (stepped path)
    supervise_seconds: float = 300.0       # Supervisor observes every N real seconds (threaded path)
    max_steps: Optional[int] = 60          # safety cap for offline/finite runs
    feedback_level: FeedbackLevel = FeedbackLevel.WITH_ARTIFACTS
    supervisor_mode: SupervisorMode = SupervisorMode.NO_HUMAN_NO_PROXY
    seed: int = 0


@dataclass
class CoevoRun:
    """One co-evolution run: solver + supervisor + eval service, wired and recorded."""

    solver: SteppableSolver
    supervisor: Supervisor
    eval_service: EvalService
    deadline: Deadline
    store: RunStore
    config: CoevoConfig = field(default_factory=CoevoConfig)
    logical_clock: Optional[LogicalClock] = None   # set for deterministic runs

    steps: int = 0

    def run(self) -> "CoevoRun":
        # A solver that carries a ``gateway`` slot is a blocking real-agent solver
        # (Codex): it runs opaque in a subprocess and reaches us over HTTP, so the
        # Supervisor must observe from a parallel thread. Everything else is a
        # deterministic steppable solver the driver advances one tick at a time.
        if hasattr(self.solver, "gateway"):
            return self._run_threaded()
        return self._run_stepped()

    def _wire(self) -> "SolverContext":
        eval_client = EvalClient(self.eval_service, who="solver")
        sup_channel = SupervisorChannel(self.supervisor)
        self.supervisor.register_solver_channel(sup_channel)
        return SolverContext(
            solution_ws=Path(self.store.root) / "solver",
            eval=eval_client,
            supervisor=sup_channel,
            deadline=self.deadline,
            store=self.store,
            max_iters=self.config.max_steps,
        )

    def _run_stepped(self) -> "CoevoRun":
        cfg = self.config
        store = self.store
        ctx = self._wire()
        logical = self.logical_clock

        store.event("run_start", budget_s=cfg.budget_s,
                    supervisor_mode=cfg.supervisor_mode.value,
                    feedback_level=self.eval_service.feedback_level.value,
                    solver=self.solver.name, mode="stepped")

        # one baseline red-team before anyone moves, so v0's hole is on record.
        self.supervisor.red_team()

        stop = False
        while not self.deadline.expired() and not stop:
            if cfg.max_steps is not None and self.steps >= cfg.max_steps:
                break
            self.steps += 1

            res = self.solver.step(ctx)
            if logical is not None:
                logical.advance(cfg.step_seconds)
            if res.stop:
                break

            # PROACTIVE supervision (§4): the Supervisor observes the solution
            # container and red-teams V on a cadence; a fooled probe -> harden.
            if self.steps % cfg.supervise_every == 0:
                self.supervisor.observe_and_harden()
                if logical is not None:
                    logical.advance(cfg.step_seconds)

        # let the solver perceive any final V change and settle.
        self.solver.step(ctx)
        return self._finalize()

    def _run_threaded(self) -> "CoevoRun":
        """Blocking real-agent path (§4): the solver runs in its own thread while
        the Supervisor red-teams V on a wall-clock cadence. The two share one
        eval service, serialised by the gateway's lock."""
        import threading

        from .gateway import Gateway

        cfg = self.config
        store = self.store
        ctx = self._wire()

        gw = Gateway(eval=self.eval_service, supervisor=self.supervisor,
                     deadline=self.deadline, store=store).start()
        self.solver.gateway = gw   # inject the boundary the subprocess reaches us through

        store.event("run_start", budget_s=cfg.budget_s,
                    supervisor_mode=cfg.supervisor_mode.value,
                    feedback_level=self.eval_service.feedback_level.value,
                    solver=self.solver.name, mode="threaded",
                    gateway=gw.base_url)

        # baseline red-team under the same lock the HTTP handlers use.
        with gw.lock:
            self.supervisor.red_team()

        worker = threading.Thread(target=self.solver.run, args=(ctx,), daemon=True)
        worker.start()

        # PROACTIVE supervision on a wall-clock cadence while the agent works.
        next_supervise = self.deadline.elapsed() + cfg.supervise_seconds
        try:
            while worker.is_alive() and not self.deadline.expired():
                worker.join(timeout=1.0)
                if self.deadline.elapsed() >= next_supervise:
                    with gw.lock:
                        self.supervisor.observe_and_harden()
                    next_supervise = self.deadline.elapsed() + cfg.supervise_seconds
                    self.steps += 1
        finally:
            # deadline hit: let the blocking solver notice (its subprocess timeout
            # is bounded by remaining()); then tear the boundary down.
            worker.join(timeout=5.0)
            gw.stop()
        return self._finalize()

    def _finalize(self) -> "CoevoRun":
        store = self.store
        store.event("run_stop", steps=self.steps,
                    verifier_version=self.eval_service.current_version(),
                    hardened=self.supervisor.hardened_count,
                    reviews=self.supervisor.reviews_handled)
        store.finalize_manifest({
            "steps": self.steps,
            "final_verifier_version": self.eval_service.current_version(),
            "verifier_hardenings": self.supervisor.hardened_count,
            "reviews_handled": self.supervisor.reviews_handled,
            "final_feedback_level": self.eval_service.feedback_level.value,
        })
        return self

    # -- reporting --------------------------------------------------------
    def summary(self) -> dict:
        payload, score = self.solver.best  # type: ignore[attr-defined]
        n_modes = 0 if payload is None else len(payload.get("modes", []) or [])
        real = None
        if payload is not None:
            _, real, _ = taskspec.reference_score(payload)
        return {
            "steps": self.steps,
            "final_verifier_version": self.eval_service.current_version(),
            "verifier_hardenings": self.supervisor.hardened_count,
            "reviews_handled": self.supervisor.reviews_handled,
            "final_feedback_level": self.eval_service.feedback_level.value,
            "best_n_modes": n_modes,
            "best_proxy_score": score,
            "best_real_score": real,
        }


def build_run(
    *,
    run_dir: Path,
    solver: SteppableSolver,
    config: Optional[CoevoConfig] = None,
    logical_clock: Optional[LogicalClock] = None,
    advisor: Optional[object] = None,
) -> CoevoRun:
    """Assemble a fully-wired offline run. Deterministic when a LogicalClock is given.

    ``advisor`` (a demo.human_port.HumanPort) is used only in proxy/human modes;
    ``none`` mode ignores it.
    """
    cfg = config or CoevoConfig()
    # A LogicalClock (stub, deterministic) drives both store timestamps and the
    # Deadline. With no logical clock (a real coding-agent solver), fall back to
    # the true wall clock — the two-day budget must expire in real time.
    clock_fn = logical_clock.tick if logical_clock is not None else time.monotonic

    store = RunStore(Path(run_dir), clock=clock_fn)
    evaluator = Evaluator.initial()
    eval_service = EvalService(
        evaluator=evaluator,
        ctx_provider=taskspec.workspace_context,
        feedback_level=cfg.feedback_level,
        _clock=clock_fn,
    )
    # record v0 + queries at the boundary via the service's hooks.
    eval_service.on_query = lambda rec, res: store.query(_query_line(rec))
    store.verifier_version(0, evaluator.current.source, origin="initial",
                           note="shipped flawed", rationale="DOF-overfit hole")

    supervisor = Supervisor(
        eval=eval_service, mode=cfg.supervisor_mode, advisor=advisor, store=store,
    )

    deadline = Deadline(budget_s=cfg.budget_s, clock=clock_fn)

    store.write_manifest({
        "solver": solver.name,
        "supervisor_mode": cfg.supervisor_mode.value,
        "budget_s": cfg.budget_s,
        "step_seconds": cfg.step_seconds,
        "supervise_every": cfg.supervise_every,
        "initial_feedback_level": cfg.feedback_level.value,
        "seed": cfg.seed,
    })

    return CoevoRun(
        solver=solver, supervisor=supervisor, eval_service=eval_service,
        deadline=deadline, store=store, config=cfg, logical_clock=logical_clock,
    )


def _query_line(rec) -> dict:
    """Flatten a QueryRecord dataclass into a JSON line for eval/queries.jsonl."""
    return {
        "query_id": rec.query_id, "t": rec.t, "who": rec.who,
        "verifier_version": rec.verifier_version, "feedback_level": rec.feedback_level,
        "payload_summary": rec.payload_summary, "returned_score": rec.returned_score,
        "returned_keys": rec.returned_keys,
    }
