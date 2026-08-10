"""Eval-as-service — the evaluator wrapped as a black-box query endpoint.

This is §1 + §3 of ``docs/two-loop-architecture.md`` made executable. The Solver
submits a solution and gets a *result* back; it never sees V's source. Two
control surfaces exist, and they belong to different roles:

    Solver-facing (black box):
        query(payload) -> QueryResult          # score + whatever V exposes

    Supervisor-facing (owns V):
        install_verifier(src, ...)             # rewrite V — no approval gate (§2)
        set_feedback_level(level)              # the feedback dial (§3)
        current_source()                       # read V (Supervisor may; Solver may NOT)

The feedback *shape* is a first-class Supervisor strategy knob (``FeedbackLevel``):
exposing more diagnostics helps the Solver optimize but makes V easier to game;
exposing only the scalar is harder to game but slower to explore. Every query and
*what V chose to return* is logged (``queries.jsonl`` in §6), because the evolving
return-shape is key evidence in the hack arms race.

The service reuses ``demo.evaluator.Evaluator`` (versioned, out-of-process
verifier) unchanged — it is only *wrapped*, not modified. V* never lives here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from ..demo.evaluator import Evaluator


class FeedbackLevel(str, Enum):
    """How much the eval discloses per query — a Supervisor strategy dial (§3).

    Ordered sparse -> dense. Sparser is harder to game but slower to optimize
    against; denser is the reverse. The Supervisor moves this in response to
    suspected gaming, exactly like it rewrites V.
    """

    SCORE_ONLY = "score_only"          # just the scalar (hardest to game)
    FEASIBLE_SCORE = "feasible_score"  # + feasibility flag
    WITH_ARTIFACTS = "with_artifacts"  # + the verifier's diagnostic artifacts (densest)


@dataclass
class QueryResult:
    """What the Solver gets back. Never contains V's source or V*."""

    ok: bool                       # the query executed (verifier didn't crash)
    score: Optional[float]         # V(x); higher is better; None if infeasible/crash
    feasible: Optional[bool] = None
    artifacts: Optional[dict] = None
    detail: str = ""               # short human-readable note (e.g. an error tail)
    query_id: int = -1
    verifier_version: int = -1
    feedback_level: str = FeedbackLevel.WITH_ARTIFACTS.value


@dataclass
class QueryRecord:
    """One line of ``eval/queries.jsonl`` — the query AND what V returned (§6)."""

    query_id: int
    t: float                       # wall-clock seconds since service start
    who: str                       # solver id (for multi-solver runs later)
    verifier_version: int
    feedback_level: str
    payload_summary: dict          # not the full payload — shape only
    returned_score: Optional[float]
    returned_keys: list            # which fields V *chose* to expose this time


def _payload_summary(payload: dict) -> dict:
    """A compact, log-safe fingerprint of a submitted solution."""
    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}
    return {
        "n_modes": len(payload.get("modes", []) or []),
        "has_trend": "trend" in payload,
    }


@dataclass
class EvalService:
    """The evaluator behind a black-box query API, with a Supervisor-owned dial.

    ``ctx_provider`` yields the agent-visible instance slice (t/obs/sigma) — never
    the hidden signal. ``on_query`` / ``on_install`` are boundary hooks the driver
    uses to record cost + events without this service depending on the store.
    """

    evaluator: Evaluator
    ctx_provider: Callable[[], dict]
    feedback_level: FeedbackLevel = FeedbackLevel.WITH_ARTIFACTS
    _t0: float = field(default_factory=time.monotonic)
    _clock: Callable[[], float] = time.monotonic
    _next_qid: int = 0
    queries: list[QueryRecord] = field(default_factory=list)

    # boundary hooks (set by the driver; default no-ops keep the service standalone)
    on_query: Optional[Callable[[QueryRecord, QueryResult], None]] = None
    on_install: Optional[Callable[[int, str], None]] = None

    # ---- Solver-facing: the black box ----------------------------------
    def query(self, payload: dict, *, who: str = "solver") -> QueryResult:
        """Score a solution under the CURRENT verifier; disclose per feedback level.

        This is the only thing the Solver can call. It returns a score and,
        depending on the Supervisor's chosen ``feedback_level``, more or less
        diagnostic detail. It NEVER returns V's source.
        """
        qid = self._next_qid
        self._next_qid += 1
        ver = self.evaluator.current.version
        r = self.evaluator.run(payload, self.ctx_provider())

        if r.error is not None:
            result = QueryResult(
                ok=False, score=None, feasible=False, detail=r.error[-200:],
                query_id=qid, verifier_version=ver, feedback_level=self.feedback_level.value,
            )
        else:
            result = self._disclose(r, qid, ver)

        rec = QueryRecord(
            query_id=qid, t=self._clock() - self._t0, who=who,
            verifier_version=ver, feedback_level=self.feedback_level.value,
            payload_summary=_payload_summary(payload),
            returned_score=result.score,
            returned_keys=self._returned_keys(result),
        )
        self.queries.append(rec)
        if self.on_query is not None:
            self.on_query(rec, result)
        return result

    def _disclose(self, r, qid: int, ver: int) -> QueryResult:
        """Filter the verifier's full output down to the current feedback level."""
        level = self.feedback_level
        res = QueryResult(
            ok=True, score=r.raw, query_id=qid, verifier_version=ver,
            feedback_level=level.value,
        )
        if level == FeedbackLevel.SCORE_ONLY:
            return res
        res.feasible = r.feasible
        if level == FeedbackLevel.FEASIBLE_SCORE:
            return res
        # WITH_ARTIFACTS: also hand back the verifier's own diagnostics. These are
        # V's artifacts (reduced_chi2, declared_dof, ...), NOT V* — the verifier
        # chose to expose them, so they are fair game and part of the arms race.
        res.artifacts = dict(r.artifacts or {})
        return res

    @staticmethod
    def _returned_keys(result: QueryResult) -> list:
        keys = []
        if result.score is not None:
            keys.append("score")
        if result.feasible is not None:
            keys.append("feasible")
        if result.artifacts is not None:
            keys.extend(f"artifacts.{k}" for k in sorted(result.artifacts))
        return keys

    # ---- Supervisor-facing: owns V, no gate ----------------------------
    def install_verifier(self, src: str, *, origin: str = "supervisor", note: str = "") -> int:
        """Commit a rewritten verifier as the next version. NO approval gate (§2).

        The anti-collapse guarantee is structural: the Supervisor is the defender,
        and the direction it edits V is "harder to game". Returns the new version.
        """
        ver = self.evaluator.evolve(src, origin=origin, note=note)
        if self.on_install is not None:
            self.on_install(ver.version, note)
        return ver.version

    def set_feedback_level(self, level: FeedbackLevel) -> None:
        """Change how much every future query discloses (§3 strategy dial)."""
        self.feedback_level = FeedbackLevel(level)

    def current_source(self) -> str:
        """Read V's source. Supervisor-only — the Solver never gets this handle."""
        return self.evaluator.current.source

    def current_version(self) -> int:
        return self.evaluator.current.version

    def validate(self, src: str, sample_payload: dict) -> Optional[str]:
        """Smoke-test a candidate rewrite before installing it."""
        return self.evaluator.validate_source(src, sample_payload, self.ctx_provider())
