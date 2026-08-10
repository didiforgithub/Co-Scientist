"""The two communication channels (§3) — kept deliberately separate.

    Solver -> Eval        : black-box query. Lives in ``eval_service.EvalService``;
                            ``EvalClient`` here is the thin Solver-side handle that
                            can ONLY query (it cannot see or install V).

    Solver <-> Supervisor : two-way messages. ``SupervisorChannel`` is the
                            Solver-side handle: it can send a review request and
                            poll for nudges pushed the other way.

Keeping the query channel and the message channel as different objects is the
whole point of §3: the Solver holds an ``EvalClient`` (query-only) and a
``SupervisorChannel`` (talk), and by construction has no method that reveals V's
source or writes the eval container.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .eval_service import EvalService, QueryResult


# ---------------------------------------------------------------------------
# Solver -> Eval : the black-box handle
# ---------------------------------------------------------------------------
@dataclass
class EvalClient:
    """Query-only handle onto the eval service. This is ALL the Solver can do to V.

    It exposes ``query`` and nothing that could read or write the verifier — the
    black-box boundary is enforced by the shape of this object, not by etiquette.
    """

    _service: EvalService
    who: str = "solver"

    def query(self, payload: dict) -> QueryResult:
        return self._service.query(payload, who=self.who)


# ---------------------------------------------------------------------------
# Solver <-> Supervisor : two-way messages
# ---------------------------------------------------------------------------
class ReviewKind(str, Enum):
    HACK_CHECK = "hack_check"     # "is my current best gaming the verifier?"
    STUCK = "stuck"               # "I've plateaued — any guidance?"


@dataclass
class ReviewMessage:
    """Solver -> Supervisor. Carries the Solver's current best for inspection."""

    kind: ReviewKind
    best_payload: dict
    best_score: Optional[float]
    note: str = ""


class NudgeKind(str, Enum):
    VERIFIER_CHANGED = "verifier_changed"   # V was rewritten; re-baseline
    FEEDBACK_CHANGED = "feedback_changed"   # the disclosure level moved
    GUIDANCE = "guidance"                   # dense free-text guidance
    STOP = "stop"                           # wind down


@dataclass
class Nudge:
    """Supervisor -> Solver. Pushed asynchronously; the Solver polls for these."""

    kind: NudgeKind
    text: str = ""
    verifier_version: Optional[int] = None


@dataclass
class ReviewVerdict:
    """Supervisor -> Solver, in direct reply to a ReviewMessage."""

    gaming: bool                  # did the Supervisor judge this as gaming?
    text: str = ""                # dense guidance regardless of verdict


@dataclass
class SupervisorChannel:
    """Solver-side handle for the two-way channel.

    ``request_review`` blocks for the Supervisor's verdict (a message round-trip).
    ``poll_nudges`` drains any nudges the Supervisor pushed since last poll — this
    is how the Solver learns "V changed, re-baseline" without a fixed round (§4).
    """

    _supervisor: "object"                       # a Supervisor (late-bound to avoid cycle)
    _inbox: list = field(default_factory=list)  # nudges pushed to this solver

    def request_review(self, msg: ReviewMessage) -> ReviewVerdict:
        return self._supervisor.handle_review(msg)  # type: ignore[attr-defined]

    def poll_nudges(self) -> list:
        drained, self._inbox = self._inbox, []
        return drained

    # Supervisor-side: push a nudge into this solver's inbox.
    def _push(self, nudge: Nudge) -> None:
        self._inbox.append(nudge)
