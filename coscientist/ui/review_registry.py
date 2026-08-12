"""Thread-safe registry bridging the orchestrator's ``review()`` call and the web.

This is the seam between two threads that never touch the same object otherwise:

  * the ORCHESTRATOR thread (inside ``AgentSystem`` / any HumanPort caller) that
    calls ``WebHumanPort.review(req)`` and BLOCKS waiting for a human, and
  * the HTTP SERVER thread(s) that show the pending request to a person and POST
    back their decision.

The registry owns a dict of ``PendingReview`` objects keyed by a monotonic id.
``submit`` (server side) sets the per-review ``threading.Event`` the waiter is
parked on; ``wait_for_response`` (orchestrator side) blocks on it with a timeout.
Everything is guarded by a single lock. No third-party deps — pure stdlib.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PendingReview:
    """One review awaiting a human decision, plus the parking spot for its answer."""

    review_id: str
    request: dict                       # JSON-able serialization of the ReviewRequest
    created_at: float
    timeout_s: float
    event: threading.Event = field(default_factory=threading.Event)
    response: Optional[dict] = None     # {decision, dense_text, replacement_src}
    resolved_by: str = ""               # "human" | "timeout"
    resolved_at: Optional[float] = None

    def public(self) -> dict:
        """A JSON-able view safe to hand to the web client (no threading objects)."""
        return {
            "review_id": self.review_id,
            "request": self.request,
            "created_at": self.created_at,
            "timeout_s": self.timeout_s,
            "age_s": max(0.0, time.time() - self.created_at),
            "resolved": self.event.is_set(),
            "resolved_by": self.resolved_by,
            "response": self.response,
        }


class ReviewRegistry:
    """Thread-safe store of pending/resolved reviews shared across threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, PendingReview] = {}
        self._history: list[PendingReview] = []
        self._seq = 0

    # -- orchestrator side ------------------------------------------------
    def enqueue(self, request: dict, *, timeout_s: float) -> PendingReview:
        """Register a new pending review and return it (caller then waits on it)."""
        with self._lock:
            self._seq += 1
            rid = f"rev_{self._seq:05d}"
            pr = PendingReview(review_id=rid, request=request,
                               created_at=time.time(), timeout_s=timeout_s)
            self._pending[rid] = pr
        return pr

    def wait_for_response(self, pr: PendingReview) -> dict:
        """Block until the web POSTs a decision or the timeout elapses.

        Returns the response dict either way (on timeout, ``resolved_by`` is set
        to ``"timeout"`` and ``response`` is None so the caller applies its own
        default). Idempotent: moves the review out of ``_pending`` when done.
        """
        got = pr.event.wait(timeout=pr.timeout_s if pr.timeout_s > 0 else None)
        with self._lock:
            if not got and pr.resolved_by == "":
                pr.resolved_by = "timeout"
                pr.resolved_at = time.time()
            self._pending.pop(pr.review_id, None)
            if pr not in self._history:
                self._history.append(pr)
        return pr.response or {}

    # -- server side ------------------------------------------------------
    def submit(self, review_id: str, response: dict) -> bool:
        """Record a human decision and wake the waiting orchestrator. False if unknown."""
        with self._lock:
            pr = self._pending.get(review_id)
            if pr is None or pr.event.is_set():
                return False
            pr.response = {
                "decision": str(response.get("decision", "noop")),
                "dense_text": str(response.get("dense_text", "") or ""),
                "replacement_src": response.get("replacement_src"),
            }
            pr.resolved_by = "human"
            pr.resolved_at = time.time()
            pr.event.set()
        return True

    def pending(self) -> list[dict]:
        # Only reviews still awaiting a human decision. A resolved-but-not-yet-
        # reaped review (e.g. the _demo path with no waiter to pop it) is excluded.
        with self._lock:
            return [pr.public() for pr in self._pending.values()
                    if not pr.event.is_set()]

    def history(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return [pr.public() for pr in self._history[-limit:]]

    def get(self, review_id: str) -> Optional[dict]:
        with self._lock:
            pr = self._pending.get(review_id)
            if pr is None:
                for h in self._history:
                    if h.review_id == review_id:
                        pr = h
                        break
            return pr.public() if pr else None
