"""``WebHumanPort`` — the web equivalent of ``demo.human_port.CliHuman``.

Implements the ``HumanPort`` Protocol (``review(req) -> ReviewResponse``). Instead
of prompting at a terminal, it enqueues the request into a shared, thread-safe
``ReviewRegistry`` and BLOCKS until the web side POSTs a decision — then returns
the matching ``ReviewResponse``. A configurable timeout guarantees a headless run
never hangs forever: with no human in ``timeout_s`` seconds it returns the
configured default decision (``GUIDE`` with empty text by default, which the
orchestrator records as dense guidance without moving the verifier).

This is the seam a future ``--supervisor web`` mode would wire into
``build()`` / ``run_agent_system`` (see coscientist/ui/README.md). It is NOT
wired into the live orchestrator here (that would touch existing source); it is
constructed standalone and driven against the server's control-plane endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..demo.human_port import Decision, ReviewRequest, ReviewResponse
from .review_registry import ReviewRegistry


def serialize_request(req: ReviewRequest) -> dict:
    """Turn a ReviewRequest into a JSON-able dict for the web client.

    Includes the full ``proposed_verifier_src`` so the UI can diff it against the
    current verifier; the rest mirrors the dataclass fields verbatim.
    """
    return {
        "tick": req.tick,
        "reason": req.reason,
        "top_solutions": list(req.top_solutions or []),
        "current_verifier_note": req.current_verifier_note,
        "proposed_verifier_src": req.proposed_verifier_src,
        "evidence": dict(req.evidence or {}),
    }


@dataclass
class WebHumanPort:
    """A blocking HumanPort backed by a web control plane.

    Parameters
    ----------
    registry:
        The shared ``ReviewRegistry`` the HTTP server also holds. The server's
        ``GET /api/reviews/pending`` reads it and ``POST /api/reviews/<id>``
        writes into it, unblocking ``review()``.
    timeout_s:
        Seconds to wait for a human before falling back. ``0`` or negative means
        wait forever (use only when a human is guaranteed present).
    default_decision:
        The ``Decision`` returned on timeout. Defaults to ``GUIDE`` (record
        empty guidance, keep the current verifier) so a stalled human is a no-op
        on the evaluation rather than a spurious approve/reject.
    default_text:
        The ``dense_text`` returned on timeout.
    """

    registry: ReviewRegistry
    timeout_s: float = 900.0
    default_decision: Decision = Decision.GUIDE
    default_text: str = "(no human responded within the review window)"
    calls: int = field(default=0)

    def review(self, req: ReviewRequest) -> ReviewResponse:
        self.calls += 1
        pr = self.registry.enqueue(serialize_request(req), timeout_s=self.timeout_s)
        resp = self.registry.wait_for_response(pr)
        if not resp:  # timed out — apply the configured default
            return ReviewResponse(self.default_decision, dense_text=self.default_text)
        return _response_from_dict(resp)


def _response_from_dict(resp: dict) -> ReviewResponse:
    """Map a POSTed ``{decision, dense_text, replacement_src}`` to a ReviewResponse.

    Unknown/garbage decisions degrade to ``NOOP`` rather than raising, so a
    malformed client can never crash the orchestrator thread.
    """
    raw = str(resp.get("decision", "noop")).strip().lower()
    try:
        decision = Decision(raw)
    except ValueError:
        decision = Decision.NOOP
    replacement: Optional[str] = resp.get("replacement_src")
    if replacement is not None:
        replacement = str(replacement)
    # A REPLACE with no source is meaningless — treat it as guidance instead.
    if decision == Decision.REPLACE and not replacement:
        decision = Decision.GUIDE
    return ReviewResponse(
        decision,
        dense_text=str(resp.get("dense_text", "") or ""),
        replacement_src=replacement,
    )
