"""Core data types shared across the Co-Scientist MVP.

These mirror the abstractions found in the two reference systems so that the
toy implementations here can be swapped for the real ones later:

- ``Candidate``  ~ ExplorationHarness ``Node`` / autosciworld ``Candidate``
- ``EvalResult`` ~ ASW ``run_verification`` return ``{feasible, score}``
                   and EH evaluator ``{combined_score, validity, ...}``
- ``Note``       ~ EH ``Node.reflection`` (SimpleTES note-share)
- ``Feedback``   ~ the Human-Proxy response: dense eval guidance + an
                   optional verifier patch (the human's Evaluation Guidance)

Nothing here depends on numpy, LLMs, or docker; it is the vocabulary the
rest of the package speaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# A candidate solution. In the real Solver this is source code delimited by
# EVOLVE-BLOCK markers; in the MVP toy task it is a structured fit spec. We keep
# it as an opaque ``payload`` plus a human-readable ``summary`` so the Solver,
# evaluators, and Human Proxy all agree on the same object without caring what
# domain it comes from.
@dataclass
class Candidate:
    payload: Any                      # domain-specific solution object
    summary: str = ""                 # short natural-language description
    parent_id: Optional[int] = None   # DAG lineage (EH ``parent_ids``)
    origin: str = "solver"            # "seed" | "solver" | "human_repair"

    # filled in by the environment after evaluation
    id: Optional[int] = None


@dataclass
class EvalResult:
    """Uniform evaluator return, matching ASW's ``{feasible, score}`` contract.

    ``score`` is the *proxy* score the Solver optimizes (higher is better).
    ``feasible`` gates invalid submissions (correctness is a gate, not a score,
    exactly as in EH's ``eval_contract.md``). ``artifacts`` carries whatever
    evidence the Human Proxy may later inspect (traces, per-instance residuals,
    declared vs. effective parameter counts, ...).
    """

    feasible: bool
    score: float
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class Note:
    """A shared-memory note (EH reflection / SimpleTES note-share).

    Notes are the Solver's own memory. Human feedback is deliberately a
    *different* channel (``Feedback``) so we can meter and study it separately.
    """

    text: str
    candidate_id: Optional[int] = None
    kind: str = "reflection"          # "reflection" | "failure" | "human_hint"


@dataclass
class Feedback:
    """Human-Proxy response to a Solver query ``(candidate x_t, question q_t)``.

    This is the crux of the design doc's "how to scale human effort" question.
    The Human Proxy holds the hidden reference evaluator V* and answers with
    *dense* evaluation guidance rather than a bare yes/no, and — when it detects
    that the proxy evaluator has been gamed — it emits a ``verifier_patch`` that
    hardens the proxy environment (Evaluation Guidance that actually changes V).
    """

    dense_text: str                        # the dense, solution-agnostic feedback
    hack_detected: bool = False            # did V* disagree with V for this x_t?
    locus: str = "none"                    # "verifier" | "environment" | "none"
    verifier_patch: Optional[dict] = None  # concrete hardening applied to V
    # bookkeeping
    real_score: Optional[float] = None     # V*(x_t), for logging only (never leaked to solver text)


@dataclass
class Step:
    """One evaluated candidate on the run timeline (one 'tick' of the clock)."""

    tick: int
    candidate_id: int
    proxy_score: float          # V(x)   — what the solver sees
    real_score: float           # V*(x)  — hidden ground truth, for metrics only
    feasible: bool
    is_hack: bool               # proxy >> real by more than the hack margin
    human_interventions: int    # cumulative human-proxy interventions so far
    note: str = ""
