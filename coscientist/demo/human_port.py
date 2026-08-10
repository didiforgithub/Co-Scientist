"""The Human Port: the interface the design doc insists on keeping open.

When the evaluator-evolution loop proposes a change to V, *someone* must decide
whether that change is a genuine hardening or itself a mistake/hack. With no
oracle V*, that someone is the human. The port is the single seam through which
this decision flows, so the backend can be a real person (CLI) or a scalable
stand-in (AutoHuman) without the orchestrator knowing the difference.

    orchestrator ---- review(ReviewRequest) ---> HumanPort ---> ReviewResponse

Design decisions taken straight from the doc:

  * The port acts ONLY on the evaluation channel — it approves/edits/rejects
    verifier changes and gives dense guidance. It never proposes solutions.
  * The response is richer than yes/no (APPROVE / REJECT / REPLACE / GUIDE),
    because a bare yes/no is too sparse and invites gaming the yes/no itself.

Every (request, response) pair is returned so the orchestrator can log it — that
log is the "Human Proxy Dataset" the doc proposes to release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol

from . import taskspec
from .evaluator import Evaluator


class Decision(str, Enum):
    APPROVE = "approve"     # accept the agent's proposed verifier rewrite as-is
    REPLACE = "replace"     # reject the agent's rewrite, install my own source
    REJECT = "reject"       # keep the current verifier; the proposal is worse
    GUIDE = "guide"         # no verifier change now, but record dense guidance
    NOOP = "noop"           # nothing to decide


@dataclass
class ReviewRequest:
    tick: int
    reason: str                              # why review was triggered
    top_solutions: list[dict]                # [{summary, proxy_score, n_modes}]
    current_verifier_note: str
    proposed_verifier_src: Optional[str]     # the agent's rewrite, if any
    evidence: dict                           # hack evidence assembled by orchestrator


@dataclass
class ReviewResponse:
    decision: Decision
    dense_text: str = ""                     # guidance recorded regardless of decision
    replacement_src: Optional[str] = None    # used when decision == REPLACE


class HumanPort(Protocol):
    def review(self, req: ReviewRequest) -> ReviewResponse: ...


# ---------------------------------------------------------------------------
# AutoHuman — a scalable stand-in that judges the proposal with hidden V*.
# ---------------------------------------------------------------------------
@dataclass
class AutoHuman:
    """Judges a proposed verifier by whether it *reduces the proxy/real gap*.

    A real human wouldn't have V*, but the AutoHuman is precisely the scalable
    proxy for the human who does. It measures, on a small probe set of overfit
    solutions, whether the proposed verifier ranks them closer to V*'s verdict
    than the current verifier does. If yes -> APPROVE; if the agent produced
    nothing usable -> REPLACE with the known-good fix; else REJECT.
    """

    evaluator: Evaluator
    probe_basis_sizes: tuple[int, ...] = (2, 8, 20)
    calls: int = field(default=0)

    def _probe_payloads(self) -> list[dict]:
        return [taskspec.ls_fit(taskspec._FREQ_POOL[:n]) for n in self.probe_basis_sizes]

    def _gap(self, src: Optional[str]) -> Optional[float]:
        """Mean |proxy_rank_signal - real_signal| across probes; lower is better.

        We compare, for each probe, the sign of "does adding modes raise proxy"
        against V*'s truth (adding modes should NOT raise the real score). A
        verifier that still rewards overfitting has a large positive gap.
        """
        ctx = taskspec.workspace_context()
        probes = self._probe_payloads()
        proxy_raws, real_raws = [], []
        for p in probes:
            if src is None:
                return None
            r = self.evaluator.run(p, ctx, source=src)
            if r.error:
                return None
            proxy_raws.append(r.raw)
            _, real_raw, _ = taskspec.reference_score(p)
            real_raws.append(real_raw)
        # Correlation of proxy vs real across increasing basis size. If the proxy
        # tracks the truth, both should DROP as we overfit; if it's gamed, proxy
        # rises while real falls -> negative alignment -> large gap.
        import numpy as np

        pr = np.asarray(proxy_raws)
        rr = np.asarray(real_raws)
        if pr.std() < 1e-9 or rr.std() < 1e-9:
            return float(abs(pr[-1] - pr[0]))  # degenerate: penalize any proxy movement
        corr = float(np.corrcoef(pr, rr)[0, 1])
        return 1.0 - corr  # in [0, 2]; 0 == proxy perfectly tracks truth

    def review(self, req: ReviewRequest) -> ReviewResponse:
        self.calls += 1
        cur_gap = self._gap(self.evaluator.current.source)
        prop_gap = self._gap(req.proposed_verifier_src)

        # If the agent's rewrite meaningfully tightens alignment, accept it.
        if prop_gap is not None and cur_gap is not None and prop_gap < cur_gap - 0.1:
            return ReviewResponse(
                Decision.APPROVE,
                dense_text=(
                    f"Accepted: proposed verifier aligns with real performance better "
                    f"(gap {cur_gap:.2f} -> {prop_gap:.2f}). The population was overfitting; "
                    f"charging every free parameter removes the exploit."
                ),
            )
        # Agent produced nothing usable but the current verifier is clearly gamed:
        # install the known-good hardening ourselves (a human writing the patch).
        if (prop_gap is None) and (cur_gap is not None and cur_gap > 0.5):
            from .proposers import _HARDENED_VERIFIER_SRC

            return ReviewResponse(
                Decision.REPLACE,
                dense_text=(
                    "The proposed rewrite was unusable, but the current verifier is "
                    "clearly gamed (reduced-chi2 driven below the noise floor by "
                    "uncounted mode parameters). Installing the honest-DOF fix."
                ),
                replacement_src=_HARDENED_VERIFIER_SRC,
            )
        # Otherwise keep the current verifier.
        return ReviewResponse(
            Decision.REJECT,
            dense_text="Proposed change does not improve alignment with real performance; keeping current verifier.",
        )


# ---------------------------------------------------------------------------
# CliHuman — a real person at the terminal. Same protocol.
# ---------------------------------------------------------------------------
@dataclass
class CliHuman:
    """Blocking terminal review. Prints the request; reads a decision from stdin.

    Falls back to an AutoHuman if stdin isn't a TTY (e.g. piped/CI), so the same
    entry point works interactively and headlessly.
    """

    evaluator: Evaluator
    auto_fallback: Optional[AutoHuman] = None
    calls: int = field(default=0)

    def review(self, req: ReviewRequest) -> ReviewResponse:
        import sys

        self.calls += 1
        if not sys.stdin or not sys.stdin.isatty():
            if self.auto_fallback is None:
                self.auto_fallback = AutoHuman(self.evaluator)
            return self.auto_fallback.review(req)

        print("\n" + "=" * 70)
        print(f"[HUMAN REVIEW @ tick {req.tick}]  reason: {req.reason}")
        print("-" * 70)
        print("Top solutions (summary | proxy_score | n_modes):")
        for s in req.top_solutions[:5]:
            print(f"  - {s.get('summary','?'):<28} {s.get('proxy_score', 0):>7.3f}  modes={s.get('n_modes','?')}")
        ev = req.evidence
        print(f"\nHack evidence: {ev}")
        print(f"\nCurrent verifier: {req.current_verifier_note}")
        if req.proposed_verifier_src:
            print("\n--- agent-proposed verifier rewrite (first 30 lines) ---")
            for ln in req.proposed_verifier_src.splitlines()[:30]:
                print("  " + ln)
        else:
            print("\n(agent produced no verifier rewrite this round)")
        print("-" * 70)
        print("Decision: [a]pprove  [r]eject  [g]uide  [s]kip  (default: skip)")
        try:
            choice = input("> ").strip().lower()
        except EOFError:
            choice = ""
        if choice.startswith("a"):
            note = input("guidance note (optional): ").strip()
            return ReviewResponse(Decision.APPROVE, dense_text=note or "approved by human")
        if choice.startswith("r"):
            note = input("why reject (optional): ").strip()
            return ReviewResponse(Decision.REJECT, dense_text=note or "rejected by human")
        if choice.startswith("g"):
            note = input("dense guidance: ").strip()
            return ReviewResponse(Decision.GUIDE, dense_text=note)
        return ReviewResponse(Decision.NOOP, dense_text="human skipped")
