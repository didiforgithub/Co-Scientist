"""The Supervisor (§2) — the Eval Proposer, a self-contained red-teamer.

The Supervisor owns V. It can ALWAYS rewrite V — there is **no approval gate**.
Collapse is prevented structurally: the Supervisor is the defender, and the
direction it edits V is "harder to game," inherently adversarial to the Solver's
gaming. Its three modes are *advisor* autonomy levels, not gatekeepers:

    none/no-human-no-proxy — fully autonomous; decides from its own probes + observations.
    proxy   — a V*-holding proxy agent advises (reuses demo.human_port.AutoHuman).
    human   — a real person advises at the CLI (reuses demo.human_port.CliHuman).

Hack detection folds three sources, none of which requires V* (§2):

    1. probes it crafts itself     — build a solution it KNOWS should score low
                                     (an obvious overfit), submit it to the
                                     current V, see if V is fooled. This is the
                                     concrete "runs red-team in the eval
                                     container": it actually executes attacks.
    2. the Solver's review messages — reasons about a submitted best.
    3. the submission trajectory    — score climbing while method drifts overfit.

Source (1) alone lets the autonomous mode stand on its own. proxy/human layer more
accurate calibration on top.

Producing the hardened V is pluggable (``VerifierSmith``), per the project's
delegate-to-agent principle: offline it emits the known honest-DOF fix; a real
run can drive an agent to rewrite V. The Supervisor decides WHEN and installs the
result with no gate; the smith decides WHAT the rewrite is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Protocol

from ..demo import taskspec
from .channels import Nudge, NudgeKind, ReviewMessage, ReviewVerdict, SupervisorChannel
from .eval_service import EvalService, FeedbackLevel


class SupervisorMode(str, Enum):
    # The Supervisor ROLE (red-team + harden) is ALWAYS active and orchestrator-driven;
    # this enum only selects whether an EXTERNAL advisor sits beside it. The autonomous
    # value is named for what it actually is — no human, no proxy — because "none" wrongly
    # read as "no Supervisor at all". "none" stays accepted as a back-compat alias below.
    NO_HUMAN_NO_PROXY = "no-human-no-proxy"
    PROXY = "proxy"
    HUMAN = "human"

    @classmethod
    def _missing_(cls, value):
        # Back-compat: the legacy spelling "none" maps to the autonomous mode, so
        # in-flight runs launched with `--supervisor none` still resume cleanly.
        if isinstance(value, str) and value.strip().lower() == "none":
            return cls.NO_HUMAN_NO_PROXY
        return None


# ---------------------------------------------------------------------------
# Pluggable "how do I rewrite V harder?" — the smith
# ---------------------------------------------------------------------------
class VerifierSmith(Protocol):
    def harden(self, current_src: str, evidence: dict) -> Optional[str]:
        """Return a hardened verifier source, or None if it can't produce one."""


@dataclass
class KnownGoodSmith:
    """Offline default: emit the honest-DOF fix (debits every free parameter).

    This is the deterministic stand-in so the ``none`` mode runs fully offline. A
    real run swaps in an agent-backed smith; the Supervisor's decision logic is
    unchanged.
    """

    def harden(self, current_src: str, evidence: dict) -> Optional[str]:
        from ..demo.proposers import _HARDENED_VERIFIER_SRC
        return _HARDENED_VERIFIER_SRC


# ---------------------------------------------------------------------------
# The Supervisor
# ---------------------------------------------------------------------------
@dataclass
class Supervisor:
    """Owns V; red-teams it; rewrites it without a gate; advises the Solver.

    ``advisor`` is a ``demo.human_port.HumanPort`` when mode is proxy/human, else
    None. ``store`` is the boundary sink (probes/reviews/versions/events) — may be
    None in a bare unit test. ``probe_basis_sizes`` are the increasing overfit
    bases the red-team crafts.
    """

    eval: EvalService
    mode: SupervisorMode = SupervisorMode.NO_HUMAN_NO_PROXY
    smith: VerifierSmith = field(default_factory=KnownGoodSmith)
    advisor: Optional[object] = None          # demo.human_port.HumanPort or None
    store: Optional[object] = None            # coevo.store.RunStore or None

    probe_basis_sizes: tuple[int, ...] = (2, 8, 18)
    hack_score_rise_eps: float = 1e-3         # a probe "fools" V if a bigger basis scores higher

    # solver channels the Supervisor can push nudges to (registered by the driver)
    _channels: list = field(default_factory=list)
    hardened_count: int = 0
    reviews_handled: int = 0

    # -- driver wiring ----------------------------------------------------
    def register_solver_channel(self, ch: SupervisorChannel) -> None:
        self._channels.append(ch)

    def _broadcast(self, nudge: Nudge) -> None:
        for ch in self._channels:
            ch._push(nudge)

    # -- (1) red-team probes IN THE EVAL CONTAINER ------------------------
    def red_team(self) -> dict:
        """Craft increasing-overfit probes, submit to current V, detect a hole.

        A verifier is FOOLED when a deeper overfit (more modes) scores HIGHER — an
        overfit that fits the noise should not be rewarded. Needs no V*: it is the
        internal monotonicity of "more modes -> higher score" that betrays the
        hole. Returns evidence usable by ``maybe_harden``.
        """
        probes = [(n, taskspec.ls_fit(taskspec._FREQ_POOL[:n])) for n in self.probe_basis_sizes]
        scored = []
        for n, payload in probes:
            r = self.eval.query(payload, who="supervisor-probe")
            scored.append((n, payload, r.score))

        # Compare the smallest basis to the largest: if the deep overfit scores
        # meaningfully higher, V rewards overfitting -> fooled.
        small_n, small_p, small_s = scored[0]
        deep_n, deep_p, deep_s = scored[-1]
        fooled = (
            small_s is not None and deep_s is not None
            and deep_s > small_s + self.hack_score_rise_eps
        )

        if self.store is not None:
            self.store.probe(
                description=f"overfit-monotonicity: modes {small_n} vs {deep_n}",
                payload=deep_p, score=deep_s,
                expected_low=True, fooled=fooled,
            )
            self.store.event(
                "supervisor_red_team", fooled=fooled,
                small=(small_n, small_s), deep=(deep_n, deep_s),
                verifier_version=self.eval.current_version(),
            )

        return {
            "fooled": fooled,
            "small": {"n_modes": small_n, "score": small_s},
            "deep": {"n_modes": deep_n, "score": deep_s},
            "score_rise": (None if (small_s is None or deep_s is None) else round(deep_s - small_s, 4)),
            "verifier_version": self.eval.current_version(),
            "hint": "deeper overfit scores higher — V is not charging mode DOF.",
        }

    # -- (2) handle a Solver review request -------------------------------
    def handle_review(self, msg: ReviewMessage) -> ReviewVerdict:
        """Judge whether the Solver's current best is gaming V.

        Uses the same overfit signal: score the best under the current V and ask
        whether a same-shape deep overfit is what's winning. Advisor (proxy/human)
        calibrates the wording/decision when present, but the Supervisor never
        needs it to answer.
        """
        self.reviews_handled += 1
        n_modes = len(msg.best_payload.get("modes", []) or [])
        rt = self.red_team()
        gaming = bool(rt["fooled"]) and n_modes >= self.probe_basis_sizes[1]

        text = (
            f"Reviewed best (modes={n_modes}, score={msg.best_score}). "
            + ("Judged GAMING: the current verifier rewards deeper overfits "
               "(probe score rose with basis size). Hardening V." if gaming
               else "No clear gaming under the current verifier; keep exploring.")
        )
        verdict = ReviewVerdict(gaming=gaming, text=text)

        if self.store is not None:
            self.store.review(
                kind=msg.kind.value, n_modes=n_modes, best_score=msg.best_score,
                gaming=gaming, evidence=rt, text=text, mode=self.mode.value,
            )
        # A review that finds gaming immediately triggers a (gate-free) harden.
        if gaming:
            self.maybe_harden(rt, reason="solver_review")
        return verdict

    # -- (3) autonomous / advised hardening -------------------------------
    def observe_and_harden(self) -> bool:
        """Proactive path (§4): red-team on its own, harden if V is fooled."""
        rt = self.red_team()
        if rt["fooled"]:
            return self.maybe_harden(rt, reason="proactive_probe")
        return False

    def maybe_harden(self, evidence: dict, *, reason: str) -> bool:
        """Decide (advisor-calibrated) and install a harder V. No approval gate.

        Returns True if V was rewritten. The advisor, when present, can *refine*
        the decision, but in ``none`` mode the probe evidence alone drives it.
        """
        advised_ok = self._consult_advisor(evidence)
        if not advised_ok:
            if self.store is not None:
                self.store.event("supervisor_hold", reason=reason, mode=self.mode.value)
            return False

        current = self.eval.current_source()
        new_src = self.smith.harden(current, evidence)
        if new_src is None:
            if self.store is not None:
                self.store.event("supervisor_smith_empty", reason=reason)
            return False

        # smoke-test before install
        err = self.eval.validate(new_src, taskspec.ls_fit(taskspec._FREQ_POOL[:2]))
        if err:
            if self.store is not None:
                self.store.event("supervisor_rewrite_invalid", reason=reason, error=err)
            return False

        ver = self.eval.install_verifier(
            new_src, origin=("supervisor/" + self.mode.value),
            note=f"hardened ({reason}) @ v{self.eval.current_version()}->",
        )
        self.hardened_count += 1
        if self.store is not None:
            self.store.verifier_version(
                ver, new_src, origin="supervisor", note=reason,
                rationale=evidence.get("hint", ""),
            )
            self.store.event("supervisor_hardened", reason=reason, version=ver,
                             mode=self.mode.value)
        # tell every solver: re-baseline, V moved under you (§4).
        self._broadcast(Nudge(NudgeKind.VERIFIER_CHANGED,
                              text=f"verifier hardened ({reason})", verifier_version=ver))
        return True

    def _consult_advisor(self, evidence: dict) -> bool:
        """proxy/human calibration. In ``none`` mode, act on probes alone.

        Reuses the demo HumanPort: APPROVE/REPLACE -> proceed; REJECT/other ->
        hold. The advisor here does NOT gate structurally — it is an autonomy dial
        the operator chose; ``none`` skips it entirely.
        """
        if self.mode == SupervisorMode.NO_HUMAN_NO_PROXY or self.advisor is None:
            return True
        from ..demo.human_port import Decision, ReviewRequest

        req = ReviewRequest(
            tick=self.eval.current_version(),
            reason="supervisor proposes hardening V",
            top_solutions=[],
            current_verifier_note=f"v{self.eval.current_version()}",
            proposed_verifier_src=self.smith.harden(self.eval.current_source(), evidence),
            evidence=evidence,
        )
        resp = self.advisor.review(req)
        if self.store is not None:
            self.store.event("supervisor_advisor", mode=self.mode.value,
                             decision=resp.decision.value, dense_text=resp.dense_text)
        return resp.decision in (Decision.APPROVE, Decision.REPLACE)

    # -- feedback dial (§3) ----------------------------------------------
    def tighten_feedback(self) -> None:
        """Move the disclosure dial sparser — a strategy response to gaming (§3)."""
        order = [FeedbackLevel.WITH_ARTIFACTS, FeedbackLevel.FEASIBLE_SCORE,
                 FeedbackLevel.SCORE_ONLY]
        cur = self.eval.feedback_level
        idx = order.index(cur) if cur in order else 0
        if idx + 1 < len(order):
            self.eval.set_feedback_level(order[idx + 1])
            if self.store is not None:
                self.store.event("supervisor_tightened_feedback",
                                 level=self.eval.feedback_level.value)
            self._broadcast(Nudge(NudgeKind.FEEDBACK_CHANGED,
                                  text=f"feedback now {self.eval.feedback_level.value}"))
