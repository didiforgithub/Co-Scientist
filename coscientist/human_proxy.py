"""The Human Proxy: a scalable stand-in for a human domain expert.

This is the design doc's answer to *"how do we Scale Human Effort experimentally?"*
A real expert can give dense Evaluation Guidance but is expensive and cannot be
run across hundreds of tasks and repeated trials. So we replace them with an
agent that holds a **hidden Reference Evaluator V\*** and answers the Solver's
queries from V*'s real results.

Design decisions taken straight from the doc's margin notes:

  * The intervention is restricted to the **Evaluation** channel. The Human
    Proxy never proposes solutions — it only judges the environment/evaluator.
  * It is **not** a bare yes/no. Yes/No is too sparse and, per the doc, inviting
    the Solver to hack the yes/no signal. Instead it returns *dense* feedback:
    what V* actually measured, and *why* the proxy score was or wasn't earned.
  * But dense context alone is dangerous (the doc's other note: "只给 Context
    会让 system 去跟你做 Hack"). So when the Proxy detects that the gap between
    V and V* was produced by gaming, it does not merely describe it — it **emits
    a concrete verifier patch** that hardens V. The feedback that changes the
    world is a *patch*, not a paragraph the Solver could learn to talk around.

The Proxy is *metered*: every call is a unit of human effort. The whole
experiment measures outcomes against the number of these calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .environment import ProxyEnvironment
from .types import Candidate, Feedback


# A hack-diagnosis strategy inspects (proxy result, reference result, artifacts)
# and, if it finds the proxy was gamed, returns a (locus, patch, explanation).
# Different tasks plug in different diagnosers; the toy task ships a DOF-overfit
# diagnoser matching ASW's TTV case.
HackDiagnoser = Callable[[dict, dict, dict], Optional[tuple[str, dict, str]]]


@dataclass
class HumanProxyConfig:
    hack_margin: float = 0.15     # |V - V*| above this (with V>V*) == suspicious
    budget: Optional[int] = None  # max interventions the "human" can afford (None = unlimited)


class HumanProxy:
    """Holds V*, answers ``(x_t, q_t)`` with dense Evaluation Guidance + patches."""

    def __init__(
        self,
        env: ProxyEnvironment,
        diagnoser: HackDiagnoser,
        config: Optional[HumanProxyConfig] = None,
    ) -> None:
        self.env = env
        self.diagnoser = diagnoser
        self.config = config or HumanProxyConfig()
        self.calls = 0            # total interventions == the human-effort meter
        self.patches_emitted = 0

    @property
    def exhausted(self) -> bool:
        return self.config.budget is not None and self.calls >= self.config.budget

    def query(self, x: Candidate, question: str = "") -> Feedback:
        """Answer one Solver query. Consumes one unit of human effort.

        The Solver submits a candidate ``x`` and a question ``q`` about the
        current solution or evaluator. The Proxy consults V* (which the Solver
        cannot) and returns dense feedback; if V and V* disagree in the
        gaming direction, it also hardens V in place.
        """
        if self.exhausted:
            return Feedback(dense_text="[human budget exhausted]", hack_detected=False)

        self.calls += 1

        proxy = self.env.proxy_score(x)
        ref = self.env.reference_score(x)
        gap = proxy.score - ref.score

        # The diagnoser inspects V vs V* evidence for a *structural* evaluator
        # flaw (e.g. reduced-chi2 below the noise floor achieved with uncounted
        # DOF). A Human Proxy holding V* can recognize the hole from this
        # signature BEFORE real performance has visibly collapsed — that early
        # catch is exactly what buys back exploration horizon. We therefore act
        # on the diagnosis itself; ``gap`` and ``hack_margin`` only inform the
        # wording, they do not gate the patch.
        diagnosis = self.diagnoser(proxy.artifacts, ref.artifacts, {"gap": gap, "question": question})

        if diagnosis is not None:
            locus, patch, explanation = diagnosis
            self.env.apply_patch(patch)
            self.patches_emitted += 1
            return Feedback(
                dense_text=(
                    f"Your candidate reports proxy score {proxy.score:.3f} but the reference "
                    f"evaluator measures real performance {ref.score:.3f}. {explanation} "
                    f"The evaluator has been hardened ({locus}) so this route no longer scores."
                ),
                hack_detected=True,
                locus=locus,
                verifier_patch=patch,
                real_score=ref.score,
            )

        # No gaming detected: still give *dense* guidance, grounded in V*, that
        # points toward real improvement without handing over the answer.
        direction = "closer to" if gap <= 0 else "still short of"
        return Feedback(
            dense_text=(
                f"This looks like an honest attempt: proxy {proxy.score:.3f} is {direction} the "
                f"reference {ref.score:.3f}. No evaluator flaw found. To improve real performance, "
                f"reduce the true residual rather than adding representational capacity."
            ),
            hack_detected=False,
            locus="none",
            verifier_patch=None,
            real_score=ref.score,
        )
