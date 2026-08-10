"""The Solver: a stripped-down evolutionary search loop.

This is layer-1 of ExplorationHarness distilled to its essentials: a single
chain of best-of-K generation, scored by the (proxy) evaluator, with the winner
committed and a note/reflection kept for the next round. What EH does with an
LLM/agent Generator + RPUCG selection, we do with a pluggable ``propose``
function and top-N selection — the loop shape is identical:

    seed -> [propose K children -> proxy-evaluate -> select best -> note] * L

The one addition that makes this a *Co-Scientist* rather than a plain solver:
the Solver may spend a query on the Human Proxy. It does so exactly when it is
locally stuck on the proxy signal (its recent best proxy score has plateaued) —
the natural moment a real system would ask "is this evaluator actually
measuring what I think, or am I gaming it?". The Human Proxy's dense feedback
(and any verifier patch it triggers) then reshapes subsequent search.

Importantly the Solver is *rational about the proxy*: ``propose`` is free to
return exploit candidates, and the Solver keeps whatever scores highest under
the current V. This is what lets the environment be gamed — and what makes the
human-refinement loop necessary rather than decorative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .environment import ProxyEnvironment
from .human_proxy import HumanProxy
from .types import Candidate, Note, Step


# propose(parent, notes, feedback, rng_seed) -> list[Candidate]
# The proposer sees the committed parent, the Solver's own notes, and the most
# recent human Feedback text (guidance), and returns K children. It has NO
# access to V*. Toy tasks supply a domain-specific proposer.
Proposer = Callable[[Candidate, list[Note], Optional[str], int], list[Candidate]]


@dataclass
class SolverConfig:
    max_iters: int = 40             # L: evolution rounds
    k_candidates: int = 4           # K: children per round
    plateau_patience: int = 3       # rounds of no proxy gain before asking the human
    ask_human: bool = True          # whether the human-in-the-loop channel is on
    plateau_eps: float = 1e-3
    review_every: Optional[int] = None  # audit the env every N rounds (cadence).
    #   None -> only ask on plateau. A smaller N == a more attentive human who
    #   audits the evaluator more often, catching hacks earlier. This is the main
    #   "human effort" dial in the experiments: more audits -> later collapse.


@dataclass
class SolverState:
    steps: list[Step] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    best: Optional[Candidate] = None
    best_proxy: float = -float("inf")
    _next_id: int = 0
    _tick: int = 0
    _human_calls: int = 0


class Solver:
    def __init__(
        self,
        env: ProxyEnvironment,
        proposer: Proposer,
        human: Optional[HumanProxy] = None,
        config: Optional[SolverConfig] = None,
    ) -> None:
        self.env = env
        self.proposer = proposer
        self.human = human
        self.config = config or SolverConfig()
        self.state = SolverState()

    def _register(self, x: Candidate) -> Candidate:
        x.id = self.state._next_id
        self.state._next_id += 1
        return x

    def _record_step(self, x: Candidate, proxy_score: float, feasible: bool, note: str = "") -> None:
        # The reference score is read here ONLY for metrics/logging. The Solver
        # never sees it and never optimizes it directly — it is the hidden
        # ground truth used to detect hacking and measure Real Performance.
        ref = self.env.reference_score(x)
        is_hack = feasible and (proxy_score - ref.score) > _HACK_MARGIN
        self.state._tick += 1
        self.state.steps.append(
            Step(
                tick=self.state._tick,
                candidate_id=x.id if x.id is not None else -1,
                proxy_score=proxy_score,
                real_score=ref.score,
                feasible=feasible,
                is_hack=is_hack,
                human_interventions=self.state._human_calls,
                note=note,
            )
        )

    def run(self) -> SolverState:
        st = self.state
        cfg = self.config

        # seed
        seed = self._register(Candidate(payload=self.env.task.seed_payload, summary="seed x0", origin="seed"))
        seed_eval = self.env.proxy_score(seed)
        st.best, st.best_proxy = seed, seed_eval.score
        self._record_step(seed, seed_eval.score, seed_eval.feasible, note="seed")

        plateau = 0
        last_feedback: Optional[str] = None

        for round_idx in range(cfg.max_iters):
            children = self.proposer(st.best, st.notes, last_feedback, st._tick)
            round_best, round_best_eval = None, None

            for child in children[: cfg.k_candidates]:
                child = self._register(child)
                child.parent_id = st.best.id
                ev = self.env.proxy_score(child)
                self._record_step(child, ev.score, ev.feasible)
                if ev.feasible and (round_best_eval is None or ev.score > round_best_eval.score):
                    round_best, round_best_eval = child, ev

            # commit best-of-K if it beats the current proxy best
            improved = False
            if round_best is not None and round_best_eval.score > st.best_proxy + cfg.plateau_eps:
                st.best, st.best_proxy = round_best, round_best_eval.score
                st.notes.append(
                    Note(text=f"proxy improved to {st.best_proxy:.3f}: {round_best.summary}",
                         candidate_id=round_best.id, kind="reflection")
                )
                improved = True

            plateau = 0 if improved else plateau + 1

            # Decide whether to spend a human intervention this round. Two
            # triggers: (a) a scheduled audit cadence (``review_every``) — an
            # attentive human periodically checks the evaluator regardless of
            # solver state; (b) the solver is locally stuck on the proxy signal.
            due_for_audit = (
                cfg.review_every is not None
                and cfg.review_every > 0
                and (round_idx + 1) % cfg.review_every == 0
            )
            stuck = plateau >= cfg.plateau_patience
            if (
                cfg.ask_human
                and self.human is not None
                and not self.human.exhausted
                and (due_for_audit or stuck)
            ):
                fb = self.human.query(
                    st.best,
                    question="Audit: is the evaluator sound for my current best, or is it being gamed?",
                )
                st._human_calls = self.human.calls
                last_feedback = fb.dense_text
                st.notes.append(Note(text=f"[human] {fb.dense_text}", kind="human_hint"))
                plateau = 0

                # If the human hardened V, the committed best must be re-scored
                # under the new evaluator — a gamed 'best' can lose its score,
                # which is exactly how the environment reclaims the search.
                if fb.hack_detected:
                    re_eval = self.env.proxy_score(st.best)
                    st.best_proxy = re_eval.score

        return st


# Kept module-level so metrics and solver agree on what "a hack" means: the
# proxy overstates real performance by more than this margin.
_HACK_MARGIN = 0.15
