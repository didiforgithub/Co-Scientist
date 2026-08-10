"""The orchestrator: interleave the two evolution loops around a shared
blackboard, routing every proposed evaluator change through the Human Port.

Blackboard (shared state):
    archive     — evaluated solutions (payload + proxy score + lineage)
    evaluator   — the versioned, evolvable verifier V
    eventlog    — append-only action stream

Control flow, per outer round:

    LOOP A (fast, solutions evolve):
        for a few inner steps:
            child = SolutionProposer.propose(best, evaluator)   # agent edits solution
            score child under CURRENT verifier; commit if best-of-K
            record the step + hidden V* (metrics only)

    Trigger check: is the population gaming V?  (proxy rising while the basis
        grows / proxy far above what V* would say)

    LOOP B (slow, evaluator evolves) — only when triggered:
        proposed_src = EvaluatorProposer.propose(evaluator, evidence)  # agent rewrites verifier
        response     = human_port.review(request(proposed_src, evidence))  # HUMAN ARBITRATES
        if APPROVE  -> install agent's rewrite as a new verifier version
        if REPLACE  -> install the human's own source
        if REJECT/GUIDE/NOOP -> keep current verifier (guidance still logged)
        if the verifier changed: RE-SCORE the archive under the new V

The human port is the only place an evaluator change can be ratified. That is
the doc's thesis made executable: the environment is refined under human
guidance, and the refinement is what keeps exploration effective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import taskspec
from .evaluator import Evaluator
from .eventlog import EventLog
from .human_port import Decision, HumanPort, ReviewRequest
from .proposers import EvaluatorProposer, SolutionProposer


@dataclass
class ArchiveEntry:
    id: int
    payload: dict
    proxy_score: float          # V(x) under the verifier version current at commit
    real_score: float           # V*(x), hidden, metrics only
    n_modes: int
    parent_id: Optional[int]
    summary: str = ""


@dataclass
class OrchestratorConfig:
    outer_rounds: int = 8
    inner_steps: int = 3
    k_candidates: int = 3
    trigger_hack_gap: float = 0.4      # commit proxy - real gap that flags gaming
    review_every: Optional[int] = 2    # force a review cadence (human effort dial)
    seed: int = 0


@dataclass
class Orchestrator:
    evaluator: Evaluator
    solution_proposer: SolutionProposer
    evaluator_proposer: EvaluatorProposer
    human_port: HumanPort
    config: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    log: EventLog = field(default_factory=EventLog)

    archive: list[ArchiveEntry] = field(default_factory=list)
    _tick: int = 0
    _next_id: int = 0
    reviews: int = 0
    verifier_changes: int = 0

    # -- scoring helpers --------------------------------------------------
    def _score(self, payload: dict) -> tuple[bool, float, dict]:
        ctx = taskspec.workspace_context()
        r = self.evaluator.run(payload, ctx)
        return (r.feasible and r.error is None), r.raw, r.artifacts

    def _real(self, payload: dict) -> float:
        _, raw, _ = taskspec.reference_score(payload)
        return raw

    def _commit(self, payload: dict, parent_id: Optional[int], summary: str) -> Optional[ArchiveEntry]:
        feasible, proxy_raw, art = self._score(payload)
        if not feasible:
            self.log.emit("infeasible", tick=self._tick, summary=summary, art=art)
            return None
        entry = ArchiveEntry(
            id=self._next_id,
            payload=payload,
            proxy_score=proxy_raw,
            real_score=self._real(payload),
            n_modes=len(payload.get("modes", [])),
            parent_id=parent_id,
            summary=summary,
        )
        self._next_id += 1
        self.archive.append(entry)
        return entry

    @property
    def best(self) -> ArchiveEntry:
        return max(self.archive, key=lambda e: e.proxy_score)

    # -- LOOP A -----------------------------------------------------------
    def _evolve_solution(self) -> None:
        cfg = self.config
        parent = self.best
        best_child: Optional[ArchiveEntry] = None
        for k in range(cfg.k_candidates):
            self._tick += 1
            payload = self.solution_proposer.propose(parent.payload, self.evaluator, seed=self._tick + cfg.seed)
            if payload is None:
                self.log.emit("solution_session_failed", tick=self._tick)
                continue
            entry = self._commit(payload, parent_id=parent.id, summary=f"child(modes={len(payload.get('modes', []))})")
            if entry is None:
                continue
            self.log.emit(
                "solution_committed", tick=self._tick, id=entry.id,
                proxy=entry.proxy_score, real=entry.real_score, n_modes=entry.n_modes,
            )
            if best_child is None or entry.proxy_score > best_child.proxy_score:
                best_child = entry

    # -- trigger ----------------------------------------------------------
    def _hack_evidence(self) -> Optional[dict]:
        """Assemble evidence that the population is gaming V (proxy >> real)."""
        if not self.archive:
            return None
        top = self.best
        gap = top.proxy_score - top.real_score
        # A verifier is being gamed when its best-scoring solution carries many
        # modes yet its real performance is poor (proxy high, real low).
        if gap >= self.config.trigger_hack_gap and top.n_modes >= 4:
            return {
                "top_proxy_score": round(top.proxy_score, 4),
                "top_real_score": round(top.real_score, 4),
                "proxy_minus_real": round(gap, 4),
                "top_n_modes": top.n_modes,
                "verifier_version": self.evaluator.current.version,
                "hint": "proxy score rises with mode count while real performance does not — DOF undercounting.",
            }
        return None

    # -- LOOP B (gated by the human port) --------------------------------
    def _evolve_evaluator(self, reason: str, evidence: dict) -> None:
        self.log.emit("eval_evolution_triggered", tick=self._tick, reason=reason, evidence=evidence)
        proposed_src = self.evaluator_proposer.propose(self.evaluator, evidence, seed=self._tick + self.config.seed)

        # validate the agent's rewrite before it ever reaches review
        if proposed_src is not None:
            err = self.evaluator.validate_source(
                proposed_src, taskspec.ls_fit(taskspec._FREQ_POOL[:2]), taskspec.workspace_context()
            )
            if err:
                self.log.emit("proposed_verifier_invalid", tick=self._tick, error=err)
                proposed_src = None

        top = self.best
        req = ReviewRequest(
            tick=self._tick,
            reason=reason,
            top_solutions=[
                {"summary": e.summary, "proxy_score": e.proxy_score, "n_modes": e.n_modes}
                for e in sorted(self.archive, key=lambda e: e.proxy_score, reverse=True)[:5]
            ],
            current_verifier_note=f"v{self.evaluator.current.version} ({self.evaluator.current.note})",
            proposed_verifier_src=proposed_src,
            evidence=evidence,
        )
        self.reviews += 1
        resp = self.human_port.review(req)
        self.log.emit("human_review", tick=self._tick, decision=resp.decision.value, dense_text=resp.dense_text)

        new_src = None
        origin = ""
        if resp.decision == Decision.APPROVE and proposed_src is not None:
            new_src, origin = proposed_src, "agent"
        elif resp.decision == Decision.REPLACE and resp.replacement_src is not None:
            new_src, origin = resp.replacement_src, "human"

        if new_src is not None:
            ver = self.evaluator.evolve(new_src, origin=origin, note=f"hardened via {origin} @ tick {self._tick}")
            self.verifier_changes += 1
            self.log.emit("verifier_evolved", tick=self._tick, version=ver.version, origin=origin)
            self._rescore_archive()

    def _rescore_archive(self) -> None:
        """Re-evaluate committed solutions under the new verifier version."""
        for e in self.archive:
            feasible, proxy_raw, _ = self._score(e.payload)
            e.proxy_score = proxy_raw if feasible else float("-inf")
        self.log.emit("archive_rescored", tick=self._tick, size=len(self.archive), best_proxy=self.best.proxy_score)

    # -- main -------------------------------------------------------------
    def run(self) -> "Orchestrator":
        cfg = self.config
        # seed
        self._tick += 1
        seed_entry = self._commit({"trend": (0.0, 0.0), "modes": []}, None, "seed")
        self.log.emit("seed", tick=self._tick, id=seed_entry.id if seed_entry else None)

        for rnd in range(cfg.outer_rounds):
            for _ in range(cfg.inner_steps):
                self._evolve_solution()

            due = cfg.review_every is not None and (rnd + 1) % cfg.review_every == 0
            evidence = self._hack_evidence()
            if evidence is not None and (due or evidence["proxy_minus_real"] > 2 * cfg.trigger_hack_gap):
                self._evolve_evaluator(
                    reason=("scheduled audit" if due else "hack gap exceeded"), evidence=evidence
                )
            self.log.emit(
                "round_end", round=rnd, tick=self._tick,
                best_proxy=self.best.proxy_score, best_real=self.best.real_score,
                verifier_version=self.evaluator.current.version,
            )
        return self
