"""Experiment driver: the design doc's flagship curve.

Runs the toy TTV-overfit task under increasing amounts of human effort and shows
the three headline relationships the doc predicts:

    as human interventions increase:
        Hack Rate            decreases
        Effective Horizon    increases  ->  approaches the Reference Horizon
        Real Performance     increases

We contrast three regimes on the SAME task/proxy/solver, changing only the human
channel:

  1. no-human      : the proxy is never refined. The rational solver games the
                     DOF hole; the evaluator collapses early (short EEH).
  2. budgeted-human: a Human Proxy with a small intervention budget hardens V
                     when it detects the overfit; EEH extends.
  3. reference     : an oracle solver driven directly by V* (no proxy to game).
                     Its horizon is the ceiling every method is trying to reach.

Run:  python -m coscientist.experiments.run_curve
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..human_proxy import HumanProxy, HumanProxyConfig
from ..metrics import compute_metrics, horizon_gap
from ..solver import Solver, SolverConfig
from ..tasks.ttv_overfit import (
    dof_overfit_diagnoser,
    make_env,
    make_proposer,
)


@dataclass
class RegimeResult:
    label: str
    human_budget: Optional[int]
    interventions: int
    hack_rate_final: float
    collapse_tick: Optional[int]
    effective_horizon: int
    real_performance: float
    proxy_performance: float


def _run_once(
    review_every: Optional[int],
    ask_human: bool,
    max_iters: int = 30,
    human_budget: Optional[int] = None,
) -> RegimeResult:
    env = make_env()
    proposer = make_proposer(k=4)
    human = None
    if ask_human:
        human = HumanProxy(
            env,
            diagnoser=dof_overfit_diagnoser,
            config=HumanProxyConfig(hack_margin=0.15, budget=human_budget),
        )
    solver = Solver(
        env,
        proposer,
        human=human,
        config=SolverConfig(
            max_iters=max_iters,
            k_candidates=4,
            plateau_patience=2,
            ask_human=ask_human,
            review_every=review_every,
        ),
    )
    st = solver.run()
    m = compute_metrics(st.steps, window=8, hack_threshold=0.5, eps=0.02)
    label = "no-human" if not ask_human else f"audit/{review_every}"
    return RegimeResult(
        label=label,
        human_budget=human_budget,
        interventions=human.calls if human else 0,
        hack_rate_final=m.hack_rate_final,
        collapse_tick=m.collapse_tick,
        effective_horizon=m.effective_horizon,
        real_performance=m.real_performance,
        proxy_performance=m.proxy_performance,
    )


def _reference_horizon(max_iters: int = 30) -> int:
    """Oracle solver driven directly by V*: the horizon ceiling.

    We simulate the doc's 'Reference' by fully crediting the mode degrees of
    freedom from the start (``dof_credit_fraction = 1.0`` == the environment is
    already perfect) and letting the same solver run with the human channel off.
    With no exploitable hole, every tick is useful, so the effective horizon
    spans the whole run.
    """
    env = make_env()
    env.apply_patch({"dof_credit_fraction": 1.0})   # perfect environment == reference
    solver = Solver(
        env,
        make_proposer(k=4),
        human=None,
        config=SolverConfig(max_iters=max_iters, k_candidates=4, ask_human=False),
    )
    st = solver.run()
    m = compute_metrics(st.steps, window=8, hack_threshold=0.5, eps=0.02)
    return m.effective_horizon


def main() -> None:
    max_iters = 40
    ref_h = _reference_horizon(max_iters)

    regimes = [
        _run_once(review_every=None, ask_human=False, max_iters=max_iters),  # no human
        _run_once(review_every=6, ask_human=True, max_iters=max_iters),      # rare audits
        _run_once(review_every=3, ask_human=True, max_iters=max_iters),      # moderate
        _run_once(review_every=1, ask_human=True, max_iters=max_iters),      # attentive
    ]

    print("=" * 84)
    print("Human-Guided Environment Refinement — TTV DOF-overfit toy task")
    print("=" * 84)
    print(f"Reference Horizon (oracle V*, perfect env): {ref_h} useful ticks\n")
    hdr = f"{'regime':<20}{'human':>7}{'hackrate':>10}{'collapse':>10}{'EEH':>6}{'horizon_gap':>13}{'realperf':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in regimes:
        gap = horizon_gap(r.effective_horizon, ref_h)
        collapse = "-" if r.collapse_tick is None else str(r.collapse_tick)
        print(
            f"{r.label:<20}{r.interventions:>7}{r.hack_rate_final:>10.2f}{collapse:>10}"
            f"{r.effective_horizon:>6}{gap:>13.2f}{r.real_performance:>10.3f}"
        )
    print()
    print("Reading the table (the design doc's predicted trends):")
    print("  * more human interventions -> lower final Hack Rate")
    print("  * more human interventions -> later/なし collapse -> larger EEH -> horizon_gap -> 0")
    print("  * more human interventions -> higher Real Performance (V*)")


if __name__ == "__main__":
    main()
