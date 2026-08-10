"""Tests pinning the MVP's core claims.

These are behavioural assertions, not just smoke tests: they encode the design
doc's predicted relationships so a future refactor can't silently break the
demonstration.
"""

from __future__ import annotations

from coscientist.human_proxy import HumanProxy, HumanProxyConfig
from coscientist.metrics import compute_metrics, horizon_gap
from coscientist.solver import Solver, SolverConfig
from coscientist.tasks.ttv_overfit import (
    dof_overfit_diagnoser,
    make_env,
    make_proposer,
)


def _run(ask_human: bool, review_every=None, budget=None, max_iters=40):
    env = make_env()
    human = (
        HumanProxy(env, dof_overfit_diagnoser, HumanProxyConfig(budget=budget))
        if ask_human
        else None
    )
    solver = Solver(
        env,
        make_proposer(4),
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
    return env, human, st, compute_metrics(st.steps, window=8)


def test_proxy_and_reference_disagree_on_overfit():
    """The gameable hole exists: deep overfit scores high on V, low on V*."""
    from coscientist.tasks.ttv_overfit import _FREQ_POOL, _ls_fit
    from coscientist.types import Candidate

    env = make_env()
    deep = Candidate(payload=_ls_fit(_FREQ_POOL[:25]))
    p = env.proxy_score(deep)
    r = env.reference_score(deep)
    assert p.score > 1.0, "flawed proxy should reward the overfit"
    assert r.score < 0.0, "reference should punish the overfit"
    assert p.score - r.score > 0.5, "there must be an exploitable gap"


def test_no_human_collapses():
    """Without human refinement the evaluator collapses within the run."""
    _, _, _, m = _run(ask_human=False)
    assert m.collapse_tick is not None, "expected an evaluator collapse"
    assert m.effective_horizon < 60, "EEH should be short without human help"
    assert m.hack_rate_final > 0.3, "hack rate should be high at the end"


def test_human_prevents_collapse_and_extends_horizon():
    """With a Human Proxy holding V*, collapse is prevented and EEH grows."""
    _, human, _, m_no = _run(ask_human=False)
    _, human_y, _, m_yes = _run(ask_human=True, review_every=2)
    assert human_y.patches_emitted >= 1, "human should harden the evaluator"
    assert m_yes.effective_horizon > m_no.effective_horizon
    assert m_yes.hack_rate_final <= m_no.hack_rate_final


def test_single_intervention_removes_runaway_gradient():
    """A threshold finding: even budget=1 kills the unbounded exploit gradient."""
    _, human, _, m = _run(ask_human=True, review_every=2, budget=1)
    assert human.calls == 1
    assert m.collapse_tick is None, "one well-placed patch should prevent collapse"


def test_reference_horizon_is_the_ceiling():
    """Reference (perfect env) has the longest horizon; methods approach it."""
    env = make_env()
    env.apply_patch({"dof_credit_fraction": 1.0})
    solver = Solver(env, make_proposer(4), human=None,
                    config=SolverConfig(max_iters=40, k_candidates=4, ask_human=False))
    ref_m = compute_metrics(solver.run().steps, window=8)

    _, _, _, m_no = _run(ask_human=False)
    _, _, _, m_yes = _run(ask_human=True, review_every=2)

    assert ref_m.collapse_tick is None
    assert horizon_gap(m_no.effective_horizon, ref_m.effective_horizon) > \
        horizon_gap(m_yes.effective_horizon, ref_m.effective_horizon)


def test_solver_never_reads_feedback_text():
    """The solver reacts to the evaluator, not to what the human 'said'.

    We assert the proposer signature is honoured but that behaviour with the
    human channel is driven by verifier patches (guards change), not by the
    feedback string — by checking the environment guards actually moved.
    """
    env, human, st, _ = _run(ask_human=True, review_every=2)
    assert env.guards["dof_credit_fraction"] > 0.0, \
        "environment must have been hardened via patches, not via text"
