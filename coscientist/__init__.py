"""Co-Scientist MVP: Human-Guided Environment Refinement for Automated Research.

A minimal, runnable skeleton of the design doc's second-layer thesis — that the
bottleneck in automated research is not solution discovery but constructing and
*continuously refining* a Proxy Environment that is hard to game.

The pieces map 1:1 onto the two reference systems:

    ProxyEnvironment  <- autosciworld (verifier.py = V, _meta/reference.py = V*)
    Solver            <- ExplorationHarness (evolutionary best-of-K + notes)
    HumanProxy        <- the novel middle: a V*-holding, dense-feedback,
                         evaluator-patching stand-in for a human expert
    metrics           <- Hack Rate / Effective Exploration Horizon / Real Perf
"""

from .environment import ProxyEnvironment, Task
from .human_proxy import HumanProxy, HumanProxyConfig
from .metrics import RunMetrics, compute_metrics, horizon_gap
from .solver import Solver, SolverConfig, SolverState
from .types import Candidate, EvalResult, Feedback, Note, Step

__all__ = [
    "ProxyEnvironment",
    "Task",
    "HumanProxy",
    "HumanProxyConfig",
    "RunMetrics",
    "compute_metrics",
    "horizon_gap",
    "Solver",
    "SolverConfig",
    "SolverState",
    "Candidate",
    "EvalResult",
    "Feedback",
    "Note",
    "Step",
]
