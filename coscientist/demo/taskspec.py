"""The fixed demo task instance + a hidden reference — shared, agent-invisible.

The demo domain is the same TTV-overfit curve fit used by the single-loop MVP,
but here the *verifier* is a rewritable source string (see ``evaluator.py``),
so the "evaluator evolves" loop has something concrete to edit.

Two things live here and are **never written into the agent workspace**:

  * ``INSTANCE`` — the deterministic (t, obs, signal) instance. The agents see
    only ``t`` and ``obs`` (via the workspace); ``signal`` is the hidden truth.
  * ``reference_score`` — V*, honest reduced-chi-square against the true signal.
    Used only for metrics and by the AutoHuman stand-in when deciding whether a
    proposed evaluator change is an improvement. Real humans wouldn't have this;
    the AutoHuman is a scalable proxy for the human that does.

Keeping these out of the workspace is the whole point: an agent tasked with
improving the solution (or the verifier) cannot read the answer key.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

_N = 60
_SEED = 7
_NOISE_SIGMA = 1.0


@dataclass(frozen=True)
class Instance:
    t: list[float]
    obs: list[float]
    signal: list[float]        # HIDDEN — never serialized into the workspace
    noise_sigma: float


def _build_instance() -> Instance:
    rng = np.random.default_rng(_SEED)
    t = np.linspace(0.0, 10.0, _N)
    signal = 0.5 * t + 2.0
    for f, a, p in [(0.6, 3.0, 0.4), (1.1, 1.5, 2.0)]:
        signal = signal + a * np.sin(f * t + p)
    obs = signal + rng.normal(0.0, _NOISE_SIGMA, size=_N)
    return Instance(t.tolist(), obs.tolist(), signal.tolist(), _NOISE_SIGMA)


INSTANCE = _build_instance()


# --- prediction + honest reference (hidden V*) ------------------------------
def predict(payload: dict, t: list[float] | None = None) -> np.ndarray:
    tt = np.asarray(INSTANCE.t if t is None else t, dtype=float)
    a, b = payload.get("trend", (0.0, 0.0))
    pred = a * tt + b
    for mode in payload.get("modes", []):
        f, amp, ph = mode
        pred = pred + amp * np.sin(f * tt + ph)
    return pred


def num_free_params(payload: dict) -> int:
    return 2 + 3 * len(payload.get("modes", []))


def reference_score(payload: dict) -> tuple[bool, float, dict]:
    """V*: reduced chi-square vs the TRUE signal, honest DOF. Cannot be gamed.

    Returned raw is ``-reduced_chi2`` (higher is better). Overfitting the noise
    moves the prediction away from the true signal, so piling on modes lowers
    this — the property that makes it the ground truth.
    """
    pred = predict(payload)
    truth = np.asarray(INSTANCE.signal, dtype=float)
    resid2 = float(np.sum((truth - pred) ** 2)) / (INSTANCE.noise_sigma ** 2)
    k = num_free_params(payload)
    dof = max(1, _N - k)
    red = resid2 / dof
    return True, -red, {"reduced_chi2_vs_truth": red, "num_free_params": k}


# --- least-squares fit used by the stub solution proposer -------------------
_FREQ_POOL = list(np.linspace(0.2, 6.0, 40))


def ls_fit(freqs: list[float]) -> dict:
    """Genuine least-squares fit of trend + sinusoids at ``freqs`` to the obs.

    More frequencies drive the residual against the *observations* arbitrarily
    low (overfitting the noise) — the exploit the flawed verifier rewards.
    """
    t = np.asarray(INSTANCE.t, dtype=float)
    obs = np.asarray(INSTANCE.obs, dtype=float)
    cols = [t, np.ones_like(t)]
    for f in freqs:
        cols += [np.sin(f * t), np.cos(f * t)]
    A = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(A, obs, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    modes = []
    for i, f in enumerate(freqs):
        s, c = float(coef[2 + 2 * i]), float(coef[3 + 2 * i])
        modes.append((float(f), float(math.hypot(s, c)), float(math.atan2(c, s))))
    return {"trend": (a, b), "modes": modes}


def workspace_context() -> dict:
    """The agent-visible slice of the instance (no hidden ``signal``)."""
    return {"t": INSTANCE.t, "obs": INSTANCE.obs, "noise_sigma": INSTANCE.noise_sigma}


def context_json() -> str:
    return json.dumps(workspace_context(), indent=2)
