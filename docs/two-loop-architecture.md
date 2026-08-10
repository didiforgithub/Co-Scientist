# Two-Loop Co-Evolution — Target Multi-Agent Architecture

> Status: **design, pre-implementation.** This is the aligned target for refactoring
> `coscientist/demo/` from a single-process orchestrator into a live multi-agent
> system. Every decision here was settled in conversation; the "Open, deliberately
> deferred" section lists what is *not* yet decided. Nothing in `coscientist/demo/`
> is changed until this doc is approved.

## 0. Why this is a rewrite, not a feature

Today the demo is **one process**: `Orchestrator.run()` is a synchronous
`for outer_round: for inner_step:` loop that *calls* stateless proposers, which
each spin up a throwaway workspace, run one agent session, and return. The target
is **several long-lived containers that send each other messages**. That inversion
— synchronous call-tree → asynchronous message loop, throwaway workspace →
persistent container, in-process verifier → verifier-as-a-network-service — is the
work. There is (by the user's choice) **no offline-testable intermediate**; we go
to containers directly.

## 1. Roles & the container topology

Three roles. Two are agents in their own containers; one is plain code.

```
┌─────────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR  (plain code — NOT an LLM)                              │
│  starts containers · owns the message bus · owns the 2-day wall clock │
│  extracts traj/artifacts at boundaries · records cost at the gateway  │
│  ENFORCES the permission matrix (the boundaries below are real, not   │
│  conventions)                                                         │
└───────────────┬───────────────────────────────────┬─────────────────┘
                │                                     │
   ┌────────────▼────────────┐          ┌─────────────▼─────────────────┐
   │  SOLUTION CONTAINER      │          │  EVAL CONTAINER                │
   │  ── Solver (pluggable) ──│          │  ── Supervisor (=Eval Proposer)│
   │  reads/writes solution   │          │  owns & rewrites verifier V    │
   │  queries eval as a        │  query  │  runs PROBE red-team here      │
   │  BLACK BOX ───────────────┼────────▶│  (crafts attacks, reproduces   │
   │  (never sees V source)    │◀────────┤   solver's best, checks holes) │
   │                           │ result  │  holds V* (proxy mode only)    │
   └───────────┬───────────────┘         └─────────────┬─────────────────┘
               │      Solver ↔ Supervisor two-way msgs  │
               └───────────────────────────────────────┘
                                   │
                    Supervisor consults one of:
                    ┌──────────┬───────────────┬──────────┐
                    │  human   │  V*-proxy agent│  none     │
                    └──────────┴───────────────┴──────────┘
```

### Permission matrix (final — the last cell was the hard one)

| | solution container | eval container |
|---|---|---|
| **Solver** | read/write (**exclusive**) | **black-box query only** — cannot see V source |
| **Supervisor** | **does not touch it at all** | read/write V **+ runs probe red-team here** |
| **Orchestrator** | starts / extracts / enforces | starts / extracts / enforces |

Why the Supervisor's probe work lives in the **eval** container (chosen over
using the solver's workspace, or opening a third one): testing whether V has a
hole belongs where V lives, and it keeps the solution container a hard black box
the Supervisor never breaches. Writing into the solver's live workspace was
rejected — it dissolves the isolation boundary and races the running solver.

## 2. The Supervisor is a self-contained red-teamer (no gate)

The Supervisor **always has the power to directly modify V** — there is **no
approval gate**. Collapse is prevented not by a human holding a gate, but by the
structural fact that **the Supervisor is the defender**: the direction it edits V
is "harder to game," which is inherently adversarial to the Solver. It does not
need to be constrained by a gate.

Its three "human" modes are **autonomy levels for how it decides what to change**,
not gatekeepers:

- **human** — a real person advises (via a blocking CLI).
- **V\*-proxy** — an agent holding the hidden reference V\* advises.
- **none** — fully autonomous.

Hack detection folds **three information sources**, none of which requires V\*:

1. **Probes it crafts itself** — build a solution it *knows* should score low
   (an obvious overfit), submit it to the current V, and see if V is fooled into a
   high score. A fooled probe = a hole. This is the concrete meaning of "the
   Supervisor experiments": it has a workspace in the eval container, so it can
   actually *run* attacks, not merely reason about them.
2. **The Solver's messages** — when the Solver asks for a review it includes its
   current best solution; the Supervisor reasons about whether it's gaming.
3. **The submission trajectory** — is the score climbing while the method drifts
   toward overfitting?

V\*-proxy / human are *additional*, more-accurate calibration sources layered on
top — the **none** mode stands on its own via source (1).

## 3. Two channels (keep them separate)

- **Solver → Eval : black-box query.** The Solver submits a solution, gets a
  result. It **never sees V's source.** Crucially, **what the result contains is
  controlled by the Supervisor and evolves with V** — the feedback interface is
  *part of* the evaluator. How much diagnostic info V returns (residuals, per-term
  breakdowns, artifacts) is a **Supervisor strategy dial**: expose more → the
  Solver optimizes better but can game more easily; expose less → harder to game
  but slower exploration. (This is the same argument as the design doc's "the
  Human Proxy is neither a bare yes/no nor free-text" — made into a knob.)
- **Solver ↔ Supervisor : two-way messages.** The Solver can proactively ask
  "review my current best for hacking"; the Supervisor judges (via its active
  mode) and replies with a verdict / dense guidance. The Supervisor can also push
  the other way ("V changed, re-baseline" / "stop").

## 4. Rhythm: event-driven, no fixed rounds

The current `outer_round × inner_step` cadence is **removed.** Both agents run
long and asynchronously; **either side can trigger a conversation** (two-way,
settled):

- **Solver (proactive):** detects its own **plateau** — best eval score has not
  improved over the last *N* submissions (window by count or time) — and asks for
  a review.
- **Supervisor (proactive):** watches the solution container; on **stall** or a
  **probe-detected hack**, it intervenes — inspects, then rewrites V.

There is no "round." There is one **outer budget: a 2-day wall clock**, and the
Solver's internal plateau detector. After V changes, the Solver perceives the
score shift through the API and naturally continues — **the loop closes itself; no
external clock has to drive it.**

## 5. Pluggable Solver + the time budget

A single `Solver` interface, with `eval-client`, `supervisor-channel`, and a
`deadline` injected:

```python
class Solver(Protocol):
    def run(self, *, solution_ws: Path, eval: EvalClient,
            supervisor: SupervisorChannel, deadline: Deadline) -> None: ...
```

- **SimpleTES-agent** — its internal best-of-K / notes loop is kept, but the stop
  condition flips from **submission-count** (`while submissions < N`) to
  **wall-clock** (`while not deadline.expired()`). Each candidate is scored via
  `eval.score()`; it periodically asks the Supervisor for a review.
- **Codex direct-run** — Codex runs in the container with `eval` and `supervisor`
  exposed to it as tools/CLI, deciding for itself when to score and when to ask
  for review, until the deadline.

**Budget = wall-clock, not submissions** (≤ 2 days). This is a metric change too:
the design doc's **EEH horizon axis moves from tick-count to wall-clock time**, and
per-solver **cost curves** (SimpleTES vs Codex) become a first-class comparison.

## 6. Storage: free inside the container,规范 at the boundary

Hard principle: **do not impose a write format on the agent.** Inside its
container the agent scratches however it likes. **Normalization happens at the
boundary** — at session end / checkpoint the Orchestrator *extracts* trajectory +
artifacts (reusing Harbor's declarative `artifacts` mechanism), and **cost is
recorded at the gateway** (every eval-API call and every LLM generation passes
through our proxy; token + seconds are logged there, not left to the agent's
goodwill).

```
runs/<run_id>/
  manifest.json          # config: solver type, supervisor mode (human/proxy/none),
                         #         start/stop wall-clock, budget
  events.jsonl           # global event stream (today's EventLog, + wall-clock + cost)
  cost.jsonl             # one line per LLM/eval/consult call: {who,kind,tokens,seconds,$}
  solver/
    trajectory.jsonl     # the solver's full trajectory (extracted, not agent-authored)
    candidates/          # each submitted solution + the feedback it got back
  supervisor/
    reviews.jsonl        # each consult: evidence, verdict, dense feedback
    verifier_versions/   # v0.py, v1.py, ... + per-version diff & rationale
    probes/              # red-team probes crafted + whether V was fooled
  eval/
    queries.jsonl        # every Solver→Eval query AND what V chose to return  ← see §3
```

`eval/queries.jsonl` is load-bearing: since **what V returns evolves and is the
key evidence in the hack arms race**, every return-shape decision must be logged.
Cost/traj storage and "the feedback interface is part of V" are therefore coupled,
not two separate asks.

This is the operational form of the design doc's **"Human Proxy Dataset"**
deliverable — the `(x_t, query, feedback, patch)` tuples for others to reproduce.

## 7. What is reused vs newly built

| Existing piece | Fate |
|---|---|
| `HumanPort` (`AutoHuman` / `CliHuman`) | reused as Supervisor's **advisor** modes (V\*-proxy / human); the *gate* semantics drop (§2) |
| `Evaluator` (versioned verifier, out-of-process) | core reused, but **wrapped as a network service** (Solver queries it, Supervisor rewrites it) |
| `taskspec` V\* | reused as the proxy-mode ground truth |
| `EvaluatorProposer` rewrite logic | becomes a Supervisor action (one-shot session; `HarborBackend` suffices) |
| `HarborBackend` / `harbor_runtime` | reused, but upgraded from one-shot exec-and-recover to a **persistent-container** mode for the long-lived Solver |
| `EventLog` (append-only JSONL) | reused as the seed of `events.jsonl`, extended with wall-clock + cost + cross-container origin |
| **message bus** (Solver ↔ Supervisor) | **new** |
| **persistent solver runtime + `Solver` interface** | **new** |
| **eval-as-service + permission enforcement** | **new** |

## 8. Migration path (order of work, once approved)

1. **Eval-as-service** — wrap `Evaluator` behind a query API returning a
   Supervisor-controlled result shape; the return shape itself versions with V.
2. **`Solver` interface + wall-clock deadline** — define the Protocol; port the
   stub + SimpleTES loop to wall-clock; add the plateau detector.
3. **Message bus** — the two-way Solver ↔ Supervisor channel; retire the
   `outer_round × inner_step` loop for an event loop.
4. **Persistent-container runtime** — extend `HarborBackend` so the Solver lives
   across many queries in one container; Supervisor gets its eval-container probe
   workspace.
5. **Storage layer** — the `runs/<run_id>/` layout; boundary extraction; gateway
   cost logging.
6. **Codex direct-run solver** — the second `Solver` implementation.

## Open, deliberately deferred

- **Message bus mechanism** — files/sockets/HTTP/a broker? (implementation detail)
- **How the eval service is started & addressed** across containers.
- **Plateau `N`** — exact window (count vs time) and threshold.
- **Second setting with a real (non-toy) evaluator** — explicitly deferred earlier.
