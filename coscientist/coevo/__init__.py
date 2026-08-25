"""Co-Scientist target architecture — the live multi-agent co-evolution system.

This subpackage is the target described in ``docs/two-loop-architecture.md``: two
long-lived roles that pass messages, rather than the single-process
``for round: for step:`` reference implementation in ``coscientist.demo``.

    demo/   — one process, synchronous call-tree, throwaway workspaces. The
              reference implementation and the home of the reused primitives
              (Evaluator, taskspec, Workspace, human_port, harbor_runtime, ...).
    coevo/  — Solver (in the solution container) and Supervisor (=Eval Proposer,
              in the eval container) talking over two channels, on a wall-clock
              budget, with cost + trajectory recorded at the boundary.

Roles
-----
    Solver       — read/writes the solution; queries eval as a BLACK BOX (never
                   sees V's source). Pluggable: StubSolver (offline) and
                   CodexSolver (a real coding agent) share one interface.
    Supervisor   — = Eval Proposer. Owns and rewrites V; runs red-team probes in
                   the eval container. Autonomous mode installs validated changes;
                   optional Human Sessions freeze a proposed change until a Feishu
                   expert or model-backed Human Proxy finishes the shared conversation
                   contract. The Proxy receives evaluator context as text and has no
                   solution/evaluator execution environment.
    driver       — plain code (not an LLM): starts the roles, owns the two-day
                   wall clock, routes the two channels, records everything.

Channels
--------
    Solver -> Eval        : black-box query; the RESULT SHAPE is a Supervisor knob
                            (``FeedbackLevel``) that evolves with V.
    Solver <-> Supervisor : two-way messages (plateau review request; proactive
                            nudge / stop).
"""
