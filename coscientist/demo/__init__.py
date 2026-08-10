"""The co-evolution demo: two evolving loops around a shared blackboard, with a
human port arbitrating every change to the evaluator.

This subpackage is the *functionally complete* system sketch the design doc
asks for — as opposed to the single-loop `experiments/run_curve.py` MVP. The
difference:

    run_curve.py  : solutions evolve; the evaluator is only *patched* by a
                    HumanProxy holding a hidden V*. One loop.
    demo/         : solutions evolve AND the evaluator evolves, both driven by
                   coding *agents* working in a real file workspace; a HumanPort
                   arbitrates every proposed change to the evaluator. Two loops.

Nothing here needs a hidden reference evaluator V*. When there is no oracle,
the human (or the AutoHuman stand-in) *is* the source of truth for "is this
evaluator change good?" — which is exactly the interface we want to exercise.

Layout
------
    workspace.py        the shared file workspace (solution.py + verifier.py)
    evaluator.py        the evolvable evaluator: run a verifier source string
    agent_backend.py    codex / claude-code headless backends + a stub backend
    proposers.py        SolutionProposer + EvaluatorProposer (agent-driven)
    human_port.py       HumanPort protocol + CliHuman + AutoHuman
    eventlog.py         append-only JSONL event stream (for CLI + dataset)
    orchestrator.py     interleave the two loops, route patches through review
    cli.py              `coscientist-demo` entry point
"""
