"""`coscientist-demo` — run the two-loop co-evolution with a human port.

Examples
--------
    # Fully offline, deterministic (stub proposers, AutoHuman). Always runs.
    python -m coscientist.demo.cli

    # Real coding agent drives BOTH evolution loops; you arbitrate at the CLI.
    python -m coscientist.demo.cli --agent codex --human cli
    python -m coscientist.demo.cli --agent claude-code --human cli

    # Real agent, but let the scalable AutoHuman stand in for the human.
    python -m coscientist.demo.cli --agent codex --human auto

The default (`--agent stub --human auto`) is the one guaranteed to run anywhere;
it exercises the entire control flow — two evolving loops, a triggered
evaluator rewrite, a human review, a verifier version bump, an archive rescore —
without any network or API key.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .agent_backend import make_backend
from .evaluator import Evaluator
from .eventlog import EventLog
from .human_port import AutoHuman, CliHuman
from .orchestrator import Orchestrator, OrchestratorConfig
from .proposers import EvaluatorProposer, SolutionProposer, default_stub_handlers


def build(args) -> Orchestrator:
    evaluator = Evaluator.initial()

    # One backend per proposer role. The stub needs handlers to dispatch on role.
    # Harbor backends carry the isolation knob + any site compose overlays; other
    # backends ignore those kwargs.
    handlers = default_stub_handlers()
    extra_compose = list(args.extra_compose or [])
    sol_backend = make_backend(args.agent, handlers=handlers,
                               isolation=args.isolation, extra_compose=extra_compose)
    eval_backend = make_backend(args.agent, handlers=handlers,
                                isolation=args.isolation, extra_compose=extra_compose)

    solution_proposer = SolutionProposer(sol_backend, model=args.model, timeout_s=args.timeout)
    evaluator_proposer = EvaluatorProposer(eval_backend, model=args.model, timeout_s=args.timeout)

    if args.human == "cli":
        human = CliHuman(evaluator, auto_fallback=AutoHuman(evaluator))
    else:
        human = AutoHuman(evaluator)

    log_path = Path(args.log) if args.log else None
    if log_path and log_path.exists():
        log_path.unlink()

    return Orchestrator(
        evaluator=evaluator,
        solution_proposer=solution_proposer,
        evaluator_proposer=evaluator_proposer,
        human_port=human,
        config=OrchestratorConfig(
            outer_rounds=args.rounds,
            inner_steps=args.inner,
            k_candidates=args.k,
            review_every=args.review_every,
            seed=args.seed,
        ),
        log=EventLog(path=log_path),
    )


def summarize(orch: Orchestrator) -> None:
    ev = orch.evaluator
    best = orch.best
    print("\n" + "=" * 72)
    print("Co-Scientist demo — two-loop co-evolution with a human port")
    print("=" * 72)
    print(f"agent backend rounds : {orch.config.outer_rounds} outer x {orch.config.inner_steps} inner")
    print(f"verifier versions    : {len(ev.versions)}  (0=initial flawed -> {ev.current.version}={ev.current.note})")
    print(f"human reviews         : {orch.reviews}")
    print(f"verifier changes      : {orch.verifier_changes}")
    print(f"solutions in archive  : {len(orch.archive)}")
    print(f"best solution modes   : {best.n_modes}")
    print(f"best proxy score (now): {best.proxy_score:.4f}   (under final verifier v{ev.current.version})")
    print(f"best real  score (V*) : {best.real_score:.4f}   (hidden ground truth)")
    print("-" * 72)
    print("Verifier lineage:")
    for v in ev.versions:
        print(f"  v{v.version}: origin={v.origin:<7} {v.note}")
    print("-" * 72)
    print("Event tally:")
    kinds: dict[str, int] = {}
    for e in orch.log.events:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    for k, c in kinds.items():
        print(f"  {k:<28} {c}")
    print("=" * 72)
    print("Read it as: the solution loop games verifier v0 (proxy score climbs on")
    print("uncounted mode DOF); the evaluator loop proposes a hardened verifier;")
    print("the HUMAN PORT ratifies it; the archive is rescored under the new V and")
    print("the overfit exploit stops paying. Both loops ran; the human gated every")
    print("change to the evaluator.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Co-Scientist two-loop co-evolution demo")
    ap.add_argument("--agent", default="stub",
                    choices=["stub", "none", "no-agent", "codex", "claude", "claude-code",
                             "harbor", "harbor-claude", "harbor-claude-code", "harbor-codex", "dsh"],
                    help="proposer backend for BOTH loops (default: stub, offline). "
                         "harbor* backends run each session in a container (needs docker+harbor).")
    ap.add_argument("--isolation", default="separate", choices=["separate", "shared"],
                    help="harbor grading topology: separate (grader in its own container, "
                         "TB2/TB3 model) or shared (agent+grader together, AutoLab model)")
    ap.add_argument("--extra-compose", action="append", default=None, metavar="PATH",
                    help="extra docker-compose overlay for harbor (repeatable), e.g. a DNS-pin "
                         "hosts_overlay.yaml. Ignored by non-harbor backends.")
    ap.add_argument("--human", default="auto", choices=["auto", "cli"],
                    help="human port backend (default: auto = scalable stand-in)")
    ap.add_argument("--model", default=None, help="model name passed to the agent CLI")
    ap.add_argument("--rounds", type=int, default=8, help="outer rounds")
    ap.add_argument("--inner", type=int, default=3, help="inner solution steps per round")
    ap.add_argument("--k", type=int, default=3, help="candidates per inner step")
    ap.add_argument("--review-every", type=int, default=2, help="force a human review every N rounds")
    ap.add_argument("--timeout", type=float, default=120.0, help="per agent-session timeout (s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None, help="path to write the JSONL event log")
    args = ap.parse_args()

    orch = build(args).run()
    summarize(orch)


if __name__ == "__main__":
    main()
