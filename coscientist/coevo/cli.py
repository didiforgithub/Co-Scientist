"""`coscientist-coevo` — run the target multi-agent co-evolution (docs §, coevo/).

This is the live-multi-agent counterpart to ``coscientist-demo``. It wires a
Solver (in its solution container) and a Supervisor (=Eval Proposer, owning V and
red-teaming it in the eval container) across the two channels, on a wall-clock
budget, recording everything into ``runs/<run_id>/``.

Examples
--------
    # Fully offline, deterministic: stub solver + autonomous (no-human) supervisor.
    python -m coscientist.coevo.cli

    # Autonomy modes for the Supervisor (§2):
    python -m coscientist.coevo.cli --supervisor none     # decides from its own probes
    python -m coscientist.coevo.cli --supervisor proxy    # a V*-holding proxy advises
    python -m coscientist.coevo.cli --supervisor human    # a person advises at the CLI

    # A real coding agent as the Solver (needs codex on PATH):
    python -m coscientist.coevo.cli --solver codex

The default (`--solver stub --supervisor none`) runs anywhere with no network,
no keys, and demonstrates the whole arc: the solver games v0, the supervisor's
red-team catches it, V is hardened WITHOUT a gate, the solver re-baselines and
retreats to an honest fit.

The GENERAL path — the SYSTEM builds its own evaluator from raw input (needs
docker + the claude agent binary + a model gateway):

    # Drop a raw problem in and let the system research, author V, and attempt it:
    python -m coscientist.coevo.cli --problem chowla --budget-s 1800
    python -m coscientist.coevo.cli --input path/to/raw_input_dir --budget-s 1800

Here NOTHING is hardcoded about the domain: a Supervisor agent reads the raw input,
authors the verifier + probes, a Solver agent attempts it against that black box in
a long-lived container, and the Supervisor agent-hardens V when its probes are
rewarded. Clean-fails with an explanation if docker/agent/gateway are absent.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..demo import taskspec
from ..demo.evaluator import Evaluator
from ..demo.human_port import AutoHuman, CliHuman
from .driver import CoevoConfig, LogicalClock, build_run
from .eval_service import FeedbackLevel
from .solver import StubSolver
from .supervisor import SupervisorMode


def _make_solver(name: str):
    if name in ("stub", "none", "no-agent"):
        return StubSolver()
    if name == "codex":
        from .codex_solver import CodexSolver
        return CodexSolver()
    raise ValueError(f"unknown solver: {name!r}")


def _make_advisor(mode: SupervisorMode):
    """proxy/human advisors reuse the demo HumanPort (it holds V* for proxy)."""
    if mode == SupervisorMode.NONE:
        return None
    evaluator = Evaluator.initial()   # advisor's own handle to score probes
    if mode == SupervisorMode.HUMAN:
        return CliHuman(evaluator, auto_fallback=AutoHuman(evaluator))
    return AutoHuman(evaluator)       # proxy = V*-holding scalable stand-in


def build(args):
    mode = SupervisorMode(args.supervisor)
    cfg = CoevoConfig(
        budget_s=args.budget_hours * 3600.0,
        step_seconds=args.step_seconds,
        supervise_every=args.supervise_every,
        max_steps=args.max_steps,
        feedback_level=FeedbackLevel(args.feedback),
        supervisor_mode=mode,
        seed=args.seed,
    )
    run_dir = Path(args.runs_dir) / args.run_id
    solver = _make_solver(args.solver)
    advisor = _make_advisor(mode)
    # A real coding-agent solver needs true wall-clock time; the stub is
    # deterministic under a logical clock. Only the stub gets the logical clock.
    logical = LogicalClock() if args.solver in ("stub", "none", "no-agent") else None
    return build_run(run_dir=run_dir, solver=solver, config=cfg,
                     logical_clock=logical, advisor=advisor)


def summarize(run) -> None:
    s = run.summary()
    print("\n" + "=" * 72)
    print("Co-Scientist coevo — live two-role co-evolution (Solver <-> Supervisor)")
    print("=" * 72)
    print(f"solver / supervisor    : {run.solver.name} / {run.config.supervisor_mode.value}")
    print(f"steps (wall-clock bound): {s['steps']}   budget={run.config.budget_s:.0f}s")
    print(f"verifier versions       : v0 -> v{s['final_verifier_version']}  "
          f"({s['verifier_hardenings']} hardening(s), NO approval gate)")
    print(f"supervisor reviews      : {s['reviews_handled']}")
    print(f"final feedback level    : {s['final_feedback_level']}")
    print("-" * 72)
    print(f"best solution modes     : {s['best_n_modes']}")
    print(f"best proxy score (now)  : {fmt(s['best_proxy_score'])}   (under final V)")
    print(f"best real  score (V*)   : {fmt(s['best_real_score'])}   (hidden ground truth)")
    print("-" * 72)
    print(f"run recorded under      : {run.store.root}")
    print("Read it as: the solver games v0 (piling on modes drives proxy up); the")
    print("supervisor red-teams V in the eval container, catches the hole, and")
    print("hardens V with NO gate; the solver perceives V move through the black box")
    print("and retreats to an honest fit. Two roles, two channels, wall-clock bound.")


def fmt(x):
    return "n/a" if x is None else f"{x:.4f}"


# ---------------------------------------------------------------------------
# the general agent-system path — the SYSTEM builds its own evaluator
# ---------------------------------------------------------------------------
_BUILTIN_PROBLEMS = {
    "chowla": Path(__file__).parent / "problems" / "chowla",
}


def _resolve_raw_input(args) -> Path:
    if args.input:
        return Path(args.input)
    if args.problem and args.problem != "curvefit":
        d = _BUILTIN_PROBLEMS.get(args.problem)
        if d is None:
            raise ValueError(f"unknown built-in problem: {args.problem!r}")
        return d
    raise ValueError("agent-system path needs --input <dir> or --problem <name>")


def _load_llm_config(args) -> dict:
    """Assemble the per-problem LLM config for HOST-SIDE eval/feedback code.

    Precedence: ``--llm-config <path>`` (a JSON file kept OUTSIDE the repo/runs
    tree, ``{"api_key","base_url","model"}``) then the ``LLM_API_KEY`` /
    ``LLM_BASE_URL`` / ``LLM_MODEL`` process-env fallback. Returns a plain dict
    with only the present keys (``api_key``/``base_url``/``model``); empty when
    nothing is configured. These creds are injected ONLY into the host-side
    Evaluator subprocess — never into the Solver container — and are never
    written to disk (events record just ``llm_config_present``).
    """
    cfg: dict = {}
    path = getattr(args, "llm_config", None)
    if path:
        p = Path(path)
        if not p.is_file():
            raise ValueError(f"--llm-config file not found: {p}")
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"--llm-config is not readable JSON: {e}") from e
        if not isinstance(raw, dict):
            raise ValueError("--llm-config must be a JSON object")
        for k in ("api_key", "base_url", "model"):
            v = raw.get(k)
            if v:
                cfg[k] = str(v)
    # env fallback fills only keys not already set by the file
    for k, envk in (("api_key", "LLM_API_KEY"), ("base_url", "LLM_BASE_URL"),
                    ("model", "LLM_MODEL")):
        if k not in cfg and os.environ.get(envk):
            cfg[k] = os.environ[envk]
    return cfg


def _build_resource_overrides(args):
    """Build a ResourceSpec of CLI overrides (non-None fields fold over the file).

    Solver slice: --solver-cpus/--solver-memory-mb/--solver-allow-internet (plus the
    existing --solver-image/--solver-gpus, folded in AgentSystem). Verifier slice:
    --verifier-image/--verifier-gpus/--verifier-cpus/--verifier-memory-mb/
    --verifier-timeout-s. Returns None when no override flag was given."""
    from . import resources as R
    sol = R.ContainerResources(
        cpus=getattr(args, "solver_cpus", None),
        memory_mb=getattr(args, "solver_memory_mb", None),
        allow_internet=bool(getattr(args, "solver_allow_internet", False)),
    )
    ver = R.ContainerResources(
        image=getattr(args, "verifier_image", None),
        gpus=R._as_gpus(getattr(args, "verifier_gpus", None)),
        cpus=getattr(args, "verifier_cpus", None),
        memory_mb=getattr(args, "verifier_memory_mb", None),
        timeout_sec=getattr(args, "verifier_timeout_s", None),
        allow_internet=bool(getattr(args, "verifier_allow_internet", False)),
    )
    if sol.is_empty() and ver.is_empty():
        return None
    return R.ResourceSpec(solver=sol, verifier=ver)


def run_agent_system(args) -> None:
    from .agent_system import AgentSystem, AgentSystemUnavailable

    raw = _resolve_raw_input(args)
    run_dir = Path(args.runs_dir) / args.run_id
    llm_config = _load_llm_config(args)
    opt = {}
    if getattr(args, "bootstrap_timeout_s", None):
        opt["bootstrap_timeout_s"] = args.bootstrap_timeout_s
    if getattr(args, "harden_timeout_s", None):
        opt["harden_timeout_s"] = args.harden_timeout_s
    if getattr(args, "post_harden_solve_s", None):
        opt["post_harden_solve_s"] = args.post_harden_solve_s
    if llm_config:
        opt["llm_config"] = llm_config
    if getattr(args, "solver_image", None):
        opt["solver_image"] = args.solver_image
    if getattr(args, "solver_gpus", None):
        opt["solver_gpus"] = args.solver_gpus
    if getattr(args, "resource_config", None):
        opt["resource_config_path"] = Path(args.resource_config)
    ov = _build_resource_overrides(args)
    if ov is not None:
        opt["resource_overrides"] = ov
    system = AgentSystem(
        raw_input_dir=raw, run_dir=run_dir,
        budget_s=args.budget_s if args.budget_s is not None
        else args.budget_hours * 3600.0,
        feedback_level=FeedbackLevel(args.feedback),
        **opt,
    )
    print("=" * 72)
    print("Co-Scientist coevo — GENERAL agent system (the system builds its own eval)")
    print("=" * 72)
    print(f"raw input : {raw}")
    print(f"run dir   : {run_dir}")
    print(f"budget    : {system.budget_s:.0f}s")
    print(f"llm config: {'present' if llm_config else 'absent'} "
          f"(host-side eval only; never in the Solver container)")
    if args.solver_image or args.solver_gpus:
        print(f"solver env: image={args.solver_image or 'default'} "
              f"gpus={args.solver_gpus or 'none'}")
    if getattr(args, "resource_config", None):
        print(f"resource  : {args.resource_config} "
              "(resource.toml; Eval/Solve containers isolated)")
    if ov is not None:
        print(f"res overr : solver={ov.solver.to_manifest()} "
              f"verifier={ov.verifier.to_manifest()}")
    print("-" * 72)
    try:
        system.run(max_turns=args.max_turns, resume=getattr(args, "resume", False))
    except AgentSystemUnavailable as e:
        print(f"\nagent system unavailable (clean stop): {e}")
        print("This path needs docker + the claude agent binary + a model gateway.")
        return
    s = system.summary()
    print("\n" + "=" * 72)
    print("agent system — result")
    print("=" * 72)
    print(f"verifier versions  : v0 -> v{s['final_verifier_version']}  "
          f"({s['verifier_hardenings']} agent-authored hardening(s))")
    print(f"supervisor reviews : {s['reviews_handled']}")
    print(f"best score (final V): {fmt(s['best_score'])}")
    print(f"run recorded under : {run_dir}")
    print("Inspect: supervisor/verifier_versions/ (agent-authored V), "
          "solver/candidates/, eval/queries.jsonl, events.jsonl.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Co-Scientist coevo — target multi-agent system")
    ap.add_argument("--solver", default="stub", choices=["stub", "none", "no-agent", "codex"],
                    help="solver logic (default: stub, offline & deterministic)")
    ap.add_argument("--supervisor", default="none", choices=["none", "proxy", "human"],
                    help="Supervisor autonomy mode (§2): none=autonomous, "
                         "proxy=V*-holding proxy advises, human=person advises at CLI")
    ap.add_argument("--problem", default="curvefit",
                    help="curvefit (built-in demo, offline) or a built-in raw problem "
                         "name (e.g. chowla) that routes to the general agent system")
    ap.add_argument("--input", default=None,
                    help="path to a raw-input dir/file -> the general agent system "
                         "builds its own evaluator from it (docker + agent required)")
    ap.add_argument("--feedback", default="with_artifacts",
                    choices=[f.value for f in FeedbackLevel],
                    help="initial eval disclosure level (§3 strategy dial)")
    ap.add_argument("--budget-hours", type=float, default=48.0,
                    help="wall-clock budget in hours (ceiling: 48 = two days)")
    ap.add_argument("--budget-s", type=float, default=None,
                    help="wall-clock budget in seconds (overrides --budget-hours; "
                         "used by the agent-system path)")
    ap.add_argument("--step-seconds", type=float, default=60.0,
                    help="logical seconds charged per solver step (stub runs)")
    ap.add_argument("--supervise-every", type=int, default=3,
                    help="Supervisor observes the solution container every N steps")
    ap.add_argument("--max-steps", type=int, default=60,
                    help="safety cap on solver steps for finite offline runs")
    ap.add_argument("--max-turns", type=int, default=12,
                    help="max Solver agent turns in the general agent-system path")
    ap.add_argument("--bootstrap-timeout-s", type=float, default=None,
                    help="wall-clock cap for the Supervisor bootstrap agent turn "
                         "(agent-system path; default in AgentSystem, ~900s)")
    ap.add_argument("--harden-timeout-s", type=float, default=None,
                    help="protected wall-clock cap for each Supervisor harden turn "
                         "(agent-system path; default ~600s)")
    ap.add_argument("--post-harden-solve-s", type=float, default=None,
                    help="protected budget for the guaranteed post-harden Solver turn "
                         "(agent-system path; default ~900s)")
    ap.add_argument("--resume", action="store_true",
                    help="resume an interrupted agent-system run from runs/<run-id>/ "
                         "(rebuilds V chain, hardenings, and best-so-far from disk; "
                         "skips re-bootstrap). No-op if no completed bootstrap is found.")
    ap.add_argument("--llm-config", default=None,
                    help="path to a JSON file (kept OUTSIDE repo/runs) with "
                         '{"api_key","base_url","model"} for host-side eval/feedback '
                         "code that consults an LLM (e.g. an LLM-verifier). Falls back "
                         "to LLM_API_KEY/LLM_BASE_URL/LLM_MODEL env vars. NEVER written "
                         "to disk and NEVER injected into the Solver container.")
    ap.add_argument("--solver-image", default=None,
                    help="docker image for the long-lived Solver container "
                         "(agent-system path; overrides the default python:3.11-slim). "
                         "Use a task-provided image for kernel/GPU tasks.")
    ap.add_argument("--solver-gpus", default=None,
                    help="GPU spec passed to `docker run --gpus` for the Solver "
                         "container (e.g. 'all' or '1'). Needs the host nvidia docker "
                         "runtime. Default: no GPU. No creds are added to the container.")
    # ---- resource config: Harbor-style resource.toml + fine CLI overrides ----
    ap.add_argument("--resource-config", default=None,
                    help="path to a Harbor-style resource.toml (or a K3 task.toml) with "
                         "isolated [solver]/[verifier] resource slices. Wins over a "
                         "resource.toml auto-discovered inside the input dir. Missing/"
                         "unparseable falls back to defaults (never fatal).")
    ap.add_argument("--solver-cpus", type=float, default=None,
                    help="CPU cap for the Solver container (docker --cpus).")
    ap.add_argument("--solver-memory-mb", type=int, default=None,
                    help="memory cap in MiB for the Solver container (docker --memory).")
    ap.add_argument("--solver-allow-internet", action="store_true",
                    help="allow network in the Solver container (default: --network none "
                         "once any solver resource cap is set).")
    ap.add_argument("--verifier-image", default=None,
                    help="docker image for the ISOLATED verifier (Eval) container. "
                         "Default: the solver image (same torch/CUDA base). Only used "
                         "when the task ships a checker or a verifier resource is set.")
    ap.add_argument("--verifier-gpus", default=None,
                    help="GPU spec for the verifier container (docker --gpus). Isolated "
                         "from --solver-gpus — this is the Eval/Solve resource split.")
    ap.add_argument("--verifier-cpus", type=float, default=None,
                    help="CPU cap for the verifier container (docker --cpus).")
    ap.add_argument("--verifier-memory-mb", type=int, default=None,
                    help="memory cap in MiB for the verifier container (docker --memory).")
    ap.add_argument("--verifier-timeout-s", type=float, default=None,
                    help="wall-clock cap for one verify run in the verifier container "
                         "(a real GPU scorer needs minutes, not the 10s host default).")
    ap.add_argument("--verifier-allow-internet", action="store_true",
                    help="allow network in the verifier container (default: no-network).")
    ap.add_argument("--runs-dir", default="runs", help="parent dir for run storage")
    ap.add_argument("--run-id", default="coevo_demo", help="run id (subdir under runs-dir)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Route: --input or a non-curvefit --problem -> the general agent system.
    if args.input or (args.problem and args.problem != "curvefit"):
        run_agent_system(args)
        return

    run = build(args).run()
    summarize(run)


if __name__ == "__main__":
    main()
