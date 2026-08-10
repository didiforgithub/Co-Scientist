"""Behavioural tests for the target multi-agent co-evolution (``coscientist.coevo``).

These pin the *shape* of the two-role system the same way ``test_demo.py`` pins
the single-process demo: both channels exist and are enforced, the Supervisor can
always harden V (NO gate), the Solver perceives V move through the black box and
retreats, cost/queries/versions land in the standardized ``runs/<id>/`` layout,
and the real-agent (codex) path clean-fails when its binary is absent.

Everything here runs offline, deterministic, no network and no keys:
  * the StubSolver drives the stepped, logical-clock path;
  * the Gateway is exercised directly with the stdlib client the shims use;
  * the codex path is checked only for its availability guard (no codex needed).
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from coscientist.demo import taskspec
from coscientist.demo.evaluator import Evaluator
from coscientist.coevo.budget import Deadline
from coscientist.coevo.driver import CoevoConfig, LogicalClock, build_run
from coscientist.coevo.eval_service import EvalService, FeedbackLevel
from coscientist.coevo.gateway import Gateway
from coscientist.coevo.solver import StubSolver
from coscientist.coevo.supervisor import Supervisor, SupervisorMode


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _service(level=FeedbackLevel.WITH_ARTIFACTS) -> EvalService:
    return EvalService(evaluator=Evaluator.initial(),
                       ctx_provider=taskspec.workspace_context,
                       feedback_level=level)


def _deep():
    return taskspec.ls_fit(taskspec._FREQ_POOL[:18])


def _small():
    return taskspec.ls_fit(taskspec._FREQ_POOL[:2])


def _offline_run(tmp_path: Path, *, mode=SupervisorMode.NONE, budget_s=None,
                 step_seconds=60.0, max_steps=60):
    """A fully deterministic stub run under a logical clock."""
    cfg = CoevoConfig(
        budget_s=budget_s if budget_s is not None else 2 * 24 * 3600,
        step_seconds=step_seconds, supervise_every=3, max_steps=max_steps,
        supervisor_mode=mode,
    )
    return build_run(run_dir=tmp_path / "run", solver=StubSolver(), config=cfg,
                     logical_clock=LogicalClock()).run()


# ---------------------------------------------------------------------------
# eval service: black box + feedback dial + query logging
# ---------------------------------------------------------------------------
def test_feedback_level_controls_disclosure():
    """The §3 dial: sparser levels expose strictly less, densest adds artifacts."""
    sol = _small()
    only = _service(FeedbackLevel.SCORE_ONLY).query(sol)
    feas = _service(FeedbackLevel.FEASIBLE_SCORE).query(sol)
    full = _service(FeedbackLevel.WITH_ARTIFACTS).query(sol)

    assert only.score is not None and only.feasible is None and only.artifacts is None
    assert feas.feasible is not None and feas.artifacts is None
    assert full.artifacts is not None and "reduced_chi2" in full.artifacts
    # the scalar is identical across levels — only disclosure changes.
    assert only.score == feas.score == full.score


def test_query_hook_logs_returned_shape():
    """Every query records what V *chose* to return (the arms-race evidence, §6)."""
    logged = []
    svc = _service()
    svc.on_query = lambda rec, res: logged.append(rec)
    svc.query(_deep())
    assert len(logged) == 1
    rec = logged[0]
    assert rec.returned_score is not None
    assert "score" in rec.returned_keys
    assert any(k.startswith("artifacts.") for k in rec.returned_keys)
    assert rec.payload_summary["n_modes"] == 18


def test_eval_client_is_query_only():
    """The Solver's handle exposes query and nothing that reads/writes V (§3)."""
    from coscientist.coevo.channels import EvalClient

    client = EvalClient(_service())
    assert hasattr(client, "query")
    for forbidden in ("install_verifier", "current_source", "set_feedback_level"):
        assert not hasattr(client, forbidden), f"black box leaked {forbidden}"


# ---------------------------------------------------------------------------
# supervisor: red-team detects the hole, hardening flips it, NO gate
# ---------------------------------------------------------------------------
def test_red_team_detects_v0_hole_then_clean_after_harden():
    svc = _service()
    sup = Supervisor(eval=svc)
    before = sup.red_team()
    assert before["fooled"] is True, "v0 should be fooled by the deep overfit"

    hardened = sup.observe_and_harden()
    assert hardened is True
    assert svc.current_version() == 1, "V must have evolved with no approval gate"

    after = sup.red_team()
    assert after["fooled"] is False, "the hardened V must no longer reward overfitting"


def test_none_mode_hardens_without_advisor():
    """Autonomy: with no advisor the Supervisor still hardens on probe evidence."""
    sup = Supervisor(eval=_service(), mode=SupervisorMode.NONE, advisor=None)
    assert sup.observe_and_harden() is True
    assert sup.hardened_count == 1


def test_rejecting_advisor_holds_the_harden():
    """proxy/human is an autonomy dial: a REJECT verdict holds V (but never a
    structural gate — that role is the Supervisor's own defence)."""
    from coscientist.demo.human_port import Decision, ReviewResponse

    class Rejecting:
        def review(self, req):
            return ReviewResponse(Decision.REJECT, dense_text="hold")

    svc = _service()
    sup = Supervisor(eval=svc, mode=SupervisorMode.PROXY, advisor=Rejecting())
    assert sup.observe_and_harden() is False
    assert svc.current_version() == 0


# ---------------------------------------------------------------------------
# gateway: the HTTP boundary a subprocess solver reaches us through (§6)
# ---------------------------------------------------------------------------
def _post(base, path, obj):
    req = urllib.request.Request(base + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.loads(r.read().decode())


def test_gateway_query_review_and_status():
    svc = _service()
    sup = Supervisor(eval=svc)
    with Gateway(eval=svc, supervisor=sup, deadline=Deadline(budget_s=100)) as gw:
        base = gw.base_url
        status = _get(base, "/status")
        assert status["verifier_version"] == 0
        assert status["deadline_remaining_s"] is not None

        q = _post(base, "/query", {"solution": _deep()})
        assert q["ok"] and q["score"] is not None and q["verifier_version"] == 0

        # a deep overfit sent to /review is judged gaming and hardens V with no gate
        rv = _post(base, "/review", {"solution": _deep()})
        assert rv["gaming"] is True
        assert rv["verifier_version"] == 1
    assert gw.queries >= 1 and gw.reviews == 1


def test_gateway_records_cost_at_the_boundary(tmp_path):
    """Cost is logged at the gateway, not from any agent self-report (§6)."""
    from coscientist.coevo.store import RunStore

    svc = _service()
    store = RunStore(tmp_path / "run")
    with Gateway(eval=svc, supervisor=Supervisor(eval=svc), store=store) as gw:
        _post(gw.base_url, "/query", {"solution": _small()})
    lines = (store.root / "cost.jsonl").read_text().strip().splitlines()
    assert any(json.loads(l)["kind"] == "eval_query" for l in lines)


# ---------------------------------------------------------------------------
# full stub co-evolution: games v0, supervisor hardens, solver retreats
# ---------------------------------------------------------------------------
def test_stub_run_games_then_retreats(tmp_path):
    run = _offline_run(tmp_path)
    s = run.summary()
    assert s["final_verifier_version"] >= 1, "V must have hardened at least once"
    assert s["verifier_hardenings"] >= 1
    # after re-baselining under the hardened V, the solver settles on a modest fit
    assert s["best_n_modes"] <= 6, "the deep overfit must be abandoned"
    assert s["best_real_score"] is not None and s["best_real_score"] > -1.0


def test_wall_clock_is_the_real_bound(tmp_path):
    """With a tiny budget and a huge step cap, the DEADLINE stops the run."""
    run = _offline_run(tmp_path, budget_s=600, step_seconds=60.0, max_steps=100000)
    assert run.deadline.expired()
    assert run.steps < 100000, "wall-clock, not the step cap, ended the run"


def test_storage_layout_written(tmp_path):
    run = _offline_run(tmp_path)
    root = Path(run.store.root)
    assert (root / "manifest.json").is_file()
    assert (root / "events.jsonl").is_file()
    assert (root / "eval" / "queries.jsonl").is_file()
    assert (root / "solver" / "trajectory.jsonl").is_file()
    # v0 and the hardened v1 are both persisted with rationale
    versions = sorted((root / "supervisor" / "verifier_versions").glob("v*.py"))
    assert [p.name for p in versions][:2] == ["v0.py", "v1.py"]
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["final_verifier_version"] >= 1


def test_no_gate_the_supervisor_always_changes_v(tmp_path):
    """Structural anti-collapse: even in none mode (no human/proxy) V changes."""
    run = _offline_run(tmp_path, mode=SupervisorMode.NONE)
    assert run.supervisor.hardened_count >= 1
    events = [json.loads(l) for l in
              (Path(run.store.root) / "events.jsonl").read_text().strip().splitlines()]
    assert any(e["kind"] == "supervisor_hardened" for e in events)


# ---------------------------------------------------------------------------
# codex path: availability guard (no codex binary required to run this test)
# ---------------------------------------------------------------------------
def test_codex_solver_clean_fails_without_binary(tmp_path, monkeypatch):
    """CodexSolver with no codex on PATH records unavailability, never raises."""
    from coscientist.coevo import codex_solver as cs
    from coscientist.coevo.solver import SolverContext
    from coscientist.coevo.channels import EvalClient, SupervisorChannel

    monkeypatch.setattr(cs.shutil, "which", lambda *_: None, raising=False)

    svc = _service()
    sup = Supervisor(eval=svc)
    ctx = SolverContext(
        solution_ws=tmp_path / "solver",
        eval=EvalClient(svc),
        supervisor=SupervisorChannel(sup),
        deadline=Deadline(budget_s=100),
    )
    solver = cs.CodexSolver(gateway=None)   # no gateway either
    solver.run(ctx)                          # must not raise
    payload, score = solver.best
    assert payload is None and score == float("-inf")
    assert "unavailable" in solver._note


# ---------------------------------------------------------------------------
# the general agent-system path — the SYSTEM builds its own evaluator.
# These run offline: no docker, no agent binary, no network. They exercise the
# transport (unix-socket shim round-trip) and the clean-fail guards, NOT a real
# container (that is the 30-min live run, done separately).
# ---------------------------------------------------------------------------
def test_control_socket_round_trips_the_shim_protocol(tmp_path):
    """The synchronous shim channel: an in-container client's newline-JSON request
    reaches the host handler and the response comes back — the exact transport
    container-eval/container-status ride on, tested without docker."""
    import socket

    from coscientist.coevo.container import ControlSocket

    seen = []

    def handler(cmd, arg):
        seen.append((cmd, arg))
        if cmd == "eval":
            return {"ok": True, "score": 1.5, "verifier_version": 0}
        if cmd == "status":
            return {"verifier_version": 0, "deadline_remaining_s": 42.0}
        return {"ok": False, "error": "unknown"}

    cs = ControlSocket(workdir=tmp_path, handler=handler).start()
    try:
        cs.seed_shims(tmp_path)
        # the three shims + client landed and are executable
        for name in ("container-eval", "container-ask-supervisor", "container-status",
                     "_control_client.py"):
            assert (tmp_path / name).is_file()
        assert (tmp_path / "container-eval").stat().st_mode & 0o111

        # speak the client's protocol directly against the host socket
        def call(obj):
            c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            c.connect(str(cs.host_path))
            c.sendall((json.dumps(obj) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = c.recv(4096)
                if not chunk:
                    break
                buf += chunk
            c.close()
            return json.loads(buf.decode())

        r = call({"cmd": "eval", "arg": {"solution": {"x": 1}}})
        assert r["ok"] and r["score"] == 1.5 and r["verifier_version"] == 0
        st = call({"cmd": "status", "arg": {}})
        assert st["deadline_remaining_s"] == 42.0
    finally:
        cs.stop()
    assert ("eval", {"solution": {"x": 1}}) in seen
    assert not cs.host_path.exists(), "socket file cleaned up on stop"


def test_control_socket_never_dies_on_a_bad_request(tmp_path):
    """A handler exception is turned into an error response, not a dead listener."""
    import socket

    from coscientist.coevo.container import ControlSocket

    def boom(cmd, arg):
        raise ValueError("kaboom")

    cs = ControlSocket(workdir=tmp_path, handler=boom).start()
    try:
        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c.connect(str(cs.host_path))
        c.sendall(b'{"cmd":"eval","arg":{}}\n')
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = c.recv(4096)
            if not chunk:
                break
            buf += chunk
        c.close()
        resp = json.loads(buf.decode())
        assert resp["ok"] is False and "kaboom" in resp["error"]
    finally:
        cs.stop()


def test_gateway_config_reads_codex_home_from_host(tmp_path, monkeypatch):
    """The codex auth carrier is discovered from ~/.codex (auth.json + model)."""
    from coscientist.coevo import container as C

    fake_home = tmp_path / ".codex"
    fake_home.mkdir()
    (fake_home / "auth.json").write_text('{"OPENAI_API_KEY": "sk-test"}')
    (fake_home / "config.toml").write_text(
        'model = "gpt-5.6-sol"\nmodel_reasoning_effort = "high"\n')
    monkeypatch.setattr(C.Path, "home", classmethod(lambda cls: tmp_path),
                        raising=False)

    gw = C.GatewayConfig.from_host()
    assert gw is not None
    assert gw.codex_home == fake_home.resolve()
    assert gw.model == "gpt-5.6-sol"
    assert gw.reasoning_effort == "high"
    assert 'model = "gpt-5.6-sol"' in gw.container_config_toml()

    # no auth.json -> None (clean, not a crash)
    (fake_home / "auth.json").unlink()
    assert C.GatewayConfig.from_host() is None


def test_agent_system_clean_fails_without_docker(tmp_path, monkeypatch):
    """AgentSystem.preflight with no docker raises a clean, explained error — never
    a stack trace (mirrors the codex availability guard)."""
    from coscientist.coevo import agent_system as A

    monkeypatch.setattr(A, "docker_unavailable", lambda: "docker CLI not found on PATH")

    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "problem.md").write_text("solve this")
    system = A.AgentSystem(raw_input_dir=tmp_path / "raw", run_dir=tmp_path / "run",
                           budget_s=60)
    import pytest
    with pytest.raises(A.AgentSystemUnavailable) as ei:
        system.run()
    assert "docker" in str(ei.value)


def test_agent_system_clean_fails_without_agent_binary(tmp_path, monkeypatch):
    """docker present but no codex binary -> clean AgentSystemUnavailable."""
    from coscientist.coevo import agent_system as A

    monkeypatch.setattr(A, "docker_unavailable", lambda: None)
    monkeypatch.setattr(A, "agent_elf_path", lambda: None)

    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "problem.md").write_text("solve this")
    system = A.AgentSystem(raw_input_dir=tmp_path / "raw", run_dir=tmp_path / "run",
                           budget_s=60)
    import pytest
    with pytest.raises(A.AgentSystemUnavailable) as ei:
        system.run()
    assert "agent binary" in str(ei.value)


def test_agent_bootstrap_recovers_authored_verifier_offline(tmp_path, monkeypatch):
    """The bootstrap boundary recovery: given a workspace an agent 'authored'
    (verifier.py + ctx/seed/probes), the orchestrator recovers it as V0, smoke-tests
    it on the seed, and wires a working black box — with the agent turn stubbed so
    no docker/model is touched. This pins the domain-agnostic bootstrap contract."""
    from coscientist.coevo import agent_system as A
    from coscientist.coevo.container import GatewayConfig
    from coscientist.coevo.eval_service import FeedbackLevel

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "problem.md").write_text("maximize the value field, honestly")

    system = A.AgentSystem(raw_input_dir=raw, run_dir=tmp_path / "run", budget_s=120,
                           feedback_level=FeedbackLevel.WITH_ARTIFACTS)
    system.gateway = GatewayConfig(codex_home=Path("/nonexistent/.codex"), model="m")
    system.agent_elf = Path("/nonexistent/codex")

    authored_verifier = (
        "def verify(payload, ctx):\n"
        "    try:\n"
        "        v = float(payload.get('value', 0))\n"
        "    except Exception:\n"
        "        return {'feasible': False, 'raw': -1e9, 'artifacts': {}}\n"
        "    cap = ctx.get('cap', 10)\n"
        "    feasible = 0 <= v <= cap\n"
        "    raw = v if feasible else -1e9\n"
        "    return {'feasible': feasible, 'raw': raw, 'artifacts': {'value': v}}\n"
    )

    # stub the one-shot agent: instead of a container, write the files an agent would.
    def fake_one_shot(ws, gateway, agent_elf, prompt, *, timeout_s, image,
                      disallowed_tools=None):
        ws = Path(ws)
        (ws / "verifier.py").write_text(authored_verifier)
        (ws / "ctx.json").write_text(json.dumps({"cap": 10}))
        (ws / "seed_solution.json").write_text(json.dumps({"value": 1}))
        (ws / "probes.json").write_text(json.dumps([
            {"description": "over the cap should score low", "solution": {"value": 9999}},
            {"description": "garbage", "solution": {"nope": True}},
        ]))
        (ws / "SOLVER_BRIEF.md").write_text("produce {\"value\": float in [0,cap]}")
        (ws / "BOOTSTRAP_DONE").write_text("done")
        from coscientist.coevo.container import AgentSession
        return AgentSession(ok=True, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(A, "one_shot_agent", fake_one_shot)

    bs = system.bootstrap()
    assert "def verify" in bs.verifier_src
    assert bs.ctx == {"cap": 10} and bs.seed_solution == {"value": 1}
    assert len(bs.probes) == 2

    # the recovered V0 is a live black box: a good solution scores high, the
    # over-cap probe scores low — exactly the red-team contract.
    good = system.eval_service.query({"value": 8})
    bad = system.eval_service.query({"value": 9999})
    assert good.ok and good.score == 8
    assert bad.score <= 0
    # v0 recorded at the boundary with agent origin
    v0 = (Path(system.run_dir) / "supervisor" / "verifier_versions" / "v0.py")
    assert v0.is_file() and "def verify" in v0.read_text()
