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

    (tmp_path / "raw").mkdir(exist_ok=True)
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

    (tmp_path / "raw").mkdir(exist_ok=True)
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
    raw.mkdir(exist_ok=True)
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


def _bootstrapped_system(tmp_path, monkeypatch, *, budget_s=120):
    """An AgentSystem past bootstrap (V0 live), with docker/codex stubbed out, ready
    to drive solve_and_evolve. Shared by the loop-guarantee tests below."""
    from coscientist.coevo import agent_system as A
    from coscientist.coevo.container import AgentSession, GatewayConfig
    from coscientist.coevo.eval_service import FeedbackLevel

    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    (raw / "problem.md").write_text("maximize the value field, honestly")

    system = A.AgentSystem(raw_input_dir=raw, run_dir=tmp_path / "run", budget_s=budget_s,
                           feedback_level=FeedbackLevel.WITH_ARTIFACTS)
    system.gateway = GatewayConfig(codex_home=Path("/nonexistent/.codex"), model="m")
    system.agent_elf = Path("/nonexistent/codex")

    verifier = (
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

    def fake_one_shot(ws, gateway, agent_elf, prompt, *, timeout_s, image,
                      disallowed_tools=None):
        ws = Path(ws)
        (ws / "verifier.py").write_text(verifier)
        (ws / "ctx.json").write_text(json.dumps({"cap": 10}))
        (ws / "seed_solution.json").write_text(json.dumps({"value": 1}))
        (ws / "probes.json").write_text(json.dumps([
            {"description": "over cap", "solution": {"value": 9999}}]))
        (ws / "SOLVER_BRIEF.md").write_text("produce {\"value\": float}")
        (ws / "BOOTSTRAP_DONE").write_text("done")
        return AgentSession(ok=True, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(A, "one_shot_agent", fake_one_shot)
    system.bootstrap()
    return system, A


def test_loop_guarantees_a_solve_turn_after_every_harden(tmp_path, monkeypatch):
    """The load-bearing co-evolution invariant: solve -> harden -> solve-AGAIN. When
    a review moves V, the loop MUST run one more Solver turn so it re-baselines under
    the new verifier. This is what the first Chowla run failed to do (harden fired at
    the deadline and the loop broke before re-solving). Driven fully offline with a
    fake container + fake harden — no docker, no codex."""
    from coscientist.coevo.container import AgentSession

    system, A = _bootstrapped_system(tmp_path, monkeypatch)

    turns_seen = []
    review_written = {"done": False}

    class FakeContainer:
        def __init__(self, *, workdir, gateway, agent_elf, image, **kw):
            self.ws = Path(workdir)
        def start(self):
            return self
        def stop(self):
            pass
        def exec_agent(self, prompt, *, timeout_s, **kw):
            turns_seen.append({"budget": timeout_s})
            # turn 1: leave a review request (the agent's reason for exiting).
            if not review_written["done"]:
                review_written["done"] = True
                (self.ws / "review_request.json").write_text(json.dumps(
                    {"solution": {"value": 8}, "question": "am I gaming V?"}))
                (self.ws / "solution_out.json").write_text(json.dumps({"value": 8}))
            else:
                # post-harden turn: re-baseline, write a fresh best, DON'T ask again.
                (self.ws / "solution_out.json").write_text(json.dumps({"value": 6}))
            return AgentSession(ok=True, returncode=0, stdout="", stderr="")

    class FakeControl:
        def __init__(self, *, workdir, handler, **kw):
            pass
        def seed_shims(self, ws):
            pass
        def start(self):
            return self
        def stop(self):
            pass

    monkeypatch.setattr(A, "DockerContainer", FakeContainer)
    monkeypatch.setattr(A, "ControlSocket", FakeControl)

    # a fake harden that always installs a stricter v1 (caps at 5) exactly once.
    stricter = (
        "def verify(payload, ctx):\n"
        "    try:\n"
        "        v = float(payload.get('value', 0))\n"
        "    except Exception:\n"
        "        return {'feasible': False, 'raw': -1e9, 'artifacts': {}}\n"
        "    feasible = 0 <= v <= 5\n"
        "    return {'feasible': feasible, 'raw': (v if feasible else -1e9),"
        " 'artifacts': {'value': v}}\n"
    )

    def fake_harden(best, sol_ws, *, trigger):
        if system.eval_service.current_version() == 0:
            system.eval_service.install_verifier(stricter, origin="agent", note="harden")
            system.hardenings += 1
            system.store.event("harden_start", trigger=trigger, remaining_s=0.0)
    monkeypatch.setattr(system, "_run_supervisor_harden", fake_harden)

    system.solve_and_evolve(max_turns=8)

    # exactly one harden happened, and a solve turn RAN after it.
    assert system.hardenings == 1
    assert system._post_harden_solves >= 1, "no re-baseline solve turn after harden"
    assert len(turns_seen) >= 2, "loop stopped before the post-harden solve turn"
    # V evolved and the run finalized recording the post-harden solve.
    manifest = json.loads((Path(system.run_dir) / "manifest.json").read_text())
    assert manifest["final_verifier_version"] == 1
    assert manifest["post_harden_solves"] >= 1


def test_wall_clock_turn_still_drives_a_harden(tmp_path, monkeypatch):
    """The smoke_peak gap: a NORMAL turn that hits its wall-clock cap (codex just kept
    working, never left a review_request) must STILL let the orchestrator drive the
    evolve step. The old loop `break`ed on a wall-clock hit before _proactive_supervise
    could run, so hardenings stayed 0 and the second loop never engaged. Here the fake
    agent never asks for review and always times out; the loop must harden anyway."""
    from coscientist.coevo.container import AgentSession

    system, A = _bootstrapped_system(tmp_path, monkeypatch)

    turns = {"n": 0}

    class FakeContainer:
        def __init__(self, *, workdir, gateway, agent_elf, image, **kw):
            self.ws = Path(workdir)
        def start(self):
            return self
        def stop(self):
            pass
        def exec_agent(self, prompt, *, timeout_s, **kw):
            turns["n"] += 1
            # never write review_request; always report a wall-clock timeout.
            (self.ws / "solution_out.json").write_text(json.dumps({"value": 7}))
            return AgentSession(ok=True, returncode=124, stdout="", stderr="",
                                note="agent hit the wall-clock budget")

    class FakeControl:
        def __init__(self, *, workdir, handler, **kw):
            pass
        def seed_shims(self, ws):
            pass
        def start(self):
            return self
        def stop(self):
            pass

    monkeypatch.setattr(A, "DockerContainer", FakeContainer)
    monkeypatch.setattr(A, "ControlSocket", FakeControl)

    stricter = (
        "def verify(payload, ctx):\n"
        "    try:\n"
        "        v = float(payload.get('value', 0))\n"
        "    except Exception:\n"
        "        return {'feasible': False, 'raw': -1e9, 'artifacts': {}}\n"
        "    feasible = 0 <= v <= 5\n"
        "    return {'feasible': feasible, 'raw': (v if feasible else -1e9),"
        " 'artifacts': {'value': v}}\n"
    )
    hardened = {"n": 0}

    def fake_harden(best, sol_ws, *, trigger):
        hardened["n"] += 1
        # harden exactly once so the loop terminates (owed solve, then stable V).
        if system.eval_service.current_version() == 0:
            system.eval_service.install_verifier(stricter, origin="agent", note="harden")
            system.hardenings += 1
    monkeypatch.setattr(system, "_run_supervisor_harden", fake_harden)

    system.solve_and_evolve(max_turns=4)

    # the wall-clock turn drove a harden even though the agent never asked for review.
    assert hardened["n"] >= 1, "wall-clock turn did not drive a harden (the smoke_peak bug)"
    assert system.hardenings == 1
    assert system._post_harden_solves >= 1, "no re-baseline solve after the wall-clock harden"


def test_last_review_before_deadline_still_hardens_and_resolves(tmp_path, monkeypatch):
    """Even when the review fires with almost no budget left, the reserved slices let
    the harden complete AND the owed post-harden solve run — the exact deadline-edge
    case the first Chowla run hit. We force it by exhausting the clock on turn 1."""
    from coscientist.coevo.container import AgentSession
    from coscientist.coevo.budget import Deadline

    system, A = _bootstrapped_system(tmp_path, monkeypatch, budget_s=1000)

    # a hand-driven clock: turn 1 consumes almost the whole budget before the review.
    clock = {"t": 0.0}
    system.deadline = Deadline(budget_s=1000, clock=lambda: clock["t"])

    calls = {"n": 0}

    class FakeContainer:
        def __init__(self, *, workdir, gateway, agent_elf, image, **kw):
            self.ws = Path(workdir)
        def start(self):
            return self
        def stop(self):
            pass
        def exec_agent(self, prompt, *, timeout_s, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                clock["t"] = 995.0   # burn the budget: only 5s nominal remaining
                (self.ws / "review_request.json").write_text(json.dumps(
                    {"solution": {"value": 8}, "question": "?"}))
                (self.ws / "solution_out.json").write_text(json.dumps({"value": 8}))
            else:
                (self.ws / "solution_out.json").write_text(json.dumps({"value": 4}))
            return AgentSession(ok=True, returncode=0, stdout="", stderr="")

    class FakeControl:
        def __init__(self, *, workdir, handler, **kw):
            pass
        def seed_shims(self, ws):
            pass
        def start(self):
            return self
        def stop(self):
            pass

    monkeypatch.setattr(A, "DockerContainer", FakeContainer)
    monkeypatch.setattr(A, "ControlSocket", FakeControl)

    hardened = {"n": 0}

    def fake_harden(best, sol_ws, *, trigger):
        # harden must be REACHED despite the near-zero remaining budget.
        hardened["n"] += 1
        if system.eval_service.current_version() == 0:
            system.eval_service.install_verifier(
                system.eval_service.current_source(), origin="agent", note="harden")
            system.hardenings += 1
    monkeypatch.setattr(system, "_run_supervisor_harden", fake_harden)

    system.solve_and_evolve(max_turns=8)

    assert hardened["n"] >= 1, "harden was skipped at the deadline edge (the old bug)"
    assert system._post_harden_solves >= 1, "no post-harden solve at the deadline edge"


def test_resume_rebuilds_verifier_chain_and_best_from_disk(tmp_path):
    """8h runs must survive a crash. A resume reconstructs the evolving evaluator, the
    hardening count, and the best-so-far entirely from runs/<id>/ — no re-bootstrap and
    no in-memory state. We lay down a realistic tree (v0 authored, v1 hardened, one
    candidate best under v0 that becomes infeasible under v1) and assert the resume
    lands on v1, counts one hardening, and re-scores the best under the CURRENT V."""
    from coscientist.coevo import agent_system as A
    from coscientist.coevo.eval_service import FeedbackLevel

    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    (raw / "problem.md").write_text("maximize value up to a cap")
    run_dir = tmp_path / "run"

    # v0: caps at 10.  v1 (a harden): caps at 5 — so a value=8 best goes infeasible.
    def verifier(cap):
        return (
            "def verify(payload, ctx):\n"
            "    try:\n"
            "        v = float(payload.get('value', 0))\n"
            "    except Exception:\n"
            "        return {'feasible': False, 'raw': -1e9, 'artifacts': {}}\n"
            f"    feasible = 0 <= v <= {cap}\n"
            "    return {'feasible': feasible, 'raw': (v if feasible else -1e9),"
            " 'artifacts': {}}\n"
        )

    # bootstrap_ws (the instance + authored v0), as bootstrap() would have left it.
    bws = run_dir / "bootstrap_ws"
    bws.mkdir(parents=True)
    (bws / "verifier.py").write_text(verifier(10))
    (bws / "ctx.json").write_text(json.dumps({}))
    (bws / "seed_solution.json").write_text(json.dumps({"value": 1}))
    (bws / "probes.json").write_text(json.dumps(
        [{"description": "over cap", "solution": {"value": 9999}}]))
    (bws / "BOOTSTRAP_DONE").write_text("done")

    # the recorded version chain: v0 (bootstrap) then v1 (harden).
    vdir = run_dir / "supervisor" / "verifier_versions"
    vdir.mkdir(parents=True)
    (vdir / "v0.py").write_text(verifier(10))
    (vdir / "v1.py").write_text(verifier(5))
    (run_dir / "supervisor").mkdir(exist_ok=True)
    (run_dir / "supervisor" / "versions.jsonl").write_text(
        json.dumps({"version": 0, "origin": "agent", "note": "bootstrap"}) + "\n"
        + json.dumps({"version": 1, "origin": "agent", "note": "harden"}) + "\n")

    # a couple of prior events (one review already handled) + a best candidate under v0.
    (run_dir / "events.jsonl").write_text(
        json.dumps({"t": 1.0, "kind": "review_request", "question": "?"}) + "\n")
    cdir = run_dir / "solver" / "candidates"
    cdir.mkdir(parents=True)
    (cdir / "cand_00000.json").write_text(json.dumps(
        {"id": "cand_00000", "payload": {"value": 8}, "score": 8.0}))  # good under v0
    (cdir / "cand_00001.json").write_text(json.dumps(
        {"id": "cand_00001", "payload": {"value": 4}, "score": 4.0}))  # good under v1
    (run_dir / "manifest.json").write_text(json.dumps(
        {"mode": "agent_system", "best_solution": {"value": 8}, "best_score": 8.0}))

    system = A.AgentSystem(raw_input_dir=raw, run_dir=run_dir, budget_s=60,
                           feedback_level=FeedbackLevel.WITH_ARTIFACTS)
    assert system.can_resume()
    system._resume_from_disk()

    # chain rebuilt to v1; exactly one hardening; the earlier review counted.
    assert system.eval_service.current_version() == 1
    assert system.hardenings == 1
    assert system.reviews_handled == 1
    # best re-scored under CURRENT V: value=8 is infeasible at cap 5, so the honest
    # best is value=4 (=4.0), NOT the stale 8.0 the manifest recorded under v0.
    assert system._best_payload == {"value": 4}
    assert system._best_score == 4.0


def test_can_resume_false_before_bootstrap(tmp_path):
    """A fresh run_dir with no completed bootstrap must NOT be treated as resumable
    (else --resume on a first launch would silently skip authoring the evaluator)."""
    from coscientist.coevo import agent_system as A
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    (raw / "p.md").write_text("x")
    system = A.AgentSystem(raw_input_dir=raw, run_dir=tmp_path / "run", budget_s=60)
    assert system.can_resume() is False


# ---------------------------------------------------------------------------
# parallel launcher: unique run-ids, resume-friendly spawn, disk-based status
# ---------------------------------------------------------------------------
def test_launcher_assigns_unique_run_ids_and_resume(tmp_path, monkeypatch):
    """start() spawns one detached child per input with a distinct run_id, and every
    child launches with --resume so a re-run continues from disk. We capture the argv
    instead of really spawning (no codex/docker), and feed two inputs whose basenames
    collide to prove run-ids are de-duplicated."""
    from coscientist.coevo import launcher as L

    calls = []

    class FakePopen:
        def __init__(self, argv, **kw):
            calls.append({"argv": argv, "kw": kw})
            self.pid = 1000 + len(calls)

    monkeypatch.setattr(L.subprocess, "Popen", FakePopen)

    a = tmp_path / "dirA" / "chowla"; a.mkdir(parents=True)
    b = tmp_path / "dirB" / "chowla"; b.mkdir(parents=True)   # same basename
    batch = L.Batch("case2026", runs_dir=tmp_path / "runs")
    specs = batch.start([a, b], hours=8.0)

    assert len(specs) == 2
    ids = [s.run_id for s in specs]
    assert len(set(ids)) == 2, f"run-ids collided: {ids}"
    assert ids[0] == "case2026__chowla" and ids[1] == "case2026__chowla_2"
    for c in calls:
        assert "--resume" in c["argv"], "every child must launch resume-able"
        assert "--solver" in c["argv"] and "codex" in c["argv"]
        # 8h budget threaded through as seconds
        assert str(8.0 * 3600.0) in c["argv"]
        assert c["kw"].get("start_new_session") is True, "child must be detached"
    # the batch manifest records pids + run-ids for later monitoring.
    man = json.loads((tmp_path / "runs" / "case2026" / "batch.json").read_text())
    assert {r["run_id"] for r in man["runs"]} == set(ids)
    assert all(isinstance(r["pid"], int) for r in man["runs"])


def test_launcher_status_reads_progress_from_disk(tmp_path):
    """status() renders one row per run purely from each run's own tree — so it works
    on a detached run, after a crash, or from another shell. We synthesize a run that
    bootstrapped, took two turns (one post-harden) and stopped with one hardening."""
    from coscientist.coevo import launcher as L

    runs = tmp_path / "runs"
    batch = L.Batch("b", runs_dir=runs)
    run_id = "b__demo"
    rd = runs / run_id
    rd.mkdir(parents=True)
    (rd / "manifest.json").write_text(json.dumps({
        "final_verifier_version": 1, "verifier_hardenings": 1,
        "post_harden_solves": 1, "best_score": -0.42}))
    (rd / "events.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"kind": "run_start"}, {"kind": "bootstrap_done"},
        {"kind": "solver_turn_start", "post_harden": False},
        {"kind": "harden_start"},
        {"kind": "solver_turn_start", "post_harden": True},
        {"kind": "run_stop", "hardenings": 1, "post_harden_solves": 1,
         "best_score": -0.42},
    ]) + "\n")
    # a batch manifest pointing at a dead pid (0 => not alive).
    (runs / "b").mkdir(parents=True, exist_ok=True)
    (runs / "b" / "batch.json").write_text(json.dumps(
        {"batch": "b", "runs": [{"run_id": run_id, "pid": 0}]}))

    rows = batch.status()
    assert len(rows) == 1
    r = rows[0]
    assert r["bootstrapped"] is True
    assert r["solver_turns"] == 2
    assert r["post_harden_solves"] == 1
    assert r["hardenings"] == 1
    assert r["final_verifier_version"] == 1
    assert r["best_score"] == -0.42
    assert r["stopped"] is True
    assert r["alive"] is False

