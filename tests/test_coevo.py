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


def _offline_run(tmp_path: Path, *, mode=SupervisorMode.NO_HUMAN_NO_PROXY, budget_s=None,
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
    sup = Supervisor(eval=_service(), mode=SupervisorMode.NO_HUMAN_NO_PROXY, advisor=None)
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
    run = _offline_run(tmp_path, mode=SupervisorMode.NO_HUMAN_NO_PROXY)
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
    raw.mkdir(parents=True, exist_ok=True)
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
    raw.mkdir(parents=True, exist_ok=True)
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
            self.workdir = workdir
        def seed_shims(self, ws):
            pass
        def start(self):
            return self
        def stop(self):
            pass
        @property
        def needs_explicit_mount(self):
            return False
        @property
        def host_path(self):
            from pathlib import Path as _P
            return _P(self.workdir) / ".control.sock"

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

    def fake_harden(best, sol_ws, *, trigger, plateau=False):
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
            self.workdir = workdir
        def seed_shims(self, ws):
            pass
        def start(self):
            return self
        def stop(self):
            pass
        @property
        def needs_explicit_mount(self):
            return False
        @property
        def host_path(self):
            from pathlib import Path as _P
            return _P(self.workdir) / ".control.sock"

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

    def fake_harden(best, sol_ws, *, trigger, plateau=False):
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


def test_voluntary_exit_does_not_stop_run(tmp_path, monkeypatch):
    """Wall-clock is the ONLY stop signal. A solver that exits VOLUNTARILY (no
    review_request, note="" — not a wall-clock hit) with a STABLE verifier used to end
    the whole run after turn 1. Now the loop must keep giving turns until max_turns (the
    deadline stand-in here), and each turn after the first must carry the KEEP PUSHING
    addendum, since the Supervisor looked and deliberately left the game unchanged."""
    from coscientist.coevo.container import AgentSession

    system, A = _bootstrapped_system(tmp_path, monkeypatch)

    prompts = []

    class FakeContainer:
        def __init__(self, *, workdir, gateway, agent_elf, image, **kw):
            self.ws = Path(workdir)
        def start(self):
            return self
        def stop(self):
            pass
        def exec_agent(self, prompt, *, timeout_s, **kw):
            prompts.append(prompt)
            # voluntary exit: never ask for review, report a clean stop (NOT a timeout).
            (self.ws / "solution_out.json").write_text(json.dumps({"value": 3}))
            return AgentSession(ok=True, returncode=0, stdout="", stderr="", note="")

    class FakeControl:
        def __init__(self, *, workdir, handler, **kw):
            self.workdir = workdir
        def seed_shims(self, ws):
            pass
        def start(self):
            return self
        def stop(self):
            pass
        @property
        def needs_explicit_mount(self):
            return False
        @property
        def host_path(self):
            from pathlib import Path as _P
            return _P(self.workdir) / ".control.sock"

    monkeypatch.setattr(A, "DockerContainer", FakeContainer)
    monkeypatch.setattr(A, "ControlSocket", FakeControl)
    # Supervisor looks every normal turn but never moves V (stable game).
    monkeypatch.setattr(system, "_run_supervisor_harden",
                        lambda best, sol_ws, *, trigger: None)

    system.solve_and_evolve(max_turns=5)

    # the old loop broke after turn 1; now all 5 turns run, stopped only by max_turns.
    assert len(prompts) == 5, "voluntary exit stopped the run early"
    assert system.hardenings == 0
    assert system._post_harden_solves == 0
    # turn 1 has no push; every turn after a stable-V voluntary exit carries KEEP PUSHING.
    assert "KEEP PUSHING" not in prompts[0]
    assert all("KEEP PUSHING" in p for p in prompts[1:]), "repush not applied after exit"


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
            self.workdir = workdir
        def seed_shims(self, ws):
            pass
        def start(self):
            return self
        def stop(self):
            pass
        @property
        def needs_explicit_mount(self):
            return False
        @property
        def host_path(self):
            from pathlib import Path as _P
            return _P(self.workdir) / ".control.sock"

    monkeypatch.setattr(A, "DockerContainer", FakeContainer)
    monkeypatch.setattr(A, "ControlSocket", FakeControl)

    hardened = {"n": 0}

    def fake_harden(best, sol_ws, *, trigger, plateau=False):
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


# ---------------------------------------------------------------------------
# free-form evaluator evolution: authored feedback, host-side creds, plateau→proof
# modality switch, direction-neutral separation invariant, resume, checker ingest,
# and the GPU/image knob. All offline: no docker, no codex, no network, no GPU —
# LLM creds are just env strings the host-side subprocess reads, the llm_client is
# never actually called, and the GPU is asserted only in argv/constructor wiring.
# ---------------------------------------------------------------------------
def _authored_bootstrap(tmp_path, monkeypatch, *, verifier, ctx=None, seed=None,
                        probes=None, feedback_src=None, solver_env=None,
                        checker_files=None, llm_config=None, budget_s=120):
    """Bootstrap an AgentSystem with an agent-authored evaluation, docker/codex stubbed.

    Unlike ``_bootstrapped_system`` this lets each test dictate the verifier and the
    OPTIONAL free-form pieces (feedback.py, solver_env.json, a bundled checker/), so a
    single helper drives the feedback / creds / checker-ingest / knob tests. Returns
    ``(system, A, bs)``."""
    from coscientist.coevo import agent_system as A
    from coscientist.coevo.container import AgentSession, GatewayConfig
    from coscientist.coevo.eval_service import FeedbackLevel

    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "problem.md").write_text("maximize the value field, honestly")

    ctx = {} if ctx is None else ctx
    seed = {"value": 1} if seed is None else seed
    probes = ([{"description": "over cap", "solution": {"value": 9999}}]
              if probes is None else probes)

    kw = {}
    if llm_config is not None:
        kw["llm_config"] = llm_config
    system = A.AgentSystem(raw_input_dir=raw, run_dir=tmp_path / "run", budget_s=budget_s,
                           feedback_level=FeedbackLevel.WITH_ARTIFACTS, **kw)
    system.gateway = GatewayConfig(codex_home=Path("/nonexistent/.codex"), model="m")
    system.agent_elf = Path("/nonexistent/codex")

    def fake_one_shot(ws, gateway, agent_elf, prompt, *, timeout_s, image,
                      disallowed_tools=None):
        ws = Path(ws)
        (ws / "verifier.py").write_text(verifier)
        (ws / "ctx.json").write_text(json.dumps(ctx))
        (ws / "seed_solution.json").write_text(json.dumps(seed))
        (ws / "probes.json").write_text(json.dumps(probes))
        (ws / "SOLVER_BRIEF.md").write_text("produce a candidate")
        if feedback_src is not None:
            (ws / "feedback.py").write_text(feedback_src)
        if solver_env is not None:
            (ws / "solver_env.json").write_text(json.dumps(solver_env))
        if checker_files:
            cdir = ws / "checker"
            cdir.mkdir(exist_ok=True)
            for name, content in checker_files.items():
                (cdir / name).write_text(content)
        (ws / "BOOTSTRAP_DONE").write_text("done")
        return AgentSession(ok=True, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(A, "one_shot_agent", fake_one_shot)
    bs = system.bootstrap()
    return system, A, bs


# A cap-10 verifier reused across several tests (value in [0,cap] scores = value).
_CAP_VERIFIER = (
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

# A proof-representation verifier: payload is a structural argument (a string). It is
# an LLM-verifier that DEGRADES GRACEFULLY when no creds are present (llm_client
# .available() is False here), scoring rigor by the argument's substance instead of
# crashing. Longer/nonempty argument = more progress; empty/garbage = degenerate low.
_PROOF_VERIFIER = (
    "import llm_client\n"
    "def verify(payload, ctx):\n"
    "    arg = payload.get('argument', '')\n"
    "    if not isinstance(arg, str) or not arg.strip():\n"
    "        return {'feasible': False, 'raw': -1e9, 'artifacts': {}}\n"
    "    # would consult an LLM if creds were present; degrade to a stdlib proxy.\n"
    "    if llm_client.available():\n"
    "        pass\n"
    "    score = float(len(arg.strip()))\n"
    "    return {'feasible': True, 'raw': score, 'artifacts': {'len': len(arg)}}\n"
)


def test_authored_feedback_is_versioned_and_shapes_disclosure(tmp_path, monkeypatch):
    """A bootstrap that ships feedback.py: it is recovered, versioned as a v0.feedback.py
    sibling, and its dict shapes what the Solver sees (detail/artifacts) while the
    verifier's score stays authoritative. Absent feedback.py -> the enum path, unchanged."""
    feedback_src = (
        "def feedback(payload, ctx, verify_result, history):\n"
        "    return {'detail': 'hi', 'artifacts': {'k': 1}}\n"
    )
    system, A, bs = _authored_bootstrap(tmp_path, monkeypatch,
                                        verifier=_CAP_VERIFIER, feedback_src=feedback_src)
    assert bs.feedback_src is not None and "def feedback" in bs.feedback_src
    vfb = tmp_path / "run" / "supervisor" / "verifier_versions" / "v0.feedback.py"
    assert vfb.is_file(), "authored feedback must be versioned beside v0.py"

    r = system.eval_service.query({"value": 8})
    assert r.ok and r.score == 8, "verifier score stays authoritative"
    assert r.detail == "hi"
    assert r.artifacts == {"k": 1}

    # no feedback.py -> the frozen disclosure (no detail, verifier artifacts only).
    plain, _, _ = _authored_bootstrap(tmp_path / "b", monkeypatch, verifier=_CAP_VERIFIER)
    assert not (tmp_path / "b" / "run" / "supervisor" / "verifier_versions"
                / "v0.feedback.py").is_file()
    pr = plain.eval_service.query({"value": 8})
    assert pr.detail == "" and pr.artifacts == {"value": 8.0}


def test_validate_evaluation_accepts_expose_more_rejects_collapse(tmp_path, monkeypatch):
    """The separation invariant is direction-neutral: an unchanged verifier with a
    RICHER feedback module (expose-more) validates; a verifier that COLLAPSES good vs
    degenerate is rejected."""
    system, A, _ = _authored_bootstrap(tmp_path, monkeypatch, verifier=_CAP_VERIFIER)
    svc = system.eval_service

    richer = (
        "def feedback(payload, ctx, verify_result, history):\n"
        "    return {'detail': 'try a bigger value', 'artifacts': verify_result}\n"
    )
    # expose-more: same verifier, add guiding feedback, probes unchanged -> accepted.
    err = svc.validate_evaluation(
        verify_src=svc.current_source(), feedback_src=richer,
        seed=system._seed, reference=system._seed, probes=system._probes,
        ctx=system._ctx)
    assert err is None, f"expose-more must be accepted, got: {err}"

    # a verifier that scores everything the same collapses the separation -> rejected.
    collapse = (
        "def verify(payload, ctx):\n"
        "    return {'feasible': True, 'raw': 5.0, 'artifacts': {}}\n"
    )
    err2 = svc.validate_evaluation(
        verify_src=collapse, feedback_src=None,
        seed=system._seed, reference=system._seed, probes=system._probes,
        ctx=system._ctx)
    assert err2 is not None and "separation" in err2


def test_llm_creds_reach_eval_subprocess_never_the_container(tmp_path, monkeypatch):
    """Host-side cred boundary: LLM creds are injected into the Evaluator subprocess
    (an env-sensing verifier can see them) but NEVER into the Solver container's argv."""
    env_verifier = (
        "import os\n"
        "def verify(payload, ctx):\n"
        "    if os.environ.get('LLM_API_KEY') == 'sk-x':\n"
        "        return {'feasible': True, 'raw': 1.0, 'artifacts': {}}\n"
        "    return {'feasible': False, 'raw': -1e9, 'artifacts': {}}\n"
    )
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    system, A, _ = _authored_bootstrap(
        tmp_path, monkeypatch, verifier=env_verifier, seed={"value": 1},
        llm_config={"api_key": "sk-x", "base_url": "http://x", "model": "m"})

    # (a) the host-side subprocess sees the creds -> score 1.
    assert system.eval_service.query({"value": 1}).score == 1.0
    # (b) WITHOUT the injected env, the same verifier can't see them -> infeasible.
    #     (proves it's the injection, not an ambient env var, doing the work.)
    assert system.evaluator.run({"value": 1}, system._ctx).feasible is False

    # (c) the container argv carries NO creds — even with a GPU requested. Drive the
    #     real DockerContainer.start() with docker stubbed so we capture its argv.
    from coscientist.coevo import container as C
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text('{"OPENAI_API_KEY": "irrelevant"}')
    gw = C.GatewayConfig(codex_home=codex_home, model="m")

    captured = {}

    class _Proc:
        returncode = 0
        stdout = "cid123"
        stderr = ""

    def fake_run(argv, **kw):
        captured.setdefault("argv", argv)
        return _Proc()

    monkeypatch.setattr(C.subprocess, "run", fake_run)
    cont = C.DockerContainer(workdir=tmp_path / "cw", gateway=gw,
                             agent_elf=Path("/nonexistent/codex"), image="img:x",
                             gpus="1")
    cont.start()
    argv = captured["argv"]
    joined = " ".join(argv)
    assert "sk-x" not in joined and "LLM_API_KEY" not in joined
    assert "--gpus" in argv and argv[argv.index("--gpus") + 1] == "1"
    assert "img:x" in argv
    # the only -e envs are the container plumbing, never a cred.
    envs = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
    assert all(e.split("=")[0] in ("HOME", "CODEX_HOME", "CONTROL_SOCK") for e in envs)
    cont.stop()


def test_supervisor_can_switch_construction_to_proof(tmp_path, monkeypatch):
    """The motivating arc: the Supervisor may REFRAME the game to proof-guidance. The
    full menu (incl. mode_switch) is ALWAYS in the harden prompt — no orchestrator
    "plateau" gate. Driven through the real loop with a fake harden agent that writes
    mode_switch.json + a proof verifier + a new contract; the switch must install,
    _mode flip to 'proof', the brief be rewritten, and the owed re-baseline solve run
    in the NEW representation."""
    from coscientist.coevo.container import AgentSession

    system, A = _bootstrapped_system(tmp_path, monkeypatch)

    # a flat trajectory is just CONTEXT now (progress.json), not a trigger.
    system._score_trajectory = [3.0, 3.0, 3.0]

    class FakeContainer:
        def __init__(self, *, workdir, gateway, agent_elf, image, gpus=None, **kw):
            self.ws = Path(workdir)
            self.turns = 0
        def start(self):
            return self
        def stop(self):
            pass
        def exec_agent(self, prompt, *, timeout_s, **kw):
            self.turns += 1
            if self.turns == 1:
                (self.ws / "review_request.json").write_text(json.dumps(
                    {"solution": {"value": 3}, "question": "am I plateaued?"}))
                (self.ws / "solution_out.json").write_text(json.dumps({"value": 3}))
            else:
                # post-switch turns speak the PROOF representation.
                (self.ws / "solution_out.json").write_text(
                    json.dumps({"argument": "induction: base case then step"}))
            return AgentSession(ok=True, returncode=0, stdout="", stderr="")

    class FakeControl:
        def __init__(self, *, workdir, handler, **kw):
            self.workdir = workdir
        def seed_shims(self, ws):
            pass
        def start(self):
            return self
        def stop(self):
            pass
        @property
        def needs_explicit_mount(self):
            return False
        @property
        def host_path(self):
            from pathlib import Path as _P
            return _P(self.workdir) / ".control.sock"

    monkeypatch.setattr(A, "DockerContainer", FakeContainer)
    monkeypatch.setattr(A, "ControlSocket", FakeControl)

    harden_calls = {"n": 0}

    def fake_harden_agent(ws, gateway, agent_elf, prompt, *, timeout_s, image,
                          disallowed_tools=None):
        ws = Path(ws)
        harden_calls["n"] += 1
        (ws / "verdict.json").write_text(json.dumps(
            {"gaming": False, "reasoning": "construction exhausted; switch to proof"}))
        if harden_calls["n"] == 1:
            # the full menu — incl. the reframe mechanism — is ALWAYS present.
            assert "mode_switch.json" in prompt, "switch mechanism not in harden prompt"
            (ws / "mode_switch.json").write_text(json.dumps(
                {"switch": True, "to_mode": "proof", "reasoning": "scalar exhausted"}))
            (ws / "verifier.py").write_text(_PROOF_VERIFIER)
            (ws / "seed_solution.json").write_text(json.dumps(
                {"argument": "a minimal honest argument"}))
            (ws / "probes.json").write_text(json.dumps([
                {"description": "empty", "solution": {"argument": ""}},
                {"description": "circular", "solution": {"nope": 1}}]))
            (ws / "SOLVER_BRIEF.md").write_text("PROOF MODE: produce {\"argument\": str}")
        (ws / "HARDEN_DONE").write_text("done")
        return AgentSession(ok=True, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(A, "one_shot_agent", fake_harden_agent)

    system.solve_and_evolve(max_turns=3)

    assert system._mode == "proof", "the plateau harden must have switched the game"
    assert system.hardenings == 1
    assert system._post_harden_solves >= 1, "no re-baseline solve after the switch"
    # the solver brief (single source of truth) was rewritten to the proof game.
    assert "PROOF MODE" in (system.run_dir / "bootstrap_ws" / "SOLVER_BRIEF.md").read_text()
    assert "PROOF MODE" in (system.run_dir / "solver_ws" / "PROBLEM_BRIEF.md").read_text()
    # the in-memory contract is the new representation.
    assert system._seed == {"argument": "a minimal honest argument"}
    events = [json.loads(l) for l in
              (system.run_dir / "events.jsonl").read_text().strip().splitlines()]
    assert any(e["kind"] == "mode_switch" and e.get("to_mode") == "proof" for e in events)


def test_validate_evaluation_accepts_a_separating_modality_change(tmp_path, monkeypatch):
    """The invariant accepts a full representation change (proof game) as long as a
    genuine argument outscores the degenerate probes; a verifier that ties them fails."""
    system, A, _ = _authored_bootstrap(tmp_path, monkeypatch, verifier=_CAP_VERIFIER)
    svc = system.eval_service

    proof_seed = {"argument": "base case, then the inductive step closes it"}
    proof_probes = [
        {"description": "empty", "solution": {"argument": ""}},
        {"description": "hand-wavy", "solution": {"nope": True}},
    ]
    err = svc.validate_evaluation(
        verify_src=_PROOF_VERIFIER, feedback_src=None,
        seed=proof_seed, reference=proof_seed, probes=proof_probes, ctx={})
    assert err is None, f"a separating modality change must be accepted, got: {err}"

    # a proof verifier that scores every argument the same collapses separation.
    flat_proof = (
        "def verify(payload, ctx):\n"
        "    return {'feasible': True, 'raw': 7.0, 'artifacts': {}}\n"
    )
    err2 = svc.validate_evaluation(
        verify_src=flat_proof, feedback_src=None,
        seed=proof_seed, reference=proof_seed,
        probes=[{"description": "x", "solution": {"argument": "y"}}], ctx={})
    assert err2 is not None and "separation" in err2


def test_resume_recovers_mode_and_feedback_source(tmp_path):
    """A resume must rebuild BOTH the game mode (from the last mode_switch event) and
    each version's agent-authored feedback module (from its v{n}.feedback.py sibling),
    so a crashed proof-mode run continues with the right game and disclosure."""
    from coscientist.coevo import agent_system as A
    from coscientist.coevo.eval_service import FeedbackLevel

    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    (raw / "problem.md").write_text("x")
    run_dir = tmp_path / "run"

    def cap_verifier(cap):
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

    bws = run_dir / "bootstrap_ws"
    bws.mkdir(parents=True)
    (bws / "verifier.py").write_text(cap_verifier(10))
    (bws / "ctx.json").write_text(json.dumps({}))
    (bws / "seed_solution.json").write_text(json.dumps({"value": 1}))
    (bws / "probes.json").write_text(json.dumps(
        [{"description": "over cap", "solution": {"value": 9999}}]))
    (bws / "BOOTSTRAP_DONE").write_text("done")

    vdir = run_dir / "supervisor" / "verifier_versions"
    vdir.mkdir(parents=True)
    (vdir / "v0.py").write_text(cap_verifier(10))
    (vdir / "v1.py").write_text(cap_verifier(5))
    # v1 shipped an authored feedback module; v0 did not.
    (vdir / "v1.feedback.py").write_text(
        "def feedback(payload, ctx, verify_result, history):\n"
        "    return {'detail': 'guidance-42', 'artifacts': {'h': len(history)}}\n")
    (run_dir / "supervisor" / "versions.jsonl").write_text(
        json.dumps({"version": 0, "origin": "agent", "note": "bootstrap",
                    "has_feedback": False}) + "\n"
        + json.dumps({"version": 1, "origin": "agent", "note": "harden [mode_switch]",
                      "has_feedback": True}) + "\n")

    # events include the mode_switch that must be recovered.
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"t": 1.0, "kind": "review_request", "question": "?"},
        {"t": 2.0, "kind": "mode_switch", "to_mode": "proof", "reasoning": "switch"},
    ]) + "\n")
    (run_dir / "manifest.json").write_text(json.dumps(
        {"mode": "agent_system", "best_solution": {"value": 4}, "best_score": 4.0}))

    system = A.AgentSystem(raw_input_dir=raw, run_dir=run_dir, budget_s=60,
                           feedback_level=FeedbackLevel.WITH_ARTIFACTS)
    assert system.can_resume()
    system._resume_from_disk()

    assert system.eval_service.current_version() == 1
    assert system._mode == "proof", "mode_switch event not recovered on resume"
    # the v1 feedback module is rebuilt and shapes disclosure.
    assert system.eval_service.current_feedback_source() is not None
    assert system.eval_service.query({"value": 4}).detail == "guidance-42"


def test_bootstrap_ingests_an_existing_checker_as_v0(tmp_path, monkeypatch):
    """The 'text + files' input form: when the raw input already ships a runnable
    checker, V0 WRAPS it (verify reads ctx['checker_dir']) and the checker is
    materialized under run_dir/checker/ so host-side verify can reach it."""
    # a bundled 'reference model' the wrapping verifier consults.
    checker_files = {"reference_model.py": "TARGET = 42\n"}
    wrap_verifier = (
        "import os, importlib.util\n"
        "def _load(cd):\n"
        "    spec = importlib.util.spec_from_file_location(\n"
        "        'reference_model', os.path.join(cd, 'reference_model.py'))\n"
        "    mod = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(mod)\n"
        "    return mod\n"
        "def verify(payload, ctx):\n"
        "    cd = ctx.get('checker_dir')\n"
        "    if not cd:\n"
        "        return {'feasible': False, 'raw': -1e9, 'artifacts': {}}\n"
        "    target = _load(cd).TARGET\n"
        "    try:\n"
        "        v = int(payload.get('value'))\n"
        "    except Exception:\n"
        "        return {'feasible': False, 'raw': -1e9, 'artifacts': {}}\n"
        "    ok = (v == target)\n"
        "    return {'feasible': ok, 'raw': (100.0 if ok else -1.0),"
        " 'artifacts': {'target': target}}\n"
    )
    system, A, bs = _authored_bootstrap(
        tmp_path, monkeypatch, verifier=wrap_verifier, seed={"value": 42},
        probes=[{"description": "wrong answer", "solution": {"value": 0}}],
        checker_files=checker_files, ctx={"checker_dir": "PLACEHOLDER"})

    # the checker was copied to a stable host path under run_dir and pinned into ctx.
    materialized = tmp_path / "run" / "checker" / "reference_model.py"
    assert materialized.is_file(), "bundled checker not materialized under run_dir"
    assert system._checker_dir == str((tmp_path / "run" / "checker").resolve())
    assert system._ctx.get("checker_dir") == system._checker_dir

    # V0 wraps the checker: the reference answer scores high, a wrong answer low.
    good = system.eval_service.query({"value": 42})
    bad = system.eval_service.query({"value": 0})
    assert good.ok and good.score == 100.0 and good.feasible is True
    assert bad.score == -1.0


def test_solver_container_gpu_and_image_knob(tmp_path, monkeypatch):
    """The Solver container honors an operator/bootstrap image + GPU request and adds
    NO creds. Default (unset) leaves the argv/kwargs unchanged. Also proves solver_env
    .json authored at bootstrap flows into the knobs."""
    from coscientist.coevo.container import AgentSession

    # (a) solver_env.json authored by the bootstrap agent flows into the knobs.
    sysA, A, _ = _authored_bootstrap(tmp_path / "k", monkeypatch, verifier=_CAP_VERIFIER,
                                     solver_env={"image": "k:1", "gpus": "all"})
    assert sysA.solver_image == "k:1" and sysA.solver_gpus == "all"

    # (b) explicit overrides reach the long-lived container's constructor; default
    #     leaves them at (default image, no gpu). Capture the constructor kwargs.
    def run_capturing(system, monkeypatch_):
        captured = {}

        class FakeContainer:
            def __init__(self, *, workdir, gateway, agent_elf, image, gpus=None, **kw):
                captured["image"] = image
                captured["gpus"] = gpus
                self.ws = Path(workdir)
            def start(self):
                return self
            def stop(self):
                pass
            def exec_agent(self, prompt, *, timeout_s, **kw):
                (self.ws / "solution_out.json").write_text(json.dumps({"value": 3}))
                return AgentSession(ok=True, returncode=0, stdout="", stderr="")

        class FakeControl:
            def __init__(self, *, workdir, handler, **kw):
                self.workdir = workdir
            def seed_shims(self, ws):
                pass
            def start(self):
                return self
            def stop(self):
                pass
            @property
            def needs_explicit_mount(self):
                return False
            @property
            def host_path(self):
                from pathlib import Path as _P
                return _P(self.workdir) / ".control.sock"

        monkeypatch_.setattr(A, "DockerContainer", FakeContainer)
        monkeypatch_.setattr(A, "ControlSocket", FakeControl)
        # keep the loop to a single normal turn: a no-op harden never moves V.
        monkeypatch_.setattr(system, "_run_supervisor_harden",
                             lambda *a, **k: None)
        system.solve_and_evolve(max_turns=1)
        return captured

    sysB, _, _ = _authored_bootstrap(tmp_path / "x", monkeypatch, verifier=_CAP_VERIFIER)
    sysB.solver_image, sysB.solver_gpus = "img:x", "1"
    cap = run_capturing(sysB, monkeypatch)
    assert cap["image"] == "img:x" and cap["gpus"] == "1"

    sysC, _, _ = _authored_bootstrap(tmp_path / "y", monkeypatch, verifier=_CAP_VERIFIER)
    cap2 = run_capturing(sysC, monkeypatch)
    assert cap2["image"] == "python:3.11-slim" and cap2["gpus"] is None


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


# ===========================================================================
# ResourceSpec — isolated Solver vs Verifier container resources (+ GPU seam)
# ===========================================================================
def test_resource_toml_parses_into_two_isolated_slices(tmp_path):
    """A resource.toml with distinct [solver]/[verifier] blocks -> the right cpus/mem/
    gpus/gpu_types/timeout per slice. A K3 task.toml [environment] seeds the SOLVER
    slice only (verifier stays default — isolation is explicit, never inferred). A
    missing file -> defaults()."""
    from coscientist.coevo import resources as R

    # (a) explicit two-slice resource.toml — each slice keeps its own resources.
    raw = tmp_path / "twoslice"
    raw.mkdir()
    (raw / "resource.toml").write_text(
        "[solver]\n"
        "image = 'solver:img'\n"
        "cpus = 4\n"
        "memory_mb = 8192\n"
        "gpus = 1\n"
        "\n"
        "[verifier]\n"
        "image = 'verifier:img'\n"
        "cpus = 8\n"
        "memory_mb = 16384\n"
        "gpus = 2\n"
        "gpu_types = ['A100']\n"
        "timeout_sec = 3600\n"
    )
    spec = R.load(raw)
    assert spec.solver.image == "solver:img" and spec.solver.cpus == 4
    assert spec.solver.memory_mb == 8192 and spec.solver.gpus == "1"
    assert spec.solver.timeout_sec is None      # solver did not set it
    assert spec.verifier.image == "verifier:img" and spec.verifier.cpus == 8
    assert spec.verifier.memory_mb == 16384 and spec.verifier.gpus == "2"
    assert spec.verifier.gpu_types == ["A100"] and spec.verifier.timeout_sec == 3600
    # the two slices are genuinely independent objects.
    assert spec.solver.cpus != spec.verifier.cpus

    # (b) K3 task.toml: [environment] seeds SOLVER; verifier stays default.
    k3 = tmp_path / "k3"
    k3.mkdir()
    (k3 / "task.toml").write_text(
        "[environment]\n"
        "cpus = 16\n"
        "memory_mb = 32768\n"
        "gpus = 1\n"
        "gpu_types = ['H100']\n"
        "allow_internet = false\n"
        "\n"
        "[agent]\n"
        "timeout_sec = 900\n"
        "\n"
        "[verifier]\n"
        "timeout_sec = 1800\n"
    )
    spec2 = R.load(k3)
    assert spec2.solver.cpus == 16 and spec2.solver.gpus == "1"
    assert spec2.solver.gpu_types == ["H100"]
    assert spec2.solver.timeout_sec == 900           # from [agent]
    # verifier slice only got its timeout; it did NOT inherit the solver's cpus/gpus.
    assert spec2.verifier.cpus is None and spec2.verifier.gpus is None
    assert spec2.verifier.timeout_sec == 1800

    # (c) missing file -> defaults (both empty).
    empty = tmp_path / "none"
    empty.mkdir()
    d = R.load(empty)
    assert d.solver.is_empty() and d.verifier.is_empty()


def test_cli_overrides_beat_file_and_solver_env_is_last_resort(tmp_path):
    """Precedence: defaults < resource.toml < CLI overrides; the agent-authored
    solver_env.json only fills a field still unset after the structured layers."""
    from coscientist.coevo import resources as R

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "resource.toml").write_text(
        "[solver]\nimage = 'file:img'\ncpus = 4\n"
        "[verifier]\ncpus = 4\n")

    # CLI overrides fold over the file (non-None wins).
    ov = R.ResourceSpec(
        solver=R.ContainerResources(cpus=99.0),
        verifier=R.ContainerResources(cpus=42.0))
    spec = R.load(raw).merge_overrides(solver=ov.solver, verifier=ov.verifier)
    assert spec.solver.cpus == 99.0            # CLI beat the file's 4
    assert spec.solver.image == "file:img"     # untouched-by-CLI field kept from file
    assert spec.verifier.cpus == 42.0

    # solver_env.json is last-resort: it must NOT overwrite a field the file set, but
    # may fill a still-unset one. Mirror _apply_solver_env_fallback's logic.
    file_spec = R.load(raw)                     # solver.image='file:img', gpus unset
    solver_env = {"image": "env:img", "gpus": "all"}
    sol = file_spec.solver
    img = sol.image if sol.image is not None else solver_env.get("image")
    gpus = sol.gpus if sol.gpus is not None else R._as_gpus(solver_env.get("gpus"))
    filled = file_spec.merge_overrides(
        solver=R.ContainerResources(image=img, gpus=gpus))
    assert filled.solver.image == "file:img"   # file wins over solver_env
    assert filled.solver.gpus == "all"         # solver_env filled the unset gpus


def test_solver_container_gets_the_solver_slice_argv(tmp_path, monkeypatch):
    """DockerContainer.start argv carries --gpus/--cpus/--memory/--network none from the
    solver slice; a default (unset) container's argv is byte-identical to today — no
    --cpus/--memory/--network — and never contains LLM creds."""
    from coscientist.coevo import container as C
    from coscientist.coevo.container import DockerContainer, GatewayConfig

    captured = {}

    def fake_run(argv, **kw):
        captured["argv"] = list(argv)
        class P: pass
        p = P(); p.returncode = 0; p.stdout = "cid123\n"; p.stderr = ""
        return p

    monkeypatch.setattr(C.subprocess, "run", fake_run)
    monkeypatch.setattr(DockerContainer, "_prepare_codex_home",
                        lambda self: tmp_path / "ch")
    (tmp_path / "ch").mkdir()
    gw = GatewayConfig(codex_home=tmp_path / ".codex", model="m")

    # (a) resourced solver slice -> the caps appear.
    c = DockerContainer(workdir=tmp_path / "ws", gateway=gw,
                        agent_elf=tmp_path / "codex", image="solver:img",
                        gpus="1", cpus=4.0, memory_mb=8192, allow_internet=False)
    c.start()
    argv = captured["argv"]
    assert "--gpus" in argv and argv[argv.index("--gpus") + 1] == "1"
    assert "--cpus" in argv and argv[argv.index("--cpus") + 1] == "4.0"
    assert "--memory" in argv and argv[argv.index("--memory") + 1] == "8192m"
    assert argv[argv.index("--network") + 1] == "none"
    assert "solver:img" in argv
    assert not any("LLM_API_KEY" in str(a) for a in argv)

    # (b) default container: no resource flags at all (byte-for-byte legacy argv).
    captured.clear()
    c2 = DockerContainer(workdir=tmp_path / "ws2", gateway=gw,
                         agent_elf=tmp_path / "codex", image="python:3.11-slim")
    c2.start()
    argv2 = captured["argv"]
    assert "--cpus" not in argv2 and "--memory" not in argv2
    assert "--network" not in argv2 and "--gpus" not in argv2


def test_verifier_backend_is_isolated_from_solver(tmp_path, monkeypatch):
    """LOAD-BEARING: with a checker task and a verifier slice of DIFFERENT cpus/gpus than
    the solver, the verify `docker run` argv carries the VERIFIER slice's resources (not
    the solver's), while the solver container argv carries the SOLVER slice's. Neither
    argv contains LLM creds; the checker_dir is mounted read-only into the verifier
    container as /checker."""
    from coscientist.coevo import resources as R
    from coscientist.demo import evaluator as EV
    from coscientist.demo.evaluator import Evaluator

    # A checker task, plus CLI overrides giving the verifier a DIFFERENT slice.
    overrides = R.ResourceSpec(
        solver=R.ContainerResources(cpus=2.0, gpus="1"),
        verifier=R.ContainerResources(cpus=16.0, gpus="2", memory_mb=32768,
                                      timeout_sec=1800.0))
    checker_files = {"reference_model.py": "TARGET = 42\n"}
    wrap_verifier = (
        "import os\n"
        "def verify(payload, ctx):\n"
        "    cd = ctx.get('checker_dir')\n"
        "    ok = bool(cd) and os.path.isdir(cd)\n"
        "    return {'feasible': ok, 'raw': (1.0 if ok else -1e9), 'artifacts': {'cd': cd}}\n"
    )
    system, A, _ = _authored_bootstrap(
        tmp_path, monkeypatch, verifier=wrap_verifier, seed={"value": 42},
        probes=[{"description": "x", "solution": {"value": 0}}],
        checker_files=checker_files, ctx={"checker_dir": "PLACEHOLDER"},
        llm_config={"api_key": "SECRET", "base_url": "http://x", "model": "m"})
    system.resource_overrides = overrides
    system._resolve_resources()

    # the verifier backend is built from the VERIFIER slice, defaulting its image to
    # the solver image — NOT sharing the solver's cpus/gpus.
    backend = system._verify_backend()
    assert backend is not None
    assert backend["cpus"] == 16.0 and backend["gpus"] == "2"
    assert backend["memory_mb"] == 32768 and backend["timeout_s"] == 1800.0

    # capture the verify `docker run` argv.
    seen = {}
    real_run = EV.subprocess.run

    def fake_run(argv, **kw):
        if argv and argv[0] == "docker":
            seen["argv"] = list(argv)
            seen["timeout"] = kw.get("timeout")
            class P: pass
            p = P(); p.returncode = 0
            p.stdout = '__VERIFY_RESULT__{"feasible": true, "raw": 1.0, "artifacts": {}}\n'
            p.stderr = ""
            return p
        return real_run(argv, **kw)

    monkeypatch.setattr(EV.subprocess, "run", fake_run)
    ev = Evaluator(exec_backend=backend)
    ev.versions.append(EV.VerifierVersion(0, wrap_verifier, "t", ""))
    ev.run({"value": 42}, {"checker_dir": system._checker_dir},
           env={"LLM_API_KEY": "SECRET"})

    argv = seen["argv"]
    # verifier's OWN slice, not the solver's (2.0/'1').
    assert argv[argv.index("--cpus") + 1] == "16.0"
    assert argv[argv.index("--gpus") + 1] == "2"
    assert argv[argv.index("--memory") + 1] == "32768m"
    assert argv[argv.index("--network") + 1] == "none"
    assert seen["timeout"] == 1800.0
    # checker mounted read-only as /checker.
    assert any(a.endswith("/checker:ro") for a in argv)
    # NO creds cross into the verifier container (no -e LLM_API_KEY).
    assert not any("LLM_API_KEY" in str(a) for a in argv)
    assert "SECRET" not in " ".join(str(a) for a in argv)


def test_verify_timeout_comes_from_verifier_slice(tmp_path, monkeypatch):
    """A backend timeout_s (from verifier.timeout_sec) bounds the verify call, not the
    10s host default."""
    from coscientist.demo import evaluator as EV
    from coscientist.demo.evaluator import Evaluator

    seen = {}
    real_run = EV.subprocess.run

    def fake_run(argv, **kw):
        if argv and argv[0] == "docker":
            seen["timeout"] = kw.get("timeout")
            class P: pass
            p = P(); p.returncode = 0
            p.stdout = '__VERIFY_RESULT__{"feasible": true, "raw": 1.0, "artifacts": {}}\n'
            p.stderr = ""
            return p
        return real_run(argv, **kw)

    monkeypatch.setattr(EV.subprocess, "run", fake_run)
    ev = Evaluator(exec_backend={"image": "img", "timeout_s": 300.0})
    ev.versions.append(EV.VerifierVersion(
        0, "def verify(p, c):\n    return {'feasible': True, 'raw': 1.0, 'artifacts': {}}\n",
        "t", ""))
    ev.run({}, {})
    assert seen["timeout"] == 300.0


def test_host_subprocess_verify_unchanged_without_backend(tmp_path, monkeypatch):
    """exec_backend=None -> run/run_feedback still shell to sys.executable (never docker);
    the host path is byte-for-byte the legacy path."""
    from coscientist.demo import evaluator as EV
    from coscientist.demo.evaluator import Evaluator

    seen = {"docker": False, "python": False}
    real_run = EV.subprocess.run

    def fake_run(argv, **kw):
        if argv and argv[0] == "docker":
            seen["docker"] = True
        if argv and argv[0] == EV.sys.executable:
            seen["python"] = True
        return real_run(argv, **kw)

    monkeypatch.setattr(EV.subprocess, "run", fake_run)
    ev = Evaluator()  # exec_backend defaults to None
    ev.versions.append(EV.VerifierVersion(
        0, "def verify(p, c):\n    return {'feasible': True, 'raw': 2.0, 'artifacts': {}}\n",
        "t", ""))
    r = ev.run({"value": 1}, {})
    assert r.feasible and r.raw == 2.0
    assert seen["python"] is True and seen["docker"] is False


def test_manifest_records_both_resource_slices(tmp_path, monkeypatch):
    """After a bootstrapped run, manifest.json['resource_spec'] carries solver AND
    verifier sub-dicts (the isolated slices, for audit + UI)."""
    from coscientist.coevo import resources as R

    system, A, _ = _authored_bootstrap(
        tmp_path, monkeypatch, verifier=_CAP_VERIFIER)
    system.resource_overrides = R.ResourceSpec(
        solver=R.ContainerResources(cpus=4.0, image="s:img"),
        verifier=R.ContainerResources(cpus=8.0, gpus="1"))
    system._resolve_resources()
    system._finalize()

    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    rs = manifest["resource_spec"]
    assert set(rs.keys()) >= {"solver", "verifier"}
    assert rs["solver"]["cpus"] == 4.0 and rs["solver"]["image"] == "s:img"
    assert rs["verifier"]["cpus"] == 8.0 and rs["verifier"]["gpus"] == "1"



# ---------------------------------------------------------------------------
# anti-interruption (SForge-style): AF_UNIX short-path bind + respawn supervisor
# ---------------------------------------------------------------------------
def test_control_socket_uses_short_path_when_workdir_too_long(tmp_path):
    """A deep runs/<long-id>/solver_ws/.control.sock exceeds the ~108B AF_UNIX limit
    and would crash at bind(). ControlSocket falls back to a short /tmp bind path and
    flags that the container must mount that file explicitly at /work/.control.sock."""
    from coscientist.coevo.container import ControlSocket, _AF_UNIX_SAFE_LEN

    # a workdir whose natural socket path blows past the limit
    long_ws = tmp_path / ("x" * 120) / "solver_ws"
    long_ws.mkdir(parents=True)
    cs = ControlSocket(workdir=long_ws, handler=lambda c, a: {"ok": True})
    assert len(str(cs._natural_path)) > _AF_UNIX_SAFE_LEN
    assert cs.needs_explicit_mount is True
    assert len(str(cs.host_path)) <= _AF_UNIX_SAFE_LEN
    # stable/deterministic: a second handle on the same workdir binds the same path
    cs2 = ControlSocket(workdir=long_ws, handler=lambda c, a: {"ok": True})
    assert cs.host_path == cs2.host_path
    # actually binds + serves + cleans up at the short path
    cs.start()
    try:
        assert cs.host_path.exists()
        import socket as _s, json as _j
        c = _s.socket(_s.AF_UNIX, _s.SOCK_STREAM); c.settimeout(5)
        c.connect(str(cs.host_path))
        c.sendall((_j.dumps({"cmd": "status", "arg": {}}) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            ch = c.recv(4096)
            if not ch:
                break
            buf += ch
        c.close()
        assert _j.loads(buf.decode())["ok"] is True
    finally:
        cs.stop()
    assert not cs.host_path.exists()


def test_control_socket_short_workdir_stays_in_workdir(tmp_path):
    """When the workdir path is short, nothing changes: the socket lives in the
    workdir (reachable via -v workdir:/work) and no explicit mount is needed."""
    from coscientist.coevo.container import ControlSocket

    ws = tmp_path / "ws"
    ws.mkdir()
    cs = ControlSocket(workdir=ws, handler=lambda c, a: {"ok": True})
    assert cs.needs_explicit_mount is False
    assert cs.host_path == cs._natural_path == ws / ".control.sock"


def test_docker_container_mounts_socket_when_bound_outside_workdir(tmp_path, monkeypatch):
    """When the control socket is bound outside the workdir, DockerContainer adds a
    -v <host_sock>:/work/.control.sock mount so the in-container shim still finds it."""
    from coscientist.coevo import container as C

    captured = {}

    class _Proc:
        returncode = 0
        stdout = "cid123"
        stderr = ""

    def fake_run(argv, **kw):
        if argv[:2] == ["docker", "run"]:
            captured["argv"] = argv
        return _Proc()

    monkeypatch.setattr(C.subprocess, "run", fake_run)
    ws = tmp_path / "ws"; ws.mkdir()
    gw = C.GatewayConfig(codex_home=tmp_path)
    # a fake codex_home so _prepare_codex_home doesn't need a real auth.json
    monkeypatch.setattr(C.shutil, "copy2", lambda *a, **k: None)
    host_sock = tmp_path / "cs_abcdef012345.sock"
    dc = C.DockerContainer(workdir=ws, gateway=gw, agent_elf=tmp_path / "codex",
                           control_sock_host_path=host_sock)
    dc.start()
    argv = captured["argv"]
    joined = " ".join(argv)
    assert f"{host_sock.resolve()}:/work/.control.sock" in joined
    dc.stop()

    # and WITHOUT an explicit path, no such extra mount appears.
    captured.clear()
    dc2 = C.DockerContainer(workdir=ws, gateway=gw, agent_elf=tmp_path / "codex")
    dc2.start()
    assert ":/work/.control.sock" not in " ".join(captured["argv"])
    dc2.stop()


def test_launcher_respawns_crashed_child_with_remaining_budget(tmp_path, monkeypatch):
    """A child that dies BEFORE its budget (no run_stop event) and ran long enough to
    not look systematic is respawned with --resume and the REMAINING budget. The
    respawn count + new pid are persisted so the watcher is itself crash-safe."""
    from coscientist.coevo import launcher as L

    spawned = []

    def fake_spawn(self, spec, *, budget_s, python, extra_args):
        spawned.append({"run_id": spec.run_id, "budget_s": budget_s,
                        "gpu": spec.gpu_device})
        return 4000 + len(spawned)

    monkeypatch.setattr(L.Batch, "_spawn", fake_spawn)
    # child looks dead
    monkeypatch.setattr(L, "_alive", lambda pid: False)

    runs = tmp_path / "runs"
    batch = L.Batch("b", runs_dir=runs)
    rid = "b__demo"
    (runs / rid).mkdir(parents=True)          # run dir exists, but no run_stop event
    now = 1_000_000.0
    (runs / "b").mkdir(parents=True, exist_ok=True)
    (runs / "b" / "batch.json").write_text(json.dumps({
        "batch": "b", "python": "py", "extra_args": [], "gpu_devices": ["3"],
        "runs": [{"run_id": rid, "pid": 999, "status": "running",
                  "input_dir": str(runs / rid),
                  "log": str(runs / "b" / f"{rid}.log"),
                  "budget_s": 28800.0, "deadline_epoch": now + 20000.0,
                  "last_launch_epoch": now - 5000.0, "respawns": 0,
                  "resource_config": None, "gpu_device": "3"}]}))

    actions = batch._wave_tick(now=now)
    assert actions[0]["action"] == "respawned"
    assert len(spawned) == 1
    # remaining budget ~= deadline - now (20000), NOT the full 28800
    assert abs(spawned[0]["budget_s"] - 20000.0) < 1.0
    assert spawned[0]["gpu"] == "3"                   # resource pin preserved
    man = json.loads((runs / "b" / "batch.json").read_text())
    assert man["runs"][0]["respawns"] == 1
    assert man["runs"][0]["pid"] == 4001
    assert man["runs"][0]["gpu_device"] == "3"        # respawn keeps its card


def test_launcher_respawn_respects_done_budget_and_systematic_floors(tmp_path, monkeypatch):
    """The respawn tick does NOT relaunch when: the run finished (run_stop present),
    the absolute budget deadline passed, or the child died faster than the
    MIN_RUNTIME floor (systematic failure — bad ELF/creds)."""
    from coscientist.coevo import launcher as L

    monkeypatch.setattr(L.Batch, "_spawn",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("must not respawn")))
    monkeypatch.setattr(L, "_alive", lambda pid: False)

    runs = tmp_path / "runs"
    batch = L.Batch("b", runs_dir=runs)
    now = 2_000_000.0

    def _mk(rid, *, stopped, deadline, last_launch):
        rd = runs / rid
        rd.mkdir(parents=True, exist_ok=True)
        events = [{"kind": "bootstrap_done"}]
        if stopped:
            events.append({"kind": "run_stop", "best_score": 1.0})
        (rd / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n")
        return {"run_id": rid, "pid": 1, "status": "running", "input_dir": str(rd),
                "log": str(rd / "l.log"), "budget_s": 28800.0,
                "deadline_epoch": deadline, "last_launch_epoch": last_launch,
                "respawns": 0, "resource_config": None, "gpu_device": None}

    (runs / "b").mkdir(parents=True, exist_ok=True)
    (runs / "b" / "batch.json").write_text(json.dumps({
        "batch": "b", "python": "py", "extra_args": [], "gpu_devices": [], "runs": [
            _mk("b__done", stopped=True, deadline=now + 5000, last_launch=now - 5000),
            _mk("b__expired", stopped=False, deadline=now - 10, last_launch=now - 5000),
            _mk("b__systematic", stopped=False, deadline=now + 5000, last_launch=now - 5),
        ]}))

    actions = {a["run_id"]: a["action"] for a in batch._wave_tick(now=now)}
    assert actions["b__done"] == "done"
    assert actions["b__expired"] == "budget_spent"
    assert actions["b__systematic"] == "held_systematic"


# ---------------------------------------------------------------------------
# wave scheduling: 8-GPU pool, first-wave launch + pending queue, card handoff on
# completion, held_systematic keeps its card, idempotent resume. All in-memory:
# _spawn / _alive / _progress are stubbed, no real process, docker, or GPU.
# ---------------------------------------------------------------------------
def _wave_batch(tmp_path, monkeypatch, *, n_inputs, gpus, spawn_log):
    """Build a Batch whose _spawn is a fake returning incrementing pids, with n_inputs
    problem dirs and a pool of `gpus`. Returns (L, batch, inputs)."""
    from coscientist.coevo import launcher as L

    def fake_spawn(self, spec, *, budget_s, python, extra_args):
        spawn_log.append({"run_id": spec.run_id, "gpu": spec.gpu_device,
                          "budget_s": budget_s})
        return 5000 + len(spawn_log)

    monkeypatch.setattr(L.Batch, "_spawn", fake_spawn)
    inputs = []
    for i in range(n_inputs):
        d = tmp_path / "probs" / f"task{i}"
        d.mkdir(parents=True)
        inputs.append(d)
    batch = L.Batch("wave", runs_dir=tmp_path / "runs")
    return L, batch, inputs


def test_wave_first_wave_fills_pool_and_queues_the_rest(tmp_path, monkeypatch):
    """8-GPU pool + 10 inputs: start() launches exactly 8 runs (each on a distinct card
    0..7) and records the other 2 as pending with no card, no pid, no charged deadline."""
    spawn_log = []
    L, batch, inputs = _wave_batch(tmp_path, monkeypatch, n_inputs=10, gpus=8,
                                   spawn_log=spawn_log)
    pool = [str(i) for i in range(8)]
    batch.start(inputs, hours=4.0, gpu_devices=pool)

    assert len(spawn_log) == 8, "first wave must be exactly pool-size"
    assert sorted(s["gpu"] for s in spawn_log) == pool, "distinct card each, 0..7"
    man = json.loads((tmp_path / "runs" / "wave" / "batch.json").read_text())
    assert man["gpu_devices"] == pool
    running = [r for r in man["runs"] if r["status"] == "running"]
    pending = [r for r in man["runs"] if r["status"] == "pending"]
    assert len(running) == 8 and len(pending) == 2
    for r in running:
        assert r["gpu_device"] in pool and r["pid"] and r["deadline_epoch"]
    for r in pending:
        assert r["gpu_device"] is None and r["pid"] is None
        assert r["deadline_epoch"] is None      # budget not charged while queued
    # every card used once, no double-assignment
    assert len({r["gpu_device"] for r in running}) == 8


def test_wave_freed_card_is_handed_to_next_pending(tmp_path, monkeypatch):
    """When a running run finishes (run_stop), its card is freed and the next pending
    run is launched onto THAT card — never a double-assignment."""
    spawn_log = []
    L, batch, inputs = _wave_batch(tmp_path, monkeypatch, n_inputs=3, gpus=2,
                                   spawn_log=spawn_log)
    pool = ["0", "1"]
    batch.start(inputs, hours=4.0, gpu_devices=pool)
    assert len(spawn_log) == 2                 # wave of 2, one pending
    man = json.loads((tmp_path / "runs" / "wave" / "batch.json").read_text())
    running = [r for r in man["runs"] if r["status"] == "running"]
    # pick the run on card "0" and make it finish cleanly.
    done_run = next(r for r in running if r["gpu_device"] == "0")
    rd = tmp_path / "runs" / done_run["run_id"]
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "events.jsonl").write_text(
        json.dumps({"kind": "run_stop", "best_score": 1.0}) + "\n")
    # its pid is now dead; the other running pid + any future pid are alive.
    monkeypatch.setattr(L, "_alive", lambda pid: pid != done_run["pid"])

    now = 3_000_000.0
    actions = {a["run_id"]: a["action"] for a in batch._wave_tick(now=now)}
    assert actions[done_run["run_id"]] == "done"
    man2 = json.loads((tmp_path / "runs" / "wave" / "batch.json").read_text())
    # the freed card "0" was handed to the previously-pending run.
    launched = [s for s in spawn_log[2:]]
    assert len(launched) == 1 and launched[0]["gpu"] == "0"
    assert launched[0]["budget_s"] == 4.0 * 3600.0, "pending gets the FULL budget"
    now_running = [r for r in man2["runs"] if r["status"] == "running"]
    cards = [r["gpu_device"] for r in now_running]
    assert sorted(cards) == ["0", "1"], f"no double-assignment: {cards}"
    # the newly-launched run got its deadline anchored now.
    newr = next(r for r in man2["runs"] if r["run_id"] == launched[0]["run_id"])
    assert abs(newr["deadline_epoch"] - (now + 4.0 * 3600.0)) < 1.0


def test_wave_held_systematic_keeps_its_card_no_pending_steal(tmp_path, monkeypatch):
    """A run that dies within MIN_RUNTIME is held_systematic and KEEPS its card; a
    pending run must NOT be launched onto that still-occupied device."""
    spawn_log = []
    L, batch, inputs = _wave_batch(tmp_path, monkeypatch, n_inputs=2, gpus=1,
                                   spawn_log=spawn_log)
    pool = ["0"]
    batch.start(inputs, hours=4.0, gpu_devices=pool)
    assert len(spawn_log) == 1                 # one running on "0", one pending
    man = json.loads((tmp_path / "runs" / "wave" / "batch.json").read_text())
    running = next(r for r in man["runs"] if r["status"] == "running")
    rd = tmp_path / "runs" / running["run_id"]
    rd.mkdir(parents=True, exist_ok=True)       # no run_stop => not done
    monkeypatch.setattr(L, "_alive", lambda pid: False)   # its pid looks dead

    # died just now (< MIN_RUNTIME since last_launch) => systematic hold.
    now = running["last_launch_epoch"] + 1.0
    before = len(spawn_log)
    actions = {a["run_id"]: a["action"] for a in batch._wave_tick(now=now)}
    assert actions[running["run_id"]] == "held_systematic"
    assert len(spawn_log) == before, "pending must NOT steal the held card"
    man2 = json.loads((tmp_path / "runs" / "wave" / "batch.json").read_text())
    held = next(r for r in man2["runs"] if r["status"] == "held_systematic")
    assert held["gpu_device"] == "0", "held_systematic keeps its card"
    assert any(r["status"] == "pending" for r in man2["runs"])


def test_wave_start_is_idempotent_resume(tmp_path, monkeypatch):
    """A second start() against an existing manifest must NOT re-plan: it resumes via a
    single _wave_tick without resetting status / re-anchoring deadlines / clearing
    respawns. Live runs stay put; only genuinely-freed cards get filled."""
    spawn_log = []
    L, batch, inputs = _wave_batch(tmp_path, monkeypatch, n_inputs=3, gpus=2,
                                   spawn_log=spawn_log)
    pool = ["0", "1"]
    batch.start(inputs, hours=4.0, gpu_devices=pool)
    man1 = json.loads((tmp_path / "runs" / "wave" / "batch.json").read_text())
    running1 = {r["run_id"]: dict(r) for r in man1["runs"]
                if r["status"] == "running"}
    assert len(spawn_log) == 2

    # everything is alive; a re-run must spawn nothing new and change nothing.
    monkeypatch.setattr(L, "_alive", lambda pid: True)
    batch.start(inputs, hours=4.0, gpu_devices=pool)
    assert len(spawn_log) == 2, "idempotent resume must not spawn while all alive"
    man2 = json.loads((tmp_path / "runs" / "wave" / "batch.json").read_text())
    for rid, r1 in running1.items():
        r2 = next(r for r in man2["runs"] if r["run_id"] == rid)
        assert r2["status"] == "running" and r2["pid"] == r1["pid"]
        assert r2["deadline_epoch"] == r1["deadline_epoch"]   # NOT re-anchored
        assert r2["gpu_device"] == r1["gpu_device"]
    # the pending run is still pending (both cards still occupied by live runs).
    assert sum(r["status"] == "pending" for r in man2["runs"]) == 1


def test_alive_treats_zombie_as_dead(tmp_path):
    """Regression: os.kill(pid,0) SUCCEEDS for a zombie (exited-but-unreaped) child, so a
    naive liveness check reports a finished run as forever-alive and wave scheduling
    deadlocks on the first wave (cards never freed, pending never launched). _alive must
    read /proc and treat state 'Z' as dead."""
    import os
    import time as _time
    from coscientist.coevo import launcher as L

    pid = os.fork()
    if pid == 0:
        os._exit(0)                      # child exits immediately -> zombie
    try:
        for _ in range(50):              # let it exit; now a zombie (we haven't waited)
            _time.sleep(0.01)
            stat = open(f"/proc/{pid}/stat").read()
            if stat[stat.rfind(")") + 1:].split()[0] == "Z":
                break
        assert L._alive(pid) is False, "a zombie child must read as DEAD, not alive"
    finally:
        os.waitpid(pid, 0)               # reap so we don't leak the zombie


def test_reap_children_clears_exited_children(tmp_path):
    """_reap_children must non-blockingly reap exited children so they stop being
    zombies (and stop keeping os.kill(pid,0) alive), and must not raise with no
    children."""
    import os
    import time as _time
    from coscientist.coevo import launcher as L

    pid = os.fork()
    if pid == 0:
        os._exit(0)
    for _ in range(50):
        _time.sleep(0.01)
        try:
            stat = open(f"/proc/{pid}/stat").read()
            if stat[stat.rfind(")") + 1:].split()[0] == "Z":
                break
        except OSError:
            break
    L._reap_children()                    # reap our zombie
    assert L._alive(pid) is False         # pid gone entirely now
    L._reap_children()                    # no children left -> harmless no-op


def test_package_k3_task_is_idempotent(tmp_path):
    """package_k3_task copies the raw task + writes the resource.toml template, and a
    second call is a no-op that neither re-copies nor clobbers the resource.toml."""
    from coscientist.coevo import launcher as L

    tasks_root = tmp_path / "tasks"
    src = tasks_root / "K3_demo"
    (src / "environment").mkdir(parents=True)
    (src / "instruction.md").write_text("solve it")
    (src / "task.toml").write_text("gpus=1\n")
    template = tmp_path / "resource.toml"
    template.write_text('[solver]\ngpus="device=0"\n')

    problems = tmp_path / "problems"
    dst = L.package_k3_task("K3_demo", tasks_root=tasks_root,
                            problems_root=problems, template=template)
    assert (dst / "instruction.md").read_text() == "solve it"
    assert (dst / "resource.toml").read_text() == template.read_text()
    assert dst.name == "k3_demo"

    # mutate the packaged copy, then re-package: it must NOT be overwritten.
    (dst / "resource.toml").write_text("EDITED")
    dst2 = L.package_k3_task("K3_demo", tasks_root=tasks_root,
                             problems_root=problems, template=template)
    assert dst2 == dst
    assert (dst / "resource.toml").read_text() == "EDITED", "idempotent: no clobber"
