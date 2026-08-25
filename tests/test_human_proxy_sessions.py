from __future__ import annotations

from pathlib import Path

from coscientist.coevo.agent_system import AgentSystem, _version0
from coscientist.coevo.eval_service import EvalService, FeedbackLevel
from coscientist.coevo.feishu_human import FeishuHumanSessionService
from coscientist.coevo.human_evidence import EvidenceReply
from coscientist.coevo.human_proxy_sessions import (
    EvaluatorBackedHumanProxyAgent,
    HumanProxyConversation,
    HumanProxyTurn,
    HumanProxySessionPort,
    LoopbackTransport,
    PythonReferenceEvaluator,
)
from coscientist.coevo.human_sessions import (
    HumanSessionStore,
    SessionOutcome,
    SessionState,
)
from coscientist.demo.evaluator import Evaluator


class FakeCoSessionAgent:
    def reply(self, **kwargs):
        instruction = kwargs["agent_instruction"]
        if "会话开场" in instruction:
            return EvidenceReply(text="我先说明当前评估变化，再请你判断是否更接近真实目标。")
        if "总结" in instruction:
            return EvidenceReply(
                text="我已总结你的判断；如果准确请回复确认结束。",
                proposed_outcome=SessionOutcome(decision="none"),
            )
        return EvidenceReply(text="我收到了你的 V* 对照结论，并记录了依据。")


class IndependentConversationAgent:
    """A distinct evaluator-backed agent that reacts to the live transcript."""

    def __init__(self, reference_evaluator, store):
        self.evaluator_agent = EvaluatorBackedHumanProxyAgent(reference_evaluator)
        self.store = store
        self.assessment = None
        self.observed_agent_replies = []

    def start_conversation(self, *, purpose, context):
        self.assessment = self.evaluator_agent.assess(
            purpose=purpose, context=context
        )
        return HumanProxyConversation(
            outcome=self.assessment.outcome,
            next_turn=self.next_turn,
        )

    def next_turn(self, *, session, transcript):
        assert self.assessment is not None
        agent_replies = [item.text for item in transcript if item.role == "agent"]
        self.observed_agent_replies = agent_replies
        human_messages = [item.text for item in transcript if item.role == "human"]

        if not human_messages:
            return HumanProxyTurn(self.assessment.message)
        if len(human_messages) == 1:
            assert "V* 对照结论" in agent_replies[-1]
            return HumanProxyTurn("请再解释这个结论覆盖了哪些边界情况？")
        if len(human_messages) == 2:
            return HumanProxyTurn("信息基本充分，我请求结束这轮。")
        if len(human_messages) == 3:
            assert session.state is SessionState.CLOSE_REQUESTED
            return HumanProxyTurn("先别结束，我还要确认继续聊天不会消耗第二次机会。")
        if len(human_messages) == 4:
            assert session.state is SessionState.ACTIVE
            assert self.store.opened_count == 1
            assert self.store.outcome(session.session_id) is None
            assert self.store.staged_outcome(session.session_id) is None
            return HumanProxyTurn("现在还剩几次会话机会？")
        if len(human_messages) == 5:
            return HumanProxyTurn("这轮可以结束了。")
        if len(human_messages) == 6:
            assert session.state is SessionState.CLOSE_REQUESTED
            return HumanProxyTurn("确认结束")
        raise AssertionError("conversation driver asked for a turn after explicit close")


def _reference_module(tmp_path: Path) -> Path:
    path = tmp_path / "hidden_reference.py"
    path.write_text(
        "TOP_SECRET_VSTAR_SOURCE = 'must never enter the run'\n"
        "def verify(payload, ctx):\n"
        "    value = float(payload['value'])\n"
        "    raw = -abs(value - float(ctx.get('target', 3)))\n"
        "    return {'feasible': True, 'raw': raw, 'artifacts': {'hidden': True}}\n"
    )
    return path


def _improving_context():
    return {
        "comparison_cases": [
            {
                "label": "seed",
                "payload": {"value": 1},
                "current": {"ok": True, "feasible": True, "score": 1.0},
                "proposed": {"ok": True, "feasible": True, "score": -2.0},
            },
            {
                "label": "adversarial_probe",
                "payload": {"value": 10},
                "current": {"ok": True, "feasible": True, "score": 10.0},
                "proposed": {"ok": True, "feasible": True, "score": -7.0},
            },
        ],
        "current_version": 0,
        "diff": "tighten toward the honest target",
    }


def _proxy_port(tmp_path, *, context=None):
    run_dir = tmp_path / "run"
    store = HumanSessionStore(run_dir)
    transport = LoopbackTransport()
    service = FeishuHumanSessionService(
        store=store,
        transport=transport,
        agent=FakeCoSessionAgent(),
    )
    reference = PythonReferenceEvaluator(
        _reference_module(tmp_path), context={"target": 3}
    )
    proxy_agent = EvaluatorBackedHumanProxyAgent(reference)
    port = HumanProxySessionPort(
        service=service,
        proxy_agent=proxy_agent,
        expert_id="human_proxy_vstar",
    )
    return port, store, transport, reference


def test_proxy_uses_real_evaluator_and_full_human_session_contract(tmp_path):
    port, store, transport, reference = _proxy_port(tmp_path)

    outcome = port.consult(purpose="verifier_change", context=_improving_context())

    session = store.sessions()[0]
    assert session.state is SessionState.CLOSED
    assert store.opened_count == 1
    assert outcome == store.outcome(session.session_id)
    assert outcome.decision == "approve"
    assert reference.calls == 2

    transcript = store.transcript(session.session_id)
    assert [item.role for item in transcript] == [
        "agent",
        "human",
        "agent",
        "human",
        "agent",
        "human",
        "agent",
        "human",
        "agent",
    ]
    assert "请继续说明" in transcript[3].text
    assert "这轮可以结束" in transcript[5].text
    assert transcript[7].text == "确认结束"
    assert "第 1/5 次" in transcript[0].text
    assert len(transport.sent) == 5


def test_proxy_port_runs_an_independent_multi_turn_agent_through_same_contract(
    tmp_path,
):
    port, store, transport, reference = _proxy_port(tmp_path)
    dialogue_agent = IndependentConversationAgent(reference, store)
    port.proxy_agent = dialogue_agent

    outcome = port.consult(purpose="verifier_change", context=_improving_context())

    session = store.sessions()[0]
    transcript = store.transcript(session.session_id)
    assert outcome == dialogue_agent.assessment.outcome
    assert session.state is SessionState.CLOSED
    assert store.opened_count == 1
    assert store.remaining_count == 4
    assert reference.calls == 2
    assert [item.role for item in transcript] == [
        "agent", "human", "agent", "human", "agent", "human", "agent",
        "human", "agent", "human", "agent", "human", "agent", "human", "agent",
    ]
    assert "先别结束" in transcript[7].text
    assert "还剩几次" in transcript[9].text
    assert len(dialogue_agent.observed_agent_replies) >= 7
    assert len(transport.sent) == 8


def test_proxy_rejects_a_proposal_that_moves_away_from_real_evaluator(tmp_path):
    port, _, _, _ = _proxy_port(tmp_path)
    context = _improving_context()
    for case in context["comparison_cases"]:
        case["current"], case["proposed"] = case["proposed"], case["current"]

    outcome = port.consult(purpose="verifier_change", context=context)

    assert outcome.decision == "reject"
    assert outcome.rejected_changes
    assert "真实 evaluator" in outcome.human_rationale


def test_proxy_never_leaks_reference_source_or_raw_results_into_run(tmp_path):
    port, store, _, _ = _proxy_port(tmp_path)
    port.consult(purpose="verifier_change", context=_improving_context())

    blobs = []
    for path in store.run_dir.rglob("*"):
        if path.is_file():
            blobs.append(path.read_text(encoding="utf-8", errors="ignore"))
    joined = "\n".join(blobs)
    assert "TOP_SECRET_VSTAR_SOURCE" not in joined
    assert "must never enter the run" not in joined
    assert "'hidden': True" not in joined
    assert '"hidden": true' not in joined.lower()


def test_proxy_has_the_same_five_session_budget_as_a_human(tmp_path):
    port, store, _, _ = _proxy_port(tmp_path)

    outcomes = [
        port.consult(purpose=f"checkpoint_{i}", context=_improving_context())
        for i in range(6)
    ]

    assert all(outcome is not None for outcome in outcomes[:5])
    assert outcomes[5] is None
    assert store.opened_count == 5
    assert store.remaining_count == 0


def test_proxy_task_definition_session_still_talks_and_closes_without_cases(tmp_path):
    port, store, _, reference = _proxy_port(tmp_path)

    outcome = port.consult(
        purpose="task_definition", context={"phase": "before_bootstrap"}
    )

    assert outcome.decision == "guide"
    assert outcome.evidence_requested
    assert reference.calls == 0
    assert store.sessions()[0].state is SessionState.CLOSED


def test_proxy_end_to_end_approves_then_agent_system_installs_after_close(tmp_path):
    current_source = (
        "def verify(payload, ctx):\n"
        "    value = float(payload.get('value', -1))\n"
        "    ok = 0 <= value <= 10\n"
        "    return {'feasible': ok, 'raw': value if ok else -1e9, 'artifacts': {}}\n"
    )
    proposed_source = (
        "def verify(payload, ctx):\n"
        "    value = float(payload.get('value', -1))\n"
        "    ok = 0 <= value <= 10\n"
        "    return {'feasible': ok, 'raw': -abs(value - 3) if ok else -1e9, "
        "'artifacts': {}}\n"
    )
    raw = tmp_path / "raw_system"
    raw.mkdir()
    (raw / "instruction.md").write_text("target value is three")
    system = AgentSystem(raw_input_dir=raw, run_dir=tmp_path / "system_run")
    evaluator = Evaluator()
    evaluator.versions.append(_version0(current_source))
    system.evaluator = evaluator
    system._ctx = {}
    system._seed = {"value": 1}
    system._probes = [
        {"description": "proxy exploit", "solution": {"value": 10}}
    ]
    system.eval_service = EvalService(
        evaluator=evaluator,
        ctx_provider=lambda: system._ctx,
        feedback_level=FeedbackLevel.WITH_ARTIFACTS,
    )
    system.store.verifier_version(
        0, current_source, origin="agent", note="bootstrap", rationale=""
    )

    reference = PythonReferenceEvaluator(
        _reference_module(tmp_path), context={"target": 3}
    )
    human_store = HumanSessionStore(system.run_dir)
    transport = LoopbackTransport()
    service = FeishuHumanSessionService(
        store=human_store, transport=transport, agent=FakeCoSessionAgent()
    )
    system.human_port = HumanProxySessionPort(
        service=service,
        proxy_agent=EvaluatorBackedHumanProxyAgent(reference),
    )
    proposal = tmp_path / "system_proposal"
    proposal.mkdir()
    (proposal / "verifier.py").write_text(proposed_source)

    system._apply_harden(proposal, trigger="proxy_test", verdict={"gaming": True})

    assert system.eval_service.current_version() == 1
    assert system.hardenings == 1
    assert human_store.opened_count == 1
    assert human_store.sessions()[0].state is SessionState.CLOSED
    assert human_store.outcome("session_001").decision == "approve"


def test_agent_system_preflight_builds_proxy_port_from_hidden_evaluator_config(
    tmp_path, monkeypatch
):
    from coscientist.coevo import agent_system as agent_system_module
    from coscientist.coevo.container import GatewayConfig

    raw = tmp_path / "raw_preflight"
    raw.mkdir()
    (raw / "instruction.md").write_text("target value is three")
    reference_path = _reference_module(tmp_path)
    context_path = tmp_path / "reference_context.json"
    context_path.write_text('{"target": 3}')
    system = AgentSystem(
        raw_input_dir=raw,
        run_dir=tmp_path / "preflight_run",
        human_proxy_evaluator_path=reference_path,
        human_proxy_evaluator_context_path=context_path,
    )
    system.gateway = GatewayConfig(codex_home=tmp_path / "fake_codex_home")
    system.agent_elf = tmp_path / "fake_codex"
    monkeypatch.setattr(agent_system_module, "docker_unavailable", lambda: None)

    system.preflight()

    assert isinstance(system.human_port, HumanProxySessionPort)
    assert system.human_port.proxy_agent.reference_evaluator.context == {"target": 3}
    assert system.human_port.service.agent.gateway.model == "gpt-5.6-luna"
    assert system.human_port.service.agent.gateway.reasoning_effort == "low"
