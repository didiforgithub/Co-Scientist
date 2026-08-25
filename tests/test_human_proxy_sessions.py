from __future__ import annotations

import json

import pytest

from coscientist.coevo.feishu_human import FeishuHumanSessionService
from coscientist.coevo.human_evidence import EvidenceReply
from coscientist.coevo.human_proxy_sessions import (
    HumanProxyConversation,
    HumanProxySessionPort,
    HumanProxyTurn,
    LoopbackTransport,
)
from coscientist.coevo.human_sessions import (
    HumanSessionStore,
    SessionOutcome,
    SessionState,
)


FINAL_OUTCOME = SessionOutcome(
    decision="guide",
    human_guidance=["先让环境返回失败类别，再优化性能。"],
    evidence_requested=["mask、数值稳定性和大规模性能 probe"],
    human_rationale="Human Proxy 与 Co-side 共同识别出总分反馈无法指导 Solver。",
)


class FakeCoSessionAgent:
    def reply(self, **kwargs):
        instruction = kwargs["agent_instruction"]
        if "会话开场" in instruction:
            return EvidenceReply(
                text="当前反馈只有总分。你认为怎样设计环境，能让 Solver 更快定位错误？"
            )
        if "总结" in instruction:
            return EvidenceReply(
                text="总结：增加失败分类与针对性 probe。若准确请确认结束。",
                proposed_outcome=SessionOutcome(decision="none"),
            )
        return EvidenceReply(text="收到。我会把这个建议转化为 evaluator 和环境设计。")


class NaturalDialogueAgent:
    """Free dialogue: ask, request close, reject close, continue, then confirm."""

    def __init__(self, store: HumanSessionStore):
        self.store = store
        self.frozen_inputs: list[SessionOutcome | None] = []

    def start_conversation(self, *, purpose, frozen_outcome=None):
        del purpose
        self.frozen_inputs.append(frozen_outcome)

        def next_turn(*, session, transcript):
            human_count = sum(item.role == "human" for item in transcript)
            if session.state is SessionState.CLOSE_REQUESTED:
                if human_count == 2:
                    return HumanProxyTurn(
                        "先别结束，我还想确认：失败分类会不会泄露隐藏样例？"
                    )
                return HumanProxyTurn("确认结束", outcome=FINAL_OUTCOME)
            if human_count == 0:
                return HumanProxyTurn(
                    "先解释总分由哪些失败模式混合而成，我们一起设计 probe。"
                )
            if human_count == 1:
                return HumanProxyTurn(
                    "这个方向基本明确，请总结后结束这轮。",
                    outcome=FINAL_OUTCOME,
                )
            if human_count == 3:
                assert self.store.opened_count == 1
                assert self.store.remaining_count == 4
                return HumanProxyTurn("明白了。请把保密边界也写进反馈契约。")
            if human_count == 4:
                return HumanProxyTurn(
                    "现在信息充分，结束这轮。",
                    outcome=FINAL_OUTCOME,
                )
            raise AssertionError(f"unexpected proxy state: {session.state}/{human_count}")

        return HumanProxyConversation(next_turn=next_turn)


class ClosingAgent:
    def start_conversation(self, *, purpose, frozen_outcome=None):
        del purpose

        def next_turn(*, session, transcript):
            del transcript
            if session.state is SessionState.CLOSE_REQUESTED:
                return HumanProxyTurn(
                    "确认结束", outcome=frozen_outcome or FINAL_OUTCOME
                )
            return HumanProxyTurn(
                "建议已明确，结束这轮。",
                outcome=frozen_outcome or FINAL_OUTCOME,
            )

        return HumanProxyConversation(next_turn=next_turn)


class FailOnceAgent:
    def __init__(self):
        self.failed = False
        self.frozen_inputs: list[SessionOutcome | None] = []

    def start_conversation(self, *, purpose, frozen_outcome=None):
        del purpose
        self.frozen_inputs.append(frozen_outcome)
        if not self.failed:
            self.failed = True

            def fail(**kwargs):
                del kwargs
                raise ValueError("synthetic model failure")

            return HumanProxyConversation(next_turn=fail)
        return ClosingAgent().start_conversation(
            purpose="resume", frozen_outcome=frozen_outcome
        )


class EndlessAgent:
    def start_conversation(self, *, purpose, frozen_outcome=None):
        del purpose, frozen_outcome

        def next_turn(*, session, transcript):
            del session
            count = sum(item.role == "human" for item in transcript)
            return HumanProxyTurn(f"继续共同设计第 {count + 1} 个环境 probe。")

        return HumanProxyConversation(next_turn=next_turn)


def _port(tmp_path, agent, *, max_turns=64):
    store = HumanSessionStore(tmp_path / "run")
    transport = LoopbackTransport()
    service = FeishuHumanSessionService(
        store=store,
        transport=transport,
        agent=FakeCoSessionAgent(),
    )
    port = HumanProxySessionPort(
        service=service,
        proxy_agent=agent,
        expert_id="human_proxy_agent",
        max_turns_per_consult=max_turns,
    )
    return port, store, transport


def test_proxy_uses_same_free_multi_turn_contract_and_can_refuse_close(tmp_path):
    store = HumanSessionStore(tmp_path / "run")
    agent = NaturalDialogueAgent(store)
    transport = LoopbackTransport()
    port = HumanProxySessionPort(
        service=FeishuHumanSessionService(
            store=store,
            transport=transport,
            agent=FakeCoSessionAgent(),
        ),
        proxy_agent=agent,
        expert_id="human_proxy_agent",
    )

    outcome = port.consult(
        purpose="verifier_change",
        context={"solver_workspace_secret": "never passed to proxy policy"},
    )

    session = store.sessions()[0]
    transcript = store.transcript(session.session_id)
    assert session.state is SessionState.CLOSED
    assert outcome == FINAL_OUTCOME
    assert store.outcome(session.session_id) == FINAL_OUTCOME
    assert store.opened_count == 1
    assert store.remaining_count == 4
    assert "先别结束" in transcript[5].text
    assert any("保密边界" in item.text for item in transcript)
    assert len([item for item in transcript if item.role == "human"]) == 6
    assert len(transport.sent) == 7


def test_proxy_failure_pauses_and_resume_reuses_same_session(tmp_path):
    agent = FailOnceAgent()
    port, store, _ = _port(tmp_path, agent)

    with pytest.raises(RuntimeError, match="Human Proxy driver failed"):
        port.consult(purpose="verifier_change", context={})

    assert store.sessions()[0].state is SessionState.PAUSED
    assert store.opened_count == 1

    outcome = port.consult(purpose="verifier_change", context={})

    assert outcome == FINAL_OUTCOME
    assert store.sessions()[0].state is SessionState.CLOSED
    assert store.opened_count == 1
    assert agent.frozen_inputs == [None, None]


def test_proxy_watchdog_pauses_one_consult_without_spending_another_session(
    tmp_path,
):
    port, store, _ = _port(tmp_path, EndlessAgent(), max_turns=2)

    with pytest.raises(RuntimeError, match="watchdog reached 2 turns"):
        port.consult(purpose="verifier_change", context={})

    assert store.sessions()[0].state is SessionState.PAUSED
    assert store.opened_count == 1
    first_length = len(store.transcript("session_001"))

    with pytest.raises(RuntimeError, match="watchdog reached 2 turns"):
        port.consult(purpose="verifier_change", context={})

    assert store.sessions()[0].state is SessionState.PAUSED
    assert store.opened_count == 1
    assert len(store.transcript("session_001")) == first_length + 4


def test_proxy_has_the_same_five_session_budget_as_a_human(tmp_path):
    port, store, _ = _port(tmp_path, ClosingAgent())

    outcomes = [
        port.consult(purpose=f"checkpoint_{index}", context={})
        for index in range(6)
    ]

    assert outcomes[:5] == [FINAL_OUTCOME] * 5
    assert outcomes[5] is None
    assert store.opened_count == 5
    assert store.remaining_count == 0


def test_final_outcome_is_frozen_before_confirmation_and_reused_on_resume(tmp_path):
    original = SessionOutcome(
        decision="guide",
        human_guidance=["original, already durable"],
        human_rationale="first confirmation attempt",
    )
    changed = SessionOutcome(
        decision="approve",
        approved_changes=["must not replace durable outcome"],
    )

    class ResumeWithChangedJudgement:
        def start_conversation(self, *, purpose, frozen_outcome=None):
            del purpose
            assert frozen_outcome == original

            def next_turn(*, session, transcript):
                del session, transcript
                return HumanProxyTurn("确认结束", outcome=changed)

            return HumanProxyConversation(next_turn=next_turn)

    port, store, _ = _port(tmp_path, ResumeWithChangedJudgement())
    session = port.service.open_session(
        expert_id="human_proxy_agent",
        purpose="verifier_change",
        context={},
    )
    store._transition(session.session_id, SessionState.CLOSE_REQUESTED)
    state_path = store.root / session.session_id / "proxy_state.json"
    state_path.write_text(
        json.dumps({"outcome": original.to_dict()}, ensure_ascii=False),
        encoding="utf-8",
    )

    outcome = port.consult(purpose="verifier_change", context={})

    assert outcome == original
    assert store.outcome(session.session_id) == original
