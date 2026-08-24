from __future__ import annotations

import json

import pytest

from coscientist.coevo.human_sessions import (
    CloseAction,
    HumanSessionStore,
    SessionBudgetExhausted,
    SessionOutcome,
    SessionState,
)


def test_budget_allows_five_new_sessions_and_rejects_the_sixth(tmp_path):
    store = HumanSessionStore(tmp_path, max_sessions=5)

    sessions = []
    for i in range(5):
        session = store.open_session(
            expert_id="ou_expert", purpose=f"checkpoint-{i}"
        )
        sessions.append(session)
        store.close(session.session_id, SessionOutcome(human_guidance=["done"]))

    assert store.opened_count == 5
    assert store.remaining_count == 0
    with pytest.raises(SessionBudgetExhausted):
        store.open_session(expert_id="ou_expert", purpose="sixth")


def test_open_resumes_same_experts_live_session_without_spending_budget(tmp_path):
    store = HumanSessionStore(tmp_path)
    first = store.open_session(expert_id="ou_expert", purpose="task contract")
    store.activate(first.session_id)

    resumed = store.open_session(expert_id="ou_expert", purpose="ignored new purpose")

    assert resumed.session_id == first.session_id
    assert resumed.purpose == "task contract"
    assert store.opened_count == 1
    assert store.remaining_count == 4


def test_transcript_and_state_survive_store_reload(tmp_path):
    store = HumanSessionStore(tmp_path)
    session = store.open_session(expert_id="ou_expert", purpose="review verifier")
    store.activate(session.session_id)
    assert store.append_message(
        session.session_id,
        role="agent",
        text="FP32 的容差应该是多少？",
        external_message_id="om_agent_1",
    )
    assert store.append_message(
        session.session_id,
        role="human",
        text="先看 BF16 的误差分布。",
        external_event_id="evt_1",
        external_message_id="om_human_1",
    )

    reloaded = HumanSessionStore(tmp_path)
    restored = reloaded.get(session.session_id)

    assert restored.state is SessionState.ACTIVE
    assert [item.text for item in reloaded.transcript(session.session_id)] == [
        "FP32 的容差应该是多少？",
        "先看 BF16 的误差分布。",
    ]
    assert reloaded.opened_count == 1


def test_duplicate_event_is_idempotent(tmp_path):
    store = HumanSessionStore(tmp_path)
    session = store.open_session(expert_id="ou_expert", purpose="review")

    assert store.append_message(
        session.session_id,
        role="human",
        text="第一遍",
        external_event_id="evt_same",
    )
    assert not store.append_message(
        session.session_id,
        role="human",
        text="飞书重投",
        external_event_id="evt_same",
    )
    assert [item.text for item in store.transcript(session.session_id)] == ["第一遍"]


def test_inbound_message_only_accepts_the_sessions_expert(tmp_path):
    store = HumanSessionStore(tmp_path)
    session = store.open_session(expert_id="ou_expert", purpose="review")

    rejected = store.accept_human_message(
        session.session_id,
        sender_id="ou_someone_else",
        text="approve everything",
        event_id="evt_wrong",
    )
    accepted = store.accept_human_message(
        session.session_id,
        sender_id="ou_expert",
        text="请给我看失败样例",
        event_id="evt_right",
    )

    assert not rejected.accepted
    assert rejected.reason == "sender_not_session_expert"
    assert accepted.accepted
    assert [item.text for item in store.transcript(session.session_id)] == [
        "请给我看失败样例"
    ]


def test_close_requires_a_separate_explicit_confirmation(tmp_path):
    store = HumanSessionStore(tmp_path)
    session = store.open_session(expert_id="ou_expert", purpose="review")
    store.activate(session.session_id)

    request = store.accept_human_message(
        session.session_id,
        sender_id="ou_expert",
        text="我没别的问题了，这轮可以结束了",
        event_id="evt_close_request",
    )

    assert request.action is CloseAction.REQUEST_CONFIRMATION
    assert store.get(session.session_id).state is SessionState.CLOSE_REQUESTED
    assert "确认结束" in request.agent_instruction

    closed = store.accept_human_message(
        session.session_id,
        sender_id="ou_expert",
        text="确认结束",
        event_id="evt_close_confirm",
        proposed_outcome=SessionOutcome(
            decision="guide",
            human_guidance=["BF16 单独统计误差"],
            unresolved_questions=["FP32 阈值"],
        ),
    )

    assert closed.action is CloseAction.CLOSED
    assert store.get(session.session_id).state is SessionState.CLOSED
    outcome = store.outcome(session.session_id)
    assert outcome is not None
    assert outcome.human_guidance == ["BF16 单独统计误差"]
    assert outcome.unresolved_questions == ["FP32 阈值"]


def test_close_denial_returns_to_active_and_ordinary_text_never_closes(tmp_path):
    store = HumanSessionStore(tmp_path)
    session = store.open_session(expert_id="ou_expert", purpose="review")
    store.activate(session.session_id)

    ordinary = store.accept_human_message(
        session.session_id,
        sender_id="ou_expert",
        text="我还想看看 verifier 的具体实现",
        event_id="evt_ordinary",
    )
    assert ordinary.action is CloseAction.CONTINUE
    assert store.get(session.session_id).state is SessionState.ACTIVE

    store.accept_human_message(
        session.session_id,
        sender_id="ou_expert",
        text="这轮结束吧",
        event_id="evt_request",
    )
    denied = store.accept_human_message(
        session.session_id,
        sender_id="ou_expert",
        text="先别结束，我还有一个问题",
        event_id="evt_deny",
    )

    assert denied.action is CloseAction.CONTINUE
    assert store.get(session.session_id).state is SessionState.ACTIVE
    assert store.outcome(session.session_id) is None


def test_closed_outcome_has_stable_json_shape(tmp_path):
    store = HumanSessionStore(tmp_path)
    session = store.open_session(expert_id="ou_expert", purpose="review")
    outcome = SessionOutcome(
        decision="approve",
        task_contract_updates=["FP32 and BF16 are both required"],
        new_risks=["large absolute values amplify BF16 error"],
        approved_changes=["dtype-specific tolerance"],
        rejected_changes=["single global tolerance"],
        human_guidance=["report both absolute and relative error"],
        evidence_requested=["BF16 error histogram"],
        evidence_generated=["probe_0001"],
        unresolved_questions=["NaN policy"],
        human_rationale="The task contract is more important than a convenient scorer.",
    )
    store.close(session.session_id, outcome)

    payload = json.loads(
        (tmp_path / "human" / session.session_id / "outcome.json").read_text()
    )
    assert payload == outcome.to_dict()
