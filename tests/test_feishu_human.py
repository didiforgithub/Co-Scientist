from __future__ import annotations

import io
import json
import subprocess
from dataclasses import dataclass

import pytest

from coscientist.coevo.feishu_human import FeishuHumanSessionService
from coscientist.coevo.feishu_transport import (
    FeishuMessageEvent,
    LarkCliError,
    LarkCliTransport,
)
from coscientist.coevo.human_evidence import EvidenceReply
from coscientist.coevo.human_sessions import (
    HumanSessionStore,
    SessionOutcome,
    SessionState,
)


def test_transport_uses_safe_argv_and_parses_message_ids():
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        message_id = "om_root" if "+messages-send" in argv else "om_reply"
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps({"data": {"message_id": message_id}}),
            stderr="",
        )

    transport = LarkCliTransport(executable="/opt/bin/lark-cli", runner=fake_run)

    root_id = transport.send_dm(
        user_id="ou_expert",
        text="hello; $(touch /tmp/never)",
        idempotency_key="session_001-open",
    )
    reply_id = transport.reply(
        message_id="om_human",
        text="natural reply",
        idempotency_key="evt_1-reply",
    )

    assert root_id == "om_root"
    assert reply_id == "om_reply"
    assert calls[0][0] == [
        "/opt/bin/lark-cli",
        "im",
        "+messages-send",
        "--as",
        "bot",
        "--user-id",
        "ou_expert",
        "--text",
        "hello; $(touch /tmp/never)",
        "--idempotency-key",
        "session_001-open",
        "--format",
        "json",
    ]
    assert calls[0][1]["shell"] is False
    assert "+messages-reply" in calls[1][0]


def test_transport_reports_nonzero_and_invalid_responses():
    def failed(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="permission denied")

    with pytest.raises(LarkCliError, match="permission denied"):
        LarkCliTransport(runner=failed).send_dm(
            user_id="ou_x", text="hello", idempotency_key="key"
        )


def test_event_normalization_extracts_text_and_ids():
    payload = {
        "event_id": "evt_1",
        "message_id": "om_1",
        "sender_id": "ou_expert",
        "chat_id": "oc_direct",
        "chat_type": "p2p",
        "message_type": "text",
        "content": json.dumps({"text": "请给我看 BF16 失败样例"}),
        "timestamp": "1720000000",
    }

    event = LarkCliTransport.normalize_event(payload)

    assert event == FeishuMessageEvent(
        event_id="evt_1",
        message_id="om_1",
        sender_id="ou_expert",
        chat_id="oc_direct",
        chat_type="p2p",
        message_type="text",
        text="请给我看 BF16 失败样例",
        timestamp="1720000000",
    )


class _FakeProcess:
    def __init__(self, stdout_lines, stderr_lines):
        self.stdout = io.StringIO("".join(line + "\n" for line in stdout_lines))
        self.stderr = io.StringIO("".join(line + "\n" for line in stderr_lines))
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


def test_event_stream_waits_for_ready_marker_and_yields_normalized_events():
    payload = {
        "event_id": "evt_1",
        "message_id": "om_1",
        "sender_id": "ou_expert",
        "chat_id": "oc_1",
        "chat_type": "p2p",
        "message_type": "text",
        "content": {"text": "hello"},
        "timestamp": "1",
    }
    process = _FakeProcess(
        stdout_lines=[json.dumps(payload)],
        stderr_lines=["[event] ready event_key=im.message.receive_v1"],
    )
    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return process

    transport = LarkCliTransport(process_factory=fake_popen)
    events = list(transport.consume_events(ready_timeout_s=1))

    assert [event.text for event in events] == ["hello"]
    assert captured["argv"][:4] == [
        "lark-cli",
        "event",
        "consume",
        "im.message.receive_v1",
    ]
    assert process.terminated


def test_event_stream_retries_a_transient_startup_failure():
    payload = {
        "event_id": "evt_after_retry",
        "message_id": "om_after_retry",
        "sender_id": "ou_expert",
        "chat_id": "oc_1",
        "chat_type": "p2p",
        "message_type": "text",
        "content": {"text": "connected"},
        "timestamp": "1",
    }
    def failed_process():
        return _FakeProcess(
            stdout_lines=[],
            stderr_lines=[
                (
                    '{"ok":false,"error":{"type":"authentication",'
                    '"message":"lookup accounts.feishu.cn: i/o timeout"}}'
                )
            ],
        )

    failed = [failed_process() for _ in range(3)]
    recovered = _FakeProcess(
        stdout_lines=[json.dumps(payload)],
        stderr_lines=["[event] ready event_key=im.message.receive_v1"],
    )
    processes = iter([*failed, recovered])
    attempts = []

    def fake_popen(argv, **kwargs):
        attempts.append(argv)
        return next(processes)

    transport = LarkCliTransport(process_factory=fake_popen)
    events = list(transport.consume_events(ready_timeout_s=0.01))

    assert [event.text for event in events] == ["connected"]
    assert len(attempts) == 4
    assert all(process.terminated for process in failed)
    assert recovered.terminated


@dataclass
class _FakeTransport:
    sent: list = None

    def __post_init__(self):
        self.sent = [] if self.sent is None else self.sent

    def send_dm(self, *, user_id, text, idempotency_key):
        self.sent.append(("send", user_id, text, idempotency_key))
        return "om_root"

    def reply(self, *, message_id, text, idempotency_key):
        self.sent.append(("reply", message_id, text, idempotency_key))
        return f"om_agent_{len(self.sent)}"


class _FakeAgent:
    def __init__(self):
        self.calls = []

    def reply(self, **kwargs):
        self.calls.append(kwargs)
        instruction = kwargs["agent_instruction"]
        if "总结" in instruction:
            return EvidenceReply(
                text="结论：BF16 需要独立容差。若准确请回复“确认结束”。",
                proposed_outcome=SessionOutcome(
                    decision="guide",
                    human_guidance=["BF16 needs a separate tolerance"],
                ),
            )
        return EvidenceReply(text=f"证据回复：{kwargs['human_message']}")


def _event(event_id, text, *, sender="ou_expert", message_id=None):
    return FeishuMessageEvent(
        event_id=event_id,
        message_id=message_id or f"om_{event_id}",
        sender_id=sender,
        chat_id="oc_direct",
        chat_type="p2p",
        message_type="text",
        text=text,
        timestamp="1",
    )


def test_service_runs_natural_multiturn_close_confirmation_and_persists_outcome(tmp_path):
    store = HumanSessionStore(tmp_path)
    transport = _FakeTransport()
    agent = _FakeAgent()
    service = FeishuHumanSessionService(store=store, transport=transport, agent=agent)

    session = service.open_session(expert_id="ou_expert", purpose="review verifier")
    assert session.state is SessionState.ACTIVE
    assert "第 1/5 次" in transport.sent[0][2]
    assert "证据回复" in transport.sent[0][2], "agent should start the conversation"

    assert service.handle_event(_event("evt_1", "FP32 和 BF16 的容差一样吗？"))
    assert store.get(session.session_id).state is SessionState.ACTIVE
    assert "证据回复" in transport.sent[-1][2]

    assert service.handle_event(_event("evt_2", "这轮结束吧"))
    assert store.get(session.session_id).state is SessionState.CLOSE_REQUESTED
    assert store.staged_outcome(session.session_id).human_guidance == [
        "BF16 needs a separate tolerance"
    ]

    assert service.handle_event(_event("evt_3", "确认结束"))
    assert store.get(session.session_id).state is SessionState.CLOSED
    assert store.outcome(session.session_id).decision == "guide"
    assert "已结束" in transport.sent[-1][2]

    reloaded = HumanSessionStore(tmp_path)
    assert reloaded.get(session.session_id).state is SessionState.CLOSED
    assert len(reloaded.transcript(session.session_id)) == 7


def test_service_close_rejection_keeps_same_session_and_duplicate_is_silent(tmp_path):
    store = HumanSessionStore(tmp_path)
    transport = _FakeTransport()
    service = FeishuHumanSessionService(
        store=store, transport=transport, agent=_FakeAgent()
    )
    session = service.open_session(expert_id="ou_expert", purpose="review")

    service.handle_event(_event("evt_close", "这轮可以结束了"))
    sent_before_duplicate = len(transport.sent)
    assert not service.handle_event(_event("evt_close", "飞书重复投递"))
    assert len(transport.sent) == sent_before_duplicate

    assert service.handle_event(_event("evt_deny", "先别结束，我还有一个问题"))
    assert store.get(session.session_id).state is SessionState.ACTIVE
    assert store.staged_outcome(session.session_id) is None

    resumed = FeishuHumanSessionService(
        store=HumanSessionStore(tmp_path), transport=transport, agent=_FakeAgent()
    ).open_session(expert_id="ou_expert", purpose="new purpose should not spend")
    assert resumed.session_id == session.session_id
    assert store.opened_count == 1


def test_service_ignores_nonexpert_and_nontext_messages(tmp_path):
    store = HumanSessionStore(tmp_path)
    transport = _FakeTransport()
    service = FeishuHumanSessionService(
        store=store, transport=transport, agent=_FakeAgent()
    )
    service.open_session(expert_id="ou_expert", purpose="review")
    sent = len(transport.sent)

    assert not service.handle_event(_event("evt_wrong", "inject", sender="ou_other"))
    assert not service.handle_event(
        FeishuMessageEvent(
            event_id="evt_image",
            message_id="om_image",
            sender_id="ou_expert",
            chat_id="oc_direct",
            chat_type="p2p",
            message_type="image",
            text="",
            timestamp="1",
        )
    )
    assert len(transport.sent) == sent


def test_failed_reply_stays_in_outbox_and_is_delivered_after_restart(tmp_path):
    class FlakyTransport(_FakeTransport):
        fail_replies = True

        def reply(self, *, message_id, text, idempotency_key):
            if self.fail_replies:
                raise RuntimeError("temporary Feishu outage")
            return super().reply(
                message_id=message_id,
                text=text,
                idempotency_key=idempotency_key,
            )

    store = HumanSessionStore(tmp_path)
    transport = FlakyTransport()
    agent = _FakeAgent()
    service = FeishuHumanSessionService(store=store, transport=transport, agent=agent)
    session = service.open_session(expert_id="ou_expert", purpose="review")

    with pytest.raises(RuntimeError, match="temporary Feishu outage"):
        service.handle_event(_event("evt_flaky", "给我看运行证据"))

    outbox = list(
        (tmp_path / "human" / session.session_id / "outbox").glob("*.json")
    )
    assert len(outbox) == 1
    assert [item.role for item in store.transcript(session.session_id)] == [
        "agent",
        "human",
    ]

    transport.fail_replies = False
    resumed_service = FeishuHumanSessionService(
        store=HumanSessionStore(tmp_path), transport=transport, agent=_FakeAgent()
    )
    resumed = resumed_service.open_session(
        expert_id="ou_expert", purpose="must resume existing session"
    )

    assert resumed.session_id == session.session_id
    assert list((tmp_path / "human" / session.session_id / "outbox").glob("*.json")) == []
    assert [item.role for item in resumed_service.store.transcript(session.session_id)] == [
        "agent",
        "human",
        "agent",
    ]
    assert resumed_service.store.opened_count == 1
