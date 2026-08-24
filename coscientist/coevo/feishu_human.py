"""Feishu-facing orchestration for durable Human Sessions."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from .feishu_transport import FeishuMessageEvent
from .human_evidence import EvidenceAgent, EvidenceAgentError
from .human_sessions import (
    CloseAction,
    HumanSession,
    HumanSessionStore,
    SessionBudgetExhausted,
    SessionOutcome,
    SessionState,
)


class FeishuTransport(Protocol):
    def send_dm(self, *, user_id: str, text: str, idempotency_key: str) -> str: ...

    def reply(self, *, message_id: str, text: str, idempotency_key: str) -> str: ...

    def consume_events(self): ...


class BlockingFeishuHumanPort:
    """The AgentSystem port: freeze its checkpoint until the expert closes the DM."""

    def __init__(
        self,
        *,
        service: FeishuHumanSessionService,
        transport: FeishuTransport,
        expert_id: str,
    ):
        self.service = service
        self.transport = transport
        self.expert_id = expert_id

    def consult(self, *, purpose: str, context: dict) -> SessionOutcome | None:
        try:
            session = self.service.open_session(
                expert_id=self.expert_id,
                purpose=purpose,
                context=context,
            )
        except SessionBudgetExhausted:
            return None
        events = self.transport.consume_events()
        try:
            for event in events:
                self.service.handle_event(event)
                session = self.service.store.get(session.session_id)
                if session.state is SessionState.CLOSED:
                    return self.service.store.outcome(session.session_id)
        except BaseException:
            session = self.service.store.get(session.session_id)
            if session.state is not SessionState.CLOSED:
                self.service.store.pause(session.session_id)
            raise
        finally:
            close = getattr(events, "close", None)
            if callable(close):
                close()
        # A clean end without an explicit human close is not approval.  Persist the
        # resumable state and stop the Co run rather than mutating its evaluator.
        self.service.store.pause(session.session_id)
        raise RuntimeError("Feishu event stream ended before the human closed the session")


class FeishuHumanSessionService:
    """Route Feishu DMs into one natural-language expert session at a time."""

    def __init__(
        self,
        *,
        store: HumanSessionStore,
        transport: FeishuTransport,
        agent: EvidenceAgent,
    ):
        self.store = store
        self.transport = transport
        self.agent = agent

    def open_session(
        self,
        *,
        expert_id: str,
        purpose: str,
        context: dict | None = None,
    ) -> HumanSession:
        session = self.store.open_session(
            expert_id=expert_id,
            purpose=purpose,
            context=context,
        )
        if session.state is SessionState.OPEN:
            opening_path = self.store.root / session.session_id / "opening.json"
            if opening_path.is_file():
                welcome = json.loads(opening_path.read_text(encoding="utf-8"))["text"]
            else:
                purpose_label = {
                    "task_definition": "确认这次任务的真实目标、约束和验收方式",
                    "verifier_change": "检查 Co-Scientist 准备采用的一次评估规则变更",
                }.get(session.purpose, session.purpose)
                try:
                    opener = self.agent.reply(
                        session=session,
                        transcript=[],
                        human_message="（系统正在发起本轮 Human Session，尚无人类消息。）",
                        agent_instruction=(
                            "这是会话开场。结合现有证据，用两三句话说明当前理解、"
                            "尚未确定的关键点，并主动提出最值得专家澄清的一到三个问题。"
                            "不要要求专家使用命令。"
                        ),
                    ).text
                except EvidenceAgentError:
                    opener = "我会基于当前运行证据回答问题；如果证据不足，我会明确说明。"
                welcome = (
                    f"Co-Scientist 想和你讨论：{purpose_label}\n\n{opener}\n\n"
                    f"这是本次运行的第 {self.store.opened_count}/{self.store.max_sessions} 次"
                    "人类会话。你可以像平常聊天一样连续提问或补充信息，不限消息条数。"
                    "当你认为这一轮信息已经充分时，直接说“结束这轮”即可；我会先总结，"
                    "只有你再次确认后才真正关闭。"
                )
                self._atomic_json(opening_path, {"text": welcome})
            message_id = self.transport.send_dm(
                user_id=expert_id,
                text=welcome,
                idempotency_key=f"{session.session_id}-open",
            )
            self.store.append_message(
                session.session_id,
                role="agent",
                text=welcome,
                external_event_id=f"outbound:{session.session_id}:open",
                external_message_id=message_id,
            )
            session = self.store.activate(session.session_id)
            opening_path.unlink(missing_ok=True)
        self.flush_outbox(session.session_id)
        return self.store.get(session.session_id)

    def handle_event(self, event: FeishuMessageEvent) -> bool:
        if event.message_type != "text" or not event.text.strip():
            return False
        session = self.store.live_session_for_expert(event.sender_id)
        if session is None:
            return False

        # Recover an answer that was generated and persisted before an earlier send
        # failed.  The lark idempotency key makes replay safe.
        self.flush_outbox(session.session_id)
        staged = self.store.staged_outcome(session.session_id)
        turn = self.store.accept_human_message(
            session.session_id,
            sender_id=event.sender_id,
            text=event.text,
            event_id=event.event_id,
            external_message_id=event.message_id,
            proposed_outcome=staged,
        )
        if not turn.accepted:
            return False

        if turn.action is CloseAction.CLOSED:
            acknowledgement = (
                "好的，本轮 Human Session 已结束，结论和依据已经写入运行记录。"
                f"本次运行还剩 {self.store.remaining_count} 次可发起会话。"
            )
            self._durable_reply(session.session_id, event, acknowledgement)
            return True

        try:
            evidence_reply = self.agent.reply(
                session=self.store.get(session.session_id),
                transcript=self.store.transcript(session.session_id),
                human_message=event.text,
                agent_instruction=turn.agent_instruction,
            )
            reply_text = evidence_reply.text
            if turn.action in {
                CloseAction.REQUEST_CONFIRMATION,
                CloseAction.AWAIT_CONFIRMATION,
            }:
                proposal = evidence_reply.proposed_outcome or SessionOutcome(
                    decision="none",
                    unresolved_questions=[
                        "会话 Agent 未生成结构化结论；关闭前需以对话总结为准。"
                    ],
                    human_rationale=reply_text,
                )
                self.store.stage_outcome(session.session_id, proposal)
        except EvidenceAgentError as exc:
            reply_text = (
                "我暂时没能读取运行证据，但本轮会话仍然保持开启。"
                f"你可以继续发消息，或稍后重试。错误摘要：{str(exc)[-160:]}"
            )
        self._durable_reply(session.session_id, event, reply_text)
        return True

    def serve(self, events: Iterable[FeishuMessageEvent]) -> None:
        """Consume events until the transport iterator exits or is interrupted."""
        for session in self.store.sessions():
            if session.state is not SessionState.CLOSED:
                self.flush_outbox(session.session_id)
        for event in events:
            self.handle_event(event)

    def _durable_reply(
        self,
        session_id: str,
        event: FeishuMessageEvent,
        text: str,
    ) -> None:
        outbox = self.store.root / session_id / "outbox"
        outbox.mkdir(parents=True, exist_ok=True)
        safe_event_name = hashlib.sha256(event.event_id.encode("utf-8")).hexdigest()
        path = outbox / f"{safe_event_name}.json"
        payload = {
            "event_id": event.event_id,
            "reply_to_message_id": event.message_id,
            "text": text,
            "idempotency_key": f"{session_id}-{event.event_id}-reply",
        }
        self._atomic_json(path, payload)
        self._deliver_outbox_file(session_id, path)

    def flush_outbox(self, session_id: str) -> None:
        outbox = self.store.root / session_id / "outbox"
        if not outbox.is_dir():
            return
        for path in sorted(outbox.glob("*.json")):
            self._deliver_outbox_file(session_id, path)

    def _deliver_outbox_file(self, session_id: str, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        message_id = self.transport.reply(
            message_id=payload["reply_to_message_id"],
            text=payload["text"],
            idempotency_key=payload["idempotency_key"],
        )
        self.store.append_message(
            session_id,
            role="agent",
            text=payload["text"],
            external_event_id=f"outbound:{payload['event_id']}",
            external_message_id=message_id,
        )
        path.unlink(missing_ok=True)

    @staticmethod
    def _atomic_json(path: Path, payload: dict) -> None:
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
