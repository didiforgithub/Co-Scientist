"""Durable, transport-independent human expert conversations.

A *session* is the scarce unit: a run may open at most five sessions, while an
open session may contain any number of natural-language turns.  Transcripts and
state are written below ``<run_dir>/human`` so a crashed Feishu listener can
resume without spending another session.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class SessionBudgetExhausted(RuntimeError):
    """Raised when a run attempts to open more than its session allowance."""


class SessionState(str, Enum):
    OPEN = "open"
    ACTIVE = "active"
    CLOSE_REQUESTED = "close_requested"
    PAUSED = "paused"
    CLOSED = "closed"


class CloseAction(str, Enum):
    CONTINUE = "continue"
    REQUEST_CONFIRMATION = "request_confirmation"
    AWAIT_CONFIRMATION = "await_confirmation"
    CLOSED = "closed"
    IGNORED = "ignored"


@dataclass(frozen=True)
class SessionOutcome:
    decision: str = "guide"
    task_contract_updates: list[str] = field(default_factory=list)
    new_risks: list[str] = field(default_factory=list)
    approved_changes: list[str] = field(default_factory=list)
    rejected_changes: list[str] = field(default_factory=list)
    human_guidance: list[str] = field(default_factory=list)
    evidence_requested: list[str] = field(default_factory=list)
    evidence_generated: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    human_rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SessionOutcome:
        known = {item.name for item in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in payload.items() if key in known})


@dataclass(frozen=True)
class HumanSession:
    session_id: str
    expert_id: str
    purpose: str
    state: SessionState
    opened_at: str
    updated_at: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionMessage:
    sequence: int
    role: str
    text: str
    timestamp: str
    external_event_id: str | None = None
    external_message_id: str | None = None


@dataclass(frozen=True)
class HumanTurnResult:
    accepted: bool
    action: CloseAction
    state: SessionState
    reason: str = ""
    duplicate: bool = False
    agent_instruction: str = ""


_CLOSE_INTENT = (
    "结束这轮",
    "这轮结束",
    "可以结束",
    "没别的问题",
    "没有别的问题",
    "没有其他问题",
    "先到这里",
    "就到这里",
    "close this session",
    "end this session",
    "we are done",
)
_CLOSE_CONFIRM = (
    "确认结束",
    "确定结束",
    "可以结束",
    "结束吧",
    "是的结束",
    "对结束",
    "confirm close",
    "yes close",
    "yes end",
)
_CLOSE_DENY = (
    "别结束",
    "不结束",
    "不能结束",
    "还有问题",
    "等一下",
    "等等",
    "继续聊",
    "not yet",
    "do not close",
    "don't close",
    "keep talking",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized(text: str) -> str:
    return re.sub(r"[\s，。！？、,.!?;；:：'\"]+", "", text.strip().lower())


def _contains(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = _normalized(text)
    return any(_normalized(phrase) in normalized for phrase in phrases)


class HumanSessionStore:
    """Own session budget, lifecycle, transcript, deduplication, and outcomes."""

    def __init__(self, run_dir: Path | str, *, max_sessions: int = 5):
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        self.run_dir = Path(run_dir)
        self.root = self.run_dir / "human"
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_sessions = max_sessions
        self._lock = threading.RLock()
        self._index_path = self.root / "index.json"
        if not self._index_path.exists():
            self._atomic_json(
                self._index_path,
                {"max_sessions": max_sessions, "opened_count": 0, "session_ids": []},
            )
        else:
            index = self._load_json(self._index_path)
            persisted_max = int(index.get("max_sessions", max_sessions))
            if persisted_max != max_sessions:
                # Persisted accounting is authoritative on resume.  In particular,
                # restarting must never increase an exhausted run's allowance.
                self.max_sessions = persisted_max

    @property
    def opened_count(self) -> int:
        return int(self._index()["opened_count"])

    @property
    def remaining_count(self) -> int:
        return max(0, self.max_sessions - self.opened_count)

    def open_session(
        self,
        *,
        expert_id: str,
        purpose: str,
        context: dict[str, Any] | None = None,
    ) -> HumanSession:
        if not expert_id.strip():
            raise ValueError("expert_id is required")
        if not purpose.strip():
            raise ValueError("purpose is required")
        with self._lock:
            index = self._index()
            for session_id in index["session_ids"]:
                session = self.get(session_id)
                if session.expert_id == expert_id and session.state is not SessionState.CLOSED:
                    return session
            if int(index["opened_count"]) >= self.max_sessions:
                raise SessionBudgetExhausted(
                    f"human session budget exhausted ({self.max_sessions}/{self.max_sessions})"
                )
            sequence = int(index["opened_count"]) + 1
            session_id = f"session_{sequence:03d}"
            timestamp = _now()
            session = HumanSession(
                session_id=session_id,
                expert_id=expert_id,
                purpose=purpose,
                state=SessionState.OPEN,
                opened_at=timestamp,
                updated_at=timestamp,
                context=dict(context or {}),
            )
            session_dir = self.root / session_id
            session_dir.mkdir(parents=True, exist_ok=False)
            self._write_session(session)
            index["opened_count"] = sequence
            index["session_ids"].append(session_id)
            self._atomic_json(self._index_path, index)
            return session

    def get(self, session_id: str) -> HumanSession:
        payload = self._load_json(self._session_dir(session_id) / "session.json")
        payload["state"] = SessionState(payload["state"])
        return HumanSession(**payload)

    def activate(self, session_id: str) -> HumanSession:
        return self._transition(session_id, SessionState.ACTIVE)

    def pause(self, session_id: str) -> HumanSession:
        return self._transition(session_id, SessionState.PAUSED)

    def transcript(self, session_id: str) -> list[SessionMessage]:
        path = self._session_dir(session_id) / "transcript.jsonl"
        if not path.exists():
            return []
        messages: list[SessionMessage] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                messages.append(SessionMessage(**json.loads(line)))
        return messages

    def append_message(
        self,
        session_id: str,
        *,
        role: str,
        text: str,
        external_event_id: str | None = None,
        external_message_id: str | None = None,
    ) -> bool:
        if role not in {"human", "agent", "system"}:
            raise ValueError(f"unsupported message role: {role}")
        if not text.strip():
            raise ValueError("message text must not be empty")
        with self._lock:
            existing = self.transcript(session_id)
            if external_event_id and any(
                item.external_event_id == external_event_id for item in existing
            ):
                return False
            message = SessionMessage(
                sequence=len(existing) + 1,
                role=role,
                text=text,
                timestamp=_now(),
                external_event_id=external_event_id,
                external_message_id=external_message_id,
            )
            path = self._session_dir(session_id) / "transcript.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(asdict(message), ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            session = self.get(session_id)
            self._write_session(
                HumanSession(**{**asdict(session), "state": session.state, "updated_at": _now()})
            )
            return True

    def accept_human_message(
        self,
        session_id: str,
        *,
        sender_id: str,
        text: str,
        event_id: str,
        external_message_id: str | None = None,
        proposed_outcome: SessionOutcome | None = None,
    ) -> HumanTurnResult:
        with self._lock:
            session = self.get(session_id)
            if sender_id != session.expert_id:
                return HumanTurnResult(
                    accepted=False,
                    action=CloseAction.IGNORED,
                    state=session.state,
                    reason="sender_not_session_expert",
                )
            if session.state is SessionState.CLOSED:
                return HumanTurnResult(
                    accepted=False,
                    action=CloseAction.IGNORED,
                    state=session.state,
                    reason="session_closed",
                )
            appended = self.append_message(
                session_id,
                role="human",
                text=text,
                external_event_id=event_id,
                external_message_id=external_message_id,
            )
            if not appended:
                return HumanTurnResult(
                    accepted=False,
                    action=CloseAction.IGNORED,
                    state=self.get(session_id).state,
                    reason="duplicate_event",
                    duplicate=True,
                )
            session = self.get(session_id)
            if session.state in {SessionState.OPEN, SessionState.PAUSED}:
                session = self.activate(session_id)

            if session.state is SessionState.CLOSE_REQUESTED:
                if _contains(text, _CLOSE_DENY):
                    self.clear_staged_outcome(session_id)
                    session = self.activate(session_id)
                    return HumanTurnResult(
                        accepted=True,
                        action=CloseAction.CONTINUE,
                        state=session.state,
                        agent_instruction="专家决定继续本轮会话；直接回答其后续问题。",
                    )
                if _contains(text, _CLOSE_CONFIRM):
                    self.close(session_id, proposed_outcome or SessionOutcome())
                    return HumanTurnResult(
                        accepted=True,
                        action=CloseAction.CLOSED,
                        state=SessionState.CLOSED,
                        agent_instruction="专家已明确确认结束；发送简短结束回执。",
                    )
                return HumanTurnResult(
                    accepted=True,
                    action=CloseAction.AWAIT_CONFIRMATION,
                    state=SessionState.CLOSE_REQUESTED,
                    agent_instruction=(
                        "专家尚未明确确认结束。继续回答消息，并再次说明："
                        "若本轮信息已充分，请回复“确认结束”；否则可以直接继续提问。"
                    ),
                )

            if _contains(text, _CLOSE_INTENT):
                session = self._transition(session_id, SessionState.CLOSE_REQUESTED)
                return HumanTurnResult(
                    accepted=True,
                    action=CloseAction.REQUEST_CONFIRMATION,
                    state=session.state,
                    agent_instruction=(
                        "先用简短要点总结本轮结论和未决问题，然后询问："
                        "“以上总结是否准确？若本轮可以关闭，请回复‘确认结束’；"
                        "如需继续，直接补充问题即可。”"
                    ),
                )

            return HumanTurnResult(
                accepted=True,
                action=CloseAction.CONTINUE,
                state=session.state,
                agent_instruction="继续自然对话，并基于运行证据回答专家的问题。",
            )

    def close(self, session_id: str, outcome: SessionOutcome) -> HumanSession:
        with self._lock:
            session = self.get(session_id)
            if session.state is SessionState.CLOSED:
                existing = self.outcome(session_id)
                if existing is not None and existing != outcome:
                    raise ValueError("closed session already has a different outcome")
                return session
            self._atomic_json(
                self._session_dir(session_id) / "outcome.json", outcome.to_dict()
            )
            self.clear_staged_outcome(session_id)
            return self._transition(session_id, SessionState.CLOSED)

    def outcome(self, session_id: str) -> SessionOutcome | None:
        path = self._session_dir(session_id) / "outcome.json"
        if not path.exists():
            return None
        return SessionOutcome.from_dict(self._load_json(path))

    def stage_outcome(self, session_id: str, outcome: SessionOutcome) -> None:
        """Persist the agent's close summary until the expert confirms it."""
        with self._lock:
            session = self.get(session_id)
            if session.state is not SessionState.CLOSE_REQUESTED:
                raise ValueError("an outcome can only be staged while close is requested")
            self._atomic_json(
                self._session_dir(session_id) / "pending_outcome.json",
                outcome.to_dict(),
            )

    def staged_outcome(self, session_id: str) -> SessionOutcome | None:
        path = self._session_dir(session_id) / "pending_outcome.json"
        if not path.exists():
            return None
        return SessionOutcome.from_dict(self._load_json(path))

    def clear_staged_outcome(self, session_id: str) -> None:
        with self._lock:
            (self._session_dir(session_id) / "pending_outcome.json").unlink(
                missing_ok=True
            )

    def sessions(self) -> list[HumanSession]:
        return [self.get(session_id) for session_id in self._index()["session_ids"]]

    def live_session_for_expert(self, expert_id: str) -> HumanSession | None:
        for session in reversed(self.sessions()):
            if session.expert_id == expert_id and session.state is not SessionState.CLOSED:
                return session
        return None

    def _transition(self, session_id: str, state: SessionState) -> HumanSession:
        with self._lock:
            session = self.get(session_id)
            if session.state is SessionState.CLOSED and state is not SessionState.CLOSED:
                raise ValueError("a closed human session cannot be reopened")
            updated = HumanSession(
                **{**asdict(session), "state": state, "updated_at": _now()}
            )
            self._write_session(updated)
            return updated

    def _session_dir(self, session_id: str) -> Path:
        path = self.root / session_id
        if not path.is_dir():
            raise KeyError(f"unknown human session: {session_id}")
        return path

    def _index(self) -> dict[str, Any]:
        return self._load_json(self._index_path)

    def _write_session(self, session: HumanSession) -> None:
        payload = asdict(session)
        payload["state"] = session.state.value
        self._atomic_json(
            self.root / session.session_id / "session.json",
            payload,
        )

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
