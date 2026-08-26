"""Model-backed Human Proxy using the same durable contract as a human expert.

The proxy receives private evaluator *context as text*. It has no reference
evaluator callable, solution runner, Solver workspace, or GPU environment.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import stat
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Protocol

from .container import AgentSession, GatewayConfig, one_shot_agent
from .feishu_human import FeishuHumanSessionService
from .feishu_transport import FeishuMessageEvent
from .human_sessions import (
    HumanSession,
    SessionMessage,
    SessionBudgetExhausted,
    SessionOutcome,
    SessionState,
)

@dataclass(frozen=True)
class HumanProxyTurn:
    """One natural-language turn chosen by the expert-side dialogue agent."""

    text: str
    outcome: SessionOutcome | None = None


class HumanProxyTurnPolicy(Protocol):
    def __call__(
        self, *, session: HumanSession, transcript: list[SessionMessage]
    ) -> HumanProxyTurn: ...


@dataclass(frozen=True)
class HumanProxyConversation:
    """A live expert policy reconstructed from the durable transcript."""

    next_turn: HumanProxyTurnPolicy
    outcome: SessionOutcome | None = None


class HumanProxyAgent(Protocol):
    """Independent expert agent driven only through the public chat contract."""

    def start_conversation(
        self,
        *,
        purpose: str,
        frozen_outcome: SessionOutcome | None = None,
    ) -> HumanProxyConversation: ...


ProxyRunner = Callable[..., AgentSession]
MAX_TURN_JSON_BYTES = 256 * 1024

HUMAN_PROXY_E_RUN_FAILED = "HUMAN_PROXY_E_RUN_FAILED"
HUMAN_PROXY_E_TURN_MISSING = "HUMAN_PROXY_E_TURN_MISSING"
HUMAN_PROXY_E_TURN_TOO_LARGE = "HUMAN_PROXY_E_TURN_TOO_LARGE"
HUMAN_PROXY_E_TURN_INVALID = "HUMAN_PROXY_E_TURN_INVALID"
HUMAN_PROXY_E_PRIVATE_CONTEXT_LEAK = "HUMAN_PROXY_E_PRIVATE_CONTEXT_LEAK"
HUMAN_PROXY_E_DRIVER_FAILED = "HUMAN_PROXY_E_DRIVER_FAILED"
MAX_PRIVACY_LEAVES = 10_000
MAX_PRIVACY_NODES = 100_000
MAX_PRIVACY_CANDIDATES = 10_000
MAX_PRIVACY_CANDIDATE_CHARACTERS = 1_000_000


@dataclass(frozen=True)
class PrivateContextLeak:
    """Location of a forbidden private-context span, without echoing the secret."""

    location: str
    matched_characters: int


def _normalized_private_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _collect_payload_leaves(value: Any) -> tuple[list[str], list[str], bool]:
    leaves = []
    container_encoded_aggregates = []
    leaf_characters = 0
    aggregate_characters = 0
    nodes = 0
    stack = [value]
    while stack:
        current = stack.pop()
        nodes += 1
        if nodes > MAX_PRIVACY_NODES:
            return leaves, container_encoded_aggregates, True
        if isinstance(current, str):
            leaves.append(current)
            leaf_characters += len(current)
            if (
                len(leaves) > MAX_PRIVACY_LEAVES
                or leaf_characters > MAX_PRIVACY_CANDIDATE_CHARACTERS
            ):
                return leaves, container_encoded_aggregates, True
        elif isinstance(current, dict):
            stack.extend(reversed(tuple(current.values())))
        elif isinstance(current, (list, tuple)):
            if len(current) > MAX_PRIVACY_LEAVES:
                return leaves, container_encoded_aggregates, True
            if len(current) >= 2 and all(
                isinstance(child, str)
                and bool(child)
                and re.fullmatch(r"[A-Za-z0-9+/_=-]+", child) is not None
                for child in current
            ):
                container_size = sum(len(child) for child in current)
                aggregate_characters += container_size
                if aggregate_characters > MAX_PRIVACY_CANDIDATE_CHARACTERS:
                    return leaves, container_encoded_aggregates, True
                container_encoded_aggregates.append("".join(current))
            stack.extend(reversed(current))
    return leaves, container_encoded_aggregates, False


def _decoded_text_candidates(value: str) -> Iterator[str]:
    compact = "".join(value.split())
    if len(compact) >= 2 and len(compact) % 2 == 0 and re.fullmatch(
        r"[0-9a-fA-F]+", compact
    ):
        try:
            yield bytes.fromhex(compact).decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            pass
    if len(compact) >= 4 and re.fullmatch(r"[A-Za-z0-9+/_-]*={0,2}", compact):
        try:
            padded = compact + "=" * (-len(compact) % 4)
            yield base64.b64decode(
                padded, altchars=b"-_", validate=True
            ).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            pass


def scan_private_context_leaks(
    payload: Any,
    private_context: str,
    *,
    min_span_characters: int = 160,
    forbidden_literals: Iterable[str] = (),
) -> list[PrivateContextLeak]:
    """Find normalized verbatim context spans in nested model or audit payloads.

    The return value intentionally identifies only the payload location and span
    length, so logging a finding cannot itself disclose private evaluator text.
    Short exact literals (for example host context paths) can be supplied by a
    later filesystem audit through ``forbidden_literals``.
    """

    if min_span_characters < 1:
        raise ValueError("min_span_characters must be positive")
    normalized_private = _normalized_private_text(private_context)
    normalized_literals = tuple(
        normalized
        for literal in forbidden_literals
        if (normalized := _normalized_private_text(literal))
    )
    leaves, container_encoded_aggregates, exceeded = _collect_payload_leaves(payload)
    if exceeded:
        return [PrivateContextLeak(location="budget", matched_characters=0)]

    candidates: list[tuple[str, str]] = []
    normalized_seen = set()
    candidate_characters = 0

    def add_candidate(location: str, raw_text: str) -> bool:
        nonlocal candidate_characters
        normalized = _normalized_private_text(raw_text)
        if not normalized or normalized in normalized_seen:
            return True
        normalized_seen.add(normalized)
        candidate_characters += len(raw_text)
        if (
            len(candidates) >= MAX_PRIVACY_CANDIDATES
            or candidate_characters > MAX_PRIVACY_CANDIDATE_CHARACTERS
        ):
            return False
        candidates.append((location, raw_text))
        return True

    for leaf in leaves:
        if not add_candidate("leaf", leaf):
            return [PrivateContextLeak(location="budget", matched_characters=0)]
    if len(leaves) > 1:
        for aggregate in ("".join(leaves), " ".join(leaves)):
            if not add_candidate("root_aggregate", aggregate):
                return [PrivateContextLeak(location="budget", matched_characters=0)]
        encoded_leaves = [
            leaf
            for leaf in leaves
            if len(leaf) >= 16
            and re.fullmatch(r"[A-Za-z0-9+/_=-]+", leaf)
        ]
        if len(encoded_leaves) > 1 and not add_candidate(
            "root_encoded_aggregate", "".join(encoded_leaves)
        ):
            return [PrivateContextLeak(location="budget", matched_characters=0)]
    for aggregate in container_encoded_aggregates:
        if not add_candidate("container_encoded_aggregate", aggregate):
            return [PrivateContextLeak(location="budget", matched_characters=0)]
    if len(container_encoded_aggregates) > 1 and not add_candidate(
        "global_encoded_container_aggregate",
        "".join(container_encoded_aggregates),
    ):
        return [PrivateContextLeak(location="budget", matched_characters=0)]

    encoded_sources = tuple(candidates)
    for _location, raw_text in encoded_sources:
        for decoded in _decoded_text_candidates(raw_text):
            if not add_candidate("decoded", decoded):
                return [PrivateContextLeak(location="budget", matched_characters=0)]

    if len(normalized_private) >= min_span_characters:
        private_windows = {
            normalized_private[start : start + min_span_characters]
            for start in range(len(normalized_private) - min_span_characters + 1)
        }
    else:
        private_windows = set()
    for location, raw_text in candidates:
        normalized = _normalized_private_text(raw_text)
        matched = 0
        if len(normalized) >= min_span_characters and any(
            normalized[start : start + min_span_characters] in private_windows
            for start in range(len(normalized) - min_span_characters + 1)
        ):
            matched = min_span_characters
        if not matched:
            matched = max(
                (
                    len(literal)
                    for literal in normalized_literals
                    if literal in normalized
                ),
                default=0,
            )
        if matched:
            return [
                PrivateContextLeak(
                    location=location,
                    matched_characters=matched,
                )
            ]
    return []


def assert_no_private_context_leak(
    payload: Any,
    private_context: str,
    *,
    min_span_characters: int = 160,
    forbidden_literals: Iterable[str] = (),
) -> None:
    """Reject a payload without including private text in the exception."""

    findings = scan_private_context_leaks(
        payload,
        private_context,
        min_span_characters=min_span_characters,
        forbidden_literals=forbidden_literals,
    )
    if findings:
        raise RuntimeError(HUMAN_PROXY_E_PRIVATE_CONTEXT_LEAK) from None


@dataclass
class ModelBackedHumanProxyAgent:
    """A human-like expert agent with private evaluator *context*, not execution.

    The agent gets no Solver payload, run workspace, evaluator callable, GPU, or
    evaluator endpoint from this class.  Every turn is reconstructed from a private
    text context plus the public Human Session transcript in a throw-away workspace.
    """

    evaluator_context: str
    gateway: GatewayConfig
    agent_elf: Path
    image: str = "python:3.11-slim"
    timeout_s: float = 180.0
    runner: ProxyRunner | None = None

    def __post_init__(self) -> None:
        self.evaluator_context = self.evaluator_context.strip()
        if not self.evaluator_context:
            raise ValueError("Human Proxy evaluator context must not be empty")
        if len(self.evaluator_context) > 200_000:
            raise ValueError("Human Proxy evaluator context exceeds 200000 characters")
        self.agent_elf = Path(self.agent_elf)

    def start_conversation(
        self,
        *,
        purpose: str,
        frozen_outcome: SessionOutcome | None = None,
    ) -> HumanProxyConversation:
        # A real human learns run details only through Co-side chat messages.
        # The port never passes the orchestrator's private session context here.
        del frozen_outcome

        def next_turn(
            *, session: HumanSession, transcript: list[SessionMessage]
        ) -> HumanProxyTurn:
            return self._next_turn(
                purpose=purpose,
                session=session,
                transcript=transcript,
            )

        return HumanProxyConversation(next_turn=next_turn)

    def _next_turn(
        self,
        *,
        purpose: str,
        session: HumanSession,
        transcript: list[SessionMessage],
    ) -> HumanProxyTurn:
        with tempfile.TemporaryDirectory(prefix="coscientist-human-proxy-") as raw:
            workspace = Path(raw)
            (workspace / "evaluator_context.md").write_text(
                self.evaluator_context, encoding="utf-8"
            )
            (workspace / "transcript.json").write_text(
                json.dumps(
                    [
                        {
                            "sequence": item.sequence,
                            "role": item.role,
                            "text": item.text,
                            "timestamp": item.timestamp,
                        }
                        for item in transcript
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            try:
                result = self._run(
                    workspace,
                    self._prompt(purpose=purpose, state=session.state),
                )
            except Exception:
                raise RuntimeError(HUMAN_PROXY_E_RUN_FAILED) from None
            if not result.ok:
                raise RuntimeError(HUMAN_PROXY_E_RUN_FAILED) from None
            turn_path = workspace / "turn.json"
            try:
                turn_stat = turn_path.lstat()
            except OSError:
                raise RuntimeError(HUMAN_PROXY_E_TURN_MISSING) from None
            if stat.S_ISLNK(turn_stat.st_mode) or not stat.S_ISREG(turn_stat.st_mode):
                raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None
            if turn_stat.st_size > MAX_TURN_JSON_BYTES:
                raise RuntimeError(HUMAN_PROXY_E_TURN_TOO_LARGE) from None
            try:
                with turn_path.open("rb") as stream:
                    turn_raw = stream.read(MAX_TURN_JSON_BYTES + 1)
            except OSError:
                raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None
            if len(turn_raw) > MAX_TURN_JSON_BYTES:
                raise RuntimeError(HUMAN_PROXY_E_TURN_TOO_LARGE) from None
            try:
                payload = json.loads(turn_raw.decode("utf-8"))
            except Exception:
                raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None
            try:
                assert_no_private_context_leak(payload, self.evaluator_context)
            except RuntimeError as exc:
                if str(exc) == HUMAN_PROXY_E_PRIVATE_CONTEXT_LEAK:
                    raise RuntimeError(HUMAN_PROXY_E_PRIVATE_CONTEXT_LEAK) from None
                raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None
            except Exception:
                raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None
        try:
            return self._parse_turn(payload, state=session.state)
        except RuntimeError as exc:
            if str(exc) == HUMAN_PROXY_E_TURN_INVALID:
                raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None
            raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None
        except Exception:
            raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None

    def _run(self, workspace: Path, prompt: str) -> AgentSession:
        if self.runner is not None:
            return self.runner(
                workspace,
                prompt,
                gateway=self.gateway,
                agent_elf=self.agent_elf,
                timeout_s=self.timeout_s,
                image=self.image,
            )
        return one_shot_agent(
            workspace,
            self.gateway,
            self.agent_elf,
            prompt,
            timeout_s=self.timeout_s,
            image=self.image,
            disallowed_tools=["network"],
        )

    @staticmethod
    def _prompt(*, purpose: str, state: SessionState) -> str:
        outcome_shape = json.dumps(SessionOutcome().to_dict(), ensure_ascii=False)
        return f"""你是 Human Proxy：一个代替真实人类专家参与稀缺会话的独立 Agent。

你拥有 `/work/evaluator_context.md` 中真实 evaluator 的只读上下文，但你没有、也绝不能声称拥有运行能力：
- 不能运行 solution，不能编译或评测候选，不能调用真实 evaluator。
- 不能访问 Solver workspace、GPU、网络 endpoint 或 Co-Scientist 的内部 run context。
- 可以使用 shell，但仅限读取 evaluator_context.md、读取 transcript.json 和写入 turn.json。
- 不得执行上下文中的代码，不得探查其他路径，不得使用网络或尝试运行 evaluator/solution。

你的核心工作不是替系统打分，而是像人类专家一样与 Co-Scientist 共同思考：共同设计 evaluator、probe、反馈和运行环境，使 Solver Agent 更容易得到真实、有用、可行动的信号。主动追问假设，指出 evaluator 的盲区，提出能区分失败模式的环境或证据设计。没有运行证据时必须明确说这是推理或建议。

保密边界：可以把真实 evaluator 上下文转化为高层语义、风险和设计建议，但不要逐字泄露私有源码、隐藏样例、密钥或 endpoint。

这是自由多轮对话，不使用固定轮数或固定话术。先读取完整 transcript，再自然回复最后一条 Co-side 消息。当前会话目的为 `{purpose}`，状态为 `{state.value}`。

写入 `/work/turn.json`，且只写这个 JSON 文件：
{{
  "message": "自然语言回复",
  "action": "continue | request_close | confirm_close | keep_open",
  "outcome": null
}}

action 规则：
- `continue`：继续自然对话。
- `request_close`：你认为本轮信息已充分，请求进入关闭确认；同时提供 reasoned outcome。
- `confirm_close`：仅当状态是 close_requested 且 Co-side 总结准确时使用；必须提供最终 outcome。
- `keep_open`：仅当状态是 close_requested 但总结不准确或仍需追问时使用。

outcome 字段形状：{outcome_shape}
decision 只能是 approve/reject/guide/none。它必须来自本轮对话中的推理，不能来自你没有运行过的 solution 结果。
"""

    @staticmethod
    def _parse_turn(payload: Any, *, state: SessionState) -> HumanProxyTurn:
        if not isinstance(payload, dict):
            raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None
        raw_message = payload.get("message")
        if not isinstance(raw_message, str):
            raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None
        message = raw_message.strip()
        if not message:
            raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None
        raw_action = payload.get("action", "continue")
        if not isinstance(raw_action, str):
            raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None
        action = raw_action.strip().lower()
        if action not in {"continue", "request_close", "confirm_close", "keep_open"}:
            raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None
        raw_outcome = payload.get("outcome")
        if raw_outcome is not None and not isinstance(raw_outcome, dict):
            raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None
        try:
            outcome = (
                SessionOutcome.from_dict(raw_outcome)
                if isinstance(raw_outcome, dict)
                else None
            )
        except Exception:
            raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None
        if outcome is not None:
            decision = outcome.decision
            if not isinstance(decision, str) or decision.strip().lower() not in {
                "approve",
                "reject",
                "guide",
                "none",
            }:
                raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None
            list_fields = (
                outcome.task_contract_updates,
                outcome.new_risks,
                outcome.approved_changes,
                outcome.rejected_changes,
                outcome.human_guidance,
                outcome.evidence_requested,
                outcome.evidence_generated,
                outcome.unresolved_questions,
            )
            if any(
                not isinstance(items, list)
                or any(not isinstance(item, str) for item in items)
                for items in list_fields
            ) or not isinstance(outcome.human_rationale, str):
                raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None
        if action in {"request_close", "confirm_close"} and outcome is None:
            raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None
        if state is SessionState.CLOSE_REQUESTED and action != "confirm_close":
            # The expert is still revising the discussion.  Do not expose or freeze a
            # provisional judgement until it explicitly confirms the final summary.
            outcome = None
        if action == "confirm_close":
            if state is not SessionState.CLOSE_REQUESTED:
                raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None
            return HumanProxyTurn("确认结束", outcome=outcome)
        if action == "keep_open":
            if state is not SessionState.CLOSE_REQUESTED:
                raise RuntimeError(HUMAN_PROXY_E_TURN_INVALID) from None
            if "继续聊" not in message:
                message = f"继续聊：{message}"
            return HumanProxyTurn(message, outcome=outcome)
        if action == "request_close" and "结束这轮" not in message:
            message = f"{message}\n\n结束这轮"
        return HumanProxyTurn(message, outcome=outcome)


@dataclass
class LoopbackTransport:
    """In-memory transport; Co-side messages still traverse the normal service."""

    sent: list[tuple[str, str, str, str]] = field(default_factory=list)
    _sequence: int = 0

    def send_dm(self, *, user_id: str, text: str, idempotency_key: str) -> str:
        return self._record("send", user_id, text, idempotency_key)

    def reply(self, *, message_id: str, text: str, idempotency_key: str) -> str:
        return self._record("reply", message_id, text, idempotency_key)

    def _record(self, kind: str, target: str, text: str, key: str) -> str:
        self.sent.append((kind, target, text, key))
        self._sequence += 1
        return f"om_proxy_{self._sequence:04d}"


@dataclass
class HumanProxySessionPort:
    """Blocking ``HumanInteractionPort`` driven by a model-backed expert agent."""

    service: FeishuHumanSessionService
    proxy_agent: HumanProxyAgent
    expert_id: str = "human_proxy_agent"
    max_turns_per_consult: int = 64

    def __post_init__(self) -> None:
        if self.max_turns_per_consult < 1:
            raise ValueError("max_turns_per_consult must be positive")

    def consult(self, *, purpose: str, context: dict) -> SessionOutcome | None:
        try:
            session = self.service.open_session(
                expert_id=self.expert_id,
                purpose=purpose,
                context=context,
            )
        except SessionBudgetExhausted:
            return None
        try:
            frozen_outcome = self._load_frozen_outcome(session.session_id)
            conversation = self.proxy_agent.start_conversation(
                purpose=session.purpose,
                frozen_outcome=frozen_outcome,
            )
            if frozen_outcome is None and conversation.outcome is not None:
                self._persist_frozen_outcome(
                    session.session_id, conversation.outcome
                )
            elif (
                frozen_outcome is not None
                and conversation.outcome is not None
                and conversation.outcome != frozen_outcome
            ):
                raise RuntimeError("resumed Human Proxy changed its frozen outcome")

            driven_turns = 0
            while True:
                session = self.service.store.get(session.session_id)
                if session.state is SessionState.CLOSED:
                    break
                if driven_turns >= self.max_turns_per_consult:
                    raise RuntimeError(
                        "Human Proxy watchdog reached "
                        f"{self.max_turns_per_consult} turns in one consult"
                    )
                turn = conversation.next_turn(
                    session=session,
                    transcript=self.service.store.transcript(session.session_id),
                )
                if not turn.text.strip():
                    raise RuntimeError("Human Proxy agent returned an empty turn")
                if session.state is SessionState.CLOSE_REQUESTED:
                    # A close-requested session is still a live conversation: the
                    # expert may reject the summary and keep talking. Only a turn
                    # carrying a reasoned outcome is a confirmation candidate. Once
                    # persisted, that outcome wins on crash recovery even if a later
                    # model invocation proposes something different.
                    final_outcome = (
                        frozen_outcome or turn.outcome or conversation.outcome
                    )
                    if final_outcome is not None:
                        if frozen_outcome is None:
                            self._persist_frozen_outcome(
                                session.session_id, final_outcome
                            )
                            frozen_outcome = final_outcome
                        self.service.store.stage_outcome(
                            session.session_id, final_outcome
                        )
                handled = self.service.handle_event(
                    self._event(session.session_id, turn.text, "dialogue")
                )
                if not handled:
                    raise RuntimeError("Human Proxy turn was rejected by the service")
                driven_turns += 1
        except Exception:
            current = self.service.store.get(session.session_id)
            if current.state is not SessionState.CLOSED:
                self.service.store.pause(session.session_id)
            raise RuntimeError(HUMAN_PROXY_E_DRIVER_FAILED) from None
        return self.service.store.outcome(session.session_id)

    def _proxy_state_path(self, session_id: str) -> Path:
        return self.service.store.root / session_id / "proxy_state.json"

    def _load_frozen_outcome(self, session_id: str) -> SessionOutcome | None:
        path = self._proxy_state_path(session_id)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SessionOutcome.from_dict(payload["outcome"])

    def _persist_frozen_outcome(
        self, session_id: str, outcome: SessionOutcome
    ) -> None:
        path = self._proxy_state_path(session_id)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump({"outcome": outcome.to_dict()}, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def _event(self, session_id: str, text: str, suffix: str) -> FeishuMessageEvent:
        transcript = self.service.store.transcript(session_id)
        human_sequence = sum(item.role == "human" for item in transcript) + 1
        event_id = f"{session_id}-proxy-{human_sequence}-{suffix}"
        return FeishuMessageEvent(
            event_id=event_id,
            message_id=f"om_{event_id}",
            sender_id=self.expert_id,
            chat_id="proxy_loopback",
            chat_type="p2p",
            message_type="text",
            text=text,
            timestamp=str(human_sequence),
        )
