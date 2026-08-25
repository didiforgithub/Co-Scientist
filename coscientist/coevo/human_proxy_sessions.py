"""Evaluator-backed Human Proxy that obeys the real Human Session contract.

The proxy is the *expert participant*, not a privileged evaluator mutation path.
It consults a hidden reference evaluator (V*), talks to the same Co-side evidence
agent through the same durable session store, requests closure, and explicitly
confirms it.  AgentSystem therefore sees only the ordinary ``consult`` seam used
by a Feishu human.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

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
class ReferenceEvaluation:
    ok: bool
    feasible: bool = False
    score: float | None = None
    error: str = ""


class ReferenceEvaluator(Protocol):
    calls: int

    def evaluate(self, payload: dict) -> ReferenceEvaluation: ...


class PythonReferenceEvaluator:
    """Trusted control-plane adapter for a hidden Python V* module.

    The module path and source are never copied into the run. Its callable may be
    ``verify(payload, ctx)`` or ``verify(payload)`` and should return a mapping with
    ``feasible`` plus ``raw`` or ``score`` (higher is better), matching Co's verifier
    convention.
    """

    def __init__(
        self,
        module_path: Path | str,
        *,
        function: str = "verify",
        context: dict[str, Any] | None = None,
    ):
        self.module_path = Path(module_path).resolve()
        self.function = function
        self.context = dict(context or {})
        self.calls = 0
        self._callable = self._load()

    def _load(self):
        if not self.module_path.is_file():
            raise FileNotFoundError(f"reference evaluator not found: {self.module_path}")
        digest = hashlib.sha256(str(self.module_path).encode("utf-8")).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(
            f"coscientist_hidden_reference_{digest}", self.module_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load the reference evaluator module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = getattr(module, self.function, None)
        if not callable(function):
            raise TypeError(
                f"reference evaluator has no callable {self.function!r}"
            )
        return function

    def evaluate(self, payload: dict) -> ReferenceEvaluation:
        self.calls += 1
        try:
            parameters = inspect.signature(self._callable).parameters
            raw_result = (
                self._callable(payload, dict(self.context))
                if len(parameters) >= 2
                else self._callable(payload)
            )
            if not isinstance(raw_result, dict):
                raise TypeError("reference evaluator must return a dict")
            raw_score = raw_result.get("raw", raw_result.get("score"))
            score = float(raw_score) if raw_score is not None else None
            if score is not None and not math.isfinite(score):
                raise ValueError("reference evaluator returned a non-finite score")
            feasible = bool(raw_result.get("feasible", score is not None))
            return ReferenceEvaluation(
                ok=True,
                feasible=feasible,
                score=score,
            )
        except Exception as exc:  # noqa: BLE001 - V* errors become safe proxy guidance
            return ReferenceEvaluation(ok=False, error=type(exc).__name__)


@dataclass(frozen=True)
class ProxyAssessment:
    message: str
    outcome: SessionOutcome


@dataclass(frozen=True)
class HumanProxyTurn:
    """One natural-language turn chosen by the expert-side dialogue agent."""

    text: str


class HumanProxyTurnPolicy(Protocol):
    def __call__(
        self, *, session: HumanSession, transcript: list[SessionMessage]
    ) -> HumanProxyTurn: ...


@dataclass(frozen=True)
class HumanProxyConversation:
    """A frozen V* judgement plus a live policy for choosing subsequent turns."""

    outcome: SessionOutcome
    next_turn: HumanProxyTurnPolicy


class HumanProxyAgent(Protocol):
    """Independent expert agent driven only through the public chat contract."""

    def start_conversation(
        self, *, purpose: str, context: dict[str, Any]
    ) -> HumanProxyConversation: ...


@dataclass
class EvaluatorBackedHumanProxyAgent:
    """Expert-side agent that judges V changes against hidden V* evaluations."""

    reference_evaluator: ReferenceEvaluator
    assessments: int = 0

    def start_conversation(
        self, *, purpose: str, context: dict[str, Any]
    ) -> HumanProxyConversation:
        """Freeze one hidden evaluation, then converse from its safe assessment.

        The returned policy reacts to the durable transcript rather than prescribing
        messages in the transport port.  A deployment may replace this agent with a
        model-backed implementation of :class:`HumanProxyAgent` without changing the
        Human Session service or its lifecycle rules.
        """
        assessment = self.assess(purpose=purpose, context=context)

        def next_turn(
            *, session: HumanSession, transcript: list[SessionMessage]
        ) -> HumanProxyTurn:
            human_turns = [item for item in transcript if item.role == "human"]
            agent_turns = [item for item in transcript if item.role == "agent"]
            if session.state is SessionState.CLOSE_REQUESTED:
                return HumanProxyTurn("确认结束")
            if not human_turns:
                return HumanProxyTurn(assessment.message)
            if len(human_turns) == 1:
                last_reply = agent_turns[-1].text if agent_turns else ""
                focus = (
                    assessment.outcome.unresolved_questions[0]
                    if assessment.outcome.unresolved_questions
                    else "尚未覆盖的边界条件"
                )
                if focus in last_reply:
                    focus = "这个判断在对抗样例上的依据"
                return HumanProxyTurn(f"请继续说明：{focus}。")
            return HumanProxyTurn(
                f"我的最终判断是 {assessment.outcome.decision}，依据已经说明。"
                "现在这轮可以结束了。"
            )

        return HumanProxyConversation(
            outcome=assessment.outcome,
            next_turn=next_turn,
        )

    def assess(self, *, purpose: str, context: dict[str, Any]) -> ProxyAssessment:
        self.assessments += 1
        cases = context.get("comparison_cases", [])
        if not isinstance(cases, list) or not cases:
            return ProxyAssessment(
                message=(
                    "我目前还没有可送入真实 evaluator 的候选样例。请先说明任务契约中"
                    "最关键的正确性条件、数据类型和容差，并记录后续需要生成的对照 probe。"
                ),
                outcome=SessionOutcome(
                    decision="guide",
                    human_guidance=[
                        "先明确任务契约，再用真实 evaluator 对代表性候选做隔离对照。"
                    ],
                    evidence_requested=[
                        "至少一个诚实基线、一个当前最优候选和一个对抗 probe"
                    ],
                    unresolved_questions=["尚无候选可供真实 evaluator 比较"],
                    human_rationale="Human Proxy 不在缺少 V* 对照样例时伪造判断。",
                ),
            )

        evaluated: list[dict[str, Any]] = []
        failed = 0
        for index, case in enumerate(cases):
            if not isinstance(case, dict) or not isinstance(case.get("payload"), dict):
                failed += 1
                continue
            reference = self.reference_evaluator.evaluate(case["payload"])
            if not reference.ok or reference.score is None:
                failed += 1
                continue
            evaluated.append(
                {
                    "label": str(case.get("label", f"case_{index}")),
                    "reference": reference,
                    "current": self._candidate_result(case.get("current")),
                    "proposed": self._candidate_result(case.get("proposed")),
                }
            )

        if not evaluated:
            return ProxyAssessment(
                message=(
                    "真实 evaluator 没有成功完成任何对照样例，所以我不会批准这次变化。"
                    "请先修复 V* 执行路径或补充可运行候选。"
                ),
                outcome=SessionOutcome(
                    decision="guide",
                    human_guidance=["修复真实 evaluator 对照执行后再判断 verifier 变化。"],
                    evidence_requested=["可成功执行的 V* 对照结果"],
                    unresolved_questions=["真实 evaluator 对全部候选执行失败"],
                    human_rationale="没有真实 evaluator 证据时保持 evaluator 冻结。",
                ),
            )

        current = self._alignment(evaluated, "current")
        proposed = self._alignment(evaluated, "proposed")
        evidence_summary = (
            f"真实 evaluator 已隔离检查 {len(evaluated)} 个代表性样例"
            + (f"，另有 {failed} 个样例无法执行" if failed else "")
            + "。"
        )
        if proposed["valid"] < len(evaluated):
            decision = "reject"
            comparison = "拟议 verifier 不能稳定评估全部 V* 对照样例"
        elif self._strictly_better(proposed, current):
            decision = "approve"
            comparison = "拟议 verifier 与真实排序/可行性的对齐优于当前版本"
        elif self._strictly_better(current, proposed):
            decision = "reject"
            comparison = "拟议 verifier 相比当前版本更偏离真实排序/可行性"
        else:
            decision = "guide"
            comparison = "拟议 verifier 没有显示出可验证的 V* 对齐增益"

        message = (
            f"{evidence_summary}{comparison}。我不会暴露 V* 源码或逐样例原始分数；"
            "请说明这次变化如何处理仍未覆盖的边界条件。"
        )
        common = {
            "decision": decision,
            "human_guidance": [comparison + "。"],
            "evidence_generated": [
                f"reference_evaluator_cases:{len(evaluated)}",
                (
                    "pairwise_alignment:"
                    f"current={current['pairwise']:.3f},"
                    f"proposed={proposed['pairwise']:.3f}"
                ),
            ],
            "unresolved_questions": (
                [f"{failed} 个 V* 对照样例未成功执行"] if failed else []
            ),
            "human_rationale": (
                f"真实 evaluator 的隔离对照表明：{comparison}；"
                "V* 源码和逐样例原始结果未提供给 Solver。"
            ),
        }
        if decision == "approve":
            common["approved_changes"] = ["采用本轮经 V* 对照的 verifier 变化"]
        elif decision == "reject":
            common["rejected_changes"] = ["拒绝本轮 verifier 变化"]
        else:
            common["evidence_requested"] = ["增加能区分当前版与拟议版的边界 probe"]
        return ProxyAssessment(message=message, outcome=SessionOutcome(**common))

    @staticmethod
    def _candidate_result(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {"ok": False, "feasible": False, "score": None}
        score = value.get("score")
        try:
            parsed_score = float(score) if score is not None else None
        except (TypeError, ValueError):
            parsed_score = None
        return {
            "ok": bool(value.get("ok", parsed_score is not None)),
            "feasible": bool(value.get("feasible", parsed_score is not None)),
            "score": parsed_score,
        }

    @staticmethod
    def _alignment(evaluated: list[dict[str, Any]], key: str) -> dict[str, float]:
        valid = [
            item
            for item in evaluated
            if item[key]["ok"] and item[key]["score"] is not None
        ]
        feasibility = (
            sum(
                item[key]["feasible"] == item["reference"].feasible
                for item in valid
            )
            / len(valid)
            if valid
            else 0.0
        )
        agreements = 0
        pairs = 0
        for left_index, left in enumerate(valid):
            for right in valid[left_index + 1 :]:
                ref_delta = left["reference"].score - right["reference"].score
                candidate_delta = left[key]["score"] - right[key]["score"]
                if abs(ref_delta) < 1e-12:
                    continue
                pairs += 1
                if ref_delta * candidate_delta > 0:
                    agreements += 1
                elif abs(candidate_delta) < 1e-12:
                    agreements += 0.5
        pairwise = agreements / pairs if pairs else feasibility
        mae = (
            sum(abs(item[key]["score"] - item["reference"].score) for item in valid)
            / len(valid)
            if valid
            else float("inf")
        )
        return {
            "valid": float(len(valid)),
            "feasibility": feasibility,
            "pairwise": pairwise,
            "mae": mae,
        }

    @staticmethod
    def _strictly_better(left: dict[str, float], right: dict[str, float]) -> bool:
        if left["valid"] > right["valid"]:
            return True
        if left["valid"] < right["valid"]:
            return False
        if left["feasibility"] > right["feasibility"] + 1e-9:
            return True
        if left["feasibility"] + 1e-9 < right["feasibility"]:
            return False
        if left["pairwise"] > right["pairwise"] + 1e-9:
            return True
        if left["pairwise"] + 1e-9 < right["pairwise"]:
            return False
        return left["mae"] + 1e-9 < right["mae"]


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
    """Blocking ``HumanInteractionPort`` implemented by a V*-holding expert agent."""

    service: FeishuHumanSessionService
    proxy_agent: HumanProxyAgent
    expert_id: str = "human_proxy_vstar"

    def consult(self, *, purpose: str, context: dict) -> SessionOutcome | None:
        try:
            session = self.service.open_session(
                expert_id=self.expert_id,
                purpose=purpose,
                context=context,
            )
        except SessionBudgetExhausted:
            return None
        conversation = self.proxy_agent.start_conversation(
            purpose=session.purpose,
            context=session.context,
        )
        while True:
            session = self.service.store.get(session.session_id)
            if session.state is SessionState.CLOSED:
                break
            turn = conversation.next_turn(
                session=session,
                transcript=self.service.store.transcript(session.session_id),
            )
            if not turn.text.strip():
                raise RuntimeError("Human Proxy agent returned an empty turn")
            if session.state is SessionState.CLOSE_REQUESTED:
                # V* is frozen before the first turn.  Only expose its sanitized
                # structured outcome at the ordinary explicit-confirmation seam.
                self.service.store.stage_outcome(
                    session.session_id, conversation.outcome
                )
            self.service.handle_event(
                self._event(session.session_id, turn.text, "dialogue")
            )
        return self.service.store.outcome(session.session_id)

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
