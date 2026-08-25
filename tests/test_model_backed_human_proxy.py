from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from coscientist.coevo.agent_system import AgentSystem
from coscientist.coevo.container import AgentSession, GatewayConfig
from coscientist.coevo.feishu_human import FeishuHumanSessionService
from coscientist.coevo.human_evidence import EvidenceReply
from coscientist.coevo.human_proxy_sessions import (
    HumanProxySessionPort,
    LoopbackTransport,
    ModelBackedHumanProxyAgent,
)
from coscientist.coevo.human_sessions import (
    HumanSession,
    HumanSessionStore,
    SessionMessage,
    SessionOutcome,
    SessionState,
)


class RecordingRunner:
    def __init__(self, replies: list[dict]):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def __call__(self, workspace: Path, prompt: str, **kwargs) -> AgentSession:
        files = {
            path.name: path.read_text(encoding="utf-8")
            for path in workspace.iterdir()
            if path.is_file()
        }
        self.calls.append({"prompt": prompt, "files": files, "kwargs": kwargs})
        (workspace / "turn.json").write_text(
            json.dumps(self.replies.pop(0), ensure_ascii=False),
            encoding="utf-8",
        )
        return AgentSession(ok=True, returncode=0, stdout="", stderr="")


def _session(state: SessionState = SessionState.ACTIVE) -> HumanSession:
    return HumanSession(
        session_id="session_001",
        expert_id="human_proxy_agent",
        purpose="verifier_change",
        state=state,
        opened_at="2026-08-26T00:00:00+00:00",
        updated_at="2026-08-26T00:00:00+00:00",
        context={
            "private_solver_payload": "MUST_NOT_REACH_PROXY",
            "comparison_cases": [{"payload": {"solution": "SECRET"}}],
        },
    )


def _message(sequence: int, role: str, text: str) -> SessionMessage:
    return SessionMessage(
        sequence=sequence,
        role=role,
        text=text,
        timestamp="2026-08-26T00:00:00+00:00",
    )


def _agent(tmp_path: Path, runner: RecordingRunner) -> ModelBackedHumanProxyAgent:
    gateway = GatewayConfig(codex_home=tmp_path / "codex-home")
    return ModelBackedHumanProxyAgent(
        evaluator_context=(
            "PRIVATE_EVALUATOR_CONTEXT\n"
            "Think about mask semantics and whether feedback helps the solver.\n"
            "raise RuntimeError('THIS IS TEXT, NEVER EXECUTE IT')\n"
        ),
        gateway=gateway,
        agent_elf=tmp_path / "codex",
        runner=runner,
    )


def test_model_proxy_gets_private_text_and_transcript_but_no_run_context(tmp_path):
    runner = RecordingRunner(
        [
            {
                "message": "先解释当前环境如何暴露 mask 失败，再一起设计 probe。",
                "action": "continue",
                "outcome": None,
            }
        ]
    )
    agent = _agent(tmp_path, runner)
    conversation = agent.start_conversation(
        purpose="verifier_change",
    )

    turn = conversation.next_turn(
        session=_session(),
        transcript=[_message(1, "agent", "我们怎样构造更有帮助的 evaluator？")],
    )

    assert turn.text.startswith("先解释")
    assert turn.outcome is None
    assert len(runner.calls) == 1
    files = runner.calls[0]["files"]
    assert "PRIVATE_EVALUATOR_CONTEXT" in files["evaluator_context.md"]
    assert "我们怎样构造" in files["transcript.json"]
    joined = "\n".join(files.values()) + runner.calls[0]["prompt"]
    assert "MUST_NOT_REACH_PROXY" not in joined
    assert '"solution": "SECRET"' not in joined
    assert not (tmp_path / "THIS_IS_TEXT").exists()


def test_model_proxy_freely_requests_then_confirms_close_with_reasoned_outcome(
    tmp_path,
):
    outcome = SessionOutcome(
        decision="guide",
        human_guidance=["先让环境返回逐 shape 的错误类别，再调性能。"],
        evidence_requested=["mask/Cauchy/large-N 的分类 probe"],
        human_rationale="当前反馈只有总分，无法帮助 Solver 定位失败。",
    )
    runner = RecordingRunner(
        [
            {
                "message": "我认为方向已经明确。",
                "action": "request_close",
                "outcome": outcome.to_dict(),
            },
            {
                "message": "可以落盘这个结论。",
                "action": "confirm_close",
                "outcome": outcome.to_dict(),
            },
        ]
    )
    conversation = _agent(tmp_path, runner).start_conversation(
        purpose="verifier_change"
    )
    transcript = [
        _message(1, "agent", "你觉得怎样让 Agent 跑得更好？"),
        _message(2, "human", "需要给错误分类，而不只是总分。"),
    ]

    request = conversation.next_turn(session=_session(), transcript=transcript)
    confirm = conversation.next_turn(
        session=replace(_session(), state=SessionState.CLOSE_REQUESTED),
        transcript=transcript
        + [_message(3, "human", request.text), _message(4, "agent", "请确认结束")],
    )

    assert "结束这轮" in request.text
    assert request.outcome == outcome
    assert confirm.text == "确认结束"
    assert confirm.outcome == outcome
    assert len(runner.calls) == 2


def test_model_proxy_prompt_forbids_execution_and_centers_environment_codesign(
    tmp_path,
):
    runner = RecordingRunner(
        [{"message": "先讨论环境。", "action": "continue", "outcome": None}]
    )
    conversation = _agent(tmp_path, runner).start_conversation(
        purpose="task_definition"
    )

    conversation.next_turn(session=_session(), transcript=[])

    prompt = runner.calls[0]["prompt"]
    assert "不能运行 solution" in prompt
    assert "不能调用真实 evaluator" in prompt
    assert "可以使用 shell" in prompt
    assert "不要使用 shell" not in prompt
    assert "共同设计" in prompt
    assert "evaluator、probe、反馈和运行环境" in prompt


def test_session_port_stages_the_model_agents_final_outcome_only_at_confirmation(
    tmp_path,
):
    outcome = SessionOutcome(
        decision="guide",
        human_guidance=["增加按失败类型分流的反馈。"],
        human_rationale="这会让 Solver 知道下一步该改正确性还是性能。",
    )
    runner = RecordingRunner(
        [
            {
                "message": "建议已经足够具体。",
                "action": "request_close",
                "outcome": outcome.to_dict(),
            },
            {
                "message": "确认这个总结。",
                "action": "confirm_close",
                "outcome": outcome.to_dict(),
            },
        ]
    )

    class CoSideAgent:
        def reply(self, **kwargs):
            if not kwargs["transcript"]:
                return EvidenceReply(text="请一起设计更有帮助的 evaluator 环境。")
            return EvidenceReply(
                text="总结：增加失败分类反馈。若准确请确认结束。",
                proposed_outcome=SessionOutcome(decision="none"),
            )

    store = HumanSessionStore(tmp_path / "run")
    service = FeishuHumanSessionService(
        store=store,
        transport=LoopbackTransport(),
        agent=CoSideAgent(),
    )
    port = HumanProxySessionPort(
        service=service,
        proxy_agent=_agent(tmp_path, runner),
        expert_id="human_proxy_agent",
    )

    result = port.consult(purpose="verifier_change", context={"secret": "run-only"})

    assert result == outcome
    assert store.outcome("session_001") == outcome
    assert store.sessions()[0].state is SessionState.CLOSED
    assert len(runner.calls) == 2


def test_agent_system_builds_proxy_from_text_without_importing_or_executing_it(
    tmp_path, monkeypatch
):
    from coscientist.coevo import agent_system as agent_system_module

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "instruction.md").write_text("public task")
    canary = tmp_path / "IMPORTED_EVALUATOR"
    private_context = tmp_path / "private_evaluator.py"
    private_context.write_text(
        "from pathlib import Path\n"
        f"Path({str(canary)!r}).write_text('bad')\n"
        "The evaluator rewards environments that expose useful failure modes.\n"
    )
    system = AgentSystem(
        raw_input_dir=raw,
        run_dir=tmp_path / "run",
        human_proxy_context_path=private_context,
    )
    system.gateway = GatewayConfig(codex_home=tmp_path / "codex-home")
    system.agent_elf = tmp_path / "codex"
    monkeypatch.setattr(agent_system_module, "docker_unavailable", lambda: None)

    system.preflight()

    assert isinstance(system.human_port, HumanProxySessionPort)
    assert isinstance(system.human_port.proxy_agent, ModelBackedHumanProxyAgent)
    assert "useful failure modes" in system.human_port.proxy_agent.evaluator_context
    assert not canary.exists()
    assert not list(system.run_dir.rglob("private_evaluator.py"))


def test_execution_backed_proxy_api_is_removed():
    from coscientist.coevo import human_proxy_sessions as proxy_module

    assert not hasattr(proxy_module, "PythonReferenceEvaluator")
    assert not hasattr(proxy_module, "EvaluatorBackedHumanProxyAgent")
    assert not hasattr(AgentSystem, "_human_comparison_cases")
