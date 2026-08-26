from __future__ import annotations

import base64
import hashlib
import json
import time
import traceback
from dataclasses import replace
from pathlib import Path

import pytest

from coscientist.coevo.agent_system import AgentSystem
from coscientist.coevo.container import AgentSession, GatewayConfig
from coscientist.coevo.feishu_human import FeishuHumanSessionService
from coscientist.coevo.human_evidence import EvidenceReply
from coscientist.coevo.human_proxy_sessions import (
    HumanProxySessionPort,
    LoopbackTransport,
    ModelBackedHumanProxyAgent,
    assert_no_private_context_leak,
    scan_private_context_leaks,
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


@pytest.mark.parametrize("decision", [None, 7, "allow"])
def test_model_proxy_rejects_invalid_outcome_decisions(decision):
    with pytest.raises(RuntimeError) as caught:
        ModelBackedHumanProxyAgent._parse_turn(
            {
                "message": "确认这个结论。",
                "action": "confirm_close",
                "outcome": {"decision": decision},
            },
            state=SessionState.CLOSE_REQUESTED,
        )
    assert str(caught.value) == "HUMAN_PROXY_E_TURN_INVALID"


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


def test_keep_open_does_not_freeze_an_outcome_before_later_confirmation(tmp_path):
    early = SessionOutcome(
        decision="reject",
        rejected_changes=["结论尚未讨论完整"],
    )
    final = SessionOutcome(
        decision="approve",
        approved_changes=["继续讨论后确认可以采用"],
    )
    runner = RecordingRunner(
        [
            {
                "message": "先总结当前讨论。",
                "action": "request_close",
                "outcome": early.to_dict(),
            },
            {
                "message": "总结还不准确，需要继续讨论。",
                "action": "keep_open",
                "outcome": early.to_dict(),
            },
            {
                "message": "补充信息已经充分。",
                "action": "request_close",
                "outcome": final.to_dict(),
            },
            {
                "message": "确认采用更新后的结论。",
                "action": "confirm_close",
                "outcome": final.to_dict(),
            },
        ]
    )

    class CoSideAgent:
        def reply(self, **kwargs):
            instruction = kwargs["agent_instruction"]
            if "会话开场" in instruction:
                return EvidenceReply(text="请一起检查 verifier 变化。")
            if "总结" in instruction:
                return EvidenceReply(
                    text="这是当前总结，请确认是否准确。",
                    proposed_outcome=SessionOutcome(decision="none"),
                )
            return EvidenceReply(text="收到补充信息，继续讨论。")

    store = HumanSessionStore(tmp_path / "run")
    port = HumanProxySessionPort(
        service=FeishuHumanSessionService(
            store=store,
            transport=LoopbackTransport(),
            agent=CoSideAgent(),
        ),
        proxy_agent=_agent(tmp_path, runner),
        expert_id="human_proxy_agent",
    )

    result = port.consult(purpose="verifier_change", context={})

    assert result == final
    assert store.outcome("session_001") == final


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
        human_proxy_context_sha256=hashlib.sha256(
            private_context.read_bytes()).hexdigest(),
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


def test_agent_system_hashes_single_context_read_and_uses_same_in_memory_bytes(
        tmp_path, monkeypatch):
    """Replacing the file after read_bytes cannot change what the Proxy receives."""
    from coscientist.coevo import agent_system as agent_system_module

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "instruction.md").write_text("public")
    private_context = tmp_path / "private.md"
    original_text = "ORIGINAL accepted evaluator guidance"
    private_context.write_text(original_text)
    expected = hashlib.sha256(private_context.read_bytes()).hexdigest()
    system = AgentSystem(
        raw_input_dir=raw,
        run_dir=tmp_path / "run",
        human_proxy_context_path=private_context,
        human_proxy_context_sha256=expected,
    )
    system.gateway = GatewayConfig(codex_home=tmp_path / "codex-home")
    system.agent_elf = tmp_path / "codex"
    monkeypatch.setattr(agent_system_module, "docker_unavailable", lambda: None)
    original_read_bytes = Path.read_bytes
    reads = {"n": 0}

    def swap_after_read(path):
        data = original_read_bytes(path)
        if path == private_context:
            reads["n"] += 1
            private_context.write_text("SWAPPED malicious context")
        return data

    monkeypatch.setattr(Path, "read_bytes", swap_after_read)
    system.preflight()

    assert reads["n"] == 1
    assert system.human_port.proxy_agent.evaluator_context == original_text
    assert private_context.read_text() == "SWAPPED malicious context"


def test_execution_backed_proxy_api_is_removed():
    from coscientist.coevo import human_proxy_sessions as proxy_module

    assert not hasattr(proxy_module, "PythonReferenceEvaluator")
    assert not hasattr(proxy_module, "EvaluatorBackedHumanProxyAgent")
    assert not hasattr(AgentSystem, "_human_comparison_cases")


@pytest.mark.parametrize("field", ["message", "outcome"])
def test_model_proxy_rejects_normalized_long_private_spans_before_return(
    tmp_path, field
):
    secret = (
        "PRIVATE CHECKER CONTRACT: candidates must satisfy the exact hidden "
        "matrix dimensions, sentinel ordering, nonce derivation, and rejection "
        "branches recorded only in this accepted final evaluator package. "
    )
    context = f"Evaluator overview.\n{secret}\nKnown issues."
    leaked = "  \n".join(secret.upper().split(" "))
    payload = {
        "message": "Here is high-level guidance.",
        "action": "continue",
        "outcome": None,
    }
    if field == "message":
        payload["message"] = leaked
    else:
        payload["action"] = "request_close"
        payload["outcome"] = {
            "decision": "guide",
            "human_guidance": [leaked],
        }
    runner = RecordingRunner([payload])
    agent = ModelBackedHumanProxyAgent(
        evaluator_context=context,
        gateway=GatewayConfig(codex_home=tmp_path / "codex-home"),
        agent_elf=tmp_path / "codex",
        runner=runner,
    )

    conversation = agent.start_conversation(purpose="verifier_change")

    with pytest.raises(RuntimeError) as caught:
        conversation.next_turn(session=_session(), transcript=[])
    assert str(caught.value) == "HUMAN_PROXY_E_PRIVATE_CONTEXT_LEAK"


def test_private_context_scan_recurses_and_allows_high_level_semantic_advice():
    secret = (
        "The accepted private evaluator checks exact payload framing, rejects "
        "ambiguous numeric encodings, validates every hidden sentinel, and binds "
        "the benchmark seed to a protected campaign-specific nonce before scoring."
    )

    safe = scan_private_context_leaks(
        {
            "message": "Add probes for malformed payloads and seed leakage.",
            "outcome": {"human_guidance": ["Separate correctness from speed."]},
        },
        secret,
    )
    unsafe = scan_private_context_leaks(
        {"outcome": {"nested": [{"guidance": f"prefix {secret} suffix"}]}},
        secret,
    )

    assert safe == []
    assert len(unsafe) == 1
    assert unsafe[0].matched_characters == 160


def _long_private_contract() -> str:
    return (
        "Candidate framing must match every protected dimension and sentinel; "
        "numeric encodings are canonicalized before the hidden campaign nonce "
        "binds the deterministic benchmark seed to the accepted evaluator. "
        "Malformed payload branches fail closed before any performance score."
    )


@pytest.mark.parametrize("separator", ["", " "])
def test_private_context_scan_catches_a_span_split_across_nested_leaves(separator):
    secret = _long_private_contract()
    chunks = [secret[index : index + 48] for index in range(0, len(secret), 48)]
    payload = {
        "message": chunks[0],
        "outcome": {"guidance": [{"part": chunk} for chunk in chunks[1:]]},
    }
    if separator:
        secret = separator.join(chunks)

    findings = scan_private_context_leaks(payload, secret)

    assert findings


def test_private_context_scan_catches_zero_width_and_punctuation_insertion():
    secret = _long_private_contract()
    obfuscated = "\u200b!".join(secret)

    findings = scan_private_context_leaks({"message": obfuscated}, secret)

    assert findings


@pytest.mark.parametrize("encoding", ["hex", "base64"])
def test_private_context_scan_decodes_encoded_spans_split_across_leaves(encoding):
    secret = _long_private_contract()
    raw = secret.encode("utf-8")
    encoded = raw.hex() if encoding == "hex" else base64.b64encode(raw).decode()
    chunks = [encoded[index : index + 52] for index in range(0, len(encoded), 52)]
    payload = {
        "message": "safe high-level advice",
        "action": "request_close",
        "outcome": {"decision": "guide", "human_guidance": chunks},
    }

    findings = scan_private_context_leaks(payload, secret)

    assert findings


def test_private_context_scan_decodes_unpadded_urlsafe_base64_subtree():
    secret = _long_private_contract()
    encoded = base64.urlsafe_b64encode(secret.encode()).decode().rstrip("=")
    chunks = [encoded[index : index + 47] for index in range(0, len(encoded), 47)]
    payload = {
        "message": "safe",
        "action": "request_close",
        "outcome": {"decision": "guide", "human_guidance": chunks},
    }

    findings = scan_private_context_leaks(payload, secret)

    assert findings


@pytest.mark.parametrize("chunk_size", [1, 7, 15])
@pytest.mark.parametrize("encoding", ["hex", "base64", "urlsafe"])
def test_private_context_scan_decodes_short_encoded_chunks_in_one_container(
    encoding, chunk_size
):
    secret = _long_private_contract()
    if encoding == "hex":
        encoded = secret.encode().hex()
    elif encoding == "base64":
        encoded = base64.b64encode(secret.encode()).decode()
    else:
        encoded = base64.urlsafe_b64encode(secret.encode()).decode().rstrip("=")
    chunks = [
        encoded[index : index + chunk_size]
        for index in range(0, len(encoded), chunk_size)
    ]
    payload = {
        "message": "safe high-level advice",
        "action": "request_close",
        "outcome": {"decision": "guide", "human_guidance": chunks},
    }

    findings = scan_private_context_leaks(payload, secret)

    assert findings


@pytest.mark.parametrize("encoding", ["hex", "base64", "urlsafe"])
def test_private_context_scan_decodes_one_char_chunks_across_outcome_lists(encoding):
    secret = _long_private_contract()
    if encoding == "hex":
        encoded = secret.encode().hex()
    elif encoding == "base64":
        encoded = base64.b64encode(secret.encode()).decode()
    else:
        encoded = base64.urlsafe_b64encode(secret.encode()).decode().rstrip("=")
    fields = (
        "task_contract_updates",
        "new_risks",
        "approved_changes",
        "rejected_changes",
        "human_guidance",
        "evidence_requested",
        "evidence_generated",
        "unresolved_questions",
    )
    boundaries = [len(encoded) * index // len(fields) for index in range(9)]
    outcome = {"decision": "guide"}
    for index, field in enumerate(fields):
        outcome[field] = list(encoded[boundaries[index] : boundaries[index + 1]])
    payload = {
        "message": "safe high-level advice",
        "action": "request_close",
        "outcome": outcome,
    }

    findings = scan_private_context_leaks(payload, secret)

    assert findings


def test_model_proxy_does_not_echo_runner_note_or_stderr(tmp_path):
    secret_note = "PRIVATE_NOTE_FROM_UNTRUSTED_AGENT"
    secret_stderr = "PRIVATE_STDERR_FROM_UNTRUSTED_AGENT"

    def failed_runner(*args, **kwargs):
        return AgentSession(
            ok=False,
            returncode=1,
            stdout="",
            stderr=secret_stderr,
            note=secret_note,
        )

    agent = ModelBackedHumanProxyAgent(
        evaluator_context=_long_private_contract(),
        gateway=GatewayConfig(codex_home=tmp_path / "codex-home"),
        agent_elf=tmp_path / "codex",
        runner=failed_runner,
    )

    with pytest.raises(RuntimeError) as caught:
        agent.start_conversation(purpose="verifier_change").next_turn(
            session=_session(), transcript=[]
        )

    assert str(caught.value) == "HUMAN_PROXY_E_RUN_FAILED"
    assert secret_note not in repr(caught.value)
    assert secret_stderr not in repr(caught.value)


def test_model_proxy_maps_a_runner_exception_to_a_fixed_safe_code(tmp_path):
    secret = "PRIVATE_EXCEPTION_FROM_UNTRUSTED_RUNNER"

    def exploding_runner(*args, **kwargs):
        raise RuntimeError(secret)

    agent = ModelBackedHumanProxyAgent(
        evaluator_context=_long_private_contract(),
        gateway=GatewayConfig(codex_home=tmp_path / "codex-home"),
        agent_elf=tmp_path / "codex",
        runner=exploding_runner,
    )

    with pytest.raises(RuntimeError) as caught:
        agent.start_conversation(purpose="verifier_change").next_turn(
            session=_session(), transcript=[]
        )

    assert str(caught.value) == "HUMAN_PROXY_E_RUN_FAILED"
    assert secret not in repr(caught.value)
    rendered = "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb)
    )
    assert secret not in rendered


def test_model_proxy_maps_pathologically_nested_json_to_a_fixed_safe_code(tmp_path):
    class NestedJsonRunner:
        def __call__(self, workspace, prompt, **kwargs):
            (workspace / "turn.json").write_text("[" * 2000 + "]" * 2000)
            return AgentSession(ok=True, returncode=0, stdout="", stderr="")

    agent = ModelBackedHumanProxyAgent(
        evaluator_context=_long_private_contract(),
        gateway=GatewayConfig(codex_home=tmp_path / "codex-home"),
        agent_elf=tmp_path / "codex",
        runner=NestedJsonRunner(),
    )

    with pytest.raises(RuntimeError) as caught:
        agent.start_conversation(purpose="verifier_change").next_turn(
            session=_session(), transcript=[]
        )

    assert str(caught.value) == "HUMAN_PROXY_E_TURN_INVALID"


@pytest.mark.parametrize("depth", [100, 800])
def test_private_scan_is_bounded_and_fast_for_deep_200kb_payload(depth):
    payload = "x" * 200_000
    for _ in range(depth):
        payload = {"nested": [payload]}

    started = time.monotonic()
    findings = scan_private_context_leaks(payload, _long_private_contract())
    elapsed = time.monotonic() - started

    assert findings == []
    assert elapsed < 2.0


@pytest.mark.parametrize(
    "payload",
    [
        [f"leaf-{index}" for index in range(20_000)],
        "x" * 2_100_000,
    ],
    ids=["candidate-count", "candidate-characters"],
)
def test_private_scan_fails_closed_when_candidate_budget_is_exceeded(payload):
    with pytest.raises(RuntimeError) as caught:
        assert_no_private_context_leak(payload, _long_private_contract())

    assert str(caught.value) == "HUMAN_PROXY_E_PRIVATE_CONTEXT_LEAK"


def test_model_proxy_suppresses_untrusted_json_parser_exception_chain(
    tmp_path, monkeypatch
):
    from coscientist.coevo import human_proxy_sessions as proxy_module

    canary = "PRIVATE_JSON_EXCEPTION_CANARY"
    runner = RecordingRunner(
        [{"message": "safe", "action": "continue", "outcome": None}]
    )
    agent = ModelBackedHumanProxyAgent(
        evaluator_context=_long_private_contract(),
        gateway=GatewayConfig(codex_home=tmp_path / "codex-home"),
        agent_elf=tmp_path / "codex",
        runner=runner,
    )

    def explode_json(*args, **kwargs):
        raise RuntimeError(canary)

    monkeypatch.setattr(proxy_module.json, "loads", explode_json)

    with pytest.raises(RuntimeError) as caught:
        agent.start_conversation(purpose="verifier_change").next_turn(
            session=_session(), transcript=[]
        )

    rendered = "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb)
    )
    assert str(caught.value) == "HUMAN_PROXY_E_TURN_INVALID"
    assert canary not in rendered


def test_model_proxy_suppresses_session_outcome_parser_exception_chain(
    tmp_path, monkeypatch
):
    canary = "PRIVATE_OUTCOME_EXCEPTION_CANARY"
    runner = RecordingRunner(
        [
            {
                "message": "safe",
                "action": "request_close",
                "outcome": {"decision": "guide"},
            }
        ]
    )
    agent = ModelBackedHumanProxyAgent(
        evaluator_context=_long_private_contract(),
        gateway=GatewayConfig(codex_home=tmp_path / "codex-home"),
        agent_elf=tmp_path / "codex",
        runner=runner,
    )

    def explode_outcome(cls, payload):
        raise RuntimeError(canary)

    monkeypatch.setattr(SessionOutcome, "from_dict", classmethod(explode_outcome))

    with pytest.raises(RuntimeError) as caught:
        agent.start_conversation(purpose="verifier_change").next_turn(
            session=_session(), transcript=[]
        )

    rendered = "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb)
    )
    assert str(caught.value) == "HUMAN_PROXY_E_TURN_INVALID"
    assert canary not in rendered


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (
            {"message": "safe", "action": "PRIVATE_INVALID_ACTION", "outcome": None},
            "HUMAN_PROXY_E_TURN_INVALID",
        ),
        (
            {"message": "safe", "action": "continue", "outcome": {"decision": []}},
            "HUMAN_PROXY_E_TURN_INVALID",
        ),
    ],
)
def test_model_proxy_turn_validation_uses_only_fixed_safe_codes(
    tmp_path, payload, expected_code
):
    runner = RecordingRunner([payload])
    agent = ModelBackedHumanProxyAgent(
        evaluator_context=_long_private_contract(),
        gateway=GatewayConfig(codex_home=tmp_path / "codex-home"),
        agent_elf=tmp_path / "codex",
        runner=runner,
    )

    with pytest.raises(RuntimeError) as caught:
        agent.start_conversation(purpose="verifier_change").next_turn(
            session=_session(), transcript=[]
        )

    assert str(caught.value) == expected_code
    assert "PRIVATE_INVALID_ACTION" not in repr(caught.value)


def test_model_proxy_leak_error_does_not_echo_untrusted_json_key_or_location(tmp_path):
    secret = _long_private_contract()
    untrusted_key = "PRIVATE_ATTACKER_CHOSEN_KEY"
    runner = RecordingRunner(
        [
            {
                "message": "safe",
                "action": "continue",
                "outcome": {untrusted_key: secret},
            }
        ]
    )
    agent = ModelBackedHumanProxyAgent(
        evaluator_context=secret,
        gateway=GatewayConfig(codex_home=tmp_path / "codex-home"),
        agent_elf=tmp_path / "codex",
        runner=runner,
    )

    with pytest.raises(RuntimeError) as caught:
        agent.start_conversation(purpose="verifier_change").next_turn(
            session=_session(), transcript=[]
        )

    assert str(caught.value) == "HUMAN_PROXY_E_PRIVATE_CONTEXT_LEAK"
    assert untrusted_key not in repr(caught.value)


def test_model_proxy_refuses_oversized_turn_before_reading_json(tmp_path):
    oversized = "x" * 300_000
    runner = RecordingRunner(
        [{"message": oversized, "action": "continue", "outcome": None}]
    )
    agent = ModelBackedHumanProxyAgent(
        evaluator_context=_long_private_contract(),
        gateway=GatewayConfig(codex_home=tmp_path / "codex-home"),
        agent_elf=tmp_path / "codex",
        runner=runner,
    )

    with pytest.raises(RuntimeError) as caught:
        agent.start_conversation(purpose="verifier_change").next_turn(
            session=_session(), transcript=[]
        )

    assert str(caught.value) == "HUMAN_PROXY_E_TURN_TOO_LARGE"
