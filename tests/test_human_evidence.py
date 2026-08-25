from __future__ import annotations

import json
from pathlib import Path

from coscientist.coevo.container import AgentSession
from coscientist.coevo.human_evidence import (
    CodexEvidenceAgent,
    RunEvidenceBuilder,
)
from coscientist.coevo.human_sessions import HumanSessionStore, SessionOutcome


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item) + "\n" for item in records))


def test_evidence_snapshot_is_allowlisted_bounded_and_secret_free(tmp_path):
    raw = tmp_path / "raw"
    run = tmp_path / "run"
    raw.mkdir()
    run.mkdir()
    (raw / "instruction.md").write_text("Implement ABC for FP32 and BF16.")
    (raw / "task.toml").write_text('name = "abc"\n')
    (raw / ".env").write_text("API_KEY=sk-never-include-this")
    (run / "manifest.json").write_text(
        json.dumps({"mode": "agent_system", "api_key": "sk-secret-value"})
    )
    _write_jsonl(
        run / "events.jsonl",
        [{"kind": "turn", "i": i, "detail": "x" * 300} for i in range(30)],
    )

    builder = RunEvidenceBuilder(
        run_dir=run,
        raw_input_dir=raw,
        max_total_chars=2_500,
        tail_records=4,
    )
    snapshot = builder.build()
    rendered = builder.render(snapshot)

    assert snapshot["task"]["instruction.md"].startswith("Implement ABC")
    assert len(snapshot["events"]) == 4
    assert len(rendered) <= 2_500
    assert "sk-never-include-this" not in rendered
    assert "sk-secret-value" not in rendered
    assert ".env" not in rendered
    assert "[REDACTED]" in rendered


def test_snapshot_exposes_latest_verifier_diff_candidates_and_probes(tmp_path):
    raw = tmp_path / "raw"
    run = tmp_path / "run"
    raw.mkdir()
    (raw / "instruction.md").write_text("ABC task")
    versions = run / "supervisor" / "verifier_versions"
    versions.mkdir(parents=True)
    (versions / "v0.py").write_text("TOL = 1e-2\n")
    (versions / "v1.py").write_text("TOL_FP32 = 1e-4\nTOL_BF16 = 2e-2\n")
    _write_jsonl(
        run / "supervisor" / "versions.jsonl",
        [{"version": 0}, {"version": 1, "rationale": "dtype-specific"}],
    )
    candidate_dir = run / "solver" / "candidates"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "cand_00001.json").write_text(
        json.dumps({"score": 0.8, "feedback": {"reason": "BF16 drift"}})
    )
    probe_dir = run / "supervisor" / "probes"
    probe_dir.mkdir(parents=True)
    (probe_dir / "probe_0001.json").write_text(
        json.dumps({"fooled": True, "description": "large magnitude BF16"})
    )

    snapshot = RunEvidenceBuilder(run, raw).build()

    assert snapshot["verifier"]["current_version"] == 1
    assert "TOL_BF16" in snapshot["verifier"]["current_source"]
    assert "-TOL = 1e-2" in snapshot["verifier"]["diff_from_previous"]
    assert snapshot["candidates"][-1]["feedback"]["reason"] == "BF16 drift"
    assert snapshot["probes"][-1]["fooled"] is True


def test_codex_agent_prompt_contains_evidence_and_full_session_transcript(tmp_path):
    raw = tmp_path / "raw"
    run = tmp_path / "run"
    raw.mkdir()
    (raw / "instruction.md").write_text("FP32 and BF16 are required")
    store = HumanSessionStore(run)
    session = store.open_session(expert_id="ou_expert", purpose="task contract")
    store.append_message(session.session_id, role="agent", text="你想先看哪个约束？")
    store.append_message(session.session_id, role="human", text="先看 dtype")
    captured: dict = {}

    def fake_runner(workdir, prompt, **kwargs):
        captured["workdir"] = workdir
        captured["prompt"] = prompt
        (workdir / "response.json").write_text(
            json.dumps(
                {
                    "reply": "当前任务同时要求 FP32 和 BF16。",
                    "proposed_outcome": None,
                },
                ensure_ascii=False,
            )
        )
        return AgentSession(ok=True, returncode=0, stdout="ok", stderr="")

    agent = CodexEvidenceAgent(
        run_dir=run,
        raw_input_dir=raw,
        gateway=object(),
        agent_elf=Path("/fake/codex"),
        runner=fake_runner,
    )
    reply = agent.reply(
        session=store.get(session.session_id),
        transcript=store.transcript(session.session_id),
        human_message="这两个 dtype 的容差一样吗？",
        agent_instruction="继续对话",
    )

    assert reply.text == "当前任务同时要求 FP32 和 BF16。"
    prompt = captured["prompt"]
    assert "你想先看哪个约束" in prompt
    assert "先看 dtype" in prompt
    assert "这两个 dtype 的容差一样吗" in prompt
    assert "FP32 and BF16 are required" in prompt
    assert "必须先读取并核对 `/work/context.json`" in prompt
    context = (captured["workdir"] / "context.json").read_text()
    assert "FP32 and BF16 are required" in context


def test_codex_agent_parses_structured_close_outcome(tmp_path):
    raw = tmp_path / "raw"
    run = tmp_path / "run"
    raw.mkdir()
    (raw / "instruction.md").write_text("ABC")
    store = HumanSessionStore(run)
    session = store.open_session(expert_id="ou_expert", purpose="review")

    def fake_runner(workdir, prompt, **kwargs):
        (workdir / "response.json").write_text(
            json.dumps(
                {
                    "reply": "总结如下。请回复确认结束。",
                    "proposed_outcome": {
                        "decision": "guide",
                        "human_guidance": ["split tolerance by dtype"],
                        "unresolved_questions": ["NaN policy"],
                    },
                }
            )
        )
        return AgentSession(ok=True, returncode=0, stdout="", stderr="")

    agent = CodexEvidenceAgent(
        run, raw, gateway=object(), agent_elf=Path("/fake/codex"), runner=fake_runner
    )
    reply = agent.reply(
        session=session,
        transcript=[],
        human_message="结束这轮",
        agent_instruction="总结后要求明确确认结束",
    )

    assert isinstance(reply.proposed_outcome, SessionOutcome)
    assert reply.proposed_outcome.human_guidance == ["split tolerance by dtype"]
    assert reply.proposed_outcome.unresolved_questions == ["NaN policy"]
