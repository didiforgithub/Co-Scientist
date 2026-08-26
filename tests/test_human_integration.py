from __future__ import annotations

import json

import pytest

from coscientist.coevo.agent_system import AgentSystem, _version0
from coscientist.coevo.eval_service import EvalService, FeedbackLevel
from coscientist.coevo.human_sessions import SessionOutcome
from coscientist.demo.evaluator import Evaluator

V0 = """\
def verify(payload, ctx):
    value = float(payload.get("value", -1))
    feasible = 0 <= value <= 10
    return {"feasible": feasible, "raw": value if feasible else -1e9, "artifacts": {}}
"""

V1 = """\
def verify(payload, ctx):
    value = float(payload.get("value", -1))
    feasible = 0 <= value <= 5
    return {"feasible": feasible, "raw": value if feasible else -1e9, "artifacts": {}}
"""


class FakeHumanPort:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def consult(self, *, purpose, context):
        self.calls.append({"purpose": purpose, "context": context})
        return self.outcomes.pop(0) if self.outcomes else None


def _system_with_live_v0(tmp_path, *, human_port=None):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    (raw / "instruction.md").write_text("FP32 and BF16 ABC")
    system = AgentSystem(
        raw_input_dir=raw,
        run_dir=tmp_path / "run",
        budget_s=120,
        feedback_level=FeedbackLevel.WITH_ARTIFACTS,
        human_port=human_port,
    )
    evaluator = Evaluator()
    evaluator.versions.append(_version0(V0))
    system.evaluator = evaluator
    system._ctx = {}
    system._seed = {"value": 1}
    system._probes = [{"description": "over cap", "solution": {"value": 999}}]
    system.eval_service = EvalService(
        evaluator=evaluator,
        ctx_provider=lambda: system._ctx,
        feedback_level=FeedbackLevel.WITH_ARTIFACTS,
    )
    system.store.verifier_version(
        0, V0, origin="agent", note="bootstrap", rationale=""
    )
    return system


def _proposal(tmp_path):
    ws = tmp_path / "proposal"
    ws.mkdir()
    (ws / "verifier.py").write_text(V1)
    return ws


def test_new_run_consults_task_definition_before_bootstrap_and_persists_guidance(
    tmp_path,
):
    port = FakeHumanPort(
        [
            SessionOutcome(
                decision="guide",
                task_contract_updates=["FP32 and BF16 require separate tolerances"],
                human_guidance=["show dtype-specific error evidence"],
            )
        ]
    )
    system = AgentSystem(
        raw_input_dir=tmp_path / "raw",
        run_dir=tmp_path / "run",
        budget_s=60,
        human_port=port,
    )
    system.raw_input_dir.mkdir()
    (system.raw_input_dir / "instruction.md").write_text("ABC")
    order = []
    system.preflight = lambda: order.append("preflight")

    def fake_bootstrap():
        order.append("bootstrap")
        guidance = (system.run_dir / "human_guidance.md").read_text()
        assert "separate tolerances" in guidance

    system.bootstrap = fake_bootstrap
    system.solve_and_evolve = lambda **kwargs: order.append("solve")

    system.run(max_turns=1)

    assert order == ["preflight", "bootstrap", "solve"]
    assert [call["purpose"] for call in port.calls] == ["task_definition"]
    checkpoint = json.loads(
        (system.run_dir / "human" / "checkpoints" / "task_definition.json").read_text()
    )
    assert checkpoint["task_contract_updates"] == [
        "FP32 and BF16 require separate tolerances"
    ]


def test_task_definition_checkpoint_is_not_reopened_on_resume(tmp_path):
    first = FakeHumanPort([SessionOutcome(human_guidance=["first guidance"])])
    system = _system_with_live_v0(tmp_path, human_port=first)
    result = system._consult_human(
        purpose="task_definition", context={"phase": "before_bootstrap"}, checkpoint=True
    )
    assert result is not None and len(first.calls) == 1

    second = FakeHumanPort([SessionOutcome(human_guidance=["must not be consumed"])])
    resumed = AgentSystem(
        raw_input_dir=system.raw_input_dir,
        run_dir=system.run_dir,
        human_port=second,
    )
    restored = resumed._consult_human(
        purpose="task_definition", context={"phase": "resume"}, checkpoint=True
    )

    assert restored.human_guidance == ["first guidance"]
    assert second.calls == []


@pytest.mark.parametrize("decision", ["reject", "guide", "none"])
def test_nonapproved_human_verdict_holds_valid_verifier_change(tmp_path, decision):
    port = FakeHumanPort(
        [SessionOutcome(decision=decision, human_guidance=["hold and inspect BF16"])]
    )
    system = _system_with_live_v0(tmp_path, human_port=port)

    system._apply_harden(
        _proposal(tmp_path), trigger="proactive", verdict={"gaming": True}
    )

    assert system.eval_service.current_version() == 0
    assert system.hardenings == 0
    assert "hold and inspect BF16" in (
        system.run_dir / "human_guidance.md"
    ).read_text()
    assert port.calls[0]["purpose"] == "verifier_change"
    assert port.calls[0]["context"]["current_verifier"] == V0
    assert port.calls[0]["context"]["proposed_verifier"] == V1
    assert "diff" in port.calls[0]["context"]


def test_explicit_human_approval_installs_change_only_after_consult_returns(tmp_path):
    seen_versions = []
    system = _system_with_live_v0(tmp_path)

    class ApprovingPort:
        def consult(self, *, purpose, context):
            seen_versions.append(system.eval_service.current_version())
            assert purpose == "verifier_change"
            assert "comparison_cases" not in context
            return SessionOutcome(
                decision="approve",
                approved_changes=["cap is now 5"],
                human_rationale="probe shows the old cap was gameable",
            )

    system.human_port = ApprovingPort()
    system._apply_harden(
        _proposal(tmp_path), trigger="proactive", verdict={"gaming": True}
    )

    assert seen_versions == [0], "evaluator must remain frozen while the session is open"
    assert system.eval_service.current_version() == 1
    assert system.hardenings == 1


def test_configured_human_mode_holds_change_when_no_approval_is_available(tmp_path):
    system = _system_with_live_v0(tmp_path, human_port=FakeHumanPort([None]))

    system._apply_harden(
        _proposal(tmp_path), trigger="budget_exhausted", verdict={"gaming": True}
    )

    assert system.eval_service.current_version() == 0
    assert system.hardenings == 0


def test_disabled_human_mode_preserves_autonomous_install_behavior(tmp_path):
    system = _system_with_live_v0(tmp_path, human_port=None)

    system._apply_harden(
        _proposal(tmp_path), trigger="proactive", verdict={"gaming": True}
    )

    assert system.eval_service.current_version() == 1
    assert system.hardenings == 1


def test_cli_wires_feishu_expert_and_listener_configuration(tmp_path, monkeypatch):
    import argparse

    from coscientist.coevo import agent_system as agent_system_module
    from coscientist.coevo import cli

    raw = tmp_path / "raw_cli"
    raw.mkdir()
    (raw / "instruction.md").write_text("ABC")
    captured = {}

    class FakeSystem:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.budget_s = kwargs["budget_s"]

        def run(self, **kwargs):
            return self

        def summary(self):
            return {
                "final_verifier_version": 0,
                "verifier_hardenings": 0,
                "reviews_handled": 0,
                "best_score": None,
            }

    monkeypatch.setattr(agent_system_module, "AgentSystem", FakeSystem)
    args = argparse.Namespace(
        input=str(raw),
        problem="curvefit",
        runs_dir=str(tmp_path / "runs"),
        run_id="human-cli",
        budget_s=60.0,
        budget_hours=0.0,
        feedback="with_artifacts",
        max_turns=1,
        resume=False,
        solver_strength="weak",
        solver_image=None,
        solver_gpus=None,
        llm_config=None,
        feishu_expert_id="ou_expert",
        lark_cli_executable="/opt/homebrew/bin/lark-cli",
        human_agent_timeout_s=90.0,
        human_agent_model="human-fast-model",
        human_agent_reasoning_effort="low",
    )

    cli.run_agent_system(args)

    assert captured["human_expert_id"] == "ou_expert"
    assert captured["lark_cli_executable"] == "/opt/homebrew/bin/lark-cli"
    assert captured["human_agent_timeout_s"] == 90.0
    assert captured["human_agent_model"] == "human-fast-model"
    assert captured["human_agent_reasoning_effort"] == "low"


def test_cli_wires_private_text_context_for_model_human_proxy(tmp_path, monkeypatch):
    import argparse

    from coscientist.coevo import agent_system as agent_system_module
    from coscientist.coevo import cli

    raw = tmp_path / "raw_proxy_cli"
    raw.mkdir()
    (raw / "instruction.md").write_text("ABC")
    proxy_context = tmp_path / "private_evaluator_context.md"
    proxy_context.write_text("The real evaluator cares about target semantics.")
    captured = {}

    class FakeSystem:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.budget_s = kwargs["budget_s"]

        def run(self, **kwargs):
            return self

        def summary(self):
            return {
                "final_verifier_version": 0,
                "verifier_hardenings": 0,
                "reviews_handled": 0,
                "best_score": None,
            }

    monkeypatch.setattr(agent_system_module, "AgentSystem", FakeSystem)
    args = argparse.Namespace(
        input=str(raw), problem="curvefit", runs_dir=str(tmp_path / "runs"),
        run_id="proxy-cli", budget_s=60.0, budget_hours=0.0,
        feedback="with_artifacts", max_turns=1, resume=False,
        solver_strength="weak", solver_image=None, solver_gpus=None,
        llm_config=None, feishu_expert_id=None,
        human_proxy_context=str(proxy_context),
        human_proxy_context_sha256="a" * 64,
        human_agent_timeout_s=90.0,
        human_agent_model="proxy-model",
        human_agent_reasoning_effort="medium",
    )

    cli.run_agent_system(args)

    assert captured["human_proxy_context_path"] == proxy_context
    assert captured["human_proxy_context_sha256"] == "a" * 64
    assert captured["human_agent_timeout_s"] == 90.0
    assert captured["human_agent_model"] == "proxy-model"
    assert captured["human_agent_reasoning_effort"] == "medium"
