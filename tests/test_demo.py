"""Behavioural tests for the two-loop co-evolution demo.

These pin the demo's claims the same way ``test_mvp.py`` pins the single-loop
MVP: they assert the *shape* of the co-evolution (both loops run, the human
gates the evaluator change, the fix removes the exploit) rather than exact
numbers, so refactors can't silently hollow out the demonstration.

Everything runs on the offline stub backend + AutoHuman — no network, no keys.
"""

from __future__ import annotations

from coscientist.demo import taskspec
from coscientist.demo.agent_backend import make_backend
from coscientist.demo.evaluator import Evaluator
from coscientist.demo.eventlog import EventLog
from coscientist.demo.human_port import AutoHuman
from coscientist.demo.orchestrator import Orchestrator, OrchestratorConfig
from coscientist.demo.proposers import (
    EvaluatorProposer,
    SolutionProposer,
    default_stub_handlers,
)


def _build(review_every=2, rounds=8):
    evaluator = Evaluator.initial()
    handlers = default_stub_handlers()
    backend = make_backend("stub", handlers=handlers)
    return Orchestrator(
        evaluator=evaluator,
        solution_proposer=SolutionProposer(backend),
        evaluator_proposer=EvaluatorProposer(backend),
        human_port=AutoHuman(evaluator),
        config=OrchestratorConfig(outer_rounds=rounds, inner_steps=3, k_candidates=3, review_every=review_every),
        log=EventLog(),
    )


def test_initial_verifier_is_gameable():
    """The shipped verifier rewards overfitting; V* does not."""
    ev = Evaluator.initial()
    ctx = taskspec.workspace_context()
    small = taskspec.ls_fit(taskspec._FREQ_POOL[:2])
    deep = taskspec.ls_fit(taskspec._FREQ_POOL[:18])
    proxy_small = ev.run(small, ctx).raw
    proxy_deep = ev.run(deep, ctx).raw
    _, real_small, _ = taskspec.reference_score(small)
    _, real_deep, _ = taskspec.reference_score(deep)
    assert proxy_deep > proxy_small, "flawed proxy should reward the deeper overfit"
    assert real_deep < real_small, "reference should punish the deeper overfit"


def test_hardened_verifier_kills_the_exploit():
    """The hardened rewrite makes overfitting stop paying under the proxy."""
    from coscientist.demo.proposers import _HARDENED_VERIFIER_SRC

    ev = Evaluator.initial()
    ctx = taskspec.workspace_context()
    small = taskspec.ls_fit(taskspec._FREQ_POOL[:2])
    deep = taskspec.ls_fit(taskspec._FREQ_POOL[:18])
    proxy_small = ev.run(small, ctx, source=_HARDENED_VERIFIER_SRC).raw
    proxy_deep = ev.run(deep, ctx, source=_HARDENED_VERIFIER_SRC).raw
    assert proxy_deep < proxy_small, "hardened proxy must not reward the overfit"


def test_both_loops_run_and_human_gates_the_evaluator():
    orch = _build().run()
    assert len(orch.archive) > 10, "solution loop should produce many candidates"
    assert orch.reviews >= 1, "evaluator loop should trigger at least one human review"
    assert orch.verifier_changes >= 1, "a ratified verifier change should occur"
    assert len(orch.evaluator.versions) >= 2, "the evaluator should have evolved"
    # the change must have gone through the human port (a human_review event exists)
    assert len(orch.log.of_kind("human_review")) >= 1


def test_verifier_change_requires_the_human_port():
    """With a rejecting human, the verifier never changes even though it's gamed."""
    from coscientist.demo.human_port import Decision, ReviewResponse

    class RejectingHuman:
        calls = 0

        def review(self, req):
            self.calls += 1
            return ReviewResponse(Decision.REJECT, dense_text="no")

    orch = _build()
    orch.human_port = RejectingHuman()
    orch.run()
    assert orch.reviews >= 1, "review should still be requested"
    assert orch.verifier_changes == 0, "a rejecting human must block all verifier changes"
    assert len(orch.evaluator.versions) == 1, "verifier stays at the initial flawed version"


def test_rescore_lowers_the_gamed_best_score():
    """After hardening, the archive's best proxy score drops (exploit removed)."""
    orch = _build().run()
    rescored = orch.log.of_kind("archive_rescored")
    assert rescored, "a rescore should happen after the verifier evolves"
    # the winning solution under the hardened verifier should be a modest basis,
    # not a deep overfit (real score should be decent, not catastrophic).
    assert orch.best.real_score > -1.0, "post-fix best should have good real performance"
    assert orch.best.n_modes <= 6, "post-fix best should not be a deep overfit"


# ---------------------------------------------------------------------------
# Harbor container runtime (offline: packaging + parsing, no docker required)
# ---------------------------------------------------------------------------
import json
from pathlib import Path

try:
    import tomllib as tomli  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10
    import tomli

from coscientist.demo import harbor_runtime as hr
from coscientist.demo.agent_backend import HarborBackend, make_backend
from coscientist.demo.workspace import Workspace


def _seed_solution_workspace(ws: Workspace) -> None:
    """Seed a workspace exactly as SolutionProposer.propose does."""
    ws.write("TASK", "solution")
    ws.write("PROMPT.md", "improve the fit")
    ws.write_json("context.json", taskspec.workspace_context())
    ws.write_json("solution_in.json", taskspec.ls_fit(taskspec._FREQ_POOL[:2]))
    ws.write("verifier.py", Evaluator.initial().current.source)
    ws.write("SEED", "1")


def _seed_evaluator_workspace(ws: Workspace) -> None:
    """Seed a workspace exactly as EvaluatorProposer.propose does."""
    ws.write("TASK", "evaluator")
    ws.write("PROMPT.md", "harden the verifier")
    ws.write("verifier.py", Evaluator.initial().current.source)
    ws.write_json("evidence.json", {"proxy_minus_real": 4.8, "top_n_modes": 18})
    ws.write_json("context.json", taskspec.workspace_context())
    ws.write("SEED", "1")


def test_pack_task_emits_valid_harbor_package():
    for role, seed, output_rel in (
        ("solution", _seed_solution_workspace, "solution_out.json"),
        ("evaluator", _seed_evaluator_workspace, "verifier_out.py"),
    ):
        with Workspace() as ws:
            seed(ws)
            pkg = hr.pack_task(ws.root, role=role, isolation="separate",
                               dest=ws.root.parent / f"{ws.root.name}__pkg_{role}")
            try:
                toml = tomli.loads((pkg / "task.toml").read_text())
                assert toml["schema_version"] == "1.3"
                # the recovery channel names exactly this role's output file
                assert toml["artifacts"] == [f"/app/{output_rel}"]
                assert (pkg / "tests" / "test.sh").is_file()
                dockerfile = (pkg / "environment" / "Dockerfile").read_text()
                # the seeded inputs are COPYd into the build context
                assert "COPY verifier.py /app/verifier.py" in dockerfile
                assert (pkg / "instruction.md").is_file()
            finally:
                import shutil
                shutil.rmtree(pkg, ignore_errors=True)


def test_separate_isolation_declares_verifier_environment():
    with Workspace() as ws:
        _seed_solution_workspace(ws)
        sep = hr.pack_task(ws.root, role="solution", isolation="separate",
                           dest=ws.root.parent / f"{ws.root.name}__sep")
        shared = hr.pack_task(ws.root, role="solution", isolation="shared",
                              dest=ws.root.parent / f"{ws.root.name}__shared")
        try:
            sep_toml = tomli.loads((sep / "task.toml").read_text())
            shared_toml = tomli.loads((shared / "task.toml").read_text())
            assert sep_toml["verifier"].get("environment_mode") == "separate"
            # [verifier.environment] parses as a nested table under verifier
            assert "environment" in sep_toml["verifier"]
            assert sep_toml["verifier"]["environment"]["network_mode"] == "no-network"
            # shared omits the separate-grader declaration (AutoLab model)
            assert shared_toml["verifier"].get("environment_mode") != "separate"
            assert "environment" not in shared_toml["verifier"]
        finally:
            import shutil
            shutil.rmtree(sep, ignore_errors=True)
            shutil.rmtree(shared, ignore_errors=True)


def test_vstar_never_leaks_into_task_package():
    """Load-bearing: the hidden reference evaluator (V*) must never be packaged."""
    for role, seed in (("solution", _seed_solution_workspace),
                       ("evaluator", _seed_evaluator_workspace)):
        with Workspace() as ws:
            seed(ws)
            pkg = hr.pack_task(ws.root, role=role, isolation="separate",
                               dest=ws.root.parent / f"{ws.root.name}__leak_{role}")
            try:
                # pack_task already ran assert_no_reference_leak; assert directly too.
                hr.assert_no_reference_leak(pkg)
                blob = "\n".join(
                    p.read_text(errors="ignore")
                    for p in pkg.rglob("*") if p.is_file()
                )
                for marker in ("reference_score", "reduced_chi2_vs_truth", "INSTANCE.signal"):
                    assert marker not in blob, f"V* marker {marker!r} leaked into {role} package"
            finally:
                import shutil
                shutil.rmtree(pkg, ignore_errors=True)


def test_leak_guard_trips_on_injected_vstar():
    """A workspace carrying V* source is rejected by the guard."""
    import pytest
    with Workspace() as ws:
        _seed_solution_workspace(ws)
        # simulate a mistake: someone writes the reference scorer into the workspace
        ws.write("oops.py", "def reference_score(payload):\n    return 0.0\n")
        with pytest.raises(AssertionError):
            hr.pack_task(ws.root, role="solution",
                         dest=ws.root.parent / f"{ws.root.name}__trip")


def test_parse_job_dir_recovers_output(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    trial = jobs_dir / "cosci_solution" / "trial_abc"
    (trial / "artifacts" / "app").mkdir(parents=True)
    payload = {"trend": [0.5, 2.0], "modes": []}
    (trial / "artifacts" / "app" / "solution_out.json").write_text(json.dumps(payload))
    (trial / "result.json").write_text(json.dumps({"trial_name": "trial_abc"}))

    ok, text, note = hr.parse_job_dir(jobs_dir, "cosci_solution", output_rel="solution_out.json")
    assert ok
    assert json.loads(text) == payload
    assert "trial_abc" in note

    # missing output -> clean failure, not a crash
    ok2, text2, _ = hr.parse_job_dir(jobs_dir, "cosci_solution", output_rel="verifier_out.py")
    assert not ok2 and text2 is None


def test_harbor_backend_absent_docker_is_clean(monkeypatch):
    """With no harbor/docker, run_session returns a clean failure, never raises."""
    monkeypatch.setattr(hr.shutil, "which", lambda *_: None, raising=False)
    import coscientist.demo.agent_backend as ab
    monkeypatch.setattr(ab.shutil, "which", lambda *_: None, raising=False)

    backend = make_backend("harbor", isolation="separate")
    assert isinstance(backend, HarborBackend)
    with Workspace() as ws:
        _seed_solution_workspace(ws)
        res = backend.run_session(workspace=ws.root, prompt="x", model=None, timeout_s=5)
    assert res.ok is False
    assert "unavailable" in (res.error or "")

