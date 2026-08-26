from __future__ import annotations

import hashlib
import base64
import json
import stat
import subprocess
import threading
from pathlib import Path

import pytest

from coscientist.experiments.autolab_human_proxy import (
    ACCEPTED_ARTIFACTS,
    AUTOLAB_GROUPS,
    MAX_CONTEXT_CHARACTERS,
    audit_contexts,
    audit_live_privacy,
    audit_runs,
    evaluate_results,
    main,
    prepare_contexts,
    prepare_runs,
)


FROZEN_TAIL = '''## Frozen evaluation contract

Your submission is evaluated only by this task's original shipped `tests/test.sh`.
The evaluator returns `reward.json["reward"]` unchanged as the raw score; higher is better.
The verifier is frozen at version 0 for this control run. No verifier hardening or
Bootstrap-authored hidden checks will be added during the run.
'''


def _corrected_control_fixture(root: Path, problems_root: Path) -> Path:
    root.mkdir()
    problems_root.mkdir()
    for tasks in AUTOLAB_GROUPS.values():
        for task in tasks:
            run = root / (
                "autolab_shippedv0_control17_4h_v2_20260825"
                f"__autolab_{task}"
            )
            bws = run / "bootstrap_ws"
            vdir = run / "supervisor" / "verifier_versions"
            checker = run / "checker"
            bws.mkdir(parents=True)
            vdir.mkdir(parents=True)
            (checker / "tests").mkdir(parents=True)
            (checker / "tests" / "test.sh").write_text(
                f"#!/bin/sh\n# shipped weak test for {task}\n", encoding="utf-8"
            )
            (checker / "instruction.md").write_text(
                f"task={task}\n", encoding="utf-8"
            )
            verifier = f"# mechanical shipped-V0 adapter for {task}\n"
            (bws / "verifier.py").write_text(verifier, encoding="utf-8")
            (vdir / "v0.py").write_text(verifier, encoding="utf-8")
            (bws / "seed_solution.json").write_text(
                json.dumps({"source": f"seed-{task}"}), encoding="utf-8"
            )
            (bws / "ctx.json").write_text(
                json.dumps({"adapter": "AUTOLAB_SHIPPED_TEST_ADAPTER_V1"}),
                encoding="utf-8",
            )
            (bws / "probes.json").write_text("[]\n", encoding="utf-8")
            (bws / "solver_env.json").write_text("{}\n", encoding="utf-8")
            (bws / "reframe_policy.json").write_text(
                json.dumps({"admits_proof": False, "provable_claim": ""}),
                encoding="utf-8",
            )
            (bws / "SOLVER_BRIEF.md").write_text(
                f"# AutoLab task: {task}\n\nSolve it.\n\n{FROZEN_TAIL}",
                encoding="utf-8",
            )
            (run / "manifest.json").write_text(
                json.dumps(
                    {
                        "raw_input_dir": f"/stale/main/autolab_{task}",
                        "initial_feedback_level": "with_artifacts",
                        "initial_verifier_origin": "autolab_shipped_tests/test.sh",
                        "initial_verifier_sha256": _sha256(vdir / "v0.py"),
                        "original_test_sha256": _sha256(checker / "tests" / "test.sh"),
                    }
                ),
                encoding="utf-8",
            )
            (run / "events.jsonl").write_text(
                '{"kind":"bootstrap_start"}\n{"kind":"run_stop"}\n',
                encoding="utf-8",
            )
            (run / "supervisor" / "versions.jsonl").write_text(
                '{"version":0,"origin":"old control"}\n', encoding="utf-8"
            )
            (run / "solver" / "candidates").mkdir(parents=True)
            (run / "solver" / "candidates" / "cand_00000.json").write_text(
                '{"payload":{"source":"old-result"}}', encoding="utf-8"
            )
            (run / "result.json").write_text('{"old":true}', encoding="utf-8")
            problem = problems_root / f"autolab_{task}"
            (problem / "tests").mkdir(parents=True)
            (problem / "tests" / "test.sh").write_bytes(
                (checker / "tests" / "test.sh").read_bytes()
            )
    return root


def _private_context_text(task: str) -> str:
    return (
        f"PRIVATE FINAL EVALUATOR SECTION FOR {task}. "
        + "This accepted verifier binds every hidden sentinel dimension and payload "
        "branch to the protected deterministic benchmark contract, rejects malformed "
        "numeric encodings before scoring, validates the task-specific correctness "
        "oracle on held-out inputs, and prevents benchmark recognition shortcuts. "
        "Only high-level semantic guidance may leave the Human Proxy boundary. "
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accepted_fixture(root: Path, *, oversize_task: str | None = None) -> Path:
    root.mkdir()
    task_records = []
    for group, tasks in AUTOLAB_GROUPS.items():
        for task in tasks:
            package = root / group / task
            package.mkdir(parents=True)
            contents = {
                "final_verifier.py": (
                    f"# accepted verifier for {task}\n"
                    + (
                        "x" * MAX_CONTEXT_CHARACTERS
                        if task == oversize_task
                        else "def score(payload, ctx):\n    return 1.0\n"
                    )
                ),
                "final_ctx.template.json": json.dumps(
                    {
                        "task": task,
                        "checker_dir": "REQUIRED_CHECKER_DIR",
                        "hidden_seed": "REQUIRED_PRIVATE",
                        "validation_mode": False,
                    },
                    indent=2,
                ),
                "authored_design.md": f"# Design for {task}\nReject semantic hacks.\n",
                "issues.md": f"# Issues for {task}\nTiming noise remains.\n",
            }
            for name, content in contents.items():
                (package / name).write_text(content, encoding="utf-8")
            task_records.append(
                {
                    "task": task,
                    "group": group,
                    "ast_parse": "pass",
                    "deployment_template": "pass",
                    "py_compile": "pass",
                    "validation_json_parse": "pass",
                    "validation_live_hash_pair": "pass",
                    "runtime_entropy_hits": 0,
                    "seed_or_nonce_artifact_hits": 0,
                    "fixed_formal_fallback_hits": 0,
                    "final_verifier_sha256": _sha256(
                        package / "final_verifier.py"
                    ),
                    "final_ctx_template_sha256": _sha256(
                        package / "final_ctx.template.json"
                    ),
                    "required_artifacts": [
                        f"{group}/{task}/{name}" for name in ACCEPTED_ARTIFACTS
                    ],
                }
            )
    report = {
        "schema_version": 1,
        "status": "pass",
        "failures": [],
        "aggregate": {"accepted_tasks": 17, "expected_tasks": 17},
        "results": {
            "exact_task_set": {
                group: list(tasks) for group, tasks in AUTOLAB_GROUPS.items()
            }
        },
        "tasks": task_records,
    }
    (root / "acceptance_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return root


def _evaluation_fixture(
    tmp_path: Path, *, completed: bool
) -> tuple[Path, Path, Path, str]:
    """Build a dedicated target batch plus independent trusted comparators."""

    accepted = _accepted_fixture(tmp_path / "accepted")
    runs_root = tmp_path / "target-runs"
    trusted_root = tmp_path / "trusted-runs"
    batch_name = "autolab_hp_4h_test"
    rows = []
    for group, tasks in AUTOLAB_GROUPS.items():
        checker_batch = "autolab_all_4h" if group == "old10" else "autolab_proofgate_4h"
        for task in tasks:
            trusted = trusted_root / f"{checker_batch}__autolab_{task}"
            (trusted / "checker" / "tests").mkdir(parents=True)
            (trusted / "checker" / "tests" / "test.sh").write_text(
                f"trusted checker {checker_batch} {task}\n", encoding="utf-8"
            )
            (trusted / "manifest.json").write_text(
                json.dumps(
                    {
                        "resource_spec": {
                            "verifier": {
                                "image": f"autolab_{task}:latest",
                                "cpus": 2.0,
                                "memory_mb": 4096,
                                "timeout_sec": 37.0,
                                "allow_internet": False,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            run_id = f"{batch_name}__autolab_{task}"
            run = runs_root / run_id
            run.mkdir(parents=True)
            manifest = {
                "arm": "autolab_coevolve_human_proxy",
                "freeze_verifier": False,
                "initial_verifier_origin": "autolab_shipped_tests/test.sh",
                "final_verifier_version": 3,
                "verifier_hardenings": 3,
            }
            if completed:
                manifest["best_solution"] = {
                    "source": f"final payload for {task}",
                    "task_marker": task,
                }
                manifest["best_score"] = float(len(task))
                (run / "events.jsonl").write_text(
                    '{"kind":"run_stop"}\n', encoding="utf-8"
                )
            (run / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            rows.append(
                {
                    "run_id": run_id,
                    "input_dir": f"/problems/autolab_{task}",
                    "status": "done" if completed else "pending",
                }
            )
    batch_dir = runs_root / batch_name
    batch_dir.mkdir()
    (batch_dir / "batch.json").write_text(
        json.dumps({"batch": batch_name, "runs": rows}), encoding="utf-8"
    )
    return runs_root, accepted, trusted_root, batch_name


class _FakeDocker:
    def __init__(
        self, *, seed_forbidden: str | None = None, leak_seed_in_result: bool = False
    ):
        self.commands: list[list[str]] = []
        self.seed_forbidden = seed_forbidden
        self.leak_seed_in_result = leak_seed_in_result

    def __call__(self, argv, **kwargs):
        argv = [str(value) for value in argv]
        self.commands.append(argv)
        if self.seed_forbidden is not None:
            assert self.seed_forbidden not in " ".join(argv)
        if argv[:3] == ["docker", "image", "inspect"]:
            task = argv[3].removeprefix("autolab_").removesuffix(":latest")
            digest = hashlib.sha256(task.encode()).hexdigest()
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=json.dumps(
                    {
                        "Id": f"sha256:{digest}",
                        "RepoDigests": [f"example/{task}@sha256:{digest}"],
                        "Created": "2026-08-27T00:00:00Z",
                    }
                ),
                stderr="",
            )
        assert argv[:2] == ["docker", "run"]
        assert argv[argv.index("--network") + 1] == "none"
        image = next(value for value in argv if value.startswith("sha256:"))
        assert len(image) == 71
        mounts = [argv[index + 1] for index, value in enumerate(argv) if value == "-v"]
        assert all(mount.endswith(":ro") for mount in mounts)
        runtime_mount = next(mount for mount in mounts if mount.endswith("/input/runtime_ctx.json:ro"))
        payload_mount = next(mount for mount in mounts if mount.endswith("/input/payload.json:ro"))
        runtime = json.loads(Path(runtime_mount.split(":", 1)[0]).read_text())
        payload = json.loads(Path(payload_mount.split(":", 1)[0]).read_text())
        assert runtime["checker_dir"] == "/checker"
        assert runtime["validation_mode"] is False
        task = payload["task_marker"]
        result = {
            "feasible": True,
            "raw": float(len(task)),
            "artifacts": {"stage": "scored", "task": task},
        }
        if self.leak_seed_in_result:
            result["artifacts"]["forbidden"] = runtime["hidden_seed"]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps({"result": result, "duration_seconds": 0.25}) + "\n",
            stderr="",
        )


def _assert_secret_absent(value, secret: str) -> None:
    assert secret not in json.dumps(value, ensure_ascii=False, sort_keys=True)


def test_prepare_renders_deterministic_contexts_and_complete_hash_inventory(tmp_path):
    accepted = _accepted_fixture(tmp_path / "accepted")
    first = tmp_path / "private-first"
    second = tmp_path / "private-second"

    manifest = prepare_contexts(accepted, first)
    repeated = prepare_contexts(accepted, second)

    assert manifest == repeated
    assert manifest["arm"] == "autolab_coevolve_human_proxy"
    assert manifest["accepted_report"]["sha256"] == _sha256(
        accepted / "acceptance_report.json"
    )
    assert {item["task"] for item in manifest["tasks"]} == {
        task for tasks in AUTOLAB_GROUPS.values() for task in tasks
    }
    assert len(manifest["tasks"]) == 17
    for item in manifest["tasks"]:
        context = first / item["context_path"]
        duplicate = second / item["context_path"]
        text = context.read_text(encoding="utf-8")
        assert context.read_bytes() == duplicate.read_bytes()
        assert item["context_sha256"] == _sha256(context)
        assert item["context_bytes"] == len(context.read_bytes())
        assert item["context_characters"] == len(text)
        assert item["context_characters"] <= MAX_CONTEXT_CHARACTERS
        assert Path(item["package_path"]) == (
            accepted / item["group"] / item["task"]
        ).resolve()
        assert [source["name"] for source in item["sources"]] == list(
            ACCEPTED_ARTIFACTS
        )
        for source in item["sources"]:
            source_path = Path(source["path"])
            assert source["sha256"] == _sha256(source_path)
            assert source["bytes"] == len(source_path.read_bytes())
            assert source["characters"] == len(
                source_path.read_text(encoding="utf-8")
            )
            assert f"===== BEGIN {source['name']} =====" in text
            assert f"===== END {source['name']} =====" in text
    assert json.loads((first / "manifest.json").read_text()) == manifest


def test_prepare_and_audit_enforce_private_permissions(tmp_path):
    accepted = _accepted_fixture(tmp_path / "accepted")
    output = tmp_path / "contexts"

    prepare_contexts(accepted, output)
    audited = audit_contexts(accepted, output)

    assert audited["task_count"] == 17
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for path in output.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("damage", ["missing", "context_drift", "source_drift"])
def test_prepare_refuses_to_reuse_incomplete_or_mismatched_output(tmp_path, damage):
    accepted = _accepted_fixture(tmp_path / "accepted")
    output = tmp_path / "contexts"
    prepare_contexts(accepted, output)
    task = AUTOLAB_GROUPS["old10"][0]

    if damage == "missing":
        (output / f"autolab_{task}.md").unlink()
    elif damage == "context_drift":
        (output / f"autolab_{task}.md").write_text("tampered", encoding="utf-8")
    else:
        (accepted / "old10" / task / "issues.md").write_text(
            "changed after generation", encoding="utf-8"
        )

    with pytest.raises(RuntimeError, match="refus|mismatch|missing|drift"):
        prepare_contexts(accepted, output)


@pytest.mark.parametrize(
    "problem",
    ["task_set", "status", "reported_hash", "missing_hash", "malformed_hash"],
)
def test_prepare_requires_the_exact_accepted_report_contract(tmp_path, problem):
    accepted = _accepted_fixture(tmp_path / "accepted")
    report_path = accepted / "acceptance_report.json"
    report = json.loads(report_path.read_text())
    if problem == "task_set":
        report["results"]["exact_task_set"]["new7"].pop()
    elif problem == "status":
        report["status"] = "fail"
    elif problem == "reported_hash":
        report["tasks"][0]["final_verifier_sha256"] = "0" * 64
    elif problem == "missing_hash":
        del report["tasks"][0]["final_ctx_template_sha256"]
    else:
        report["tasks"][0]["final_verifier_sha256"] = "not-a-sha256"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="accept|task|hash"):
        prepare_contexts(accepted, tmp_path / "contexts")


def test_prepare_rejects_a_context_above_the_character_limit(tmp_path):
    task = AUTOLAB_GROUPS["old10"][0]
    accepted = _accepted_fixture(tmp_path / "accepted", oversize_task=task)

    with pytest.raises(ValueError, match="200000"):
        prepare_contexts(accepted, tmp_path / "contexts")


def test_prepare_rejects_symlinks_anywhere_in_accepted_or_output_path(tmp_path):
    accepted = _accepted_fixture(tmp_path / "accepted-real")
    accepted_alias = tmp_path / "accepted-alias"
    accepted_alias.symlink_to(accepted, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        prepare_contexts(accepted_alias, tmp_path / "contexts-a")

    private_parent = tmp_path / "private-real"
    private_parent.mkdir()
    private_alias = tmp_path / "private-alias"
    private_alias.symlink_to(private_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prepare_contexts(accepted, private_alias / "contexts-b")


def test_prepare_rejects_output_below_the_accepted_package_root(tmp_path):
    accepted = _accepted_fixture(tmp_path / "accepted")

    with pytest.raises(ValueError, match="accepted"):
        prepare_contexts(accepted, accepted / "private-contexts")


def test_prepare_rejects_oversized_source_before_reading_it(tmp_path, monkeypatch):
    accepted = _accepted_fixture(tmp_path / "accepted")
    task = AUTOLAB_GROUPS["old10"][0]
    oversized = accepted / "old10" / task / "issues.md"
    oversized.write_bytes(b"x" * 1_100_000)
    original_read_bytes = Path.read_bytes

    def forbid_oversized_read(path):
        if path == oversized:
            raise AssertionError("oversized source was read")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", forbid_oversized_read)

    with pytest.raises(ValueError, match="too large"):
        prepare_contexts(accepted, tmp_path / "contexts")


def test_prepare_audits_a_private_staging_directory_before_atomic_publish(
    tmp_path, monkeypatch
):
    from coscientist.experiments import autolab_human_proxy as experiment

    accepted = _accepted_fixture(tmp_path / "accepted")
    output = tmp_path / "contexts"
    original_audit = experiment.audit_contexts
    observed = {}

    def inspect_staging(accepted_root, candidate):
        candidate = Path(candidate)
        observed["path"] = candidate
        observed["mode"] = stat.S_IMODE(candidate.stat().st_mode)
        observed["output_exists"] = output.exists()
        return original_audit(accepted_root, candidate)

    monkeypatch.setattr(experiment, "audit_contexts", inspect_staging)

    prepare_contexts(accepted, output)

    assert observed["path"].parent == output.parent
    assert observed["path"] != output
    assert observed["mode"] == 0o700
    assert observed["output_exists"] is False
    assert output.is_dir()


def test_prepare_cleans_failed_staging_and_can_retry_safely(tmp_path, monkeypatch):
    from coscientist.experiments import autolab_human_proxy as experiment

    accepted = _accepted_fixture(tmp_path / "accepted")
    output = tmp_path / "contexts"
    original_audit = experiment.audit_contexts

    def fail_audit(*args, **kwargs):
        raise RuntimeError("synthetic staging audit failure")

    monkeypatch.setattr(experiment, "audit_contexts", fail_audit)
    with pytest.raises(RuntimeError, match="synthetic"):
        prepare_contexts(accepted, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".contexts.tmp.*"))

    monkeypatch.setattr(experiment, "audit_contexts", original_audit)
    prepare_contexts(accepted, output)
    assert output.is_dir()


def test_audit_rejects_oversized_output_before_reading_it(tmp_path, monkeypatch):
    accepted = _accepted_fixture(tmp_path / "accepted")
    output = tmp_path / "contexts"
    manifest = prepare_contexts(accepted, output)
    context = output / manifest["tasks"][0]["context_path"]
    context.write_bytes(b"x" * 1_100_000)
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def forbid_read_bytes(path):
        if path == context:
            raise AssertionError("oversized output was read")
        return original_read_bytes(path)

    def forbid_read_text(path, *args, **kwargs):
        if path == context:
            raise AssertionError("oversized output was read")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", forbid_read_bytes)
    monkeypatch.setattr(Path, "read_text", forbid_read_text)

    with pytest.raises(RuntimeError, match="size|large|mismatch"):
        audit_contexts(accepted, output)


def test_prepare_runs_copies_only_authoritative_initial_state_and_rewrites_brief(
    tmp_path,
):
    controls = _corrected_control_fixture(
        tmp_path / "controls", tmp_path / "problems"
    )
    contexts = tmp_path / "private-contexts"
    contexts.mkdir()
    for tasks in AUTOLAB_GROUPS.values():
        for task in tasks:
            (contexts / f"autolab_{task}.md").write_text(
                _private_context_text(task), encoding="utf-8"
            )
    runs = tmp_path / "runs"

    summary = prepare_runs(
        controls,
        runs,
        "autolab_hp_4h_test",
        problems_root=tmp_path / "problems",
    )

    assert summary["task_count"] == 17
    assert summary["status"] == "pass"
    for tasks in AUTOLAB_GROUPS.values():
        for task in tasks:
            source = controls / (
                "autolab_shippedv0_control17_4h_v2_20260825"
                f"__autolab_{task}"
            )
            prepared = runs / f"autolab_hp_4h_test__autolab_{task}"
            assert (prepared / "bootstrap_ws" / "verifier.py").read_bytes() == (
                source / "bootstrap_ws" / "verifier.py"
            ).read_bytes()
            assert (prepared / "supervisor/verifier_versions/v0.py").read_bytes() == (
                source / "supervisor/verifier_versions/v0.py"
            ).read_bytes()
            assert (prepared / "bootstrap_ws/seed_solution.json").read_bytes() == (
                source / "bootstrap_ws/seed_solution.json"
            ).read_bytes()
            assert _tree_hash(prepared / "checker") == _tree_hash(source / "checker")
            assert (prepared / "checker/tests/test.sh").read_bytes() == (
                source / "checker/tests/test.sh"
            ).read_bytes()
            brief = (prepared / "bootstrap_ws/SOLVER_BRIEF.md").read_text()
            assert brief.startswith(f"# AutoLab task: {task}\n")
            assert "## Evolving evaluation contract" in brief
            assert "Frozen evaluation contract" not in brief
            assert "verifier hardening is enabled" in brief
            manifest = json.loads((prepared / "manifest.json").read_text())
            assert manifest["freeze_verifier"] is False
            assert manifest["initial_verifier_origin"] == (
                "autolab_shipped_tests/test.sh"
            )
            assert manifest["raw_input_dir"] == str(
                (tmp_path / "problems" / f"autolab_{task}").resolve()
            )
            assert json.loads(
                (prepared / "bootstrap_ws/reframe_policy.json").read_text()
            )["admits_proof"] is False
            event_kinds = [
                json.loads(line)["kind"]
                for line in (prepared / "events.jsonl").read_text().splitlines()
            ]
            assert event_kinds == ["shipped_verifier_preseed"]
            assert not (prepared / "solver/candidates/cand_00000.json").exists()
            assert not (prepared / "result.json").exists()

    audited = audit_runs(
        controls,
        runs,
        "autolab_hp_4h_test",
        problems_root=tmp_path / "problems",
        context_dir=contexts,
    )
    assert audited == {
        "arm": "autolab_coevolve_human_proxy",
        "batch_name": "autolab_hp_4h_test",
        "status": "pass",
        "task_count": 17,
    }


def test_prepare_runs_refuses_to_overwrite_any_existing_run(tmp_path):
    controls = _corrected_control_fixture(
        tmp_path / "controls", tmp_path / "problems"
    )
    runs = tmp_path / "runs"
    collision = runs / "autolab_hp_4h_test__autolab_adaptive_compression"
    collision.mkdir(parents=True)
    sentinel = collision / "keep.txt"
    sentinel.write_text("owned by user", encoding="utf-8")

    with pytest.raises(RuntimeError, match="refus|exist|overwrite"):
        prepare_runs(
            controls,
            runs,
            "autolab_hp_4h_test",
            problems_root=tmp_path / "problems",
        )

    assert sentinel.read_text() == "owned by user"
    assert len(list(runs.iterdir())) == 1


@pytest.mark.parametrize(
    "damage",
    [
        "v0",
        "checker",
        "seed",
        "bootstrap_event",
        "frozen",
        "proof",
        "context_path",
        "context_content",
        "context_partial_normalized",
        "oversized_text",
    ],
)
def test_audit_runs_fails_closed_on_drift_or_private_context_leak(tmp_path, damage):
    controls = _corrected_control_fixture(
        tmp_path / "controls", tmp_path / "problems"
    )
    contexts = tmp_path / "private-contexts"
    contexts.mkdir()
    for tasks in AUTOLAB_GROUPS.values():
        for task in tasks:
            (contexts / f"autolab_{task}.md").write_text(
                _private_context_text(task), encoding="utf-8"
            )
    runs = tmp_path / "runs"
    batch = "autolab_hp_4h_test"
    prepare_runs(controls, runs, batch, problems_root=tmp_path / "problems")
    run = runs / f"{batch}__autolab_adaptive_compression"

    if damage == "v0":
        (run / "supervisor/verifier_versions/v0.py").write_text("changed")
    elif damage == "checker":
        (run / "checker/tests/test.sh").write_text("changed")
    elif damage == "seed":
        (run / "bootstrap_ws/seed_solution.json").write_text("{}")
    elif damage == "bootstrap_event":
        with (run / "events.jsonl").open("a") as stream:
            stream.write('{"kind":"bootstrap_start"}\n')
    elif damage == "frozen":
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["freeze_verifier"] = True
        (run / "manifest.json").write_text(json.dumps(manifest))
    elif damage == "proof":
        (run / "bootstrap_ws/reframe_policy.json").write_text(
            json.dumps({"admits_proof": True})
        )
    elif damage == "context_path":
        (run / "leak.txt").write_text(str(contexts.resolve()))
    elif damage == "context_content":
        (run / "leak.txt").write_bytes(
            (contexts / "autolab_adaptive_compression.md").read_bytes()
        )
    elif damage == "context_partial_normalized":
        private = (contexts / "autolab_adaptive_compression.md").read_text()
        leaked_section = private[35:390].upper().replace(" ", " \n!! ")
        (run / "leak.txt").write_text(leaked_section, encoding="utf-8")
    else:
        (run / "oversized.txt").write_text("x" * 4_100_000, encoding="utf-8")

    with pytest.raises(RuntimeError, match="audit|mismatch|drift|bootstrap|frozen|proof|private|context"):
        audit_runs(
            controls,
            runs,
            batch,
            problems_root=tmp_path / "problems",
            context_dir=contexts,
        )


@pytest.mark.parametrize("damage", ["manifest_v0_sha", "tracked_test"])
def test_prepare_runs_rejects_a_forged_corrected_control_source(tmp_path, damage):
    controls = _corrected_control_fixture(
        tmp_path / "controls", tmp_path / "problems"
    )
    task = "adaptive_compression"
    source = controls / (
        "autolab_shippedv0_control17_4h_v2_20260825"
        f"__autolab_{task}"
    )
    if damage == "manifest_v0_sha":
        manifest_path = source / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["initial_verifier_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        (tmp_path / "problems" / f"autolab_{task}" / "tests/test.sh").write_text(
            "#!/bin/sh\n# forged tracked test\n", encoding="utf-8"
        )

    with pytest.raises(ValueError, match="hash|tracked|test|V0|source"):
        prepare_runs(
            controls,
            tmp_path / "runs",
            "autolab_hp_4h_test",
            problems_root=tmp_path / "problems",
        )


@pytest.mark.parametrize(
    "contamination",
    [
        "candidate",
        "eval_query",
        "human",
        "solver_ws",
        "cost",
        "v1",
        "manifest_best",
        "manifest_final",
        "result_json",
        "human_guidance",
        "outcome_json",
    ],
)
def test_audit_runs_rejects_prelaunch_runtime_contamination(
    tmp_path, contamination
):
    controls = _corrected_control_fixture(
        tmp_path / "controls", tmp_path / "problems"
    )
    runs = tmp_path / "runs"
    batch = "autolab_hp_4h_test"
    prepare_runs(controls, runs, batch, problems_root=tmp_path / "problems")
    run = runs / f"{batch}__autolab_adaptive_compression"
    if contamination == "candidate":
        path = run / "solver/candidates/cand_00000.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    elif contamination == "eval_query":
        (run / "eval/queries.jsonl").write_text("{}\n", encoding="utf-8")
    elif contamination == "human":
        (run / "human").mkdir()
    elif contamination == "solver_ws":
        (run / "solver_ws").mkdir()
    elif contamination == "cost":
        (run / "cost.jsonl").write_text("{}\n", encoding="utf-8")
    elif contamination == "v1":
        (run / "supervisor/verifier_versions/v1.py").write_text(
            "# hardened", encoding="utf-8"
        )
    elif contamination == "result_json":
        (run / "result.json").write_text('{"score": 1}', encoding="utf-8")
    elif contamination == "human_guidance":
        (run / "human_guidance.md").write_text("guidance", encoding="utf-8")
    elif contamination == "outcome_json":
        (run / "outcome.json").write_text('{"decision": "guide"}', encoding="utf-8")
    else:
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if contamination == "manifest_best":
            manifest["best_solution"] = {"source": "old result"}
        else:
            manifest["final_verifier_version"] = 0
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="prelaunch|contamin|candidate|runtime|final|best"):
        audit_runs(
            controls,
            runs,
            batch,
            problems_root=tmp_path / "problems",
        )


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_prepare_runs_and_audit_runs_cli_commands(tmp_path, capsys):
    from coscientist.experiments.autolab_human_proxy import main

    controls = _corrected_control_fixture(
        tmp_path / "controls", tmp_path / "problems"
    )
    contexts = tmp_path / "contexts"
    contexts.mkdir()
    for tasks in AUTOLAB_GROUPS.values():
        for task in tasks:
            (contexts / f"autolab_{task}.md").write_text(
                _private_context_text(task), encoding="utf-8"
            )
    runs = tmp_path / "runs"
    common = [
        "--control-runs-root", str(controls),
        "--runs-root", str(runs),
        "--batch-name", "autolab_hp_4h_test",
        "--problems-root", str(tmp_path / "problems"),
    ]

    assert main(["prepare-runs", *common]) == 0
    assert json.loads(capsys.readouterr().out)["task_count"] == 17
    assert main(["audit-runs", *common, "--context-dir", str(contexts)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"


def test_privacy_audit_allows_only_a_context_span_already_bound_in_control(tmp_path):
    controls = _corrected_control_fixture(
        tmp_path / "controls", tmp_path / "problems"
    )
    task = "adaptive_compression"
    source = controls / (
        "autolab_shippedv0_control17_4h_v2_20260825"
        f"__autolab_{task}"
    )
    public_checker_text = (
        "This public shipped checker validates canonical payload framing and the "
        "documented correctness oracle before measuring performance on deterministic "
        "inputs. It rejects malformed submissions, missing output fields, invalid "
        "numeric values, and candidates that fail the task's published semantics. "
        "This paragraph deliberately exceeds the privacy detector span threshold.\n"
    )
    source_test = source / "checker/tests/test.sh"
    source_test.write_text(public_checker_text, encoding="utf-8")
    tracked_test = tmp_path / "problems" / f"autolab_{task}" / "tests/test.sh"
    tracked_test.write_text(public_checker_text, encoding="utf-8")
    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_manifest["original_test_sha256"] = _sha256(source_test)
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    runs = tmp_path / "runs"
    batch = "autolab_hp_4h_test"
    prepare_runs(controls, runs, batch, problems_root=tmp_path / "problems")
    contexts = tmp_path / "contexts"
    contexts.mkdir()
    for candidate_task in (
        task for tasks in AUTOLAB_GROUPS.values() for task in tasks
    ):
        text = _private_context_text(candidate_task)
        if candidate_task == task:
            text += public_checker_text
        (contexts / f"autolab_{candidate_task}.md").write_text(
            text, encoding="utf-8"
        )

    assert audit_runs(
        controls,
        runs,
        batch,
        problems_root=tmp_path / "problems",
        context_dir=contexts,
    )["status"] == "pass"

    leaked = runs / f"{batch}__autolab_{task}" / "leak.txt"
    leaked.write_text(public_checker_text, encoding="utf-8")
    with pytest.raises(RuntimeError, match="private context"):
        audit_runs(
            controls,
            runs,
            batch,
            problems_root=tmp_path / "problems",
            context_dir=contexts,
        )


def test_prepare_runs_single_publish_failure_leaves_no_visible_root_and_can_retry(
    tmp_path, monkeypatch
):
    from coscientist.experiments import autolab_human_proxy as experiment

    controls = _corrected_control_fixture(
        tmp_path / "controls", tmp_path / "problems"
    )
    runs = tmp_path / "runs"
    batch = "autolab_hp_4h_test"
    original_publish = experiment._rename_noreplace

    def fail_publish(source, destination):
        raise OSError("synthetic publish failure")

    monkeypatch.setattr(experiment, "_rename_noreplace", fail_publish)
    with pytest.raises(OSError, match="synthetic publish"):
        prepare_runs(
            controls, runs, batch, problems_root=tmp_path / "problems"
        )

    assert not runs.exists()
    assert not list(tmp_path.glob(f".{batch}.runs.tmp.*"))

    monkeypatch.setattr(experiment, "_rename_noreplace", original_publish)
    assert prepare_runs(
        controls, runs, batch, problems_root=tmp_path / "problems"
    )["task_count"] == 17


def test_prepare_runs_concurrent_atomic_publish_has_exactly_one_winner(
    tmp_path, monkeypatch
):
    from coscientist.experiments import autolab_human_proxy as experiment

    controls = _corrected_control_fixture(
        tmp_path / "controls", tmp_path / "problems"
    )
    runs = tmp_path / "runs"
    batch = "autolab_hp_4h_test"
    publish_barrier = threading.Barrier(2)
    original_publish = experiment._rename_noreplace

    def rendezvous_publish(source, destination):
        publish_barrier.wait(timeout=10)
        return original_publish(source, destination)

    monkeypatch.setattr(experiment, "_rename_noreplace", rendezvous_publish)
    outcomes = []
    outcome_lock = threading.Lock()

    def prepare_one():
        try:
            outcome = prepare_runs(
                controls, runs, batch, problems_root=tmp_path / "problems"
            )["status"]
        except Exception as exc:  # pragma: no cover - asserted below
            outcome = exc
        with outcome_lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=prepare_one) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert outcomes.count("pass") == 1
    failures = [outcome for outcome in outcomes if outcome != "pass"]
    assert len(failures) == 1 and isinstance(failures[0], FileExistsError)
    assert len(list(runs.glob(f"{batch}__autolab_*"))) == 17
    assert not list(tmp_path.glob(f".{batch}.runs.tmp.*"))


def test_atomic_publish_never_overwrites_target_created_during_prepare(
    tmp_path, monkeypatch
):
    from coscientist.experiments import autolab_human_proxy as experiment

    controls = _corrected_control_fixture(
        tmp_path / "controls", tmp_path / "problems"
    )
    runs = tmp_path / "runs"
    batch = "autolab_hp_4h_test"
    original_publish = experiment._rename_noreplace
    sentinel = runs / "owned.txt"

    def inject_competing_target(source, destination):
        runs.mkdir()
        sentinel.write_text("competitor", encoding="utf-8")
        return original_publish(source, destination)

    monkeypatch.setattr(experiment, "_rename_noreplace", inject_competing_target)
    with pytest.raises(FileExistsError):
        prepare_runs(controls, runs, batch, problems_root=tmp_path / "problems")

    assert sentinel.read_text() == "competitor"
    assert list(runs.iterdir()) == [sentinel]
    assert not list(tmp_path.glob(f".{batch}.runs.tmp.*"))


@pytest.mark.parametrize("payload", ["raw_private", "unknown_binary"])
def test_privacy_audit_fails_closed_on_untrusted_non_utf8_files(tmp_path, payload):
    controls = _corrected_control_fixture(
        tmp_path / "controls", tmp_path / "problems"
    )
    runs = tmp_path / "runs"
    batch = "autolab_hp_4h_test"
    prepare_runs(controls, runs, batch, problems_root=tmp_path / "problems")
    contexts = tmp_path / "contexts"
    contexts.mkdir()
    for task in (task for tasks in AUTOLAB_GROUPS.values() for task in tasks):
        (contexts / f"autolab_{task}.md").write_text(
            _private_context_text(task), encoding="utf-8"
        )
    run = runs / f"{batch}__autolab_adaptive_compression"
    if payload == "raw_private":
        raw = (contexts / "autolab_adaptive_compression.md").read_bytes()
        (run / "leak.bin").write_bytes(b"\xff" + raw[25:380])
    else:
        (run / "unknown.bin").write_bytes(b"\xff\xfe\x00untrusted")

    with pytest.raises(RuntimeError, match="private|binary|decode|prelaunch"):
        audit_runs(
            controls,
            runs,
            batch,
            problems_root=tmp_path / "problems",
            context_dir=contexts,
        )


def test_privacy_audit_allows_byte_bound_control_binary(tmp_path):
    controls = _corrected_control_fixture(
        tmp_path / "controls", tmp_path / "problems"
    )
    task = "adaptive_compression"
    source = controls / (
        "autolab_shippedv0_control17_4h_v2_20260825"
        f"__autolab_{task}"
    )
    (source / "checker/public_fixture.bin").write_bytes(b"\xff\xfe\x00public")
    runs = tmp_path / "runs"
    batch = "autolab_hp_4h_test"
    prepare_runs(controls, runs, batch, problems_root=tmp_path / "problems")
    contexts = tmp_path / "contexts"
    contexts.mkdir()
    for candidate_task in (
        task for tasks in AUTOLAB_GROUPS.values() for task in tasks
    ):
        (contexts / f"autolab_{candidate_task}.md").write_text(
            _private_context_text(candidate_task), encoding="utf-8"
        )

    assert audit_runs(
        controls,
        runs,
        batch,
        problems_root=tmp_path / "problems",
        context_dir=contexts,
    )["status"] == "pass"


@pytest.mark.parametrize(
    "attack", ["bootstrap_parent", "checker_parent", "manifest", "tracked_tests_parent"]
)
def test_prepare_runs_rejects_symlinks_in_every_trusted_source_path(tmp_path, attack):
    controls = _corrected_control_fixture(
        tmp_path / "controls", tmp_path / "problems"
    )
    task = "adaptive_compression"
    source = controls / (
        "autolab_shippedv0_control17_4h_v2_20260825"
        f"__autolab_{task}"
    )
    if attack == "bootstrap_parent":
        target = source / "bootstrap-real"
        (source / "bootstrap_ws").rename(target)
        (source / "bootstrap_ws").symlink_to(target, target_is_directory=True)
    elif attack == "checker_parent":
        target = source / "checker-real"
        (source / "checker").rename(target)
        (source / "checker").symlink_to(target, target_is_directory=True)
    elif attack == "manifest":
        target = source / "manifest-real.json"
        (source / "manifest.json").rename(target)
        (source / "manifest.json").symlink_to(target)
    else:
        tests = tmp_path / "problems" / f"autolab_{task}" / "tests"
        target = tests.with_name("tests-real")
        tests.rename(target)
        tests.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|unsafe"):
        prepare_runs(
            controls,
            tmp_path / "runs",
            "autolab_hp_4h_test",
            problems_root=tmp_path / "problems",
        )


@pytest.mark.parametrize("oversize", ["bootstrap_file", "checker_total", "context"])
def test_preseed_and_context_reads_enforce_size_limits(
    tmp_path, monkeypatch, oversize
):
    from coscientist.experiments import autolab_human_proxy as experiment

    controls = _corrected_control_fixture(
        tmp_path / "controls", tmp_path / "problems"
    )
    task = "adaptive_compression"
    source = controls / (
        "autolab_shippedv0_control17_4h_v2_20260825"
        f"__autolab_{task}"
    )
    if oversize == "bootstrap_file":
        monkeypatch.setattr(experiment, "MAX_PRESEED_FILE_BYTES", 32)
        (source / "bootstrap_ws/seed_solution.json").write_text(
            json.dumps({"source": "x" * 100}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="large|size|limit"):
            prepare_runs(
                controls,
                tmp_path / "runs",
                "autolab_hp_4h_test",
                problems_root=tmp_path / "problems",
            )
        return
    if oversize == "checker_total":
        monkeypatch.setattr(experiment, "MAX_CHECKER_TOTAL_BYTES", 64)
        with pytest.raises(ValueError, match="checker.*large|total|limit"):
            prepare_runs(
                controls,
                tmp_path / "runs",
                "autolab_hp_4h_test",
                problems_root=tmp_path / "problems",
            )
        return

    runs = tmp_path / "runs"
    batch = "autolab_hp_4h_test"
    prepare_runs(controls, runs, batch, problems_root=tmp_path / "problems")
    contexts = tmp_path / "contexts"
    contexts.mkdir()
    for candidate_task in (
        task for tasks in AUTOLAB_GROUPS.values() for task in tasks
    ):
        text = _private_context_text(candidate_task)
        if candidate_task == task:
            text = "x" * 200_001
        (contexts / f"autolab_{candidate_task}.md").write_text(
            text, encoding="utf-8"
        )
    with pytest.raises((ValueError, RuntimeError), match="200000|large|size|limit"):
        audit_runs(
            controls,
            runs,
            batch,
            problems_root=tmp_path / "problems",
            context_dir=contexts,
        )


def test_evaluate_results_dry_run_validates_exact_contract_without_final_best(
    tmp_path,
):
    runs, accepted, trusted, batch = _evaluation_fixture(tmp_path, completed=False)
    docker = _FakeDocker()
    output = tmp_path / "dry-run-report.json"

    report = evaluate_results(
        batch_name=batch,
        runs_root=runs,
        accepted_root=accepted,
        trusted_runs_root=trusted,
        workers=4,
        repeat=2,
        output=output,
        audit_seed="shipped-v0-control17-final-replay-audit-only-20260825-v1",
        dry_run=True,
        run_command=docker,
    )

    assert report["status"] == "dry_run_pass"
    assert report["completeness"] == {
        "expected_tasks": 17,
        "contract_validated_tasks": 17,
        "replayed_tasks": 0,
        "failed_tasks": 0,
        "complete": False,
    }
    assert len(report["tasks"]) == 17
    assert len(docker.commands) == 17
    assert all(command[:3] == ["docker", "image", "inspect"] for command in docker.commands)
    old = next(row for row in report["tasks"] if row["task"] == "adaptive_compression")
    new = next(row for row in report["tasks"] if row["task"] == "aes128_ctr")
    assert old["checker"]["source_batch"] == "autolab_all_4h"
    assert new["checker"]["source_batch"] == "autolab_proofgate_4h"
    assert old["payload"] == {"mapping": "identity", "status": "not_available"}
    assert old["metric"]["direction"] == "higher_is_better"
    assert "bits per byte" in old["metric"]["native_meaning"]
    assert old["docker"]["network"] == "none"
    assert old["docker"]["image_used"].startswith("sha256:")
    assert json.loads(output.read_text()) == report


def test_evaluate_results_scores_identity_payload_repeats_and_keeps_file_seed_secret(
    tmp_path,
):
    runs, accepted, trusted, batch = _evaluation_fixture(tmp_path, completed=True)
    hidden_seed = "do-not-disclose-this-live-secret"
    seed_file = tmp_path / "hidden-seed.txt"
    seed_file.write_text(hidden_seed + "\n", encoding="utf-8")
    docker = _FakeDocker(seed_forbidden=hidden_seed)
    output = tmp_path / "final-report.json"

    report = evaluate_results(
        batch_name=batch,
        runs_root=runs,
        accepted_root=accepted,
        trusted_runs_root=trusted,
        workers=3,
        repeat=2,
        output=output,
        hidden_seed_file=seed_file,
        run_command=docker,
    )

    assert report["status"] == "pass"
    assert report["completeness"] == {
        "expected_tasks": 17,
        "contract_validated_tasks": 17,
        "replayed_tasks": 17,
        "failed_tasks": 0,
        "complete": True,
    }
    _assert_secret_absent(report, hidden_seed)
    assert report["hidden_seed"] == {
        "mode": "private_file",
        "sha256": hashlib.sha256(hidden_seed.encode()).hexdigest(),
        "persist_across_repeats": True,
        "disclosed": False,
    }
    for row in report["tasks"]:
        expected_payload = {
            "source": f"final payload for {row['task']}",
            "task_marker": row["task"],
        }
        assert row["status"] == "replayed"
        assert row["payload"]["mapping"] == "identity"
        assert row["payload"]["sha256"] == hashlib.sha256(
            json.dumps(
                expected_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        assert row["raw_values"] == [float(len(row["task"]))] * 2
        assert row["feasible_values"] == [True, True]
        assert len(row["repeats"]) == 2
        assert row["accepted_evaluator"]["verifier_sha256"] == _sha256(
            accepted / row["group"] / row["task"] / "final_verifier.py"
        )
        assert row["accepted_evaluator"]["ctx_template_sha256"] == _sha256(
            accepted / row["group"] / row["task"] / "final_ctx.template.json"
        )
        assert row["docker"]["image_reference"] == f"autolab_{row['task']}:latest"
        assert row["docker"]["image_used"].startswith("sha256:")
    run_commands = [command for command in docker.commands if command[:2] == ["docker", "run"]]
    assert len(run_commands) == 34


@pytest.mark.parametrize(
    "damage,match",
    [
        ("missing_task", "exact|17|missing"),
        ("extra_task", "exact|unexpected|ambiguous"),
        ("wrong_status", "complete|done|status"),
        ("missing_best", "best_solution|payload"),
        ("accepted_drift", "hash"),
        ("trusted_resource_drift", "internet|resource"),
    ],
)
def test_evaluate_results_rejects_incomplete_ambiguous_or_polluted_inputs(
    tmp_path, damage, match
):
    runs, accepted, trusted, batch = _evaluation_fixture(tmp_path, completed=True)
    batch_path = runs / batch / "batch.json"
    batch_doc = json.loads(batch_path.read_text())
    task = "adaptive_compression"
    target = runs / f"{batch}__autolab_{task}"
    if damage == "missing_task":
        batch_doc["runs"].pop()
        batch_path.write_text(json.dumps(batch_doc), encoding="utf-8")
    elif damage == "extra_task":
        extra = runs / f"{batch}__autolab_not_a_task"
        extra.mkdir()
        (extra / "manifest.json").write_text("{}", encoding="utf-8")
    elif damage == "wrong_status":
        batch_doc["runs"][0]["status"] = "running"
        batch_path.write_text(json.dumps(batch_doc), encoding="utf-8")
    elif damage == "missing_best":
        manifest = json.loads((target / "manifest.json").read_text())
        del manifest["best_solution"]
        (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    elif damage == "accepted_drift":
        (accepted / "old10" / task / "final_verifier.py").write_text(
            "# tampered after acceptance\n", encoding="utf-8"
        )
    else:
        manifest_path = trusted / f"autolab_all_4h__autolab_{task}" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["resource_spec"]["verifier"]["allow_internet"] = True
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises((ValueError, RuntimeError), match=match):
        evaluate_results(
            batch_name=batch,
            runs_root=runs,
            accepted_root=accepted,
            trusted_runs_root=trusted,
            workers=2,
            repeat=2,
            output=tmp_path / "report.json",
            audit_seed="shipped-v0-control17-final-replay-audit-only-20260825-v1",
            run_command=_FakeDocker(),
        )


def test_evaluate_results_cli_requires_one_explicit_seed_mode_and_redacts_file_seed(
    tmp_path, monkeypatch, capsys
):
    runs, accepted, trusted, batch = _evaluation_fixture(tmp_path, completed=False)
    secret = "cli-private-seed-that-must-not-be-printed"
    seed_file = tmp_path / "seed"
    seed_file.write_text(secret, encoding="utf-8")
    output = tmp_path / "report.json"
    from coscientist.experiments import autolab_human_proxy as experiment

    docker = _FakeDocker(seed_forbidden=secret)
    monkeypatch.setattr(experiment.subprocess, "run", docker)
    rc = main(
        [
            "evaluate-results",
            "--batch-name",
            batch,
            "--runs-root",
            str(runs),
            "--accepted-root",
            str(accepted),
            "--trusted-runs-root",
            str(trusted),
            "--workers",
            "2",
            "--repeat",
            "2",
            "--output",
            str(output),
            "--hidden-seed-file",
            str(seed_file),
            "--dry-run",
        ]
    )

    assert rc == 0
    assert secret not in capsys.readouterr().out
    _assert_secret_absent(json.loads(output.read_text()), secret)


def test_evaluate_results_rejects_a_verifier_output_that_echoes_private_seed(tmp_path):
    runs, accepted, trusted, batch = _evaluation_fixture(tmp_path, completed=True)
    secret = "private-seed-must-not-cross-output-boundary"
    seed_file = tmp_path / "seed"
    seed_file.write_text(secret, encoding="utf-8")
    output = tmp_path / "report.json"

    with pytest.raises(RuntimeError, match="failed"):
        evaluate_results(
            batch_name=batch,
            runs_root=runs,
            accepted_root=accepted,
            trusted_runs_root=trusted,
            workers=2,
            repeat=1,
            output=output,
            hidden_seed_file=seed_file,
            run_command=_FakeDocker(leak_seed_in_result=True),
        )

    _assert_secret_absent(json.loads(output.read_text()), secret)


def test_evaluate_results_manifest_hash_binds_the_same_snapshot_as_payload(tmp_path):
    runs, accepted, trusted, batch = _evaluation_fixture(tmp_path, completed=False)
    task = "adaptive_compression"
    manifest_path = runs / f"{batch}__autolab_{task}" / "manifest.json"
    initial_raw = manifest_path.read_bytes()
    docker = _FakeDocker()
    mutated = False

    def mutate_after_contract_read(argv, **kwargs):
        nonlocal mutated
        if not mutated:
            value = json.loads(manifest_path.read_text())
            value["final_verifier_version"] = 99
            manifest_path.write_text(json.dumps(value), encoding="utf-8")
            mutated = True
        return docker(argv, **kwargs)

    report = evaluate_results(
        batch_name=batch,
        runs_root=runs,
        accepted_root=accepted,
        trusted_runs_root=trusted,
        workers=1,
        repeat=1,
        output=tmp_path / "report.json",
        audit_seed="shipped-v0-control17-final-replay-audit-only-20260825-v1",
        dry_run=True,
        run_command=mutate_after_contract_read,
    )

    row = next(item for item in report["tasks"] if item["task"] == task)
    assert row["run"]["final_verifier_version"] == 3
    assert row["run"]["manifest_sha256"] == hashlib.sha256(initial_raw).hexdigest()


def _live_privacy_fixture(tmp_path: Path) -> tuple[Path, Path, str, dict[str, str]]:
    runs = tmp_path / "runs"
    contexts = tmp_path / "contexts"
    batch = "autolab_hp_live_privacy"
    runs.mkdir()
    contexts.mkdir()
    rows = []
    context_texts = {}
    for tasks in AUTOLAB_GROUPS.values():
        for task in tasks:
            private = (
                f"Private accepted evaluator context for {task}. "
                + (f"hidden-{task}-criterion " * 20)
            )
            context_texts[task] = private
            context_path = contexts / f"autolab_{task}.md"
            context_path.write_text(private, encoding="utf-8")
            run_id = f"{batch}__autolab_{task}"
            run = runs / run_id
            session = run / "human" / "session_001"
            session.mkdir(parents=True)
            (session / "session.json").write_text(
                json.dumps({"state": "closed"}), encoding="utf-8"
            )
            (session / "transcript.jsonl").write_text(
                json.dumps({"role": "human", "text": "Use an independent edge case."})
                + "\n",
                encoding="utf-8",
            )
            (run / "human_guidance.md").write_text(
                "Try a boundary case without revealing evaluator details.\n",
                encoding="utf-8",
            )
            rows.append(
                {
                    "run_id": run_id,
                    "input_dir": f"/problems/autolab_{task}",
                    "human_proxy_context": str(context_path.resolve()),
                    "human_proxy_context_sha256": hashlib.sha256(
                        private.encode("utf-8")
                    ).hexdigest(),
                }
            )
    batch_dir = runs / batch
    batch_dir.mkdir()
    (batch_dir / "batch.json").write_text(
        json.dumps(
            {
                "batch": batch,
                "human_proxy_context_dir": str(contexts.resolve()),
                "runs": rows,
            }
        ),
        encoding="utf-8",
    )
    return runs, contexts, batch, context_texts


def test_audit_live_privacy_validates_closed_sessions_without_scanning_batch_paths(
    tmp_path,
):
    runs, contexts, batch, _private = _live_privacy_fixture(tmp_path)

    summary = audit_live_privacy(
        runs_root=runs,
        batch_name=batch,
        context_dir=contexts,
        require_closed_tasks=10,
    )

    assert summary["status"] == "pass"
    assert summary["task_count"] == 17
    assert summary["closed_session_tasks"] == 17
    assert summary["scanned_files"] == 51


def test_audit_live_privacy_rejects_encoded_private_context_in_transcript(tmp_path):
    runs, contexts, batch, private = _live_privacy_fixture(tmp_path)
    task = "adaptive_compression"
    transcript = (
        runs
        / f"{batch}__autolab_{task}"
        / "human"
        / "session_001"
        / "transcript.jsonl"
    )
    encoded = base64.b64encode(private[task].encode("utf-8")).decode("ascii")
    transcript.write_text(
        json.dumps({"role": "human", "text": encoded}) + "\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="private context content leaked"):
        audit_live_privacy(
            runs_root=runs,
            batch_name=batch,
            context_dir=contexts,
            require_closed_tasks=1,
        )
