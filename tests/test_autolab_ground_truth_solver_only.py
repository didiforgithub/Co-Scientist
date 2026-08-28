from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import traceback
from datetime import datetime as real_datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from coscientist.experiments.autolab_human_proxy import (
    ACCEPTED_ARTIFACTS,
    AUTOLAB_GROUPS,
    FROZEN_BRIEF_TAIL,
)
from coscientist.experiments.autolab_ground_truth_solver_only import (
    ARM,
    audit_invariance,
    audit_privacy,
    audit_runs,
    audit_seeds,
    audit_solver_isolation_contract,
    main,
    prepare_runs,
    prepare_seeds,
    replay_results,
)
from coscientist.demo.evaluator import RunResult


BATCH = "autolab_ground_truth_test"
CONTROL_PREFIX = "autolab_shippedv0_control17_4h_v2_20260825"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image_inspect(argv, **_kwargs):
    assert argv[:3] == ["docker", "image", "inspect"]
    task = argv[3].removeprefix("autolab_").removesuffix(":latest")
    digest = hashlib.sha256(task.encode()).hexdigest()
    return SimpleNamespace(
        returncode=0,
        stdout=json.dumps(
            {
                "Id": f"sha256:{digest}",
                "RepoDigests": [f"autolab_{task}@sha256:{digest}"],
                "Created": "2026-08-25T00:00:00Z",
            }
        ),
        stderr="",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    accepted = tmp_path / "accepted"
    controls = tmp_path / "controls"
    trusted = tmp_path / "trusted"
    problems = tmp_path / "problems"
    accepted.mkdir()
    controls.mkdir()
    trusted.mkdir()
    problems.mkdir()
    records = []
    for group, tasks in AUTOLAB_GROUPS.items():
        trusted_batch = "autolab_all_4h" if group == "old10" else "autolab_proofgate_4h"
        for task in tasks:
            package = accepted / group / task
            package.mkdir(parents=True)
            verifier = (
                f"# accepted final evaluator: {task}\n"
                "def verify(payload, ctx):\n"
                "    return {'feasible': True, 'raw': 1.0, 'artifacts': {}}\n"
            )
            template = {
                "checker_dir": "REQUIRED_CHECKER_DIR",
                "hidden_seed": "REQUIRED_PRIVATE",
            }
            if group == "old10":
                template.update(
                    {
                        "deployment_requirements": {"network": "none"},
                        "validation_mode": False,
                    }
                )
            (package / "final_verifier.py").write_text(verifier, encoding="utf-8")
            (package / "final_ctx.template.json").write_text(
                json.dumps(template), encoding="utf-8"
            )
            (package / "authored_design.md").write_text("accepted\n", encoding="utf-8")
            (package / "issues.md").write_text("none\n", encoding="utf-8")
            records.append(
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
                    "final_verifier_sha256": _sha(package / "final_verifier.py"),
                    "final_ctx_template_sha256": _sha(package / "final_ctx.template.json"),
                    "required_artifacts": [
                        f"{group}/{task}/{name}" for name in ACCEPTED_ARTIFACTS
                    ],
                }
            )

            problem = problems / f"autolab_{task}"
            problem.mkdir()
            resource = (
                "[solver]\nimage = \"python:3.11-slim\"\ncpus = 1\n"
                "memory_mb = 2048\nallow_internet = false\n\n"
                "[verifier]\n"
                f"image = \"autolab_{task}:latest\"\n"
                "cpus = 2\nmemory_mb = 4096\ntimeout_sec = 37\n"
                "allow_internet = false\n"
            )
            (problem / "resource.toml").write_text(resource, encoding="utf-8")

            control = controls / f"{CONTROL_PREFIX}__autolab_{task}"
            bws = control / "bootstrap_ws"
            vdir = control / "supervisor" / "verifier_versions"
            checker = control / "checker"
            bws.mkdir(parents=True)
            vdir.mkdir(parents=True)
            (checker / "tests").mkdir(parents=True)
            shipped_test = f"#!/bin/sh\n# original shipped test for {task}\n"
            (checker / "tests/test.sh").write_text(shipped_test, encoding="utf-8")
            (checker / "instruction.md").write_text(f"task={task}\n", encoding="utf-8")
            (problem / "tests").mkdir()
            (problem / "tests/test.sh").write_text(shipped_test, encoding="utf-8")
            shipped_v0 = f"# shipped V0 adapter for {task}\n"
            (bws / "verifier.py").write_text(shipped_v0, encoding="utf-8")
            (vdir / "v0.py").write_text(shipped_v0, encoding="utf-8")
            (bws / "seed_solution.json").write_text(
                json.dumps({"source": f"original-seed-{task}"}), encoding="utf-8"
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
                f"# AutoLab task: {task}\n\nOptimize the supplied implementation.\n\n"
                f"{FROZEN_BRIEF_TAIL}",
                encoding="utf-8",
            )
            (control / "manifest.json").write_text(
                json.dumps(
                    {
                        "initial_verifier_origin": "autolab_shipped_tests/test.sh",
                        "initial_verifier_sha256": _sha(vdir / "v0.py"),
                        "original_test_sha256": _sha(checker / "tests/test.sh"),
                    }
                ),
                encoding="utf-8",
            )

            trusted_run = trusted / f"{trusted_batch}__autolab_{task}"
            checker = trusted_run / "checker"
            checker.mkdir(parents=True)
            (checker / "test.sh").write_text(
                f"#!/bin/sh\n# trusted {group} checker for {task}\n", encoding="utf-8"
            )
            (checker / "test.sh").chmod(0o755)
            (checker / "oracle.dat").write_text(f"oracle-{task}\n", encoding="utf-8")
            (trusted_run / "manifest.json").write_text(
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
    (accepted / "acceptance_report.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "failures": [],
                "aggregate": {"accepted_tasks": 17, "expected_tasks": 17},
                "results": {
                    "exact_task_set": {
                        group: list(tasks) for group, tasks in AUTOLAB_GROUPS.items()
                    }
                },
                "tasks": records,
            }
        ),
        encoding="utf-8",
    )
    return accepted, controls, trusted, problems


def _prepared(tmp_path: Path):
    accepted, controls, trusted, problems = _fixture(tmp_path)
    seeds = tmp_path / "private" / "dev_seeds"
    runs = tmp_path / "runs"
    prepare_seeds(seeds)
    prepare_runs(
        controls,
        accepted,
        trusted,
        seeds,
        runs,
        BATCH,
        problems_root=problems,
        run_command=_image_inspect,
    )
    return accepted, controls, trusted, problems, seeds, runs


def _publish_live_batch(
    runs: Path,
    *,
    statuses: dict[str, str] | None = None,
    complete: bool = False,
    query_tasks: int = 0,
) -> Path:
    tasks = [task for group in AUTOLAB_GROUPS.values() for task in group]
    statuses = statuses or {}
    rows = []
    contracts = {}
    for index, task in enumerate(tasks):
        run_id = f"{BATCH}__autolab_{task}"
        run = runs / run_id
        row = {
            "run_id": run_id,
            "input_dir": str(Path("/problems") / f"autolab_{task}"),
            "status": statuses.get(task, "done" if complete else "running"),
            "deadline_epoch": 1.0,
            "freeze_verifier": True,
            "human_proxy_context": None,
            "human_proxy_context_sha256": None,
            "human_proxy_context_dir": None,
            "human_agent_timeout_s": None,
            "human_agent_model": None,
            "human_agent_reasoning_effort": None,
        }
        rows.append(row)
        contracts[run_id] = {
            "human_proxy_context": None,
            "human_proxy_context_sha256": None,
            "freeze_verifier": True,
        }
        if complete:
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update(
                {
                    "best_solution": {"source": f"final-{task}"},
                    "best_score": float(index),
                    "final_verifier_version": 0,
                    "verifier_hardenings": 0,
                }
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with (run / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "t": 2.0,
                            "kind": "run_stop",
                            "final_verifier_version": 0,
                            "hardenings": 0,
                        }
                    )
                    + "\n"
                )
        if index < query_tasks:
            queries = run / "eval/queries.jsonl"
            queries.parent.mkdir(parents=True, exist_ok=True)
            queries.write_text(
                json.dumps(
                    {
                        "query_id": f"q-{task}",
                        "verifier_version": 0,
                        "returned_feasible": True,
                        "returned_score": 1.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
    batch_dir = runs / BATCH
    batch_dir.mkdir()
    batch_path = batch_dir / "batch.json"
    batch_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "batch": BATCH,
                "freeze_verifier": True,
                "human_proxy_context_dir": None,
                "human_agent": {
                    "timeout_s": None,
                    "model": None,
                    "reasoning_effort": None,
                },
                "runs": rows,
                "run_contracts": contracts,
            }
        ),
        encoding="utf-8",
    )
    return batch_path


def test_exact_task_set_and_arm_are_locked():
    assert ARM == "autolab_ground_truth_solver_only"
    assert tuple(AUTOLAB_GROUPS) == ("old10", "new7")
    assert [len(AUTOLAB_GROUPS[group]) for group in AUTOLAB_GROUPS] == [10, 7]
    assert len({task for tasks in AUTOLAB_GROUPS.values() for task in tasks}) == 17


def test_prepare_seeds_is_private_hash_only_and_transactional(tmp_path, capsys):
    destination = tmp_path / "private" / "dev_seeds"
    summary = prepare_seeds(destination)
    output = capsys.readouterr().out

    assert summary == {
        "arm": ARM,
        "status": "pass",
        "task_count": 17,
        "seed_manifest_sha256": summary["seed_manifest_sha256"],
    }
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    manifest = json.loads((destination / "manifest.json").read_text())
    assert set(manifest["tasks"]) == {
        task for tasks in AUTOLAB_GROUPS.values() for task in tasks
    }
    values = []
    for task, metadata in manifest["tasks"].items():
        seed_file = destination / f"autolab_{task}.seed"
        value = seed_file.read_text().strip()
        values.append(value)
        assert stat.S_IMODE(seed_file.stat().st_mode) == 0o600
        assert metadata == {"sha256": hashlib.sha256(value.encode()).hexdigest()}
        assert value not in output
        assert value not in json.dumps(manifest)
    assert len(set(values)) == 17
    assert audit_seeds(destination)["status"] == "pass"

    with pytest.raises(RuntimeError, match="exist|overwrite|refus"):
        prepare_seeds(destination)


def test_prepare_seeds_removes_staging_if_audit_fails(tmp_path, monkeypatch):
    from coscientist.experiments import autolab_ground_truth_solver_only as experiment

    destination = tmp_path / "dev_seeds"

    def fail(_path):
        raise RuntimeError("synthetic audit failure")

    monkeypatch.setattr(experiment, "audit_seeds", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        prepare_seeds(destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".dev_seeds.seeds.tmp.*"))


@pytest.mark.parametrize(
    "damage", ["permission", "value", "manifest", "extra_plaintext", "symlink"]
)
def test_audit_seeds_fails_closed(tmp_path, damage):
    seeds = tmp_path / "dev_seeds"
    prepare_seeds(seeds)
    target = seeds / "autolab_adaptive_compression.seed"
    if damage == "permission":
        target.chmod(0o644)
    elif damage == "value":
        target.write_text("changed", encoding="utf-8")
    elif damage == "manifest":
        manifest = json.loads((seeds / "manifest.json").read_text())
        manifest["tasks"].pop("adaptive_compression")
        (seeds / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    elif damage == "extra_plaintext":
        manifest = json.loads((seeds / "manifest.json").read_text())
        manifest["debug_seed"] = target.read_text().strip()
        (seeds / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (seeds / "manifest.json").chmod(0o600)
    else:
        target.unlink()
        target.symlink_to(seeds / "autolab_fft_rust.seed")
    with pytest.raises((RuntimeError, ValueError), match="seed|permission|task|symlink|hash"):
        audit_seeds(seeds)


def test_prepare_runs_binds_accepted_v0_trusted_checker_seed_and_resources(tmp_path):
    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)

    for group, tasks in AUTOLAB_GROUPS.items():
        trusted_batch = "autolab_all_4h" if group == "old10" else "autolab_proofgate_4h"
        for task in tasks:
            run = runs / f"{BATCH}__autolab_{task}"
            package = accepted / group / task
            trusted_run = trusted / f"{trusted_batch}__autolab_{task}"
            control = controls / f"{CONTROL_PREFIX}__autolab_{task}"
            assert (run / "bootstrap_ws/verifier.py").read_bytes() == (
                package / "final_verifier.py"
            ).read_bytes()
            assert (run / "supervisor/verifier_versions/v0.py").read_bytes() == (
                package / "final_verifier.py"
            ).read_bytes()
            assert (run / "bootstrap_ws/seed_solution.json").read_bytes() == (
                control / "bootstrap_ws/seed_solution.json"
            ).read_bytes()
            assert {
                p.relative_to(run / "checker").as_posix(): p.read_bytes()
                for p in (run / "checker").rglob("*") if p.is_file()
            } == {
                p.relative_to(trusted_run / "checker").as_posix(): p.read_bytes()
                for p in (trusted_run / "checker").rglob("*") if p.is_file()
            }
            assert stat.S_IMODE((run / "checker/test.sh").stat().st_mode) == 0o755
            ctx_path = run / "bootstrap_ws/ctx.json"
            ctx = json.loads(ctx_path.read_text())
            secret = (seeds / f"autolab_{task}.seed").read_text().strip()
            expected_ctx = json.loads(
                (package / "final_ctx.template.json").read_text()
            )
            expected_ctx.update(
                {
                    "task": task,
                    "checker_dir": str((run / "checker").resolve()),
                    "hidden_seed": secret,
                    "validation_mode": False,
                }
            )
            assert ctx == expected_ctx
            assert stat.S_IMODE(ctx_path.stat().st_mode) == 0o600
            manifest = json.loads((run / "manifest.json").read_text())
            assert manifest["arm"] == ARM
            assert manifest["freeze_verifier"] is True
            assert manifest["initial_feedback_level"] == "feasible_score"
            assert manifest["solver"] == {
                "model": "gpt-5.6-sol",
                "reasoning_effort": "high",
                "topology": "weak",
            }
            assert manifest["budget_s"] == 14_400.0
            assert manifest["human_proxy_enabled"] is False
            assert manifest["human_session_enabled"] is False
            assert manifest["development_seed_sha256"] == hashlib.sha256(
                secret.encode()
            ).hexdigest()
            assert secret not in json.dumps(manifest)
            assert manifest["accepted_final_verifier_sha256"] == _sha(
                package / "final_verifier.py"
            )
            assert manifest["accepted_artifact_sha256"] == {
                name: _sha(package / name) for name in ACCEPTED_ARTIFACTS
            }
            assert manifest["trusted_checker_run"] == trusted_run.name
            assert manifest["resource_contract"]["verifier"]["allow_internet"] is False
            assert manifest["resource_contract"]["immutable_image"]["id"].startswith(
                "sha256:"
            )
            assert manifest["pinned_verifier_image_id"] == (
                manifest["resource_contract"]["immutable_image"]["id"]
            )
            assert len(manifest["resource_contract_sha256"]) == 64
            brief = (run / "bootstrap_ws/SOLVER_BRIEF.md").read_text()
            assert "feasible" in brief and "score" in brief
            assert "fixed" in brief
            assert "scrubbed at the delivery boundary" in " ".join(brief.split())
            assert "Human Proxy" not in brief and "hardening is enabled" not in brief
            assert "original shipped `tests/test.sh`" not in brief
            versions = [json.loads(line) for line in (
                run / "supervisor/versions.jsonl"
            ).read_text().splitlines()]
            assert len(versions) == 1
            assert versions[0]["version"] == 0
            assert versions[0]["origin"] == "accepted_final_evaluator"

    assert audit_runs(
        controls,
        accepted,
        trusted,
        seeds,
        runs,
        BATCH,
        problems_root=problems,
        run_command=_image_inspect,
    )["task_count"] == 17
    assert audit_privacy(runs, BATCH, seeds)["scanned_tasks"] == 17


def test_prepare_runs_is_whole_root_transactional(tmp_path, monkeypatch):
    accepted, controls, trusted, problems = _fixture(tmp_path)
    seeds = tmp_path / "dev_seeds"
    prepare_seeds(seeds)
    runs = tmp_path / "runs"
    from coscientist.experiments import autolab_ground_truth_solver_only as experiment

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic run audit failure")

    monkeypatch.setattr(experiment, "audit_runs", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        prepare_runs(
            controls, accepted, trusted, seeds, runs, BATCH,
            problems_root=problems, run_command=_image_inspect,
        )
    assert not runs.exists()
    assert not list(tmp_path.glob(f".{BATCH}.runs.tmp.*"))


def test_prepare_runs_rejects_missing_required_accepted_artifact(tmp_path):
    accepted, controls, trusted, problems = _fixture(tmp_path)
    (accepted / "old10/adaptive_compression/issues.md").unlink()
    seeds = tmp_path / "dev_seeds"
    prepare_seeds(seeds)
    with pytest.raises(ValueError, match="accepted|required|missing|artifact"):
        prepare_runs(
            controls, accepted, trusted, seeds, tmp_path / "runs", BATCH,
            problems_root=problems, run_command=_image_inspect,
        )


@pytest.mark.parametrize("damage", ["origin", "v0_copy", "v0_hash", "tracked_test"])
def test_prepare_runs_rejects_forged_corrected_control_provenance(tmp_path, damage):
    accepted, controls, trusted, problems = _fixture(tmp_path)
    task = "adaptive_compression"
    control = controls / f"{CONTROL_PREFIX}__autolab_{task}"
    if damage == "origin":
        manifest = json.loads((control / "manifest.json").read_text())
        manifest["initial_verifier_origin"] = "forged"
        (control / "manifest.json").write_text(json.dumps(manifest))
    elif damage == "v0_copy":
        (control / "bootstrap_ws/verifier.py").write_text("# forged\n")
    elif damage == "v0_hash":
        manifest = json.loads((control / "manifest.json").read_text())
        manifest["initial_verifier_sha256"] = "0" * 64
        (control / "manifest.json").write_text(json.dumps(manifest))
    else:
        (problems / f"autolab_{task}/tests/test.sh").write_text("#!/bin/sh\n# forged\n")
    seeds = tmp_path / "dev_seeds"
    prepare_seeds(seeds)

    with pytest.raises(ValueError, match="Control|control|origin|V0|hash|tracked|test"):
        prepare_runs(
            controls, accepted, trusted, seeds, tmp_path / "runs", BATCH,
            problems_root=problems, run_command=_image_inspect,
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("task", "wrong_task", "task mismatch"),
        ("validation_mode", True, "validation mode"),
    ],
)
def test_prepare_runs_rejects_invalid_optional_template_fields(
    tmp_path, field, value, error
):
    accepted, controls, trusted, problems = _fixture(tmp_path)
    task = "aes128_ctr"
    template_path = accepted / "new7" / task / "final_ctx.template.json"
    template = json.loads(template_path.read_text())
    template[field] = value
    template_path.write_text(json.dumps(template), encoding="utf-8")
    report_path = accepted / "acceptance_report.json"
    report = json.loads(report_path.read_text())
    record = next(item for item in report["tasks"] if item["task"] == task)
    record["final_ctx_template_sha256"] = _sha(template_path)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    seeds = tmp_path / "dev_seeds"
    prepare_seeds(seeds)

    with pytest.raises(ValueError, match=error):
        prepare_runs(
            controls, accepted, trusted, seeds, tmp_path / "runs", BATCH,
            problems_root=problems, run_command=_image_inspect,
        )


@pytest.mark.parametrize(
    "damage",
    [
        "accepted_v0", "checker", "seed_solution", "ctx_seed", "ctx_permission",
        "resource", "v1", "human", "runtime", "manifest", "symlink",
        "review", "probe", "harden_ws", "event_meta", "versions_meta",
    ],
)
def test_audit_runs_fails_closed_on_static_drift(tmp_path, damage):
    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)
    run = runs / f"{BATCH}__autolab_adaptive_compression"
    if damage == "accepted_v0":
        (run / "supervisor/verifier_versions/v0.py").write_text("changed")
    elif damage == "checker":
        (run / "checker/test.sh").write_text("changed")
    elif damage == "seed_solution":
        (run / "bootstrap_ws/seed_solution.json").write_text("{}")
    elif damage == "ctx_seed":
        ctx = json.loads((run / "bootstrap_ws/ctx.json").read_text())
        ctx["hidden_seed"] = "wrong"
        (run / "bootstrap_ws/ctx.json").write_text(json.dumps(ctx))
        (run / "bootstrap_ws/ctx.json").chmod(0o600)
    elif damage == "ctx_permission":
        (run / "bootstrap_ws/ctx.json").chmod(0o644)
    elif damage == "resource":
        (problems / "autolab_adaptive_compression/resource.toml").write_text(
            "[verifier]\nimage='wrong:latest'\ncpus=2\nmemory_mb=4096\n"
            "timeout_sec=37\nallow_internet=false\n"
        )
    elif damage == "v1":
        (run / "supervisor/verifier_versions/v1.py").write_text("# drift")
    elif damage == "human":
        (run / "human").mkdir()
    elif damage == "runtime":
        (run / "solver_ws").mkdir()
    elif damage == "review":
        (run / "supervisor/reviews.jsonl").write_text("{}\n")
    elif damage == "probe":
        (run / "supervisor/probes").mkdir()
        (run / "supervisor/probes/probe_000.json").write_text("{}")
    elif damage == "harden_ws":
        (run / "harden_ws").mkdir()
        (run / "harden_ws/verifier.py").write_text("# runtime hardening\n")
    elif damage == "event_meta":
        event = json.loads((run / "events.jsonl").read_text())
        event["unexpected"] = "drift"
        (run / "events.jsonl").write_text(json.dumps(event) + "\n")
    elif damage == "versions_meta":
        version = json.loads((run / "supervisor/versions.jsonl").read_text())
        version["has_feedback"] = True
        (run / "supervisor/versions.jsonl").write_text(json.dumps(version) + "\n")
    elif damage == "manifest":
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["freeze_verifier"] = False
        (run / "manifest.json").write_text(json.dumps(manifest))
    else:
        target = run / "checker/oracle.dat"
        target.unlink()
        target.symlink_to(trusted / "autolab_all_4h__autolab_adaptive_compression/checker/oracle.dat")

    with pytest.raises((RuntimeError, ValueError), match="audit|mismatch|drift|contamination|permission|resource|symlink|runtime|Human|verifier|seed|manifest"):
        audit_runs(
            controls, accepted, trusted, seeds, runs, BATCH,
            problems_root=problems, run_command=_image_inspect,
        )


@pytest.mark.parametrize("encoding", ["raw", "hex", "base64", "urlsafe", "path"])
def test_privacy_scanner_rejects_seed_and_private_path_leaks(tmp_path, encoding):
    _accepted, _controls, _trusted, _problems, seeds, runs = _prepared(tmp_path)
    run = runs / f"{BATCH}__autolab_adaptive_compression"
    secret = (seeds / "autolab_adaptive_compression.seed").read_text().strip()
    raw = secret.encode()
    leak = {
        "raw": secret,
        "hex": raw.hex(),
        "base64": base64.b64encode(raw).decode(),
        "urlsafe": base64.urlsafe_b64encode(raw).decode().rstrip("="),
        "path": str(seeds.resolve()),
    }[encoding]
    destination = run / "solver/trajectory.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"stdout": leak}) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="private|seed|leak"):
        audit_privacy(runs, BATCH, seeds)


def test_privacy_scanner_includes_live_launcher_batch_manifest(tmp_path):
    _accepted, _controls, _trusted, _problems, seeds, runs = _prepared(tmp_path)
    secret = (seeds / "autolab_adaptive_compression.seed").read_text().strip()
    launcher_batch = runs / BATCH
    launcher_batch.mkdir()
    (launcher_batch / "batch.json").write_text(
        json.dumps({"exceptional_detail": base64.b64encode(secret.encode()).decode()}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="private|seed|leak"):
        audit_privacy(runs, BATCH, seeds)


def test_solver_workspace_symlinks_are_scanned_without_following_and_allow_replay(
    tmp_path,
):
    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)

    fft_ws = runs / f"{BATCH}__autolab_fft_rust/solver_ws"
    fft_ws.mkdir()
    (fft_ws / "app_link").symlink_to("/app")

    router_ws = runs / f"{BATCH}__autolab_agent_tool_routing/solver_ws"
    router_ws.mkdir()
    (router_ws / "candidate_top5.py").write_text("# candidate\n", encoding="utf-8")
    (router_ws / "retriever.py").symlink_to("candidate_top5.py")

    kv_ws = runs / f"{BATCH}__autolab_concurrent_kv_wal/solver_ws/diag.test"
    kv_ws.mkdir(parents=True)
    (kv_ws / "go.mod").symlink_to("/app/go.mod")
    (kv_ws / "index.go").symlink_to("/work/tablefirst_index.go")

    privacy = audit_privacy(runs, BATCH, seeds)
    assert privacy["status"] == "pass"
    invariance = audit_invariance(
        controls,
        accepted,
        trusted,
        seeds,
        runs,
        BATCH,
        problems_root=problems,
        run_command=_image_inspect,
    )
    assert invariance["status"] == "pass"
    replay = replay_results(
        control_runs_root=controls,
        accepted_root=accepted,
        trusted_runs_root=trusted,
        seed_dir=seeds,
        runs_root=runs,
        batch_name=BATCH,
        problems_root=problems,
        output=tmp_path / "symlink-replay-dry-run.json",
        dry_run=True,
        run_command=_image_inspect,
    )
    assert replay["status"] == "dry_run_pass"


def test_solver_workspace_symlink_target_is_privacy_scanned(tmp_path):
    _accepted, _controls, _trusted, _problems, seeds, runs = _prepared(tmp_path)
    secret = (seeds / "autolab_fft_rust.seed").read_text().strip()
    solver_ws = runs / f"{BATCH}__autolab_fft_rust/solver_ws"
    solver_ws.mkdir()
    (solver_ws / "leaking_link").symlink_to(f"/work/{secret}")

    with pytest.raises(RuntimeError, match="private|seed|leak"):
        audit_privacy(runs, BATCH, seeds)


def test_contract_symlink_remains_fail_closed(tmp_path):
    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)
    run = runs / f"{BATCH}__autolab_fft_rust"
    verifier = run / "bootstrap_ws/verifier.py"
    verifier.unlink()
    verifier.symlink_to(accepted / "old10/fft_rust/final_verifier.py")

    with pytest.raises((RuntimeError, ValueError), match="symlink|regular|unsafe"):
        audit_privacy(runs, BATCH, seeds)
    with pytest.raises((RuntimeError, ValueError), match="symlink|regular|unsafe"):
        audit_invariance(
            controls,
            accepted,
            trusted,
            seeds,
            runs,
            BATCH,
            problems_root=problems,
            run_command=_image_inspect,
        )


def test_solver_workspace_mountpoint_symlink_remains_fail_closed(tmp_path):
    _accepted, _controls, _trusted, _problems, seeds, runs = _prepared(tmp_path)
    run = runs / f"{BATCH}__autolab_fft_rust"
    external = tmp_path / "external-workspace"
    external.mkdir()
    (run / "solver_ws").symlink_to(external, target_is_directory=True)

    with pytest.raises((RuntimeError, ValueError), match="symlink|unsafe"):
        audit_privacy(runs, BATCH, seeds)


def test_solver_isolation_contract_and_refresh_exclude_private_state(tmp_path):
    summary = audit_solver_isolation_contract()
    assert summary["status"] == "pass"
    assert summary["solver_mount"] == "solver_ws_only"
    assert summary["refresh_copies_ctx"] is False


def test_solver_container_constructed_mounts_allow_only_solver_workspace(
    tmp_path, monkeypatch
):
    from coscientist.coevo import container as container_module
    from coscientist.coevo.container import DockerContainer, GatewayConfig

    solver_ws = tmp_path / "run/solver_ws"
    solver_ws.mkdir(parents=True)
    codex = tmp_path / "codex"
    codex.write_text("binary")
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    captured = {}

    monkeypatch.setattr(
        DockerContainer, "_prepare_codex_home", lambda _self: codex_home
    )

    def fake_run(argv, **_kwargs):
        captured["argv"] = list(argv)
        return SimpleNamespace(returncode=0, stdout="cid\n", stderr="")

    monkeypatch.setattr(container_module.subprocess, "run", fake_run)
    DockerContainer(
        workdir=solver_ws,
        gateway=GatewayConfig(codex_home=codex_home),
        agent_elf=codex,
    ).start()

    argv = captured["argv"]
    mounts = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "-v"]
    targets = {mount.split(":", 1)[1].removesuffix(":ro") for mount in mounts}
    assert targets == {"/work", "/usr/local/bin/codex", "/codexhome"}
    assert mounts[0].startswith(str(solver_ws.resolve()) + ":/work")
    assert not any(
        forbidden in mount
        for mount in mounts
        for forbidden in ("bootstrap_ws", "/checker", "dev_seeds")
    )


def test_cli_prepare_seeds_and_audit_seeds(tmp_path, capsys):
    seeds = tmp_path / "seeds"
    assert main(["prepare-seeds", "--seed-dir", str(seeds)]) == 0
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["task_count"] == 17
    assert main(["audit-seeds", "--seed-dir", str(seeds)]) == 0
    audited = json.loads(capsys.readouterr().out)
    assert audited["status"] == "pass"


def test_smoke_runs_uses_formal_pinned_backend_for_all_17_and_publishes_hash_only_report(
    tmp_path, monkeypatch
):
    from coscientist.demo.evaluator import Evaluator
    from coscientist.experiments import autolab_ground_truth_solver_only as experiment

    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)
    report = tmp_path / "reports" / "smoke.json"
    formal_before = {
        path.relative_to(runs).as_posix(): path.read_bytes()
        for path in runs.rglob("*")
        if path.is_file()
    }
    seen = {}

    def fake_run(self, payload, ctx, source=None, **_kwargs):
        task = ctx["task"]
        manifest = json.loads(
            (runs / f"{BATCH}__autolab_{task}" / "manifest.json").read_text()
        )
        verifier = manifest["resource_contract"]["verifier"]
        assert source is not None
        assert source.startswith(self.current.source)
        assert "malformed accepted evaluator result" in source
        assert self.current.version == 0
        assert self.current.origin == "accepted_final_evaluator"
        assert self.exec_backend == {
            "image": manifest["pinned_verifier_image_id"],
            "gpus": None,
            "cpus": verifier["cpus"],
            "memory_mb": verifier["memory_mb"],
            "allow_internet": False,
            "timeout_s": verifier["timeout_sec"],
        }
        assert ctx["checker_dir"] == str(
            (runs / f"{BATCH}__autolab_{task}" / "checker").resolve()
        )
        assert payload == {"source": f"original-seed-{task}"}
        seen[task] = ctx["hidden_seed"]
        return RunResult(
            feasible=task != "adaptive_compression",
            raw=0.0 if task == "adaptive_compression" else 1.25,
            artifacts={"must_not_persist": ctx["hidden_seed"]},
        )

    monkeypatch.setattr(Evaluator, "run", fake_run)
    summary = experiment.smoke_runs(
        controls,
        accepted,
        trusted,
        seeds,
        runs,
        BATCH,
        report,
        problems_root=problems,
        run_command=_image_inspect,
    )

    assert summary == {
        "arm": ARM,
        "batch_name": BATCH,
        "status": "pass",
        "task_count": 17,
        "report_sha256": _sha(report),
    }
    assert set(seen) == {
        task for tasks in AUTOLAB_GROUPS.values() for task in tasks
    }
    formal_after = {
        path.relative_to(runs).as_posix(): path.read_bytes()
        for path in runs.rglob("*")
        if path.is_file()
    }
    assert formal_after == formal_before
    payload = json.loads(report.read_text())
    assert payload["status"] == "pass"
    assert payload["task_count"] == 17
    assert len(payload["tasks"]) == 17
    assert next(row for row in payload["tasks"] if row["task"] == "adaptive_compression")[
        "feasible"
    ] is False
    for row in payload["tasks"]:
        assert set(row) == {
            "task",
            "manifest_sha256",
            "accepted_final_verifier_sha256",
            "seed_solution_sha256",
            "development_seed_sha256",
            "checker_inventory_sha256",
            "resource_contract_sha256",
            "formal_state_sha256",
            "feasible",
            "raw",
        }
    encoded = report.read_text()
    assert "artifacts" not in encoded and "checker_dir" not in encoded
    assert all(secret not in encoded for secret in seen.values())


@pytest.mark.parametrize(
    "result_source",
    [
        "{'feasible': True, 'artifacts': {}}",
        "{'feasible': True, 'raw': 1.0}",
        "{'feasible': 'yes', 'raw': 1.0, 'artifacts': {}}",
        "{'feasible': True, 'raw': float('nan'), 'artifacts': {}}",
    ],
)
def test_strict_smoke_source_rejects_malformed_real_evaluator_results(result_source):
    from coscientist.demo.evaluator import Evaluator, VerifierVersion
    from coscientist.experiments import autolab_ground_truth_solver_only as experiment

    accepted_source = (
        "def verify(payload, ctx):\n"
        f"    return {result_source}\n"
    )
    evaluator = Evaluator()
    evaluator.versions.append(
        VerifierVersion(0, accepted_source, "accepted_final_evaluator")
    )
    result = evaluator.run({}, {}, source=experiment._strict_smoke_source(accepted_source))
    assert result.error is not None
    assert "malformed accepted evaluator result" in result.error


@pytest.mark.parametrize("failure", ["crash", "timeout", "malformed", "non_finite"])
def test_smoke_runs_fails_closed_without_report_or_formal_mutation(
    tmp_path, monkeypatch, failure
):
    from coscientist.demo.evaluator import Evaluator
    from coscientist.experiments import autolab_ground_truth_solver_only as experiment

    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)
    report = tmp_path / "reports" / "smoke.json"
    formal_before = _sha(
        runs / f"{BATCH}__autolab_adaptive_compression" / "events.jsonl"
    )

    def fake_run(_self, _payload, ctx, **_kwargs):
        if ctx["task"] != "adaptive_compression":
            return RunResult(True, 1.0, {})
        if failure == "crash":
            return RunResult(False, 0.0, {}, error="verifier crashed")
        if failure == "timeout":
            return RunResult(False, 0.0, {}, error="verifier timeout")
        if failure == "malformed":
            return SimpleNamespace(feasible="yes", raw=1.0, artifacts={}, error=None)
        return RunResult(True, float("nan"), {})

    monkeypatch.setattr(Evaluator, "run", fake_run)
    with pytest.raises(RuntimeError, match="smoke|verifier|malformed|finite|timeout|crash"):
        experiment.smoke_runs(
            controls,
            accepted,
            trusted,
            seeds,
            runs,
            BATCH,
            report,
            problems_root=problems,
            run_command=_image_inspect,
        )
    assert not report.exists()
    assert formal_before == _sha(
        runs / f"{BATCH}__autolab_adaptive_compression" / "events.jsonl"
    )


def test_smoke_runs_does_not_echo_private_evaluator_error(tmp_path, monkeypatch):
    from coscientist.demo.evaluator import Evaluator
    from coscientist.experiments import autolab_ground_truth_solver_only as experiment

    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)
    secret = (seeds / "autolab_adaptive_compression.seed").read_text().strip()

    def fake_run(_self, _payload, ctx, **_kwargs):
        if ctx["task"] == "adaptive_compression":
            return RunResult(False, 0.0, {}, error=f"crashed with {secret}")
        return RunResult(True, 1.0, {})

    monkeypatch.setattr(Evaluator, "run", fake_run)
    with pytest.raises(RuntimeError) as captured:
        experiment.smoke_runs(
            controls,
            accepted,
            trusted,
            seeds,
            runs,
            BATCH,
            tmp_path / "smoke.json",
            problems_root=problems,
            run_command=_image_inspect,
        )
    assert secret not in str(captured.value)


def test_smoke_runs_drops_private_exception_chain_from_full_traceback(
    tmp_path, monkeypatch
):
    from coscientist.demo.evaluator import Evaluator
    from coscientist.experiments import autolab_ground_truth_solver_only as experiment

    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)
    secret = (seeds / "autolab_adaptive_compression.seed").read_text().strip()

    def fake_run(_self, _payload, ctx, **_kwargs):
        if ctx["task"] == "adaptive_compression":
            raise ValueError(f"private evaluator crash: {secret}")
        return RunResult(True, 1.0, {})

    monkeypatch.setattr(Evaluator, "run", fake_run)
    with pytest.raises(RuntimeError) as captured:
        experiment.smoke_runs(
            controls,
            accepted,
            trusted,
            seeds,
            runs,
            BATCH,
            tmp_path / "smoke.json",
            problems_root=problems,
            run_command=_image_inspect,
        )
    formatted = "".join(
        traceback.format_exception(captured.type, captured.value, captured.tb)
    )
    assert secret not in formatted
    assert captured.value.__cause__ is None


def test_smoke_runs_rejects_report_inside_runs_or_existing_report(tmp_path, monkeypatch):
    from coscientist.demo.evaluator import Evaluator
    from coscientist.experiments import autolab_ground_truth_solver_only as experiment

    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)
    monkeypatch.setattr(
        Evaluator, "run", lambda *_args, **_kwargs: RunResult(True, 1.0, {})
    )
    common = (controls, accepted, trusted, seeds, runs, BATCH)
    with pytest.raises((RuntimeError, ValueError), match="report|runs root|outside"):
        experiment.smoke_runs(
            *common,
            runs / "smoke.json",
            problems_root=problems,
            run_command=_image_inspect,
        )
    report = tmp_path / "smoke.json"
    report.write_text("keep", encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError), match="report|exist|overwrite|refus"):
        experiment.smoke_runs(
            *common,
            report,
            problems_root=problems,
            run_command=_image_inspect,
        )
    assert report.read_text() == "keep"


def test_cli_smoke_runs_dispatches_full_contract(tmp_path, monkeypatch, capsys):
    from coscientist.experiments import autolab_ground_truth_solver_only as experiment

    captured = {}

    def fake_smoke(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"status": "pass", "task_count": 17}

    monkeypatch.setattr(experiment, "smoke_runs", fake_smoke, raising=False)
    values = {
        "control-runs-root": tmp_path / "controls",
        "accepted-root": tmp_path / "accepted",
        "trusted-runs-root": tmp_path / "trusted",
        "seed-dir": tmp_path / "seeds",
        "runs-root": tmp_path / "runs",
        "batch-name": BATCH,
        "problems-root": tmp_path / "problems",
        "report": tmp_path / "smoke.json",
    }
    argv = ["smoke-runs"]
    for key, value in values.items():
        argv.extend((f"--{key}", str(value)))
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "pass", "task_count": 17}
    assert captured["args"][-1] == Path(values["report"])
    assert captured["kwargs"] == {"problems_root": Path(values["problems-root"])}


def test_invariance_prelaunch_accepts_exact_accepted_v0_without_batch(tmp_path):
    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)

    summary = audit_invariance(
        controls,
        accepted,
        trusted,
        seeds,
        runs,
        BATCH,
        problems_root=problems,
        live=False,
        run_command=_image_inspect,
    )

    assert summary == {
        "arm": ARM,
        "batch_name": BATCH,
        "status": "pass",
        "mode": "prelaunch",
        "task_count": 17,
        "query_tasks": 0,
        "query_count": 0,
        "required_query_tasks": 0,
    }


def test_invariance_live_requires_schema3_frozen_batch_and_query_floor(tmp_path):
    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)
    _publish_live_batch(runs, query_tasks=5)

    summary = audit_invariance(
        controls,
        accepted,
        trusted,
        seeds,
        runs,
        BATCH,
        problems_root=problems,
        live=True,
        require_query_tasks=5,
        run_command=_image_inspect,
    )
    assert summary["mode"] == "live"
    assert summary["query_tasks"] == 5
    assert summary["query_count"] == 5

    with pytest.raises(RuntimeError, match="quer|Query|required"):
        audit_invariance(
            controls,
            accepted,
            trusted,
            seeds,
            runs,
            BATCH,
            problems_root=problems,
            live=True,
            require_query_tasks=6,
            run_command=_image_inspect,
        )


def test_live_jsonl_tolerates_one_incomplete_trailing_record_but_not_complete_malformed(
    tmp_path,
):
    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)
    _publish_live_batch(runs, query_tasks=1)
    queries = runs / f"{BATCH}__autolab_adaptive_compression/eval/queries.jsonl"
    with queries.open("ab") as stream:
        stream.write(b'{"verifier_version":')

    summary = audit_invariance(
        controls,
        accepted,
        trusted,
        seeds,
        runs,
        BATCH,
        problems_root=problems,
        live=True,
        require_query_tasks=1,
        run_command=_image_inspect,
    )
    assert summary["query_count"] == 1

    with queries.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="JSONL|eval queries"):
        audit_invariance(
            controls,
            accepted,
            trusted,
            seeds,
            runs,
            BATCH,
            problems_root=problems,
            live=True,
            run_command=_image_inspect,
        )

    queries.write_bytes(b'{"verifier_version": 0}\n42')
    with pytest.raises(ValueError, match="non-object|JSONL|eval queries"):
        audit_invariance(
            controls,
            accepted,
            trusted,
            seeds,
            runs,
            BATCH,
            problems_root=problems,
            live=True,
            run_command=_image_inspect,
        )


def test_jsonl_snapshot_retries_append_race_and_tolerates_disappearance(
    tmp_path, monkeypatch
):
    from coscientist.experiments import autolab_ground_truth_solver_only as experiment

    path = tmp_path / "queries.jsonl"
    path.write_text('{"verifier_version": 0}\n', encoding="utf-8")
    original = experiment._read_regular_snapshot_once
    calls = {"count": 0}

    def append_once(target, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            with Path(target).open("a", encoding="utf-8") as stream:
                stream.write('{"verifier_version": 0}\n')
            raise experiment._SnapshotChanged("append race")
        return original(target, **kwargs)

    monkeypatch.setattr(experiment, "_read_regular_snapshot_once", append_once)
    rows = experiment._read_json_lines(path, label="eval queries")
    assert len(rows) == 2
    assert calls["count"] == 2

    calls["count"] = 0
    path.write_text('{"verifier_version": 0}\n', encoding="utf-8")

    def disappear_once(target, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            Path(target).unlink()
            raise experiment._SnapshotChanged("disappear race")
        return original(target, **kwargs)

    monkeypatch.setattr(experiment, "_read_regular_snapshot_once", disappear_once)
    assert experiment._read_json_lines(path, label="eval queries") == []


@pytest.mark.parametrize(
    "damage",
    [
        "batch_schema",
        "batch_freeze",
        "row_freeze",
        "contract_freeze",
        "human_top",
        "human_row",
        "human_run_manifest",
        "run_freeze",
        "accepted_bootstrap",
        "accepted_version",
        "extra_version",
        "versions_metadata",
        "versions_false",
        "query_version",
        "query_false",
        "mode_switch",
        "human_event",
        "human_dir",
        "hardenings",
        "final_version",
        "counter_false",
    ],
)
def test_invariance_fails_closed_on_runtime_drift(tmp_path, damage):
    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)
    batch_path = _publish_live_batch(runs, query_tasks=1)
    task = "adaptive_compression"
    run = runs / f"{BATCH}__autolab_{task}"
    batch = json.loads(batch_path.read_text())
    if damage == "batch_schema":
        batch["schema_version"] = 2
    elif damage == "batch_freeze":
        batch["freeze_verifier"] = False
    elif damage == "row_freeze":
        batch["runs"][0]["freeze_verifier"] = False
    elif damage == "contract_freeze":
        batch["run_contracts"][batch["runs"][0]["run_id"]]["freeze_verifier"] = False
    elif damage == "human_top":
        batch["human_agent"]["model"] = "human-agent"
    elif damage == "human_row":
        batch["runs"][0]["human_proxy_context"] = "/private/context"
    elif damage == "human_run_manifest":
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["human_expert_id"] = "ou_forbidden"
        (run / "manifest.json").write_text(json.dumps(manifest))
    elif damage == "run_freeze":
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["freeze_verifier"] = False
        (run / "manifest.json").write_text(json.dumps(manifest))
    elif damage == "accepted_bootstrap":
        (run / "bootstrap_ws/verifier.py").write_text("# drift\n")
    elif damage == "accepted_version":
        (run / "supervisor/verifier_versions/v0.py").write_text("# drift\n")
    elif damage == "extra_version":
        (run / "supervisor/verifier_versions/v1.py").write_text("# evolved\n")
    elif damage == "versions_metadata":
        with (run / "supervisor/versions.jsonl").open("a") as stream:
            stream.write(json.dumps({"version": 1, "origin": "agent"}) + "\n")
    elif damage == "versions_false":
        version = json.loads((run / "supervisor/versions.jsonl").read_text())
        version["version"] = False
        (run / "supervisor/versions.jsonl").write_text(json.dumps(version) + "\n")
    elif damage in {"query_version", "query_false"}:
        (run / "eval/queries.jsonl").write_text(
            json.dumps(
                {"verifier_version": 1 if damage == "query_version" else False}
            )
            + "\n"
        )
    elif damage in {"mode_switch", "human_event"}:
        with (run / "events.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "kind": (
                            "mode_switch"
                            if damage == "mode_switch"
                            else "human_session_requested"
                        )
                    }
                )
                + "\n"
            )
    elif damage == "human_dir":
        (run / "human").mkdir()
    else:
        manifest = json.loads((run / "manifest.json").read_text())
        field = (
            "verifier_hardenings"
            if damage in {"hardenings", "counter_false"}
            else "final_verifier_version"
        )
        manifest[field] = False if damage == "counter_false" else 1
        (run / "manifest.json").write_text(json.dumps(manifest))
    if damage.startswith(("batch", "row", "contract", "human_top", "human_row")):
        batch_path.write_text(json.dumps(batch))

    with pytest.raises((RuntimeError, ValueError), match="batch|freeze|Human|human|accepted|version|query|mode|harden|invariant|drift"):
        audit_invariance(
            controls,
            accepted,
            trusted,
            seeds,
            runs,
            BATCH,
            problems_root=problems,
            live=True,
            run_command=_image_inspect,
        )


def test_replay_dry_run_needs_no_batch_or_candidates_and_never_reports_seeds(tmp_path):
    from coscientist.experiments.autolab_human_proxy import FINAL_REPLAY_AUDIT_SEED

    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)
    output = tmp_path / "replay-dry.json"

    report = replay_results(
        control_runs_root=controls,
        accepted_root=accepted,
        trusted_runs_root=trusted,
        seed_dir=seeds,
        runs_root=runs,
        batch_name=BATCH,
        problems_root=problems,
        output=output,
        dry_run=True,
        run_command=_image_inspect,
    )

    assert report["status"] == "dry_run_pass"
    assert report["execution"] == {
        "dry_run": True,
        "parallel_tasks": 4,
        "sequential_repeats_per_task": 2,
        "network": "none",
        "mounts": "read_only",
        "payload_mapping": "identity",
    }
    assert report["completeness"]["contract_validated_tasks"] == 17
    assert all(row["status"] == "contract_validated" for row in report["tasks"])
    raw = output.read_text()
    assert FINAL_REPLAY_AUDIT_SEED not in raw
    for task in [task for group in AUTOLAB_GROUPS.values() for task in group]:
        assert (seeds / f"autolab_{task}.seed").read_text().strip() not in raw


def test_replay_captures_started_timestamp_before_contract_work(tmp_path, monkeypatch):
    from coscientist.experiments import autolab_ground_truth_solver_only as experiment

    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)
    order = []
    original_audit = experiment.audit_invariance

    class RecordingDateTime:
        @classmethod
        def now(cls, timezone):
            order.append("timestamp")
            return real_datetime.now(timezone)

    def recording_audit(*args, **kwargs):
        order.append("audit")
        return original_audit(*args, **kwargs)

    monkeypatch.setattr(experiment, "datetime", RecordingDateTime)
    monkeypatch.setattr(experiment, "audit_invariance", recording_audit)
    replay_results(
        control_runs_root=controls,
        accepted_root=accepted,
        trusted_runs_root=trusted,
        seed_dir=seeds,
        runs_root=runs,
        batch_name=BATCH,
        problems_root=problems,
        output=tmp_path / "timestamp-dry-run.json",
        dry_run=True,
        run_command=_image_inspect,
    )

    assert order[0] == "timestamp"
    assert order.count("timestamp") == 2


def test_replay_requires_repeat_two_and_audit_seed_absent_from_search_files(tmp_path):
    from coscientist.experiments.autolab_human_proxy import FINAL_REPLAY_AUDIT_SEED

    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)
    common = dict(
        control_runs_root=controls,
        accepted_root=accepted,
        trusted_runs_root=trusted,
        seed_dir=seeds,
        runs_root=runs,
        batch_name=BATCH,
        problems_root=problems,
        output=tmp_path / "replay.json",
        dry_run=True,
        run_command=_image_inspect,
    )
    with pytest.raises(ValueError, match="repeat|2"):
        replay_results(**common, repeat=1)

    leak = runs / f"{BATCH}__autolab_adaptive_compression/solver/trajectory.jsonl"
    leak.parent.mkdir(parents=True)
    leak.write_text(FINAL_REPLAY_AUDIT_SEED, encoding="utf-8")
    with pytest.raises(RuntimeError, match="audit|seed|search"):
        replay_results(**common)
    assert not common["output"].exists()


def test_replay_dry_run_scans_large_binary_checker_without_utf8_or_4mb_limit(tmp_path):
    accepted, controls, trusted, problems = _fixture(tmp_path)
    large_checker = (
        trusted
        / "autolab_all_4h__autolab_adaptive_compression"
        / "checker"
        / "benchmark_input.raw.gz"
    )
    binary = bytes(range(256)) * 16_000
    assert len(binary) > 4_000_000
    large_checker.write_bytes(binary)
    seeds = tmp_path / "private/dev_seeds"
    runs = tmp_path / "runs"
    prepare_seeds(seeds)
    prepare_runs(
        controls,
        accepted,
        trusted,
        seeds,
        runs,
        BATCH,
        problems_root=problems,
        run_command=_image_inspect,
    )

    report = replay_results(
        control_runs_root=controls,
        accepted_root=accepted,
        trusted_runs_root=trusted,
        seed_dir=seeds,
        runs_root=runs,
        batch_name=BATCH,
        problems_root=problems,
        output=tmp_path / "large-binary-dry-run.json",
        dry_run=True,
        run_command=_image_inspect,
    )

    assert report["status"] == "dry_run_pass"


def test_replay_scans_search_artifact_larger_than_32mb_without_whole_file_limit(
    tmp_path,
):
    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)
    large = runs / f"{BATCH}__autolab_adaptive_compression/solver/large-output.bin"
    large.parent.mkdir(parents=True)
    block = bytes(range(256)) * 4096
    with large.open("wb") as stream:
        for _ in range(33):
            stream.write(block)
    assert large.stat().st_size > 32_000_000

    report = replay_results(
        control_runs_root=controls,
        accepted_root=accepted,
        trusted_runs_root=trusted,
        seed_dir=seeds,
        runs_root=runs,
        batch_name=BATCH,
        problems_root=problems,
        output=tmp_path / "large-search-artifact-dry-run.json",
        dry_run=True,
        run_command=_image_inspect,
    )
    assert report["status"] == "dry_run_pass"


def test_chunked_audit_seed_scan_detects_variant_across_chunk_boundary(tmp_path):
    from coscientist.experiments import autolab_ground_truth_solver_only as experiment
    from coscientist.experiments.autolab_human_proxy import FINAL_REPLAY_AUDIT_SEED

    _accepted, _controls, _trusted, _problems, _seeds, runs = _prepared(tmp_path)
    leak = runs / f"{BATCH}__autolab_adaptive_compression/solver/boundary.bin"
    leak.parent.mkdir(parents=True)
    prefix = b"x" * (1024 * 1024 - 7)
    leak.write_bytes(prefix + FINAL_REPLAY_AUDIT_SEED.encode("utf-8") + b"tail")

    with pytest.raises(RuntimeError, match="audit seed|search"):
        experiment._assert_audit_seed_absent(runs, BATCH)


def test_audit_seed_scan_checks_solver_workspace_symlink_text_without_following(
    tmp_path,
):
    from coscientist.experiments import autolab_ground_truth_solver_only as experiment
    from coscientist.experiments.autolab_human_proxy import FINAL_REPLAY_AUDIT_SEED

    _accepted, _controls, _trusted, _problems, _seeds, runs = _prepared(tmp_path)
    solver_ws = runs / f"{BATCH}__autolab_fft_rust/solver_ws"
    solver_ws.mkdir()
    (solver_ws / "audit-seed-link").symlink_to(
        f"/work/{FINAL_REPLAY_AUDIT_SEED}"
    )

    with pytest.raises(RuntimeError, match="audit seed|search"):
        experiment._assert_audit_seed_absent(runs, BATCH)


def test_replay_completed_done_and_budget_spent_runs_twice_on_immutable_backend(
    tmp_path,
):
    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)
    _publish_live_batch(
        runs,
        statuses={"adaptive_compression": "budget_spent"},
        complete=True,
        query_tasks=17,
    )
    budget_run = runs / f"{BATCH}__autolab_adaptive_compression"
    budget_events = [
        line
        for line in (budget_run / "events.jsonl").read_text().splitlines()
        if '"run_stop"' not in line
    ]
    (budget_run / "events.jsonl").write_text("\n".join(budget_events) + "\n")
    output = tmp_path / "replay.json"
    docker_runs = []

    def runner(argv, **kwargs):
        if argv[:3] == ["docker", "image", "inspect"]:
            return _image_inspect(argv, **kwargs)
        docker_runs.append(list(argv))
        assert argv[:5] == ["docker", "run", "--rm", "--network", "none"]
        for index, value in enumerate(argv):
            if value == "-v":
                assert argv[index + 1].endswith(":ro")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "result": {"feasible": True, "raw": 7.0, "artifacts": {}},
                    "duration_seconds": 0.1,
                }
            )
            + "\n",
            stderr="",
        )

    report = replay_results(
        control_runs_root=controls,
        accepted_root=accepted,
        trusted_runs_root=trusted,
        seed_dir=seeds,
        runs_root=runs,
        batch_name=BATCH,
        problems_root=problems,
        output=output,
        workers=3,
        run_command=runner,
    )

    assert report["status"] == "pass"
    assert report["completeness"]["replayed_tasks"] == 17
    assert len(docker_runs) == 34
    assert all(row["repeat_count"] == 2 for row in report["tasks"])
    assert all(row["run"]["freeze_verifier"] is True for row in report["tasks"])
    encoded = output.read_text()
    from coscientist.experiments.autolab_human_proxy import FINAL_REPLAY_AUDIT_SEED
    assert FINAL_REPLAY_AUDIT_SEED not in encoded


@pytest.mark.parametrize(
    ("status", "damage", "error"),
    [
        ("held_systematic", None, "status|complete"),
        ("failed", None, "status|complete"),
        ("done", "best", "best_solution|best solution"),
        ("done", "run_stop", "run_stop"),
        ("budget_spent", "future_deadline", "deadline"),
    ],
)
def test_replay_rejects_invalid_completion_contract(tmp_path, status, damage, error):
    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)
    _publish_live_batch(
        runs,
        statuses={"adaptive_compression": status},
        complete=True,
        query_tasks=17,
    )
    run = runs / f"{BATCH}__autolab_adaptive_compression"
    if damage == "best":
        manifest = json.loads((run / "manifest.json").read_text())
        manifest.pop("best_solution")
        (run / "manifest.json").write_text(json.dumps(manifest))
    elif damage == "run_stop":
        events = [
            line
            for line in (run / "events.jsonl").read_text().splitlines()
            if '"run_stop"' not in line
        ]
        (run / "events.jsonl").write_text("\n".join(events) + "\n")
    elif damage == "future_deadline":
        batch_path = runs / BATCH / "batch.json"
        batch = json.loads(batch_path.read_text())
        batch["runs"][0]["deadline_epoch"] = 10**20
        batch_path.write_text(json.dumps(batch))

    with pytest.raises((RuntimeError, ValueError), match=error):
        replay_results(
            control_runs_root=controls,
            accepted_root=accepted,
            trusted_runs_root=trusted,
            seed_dir=seeds,
            runs_root=runs,
            batch_name=BATCH,
            problems_root=problems,
            output=tmp_path / "replay.json",
            run_command=_image_inspect,
        )


@pytest.mark.parametrize(
    "deadline_epoch",
    [0.0, -1.0, float("nan"), float("inf"), 10**20],
    ids=["zero", "negative", "nan", "infinity", "future"],
)
def test_budget_spent_replay_rejects_nonpositive_nonfinite_or_future_deadline(
    tmp_path, deadline_epoch
):
    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)
    batch_path = _publish_live_batch(
        runs,
        statuses={"adaptive_compression": "budget_spent"},
        complete=True,
        query_tasks=17,
    )
    batch = json.loads(batch_path.read_text())
    batch["runs"][0]["deadline_epoch"] = deadline_epoch
    batch_path.write_text(json.dumps(batch))

    with pytest.raises(RuntimeError, match="deadline"):
        replay_results(
            control_runs_root=controls,
            accepted_root=accepted,
            trusted_runs_root=trusted,
            seed_dir=seeds,
            runs_root=runs,
            batch_name=BATCH,
            problems_root=problems,
            output=tmp_path / "invalid-deadline-replay.json",
            run_command=_image_inspect,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("verifier_hardenings", None),
        ("final_verifier_version", None),
        ("verifier_hardenings", False),
        ("final_verifier_version", False),
    ],
    ids=["missing-hardenings", "missing-final-version", "false-hardenings", "false-final-version"],
)
def test_actual_replay_requires_present_strict_integer_zero_final_counters(
    tmp_path, field, value
):
    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)
    _publish_live_batch(runs, complete=True, query_tasks=17)
    run = runs / f"{BATCH}__autolab_adaptive_compression"
    manifest = json.loads((run / "manifest.json").read_text())
    if value is None:
        manifest.pop(field)
    else:
        manifest[field] = value
    (run / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="harden|final verifier|version|counter"):
        replay_results(
            control_runs_root=controls,
            accepted_root=accepted,
            trusted_runs_root=trusted,
            seed_dir=seeds,
            runs_root=runs,
            batch_name=BATCH,
            problems_root=problems,
            output=tmp_path / "invalid-counters-replay.json",
            run_command=_image_inspect,
        )


def test_completed_replay_strictly_rejects_incomplete_query_tail(tmp_path):
    accepted, controls, trusted, problems, seeds, runs = _prepared(tmp_path)
    _publish_live_batch(runs, complete=True, query_tasks=17)
    queries = runs / f"{BATCH}__autolab_adaptive_compression/eval/queries.jsonl"
    with queries.open("ab") as stream:
        stream.write(b'{"verifier_version": 1')

    with pytest.raises(ValueError, match="JSONL|incomplete|truncated|eval queries"):
        replay_results(
            control_runs_root=controls,
            accepted_root=accepted,
            trusted_runs_root=trusted,
            seed_dir=seeds,
            runs_root=runs,
            batch_name=BATCH,
            problems_root=problems,
            output=tmp_path / "truncated-final-replay.json",
            run_command=_image_inspect,
        )


def test_cli_replay_and_invariance_dispatch_full_contract(tmp_path, monkeypatch, capsys):
    from coscientist.experiments import autolab_ground_truth_solver_only as experiment

    captured = []

    def fake_invariance(*args, **kwargs):
        captured.append(("invariance", args, kwargs))
        return {"status": "pass", "task_count": 17}

    def fake_replay(**kwargs):
        captured.append(("replay", (), kwargs))
        return {"status": "dry_run_pass", "task_count": 17}

    monkeypatch.setattr(experiment, "audit_invariance", fake_invariance, raising=False)
    monkeypatch.setattr(experiment, "replay_results", fake_replay, raising=False)
    common = {
        "control-runs-root": tmp_path / "controls",
        "accepted-root": tmp_path / "accepted",
        "trusted-runs-root": tmp_path / "trusted",
        "seed-dir": tmp_path / "seeds",
        "runs-root": tmp_path / "runs",
        "batch-name": BATCH,
        "problems-root": tmp_path / "problems",
    }
    argv = ["audit-invariance"]
    for key, value in common.items():
        argv += [f"--{key}", str(value)]
    argv += ["--live", "--require-query-tasks", "7"]
    assert main(argv) == 0
    capsys.readouterr()
    assert captured[-1][0] == "invariance"
    assert captured[-1][2]["live"] is True
    assert captured[-1][2]["require_query_tasks"] == 7

    argv = ["replay-results"]
    for key, value in common.items():
        argv += [f"--{key}", str(value)]
    argv += ["--output", str(tmp_path / "report.json"), "--dry-run"]
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "dry_run_pass"
    assert captured[-1][0] == "replay"
    assert captured[-1][2]["repeat"] == 2
    assert captured[-1][2]["dry_run"] is True
