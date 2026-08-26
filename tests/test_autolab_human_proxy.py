from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from coscientist.experiments.autolab_human_proxy import (
    ACCEPTED_ARTIFACTS,
    AUTOLAB_GROUPS,
    MAX_CONTEXT_CHARACTERS,
    audit_contexts,
    prepare_contexts,
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
                    {"task": task, "hidden_seed": "REQUIRED_PRIVATE"},
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
