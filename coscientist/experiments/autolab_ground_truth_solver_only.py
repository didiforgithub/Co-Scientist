"""Prepare the fixed-Final-Evaluator AutoLab Solver-only experiment.

This module owns the full fixed-arm audit boundary: private development seeds,
transactional run materialization, static and live invariance checks, a real
evaluator smoke gate, post-write privacy scanning, and held-out replay.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import inspect
import json
import math
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from coscientist.coevo import resources as resource_api
from coscientist.experiments import autolab_human_proxy as human_proxy_replay
from coscientist.experiments.autolab_human_proxy import (
    ACCEPTED_ARTIFACTS,
    AUTOLAB_GROUPS,
    CORRECTED_CONTROL_PREFIX,
    FINAL_REPLAY_AUDIT_SEED,
    FROZEN_BRIEF_TAIL,
    MAX_CHECKER_FILE_BYTES,
    MAX_PRESEED_FILE_BYTES,
    MAX_RUN_AUDIT_FILE_BYTES,
    TRUSTED_REPLAY_BATCHES,
    _accepted_replay_contracts,
    _canonical_sha256,
    _inspect_immutable_image,
    _read_bounded_regular,
    _read_json_object,
    _read_replay_resource,
    _reject_symlink_components,
    _rename_noreplace,
    _sha256_bytes,
    _tree_inventory,
    _validate_control_source,
    _validated_batch_name,
)


ARM = "autolab_ground_truth_solver_only"
ACCEPTED_ORIGIN = "accepted_final_evaluator"
SEED_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
BUDGET_S = 14_400.0
FEEDBACK_LEVEL = "feasible_score"
SOLVER_CONTRACT = {
    "model": "gpt-5.6-sol",
    "reasoning_effort": "high",
    "topology": "weak",
}
COPIED_CONTROL_FILES = (
    "seed_solution.json",
    "probes.json",
    "solver_env.json",
    "reframe_policy.json",
)
FIXED_CONTRACT_TAIL = """## Fixed accepted Final Evaluator contract

The accepted Final Evaluator is fixed at version 0 for this entire run. Normal
evaluation queries expose only `feasible` and `score`; higher score is better
among feasible solutions. Exceptional evaluator error detail is scrubbed at the
delivery boundary for raw, hex, base64, and URL-safe development-seed variants,
then durable artifacts must also pass the post-write privacy audit gate. There is
no verifier evolution or external expert-guidance channel in this arm. Solve the
general task rather than targeting hidden cases or attempting to reconstruct the
evaluator.
"""


def _tasks() -> tuple[str, ...]:
    return tuple(task for tasks in AUTOLAB_GROUPS.values() for task in tasks)


def _group_for(task: str) -> str:
    for group, tasks in AUTOLAB_GROUPS.items():
        if task in tasks:
            return group
    raise ValueError(f"unknown AutoLab task: {task}")


def _seed_file(seed_dir: Path, task: str) -> Path:
    return seed_dir / f"autolab_{task}.seed"


def _run(runs_root: Path, batch_name: str, task: str) -> Path:
    return runs_root / f"{batch_name}__autolab_{task}"


def _regular_mode(path: Path, expected: int, *, label: str) -> None:
    safe = _reject_symlink_components(path)
    try:
        mode = os.lstat(safe).st_mode
    except OSError as exc:
        raise RuntimeError(f"{label} is missing: {safe}") from exc
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        raise RuntimeError(f"{label} is not a regular non-symlinked file: {safe}")
    if stat.S_IMODE(mode) != expected:
        raise RuntimeError(
            f"{label} permission mismatch: expected {oct(expected)}, "
            f"got {oct(stat.S_IMODE(mode))}"
        )


def _directory_mode(path: Path, expected: int, *, label: str) -> None:
    safe = _reject_symlink_components(path)
    try:
        mode = os.lstat(safe).st_mode
    except OSError as exc:
        raise RuntimeError(f"{label} is missing: {safe}") from exc
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        raise RuntimeError(f"{label} is not a regular non-symlinked directory: {safe}")
    if stat.S_IMODE(mode) != expected:
        raise RuntimeError(
            f"{label} permission mismatch: expected {oct(expected)}, "
            f"got {oct(stat.S_IMODE(mode))}"
        )


def _json_write(path: Path, value: Any, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if mode is not None:
        path.chmod(mode)


def _seed_values(seed_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    seed_dir = _reject_symlink_components(seed_dir)
    _directory_mode(seed_dir, 0o700, label="private seed directory")
    manifest_path = seed_dir / "manifest.json"
    _regular_mode(manifest_path, 0o600, label="private seed manifest")
    manifest = _read_json_object(manifest_path, label="private seed manifest")
    if set(manifest) != {"schema_version", "arm", "tasks"}:
        raise RuntimeError("private seed manifest has unexpected top-level fields")
    if manifest.get("schema_version") != SEED_SCHEMA_VERSION:
        raise RuntimeError("private seed manifest schema mismatch")
    if manifest.get("arm") != ARM:
        raise RuntimeError("private seed manifest arm mismatch")
    metadata = manifest.get("tasks")
    if not isinstance(metadata, dict) or set(metadata) != set(_tasks()):
        raise RuntimeError("private seed manifest task set mismatch")
    expected_entries = {"manifest.json", *{f"autolab_{t}.seed" for t in _tasks()}}
    actual_entries: set[str] = set()
    for path in seed_dir.iterdir():
        _reject_symlink_components(path)
        if not path.is_file():
            raise RuntimeError(f"unexpected private seed entry: {path}")
        actual_entries.add(path.name)
    if actual_entries != expected_entries:
        raise RuntimeError("private seed directory has an unexpected task/file set")
    values: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for task in _tasks():
        path = _seed_file(seed_dir, task)
        _regular_mode(path, 0o600, label=f"private seed for {task}")
        raw = _read_bounded_regular(path, max_bytes=4096, label="private seed")
        try:
            value = raw.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"private seed is not ASCII: {task}") from exc
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise RuntimeError(f"private seed has an invalid format: {task}")
        digest = _sha256_bytes(value.encode("ascii"))
        if metadata.get(task) != {"sha256": digest}:
            raise RuntimeError(f"private seed hash mismatch: {task}")
        if value == FINAL_REPLAY_AUDIT_SEED:
            raise RuntimeError("development seed aliases the held-out audit seed")
        values[task] = value
        hashes[task] = digest
    if len(set(values.values())) != 17:
        raise RuntimeError("development seeds are not unique")
    return values, hashes


def audit_seeds(seed_dir: Path | str) -> dict[str, Any]:
    """Fail closed unless ``seed_dir`` is the exact private 17-seed bundle."""

    seed_dir = _reject_symlink_components(seed_dir)
    _values, _hashes = _seed_values(seed_dir)
    manifest_raw = _read_bounded_regular(
        seed_dir / "manifest.json", max_bytes=MAX_PRESEED_FILE_BYTES,
        label="private seed manifest",
    )
    return {
        "arm": ARM,
        "status": "pass",
        "task_count": 17,
        "seed_manifest_sha256": _sha256_bytes(manifest_raw),
    }


def prepare_seeds(seed_dir: Path | str) -> dict[str, Any]:
    """Generate and atomically publish a new private 17-seed bundle."""

    seed_dir = Path(os.path.abspath(os.fspath(seed_dir)))
    _reject_symlink_components(seed_dir.parent)
    if os.path.lexists(seed_dir):
        raise RuntimeError(f"refusing to overwrite existing seed directory: {seed_dir}")
    seed_dir.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = None
    try:
        staging = Path(
            tempfile.mkdtemp(prefix=f".{seed_dir.name}.seeds.tmp.", dir=seed_dir.parent)
        )
        staging.chmod(0o700)
        task_metadata: dict[str, dict[str, str]] = {}
        for task in _tasks():
            value = secrets.token_hex(32)
            destination = _seed_file(staging, task)
            with destination.open("x", encoding="ascii") as stream:
                stream.write(value + "\n")
            destination.chmod(0o600)
            task_metadata[task] = {
                "sha256": _sha256_bytes(value.encode("ascii"))
            }
        _json_write(
            staging / "manifest.json",
            {
                "schema_version": SEED_SCHEMA_VERSION,
                "arm": ARM,
                "tasks": task_metadata,
            },
            mode=0o600,
        )
        summary = audit_seeds(staging)
        _rename_noreplace(staging, seed_dir)
        staging = None
        return summary
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def _accepted_info(accepted_root: Path, group: str, task: str) -> dict[str, Any]:
    report_sha256, records = _accepted_replay_contracts(accepted_root)
    record = records[(group, task)]
    package = _reject_symlink_components(accepted_root / group / task)
    artifact_raw: dict[str, bytes] = {}
    for name in ACCEPTED_ARTIFACTS:
        artifact_raw[name] = _read_bounded_regular(
            package / name,
            max_bytes=MAX_PRESEED_FILE_BYTES,
            label=f"accepted required artifact {name}",
        )
    verifier_raw = artifact_raw["final_verifier.py"]
    template_raw = artifact_raw["final_ctx.template.json"]
    if _sha256_bytes(verifier_raw) != record["final_verifier_sha256"]:
        raise ValueError(f"accepted Final Evaluator hash mismatch: {task}")
    if _sha256_bytes(template_raw) != record["final_ctx_template_sha256"]:
        raise ValueError(f"accepted context template hash mismatch: {task}")
    try:
        template = json.loads(template_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"accepted context template is invalid: {task}") from exc
    if not isinstance(template, dict):
        raise ValueError(f"accepted context template is not an object: {task}")
    if "task" in template and template["task"] != task:
        raise ValueError(f"accepted context template task mismatch: {task}")
    if template.get("hidden_seed") != "REQUIRED_PRIVATE":
        raise ValueError(f"accepted context template seed placeholder mismatch: {task}")
    if template.get("checker_dir") != "REQUIRED_CHECKER_DIR":
        raise ValueError(f"accepted context template checker placeholder mismatch: {task}")
    if "validation_mode" in template and template["validation_mode"] is not False:
        raise ValueError(f"accepted context template validation mode mismatch: {task}")
    return {
        "package": package,
        "verifier_raw": verifier_raw,
        "template": template,
        "verifier_sha256": _sha256_bytes(verifier_raw),
        "template_sha256": _sha256_bytes(template_raw),
        "acceptance_report_sha256": report_sha256,
        "artifact_sha256": {
            name: _sha256_bytes(raw) for name, raw in artifact_raw.items()
        },
    }


def _control_info(
    control_root: Path, problems_root: Path, task: str
) -> dict[str, Any]:
    # Reuse the hardened provenance validator: this binds the corrected Control
    # manifest, both shipped-V0 copies and hashes, original checker/test, and the
    # tracked problem source before we consume any seed-side resource.
    validated = _validate_control_source(control_root, problems_root, task)
    source = Path(validated["source"])
    bws = source / "bootstrap_ws"
    raw: dict[str, bytes] = {}
    for name in (*COPIED_CONTROL_FILES, "SOLVER_BRIEF.md"):
        raw[name] = _read_bounded_regular(
            bws / name, max_bytes=MAX_PRESEED_FILE_BYTES,
            label=f"corrected-control {name}",
        )
    policy = json.loads(raw["reframe_policy.json"].decode("utf-8"))
    if not isinstance(policy, dict) or policy.get("admits_proof") is not False:
        raise ValueError(f"corrected-control proof policy mismatch: {task}")
    try:
        source_brief = raw["SOLVER_BRIEF.md"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"corrected-control Solver brief is not UTF-8: {task}") from exc
    if source_brief.endswith(FROZEN_BRIEF_TAIL):
        source_brief = source_brief[: -len(FROZEN_BRIEF_TAIL)]
    brief = source_brief.rstrip() + "\n\n" + FIXED_CONTRACT_TAIL
    return {
        "source": source,
        "raw": raw,
        "brief": brief,
        "seed_solution_sha256": _sha256_bytes(raw["seed_solution.json"]),
    }


def _normalized_problem_verifier(problem: Path) -> tuple[dict[str, Any], str]:
    resource_path = _reject_symlink_components(problem / "resource.toml")
    resource_raw = _read_bounded_regular(
        resource_path, max_bytes=MAX_PRESEED_FILE_BYTES, label="task resource.toml"
    )
    verifier = resource_api.load(problem).verifier.to_manifest()
    required = ("image", "cpus", "memory_mb", "timeout_sec")
    if any(verifier.get(key) is None for key in required):
        raise ValueError(f"task verifier resource is incomplete: {problem}")
    if verifier.get("allow_internet") is not False:
        raise ValueError(f"task verifier resource must disable internet: {problem}")
    normalized = {
        "image": str(verifier["image"]),
        "cpus": float(verifier["cpus"]),
        "memory_mb": int(verifier["memory_mb"]),
        "timeout_sec": float(verifier["timeout_sec"]),
        "allow_internet": False,
    }
    if not normalized["image"] or any(
        normalized[key] <= 0 for key in ("cpus", "memory_mb", "timeout_sec")
    ):
        raise ValueError(f"task verifier resource is invalid: {problem}")
    return normalized, _sha256_bytes(resource_raw)


def _trusted_info(
    trusted_root: Path,
    problems_root: Path,
    group: str,
    task: str,
    *,
    run_command: Any,
) -> dict[str, Any]:
    trusted_batch = TRUSTED_REPLAY_BATCHES[group]
    trusted_run = _reject_symlink_components(
        trusted_root / f"{trusted_batch}__autolab_{task}"
    )
    if not trusted_run.is_dir():
        raise ValueError(f"trusted checker run is missing: {task}")
    inventory = _tree_inventory(trusted_run / "checker", required_entry=None)
    checker_modes = {
        relative: stat.S_IMODE(os.lstat(trusted_run / "checker" / relative).st_mode)
        for relative in inventory
    }
    trusted_resource, trusted_manifest_sha256 = _read_replay_resource(
        trusted_run / "manifest.json"
    )
    problem = _reject_symlink_components(problems_root / f"autolab_{task}")
    if not problem.is_dir():
        raise ValueError(f"AutoLab problem is missing: {task}")
    problem_resource, resource_toml_sha256 = _normalized_problem_verifier(problem)
    if problem_resource != trusted_resource:
        raise ValueError(f"task/trusted verifier resource mismatch: {task}")
    immutable_image = _inspect_immutable_image(
        trusted_resource["image"], run_command=run_command
    )
    contract = {
        "verifier": trusted_resource,
        "immutable_image": immutable_image,
        "task_resource_toml_sha256": resource_toml_sha256,
        "trusted_manifest_sha256": trusted_manifest_sha256,
    }
    return {
        "trusted_run": trusted_run,
        "problem": problem.resolve(),
        "checker_inventory": inventory,
        "checker_modes": checker_modes,
        "checker_inventory_sha256": _canonical_sha256(
            {"files": inventory, "modes": checker_modes}
        ),
        "resource_contract": contract,
        "resource_contract_sha256": _canonical_sha256(contract),
    }


def _source_info(
    *,
    accepted_root: Path,
    control_root: Path,
    trusted_root: Path,
    problems_root: Path,
    task: str,
    seed_values: dict[str, str],
    seed_hashes: dict[str, str],
    run_command: Any,
) -> dict[str, Any]:
    group = _group_for(task)
    return {
        "task": task,
        "group": group,
        "seed": seed_values[task],
        "seed_sha256": seed_hashes[task],
        "accepted": _accepted_info(accepted_root, group, task),
        "control": _control_info(control_root, problems_root, task),
        "trusted": _trusted_info(
            trusted_root, problems_root, group, task, run_command=run_command
        ),
    }


def _manifest(info: dict[str, Any]) -> dict[str, Any]:
    accepted = info["accepted"]
    control = info["control"]
    trusted = info["trusted"]
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "arm": ARM,
        "mode": "agent_system",
        "task": info["task"],
        "task_group": info["group"],
        "raw_input_dir": str(trusted["problem"]),
        "budget_s": BUDGET_S,
        "initial_feedback_level": FEEDBACK_LEVEL,
        "freeze_verifier": True,
        "initial_verifier_origin": ACCEPTED_ORIGIN,
        "accepted_final_verifier_sha256": accepted["verifier_sha256"],
        "accepted_ctx_template_sha256": accepted["template_sha256"],
        "accepted_artifact_sha256": accepted["artifact_sha256"],
        "acceptance_report_sha256": accepted["acceptance_report_sha256"],
        "trusted_checker_run": trusted["trusted_run"].name,
        "trusted_checker_inventory": trusted["checker_inventory"],
        "trusted_checker_modes": trusted["checker_modes"],
        "trusted_checker_inventory_sha256": trusted["checker_inventory_sha256"],
        "source_contract_run": control["source"].name,
        "seed_solution_sha256": control["seed_solution_sha256"],
        "development_seed_sha256": info["seed_sha256"],
        "solver": dict(SOLVER_CONTRACT),
        "human_proxy_enabled": False,
        "human_session_enabled": False,
        "exceptional_detail_policy": "delivery_redaction_plus_postwrite_privacy_audit_required",
        "private_literal_encodings": ["raw", "hex", "base64", "urlsafe_base64"],
        "privacy_scanner_semantics": "postwrite_detection_not_delivery_redaction",
        "resource_contract": trusted["resource_contract"],
        "resource_contract_sha256": trusted["resource_contract_sha256"],
        "pinned_verifier_image_id": trusted["resource_contract"]["immutable_image"]["id"],
    }


def _copy_checker(info: dict[str, Any], destination: Path) -> None:
    source = info["trusted"]["trusted_run"] / "checker"
    destination.mkdir(parents=True)
    for relative, expected_sha in info["trusted"]["checker_inventory"].items():
        raw = _read_bounded_regular(
            source / relative, max_bytes=MAX_CHECKER_FILE_BYTES, label="trusted checker file"
        )
        if _sha256_bytes(raw) != expected_sha:
            raise ValueError(f"trusted checker changed during copy: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
        target.chmod(info["trusted"]["checker_modes"][relative])


def _preseed_event(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "t": 0.0,
        "kind": "accepted_final_evaluator_preseed",
        "task": info["task"],
        "accepted_final_verifier_sha256": info["accepted"]["verifier_sha256"],
    }


def _preseed_version() -> dict[str, Any]:
    return {
        "t": 0.0,
        "version": 0,
        "origin": ACCEPTED_ORIGIN,
        "note": "accepted Final Evaluator fixed from turn zero",
        "rationale": "ground-truth Solver-only arm; verifier evolution disabled",
        "has_feedback": False,
    }


def _write_run(
    destination: Path,
    *,
    published_destination: Path,
    info: dict[str, Any],
) -> None:
    bws = destination / "bootstrap_ws"
    vdir = destination / "supervisor" / "verifier_versions"
    bws.mkdir(parents=True)
    vdir.mkdir(parents=True)
    _copy_checker(info, destination / "checker")
    verifier_raw = info["accepted"]["verifier_raw"]
    for path in (bws / "verifier.py", vdir / "v0.py"):
        with path.open("xb") as stream:
            stream.write(verifier_raw)
    for name in COPIED_CONTROL_FILES:
        raw = info["control"]["raw"][name]
        with (bws / name).open("xb") as stream:
            stream.write(raw)
    (bws / "SOLVER_BRIEF.md").write_text(
        info["control"]["brief"], encoding="utf-8"
    )
    ctx = dict(info["accepted"]["template"])
    ctx.update(
        {
            "task": info["task"],
            "checker_dir": str((published_destination / "checker").resolve()),
            "hidden_seed": info["seed"],
            "validation_mode": False,
        }
    )
    _json_write(bws / "ctx.json", ctx, mode=0o600)
    _json_write(destination / "manifest.json", _manifest(info))
    (destination / "events.jsonl").write_text(
        json.dumps(_preseed_event(info), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (destination / "supervisor" / "versions.jsonl").write_text(
        json.dumps(_preseed_version(), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _assert_pristine(run: Path) -> None:
    forbidden = (
        run / "cost.jsonl",
        run / "solver",
        run / "eval",
        run / "human",
        run / "solver_ws",
        run / "harden_ws",
        run / "result.json",
        run / "final_result.json",
        run / "run_result.json",
        run / "outcome.json",
        run / "pending_outcome.json",
        run / "progress.json",
        run / "human_guidance.md",
        run / "HUMAN_GUIDANCE.md",
        run / "supervisor/reviews.jsonl",
        run / "supervisor/probes",
    )
    if any(os.path.lexists(path) for path in forbidden):
        raise RuntimeError(f"prelaunch runtime or Human contamination detected: {run}")
    versions = sorted(
        path.name for path in (run / "supervisor/verifier_versions").iterdir()
    )
    if versions != ["v0.py"]:
        raise RuntimeError(f"fixed verifier version drift detected: {run}")


def _privacy_variants(secret: str) -> set[bytes]:
    raw = secret.encode("ascii")
    variants = {
        raw,
        raw.hex().encode("ascii"),
        base64.b64encode(raw),
        base64.b64encode(raw).rstrip(b"="),
        base64.urlsafe_b64encode(raw),
        base64.urlsafe_b64encode(raw).rstrip(b"="),
    }
    try:
        entropy = bytes.fromhex(secret)
    except ValueError:
        entropy = b""
    if entropy:
        variants.update(
            {
                entropy,
                entropy.hex().encode("ascii"),
                base64.b64encode(entropy),
                base64.b64encode(entropy).rstrip(b"="),
                base64.urlsafe_b64encode(entropy),
                base64.urlsafe_b64encode(entropy).rstrip(b"="),
            }
        )
    return {value for value in variants if len(value) >= 16}


class _SnapshotChanged(RuntimeError):
    """An otherwise-safe regular file changed while it was being inspected."""


_SNAPSHOT_RETRIES = 4
_SCAN_CHUNK_BYTES = 1024 * 1024


def _snapshot_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _read_regular_snapshot_once(
    path: Path | str, *, max_bytes: int, label: str
) -> bytes | None:
    safe_path = _reject_symlink_components(path)
    try:
        before = os.lstat(safe_path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"{label} could not be inspected safely: {safe_path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlinked file: {safe_path}")
    if before.st_size > max_bytes:
        raise ValueError(f"{label} is too large (exceeds size limit): {safe_path}")
    descriptor = -1
    try:
        descriptor = os.open(safe_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label} is not a regular file: {safe_path}")
        if _snapshot_identity(opened) != _snapshot_identity(before):
            raise _SnapshotChanged(f"{label} changed before snapshot read")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            raw = stream.read(max_bytes + 1)
            after_open = os.fstat(stream.fileno())
    except FileNotFoundError as exc:
        raise _SnapshotChanged(f"{label} disappeared during snapshot") from exc
    except OSError as exc:
        raise ValueError(f"{label} could not be opened safely: {safe_path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw) > max_bytes:
        raise ValueError(f"{label} is too large (exceeds size limit): {safe_path}")
    try:
        after_path = os.lstat(safe_path)
    except FileNotFoundError as exc:
        raise _SnapshotChanged(f"{label} disappeared after snapshot") from exc
    if (
        _snapshot_identity(after_open) != _snapshot_identity(before)
        or _snapshot_identity(after_path) != _snapshot_identity(before)
        or len(raw) != before.st_size
    ):
        raise _SnapshotChanged(f"{label} changed during snapshot read")
    return raw


def _read_stable_regular(
    path: Path | str, *, max_bytes: int, label: str
) -> bytes | None:
    for attempt in range(_SNAPSHOT_RETRIES):
        try:
            return _read_regular_snapshot_once(path, max_bytes=max_bytes, label=label)
        except _SnapshotChanged:
            if attempt + 1 == _SNAPSHOT_RETRIES:
                raise RuntimeError(f"{label} did not reach a stable snapshot") from None
    raise AssertionError("unreachable")


def _scan_regular_patterns_once(
    path: Path | str, *, patterns: set[bytes], label: str
) -> bytes | None:
    safe_path = _reject_symlink_components(path)
    try:
        before = os.lstat(safe_path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"{label} could not be inspected safely: {safe_path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlinked file: {safe_path}")
    descriptor = -1
    matched: bytes | None = None
    bytes_read = 0
    overlap = max((len(pattern) for pattern in patterns), default=1) - 1
    tail = b""
    try:
        descriptor = os.open(safe_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label} is not a regular file: {safe_path}")
        if _snapshot_identity(opened) != _snapshot_identity(before):
            raise _SnapshotChanged(f"{label} changed before streaming scan")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            while True:
                chunk = stream.read(_SCAN_CHUNK_BYTES)
                if not chunk:
                    break
                bytes_read += len(chunk)
                window = tail + chunk
                if matched is None:
                    matched = next(
                        (pattern for pattern in patterns if pattern and pattern in window),
                        None,
                    )
                tail = window[-overlap:] if overlap else b""
            after_open = os.fstat(stream.fileno())
    except FileNotFoundError as exc:
        raise _SnapshotChanged(f"{label} disappeared during streaming scan") from exc
    except OSError as exc:
        raise ValueError(f"{label} could not be opened safely: {safe_path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        after_path = os.lstat(safe_path)
    except FileNotFoundError as exc:
        raise _SnapshotChanged(f"{label} disappeared after streaming scan") from exc
    if (
        _snapshot_identity(after_open) != _snapshot_identity(before)
        or _snapshot_identity(after_path) != _snapshot_identity(before)
        or bytes_read != before.st_size
    ):
        raise _SnapshotChanged(f"{label} changed during streaming scan")
    return matched


def _scan_stable_regular_patterns(
    path: Path | str, *, patterns: set[bytes], label: str
) -> bytes | None:
    for attempt in range(_SNAPSHOT_RETRIES):
        try:
            return _scan_regular_patterns_once(path, patterns=patterns, label=label)
        except _SnapshotChanged:
            if attempt + 1 == _SNAPSHOT_RETRIES:
                raise RuntimeError(f"{label} did not reach a stable snapshot") from None
    raise AssertionError("unreachable")


def audit_privacy(
    runs_root: Path | str, batch_name: str, seed_dir: Path | str
) -> dict[str, Any]:
    """Detect persisted leaks; this is an audit gate, not delivery-time redaction."""

    batch_name = _validated_batch_name(batch_name)
    runs_root = _reject_symlink_components(runs_root)
    seed_dir = _reject_symlink_components(seed_dir)
    values, _hashes = _seed_values(seed_dir)
    if not runs_root.is_dir():
        raise RuntimeError("dedicated runs root is missing")
    try:
        seed_dir.resolve().relative_to(runs_root.resolve())
    except ValueError:
        pass
    else:
        raise RuntimeError("private seed storage must be outside the runs root")
    path_literals = {
        str(seed_dir.resolve()).encode("utf-8"),
        *{
            str(_seed_file(seed_dir, task).resolve()).encode("utf-8")
            for task in _tasks()
        },
    }
    variants = {task: _privacy_variants(secret) for task, secret in values.items()}
    scanned_files = 0

    def scan_file(path: Path, *, allowed_secret_owner: str | None = None) -> None:
        nonlocal scanned_files
        seed_patterns = {
            variant
            for owner, owner_variants in variants.items()
            if owner != allowed_secret_owner
            for variant in owner_variants
        }
        matched = _scan_stable_regular_patterns(
            path,
            patterns=path_literals | seed_patterns,
            label="privacy audit file",
        )
        if matched is None and not os.path.lexists(path):
            return
        if matched in path_literals:
            raise RuntimeError(f"private seed path leak detected: {path}")
        if matched in seed_patterns:
            raise RuntimeError(f"private development seed leak detected: {path}")
        scanned_files += 1

    for task in _tasks():
        run = _reject_symlink_components(_run(runs_root, batch_name, task))
        if not run.is_dir():
            raise RuntimeError(f"privacy audit run missing: {task}")
        allowed_secret_path = run / "bootstrap_ws/ctx.json"
        for path in run.rglob("*"):
            _reject_symlink_components(path)
            try:
                mode = os.lstat(path).st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise RuntimeError(f"privacy audit found unsafe entry: {path}")
            scan_file(
                path,
                allowed_secret_owner=(task if path == allowed_secret_path else None),
            )
    # The launcher creates ``runs_root/<batch>/batch.json`` after prelaunch.  It is
    # outside all per-task run roots, but it is still a durable manifest surface and
    # therefore belongs to the live privacy gate.
    live_batch_root = runs_root / batch_name
    if os.path.lexists(live_batch_root):
        _reject_symlink_components(live_batch_root)
        if not live_batch_root.is_dir():
            raise RuntimeError("live launcher batch root is unsafe")
        for path in live_batch_root.rglob("*"):
            _reject_symlink_components(path)
            try:
                mode = os.lstat(path).st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise RuntimeError(f"live launcher batch contains unsafe entry: {path}")
            scan_file(path)
    return {
        "arm": ARM,
        "batch_name": batch_name,
        "status": "pass",
        "scanned_tasks": 17,
        "scanned_files": scanned_files,
        "classification": "postwrite_audit_gate_not_delivery_redaction",
    }


def audit_solver_isolation_contract() -> dict[str, Any]:
    """Statically bind the two code seams that enforce Solver filesystem isolation."""

    from coscientist.coevo.agent_system import AgentSystem
    from coscientist.coevo.container import DockerContainer

    solve_source = inspect.getsource(AgentSystem.solve_and_evolve)
    refresh_source = inspect.getsource(AgentSystem._refresh_solver_ws)
    container_source = inspect.getsource(DockerContainer.start)
    if 'sol_ws = self.run_dir / "solver_ws"' not in solve_source:
        raise RuntimeError("Solver workspace derivation contract drifted")
    if "DockerContainer(workdir=sol_ws" not in solve_source:
        raise RuntimeError("Solver container no longer mounts solver_ws as its workdir")
    if 'f"{self.workdir}:/work"' not in container_source:
        raise RuntimeError("Solver container workdir mount contract drifted")
    if "ctx.json" in refresh_source or "checker" in refresh_source:
        raise RuntimeError("Solver workspace refresh exposes evaluator private state")
    if 'bootstrap_ws" / "SOLVER_BRIEF.md"' not in refresh_source:
        raise RuntimeError("Solver workspace refresh contract drifted unexpectedly")
    return {
        "arm": ARM,
        "status": "pass",
        "solver_mount": "solver_ws_only",
        "refresh_copies_ctx": False,
    }


def _audit_runs_impl(
    control_runs_root: Path,
    accepted_root: Path,
    trusted_runs_root: Path,
    seed_dir: Path,
    output_runs_root: Path,
    batch_name: str,
    *,
    problems_root: Path,
    run_command: Any,
    logical_runs_root: Path,
) -> dict[str, Any]:
    values, hashes = _seed_values(seed_dir)
    expected_names = {f"{batch_name}__autolab_{task}" for task in _tasks()}
    if not output_runs_root.is_dir():
        raise RuntimeError("dedicated runs root is missing")
    actual_names = {path.name for path in output_runs_root.iterdir()}
    if actual_names != expected_names:
        raise RuntimeError("prepared runs do not match the exact 17-task set")
    for task in _tasks():
        info = _source_info(
            accepted_root=accepted_root,
            control_root=control_runs_root,
            trusted_root=trusted_runs_root,
            problems_root=problems_root,
            task=task,
            seed_values=values,
            seed_hashes=hashes,
            run_command=run_command,
        )
        run = _reject_symlink_components(_run(output_runs_root, batch_name, task))
        logical_run = _run(logical_runs_root, batch_name, task)
        if not run.is_dir():
            raise RuntimeError(f"prepared run is missing: {task}")
        for path in run.rglob("*"):
            _reject_symlink_components(path)
        verifier_raw = info["accepted"]["verifier_raw"]
        for path in (
            run / "bootstrap_ws/verifier.py",
            run / "supervisor/verifier_versions/v0.py",
        ):
            if _read_bounded_regular(
                path, max_bytes=MAX_PRESEED_FILE_BYTES, label="prepared accepted verifier"
            ) != verifier_raw:
                raise RuntimeError(f"accepted Final Evaluator mismatch: {task}")
        for name in COPIED_CONTROL_FILES:
            actual = _read_bounded_regular(
                run / "bootstrap_ws" / name,
                max_bytes=MAX_PRESEED_FILE_BYTES,
                label=f"prepared {name}",
            )
            if actual != info["control"]["raw"][name]:
                raise RuntimeError(f"corrected-control resource mismatch: {task}/{name}")
        brief = _read_bounded_regular(
            run / "bootstrap_ws/SOLVER_BRIEF.md",
            max_bytes=MAX_PRESEED_FILE_BYTES,
            label="prepared Solver brief",
        ).decode("utf-8")
        if brief != info["control"]["brief"]:
            raise RuntimeError(f"fixed Solver brief mismatch: {task}")
        ctx_path = run / "bootstrap_ws/ctx.json"
        _regular_mode(ctx_path, 0o600, label="run-local private evaluator context")
        ctx = _read_json_object(ctx_path, label="run-local private evaluator context")
        expected_ctx = dict(info["accepted"]["template"])
        expected_ctx.update(
            {
                "task": task,
                "checker_dir": str((logical_run / "checker").resolve()),
                "hidden_seed": info["seed"],
                "validation_mode": False,
            }
        )
        if ctx != expected_ctx:
            raise RuntimeError(f"run-local seed/checker context mismatch: {task}")
        if _tree_inventory(run / "checker", required_entry=None) != info["trusted"]["checker_inventory"]:
            raise RuntimeError(f"trusted checker mismatch: {task}")
        actual_checker_modes = {
            relative: stat.S_IMODE(os.lstat(run / "checker" / relative).st_mode)
            for relative in info["trusted"]["checker_inventory"]
        }
        if actual_checker_modes != info["trusted"]["checker_modes"]:
            raise RuntimeError(f"trusted checker permission mismatch: {task}")
        manifest = _read_json_object(run / "manifest.json", label="prepared manifest")
        if manifest != _manifest(info):
            raise RuntimeError(f"prepared manifest/resource contract mismatch: {task}")
        try:
            events = [
                json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()
                if line.strip()
            ]
            versions = [
                json.loads(line)
                for line in (run / "supervisor/versions.jsonl").read_text().splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"preseed metadata is invalid: {task}") from exc
        if events != [_preseed_event(info)]:
            raise RuntimeError(f"prelaunch event contamination: {task}")
        if versions != [_preseed_version()]:
            raise RuntimeError(f"accepted verifier versions metadata drift: {task}")
        _assert_pristine(run)
        from coscientist.coevo.agent_system import AgentSystem

        # ``AgentSystem.__post_init__`` constructs RunStore and therefore creates
        # runtime directories.  Invoke the pure can_resume predicate on an uninitialised
        # instance so the static audit itself cannot contaminate the launch boundary.
        resumability_probe = object.__new__(AgentSystem)
        resumability_probe.run_dir = run
        if not AgentSystem.can_resume(resumability_probe):
            raise RuntimeError(f"prepared run cannot resume: {task}")
    audit_solver_isolation_contract()
    audit_privacy(output_runs_root, batch_name, seed_dir)
    return {
        "arm": ARM,
        "batch_name": batch_name,
        "status": "pass",
        "task_count": 17,
    }


def audit_runs(
    control_runs_root: Path | str,
    accepted_root: Path | str,
    trusted_runs_root: Path | str,
    seed_dir: Path | str,
    output_runs_root: Path | str,
    batch_name: str,
    *,
    problems_root: Path | str,
    run_command: Any = subprocess.run,
    _logical_runs_root: Path | str | None = None,
) -> dict[str, Any]:
    """Statically audit the complete published 17-run boundary."""

    batch_name = _validated_batch_name(batch_name)
    logical_runs_root = (
        _reject_symlink_components(_logical_runs_root)
        if _logical_runs_root is not None
        else _reject_symlink_components(output_runs_root)
    )
    return _audit_runs_impl(
        _reject_symlink_components(control_runs_root),
        _reject_symlink_components(accepted_root),
        _reject_symlink_components(trusted_runs_root),
        _reject_symlink_components(seed_dir),
        _reject_symlink_components(output_runs_root),
        batch_name,
        problems_root=_reject_symlink_components(problems_root),
        run_command=run_command,
        logical_runs_root=logical_runs_root,
    )


def prepare_runs(
    control_runs_root: Path | str,
    accepted_root: Path | str,
    trusted_runs_root: Path | str,
    seed_dir: Path | str,
    output_runs_root: Path | str,
    batch_name: str,
    *,
    problems_root: Path | str,
    run_command: Any = subprocess.run,
) -> dict[str, Any]:
    """Build, audit, and atomically publish all 17 fixed-evaluator runs."""

    batch_name = _validated_batch_name(batch_name)
    control_runs_root = _reject_symlink_components(control_runs_root)
    accepted_root = _reject_symlink_components(accepted_root)
    trusted_runs_root = _reject_symlink_components(trusted_runs_root)
    seed_dir = _reject_symlink_components(seed_dir)
    problems_root = _reject_symlink_components(problems_root)
    output_runs_root = Path(os.path.abspath(os.fspath(output_runs_root)))
    _reject_symlink_components(output_runs_root.parent)
    if os.path.lexists(output_runs_root):
        raise RuntimeError(
            f"refusing to overwrite existing dedicated runs root: {output_runs_root}"
        )
    try:
        seed_dir.resolve().relative_to(output_runs_root.resolve())
    except ValueError:
        pass
    else:
        raise RuntimeError("private seed storage must be outside the runs root")
    output_runs_root.parent.mkdir(parents=True, exist_ok=True)
    values, hashes = _seed_values(seed_dir)
    source_inventory = {
        task: _source_info(
            accepted_root=accepted_root,
            control_root=control_runs_root,
            trusted_root=trusted_runs_root,
            problems_root=problems_root,
            task=task,
            seed_values=values,
            seed_hashes=hashes,
            run_command=run_command,
        )
        for task in _tasks()
    }
    staging: Path | None = None
    try:
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{batch_name}.runs.tmp.", dir=output_runs_root.parent
            )
        )
        for task, info in source_inventory.items():
            _write_run(
                _run(staging, batch_name, task),
                published_destination=_run(output_runs_root, batch_name, task),
                info=info,
            )
        audit_runs(
            control_runs_root,
            accepted_root,
            trusted_runs_root,
            seed_dir,
            staging,
            batch_name,
            problems_root=problems_root,
            run_command=run_command,
            _logical_runs_root=output_runs_root,
        )
        _rename_noreplace(staging, output_runs_root)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
    return {
        "arm": ARM,
        "batch_name": batch_name,
        "status": "pass",
        "task_count": 17,
    }


def _formal_state(run: Path) -> tuple[dict[str, Any], str]:
    """Hash formal run surfaces without creating RunStore/runtime directories."""

    state: dict[str, Any] = {}
    for relative in (
        "manifest.json",
        "events.jsonl",
        "supervisor/versions.jsonl",
        "eval/queries.jsonl",
    ):
        path = run / relative
        if not os.path.lexists(path):
            state[relative] = None
            continue
        state[relative] = _sha256_bytes(
            _read_bounded_regular(
                path, max_bytes=MAX_RUN_AUDIT_FILE_BYTES,
                label=f"formal smoke snapshot {relative}",
            )
        )
    for relative in ("supervisor/verifier_versions", "solver/candidates"):
        root = _reject_symlink_components(run / relative)
        if not os.path.lexists(root):
            state[relative] = None
            continue
        if not root.is_dir():
            raise RuntimeError(f"formal smoke surface is not a directory: {root}")
        files: dict[str, str] = {}
        for path in root.rglob("*"):
            _reject_symlink_components(path)
            mode = os.lstat(path).st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise RuntimeError(f"formal smoke surface contains unsafe entry: {path}")
            files[path.relative_to(root).as_posix()] = _sha256_bytes(
                _read_bounded_regular(
                    path, max_bytes=MAX_RUN_AUDIT_FILE_BYTES,
                    label=f"formal smoke snapshot {relative}",
                )
            )
        state[relative] = files
    return state, _canonical_sha256(state)


def _smoke_backend(manifest: dict[str, Any], *, task: str) -> dict[str, Any]:
    """Reconstruct the exact verifier backend used by formal AgentSystem resume."""

    try:
        verifier = manifest["resource_contract"]["verifier"]
        pinned = manifest["pinned_verifier_image_id"]
        backend = {
            "image": pinned,
            "gpus": verifier.get("gpus"),
            "cpus": verifier["cpus"],
            "memory_mb": verifier["memory_mb"],
            "allow_internet": verifier["allow_internet"],
            "timeout_s": verifier["timeout_sec"],
        }
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"smoke resource contract is malformed: {task}") from exc
    if (
        backend["image"] != manifest.get("resource_contract", {})
        .get("immutable_image", {}).get("id")
        or backend["allow_internet"] is not False
        or not isinstance(backend["cpus"], (int, float))
        or isinstance(backend["cpus"], bool)
        or backend["cpus"] <= 0
        or not isinstance(backend["memory_mb"], int)
        or isinstance(backend["memory_mb"], bool)
        or backend["memory_mb"] <= 0
        or not isinstance(backend["timeout_s"], (int, float))
        or isinstance(backend["timeout_s"], bool)
        or backend["timeout_s"] <= 0
    ):
        raise RuntimeError(f"smoke resource contract is invalid: {task}")
    return backend


def _strict_smoke_source(accepted_source: str) -> str:
    """Add a result-shape guard without changing the accepted verifier logic."""

    return accepted_source.rstrip() + """

import math as _coscientist_smoke_math
_coscientist_smoke_accepted_verify = verify

def verify(payload, ctx):
    _coscientist_smoke_result = _coscientist_smoke_accepted_verify(payload, ctx)
    if (
        not isinstance(_coscientist_smoke_result, dict)
        or not {'feasible', 'raw', 'artifacts'} <= set(_coscientist_smoke_result)
        or type(_coscientist_smoke_result['feasible']) is not bool
        or type(_coscientist_smoke_result['raw']) not in (int, float)
        or not _coscientist_smoke_math.isfinite(
            float(_coscientist_smoke_result['raw'])
        )
        or not isinstance(_coscientist_smoke_result['artifacts'], dict)
    ):
        raise ValueError('malformed accepted evaluator result')
    return _coscientist_smoke_result
"""


def _publish_json_report(path: Path, payload: dict[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: Path | None = None
    descriptor = -1
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _rename_noreplace(temporary, path)
        temporary = None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return _sha256_bytes(raw)


def smoke_runs(
    control_runs_root: Path | str,
    accepted_root: Path | str,
    trusted_runs_root: Path | str,
    seed_dir: Path | str,
    runs_root: Path | str,
    batch_name: str,
    report: Path | str,
    *,
    problems_root: Path | str,
    run_command: Any = subprocess.run,
) -> dict[str, Any]:
    """Run every original seed through its real fixed accepted evaluator V0."""

    from coscientist.demo.evaluator import Evaluator, VerifierVersion

    batch_name = _validated_batch_name(batch_name)
    runs_root = _reject_symlink_components(runs_root)
    report = _reject_symlink_components(report)
    try:
        report.relative_to(runs_root)
    except ValueError:
        pass
    else:
        raise ValueError("smoke report must be outside the dedicated runs root")
    if os.path.lexists(report):
        raise RuntimeError(f"refusing to overwrite existing smoke report: {report}")
    report.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(report.parent)

    audit_arguments = (
        control_runs_root,
        accepted_root,
        trusted_runs_root,
        seed_dir,
        runs_root,
        batch_name,
    )
    audit_runs(
        *audit_arguments,
        problems_root=problems_root,
        run_command=run_command,
    )
    audit_privacy(runs_root, batch_name, seed_dir)

    snapshots: dict[str, tuple[dict[str, Any], str]] = {}
    rows: list[dict[str, Any]] = []
    for task in _tasks():
        run = _reject_symlink_components(_run(runs_root, batch_name, task))
        snapshots[task] = _formal_state(run)
        manifest_path = run / "manifest.json"
        verifier_path = run / "supervisor/verifier_versions/v0.py"
        seed_path = run / "bootstrap_ws/seed_solution.json"
        ctx_path = run / "bootstrap_ws/ctx.json"
        manifest = _read_json_object(manifest_path, label="smoke run manifest")
        seed_solution = _read_json_object(seed_path, label="smoke original seed solution")
        ctx = _read_json_object(ctx_path, label="smoke private evaluator context")
        verifier_raw = _read_bounded_regular(
            verifier_path, max_bytes=MAX_PRESEED_FILE_BYTES,
            label="smoke accepted Final Evaluator V0",
        )
        try:
            verifier_src = verifier_raw.decode("utf-8")
        except UnicodeDecodeError:
            raise RuntimeError(f"smoke evaluator is not UTF-8: {task}") from None
        evaluator = Evaluator(exec_backend=_smoke_backend(manifest, task=task))
        evaluator.versions.append(
            VerifierVersion(0, verifier_src, ACCEPTED_ORIGIN, "prelaunch seed smoke")
        )
        try:
            result = evaluator.run(
                seed_solution, ctx, source=_strict_smoke_source(verifier_src)
            )
        except Exception:
            raise RuntimeError(
                f"smoke evaluator crashed unexpectedly: {task}"
            ) from None
        if getattr(result, "error", None):
            failure = (
                "timeout" if "timeout" in str(result.error).lower() else "crashed"
            )
            raise RuntimeError(f"smoke evaluator {failure}: {task}")
        if type(getattr(result, "feasible", None)) is not bool:
            raise RuntimeError(f"smoke evaluator returned malformed feasibility: {task}")
        raw = getattr(result, "raw", None)
        if (
            not isinstance(raw, (int, float))
            or isinstance(raw, bool)
            or not math.isfinite(float(raw))
            or not isinstance(getattr(result, "artifacts", None), dict)
        ):
            raise RuntimeError(f"smoke evaluator returned malformed or non-finite output: {task}")
        rows.append(
            {
                "task": task,
                "manifest_sha256": _sha256_bytes(
                    _read_bounded_regular(
                        manifest_path, max_bytes=MAX_RUN_AUDIT_FILE_BYTES,
                        label="smoke manifest hash",
                    )
                ),
                "accepted_final_verifier_sha256": manifest[
                    "accepted_final_verifier_sha256"
                ],
                "seed_solution_sha256": manifest["seed_solution_sha256"],
                "development_seed_sha256": manifest["development_seed_sha256"],
                "checker_inventory_sha256": manifest[
                    "trusted_checker_inventory_sha256"
                ],
                "resource_contract_sha256": manifest["resource_contract_sha256"],
                "formal_state_sha256": snapshots[task][1],
                "feasible": result.feasible,
                "raw": float(raw),
            }
        )

    for task in _tasks():
        run = _run(runs_root, batch_name, task)
        if _formal_state(run) != snapshots[task]:
            raise RuntimeError(f"smoke mutated formal run state: {task}")
    audit_runs(
        *audit_arguments,
        problems_root=problems_root,
        run_command=run_command,
    )
    audit_privacy(runs_root, batch_name, seed_dir)

    payload = {
        "schema_version": 1,
        "arm": ARM,
        "batch_name": batch_name,
        "status": "pass",
        "task_count": 17,
        "tasks": rows,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    seed_values, _seed_hashes = _seed_values(_reject_symlink_components(seed_dir))
    if any(
        variant in encoded
        for secret in seed_values.values()
        for variant in _privacy_variants(secret)
    ):
        raise RuntimeError("private development seed leaked into smoke report")
    report_sha256 = _publish_json_report(report, payload)
    return {
        "arm": ARM,
        "batch_name": batch_name,
        "status": "pass",
        "task_count": 17,
        "report_sha256": report_sha256,
    }


def _read_json_lines(
    path: Path, *, label: str, allow_incomplete_tail: bool = False
) -> list[dict[str, Any]]:
    raw = _read_stable_regular(
        path, max_bytes=MAX_RUN_AUDIT_FILE_BYTES, label=label
    )
    if raw is None:
        return []
    rows: list[dict[str, Any]] = []
    pieces = raw.split(b"\n")
    trailing = b""
    if raw.endswith(b"\n"):
        pieces.pop()
    else:
        trailing = pieces.pop() if pieces else b""

    def parse_complete(line: bytes) -> None:
        if not line.strip():
            return
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label} is not valid JSONL") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} contains a non-object record")
        rows.append(value)

    for piece in pieces:
        parse_complete(piece)
    if trailing.strip():
        try:
            trailing_value = json.loads(trailing.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            # A writer can be between write(2) calls.  Only the single unterminated
            # final record is tolerated; every newline-terminated record above is
            # parsed strictly and fails closed.
            if not allow_incomplete_tail:
                raise ValueError(f"{label} has an incomplete JSONL tail") from exc
        else:
            if not isinstance(trailing_value, dict):
                raise ValueError(f"{label} contains a non-object record")
            rows.append(trailing_value)
    return rows


def _is_strict_integer_zero(value: Any) -> bool:
    return type(value) is int and value == 0


def _source_inventory(
    *,
    control_runs_root: Path,
    accepted_root: Path,
    trusted_runs_root: Path,
    seed_dir: Path,
    problems_root: Path,
    run_command: Any,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    seed_values, seed_hashes = _seed_values(seed_dir)
    return seed_values, {
        task: _source_info(
            accepted_root=accepted_root,
            control_root=control_runs_root,
            trusted_root=trusted_runs_root,
            problems_root=problems_root,
            task=task,
            seed_values=seed_values,
            seed_hashes=seed_hashes,
            run_command=run_command,
        )
        for task in _tasks()
    }


def _human_config_active(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_human_config_active(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_human_config_active(item) for item in value)
    return value not in (None, False, "", 0)


def _assert_no_human_config(value: dict[str, Any], *, label: str) -> None:
    for key, item in value.items():
        if "human" in str(key).lower() and _human_config_active(item):
            raise RuntimeError(f"{label} contains forbidden Human configuration: {key}")


def _validate_frozen_batch(
    runs_root: Path, batch_name: str, *, required: bool
) -> tuple[dict[str, Any] | None, str | None]:
    batch_path = runs_root / batch_name / "batch.json"
    if not batch_path.exists():
        if required:
            raise RuntimeError("live invariance audit requires a batch.json manifest")
        return None, None
    batch_raw = _read_bounded_regular(
        batch_path, max_bytes=MAX_RUN_AUDIT_FILE_BYTES, label="launcher batch manifest"
    )
    try:
        batch = json.loads(batch_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("launcher batch manifest is invalid") from exc
    if not isinstance(batch, dict):
        raise ValueError("launcher batch manifest is not an object")
    if batch.get("schema_version") != 3:
        raise RuntimeError("launcher batch must use schema_version 3")
    if batch.get("batch") != batch_name:
        raise RuntimeError("launcher batch name mismatch")
    if batch.get("freeze_verifier") is not True:
        raise RuntimeError("launcher batch is not frozen")
    human_agent = batch.get("human_agent")
    if human_agent is not None and (
        not isinstance(human_agent, dict) or any(value is not None for value in human_agent.values())
    ):
        raise RuntimeError("launcher batch contains Human Agent configuration")
    if batch.get("human_proxy_context_dir") is not None:
        raise RuntimeError("launcher batch contains Human Proxy configuration")
    _assert_no_human_config(batch, label="launcher batch")

    rows = batch.get("runs")
    contracts = batch.get("run_contracts")
    if not isinstance(rows, list) or len(rows) != 17 or not isinstance(contracts, dict):
        raise RuntimeError("launcher batch does not contain exact 17 run contracts")
    row_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("run_id"), str):
            raise RuntimeError("launcher batch contains an invalid run row")
        run_id = row["run_id"]
        if run_id in row_by_id:
            raise RuntimeError(f"launcher batch has ambiguous task mapping: {run_id}")
        row_by_id[run_id] = row
    expected_ids = {f"{batch_name}__autolab_{task}" for task in _tasks()}
    if set(row_by_id) != expected_ids or set(contracts) != expected_ids:
        raise RuntimeError("launcher batch task mapping is not the exact 17-task set")
    for task in _tasks():
        run_id = f"{batch_name}__autolab_{task}"
        row = row_by_id[run_id]
        contract = contracts[run_id]
        if not isinstance(contract, dict):
            raise RuntimeError(f"launcher run contract is malformed: {task}")
        if Path(str(row.get("input_dir", ""))).name != f"autolab_{task}":
            raise RuntimeError(f"launcher batch input mapping mismatch: {task}")
        if row.get("freeze_verifier") is not True or contract.get("freeze_verifier") is not True:
            raise RuntimeError(f"launcher freeze_verifier contract mismatch: {task}")
        forbidden_row = (
            "human_proxy_context",
            "human_proxy_context_sha256",
            "human_proxy_context_dir",
            "human_agent_timeout_s",
            "human_agent_model",
            "human_agent_reasoning_effort",
        )
        if any(row.get(field) is not None for field in forbidden_row):
            raise RuntimeError(f"launcher run contains Human configuration: {task}")
        if any(
            contract.get(field) is not None
            for field in ("human_proxy_context", "human_proxy_context_sha256")
        ):
            raise RuntimeError(f"launcher run contract contains Human Proxy data: {task}")
        _assert_no_human_config(row, label=f"launcher run {task}")
        _assert_no_human_config(contract, label=f"launcher run contract {task}")
    batch["_row_by_id"] = row_by_id
    return batch, _sha256_bytes(batch_raw)


def audit_invariance(
    control_runs_root: Path | str,
    accepted_root: Path | str,
    trusted_runs_root: Path | str,
    seed_dir: Path | str,
    runs_root: Path | str,
    batch_name: str,
    *,
    problems_root: Path | str,
    live: bool = False,
    require_query_tasks: int = 0,
    allow_incomplete_jsonl_tail: bool | None = None,
    run_command: Any = subprocess.run,
) -> dict[str, Any]:
    """Audit that the fixed accepted evaluator never moved from accepted v0."""

    if require_query_tasks < 0 or require_query_tasks > 17:
        raise ValueError("require_query_tasks must be between 0 and 17")
    allow_incomplete_tail = (
        live
        if allow_incomplete_jsonl_tail is None
        else allow_incomplete_jsonl_tail
    )
    batch_name = _validated_batch_name(batch_name)
    control_runs_root = _reject_symlink_components(control_runs_root)
    accepted_root = _reject_symlink_components(accepted_root)
    trusted_runs_root = _reject_symlink_components(trusted_runs_root)
    seed_dir = _reject_symlink_components(seed_dir)
    runs_root = _reject_symlink_components(runs_root)
    problems_root = _reject_symlink_components(problems_root)
    if not runs_root.is_dir():
        raise RuntimeError("dedicated runs root is missing")
    expected_ids = {f"{batch_name}__autolab_{task}" for task in _tasks()}
    prefixed = {
        path.name
        for path in runs_root.iterdir()
        if path.name.startswith(f"{batch_name}__autolab_")
    }
    if prefixed != expected_ids:
        raise RuntimeError("invariance audit requires the exact 17 unambiguous task runs")
    _batch, _batch_sha = _validate_frozen_batch(runs_root, batch_name, required=live)
    values, source = _source_inventory(
        control_runs_root=control_runs_root,
        accepted_root=accepted_root,
        trusted_runs_root=trusted_runs_root,
        seed_dir=seed_dir,
        problems_root=problems_root,
        run_command=run_command,
    )

    query_tasks = 0
    query_count = 0
    for task in _tasks():
        run = _reject_symlink_components(_run(runs_root, batch_name, task))
        info = source[task]
        if not run.is_dir():
            raise RuntimeError(f"invariance run is missing: {task}")
        for path in run.rglob("*"):
            _reject_symlink_components(path)
        manifest = _read_json_object(run / "manifest.json", label="run manifest")
        _assert_no_human_config(manifest, label=f"run manifest {task}")
        for key, expected in _manifest(info).items():
            if manifest.get(key) != expected:
                raise RuntimeError(f"run manifest frozen invariant mismatch: {task}/{key}")
        if "verifier_hardenings" in manifest and not _is_strict_integer_zero(
            manifest["verifier_hardenings"]
        ):
            raise RuntimeError(f"run manifest records verifier hardenings: {task}")
        if "final_verifier_version" in manifest and not _is_strict_integer_zero(
            manifest["final_verifier_version"]
        ):
            raise RuntimeError(f"run manifest final verifier version drift: {task}")

        accepted_raw = info["accepted"]["verifier_raw"]
        if _read_bounded_regular(
            run / "bootstrap_ws/verifier.py",
            max_bytes=MAX_PRESEED_FILE_BYTES,
            label="accepted bootstrap v0",
        ) != accepted_raw:
            raise RuntimeError(f"accepted bootstrap evaluator drift: {task}")
        version_dir = run / "supervisor/verifier_versions"
        version_files = {path.name for path in version_dir.iterdir()}
        if version_files != {"v0.py"}:
            raise RuntimeError(f"accepted verifier version set drift: {task}")
        if _read_bounded_regular(
            version_dir / "v0.py",
            max_bytes=MAX_PRESEED_FILE_BYTES,
            label="accepted supervisor v0",
        ) != accepted_raw:
            raise RuntimeError(f"accepted supervisor v0 drift: {task}")
        versions = _read_json_lines(
            run / "supervisor/versions.jsonl",
            label="accepted versions metadata",
            allow_incomplete_tail=allow_incomplete_tail,
        )
        if (
            len(versions) != 1
            or not _is_strict_integer_zero(versions[0].get("version"))
            or versions != [_preseed_version()]
        ):
            raise RuntimeError(f"accepted versions metadata drift: {task}")

        ctx_path = run / "bootstrap_ws/ctx.json"
        _regular_mode(ctx_path, 0o600, label="private evaluator context")
        ctx = _read_json_object(ctx_path, label="private evaluator context")
        expected_ctx = dict(info["accepted"]["template"])
        expected_ctx.update(
            {
                "task": task,
                "checker_dir": str((run / "checker").resolve()),
                "hidden_seed": values[task],
                "validation_mode": False,
            }
        )
        if ctx != expected_ctx:
            raise RuntimeError(f"accepted context/checker/seed invariant drift: {task}")
        if _tree_inventory(run / "checker", required_entry=None) != info["trusted"]["checker_inventory"]:
            raise RuntimeError(f"trusted checker invariant drift: {task}")
        for name in COPIED_CONTROL_FILES:
            if _read_bounded_regular(
                run / "bootstrap_ws" / name,
                max_bytes=MAX_PRESEED_FILE_BYTES,
                label=f"preseed {name}",
            ) != info["control"]["raw"][name]:
                raise RuntimeError(f"preseed resource invariant drift: {task}/{name}")

        events = _read_json_lines(
            run / "events.jsonl",
            label="run events",
            allow_incomplete_tail=allow_incomplete_tail,
        )
        if not events or events[0] != _preseed_event(info):
            raise RuntimeError(f"accepted preseed event invariant drift: {task}")
        for event in events:
            kind = str(event.get("kind", "")).lower()
            if kind == "mode_switch" or "human" in kind:
                raise RuntimeError(f"forbidden mode/Human event in frozen run: {task}")
        for forbidden in (
            run / "human",
            run / "human_guidance.md",
            run / "HUMAN_GUIDANCE.md",
            run / "harden_ws",
        ):
            if os.path.lexists(forbidden):
                raise RuntimeError(f"forbidden Human/hardening artifact in frozen run: {task}")
        for forbidden_file in (
            run / "supervisor/reviews.jsonl",
            run / "supervisor/probes",
        ):
            if forbidden_file.is_file() and forbidden_file.stat().st_size:
                raise RuntimeError(f"forbidden verifier hardening artifact: {task}")
            if forbidden_file.is_dir() and any(forbidden_file.iterdir()):
                raise RuntimeError(f"forbidden verifier hardening artifact: {task}")

        queries = _read_json_lines(
            run / "eval/queries.jsonl",
            label="eval queries",
            allow_incomplete_tail=allow_incomplete_tail,
        )
        if queries:
            query_tasks += 1
            query_count += len(queries)
        if any(
            not _is_strict_integer_zero(query.get("verifier_version"))
            for query in queries
        ):
            raise RuntimeError(f"eval query used a nonzero verifier version: {task}")

    if query_tasks < require_query_tasks:
        raise RuntimeError(
            f"only {query_tasks} tasks have eval queries; required {require_query_tasks}"
        )
    audit_privacy(runs_root, batch_name, seed_dir)
    return {
        "arm": ARM,
        "batch_name": batch_name,
        "status": "pass",
        "mode": "live" if live else "prelaunch",
        "task_count": 17,
        "query_tasks": query_tasks,
        "query_count": query_count,
        "required_query_tasks": require_query_tasks,
    }


def _assert_audit_seed_absent(runs_root: Path, batch_name: str) -> None:
    variants = _privacy_variants(FINAL_REPLAY_AUDIT_SEED)
    roots = [_run(runs_root, batch_name, task) for task in _tasks()]
    batch_root = runs_root / batch_name
    if batch_root.exists():
        roots.append(batch_root)
    for root in roots:
        for path in root.rglob("*"):
            _reject_symlink_components(path)
            try:
                mode = os.lstat(path).st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise RuntimeError(f"search-time artifact is unsafe: {path}")
            matched = _scan_stable_regular_patterns(
                path,
                patterns=variants,
                label="search-time artifact",
            )
            if matched is not None:
                raise RuntimeError(
                    "held-out audit seed leaked into a search-time artifact"
                )


def _ground_truth_replay_contracts(
    *,
    runs_root: Path,
    batch_name: str,
    source: dict[str, dict[str, Any]],
    dry_run: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    batch, batch_sha = _validate_frozen_batch(runs_root, batch_name, required=not dry_run)
    contracts: list[dict[str, Any]] = []
    for task in _tasks():
        info = source[task]
        run_id = f"{batch_name}__autolab_{task}"
        run = _run(runs_root, batch_name, task)
        manifest_path = run / "manifest.json"
        manifest_raw = _read_bounded_regular(
            manifest_path, max_bytes=MAX_RUN_AUDIT_FILE_BYTES, label="target run manifest"
        )
        manifest = json.loads(manifest_raw)
        best = manifest.get("best_solution")
        if not dry_run:
            assert batch is not None
            for field in ("verifier_hardenings", "final_verifier_version"):
                if field not in manifest or not _is_strict_integer_zero(manifest[field]):
                    raise RuntimeError(
                        f"target run lacks a strict zero final verifier counter: "
                        f"{task}/{field}"
                    )
            batch_row = batch["_row_by_id"][run_id]
            status = batch_row.get("status")
            if status not in {"done", "budget_spent"}:
                raise RuntimeError(f"target run status is not complete: {task}/{status}")
            if not isinstance(best, dict):
                raise RuntimeError(f"target run lacks a valid best_solution: {task}")
            events = _read_json_lines(run / "events.jsonl", label="target run events")
            if status == "done" and not any(
                event.get("kind") == "run_stop" for event in events
            ):
                raise RuntimeError(f"target run has no run_stop event: {task}")
            if status == "budget_spent":
                deadline_epoch = batch_row.get("deadline_epoch")
                if (
                    isinstance(deadline_epoch, bool)
                    or not isinstance(deadline_epoch, (int, float))
                    or not math.isfinite(float(deadline_epoch))
                    or float(deadline_epoch) <= 0.0
                    or float(deadline_epoch) > time.time()
                ):
                    raise RuntimeError(
                        f"budget_spent run lacks an elapsed launcher deadline: {task}"
                    )
        trusted = info["trusted"]
        accepted = info["accepted"]
        contracts.append(
            {
                "task": task,
                "group": info["group"],
                "run": run,
                "run_id": run_id,
                "batch_row": None if batch is None else batch["_row_by_id"][run_id],
                "manifest": manifest,
                "manifest_path": manifest_path,
                "manifest_sha256": _sha256_bytes(manifest_raw),
                "best": best if isinstance(best, dict) else None,
                "package": accepted["package"],
                "verifier": accepted["package"] / "final_verifier.py",
                "template": accepted["package"] / "final_ctx.template.json",
                "verifier_sha256": accepted["verifier_sha256"],
                "template_sha256": accepted["template_sha256"],
                "checker": run / "checker",
                "checker_batch": TRUSTED_REPLAY_BATCHES[info["group"]],
                "checker_inventory": trusted["checker_inventory"],
                "trusted_manifest": trusted["trusted_run"] / "manifest.json",
                "trusted_manifest_sha256": trusted["resource_contract"]["trusted_manifest_sha256"],
                "resource": trusted["resource_contract"]["verifier"],
                "image": trusted["resource_contract"]["immutable_image"],
            }
        )
    return {
        "batch_manifest": (
            {"status": "not_required_prelaunch", "sha256": None}
            if batch is None
            else {"status": "validated", "sha256": batch_sha}
        ),
        "accepted_report_sha256": next(iter(source.values()))["accepted"]["acceptance_report_sha256"],
    }, contracts


def _fixed_task_base(contract: dict[str, Any]) -> dict[str, Any]:
    base = human_proxy_replay._task_report_base(contract)
    base["run"]["freeze_verifier"] = True
    return base


def _fixed_replay_task(
    contract: dict[str, Any], *, repeat: int, run_command: Any
) -> dict[str, Any]:
    row = human_proxy_replay._replay_task(
        contract,
        repeat=repeat,
        hidden_seed=FINAL_REPLAY_AUDIT_SEED,
        run_command=run_command,
    )
    row["run"]["freeze_verifier"] = True
    return row


def _assert_report_has_no_seeds(
    report: dict[str, Any], development_seeds: dict[str, str]
) -> None:
    raw = json.dumps(report, ensure_ascii=False, sort_keys=True).encode("utf-8")
    for secret in (*development_seeds.values(), FINAL_REPLAY_AUDIT_SEED):
        if any(variant and variant in raw for variant in _privacy_variants(secret)):
            raise RuntimeError("private development/audit seed would leak into replay report")


def replay_results(
    *,
    control_runs_root: Path | str,
    accepted_root: Path | str,
    trusted_runs_root: Path | str,
    seed_dir: Path | str,
    runs_root: Path | str,
    batch_name: str,
    problems_root: Path | str,
    output: Path | str,
    workers: int = 4,
    repeat: int = 2,
    dry_run: bool = False,
    run_command: Any = subprocess.run,
) -> dict[str, Any]:
    """Replay exact frozen-arm finalists on the fixed held-out audit seed."""

    started_at_utc = datetime.now(timezone.utc).isoformat()
    if workers < 1:
        raise ValueError("workers must be positive")
    if repeat != 2:
        raise ValueError("held-out replay repeat is fixed and required to be 2")
    batch_name = _validated_batch_name(batch_name)
    paths = {
        "control_runs_root": _reject_symlink_components(control_runs_root),
        "accepted_root": _reject_symlink_components(accepted_root),
        "trusted_runs_root": _reject_symlink_components(trusted_runs_root),
        "seed_dir": _reject_symlink_components(seed_dir),
        "runs_root": _reject_symlink_components(runs_root),
        "problems_root": _reject_symlink_components(problems_root),
    }
    audit_invariance(
        paths["control_runs_root"],
        paths["accepted_root"],
        paths["trusted_runs_root"],
        paths["seed_dir"],
        paths["runs_root"],
        batch_name,
        problems_root=paths["problems_root"],
        live=not dry_run,
        allow_incomplete_jsonl_tail=False,
        run_command=run_command,
    )
    development_seeds, source = _source_inventory(
        control_runs_root=paths["control_runs_root"],
        accepted_root=paths["accepted_root"],
        trusted_runs_root=paths["trusted_runs_root"],
        seed_dir=paths["seed_dir"],
        problems_root=paths["problems_root"],
        run_command=run_command,
    )
    _assert_audit_seed_absent(paths["runs_root"], batch_name)
    provenance, contracts = _ground_truth_replay_contracts(
        runs_root=paths["runs_root"], batch_name=batch_name, source=source, dry_run=dry_run
    )
    failures: list[str] = []
    if dry_run:
        tasks = [
            {
                **_fixed_task_base(contract),
                "status": "contract_validated",
                "repeats": [],
                "raw_values": [],
                "feasible_values": [],
            }
            for contract in contracts
        ]
    else:
        by_task: dict[str, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _fixed_replay_task,
                    contract,
                    repeat=repeat,
                    run_command=run_command,
                ): contract
                for contract in contracts
            }
            for future in concurrent.futures.as_completed(futures):
                contract = futures[future]
                try:
                    by_task[contract["task"]] = future.result()
                except Exception as exc:
                    failures.append(contract["task"])
                    by_task[contract["task"]] = {
                        **_fixed_task_base(contract),
                        "status": "runner_error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "repeats": [],
                        "raw_values": [],
                        "feasible_values": [],
                    }
        tasks = [by_task[task] for task in _tasks()]
    replayed = sum(row["status"] == "replayed" for row in tasks)
    report = {
        "schema_version": 1,
        "arm": ARM,
        "purpose": "Ground-Truth Solver Only 17-task held-out Final Evaluator replay",
        "status": "dry_run_pass" if dry_run else ("pass" if not failures else "fail"),
        "batch_name": batch_name,
        "started_at_utc": started_at_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_seed_sha256": _sha256_bytes(FINAL_REPLAY_AUDIT_SEED.encode("utf-8")),
        "execution": {
            "dry_run": dry_run,
            "parallel_tasks": workers,
            "sequential_repeats_per_task": 2,
            "network": "none",
            "mounts": "read_only",
            "payload_mapping": "identity",
        },
        "provenance": provenance,
        "completeness": {
            "expected_tasks": 17,
            "contract_validated_tasks": len(contracts),
            "replayed_tasks": replayed,
            "failed_tasks": len(failures),
            "complete": replayed == 17 and not failures,
        },
        "failures": failures,
        "tasks": tasks,
    }
    _assert_report_has_no_seeds(report, development_seeds)
    human_proxy_replay._write_replay_report(output, report)
    if failures:
        raise RuntimeError(f"held-out replay failed for {len(failures)} task(s)")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-seeds", "audit-seeds"):
        command = commands.add_parser(name)
        command.add_argument("--seed-dir", required=True, type=Path)
    for name in ("prepare-runs", "audit-runs"):
        command = commands.add_parser(name)
        command.add_argument("--control-runs-root", required=True, type=Path)
        command.add_argument("--accepted-root", required=True, type=Path)
        command.add_argument("--trusted-runs-root", required=True, type=Path)
        command.add_argument("--seed-dir", required=True, type=Path)
        command.add_argument("--runs-root", required=True, type=Path)
        command.add_argument("--batch-name", required=True)
        command.add_argument("--problems-root", required=True, type=Path)
    privacy = commands.add_parser("audit-privacy")
    privacy.add_argument("--seed-dir", required=True, type=Path)
    privacy.add_argument("--runs-root", required=True, type=Path)
    privacy.add_argument("--batch-name", required=True)
    smoke = commands.add_parser("smoke-runs")
    smoke.add_argument("--control-runs-root", required=True, type=Path)
    smoke.add_argument("--accepted-root", required=True, type=Path)
    smoke.add_argument("--trusted-runs-root", required=True, type=Path)
    smoke.add_argument("--seed-dir", required=True, type=Path)
    smoke.add_argument("--runs-root", required=True, type=Path)
    smoke.add_argument("--batch-name", required=True)
    smoke.add_argument("--problems-root", required=True, type=Path)
    smoke.add_argument("--report", required=True, type=Path)
    for name in ("audit-invariance", "replay-results"):
        command = commands.add_parser(name)
        command.add_argument("--control-runs-root", required=True, type=Path)
        command.add_argument("--accepted-root", required=True, type=Path)
        command.add_argument("--trusted-runs-root", required=True, type=Path)
        command.add_argument("--seed-dir", required=True, type=Path)
        command.add_argument("--runs-root", required=True, type=Path)
        command.add_argument("--batch-name", required=True)
        command.add_argument("--problems-root", required=True, type=Path)
        if name == "audit-invariance":
            command.add_argument("--live", action="store_true")
            command.add_argument("--require-query-tasks", type=int, default=0)
        else:
            command.add_argument("--workers", type=int, default=4)
            command.add_argument("--repeat", type=int, default=2)
            command.add_argument("--output", required=True, type=Path)
            command.add_argument("--dry-run", action="store_true")
    commands.add_parser("audit-solver-isolation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-seeds":
        result = prepare_seeds(args.seed_dir)
    elif args.command == "audit-seeds":
        result = audit_seeds(args.seed_dir)
    elif args.command in {"prepare-runs", "audit-runs"}:
        function = prepare_runs if args.command == "prepare-runs" else audit_runs
        result = function(
            args.control_runs_root,
            args.accepted_root,
            args.trusted_runs_root,
            args.seed_dir,
            args.runs_root,
            args.batch_name,
            problems_root=args.problems_root,
        )
    elif args.command == "audit-privacy":
        result = audit_privacy(args.runs_root, args.batch_name, args.seed_dir)
    elif args.command == "smoke-runs":
        result = smoke_runs(
            args.control_runs_root,
            args.accepted_root,
            args.trusted_runs_root,
            args.seed_dir,
            args.runs_root,
            args.batch_name,
            args.report,
            problems_root=args.problems_root,
        )
    elif args.command == "audit-invariance":
        result = audit_invariance(
            args.control_runs_root,
            args.accepted_root,
            args.trusted_runs_root,
            args.seed_dir,
            args.runs_root,
            args.batch_name,
            problems_root=args.problems_root,
            live=args.live,
            require_query_tasks=args.require_query_tasks,
        )
    elif args.command == "replay-results":
        result = replay_results(
            control_runs_root=args.control_runs_root,
            accepted_root=args.accepted_root,
            trusted_runs_root=args.trusted_runs_root,
            seed_dir=args.seed_dir,
            runs_root=args.runs_root,
            batch_name=args.batch_name,
            problems_root=args.problems_root,
            output=args.output,
            workers=args.workers,
            repeat=args.repeat,
            dry_run=args.dry_run,
        )
    else:
        result = audit_solver_isolation_contract()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
