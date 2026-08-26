"""Prepare auditable private Human Proxy contexts for the 17 AutoLab tasks.

This module deliberately handles only accepted evaluator context artifacts.  Run
pre-seeding and result replay are separate experiment stages.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import errno
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ARM = "autolab_coevolve_human_proxy"
CORRECTED_CONTROL_PREFIX = "autolab_shippedv0_control17_4h_v2_20260825"
SHIPPED_VERIFIER_ORIGIN = "autolab_shipped_tests/test.sh"
FINAL_REPLAY_AUDIT_SEED = (
    "shipped-v0-control17-final-replay-audit-only-20260825-v1"
)
TRUSTED_REPLAY_BATCHES = {
    "old10": "autolab_all_4h",
    "new7": "autolab_proofgate_4h",
}
MAX_CONTEXT_CHARACTERS = 200_000
MAX_ACCEPTED_SOURCE_BYTES = 1_000_000
MAX_RUN_AUDIT_FILE_BYTES = 4_000_000
MAX_PRESEED_FILE_BYTES = 16_000_000
MAX_BOOTSTRAP_TOTAL_BYTES = 64_000_000
MAX_CHECKER_FILE_BYTES = 32_000_000
MAX_CHECKER_TOTAL_BYTES = 256_000_000
MAX_RUN_CONTEXT_FILE_BYTES = 1_000_000
MAX_RUN_CONTEXT_TOTAL_BYTES = 16_000_000
ACCEPTED_ARTIFACTS = (
    "final_verifier.py",
    "final_ctx.template.json",
    "authored_design.md",
    "issues.md",
)
AUTOLAB_GROUPS = {
    "old10": (
        "adaptive_compression",
        "fft_rust",
        "flash_attention",
        "gaussian_blur",
        "hash_join",
        "levenshtein_distance",
        "radix_sort",
        "regex_engine",
        "sstable_compaction_rs",
        "z_order_range_scan",
    ),
    "new7": (
        "aes128_ctr",
        "agent_tool_routing",
        "bm25_search_go",
        "bvh_raytracer",
        "concurrent_kv_wal",
        "discover_sorting",
        "sha256_throughput",
    ),
}

FROZEN_BRIEF_TAIL = '''## Frozen evaluation contract

Your submission is evaluated only by this task's original shipped `tests/test.sh`.
The evaluator returns `reward.json["reward"]` unchanged as the raw score; higher is better.
The verifier is frozen at version 0 for this control run. No verifier hardening or
Bootstrap-authored hidden checks will be added during the run.
'''

EVOLVING_BRIEF_TAIL = '''## Evolving evaluation contract

Your run begins with this task's original shipped `tests/test.sh` as verifier version 0.
The evaluator initially returns `reward.json["reward"]` unchanged as the raw score;
higher is better. This is a Co-Evolve run: verifier hardening is enabled, so later
versions may add checks when evidence reveals a weakness in the current evaluator.
Continue solving the stated optimization task for arbitrary valid inputs rather than
special-casing a visible benchmark or verifier version.
'''

RUN_BOOTSTRAP_FILES = (
    "verifier.py",
    "seed_solution.json",
    "ctx.json",
    "probes.json",
    "solver_env.json",
    "reframe_policy.json",
)

NATIVE_METRICS = {
    "adaptive_compression": "8.0 minus aggregate bits per byte; lower native bpb becomes a higher score",
    "fft_rust": "fixed baseline seconds divided by median runtime",
    "flash_attention": "0.75 divided by median kernel seconds",
    "gaussian_blur": "12.0 divided by median kernel seconds after correctness",
    "hash_join": "20.0 divided by median runtime seconds",
    "levenshtein_distance": "2.0845 divided by median runtime seconds",
    "radix_sort": "4.5 divided by median runtime seconds",
    "regex_engine": "fixed baseline seconds divided by median runtime",
    "sstable_compaction_rs": "scaled baseline seconds divided by median runtime",
    "z_order_range_scan": "2.0 divided by median runtime seconds",
    "aes128_ctr": "median-normalized throughput speedup mapped through the historical log reward",
    "agent_tool_routing": "3.85 divided by median benchmark wall seconds",
    "bm25_search_go": "2.1 divided by median runtime seconds",
    "bvh_raytracer": "3.8-second anchor divided by normalized median frame time",
    "concurrent_kv_wal": "logarithmic speedup score capped at 1.0",
    "discover_sorting": "61 divided by comparator count plus one",
    "sha256_throughput": "2.5 divided by effective protected benchmark seconds",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(
    path: Path,
    *,
    max_bytes: int = MAX_PRESEED_FILE_BYTES,
    label: str = "file",
) -> str:
    return _sha256_bytes(
        _read_bounded_regular(path, max_bytes=max_bytes, label=label)
    )


def _tasks() -> tuple[str, ...]:
    return tuple(task for tasks in AUTOLAB_GROUPS.values() for task in tasks)


def _absolute_lexical(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path | str) -> Path:
    """Return an absolute lexical path after rejecting every existing symlink."""

    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"path contains a symlink component: {current}")
    return absolute


def _read_bounded_regular(
    path: Path | str, *, max_bytes: int, label: str
) -> bytes:
    """Read one trusted file with component and final-component symlink defenses."""

    safe_path = _reject_symlink_components(path)
    try:
        before = os.lstat(safe_path)
    except OSError as exc:
        raise ValueError(f"{label} is missing: {safe_path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlinked file: {safe_path}")
    if before.st_size > max_bytes:
        raise ValueError(f"{label} is too large (exceeds size limit): {safe_path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(safe_path, flags)
    except OSError as exc:
        raise ValueError(f"{label} could not be opened safely: {safe_path}") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_size > max_bytes
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError(
                f"{label} changed or is too large (exceeds size limit): {safe_path}"
            )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            raw = stream.read(max_bytes + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw) > max_bytes:
        raise ValueError(f"{label} is too large (exceeds size limit): {safe_path}")
    return raw


def _rename_noreplace(source: Path | str, destination: Path | str) -> None:
    """Atomically publish a directory without ever replacing an existing target."""

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    result: int
    if sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOSYS, "renameat2 is unavailable")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            source_bytes,
            -100,
            destination_bytes,
            1,  # RENAME_NOREPLACE
        )
    elif sys.platform == "darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise OSError(errno.ENOSYS, "renamex_np is unavailable")
        renamex_np.argtypes = (
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renamex_np.restype = ctypes.c_int
        result = renamex_np(
            source_bytes,
            destination_bytes,
            0x00000004,  # RENAME_EXCL
        )
    else:
        raise OSError(errno.ENOSYS, "atomic no-replace rename is unavailable")
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), os.fspath(destination))


def _validated_accepted_root(path: Path | str) -> Path:
    root = _reject_symlink_components(path)
    if not root.is_dir():
        raise ValueError(f"accepted root is not a directory: {root}")
    return root


def _validated_output_path(accepted_root: Path, path: Path | str) -> Path:
    output = _reject_symlink_components(path)
    if output == accepted_root or accepted_root in output.parents:
        raise ValueError("private output must not be inside the accepted root")
    return output


def _require_private_regular_file(path: Path, *, label: str) -> None:
    _reject_symlink_components(path)
    try:
        mode = os.lstat(path).st_mode
    except OSError as exc:
        raise ValueError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular non-symlinked file: {path}")


def _read_source(path: Path, *, label: str) -> tuple[bytes, str]:
    raw = _read_bounded_regular(
        path, max_bytes=MAX_ACCEPTED_SOURCE_BYTES, label=label
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8: {path}") from exc
    return raw, text


def _expected_task_pairs() -> set[tuple[str, str]]:
    return {
        (group, task)
        for group, tasks in AUTOLAB_GROUPS.items()
        for task in tasks
    }


def _accepted_records(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    if report.get("status") != "pass" or report.get("failures") != []:
        raise ValueError("acceptance report is not an accepted pass")
    aggregate = report.get("aggregate", {})
    if aggregate.get("accepted_tasks") != 17 or aggregate.get("expected_tasks") != 17:
        raise ValueError("acceptance report does not accept exactly 17 tasks")
    reported_set = report.get("results", {}).get("exact_task_set")
    expected_set = {group: list(tasks) for group, tasks in AUTOLAB_GROUPS.items()}
    if reported_set != expected_set:
        raise ValueError("acceptance report exact task set does not match old10/new7")

    records: dict[tuple[str, str], dict[str, Any]] = {}
    for item in report.get("tasks", []):
        if not isinstance(item, dict):
            raise ValueError("acceptance report task record is not an object")
        key = (item.get("group"), item.get("task"))
        if key in records:
            raise ValueError(f"acceptance report repeats task record: {key}")
        records[key] = item
    if set(records) != _expected_task_pairs():
        raise ValueError("acceptance report task records do not match exact task set")

    pass_fields = (
        "ast_parse",
        "deployment_template",
        "py_compile",
        "validation_json_parse",
        "validation_live_hash_pair",
    )
    zero_fields = (
        "runtime_entropy_hits",
        "seed_or_nonce_artifact_hits",
        "fixed_formal_fallback_hits",
    )
    for (group, task), item in records.items():
        if any(item.get(field) != "pass" for field in pass_fields):
            raise ValueError(f"acceptance report has failed checks for {group}/{task}")
        if any(item.get(field) != 0 for field in zero_fields):
            raise ValueError(f"acceptance report has unsafe hits for {group}/{task}")
        required = set(item.get("required_artifacts", []))
        expected_required = {
            f"{group}/{task}/{name}" for name in ACCEPTED_ARTIFACTS
        }
        if not expected_required.issubset(required):
            raise ValueError(
                f"acceptance report is missing required artifacts for {group}/{task}"
            )
        for field in ("final_verifier_sha256", "final_ctx_template_sha256"):
            digest = item.get(field)
            if not isinstance(digest, str) or re.fullmatch(
                r"[0-9a-fA-F]{64}", digest
            ) is None:
                raise ValueError(
                    f"acceptance report has missing or malformed hash for {group}/{task}"
                )
    return records


def _render_context(
    *, task: str, group: str, report_sha256: str, sources: Iterable[tuple[str, str]]
) -> str:
    sections = [
        "# Private accepted Final Evaluator context",
        "",
        "This file is read-only Human Proxy context. Do not quote private artifacts verbatim.",
        f"Task: {task}",
        f"Acceptance group: {group}",
        f"Acceptance report SHA-256: {report_sha256}",
        "",
    ]
    for name, content in sources:
        sections.extend(
            [
                f"===== BEGIN {name} =====",
                content,
                f"===== END {name} =====",
                "",
            ]
        )
    return "\n".join(sections)


def _build_bundle(accepted_root: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    accepted_root = _validated_accepted_root(accepted_root)
    report_path = accepted_root / "acceptance_report.json"
    report_raw, report_text = _read_source(report_path, label="acceptance report")
    try:
        report = json.loads(report_text)
    except json.JSONDecodeError as exc:
        raise ValueError("acceptance report is invalid JSON") from exc
    if not isinstance(report, dict):
        raise ValueError("acceptance report must be a JSON object")
    records = _accepted_records(report)
    report_sha256 = _sha256_bytes(report_raw)

    contexts: dict[str, bytes] = {}
    task_inventory = []
    for group, tasks in AUTOLAB_GROUPS.items():
        for task in tasks:
            package = accepted_root / group / task
            if package.is_symlink() or not package.is_dir():
                raise ValueError(
                    f"accepted package must be a regular directory: {package}"
                )
            record = records[(group, task)]
            source_inventory = []
            rendered_sources = []
            for name in ACCEPTED_ARTIFACTS:
                path = package / name
                raw, text = _read_source(path, label=f"accepted artifact {group}/{task}/{name}")
                digest = _sha256_bytes(raw)
                if name == "final_verifier.py":
                    reported_digest = record.get("final_verifier_sha256")
                elif name == "final_ctx.template.json":
                    reported_digest = record.get("final_ctx_template_sha256")
                else:
                    reported_digest = None
                if reported_digest is not None and digest != reported_digest:
                    raise ValueError(
                        f"acceptance report hash mismatch for {group}/{task}/{name}"
                    )
                source_inventory.append(
                    {
                        "name": name,
                        "path": str(path),
                        "sha256": digest,
                        "bytes": len(raw),
                        "characters": len(text),
                    }
                )
                rendered_sources.append((name, text))
            context_text = _render_context(
                task=task,
                group=group,
                report_sha256=report_sha256,
                sources=rendered_sources,
            )
            if len(context_text) > MAX_CONTEXT_CHARACTERS:
                raise ValueError(
                    f"context for {task} exceeds {MAX_CONTEXT_CHARACTERS} characters"
                )
            context_raw = context_text.encode("utf-8")
            context_name = f"autolab_{task}.md"
            contexts[context_name] = context_raw
            task_inventory.append(
                {
                    "task": task,
                    "group": group,
                    "context_path": context_name,
                    "context_sha256": _sha256_bytes(context_raw),
                    "context_bytes": len(context_raw),
                    "context_characters": len(context_text),
                    "package_path": str(package),
                    "sources": source_inventory,
                }
            )

    manifest = {
        "schema_version": 1,
        "arm": ARM,
        "accepted_report": {
            "path": str(report_path),
            "sha256": report_sha256,
            "bytes": len(report_raw),
        },
        "tasks": task_inventory,
    }
    return manifest, contexts


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _read_expected_output(path: Path, expected_size: int) -> bytes:
    try:
        output_stat = path.lstat()
    except OSError as exc:
        raise RuntimeError("private context output is missing") from exc
    if not stat.S_ISREG(output_stat.st_mode) or output_stat.st_size != expected_size:
        raise RuntimeError("private context output size mismatch")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            raw = stream.read(expected_size + 1)
    except OSError as exc:
        raise RuntimeError("private context output could not be read safely") from exc
    if len(raw) != expected_size:
        raise RuntimeError("private context output size mismatch")
    return raw


def audit_contexts(accepted_root: Path | str, output_dir: Path | str) -> dict[str, Any]:
    """Fail closed unless an output directory exactly matches accepted sources."""

    accepted_root = _validated_accepted_root(accepted_root)
    output_dir = _validated_output_path(accepted_root, output_dir)
    if not output_dir.is_dir():
        raise RuntimeError(f"private context directory is missing or unsafe: {output_dir}")
    if _mode(output_dir) != 0o700:
        raise RuntimeError(f"private context directory mode mismatch: {output_dir}")

    expected_manifest, expected_contexts = _build_bundle(accepted_root)
    expected_files = {
        **expected_contexts,
        "manifest.json": _manifest_bytes(expected_manifest),
    }
    expected_names = set(expected_files)
    actual_names = {path.name for path in output_dir.iterdir()}
    if actual_names != expected_names:
        raise RuntimeError("private context directory has missing or unexpected files")
    for path in output_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"private context output is not a regular file: {path}")
        if _mode(path) != 0o600:
            raise RuntimeError(f"private context file mode mismatch: {path}")

    actual_files = {}
    for name, expected_raw in expected_files.items():
        actual_raw = _read_expected_output(output_dir / name, len(expected_raw))
        if actual_raw != expected_raw:
            raise RuntimeError(f"private context content mismatch: {name}")
        actual_files[name] = actual_raw
    return {
        "status": "pass",
        "arm": ARM,
        "task_count": len(expected_contexts),
        "accepted_report_sha256": expected_manifest["accepted_report"]["sha256"],
        "manifest_sha256": _sha256_bytes(actual_files["manifest.json"]),
    }


def prepare_contexts(
    accepted_root: Path | str, output_dir: Path | str
) -> dict[str, Any]:
    """Create the immutable 17-task context set or validate an exact reuse."""

    accepted_root = _validated_accepted_root(accepted_root)
    output_dir = _validated_output_path(accepted_root, output_dir)
    manifest, contexts = _build_bundle(accepted_root)
    if output_dir.exists():
        try:
            audit_contexts(accepted_root, output_dir)
        except Exception as exc:
            raise RuntimeError(
                f"refusing to reuse incomplete or mismatched context directory: {output_dir}"
            ) from exc
        return manifest

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output_dir.parent)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp.",
            dir=output_dir.parent,
        )
    )
    try:
        os.chmod(staging, 0o700)
        for name, raw in contexts.items():
            path = staging / name
            with path.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(path, 0o600)
        manifest_path = staging / "manifest.json"
        manifest_raw = _manifest_bytes(manifest)
        with manifest_path.open("xb") as stream:
            stream.write(manifest_raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(manifest_path, 0o600)
        directory_descriptor = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        audit_contexts(accepted_root, staging)
        if output_dir.exists() or os.path.lexists(output_dir):
            raise RuntimeError("refusing to replace an existing context directory")
        os.rename(staging, output_dir)
        staging = None
        parent_descriptor = os.open(output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
    return manifest


def _validated_batch_name(batch_name: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", batch_name) is None:
        raise ValueError("batch name must be a filesystem-safe identifier")
    return batch_name


def _control_run(control_runs_root: Path, task: str) -> Path:
    return control_runs_root / f"{CORRECTED_CONTROL_PREFIX}__autolab_{task}"


def _prepared_run(output_runs_root: Path, batch_name: str, task: str) -> Path:
    return output_runs_root / f"{batch_name}__autolab_{task}"


def _rewrite_solver_brief(source: str, *, task: str) -> str:
    if source.count(FROZEN_BRIEF_TAIL) != 1 or not source.endswith(FROZEN_BRIEF_TAIL):
        raise ValueError(
            f"corrected Control solver brief lacks the exact frozen tail for {task}"
        )
    rewritten = source[: -len(FROZEN_BRIEF_TAIL)] + EVOLVING_BRIEF_TAIL
    if "Frozen evaluation contract" in rewritten or "verifier is frozen" in rewritten:
        raise AssertionError("frozen Control language survived solver brief rewrite")
    return rewritten


def _tree_inventory(
    root: Path,
    *,
    max_file_bytes: int | None = None,
    max_total_bytes: int | None = None,
) -> dict[str, str]:
    if max_file_bytes is None:
        max_file_bytes = MAX_CHECKER_FILE_BYTES
    if max_total_bytes is None:
        max_total_bytes = MAX_CHECKER_TOTAL_BYTES
    root = _reject_symlink_components(root)
    if not root.is_dir():
        raise ValueError(f"checker is not a regular non-symlinked directory: {root}")
    inventory: dict[str, str] = {}
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        _reject_symlink_components(path)
        mode = os.lstat(path).st_mode
        if stat.S_ISREG(mode):
            raw = _read_bounded_regular(
                path, max_bytes=max_file_bytes, label="checker file"
            )
            total_bytes += len(raw)
            if total_bytes > max_total_bytes:
                raise ValueError("checker total exceeds size limit")
            inventory[path.relative_to(root).as_posix()] = _sha256_bytes(raw)
        elif not stat.S_ISDIR(mode):
            raise ValueError(f"checker contains a non-regular entry: {path}")
    if "tests/test.sh" not in inventory:
        raise ValueError(f"corrected Control checker lacks tests/test.sh: {root}")
    return inventory


def _read_json_object(
    path: Path,
    *,
    label: str,
    max_bytes: int = MAX_PRESEED_FILE_BYTES,
) -> dict[str, Any]:
    try:
        raw = _read_bounded_regular(path, max_bytes=max_bytes, label=label)
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _validate_control_source(
    control_runs_root: Path, problems_root: Path, task: str
) -> dict[str, Any]:
    source = _reject_symlink_components(_control_run(control_runs_root, task))
    if not source.is_dir():
        raise ValueError(f"corrected Control run is missing: {source}")
    problem = _reject_symlink_components(problems_root / f"autolab_{task}")
    if not problem.is_dir():
        raise ValueError(f"AutoLab problem directory is missing: {problem}")
    bws = source / "bootstrap_ws"
    v0 = source / "supervisor" / "verifier_versions" / "v0.py"
    required = [bws / name for name in RUN_BOOTSTRAP_FILES] + [
        bws / "SOLVER_BRIEF.md",
        v0,
    ]
    bootstrap_raw: dict[str, bytes] = {}
    bootstrap_total = 0
    for path in required:
        raw = _read_bounded_regular(
            path, max_bytes=MAX_PRESEED_FILE_BYTES,
            label="corrected Control initial state",
        )
        bootstrap_total += len(raw)
        if bootstrap_total > MAX_BOOTSTRAP_TOTAL_BYTES:
            raise ValueError("corrected Control bootstrap total exceeds size limit")
        bootstrap_raw[path.relative_to(source).as_posix()] = raw
    v0_raw = bootstrap_raw["supervisor/verifier_versions/v0.py"]
    if bootstrap_raw["bootstrap_ws/verifier.py"] != v0_raw:
        raise ValueError(f"corrected Control has inconsistent V0 copies: {task}")
    source_manifest = _read_json_object(
        source / "manifest.json",
        label="corrected Control manifest",
        max_bytes=MAX_PRESEED_FILE_BYTES,
    )
    if source_manifest.get("initial_verifier_origin") != SHIPPED_VERIFIER_ORIGIN:
        raise ValueError(f"corrected Control is not shipped-V0 anchored: {task}")
    policy = _read_json_object(
        bws / "reframe_policy.json", label="corrected Control reframe policy"
    )
    if policy.get("admits_proof") is not False:
        raise ValueError(f"corrected Control unexpectedly admits proof: {task}")
    try:
        brief = bootstrap_raw["bootstrap_ws/SOLVER_BRIEF.md"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"corrected Control solver brief is not UTF-8: {task}") from exc
    rewritten_brief = _rewrite_solver_brief(brief, task=task)
    checker_inventory = _tree_inventory(source / "checker")
    test_sha256 = checker_inventory["tests/test.sh"]
    v0_sha256 = _sha256_bytes(v0_raw)
    if source_manifest.get("initial_verifier_sha256") != v0_sha256:
        raise ValueError(f"corrected Control manifest V0 hash mismatch: {task}")
    recorded_test_sha = source_manifest.get("original_test_sha256")
    if recorded_test_sha != test_sha256:
        raise ValueError(f"corrected Control shipped test hash mismatch: {task}")
    tracked_test = problem / "tests" / "test.sh"
    tracked_test_raw = _read_bounded_regular(
        tracked_test,
        max_bytes=MAX_CHECKER_FILE_BYTES,
        label="tracked AutoLab test",
    )
    if _sha256_bytes(tracked_test_raw) != test_sha256:
        raise ValueError(f"corrected Control test does not match tracked source: {task}")
    return {
        "source": source,
        "problem": problem.resolve(),
        "source_manifest": source_manifest,
        "rewritten_brief": rewritten_brief,
        "checker_inventory": checker_inventory,
        "bootstrap_raw": bootstrap_raw,
        "test_sha256": test_sha256,
        "v0_sha256": v0_sha256,
        "seed_sha256": _sha256_bytes(
            bootstrap_raw["bootstrap_ws/seed_solution.json"]
        ),
    }


def _write_preseeded_run(
    destination: Path, *, task: str, source_info: dict[str, Any]
) -> None:
    source = Path(source_info["source"])
    bws = destination / "bootstrap_ws"
    vdir = destination / "supervisor" / "verifier_versions"
    bws.mkdir(parents=True)
    vdir.mkdir(parents=True)
    checker_destination = destination / "checker"
    checker_destination.mkdir()
    for relative, expected_sha256 in source_info["checker_inventory"].items():
        raw = _read_bounded_regular(
            source / "checker" / relative,
            max_bytes=MAX_CHECKER_FILE_BYTES,
            label="checker file",
        )
        if _sha256_bytes(raw) != expected_sha256:
            raise ValueError(f"checker changed during bounded copy: {relative}")
        target = checker_destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
    for name in RUN_BOOTSTRAP_FILES:
        raw = source_info["bootstrap_raw"][f"bootstrap_ws/{name}"]
        with (bws / name).open("xb") as stream:
            stream.write(raw)
    (bws / "SOLVER_BRIEF.md").write_text(
        source_info["rewritten_brief"], encoding="utf-8"
    )
    with (vdir / "v0.py").open("xb") as stream:
        stream.write(
            source_info["bootstrap_raw"]["supervisor/verifier_versions/v0.py"]
        )
    source_run_name = source.name
    manifest = {
        "schema_version": 1,
        "arm": ARM,
        "mode": "agent_system",
        "raw_input_dir": str(source_info["problem"]),
        "budget_s": 14_400.0,
        "initial_feedback_level": "with_artifacts",
        "initial_verifier_origin": SHIPPED_VERIFIER_ORIGIN,
        "initial_verifier_sha256": source_info["v0_sha256"],
        "original_test_sha256": source_info["test_sha256"],
        "source_contract_run": source_run_name,
        "solver_brief_origin": "autolab_instruction_plus_evolving_shipped_test_contract",
        "solver_brief_sha256": _sha256_bytes(
            source_info["rewritten_brief"].encode("utf-8")
        ),
        "seed_solution_sha256": source_info["seed_sha256"],
        "checker_inventory": source_info["checker_inventory"],
        "freeze_verifier": False,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "events.jsonl").write_text(
        json.dumps(
            {
                "t": 0.0,
                "kind": "shipped_verifier_preseed",
                "task": task,
                "original_test_sha256": source_info["test_sha256"],
                "source_control_run": source_run_name,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (destination / "supervisor" / "versions.jsonl").write_text(
        json.dumps(
            {
                "t": 0.0,
                "version": 0,
                "origin": SHIPPED_VERIFIER_ORIGIN,
                "note": "mechanical path/payload adapter; original reward unchanged",
                "rationale": "shipped weak V0; Co-Evolve hardening enabled",
                "has_feedback": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _assert_no_private_context(
    run: Path,
    context_dir: Path,
    contexts: tuple[str, ...],
    *,
    trusted_preseed_files: set[Path],
) -> None:
    from coscientist.coevo.human_proxy_sessions import (
        HUMAN_PROXY_E_PRIVATE_CONTEXT_LEAK,
        _normalized_private_text,
        assert_no_private_context_leak,
    )

    forbidden_paths = tuple({
        str(context_dir),
        str(context_dir.resolve()),
        *(str(path) for path in context_dir.glob("autolab_*.md")),
        *(str(path.resolve()) for path in context_dir.glob("autolab_*.md")),
    })
    forbidden_path_bytes = tuple(path.encode("utf-8") for path in forbidden_paths)
    normalized_forbidden_paths = tuple(
        value
        for path in forbidden_paths
        if (value := _normalized_private_text(path))
    )
    window_owners: dict[str, set[int]] = {}
    raw_private_windows: set[bytes] = set()
    for context_index, private_context in enumerate(contexts):
        private_raw = private_context.encode("utf-8")
        raw_private_windows.update(
            private_raw[start : start + 240]
            for start in range(max(0, len(private_raw) - 240 + 1))
        )
        normalized_context = _normalized_private_text(private_context)
        for start in range(max(0, len(normalized_context) - 240 + 1)):
            window = normalized_context[start : start + 240]
            window_owners.setdefault(window, set()).add(context_index)
    for path in run.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"prelaunch audit found a symlink below run: {path}")
        if not path.is_file():
            continue
        try:
            raw = _read_bounded_regular(
                path,
                max_bytes=MAX_RUN_AUDIT_FILE_BYTES,
                label="prelaunch audit file",
            )
        except ValueError as exc:
            raise RuntimeError(f"prelaunch audit file failed bounded read: {path}") from exc
        if any(value and value in raw for value in forbidden_path_bytes):
            raise RuntimeError(f"private context path leaked below run: {path}")
        raw_private_match = any(
            raw[start : start + 240] in raw_private_windows
            for start in range(max(0, len(raw) - 240 + 1))
        )
        if raw_private_match and path not in trusted_preseed_files:
            raise RuntimeError(f"private context raw bytes leaked below run: {path}")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            if path in trusted_preseed_files:
                continue
            raise RuntimeError(
                f"prelaunch audit found an untrusted non-UTF8 file: {path}"
            ) from None
        normalized_text = _normalized_private_text(text)
        if any(literal in normalized_text for literal in normalized_forbidden_paths):
            raise RuntimeError(f"private context path leaked below run: {path}")
        candidate_contexts: set[int] = set()
        for start in range(max(0, len(normalized_text) - 240 + 1)):
            owners = window_owners.get(normalized_text[start : start + 240])
            if owners:
                candidate_contexts.update(owners)
        for context_index in candidate_contexts:
            try:
                assert_no_private_context_leak(
                    text,
                    contexts[context_index],
                    min_span_characters=240,
                    forbidden_literals=forbidden_paths,
                )
            except RuntimeError as exc:
                if str(exc) == HUMAN_PROXY_E_PRIVATE_CONTEXT_LEAK:
                    if path in trusted_preseed_files:
                        # Some accepted Final Evaluators quote code from the public,
                        # shipped checker.  This file is byte-bound to corrected
                        # Control above, so the overlap predates Human Proxy context
                        # injection and is not an escaped private artifact.
                        continue
                    raise RuntimeError(
                        f"private context content leaked below run: {path}"
                    ) from None
                raise


def _assert_pristine_prelaunch_state(run: Path, manifest: dict[str, Any]) -> None:
    runtime_paths = (
        run / "cost.jsonl",
        run / "solver" / "trajectory.jsonl",
        run / "eval" / "queries.jsonl",
        run / "supervisor" / "reviews.jsonl",
        run / "human",
        run / "solver_ws",
        run / "result.json",
        run / "final_result.json",
        run / "run_result.json",
        run / "outcome.json",
        run / "pending_outcome.json",
        run / "human_guidance.md",
        run / "HUMAN_GUIDANCE.md",
    )
    if any(path.exists() for path in runtime_paths):
        raise RuntimeError("prelaunch runtime contamination detected")
    if any((run / "solver" / "candidates").glob("*")):
        raise RuntimeError("prelaunch candidate contamination detected")
    if any((run / "supervisor" / "probes").glob("*")):
        raise RuntimeError("prelaunch runtime contamination detected")
    version_files = sorted(
        path.name
        for path in (run / "supervisor" / "verifier_versions").glob("v*.py")
    )
    if version_files != ["v0.py"]:
        raise RuntimeError("prelaunch verifier-version contamination detected")
    forbidden_manifest_fields = {
        "best_score",
        "best_solution",
        "final_verifier_version",
        "verifier_hardenings",
        "reviews_handled",
        "post_harden_solves",
        "final_mode",
        "stop_epoch",
        "outcome",
    }
    contaminated_fields = forbidden_manifest_fields.intersection(manifest)
    if contaminated_fields:
        raise RuntimeError("prelaunch final/best manifest contamination detected")


def audit_runs(
    control_runs_root: Path | str,
    output_runs_root: Path | str,
    batch_name: str,
    *,
    problems_root: Path | str,
    context_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Audit the 17 prepared resume boundaries against corrected shipped-V0 Control."""

    batch_name = _validated_batch_name(batch_name)
    control_runs_root = _reject_symlink_components(control_runs_root)
    output_runs_root = _reject_symlink_components(output_runs_root)
    problems_root = _reject_symlink_components(problems_root)
    context_path = (
        _reject_symlink_components(context_dir) if context_dir is not None else None
    )
    private_contexts: tuple[str, ...] = ()
    if context_path is not None:
        if not context_path.is_dir():
            raise RuntimeError(f"private context directory is missing: {context_path}")
        context_texts = []
        context_total_bytes = 0
        for task in _tasks():
            context_file = context_path / f"autolab_{task}.md"
            try:
                raw = _read_bounded_regular(
                    context_file,
                    max_bytes=MAX_RUN_CONTEXT_FILE_BYTES,
                    label="private context",
                )
                context_total_bytes += len(raw)
                if context_total_bytes > MAX_RUN_CONTEXT_TOTAL_BYTES:
                    raise RuntimeError("private context total exceeds size limit")
                text = raw.decode("utf-8")
            except ValueError as exc:
                raise RuntimeError("private context size/read validation failed") from exc
            except UnicodeDecodeError as exc:
                raise RuntimeError("private context is not valid UTF-8") from exc
            if len(text) > MAX_CONTEXT_CHARACTERS:
                raise RuntimeError(
                    f"private context exceeds {MAX_CONTEXT_CHARACTERS} characters"
                )
            context_texts.append(text)
        private_contexts = tuple(context_texts)

    for task in _tasks():
        source_info = _validate_control_source(control_runs_root, problems_root, task)
        source = Path(source_info["source"])
        run = _reject_symlink_components(
            _prepared_run(output_runs_root, batch_name, task)
        )
        if not run.is_dir():
            raise RuntimeError(f"prepared run is missing or unsafe: {run}")
        expected_pairs = [
            (
                source_info["bootstrap_raw"][f"bootstrap_ws/{name}"],
                run / "bootstrap_ws" / name,
            )
            for name in RUN_BOOTSTRAP_FILES
        ] + [
            (
                source_info["bootstrap_raw"]["supervisor/verifier_versions/v0.py"],
                run / "supervisor" / "verifier_versions" / "v0.py",
            ),
        ]
        for expected_raw, actual in expected_pairs:
            try:
                actual_raw = _read_bounded_regular(
                    actual,
                    max_bytes=MAX_PRESEED_FILE_BYTES,
                    label="prepared initial state",
                )
            except ValueError as exc:
                raise RuntimeError(f"prepared initial-state mismatch: {actual}") from exc
            if actual_raw != expected_raw:
                raise RuntimeError(f"prepared initial-state mismatch: {actual}")
        trusted_preseed_files = {actual for _expected, actual in expected_pairs}
        if _tree_inventory(run / "checker") != source_info["checker_inventory"]:
            raise RuntimeError(f"prepared checker mismatch: {task}")
        trusted_preseed_files.update(
            path for path in (run / "checker").rglob("*") if path.is_file()
        )
        expected_brief = source_info["rewritten_brief"]
        brief_path = run / "bootstrap_ws" / "SOLVER_BRIEF.md"
        try:
            actual_brief = _read_bounded_regular(
                brief_path,
                max_bytes=MAX_PRESEED_FILE_BYTES,
                label="prepared solver brief",
            ).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"prepared evolving solver brief mismatch: {task}") from exc
        if actual_brief != expected_brief:
            raise RuntimeError(f"prepared evolving solver brief mismatch: {task}")
        trusted_preseed_files.add(brief_path)
        policy = _read_json_object(
            run / "bootstrap_ws" / "reframe_policy.json",
            label="prepared reframe policy",
        )
        if policy.get("admits_proof") is not False:
            raise RuntimeError(f"prepared run unexpectedly admits proof: {task}")
        manifest = _read_json_object(
            run / "manifest.json", label="prepared run manifest"
        )
        critical_manifest = {
            "arm": ARM,
            "raw_input_dir": str(source_info["problem"]),
            "budget_s": 14_400.0,
            "initial_feedback_level": "with_artifacts",
            "initial_verifier_origin": SHIPPED_VERIFIER_ORIGIN,
            "initial_verifier_sha256": source_info["v0_sha256"],
            "original_test_sha256": source_info["test_sha256"],
            "source_contract_run": source.name,
            "solver_brief_sha256": _sha256_bytes(expected_brief.encode("utf-8")),
            "seed_solution_sha256": source_info["seed_sha256"],
            "checker_inventory": source_info["checker_inventory"],
            "freeze_verifier": False,
        }
        for key, expected in critical_manifest.items():
            if manifest.get(key) != expected:
                raise RuntimeError(f"prepared manifest mismatch for {task}: {key}")
        _assert_pristine_prelaunch_state(run, manifest)
        try:
            events = [
                json.loads(line)
                for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"prepared events are invalid: {task}") from exc
        if len(events) != 1 or events[0].get("kind") != "shipped_verifier_preseed":
            raise RuntimeError(f"prelaunch bootstrap/events contamination: {task}")
        if any(event.get("kind") in {"bootstrap_start", "bootstrap_done"} for event in events):
            raise RuntimeError(f"prepared run contains a bootstrap event: {task}")
        from coscientist.coevo.agent_system import AgentSystem

        if not AgentSystem(
            raw_input_dir=source_info["problem"], run_dir=run, budget_s=14_400.0
        ).can_resume():
            raise RuntimeError(f"prepared run cannot resume: {task}")
        if context_path is not None:
            _assert_no_private_context(
                run,
                context_path,
                private_contexts,
                trusted_preseed_files=trusted_preseed_files,
            )

    return {
        "arm": ARM,
        "batch_name": batch_name,
        "status": "pass",
        "task_count": 17,
    }


def prepare_runs(
    control_runs_root: Path | str,
    output_runs_root: Path | str,
    batch_name: str,
    *,
    problems_root: Path | str,
) -> dict[str, Any]:
    """Create 17 fresh, evolvable resume boundaries from corrected Control V0."""

    batch_name = _validated_batch_name(batch_name)
    control_runs_root = _reject_symlink_components(control_runs_root)
    output_runs_root = _reject_symlink_components(output_runs_root)
    problems_root = _reject_symlink_components(problems_root)
    if os.path.lexists(output_runs_root):
        raise RuntimeError(
            f"refusing to overwrite existing dedicated runs root: {output_runs_root}"
        )
    output_runs_root.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = None
    try:
        source_inventory = {
            task: _validate_control_source(control_runs_root, problems_root, task)
            for task in _tasks()
        }
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{batch_name}.runs.tmp.", dir=output_runs_root.parent
            )
        )
        for task, source_info in source_inventory.items():
            _write_preseeded_run(
                _prepared_run(staging, batch_name, task),
                task=task,
                source_info=source_info,
            )
        audit_runs(
            control_runs_root,
            staging,
            batch_name,
            problems_root=problems_root,
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


def _load_live_private_contexts(
    context_dir: Path,
) -> dict[str, tuple[Path, str, str]]:
    if not context_dir.is_dir():
        raise RuntimeError("private context directory is missing")
    contexts: dict[str, tuple[Path, str, str]] = {}
    total_bytes = 0
    for task in _tasks():
        path = _reject_symlink_components(context_dir / f"autolab_{task}.md")
        try:
            raw = _read_bounded_regular(
                path,
                max_bytes=MAX_RUN_CONTEXT_FILE_BYTES,
                label="private context",
            )
            total_bytes += len(raw)
            if total_bytes > MAX_RUN_CONTEXT_TOTAL_BYTES:
                raise RuntimeError("private context total exceeds size limit")
            text = raw.decode("utf-8")
        except ValueError as exc:
            raise RuntimeError("private context size/read validation failed") from exc
        except UnicodeDecodeError as exc:
            raise RuntimeError("private context is not valid UTF-8") from exc
        if len(text) > MAX_CONTEXT_CHARACTERS:
            raise RuntimeError("private context exceeds character limit")
        contexts[task] = (path, text, _sha256_bytes(raw))
    return contexts


def _live_privacy_files(run: Path) -> list[Path]:
    """Return only durable Human-derived surfaces, never the private batch manifest."""

    candidates: dict[tuple[int, int], Path] = {}

    def add(path: Path) -> None:
        safe = _reject_symlink_components(path)
        info = os.stat(safe, follow_symlinks=False)
        candidates[(info.st_dev, info.st_ino)] = safe

    human = run / "human"
    if os.path.lexists(human):
        _reject_symlink_components(human)
        if not human.is_dir():
            raise RuntimeError("live Human artifact root is not a directory")
        for path in human.rglob("*"):
            _reject_symlink_components(path)
            mode = os.lstat(path).st_mode
            if stat.S_ISREG(mode):
                add(path)
            elif not stat.S_ISDIR(mode):
                raise RuntimeError("live Human artifacts contain an unsafe entry")
    guidance = run / "human_guidance.md"
    if os.path.lexists(guidance):
        add(guidance)
    for path in run.rglob("HUMAN_GUIDANCE.md"):
        add(path)
    for path in candidates.values():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("live Human artifact is not a regular file")
    return sorted(candidates.values())


def _live_privacy_payload(path: Path, raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("live Human artifact is not valid UTF-8") from exc
    try:
        if path.suffix == ".json":
            return json.loads(text)
        if path.suffix == ".jsonl":
            return [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
    except json.JSONDecodeError as exc:
        raise RuntimeError("live Human artifact contains invalid JSON") from exc
    return text


def audit_live_privacy(
    *,
    runs_root: Path | str,
    batch_name: str,
    context_dir: Path | str,
    require_closed_tasks: int = 0,
) -> dict[str, Any]:
    """Audit persisted Human Proxy output after launch without reading solver payloads.

    The launcher manifest necessarily contains private context paths and hashes, so it
    is validated as provenance but deliberately excluded from the content-leak scan.
    Only durable transcripts, outcomes, guidance, and their workspace copies are
    scanned with the same normalized/encoded detector used at the Proxy boundary.
    """

    if not 0 <= require_closed_tasks <= 17:
        raise ValueError("require_closed_tasks must be between 0 and 17")
    batch_name = _validated_batch_name(batch_name)
    runs_root = _reject_symlink_components(runs_root)
    context_dir = _reject_symlink_components(context_dir)
    if not runs_root.is_dir():
        raise RuntimeError("dedicated runs root is missing")
    try:
        context_dir.resolve().relative_to(runs_root.resolve())
    except ValueError:
        pass
    else:
        raise RuntimeError("private context directory must be outside the runs root")
    contexts = _load_live_private_contexts(context_dir)

    batch_path = runs_root / batch_name / "batch.json"
    batch, batch_sha256 = _read_json_snapshot(
        batch_path,
        label="live batch manifest",
        max_bytes=MAX_RUN_AUDIT_FILE_BYTES,
    )
    if batch.get("batch") != batch_name:
        raise RuntimeError("live batch manifest names a different batch")
    if batch.get("human_proxy_context_dir") != str(context_dir.resolve()):
        raise RuntimeError("live batch context root mismatch")
    rows = batch.get("runs")
    if not isinstance(rows, list) or len(rows) != 17:
        raise RuntimeError("live batch must contain exactly 17 runs")
    expected_ids = {f"{batch_name}__autolab_{task}" for task in _tasks()}
    row_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("run_id"), str):
            raise RuntimeError("live batch contains an invalid run record")
        if row["run_id"] in row_by_id:
            raise RuntimeError("live batch contains duplicate run records")
        row_by_id[row["run_id"]] = row
    if set(row_by_id) != expected_ids:
        raise RuntimeError("live batch task set does not match the exact 17 tasks")

    from coscientist.coevo.human_proxy_sessions import (
        HUMAN_PROXY_E_PRIVATE_CONTEXT_LEAK,
        assert_no_private_context_leak,
    )

    forbidden_paths = tuple(
        {
            str(context_dir),
            str(context_dir.resolve()),
            *(str(path) for path, _text, _sha in contexts.values()),
            *(str(path.resolve()) for path, _text, _sha in contexts.values()),
        }
    )
    scanned_files = 0
    closed_tasks = 0
    for task in _tasks():
        run_id = f"{batch_name}__autolab_{task}"
        row = row_by_id[run_id]
        context_path, private_context, context_sha256 = contexts[task]
        if Path(str(row.get("human_proxy_context", ""))) != context_path.resolve():
            raise RuntimeError(f"live batch context mapping mismatch for {task}")
        if row.get("human_proxy_context_sha256") != context_sha256:
            raise RuntimeError(f"live batch context hash mismatch for {task}")
        if Path(str(row.get("input_dir", ""))).name != f"autolab_{task}":
            raise RuntimeError(f"live batch input mapping mismatch for {task}")
        run = _reject_symlink_components(runs_root / run_id)
        if not run.is_dir():
            raise RuntimeError(f"live run is missing for {task}")

        task_has_closed_session = False
        for path in _live_privacy_files(run):
            try:
                raw = _read_bounded_regular(
                    path,
                    max_bytes=MAX_RUN_AUDIT_FILE_BYTES,
                    label="live Human artifact",
                )
            except ValueError as exc:
                raise RuntimeError("live Human artifact failed bounded read") from exc
            payload = _live_privacy_payload(path, raw)
            try:
                assert_no_private_context_leak(
                    payload,
                    private_context,
                    min_span_characters=160,
                    forbidden_literals=forbidden_paths,
                )
            except RuntimeError as exc:
                if str(exc) == HUMAN_PROXY_E_PRIVATE_CONTEXT_LEAK:
                    relative = path.relative_to(run).as_posix()
                    raise RuntimeError(
                        f"private context content leaked in live artifact: {task}/{relative}"
                    ) from None
                raise
            scanned_files += 1
            if path.name == "session.json" and isinstance(payload, dict):
                task_has_closed_session |= payload.get("state") == "closed"
        closed_tasks += int(task_has_closed_session)

    if closed_tasks < require_closed_tasks:
        raise RuntimeError(
            f"only {closed_tasks} tasks have a closed Human session; "
            f"required {require_closed_tasks}"
        )
    return {
        "arm": ARM,
        "batch_name": batch_name,
        "status": "pass",
        "task_count": 17,
        "closed_session_tasks": closed_tasks,
        "required_closed_session_tasks": require_closed_tasks,
        "scanned_files": scanned_files,
        "batch_manifest_sha256": batch_sha256,
    }


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return _sha256_bytes(raw)


def _replay_seed(
    *, hidden_seed_file: Path | str | None, audit_seed: str | None
) -> tuple[str, dict[str, Any]]:
    if (hidden_seed_file is None) == (audit_seed is None):
        raise ValueError(
            "choose exactly one hidden-seed mode: --hidden-seed-file or --audit-seed"
        )
    if audit_seed is not None:
        if audit_seed != FINAL_REPLAY_AUDIT_SEED:
            raise ValueError(
                "--audit-seed must use the explicit fixed audit-only replay seed"
            )
        return audit_seed, {
            "mode": "fixed_audit_only",
            "value": audit_seed,
            "classification": "audit-only; MUST NOT deploy",
            "persist_across_repeats": True,
            "disclosed": True,
        }

    raw = _read_bounded_regular(
        Path(hidden_seed_file), max_bytes=16_384, label="private hidden seed"
    )
    try:
        seed = raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("private hidden seed is not valid UTF-8") from exc
    if not seed or seed == "REQUIRED_PRIVATE":
        raise ValueError("private hidden seed file is empty or still a placeholder")
    return seed, {
        "mode": "private_file",
        "sha256": _sha256_bytes(seed.encode("utf-8")),
        "persist_across_repeats": True,
        "disclosed": False,
    }


def _inspect_immutable_image(
    image: str, *, run_command: Any
) -> dict[str, Any]:
    proc = run_command(
        ["docker", "image", "inspect", image, "--format", "{{json .}}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker image inspect failed for {image}: {(proc.stderr or '')[-2000:]}"
        )
    try:
        inspected = json.loads(proc.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"docker image inspect returned invalid JSON for {image}") from exc
    if isinstance(inspected, list) and len(inspected) == 1:
        inspected = inspected[0]
    if not isinstance(inspected, dict):
        raise RuntimeError(f"docker image inspect returned ambiguous data for {image}")
    immutable_id = inspected.get("Id")
    if not isinstance(immutable_id, str) or re.fullmatch(
        r"sha256:[0-9a-f]{64}", immutable_id
    ) is None:
        raise RuntimeError(f"docker image lacks an immutable sha256 ID: {image}")
    return {
        "reference": image,
        "id": immutable_id,
        "repo_digests": inspected.get("RepoDigests") or [],
        "created": inspected.get("Created"),
    }


def _read_json_snapshot(
    path: Path, *, label: str, max_bytes: int
) -> tuple[dict[str, Any], str]:
    raw = _read_bounded_regular(path, max_bytes=max_bytes, label=label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not a valid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return value, _sha256_bytes(raw)


def _read_replay_resource(manifest_path: Path) -> tuple[dict[str, Any], str]:
    manifest, manifest_sha256 = _read_json_snapshot(
        manifest_path,
        label="trusted reference manifest",
        max_bytes=MAX_RUN_AUDIT_FILE_BYTES,
    )
    resource = (manifest.get("resource_spec") or {}).get("verifier")
    if not isinstance(resource, dict):
        raise ValueError(f"trusted verifier resource is missing: {manifest_path}")
    required = ("image", "cpus", "memory_mb", "timeout_sec")
    if any(resource.get(key) is None for key in required):
        raise ValueError(f"trusted verifier resource is incomplete: {manifest_path}")
    if not isinstance(resource["image"], str) or not resource["image"].strip():
        raise ValueError(f"trusted verifier image is invalid: {manifest_path}")
    for key in ("cpus", "memory_mb", "timeout_sec"):
        if isinstance(resource[key], bool) or not isinstance(resource[key], (int, float)):
            raise ValueError(f"trusted verifier resource {key} is invalid: {manifest_path}")
        if float(resource[key]) <= 0:
            raise ValueError(f"trusted verifier resource {key} must be positive")
    if resource.get("allow_internet") is not False:
        raise ValueError("trusted verifier resource must explicitly disable internet")
    normalized = {
        "image": resource["image"],
        "cpus": float(resource["cpus"]),
        "memory_mb": int(resource["memory_mb"]),
        "timeout_sec": float(resource["timeout_sec"]),
        "allow_internet": False,
    }
    return normalized, manifest_sha256


def _accepted_replay_contracts(accepted_root: Path) -> tuple[str, dict[tuple[str, str], dict[str, Any]]]:
    report_path = accepted_root / "acceptance_report.json"
    report_raw, report_text = _read_source(report_path, label="acceptance report")
    try:
        report = json.loads(report_text)
    except json.JSONDecodeError as exc:
        raise ValueError("acceptance report is invalid JSON") from exc
    if not isinstance(report, dict):
        raise ValueError("acceptance report must be a JSON object")
    records = _accepted_records(report)
    return _sha256_bytes(report_raw), records


def _build_replay_contracts(
    *,
    batch_name: str,
    runs_root: Path | str,
    accepted_root: Path | str,
    trusted_runs_root: Path | str,
    dry_run: bool,
    run_command: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    batch_name = _validated_batch_name(batch_name)
    runs_root = _reject_symlink_components(runs_root)
    accepted_root = _validated_accepted_root(accepted_root)
    trusted_runs_root = _reject_symlink_components(trusted_runs_root)
    if not runs_root.is_dir() or not trusted_runs_root.is_dir():
        raise ValueError("replay runs roots must be existing non-symlinked directories")

    expected_run_ids = {
        f"{batch_name}__autolab_{task}" for task in _tasks()
    }
    prefixed_entries = {
        path.name
        for path in runs_root.iterdir()
        if path.name.startswith(f"{batch_name}__autolab_")
    }
    if prefixed_entries != expected_run_ids:
        raise ValueError(
            "dedicated target runs root does not contain the exact 17 unambiguous task runs"
        )
    for run_id in expected_run_ids:
        run = runs_root / run_id
        if run.is_symlink() or not run.is_dir():
            raise ValueError(f"target run is missing or unsafe: {run_id}")

    batch_path = runs_root / batch_name / "batch.json"
    batch, batch_sha256 = _read_json_snapshot(
        batch_path,
        label="target batch manifest",
        max_bytes=MAX_RUN_AUDIT_FILE_BYTES,
    )
    if batch.get("batch") != batch_name:
        raise ValueError("target batch manifest names a different batch")
    rows = batch.get("runs")
    if not isinstance(rows, list) or len(rows) != 17:
        raise ValueError("target batch manifest must contain exactly 17 runs")
    row_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("run_id"), str):
            raise ValueError("target batch contains an invalid run record")
        if row["run_id"] in row_by_id:
            raise ValueError(f"target batch contains an ambiguous run: {row['run_id']}")
        row_by_id[row["run_id"]] = row
    if set(row_by_id) != expected_run_ids:
        raise ValueError("target batch task set does not match the exact 17 tasks")

    report_sha256, accepted_records = _accepted_replay_contracts(accepted_root)
    contracts: list[dict[str, Any]] = []
    for group, tasks in AUTOLAB_GROUPS.items():
        checker_batch = TRUSTED_REPLAY_BATCHES[group]
        for task in tasks:
            run_id = f"{batch_name}__autolab_{task}"
            row = row_by_id[run_id]
            input_name = Path(str(row.get("input_dir", ""))).name
            if input_name != f"autolab_{task}":
                raise ValueError(f"target batch input mapping mismatch for {task}")
            if not dry_run and row.get("status") != "done":
                raise RuntimeError(f"target run is not complete/done: {task}")

            run = runs_root / run_id
            manifest_path = run / "manifest.json"
            manifest, manifest_sha256 = _read_json_snapshot(
                manifest_path,
                label="target run manifest",
                max_bytes=MAX_RUN_AUDIT_FILE_BYTES,
            )
            if manifest.get("arm") != ARM:
                raise ValueError(f"target run arm mismatch for {task}")
            if manifest.get("freeze_verifier") is not False:
                raise ValueError(f"target run is a frozen-control contract: {task}")
            if manifest.get("initial_verifier_origin") != SHIPPED_VERIFIER_ORIGIN:
                raise ValueError(f"target run did not begin from shipped V0: {task}")
            best = manifest.get("best_solution")
            if not dry_run and not isinstance(best, dict):
                raise RuntimeError(f"target run lacks a final best_solution payload: {task}")
            if not dry_run:
                events_path = run / "events.jsonl"
                raw_events = _read_bounded_regular(
                    events_path, max_bytes=MAX_RUN_AUDIT_FILE_BYTES,
                    label="target run events"
                )
                try:
                    events = [json.loads(line) for line in raw_events.decode("utf-8").splitlines() if line.strip()]
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(f"target run events are invalid: {task}") from exc
                if not any(event.get("kind") == "run_stop" for event in events):
                    raise RuntimeError(f"target run has no run_stop completion event: {task}")

            accepted_record = accepted_records[(group, task)]
            package = accepted_root / group / task
            verifier = package / "final_verifier.py"
            template = package / "final_ctx.template.json"
            verifier_sha = _sha256_path(
                verifier, max_bytes=MAX_ACCEPTED_SOURCE_BYTES,
                label="accepted final verifier"
            )
            template_raw = _read_bounded_regular(
                template, max_bytes=MAX_ACCEPTED_SOURCE_BYTES,
                label="accepted final context template"
            )
            template_sha = _sha256_bytes(template_raw)
            if verifier_sha != accepted_record["final_verifier_sha256"]:
                raise ValueError(f"accepted final verifier hash mismatch for {task}")
            if template_sha != accepted_record["final_ctx_template_sha256"]:
                raise ValueError(f"accepted final context template hash mismatch for {task}")
            try:
                template_value = json.loads(template_raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"accepted final context template is invalid for {task}") from exc
            if not isinstance(template_value, dict) or not {
                "checker_dir", "hidden_seed"
            }.issubset(template_value):
                raise ValueError(f"accepted final context template is incomplete for {task}")

            trusted_run = trusted_runs_root / f"{checker_batch}__autolab_{task}"
            if trusted_run.is_symlink() or not trusted_run.is_dir():
                raise ValueError(f"trusted checker source run is missing for {task}")
            checker = trusted_run / "checker"
            checker_inventory = _tree_inventory(checker)
            trusted_manifest = trusted_run / "manifest.json"
            resource, trusted_manifest_sha = _read_replay_resource(trusted_manifest)
            image = _inspect_immutable_image(resource["image"], run_command=run_command)
            contracts.append(
                {
                    "task": task,
                    "group": group,
                    "run": run,
                    "run_id": run_id,
                    "batch_row": row,
                    "manifest": manifest,
                    "manifest_path": manifest_path,
                    "manifest_sha256": manifest_sha256,
                    "best": best if isinstance(best, dict) else None,
                    "package": package,
                    "verifier": verifier,
                    "template": template,
                    "verifier_sha256": verifier_sha,
                    "template_sha256": template_sha,
                    "checker": checker,
                    "checker_batch": checker_batch,
                    "checker_inventory": checker_inventory,
                    "trusted_manifest": trusted_manifest,
                    "trusted_manifest_sha256": trusted_manifest_sha,
                    "resource": resource,
                    "image": image,
                }
            )
    return {
        "path": str(batch_path),
        "sha256": batch_sha256,
        "accepted_report_path": str(accepted_root / "acceptance_report.json"),
        "accepted_report_sha256": report_sha256,
    }, contracts


_FINAL_REPLAY_WRAPPER = r'''import importlib.util,json,time
spec=importlib.util.spec_from_file_location("final_verifier","/review/final_verifier.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
payload=json.load(open("/input/payload.json"))
template=json.load(open("/review/final_ctx.template.json"))
runtime=json.load(open("/input/runtime_ctx.json"))
if not isinstance(template,dict) or "checker_dir" not in template or "hidden_seed" not in template:
    raise RuntimeError("invalid final context template")
ctx=dict(template);ctx.update(runtime)
started=time.time();result=module.verify(payload,ctx);finished=time.time()
print(json.dumps({"result":result,"duration_seconds":finished-started},sort_keys=True))
'''


def _docker_command(
    contract: dict[str, Any], *, payload_path: str, runtime_path: str
) -> list[str]:
    resource = contract["resource"]
    return [
        "docker", "run", "--rm", "--network", "none",
        "--cpus", str(resource["cpus"]),
        "--memory", f"{resource['memory_mb']}m",
        "-v", f"{contract['verifier']}:/review/final_verifier.py:ro",
        "-v", f"{contract['template']}:/review/final_ctx.template.json:ro",
        "-v", f"{payload_path}:/input/payload.json:ro",
        "-v", f"{runtime_path}:/input/runtime_ctx.json:ro",
        "-v", f"{contract['checker']}:/checker:ro",
        contract["image"]["id"], "python3", "-c", _FINAL_REPLAY_WRAPPER,
    ]


def _parse_replay_result(output: str) -> dict[str, Any] | None:
    for line in reversed([line for line in output.splitlines() if line.strip()]):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("result"), dict):
            return value
    return None


def _task_report_base(contract: dict[str, Any]) -> dict[str, Any]:
    best = contract["best"]
    payload = (
        {"mapping": "identity", "sha256": _canonical_sha256(best), "keys": sorted(best)}
        if isinstance(best, dict)
        else {"mapping": "identity", "status": "not_available"}
    )
    placeholder_command = _docker_command(
        contract,
        payload_path="<private-temp>/payload.json",
        runtime_path="<private-temp>/runtime_ctx.json",
    )
    return {
        "task": contract["task"],
        "group": contract["group"],
        "run": {
            "run_id": contract["run_id"],
            "manifest_path": str(contract["manifest_path"]),
            "manifest_sha256": contract["manifest_sha256"],
            "original_best_score": contract["manifest"].get("best_score"),
            "final_verifier_version": contract["manifest"].get("final_verifier_version"),
            "verifier_hardenings": contract["manifest"].get("verifier_hardenings"),
            "freeze_verifier": False,
        },
        "payload": payload,
        "accepted_evaluator": {
            "package_path": str(contract["package"]),
            "verifier_sha256": contract["verifier_sha256"],
            "ctx_template_sha256": contract["template_sha256"],
        },
        "checker": {
            "path": str(contract["checker"]),
            "source_batch": contract["checker_batch"],
            "inventory": contract["checker_inventory"],
            "trusted_manifest_path": str(contract["trusted_manifest"]),
            "trusted_manifest_sha256": contract["trusted_manifest_sha256"],
        },
        "metric": {
            "direction": "higher_is_better",
            "native_meaning": NATIVE_METRICS[contract["task"]],
        },
        "resource": contract["resource"],
        "docker": {
            "network": "none",
            "mounts": "read_only",
            "image_reference": contract["image"]["reference"],
            "image_used": contract["image"]["id"],
            "image_identity": contract["image"],
            "timeout_seconds": contract["resource"]["timeout_sec"] + 120.0,
            "command_template": placeholder_command,
        },
    }


def _replay_task(
    contract: dict[str, Any], *, repeat: int, hidden_seed: str, run_command: Any
) -> dict[str, Any]:
    base = _task_report_base(contract)
    best = contract["best"]
    if not isinstance(best, dict):
        raise RuntimeError(f"target run lacks a final best_solution payload: {contract['task']}")
    rows = []
    with tempfile.TemporaryDirectory(prefix=f"autolab-final-{contract['task']}-") as temp:
        payload_path = Path(temp) / "payload.json"
        runtime_path = Path(temp) / "runtime_ctx.json"
        payload_path.write_text(
            json.dumps(best, ensure_ascii=False), encoding="utf-8"
        )
        runtime_path.write_text(
            json.dumps(
                {
                    "checker_dir": "/checker",
                    "hidden_seed": hidden_seed,
                    "validation_mode": False,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.chmod(payload_path, 0o600)
        os.chmod(runtime_path, 0o600)
        command = _docker_command(
            contract, payload_path=str(payload_path), runtime_path=str(runtime_path)
        )
        for ordinal in range(1, repeat + 1):
            started = datetime.now(timezone.utc).isoformat()
            before = time.monotonic()
            proc = run_command(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=contract["resource"]["timeout_sec"] + 120.0,
            )
            combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if hidden_seed in combined_output:
                raise RuntimeError(
                    f"{contract['task']} verifier output crossed the private-seed boundary"
                )
            parsed = _parse_replay_result(proc.stdout or "")
            if proc.returncode != 0 or parsed is None:
                detail = ((proc.stderr or "") + "\n" + (proc.stdout or ""))[-4000:]
                raise RuntimeError(
                    f"{contract['task']} repeat {ordinal} failed rc={proc.returncode}: {detail}"
                )
            result = parsed["result"]
            if not isinstance(result.get("feasible"), bool) or isinstance(
                result.get("raw"), bool
            ) or not isinstance(result.get("raw"), (int, float)):
                raise RuntimeError(
                    f"{contract['task']} repeat {ordinal} returned an invalid verifier result"
                )
            rows.append(
                {
                    "ordinal": ordinal,
                    "host_started_at_utc": started,
                    "host_duration_seconds": time.monotonic() - before,
                    "docker_exit_code": proc.returncode,
                    **parsed,
                }
            )
    return {
        **base,
        "status": "replayed",
        "repeat_count": len(rows),
        "repeats": rows,
        "raw_values": [row["result"]["raw"] for row in rows],
        "feasible_values": [row["result"]["feasible"] for row in rows],
    }


def _write_replay_report(output: Path | str, report: dict[str, Any]) -> None:
    output = _reject_symlink_components(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output.parent)
    if os.path.lexists(output) and (output.is_symlink() or not output.is_file()):
        raise ValueError(f"replay report target is not a regular file: {output}")
    raw = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def evaluate_results(
    *,
    batch_name: str,
    runs_root: Path | str,
    accepted_root: Path | str,
    trusted_runs_root: Path | str,
    workers: int,
    repeat: int,
    output: Path | str,
    hidden_seed_file: Path | str | None = None,
    audit_seed: str | None = None,
    dry_run: bool = False,
    run_command: Any = None,
) -> dict[str, Any]:
    """Replay this arm's exact 17 identity-mapped finalists on accepted evaluators."""

    if workers < 1 or repeat < 1:
        raise ValueError("workers and repeat must both be positive")
    hidden_seed, seed_report = _replay_seed(
        hidden_seed_file=hidden_seed_file, audit_seed=audit_seed
    )
    runner = subprocess.run if run_command is None else run_command
    provenance, contracts = _build_replay_contracts(
        batch_name=batch_name,
        runs_root=runs_root,
        accepted_root=accepted_root,
        trusted_runs_root=trusted_runs_root,
        dry_run=dry_run,
        run_command=runner,
    )
    started = datetime.now(timezone.utc).isoformat()
    failures: list[str] = []
    if dry_run:
        tasks = [
            {**_task_report_base(contract), "status": "contract_validated", "repeats": [], "raw_values": [], "feasible_values": []}
            for contract in contracts
        ]
    else:
        by_task: dict[str, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _replay_task,
                    contract,
                    repeat=repeat,
                    hidden_seed=hidden_seed,
                    run_command=runner,
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
                        **_task_report_base(contract),
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
        "purpose": "Co-Evolve plus Human Proxy 17-task replay against accepted Final Evaluators",
        "status": (
            "dry_run_pass" if dry_run else ("pass" if not failures else "fail")
        ),
        "batch_name": batch_name,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "hidden_seed": seed_report,
        "execution": {
            "dry_run": dry_run,
            "parallel_tasks": workers,
            "sequential_repeats_per_task": repeat,
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
    _write_replay_report(output, report)
    if failures:
        raise RuntimeError(
            f"Final Evaluator replay failed for {len(failures)} task(s); report: {output}"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or audit AutoLab Human Proxy experiment artifacts"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "audit"):
        child = subparsers.add_parser(command)
        child.add_argument("--accepted-root", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
    for command in ("prepare-runs", "audit-runs"):
        child = subparsers.add_parser(command)
        child.add_argument("--control-runs-root", type=Path, required=True)
        child.add_argument(
            "--runs-root",
            type=Path,
            required=True,
            help=(
                "dedicated experiment runs root; prepare-runs requires it not to exist"
            ),
        )
        child.add_argument("--batch-name", required=True)
        child.add_argument("--problems-root", type=Path, required=True)
        if command == "audit-runs":
            child.add_argument("--context-dir", type=Path, required=True)
    live = subparsers.add_parser("audit-live-privacy")
    live.add_argument("--runs-root", type=Path, required=True)
    live.add_argument("--batch-name", required=True)
    live.add_argument("--context-dir", type=Path, required=True)
    live.add_argument("--require-closed-tasks", type=int, default=0)
    evaluate = subparsers.add_parser("evaluate-results")
    evaluate.add_argument("--batch-name", required=True)
    evaluate.add_argument("--runs-root", type=Path, required=True)
    evaluate.add_argument("--accepted-root", type=Path, required=True)
    evaluate.add_argument("--trusted-runs-root", type=Path, required=True)
    evaluate.add_argument("--workers", type=int, default=4)
    evaluate.add_argument("--repeat", type=int, default=2)
    evaluate.add_argument("--output", type=Path, required=True)
    seed_group = evaluate.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--hidden-seed-file", type=Path)
    seed_group.add_argument("--audit-seed")
    evaluate.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_contexts(args.accepted_root, args.output_dir)
        summary = {"status": "pass", "arm": result["arm"], "task_count": 17}
    elif args.command == "audit":
        summary = audit_contexts(args.accepted_root, args.output_dir)
    elif args.command == "prepare-runs":
        summary = prepare_runs(
            args.control_runs_root,
            args.runs_root,
            args.batch_name,
            problems_root=args.problems_root,
        )
    elif args.command == "audit-runs":
        summary = audit_runs(
            args.control_runs_root,
            args.runs_root,
            args.batch_name,
            problems_root=args.problems_root,
            context_dir=args.context_dir,
        )
    elif args.command == "audit-live-privacy":
        summary = audit_live_privacy(
            runs_root=args.runs_root,
            batch_name=args.batch_name,
            context_dir=args.context_dir,
            require_closed_tasks=args.require_closed_tasks,
        )
    else:
        summary = evaluate_results(
            batch_name=args.batch_name,
            runs_root=args.runs_root,
            accepted_root=args.accepted_root,
            trusted_runs_root=args.trusted_runs_root,
            workers=args.workers,
            repeat=args.repeat,
            output=args.output,
            hidden_seed_file=args.hidden_seed_file,
            audit_seed=args.audit_seed,
            dry_run=args.dry_run,
        )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
