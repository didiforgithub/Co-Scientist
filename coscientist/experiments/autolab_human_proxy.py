"""Prepare auditable private Human Proxy contexts for the 17 AutoLab tasks.

This module deliberately handles only accepted evaluator context artifacts.  Run
pre-seeding and result replay are separate experiment stages.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


ARM = "autolab_coevolve_human_proxy"
CORRECTED_CONTROL_PREFIX = "autolab_shippedv0_control17_4h_v2_20260825"
SHIPPED_VERIFIER_ORIGIN = "autolab_shipped_tests/test.sh"
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
    else:
        summary = audit_runs(
            args.control_runs_root,
            args.runs_root,
            args.batch_name,
            problems_root=args.problems_root,
            context_dir=args.context_dir,
        )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
