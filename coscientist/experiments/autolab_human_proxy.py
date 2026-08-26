"""Prepare auditable private Human Proxy contexts for the 17 AutoLab tasks.

This module deliberately handles only accepted evaluator context artifacts.  Run
pre-seeding and result replay are separate experiment stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterable


ARM = "autolab_coevolve_human_proxy"
MAX_CONTEXT_CHARACTERS = 200_000
MAX_ACCEPTED_SOURCE_BYTES = 1_000_000
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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    _require_private_regular_file(path, label=label)
    source_stat = path.stat()
    if source_stat.st_size > MAX_ACCEPTED_SOURCE_BYTES:
        raise ValueError(f"{label} is too large")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            raw = stream.read(MAX_ACCEPTED_SOURCE_BYTES + 1)
    except OSError as exc:
        raise ValueError(f"{label} could not be read safely") from exc
    if len(raw) > MAX_ACCEPTED_SOURCE_BYTES:
        raise ValueError(f"{label} is too large")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or audit AutoLab Human Proxy evaluator contexts"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "audit"):
        child = subparsers.add_parser(command)
        child.add_argument("--accepted-root", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_contexts(args.accepted_root, args.output_dir)
        summary = {"status": "pass", "arm": result["arm"], "task_count": 17}
    else:
        summary = audit_contexts(args.accepted_root, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
