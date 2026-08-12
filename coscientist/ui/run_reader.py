"""Read-only assembler for ``runs/<id>/`` trees (the layout in ``coevo.store``).

Every parser here is defensive: a run may be growing WHILE we read it, so JSONL is
parsed line-by-line and a bad trailing line is skipped, not fatal. No third-party
deps. Nothing writes; this module only reads what the orchestrator already put on
disk.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

# A run whose events.jsonl changed within this many seconds is flagged "alive"
# (heuristic — see is_alive). Real runs stamp events on their own cadence; this is
# a recency proxy, not a definitive liveness signal.
ALIVE_WINDOW_S = 180.0


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file tolerantly; skip blank/corrupt lines (e.g. a live tail)."""
    out: list[dict] = []
    if not path.is_file():
        return out
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # partial trailing write while the run is live — skip it
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def list_run_ids(runs_dir: Path) -> list[str]:
    """Directories under runs_dir that look like a run (have events.jsonl or manifest)."""
    if not runs_dir.is_dir():
        return []
    ids = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "events.jsonl").is_file() or (child / "manifest.json").is_file():
            ids.append(child.name)
    return ids


def is_alive(run_dir: Path) -> tuple[bool, Optional[float]]:
    """Heuristic liveness: events.jsonl touched recently AND no run_stop event.

    Returns (alive, seconds_since_last_event). This is best-effort — there is no
    pid file in the store layout, so recency of the event stream is the signal.
    """
    events = run_dir / "events.jsonl"
    mt = _mtime(events)
    if mt is None:
        return False, None
    age = time.time() - mt
    # A recorded run_stop/run_complete means the run finished regardless of mtime.
    evs = _read_jsonl(events)
    if evs and evs[-1].get("kind") in ("run_stop", "run_complete", "finalize"):
        return False, age
    return age <= ALIVE_WINDOW_S, age


def _payload_summary(payload: Any) -> str:
    """A compact one-line summary of a candidate payload (shape varies by problem)."""
    if isinstance(payload, dict):
        parts = []
        for k, v in list(payload.items())[:4]:
            if isinstance(v, list):
                parts.append(f"{k}[{len(v)}]")
            elif isinstance(v, (int, float, str, bool)):
                s = str(v)
                parts.append(f"{k}={s[:24]}")
            else:
                parts.append(k)
        return ", ".join(parts) if parts else "{}"
    return str(payload)[:80]


def _current_version(run_dir: Path) -> int:
    vdir = run_dir / "supervisor" / "verifier_versions"
    n = -1
    if vdir.is_dir():
        for f in vdir.glob("v*.py"):
            stem = f.stem  # "v3"
            if stem.startswith("v") and stem[1:].isdigit():
                n = max(n, int(stem[1:]))
    return n


def summarize_run(runs_dir: Path, run_id: str) -> dict:
    """Compact summary for the run list (cheap to compute for many runs)."""
    run_dir = runs_dir / run_id
    manifest = _read_json(run_dir / "manifest.json", {})
    events = _read_jsonl(run_dir / "events.jsonl")
    versions = _read_jsonl(run_dir / "supervisor" / "versions.jsonl")
    alive, age = is_alive(run_dir)
    cand_dir = run_dir / "solver" / "candidates"
    n_candidates = len(list(cand_dir.glob("cand_*.json"))) if cand_dir.is_dir() else 0

    hardenings = manifest.get("verifier_hardenings")
    if hardenings is None:
        hardenings = max(0, _current_version(run_dir))
    best_score = manifest.get("best_score")
    mode = manifest.get("final_mode")
    if not mode:
        mode = "construction"
        for ev in events:
            if ev.get("kind") == "mode_switch" and ev.get("to_mode"):
                mode = ev["to_mode"]
    last = events[-1] if events else None
    return {
        "run_id": run_id,
        "alive": alive,
        "alive_heuristic": "events.jsonl mtime within %ds and no run_stop" % int(ALIVE_WINDOW_S),
        "seconds_since_last_event": round(age, 1) if age is not None else None,
        "mode": mode,
        "current_verifier_version": _current_version(run_dir),
        "n_versions": len(versions),
        "hardenings": hardenings,
        "best_score": best_score,
        "n_candidates": n_candidates,
        "n_events": len(events),
        "last_event": ({"kind": last.get("kind"), "t": last.get("t")} if last else None),
        "budget_s": manifest.get("budget_s"),
        "finalized": bool(manifest.get("final_verifier_version") is not None
                          or manifest.get("final_mode")),
    }


def _mode_history(events: list[dict]) -> list[dict]:
    """Timeline of the game mode: starts construction, each mode_switch appends."""
    history = [{"t": 0.0, "mode": "construction", "reason": "initial"}]
    for ev in events:
        if ev.get("kind") == "mode_switch" and ev.get("to_mode"):
            history.append({"t": ev.get("t"), "mode": ev["to_mode"],
                            "reason": str(ev.get("reasoning", ""))[:300]})
    return history


def read_verifier_source(runs_dir: Path, run_id: str, version: int) -> dict:
    """Source of v{n}.py plus its optional v{n}.feedback.py sibling (both as text)."""
    vdir = runs_dir / run_id / "supervisor" / "verifier_versions"
    vf = vdir / f"v{version}.py"
    ff = vdir / f"v{version}.feedback.py"
    return {
        "run_id": run_id,
        "version": version,
        "exists": vf.is_file(),
        "verifier_src": vf.read_text(encoding="utf-8") if vf.is_file() else None,
        "has_feedback": ff.is_file(),
        "feedback_src": ff.read_text(encoding="utf-8") if ff.is_file() else None,
    }


def detail_run(runs_dir: Path, run_id: str) -> Optional[dict]:
    """Full detail assembled from the run tree. None if the run dir does not exist."""
    run_dir = runs_dir / run_id
    if not run_dir.is_dir():
        return None

    manifest = _read_json(run_dir / "manifest.json", {})
    events = _read_jsonl(run_dir / "events.jsonl")
    version_meta = _read_jsonl(run_dir / "supervisor" / "versions.jsonl")
    reviews = _read_jsonl(run_dir / "supervisor" / "reviews.jsonl")
    trajectory = _read_jsonl(run_dir / "solver" / "trajectory.jsonl")
    queries = _read_jsonl(run_dir / "eval" / "queries.jsonl")
    alive, age = is_alive(run_dir)

    # -- versions ladder: merge metadata with which sources exist on disk ----
    vdir = run_dir / "supervisor" / "verifier_versions"
    versions = []
    cur = _current_version(run_dir)
    meta_by_v = {m.get("version"): m for m in version_meta}
    for v in range(0, cur + 1):
        m = meta_by_v.get(v, {})
        has_fb = (vdir / f"v{v}.feedback.py").is_file() or bool(m.get("has_feedback"))
        versions.append({
            "version": v,
            "origin": m.get("origin", "agent"),
            "note": m.get("note", ""),
            "rationale": m.get("rationale", ""),
            "has_feedback": has_fb,
            "t": m.get("t"),
            "src_available": (vdir / f"v{v}.py").is_file(),
        })

    # -- score trajectory: prefer explicit trajectory events, else candidates --
    score_points = []
    for tr in trajectory:
        if "best_score" in tr:
            score_points.append({
                "t": tr.get("t"),
                "score": tr.get("best_score"),
                "verifier_version": tr.get("verifier_version"),
                "mode": tr.get("mode"),
            })
    cand_dir = run_dir / "solver" / "candidates"
    candidates = []
    if cand_dir.is_dir():
        for cf in sorted(cand_dir.glob("cand_*.json")):
            obj = _read_json(cf, {})
            if not isinstance(obj, dict):
                continue
            candidates.append({
                "id": obj.get("id", cf.stem),
                "t": obj.get("t"),
                "score": obj.get("score"),
                "verifier_version": (obj.get("feedback") or {}).get("verifier_version"),
                "payload_summary": _payload_summary(obj.get("payload")),
            })
    if not score_points and candidates:
        for c in candidates:
            if c["score"] is not None:
                score_points.append({"t": c["t"], "score": c["score"],
                                     "verifier_version": c["verifier_version"],
                                     "mode": None})

    # current verifier + feedback source (text) for the inspect panel
    cur_src = read_verifier_source(runs_dir, run_id, cur) if cur >= 0 else {
        "version": cur, "exists": False, "verifier_src": None,
        "has_feedback": False, "feedback_src": None}

    final_mode = manifest.get("final_mode")
    mode_hist = _mode_history(events)
    if not final_mode:
        final_mode = mode_hist[-1]["mode"]

    return {
        "run_id": run_id,
        "alive": alive,
        "seconds_since_last_event": round(age, 1) if age is not None else None,
        "manifest": manifest,
        "events": events,
        "versions": versions,
        "current_verifier_version": cur,
        "current_verifier": cur_src,
        "score_trajectory": score_points,
        "candidates": candidates,
        "reviews": reviews,
        "n_eval_queries": len(queries),
        "mode": final_mode,
        "mode_history": mode_hist,
        "best_score": manifest.get("best_score"),
        "best_solution": manifest.get("best_solution"),
        # Two isolated resource slices (solver / verifier) from the manifest, surfaced
        # top-level so the UI's Resources panel needn't dig into the raw manifest.
        "resource_spec": manifest.get("resource_spec", {}),
    }
