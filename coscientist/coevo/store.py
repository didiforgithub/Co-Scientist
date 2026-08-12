"""Standardized run storage (§6) — free inside the container, 规范 at the boundary.

The hard principle from the design doc: **do not impose a write format on the
agent.** Inside its container the agent scratches however it likes. Normalization
happens *here*, at the boundary — the driver records trajectory, cost, queries,
reviews, and verifier versions into a fixed ``runs/<run_id>/`` layout, so runs are
comparable across solvers (SimpleTES vs Codex) and reproducible by others.

Layout (matches docs/two-loop-architecture.md §6)::

    runs/<run_id>/
      manifest.json          # config: solver, supervisor mode, budget, start/stop
      events.jsonl           # global event stream (wall-clock stamped)
      cost.jsonl             # one line per LLM/eval/consult call: who,kind,tokens,seconds
      solver/
        trajectory.jsonl     # extracted, not agent-authored
        candidates/          # each submitted solution + the feedback it got
      supervisor/
        reviews.jsonl        # each consult: evidence, verdict, dense feedback
        verifier_versions/   # v0.py, v1.py, ... (+ a versions.jsonl with rationale)
        probes/              # red-team probes + whether V was fooled
      eval/
        queries.jsonl        # every Solver->Eval query AND what V chose to return

Cost is recorded at the *gateway* (§6): every eval query and every agent
generation passes through code we own, so ``cost.jsonl`` does not rely on the
agent's cooperation. ``RunStore`` is that sink; it never sleeps and takes an
injectable clock so it stamps the same wall-clock the Deadline uses.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


@dataclass
class RunStore:
    """The boundary sink. One instance per run; creates the tree lazily on open."""

    root: Path
    clock: Callable[[], float] = time.monotonic
    _t0: float = field(init=False)
    _candidate_seq: int = field(default=0, init=False)
    _probe_seq: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self._t0 = self.clock()
        for sub in ("solver/candidates", "supervisor/verifier_versions",
                    "supervisor/probes", "eval"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # -- time -------------------------------------------------------------
    def t(self) -> float:
        return self.clock() - self._t0

    # -- low-level append -------------------------------------------------
    def _append(self, rel: str, obj: dict) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_jsonable(obj)) + "\n")

    # -- manifest ---------------------------------------------------------
    def write_manifest(self, manifest: dict) -> None:
        (self.root / "manifest.json").write_text(
            json.dumps(_jsonable(manifest), indent=2), encoding="utf-8"
        )

    def finalize_manifest(self, extra: dict) -> None:
        """Merge closing fields (stop time, outcome) into the manifest."""
        path = self.root / "manifest.json"
        base = {}
        if path.is_file():
            try:
                base = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                base = {}
        base.update(_jsonable(extra))
        path.write_text(json.dumps(base, indent=2), encoding="utf-8")

    # -- global streams ---------------------------------------------------
    def event(self, kind: str, **fields: Any) -> dict:
        ev = {"t": self.t(), "kind": kind, **{k: _jsonable(v) for k, v in fields.items()}}
        self._append("events.jsonl", ev)
        return ev

    def cost(self, *, who: str, kind: str, tokens: int = 0, seconds: float = 0.0,
             usd: float = 0.0, **extra: Any) -> None:
        """Record one cost line at the gateway (not from the agent's self-report)."""
        self._append("cost.jsonl", {
            "t": self.t(), "who": who, "kind": kind,
            "tokens": tokens, "seconds": seconds, "usd": usd,
            **{k: _jsonable(v) for k, v in extra.items()},
        })

    # -- eval channel -----------------------------------------------------
    def query(self, record: dict) -> None:
        self._append("eval/queries.jsonl", record)

    # -- solver channel ---------------------------------------------------
    def trajectory(self, **fields: Any) -> None:
        self._append("solver/trajectory.jsonl", {"t": self.t(), **fields})

    def candidate(self, payload: dict, feedback: dict, *, score: Optional[float]) -> str:
        """Persist a submitted solution + the feedback it received. Returns its id."""
        cid = f"cand_{self._candidate_seq:05d}"
        self._candidate_seq += 1
        (self.root / "solver" / "candidates" / f"{cid}.json").write_text(
            json.dumps(_jsonable({"id": cid, "t": self.t(), "score": score,
                                  "payload": payload, "feedback": feedback}), indent=2),
            encoding="utf-8",
        )
        return cid

    # -- supervisor channel ----------------------------------------------
    def review(self, **fields: Any) -> None:
        self._append("supervisor/reviews.jsonl", {"t": self.t(), **fields})

    def verifier_version(self, version: int, source: str, *, origin: str, note: str,
                         rationale: str = "", feedback_src: Optional[str] = None) -> None:
        (self.root / "supervisor" / "verifier_versions" / f"v{version}.py").write_text(
            source, encoding="utf-8"
        )
        # An agent-authored feedback module (the free-form disclosure strategy) is
        # versioned as a sibling so resume can rebuild the exact (verify, feedback)
        # pair for each version. Absent → legacy enum path, nothing written.
        if feedback_src is not None:
            (self.root / "supervisor" / "verifier_versions"
             / f"v{version}.feedback.py").write_text(feedback_src, encoding="utf-8")
        self._append("supervisor/versions.jsonl", {
            "t": self.t(), "version": version, "origin": origin,
            "note": note, "rationale": rationale,
            "has_feedback": feedback_src is not None,
        })

    def probe(self, *, description: str, payload: dict, score: Optional[float],
              expected_low: bool, fooled: bool) -> str:
        """Record a Supervisor red-team probe and whether V was fooled by it."""
        pid = f"probe_{self._probe_seq:04d}"
        self._probe_seq += 1
        (self.root / "supervisor" / "probes" / f"{pid}.json").write_text(
            json.dumps(_jsonable({
                "id": pid, "t": self.t(), "description": description,
                "payload": payload, "score": score,
                "expected_low": expected_low, "fooled": fooled,
            }), indent=2),
            encoding="utf-8",
        )
        return pid
