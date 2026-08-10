"""Append-only event stream — the run's single source of truth for the CLI,
metrics, and the exported Human-Proxy dataset.

Every meaningful action becomes one JSON line: a solution committed, a hack
detected, an evaluator-evolution round triggered, a human review request and its
response, a verifier version installed. The CLI renders these live; the
``(review_request, review_response)`` pairs are exactly the dataset the design
doc proposes to release.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Optional


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


@dataclass
class EventLog:
    path: Optional[Path] = None
    events: list[dict] = field(default_factory=list)

    def emit(self, kind: str, **fields: Any) -> dict:
        ev = {"kind": kind, **{k: _jsonable(v) for k, v in fields.items()}}
        self.events.append(ev)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev) + "\n")
        return ev

    def of_kind(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["kind"] == kind]
