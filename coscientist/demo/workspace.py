"""The shared file workspace an agent session operates in.

Each proposer session gets a fresh workspace directory seeded with exactly what
that role is allowed to see:

  * a ``TASK`` file naming the role (so the stub backend can dispatch),
  * ``context.json``  — the agent-visible instance slice (t, obs, sigma); NEVER
    the hidden signal,
  * ``PROMPT.md``     — the natural-language task,
  * role-specific inputs (current solution.json, current verifier.py, evidence),
  * a known OUTPUT path the agent must write.

The caller reads the OUTPUT path back after the session. Keeping the hidden
truth out of this directory is what makes the setting honest: an agent asked to
improve the solution literally cannot open the answer key.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional


class Workspace:
    def __init__(self, root: Optional[Path] = None) -> None:
        self._tmp = None
        if root is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="cosci_ws_")
            root = Path(self._tmp.name)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, rel: str, content: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def write_json(self, rel: str, obj) -> Path:
        return self.write(rel, json.dumps(obj, indent=2))

    def read(self, rel: str) -> Optional[str]:
        p = self.root / rel
        return p.read_text(encoding="utf-8") if p.is_file() else None

    def read_json(self, rel: str):
        txt = self.read(rel)
        if txt is None:
            return None
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            return None

    def cleanup(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()
