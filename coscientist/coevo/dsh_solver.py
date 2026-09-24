"""DeepSeek Harness-backed Solver.

This keeps the Co-Scientist Solver contract identical to ``CodexSolver`` while
using DSH's one-shot headless runner as the agent process.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

from .codex_solver import CodexSolver


@dataclass
class DshSolver(CodexSolver):
    """Drive ``dsh --profile headless`` in the existing Solver workspace."""

    name = "dsh"
    binary = "dsh"

    def _available(self):
        if shutil.which(self.binary) is None:
            return f"DSH CLI not found on PATH ({self.binary!r})"
        if self.gateway is None:
            return "no gateway wired to DshSolver"
        return None

    def run(self, ctx):
        unavailable = self._available()
        if unavailable is not None:
            self._note = f"dsh solver unavailable: {unavailable}"
            if ctx.store is not None:
                ctx.store.trajectory(event="solver_unavailable", detail=self._note)
            return

        ws = ctx.solution_ws
        self._seed_workspace(ws)
        env = dict(os.environ)
        env["COSCI_GATEWAY"] = self.gateway.base_url
        env["PATH"] = f"{ws}:{env.get('PATH', '')}"
        # DSH selects its model through its profile/configuration. The model
        # field remains part of the shared Solver API but is intentionally not
        # translated into an undocumented CLI flag.
        argv = [self.binary, "--profile", "headless", "--json", "-"]
        timeout = max(1.0, ctx.deadline.remaining())
        if ctx.store is not None:
            ctx.store.event("dsh_launch", timeout_s=round(timeout, 1), profile="headless")
        try:
            proc = subprocess.run(argv, cwd=str(ws), input=self._prompt(), env=env,
                                  capture_output=True, text=True, timeout=timeout)
            self._note = (proc.stdout or "")[-400:] if proc.returncode == 0 \
                else (proc.stderr or "dsh nonzero")[-400:]
        except subprocess.TimeoutExpired:
            self._note = "dsh hit the wall-clock budget (expected for a full run)"
        except FileNotFoundError:
            self._note = f"binary not found: {self.binary!r}"

        self._ran = True
        self._recover(ws, ctx)
