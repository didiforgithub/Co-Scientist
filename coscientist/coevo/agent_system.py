"""The agent system (§4) — the SYSTEM builds and evolves its own evaluator.

This is the general path the user asked for: drop **raw input** (some files + a
text description — nothing more, NO ProblemSpec) into the system, and the system
itself researches the problem, authors the evaluation, and attempts/advances it.
The orchestrator here is plain code — the hub in a hub-and-spoke topology. It never
understands the problem's content; three coding-agent sessions do all the thinking:

  1. **Bootstrap (Supervisor authors V0).** A one-shot Supervisor container is
     handed the raw input and the ``verify(payload, ctx) -> {feasible, raw,
     artifacts}`` contract. It researches the problem and WRITES the evaluation:
     ``verifier.py`` (the black box V0), ``ctx.json`` (the instance the verifier
     needs), ``seed_solution.json`` (a scorable starting point), and ``probes.json``
     (degenerate/adversarial solutions it KNOWS should score low — the red-team
     contract, generalized from the curve-fit "mode-count monotonicity" oracle).
     The orchestrator recovers these at the boundary as V0. Nothing is presupposed
     about the domain.

  2. **Solve (long-lived Solver container + file-state resume).** The Solver agent
     works against the black box through the ``container-eval`` / ``container-status``
     / ``container-ask-supervisor`` shims (a bind-mounted unix socket). It EXITS when
     it wants to ask the Supervisor; the orchestrator processes the request and a
     fresh ``claude -p`` in the SAME open container resumes from file state + the
     agent's scratchpad. No polling.

  3. **Evolve (Supervisor agent-authored detect + harden).** On a review request —
     or on the orchestrator's own wall-clock cadence — a one-shot Supervisor
     container is given the current V source, the solver's best solution, and the
     probe evidence, and asked to decide whether V is being gamed and, if so, to
     REWRITE a strictly-harder V that still scores the seed feasibly. Detection AND
     hardening are agent-authored, because a canned curve-fit fix would overwrite
     the problem's own verifier (the load-bearing consequence).

The evaluator mechanism (``demo.evaluator.Evaluator``), the black-box wrapper
(``EvalService``), the boundary store (``RunStore``), and the wall clock
(``Deadline``) are reused unchanged — they were already domain-agnostic. Only the
verifier *source strings* and the *ctx/solution shapes* are agent-authored now,
instead of the hardcoded curve-fit ones.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..demo.evaluator import Evaluator
from .budget import Deadline
from .container import (ControlSocket, DockerContainer, GatewayConfig,
                        agent_elf_path, docker_unavailable, one_shot_agent)
from .eval_service import EvalService, FeedbackLevel
from .store import RunStore


# ---------------------------------------------------------------------------
# recovered bootstrap artifacts
# ---------------------------------------------------------------------------
@dataclass
class Bootstrap:
    """What the Supervisor bootstrap agent authored, recovered at the boundary."""

    verifier_src: str
    ctx: dict
    seed_solution: dict
    probes: list[dict]          # each: {"description": str, "solution": {...}}
    notes: str = ""


class AgentSystemUnavailable(RuntimeError):
    """Raised when docker or the agent binary is absent — a clean, explained stop."""


# ---------------------------------------------------------------------------
# prompts — domain-agnostic; the raw input carries all the problem content
# ---------------------------------------------------------------------------
_VERIFY_CONTRACT = '''\
The evaluation you write is a Python module `verifier.py` exposing exactly:

    def verify(payload: dict, ctx: dict) -> dict:
        # payload: a candidate solution (a dict — you define its shape)
        # ctx:     the fixed instance data your verifier needs (from ctx.json)
        return {"feasible": bool, "raw": float, "artifacts": {...}}
        # `raw` is the score: HIGHER IS BETTER. `artifacts` is any diagnostics
        # you choose to expose. It must be pure-Python stdlib (no third-party
        # imports), deterministic, and finish in a few seconds.

It is run OUT OF PROCESS on untrusted input, so it must never hang or crash on a
malformed payload — guard everything and return feasible=False with raw very low
for garbage input.'''

_BOOTSTRAP_PROMPT = '''\
You are the SUPERVISOR in a co-evolution system. Raw input describing a problem has
been dropped into this directory (/work). Read EVERYTHING here first (there may be a
text description and/or other files), reason about the problem from the input and
your own knowledge, and decide how a candidate solution to it should be JUDGED.

Work primarily from the raw input and your own knowledge; do not block on external
research. Encode your best-judgment model directly into the verifier. Use Read/Write
and shell (python3) on local files.

Then AUTHOR THE EVALUATION. Write these files into /work:

1. `verifier.py` — {contract}

2. `ctx.json` — a JSON object with whatever fixed data your verifier reads from
   `ctx` (e.g. an instance, target constants, parameters). May be {{}} if none.

3. `seed_solution.json` — one concrete, valid candidate solution (in the payload
   shape your verifier expects) that a solver can score immediately as a baseline.
   It should be feasible but NOT good — a starting point.

4. `probes.json` — a JSON list of adversarial/degenerate candidates you believe a
   GOOD verifier should score LOW, each as {{"description": "...", "solution": {{...}}}}.
   These are the red-team: if a weak verifier scores any of them competitively high,
   it is being gamed. Include at least 2 that try to exploit obvious shortcuts.

5. `SOLVER_BRIEF.md` — a short, self-contained brief telling a solver agent what the
   problem is, the exact payload shape to produce, and that it is scored by a hidden
   verifier via `./container-eval <solution.json>`. Do NOT reveal verifier.py's
   internals in this brief.

Verify your verifier.py actually imports and runs on seed_solution.json and on each
probe before you finish (run it with python3). This is the ENTIRE evaluation the
rest of the system will use — be rigorous. When done, write a one-line
`BOOTSTRAP_DONE` marker file.'''

_HARDEN_PROMPT = '''\
You are the SUPERVISOR. You own the hidden verifier V for a problem. A solver has
been optimizing against V and may be GAMING it (exploiting a flaw to score high
without truly solving the problem).

In /work you have:
  * `problem/` — the original raw problem input (read it to recall what a real
    solution means),
  * `current_verifier.py` — the current V source,
  * `best_solution.json` — the solver's current best candidate,
  * `probe_report.json` — your red-team probes and the score each got under current
    V (a probe scoring competitively high is evidence of a hole).

Decide: is V being gamed? Then write into /work:

1. `verdict.json` — {{"gaming": bool, "reasoning": "..."}}.

2. If gaming is true (or you otherwise see a flaw), write `verifier.py` — a REWRITTEN
   V that closes the hole: it must score the exploit/probe solutions strictly LOWER
   while still scoring a genuine solution well. Same contract as before:
   def verify(payload, ctx) -> {{"feasible","raw","artifacts"}}, stdlib-only,
   robust to garbage. It MUST still return feasible=True with a finite raw on
   `seed_solution.json` (in /work). Verify it runs (python3) on the seed and on the
   probes before finishing. If V is NOT being gamed and needs no change, do not
   write verifier.py.

Write a one-line `HARDEN_DONE` marker when finished.'''


# ---------------------------------------------------------------------------
# the orchestrator
# ---------------------------------------------------------------------------
@dataclass
class AgentSystem:
    """Plain-code hub: builds the agent-authored evaluator and runs the two loops.

    ``raw_input_dir`` is the drop-in (files + text). ``run_dir`` is the RunStore
    root. ``budget_s`` is the wall-clock ceiling for the whole run.
    """

    raw_input_dir: Path
    run_dir: Path
    budget_s: float = 1800.0
    image: str = "python:3.11-slim"
    feedback_level: FeedbackLevel = FeedbackLevel.WITH_ARTIFACTS
    bootstrap_timeout_s: float = 900.0
    harden_timeout_s: float = 600.0

    gateway: Optional[GatewayConfig] = None
    agent_elf: Optional[Path] = None

    store: RunStore = field(init=False)
    deadline: Deadline = field(init=False)
    evaluator: Optional[Evaluator] = field(default=None, init=False)
    eval_service: Optional[EvalService] = field(default=None, init=False)
    _ctx: dict = field(default_factory=dict, init=False)
    _seed: dict = field(default_factory=dict, init=False)
    _probes: list = field(default_factory=list, init=False)
    _best_payload: Optional[dict] = field(default=None, init=False)
    _best_score: float = field(default=float("-inf"), init=False)
    hardenings: int = field(default=0, init=False)
    reviews_handled: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.raw_input_dir = Path(self.raw_input_dir).resolve()
        self.run_dir = Path(self.run_dir).resolve()
        self.store = RunStore(self.run_dir)
        self.deadline = Deadline(budget_s=self.budget_s)

    # -- preflight --------------------------------------------------------
    def preflight(self) -> None:
        reason = docker_unavailable()
        if reason:
            raise AgentSystemUnavailable(reason)
        elf = self.agent_elf or agent_elf_path()
        if elf is None:
            raise AgentSystemUnavailable(
                "codex agent binary not found (need the @openai/codex npm package "
                "with its vendored static ELF)")
        self.agent_elf = elf
        gw = self.gateway or GatewayConfig.from_host()
        if gw is None:
            raise AgentSystemUnavailable(
                "no codex auth found (need ~/.codex/auth.json)")
        self.gateway = gw

    # ================================================================
    # LOOP 0 — bootstrap: the Supervisor agent authors the evaluator
    # ================================================================
    def bootstrap(self) -> Bootstrap:
        assert self.gateway and self.agent_elf
        ws = self.run_dir / "bootstrap_ws"
        ws.mkdir(parents=True, exist_ok=True)
        # drop the raw input straight into the Supervisor's workspace.
        _copy_tree(self.raw_input_dir, ws)
        (ws / "VERIFY_CONTRACT.txt").write_text(_VERIFY_CONTRACT, encoding="utf-8")

        self.store.event("bootstrap_start", raw_input=str(self.raw_input_dir))
        prompt = _BOOTSTRAP_PROMPT.format(contract=_VERIFY_CONTRACT)
        remaining = self.deadline.remaining()
        session = one_shot_agent(
            ws, self.gateway, self.agent_elf, prompt,
            timeout_s=min(self.bootstrap_timeout_s, remaining), image=self.image)
        self.store.cost(who="supervisor", kind="bootstrap_session", calls=1,
                        seconds=0.0, ok=session.ok)

        bs = self._recover_bootstrap(ws, session.note, session.stderr)
        # commit V0 through the reused, domain-agnostic evaluator.
        self.evaluator = Evaluator()
        self.evaluator.versions.append(_version0(bs.verifier_src))
        self._ctx, self._seed, self._probes = bs.ctx, bs.seed_solution, bs.probes
        self.eval_service = EvalService(
            evaluator=self.evaluator, ctx_provider=lambda: self._ctx,
            feedback_level=self.feedback_level)
        self.eval_service.on_query = lambda rec, res: self.store.query(_query_line(rec))
        self.store.verifier_version(0, bs.verifier_src, origin="agent",
                                    note="supervisor bootstrap", rationale=bs.notes[:400])
        self.store.event("bootstrap_done", n_probes=len(bs.probes),
                         ctx_keys=sorted(bs.ctx.keys()),
                         seed_keys=sorted(bs.seed_solution.keys())
                         if isinstance(bs.seed_solution, dict) else [])
        return bs

    def _recover_bootstrap(self, ws: Path, note: str, stderr: str = "") -> Bootstrap:
        vf = ws / "verifier.py"
        if not vf.is_file():
            tail = (stderr or "").strip()[-400:]
            raise AgentSystemUnavailable(
                f"bootstrap agent did not author verifier.py (note: {note[:200]}"
                + (f"; stderr: {tail}" if tail else "") + ")")
        src = vf.read_text(encoding="utf-8")
        ctx = _read_json(ws / "ctx.json", default={})
        seed = _read_json(ws / "seed_solution.json", default={})
        probes_raw = _read_json(ws / "probes.json", default=[])
        probes = [p for p in probes_raw if isinstance(p, dict) and "solution" in p] \
            if isinstance(probes_raw, list) else []
        brief = ws / "SOLVER_BRIEF.md"
        notes = brief.read_text(encoding="utf-8")[:2000] if brief.is_file() else ""
        # smoke-test V0 on the seed before trusting it as the black box.
        probe_ev = Evaluator()
        probe_ev.versions.append(_version0(src))
        r = probe_ev.run(seed, ctx)
        if r.error:
            raise AgentSystemUnavailable(
                f"bootstrap verifier.py fails on its own seed: {r.error[:200]}")
        return Bootstrap(verifier_src=src, ctx=ctx, seed_solution=seed,
                         probes=probes, notes=notes)

    # ================================================================
    # LOOP 1 — solve: long-lived Solver container + file-state resume
    # LOOP 2 — evolve: agent-authored detect + harden on review / cadence
    # ================================================================
    def solve_and_evolve(self, *, max_turns: int = 12) -> None:
        assert self.gateway and self.agent_elf and self.eval_service
        sol_ws = self.run_dir / "solver_ws"
        sol_ws.mkdir(parents=True, exist_ok=True)
        self._seed_solver_ws(sol_ws)

        control = ControlSocket(workdir=sol_ws, handler=self._handle_shim)
        control.seed_shims(sol_ws)
        control.start()
        container = DockerContainer(workdir=sol_ws, gateway=self.gateway,
                                    agent_elf=self.agent_elf, image=self.image)
        self.store.event("solver_container_start")
        try:
            container.start()
            turn = 0
            while turn < max_turns and not self.deadline.expired():
                turn += 1
                remaining = self.deadline.remaining()
                if remaining <= 5:
                    break
                self.store.event("solver_turn_start", turn=turn,
                                 remaining_s=round(remaining, 1))
                session = container.exec_agent(
                    self._solver_prompt(turn), timeout_s=remaining)
                self.store.cost(who="solver", kind="agent_turn", calls=1,
                                ok=session.ok, turn=turn)
                self._extract_best(sol_ws)

                # did the agent leave a review request? (its reason for exiting)
                review_req = sol_ws / "review_request.json"
                if review_req.is_file():
                    self._handle_review_request(review_req, sol_ws)
                    review_req.unlink()
                    continue   # resume: next turn in the SAME container
                if session.note.startswith("agent hit the wall-clock"):
                    break
                # agent finished without asking — one proactive supervisor pass,
                # then stop if nothing changed.
                if not self._proactive_supervise(sol_ws):
                    break
        finally:
            container.stop()
            control.stop()
            self.store.event("solver_container_stop")
        self._finalize()

    # -- shim handler (synchronous socket channel) ------------------------
    def _handle_shim(self, cmd: str, arg: dict) -> dict:
        svc = self.eval_service
        assert svc is not None
        if cmd == "eval":
            r = svc.query(arg.get("solution", {}) or {})
            self.store.cost(who="solver", kind="eval_query", calls=1)
            if r.ok and r.score is not None and r.score > self._best_score:
                self._best_score = r.score
                self._best_payload = arg.get("solution", {})
            return {"ok": r.ok, "score": r.score, "feasible": r.feasible,
                    "artifacts": r.artifacts, "verifier_version": r.verifier_version,
                    "feedback_level": r.feedback_level, "detail": r.detail}
        if cmd == "status":
            return {"verifier_version": svc.current_version(),
                    "feedback_level": svc.feedback_level.value,
                    "deadline_remaining_s": round(self.deadline.remaining(), 1)}
        if cmd == "ask":
            # the agent asks synchronously but the heavy path is file-state resume;
            # acknowledge and tell it to write review_request.json + exit.
            return {"ok": True,
                    "instruction": ("write your best solution and your question to "
                                    "review_request.json (as {\"solution\":...,"
                                    "\"question\":...}) and EXIT; the supervisor will "
                                    "review and you will resume with a fresh verifier_version")}
        return {"ok": False, "error": f"unknown cmd {cmd!r}"}

    # -- the evolve loop: agent-authored review + harden ------------------
    def _handle_review_request(self, req_path: Path, sol_ws: Path) -> None:
        try:
            req = json.loads(req_path.read_text())
        except (json.JSONDecodeError, OSError):
            req = {}
        best = req.get("solution") or self._best_payload or self._seed
        self.reviews_handled += 1
        self.store.event("review_request", question=str(req.get("question", ""))[:300])
        self._run_supervisor_harden(best, sol_ws, trigger="review_request")

    def _proactive_supervise(self, sol_ws: Path) -> bool:
        """Orchestrator-driven cadence: red-team + maybe harden. Returns True if V moved."""
        best = self._best_payload or self._seed
        before = self.eval_service.current_version()   # type: ignore[union-attr]
        self._run_supervisor_harden(best, sol_ws, trigger="proactive")
        return self.eval_service.current_version() > before   # type: ignore[union-attr]

    def _run_supervisor_harden(self, best: dict, sol_ws: Path, *, trigger: str) -> None:
        assert self.gateway and self.agent_elf and self.eval_service
        if self.deadline.expired():
            return
        ws = self.run_dir / "harden_ws"
        if ws.exists():
            _rmtree(ws)
        ws.mkdir(parents=True, exist_ok=True)
        _copy_tree(self.raw_input_dir, ws / "problem")
        (ws / "current_verifier.py").write_text(
            self.eval_service.current_source(), encoding="utf-8")
        (ws / "best_solution.json").write_text(json.dumps(best, indent=2))
        (ws / "seed_solution.json").write_text(json.dumps(self._seed, indent=2))
        (ws / "probe_report.json").write_text(
            json.dumps(self._probe_report(), indent=2))

        remaining = self.deadline.remaining()
        self.store.event("harden_start", trigger=trigger,
                         remaining_s=round(remaining, 1))
        session = one_shot_agent(
            ws, self.gateway, self.agent_elf, _HARDEN_PROMPT,
            timeout_s=min(self.harden_timeout_s, remaining), image=self.image)
        self.store.cost(who="supervisor", kind="harden_session", calls=1, ok=session.ok)

        verdict = _read_json(ws / "verdict.json", default={})
        new_vf = ws / "verifier.py"
        installed = False
        if new_vf.is_file():
            new_src = new_vf.read_text(encoding="utf-8")
            err = self.eval_service.validate(new_src, self._seed)
            if err is None:
                ver = self.eval_service.install_verifier(
                    new_src, origin="agent",
                    note=f"harden ({trigger}): {str(verdict.get('reasoning',''))[:120]}")
                self.store.verifier_version(
                    ver, new_src, origin="agent", note=f"harden ({trigger})",
                    rationale=str(verdict.get("reasoning", ""))[:400])
                self.hardenings += 1
                installed = True
            else:
                self.store.event("harden_rejected", reason=err[:200])
        self.store.review(trigger=trigger, gaming=bool(verdict.get("gaming")),
                          installed=installed,
                          reasoning=str(verdict.get("reasoning", ""))[:400])

    def _probe_report(self) -> list:
        """Score each bootstrap probe under the current V — evidence for the smith."""
        out = []
        svc = self.eval_service
        for p in self._probes:
            sol = p.get("solution", {})
            r = svc.query(sol) if svc else None   # type: ignore[union-attr]
            out.append({"description": p.get("description", ""),
                        "score": (r.score if r else None),
                        "feasible": (r.feasible if r else None)})
        return out

    # -- solver workspace + prompts ---------------------------------------
    def _seed_solver_ws(self, ws: Path) -> None:
        brief = self.run_dir / "bootstrap_ws" / "SOLVER_BRIEF.md"
        if brief.is_file():
            (ws / "PROBLEM_BRIEF.md").write_text(brief.read_text(encoding="utf-8"),
                                                 encoding="utf-8")
        (ws / "seed_solution.json").write_text(json.dumps(self._seed, indent=2))
        (ws / "solution_out.json").write_text(json.dumps(self._seed))
        if not (ws / "scratchpad.md").exists():
            (ws / "scratchpad.md").write_text(
                "# Solver scratchpad\n\nRecord findings here so future turns resume "
                "fast. This file persists across turns.\n", encoding="utf-8")

    def _solver_prompt(self, turn: int) -> str:
        return (
            f"You are the SOLVER (turn {turn}). Read PROBLEM_BRIEF.md and scratchpad.md "
            "in this directory (/work) first — scratchpad.md is your own memory from "
            "previous turns, keep appending to it.\n\n"
            "Score any candidate solution with:  ./container-eval <solution.json>  "
            "(higher score is better; it returns verifier_version too). Check remaining "
            "time with:  ./container-status .\n\n"
            "Watch verifier_version in each eval response: if it INCREASED since last "
            "turn, the Supervisor hardened the hidden verifier — your old best may be "
            "stale, so re-baseline. Do NOT merely game the verifier; a Supervisor is "
            "red-teaming it.\n\n"
            "Write your single best solution as JSON to solution_out.json (overwrite "
            "when you improve it). If you plateau or suspect your best is exploiting a "
            "verifier flaw, run  ./container-ask-supervisor solution_out.json , then "
            "follow its instruction (write review_request.json and exit) so the "
            "Supervisor can review and you resume next turn with a fresh verifier. "
            "Otherwise, keep improving until you are confident, then stop."
        )

    def _extract_best(self, ws: Path) -> None:
        out = ws / "solution_out.json"
        if not out.is_file():
            return
        try:
            payload = json.loads(out.read_text())
        except (json.JSONDecodeError, OSError):
            return
        r = self.eval_service.query(payload)   # type: ignore[union-attr]
        self.store.candidate(payload, {"score": r.score,
                                       "verifier_version": r.verifier_version},
                             score=r.score)
        if r.ok and r.score is not None and r.score > self._best_score:
            self._best_score, self._best_payload = r.score, payload
        self.store.trajectory(event="best_extracted", best_score=self._best_score,
                              verifier_version=r.verifier_version)

    # -- finalize ---------------------------------------------------------
    def _finalize(self) -> None:
        svc = self.eval_service
        final_v = svc.current_version() if svc else -1
        self.store.event("run_stop", final_verifier_version=final_v,
                         hardenings=self.hardenings, reviews=self.reviews_handled,
                         best_score=self._best_score)
        self.store.finalize_manifest({
            "final_verifier_version": final_v,
            "verifier_hardenings": self.hardenings,
            "reviews_handled": self.reviews_handled,
            "best_score": self._best_score if self._best_score != float("-inf") else None,
            "best_solution": self._best_payload,
        })

    # -- top-level entry --------------------------------------------------
    def run(self, *, max_turns: int = 12) -> "AgentSystem":
        self.preflight()
        self.store.write_manifest({
            "mode": "agent_system",
            "raw_input_dir": str(self.raw_input_dir),
            "budget_s": self.budget_s,
            "image": self.image,
            "initial_feedback_level": self.feedback_level.value,
        })
        self.store.event("run_start", budget_s=self.budget_s, mode="agent_system")
        self.bootstrap()
        self.solve_and_evolve(max_turns=max_turns)
        return self

    def summary(self) -> dict:
        return {
            "final_verifier_version":
                self.eval_service.current_version() if self.eval_service else -1,
            "verifier_hardenings": self.hardenings,
            "reviews_handled": self.reviews_handled,
            "best_score": self._best_score if self._best_score != float("-inf") else None,
            "best_solution": self._best_payload,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _version0(src: str):
    from ..demo.evaluator import VerifierVersion
    return VerifierVersion(0, src, "agent", "supervisor bootstrap")


def _read_json(path: Path, *, default):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _copy_tree(src: Path, dst: Path) -> None:
    import shutil as _sh
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        _sh.copy2(src, dst / src.name)
        return
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            _sh.copy2(item, target)


def _rmtree(path: Path) -> None:
    import shutil as _sh
    _sh.rmtree(path, ignore_errors=True)


def _query_line(rec) -> dict:
    return {
        "query_id": rec.query_id, "t": rec.t, "who": rec.who,
        "verifier_version": rec.verifier_version, "feedback_level": rec.feedback_level,
        "payload_summary": rec.payload_summary, "returned_score": rec.returned_score,
        "returned_keys": rec.returned_keys,
    }
