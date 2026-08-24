"""The STRONG solver mode (§4) — N concurrent Solver agents on ONE problem.

This is the high end of a *solver-strength dial*. The existing single-agent
``AgentSystem`` (agent_system.py) is the **weak** mode; this module adds the
**strong** mode: N (4–5) Solver containers attack the SAME problem against the
SAME black-box evaluator V, sharing a blackboard (progress notes + candidate
archive), advancing in **synchronous generations**, with the Supervisor triggered
**only on a new global SOTA** (provisional → coalesced hack-check → broadcast-or-
harden). It is SimpleTES-style test-time evolution made concurrent: iteration count
drives nothing (the budget stays wall-clock), and the exploration threads re-baseline
together at generation boundaries.

Design (three fixed decisions):
  1. Carrier = N real agent containers. Each solver its own ``DockerContainer`` +
     ``solver_ws_{i}`` + ``ControlSocket``, parallelized with threads.
  2. Round model = synchronous generational barrier. All N run a generation, JOIN,
     converge notes/archive, refresh context, next generation. V is FROZEN for the
     whole generation, so all N face an identical black box; a harden lands only on
     the generation edge (after the join), and gen g+1 is the collective re-baseline.
  3. SOTA = provisional-then-broadcast. A new global best is PROVISIONAL until a
     single (coalesced) Supervisor hack-check clears it: clean → broadcast as the
     "bar" (may install NOTHING, just keep strengthening); hacked → harden V and
     recompute the bar honestly under the new V — never the tainted score.

Architecture: this **composes** an ``AgentSystem`` as ``self.base`` rather than
subclassing it. ``self.base`` stays the sole owner of the evaluator, store, deadline,
instance (ctx/seed/probes), best-so-far, mode, and resource spec; the orchestrator
reads/mutates that state through ``self.base`` under locks and overrides ONLY the
solve loop. Every already-tested primitive — ``bootstrap``, ``_run_supervisor_harden``
/``_apply_harden``, ``_refresh_solver_ws``, ``_recover_best_from_disk``, ``can_resume``
/``_resume_from_disk``, ``_finalize``, ``preflight`` — is reused verbatim. The weak
path (``AgentSystem.run``) is byte-for-byte unchanged.

Locking (deadlock-free — ``_store_lock`` is always the innermost/leaf lock; the only
other nesting is ``_sota_lock`` → ``_eval_lock``):
  * ``_eval_lock``  — every ``EvalService`` touch (the ``Gateway.lock`` discipline):
                      serializes the N concurrent shim queries and fences the SOTA
                      hack-check/harden.
  * ``_store_lock`` — every ``RunStore.*`` call from a solver/shim thread (``_append``
                      and ``candidate``'s ``_candidate_seq`` are not thread-safe).
  * ``_sota_lock``  — the provisional-SOTA compare/coalesce section. The slow docker
                      hack-check NEVER runs under it (it runs at the barrier, on the
                      orchestrator thread, where solver threads have already joined).
  * ``_bb_lock``    — blackboard file assembly at the barrier.
"""

from __future__ import annotations

import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .agent_system import AgentSystem, _read_json, _read_lines
# Imported at module scope so tests can monkeypatch
# ``concurrent_agent_system.DockerContainer`` / ``.ControlSocket`` with fakes.
from .container import ControlSocket, DockerContainer


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


_NOTE_REPAIR_PROMPT = (
    "You are SOLVER {idx}. Your previous turn (generation {gen}) ended WITHOUT writing "
    "NOTES_FOR_PEERS.md, which the other solvers rely on. Do NOT change any solution or "
    "code and do NOT re-run experiments — your candidate for this generation is already "
    "recorded (score = {score}). Your ONLY task now: read your own scratchpad.md and "
    "solution_out.json in this directory and write a short, concrete NOTES_FOR_PEERS.md "
    "for your peers — what you tried this generation, your best score, and the single "
    "most useful trick or dead end. Keep it to a few sentences. Then exit."
)


@dataclass
class SolverSlot:
    """One concurrent solver: its workspace, control socket, and live container."""

    idx: int
    ws: Path
    control: object       # ControlSocket (or a test fake)
    container: object     # DockerContainer (or a test fake)


@dataclass
class ConcurrentAgentSystem:
    """N concurrent Solver agents over one shared, evolving black box (strong mode)."""

    base: AgentSystem
    concurrency: int = 4
    gen_turn_s: float = 900.0
    max_generations: int = 8

    _eval_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _store_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _bb_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _sota_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    # The BAR = the confirmed-clean broadcast best all solvers race. -inf until the
    # first SOTA clears its hack-check.
    _bar: float = field(default=float("-inf"), init=False)
    _bar_holder: Optional[int] = field(default=None, init=False)
    # The best candidate THIS generation not yet hack-checked: (score, payload, who).
    _provisional: Optional[tuple] = field(default=None, init=False)
    _gen: int = field(default=0, init=False)
    _resumed: bool = field(default=False, init=False)
    # This generation's per-solver score (idx -> score|None), set at collect time and
    # reused when repairing a missing note so the repair prompt can cite the score.
    _gen_scores: dict = field(default_factory=dict, init=False)
    # Tight cap for a note-only repair turn (also bounded by base.harden_timeout_s so it
    # never outlasts the Supervisor hack-check it overlaps).
    _note_repair_timeout_s: float = 180.0

    # -- thin proxies to base-owned state --------------------------------
    @property
    def store(self):
        return self.base.store

    @property
    def svc(self):
        return self.base.eval_service

    @property
    def deadline(self):
        return self.base.deadline

    def _bar_or_none(self) -> Optional[float]:
        return self._bar if self._bar != float("-inf") else None

    # -- top-level entry --------------------------------------------------
    def run(self, *, max_generations: Optional[int] = None,
            resume: bool = False) -> "ConcurrentAgentSystem":
        self.base.preflight()
        mg = self.max_generations if max_generations is None else max_generations
        if resume and self.base.can_resume():
            self._resumed = True
            self.store.event("run_start", mode="concurrent_agent_system",
                             resumed=True, concurrency=self.concurrency)
            self.base._resume_from_disk()
            self._resume_concurrent_state()
        else:
            self.store.write_manifest({
                "mode": "concurrent_agent_system",
                "raw_input_dir": str(self.base.raw_input_dir),
                "budget_s": self.base.budget_s,
                "image": self.base.image,
                "concurrency": self.concurrency,
                "initial_feedback_level": self.base.feedback_level.value,
            })
            self.store.event("run_start", mode="concurrent_agent_system",
                             budget_s=self.base.budget_s, concurrency=self.concurrency)
            self.base.bootstrap()
        self.solve_and_evolve_concurrent(max_generations=mg)
        return self

    def summary(self) -> dict:
        s = self.base.summary()
        s.update(concurrency=self.concurrency, generations=self._gen,
                 final_bar=self._bar_or_none())
        return s

    # ================================================================
    # the generation loop — synchronous barrier; harden = generation edge
    # ================================================================
    def solve_and_evolve_concurrent(self, *, max_generations: int) -> None:
        assert self.base.gateway and self.base.agent_elf and self.svc
        slots = [self._make_slot(i) for i in range(self.concurrency)]
        try:
            start_gen = self._gen
            while not self.deadline.expired() and (self._gen - start_gen) < max_generations:
                if self.deadline.remaining() <= self.base.min_solver_turn_s:
                    break
                self._gen += 1
                self.store.event("generation_start", gen=self._gen,
                                 bar=self._bar_or_none(),
                                 remaining_s=round(self.deadline.remaining(), 1))
                # (A) refresh every solver's context from the shared blackboard.
                for s in slots:
                    self._inject_context(s)
                # (B) reserve harden + one future generation behind this one.
                budget = self._gen_turn_budget()
                # (C) run the generation and JOIN — the barrier.
                with ThreadPoolExecutor(max_workers=len(slots)) as ex:
                    futs = [ex.submit(self._run_one_turn, s, budget) for s in slots]
                    for f in futs:
                        try:
                            f.result()
                        except Exception as e:   # one solver's failure ≠ generation death
                            self.store.event("solver_turn_error", error=str(e)[:200])
                # (D) converge: score each solver's best + harvest peer notes.
                missing = self._collect_blackboard(slots)
                # (E) hack-check the coalesced SOTA AND repair any missing peer notes
                #     CONCURRENTLY: the solver containers are idle during the Supervisor's
                #     harden one-shot, so a note-only repair turn overlaps it for free.
                with ThreadPoolExecutor(max_workers=2) as ex:
                    fut_repair = ex.submit(self._repair_missing_notes, missing)
                    moved = self._resolve_generation_supervision(slots)
                    try:
                        fut_repair.result()
                    except Exception as e:
                        self.store.event("note_repair_error", gen=self._gen,
                                         error=str(e)[:200])
                # (F) a modality switch re-seeds ALL N workspaces before the next gen.
                if moved and self.base._pending_mode_refresh:
                    for s in slots:
                        self.base._refresh_solver_ws(s.ws)
                    self.base._pending_mode_refresh = False
                self.store.event(
                    "generation_done", gen=self._gen,
                    best_score=(self.base._best_score
                                if self.base._best_score != float("-inf") else None),
                    bar=self._bar_or_none(),
                    verifier_version=self.svc.current_version())
        finally:
            for s in slots:
                try:
                    s.container.stop()
                except Exception:
                    pass
                try:
                    s.control.stop()
                except Exception:
                    pass
            self.store.event("solver_containers_stop", n=len(slots))
        self.base._finalize()

    def _gen_turn_budget(self) -> float:
        """Per-solver wall-clock for one generation, keeping a harden in reserve.

        Adapts the weak-mode normal-turn budgeting (agent_system.py:596): leave enough
        for a harden after the join so even the LAST generation before the deadline can
        still be checked/hardened. Floored at ``min_solver_turn_s``."""
        remaining = self.deadline.remaining()
        budget = min(self.gen_turn_s, remaining - self.base.harden_timeout_s)
        if budget < self.base.min_solver_turn_s:
            budget = self.base.min_solver_turn_s
        return budget

    def _run_one_turn(self, slot: SolverSlot, budget: float):
        """One solver's turn this generation. Runs on a worker thread."""
        session = slot.container.exec_agent(
            self._solver_prompt(slot.idx, self._gen), timeout_s=budget)
        with self._store_lock:
            self.store.cost(who=f"solver_{slot.idx}", kind="agent_turn", calls=1,
                            ok=getattr(session, "ok", False), gen=self._gen)
        return session

    def _repair_missing_notes(self, missing: list[SolverSlot]) -> None:
        """Re-run the solvers that produced no ``NOTES_FOR_PEERS.md`` this generation on
        a SHORT note-only turn, then re-harvest their notes into the blackboard.

        This is deliberately scheduled to overlap the Supervisor hack-check (see the
        barrier): the solver containers are idle while the Supervisor runs its own
        ``harden_ws`` one-shot, so repairing here costs no extra wall-clock. The repair
        turn is capped tight and MUST NOT touch the solution — it only writes the note."""
        if not missing:
            return
        budget = min(self.base.harden_timeout_s, self._note_repair_timeout_s)
        with ThreadPoolExecutor(max_workers=len(missing)) as ex:
            futs = {ex.submit(self._run_repair_turn, s, budget): s for s in missing}
            for f in futs:
                s = futs[f]
                try:
                    f.result()
                except Exception as e:
                    self.store.event("note_repair_error", idx=s.idx,
                                     gen=self._gen, error=str(e)[:200])
        # Re-harvest ONLY the repaired solvers; overwrite their stub index entries.
        repaired = []
        for s in missing:
            entry = self._harvest_note(s, score=self._gen_scores.get(s.idx))
            repaired.append(entry)
            self.store.event("solver_note_repaired", idx=s.idx, gen=self._gen,
                             ok=entry["has_note"])
        self._write_blackboard(repaired)

    def _run_repair_turn(self, slot: SolverSlot, budget: float):
        """A short, solution-frozen turn whose ONLY job is to write the missing note."""
        session = slot.container.exec_agent(
            _NOTE_REPAIR_PROMPT.format(idx=slot.idx, gen=self._gen,
                                       score=self._gen_scores.get(slot.idx)),
            timeout_s=budget)
        with self._store_lock:
            self.store.cost(who=f"solver_{slot.idx}", kind="note_repair", calls=1,
                            ok=getattr(session, "ok", False), gen=self._gen)
        return session

    # ================================================================
    # provisional-then-broadcast SOTA + coalesced Supervisor trigger
    # ================================================================
    def _offer_sota(self, score: Optional[float], payload: dict, who: int) -> None:
        """A solver's candidate beat the current bar — record it as PROVISIONAL.

        Never broadcasts and never invokes the Supervisor mid-generation: the N offers
        coalesce to the single best, resolved once at the barrier."""
        with self._sota_lock:
            if score is None or score <= self._bar:
                return
            if self._provisional is None or score > self._provisional[0]:
                self._provisional = (score, payload, who)
                with self._store_lock:
                    self.store.event("sota_provisional", who=f"solver_{who}",
                                     score=score, bar=self._bar_or_none(), gen=self._gen)

    def _resolve_generation_supervision(self, slots: list[SolverSlot]) -> bool:
        """At the barrier: hack-check the single coalesced provisional SOTA (if any).

        Returns True iff the hack-check moved V (hardened). No provisional ⇒ the
        Supervisor is NOT invoked this generation (the user's rule)."""
        prov = self._provisional
        self._provisional = None
        if prov is None:
            return False
        score, payload, who = prov
        before = self.svc.current_version()
        with self._eval_lock:
            self.base._run_supervisor_harden(payload, slots[0].ws,
                                             trigger="sota_hackcheck")
        moved = self.svc.current_version() > before
        if moved:
            # HACKED: V hardened. Recompute the bar HONESTLY under the new V — the
            # provisional score is stale/tainted. Reset first so a stale high best
            # from the old V cannot survive the re-score.
            self.base._best_score = float("-inf")
            self.base._best_payload = None
            self.base._recover_best_from_disk()
            self._bar = self.base._best_score
            self.store.event("sota_rejected_hardened", who=f"solver_{who}",
                             prov_score=score,
                             new_version=self.svc.current_version(),
                             bar=self._bar_or_none(), gen=self._gen)
        else:
            # CLEAN (the Supervisor may have installed NOTHING): broadcast as the bar.
            self.base._best_score, self.base._best_payload = score, payload
            self._bar = score
            self._bar_holder = who
            self.store.event("sota_broadcast", who=f"solver_{who}", score=score,
                             gen=self._gen)
        return moved

    # ================================================================
    # blackboard: converge (barrier) + inject (top of generation)
    # ================================================================
    def _collect_blackboard(self, slots: list[SolverSlot]) -> list[SolverSlot]:
        """Score each solver's ``solution_out.json`` under the CURRENT V, archive it as
        a candidate, feed the SAME ``_offer_sota`` pipeline (so an end-of-gen best is
        SOTA-eligible even if it was never offered mid-turn), and harvest peer notes.

        Every solver leaves a blackboard entry EACH generation, note or not: the full
        note text (when written) is stored under ``run_dir/peer_notes/`` and the entry
        carries only INDEX metadata (who / gen / score / path / has_note). A solver that
        skipped its note is still recorded (``has_note=False``) so peers can see it was
        silent rather than absent.

        Returns the slots that produced NO note this generation — the caller repairs
        them (a short note-only turn) IN PARALLEL with the Supervisor hack-check."""
        notes_dir = self.base.run_dir / "peer_notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        new_notes, missing = [], []
        for s in slots:
            payload = _read_json(s.ws / "solution_out.json", default=None)
            score = None
            if isinstance(payload, dict):
                with self._eval_lock:
                    r = self.svc.query(payload, who=f"solver_{s.idx}")
                score = r.score
                with self._store_lock:
                    self.store.candidate(
                        payload, {"score": r.score,
                                  "verifier_version": r.verifier_version},
                        score=r.score)
                if r.ok and r.score is not None:
                    self._offer_sota(r.score, payload, s.idx)
            self._gen_scores[s.idx] = score
            entry = self._harvest_note(s, score=score)
            new_notes.append(entry)
            if not entry["has_note"]:
                missing.append(s)
                self.store.event("solver_note_missing", idx=s.idx, gen=self._gen)
        self._write_blackboard(new_notes)
        return missing

    def _harvest_note(self, slot: SolverSlot, *, score) -> dict:
        """Read a solver's ``NOTES_FOR_PEERS.md``; if present, store the full text as a
        file under ``peer_notes/`` and return an INDEX entry pointing at it. If absent,
        return an entry flagged ``has_note=False`` (still a trace for peers)."""
        notes_dir = self.base.run_dir / "peer_notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        note = _read_text(slot.ws / "NOTES_FOR_PEERS.md").strip()
        fname = f"solver_{slot.idx}_gen{self._gen}.md"
        has_note = bool(note)
        if has_note:
            (notes_dir / fname).write_text(note, encoding="utf-8")
        return {"who": f"solver_{slot.idx}", "gen": self._gen, "score": score,
                "has_note": has_note, "path": f"peer_notes/{fname}"}

    def _write_blackboard(self, new_notes: list[dict]) -> None:
        with self._bb_lock:
            bb_path = self.base.run_dir / "blackboard.json"
            prev = _read_json(bb_path, default={})
            notes = prev.get("notes", []) if isinstance(prev, dict) else []
            if not isinstance(notes, list):
                notes = []
            # A repaired note re-emitted for the same (who, gen) supersedes the earlier
            # has_note=False stub — index by that key so the repair overwrites it.
            by_key = {}
            for n in notes:
                if isinstance(n, dict):
                    by_key[(n.get("who"), n.get("gen"))] = n
            for n in new_notes:
                by_key[(n.get("who"), n.get("gen"))] = n
            merged = sorted(by_key.values(),
                            key=lambda n: (n.get("gen") or 0, str(n.get("who"))))
            merged = merged[-50:]   # bound the tail
            bb = {
                "gen": self._gen,
                "bar": self._bar_or_none(),
                "bar_holder": (f"solver_{self._bar_holder}"
                               if self._bar_holder is not None else None),
                "notes": merged,
            }
            bb_path.write_text(json.dumps(bb, indent=2), encoding="utf-8")

    def _inject_context(self, slot: SolverSlot) -> None:
        """Render the shared blackboard INDEX into the solver's ``BLACKBOARD.md`` and
        copy the peer-note files into ``slot.ws/peer_notes/`` so the agent can READ the
        ones it cares about — we point at where each note lives rather than dumping every
        note's full text into the prompt/context."""
        bb = _read_json(self.base.run_dir / "blackboard.json", default={})
        notes = bb.get("notes", []) if isinstance(bb, dict) else []
        # Mirror the canonical peer_notes/ dir into this solver's ws (its only mount).
        src_dir = self.base.run_dir / "peer_notes"
        dst_dir = slot.ws / "peer_notes"
        dst_dir.mkdir(parents=True, exist_ok=True)
        if src_dir.is_dir():
            for f in src_dir.glob("*.md"):
                (dst_dir / f.name).write_text(
                    f.read_text(encoding="utf-8"), encoding="utf-8")
        lines = [f"# Blackboard — generation {self._gen}", ""]
        lines.append(f"Current BAR (clean, Supervisor-verified best to beat): "
                     f"{self._bar_or_none()}")
        lines.append("")
        if isinstance(notes, list) and notes:
            lines.append("## Peer notes index (read the files under ./peer_notes/)")
            lines.append("")
            for n in notes[-24:]:
                if not isinstance(n, dict):
                    continue
                if n.get("has_note"):
                    lines.append(f"- {n.get('who')} gen{n.get('gen')} "
                                 f"score={n.get('score')} → ./{n.get('path')}")
                else:
                    lines.append(f"- {n.get('who')} gen{n.get('gen')} "
                                 f"score={n.get('score')} (no note this generation)")
        (slot.ws / "BLACKBOARD.md").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")

    def _solver_prompt(self, idx: int, gen: int) -> str:
        """The weak-mode solver prompt + a strong-mode concurrency header."""
        base_prompt = self.base._solver_prompt(gen)
        header = (
            f"You are SOLVER {idx} of {self.concurrency} — one of several agents "
            f"attacking the SAME hidden verifier CONCURRENTLY (generation {gen}). Read "
            "BLACKBOARD.md in this directory FIRST: it holds the current BAR (a CLEAN, "
            f"Supervisor-verified best = {self._bar_or_none()}) and an INDEX of your "
            "peers' notes. The notes themselves are files under ./peer_notes/ — open the "
            "ones whose score or author looks worth learning from; do not assume the "
            "index lines are the whole note. Your job is to BEAT the bar.\n"
            "MANDATORY: before your turn ends you MUST (over)write NOTES_FOR_PEERS.md in "
            "this directory with a short, concrete note for the other solvers — what you "
            "tried, your best score, and the key trick or dead end. This is required "
            "every generation even if progress was small; peers rely on it. "
            "A RISING verifier_version means the Supervisor hardened the hidden verifier "
            "— re-baseline against it.\n\n")
        return header + base_prompt

    # ================================================================
    # N solver containers
    # ================================================================
    def _make_slot(self, i: int) -> SolverSlot:
        ws = self.base.run_dir / f"solver_ws_{i}"
        ws.mkdir(parents=True, exist_ok=True)
        self.base._refresh_solver_ws(ws)
        control = ControlSocket(workdir=ws, handler=self._make_shim(i))
        control.seed_shims(ws)
        control.start()
        # If the socket had to bind outside the workdir (AF_UNIX path too long), hand
        # the container the real bind path so it mounts it at /work/.control.sock.
        sock_mount = control.host_path if control.needs_explicit_mount else None
        sol = self.base.resource_spec.solver
        name = f"{self.base.run_dir.name}_solver_{i}"
        if self._resumed:
            # A crashed run may have left a same-named container; clear it before start.
            self._rm_stale_container(name)
        container = DockerContainer(
            workdir=ws, gateway=self.base.gateway, agent_elf=self.base.agent_elf,
            image=self.base.solver_image or self.base.image, gpus=self.base.solver_gpus,
            cpus=sol.cpus, memory_mb=sol.memory_mb, allow_internet=sol.allow_internet,
            control_sock_host_path=sock_mount, name=name)
        container.start()
        self.store.event("solver_container_start", idx=i, name=name,
                         image=self.base.solver_image or self.base.image,
                         gpus=self.base.solver_gpus, cpus=sol.cpus,
                         memory_mb=sol.memory_mb, allow_internet=sol.allow_internet)
        return SolverSlot(idx=i, ws=ws, control=control, container=container)

    @staticmethod
    def _rm_stale_container(name: str) -> None:
        try:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True,
                           text=True, timeout=30)
        except Exception:
            pass   # best-effort only

    def _make_shim(self, i: int):
        """A per-solver socket handler shaped like ``AgentSystem._handle_shim`` but
        thread-safe (locks) and routing bests through the provisional-SOTA pipeline —
        it NEVER writes ``base._best_*`` directly."""
        who = f"solver_{i}"

        def handler(cmd: str, arg: dict) -> dict:
            svc = self.svc
            if cmd == "eval":
                sol = arg.get("solution", {}) or {}
                with self._eval_lock:
                    r = svc.query(sol, who=who)
                with self._store_lock:
                    self.store.cost(who=who, kind="eval_query", calls=1)
                if r.ok and r.score is not None:
                    self._offer_sota(r.score, sol, i)
                return {"ok": r.ok, "score": r.score, "feasible": r.feasible,
                        "artifacts": r.artifacts, "verifier_version": r.verifier_version,
                        "feedback_level": r.feedback_level, "detail": r.detail}
            if cmd == "status":
                return {"verifier_version": svc.current_version(),
                        "feedback_level": svc.feedback_level.value,
                        "deadline_remaining_s": round(self.deadline.remaining(), 1),
                        "bar": self._bar_or_none()}
            if cmd == "ask":
                return {"ok": True,
                        "instruction": ("write your best solution and your question to "
                                        "review_request.json (as {\"solution\":...,"
                                        "\"question\":...}) and EXIT; the supervisor will "
                                        "review at the generation boundary and you will "
                                        "resume next generation with a fresh verifier")}
            return {"ok": False, "error": f"unknown cmd {cmd!r}"}

        return handler

    # ================================================================
    # resume (reuses base.can_resume / base._resume_from_disk unchanged)
    # ================================================================
    def _resume_concurrent_state(self) -> None:
        """Restore the bar + generation counter from disk after the base resume.

        The provisional is intentionally NOT persisted — an unconfirmed provisional is
        correctly discarded on crash."""
        bb = _read_json(self.base.run_dir / "blackboard.json", default={})
        bar = bb.get("bar") if isinstance(bb, dict) else None
        if bar is not None:
            try:
                self._bar = float(bar)
            except (TypeError, ValueError):
                self._bar = float("-inf")
        if self._bar == float("-inf") and self.base._best_score != float("-inf"):
            self._bar = self.base._best_score
        self._gen = self._recover_gen_from_events()

    def _recover_gen_from_events(self) -> int:
        gen = 0
        for ln in _read_lines(self.base.run_dir / "events.jsonl"):
            if '"generation_done"' not in ln:
                continue
            try:
                ev = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if ev.get("kind") == "generation_done":
                g = ev.get("gen")
                if isinstance(g, int) and g > gen:
                    gen = g
        return gen
