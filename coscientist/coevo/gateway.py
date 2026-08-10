"""The gateway (§6) — the boundary a subprocess Solver reaches eval + Supervisor through.

A real coding-agent Solver (Codex, Claude Code) runs as an opaque subprocess in
its solution container. It cannot hold Python channel objects, so the two channels
are exposed to it as a **local HTTP endpoint** it calls via CLI shims:

    POST /query   {"solution": {...}}      -> score + Supervisor-chosen feedback
    POST /review  {"solution": {...}}      -> Supervisor hack-check verdict
    GET  /status                            -> current verifier_version, deadline

This is exactly the same two channels as the in-process handles (``EvalClient`` /
``SupervisorChannel``), just over a socket. Two properties matter:

  * The Solver perceives V move through the ``verifier_version`` in each /query
    response — the "perceives the score shift through the API" mechanism of §4.
    No push needed: the subprocess pulls, and a changed version == re-baseline.
  * COST IS RECORDED HERE (§6), at the gateway, not from the agent's self-report:
    every /query and /review logs a cost line. The agent's cooperation is not
    required for the accounting to be complete.

Pure stdlib (``http.server``), binds 127.0.0.1 on an ephemeral port, single
worker thread. Start it, read ``.base_url``, hand that to the subprocess, stop it
when the session ends.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .budget import Deadline
from .channels import ReviewKind, ReviewMessage
from .eval_service import EvalService
from .supervisor import Supervisor


@dataclass
class Gateway:
    """A local HTTP boundary onto one eval service + one supervisor.

    ``store`` (RunStore or None) receives a cost line per call. ``deadline`` lets
    ``/status`` report remaining wall clock so the agent can pace itself.
    """

    eval: EvalService
    supervisor: Supervisor
    deadline: Optional[Deadline] = None
    store: Optional[object] = None
    host: str = "127.0.0.1"
    port: int = 0                                  # 0 => OS picks an ephemeral port

    _server: Optional[ThreadingHTTPServer] = field(default=None, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    # One lock serialises every touch of the eval service + supervisor. The HTTP
    # server is threaded and a supervision thread red-teams/hardens V in parallel,
    # so evaluator.run() and evaluator.evolve() must not interleave. The driver's
    # threaded path acquires this SAME lock around its supervision tick.
    lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    queries: int = field(default=0, init=False)
    reviews: int = field(default=0, init=False)

    @property
    def base_url(self) -> str:
        assert self._server is not None, "gateway not started"
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "Gateway":
        gw = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):  # silence stdlib request logging
                pass

            def _send(self, code: int, obj: dict) -> None:
                body = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict:
                n = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(n) if n else b"{}"
                try:
                    return json.loads(raw.decode("utf-8") or "{}")
                except json.JSONDecodeError:
                    return {}

            def do_GET(self):
                if self.path.rstrip("/") == "/status":
                    self._send(200, gw._status())
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                path = self.path.rstrip("/")
                body = self._read_json()
                if path == "/query":
                    self._send(200, gw._handle_query(body))
                elif path == "/review":
                    self._send(200, gw._handle_review(body))
                else:
                    self._send(404, {"error": "not found"})

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "Gateway":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- handlers ---------------------------------------------------------
    def _status(self) -> dict:
        rem = None if self.deadline is None else round(self.deadline.remaining(), 2)
        return {
            "verifier_version": self.eval.current_version(),
            "feedback_level": self.eval.feedback_level.value,
            "deadline_remaining_s": rem,
        }

    def _handle_query(self, body: dict) -> dict:
        solution = body.get("solution", {})
        with self.lock:
            r = self.eval.query(solution, who=body.get("who", "solver"))
        self.queries += 1
        if self.store is not None:
            self.store.cost(who="solver", kind="eval_query", calls=1)
        return {
            "ok": r.ok, "score": r.score, "feasible": r.feasible,
            "artifacts": r.artifacts, "detail": r.detail,
            "verifier_version": r.verifier_version, "feedback_level": r.feedback_level,
        }

    def _handle_review(self, body: dict) -> dict:
        solution = body.get("solution", {})
        with self.lock:
            # score it once so the verdict carries the current proxy score, then
            # judge + (gate-free) harden — all under the lock so a concurrent
            # /query never observes a half-installed V.
            r = self.eval.query(solution, who="solver-review-score")
            verdict = self.supervisor.handle_review(ReviewMessage(
                kind=ReviewKind.HACK_CHECK, best_payload=solution,
                best_score=r.score, note=body.get("note", ""),
            ))
        self.reviews += 1
        if self.store is not None:
            self.store.cost(who="solver", kind="supervisor_review", calls=1)
        return {
            "gaming": verdict.gaming, "text": verdict.text,
            "verifier_version": self.eval.current_version(),
        }
