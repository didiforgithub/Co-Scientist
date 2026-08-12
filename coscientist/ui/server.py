"""Stdlib-only HTTP server for the Co-Scientist run dashboard + human-review plane.

STDLIB ONLY (http.server, socketserver, json, threading, argparse) — no flask,
no fastapi, no third-party anything, so it starts on a bare box with no installs.

Routes
------
  GET  /                         -> the single-page dashboard (index.html sibling)
  GET  /api/runs                 -> [summary, ...] for every run under runs-dir
  GET  /api/runs/<run_id>        -> full assembled detail for one run
  GET  /api/runs/<run_id>/verifier/<n>  -> source of v{n}.py (+ feedback sibling)
  GET  /api/reviews/pending      -> pending human ReviewRequests (from WebHumanPort)
  GET  /api/reviews/history      -> resolved reviews (audit trail)
  POST /api/reviews/<review_id>  -> {decision, dense_text, replacement_src}; unblocks
  POST /api/reviews/_demo        -> enqueue a synthetic pending review (for testing)

The review registry is a process-wide singleton shared with any ``WebHumanPort``
constructed against ``get_registry()`` — that is the seam a future ``--supervisor
web`` mode wires into. Read endpoints tolerate partial/growing files.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from . import run_reader
from .review_registry import ReviewRegistry

_HERE = Path(__file__).resolve().parent

# Process-wide shared registry: the server's control-plane endpoints and any
# WebHumanPort share THIS instance so review() blocks until a POST arrives.
_REGISTRY = ReviewRegistry()


def get_registry() -> ReviewRegistry:
    """The process-wide ReviewRegistry a WebHumanPort should be constructed against."""
    return _REGISTRY


def _load_index_html() -> str:
    """Serve the sibling index.html; fall back to a minimal page if it's missing."""
    idx = _HERE / "index.html"
    if idx.is_file():
        return idx.read_text(encoding="utf-8")
    return "<!doctype html><meta charset=utf-8><title>Co-Scientist UI</title>" \
           "<p>index.html not found next to server.py.</p>"


class _Handler(BaseHTTPRequestHandler):
    server_version = "CoScientistUI/1.0"
    # injected by make_server via the server instance
    runs_dir: Path = Path("runs")

    # -- helpers ----------------------------------------------------------
    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, msg: str) -> None:
        self._send_json({"error": msg}, status=status)

    def _read_body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            n = 0
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            obj = json.loads(raw.decode("utf-8"))
            return obj if isinstance(obj, dict) else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    @property
    def _runs_dir(self) -> Path:
        return getattr(self.server, "runs_dir", Path("runs"))

    # quiet the default noisy logging
    def log_message(self, fmt, *args):  # noqa: A003
        pass

    # -- routing ----------------------------------------------------------
    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        parts = [p for p in path.split("/") if p]

        if path == "/":
            self._send_html(_load_index_html())
            return
        if parts == ["api", "runs"]:
            ids = run_reader.list_run_ids(self._runs_dir)
            self._send_json({"runs": [run_reader.summarize_run(self._runs_dir, r)
                                      for r in ids]})
            return
        if parts == ["api", "reviews", "pending"]:
            self._send_json({"pending": _REGISTRY.pending()})
            return
        if parts == ["api", "reviews", "history"]:
            self._send_json({"history": _REGISTRY.history()})
            return
        # /api/runs/<id>/verifier/<n>
        if len(parts) == 5 and parts[:2] == ["api", "runs"] and parts[3] == "verifier":
            run_id, ver = parts[2], parts[4]
            if not ver.isdigit():
                self._send_error_json(400, "version must be an integer")
                return
            data = run_reader.read_verifier_source(self._runs_dir, run_id, int(ver))
            if not data.get("exists"):
                self._send_error_json(404, f"verifier v{ver} not found for {run_id}")
                return
            self._send_json(data)
            return
        # /api/runs/<id>
        if len(parts) == 3 and parts[:2] == ["api", "runs"]:
            detail = run_reader.detail_run(self._runs_dir, parts[2])
            if detail is None:
                self._send_error_json(404, f"run not found: {parts[2]}")
                return
            self._send_json(detail)
            return
        self._send_error_json(404, f"no route for GET {path}")

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        parts = [p for p in path.split("/") if p]
        body = self._read_body()

        # POST /api/runs -> DOCUMENTED STUB (501). This is the "留接口" the user asked
        # for: the web-submit interface placeholder. It does NOT create/launch a run
        # this pass (data path + CLI override only). The response body documents the
        # intended submit contract so a future web-submit form knows the shape:
        #   {instruction, resource_spec:{solver{...},verifier{...}}, budget_s}
        # A submit would then map onto AgentSystem + ResourceSpec (isolated slices),
        # exactly as the CLI --resource-config / --solver-* / --verifier-* flags do.
        if parts == ["api", "runs"]:
            self._send_json({
                "error": "not implemented",
                "detail": "web submit is a stub this pass; use the CLI "
                          "(coscientist.coevo.cli --input ... --resource-config ...).",
                "expected_contract": {
                    "instruction": "<str: the raw problem / task instruction>",
                    "resource_spec": {
                        "solver": {"image": "<str?>", "cpus": "<float?>",
                                   "memory_mb": "<int?>", "gpus": "<str?>",
                                   "gpu_types": "<list[str]?>", "allow_internet": "<bool>"},
                        "verifier": {"image": "<str?>", "cpus": "<float?>",
                                     "memory_mb": "<int?>", "gpus": "<str?>",
                                     "gpu_types": "<list[str]?>",
                                     "timeout_sec": "<float?>", "allow_internet": "<bool>"},
                    },
                    "budget_s": "<float: wall-clock budget in seconds>",
                },
                "note": "Eval (verifier) and Solve (solver) resources are ISOLATED "
                        "slices — the central design requirement.",
            }, status=501)
            return
        # POST /api/reviews/_demo -> enqueue a synthetic pending review (test aid).
        if parts == ["api", "reviews", "_demo"]:
            timeout_s = float(body.get("timeout_s", 120.0))
            pr = _REGISTRY.enqueue(body.get("request") or _demo_request(),
                                   timeout_s=timeout_s)
            self._send_json({"review_id": pr.review_id, "enqueued": True})
            return
        # POST /api/reviews/<review_id> -> record a decision, unblock the waiter.
        if len(parts) == 3 and parts[:2] == ["api", "reviews"]:
            review_id = parts[2]
            decision = str(body.get("decision", "")).strip().lower()
            valid = {"approve", "reject", "guide", "replace", "noop"}
            if decision not in valid:
                self._send_error_json(400, f"decision must be one of {sorted(valid)}")
                return
            ok = _REGISTRY.submit(review_id, {
                "decision": decision,
                "dense_text": body.get("dense_text", ""),
                "replacement_src": body.get("replacement_src"),
            })
            if not ok:
                self._send_error_json(
                    404, f"no pending review {review_id!r} (already resolved/unknown)")
                return
            self._send_json({"ok": True, "review_id": review_id, "decision": decision})
            return
        self._send_error_json(404, f"no route for POST {path}")


def _demo_request() -> dict:
    """A synthetic ReviewRequest dict for the _demo endpoint / manual UI testing."""
    return {
        "tick": 0,
        "reason": "synthetic demo review (no real orchestrator attached)",
        "top_solutions": [{"summary": "example candidate", "proxy_score": 0.42,
                           "n_modes": 3}],
        "current_verifier_note": "demo v0",
        "proposed_verifier_src": "def verify(payload, ctx):\n    return {'feasible': True, 'raw': 1.0, 'artifacts': {}}\n",
        "evidence": {"note": "this is a demonstration request"},
    }


def make_server(runs_dir: Path, host: str, port: int) -> ThreadingHTTPServer:
    handler = _Handler
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.runs_dir = Path(runs_dir).resolve()  # type: ignore[attr-defined]
    return httpd


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Co-Scientist coevo web UI — visualize runs + human review plane")
    ap.add_argument("--runs-dir", default="runs", help="parent dir of run storage")
    ap.add_argument("--host", default="127.0.0.1", help="bind host (default localhost)")
    ap.add_argument("--port", type=int, default=8765, help="bind port (default 8765)")
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs_dir).resolve()
    httpd = make_server(runs_dir, args.host, args.port)
    print(f"Co-Scientist UI serving runs from {runs_dir}")
    print(f"  http://{args.host}:{args.port}/")
    print("  Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
