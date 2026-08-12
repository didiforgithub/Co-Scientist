# Co-Scientist coevo — Web UI (Task 2)

A **stdlib-only** web dashboard for observing the co-evolution multi-agent system
and a **human-review control plane** (the web equivalent of
`demo.human_port.CliHuman`). No flask/fastapi, no npm, no CDN, no external network
— the server is `http.server`, the UI is one self-contained HTML page that only
talks to its own localhost server.

This package is **purely additive**: it never modifies existing source. It READS
the on-disk `runs/<id>/` trees produced by `coevo.store.RunStore` and exposes
control endpoints; it never handles API keys.

## Launch

```bash
PYTHONPATH=. python3 -m coscientist.ui.server --runs-dir runs --host 127.0.0.1 --port 8765
# then open http://127.0.0.1:8765/
```

Flags: `--runs-dir` (default `runs`), `--host` (default `127.0.0.1`),
`--port` (default `8765`).

## What it shows

- **Run list** (left) with a LIVE badge (heuristic: `events.jsonl` touched within
  180s and no terminal `run_stop`), mode, current verifier version, hardenings,
  best score, last event.
- **Per-run detail**: stat tiles, a **score-trajectory** SVG sparkline (no chart
  library), a **mode indicator** (construction vs proof) that flips on a
  `mode_switch`, a **verifier version ladder** v0→vN (click a version to view its
  `verifier.py` and any `v{n}.feedback.py` sibling), a **co-evolution timeline**
  (icons for bootstrap/solve/harden/plateau/mode_switch/…), a **candidates**
  table, and eval-query count.
- **Human review panel**: pending `ReviewRequest`s with APPROVE / REJECT / GUIDE /
  REPLACE controls (GUIDE opens a dense-guidance textarea; REPLACE opens a
  replacement-source textarea). Submitting POSTs a `ReviewResponse`.
- Auto-refreshes every 2.5s via polling (no websockets).

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | the dashboard (served from `index.html`) |
| GET | `/api/runs` | summaries of every run under `--runs-dir` |
| GET | `/api/runs/<run_id>` | full assembled detail for one run |
| GET | `/api/runs/<run_id>/verifier/<n>` | source of `v{n}.py` (+ feedback sibling) |
| GET | `/api/reviews/pending` | pending human reviews (from `WebHumanPort`) |
| GET | `/api/reviews/history` | resolved reviews (audit trail) |
| POST | `/api/reviews/<review_id>` | `{decision, dense_text, replacement_src}` — unblocks the orchestrator |
| POST | `/api/reviews/_demo` | enqueue a synthetic pending review (manual testing) |

All read endpoints tolerate partial/growing files (a live run is being written) —
JSONL is parsed line-by-line and a corrupt trailing line is skipped.

## The `WebHumanPort` seam

`web_human_port.WebHumanPort` implements the `HumanPort` protocol
(`review(req) -> ReviewResponse`). On `review()` it enqueues a pending review into
a shared, thread-safe `ReviewRegistry` and **blocks on a `threading.Event`** until
the web side POSTs a decision, then returns the matching `ReviewResponse`. A
configurable `timeout_s` guarantees a headless run never hangs — with no human it
returns `default_decision` (default `GUIDE`, empty text = a no-op on the verifier).

The server holds a **process-wide registry** exposed via `server.get_registry()`;
construct the `WebHumanPort` against that same instance so the control-plane
endpoints unblock it.

### Wiring a future `--supervisor web` mode (NOT done here — touches existing source)

This UI does not modify the orchestrator. To let a live agent-system run route its
`ReviewRequest`s to the web, a later change would:

1. **`coscientist/coevo/cli.py`** — add `"web"` to the `--supervisor` choices
   (line ~237) and, in the agent-system path, build a `WebHumanPort` against
   `coscientist.ui.server.get_registry()`, start the server thread, and pass the
   port down to `AgentSystem`.
2. **`coscientist/coevo/agent_system.py`** — the current harden path is fully
   autonomous (`_run_supervisor_harden` / `_apply_harden`, lines ~623–742): it
   never calls a `HumanPort`. Add an optional `human_port: Optional[HumanPort] =
   None` field on `AgentSystem`, and in `_apply_harden`, before `install_verifier`
   (line ~720), call `human_port.review(ReviewRequest(...))` assembled from the
   verdict + `_probe_report()` + proposed `verify_src`, then branch on the
   returned `Decision` (APPROVE = install as-is; REPLACE = install
   `replacement_src`; REJECT = keep current; GUIDE/NOOP = record `dense_text`).

That is the only integration surface. Until then, the control plane is exercised
standalone (see `tests/test_ui.py` and the `/api/reviews/_demo` endpoint).
