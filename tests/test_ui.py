"""Offline, hermetic tests for the web UI (coscientist/ui/).

Stdlib-only. Spins the real server on an ephemeral port against a tmp runs-dir we
populate to mimic the store.py layout, checks the read endpoints, and drives the
human-review round-trip end to end (WebHumanPort.review blocks in one thread; a
POST from another thread unblocks it and the returned ReviewResponse matches).

Adding this FILE does not disturb the existing suite.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from coscientist.demo.human_port import Decision, ReviewRequest
from coscientist.ui import server as ui_server
from coscientist.ui.review_registry import ReviewRegistry
from coscientist.ui.web_human_port import WebHumanPort


# --------------------------------------------------------------------------
# fixtures: a fake runs-dir mimicking coevo/store.py layout
# --------------------------------------------------------------------------
def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_fake_run(root: Path, run_id: str, *, hardened: bool) -> None:
    r = root / run_id
    (r / "supervisor" / "verifier_versions").mkdir(parents=True, exist_ok=True)
    (r / "solver" / "candidates").mkdir(parents=True, exist_ok=True)
    (r / "eval").mkdir(parents=True, exist_ok=True)

    _write(r / "manifest.json", json.dumps({
        "mode": "agent_system", "budget_s": 1800.0,
        "raw_input_dir": "/x/problems/" + run_id,
        "best_score": 0.75 if hardened else 0.5,
        "final_verifier_version": 1 if hardened else 0,
        "verifier_hardenings": 1 if hardened else 0,
        "final_mode": "construction",
        # isolated resource slices (only the hardened run records them, to prove the
        # UI tolerates both a present and an absent resource_spec).
        **({"resource_spec": {
            "solver": {"image": "s:img", "cpus": 4.0, "allow_internet": False},
            "verifier": {"image": "v:img", "cpus": 8.0, "gpus": "1",
                         "gpu_types": ["A100"], "timeout_sec": 1800.0,
                         "allow_internet": False},
        }} if hardened else {}),
    }))
    events = [
        {"t": 0.1, "kind": "run_start", "budget_s": 1800.0},
        {"t": 1.0, "kind": "bootstrap_done", "n_probes": 3},
        {"t": 2.0, "kind": "solver_turn_start", "turn": 1},
    ]
    if hardened:
        events += [
            {"t": 3.0, "kind": "review_request", "question": "is v0 gamed?"},
            {"t": 4.0, "kind": "harden_start", "trigger": "review_request"},
        ]
    # include a deliberately corrupt trailing line to exercise tolerant parsing
    lines = [json.dumps(e) for e in events] + ['{"t": 5.0, "kind": "trunc']
    _write(r / "events.jsonl", "\n".join(lines) + "\n")

    _write(r / "supervisor" / "verifier_versions" / "v0.py",
           "def verify(payload, ctx):\n    return {'feasible': True, 'raw': 0.5, 'artifacts': {}}\n")
    vmeta = [{"t": 1.0, "version": 0, "origin": "agent", "note": "supervisor bootstrap",
              "rationale": "seed", "has_feedback": False}]
    if hardened:
        _write(r / "supervisor" / "verifier_versions" / "v1.py",
               "def verify(payload, ctx):\n    return {'feasible': True, 'raw': 0.75, 'artifacts': {}}\n")
        _write(r / "supervisor" / "verifier_versions" / "v1.feedback.py",
               "def feedback(payload, ctx, verify_result, history):\n    return {'detail': 'try harder', 'artifacts': {}}\n")
        vmeta.append({"t": 4.0, "version": 1, "origin": "agent", "note": "harden (review_request)",
                      "rationale": "closed a hole", "has_feedback": True})
    _write(r / "supervisor" / "versions.jsonl",
           "\n".join(json.dumps(m) for m in vmeta) + "\n")

    _write(r / "solver" / "candidates" / "cand_00000.json", json.dumps({
        "id": "cand_00000", "t": 2.5, "score": 0.5,
        "payload": {"frequencies": [1, 3, 5], "amplitudes": [0.1, 0.2, 0.3]},
        "feedback": {"score": 0.5, "verifier_version": 0}}))
    _write(r / "solver" / "trajectory.jsonl", json.dumps({
        "t": 2.5, "event": "best_extracted", "best_score": 0.5, "verifier_version": 0}) + "\n")
    _write(r / "eval" / "queries.jsonl",
           json.dumps({"query_id": 0, "t": 0.5, "who": "solver",
                       "verifier_version": 0, "returned_score": 0.5}) + "\n")


@pytest.fixture()
def runs_dir(tmp_path: Path) -> Path:
    _make_fake_run(tmp_path, "alpha", hardened=False)
    _make_fake_run(tmp_path, "beta", hardened=True)
    return tmp_path


@pytest.fixture()
def live_server(runs_dir: Path):
    # fresh registry per test so pending reviews don't leak between tests
    ui_server._REGISTRY = ReviewRegistry()
    httpd = ui_server.make_server(runs_dir, "127.0.0.1", 0)  # ephemeral port
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    try:
        yield base
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(base: str, path: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(base + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:  # noqa: F821
        return e.code, json.loads(e.read().decode("utf-8"))


# --------------------------------------------------------------------------
# read endpoints
# --------------------------------------------------------------------------
def test_index_served(live_server):
    with urllib.request.urlopen(live_server + "/", timeout=5) as r:
        html = r.read().decode("utf-8")
    assert "Co-Scientist" in html and "<html" in html.lower()


def test_api_runs_lists_both(live_server):
    j = _get(live_server, "/api/runs")
    ids = {r["run_id"] for r in j["runs"]}
    assert ids == {"alpha", "beta"}
    beta = next(r for r in j["runs"] if r["run_id"] == "beta")
    assert beta["current_verifier_version"] == 1
    assert beta["hardenings"] == 1
    assert beta["best_score"] == 0.75
    assert beta["n_candidates"] == 1


def test_api_run_detail(live_server):
    d = _get(live_server, "/api/runs/beta")
    assert d["run_id"] == "beta"
    assert len(d["versions"]) == 2
    assert d["versions"][1]["has_feedback"] is True
    assert d["current_verifier_version"] == 1
    assert d["n_eval_queries"] == 1
    assert len(d["candidates"]) == 1
    # tolerant parsing: the corrupt trailing events line was skipped, not fatal
    kinds = [e["kind"] for e in d["events"]]
    assert "review_request" in kinds and "harden_start" in kinds


def test_verifier_source_endpoint(live_server):
    v = _get(live_server, "/api/runs/beta/verifier/1")
    assert v["exists"] is True and "def verify" in v["verifier_src"]
    assert v["has_feedback"] is True and "def feedback" in v["feedback_src"]


def test_missing_run_404(live_server):
    status, body = _post(live_server, "/api/runs/nope", {})  # wrong method anyway
    # GET the missing run:
    try:
        _get(live_server, "/api/runs/does_not_exist")
        assert False, "expected 404"
    except urllib.error.HTTPError as e:  # noqa: F821
        assert e.code == 404


# --------------------------------------------------------------------------
# human-review control plane round-trip
# --------------------------------------------------------------------------
def test_web_human_port_roundtrip(live_server):
    port_obj = WebHumanPort(ui_server.get_registry(), timeout_s=10.0)
    req = ReviewRequest(
        tick=7, reason="probe rewarded a shortcut",
        top_solutions=[{"summary": "n=2047", "proxy_score": 0.001, "n_modes": 1}],
        current_verifier_note="v0 partial-credit tier",
        proposed_verifier_src="def verify(p,c):\n    return {'feasible':True,'raw':0.0,'artifacts':{}}\n",
        evidence={"probe_3_score": 0.001})

    result = {}

    def call_review():
        result["resp"] = port_obj.review(req)

    th = threading.Thread(target=call_review)
    th.start()

    # wait for it to show up as pending
    review_id = None
    for _ in range(50):
        pend = _get(live_server, "/api/reviews/pending")["pending"]
        if pend:
            review_id = pend[0]["review_id"]
            assert pend[0]["request"]["tick"] == 7
            assert pend[0]["request"]["proposed_verifier_src"] is not None
            break
        time.sleep(0.05)
    assert review_id is not None, "review never became pending"

    status, body = _post(live_server, "/api/reviews/" + review_id, {
        "decision": "replace",
        "dense_text": "installing the honest-DOF fix",
        "replacement_src": "def verify(p,c):\n    return {'feasible':True,'raw':1.0,'artifacts':{}}\n"})
    assert status == 200 and body["ok"] is True

    th.join(timeout=5)
    assert not th.is_alive()
    resp = result["resp"]
    assert resp.decision == Decision.REPLACE
    assert resp.dense_text == "installing the honest-DOF fix"
    assert "raw':1.0" in resp.replacement_src


def test_web_human_port_timeout_default():
    reg = ReviewRegistry()
    port_obj = WebHumanPort(reg, timeout_s=0.1, default_text="nobody home")
    req = ReviewRequest(tick=1, reason="x", top_solutions=[],
                        current_verifier_note="v0", proposed_verifier_src=None,
                        evidence={})
    resp = port_obj.review(req)  # no one answers -> default GUIDE
    assert resp.decision == Decision.GUIDE
    assert resp.dense_text == "nobody home"


def test_submit_unknown_review_404(live_server):
    status, body = _post(live_server, "/api/reviews/rev_99999",
                         {"decision": "approve"})
    assert status == 404 and "error" in body


def test_replace_without_source_becomes_guide():
    """A REPLACE decision with no source degrades to GUIDE (never a no-op install)."""
    reg = ReviewRegistry()
    port_obj = WebHumanPort(reg, timeout_s=5.0)
    req = ReviewRequest(tick=1, reason="x", top_solutions=[],
                        current_verifier_note="v0", proposed_verifier_src=None,
                        evidence={})
    res = {}
    th = threading.Thread(target=lambda: res.update(r=port_obj.review(req)))
    th.start()
    for _ in range(50):
        if reg.pending():
            break
        time.sleep(0.02)
    rid = reg.pending()[0]["review_id"]
    reg.submit(rid, {"decision": "replace", "dense_text": "", "replacement_src": None})
    th.join(timeout=3)
    assert res["r"].decision == Decision.GUIDE


# --------------------------------------------------------------------------
# resource slices in the detail payload + web-submit interface stub
# --------------------------------------------------------------------------
def test_detail_surfaces_resource_spec(live_server):
    """The run-detail payload carries resource_spec (both isolated slices) top-level;
    a run without one gets an empty dict (backward compatible)."""
    beta = _get(live_server, "/api/runs/beta")
    rs = beta["resource_spec"]
    assert rs["solver"]["cpus"] == 4.0 and rs["solver"]["image"] == "s:img"
    assert rs["verifier"]["gpus"] == "1" and rs["verifier"]["timeout_sec"] == 1800.0
    assert rs["verifier"]["gpu_types"] == ["A100"]
    # a run that never recorded resources still yields a (empty) dict, not a KeyError.
    alpha = _get(live_server, "/api/runs/alpha")
    assert alpha["resource_spec"] == {}


def test_post_runs_is_documented_501_stub(live_server):
    """POST /api/runs is the '留接口' web-submit stub: 501 with the documented submit
    contract (instruction / resource_spec{solver,verifier} / budget_s). No run created."""
    status, body = _post(live_server, "/api/runs",
                         {"instruction": "solve X", "budget_s": 1800})
    assert status == 501
    contract = body["expected_contract"]
    assert "instruction" in contract and "budget_s" in contract
    assert set(contract["resource_spec"].keys()) == {"solver", "verifier"}
    # no new run appeared — this pass is data-path + CLI only.
    ids = {r["run_id"] for r in _get(live_server, "/api/runs")["runs"]}
    assert ids == {"alpha", "beta"}

