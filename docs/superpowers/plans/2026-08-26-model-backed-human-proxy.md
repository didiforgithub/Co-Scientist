# Model-Backed Human Proxy Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Replace the execution-backed scripted Human Proxy with a model agent that can read private evaluator context and converse like a human expert, but cannot run solutions or invoke the evaluator.

**Architecture:** The Co-side evidence agent keeps the existing durable Human Session contract. A separate `ModelBackedHumanProxyAgent` receives only a private textual evaluator context plus the durable transcript in a throw-away workspace and returns one structured natural-language turn at a time. The AgentSystem no longer builds comparison cases or imports/calls a reference evaluator; verifier changes are approved only through the proxy's reasoned multi-turn outcome.

**Tech Stack:** Python 3.10+, dataclasses/protocols, existing Codex container runner, pytest.

---

## Chunk 1: Model dialogue contract

### Task 1: Specify the non-executing model agent

**Files:**
- Modify: `tests/test_human_proxy_sessions.py`
- Modify: `coscientist/coevo/human_proxy_sessions.py`

- [x] Write failing tests proving the proxy runner receives private evaluator text and the transcript, produces non-scripted multi-turn messages, and never receives a solution/evaluator callable.
- [x] Run `pytest tests/test_human_proxy_sessions.py -q` and confirm failures are caused by the missing `ModelBackedHumanProxyAgent` contract.
- [x] Implement `ModelBackedHumanProxyAgent`, a strict JSON turn schema, a prompt that forbids execution claims, and a throw-away per-turn workspace.
- [x] Extend `HumanProxyTurn` with an optional final `SessionOutcome`; stage that outcome only at explicit close confirmation.
- [x] Re-run the focused tests and keep the five-session budget, close confirmation, pause/resume, and watchdog behavior green.

## Chunk 2: Remove V* execution

### Task 2: Make evaluator context text-only

**Files:**
- Modify: `tests/test_human_integration.py`
- Modify: `coscientist/coevo/agent_system.py`
- Modify: `coscientist/coevo/cli.py`

- [x] Write failing integration tests asserting `AgentSystem` accepts a private context file, constructs a model-backed proxy, does not import a Python evaluator, and omits `comparison_cases` from Human Session context.
- [x] Run the new integration tests and confirm they fail on the old execution-backed wiring.
- [x] Replace `human_proxy_evaluator_*` configuration with `human_proxy_context_path` and `--human-proxy-context`.
- [x] Remove `_human_comparison_cases` and all automatic current/proposed/V* evaluation calls.
- [x] Verify that non-approve outcomes still hold verifier changes and explicit approval still installs only after the conversation closes.

## Chunk 3: Documentation and verification

### Task 3: Document the corrected contract

**Files:**
- Modify: `docs/feishu-human-sessions.md`
- Modify: `README.md`

- [x] Document that Human Proxy is a separate model agent with private text context, no solution environment, no evaluator endpoint, and the same five-session lifecycle as a human.
- [x] Document the JSON turn contract and the distinction between Co-side run evidence and Proxy-side evaluator context.
- [x] Run `pytest -q tests/test_human_proxy_sessions.py tests/test_human_sessions.py tests/test_human_integration.py`.
- [x] Run the full project test suite with `pytest -q tests`.
- [x] Run `python -m compileall -q coscientist tests` and `git diff --check`.
