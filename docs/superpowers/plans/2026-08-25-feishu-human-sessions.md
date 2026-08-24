# Feishu Human Sessions Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every Co-Scientist run up to five scarce, durable human-expert conversations over Feishu, where each conversation remains open across natural-language turns until the human explicitly confirms closure.

**Architecture:** A transport-independent Human Session domain owns budget, lifecycle, transcript, deduplication, and closure confirmation. A read-only evidence agent prepares run evidence and answers expert questions. A `lark-cli` adapter provides direct-message send/reply and long-connection event ingestion. The live AgentSystem invokes the service only at explicit interaction points and freezes evaluator mutation while a session is open.

**Tech Stack:** Python 3 standard library, pytest, existing Co-Scientist Docker/Codex gateway, `lark-cli` Feishu adapter.

---

## Task 1: Durable session domain and five-session budget

**Files:**
- Create: `coscientist/coevo/human_sessions.py`
- Create: `tests/test_human_sessions.py`

- [x] Write failing tests for opening sessions, rejecting the sixth session, resuming without spending budget, durable transcript reload, event deduplication, and expert identity filtering.
- [x] Run `pytest -q tests/test_human_sessions.py` and confirm the tests fail for the missing implementation.
- [x] Implement `HumanSessionStore`, explicit lifecycle states, atomic metadata writes, append-only transcript, and `SessionBudgetExhausted`.
- [x] Run the focused tests and confirm they pass.
- [ ] Commit the domain layer as part of the audited offline implementation commit.

## Task 2: Natural close-confirmation protocol

**Files:**
- Modify: `coscientist/coevo/human_sessions.py`
- Modify: `tests/test_human_sessions.py`

- [x] Write failing tests proving close intent only enters `CLOSE_REQUESTED`, explicit confirmation closes, denial returns to `ACTIVE`, idle/ordinary text never closes, and a structured outcome is persisted.
- [x] Run the focused tests and confirm the new cases fail.
- [x] Implement deterministic close-intent and confirmation handling with human-visible confirmation copy.
- [x] Run the focused tests and confirm they pass.
- [ ] Commit the close protocol as part of the audited offline implementation commit.

## Task 3: Read-only run evidence and session agent

**Files:**
- Create: `coscientist/coevo/human_evidence.py`
- Create: `tests/test_human_evidence.py`

- [x] Write failing tests for bounded evidence snapshots, secret exclusion, transcript-aware prompts, and structured final outcomes.
- [x] Run `pytest -q tests/test_human_evidence.py` and confirm failure.
- [x] Implement a deterministic `RunEvidenceBuilder`, an `EvidenceAgent` protocol, and a production Codex-backed agent using the existing one-shot container gateway.
- [x] Run focused tests and confirm they pass.
- [ ] Commit the evidence layer as part of the audited offline implementation commit.

## Task 4: Feishu CLI transport and Human Session service

**Files:**
- Create: `coscientist/coevo/feishu_transport.py`
- Create: `coscientist/coevo/feishu_human.py`
- Create: `tests/test_feishu_human.py`

- [x] Write failing tests for safe argv construction, JSON response parsing, event-ready handshake, inbound event normalization, event deduplication, non-expert filtering, natural multi-turn routing, and crash-safe resume.
- [x] Run `pytest -q tests/test_feishu_human.py` and confirm failure.
- [x] Implement `LarkCliTransport` without shell execution and a synchronous Human Session service that sends a root DM, replies to each expert message, and resumes an already-open session.
- [x] Run focused tests and confirm they pass.
- [ ] Commit the Feishu service as part of the audited offline implementation commit.

## Task 5: Live Co-Scientist integration and evaluator safety gate

**Files:**
- Modify: `coscientist/coevo/agent_system.py`
- Modify: `coscientist/coevo/cli.py`
- Modify: `tests/test_coevo.py`
- Create: `tests/test_human_integration.py`

- [x] Write failing integration tests for task-definition consultation, verifier-change consultation before installation, evaluator freeze while a session is open, and disabled-mode backward compatibility.
- [x] Run focused integration tests and confirm failure.
- [x] Add optional CLI configuration for expert identity and five-session Human Session service.
- [x] Wire the task-definition hook before solving and the verifier-change hook before `_apply_harden` mutates the evaluator.
- [x] Run focused integration tests and confirm they pass.
- [ ] Commit the live integration as part of the audited offline implementation commit.

## Task 6: Evaluator-backed Human Proxy Agent

**Files:**
- Create: `coscientist/coevo/human_proxy_sessions.py`
- Create: `tests/test_human_proxy_sessions.py`
- Modify: `coscientist/coevo/agent_system.py`
- Modify: `coscientist/coevo/cli.py`

- [x] Write failing tests proving the Proxy uses the same five-session store, natural multi-turn transcript, close-request/confirmation states, durable outcome, evaluator freeze, and approval policy as a Feishu human.
- [x] Write failing tests showing the Proxy's decisions are grounded in a hidden real evaluator and that V* source/results do not leak to Solver or ordinary run evidence.
- [x] Implement a reference-evaluator adapter, evaluator-backed Proxy expert agent, and loopback Human Session port that reuses `FeishuHumanSessionService` rather than bypassing the human contract.
- [x] Add a mutually exclusive CLI configuration for Feishu-human mode or Human-Proxy mode.
- [x] Run focused tests and confirm they pass.
- [ ] Commit the Proxy implementation as part of the audited offline implementation commit.

## Task 7: Deployment, real Feishu verification, and A100 handoff

**Files:**
- Create: `docs/feishu-human-sessions.md`
- Modify: deployment/configuration files only as required by the verified runtime.

- [x] Document exact Feishu scopes, bot capability, long-connection event, visibility/publish steps, runtime command, storage layout, and recovery behavior.
- [ ] Run `pytest -q tests` locally and inspect the final diff for unrelated/UI changes.
- [ ] Push the feature branch and fast-forward the clean tracked files on A100 without touching its UI worktree changes.
- [ ] Run `pytest -q tests` on A100.
- [ ] With Feishu permissions published, run a real direct-message session: open, exchange multiple natural-language turns, query run evidence, request close, reject once, then confirm close.
- [ ] Verify on-disk transcript/outcome, session budget accounting, event deduplication, and sixth-session rejection on A100.
- [ ] Record exact verification evidence and commit the deployment documentation.
